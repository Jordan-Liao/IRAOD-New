#!/usr/bin/env python
"""Audit VLST arm C: pseudo-label purity and VLM agreement on a corruption.

Loads the arm C EMA teacher, runs pseudo-label inference on the adaptation split
(corrupt val), and for every accepted box (score >= pseudo thr) asks:
  1. Is the pseudo label correct?            (pseudo precision, per class)
  2. Does base SARCLIP agree with it?        (VLM agreement, the spec metric)
  3. Does base SARCLIP agree with GT?        (is the VLM informative here?)
  4. Does the LoRA SARCLIP agree with GT?    (is base-SARCLIP weakness the issue?)
Also, among WRONG pseudo labels: does the VLM top-1 match GT (would VLM-consistency
gating have helped)?

This answers the prototype-purity / contamination diagnosis items and the VLM
agreement metric, separating the "visual prototypes contaminated by wrong pseudo
labels" hypothesis from the "base SARCLIP is uninformative on corrupt crops"
hypothesis for why ST+Proto < ST.
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from iraod_runtime import ensure_iraod_runtime  # noqa: E402

ensure_iraod_runtime()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from mmcv import Config  # noqa: E402
from mmdet.datasets import build_dataloader  # noqa: E402
from mmrotate.models import build_detector  # noqa: E402

import mmdet_extension  # noqa: E402,F401
import sfod  # noqa: E402,F401
from sfod.cga import CGA, RSAR_CLASSES  # noqa: E402
from tools.analysis.sogc_runtime import (  # noqa: E402
    build_dataset_checked, dump_json, load_checkpoint_sogc_compatible,
    make_detector_runner, resolve_device, seed_everything)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config',
        default=('configs/baseline/ema_config/'
                 'baseline_oriented_rcnn_ema_rsar_cga_orthonet.py'))
    parser.add_argument(
        '--data-config',
        default=('configs/unbiased_teacher/sfod/'
                 'unbiased_teacher_oriented_rcnn_selftraining_vlst_rsar_orthonet.py'),
        help='config providing the data section (EMA baseline configs have none)')
    parser.add_argument(
        '--checkpoint',
        default=('work_dirs/vlst_baseline_noise_suppression/iter_4235_ema.pth'))
    parser.add_argument('--corruption', default='noise_suppression')
    parser.add_argument('--split', default='val')
    parser.add_argument('--score-thr', type=float, default=0.7)
    parser.add_argument('--iou-thr', type=float, default=0.5)
    parser.add_argument('--max-images', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--out', default=None)
    return parser.parse_args()


def rotated_iou_matrix(predicted, truth, device):
    """Pairwise rotated IoU; empty-safe."""
    from mmcv.ops import box_iou_rotated  # noqa: PLC0415

    if predicted.size == 0 or truth.size == 0:
        return np.zeros((predicted.shape[0], truth.shape[0]), dtype=np.float32)
    left = torch.as_tensor(predicted[:, :5], dtype=torch.float32, device=device)
    right = torch.as_tensor(truth[:, :5], dtype=torch.float32, device=device)
    return box_iou_rotated(left, right).cpu().numpy()


def build_vlm(use_lora):
    """Build the same SARCLIP scorer VLST/CGA use, optionally LoRA-tuned."""
    os.environ.pop('SARCLIP_LORA', None)
    if use_lora:
        os.environ['SARCLIP_LORA'] = (
            '/myfile/mycode/IRAOD-New/work_dirs/'
            'sarclip_lora_rsar_train_corrupt_aabb_v1/lora_rsar.pth')
    return CGA(
        class_names=list(RSAR_CLASSES),
        backend='sarclip',
        model='ViT-B-32',
        pretrained='/myfile/pretrain/SARCLIP/ViT-B-32/vit_b_32_model.safetensors',
        cache_dir='/myfile/pretrain/SARCLIP/ViT-B-32',
    )


def main():
    args = parse_args()
    seed_everything(42)
    device, device_ids = resolve_device(args.device)
    cfg = Config.fromfile(Path(args.config).expanduser().resolve())
    data_cfg = Config.fromfile(Path(args.data_config).expanduser().resolve())

    root = Path('/myfile/dataset/RSAR')
    dataset_cfg = copy.deepcopy(data_cfg.data.test)
    dataset_cfg.img_prefix = str(
        root / 'corruptions' / args.corruption / args.split / 'images') + '/'
    dataset_cfg.ann_file = str(root / args.split / 'annfiles') + '/'
    dataset = build_dataset_checked(dataset_cfg)

    model_cfg = copy.deepcopy(cfg.model)
    model_cfg.train_cfg = None
    model = build_detector(model_cfg, test_cfg=cfg.get('test_cfg'))
    load_checkpoint_sogc_compatible(
        model, Path(args.checkpoint).expanduser().resolve(),
        allow_missing_sogc=True)
    model.eval()
    model.CLASSES = list(RSAR_CLASSES)
    runner = make_detector_runner(model, device, device_ids)

    loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=2, num_gpus=1,
        dist=False, shuffle=False)

    vlm_base = build_vlm(use_lora=False)
    vlm_lora = build_vlm(use_lora=True)

    num_classes = len(RSAR_CLASSES)
    # Counters
    kept_total = 0
    correct_pseudo = 0
    gt_by_class = np.zeros(num_classes, dtype=np.int64)
    agree_base_pseudo = 0
    agree_base_gt = 0
    agree_lora_gt = 0
    agree_base_gt_among_wrong = 0
    wrong_pseudo_total = 0
    per_class = {
        c: {'kept': 0, 'correct': 0, 'vlm_base_gt': 0} for c in range(num_classes)
    }
    matched_gt_boxes = 0
    gt_total = 0
    scored_total = 0
    scored_matched = 0

    images = 0
    with torch.no_grad():
        for index, data in enumerate(loader):
            if args.max_images and index >= args.max_images:
                break
            filename = os.path.join(
                dataset_cfg.img_prefix, dataset.data_infos[index]['filename'])
            results = runner(return_loss=False, rescale=True, **data)[0]
            annotation = dataset.get_ann_info(index)
            truth = annotation['bboxes']
            if truth is None or len(truth) == 0:
                continue
            truth = np.asarray(truth, dtype=np.float32).reshape(-1, 5)
            truth_labels = np.asarray(
                annotation['labels'], dtype=np.int64).reshape(-1)
            gt_total += len(truth)
            for label in truth_labels:
                if 0 <= label < num_classes:
                    gt_by_class[label] += 1

            # Accepted pseudo boxes (per-class detections >= thr)
            kept = []
            kept_labels = []
            for class_id, detections in enumerate(results):
                detections = np.asarray(detections, dtype=np.float32)
                if detections.size == 0:
                    continue
                detections = detections.reshape(-1, 6)
                above = detections[detections[:, 5] >= args.score_thr]
                if above.size:
                    kept.append(above)
                    kept_labels.append(
                        np.full(len(above), class_id, dtype=np.int64))
            if not kept:
                continue
            kept = np.concatenate(kept, axis=0)
            kept_labels = np.concatenate(kept_labels)
            kept_total += len(kept)

            # Match pseudo boxes to GT (rotated IoU)
            ious = rotated_iou_matrix(kept, truth, device)
            best_gt = ious.argmax(axis=1) if ious.size else None
            best_iou = ious.max(axis=1) if ious.size else np.zeros(len(kept))
            is_matched = best_iou >= args.iou_thr
            matched_gt_boxes += int((best_iou >= args.iou_thr).sum())

            pseudo_correct = np.zeros(len(kept), dtype=bool)
            gt_of_kept = np.full(len(kept), -1, dtype=np.int64)
            for i, matched in enumerate(is_matched):
                if not matched:
                    continue
                gt_of_kept[i] = truth_labels[best_gt[i]]
                if truth_labels[best_gt[i]] == kept_labels[i]:
                    pseudo_correct[i] = True
                    correct_pseudo += 1
                    per_class[kept_labels[i]]['correct'] += 1
            wrong_pseudo_total += int((~pseudo_correct).sum())
            for c in np.unique(kept_labels):
                per_class[int(c)]['kept'] += int((kept_labels == c).sum())

            # VLM scoring on the same crops both VLST and CGA use. Kept boxes
            # are rotated OBBs (cx,cy,w,h,theta); the crop path takes AABBs.
            from mmrotate.core import obb2xyxy  # noqa: PLC0415
            # mmrotate's obb2xyxy implementation expects a torch Tensor in
            # this environment, while detections above are NumPy arrays.
            boxes_xyxy = obb2xyxy(
                torch.as_tensor(kept[:, :5], dtype=torch.float32,
                                device=device)
            ).detach().cpu().numpy()[:, :4].copy()
            scores = kept[:, 5].copy()
            labels_in = kept_labels.copy()

            # Drop degenerate AABBs the crop path cannot open
            widths = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
            heights = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
            crop_valid = (widths >= 1.0) & (heights >= 1.0)
            if not crop_valid.any():
                continue
            scored_total += int(crop_valid.sum())
            scored_matched += int(
                ((gt_of_kept >= 0) & crop_valid).sum())
            probs_base = vlm_base(
                filename, boxes_xyxy[crop_valid], scores[crop_valid],
                labels_in[crop_valid])[0]
            probs_lora = vlm_lora(
                filename, boxes_xyxy[crop_valid], scores[crop_valid],
                labels_in[crop_valid])[0]
            top_base = probs_base.argmax(axis=1)
            top_lora = probs_lora.argmax(axis=1)

            # Restrict agreement counters to boxes that were actually scored
            agree_base_pseudo += int(
                (top_base == kept_labels[crop_valid]).sum())
            agree_base_gt += int(
                ((top_base == gt_of_kept[crop_valid]) & (gt_of_kept[crop_valid] >= 0)
                 ).sum())
            agree_lora_gt += int(
                ((top_lora == gt_of_kept[crop_valid]) & (gt_of_kept[crop_valid] >= 0)
                 ).sum())
            wrong_mask = (
                ~pseudo_correct[crop_valid] & (gt_of_kept[crop_valid] >= 0))
            agree_base_gt_among_wrong += int(
                ((top_base == gt_of_kept[crop_valid]) & wrong_mask).sum())
            for c in range(num_classes):
                per_class[c]['vlm_base_gt'] += int(
                    ((top_base == c) & (gt_of_kept[crop_valid] == c)
                     & (gt_of_kept[crop_valid] >= 0)).sum())
            images += 1

    summary = {
        'settings': {
            'corruption': args.corruption,
            'split': args.split,
            'score_thr': args.score_thr,
            'iou_thr': args.iou_thr,
            'images': images,
            'checkpoint': args.checkpoint,
        },
        'gt_total': gt_total,
        'kept_total': kept_total,
        'teacher_recall': matched_gt_boxes / gt_total if gt_total else None,
        'pseudo_precision': correct_pseudo / kept_total if kept_total else None,
        # VLM agreement rates are over crop-valid boxes (degenerate AABBs
        # cannot be scored by the crop path at all).
        'vlm_base_agree_pseudo': (
            agree_base_pseudo / scored_total if scored_total else None),
        'vlm_base_agree_gt': (
            agree_base_gt / scored_matched if scored_matched else None),
        'vlm_lora_agree_gt': (
            agree_lora_gt / scored_matched if scored_matched else None),
        'vlm_base_agree_gt_among_wrong_pseudo': (
            agree_base_gt_among_wrong / wrong_pseudo_total
            if wrong_pseudo_total else None),
        'scored_total': scored_total,
        'scored_matched': scored_matched,
        'gt_by_class': {
            RSAR_CLASSES[c]: int(gt_by_class[c]) for c in range(num_classes)
        },
        'per_class': {
            RSAR_CLASSES[c]: per_class[c] for c in range(num_classes)
        },
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        dump_json(Path(args.out), summary)
    return 0


if __name__ == '__main__':
    sys.exit(main())
