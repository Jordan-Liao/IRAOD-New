#!/usr/bin/env python
"""Compare pseudo-label count against ground-truth density on the adaptation set.

`pseudo_num` in the training log is pseudo-labels per image. On its own it says
nothing -- 0.87 is healthy if images hold one object and catastrophic if they hold
six, because in the Unbiased Teacher recipe every undetected object becomes implicit
*background* supervision. With `use_bbox_reg=False` the only unsupervised signal is
classification, so a low recall teacher does not merely fail to help: it actively
teaches the student to suppress the objects it missed.

Reports GT boxes per image on the split SFOD adapts over, the teacher's recall at
the pseudo-label threshold, and the resulting count of objects per image that are
being mislabelled as background.
"""

import argparse
import copy
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
from mmrotate.core import obb2poly_np, poly2obb_np  # noqa: E402
from mmrotate.models import build_detector  # noqa: E402

import mmdet_extension  # noqa: E402,F401
import sfod  # noqa: E402,F401
from tools.analysis.sogc_runtime import (  # noqa: E402
    build_dataset_checked, dump_json, load_checkpoint_sogc_compatible,
    make_detector_runner, resolve_device, seed_everything)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config',
        default='configs/baseline/oriented_rcnn_orthonet_rsar.py')
    parser.add_argument(
        '--checkpoint',
        default='/myfile/pretrain/oriented_rcnn_orthonet_rsar_epoch_100.pth')
    parser.add_argument('--corruption', default='chaff')
    parser.add_argument(
        '--split', default='val',
        help='SFOD adapts over the val split, so that is the default')
    parser.add_argument(
        '--score-thr', type=float, default=0.82,
        help='the pseudo-label threshold the SFOD run used')
    parser.add_argument('--iou-thr', type=float, default=0.5)
    parser.add_argument('--max-images', type=int, default=300)
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


def main():
    args = parse_args()
    seed_everything(42)
    device, device_ids = resolve_device(args.device)
    cfg = Config.fromfile(Path(args.config).expanduser().resolve())

    root = Path('/myfile/dataset/RSAR')
    dataset_cfg = copy.deepcopy(cfg.data.test)
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
    model.CLASSES = cfg.get('classes')
    runner = make_detector_runner(model, device, device_ids)

    loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=2, num_gpus=1,
        dist=False, shuffle=False)

    gt_total = 0
    kept_total = 0
    matched_total = 0
    images = 0
    gt_per_image = []
    kept_per_image = []
    missed_per_image = []
    # Per-class counts: self-training under imbalance can starve minority classes
    # of positive supervision while still looking healthy in aggregate.
    class_names = list(cfg.get('classes') or [])
    gt_by_class = np.zeros(len(class_names), dtype=np.int64)
    kept_by_class = np.zeros(len(class_names), dtype=np.int64)
    matched_by_class = np.zeros(len(class_names), dtype=np.int64)

    with torch.no_grad():
        for index, data in enumerate(loader):
            if index >= args.max_images:
                break
            results = runner(return_loss=False, rescale=True, **data)[0]
            annotation = dataset.get_ann_info(index)
            truth = annotation['bboxes']
            if truth is None or len(truth) == 0:
                continue
            truth = np.asarray(truth, dtype=np.float32).reshape(-1, 5)
            truth_labels = np.asarray(
                annotation['labels'], dtype=np.int64).reshape(-1)
            for label in truth_labels:
                if 0 <= label < len(gt_by_class):
                    gt_by_class[label] += 1

            # Concatenate per-class detections, keeping only those the SFOD run
            # would have promoted to pseudo-labels.
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
                    if class_id < len(kept_by_class):
                        kept_by_class[class_id] += len(above)
            kept = (np.concatenate(kept, axis=0) if kept
                    else np.zeros((0, 6), dtype=np.float32))
            kept_labels = (np.concatenate(kept_labels) if kept_labels
                           else np.zeros(0, dtype=np.int64))

            ious = rotated_iou_matrix(kept, truth, device)
            if ious.size:
                best = ious.max(axis=0)
                matched = int((best >= args.iou_thr).sum())
                # Credit a match to the class of the pseudo-label that found it,
                # so a class with pseudo-labels but no correct ones is visible.
                assignment = ious.argmax(axis=0)
                for truth_index, iou in enumerate(best):
                    if iou >= args.iou_thr:
                        label = int(kept_labels[assignment[truth_index]])
                        if 0 <= label < len(matched_by_class):
                            matched_by_class[label] += 1
            else:
                matched = 0

            images += 1
            gt_total += len(truth)
            kept_total += len(kept)
            matched_total += matched
            gt_per_image.append(len(truth))
            kept_per_image.append(len(kept))
            missed_per_image.append(len(truth) - matched)

    if images == 0:
        print('FATAL: no annotated images read')
        return 1

    summary = {
        'settings': {
            'corruption': args.corruption,
            'split': args.split,
            'score_thr': args.score_thr,
            'iou_thr': args.iou_thr,
            'images': images,
        },
        'gt_per_image': gt_total / images,
        'pseudo_per_image': kept_total / images,
        'matched_per_image': matched_total / images,
        'missed_per_image': (gt_total - matched_total) / images,
        'teacher_recall': matched_total / gt_total if gt_total else None,
        'pseudo_precision': matched_total / kept_total if kept_total else None,
        'images_with_zero_pseudo': float(
            np.mean([count == 0 for count in kept_per_image])),
    }

    print(f'corruption={args.corruption} split={args.split} '
          f'score_thr={args.score_thr} images={images}')
    print()
    print(f'GT objects per image        : {summary["gt_per_image"]:.3f}')
    print(f'pseudo-labels per image     : {summary["pseudo_per_image"]:.3f}')
    print(f'  of which match a GT box   : {summary["matched_per_image"]:.3f}')
    print(f'GT objects MISSED per image : {summary["missed_per_image"]:.3f}'
          '   <- these become background supervision')
    print()
    print(f'teacher recall @thr         : {summary["teacher_recall"]:.4f}')
    print(f'pseudo-label precision      : {summary["pseudo_precision"]:.4f}'
          if summary['pseudo_precision'] is not None else
          'pseudo-label precision      : n/a (no pseudo-labels)')
    print(f'images with zero pseudo-lbl : '
          f'{summary["images_with_zero_pseudo"]:.3f}')

    if class_names:
        print()
        print('per class (the supervision each class actually receives):')
        print(f'{"class":10s} {"GT":>7s} {"pseudo":>7s} {"correct":>8s} '
              f'{"GT share":>9s} {"pseudo share":>13s} {"recall":>7s}')
        gt_sum = max(1, int(gt_by_class.sum()))
        kept_sum = max(1, int(kept_by_class.sum()))
        starved = []
        for index, name in enumerate(class_names):
            gt_count = int(gt_by_class[index])
            kept_count = int(kept_by_class[index])
            hit_count = int(matched_by_class[index])
            gt_share = gt_count / gt_sum
            kept_share = kept_count / kept_sum
            recall = hit_count / gt_count if gt_count else float('nan')
            print(f'{name:10s} {gt_count:7d} {kept_count:7d} {hit_count:8d} '
                  f'{gt_share:9.3f} {kept_share:13.3f} {recall:7.3f}')
            # A class whose share of the supervision is far below its share of
            # the objects is being trained mostly as background.
            if gt_count >= 10 and kept_share < 0.5 * gt_share:
                starved.append((name, gt_share, kept_share))
        summary['per_class'] = {
            name: {
                'gt': int(gt_by_class[index]),
                'pseudo': int(kept_by_class[index]),
                'correct': int(matched_by_class[index]),
                'gt_share': float(gt_by_class[index] / gt_sum),
                'pseudo_share': float(kept_by_class[index] / kept_sum),
            }
            for index, name in enumerate(class_names)
        }
        if starved:
            print()
            print('STARVED CLASSES (pseudo share below half the GT share):')
            for name, gt_share, kept_share in starved:
                print(f'  {name:10s} GT {gt_share:.3f} -> pseudo {kept_share:.3f}'
                      f'  ({kept_share / gt_share:.2f}x its due share)')
            print('  These receive little positive supervision while every missed')
            print('  instance acts as background, so classification-only')
            print('  self-training drives their logits down each iteration.')

    missed = summary['missed_per_image']
    kept = summary['pseudo_per_image']
    print()
    if kept > 0 and missed > kept:
        print(f'DIAGNOSIS: {missed:.2f} objects per image are taught as background '
              f'against {kept:.2f} taught as foreground -- a {missed / kept:.1f}:1 '
              'ratio against the signal. Classification-only self-training at this '
              'recall suppresses more than it reinforces.')
    elif kept > 0:
        print(f'Foreground signal ({kept:.2f}/img) outweighs the false-negative '
              f'background ({missed:.2f}/img).')

    if args.out:
        dump_json(Path(args.out).expanduser().resolve(), summary)
    return 0


if __name__ == '__main__':
    sys.exit(main())
