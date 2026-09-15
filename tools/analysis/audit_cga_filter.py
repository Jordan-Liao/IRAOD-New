#!/usr/bin/env python
"""Audit what CGA's filter actually does to real pseudo-labels, against GT.

Calls the real `refine_test` path rather than reimplementing the rule, so what is
measured is what the SFOD run executes. For every box the filter touches, checks
whether the detector's label was already correct: suppressing a correct label
injects an error, suppressing a wrong one removes one. The ratio between those is
the only number that says whether the filter helps.

Reports the drop and blend paths separately -- in veto_soft they have completely
different gates (drop needs three conditions, blend fires on any disagreement), so
an aggregate figure hides which one is actually active.
"""

import argparse
import copy
import os
import sys
from collections import Counter
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
from tools.analysis.sogc_runtime import (  # noqa: E402
    build_dataset_checked, dump_json, load_checkpoint_sogc_compatible,
    make_detector_runner, resolve_device, seed_everything)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config', default='configs/baseline/oriented_rcnn_orthonet_rsar.py')
    parser.add_argument(
        '--checkpoint',
        default='/myfile/pretrain/oriented_rcnn_orthonet_rsar_epoch_100.pth')
    parser.add_argument('--corruption', default='chaff')
    parser.add_argument('--split', default='val')
    parser.add_argument('--score-thr', type=float, default=0.82)
    parser.add_argument('--iou-thr', type=float, default=0.5)
    parser.add_argument('--max-images', type=int, default=250)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--out', default=None)
    return parser.parse_args()


def rotated_iou(predicted, truth, device):
    from mmcv.ops import box_iou_rotated  # noqa: PLC0415

    if predicted.size == 0 or truth.size == 0:
        return np.zeros((predicted.shape[0], truth.shape[0]), dtype=np.float32)
    left = torch.as_tensor(predicted[:, :5], dtype=torch.float32, device=device)
    right = torch.as_tensor(truth[:, :5], dtype=torch.float32, device=device)
    return box_iou_rotated(left, right).cpu().numpy()


def label_is_correct(boxes, labels, truth_boxes, truth_labels, iou_thr, device):
    """Per box: does it hit a GT box of the same class?"""
    if boxes.size == 0:
        return np.zeros(0, dtype=bool)
    ious = rotated_iou(boxes, truth_boxes, device)
    correct = np.zeros(len(boxes), dtype=bool)
    if ious.size == 0:
        return correct
    for index in range(len(boxes)):
        order = np.argsort(-ious[index])
        for truth_index in order:
            if ious[index, truth_index] < iou_thr:
                break
            if truth_labels[truth_index] == labels[index]:
                correct[index] = True
                break
    return correct


def main():
    args = parse_args()
    seed_everything(42)
    device, device_ids = resolve_device(args.device)
    cfg = Config.fromfile(Path(args.config).expanduser().resolve())
    class_names = list(cfg.get('classes') or [])

    root = Path('/myfile/dataset/RSAR')
    image_dir = root / 'corruptions' / args.corruption / args.split / 'images'
    dataset_cfg = copy.deepcopy(cfg.data.test)
    dataset_cfg.img_prefix = str(image_dir) + '/'
    dataset_cfg.ann_file = str(root / args.split / 'annfiles') + '/'
    dataset = build_dataset_checked(dataset_cfg)

    # The filter lives on the detector: TestMixins._build_cga sets the veto /
    # blend parameters on `self`, and refine_test reads them from there.
    # Constructing a bare CGA object skips all of it, so the audit has to use the
    # detector as the filter host or it measures a differently-configured filter.
    model_cfg = copy.deepcopy(cfg.model)
    model_cfg.train_cfg = None
    detector_type = model_cfg.get('type')
    if detector_type != 'OrientedRCNN_CGA':
        model_cfg.type = 'OrientedRCNN_CGA'
        print(f'[audit] detector type {detector_type} -> OrientedRCNN_CGA '
              '(needed for the CGA filter path)')
    model = build_detector(model_cfg, test_cfg=cfg.get('test_cfg'))
    load_checkpoint_sogc_compatible(
        model, Path(args.checkpoint).expanduser().resolve(),
        allow_missing_sogc=True)
    model.eval()
    model.CLASSES = class_names
    if not hasattr(model, '_build_cga'):
        print('FATAL: detector does not inherit TestMixins; no CGA filter to audit')
        return 1
    model._build_cga(len(class_names))
    scorer = model
    print(f'filter mode = {scorer.cga_filter_mode}')
    print(f'protect_det_score = {scorer.cga_protect_det_score}  '
          f'veto_pred_thr = {scorer.cga_veto_pred_thr}  '
          f'veto_label_thr = {scorer.cga_veto_label_thr}  '
          f'blend_det_weight = {scorer.cga_blend_detector_weight}')
    runner = make_detector_runner(model, device, device_ids)

    loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=2, num_gpus=1,
        dist=False, shuffle=False)

    total = 0
    changed = Counter()          # path -> count
    changed_correct = Counter()  # path -> count where detector label was right
    images = 0

    with torch.no_grad():
        for index, data in enumerate(loader):
            if index >= args.max_images:
                break
            results = runner(return_loss=False, rescale=True, **data)[0]
            annotation = dataset.get_ann_info(index)
            truth_boxes = np.asarray(
                annotation['bboxes'], dtype=np.float32).reshape(-1, 5)
            truth_labels = np.asarray(
                annotation['labels'], dtype=np.int64).reshape(-1)

            # Keep only what the SFOD run would promote, in the per-class list
            # layout refine_test expects.
            thresholded = []
            for detections in results:
                detections = np.asarray(
                    detections, dtype=np.float32).reshape(-1, 6)
                thresholded.append(
                    detections[detections[:, 5] >= args.score_thr])
            if not any(len(part) for part in thresholded):
                continue

            boxes = np.concatenate(
                [part[:, :5] for part in thresholded if len(part)], axis=0)
            before = np.concatenate(
                [part[:, 5] for part in thresholded if len(part)], axis=0)
            labels = np.concatenate([
                np.full(len(part), class_id, dtype=np.int64)
                for class_id, part in enumerate(thresholded) if len(part)])

            correct = label_is_correct(
                boxes, labels, truth_boxes, truth_labels, args.iou_thr, device)

            meta = data['img_metas'][0].data[0][0]
            # refine_test mutates in place, so hand it a copy.
            refined = scorer.refine_test(
                [copy.deepcopy(thresholded)], [meta])
            after = np.concatenate([
                np.asarray(part, dtype=np.float32).reshape(-1, 6)[:, 5]
                for part in refined[0] if len(part)])
            if after.shape != before.shape:
                print('FATAL: filter changed box count; cannot align scores')
                return 1

            images += 1
            total += len(before)
            lowered = after < before - 1e-6
            for position in np.flatnonzero(lowered):
                path = 'dropped' if after[position] <= 1e-6 else 'blended'
                changed[path] += 1
                if correct[position]:
                    changed_correct[path] += 1
                # create_pseudo_results thresholds the *rescored* score, so a
                # blend that lands below score_thr is a silent hard drop, not the
                # soft downweight the code comment describes.
                if after[position] < args.score_thr:
                    changed[f'{path}_below_thr'] += 1
                    if correct[position]:
                        changed_correct[f'{path}_below_thr'] += 1

    if images == 0:
        print('FATAL: no images produced pseudo-labels')
        return 1

    print(f'\nimages={images}  boxes above thr={total}')
    print(f'{"path":10s} {"touched":>8s} {"was correct":>12s} '
          f'{"was wrong":>10s} {"precision":>10s}')
    summary = {'images': images, 'boxes': total, 'score_thr': args.score_thr,
               'paths': {}}
    for path in ('dropped', 'blended',
                 'dropped_below_thr', 'blended_below_thr'):
        touched = changed[path]
        harmful = changed_correct[path]
        helpful = touched - harmful
        precision = helpful / touched if touched else float('nan')
        print(f'{path:10s} {touched:8d} {harmful:12d} {helpful:10d} '
              f'{precision:10.3f}')
        summary['paths'][path] = {
            'touched': touched, 'suppressed_correct': harmful,
            'suppressed_wrong': helpful,
            'precision': None if not touched else precision}

    evicted = changed['dropped_below_thr'] + changed['blended_below_thr']
    evicted_correct = (changed_correct['dropped_below_thr']
                       + changed_correct['blended_below_thr'])
    print()
    print(f'boxes pushed below score_thr={args.score_thr} by CGA: {evicted}'
          f'  ({evicted_correct} of them correctly labelled)')
    if changed['blended'] and changed['blended_below_thr'] == changed['blended']:
        print('  EVERY blended box fell below threshold: with '
              f'blend_det_weight={scorer.cga_blend_detector_weight} a blend '
              'cannot survive this threshold, so "soft downweight" is a hard '
              'drop here.')

    touched_all = changed['dropped'] + changed['blended']
    harmful_all = (changed_correct['dropped'] + changed_correct['blended'])
    print()
    if touched_all == 0:
        print('CGA never modified a single score at these thresholds -- it '
              'cannot account for any accuracy change.')
    else:
        helpful_all = touched_all - harmful_all
        print(f'overall: {touched_all} boxes suppressed, {harmful_all} of them '
              f'correct -> precision {helpful_all / touched_all:.3f}')
        if harmful_all > helpful_all:
            print('The filter removes more correct labels than wrong ones, so it '
                  'injects net label noise into self-training.')

    if args.out:
        dump_json(Path(args.out).expanduser().resolve(), summary)
    return 0


if __name__ == '__main__':
    sys.exit(main())
