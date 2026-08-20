#!/usr/bin/env python
"""Measure how much each RSAR corruption actually degrades detection.

A separability gate is only interpretable next to this: an observable that
cannot see a corruption which does not degrade detection either has not failed.
Reuses the rotated box matching of ``analyze_sogc_geometry`` so the recall and
false-positive definitions stay identical across the analysis scripts.
"""

import argparse
import copy
import sys
from collections import OrderedDict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from iraod_runtime import ensure_iraod_runtime  # noqa: E402

ensure_iraod_runtime()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from mmdet.datasets import build_dataloader  # noqa: E402
from mmrotate.models import build_detector  # noqa: E402

import mmdet_extension  # noqa: E402,F401
import sfod  # noqa: E402,F401
from tools.analysis.analyze_sogc_geometry import _match_detections  # noqa: E402
from tools.analysis.gate_spatial_geometry import (  # noqa: E402
    DEFAULT_CORRUPTIONS, _corruption_root, _domain_dataset_cfg, _load_config,
    _unwrap_metas)
from tools.analysis.sogc_runtime import (  # noqa: E402
    build_dataset_checked, dump_json, load_checkpoint_sogc_compatible,
    make_detector_runner, resolve_device, seed_everything)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Measure per-corruption detection degradation on RSAR')
    parser.add_argument(
        '--config', default='configs/baseline/oriented_rcnn_orthonet_rsar.py')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--corruptions', default=','.join(DEFAULT_CORRUPTIONS))
    parser.add_argument('--corruption-root', default=None)
    parser.add_argument('--max-images', type=int, default=300)
    parser.add_argument('--score-thr', type=float, default=0.5)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def _validate_args(args):
    config = Path(args.config).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    for label, path in (('config', config), ('checkpoint', checkpoint)):
        if not path.is_file():
            raise FileNotFoundError(f'{label} not found: {path}')
    if args.max_images <= 0:
        raise ValueError('--max-images must be positive')
    if not 0.0 <= args.score_thr <= 1.0:
        raise ValueError('--score-thr must be in [0, 1]')
    corruptions = [
        name.strip() for name in args.corruptions.split(',') if name.strip()]
    if not corruptions:
        raise ValueError('--corruptions must name at least one corruption')
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / 'summary.json'
    if target.exists() and not args.overwrite:
        raise FileExistsError(
            f'refusing to overwrite {target} (use --overwrite)')
    return config, checkpoint, out_dir, corruptions


@torch.no_grad()
def _evaluate_domain(label, runner, dataset_cfg, args, device):
    dataset = build_dataset_checked(dataset_cfg)
    loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=2, num_gpus=1,
        dist=False, shuffle=False)
    totals = {'tp': 0, 'fp': 0, 'fn': 0, 'gt': 0, 'detections': 0}
    recalls = []
    for index, data in enumerate(loader):
        if index >= args.max_images:
            break
        results = runner(return_loss=False, rescale=True, **data)
        if len(results) != 1:
            raise RuntimeError('severity scan requires one result per batch')
        metrics = _match_detections(
            results[0], dataset.get_ann_info(index), args.score_thr, device)
        if metrics is None:
            raise RuntimeError(
                f'{label} image {index} has no usable ground truth; the split '
                'must carry annotations for a severity measurement')
        totals['tp'] += metrics['tp']
        totals['fp'] += metrics['fp']
        totals['fn'] += metrics['fn']
        totals['gt'] += metrics['tp'] + metrics['fn']
        totals['detections'] += metrics['high_conf_detections']
        if np.isfinite(metrics['gt_recall']):
            recalls.append(metrics['gt_recall'])
        # Guard against a silent no-op domain: the loader must be reading the
        # corrupt prefix, not the clean one.
        if index == 0:
            metas = _unwrap_metas(data)
            filename = str(metas.get('filename', ''))
            if label != 'clean' and label not in filename:
                raise RuntimeError(
                    f'domain {label} is reading {filename}, which is not under '
                    'the corruption prefix')
    images = min(args.max_images, len(dataset))
    return {
        'images': int(images),
        'gt_instances': int(totals['gt']),
        'detections': int(totals['detections']),
        'tp': int(totals['tp']),
        'fp': int(totals['fp']),
        'fn': int(totals['fn']),
        'micro_recall': (
            totals['tp'] / totals['gt'] if totals['gt'] else None),
        'precision': (
            totals['tp'] / totals['detections']
            if totals['detections'] else None),
        'macro_recall': float(np.mean(recalls)) if recalls else None,
        'fp_per_image': totals['fp'] / images if images else None,
    }


def main():
    args = parse_args()
    config, checkpoint, out_dir, corruptions = _validate_args(args)
    seed_everything(42)
    device, device_ids = resolve_device(args.device)
    cfg = _load_config(config)
    corruption_root = _corruption_root(cfg, args.corruption_root)

    model_cfg = copy.deepcopy(cfg.model)
    model_cfg.train_cfg = None
    model = build_detector(model_cfg, test_cfg=cfg.get('test_cfg'))
    load_checkpoint_sogc_compatible(
        model, checkpoint, allow_missing_sogc=True)
    model.eval()
    runner = make_detector_runner(model, device, device_ids)

    domains = OrderedDict()
    domains['clean'] = _evaluate_domain(
        'clean', runner, _domain_dataset_cfg(cfg, None, corruption_root),
        args, device)
    print(f"clean: recall={domains['clean']['micro_recall']:.4f}")
    for corruption in corruptions:
        domains[corruption] = _evaluate_domain(
            corruption, runner,
            _domain_dataset_cfg(cfg, corruption, corruption_root),
            args, device)
        print(f"{corruption}: recall={domains[corruption]['micro_recall']:.4f}")

    baseline = domains['clean']['micro_recall']
    severity = OrderedDict()
    for name, block in domains.items():
        if name == 'clean':
            continue
        recall = block['micro_recall']
        severity[name] = {
            'micro_recall': recall,
            'recall_drop': (
                None if recall is None or baseline is None
                else baseline - recall),
            'relative_recall_drop': (
                None if not baseline else (baseline - recall) / baseline),
            'fp_per_image': block['fp_per_image'],
            'fp_per_image_delta': (
                block['fp_per_image'] - domains['clean']['fp_per_image']),
            'precision': block['precision'],
        }

    summary = {
        'settings': {
            'max_images': args.max_images,
            'score_thr': args.score_thr,
            'config': str(config),
            'checkpoint': str(checkpoint),
        },
        'clean': domains['clean'],
        'per_corruption_raw': domains,
        'severity': severity,
    }
    dump_json(out_dir / 'summary.json', summary)

    print(f'\n=== severity (clean micro recall = {baseline:.4f}) ===')
    print(f'{"corruption":24s}{"recall":>9s}{"drop":>9s}'
          f'{"rel_drop":>10s}{"fp/img":>9s}{"d_fp/img":>10s}')
    for name, entry in sorted(
            severity.items(), key=lambda item: -(item[1]['recall_drop'] or 0)):
        print(f'{name:24s}{entry["micro_recall"]:>9.4f}'
              f'{entry["recall_drop"]:>9.4f}'
              f'{entry["relative_recall_drop"]:>10.1%}'
              f'{entry["fp_per_image"]:>9.2f}'
              f'{entry["fp_per_image_delta"]:>10.2f}')
    print(f'\nout_dir={out_dir}')


if __name__ == '__main__':
    main()
