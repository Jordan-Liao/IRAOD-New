#!/usr/bin/env python
"""Check that SLRP's own descriptor carries the signal the gate measured.

``depth_profile.py`` establishes that the post-maxpool tap is where the RSAR
corruptions are most visible, using the ``spatial_geometry`` observables. Those
observables are diagnostic code, not what the module consumes. This script
scores the four channels the module actually reads -- separable share, sparse
share and the two log band ratios of ``interference_descriptor`` -- at the same
tap, so the module's input is verified rather than assumed.

An untrained, zero-initialized SLRP is used: the point is whether the descriptor
is informative before any fitting, not whether a trained gate helps.
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
from mmdet_extension.models.utils.slrp import (  # noqa: E402
    interference_descriptor)
from mmdet_extension.models.utils.spatial_geometry import (  # noqa: E402
    valid_feature_extent)
from tools.analysis.gate_spatial_geometry import (  # noqa: E402
    DEFAULT_CORRUPTIONS, _auroc, _corruption_root, _domain_dataset_cfg,
    _load_config, _paired_stats, _unwrap_image, _unwrap_metas)
from tools.analysis.sogc_runtime import (  # noqa: E402
    build_dataset_checked, dump_json, load_checkpoint_sogc_compatible,
    resolve_device, seed_everything)

DESCRIPTOR_FIELDS = (
    'separable_share', 'sparse_share', 'row_log_ratio', 'col_log_ratio',
    'local_cv', 'cv_contrast')
# Pooling the per-pixel descriptor two ways, because a localized corruption and
# a global one concentrate in different summaries.
SUMMARIES = ('mean', 'p90', 'p10', 'std')


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score SLRP's descriptor on clean/corrupt separability")
    parser.add_argument(
        '--config',
        default='configs/baseline/oriented_rcnn_slrp_orthonet_rsar.py')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--corruptions', default=','.join(DEFAULT_CORRUPTIONS))
    parser.add_argument('--corruption-root', default=None)
    parser.add_argument('--max-images', type=int, default=400)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gate-auroc', type=float, default=0.80)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def _field_names():
    return [f'{field}_{summary}'
            for field in DESCRIPTOR_FIELDS for summary in SUMMARIES]


def _summarize(descriptor):
    """Reduce a ``[1, 4, H, W]`` descriptor to scalar summaries per channel."""
    values = {}
    for index, field in enumerate(DESCRIPTOR_FIELDS):
        channel = descriptor[0, index].flatten().to(dtype=torch.float64)
        values[f'{field}_mean'] = channel.mean().item()
        values[f'{field}_p90'] = torch.quantile(channel, 0.9).item()
        values[f'{field}_p10'] = torch.quantile(channel, 0.1).item()
        values[f'{field}_std'] = channel.std(unbiased=False).item()
    return values


@torch.no_grad()
def _collect_domain(label, backbone, module, dataset_cfg, args, device):
    dataset = build_dataset_checked(dataset_cfg)
    loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=2, num_gpus=1,
        dist=False, shuffle=False)
    rows = []
    for index, data in enumerate(loader):
        if index >= args.max_images:
            break
        metas = _unwrap_metas(data)
        image = _unwrap_image(data).to(device)
        if backbone.deep_stem:
            feature = backbone.stem(image)
        else:
            feature = backbone.relu(backbone.norm1(backbone.conv1(image)))
        feature = backbone.maxpool(feature)
        keep_h, keep_w = valid_feature_extent(
            feature.shape[2:], metas['img_shape'][:2],
            tuple(image.shape[2:]))
        descriptor = interference_descriptor(
            feature[:, :, :keep_h, :keep_w], module.window,
            sparse_multiplier=module.sparse_multiplier, eps=module.eps)
        rows.append({
            'domain': label,
            'filename': metas.get('ori_filename', metas.get('filename', '')),
            **_summarize(descriptor),
        })
    if not rows:
        raise RuntimeError(f'domain {label} produced no rows')
    return rows


def _finite(rows, field):
    values = np.asarray([row[field] for row in rows], dtype=np.float64)
    return values[np.isfinite(values)]


def _paired_arrays(clean_rows, corrupt_rows, field):
    clean_by_name = {row['filename']: row for row in clean_rows}
    pairs = [
        (clean_by_name[row['filename']][field], row[field])
        for row in corrupt_rows
        if row['filename'] in clean_by_name
        and np.isfinite(clean_by_name[row['filename']][field])
        and np.isfinite(row[field])]
    if not pairs:
        return np.asarray([]), np.asarray([])
    clean_values, corrupt_values = zip(*pairs)
    return (np.asarray(clean_values, dtype=np.float64),
            np.asarray(corrupt_values, dtype=np.float64))


def _score(clean_rows, corrupt_rows):
    scores = OrderedDict()
    for field in _field_names():
        auroc = _auroc(_finite(clean_rows, field), _finite(corrupt_rows, field))
        scores[field] = {
            'auroc': auroc,
            'separability': None if auroc is None else max(auroc, 1.0 - auroc),
            'paired': _paired_stats(
                *_paired_arrays(clean_rows, corrupt_rows, field)),
        }
    return scores


def main():
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / 'summary.json'
    if target.exists() and not args.overwrite:
        raise FileExistsError(f'refusing to overwrite {target}')
    corruptions = [
        name.strip() for name in args.corruptions.split(',') if name.strip()]
    seed_everything(42)
    device, _ = resolve_device(args.device)
    cfg = _load_config(Path(args.config).expanduser().resolve())
    corruption_root = _corruption_root(cfg, args.corruption_root)

    model_cfg = copy.deepcopy(cfg.model)
    model_cfg.train_cfg = None
    model = build_detector(model_cfg, test_cfg=cfg.get('test_cfg'))
    load_checkpoint_sogc_compatible(
        model, Path(args.checkpoint).expanduser().resolve(),
        allow_missing_sogc=True)
    backbone = model.backbone.to(device).eval()
    modules = dict(backbone.iter_slrp_modules())
    if backbone.STEM_STAGE not in modules:
        raise RuntimeError(
            'config must place SLRP at the post-maxpool tap (stages=(-1,))')
    module = modules[backbone.STEM_STAGE]

    clean_rows = _collect_domain(
        'clean', backbone, module,
        _domain_dataset_cfg(cfg, None, corruption_root), args, device)
    print(f'clean: {len(clean_rows)} images')

    blocks = OrderedDict()
    for corruption in corruptions:
        rows = _collect_domain(
            corruption, backbone, module,
            _domain_dataset_cfg(cfg, corruption, corruption_root),
            args, device)
        scores = _score(clean_rows, rows)
        ranked = [(entry['separability'], field)
                  for field, entry in scores.items()
                  if entry['separability'] is not None]
        best_sep, best_field = max(ranked)
        blocks[corruption] = {
            'sample_count': len(rows),
            'best': {
                'field': best_field,
                'separability': best_sep,
                'auroc': scores[best_field]['auroc'],
                'paired_win_rate': scores[best_field]['paired']['win_rate'],
            },
            'passes_gate': bool(best_sep >= args.gate_auroc),
            'descriptor_channels': scores,
        }
        print(f'{corruption}: {len(rows)} images')

    summary = {
        'settings': {
            'max_images': args.max_images,
            'tap': 'stem_maxpool',
            'window': module.window,
            'sparse_multiplier': module.sparse_multiplier,
        },
        'per_corruption': blocks,
    }
    dump_json(target, summary)

    print('\n=== SLRP descriptor separability at the post-maxpool tap ===')
    print(f'{"corruption":24s}{"best field":26s}{"sep":>8s}{"paired":>9s}')
    for corruption, block in blocks.items():
        best = block['best']
        flag = ' PASS' if block['passes_gate'] else ''
        print(f'{corruption:24s}{best["field"]:26s}'
              f'{best["separability"]:>8.3f}{best["paired_win_rate"]:>9.3f}'
              f'{flag}')
    print(f'\nout_dir={out_dir}')


if __name__ == '__main__':
    main()
