#!/usr/bin/env python
"""Measure an image-level bypass gate: can clean images be left bit-identical?

A per-pixel deadband was tried first and failed (``work_dirs/deadband_200``):
selectivity sat at ~1.0 and went *below* 1 for the AM stripes, because the CFAR
statistic references the scene's own median, so a corruption that makes the whole
scene more homogeneous moves numerator and denominator together and is invisible.

The descriptor is discriminative at the **image** level, not the pixel level --
that is where the 0.994 / 0.953 separability figures came from. So the gate has to
be per-image: compute one statistic per image, and if it says "no interference",
bypass the module entirely. Clean input then passes through byte-identical and its
accuracy is preserved by construction, not by hoping the training balanced it.

The number that decides whether this is usable is the clean false-positive rate at
a threshold that still catches the corrupted images: every clean image above the
threshold pays the accuracy cost, every corrupt image below it forfeits the gain.
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
    DEFAULT_CORRUPTIONS, _corruption_root, _domain_dataset_cfg, _load_config,
    _unwrap_image, _unwrap_metas)
from tools.analysis.sogc_runtime import (  # noqa: E402
    build_dataset_checked, dump_json, load_checkpoint_sogc_compatible,
    resolve_device, seed_everything)

# One scalar per descriptor channel per image. These are the summaries that
# produced the image-level separability figures; the per-pixel values did not
# separate at all.
FIELDS = ('separable_share', 'sparse_share', 'row_log_ratio', 'col_log_ratio',
          'local_cv', 'cv_contrast')
SUMMARIES = ('mean', 'std', 'p90', 'p10')
# Clean false-positive rates worth reporting: the fraction of clean images that
# would be touched, i.e. the fraction that pays any accuracy cost at all.
TARGET_FPR = (0.0, 0.01, 0.02, 0.05, 0.10)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Measure an image-level interference bypass gate')
    parser.add_argument(
        '--config',
        default='configs/baseline/oriented_rcnn_slrp_orthonet_rsar.py')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument(
        '--corruptions',
        default='noise_suppression,am_noise_vertical,am_noise_horizontal,'
                'smart_suppression,chaff')
    parser.add_argument('--corruption-root', default=None)
    parser.add_argument('--max-images', type=int, default=300)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def _field_names():
    return [f'{field}_{summary}'
            for field in FIELDS for summary in SUMMARIES]


@torch.no_grad()
def _collect(label, backbone, dataset_cfg, args, device, window):
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
            feature.shape[2:], metas['img_shape'][:2], tuple(image.shape[2:]))
        descriptor = interference_descriptor(
            feature[:, :, :keep_h, :keep_w], window)
        row = {}
        for channel, field in enumerate(FIELDS):
            values = descriptor[0, channel].flatten().to(dtype=torch.float64)
            row[f'{field}_mean'] = values.mean().item()
            row[f'{field}_std'] = values.std(unbiased=False).item()
            row[f'{field}_p90'] = torch.quantile(values, 0.9).item()
            row[f'{field}_p10'] = torch.quantile(values, 0.1).item()
        rows.append(row)
    if not rows:
        raise RuntimeError(f'domain {label} produced no rows')
    return rows


def _values(rows, field):
    values = np.asarray([row[field] for row in rows], dtype=np.float64)
    return values[np.isfinite(values)]


def _operating_points(clean, corrupt):
    """True-positive rate at each target clean false-positive rate.

    Two-sided: the corrupt distribution may sit either above or below clean, so
    the threshold is taken on whichever tail separates.
    """
    points = {}
    for fpr in TARGET_FPR:
        best = None
        for side in ('upper', 'lower'):
            if side == 'upper':
                quantile = 100.0 * (1.0 - fpr)
                threshold = np.percentile(clean, min(quantile, 100.0))
                # Strictly above, so fpr=0 means no clean image is touched.
                clean_rate = float(np.mean(clean > threshold))
                tpr = float(np.mean(corrupt > threshold))
            else:
                threshold = np.percentile(clean, max(100.0 * fpr, 0.0))
                clean_rate = float(np.mean(clean < threshold))
                tpr = float(np.mean(corrupt < threshold))
            if best is None or tpr > best['tpr']:
                best = {'side': side, 'threshold': float(threshold),
                        'clean_fpr': clean_rate, 'tpr': tpr}
        points[f'{fpr:g}'] = best
    return points


def main():
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / 'summary.json'
    if target.exists() and not args.overwrite:
        raise FileExistsError(f'refusing to overwrite {target}')
    corruptions = [name.strip() for name in args.corruptions.split(',')
                   if name.strip()]
    unknown = set(corruptions) - set(DEFAULT_CORRUPTIONS)
    if unknown:
        raise ValueError(f'unknown corruptions: {sorted(unknown)}')

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
    if not modules:
        raise RuntimeError('config enables no SLRP module')
    window = next(iter(modules.values())).window

    clean_rows = _collect(
        'clean', backbone, _domain_dataset_cfg(cfg, None, corruption_root),
        args, device, window)
    print(f'clean: {len(clean_rows)} images')

    blocks = OrderedDict()
    for corruption in corruptions:
        rows = _collect(
            corruption, backbone,
            _domain_dataset_cfg(cfg, corruption, corruption_root),
            args, device, window)
        per_field = {}
        for field in _field_names():
            clean = _values(clean_rows, field)
            corrupt = _values(rows, field)
            if clean.size == 0 or corrupt.size == 0:
                continue
            per_field[field] = _operating_points(clean, corrupt)
        blocks[corruption] = {'sample_count': len(rows), 'fields': per_field}
        print(f'{corruption}: {len(rows)} images')

    dump_json(target, {
        'settings': {'window': window, 'max_images': args.max_images},
        'per_corruption': blocks,
    })

    print('\n=== best single field: TPR at each clean false-positive rate ===')
    header = ''.join(f'{f"fpr={f:g}":>11s}' for f in TARGET_FPR)
    print(f'{"corruption":22s}{header}   best field')
    for corruption, block in blocks.items():
        # Rank by TPR at zero clean cost: that is the operating point that
        # preserves clean accuracy exactly.
        ranked = sorted(
            block['fields'].items(),
            key=lambda item: item[1]['0']['tpr'], reverse=True)
        field, points = ranked[0]
        cells = ''.join(
            f'{points[f"{f:g}"]["tpr"]:>11.3f}' for f in TARGET_FPR)
        print(f'{corruption:22s}{cells}   {field}')
    print(f'\nout_dir={out_dir}')


if __name__ == '__main__':
    main()
