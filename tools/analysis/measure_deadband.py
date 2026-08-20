#!/usr/bin/env python
"""Measure whether a CFAR-style deadband can leave clean input bit-identical.

The question this answers: is there a per-pixel threshold on the interference
descriptor below which clean imagery almost never fires, while corrupted imagery
still fires on the interference itself?

If yes, the suppression module can be given a hard deadband -- output *exactly*
zero correction below threshold, not a small one -- and clean-input accuracy is
preserved by construction rather than by hoping the training balanced it. That
matters because both attempts at protecting clean accuracy through the objective
(feature invariance, then a clean-sample gate penalty) failed at roughly a 1:22
exchange rate against the corrupt side.

Reported per threshold:
  clean_fire_rate    fraction of clean pixels that would be touched
  corrupt_fire_rate  fraction of corrupt pixels that would be touched
  clean_image_clean  fraction of clean images left completely untouched
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


def parse_args():
    parser = argparse.ArgumentParser(
        description='Measure a CFAR deadband on the interference descriptor')
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
    parser.add_argument('--max-images', type=int, default=200)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


# Channels 4 and 5 of the descriptor: local CV, and log CV relative to the
# scene median. The CFAR statistic is the deviation of the latter from zero --
# zero means "as homogeneous as the rest of this scene".
CFAR_CHANNEL = 5
THRESHOLDS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0)


@torch.no_grad()
def _collect(label, backbone, dataset_cfg, args, device, window):
    """Per-image fire rates of the CFAR statistic at every threshold."""
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
        statistic = descriptor[0, CFAR_CHANNEL].abs()
        total = float(statistic.numel())
        rows.append({
            threshold: float((statistic > threshold).sum().item()) / total
            for threshold in THRESHOLDS
        })
    if not rows:
        raise RuntimeError(f'domain {label} produced no rows')
    return rows


def _summarize(rows):
    return {
        f'{threshold:g}': {
            'mean_fire_rate': float(
                np.mean([row[threshold] for row in rows])),
            'median_fire_rate': float(
                np.median([row[threshold] for row in rows])),
            'images_untouched': float(
                np.mean([row[threshold] == 0.0 for row in rows])),
        }
        for threshold in THRESHOLDS
    }


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

    blocks = OrderedDict()
    blocks['clean'] = _summarize(_collect(
        'clean', backbone, _domain_dataset_cfg(cfg, None, corruption_root),
        args, device, window))
    print('clean collected')
    for corruption in corruptions:
        blocks[corruption] = _summarize(_collect(
            corruption, backbone,
            _domain_dataset_cfg(cfg, corruption, corruption_root),
            args, device, window))
        print(f'{corruption} collected')

    dump_json(target, {
        'settings': {
            'cfar_channel': CFAR_CHANNEL,
            'window': window,
            'max_images': args.max_images,
        },
        'per_domain': blocks,
    })

    domains = list(blocks)
    print('\n=== mean fraction of pixels above threshold ===')
    header = ''.join(f'{f"t={t:g}":>9s}' for t in THRESHOLDS)
    print(f'{"domain":22s}{header}')
    for domain in domains:
        cells = ''.join(
            f'{blocks[domain][f"{t:g}"]["mean_fire_rate"]:>9.4f}'
            for t in THRESHOLDS)
        print(f'{domain:22s}{cells}')

    print('\n=== fraction of images left completely untouched ===')
    print(f'{"domain":22s}{header}')
    for domain in domains:
        cells = ''.join(
            f'{blocks[domain][f"{t:g}"]["images_untouched"]:>9.3f}'
            for t in THRESHOLDS)
        print(f'{domain:22s}{cells}')

    print('\n=== selectivity: corrupt fire rate / clean fire rate ===')
    print(f'{"domain":22s}{header}')
    for domain in domains[1:]:
        cells = ''
        for t in THRESHOLDS:
            clean = blocks['clean'][f'{t:g}']['mean_fire_rate']
            corrupt = blocks[domain][f'{t:g}']['mean_fire_rate']
            cells += f'{(corrupt / clean):>9.2f}' if clean > 0 else f'{"inf":>9s}'
        print(f'{domain:22s}{cells}')
    print(f'\nout_dir={out_dir}')


if __name__ == '__main__':
    main()
