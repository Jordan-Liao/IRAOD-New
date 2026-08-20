#!/usr/bin/env python
"""Trace the clean/corrupt trade-off by scaling SLRP's gate at inference.

The gate output is ``1 + gamma * tanh(...)``, so ``gamma`` scales the whole
correction linearly and can be swept *without retraining*: one calibrated
checkpoint yields the entire curve. That matters because the alternative --
retraining per setting -- costs 1.8 hours a point, and the two objective-level
attempts at protecting clean recall (a feature-invariance loss, then a clean-
sample gate quiescence penalty) both failed at a ~1:22 exchange rate. A capacity
limit is a different lever from an objective term: it bounds the worst-case
perturbation on clean input directly.

Reports recall per domain at each gamma so the knee of the curve is visible,
rather than assuming the trained gamma is the right operating point.
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

import torch  # noqa: E402
from mmdet.datasets import build_dataloader  # noqa: E402
from mmrotate.models import build_detector  # noqa: E402

import mmdet_extension  # noqa: E402,F401
import sfod  # noqa: E402,F401
from tools.analysis.corruption_severity import (  # noqa: E402
    _evaluate_domain)
from tools.analysis.gate_spatial_geometry import (  # noqa: E402
    DEFAULT_CORRUPTIONS, _corruption_root, _domain_dataset_cfg, _load_config)
from tools.analysis.sogc_runtime import (  # noqa: E402
    build_dataset_checked, dump_json, load_checkpoint_sogc_compatible,
    make_detector_runner, resolve_device, seed_everything)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Sweep SLRP gamma at inference and report recall')
    parser.add_argument(
        '--config',
        default='configs/baseline/oriented_rcnn_slrp_orthonet_rsar.py')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument(
        '--gammas', default='0.0,0.05,0.1,0.2,0.3,0.5',
        help='comma-separated gamma values; 0.0 is the no-SLRP control')
    parser.add_argument(
        '--corruptions',
        default='noise_suppression,am_noise_vertical,smart_suppression,'
                'am_noise_horizontal,chaff')
    parser.add_argument('--corruption-root', default=None)
    parser.add_argument('--max-images', type=int, default=300)
    parser.add_argument('--score-thr', type=float, default=0.5)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def _validate(args):
    gammas = []
    for token in args.gammas.split(','):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        if value < 0.0:
            raise ValueError('gamma must be non-negative')
        gammas.append(value)
    if not gammas:
        raise ValueError('--gammas must name at least one value')
    corruptions = [name.strip() for name in args.corruptions.split(',')
                   if name.strip()]
    unknown = set(corruptions) - set(DEFAULT_CORRUPTIONS)
    if unknown:
        raise ValueError(f'unknown corruptions: {sorted(unknown)}')
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / 'summary.json'
    if target.exists() and not args.overwrite:
        raise FileExistsError(f'refusing to overwrite {target}')
    return sorted(set(gammas)), corruptions, out_dir


def _set_gamma(backbone, gamma):
    """Override gamma on every SLRP module; gamma=0 makes the module identity."""
    count = 0
    for _, module in backbone.iter_slrp_modules():
        module.gamma = float(gamma)
        count += 1
    if count == 0:
        raise RuntimeError('config enables no SLRP module to sweep')
    return count


def main():
    args = parse_args()
    gammas, corruptions, out_dir = _validate(args)
    seed_everything(42)
    device, device_ids = resolve_device(args.device)
    cfg = _load_config(Path(args.config).expanduser().resolve())
    corruption_root = _corruption_root(cfg, args.corruption_root)

    model_cfg = copy.deepcopy(cfg.model)
    model_cfg.train_cfg = None
    model = build_detector(model_cfg, test_cfg=cfg.get('test_cfg'))
    load_checkpoint_sogc_compatible(
        model, Path(args.checkpoint).expanduser().resolve(),
        allow_missing_sogc=True)
    model.CLASSES = cfg.get('classes')
    backbone = model.backbone
    modules = dict(backbone.iter_slrp_modules())
    if not modules:
        raise RuntimeError('config enables no SLRP module to sweep')
    trained_gamma = next(iter(modules.values())).gamma
    runner = make_detector_runner(model, device, device_ids)
    print(f'trained gamma={trained_gamma}, sweeping {gammas}')

    domains = ['clean'] + corruptions
    results = OrderedDict()
    with torch.no_grad():
        for gamma in gammas:
            _set_gamma(backbone, gamma)
            per_domain = OrderedDict()
            for domain in domains:
                label = None if domain == 'clean' else domain
                per_domain[domain] = _evaluate_domain(
                    domain, runner,
                    _domain_dataset_cfg(cfg, label, corruption_root),
                    args, device)
                recall = per_domain[domain]['micro_recall']
                print(f'  gamma={gamma:.2f} {domain}: recall={recall:.4f}')
            results[f'{gamma:g}'] = per_domain

    summary = {
        'settings': {
            'checkpoint': str(Path(args.checkpoint).resolve()),
            'trained_gamma': trained_gamma,
            'gammas': gammas,
            'max_images': args.max_images,
            'score_thr': args.score_thr,
        },
        'per_gamma': results,
    }
    dump_json(out_dir / 'summary.json', summary)

    print('\n=== recall vs gamma (gamma=0 is the no-SLRP control) ===')
    header = ''.join(f'{f"g={g:g}":>10s}' for g in gammas)
    print(f'{"domain":24s}{header}')
    for domain in domains:
        cells = ''.join(
            f'{results[f"{g:g}"][domain]["micro_recall"]:>10.4f}'
            for g in gammas)
        print(f'{domain:24s}{cells}')

    print('\n=== delta from the no-SLRP control ===')
    base_key = f'{gammas[0]:g}'
    if gammas[0] != 0.0:
        print('(gamma=0 not swept; deltas unavailable)')
    else:
        print(f'{"domain":24s}{header}')
        for domain in domains:
            reference = results[base_key][domain]['micro_recall']
            cells = ''.join(
                f'{results[f"{g:g}"][domain]["micro_recall"] - reference:>+10.4f}'
                for g in gammas)
            print(f'{domain:24s}{cells}')
    print(f'\nout_dir={out_dir}')


if __name__ == '__main__':
    main()
