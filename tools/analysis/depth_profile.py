#!/usr/bin/env python
"""Locate the depth at which each corruption's signature is still visible.

``gate_spatial_geometry.py`` scores the four output stages, and
``gate_amplitude_cv.py`` scores the input amplitude, but the two use different
observable definitions, so their numbers cannot be compared directly. This
script applies one definition -- the ``spatial_geometry`` observables -- at every
tap along the backbone, from the normalized input tensor through the stem and
maxpool to C2..C5, which makes the decay curve directly readable.

The question it answers: is a corruption's signature destroyed by depth (so a
suppression module must sit shallow), or absent from the features to begin with
(so it needs an input-domain front end)?
"""

import argparse
import copy
import csv
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
from mmdet_extension.models.utils.spatial_geometry import (  # noqa: E402
    OBSERVABLE_NAMES, compute_observables, valid_feature_extent)
from tools.analysis.gate_spatial_geometry import (  # noqa: E402
    DEFAULT_CORRUPTIONS, _auroc, _corruption_root, _domain_dataset_cfg,
    _load_config, _paired_stats, _unwrap_image, _unwrap_metas)
from tools.analysis.sogc_runtime import (  # noqa: E402
    build_dataset_checked, dump_json, load_checkpoint_sogc_compatible,
    resolve_device, seed_everything)

TAP_NAMES = ('input', 'stem', 'maxpool', 'C2', 'C3', 'C4', 'C5')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Profile corruption visibility against backbone depth')
    parser.add_argument(
        '--config', default='configs/baseline/oriented_rcnn_orthonet_rsar.py')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--corruptions', default=','.join(DEFAULT_CORRUPTIONS))
    parser.add_argument('--corruption-root', default=None)
    parser.add_argument('--max-images', type=int, default=400)
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


def _taps(backbone, image):
    """Yield ``(name, tensor)`` for every tap along the backbone."""
    yield 'input', image
    if backbone.deep_stem:
        x = backbone.stem(image)
    else:
        x = backbone.conv1(image)
        x = backbone.norm1(x)
        x = backbone.relu(x)
    yield 'stem', x
    x = backbone.maxpool(x)
    yield 'maxpool', x
    for index, layer_name in enumerate(backbone.res_layers):
        x = getattr(backbone, layer_name)(x)
        yield f'C{index + 2}', x


def _field_names():
    return [f'{tap}_{name}'
            for tap in TAP_NAMES for name in OBSERVABLE_NAMES]


@torch.no_grad()
def _collect_domain(label, backbone, dataset_cfg, args, device):
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
        row = {
            'domain': label,
            'image_index': index,
            'filename': metas.get('ori_filename', metas.get('filename', '')),
        }
        for tap, tensor in _taps(backbone, image):
            keep_h, keep_w = valid_feature_extent(
                tensor.shape[2:], metas['img_shape'][:2],
                tuple(image.shape[2:]))
            values = compute_observables(tensor[0, :, :keep_h, :keep_w])
            for name, value in values.items():
                row[f'{tap}_{name}'] = value
        rows.append(row)
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


def _profile(clean_rows, corrupt_rows):
    """Separability and paired win rate of every observable at every tap."""
    profile = OrderedDict()
    for name in OBSERVABLE_NAMES:
        per_tap = OrderedDict()
        for tap in TAP_NAMES:
            field = f'{tap}_{name}'
            auroc = _auroc(
                _finite(clean_rows, field), _finite(corrupt_rows, field))
            per_tap[tap] = {
                'auroc': auroc,
                'separability': (
                    None if auroc is None else max(auroc, 1.0 - auroc)),
                'paired': _paired_stats(
                    *_paired_arrays(clean_rows, corrupt_rows, field)),
            }
        profile[name] = per_tap
    return profile


def _peak_tap(profile):
    """Tap and observable where the corruption is most visible."""
    best = None
    for name, per_tap in profile.items():
        for tap, entry in per_tap.items():
            if entry['separability'] is None:
                continue
            if best is None or entry['separability'] > best['separability']:
                best = {
                    'observable': name,
                    'tap': tap,
                    'separability': entry['separability'],
                    'auroc': entry['auroc'],
                    'paired_win_rate': entry['paired']['win_rate'],
                }
    return best


def _retention(profile, reference_tap='input'):
    """Share of the input-domain separability that survives to each tap.

    Computed on the best observable at the reference tap, so a value well below
    one at C4 means depth destroyed a signature that was present at the input.
    """
    scored = [
        (name, per_tap[reference_tap]['separability'])
        for name, per_tap in profile.items()
        if per_tap[reference_tap]['separability'] is not None]
    if not scored:
        return None
    name, reference = max(scored, key=lambda item: item[1])
    excess = reference - 0.5
    if excess <= 1e-9:
        return {'observable': name, 'reference_separability': reference,
                'by_tap': None}
    return {
        'observable': name,
        'reference_separability': reference,
        'by_tap': OrderedDict(
            (tap, (profile[name][tap]['separability'] - 0.5) / excess)
            for tap in TAP_NAMES),
    }


def _best_per_tap(profile):
    """Best separability available at each tap, over all observables."""
    result = OrderedDict()
    for tap in TAP_NAMES:
        scored = [
            (name, per_tap[tap]['separability'])
            for name, per_tap in profile.items()
            if per_tap[tap]['separability'] is not None]
        if not scored:
            result[tap] = None
            continue
        name, separability = max(scored, key=lambda item: item[1])
        result[tap] = {'observable': name, 'separability': separability}
    return result


def _write_csv(path, rows):
    fieldnames = ['domain', 'image_index', 'filename'] + _field_names()
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_report(summary):
    print('\n=== best separability at each tap ===')
    header = ''.join(f'{tap:>10s}' for tap in TAP_NAMES)
    print(f'{"corruption":24s}{header}')
    for corruption, block in summary['per_corruption'].items():
        cells = ''.join(
            f'{entry["separability"]:>10.3f}' if entry else f'{"-":>10s}'
            for entry in block['best_per_tap'].values())
        print(f'{corruption:24s}{cells}')

    print('\n=== which observable peaks, and where ===')
    for corruption, block in summary['per_corruption'].items():
        peak = block['peak']
        if peak is None:
            print(f'{corruption:24s} (none)')
            continue
        print(f'{corruption:24s} {peak["tap"]:8s} {peak["observable"]:28s} '
              f'sep={peak["separability"]:.3f} '
              f'paired={peak["paired_win_rate"]:.3f}')

    print('\n=== retention of the input-domain signature ===')
    print(f'{"corruption":24s}{header}')
    for corruption, block in summary['per_corruption'].items():
        retention = block['retention']
        if retention is None or retention['by_tap'] is None:
            print(f'{corruption:24s} (no input-domain signal to retain)')
            continue
        cells = ''.join(
            f'{value:>10.2f}' for value in retention['by_tap'].values())
        print(f'{corruption:24s}{cells}')


def main():
    args = parse_args()
    config, checkpoint, out_dir, corruptions = _validate_args(args)
    seed_everything(42)
    device, _ = resolve_device(args.device)
    cfg = _load_config(config)
    corruption_root = _corruption_root(cfg, args.corruption_root)

    model_cfg = copy.deepcopy(cfg.model)
    model_cfg.train_cfg = None
    model = build_detector(model_cfg, test_cfg=cfg.get('test_cfg'))
    load_checkpoint_sogc_compatible(
        model, checkpoint, allow_missing_sogc=True)
    backbone = model.backbone.to(device).eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)

    clean_rows = _collect_domain(
        'clean', backbone, _domain_dataset_cfg(cfg, None, corruption_root),
        args, device)
    print(f'clean: {len(clean_rows)} images')

    blocks = OrderedDict()
    all_rows = list(clean_rows)
    for corruption in corruptions:
        rows = _collect_domain(
            corruption, backbone,
            _domain_dataset_cfg(cfg, corruption, corruption_root),
            args, device)
        all_rows.extend(rows)
        profile = _profile(clean_rows, rows)
        blocks[corruption] = {
            'sample_count': len(rows),
            'peak': _peak_tap(profile),
            'best_per_tap': _best_per_tap(profile),
            'retention': _retention(profile),
            'profile': profile,
        }
        print(f'{corruption}: {len(rows)} images')

    summary = {
        'settings': {
            'max_images': args.max_images,
            'taps': list(TAP_NAMES),
            'clean_sample_count': len(clean_rows),
        },
        'per_corruption': blocks,
    }
    _write_csv(out_dir / 'per_image_depth_profile.csv', all_rows)
    dump_json(out_dir / 'summary.json', summary)
    _print_report(summary)
    print(f'\nout_dir={out_dir}')


if __name__ == '__main__':
    main()
