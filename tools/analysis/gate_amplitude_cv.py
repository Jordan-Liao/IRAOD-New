#!/usr/bin/env python
"""Separability gate for amplitude-domain CV observables.

Companion to ``gate_spatial_geometry.py`` for the corruptions that a low-rank
separable model cannot see by construction: ``gaussian_white_noise``,
``noise_suppression`` and ``point_target`` are full-rank additive or sub-pixel.
No network forward pass is involved — this reads images only.
"""

import argparse
import csv
import sys
from collections import OrderedDict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from tools.analysis.amplitude_cv import (  # noqa: E402
    OBSERVABLE_NAMES, RAYLEIGH_AMPLITUDE_CV, compute_observables)
from tools.analysis.gate_spatial_geometry import (  # noqa: E402
    DEFAULT_CORRUPTIONS, _auroc, _paired_stats)

SUPPORTED_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Gate amplitude-domain CV observables on clean/corrupt RSAR')
    parser.add_argument('--data-root', default='/myfile/dataset/RSAR/')
    parser.add_argument('--split', default='test')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--corruptions', default=','.join(DEFAULT_CORRUPTIONS))
    parser.add_argument('--max-images', type=int, default=800)
    parser.add_argument('--window', type=int, default=5)
    parser.add_argument('--gate-auroc', type=float, default=0.80)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def _validate_args(args):
    data_root = Path(args.data_root).expanduser().resolve()
    clean_dir = data_root / args.split / 'images'
    if not clean_dir.is_dir():
        raise FileNotFoundError(f'clean images not found: {clean_dir}')
    if args.max_images <= 0:
        raise ValueError('--max-images must be positive')
    if args.window < 1 or args.window % 2 == 0:
        raise ValueError('--window must be a positive odd integer')
    if not 0.5 <= args.gate_auroc <= 1.0:
        raise ValueError('--gate-auroc must be in [0.5, 1]')
    corruptions = [
        name.strip() for name in args.corruptions.split(',') if name.strip()]
    if not corruptions:
        raise ValueError('--corruptions must name at least one corruption')
    if len(set(corruptions)) != len(corruptions):
        raise ValueError('--corruptions contains duplicates')
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = [
        str(out_dir / name)
        for name in ('per_image_amplitude_cv.csv', 'summary.json')
        if (out_dir / name).exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            'refusing to overwrite existing outputs (use --overwrite): '
            + ', '.join(existing))
    return data_root, clean_dir, out_dir, corruptions


def _listing(directory, limit):
    names = sorted(
        path.name for path in directory.iterdir()
        if path.suffix.lower() in SUPPORTED_EXTS)
    if not names:
        raise RuntimeError(f'no images found in {directory}')
    return names[:limit]


def _read_amplitude(path):
    with Image.open(path) as handle:
        image = np.asarray(handle.convert('L'), dtype=np.float64)
    if image.ndim != 2 or min(image.shape) < 2:
        raise ValueError(f'unusable image {path}: shape {image.shape}')
    return image


def _collect_domain(label, directory, names, args):
    rows = []
    for index, name in enumerate(names):
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f'{label} is missing {name}')
        values = compute_observables(
            _read_amplitude(path), window=args.window)
        rows.append({
            'domain': label, 'image_index': index, 'filename': name, **values})
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


def _score(clean_rows, corrupt_rows, gate_auroc):
    scores = OrderedDict()
    for field in OBSERVABLE_NAMES:
        auroc = _auroc(_finite(clean_rows, field), _finite(corrupt_rows, field))
        separability = None if auroc is None else max(auroc, 1.0 - auroc)
        scores[field] = {
            'auroc': auroc,
            'separability': separability,
            'passes_gate': (
                None if separability is None
                else bool(separability >= gate_auroc)),
            'paired': _paired_stats(
                *_paired_arrays(clean_rows, corrupt_rows, field)),
        }
    return scores


def _make_summary(clean_rows, per_corruption, args):
    blocks = OrderedDict()
    for corruption, rows in per_corruption.items():
        scores = _score(clean_rows, rows, args.gate_auroc)
        ranked = [
            (field, entry['separability']) for field, entry in scores.items()
            if entry['separability'] is not None]
        best = None
        if ranked:
            field, separability = max(ranked, key=lambda item: item[1])
            best = {
                'field': field,
                'separability': separability,
                'auroc': scores[field]['auroc'],
                'paired_win_rate': scores[field]['paired']['win_rate'],
            }
        blocks[corruption] = {
            'sample_count': len(rows),
            'best_observable': best,
            'passes_gate': bool(
                best is not None and best['separability'] >= args.gate_auroc),
            'observables': scores,
        }
    clean_cv = _finite(clean_rows, 'cv_median')
    return {
        'settings': {
            'window': args.window,
            'gate_auroc': args.gate_auroc,
            'max_images': args.max_images,
            'clean_sample_count': len(clean_rows),
        },
        'clean_reference': {
            # The empirical anchor to use in place of the theoretical value,
            # since RSAR is 8-bit and JPEG-compressed.
            'cv_median_median': (
                float(np.median(clean_cv)) if clean_cv.size else None),
            'cv_median_mean': (
                float(clean_cv.mean()) if clean_cv.size else None),
            'rayleigh_amplitude_cv': RAYLEIGH_AMPLITUDE_CV,
        },
        'verdict': {
            'corruptions_passing': sorted(
                name for name, block in blocks.items()
                if block['passes_gate']),
            'corruptions_failing': sorted(
                name for name, block in blocks.items()
                if not block['passes_gate']),
        },
        'per_corruption': blocks,
    }


def _write_csv(path, rows):
    fieldnames = ['domain', 'image_index', 'filename'] + list(OBSERVABLE_NAMES)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_report(summary):
    reference = summary['clean_reference']
    print('\n=== clean reference ===')
    print(f'source cv_median (median over images): '
          f'{reference["cv_median_median"]:.4f}')
    print(f'Rayleigh amplitude CV (theory):        '
          f'{reference["rayleigh_amplitude_cv"]:.4f}')
    print('\n=== gate verdict ===')
    print('passing: '
          + (', '.join(summary['verdict']['corruptions_passing']) or '(none)'))
    print('failing: '
          + (', '.join(summary['verdict']['corruptions_failing']) or '(none)'))
    print('\n=== best observable per corruption ===')
    for corruption, block in summary['per_corruption'].items():
        best = block['best_observable']
        if best is None:
            print(f'{corruption:24s} (no finite observable)')
            continue
        print(f'{corruption:24s} {best["field"]:26s} '
              f'sep={best["separability"]:.3f} '
              f'auroc={best["auroc"]:.3f} '
              f'paired={best["paired_win_rate"]:.3f}')
    print('\n=== full matrix (separability) ===')
    header = ''.join(f'{name[:11]:>13s}' for name in OBSERVABLE_NAMES)
    print(f'{"corruption":24s}{header}')
    for corruption, block in summary['per_corruption'].items():
        cells = ''.join(
            f'{block["observables"][name]["separability"]:>13.3f}'
            for name in OBSERVABLE_NAMES)
        print(f'{corruption:24s}{cells}')


def main():
    args = parse_args()
    data_root, clean_dir, out_dir, corruptions = _validate_args(args)
    names = _listing(clean_dir, args.max_images)
    clean_rows = _collect_domain('clean', clean_dir, names, args)
    print(f'clean: {len(clean_rows)} images')

    per_corruption = OrderedDict()
    for corruption in corruptions:
        directory = data_root / 'corruptions' / corruption / args.split / 'images'
        if not directory.is_dir():
            raise FileNotFoundError(f'corrupt images not found: {directory}')
        # Same filenames as the clean split, so the pairing is exact.
        per_corruption[corruption] = _collect_domain(
            corruption, directory, names, args)
        print(f'{corruption}: {len(per_corruption[corruption])} images')

    summary = _make_summary(clean_rows, per_corruption, args)
    all_rows = clean_rows + [
        row for rows in per_corruption.values() for row in rows]
    _write_csv(out_dir / 'per_image_amplitude_cv.csv', all_rows)
    from tools.analysis.sogc_runtime import dump_json
    dump_json(out_dir / 'summary.json', summary)
    _print_report(summary)
    print(f'\nout_dir={out_dir}')


if __name__ == '__main__':
    main()
