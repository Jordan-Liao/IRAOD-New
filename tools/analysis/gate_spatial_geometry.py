#!/usr/bin/env python
"""Separability gate for spatially resolved backbone geometry observables.

This is the admission test that ``SOGCCalibrator`` never had to pass. It scores
each observable in ``mmdet_extension.models.utils.spatial_geometry`` on how well
it separates clean RSAR test images from each corrupted variant, using a frozen
backbone and no training whatsoever.

Two statistics are reported per observable:

``auroc``
    Unpaired, matching the deployment situation where no clean reference of the
    same scene exists. This is the number the gate threshold applies to.
``paired_win_rate``
    Share of image pairs whose corrupt value exceeds its own clean value. The
    corruptions are generated per image from the clean set, so this removes
    scene-to-scene variance and answers the weaker question of whether the
    observable responds to the interference at all.

An observable that responds in either direction is credited by the two-sided
``separability`` = ``max(auroc, 1 - auroc)``.
"""

import argparse
import copy
import csv
import math
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
from mmcv import Config  # noqa: E402
from mmdet.datasets import build_dataloader  # noqa: E402
from mmrotate.models import build_detector  # noqa: E402
from mmrotate.utils import compat_cfg  # noqa: E402

import mmdet_extension  # noqa: E402,F401
import sfod  # noqa: E402,F401
from mmdet_extension.models.utils.spatial_geometry import (  # noqa: E402
    OBSERVABLE_NAMES, compute_observables, valid_feature_extent)
from sfod.utils import patch_config  # noqa: E402
from tools.analysis.sogc_runtime import (  # noqa: E402
    build_dataset_checked, dump_json, evaluation_dataset_cfg,
    load_checkpoint_sogc_compatible, resolve_device, seed_everything)

DEFAULT_CORRUPTIONS = (
    'chaff', 'gaussian_white_noise', 'point_target', 'noise_suppression',
    'am_noise_horizontal', 'smart_suppression', 'am_noise_vertical')
STAGE_NAMES = ('C2', 'C3', 'C4', 'C5')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Gate spatial geometry observables on clean/corrupt RSAR')
    parser.add_argument(
        '--config', default='configs/baseline/oriented_rcnn_orthonet_rsar.py')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument(
        '--corruptions', default=','.join(DEFAULT_CORRUPTIONS),
        help='comma-separated corruption directory names')
    parser.add_argument(
        '--corruption-root', default=None,
        help='defaults to <data_root>/corruptions')
    parser.add_argument('--max-images', type=int, default=300)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument(
        '--gate-auroc', type=float, default=0.80,
        help='two-sided separability an observable must reach to pass')
    parser.add_argument(
        '--sparse-multiplier', type=float, default=1.0,
        help='soft-threshold multiple of the residual median')
    parser.add_argument(
        '--shrinkage', type=float, default=0.2,
        help='identity shrinkage of the Mahalanobis correlation matrix')
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
    if not 0.5 <= args.gate_auroc <= 1.0:
        raise ValueError('--gate-auroc must be in [0.5, 1]')
    if args.sparse_multiplier < 0.0:
        raise ValueError('--sparse-multiplier must be non-negative')
    if not 0.0 <= args.shrinkage <= 1.0:
        raise ValueError('--shrinkage must be in [0, 1]')
    corruptions = [
        name.strip() for name in args.corruptions.split(',') if name.strip()]
    if not corruptions:
        raise ValueError('--corruptions must name at least one corruption')
    if len(set(corruptions)) != len(corruptions):
        raise ValueError('--corruptions contains duplicates')
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = [
        str(out_dir / name) for name in ('per_image_observables.csv',
                                        'summary.json')
        if (out_dir / name).exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            'refusing to overwrite existing gate outputs (use --overwrite): '
            + ', '.join(existing))
    return config, checkpoint, out_dir, corruptions


def _load_config(config_path):
    return patch_config(compat_cfg(Config.fromfile(str(config_path))))


def _corruption_root(cfg, override):
    if override is not None:
        root = Path(override).expanduser().resolve()
    else:
        data_root = cfg.get('data_root')
        if not data_root:
            raise KeyError('config has no data_root; pass --corruption-root')
        root = Path(data_root).expanduser().resolve() / 'corruptions'
    if not root.is_dir():
        raise FileNotFoundError(f'corruption root not found: {root}')
    return root


def _domain_dataset_cfg(cfg, corruption, corruption_root):
    dataset_cfg = evaluation_dataset_cfg(cfg)
    if corruption is None:
        return dataset_cfg
    prefix = corruption_root / corruption / 'test' / 'images'
    if not prefix.is_dir():
        raise FileNotFoundError(f'corrupt images not found: {prefix}')
    dataset_cfg = copy.deepcopy(dataset_cfg)
    dataset_cfg['img_prefix'] = str(prefix) + '/'
    return dataset_cfg


def _build_backbone(cfg, checkpoint_path, device):
    model_cfg = copy.deepcopy(cfg.model)
    model_cfg.train_cfg = None
    model = build_detector(model_cfg, test_cfg=cfg.get('test_cfg'))
    load_checkpoint_sogc_compatible(
        model, checkpoint_path, allow_missing_sogc=True)
    backbone = getattr(model, 'backbone', None)
    if backbone is None:
        raise TypeError('configured model does not expose a backbone')
    backbone = backbone.to(device).eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    return backbone


def _unwrap_singleton(value, what):
    """Peel the DataContainer/list nesting of a batch-size-one test sample.

    mmcv alternates the two wrappers (a DataContainer whose payload is a list of
    DataContainers), so the two cases have to be tried in turn rather than in
    sequence.
    """
    for _ in range(8):
        # Tensors also expose ``.data``, so they have to terminate the peel.
        if isinstance(value, (torch.Tensor, dict)):
            return value
        if hasattr(value, 'data'):
            value = value.data
            continue
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise RuntimeError(
                    f'gate requires a single {what} per batch without '
                    f'test-time augmentation; got {len(value)} entries')
            value = value[0]
            continue
        return value
    raise RuntimeError(f'could not unwrap {what} from the data batch')


def _unwrap_metas(data):
    metas = _unwrap_singleton(data['img_metas'], 'img_metas')
    if not isinstance(metas, dict):
        raise TypeError(f'unexpected img_metas payload: {type(metas)}')
    return metas


def _unwrap_image(data):
    image = _unwrap_singleton(data['img'], 'image')
    if not isinstance(image, torch.Tensor):
        raise TypeError(f'unexpected image payload: {type(image)}')
    if image.ndim != 4 or image.shape[0] != 1:
        raise RuntimeError(
            f'gate requires a [1, 3, H, W] image; got {tuple(image.shape)}')
    return image


@torch.no_grad()
def _collect_domain(label, backbone, dataset_cfg, args, device):
    dataset = build_dataset_checked(dataset_cfg)
    loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=2,
        num_gpus=1,
        dist=False,
        shuffle=False)
    rows = []
    for index, data in enumerate(loader):
        if index >= args.max_images:
            break
        metas = _unwrap_metas(data)
        image = _unwrap_image(data).to(device)
        features = backbone(image)
        if len(features) != len(STAGE_NAMES):
            raise RuntimeError(
                f'expected {len(STAGE_NAMES)} backbone outputs; '
                f'got {len(features)}')
        row = {
            'domain': label,
            'image_index': index,
            'filename': metas.get('ori_filename', metas.get('filename', '')),
        }
        for stage, feature in zip(STAGE_NAMES, features):
            if feature.shape[0] != 1:
                raise RuntimeError('gate requires batch size one')
            keep_h, keep_w = valid_feature_extent(
                feature.shape[2:], metas['img_shape'][:2],
                tuple(image.shape[2:]))
            cropped = feature[0, :, :keep_h, :keep_w]
            values = compute_observables(
                cropped, sparse_multiplier=args.sparse_multiplier)
            for name, value in values.items():
                row[f'{stage}_{name}'] = value
        rows.append(row)
    if not rows:
        raise RuntimeError(f'domain {label} produced no rows')
    return rows


def _rankdata(values):
    order = np.argsort(values, kind='mergesort')
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _auroc(clean_values, corrupt_values):
    """Probability that a corrupt sample scores above a clean one."""
    if clean_values.size == 0 or corrupt_values.size == 0:
        return None
    combined = np.concatenate((clean_values, corrupt_values))
    ranks = _rankdata(combined)
    n_clean = clean_values.size
    n_corrupt = corrupt_values.size
    rank_sum = ranks[n_clean:].sum()
    u_value = rank_sum - n_corrupt * (n_corrupt + 1) / 2.0
    return float(u_value / (n_clean * n_corrupt))


def _paired_stats(clean_values, corrupt_values):
    """Win rate and median relative shift over per-image pairs."""
    if clean_values.size == 0:
        return {'count': 0, 'win_rate': None, 'median_relative_shift': None}
    difference = corrupt_values - clean_values
    wins = float((difference > 0).mean())
    scale = np.maximum(np.abs(clean_values), 1e-12)
    shift = float(np.median(difference / scale))
    return {
        'count': int(clean_values.size),
        'win_rate': wins,
        'median_relative_shift': shift if math.isfinite(shift) else None,
    }


def _paired_arrays(clean_rows, corrupt_rows, field):
    """Align two domains by filename and drop pairs with non-finite values."""
    clean_by_name = {row['filename']: row for row in clean_rows}
    clean_values = []
    corrupt_values = []
    for row in corrupt_rows:
        match = clean_by_name.get(row['filename'])
        if match is None:
            continue
        clean_value = match[field]
        corrupt_value = row[field]
        if not (math.isfinite(clean_value) and math.isfinite(corrupt_value)):
            continue
        clean_values.append(clean_value)
        corrupt_values.append(corrupt_value)
    return (np.asarray(clean_values, dtype=np.float64),
            np.asarray(corrupt_values, dtype=np.float64))


def _finite(rows, field):
    values = np.asarray([row[field] for row in rows], dtype=np.float64)
    return values[np.isfinite(values)]


def _field_names():
    return [f'{stage}_{name}'
            for stage in STAGE_NAMES for name in OBSERVABLE_NAMES]


def _matrix(rows, fields):
    return np.asarray(
        [[row[field] for field in fields] for row in rows], dtype=np.float64)


def _usable_fields(fit_matrix, fields, min_std=1e-9):
    """Drop observables that are non-finite or degenerate on the fit split."""
    finite = np.isfinite(fit_matrix).all(axis=0)
    varying = fit_matrix.std(axis=0, ddof=0) > min_std
    keep = finite & varying
    return [field for field, flag in zip(fields, keep) if flag], keep


def _mahalanobis_scorer(fit_matrix, shrinkage):
    """Source-anchored Mahalanobis scorer with Ledoit-Wolf style shrinkage.

    The correlation form is used (standardize, then whiten) so observables on
    wildly different scales contribute comparably, and the shrinkage keeps the
    covariance invertible when the fit split is smaller than a comfortable
    multiple of the dimension.
    """
    mean = fit_matrix.mean(axis=0)
    std = fit_matrix.std(axis=0, ddof=0)
    std = np.where(std > 1e-12, std, 1.0)
    standardized = (fit_matrix - mean) / std
    dimension = standardized.shape[1]
    correlation = np.cov(standardized, rowvar=False, ddof=1)
    correlation = np.atleast_2d(correlation)
    regularized = ((1.0 - shrinkage) * correlation
                   + shrinkage * np.eye(dimension))
    precision = np.linalg.pinv(regularized)

    def score(matrix):
        centered = (matrix - mean) / std
        return np.einsum('ij,jk,ik->i', centered, precision, centered)

    return score


def _multivariate_scores(clean_rows, corrupt_rows, args):
    """AUROC of a source-anchored Mahalanobis score, per stage and pooled.

    The clean set is split in half: the first half fits the source statistics
    and the second half is the clean evaluation set, so no clean sample is ever
    both fitted and scored.
    """
    split = len(clean_rows) // 2
    if split < 8:
        return {}
    fit_rows = clean_rows[:split]
    eval_rows = clean_rows[split:]
    groups = OrderedDict(
        (stage, [f'{stage}_{name}' for name in OBSERVABLE_NAMES])
        for stage in STAGE_NAMES)
    groups['all_stages'] = _field_names()

    results = OrderedDict()
    for group, fields in groups.items():
        fit_matrix = _matrix(fit_rows, fields)
        kept_fields, keep = _usable_fields(fit_matrix, fields)
        if len(kept_fields) < 2:
            continue
        scorer = _mahalanobis_scorer(fit_matrix[:, keep], args.shrinkage)
        clean_eval = _matrix(eval_rows, kept_fields)
        corrupt_eval = _matrix(corrupt_rows, kept_fields)
        valid_clean = np.isfinite(clean_eval).all(axis=1)
        valid_corrupt = np.isfinite(corrupt_eval).all(axis=1)
        if valid_clean.sum() < 4 or valid_corrupt.sum() < 4:
            continue
        clean_scores = scorer(clean_eval[valid_clean])
        corrupt_scores = scorer(corrupt_eval[valid_corrupt])
        auroc = _auroc(clean_scores, corrupt_scores)
        results[group] = {
            'auroc': auroc,
            'separability': None if auroc is None else max(auroc, 1.0 - auroc),
            'passes_gate': (
                None if auroc is None
                else bool(max(auroc, 1.0 - auroc) >= args.gate_auroc)),
            'dimension': len(kept_fields),
            'fit_samples': int(len(fit_rows)),
            'clean_eval_samples': int(valid_clean.sum()),
            'corrupt_eval_samples': int(valid_corrupt.sum()),
        }
    return results


def _score_corruption(clean_rows, corrupt_rows, gate_auroc):
    scores = OrderedDict()
    for field in _field_names():
        auroc = _auroc(_finite(clean_rows, field), _finite(corrupt_rows, field))
        paired = _paired_stats(*_paired_arrays(clean_rows, corrupt_rows, field))
        separability = None if auroc is None else max(auroc, 1.0 - auroc)
        scores[field] = {
            'auroc': auroc,
            'separability': separability,
            'passes_gate': (
                None if separability is None
                else bool(separability >= gate_auroc)),
            'paired': paired,
        }
    return scores


def _best_observable(scores):
    ranked = [
        (field, entry['separability'])
        for field, entry in scores.items()
        if entry['separability'] is not None]
    if not ranked:
        return None
    field, separability = max(ranked, key=lambda item: item[1])
    return {
        'field': field,
        'separability': separability,
        'auroc': scores[field]['auroc'],
        'paired_win_rate': scores[field]['paired']['win_rate'],
    }


def _make_summary(clean_rows, per_corruption, args):
    corruption_blocks = OrderedDict()
    for corruption, rows in per_corruption.items():
        scores = _score_corruption(clean_rows, rows, args.gate_auroc)
        best = _best_observable(scores)
        multivariate = _multivariate_scores(clean_rows, rows, args)
        best_multivariate = None
        if multivariate:
            group, entry = max(
                multivariate.items(),
                key=lambda item: item[1]['separability'] or 0.0)
            best_multivariate = {'group': group, **entry}
        univariate_passes = bool(
            best is not None and best['separability'] >= args.gate_auroc)
        multivariate_passes = bool(
            best_multivariate is not None
            and best_multivariate['separability'] >= args.gate_auroc)
        corruption_blocks[corruption] = {
            'sample_count': len(rows),
            'best_observable': best,
            'best_multivariate': best_multivariate,
            'passes_gate_univariate': univariate_passes,
            'passes_gate': univariate_passes or multivariate_passes,
            'observables': scores,
            'multivariate': multivariate,
        }

    # Worst-case view: an observable is only useful for the backbone if it
    # holds up across corruptions, not just on the one it was tuned for.
    per_field = OrderedDict()
    for field in _field_names():
        separabilities = [
            block['observables'][field]['separability']
            for block in corruption_blocks.values()
            if block['observables'][field]['separability'] is not None]
        win_rates = [
            block['observables'][field]['paired']['win_rate']
            for block in corruption_blocks.values()
            if block['observables'][field]['paired']['win_rate'] is not None]
        if not separabilities:
            continue
        per_field[field] = {
            'min_separability': float(np.min(separabilities)),
            'mean_separability': float(np.mean(separabilities)),
            'max_separability': float(np.max(separabilities)),
            'corruptions_passing': int(sum(
                value >= args.gate_auroc for value in separabilities)),
            'mean_paired_win_rate': (
                float(np.mean(win_rates)) if win_rates else None),
        }

    ranked_fields = sorted(
        per_field.items(),
        key=lambda item: item[1]['mean_separability'],
        reverse=True)
    return {
        'settings': {
            'gate_auroc': args.gate_auroc,
            'max_images': args.max_images,
            'sparse_multiplier': args.sparse_multiplier,
            'shrinkage': args.shrinkage,
            'clean_sample_count': len(clean_rows),
        },
        'verdict': {
            'corruptions_passing': sorted(
                name for name, block in corruption_blocks.items()
                if block['passes_gate']),
            'corruptions_failing': sorted(
                name for name, block in corruption_blocks.items()
                if not block['passes_gate']),
            'corruptions_passing_univariate_only': sorted(
                name for name, block in corruption_blocks.items()
                if block['passes_gate_univariate']),
            'top_observables_by_mean_separability': [
                {'field': field, **stats}
                for field, stats in ranked_fields[:10]],
        },
        'per_corruption': corruption_blocks,
        'per_observable': per_field,
    }


def _write_csv(path, rows):
    fieldnames = ['domain', 'image_index', 'filename'] + _field_names()
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_report(summary):
    verdict = summary['verdict']
    print('\n=== gate verdict ===')
    print('passing corruptions: '
          + (', '.join(verdict['corruptions_passing']) or '(none)'))
    print('failing corruptions: '
          + (', '.join(verdict['corruptions_failing']) or '(none)'))
    print('\n=== best observable per corruption ===')
    for corruption, block in summary['per_corruption'].items():
        best = block['best_observable']
        if best is None:
            print(f'{corruption:24s} (no finite observable)')
            continue
        print(f'{corruption:24s} {best["field"]:34s} '
              f'sep={best["separability"]:.3f} '
              f'auroc={best["auroc"]:.3f} '
              f'paired={best["paired_win_rate"]:.3f}')
    print('\n=== source-anchored Mahalanobis score ===')
    for corruption, block in summary['per_corruption'].items():
        best = block['best_multivariate']
        if best is None:
            print(f'{corruption:24s} (unavailable)')
            continue
        print(f'{corruption:24s} {best["group"]:12s} '
              f'sep={best["separability"]:.3f} '
              f'auroc={best["auroc"]:.3f} dim={best["dimension"]}')
    print('\n=== most consistent observables ===')
    for entry in verdict['top_observables_by_mean_separability']:
        print(f'{entry["field"]:34s} mean={entry["mean_separability"]:.3f} '
              f'min={entry["min_separability"]:.3f} '
              f'pass={entry["corruptions_passing"]}'
              f'/{len(summary["per_corruption"])}')


def main():
    args = parse_args()
    config, checkpoint, out_dir, corruptions = _validate_args(args)
    seed_everything(42)
    device, _ = resolve_device(args.device)
    cfg = _load_config(config)
    corruption_root = _corruption_root(cfg, args.corruption_root)
    backbone = _build_backbone(cfg, checkpoint, device)

    clean_rows = _collect_domain(
        'clean', backbone, _domain_dataset_cfg(cfg, None, corruption_root),
        args, device)
    print(f'clean: {len(clean_rows)} images')

    per_corruption = OrderedDict()
    for corruption in corruptions:
        rows = _collect_domain(
            corruption, backbone,
            _domain_dataset_cfg(cfg, corruption, corruption_root),
            args, device)
        per_corruption[corruption] = rows
        print(f'{corruption}: {len(rows)} images')

    all_rows = clean_rows + [
        row for rows in per_corruption.values() for row in rows]
    _write_csv(out_dir / 'per_image_observables.csv', all_rows)
    summary = _make_summary(clean_rows, per_corruption, args)
    dump_json(out_dir / 'summary.json', summary)
    _print_report(summary)
    print(f'\nout_dir={out_dir}')


if __name__ == '__main__':
    main()
