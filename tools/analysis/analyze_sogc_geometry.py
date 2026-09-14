#!/usr/bin/env python
"""Compare clean/corrupt SOGC geometry before adaptation training."""

import argparse
import csv
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from iraod_runtime import ensure_iraod_runtime  # noqa: E402

ensure_iraod_runtime()

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from mmcv import Config  # noqa: E402
from mmdet.datasets import build_dataloader  # noqa: E402
from mmrotate.core.bbox import rbbox_overlaps  # noqa: E402
from mmrotate.models import build_detector  # noqa: E402
from mmrotate.utils import compat_cfg  # noqa: E402

import mmdet_extension  # noqa: E402,F401
import sfod  # noqa: E402,F401
from sfod.utils import patch_config  # noqa: E402
from tools.analysis.sogc_runtime import (  # noqa: E402
    build_dataset_checked, detector_backbone, dump_json,
    evaluation_dataset_cfg, load_checkpoint_sogc_compatible,
    make_detector_runner, resolve_device, seed_everything)


CSV_FIELDS = [
    'domain', 'image_index', 'filename',
    'C4_mean_sq_z', 'C5_mean_sq_z',
    'C4_mean_abs_z', 'C5_mean_abs_z',
    'C4_z_clip_fraction', 'C5_z_clip_fraction',
    'tp', 'fp', 'fn', 'gt_recall', 'high_conf_detections',
]

GEOMETRY_METRICS = ('mean_sq_z', 'mean_abs_z', 'z_clip_fraction')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyze clean/corrupt SOGC response geometry')
    parser.add_argument('--clean-config', required=True)
    parser.add_argument('--corrupt-config', required=True)
    parser.add_argument('--checkpoint-with-stats', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--max-images', type=int, default=None)
    parser.add_argument('--score-thr', type=float, default=0.7)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument(
        '--corruption', default=None,
        help='override the corrupt config ${corrupt} value')
    return parser.parse_args()


def _validate_args(args):
    clean = Path(args.clean_config).expanduser().resolve()
    corrupt = Path(args.corrupt_config).expanduser().resolve()
    checkpoint = Path(args.checkpoint_with_stats).expanduser().resolve()
    for label, path in (
            ('clean config', clean), ('corrupt config', corrupt),
            ('checkpoint', checkpoint)):
        if not path.is_file():
            raise FileNotFoundError(f'{label} not found: {path}')
    if args.max_images is not None and args.max_images <= 0:
        raise ValueError('--max-images must be positive')
    if not 0.0 <= args.score_thr <= 1.0:
        raise ValueError('--score-thr must be in [0, 1]')
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = [out_dir / name for name in (
        'per_image_metrics.csv', 'summary.json',
        'clean_corrupt_geometry_boxplot.png', 'geometry_vs_fp.png',
        'geometry_vs_recall.png')]
    existing = [str(path) for path in expected if path.exists()]
    if existing:
        raise FileExistsError(
            'refusing to overwrite existing analysis outputs: ' +
            ', '.join(existing))
    return clean, corrupt, checkpoint, out_dir


def _diag_scalar(diagnostics, stage, metric):
    if stage not in diagnostics or metric not in diagnostics[stage]:
        raise RuntimeError(f'missing SOGC diagnostic {stage}.{metric}')
    value = diagnostics[stage][metric]
    if value.numel() != 1:
        raise RuntimeError(
            'geometry analysis requires samples_per_gpu=1; got diagnostic '
            f'shape {tuple(value.shape)}')
    scalar = float(value.item())
    if not math.isfinite(scalar):
        raise FloatingPointError(
            f'non-finite SOGC diagnostic {stage}.{metric}: {scalar}')
    return scalar


def _flatten_detections(result, score_thr):
    if isinstance(result, tuple):
        result = result[0]
    if not isinstance(result, list):
        raise TypeError('detector result must be a list of per-class arrays')
    selected = []
    for class_index, class_result in enumerate(result):
        array = np.asarray(class_result)
        if array.size == 0:
            continue
        if array.ndim != 2 or array.shape[1] < 6:
            raise ValueError(
                f'invalid rotated detection shape: {array.shape}')
        array = array[array[:, -1] >= score_thr]
        for row in array:
            selected.append((class_index, row[:5], float(row[-1])))
    selected.sort(key=lambda item: item[2], reverse=True)
    return selected


def _match_detections(result, annotation, score_thr, device):
    gt_bboxes = annotation.get('bboxes', [])
    gt_labels = annotation.get('labels', [])
    # DOTA's annotation-free test branch stores Python lists; validation
    # annotations, including genuinely empty images, use ndarrays.
    if not isinstance(gt_bboxes, np.ndarray):
        return None
    gt_bboxes = np.asarray(gt_bboxes, dtype=np.float32).reshape(-1, 5)
    gt_labels = np.asarray(gt_labels, dtype=np.int64).reshape(-1)
    detections = _flatten_detections(result, score_thr)

    true_positive = 0
    for class_index in np.unique(np.concatenate((
            gt_labels,
            np.asarray([item[0] for item in detections], dtype=np.int64)))):
        class_gt_indices = np.flatnonzero(gt_labels == class_index)
        class_detections = [
            item for item in detections if item[0] == class_index]
        if not class_detections or class_gt_indices.size == 0:
            continue
        det_tensor = torch.as_tensor(
            np.stack([item[1] for item in class_detections]),
            dtype=torch.float32,
            device=device)
        gt_tensor = torch.as_tensor(
            gt_bboxes[class_gt_indices], dtype=torch.float32, device=device)
        overlaps = rbbox_overlaps(det_tensor, gt_tensor).detach().cpu().numpy()
        matched = set()
        for detection_index in range(len(class_detections)):
            order = np.argsort(overlaps[detection_index])[::-1]
            for gt_index in order:
                if overlaps[detection_index, gt_index] < 0.5:
                    break
                if int(gt_index) not in matched:
                    matched.add(int(gt_index))
                    true_positive += 1
                    break

    detection_count = len(detections)
    gt_count = len(gt_bboxes)
    false_positive = detection_count - true_positive
    false_negative = gt_count - true_positive
    recall = true_positive / gt_count if gt_count else float('nan')
    return {
        'tp': true_positive,
        'fp': false_positive,
        'fn': false_negative,
        'gt_recall': recall,
        'high_conf_detections': detection_count,
    }


def _load_config(path, corruption=None):
    cfg = Config.fromfile(str(path))
    if corruption is not None:
        cfg.corrupt = corruption
    return patch_config(compat_cfg(cfg))


def _analyze_domain(label, config_path, checkpoint_path, args,
                    device, device_ids):
    corruption = args.corruption if label == 'corrupt' else None
    cfg = _load_config(config_path, corruption=corruption)
    dataset = build_dataset_checked(evaluation_dataset_cfg(cfg))
    loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        num_gpus=1,
        dist=False,
        shuffle=False)
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint_sogc_compatible(
        model, checkpoint_path, allow_missing_sogc=False)
    backbone = detector_backbone(model)
    if not backbone.has_valid_source_stats():
        raise RuntimeError(
            f'{label} model does not have valid finalized source statistics')
    backbone.set_sogc_mode('calibrate')
    model.eval()
    runner = make_detector_runner(model, device, device_ids)

    rows = []
    with torch.no_grad():
        for index, data in enumerate(loader):
            if args.max_images is not None and index >= args.max_images:
                break
            batch_results = runner(return_loss=False, rescale=True, **data)
            if len(batch_results) != 1:
                raise RuntimeError('analysis requires one result per batch')
            diagnostics = backbone.get_sogc_diagnostics()
            row = {
                'domain': label,
                'image_index': index,
                'filename': dataset.data_infos[index].get('filename', ''),
                'C4_mean_sq_z': _diag_scalar(
                    diagnostics, 'C4', 'mean_sq_z'),
                'C5_mean_sq_z': _diag_scalar(
                    diagnostics, 'C5', 'mean_sq_z'),
                'C4_mean_abs_z': _diag_scalar(
                    diagnostics, 'C4', 'mean_abs_z'),
                'C5_mean_abs_z': _diag_scalar(
                    diagnostics, 'C5', 'mean_abs_z'),
                'C4_z_clip_fraction': _diag_scalar(
                    diagnostics, 'C4', 'z_clip_fraction'),
                'C5_z_clip_fraction': _diag_scalar(
                    diagnostics, 'C5', 'z_clip_fraction'),
                'tp': None,
                'fp': None,
                'fn': None,
                'gt_recall': None,
                'high_conf_detections': None,
            }
            annotation = dataset.get_ann_info(index)
            box_metrics = _match_detections(
                batch_results[0], annotation, args.score_thr, device)
            if box_metrics is not None:
                row.update(box_metrics)
            rows.append(row)
    return rows


def _finite_values(rows, field):
    values = np.asarray([
        row[field] for row in rows if row[field] is not None
    ], dtype=np.float64)
    return values[np.isfinite(values)]


def _describe(values):
    if values.size == 0:
        return {'count': 0, 'mean': None, 'median': None, 'std': None}
    return {
        'count': int(values.size),
        'mean': float(values.mean()),
        'median': float(np.median(values)),
        'std': float(values.std(ddof=0)),
    }


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
    clean_values = clean_values[np.isfinite(clean_values)]
    corrupt_values = corrupt_values[np.isfinite(corrupt_values)]
    if clean_values.size == 0 or corrupt_values.size == 0:
        return None
    combined = np.concatenate((clean_values, corrupt_values))
    ranks = _rankdata(combined)
    n_clean = clean_values.size
    n_corrupt = corrupt_values.size
    rank_sum = ranks[n_clean:].sum()
    u_value = rank_sum - n_corrupt * (n_corrupt + 1) / 2.0
    return float(u_value / (n_clean * n_corrupt))


def _spearman(x_values, y_values):
    x_values = np.asarray(x_values, dtype=np.float64)
    y_values = np.asarray(y_values, dtype=np.float64)
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[valid]
    y_values = y_values[valid]
    if x_values.size < 2:
        return {'rho': None, 'pvalue': None, 'count': int(x_values.size)}
    try:
        from scipy.stats import spearmanr
        result = spearmanr(x_values, y_values)
        rho = float(result.statistic)
        pvalue = float(result.pvalue)
    except ImportError:
        rho = float(np.corrcoef(
            _rankdata(x_values), _rankdata(y_values))[0, 1])
        pvalue = None
    if not math.isfinite(rho):
        rho = None
    if pvalue is not None and not math.isfinite(pvalue):
        pvalue = None
    return {'rho': rho, 'pvalue': pvalue, 'count': int(x_values.size)}


def _write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _plot_boxplot(path, clean_rows, corrupt_rows):
    fields = [
        ('C4_mean_sq_z', 'C4 mean_sq_z'),
        ('C5_mean_sq_z', 'C5 mean_sq_z'),
        ('C4_mean_abs_z', 'C4 mean_abs_z'),
        ('C5_mean_abs_z', 'C5 mean_abs_z'),
        ('C4_z_clip_fraction', 'C4 z_clip_fraction'),
        ('C5_z_clip_fraction', 'C5 z_clip_fraction'),
    ]
    figure, axes = plt.subplots(3, 2, figsize=(10, 10))
    for axis, (field, title) in zip(axes.flat, fields):
        axis.boxplot([
            _finite_values(clean_rows, field),
            _finite_values(corrupt_rows, field),
        ], showfliers=False)
        axis.set_xticklabels(['clean', 'corrupt'])
        axis.set_title(title)
        axis.set_ylabel('standardized deviation')
        axis.grid(axis='y', alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_relation(path, rows, target_field, target_label):
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    plotted = False
    for axis, stage in zip(axes, ('C4', 'C5')):
        for domain, color in (('clean', '#287271'), ('corrupt', '#c8553d')):
            selected = [row for row in rows if row['domain'] == domain]
            x_values = np.asarray(
                [row[f'{stage}_mean_sq_z'] for row in selected],
                dtype=np.float64)
            y_values = np.asarray([
                np.nan if row[target_field] is None else row[target_field]
                for row in selected
            ], dtype=np.float64)
            valid = np.isfinite(x_values) & np.isfinite(y_values)
            if valid.any():
                axis.scatter(x_values[valid], y_values[valid], s=18,
                             alpha=0.65, label=domain, color=color)
                plotted = True
        axis.set_title(stage)
        axis.set_xlabel('mean_sq_z')
        axis.set_ylabel(target_label)
        axis.grid(alpha=0.25)
        axis.legend(loc='best')
    if not plotted:
        figure.text(0.5, 0.5, 'GT metrics unavailable',
                    ha='center', va='center')
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _domain_correlations(rows):
    """Spearman correlations between deviation and detection failure."""
    correlations = {}
    for stage in ('C4', 'C5'):
        x_values = [row[f'{stage}_mean_sq_z'] for row in rows]
        correlations[stage] = {
            'mean_sq_z_vs_fp_per_image': _spearman(
                x_values,
                [np.nan if row['fp'] is None else row['fp']
                 for row in rows]),
            'mean_sq_z_vs_gt_recall': _spearman(
                x_values,
                [np.nan if row['gt_recall'] is None else row['gt_recall']
                 for row in rows]),
        }
    return correlations


def _make_summary(clean_rows, corrupt_rows):
    all_rows = clean_rows + corrupt_rows
    geometry = {}
    for domain, rows in (('clean', clean_rows), ('corrupt', corrupt_rows)):
        geometry[domain] = {}
        for stage in ('C4', 'C5'):
            geometry[domain][stage] = {
                metric: _describe(_finite_values(
                    rows, f'{stage}_{metric}'))
                for metric in GEOMETRY_METRICS
            }

    auroc = {}
    for stage in ('C4', 'C5'):
        for metric in GEOMETRY_METRICS:
            field = f'{stage}_{metric}'
            auroc[field] = _auroc(
                _finite_values(clean_rows, field),
                _finite_values(corrupt_rows, field))

    # Pooling both domains conflates "corrupt images deviate more" with
    # "deviation predicts failure", so the within-domain splits are reported
    # separately and the corrupt-only block is the one that matters.
    correlations = {
        'combined': _domain_correlations(all_rows),
        'clean': _domain_correlations(clean_rows),
        'corrupt': _domain_correlations(corrupt_rows),
    }
    return {
        'sample_counts': {
            'clean': len(clean_rows),
            'corrupt': len(corrupt_rows),
            'total': len(all_rows),
        },
        'geometry': geometry,
        'clean_vs_corrupt_auroc': auroc,
        'correlations': correlations,
    }


def main():
    args = parse_args()
    clean, corrupt, checkpoint, out_dir = _validate_args(args)
    seed_everything(42)
    device, device_ids = resolve_device(args.device)
    clean_rows = _analyze_domain(
        'clean', clean, checkpoint, args, device, device_ids)
    corrupt_rows = _analyze_domain(
        'corrupt', corrupt, checkpoint, args, device, device_ids)
    rows = clean_rows + corrupt_rows

    _write_csv(out_dir / 'per_image_metrics.csv', rows)
    dump_json(out_dir / 'summary.json', _make_summary(
        clean_rows, corrupt_rows))
    _plot_boxplot(
        out_dir / 'clean_corrupt_geometry_boxplot.png',
        clean_rows, corrupt_rows)
    _plot_relation(
        out_dir / 'geometry_vs_fp.png', rows, 'fp', 'FP / image')
    _plot_relation(
        out_dir / 'geometry_vs_recall.png', rows, 'gt_recall', 'GT recall')
    print(f'clean_images={len(clean_rows)}')
    print(f'corrupt_images={len(corrupt_rows)}')
    print(f'out_dir={out_dir}')


if __name__ == '__main__':
    main()
