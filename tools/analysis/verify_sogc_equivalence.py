#!/usr/bin/env python
"""Verify that zero-initialized SOGC leaves the source detector unchanged.

Model A is the plain OrthoNet detector with its original checkpoint; model B is
the SOGC detector with collected source statistics. Both run on the same input
batch inside one process, so any difference isolates to the SOGC branch.
"""

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from iraod_runtime import ensure_iraod_runtime  # noqa: E402

ensure_iraod_runtime()

import copy  # noqa: E402

import mmcv  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from mmcv import Config  # noqa: E402
from mmdet.datasets import build_dataloader  # noqa: E402
from mmrotate.models import build_detector  # noqa: E402
from mmrotate.utils import compat_cfg  # noqa: E402

import mmdet_extension  # noqa: E402,F401
import sfod  # noqa: E402,F401
from sfod.utils import patch_config  # noqa: E402
from tools.analysis.sogc_runtime import (  # noqa: E402
    build_dataset_checked, detector_backbone, dump_json,
    evaluation_dataset_cfg, load_checkpoint_sogc_compatible,
    make_detector_runner, resolve_device, seed_everything)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Verify SOGC zero-initialization detector equivalence')
    parser.add_argument('--baseline-config', required=True)
    parser.add_argument('--baseline-checkpoint', required=True)
    parser.add_argument('--sogc-config', required=True)
    parser.add_argument('--sogc-checkpoint', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--tolerance', type=float, default=1e-6)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-images', type=int, default=None)
    parser.add_argument(
        '--save-predictions', action='store_true',
        help='dump both prediction sets as pickles for offline comparison')
    parser.add_argument('--log-interval', type=int, default=250)
    return parser.parse_args()


def _validate_args(args):
    paths = {}
    for name in ('baseline_config', 'baseline_checkpoint',
                 'sogc_config', 'sogc_checkpoint'):
        path = Path(getattr(args, name)).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f'{name} not found: {path}')
        paths[name] = path
    if args.tolerance < 0.0:
        raise ValueError('--tolerance must be non-negative')
    if args.max_images is not None and args.max_images <= 0:
        raise ValueError('--max-images must be positive')
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / 'equivalence_summary.json'
    if summary_path.exists():
        raise FileExistsError(f'refusing to overwrite {summary_path}')
    return paths, out_dir


def _as_class_list(result):
    if isinstance(result, tuple):
        result = result[0]
    if not isinstance(result, list):
        raise TypeError('detector result must be a list of per-class arrays')
    return result


def _compare_result(result_a, result_b, tolerance):
    """Compare one image's per-class rotated detections."""
    classes_a = _as_class_list(result_a)
    classes_b = _as_class_list(result_b)
    if len(classes_a) != len(classes_b):
        return {'shape_mismatch': True, 'max_abs_diff': float('inf'),
                'over_tolerance': -1, 'bitwise_equal': False,
                'counts_a': [], 'counts_b': []}

    max_abs_diff = 0.0
    over_tolerance = 0
    bitwise_equal = True
    shape_mismatch = False
    counts_a = []
    counts_b = []
    for class_a, class_b in zip(classes_a, classes_b):
        array_a = np.asarray(class_a)
        array_b = np.asarray(class_b)
        counts_a.append(int(array_a.shape[0]) if array_a.size else 0)
        counts_b.append(int(array_b.shape[0]) if array_b.size else 0)
        if array_a.shape != array_b.shape:
            shape_mismatch = True
            bitwise_equal = False
            max_abs_diff = float('inf')
            continue
        if array_a.size == 0:
            continue
        if not np.array_equal(array_a, array_b):
            bitwise_equal = False
        difference = np.abs(
            array_a.astype(np.float64) - array_b.astype(np.float64))
        max_abs_diff = max(max_abs_diff, float(difference.max()))
        over_tolerance += int((difference > tolerance).sum())
    return {'shape_mismatch': shape_mismatch, 'max_abs_diff': max_abs_diff,
            'over_tolerance': over_tolerance, 'bitwise_equal': bitwise_equal,
            'counts_a': counts_a, 'counts_b': counts_b}


def _build_model(config_path, checkpoint_path, device, device_ids,
                 expect_sogc):
    cfg = patch_config(compat_cfg(Config.fromfile(str(config_path))))
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    _, load_summary = load_checkpoint_sogc_compatible(
        model, checkpoint_path, allow_missing_sogc=not expect_sogc)
    backbone = None
    if expect_sogc:
        backbone = detector_backbone(model)
        if not backbone.has_valid_source_stats():
            raise RuntimeError(
                'SOGC model lacks finalized source statistics; run '
                'tools/analysis/collect_sogc_source_stats.py first')
        backbone.set_sogc_mode('calibrate')
    model.eval()
    return cfg, make_detector_runner(model, device, device_ids), \
        backbone, load_summary


def _assert_same_test_data(cfg_a, cfg_b):
    data_a = evaluation_dataset_cfg(cfg_a)
    data_b = evaluation_dataset_cfg(cfg_b)
    for key in ('type', 'ann_file', 'img_prefix', 'classes', 'pipeline'):
        if data_a.get(key) != data_b.get(key):
            raise ValueError(
                f'baseline and SOGC configs disagree on data.test.{key}; '
                'equivalence requires an identical evaluation pipeline')
    return data_a


def _identity_calibration(backbone, tolerance):
    """Confirm the SOGC branch reported an exactly multiplicative identity."""
    findings = {}
    for stage, diagnostics in backbone.get_sogc_diagnostics().items():
        magnitude = float(diagnostics['calibration_magnitude'].max())
        factor_min = float(diagnostics['calibration_factor_min'].min())
        factor_max = float(diagnostics['calibration_factor_max'].max())
        findings[stage] = {
            'calibration_magnitude_max': magnitude,
            'calibration_factor_min': factor_min,
            'calibration_factor_max': factor_max,
            'is_identity': bool(
                magnitude <= tolerance
                and abs(factor_min - 1.0) <= tolerance
                and abs(factor_max - 1.0) <= tolerance),
        }
    return findings


def _worst(entries, key, default=0.0):
    values = [entry[key] for entry in entries]
    return max(values) if values else default


def _evaluate_map(dataset, results, label):
    metrics = dataset.evaluate(results, metric='mAP')
    parsed = {}
    for key, value in metrics.items():
        try:
            parsed[str(key)] = float(value)
        except (TypeError, ValueError):
            parsed[str(key)] = value
    if 'mAP' not in parsed:
        raise RuntimeError(f'{label} evaluation returned no mAP entry')
    return parsed


def main():
    args = parse_args()
    paths, out_dir = _validate_args(args)
    seed_everything(42)
    device, device_ids = resolve_device(args.device)
    # Equivalence is a bitwise claim, so algorithm selection must not drift
    # between the two forward passes.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    cfg_a, runner_a, _, load_a = _build_model(
        paths['baseline_config'], paths['baseline_checkpoint'],
        device, device_ids, expect_sogc=False)
    cfg_b, runner_b, backbone_b, load_b = _build_model(
        paths['sogc_config'], paths['sogc_checkpoint'],
        device, device_ids, expect_sogc=True)
    if load_a['missing_keys']:
        raise RuntimeError(
            'baseline checkpoint did not fully populate the baseline model: '
            f"{load_a['missing_keys'][:10]}")
    if load_b['missing_keys']:
        raise RuntimeError(
            'SOGC checkpoint is missing SOGC keys: '
            f"{load_b['missing_keys'][:10]}")

    dataset_cfg = _assert_same_test_data(cfg_a, cfg_b)
    dataset = build_dataset_checked(dataset_cfg)
    loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=0, num_gpus=1,
        dist=False, shuffle=False)

    comparisons = []
    results_a = []
    results_b = []
    calibration = {}
    with torch.no_grad():
        for index, data in enumerate(loader):
            if args.max_images is not None and index >= args.max_images:
                break
            batch_a = runner_a(
                return_loss=False, rescale=True, **copy.deepcopy(data))
            batch_b = runner_b(
                return_loss=False, rescale=True, **copy.deepcopy(data))
            if len(batch_a) != 1 or len(batch_b) != 1:
                raise RuntimeError('equivalence requires one result per batch')
            if not calibration:
                calibration = _identity_calibration(
                    backbone_b, args.tolerance)
            comparison = _compare_result(
                batch_a[0], batch_b[0], args.tolerance)
            comparison['image_index'] = index
            comparison['filename'] = dataset.data_infos[index].get(
                'filename', '')
            comparisons.append(comparison)
            results_a.append(batch_a[0])
            results_b.append(batch_b[0])
            if args.log_interval and (index + 1) % args.log_interval == 0:
                print(f'compared {index + 1} images, '
                      f"worst_abs_diff={_worst(comparisons, 'max_abs_diff')}",
                      flush=True)

    processed = len(comparisons)
    if processed == 0:
        raise RuntimeError('no images were compared')
    return _report(args, out_dir, dataset, comparisons, results_a, results_b,
                   calibration, load_a, load_b, processed, paths)


def _class_totals(comparisons, key, num_classes):
    totals = [0] * num_classes
    for comparison in comparisons:
        for class_index, count in enumerate(comparison[key]):
            totals[class_index] += count
    return totals


def _report(args, out_dir, dataset, comparisons, results_a, results_b,
            calibration, load_a, load_b, processed, paths):
    num_classes = max(
        (len(comparison['counts_a']) for comparison in comparisons),
        default=0)
    totals_a = _class_totals(comparisons, 'counts_a', num_classes)
    totals_b = _class_totals(comparisons, 'counts_b', num_classes)
    class_names = list(getattr(dataset, 'CLASSES', []) or [])

    worst_diff = _worst(comparisons, 'max_abs_diff')
    over_tolerance_images = [
        comparison['image_index'] for comparison in comparisons
        if comparison['over_tolerance'] != 0]
    shape_mismatch_images = [
        comparison['image_index'] for comparison in comparisons
        if comparison['shape_mismatch']]
    bitwise_equal_images = sum(
        1 for comparison in comparisons if comparison['bitwise_equal'])

    prediction_files = {}
    if args.save_predictions:
        for label, results in (('baseline', results_a), ('sogc', results_b)):
            path = out_dir / f'predictions_{label}.pkl'
            mmcv.dump(results, str(path))
            prediction_files[label] = str(path)

    evaluated_full_split = processed == len(dataset)
    evaluation = {}
    if evaluated_full_split:
        evaluation = {
            'baseline': _evaluate_map(dataset, results_a, 'baseline'),
            'sogc': _evaluate_map(dataset, results_b, 'sogc'),
        }
        evaluation['map_abs_diff'] = abs(
            evaluation['baseline']['mAP'] - evaluation['sogc']['mAP'])
    else:
        evaluation['skipped'] = (
            f'mAP needs the full split; compared {processed} of '
            f'{len(dataset)} images')

    identity_ok = all(
        entry['is_identity'] for entry in calibration.values())
    detections_match = totals_a == totals_b
    passed = bool(
        not shape_mismatch_images
        and not over_tolerance_images
        and detections_match
        and identity_ok
        and (not evaluated_full_split
             or evaluation['map_abs_diff'] <= args.tolerance))

    summary = {
        'passed': passed,
        'tolerance': args.tolerance,
        'compared_images': processed,
        'dataset_size': len(dataset),
        'evaluated_full_split': evaluated_full_split,
        'inputs': {name: str(path) for name, path in paths.items()},
        'load_summary': {
            'baseline_missing_keys': load_a['missing_keys'],
            'baseline_unexpected_keys': load_a['unexpected_keys'],
            'sogc_missing_keys': load_b['missing_keys'],
            'sogc_unexpected_keys': load_b['unexpected_keys'],
        },
        'prediction_agreement': {
            'max_abs_diff': worst_diff,
            'bitwise_equal_images': bitwise_equal_images,
            'over_tolerance_image_count': len(over_tolerance_images),
            'over_tolerance_images': over_tolerance_images[:50],
            'shape_mismatch_image_count': len(shape_mismatch_images),
            'shape_mismatch_images': shape_mismatch_images[:50],
        },
        'detection_counts': {
            'total_baseline': int(sum(totals_a)),
            'total_sogc': int(sum(totals_b)),
            'per_class_match': detections_match,
            'per_class': [
                {
                    'class_index': index,
                    'class_name': class_names[index]
                    if index < len(class_names) else None,
                    'baseline': totals_a[index],
                    'sogc': totals_b[index],
                }
                for index in range(num_classes)
            ],
        },
        'sogc_calibration_identity': calibration,
        'evaluation': evaluation,
        'prediction_files': prediction_files,
    }
    dump_json(out_dir / 'equivalence_summary.json', summary)

    print(f'compared_images={processed}')
    print(f'max_abs_diff={worst_diff}')
    print(f'bitwise_equal_images={bitwise_equal_images}/{processed}')
    if evaluated_full_split:
        print(f"baseline_mAP={evaluation['baseline']['mAP']:.6f}")
        print(f"sogc_mAP={evaluation['sogc']['mAP']:.6f}")
        print(f"map_abs_diff={evaluation['map_abs_diff']:.3e}")
    print(f'passed={passed}')
    print(f'summary={out_dir / "equivalence_summary.json"}')
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
