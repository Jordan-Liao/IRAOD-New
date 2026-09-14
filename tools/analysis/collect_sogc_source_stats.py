#!/usr/bin/env python
"""Collect clean-source SOGC response statistics into a checkpoint."""

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from iraod_runtime import ensure_iraod_runtime  # noqa: E402

ensure_iraod_runtime()

import torch  # noqa: E402
from mmcv import Config  # noqa: E402
from mmdet.datasets import build_dataloader  # noqa: E402
from mmrotate.models import build_detector  # noqa: E402

import mmdet_extension  # noqa: E402,F401
import sfod  # noqa: E402,F401
from sfod.utils import patch_config  # noqa: E402
from tools.analysis.sogc_runtime import (  # noqa: E402
    build_dataset_checked, clean_source_dataset_cfg, detector_backbone,
    dump_json, load_checkpoint_sogc_compatible, make_backbone_runner,
    resolve_device, seed_everything)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Collect source-domain SOGC response statistics')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-images', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--work-dir', default='work_dirs/sogc_source_stats')
    return parser.parse_args()


def _validate_args(args):
    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_path = Path(args.out).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f'config not found: {config_path}')
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'checkpoint not found: {checkpoint_path}')
    if output_path == checkpoint_path:
        raise ValueError('output checkpoint must not overwrite the input')
    if output_path.exists():
        raise FileExistsError(f'output checkpoint already exists: {output_path}')
    if args.max_images is not None and args.max_images <= 1:
        raise ValueError('--max-images must be greater than one')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return config_path, checkpoint_path, output_path


def _module_summary(backbone):
    modules = []
    for stage_index, block_index, calibrator in \
            backbone.iter_sogc_calibrators():
        modules.append({
            'stage_index': stage_index,
            'feature_stage': f'C{stage_index + 2}',
            'block_index': block_index,
            'channels': calibrator.channels,
            'source_count': int(calibrator.source_count.item()),
            'stats_ready': bool(calibrator.stats_ready.item()),
            'mean_finite': bool(torch.isfinite(
                calibrator.source_mean).all().item()),
            'var_finite': bool(torch.isfinite(
                calibrator.source_var).all().item()),
            'var_nonnegative': bool((
                calibrator.source_var >= 0).all().item()),
            'var_min': float(calibrator.source_var.min().item()),
            'var_max': float(calibrator.source_var.max().item()),
        })
    return modules


def main():
    args = parse_args()
    config_path, checkpoint_path, output_path = _validate_args(args)
    work_dir = Path(args.work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device, device_ids = resolve_device(args.device)

    cfg = patch_config(Config.fromfile(str(config_path)))
    dataset_cfg = clean_source_dataset_cfg(cfg)
    dataset = build_dataset_checked(dataset_cfg)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        num_gpus=1,
        dist=False,
        shuffle=False,
        seed=args.seed)

    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    source_checkpoint, load_summary = load_checkpoint_sogc_compatible(
        model, checkpoint_path, allow_missing_sogc=True)
    backbone = detector_backbone(model)
    backbone.reset_source_stats()
    backbone.set_sogc_mode('collect')
    model.eval()
    runner = make_backbone_runner(backbone, device, device_ids)

    processed = 0
    with torch.no_grad():
        for data in data_loader:
            if args.max_images is not None and processed >= args.max_images:
                break
            runner(**data)
            processed += 1

    backbone.finalize_source_stats()
    modules = _module_summary(backbone)
    for module in modules:
        if module['source_count'] <= 1:
            raise RuntimeError(f'insufficient samples for {module}')
        if not all((module['stats_ready'], module['mean_finite'],
                    module['var_finite'], module['var_nonnegative'])):
            raise RuntimeError(f'invalid finalized SOGC statistics: {module}')
    backbone.set_sogc_mode('calibrate')

    git_commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, text=True).strip()
    active = backbone.get_sogc_metadata()
    meta = dict(source_checkpoint.get('meta', {}))
    meta.update({
        'source_config': str(config_path),
        'source_checkpoint': str(checkpoint_path),
        'source_sample_count': processed,
        'sogc_hyperparameters': active,
        'git_commit': git_commit,
        'active_stages_blocks': [
            {
                'stage_index': item['stage_index'],
                'feature_stage': item['feature_stage'],
                'block_index': item['block_index'],
            } for item in active
        ],
    })
    checkpoint = dict(meta=meta, state_dict=model.state_dict())
    torch.save(checkpoint, str(output_path))

    summary = {
        'status': 'ok',
        'output_checkpoint': str(output_path),
        'processed_images': processed,
        'dataset_size': len(dataset),
        'load_summary': load_summary,
        'modules': modules,
        'meta': meta,
    }
    summary_path = work_dir / 'collect_sogc_source_stats_summary.json'
    dump_json(summary_path, summary)
    print(f'processed_images={processed}')
    print(f'output_checkpoint={output_path}')
    print(f'summary={summary_path}')


if __name__ == '__main__':
    main()
