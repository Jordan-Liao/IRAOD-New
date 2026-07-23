"""One-step real-RSAR training smoke test for the StripNet configuration."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from iraod_runtime import ensure_iraod_runtime  # noqa: E402

ensure_iraod_runtime()

import argparse
import math

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import build_optimizer
from mmdet.datasets import build_dataloader
from mmrotate.datasets import build_dataset
from mmrotate.models import build_detector

import mmdet_extension  # noqa: F401
import sfod  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='configs/baseline/oriented_rcnn_stripnet_rsar.py')
    parser.add_argument('--ann-dir', required=True)
    parser.add_argument('--img-prefix', required=True)
    parser.add_argument('--samples-per-gpu', type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.data.train.ann_file = args.ann_dir
    cfg.data.train.img_prefix = args.img_prefix
    cfg.data.samples_per_gpu = args.samples_per_gpu
    cfg.data.workers_per_gpu = 0

    dataset = build_dataset(cfg.data.train)
    if len(dataset) != args.samples_per_gpu:
        raise RuntimeError(
            'smoke dataset size must match samples_per_gpu: '
            f'{len(dataset)} != {args.samples_per_gpu}')
    loader = build_dataloader(
        dataset,
        samples_per_gpu=args.samples_per_gpu,
        workers_per_gpu=0,
        num_gpus=1,
        dist=False,
        shuffle=False,
        seed=42)

    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    optimizer = build_optimizer(model, cfg.optimizer)
    model = MMDataParallel(model.cuda(), device_ids=[0])
    model.train()

    batch = next(iter(loader))
    optimizer.zero_grad()
    outputs = model.train_step(batch, optimizer)
    loss = outputs['loss']
    if not torch.isfinite(loss):
        raise FloatingPointError(f'non-finite training loss: {loss.item()}')
    loss.backward()
    optimizer.step()

    log_vars = {
        key: float(value)
        for key, value in outputs['log_vars'].items()
        if math.isfinite(float(value))
    }
    print('stripnet_rsar_smoke=PASS')
    print(f'loss={float(loss.detach()):.6f}')
    print(f'log_vars={log_vars}')
    print(f'max_cuda_memory_mib={torch.cuda.max_memory_allocated() / 2**20:.1f}')


if __name__ == '__main__':
    main()
