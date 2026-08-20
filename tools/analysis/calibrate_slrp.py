#!/usr/bin/env python
"""Calibrate SLRP for interference invariance, with the rest of the net frozen.

Why this stage exists
---------------------
Inside SFOD self-training, SLRP has no usable gradient: the pseudo-labels come
from a teacher looking at the same corrupted image, and a zero-initialized SLRP
is the identity, so the supervision it would learn from is already degraded by
the very interference it is supposed to remove. Nothing in that loop tells it
what to suppress.

This stage supplies the missing signal without labels and without touching the
target domain. Let ``F0`` be the frozen backbone with SLRP bypassed. For a clean
source image ``x`` and a synthetically corrupted ``x'``, SLRP is trained so that

    F(x') ~= F0(x)      (invariance: cancel the interference)
    F(x)  ~= F0(x)      (fidelity: stay the identity on clean input)

Both terms distill towards the *frozen clean* reference, so the target cannot
drift and the trivial "suppress everything" solution is penalised by the second
term. Only SLRP's parameters require grad -- 1200 of them for the post-maxpool
placement.

The corruption family is ``tools/dataset/random_interference.py``, whose
parameters are held out from the seven fixed evaluation specs.
"""

import argparse
import copy
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from iraod_runtime import ensure_iraod_runtime  # noqa: E402

ensure_iraod_runtime()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from mmrotate.models import build_detector  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

import mmdet_extension  # noqa: E402,F401
import sfod  # noqa: E402,F401
from tools.analysis.gate_spatial_geometry import _load_config  # noqa: E402
from tools.analysis.sogc_runtime import (  # noqa: E402
    dump_json, load_checkpoint_sogc_compatible, resolve_device,
    seed_everything)
from tools.dataset.random_interference import (  # noqa: E402
    apply_random_interference)

SUPPORTED_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Calibrate SLRP for interference invariance')
    parser.add_argument(
        '--config',
        default='configs/baseline/oriented_rcnn_slrp_orthonet_rsar.py')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--image-dir', default='/myfile/dataset/RSAR/train/images')
    parser.add_argument('--image-size', type=int, default=512)
    parser.add_argument('--max-images', type=int, default=8000)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument(
        '--fidelity-weight', type=float, default=30.0,
        help='weight of the clean-input gate-quiescence penalty')
    parser.add_argument(
        '--eval-images', type=int, default=256,
        help='held-out images used only for the before/after measurement')
    parser.add_argument('--log-every', type=int, default=50)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def _validate_args(args):
    config = Path(args.config).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    image_dir = Path(args.image_dir).expanduser().resolve()
    for label, path in (('config', config), ('checkpoint', checkpoint)):
        if not path.is_file():
            raise FileNotFoundError(f'{label} not found: {path}')
    if not image_dir.is_dir():
        raise FileNotFoundError(f'image dir not found: {image_dir}')
    for name, value in (('--max-images', args.max_images),
                        ('--epochs', args.epochs),
                        ('--batch-size', args.batch_size),
                        ('--image-size', args.image_size)):
        if value <= 0:
            raise ValueError(f'{name} must be positive')
    if args.image_size % 32 != 0:
        raise ValueError('--image-size must be a multiple of 32')
    if args.lr <= 0.0:
        raise ValueError('--lr must be positive')
    if args.fidelity_weight < 0.0:
        raise ValueError('--fidelity-weight must be non-negative')
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / 'slrp_calibrated.pth'
    if target.exists() and not args.overwrite:
        raise FileExistsError(
            f'refusing to overwrite {target} (use --overwrite)')
    return config, checkpoint, image_dir, out_dir


class PairedInterferenceDataset(Dataset):
    """Yields a clean/corrupt pair of the same source image.

    The corruption is drawn per sample from the held-out calibration family and
    applied in the 8-bit amplitude domain, before normalization, which is where
    the interference physically acts.
    """

    def __init__(self, image_dir, names, image_size, mean, std, seed):
        self.image_dir = Path(image_dir)
        self.names = list(names)
        self.image_size = int(image_size)
        self.mean = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
        self.seed = int(seed)

    def __len__(self):
        return len(self.names)

    def _normalize(self, image):
        tensor = np.ascontiguousarray(
            image.astype(np.float32).transpose(2, 0, 1))
        return torch.from_numpy((tensor - self.mean) / self.std)

    def __getitem__(self, index):
        path = self.image_dir / self.names[index]
        with Image.open(path) as handle:
            image = handle.convert('RGB').resize(
                (self.image_size, self.image_size), Image.BILINEAR)
        clean = np.asarray(image, dtype=np.uint8)
        # Seeded per (epoch-independent) index so a run is reproducible while
        # still covering the family across epochs via the epoch offset.
        rng = np.random.default_rng(self.seed + index)
        corrupt, itype = apply_random_interference(clean, rng)
        corrupt = np.clip(corrupt, 0, 255).astype(np.uint8)
        return {
            'clean': self._normalize(clean),
            'corrupt': self._normalize(corrupt),
            'itype': itype,
        }


def _listing(image_dir, limit):
    names = sorted(
        path.name for path in Path(image_dir).iterdir()
        if path.suffix.lower() in SUPPORTED_EXTS)
    if not names:
        raise RuntimeError(f'no images found in {image_dir}')
    return names[:limit]


def _gate_energy(backbone):
    """Mean squared gate deviation of the most recent forward pass.

    Applied to a clean-input pass it is the quiescence constraint: SLRP must be
    the identity where there is no interference to remove.
    """
    total = 0.0
    count = 0
    for _, module in backbone.iter_slrp_modules():
        total = total + module.last_gate_deviation().square().mean()
        count += 1
    if count == 0:
        raise RuntimeError('backbone exposes no SLRP modules')
    return total / count


def _feature_distance(features, references):
    """Mean cosine distance over stages, positions and samples.

    Channel-normalized so the loss measures feature *direction* and cannot be
    trivially reduced by shrinking the activation magnitude.
    """
    total = 0.0
    for feature, reference in zip(features, references):
        normalized = F.normalize(feature, dim=1, eps=1e-6)
        target = F.normalize(reference, dim=1, eps=1e-6)
        total = total + (1.0 - (normalized * target).sum(dim=1)).mean()
    return total / len(features)


@torch.no_grad()
def _evaluate(backbone, loader, device, max_batches):
    """Invariance and fidelity on a fixed held-out split.

    Measured on the same batches with the same corruption seeds before and after
    training, so the reported change is not confounded by which corruptions a
    given epoch happened to draw.
    """
    totals = {'invariance': 0.0, 'fidelity': 0.0, 'bypass': 0.0,
              'clean_gate': 0.0, 'corrupt_gate': 0.0, 'count': 0}
    for index, batch in enumerate(loader):
        if index >= max_batches:
            break
        clean = batch['clean'].to(device, non_blocking=True)
        corrupt = batch['corrupt'].to(device, non_blocking=True)
        with backbone.slrp_bypassed():
            reference = [feature.detach() for feature in backbone(clean)]
            totals['bypass'] += _feature_distance(
                backbone(corrupt), reference).item()
        totals['invariance'] += _feature_distance(
            backbone(corrupt), reference).item()
        totals['corrupt_gate'] += _gate_energy(backbone).item()
        totals['fidelity'] += _feature_distance(
            backbone(clean), reference).item()
        # The ratio of these two gate energies is what decides whether SLRP is
        # selective or just a global perturbation.
        totals['clean_gate'] += _gate_energy(backbone).item()
        totals['count'] += 1
    count = max(1, totals['count'])
    clean_gate = totals['clean_gate'] / count
    corrupt_gate = totals['corrupt_gate'] / count
    return {
        'invariance': totals['invariance'] / count,
        'fidelity': totals['fidelity'] / count,
        'invariance_without_slrp': totals['bypass'] / count,
        'clean_gate_energy': clean_gate,
        'corrupt_gate_energy': corrupt_gate,
        'selectivity': (corrupt_gate / clean_gate) if clean_gate > 0 else None,
        'batches': totals['count'],
    }


def _build_model(cfg, checkpoint, device):
    model_cfg = copy.deepcopy(cfg.model)
    model_cfg.train_cfg = None
    model = build_detector(model_cfg, test_cfg=cfg.get('test_cfg'))
    load_checkpoint_sogc_compatible(model, checkpoint, allow_missing_sogc=True)
    backbone = model.backbone.to(device)
    if not hasattr(backbone, 'slrp'):
        raise RuntimeError('config does not enable SLRP; nothing to calibrate')
    # Everything frozen and in eval mode: BN statistics must not move, and the
    # reference features must stay exactly the pretrained ones.
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    trainable = []
    for _, module in backbone.iter_slrp_modules():
        for parameter in module.parameters():
            parameter.requires_grad_(True)
            trainable.append(parameter)
    if not trainable:
        raise RuntimeError('SLRP exposes no trainable parameters')
    return backbone, trainable


def _slrp_state_dict(backbone):
    return {
        key: value.detach().cpu().clone()
        for key, value in backbone.state_dict().items() if '.slrp.' in key
        or key.startswith('slrp.')
    }


def main():
    args = parse_args()
    config, checkpoint, image_dir, out_dir = _validate_args(args)
    seed_everything(args.seed)
    device, _ = resolve_device(args.device)
    cfg = _load_config(config)
    backbone, trainable = _build_model(cfg, checkpoint, device)
    parameter_count = sum(parameter.numel() for parameter in trainable)
    print(f'trainable SLRP parameters: {parameter_count}')

    norm_cfg = cfg.get('img_norm_cfg', dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]))
    names = _listing(image_dir, args.max_images + args.eval_images)
    if len(names) <= args.eval_images:
        raise RuntimeError(
            f'need more than {args.eval_images} images to hold an eval split')
    # Held-out split, never trained on, so the reported invariance change is
    # not measured on the images the gate was fitted to.
    eval_names = names[:args.eval_images]
    train_names = names[args.eval_images:]
    train_dataset = PairedInterferenceDataset(
        image_dir, train_names, args.image_size,
        norm_cfg['mean'], norm_cfg['std'], args.seed)
    eval_dataset = PairedInterferenceDataset(
        image_dir, eval_names, args.image_size,
        norm_cfg['mean'], norm_cfg['std'], args.seed + 10 ** 6)
    loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    eval_loader = DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=max(1, args.workers // 2), pin_memory=True)
    eval_batches = max(1, len(eval_dataset) // args.batch_size)
    print(f'calibration images: {len(train_dataset)} train / '
          f'{len(eval_dataset)} held out, iterations/epoch: {len(loader)}')

    before = _evaluate(backbone, eval_loader, device, eval_batches)
    print(f'before: inv={before["invariance"]:.5f} '
          f'(no SLRP {before["invariance_without_slrp"]:.5f}) '
          f'fid={before["fidelity"]:.5f}')

    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs * len(loader)))

    history = []
    started = time.time()
    for epoch in range(args.epochs):
        running = {'loss': 0.0, 'invariance': 0.0, 'fidelity': 0.0,
                   'quiescence': 0.0, 'gate': 0.0, 'count': 0}
        for step, batch in enumerate(loader):
            clean = batch['clean'].to(device, non_blocking=True)
            corrupt = batch['corrupt'].to(device, non_blocking=True)

            with torch.no_grad(), backbone.slrp_bypassed():
                reference = [
                    feature.detach() for feature in backbone(clean)]

            invariance = _feature_distance(backbone(corrupt), reference)
            # Penalise the gate itself on clean input rather than the resulting
            # feature distance: a 0.0045 cosine deviation looked negligible but
            # cost 4 points of clean recall through the frozen detection head.
            clean_features = backbone(clean)
            quiescence = _gate_energy(backbone)
            fidelity = _feature_distance(clean_features, reference)
            loss = invariance + args.fidelity_weight * quiescence

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
            schedule.step()

            diagnostics = backbone.get_slrp_diagnostics()
            gate = float(np.mean([
                stage['gate_magnitude'].mean().item()
                for stage in diagnostics.values()]))
            running['loss'] += loss.item()
            running['invariance'] += invariance.item()
            running['fidelity'] += fidelity.item()
            running['quiescence'] += quiescence.item()
            running['gate'] += gate
            running['count'] += 1
            if (step + 1) % args.log_every == 0:
                count = running['count']
                print(f'epoch {epoch + 1}/{args.epochs} '
                      f'step {step + 1}/{len(loader)} '
                      f'loss={running["loss"] / count:.5f} '
                      f'inv={running["invariance"] / count:.5f} '
                      f'fid={running["fidelity"] / count:.5f} '
                      f'quiesc={running["quiescence"] / count:.5f} '
                      f'gate={running["gate"] / count:.4f}')
        count = max(1, running['count'])
        held_out = _evaluate(backbone, eval_loader, device, eval_batches)
        history.append({
            'epoch': epoch + 1,
            'train_loss': running['loss'] / count,
            'train_invariance': running['invariance'] / count,
            'train_fidelity': running['fidelity'] / count,
            'train_quiescence': running['quiescence'] / count,
            'gate_magnitude': running['gate'] / count,
            'held_out': held_out,
        })
        selectivity = held_out['selectivity']
        print(f'== epoch {epoch + 1}: held-out inv='
              f'{held_out["invariance"]:.5f} '
              f'(before {before["invariance"]:.5f}) '
              f'fid={held_out["fidelity"]:.5f} '
              f'clean_gate={held_out["clean_gate_energy"]:.6f} '
              + (f'selectivity={selectivity:.2f}x'
                 if selectivity is not None else 'selectivity=n/a'))

    # Write a full detector checkpoint so the SFOD stage can load it directly.
    full = torch.load(str(checkpoint), map_location='cpu')
    state_dict = full.get('state_dict', full)
    for key, value in _slrp_state_dict(backbone).items():
        state_dict[f'backbone.{key}' if not key.startswith('backbone.')
                   else key] = value
    if 'state_dict' in full:
        full['state_dict'] = state_dict
    else:
        full = state_dict
    torch.save(full, str(out_dir / 'slrp_calibrated.pth'))

    summary = {
        'settings': {
            'config': str(config),
            'checkpoint': str(checkpoint),
            'image_dir': str(image_dir),
            'image_size': args.image_size,
            'max_images': args.max_images,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'fidelity_weight': args.fidelity_weight,
            'trainable_parameters': parameter_count,
        },
        'held_out_before': before,
        'held_out_after': history[-1]['held_out'],
        'history': history,
        'wall_clock_seconds': time.time() - started,
    }
    dump_json(out_dir / 'summary.json', summary)
    after = history[-1]['held_out']
    reduction = (
        (before['invariance'] - after['invariance']) / before['invariance']
        if before['invariance'] else None)
    print(f'\nheld-out invariance {before["invariance"]:.5f} -> '
          f'{after["invariance"]:.5f}'
          + (f' ({reduction:.1%} reduction)' if reduction is not None else ''))
    print(f'clean-input fidelity error: {after["fidelity"]:.5f}')
    print(f'gate energy clean={after["clean_gate_energy"]:.6f} '
          f'corrupt={after["corrupt_gate_energy"]:.6f}'
          + (f' ({after["selectivity"]:.2f}x selective)'
             if after['selectivity'] is not None else ''))
    print(f'out_dir={out_dir}')


if __name__ == '__main__':
    main()
