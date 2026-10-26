#!/usr/bin/env python
"""Audit what OrthoNet's orthogonal channel attention actually computes.

Three independent checks, none of which need a training run:

``keys``
    Compares ``OrthoNet(depth=50)`` against ``ResNet(depth=50)`` key by key and
    then loads a torchvision ResNet-50 state dict into the OrthoNet. This is
    the evidence behind the claim that an ImageNet inflation of OrthoNet is a
    drop-in change: the only extra keys are ``*.attn.*``.

``projection``
    Measures how much of the pooled channel vector the "orthogonal response"
    actually removes. Removing one unit-norm direction from a ``C``-dimensional
    vector removes ``1/C`` of its energy in expectation, so the quantity is
    expected to be 0.4% at C=256 and 0.05% at C=2048 unless the learned vector
    has aligned with a dominant channel.

``checkpoint``
    Reports ``||ortho_vector||`` and the magnitude of the attention's sigmoid
    output layer for each stage of a trained checkpoint. ``ortho_vector`` is
    only ever used through ``F.normalize``, so its magnitude carries no
    function; weight decay on it is pure noise on the parameterisation. Where
    the norm has collapsed, the learned direction is floating-point noise and
    the output-layer magnitude is what is left of the mechanism.

Run from the repository root:

    /opt/conda/envs/iraod/bin/python \
        tools/analysis/audit_orthonet_attention.py \
        --checkpoints /myfile/pretrain/oriented_rcnn_orthonet_rsar_epoch_100.pth \
        --out-json work_dirs/orthonet_attention_audit.json
"""

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from iraod_runtime import ensure_iraod_runtime  # noqa: E402

ensure_iraod_runtime()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import mmdet_extension  # noqa: E402,F401
import sfod  # noqa: E402,F401
from mmdet.models import build_backbone  # noqa: E402
from mmdet_extension.models.backbones.orthonet import (  # noqa: E402
    OrthogonalChannelAttention)

ORTHONET_CFG = dict(
    type='OrthoNet', depth=50, reduction=16, in_channels=3, num_stages=4,
    out_indices=(0, 1, 2, 3), frozen_stages=-1, norm_eval=False,
    style='pytorch', init_cfg=None)
RESNET_CFG = dict(
    type='ResNet', depth=50, num_stages=4, out_indices=(0, 1, 2, 3),
    style='pytorch')
STAGE_KEYS = ('layer1', 'layer2', 'layer3', 'layer4')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Audit OrthoNet orthogonal channel attention')
    parser.add_argument(
        '--checkpoints', nargs='*', default=[
            '/myfile/pretrain/oriented_rcnn_orthonet_rsar_epoch_100.pth'],
        help='trained OrthoNet checkpoints to inspect')
    parser.add_argument('--probe-batch', type=int, default=8)
    parser.add_argument('--probe-size', type=int, default=25,
                        help='spatial size of the synthetic probe activation')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out-json', default=None)
    return parser.parse_args()


def build_orthonet():
    return build_backbone(ORTHONET_CFG).eval()


def strip_backbone_prefix(state_dict):
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    return {
        key[len('backbone.'):]: value
        for key, value in state_dict.items()
        if key.startswith('backbone.')
    }


def audit_keys():
    """Check that OrthoNet is ResNet-50 plus attention parameters."""
    orthonet = build_orthonet()
    resnet = build_backbone(RESNET_CFG)
    ortho_keys = {k: tuple(v.shape) for k, v in orthonet.state_dict().items()}
    resnet_keys = {k: tuple(v.shape) for k, v in resnet.state_dict().items()}

    only_orthonet = sorted(set(ortho_keys) - set(resnet_keys))
    only_resnet = sorted(set(resnet_keys) - set(ortho_keys))
    mismatch = sorted(
        key for key in set(ortho_keys) & set(resnet_keys)
        if ortho_keys[key] != resnet_keys[key])

    from torchvision.models import resnet50
    torchvision_state = {
        key: value for key, value in resnet50(pretrained=False).state_dict().items()
        if not key.startswith('fc.')}
    result = orthonet.load_state_dict(torchvision_state, strict=False)

    non_attention = [key for key in only_orthonet if '.attn.' not in key]
    report = OrderedDict(
        orthonet_params=sum(p.numel() for p in orthonet.parameters()),
        resnet_params=sum(p.numel() for p in resnet.parameters()),
        orthonet_non_attn_params=sum(
            p.numel() for key, p in orthonet.named_parameters()
            if '.attn.' not in key),
        attention_params=sum(
            p.numel() for key, p in orthonet.named_parameters()
            if '.attn.' in key),
        keys_only_in_orthonet=len(only_orthonet),
        keys_only_in_resnet=len(only_resnet),
        shape_mismatches=len(mismatch),
        non_attention_extra_keys=non_attention,
        torchvision_missing_keys=len(result.missing_keys),
        torchvision_missing_all_attention=all(
            '.attn.' in key for key in result.missing_keys),
        torchvision_unexpected_keys=len(result.unexpected_keys),
    )
    print('=== keys')
    print(f'  OrthoNet {report["orthonet_params"] / 1e6:.3f}M params '
          f'({report["attention_params"] / 1e6:.3f}M in attention)')
    print(f'  ResNet50 {report["resnet_params"] / 1e6:.3f}M params; '
          f'OrthoNet non-attention {report["orthonet_non_attn_params"] / 1e6:.3f}M')
    print(f'  keys only in OrthoNet: {report["keys_only_in_orthonet"]} '
          f'(non-attn: {len(non_attention)}) | only in ResNet: '
          f'{report["keys_only_in_resnet"]} | shape mismatches: '
          f'{report["shape_mismatches"]}')
    print(f'  torchvision resnet50 -> OrthoNet: '
          f'{report["torchvision_missing_keys"]} missing '
          f'(all attn: {report["torchvision_missing_all_attention"]}), '
          f'{report["torchvision_unexpected_keys"]} unexpected')
    return report


def audit_projection(model, batch, size, seed):
    """Measure the energy the orthogonal projection removes, per stage."""
    generator = torch.Generator().manual_seed(seed)
    per_stage = OrderedDict()
    for name, module in model.named_modules():
        if not isinstance(module, OrthogonalChannelAttention):
            continue
        channels = module.channels
        # Unit-scale activation: what a post-norm feature map looks like.
        activation = torch.randn(
            batch, channels, size, size, generator=generator) * 0.5 + 0.2
        with torch.no_grad():
            pooled = F.adaptive_avg_pool2d(activation, 1).flatten(1)
            response = module.orthogonal_response(activation).flatten(1)
            gate = module.fc(response)
        removed = (1.0 - response.pow(2).sum(1)
                   / pooled.pow(2).sum(1).clamp_min(1e-12)).mean().item()
        per_stage.setdefault(name.split('.')[0], []).append(OrderedDict(
            module=name,
            channels=channels,
            energy_removed=removed,
            random_reference=1.0 / channels,
            gate_mean=gate.mean().item(),
            gate_std=gate.std().item(),
            response_std=response.std().item(),
            activation_scale=activation.abs().mean().item(),
        ))
    return per_stage


def summarise(stages):
    summary = OrderedDict()
    for stage, entries in stages.items():
        removed = torch.tensor([e['energy_removed'] for e in entries])
        gate_std = torch.tensor([e['gate_std'] for e in entries])
        gate_mean = torch.tensor([e['gate_mean'] for e in entries])
        summary[stage] = OrderedDict(
            modules=len(entries),
            channels=entries[0]['channels'],
            energy_removed_mean=removed.mean().item(),
            random_reference=entries[0]['random_reference'],
            gate_mean=gate_mean.mean().item(),
            gate_std_mean=gate_std.mean().item(),
        )
    return summary


def print_projection(title, stages):
    print(f'=== projection ({title})')
    print(f'  {"stage":8s} {"C":>5s} {"removed":>9s} {"1/C":>9s} '
          f'{"gate mean":>10s} {"gate std":>10s}')
    for stage, entry in summarise(stages).items():
        print(f'  {stage:8s} {entry["channels"]:5d} '
              f'{entry["energy_removed_mean"]:9.5f} '
              f'{entry["random_reference"]:9.5f} '
              f'{entry["gate_mean"]:10.4f} {entry["gate_std_mean"]:10.5f}')


def audit_checkpoint(model, path):
    """Report the trained attention magnitudes per stage."""
    try:
        raw = torch.load(path, map_location='cpu')
    except Exception as error:  # pragma: no cover - environment dependent
        print(f'  {path}: load failed ({error})')
        return None
    state = strip_backbone_prefix(raw)
    model.load_state_dict(state, strict=False)
    print(f'=== checkpoint {path}')
    print(f'  {"stage":8s} {"||ortho_v||":>12s} {"fc1 std":>9s} '
          f'{"fc2 std":>9s} {"fc2 absmax":>11s}')
    report = OrderedDict()
    for stage in STAGE_KEYS:
        vector = state.get(f'{stage}.0.attn.ortho_vector')
        fc1 = state.get(f'{stage}.0.attn.fc.0.weight')
        fc2 = state.get(f'{stage}.0.attn.fc.2.weight')
        if vector is None:
            continue
        entry = OrderedDict(
            ortho_vector_norm=vector.norm().item(),
            ortho_vector_init_norm=float(vector.numel()) ** 0.5,
            fc1_std=fc1.std().item() if fc1 is not None else None,
            fc2_std=fc2.std().item() if fc2 is not None else None,
            fc2_absmax=fc2.abs().max().item() if fc2 is not None else None,
        )
        report[stage] = entry
        print(f'  {stage:8s} {entry["ortho_vector_norm"]:12.4f} '
              f'{(entry["fc1_std"] or 0.0):9.4f} {(entry["fc2_std"] or 0.0):9.5f} '
              f'{(entry["fc2_absmax"] or 0.0):11.5f} '
              f'(init ||v|| {entry["ortho_vector_init_norm"]:.1f})')
    return report


def main():
    args = parse_args()
    if args.probe_batch <= 0 or args.probe_size <= 0:
        raise ValueError('--probe-batch and --probe-size must be positive')

    report = OrderedDict(keys=audit_keys())

    model = build_orthonet()
    torch.manual_seed(args.seed)
    stages = audit_projection(model, args.probe_batch, args.probe_size,
                              args.seed)
    print_projection('fresh random attention', stages)
    report['projection_random_init'] = summarise(stages)

    report['checkpoints'] = OrderedDict()
    for path in args.checkpoints:
        if not Path(path).is_file():
            print(f'=== checkpoint {path}: not found, skipped')
            continue
        report['checkpoints'][path] = audit_checkpoint(model, path)
        if report['checkpoints'][path] is None:
            continue
        stages = audit_projection(model, args.probe_batch, args.probe_size,
                                  args.seed)
        print_projection(f'trained attention from {Path(path).name}', stages)
        report['checkpoints'][path]['projection'] = summarise(stages)

    print()
    print('Interpretation')
    print('  * "removed" near "1/C" means the orthogonal projection removes '
          'only the')
    print('    share a random direction would, i.e. it is an SE block with a')
    print('    numerically negligible extra term.')
    print('  * gate mean ~0.5 with gate std ~0 means the attention is a '
          'constant')
    print('    gain: out = relu(0.5 * F(x) + x) for every sample and channel.')
    print('  * ||ortho_vector|| is not a functional parameter (the forward '
          'uses only')
    print('    its direction), so weight decay on it is pure noise on the '
          'parameterisation.')

    if args.out_json:
        target = Path(args.out_json)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('w') as handle:
            json.dump(report, handle, indent=2)
        print(f'\nwrote {target}')


if __name__ == '__main__':
    main()
