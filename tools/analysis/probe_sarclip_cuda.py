#!/usr/bin/env python
"""Isolate where SARCLIP + LoRA hits a CUDA illegal instruction on this GPU.

`optimizer.step()` was where the error surfaced, but CUDA reports asynchronously,
so the real fault is usually earlier. This walks the stages in order -- construct,
forward, backward, step -- with a synchronize after each, so the first stage that
actually fails is the one that reports.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--pretrained',
        default='/myfile/dataset/SARCLIP/ViT-B-32/vit_b_32_model.safetensors')
    parser.add_argument(
        '--cache-dir', default='/myfile/dataset/SARCLIP/ViT-B-32')
    parser.add_argument('--model', default='ViT-B-32')
    parser.add_argument('--precision', default='fp32')
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


def stage(name, function):
    print(f'--- {name}', flush=True)
    try:
        result = function()
        torch.cuda.synchronize()
        print(f'    OK', flush=True)
        return result
    except Exception as error:  # noqa: BLE001
        print(f'    FAILED: {type(error).__name__}: {error}', flush=True)
        raise SystemExit(1) from error


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f'torch {torch.__version__}  cuda {torch.version.cuda}')
    print(f'device capability {torch.cuda.get_device_capability(device)}')
    print(f'arch list {torch.cuda.get_arch_list()}')
    print()

    import sar_clip  # noqa: PLC0415

    model = stage('create_model', lambda: sar_clip.create_model(
        args.model, pretrained=args.pretrained, precision=args.precision,
        cache_dir=args.cache_dir))
    stage('to(device)', lambda: model.to(device))
    model.eval()

    image = torch.randn(4, 3, 224, 224, device=device)

    def encode_image_nograd():
        # A bare `torch.no_grad().__enter__()` here leaks the context and
        # disables grad for every later stage, which masks the real fault.
        with torch.no_grad():
            return model.encode_image(image)

    stage('encode_image (no grad)', encode_image_nograd)

    tokenizer = stage('get_tokenizer', lambda: sar_clip.get_tokenizer(
        args.model, cache_dir=args.cache_dir))
    tokens = stage('tokenize', lambda: tokenizer(
        ['a SAR image of a ship', 'a SAR image of a car']).to(device))
    with torch.no_grad():
        stage('encode_text (no grad)', lambda: model.encode_text(tokens))

    # The forward that training actually runs: grad enabled, visual tower only.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    visual_last = None
    for name, module in model.visual.named_modules():
        if isinstance(module, torch.nn.Linear):
            visual_last = (name, module)
    if visual_last is None:
        raise SystemExit('no Linear layer found in the visual tower')
    name, module = visual_last
    module.weight.requires_grad_(True)
    print(f'\ntraining probe on visual.{name} '
          f'({module.weight.numel()} params)')

    features = stage('encode_image (grad)', lambda: model.encode_image(image))
    loss = stage('loss', lambda: features.square().mean())
    stage('backward', lambda: loss.backward())
    optimizer = torch.optim.AdamW([module.weight], lr=1e-4)
    stage('optimizer.step', lambda: optimizer.step())

    print('\nall stages passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
