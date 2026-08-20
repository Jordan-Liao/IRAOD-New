#!/usr/bin/env python
"""Verify the trained LoRA loads through the real CGA scorer path.

Training accuracy says the adapter learned something; it does not say CGA can load
it. The scorer resolves weights through environment variables and a separate
adapter-format check, so a mismatch there surfaces as a silently unadapted scorer
that quietly degrades every pseudo-label decision rather than as an error.

This scores real patches of known class and reports whether the argmax matches, so
a wrongly-loaded adapter shows up as chance-level accuracy before any SFOD arm runs.
"""

import argparse
import csv
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from iraod_runtime import ensure_iraod_runtime  # noqa: E402

ensure_iraod_runtime()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--metadata',
        default='work_dirs/sarclip_patches_train_corrupt_aabb/metadata.csv')
    parser.add_argument('--per-class', type=int, default=8)
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


def load_samples(metadata, per_class):
    """A few patches per class, spread across corruptions."""
    buckets = defaultdict(list)
    with Path(metadata).expanduser().open(newline='') as handle:
        for row in csv.DictReader(handle):
            buckets[row['class_name']].append(
                (row['patch_path'], row['corruption']))
    rng = random.Random(0)
    samples = []
    for class_name in sorted(buckets):
        pool = buckets[class_name]
        chosen = pool if len(pool) <= per_class else rng.sample(pool, per_class)
        samples.extend(
            (path, class_name, corruption) for path, corruption in chosen)
    return samples


def main():
    args = parse_args()
    lora = (REPO_ROOT / 'work_dirs' / 'sarclip_lora_rsar_train_corrupt_aabb_v1'
            / 'lora_rsar.pth')
    if not lora.is_file():
        print(f'FATAL: LoRA not found at {lora}')
        return 1

    os.environ.setdefault('CGA_SCORER', 'sarclip')
    os.environ.setdefault('CGA_BACKEND', 'sarclip')
    os.environ['SARCLIP_LORA'] = str(lora)
    os.environ['SARCLIP_PRETRAINED'] = (
        '/myfile/dataset/SARCLIP/ViT-B-32/vit_b_32_model.safetensors')
    os.environ['SARCLIP_CACHE_DIR'] = '/myfile/dataset/SARCLIP/ViT-B-32'

    from sfod.cga import ClipGuidedScorer  # noqa: PLC0415

    classes = ['ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor']
    scorer = ClipGuidedScorer.build(
        class_names=classes, device=torch.device(args.device))
    print(f'scorer built: {type(scorer).__name__}')
    print(f'lora = {lora}')

    samples = load_samples(args.metadata, args.per_class)
    if not samples:
        print('FATAL: no samples read from metadata')
        return 1

    correct = 0
    per_class_correct = defaultdict(lambda: [0, 0])
    for path, truth, _corruption in samples:
        patch = np.asarray(Image.open(path).convert('RGB'))
        scores = scorer.score_patches([patch])
        vector = np.asarray(scores)[0]
        if not np.all(np.isfinite(vector)):
            print(f'FATAL: non-finite scores for {path}')
            return 1
        predicted = classes[int(np.argmax(vector))]
        hit = predicted == truth
        correct += hit
        per_class_correct[truth][0] += hit
        per_class_correct[truth][1] += 1

    total = len(samples)
    accuracy = correct / total
    print(f'\npatches scored : {total}')
    print(f'argmax accuracy: {accuracy:.3f}  (chance = {1.0 / len(classes):.3f})')
    print('\nper class:')
    for class_name in classes:
        hits, count = per_class_correct[class_name]
        if count:
            print(f'  {class_name:10s} {hits}/{count}')

    # Chance level means the adapter did not actually load, whatever the training
    # log said.
    if accuracy < 2.0 / len(classes):
        print('\nBLOCKER: accuracy is near chance -- the adapter is probably '
              'not being applied by the scorer')
        return 1
    print('\nCGA + LoRA OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
