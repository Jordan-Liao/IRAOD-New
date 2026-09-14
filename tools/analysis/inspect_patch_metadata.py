#!/usr/bin/env python
"""Check a SARCLIP patch metadata CSV before committing GPU hours to LoRA.

Two things silently ruin a LoRA run and neither shows up as an error: a class with
no patches (the contrastive objective then has nothing to separate it from), and a
corruption mix that is skewed (the adapter specialises on whichever corruption
dominates). Both are cheap to check and expensive to discover afterwards.
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('metadata')
    parser.add_argument(
        '--expected-classes', default='ship,aircraft,car,tank,bridge,harbor')
    parser.add_argument(
        '--expected-corruptions',
        default='am_noise_horizontal,am_noise_vertical,chaff,'
                'gaussian_white_noise,noise_suppression,point_target,'
                'smart_suppression')
    parser.add_argument(
        '--skew-tolerance', type=float, default=3.0,
        help='fail if the most common corruption exceeds the least by this factor')
    return parser.parse_args()


def main():
    args = parse_args()
    path = Path(args.metadata).expanduser().resolve()
    if not path.is_file():
        print(f'FATAL: {path} does not exist')
        return 1

    classes = Counter()
    corruptions = Counter()
    crop_modes = Counter()
    missing_files = 0
    total = 0
    with path.open(newline='') as handle:
        for row in csv.DictReader(handle):
            total += 1
            classes[row['class_name']] += 1
            corruptions[row['corruption']] += 1
            crop_modes[row['crop_mode']] += 1
            # Spot-check existence on a sample; stat-ing 4M files is not worth it.
            if total % 50000 == 0 and not Path(row['patch_path']).is_file():
                missing_files += 1

    print(f'total patches      : {total}')
    print(f'crop modes         : {dict(crop_modes)}')
    print(f'sampled missing    : {missing_files}')
    print()
    print('per class:')
    for name, count in classes.most_common():
        print(f'  {name:12s} {count:9d}  ({100.0 * count / total:5.2f}%)')
    print()
    print('per corruption:')
    for name, count in corruptions.most_common():
        print(f'  {name:22s} {count:9d}  ({100.0 * count / total:5.2f}%)')

    failures = []
    expected_classes = set(args.expected_classes.split(','))
    absent = sorted(expected_classes - set(classes))
    if absent:
        failures.append(f'classes with no patches: {absent}')
    expected_corruptions = set(args.expected_corruptions.split(','))
    absent = sorted(expected_corruptions - set(corruptions))
    if absent:
        failures.append(f'corruptions with no patches: {absent}')
    if corruptions:
        ratio = max(corruptions.values()) / max(1, min(corruptions.values()))
        if ratio > args.skew_tolerance:
            failures.append(
                f'corruption mix skewed {ratio:.1f}x '
                f'(tolerance {args.skew_tolerance:.1f}x)')
    if missing_files:
        failures.append(f'{missing_files} sampled patch files are missing')

    print()
    if failures:
        print('BLOCKERS:')
        for item in failures:
            print(f'  - {item}')
        return 1
    print('metadata OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
