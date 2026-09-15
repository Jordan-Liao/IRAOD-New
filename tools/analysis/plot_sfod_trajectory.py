#!/usr/bin/env python
"""Extract the iteration trajectory of an SFOD run from its training log.

The collapse is already recorded: pseudo_num, pseudo_num(acc) and the unsupervised
losses are logged every 10 iterations. Reading their shape says whether the failure
is a sudden divergence (a bad learning rate or a broken first step) or a slow drift
(self-training feedback), and those have different fixes.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

PATTERN = re.compile(
    r'Epoch \[(?P<epoch>\d+)\]\[(?P<iter>\d+)/(?P<total>\d+)\].*?'
    r'lr: (?P<lr>[\d.e+-]+)')
FIELD = re.compile(r'(?P<name>[a-zA-Z_()]+): (?P<value>[-\d.]+)')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logs', nargs='+')
    parser.add_argument('--buckets', type=int, default=12)
    parser.add_argument(
        '--fields',
        default='pseudo_num,pseudo_num(acc),loss_cls_unlabeled,'
                'loss_rpn_cls_unlabeled,acc_unlabeled')
    return parser.parse_args()


def read_log(path):
    rows = []
    seen = set()
    with Path(path).expanduser().open(errors='ignore') as handle:
        for line in handle:
            match = PATTERN.search(line)
            if not match:
                continue
            iteration = int(match.group('iter'))
            # Each iteration is logged twice (mmcv logger plus INFO duplicate).
            key = (iteration, line.count(','))
            if key in seen:
                continue
            seen.add(key)
            row = {'iter': iteration, 'lr': float(match.group('lr'))}
            tail = line[match.end():]
            for field in FIELD.finditer(tail):
                row[field.group('name')] = float(field.group('value'))
            rows.append(row)
    return rows


def main():
    args = parse_args()
    fields = [name.strip() for name in args.fields.split(',') if name.strip()]

    for path in args.logs:
        rows = read_log(path)
        if not rows:
            print(f'{path}: no iteration lines found')
            continue
        rows.sort(key=lambda row: row['iter'])
        iterations = np.asarray([row['iter'] for row in rows])
        total = iterations.max()
        edges = np.linspace(0, total, args.buckets + 1)

        print(f'=== {Path(path).name}  ({len(rows)} logged iterations, '
              f'max iter {total})')
        header = f'{"iters":>13s}'
        for name in fields:
            header += f'{name[:16]:>17s}'
        print(header)
        for index in range(args.buckets):
            low, high = edges[index], edges[index + 1]
            mask = (iterations > low) & (iterations <= high)
            if not mask.any():
                continue
            line = f'{int(low):5d}-{int(high):<7d}'
            for name in fields:
                values = [row.get(name) for row in rows
                          if low < row['iter'] <= high and name in row]
                values = [value for value in values if value is not None]
                line += (f'{np.mean(values):>17.4f}' if values
                         else f'{"-":>17s}')
            print(line)
        print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
