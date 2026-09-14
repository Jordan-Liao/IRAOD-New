#!/usr/bin/env python
"""Diff two mmcv configs after inheritance and ${...} resolution.

Comparing config *files* is misleading when one is self-contained and the other
inherits through a base: the meaningful comparison is what the runner actually
sees. This resolves both the way train.py does and reports only the keys that
differ.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from iraod_runtime import ensure_iraod_runtime  # noqa: E402

ensure_iraod_runtime()

from mmcv import Config  # noqa: E402

import mmdet_extension  # noqa: E402,F401
import sfod  # noqa: E402,F401
from sfod.utils import patch_config  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left')
    parser.add_argument('right')
    parser.add_argument('--corrupt', default='chaff')
    parser.add_argument(
        '--prefix', default='',
        help='only report keys under this dotted prefix, e.g. "model"')
    return parser.parse_args()


def flatten(value, prefix=''):
    if isinstance(value, dict):
        items = {}
        for key, item in value.items():
            items.update(flatten(item, f'{prefix}.{key}' if prefix else str(key)))
        return items
    if isinstance(value, (list, tuple)):
        # Lists of dicts are pipelines; index them so a reordering shows up.
        if any(isinstance(item, dict) for item in value):
            items = {}
            for index, item in enumerate(value):
                items.update(flatten(item, f'{prefix}[{index}]'))
            return items
        return {prefix: repr(value)}
    return {prefix: repr(value)}


def resolve(path, corrupt):
    cfg = Config.fromfile(path)
    cfg.corrupt = corrupt
    return patch_config(cfg)


def main():
    args = parse_args()
    left = flatten(resolve(args.left, args.corrupt)._cfg_dict.to_dict())
    right = flatten(resolve(args.right, args.corrupt)._cfg_dict.to_dict())

    keys = sorted(set(left) | set(right))
    if args.prefix:
        keys = [key for key in keys
                if key == args.prefix or key.startswith(args.prefix + '.')]

    differences = 0
    for key in keys:
        if key in ('cfg_name', 'filename'):
            continue
        lvalue = left.get(key, '<absent>')
        rvalue = right.get(key, '<absent>')
        if lvalue != rvalue:
            differences += 1
            print(f'{key}')
            print(f'  L: {lvalue}')
            print(f'  R: {rvalue}')

    print(f'\n{differences} differing keys')
    print(f'L = {args.left}')
    print(f'R = {args.right}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
