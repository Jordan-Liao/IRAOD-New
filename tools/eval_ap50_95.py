"""AP50 and AP50:95 of a saved `test.py --out` result with the project evaluator.

Calls the config's test dataset `evaluate(metric='mAP', iou_thr=t)` for
t = 0.50, 0.55, ..., 0.95 and prints the per-threshold mAP and their mean.

Usage: python tools/eval_ap50_95.py CONFIG RESULTS.pkl [--out metrics.json]
"""
import argparse
import json

import mmcv
import numpy as np
from mmcv import Config
from mmrotate.datasets import build_dataset

from sfod.utils import patch_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('results')
    parser.add_argument('--out')
    args = parser.parse_args()

    cfg = patch_config(Config.fromfile(args.config))
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    results = mmcv.load(args.results)
    assert len(results) == len(dataset), (len(results), len(dataset))

    per_thr = {}
    for thr in np.round(np.arange(0.5, 0.96, 0.05), 2):
        per_thr[f'AP{int(round(thr * 100))}'] = float(
            dataset.evaluate(results, metric='mAP', iou_thr=float(thr))['mAP'])
    metrics = dict(AP50=per_thr['AP50'],
                   AP50_95=float(np.mean(list(per_thr.values()))),
                   per_iou=per_thr)
    print(json.dumps(metrics, indent=2))
    if args.out:
        mmcv.dump(metrics, args.out)


if __name__ == '__main__':
    main()
