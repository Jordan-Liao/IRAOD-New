"""Rotated mAP50-95 from saved detections, CPU only.

AP at IoU 0.50:0.05:0.95 comes from the test dataset's own ``evaluate``
(``eval_rbbox_map`` for DOTA/RSAR and DIOR-R), so AP50 here is the exact value
the standard ``test.py --eval mAP`` path reports; a backfill can check it
against the stored AP50 before trusting mAP50-95.

    python tools/eval_map50_95.py TEST_CONFIG RESULTS.pkl [--out metrics.json] \
        [--cfg-options data.test.ann_file=... data.test.img_prefix=...]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

IOU_THRS = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))


# test.py strips these EvalHook-only keys from cfg.evaluation before evaluate().
_HOOK_KEYS = ('interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'rule',
              'dynamic_intervals', 'type')


def eval_kwargs_from_cfg(cfg):
    """The kwargs test.py --eval mAP passes to dataset.evaluate()."""
    kwargs = {k: v for k, v in dict(cfg.get('evaluation', None) or {}).items()
              if k not in _HOOK_KEYS}
    kwargs['metric'] = 'mAP'
    return kwargs


def rotated_map50_95(dataset, results, eval_kwargs=None):
    """Return {'AP50', 'mAP50-95', 'AP_per_iou'} using ``dataset.evaluate``."""
    aps = {}
    for thr in IOU_THRS:
        kwargs = dict(eval_kwargs or {'metric': 'mAP'}, iou_thr=thr)
        if thr != 0.5:
            kwargs['logger'] = 'silent'
        aps[f'{thr:.2f}'] = float(dataset.evaluate(results, **kwargs)['mAP'])
    return {'AP50': aps['0.50'], 'mAP50-95': float(np.mean(list(aps.values()))),
            'AP_per_iou': aps}


def main():
    import mmcv
    from mmcv import Config, DictAction
    from mmrotate.datasets import build_dataset
    from sfod.utils import patch_config

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('config', help='test config (same one test.py used)')
    parser.add_argument('results', help='test.py --out pickle')
    parser.add_argument('--out', help='write metrics json here')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    cfg = patch_config(cfg)
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    results = mmcv.load(args.results)
    if len(results) != len(dataset):
        raise ValueError(f'{len(results)} results for {len(dataset)} test images')
    metrics = rotated_map50_95(dataset, results, eval_kwargs_from_cfg(cfg))
    metrics.update(config=args.config, results=args.results, n_images=len(dataset))
    print(json.dumps(metrics, indent=1))
    if args.out:
        mmcv.dump(metrics, args.out)


if __name__ == '__main__':
    main()
