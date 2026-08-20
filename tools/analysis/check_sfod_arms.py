#!/usr/bin/env python
"""Print the settings of each SFOD ablation arm before a long run.

Building both arms is cheap and catches the failure modes that otherwise only
surface an hour into training: a stale ``load_from``, an SLRP module attached at
the wrong stage, or an unsupervised prefix that still points at clean images.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from iraod_runtime import ensure_iraod_runtime  # noqa: E402

ensure_iraod_runtime()

from mmcv import Config  # noqa: E402
from mmrotate.models import build_detector  # noqa: E402
from mmrotate.utils import compat_cfg  # noqa: E402

import mmdet_extension  # noqa: E402,F401
import sfod  # noqa: E402,F401
from sfod.utils import patch_config  # noqa: E402


ARMS = {
    'no-slrp': ('configs/unbiased_teacher/sfod/'
                'unbiased_teacher_oriented_rcnn_selftraining_cga_rsar_orthonet.py'),
    'slrp-maxpool': ('configs/unbiased_teacher/sfod/'
                     'unbiased_teacher_oriented_rcnn_selftraining_slrp_rsar_orthonet.py'),
    'slrp-deep': ('configs/unbiased_teacher/sfod/'
                  'unbiased_teacher_oriented_rcnn_selftraining_slrp_deep_rsar_orthonet.py'),
}


def main():
    corruption = sys.argv[1] if len(sys.argv) > 1 else 'noise_suppression'
    arms = sys.argv[2].split(',') if len(sys.argv) > 2 else list(ARMS)

    failures = []
    for name in arms:
        cfg_path = ARMS[name]
        try:
            cfg = Config.fromfile(cfg_path)
            # patch_config resolves ${corrupt}, so the value must be set first.
            cfg.corrupt = corruption
            cfg = patch_config(compat_cfg(cfg))
            model = build_detector(cfg.model)
            host = model.model if hasattr(model, 'model') else model
            backbone = host.backbone
            meta = (backbone.get_slrp_metadata()
                    if hasattr(backbone, 'get_slrp_metadata') else [])
            stages = [entry['feature_stage'] for entry in meta]
            params = sum(
                p.numel() for n, p in backbone.named_parameters()
                if '.slrp.' in n or n.startswith('slrp.'))
            load_from = cfg.get('load_from')
            resolved = Path(load_from).expanduser() if load_from else None
            exists = bool(resolved and resolved.is_file())

            print(f'--- {name}')
            print(f'    load_from   = {load_from}')
            print(f'    exists      = {exists}')
            print(f'    slrp stages = {stages} ({params} params)')
            print(f'    ema_ckpt    = {cfg.model.get("ema_ckpt")}')
            print(f'    epochs      = {cfg.runner.max_epochs}'
                  f'  lr = {cfg.optimizer.lr}'
                  f'  score_thr = {cfg.model.cfg.score_thr}')
            print(f'    unsup imgs  = {cfg.data.train.img_prefix_u}')
            print(f'    test imgs   = {cfg.data.test.img_prefix}')
            print(f'    work_dir    = {cfg.get("work_dir")}')
            if not exists:
                failures.append(f'{name}: load_from missing ({load_from})')
            if corruption not in str(cfg.data.test.img_prefix):
                failures.append(f'{name}: test prefix is not the corrupt split')
        except Exception as error:  # noqa: BLE001
            print(f'--- {name}\n    BUILD FAILED: {error}')
            failures.append(f'{name}: build failed')

    print()
    if failures:
        print('BLOCKERS:')
        for item in failures:
            print(f'  - {item}')
        return 1
    print('all arms OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
