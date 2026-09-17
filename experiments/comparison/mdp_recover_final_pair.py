"""Complete a retained last MDP step and its native epoch-final checkpoint pair."""

import argparse
from math import ceil
import os
from pathlib import Path
import runpy
import sys


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-root', type=Path, required=True)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('train_arguments', nargs=argparse.REMAINDER)
    return parser.parse_args()


if __name__ == '__main__':
    options = arguments()
    for name in ('code_root', 'capture', 'output'):
        setattr(options, name, getattr(options, name).resolve())
    os.chdir(options.code_root)
    sys.path.insert(0, str(options.code_root))
    sys.argv[0] = str(Path(__file__).resolve())
    from iraod_runtime import ensure_iraod_runtime
    ensure_iraod_runtime()

import json
import logging

import numpy as np
import torch
from mmcv.runner import OptimizerHook, build_optimizer
from mmdet_extension.core.runner.semi_runner import SemiEpochBasedRunner

import mdp_first_invalid as diagnostic
import numerical_integrity as finite
from sfod.extensions.proposal_teacher import EpochFinalTeacherHook


def finalize_epoch(runner, expected_updates):
    if runner.iter != expected_updates - 1 or runner.model.cur_iter != expected_updates:
        raise RuntimeError('Recovered step does not complete the prescribed epoch budget')
    runner._iter += 1
    hook = EpochFinalTeacherHook()
    hook.before_run(runner)
    hook.after_train_epoch(runner)
    with finite.native_boundaries():
        runner.save_checkpoint(runner.work_dir, create_symlink=False)


def cuda_tree(value):
    if torch.is_tensor(value):
        return value.cuda()
    if isinstance(value, dict):
        return {key: cuda_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cuda_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cuda_tree(item) for item in value)
    return value


def run(options):
    entry = options.code_root / 'train.py'
    namespace = runpy.run_path(str(entry), run_name='mdp_recovery_native')
    forwarded = options.train_arguments
    if forwarded[:1] == ['--']:
        forwarded = forwarded[1:]
    sys.argv = [str(entry), *forwarded]
    args = namespace['parse_args']()
    cfg = namespace['Config'].fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    namespace['setup_multi_processes'](cfg)
    cfg.device = namespace['get_device']()
    cfg.work_dir = str(options.output / 'work')
    cfg = namespace['patch_config'](cfg)
    if (cfg.runner.type != 'SemiEpochBasedRunner' or cfg.runner.max_epochs != 1
            or args.launcher != 'none' or args.gpus != 1
            or cfg.data.train.type != 'StrictSourceFreeDOTADataset'):
        raise RuntimeError('Recovery requires the retained one-epoch, one-device image-only protocol')
    expected_updates = ceil(cfg.data.train.unlabeled_epoch_size / cfg.data.samples_per_gpu)
    payload = torch.load(options.capture, map_location='cpu', weights_only=False)
    pre = payload['pre_forward']
    if (payload['stage'] != 'finite_limit' or payload['update'] != expected_updates
            or pre['python_state']['cur_iter'] != expected_updates - 1):
        raise RuntimeError('Capture is not the finite final step of this configured epoch')
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    namespace['set_random_seed'](args.seed, deterministic=args.deterministic)
    model = namespace['build_detector'](
        cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    source_teacher = model.ema_model.state_dict()
    if (source_teacher.keys() != payload['teacher_state'].keys()
            or any(not torch.equal(value, payload['teacher_state'][key])
                   for key, value in source_teacher.items())):
        raise RuntimeError('Captured frozen teacher differs from the configured source checkpoint')
    model.load_state_dict(pre['model'], strict=True)
    for name, value in pre['python_state'].items():
        setattr(model, name, value)
    model.set_epoch(0)
    model.CLASSES = cfg.classes
    model.train()
    model.ema_model.requires_grad_(False).eval()
    model.cuda()
    optimizer = build_optimizer(model, cfg.optimizer)
    optimizer.load_state_dict(pre['optimizer'])
    options.output.mkdir(parents=True, exist_ok=False)
    cfg.dump(str(options.output / 'mdp_rsar.py'))
    runner = SemiEpochBasedRunner(
        max_epochs=cfg.runner.max_epochs,
        model=model, optimizer=optimizer, work_dir=cfg.work_dir,
        logger=logging.getLogger('mdp-final-recovery'),
        meta=dict(
            seed=args.seed, config=cfg.pretty_text, CLASSES=cfg.classes,
            recovery_source=str(options.capture),
            prior_optimizer_updates=expected_updates - 1,
            recovered_optimizer_updates=1,
            original_nan_fixed=False,
        ))
    runner._iter = runner._inner_iter = expected_updates - 1
    runner._epoch = 0
    batch = cuda_tree(payload['actual_batch'])
    rng = pre['rng']
    diagnostic.random.setstate(rng['python'])
    np.random.set_state(rng['numpy'])
    torch.set_rng_state(rng['torch'])
    torch.cuda.set_rng_state_all(rng['cuda'])
    observer = diagnostic.FirstInvalidCapture(
        options.output / 'step_capture', finite, max_updates=1, anomaly_from=1)
    try:
        with observer.install(type(model), OptimizerHook):
            runner.outputs = model.train_step(batch, optimizer)
            OptimizerHook(**cfg.optimizer_config).after_train_iter(runner)
    except diagnostic.DiagnosticLimitReached:
        if observer.completed != 1:
            raise RuntimeError('Recovery did not complete exactly one optimizer update')
    else:
        raise RuntimeError('Recovery returned without its single-step boundary')
    finalize_epoch(runner, expected_updates)
    summary = {
        'status': 'RECOVERED_NATIVE_FINAL_PAIR',
        'completed_lineage_optimizer_updates': expected_updates,
        'new_optimizer_updates': 1,
        'source_teacher_exact_equal': True,
        'epoch_teacher_momentum': model.epoch_teacher_momentum,
        'student': str(Path(runner.work_dir) / f'iter_{expected_updates + 1}.pth'),
        'ema': str(Path(runner.work_dir) / f'iter_{expected_updates + 1}_ema.pth'),
        'saved_optimizer_lr': [group['lr'] for group in optimizer.param_groups],
        'step_loss': observer.logs['loss'],
        'original_nan_fixed': False,
        'test_or_roi_started': False,
        'recovery_source': str(options.capture),
    }
    (options.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('[MDP-RECOVERY] ' + json.dumps(summary), flush=True)


if __name__ == '__main__':
    run(options)
