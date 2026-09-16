"""Opt-in first-invalid observer; never changes MDP computation or its schedule."""

import argparse
from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import runpy
import sys
import time
import traceback


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-root', type=Path, required=True)
    parser.add_argument('--finite-helper', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-updates', type=int, default=80)
    parser.add_argument('train_arguments', nargs=argparse.REMAINDER)
    options = parser.parse_args()
    if options.max_updates < 1:
        parser.error('--max-updates must be positive')
    return options


if __name__ == '__main__':
    options = arguments()
    options.code_root = options.code_root.resolve()
    options.finite_helper = options.finite_helper.resolve()
    options.output = options.output.resolve()
    sys.argv[0] = str(Path(__file__).resolve())
    os.chdir(options.code_root)
    sys.path.insert(0, str(options.code_root))
    from iraod_runtime import ensure_iraod_runtime
    ensure_iraod_runtime()

import numpy as np
import torch


class DiagnosticLimitReached(RuntimeError):
    pass


def copy_tree(value, *, to_cpu):
    if torch.is_tensor(value):
        return value.detach().to(device='cpu', copy=True) if to_cpu else value.detach()
    if isinstance(value, Mapping):
        return {key: copy_tree(item, to_cpu=to_cpu) for key, item in value.items()}
    if isinstance(value, list):
        return [copy_tree(item, to_cpu=to_cpu) for item in value]
    if isinstance(value, tuple):
        return tuple(copy_tree(item, to_cpu=to_cpu) for item in value)
    return deepcopy(value)


def rng_state():
    return {
        'python': random.getstate(), 'numpy': np.random.get_state(),
        'torch': torch.get_rng_state().clone(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
    }


def json_values(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, Mapping):
        return {key: json_values(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_values(item) for item in value]
    return value


def maximum(tensors):
    values = [tensor.detach().abs().max() for tensor in tensors
              if tensor is not None and tensor.numel()]
    return float(torch.stack(values).max()) if values else None


class FirstInvalidCapture:
    maximum = staticmethod(maximum)

    def __init__(self, output, finite, max_updates=80, anomaly_from=55):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.finite = finite
        self.max_updates, self.anomaly_from = max_updates, anomaly_from
        self.step = self.completed = 0
        self.model = self.optimizer = self.prestate = self.batch = None
        self.teacher_state = None
        self.handles = []
        self.anomaly = None
        self.failure_saved = False
        self.backward_started = False
        self.logs = {}
        self.started = time.monotonic()

    def begin(self, model, optimizer, batch):
        if self.model is None:
            self.model, self.optimizer = model, optimizer
            teacher = getattr(model, 'ema_model', None)
            if teacher is not None:
                teacher = getattr(teacher, 'module', teacher)
                self.teacher_state = copy_tree(teacher.state_dict(), to_cpu=True)
            self.handles.extend((
                optimizer.register_step_pre_hook(self.before_step),
                optimizer.register_step_post_hook(self.after_step),
            ))
            watched = {
                'mdp.afsp', 'mdp.afsp.style', 'mdp.transform', 'mdp.history',
                'backbone.layer1', 'backbone.layer2', 'backbone.layer3', 'backbone.layer4',
                'rpn_head', 'roi_head.bbox_head',
            }
            for name, module in model.named_modules():
                if name in watched:
                    self.handles.append(module.register_forward_hook(self.forward_hook(name)))
        if model is not self.model or optimizer is not self.optimizer:
            raise RuntimeError('The diagnostic is restricted to one model and optimizer')
        self.step += 1
        self.backward_started = False
        self.logs = {}
        self.batch = copy_tree(batch, to_cpu=False)
        self.prestate = {
            'model': copy_tree(model.state_dict(), to_cpu=True),
            'optimizer': copy_tree(optimizer.state_dict(), to_cpu=True),
            'rng': rng_state(),
            'python_state': {
                name: deepcopy(getattr(model, name))
                for name in ('cur_iter', 'image_num', 'pseudo_num', 'pseudo_sem_weight')
                if hasattr(model, name)
            },
        }
        if self.step == 1:
            self.save_capture('initial', filename='initial.pt')
        if self.step >= self.anomaly_from and self.anomaly is None:
            self.anomaly = torch.autograd.detect_anomaly(check_nan=True)
            self.anomaly.__enter__()

    def forward_hook(self, name):
        def observe(_module, inputs, output):
            if self.step < self.anomaly_from:
                return
            try:
                self.finite.require_finite(output, f'forward {name}')
            except FloatingPointError as error:
                self.fail('forward', error, {'module': name, 'inputs': inputs, 'output': output})
                raise
        return observe

    def before_step(self, optimizer, args, kwargs):
        try:
            self.finite._require_finite(
                ((name + '.grad', parameter.grad)
                 for name, parameter in self.model.named_parameters()
                 if parameter.grad is not None),
                'gradient before optimizer.step',
            )
        except FloatingPointError as error:
            self.fail('before_optimizer', error)
            raise
        self.logs['gradient_max_abs'] = self.maximum(
            parameter.grad for parameter in self.model.parameters())

    def after_step(self, optimizer, args, kwargs):
        try:
            self.finite.require_finite(
                {'model': self.model.state_dict(), 'optimizer': optimizer.state_dict()},
                'state after optimizer.step')
        except FloatingPointError as error:
            self.fail('after_optimizer', error)
            raise
        self.completed += 1
        self.logs.update(
            update=self.step, completed_updates=self.completed,
            parameter_max_abs=self.maximum(self.model.parameters()),
            elapsed_seconds=time.monotonic() - self.started)
        with (self.output / 'steps.jsonl').open('a') as stream:
            stream.write(json.dumps(json_values(self.logs)) + '\n')
        if self.completed >= self.max_updates:
            self.save_capture('finite_limit')
            summary = {
                'status': 'DIAGNOSTIC_LIMIT_WITHOUT_NONFINITE',
                'completed_updates': self.completed,
                'valid_full_budget_model': False,
                'capture': str(self.output / 'capture.pt'),
                'replay_scope': 'saved optimizer step only; no dataloader or epoch resume',
            }
            (self.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
            raise DiagnosticLimitReached('Diagnostic update boundary reached without a failure')

    def save_capture(self, stage, error=None, operands=None, filename='capture.pt'):
        payload = {
            'update': self.step, 'stage': stage, 'pre_forward': self.prestate,
            'teacher_state': self.teacher_state,
            'actual_batch': copy_tree(self.batch, to_cpu=True),
            'operands': copy_tree(operands, to_cpu=True),
            'gradients': {
                name: copy_tree(parameter.grad, to_cpu=True)
                for name, parameter in self.model.named_parameters()
                if self.backward_started and parameter.grad is not None
            },
            'backward_started': self.backward_started,
            'error': str(error) if error is not None else None,
        }
        torch.save(payload, self.output / filename)

    def fail(self, stage, error, operands=None):
        if self.failure_saved:
            return
        numeric = isinstance(error, FloatingPointError) or 'returned nan values' in str(error)
        self.save_capture(stage, error, operands)
        summary = {
            'status': 'CAPTURED_NONFINITE' if numeric else 'CAPTURED_RUNTIME_FAILURE',
            'update': self.step, 'completed_updates': self.completed, 'stage': stage,
            'module': operands['module'] if operands is not None else None,
            'exception_type': type(error).__name__, 'error': str(error),
            'losses': json_values(self.logs),
            'capture': str(self.output / 'capture.pt'),
            'elapsed_seconds': time.monotonic() - self.started,
            'valid_full_budget_model': False,
        }
        (self.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        (self.output / 'traceback.txt').write_text(traceback.format_exc())
        self.failure_saved = True
        print('[MDP-DIAG] ' + json.dumps(summary), flush=True)

    @contextmanager
    def install(self, model_class, optimizer_hook):
        own_train_step = 'train_step' in model_class.__dict__
        original_train_step = model_class.train_step
        original_after_iter = optimizer_hook.after_train_iter

        @wraps(original_train_step)
        def train_step(model, batch, optimizer, *args, **kwargs):
            self.begin(model, optimizer, batch)
            try:
                output = original_train_step(model, batch, optimizer, *args, **kwargs)
                self.logs.update(output.get('log_vars', {}))
                self.finite.require_finite(output['loss'], 'loss before backward')
                return output
            except (FloatingPointError, RuntimeError) as error:
                self.fail('forward_or_loss', error)
                raise

        @wraps(original_after_iter)
        def after_train_iter(hook, runner):
            self.backward_started = True
            try:
                return original_after_iter(hook, runner)
            except DiagnosticLimitReached:
                raise
            except (FloatingPointError, RuntimeError) as error:
                self.fail('backward_or_update', error)
                raise

        model_class.train_step = train_step
        optimizer_hook.after_train_iter = after_train_iter
        try:
            yield self
        finally:
            if own_train_step:
                model_class.train_step = original_train_step
            else:
                delattr(model_class, 'train_step')
            optimizer_hook.after_train_iter = original_after_iter
            for handle in self.handles:
                handle.remove()
            if self.anomaly is not None:
                self.anomaly.__exit__(None, None, None)


def run(options):
    from mmcv.runner import OptimizerHook
    from sfod.extensions.mdp import MDPOBB

    spec = importlib.util.spec_from_file_location('existing_finite_boundary', options.finite_helper)
    finite = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(finite)
    forwarded = options.train_arguments
    if forwarded[:1] == ['--']:
        forwarded = forwarded[1:]
    entry = options.code_root / 'train.py'
    sys.argv = [str(entry), *forwarded]
    observer = FirstInvalidCapture(options.output, finite, max_updates=options.max_updates)
    try:
        with observer.install(MDPOBB, OptimizerHook):
            runpy.run_path(str(entry), run_name='__main__')
    except DiagnosticLimitReached:
        return 3
    raise RuntimeError('Native training returned before the diagnostic boundary')


if __name__ == '__main__':
    raise SystemExit(run(options))
