"""Check copied MDP tensors on CPU without adding diagnostic GPU arithmetic."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-root', type=Path, required=True)
    parser.add_argument('--finite-helper', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-updates', type=int, required=True)
    parser.add_argument('train_arguments', nargs=argparse.REMAINDER)
    options = parser.parse_args()
    if options.max_updates < 1:
        parser.error('--max-updates must be positive')
    return options


if __name__ == '__main__':
    options = arguments()
    for name in ('code_root', 'finite_helper', 'output'):
        setattr(options, name, getattr(options, name).resolve())
    os.chdir(options.code_root)
    sys.path.insert(0, str(options.code_root))
    sys.argv[0] = str(Path(__file__).resolve())
    from iraod_runtime import ensure_iraod_runtime
    ensure_iraod_runtime()

import numpy as np

from mdp_first_invalid import DiagnosticLimitReached, FirstInvalidCapture


class CpuFinite:
    def __init__(self, native):
        self.native = native

    def _require_finite(self, named_tensors, boundary):
        for name, tensor in named_tensors:
            if tensor.is_floating_point() or tensor.is_complex():
                array = tensor.detach().cpu().numpy()
                if not np.isfinite(array).all():
                    raise FloatingPointError(f'Nonfinite {boundary}: {name}')

    def require_finite(self, value, boundary):
        self._require_finite(self.native._tensors(value, boundary), boundary)


class CpuFirstInvalidCapture(FirstInvalidCapture):
    def __init__(self, output, finite, max_updates):
        super().__init__(
            output, CpuFinite(finite), max_updates=max_updates, anomaly_from=float('inf'))

    @staticmethod
    def maximum(tensors):
        values = [
            float(np.abs(tensor.detach().cpu().numpy()).max())
            for tensor in tensors if tensor is not None and tensor.numel()
        ]
        return max(values) if values else None

    def begin(self, model, optimizer, batch):
        if self.completed >= self.max_updates:
            raise DiagnosticLimitReached('CPU diagnostic update boundary reached')
        super().begin(model, optimizer, batch)

    def after_step(self, optimizer, args, kwargs):
        try:
            super().after_step(optimizer, args, kwargs)
        except DiagnosticLimitReached:
            # Allow native epoch-end EMA/saving, but never another train_step.
            pass


def run(options):
    from mmcv.runner import OptimizerHook
    from sfod.extensions.mdp import MDPOBB

    spec = importlib.util.spec_from_file_location('mdp_cpu_finite_boundary', options.finite_helper)
    finite = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(finite)
    observer = CpuFirstInvalidCapture(options.output, finite, options.max_updates)
    forwarded = options.train_arguments
    if forwarded[:1] == ['--']:
        forwarded = forwarded[1:]
    entry = options.code_root / 'train.py'
    sys.argv = [str(entry), *forwarded]
    try:
        with observer.install(MDPOBB, OptimizerHook):
            runpy.run_path(str(entry), run_name='__main__')
    except DiagnosticLimitReached:
        return 3
    if observer.completed != options.max_updates:
        raise RuntimeError('Native training returned before the prescribed diagnostic budget')
    summary_path = options.output / 'summary.json'
    summary = json.loads(summary_path.read_text())
    summary.update(
        status='FINITE_NATIVE_TRAINING_RETURN',
        native_training_returned=True,
        final_pair_validation='pending_supervisor',
        diagnostic_checks='CPU tensor copies; no GPU finite/max kernels or anomaly detection',
    )
    summary_path.write_text(json.dumps(summary, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(run(options))
