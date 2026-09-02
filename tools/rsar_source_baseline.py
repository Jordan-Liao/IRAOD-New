#!/usr/bin/env python3
"""Plan and record the portable clean-source RSAR OrthoNet baseline.

The artifacts intentionally use the established ``command.json``,
``launch_environment.json``, and ``run_result.json`` layout used by
``tools.cga_research.run_experiment``.  The final-checkpoint policy has a
specific trigger: validation metrics could otherwise select an intermediate
epoch.  Normal periodic checkpoints do not state that decision, so this module
binds the run to ``epoch_100.pth``; the experiment supervisor consumes the
recorded identity when reporting the run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


SEED = 42
TOTAL_EPOCHS = 100
CLASSES = ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor')


class SourceBaselineError(ValueError):
    """Raised when the source-only baseline cannot be bound to clean RSAR."""


@dataclass(frozen=True)
class SourceDatasetBindings:
    root: Path
    train_images: Path
    train_annotations: Path
    val_images: Path
    val_annotations: Path

    def to_dict(self) -> dict[str, str]:
        return {
            key: str(value)
            for key, value in asdict(self).items()
        }


def resolve_rsar_root(raw_root: str | Path | None) -> Path:
    """Resolve the explicitly supplied RSAR root."""

    if raw_root is None or not str(raw_root).strip():
        raise SourceBaselineError(
            'RSAR_ROOT is required and must point to the clean RSAR dataset.'
        )
    root = Path(raw_root).expanduser().resolve()
    if not root.is_dir():
        raise SourceBaselineError(f'RSAR_ROOT is not a directory: {root}')
    return root


def source_dataset_bindings(
    raw_root: str | Path | None,
) -> SourceDatasetBindings:
    """Return the only clean source paths the baseline is allowed to use."""

    root = resolve_rsar_root(raw_root)
    bindings = SourceDatasetBindings(
        root=root,
        train_images=root / 'train' / 'images',
        train_annotations=root / 'train' / 'annfiles',
        val_images=root / 'val' / 'images',
        val_annotations=root / 'val' / 'annfiles',
    )
    missing = [
        path for path in asdict(bindings).values()
        if path != root and not path.is_dir()
    ]
    if missing:
        missing_paths = ', '.join(str(path) for path in missing)
        raise SourceBaselineError(
            f'clean RSAR source paths are missing: {missing_paths}'
        )
    return bindings


def build_train_command(
    python: str | Path,
    config: str | Path,
    work_dir: str | Path,
) -> list[str]:
    """Build the deterministic non-distributed command for one masked GPU."""

    return [
        str(python),
        'train.py',
        str(config),
        '--work-dir',
        str(Path(work_dir).resolve()),
        '--gpus',
        '1',
        '--seed',
        str(SEED),
        '--deterministic',
    ]


def build_smoke_command(
    python: str | Path,
    config: str | Path,
    work_dir: str | Path,
    rsar_root: str | Path,
    iterations: int,
) -> list[str]:
    """Build the bounded clean-source smoke command for one masked GPU."""

    if iterations <= 0:
        raise SourceBaselineError('smoke iterations must be positive')
    return [
        str(python),
        'tools/rsar_source_baseline.py',
        'smoke',
        '--config',
        str(config),
        '--work-dir',
        str(Path(work_dir).resolve()),
        '--rsar-root',
        str(Path(rsar_root).resolve()),
        '--iterations',
        str(iterations),
    ]


def final_epoch_checkpoint(work_dir: str | Path) -> Path:
    """Return the checkpoint selected independently of validation metrics."""

    return Path(work_dir).resolve() / f'epoch_{TOTAL_EPOCHS}.pth'


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise SourceBaselineError(f'expected an object in {path}')
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise SourceBaselineError(f'expected file does not exist: {resolved}')
    return {
        'path': str(resolved),
        'sha256': _sha256(resolved),
        'size_bytes': resolved.stat().st_size,
    }


def write_prelaunch_artifacts(
    artifact_dir: str | Path,
    bindings: SourceDatasetBindings,
    command: Sequence[str],
    environment: dict[str, str],
    git_commit: str,
) -> None:
    """Record the resolved inputs before the training process starts."""

    output = Path(artifact_dir).resolve()
    _write_json(
        output / 'source_dataset_bindings.json',
        {
            'train': {
                'annotations': str(bindings.train_annotations),
                'images': str(bindings.train_images),
            },
            'val': {
                'annotations': str(bindings.val_annotations),
                'images': str(bindings.val_images),
            },
        },
    )
    _write_json(output / 'command.json', {'argv': list(command)})
    _write_json(output / 'launch_environment.json', environment)
    _write_json(output / 'git_commit.json', {'commit': git_commit.strip()})


def run_recorded_command(
    artifact_dir: str | Path,
    project_root: str | Path,
) -> int:
    """Run the pre-recorded argv without introducing shell interpretation."""

    payload = _read_json(Path(artifact_dir).resolve() / 'command.json')
    command = payload.get('argv')
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(argument, str) for argument in command)
    ):
        raise SourceBaselineError('command.json must contain a non-empty argv list')
    return subprocess.run(
        command,
        cwd=str(Path(project_root).resolve()),
        check=False,
    ).returncode


def write_terminal_status(
    artifact_dir: str | Path,
    work_dir: str | Path,
    dataset_manifest: str | Path,
    exit_code: int,
) -> bool:
    """Write terminal status and the identities of the run's final artifacts."""

    output = Path(artifact_dir).resolve()
    result: dict[str, Any] = {
        'dataset_manifest': _file_identity(Path(dataset_manifest)),
        'exit_code': int(exit_code),
        'final_checkpoint': None,
        'seed': SEED,
        'status': 'failed',
    }
    if exit_code == 0:
        checkpoint = final_epoch_checkpoint(work_dir)
        if checkpoint.is_file():
            result['final_checkpoint'] = _file_identity(checkpoint)
            result['status'] = 'succeeded'
        else:
            result['exit_code'] = 1
            result['failure_detail'] = (
                f'final epoch checkpoint missing: {checkpoint}'
            )
    _write_json(output / 'run_result.json', result)
    return result['status'] == 'succeeded'


def write_smoke_terminal_status(
    artifact_dir: str | Path,
    dataset_manifest: str | Path,
    smoke_result: str | Path,
    exit_code: int,
    expected_iterations: int,
) -> bool:
    """Record a successful smoke only after its exact optimizer-step count."""

    if expected_iterations <= 0:
        raise SourceBaselineError('expected smoke iterations must be positive')
    output = Path(artifact_dir).resolve()
    result: dict[str, Any] = {
        'dataset_manifest': _file_identity(Path(dataset_manifest)),
        'exit_code': int(exit_code),
        'final_checkpoint': None,
        'optimizer_iterations': None,
        'seed': SEED,
        'selected_checkpoint': None,
        'status': 'failed',
        'validation_best_checkpoint': None,
    }
    if exit_code == 0:
        payload = _read_json(Path(smoke_result))
        completed_iterations = payload.get('optimizer_iterations')
        if completed_iterations == expected_iterations:
            result['optimizer_iterations'] = completed_iterations
            result['status'] = 'succeeded'
        else:
            result['exit_code'] = 1
            result['failure_detail'] = (
                'smoke optimizer iteration mismatch: '
                f'expected {expected_iterations}, got {completed_iterations!r}'
            )
    _write_json(output / 'run_result.json', result)
    return result['status'] == 'succeeded'


def run_finite_optimizer_steps(
    model: Any,
    optimizer: Any,
    data_loader: Any,
    iterations: int,
    torch_module: Any,
) -> list[float]:
    """Perform an exact number of finite-loss forward/backward optimizer steps."""

    if iterations <= 0:
        raise SourceBaselineError('smoke iterations must be positive')

    losses: list[float] = []
    batches = iter(data_loader)
    for _ in range(iterations):
        try:
            batch = next(batches)
        except StopIteration:
            batches = iter(data_loader)
            try:
                batch = next(batches)
            except StopIteration as error:
                raise SourceBaselineError(
                    'smoke dataset must yield at least one batch'
                ) from error

        optimizer.zero_grad()
        outputs = model.train_step(batch, optimizer)
        loss = outputs['loss']
        if not torch_module.isfinite(loss).all().item():
            raise FloatingPointError(f'non-finite training loss: {loss.item()}')
        loss.backward()
        for name, parameter in model.named_parameters():
            if (
                parameter.grad is not None
                and not torch_module.isfinite(parameter.grad).all().item()
            ):
                raise FloatingPointError(f'non-finite gradient: {name}')
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


def run_bounded_smoke(
    config_path: str | Path,
    work_dir: str | Path,
    raw_rsar_root: str | Path,
    iterations: int,
) -> None:
    """Run exactly ``iterations`` finite-loss optimizer steps on clean RSAR."""

    if iterations <= 0:
        raise SourceBaselineError('smoke iterations must be positive')

    project_root = Path(__file__).resolve().parents[1]
    import sys

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from iraod_runtime import ensure_iraod_runtime

    ensure_iraod_runtime()

    import torch
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import build_optimizer
    from mmdet.datasets import build_dataloader
    from mmrotate.datasets import build_dataset
    from mmrotate.models import build_detector

    import mmdet_extension  # noqa: F401
    import sfod  # noqa: F401
    from sfod.utils import patch_config

    bindings = source_dataset_bindings(raw_rsar_root)
    os.environ['RSAR_ROOT'] = str(bindings.root)
    config = patch_config(Config.fromfile(str(config_path)))
    expected_paths = {
        'train.ann_file': bindings.train_annotations,
        'train.img_prefix': bindings.train_images,
        'val.ann_file': bindings.val_annotations,
        'val.img_prefix': bindings.val_images,
    }
    configured_paths = {
        'train.ann_file': Path(config.data.train.ann_file).resolve(),
        'train.img_prefix': Path(config.data.train.img_prefix).resolve(),
        'val.ann_file': Path(config.data.val.ann_file).resolve(),
        'val.img_prefix': Path(config.data.val.img_prefix).resolve(),
    }
    for name, expected_path in expected_paths.items():
        if configured_paths[name] != expected_path:
            raise SourceBaselineError(
                f'smoke config {name} is not clean RSAR: {configured_paths[name]}'
            )

    dataset = build_dataset(config.data.train)
    data_loader = build_dataloader(
        dataset,
        config.data.samples_per_gpu,
        config.data.workers_per_gpu,
        num_gpus=1,
        dist=False,
        shuffle=False,
        seed=SEED,
    )
    model = build_detector(
        config.model,
        train_cfg=config.get('train_cfg'),
        test_cfg=config.get('test_cfg'),
    )
    model.init_weights()
    optimizer = build_optimizer(model, config.optimizer)
    model = MMDataParallel(model.cuda(), device_ids=[0])
    model.train()

    losses = run_finite_optimizer_steps(
        model,
        optimizer,
        data_loader,
        iterations,
        torch,
    )

    output = Path(work_dir).resolve() / 'smoke_result.json'
    _write_json(
        output,
        {
            'losses': losses,
            'optimizer_iterations': len(losses),
            'seed': SEED,
        },
    )
    print(
        json.dumps(
            {
                'optimizer_iterations': len(losses),
                'status': 'succeeded',
            },
            sort_keys=True,
        )
    )


def _prepare(args: argparse.Namespace) -> int:
    bindings = source_dataset_bindings(args.rsar_root)
    if args.smoke_iters is None:
        command = build_train_command(args.python, args.config, args.work_dir)
    else:
        command = build_smoke_command(
            args.python,
            args.config,
            args.work_dir,
            bindings.root,
            args.smoke_iters,
        )
    environment = {
        'CUDA_VISIBLE_DEVICES': str(args.gpu_index),
        'MASTER_PORT': os.environ.get('MASTER_PORT', ''),
        'PYTHONNOUSERSITE': '1',
        'PYTHONUNBUFFERED': '1',
        'RSAR_ROOT': str(bindings.root),
    }
    write_prelaunch_artifacts(
        args.artifact_dir,
        bindings,
        command,
        environment,
        args.git_commit,
    )
    print(json.dumps({'status': 'prepared', 'command': command}, sort_keys=True))
    return 0


def _execute(args: argparse.Namespace) -> int:
    return run_recorded_command(args.artifact_dir, args.project_root)


def _finalize(args: argparse.Namespace) -> int:
    succeeded = write_terminal_status(
        args.artifact_dir,
        args.work_dir,
        args.dataset_manifest,
        args.exit_code,
    )
    print(
        json.dumps(
            {'status': 'succeeded' if succeeded else 'failed'},
            sort_keys=True,
        )
    )
    return 0 if succeeded or args.exit_code != 0 else 1


def _smoke(args: argparse.Namespace) -> int:
    run_bounded_smoke(
        args.config,
        args.work_dir,
        args.rsar_root,
        args.iterations,
    )
    return 0


def _finalize_smoke(args: argparse.Namespace) -> int:
    succeeded = write_smoke_terminal_status(
        args.artifact_dir,
        args.dataset_manifest,
        Path(args.work_dir) / 'smoke_result.json',
        args.exit_code,
        args.expected_iterations,
    )
    print(
        json.dumps(
            {'status': 'succeeded' if succeeded else 'failed'},
            sort_keys=True,
        )
    )
    return 0 if succeeded or args.exit_code != 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)

    prepare = subparsers.add_parser('prepare')
    prepare.add_argument('--rsar-root', required=True)
    prepare.add_argument('--python', required=True)
    prepare.add_argument('--config', required=True)
    prepare.add_argument('--work-dir', required=True)
    prepare.add_argument('--artifact-dir', required=True)
    prepare.add_argument('--gpu-index', type=int, required=True)
    prepare.add_argument('--git-commit', required=True)
    prepare.add_argument('--smoke-iters', type=int)
    prepare.set_defaults(handler=_prepare)

    execute = subparsers.add_parser('execute')
    execute.add_argument('--artifact-dir', required=True)
    execute.add_argument('--project-root', required=True)
    execute.set_defaults(handler=_execute)

    finalize = subparsers.add_parser('finalize')
    finalize.add_argument('--artifact-dir', required=True)
    finalize.add_argument('--work-dir', required=True)
    finalize.add_argument('--dataset-manifest', required=True)
    finalize.add_argument('--exit-code', type=int, required=True)
    finalize.set_defaults(handler=_finalize)

    smoke = subparsers.add_parser('smoke')
    smoke.add_argument('--config', required=True)
    smoke.add_argument('--work-dir', required=True)
    smoke.add_argument('--rsar-root', required=True)
    smoke.add_argument('--iterations', type=int, required=True)
    smoke.set_defaults(handler=_smoke)

    finalize_smoke = subparsers.add_parser('finalize-smoke')
    finalize_smoke.add_argument('--artifact-dir', required=True)
    finalize_smoke.add_argument('--work-dir', required=True)
    finalize_smoke.add_argument('--dataset-manifest', required=True)
    finalize_smoke.add_argument('--exit-code', type=int, required=True)
    finalize_smoke.add_argument('--expected-iterations', type=int, required=True)
    finalize_smoke.set_defaults(handler=_finalize_smoke)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == '__main__':
    raise SystemExit(main())
