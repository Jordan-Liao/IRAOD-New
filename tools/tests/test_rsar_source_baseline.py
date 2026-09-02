from __future__ import annotations

import json
import os
import runpy
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.rsar_source_baseline import (
    SEED,
    TOTAL_EPOCHS,
    SourceBaselineError,
    build_smoke_command,
    build_train_command,
    final_epoch_checkpoint,
    resolve_rsar_root,
    run_finite_optimizer_steps,
    source_dataset_bindings,
    write_prelaunch_artifacts,
    write_smoke_terminal_status,
)


class RsarSourceBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / 'RSAR'
        for relative in (
            'train/images',
            'train/annfiles',
            'val/images',
            'val/annfiles',
        ):
            (self.root / relative).mkdir(parents=True)
        self.root = self.root.resolve()

    def test_rsar_root_is_required_and_resolved(self) -> None:
        with self.assertRaisesRegex(SourceBaselineError, 'RSAR_ROOT is required'):
            resolve_rsar_root(None)
        self.assertEqual(resolve_rsar_root(self.root), self.root)

    def test_source_bindings_are_clean_train_and_val_only(self) -> None:
        bindings = source_dataset_bindings(self.root)
        self.assertEqual(bindings.train_images, self.root / 'train/images')
        self.assertEqual(bindings.train_annotations, self.root / 'train/annfiles')
        self.assertEqual(bindings.val_images, self.root / 'val/images')
        self.assertEqual(bindings.val_annotations, self.root / 'val/annfiles')
        self.assertNotIn('corruptions', json.dumps(bindings.to_dict()))

    def test_config_binds_only_clean_source_data_and_no_checkpoint(self) -> None:
        config_path = (
            Path(__file__).resolve().parents[2]
            / 'configs/baseline/oriented_rcnn_orthonet_rsar.py'
        )
        with patch.dict(os.environ, {'RSAR_ROOT': str(self.root)}, clear=True):
            config = runpy.run_path(str(config_path))

        self.assertEqual(config['train_img'], str(self.root / 'train/images') + '/')
        self.assertEqual(
            config['train_ann'], str(self.root / 'train/annfiles') + '/'
        )
        self.assertEqual(config['val_img'], str(self.root / 'val/images') + '/')
        self.assertNotIn('corruptions', json.dumps(config['data']['train']))
        self.assertIsNone(config['load_from'])

    def test_train_command_masks_to_one_logical_gpu(self) -> None:
        command = build_train_command(
            '/opt/iraod/bin/python',
            'configs/baseline/oriented_rcnn_orthonet_rsar.py',
            self.root / 'run',
        )
        self.assertEqual(command[0], '/opt/iraod/bin/python')
        self.assertEqual(command[1], 'train.py')
        self.assertEqual(command[command.index('--gpus') + 1], '1')
        self.assertEqual(command[command.index('--seed') + 1], str(SEED))
        self.assertIn('--deterministic', command)
        self.assertNotIn('--gpu-ids', command)
        self.assertNotIn('--cfg-options', command)
        self.assertNotIn('test.py', command)

    def test_smoke_command_uses_exact_optimizer_iterations(self) -> None:
        command = build_smoke_command(
            '/opt/iraod/bin/python',
            'configs/baseline/oriented_rcnn_orthonet_rsar.py',
            self.root / 'smoke',
            self.root,
            4,
        )
        self.assertEqual(command[0], '/opt/iraod/bin/python')
        self.assertEqual(command[1:3], ['tools/rsar_source_baseline.py', 'smoke'])
        self.assertEqual(command[command.index('--iterations') + 1], '4')
        self.assertEqual(command[command.index('--rsar-root') + 1], str(self.root))
        self.assertNotIn('train.py', command)
        self.assertNotIn('test.py', command)
        self.assertNotIn('corruptions', json.dumps(command))
        self.assertNotIn('target', json.dumps(command))
        with self.assertRaisesRegex(SourceBaselineError, 'must be positive'):
            build_smoke_command('python', 'config.py', self.root, self.root, 0)

    def test_smoke_runs_exact_finite_forward_backward_steps(self) -> None:
        class FakeFinite:
            def __init__(self, value: bool) -> None:
                self.value = value

            def all(self) -> 'FakeFinite':
                return self

            def item(self) -> bool:
                return self.value

        class FakeLoss:
            def __init__(self, model: 'FakeModel', finite: bool) -> None:
                self.model = model
                self.finite = finite

            def item(self) -> float:
                return 1.0

            def backward(self) -> None:
                self.model.backward_steps += 1

            def detach(self) -> float:
                return 1.0

        class FakeModel:
            def __init__(self, finite: bool = True) -> None:
                self.backward_steps = 0
                self.forward_steps = 0
                self.finite = finite

            def train_step(self, batch: object, optimizer: object) -> dict[str, FakeLoss]:
                del batch, optimizer
                self.forward_steps += 1
                return {'loss': FakeLoss(self, self.finite)}

            def named_parameters(self) -> list[tuple[str, object]]:
                return []

        class FakeOptimizer:
            def __init__(self) -> None:
                self.zero_grad_steps = 0
                self.steps = 0

            def zero_grad(self) -> None:
                self.zero_grad_steps += 1

            def step(self) -> None:
                self.steps += 1

        class FakeTorch:
            @staticmethod
            def isfinite(value: FakeLoss) -> FakeFinite:
                return FakeFinite(value.finite)

        model = FakeModel()
        optimizer = FakeOptimizer()
        losses = run_finite_optimizer_steps(
            model, optimizer, [object()], iterations=4, torch_module=FakeTorch
        )
        self.assertEqual(losses, [1.0, 1.0, 1.0, 1.0])
        self.assertEqual(model.forward_steps, 4)
        self.assertEqual(model.backward_steps, 4)
        self.assertEqual(optimizer.zero_grad_steps, 4)
        self.assertEqual(optimizer.steps, 4)

        nonfinite_model = FakeModel(finite=False)
        nonfinite_optimizer = FakeOptimizer()
        with self.assertRaisesRegex(FloatingPointError, 'non-finite training loss'):
            run_finite_optimizer_steps(
                nonfinite_model,
                nonfinite_optimizer,
                [object()],
                iterations=4,
                torch_module=FakeTorch,
            )
        self.assertEqual(nonfinite_model.backward_steps, 0)
        self.assertEqual(nonfinite_optimizer.steps, 0)

    def test_final_checkpoint_is_fixed_to_epoch_100(self) -> None:
        checkpoint = final_epoch_checkpoint(self.root / 'run')
        self.assertEqual(checkpoint.name, f'epoch_{TOTAL_EPOCHS}.pth')
        self.assertNotEqual(checkpoint.name, 'latest.pth')

    def test_prelaunch_artifacts_record_clean_bindings_and_command(self) -> None:
        output = self.root / 'run' / 'reproducibility'
        command = build_train_command(
            'python',
            'configs/baseline/oriented_rcnn_orthonet_rsar.py',
            self.root / 'run/train',
        )
        write_prelaunch_artifacts(
            output,
            source_dataset_bindings(self.root),
            command,
            {
                'CUDA_VISIBLE_DEVICES': '4',
                'RSAR_ROOT': str(self.root),
            },
            '2d2e60b275dcaaf7a1c8993d3341a33eb9754c9c',
        )
        bindings = json.loads(
            (output / 'source_dataset_bindings.json').read_text(encoding='utf-8')
        )
        self.assertEqual(bindings['train']['images'], str(self.root / 'train/images'))
        self.assertEqual(
            json.loads((output / 'command.json').read_text(encoding='utf-8'))['argv'],
            command,
        )
        self.assertEqual(
            json.loads(
                (output / 'launch_environment.json').read_text(encoding='utf-8')
            )['CUDA_VISIBLE_DEVICES'],
            '4',
        )

    def test_smoke_status_requires_exact_count_and_selects_no_checkpoint(self) -> None:
        output = self.root / 'smoke' / 'reproducibility'
        manifest = output / 'dataset_manifest.json'
        manifest.parent.mkdir(parents=True)
        manifest.write_text('{}\n', encoding='utf-8')
        smoke_result = self.root / 'smoke' / 'train' / 'smoke_result.json'
        smoke_result.parent.mkdir(parents=True)
        smoke_result.write_text(
            json.dumps({'optimizer_iterations': 4}) + '\n',
            encoding='utf-8',
        )

        self.assertTrue(
            write_smoke_terminal_status(
                output, manifest, smoke_result, exit_code=0, expected_iterations=4
            )
        )
        status = json.loads((output / 'run_result.json').read_text(encoding='utf-8'))
        self.assertEqual(status['optimizer_iterations'], 4)
        self.assertIsNone(status['final_checkpoint'])
        self.assertIsNone(status['selected_checkpoint'])
        self.assertIsNone(status['validation_best_checkpoint'])

        smoke_result.write_text(
            json.dumps({'optimizer_iterations': 3}) + '\n',
            encoding='utf-8',
        )
        self.assertFalse(
            write_smoke_terminal_status(
                output, manifest, smoke_result, exit_code=0, expected_iterations=4
            )
        )
        failed = json.loads((output / 'run_result.json').read_text(encoding='utf-8'))
        self.assertEqual(failed['status'], 'failed')
        self.assertIn('iteration mismatch', failed['failure_detail'])

    def test_launcher_rejects_positional_gpu_and_documents_smoke(self) -> None:
        script = (
            Path(__file__).resolve().parents[2]
            / 'scripts/run_orthonet_rsar_source_seed42.sh'
        )
        rejected = subprocess.run(
            ['bash', str(script), '4'],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(rejected.returncode, 2)
        self.assertIn('unknown argument: 4', rejected.stderr)
        self.assertIn('use --gpu GPU', rejected.stderr)

        help_result = subprocess.run(
            ['bash', str(script), '--help'],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(help_result.returncode, 0)
        self.assertIn('--gpu GPU', help_result.stdout)
        self.assertIn('--smoke-iters N', help_result.stdout)
        self.assertIn('epoch_100.pth', help_result.stdout)

    def test_launcher_requires_explicit_iraod_python(self) -> None:
        script = (
            Path(__file__).resolve().parents[2]
            / 'scripts/run_orthonet_rsar_source_seed42.sh'
        )
        environment = os.environ.copy()
        environment.pop('IRAOD_PYTHON', None)
        environment.pop('RSAR_ROOT', None)
        result = subprocess.run(
            ['bash', str(script), '--gpu', '6'],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('IRAOD_PYTHON', result.stderr)


if __name__ == '__main__':
    unittest.main()
