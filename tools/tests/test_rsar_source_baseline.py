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
    DISTRIBUTED_SAMPLES_PER_GPU,
    DISTRIBUTED_WORLD_SIZE,
    EXPECTED_DISTRIBUTED_PHYSICAL_GPUS,
    OPTIMIZER_LR,
    REFERENCE_GLOBAL_BATCH,
    SEED,
    SINGLE_GPU_SAMPLES_PER_GPU,
    TOTAL_EPOCHS,
    SourceBaselineError,
    build_distributed_train_command,
    build_launch_metadata,
    build_smoke_command,
    build_train_command,
    final_epoch_checkpoint,
    parse_physical_gpus,
    resolve_rsar_root,
    run_finite_optimizer_steps,
    source_config_contract,
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
        self.assertEqual(
            config['data']['samples_per_gpu'],
            SINGLE_GPU_SAMPLES_PER_GPU,
        )
        self.assertEqual(config['optimizer']['lr'], OPTIMIZER_LR)
        self.assertEqual(config['runner']['max_epochs'], TOTAL_EPOCHS)

    def test_train_command_masks_to_one_logical_gpu_without_changes(self) -> None:
        command = build_train_command(
            '/opt/iraod/bin/python',
            'configs/baseline/oriented_rcnn_orthonet_rsar.py',
            self.root / 'run',
        )
        self.assertEqual(
            command,
            [
                '/opt/iraod/bin/python',
                'train.py',
                'configs/baseline/oriented_rcnn_orthonet_rsar.py',
                '--work-dir',
                str((self.root / 'run').resolve()),
                '--gpus',
                '1',
                '--seed',
                str(SEED),
                '--deterministic',
            ],
        )

    def test_distributed_command_preserves_global_batch_lr_and_seed(self) -> None:
        command = build_distributed_train_command(
            '/opt/iraod/bin/python',
            'configs/baseline/oriented_rcnn_orthonet_rsar.py',
            self.root / 'run',
            20067,
        )
        self.assertEqual(
            command[:5],
            [
                '/opt/iraod/bin/python',
                '-m',
                'torch.distributed.launch',
                f'--nproc_per_node={DISTRIBUTED_WORLD_SIZE}',
                '--master_port=20067',
            ],
        )
        self.assertEqual(command[command.index('--launcher') + 1], 'pytorch')
        self.assertEqual(command[command.index('--seed') + 1], str(SEED))
        self.assertIn('--deterministic', command)
        self.assertEqual(
            command[command.index('--cfg-options') + 1],
            f'data.samples_per_gpu={DISTRIBUTED_SAMPLES_PER_GPU}',
        )
        self.assertNotIn('optimizer.lr', json.dumps(command))
        self.assertNotIn('corruptions', json.dumps(command))
        self.assertNotIn('target', json.dumps(command))
        self.assertNotIn('rsar_source_baseline.py', command)
        self.assertNotIn('build_data_manifest.py', command)

        config_path = (
            Path(__file__).resolve().parents[2]
            / 'configs/baseline/oriented_rcnn_orthonet_rsar.py'
        )
        bindings = source_dataset_bindings(self.root)
        contract = source_config_contract(config_path, bindings)
        launch = build_launch_metadata(
            parse_physical_gpus('4,5,6,7'),
            command,
            contract,
        )
        self.assertTrue(launch['distributed'])
        self.assertEqual(launch['world_size'], DISTRIBUTED_WORLD_SIZE)
        self.assertEqual(
            launch['samples_per_gpu'],
            DISTRIBUTED_SAMPLES_PER_GPU,
        )
        self.assertEqual(
            launch['effective_global_batch'],
            REFERENCE_GLOBAL_BATCH,
        )
        self.assertEqual(launch['optimizer_lr'], OPTIMIZER_LR)
        self.assertEqual(launch['seed'], SEED)
        self.assertEqual(launch['total_epochs'], TOTAL_EPOCHS)
        self.assertEqual(launch['distributed_argv'], command)
        self.assertEqual(
            launch['logical_to_physical'],
            [
                {'logical_rank': 0, 'physical_gpu': 4},
                {'logical_rank': 1, 'physical_gpu': 5},
                {'logical_rank': 2, 'physical_gpu': 6},
                {'logical_rank': 3, 'physical_gpu': 7},
            ],
        )
        self.assertEqual(
            tuple(launch['physical_gpus']),
            EXPECTED_DISTRIBUTED_PHYSICAL_GPUS,
        )
        self.assertTrue(all(launch['invariants'].values()))

    def test_distributed_mapping_requires_four_unique_physical_gpus(self) -> None:
        self.assertEqual(parse_physical_gpus('4,5,6,7'), (4, 5, 6, 7))
        with self.assertRaisesRegex(SourceBaselineError, 'must be unique'):
            parse_physical_gpus('4,5,5,7')
        with self.assertRaisesRegex(SourceBaselineError, 'non-negative'):
            build_launch_metadata(
                (-1,),
                ['python', 'train.py', '--seed', '42', '--deterministic'],
                {
                    'clean_source_only': True,
                    'optimizer_lr': OPTIMIZER_LR,
                    'total_epochs': TOTAL_EPOCHS,
                },
            )
        with self.assertRaisesRegex(SourceBaselineError, 'exactly 4,5,6,7'):
            build_launch_metadata(
                parse_physical_gpus('7,6,5,4'),
                ['python', 'train.py', '--seed', '42', '--deterministic'],
                {
                    'clean_source_only': True,
                    'optimizer_lr': OPTIMIZER_LR,
                    'total_epochs': TOTAL_EPOCHS,
                },
            )
        with self.assertRaisesRegex(SourceBaselineError, 'exactly four GPUs'):
            build_launch_metadata(
                parse_physical_gpus('4,5'),
                ['python', 'train.py', '--seed', '42', '--deterministic'],
                {
                    'clean_source_only': True,
                    'optimizer_lr': OPTIMIZER_LR,
                    'total_epochs': TOTAL_EPOCHS,
                },
            )

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
            {
                'distributed': False,
                'world_size': 1,
            },
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
        self.assertEqual(
            json.loads(
                (output / 'launch_plan.json').read_text(encoding='utf-8')
            )['world_size'],
            1,
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
        self.assertIn('--gpus GPU,GPU,GPU,GPU', help_result.stdout)
        self.assertIn('--smoke-iters N', help_result.stdout)
        self.assertIn('epoch_100.pth', help_result.stdout)
        self.assertIn(
            'work_dirs/orthonet_rsar_source_seed42_4gpu',
            help_result.stdout,
        )

        wrong_mapping = subprocess.run(
            ['bash', str(script), '--gpus', '7,6,5,4'],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(wrong_mapping.returncode, 2)
        self.assertIn('exactly 4,5,6,7', wrong_mapping.stderr)
        script_text = script.read_text(encoding='utf-8')
        self.assertEqual(
            script_text.count(
                '"${PYTHON_BIN}" tools/rsar_source_baseline.py prepare'
            ),
            1,
        )
        self.assertEqual(
            script_text.count(
                '"${PYTHON_BIN}" tools/cga_research/build_data_manifest.py build'
            ),
            1,
        )
        self.assertNotRegex(script_text, r'\b(?:kill|pkill|killall)\b')

    def test_launcher_rejects_unsafe_distributed_run_roots_and_smoke(self) -> None:
        script = (
            Path(__file__).resolve().parents[2]
            / 'scripts/run_orthonet_rsar_source_seed42.sh'
        )
        environment = {
            **os.environ,
            'IRAOD_PYTHON': '/usr/bin/true',
            'RSAR_ROOT': str(self.root),
            'RUN_ROOT': 'work_dirs/orthonet_rsar_source_seed42',
        }
        reused_single_root = subprocess.run(
            ['bash', str(script), '--gpus', '4,5,6,7'],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(reused_single_root.returncode, 2)
        self.assertIn(
            'distributed RUN_ROOT must differ from the one-GPU full run',
            reused_single_root.stderr,
        )

        distributed_smoke = subprocess.run(
            [
                'bash',
                str(script),
                '--gpus',
                '4,5,6,7',
                '--smoke-iters',
                '4',
            ],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(distributed_smoke.returncode, 2)
        self.assertIn('one-GPU --gpu path', distributed_smoke.stderr)

        existing_run_root = self.root / 'existing-distributed-run'
        existing_run_root.mkdir()
        environment['RUN_ROOT'] = str(existing_run_root)
        reused_distributed_root = subprocess.run(
            ['bash', str(script), '--gpus', '4,5,6,7'],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(reused_distributed_root.returncode, 2)
        self.assertIn('refusing to reuse run directory', reused_distributed_root.stderr)

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
