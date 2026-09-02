from __future__ import annotations

import json
import os
import runpy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.rsar_source_baseline import (
    SEED,
    TOTAL_EPOCHS,
    SourceBaselineError,
    build_train_command,
    final_epoch_checkpoint,
    resolve_rsar_root,
    source_dataset_bindings,
    write_prelaunch_artifacts,
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


if __name__ == '__main__':
    unittest.main()
