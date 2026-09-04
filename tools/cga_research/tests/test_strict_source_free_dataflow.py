from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

try:
    from mmcv import Config
    from sfod.rotated_unbiased_teacher import UnbiasedTeacher
    from sfod.semi_dota_dataset import StrictSourceFreeDOTADataset
except ModuleNotFoundError:
    Config = None
    UnbiasedTeacher = None
    StrictSourceFreeDOTADataset = None


STRICT_CONFIG = (
    'configs/unbiased_teacher/sfod/'
    'unbiased_teacher_oriented_rcnn_selftraining_'
    'st_baseline_rsar_orthonet_strict.py')


class StrictSourceFreeStaticContractTests(unittest.TestCase):
    def test_strict_config_and_dataset_do_not_bind_annotations(self):
        config_text = Path(STRICT_CONFIG).read_text(encoding='utf-8')
        dataset_text = Path('sfod/semi_dota_dataset.py').read_text(
            encoding='utf-8')

        self.assertNotIn('ann_file=', config_text)
        self.assertNotIn("os.environ['CGA_SCORER']", config_text)
        self.assertIn("type='StrictSourceFreeDOTADataset'", config_text)
        class_text = dataset_text.split(
            'class StrictSourceFreeDOTADataset', 1)[1].split(
                '@ROTATED_DATASETS.register_module()', 1)[0]
        constructor = class_text.split('def __init__', 1)[1].split(
            '):', 1)[0]
        self.assertNotIn('ann_file', constructor)
        self.assertNotIn('ROTATED_DATASETS.build', class_text)

    def test_strict_model_branch_skips_supervised_loss_and_gt_diagnostics(self):
        model_text = Path('sfod/rotated_unbiased_teacher.py').read_text(
            encoding='utf-8')
        strict_branch = model_text.split(
            'if strict_source_free:', 1)[1].split(
                '# # -------------------unlabeled data', 1)[0]
        self.assertIn(
            'if strict_source_free:\n'
            '            losses = {}\n'
            '        else:\n'
            '            losses = self.forward_train(',
            strict_branch)
        self.assertIn(
            'None if strict_source_free else gt_bboxes_unlabeled', model_text)
        self.assertIn(
            'if proto_v2:\n'
            '                if not strict_source_free:', model_text)
        proto_v2_block = model_text.split(
            'if proto_v2:', 1)[1].split(
                'elif self.semantic_reweight:', 1)[0]
        self.assertIn(
            'self._prototype_bank_update(ema_host, strong_results)',
            proto_v2_block)


def _format_target(results):
    return {
        'img': results['img'],
        'img_metas': {
            'filename': str(Path(results['img_prefix'])
                            / results['img_info']['filename'])
        },
        'gt_bboxes': torch.from_numpy(results['gt_bboxes']),
        'gt_labels': torch.from_numpy(results['gt_labels']),
    }


@unittest.skipUnless(Config is not None, 'MMCV/MMRotate runtime is unavailable')
class StrictSourceFreeDataflowTests(unittest.TestCase):
    def test_config_binds_only_target_images_for_adaptation(self):
        cfg = Config.fromfile(STRICT_CONFIG)
        train = cfg.data.train

        self.assertEqual(train.type, 'StrictSourceFreeDOTADataset')
        self.assertIn(
            '/corruptions/${corrupt}/val/images/', train.img_prefix)
        self.assertNotIn('ann_file', train)
        self.assertNotIn('ann_file_u', train)
        self.assertNotIn('val', cfg.data)
        self.assertNotIn('test', cfg.data)
        self.assertTrue(cfg.model.cfg.strict_source_free)
        self.assertEqual(cfg.model.cfg.weight_l, 0.0)
        self.assertEqual(cfg.model.cfg.weight_u, 1.0)
        self.assertEqual(cfg.load_from, cfg.model.ema_ckpt)

    def test_dataset_enumerates_images_without_opening_annotations_or_source(self):
        with tempfile.TemporaryDirectory() as directory:
            image_root = Path(directory) / 'target'
            image_root.mkdir()
            (image_root / 'one.png').write_bytes(b'not decoded')
            (image_root / 'two.JPG').write_bytes(b'not decoded')

            def load_image(results):
                results['img'] = torch.zeros(3, 4, 4)
                return results

            with mock.patch(
                    'builtins.open',
                    side_effect=AssertionError('dataset must not open a file')):
                dataset = StrictSourceFreeDOTADataset(
                    img_prefix=str(image_root),
                    pipeline_share=[load_image],
                    pipeline_weak=[_format_target],
                    pipeline_strong=[_format_target],
                    classes=('ship',),
                )
                sample = dataset[0]

        self.assertEqual(len(dataset), 2)
        self.assertFalse(hasattr(dataset, 'dota_labeled'))
        self.assertFalse(hasattr(dataset, 'dota_unlabeled'))
        self.assertEqual(sample['gt_bboxes'].shape, (0, 5))
        self.assertEqual(sample['gt_labels'].shape, (0,))
        self.assertEqual(sample['gt_bboxes_unlabeled_1'].shape, (0, 5))
        self.assertEqual(sample['gt_labels_unlabeled_1'].shape, (0,))

    @staticmethod
    def _fake_teacher():
        teacher = SimpleNamespace(
            strict_source_free=True,
            image_num=0,
            cur_iter=0,
            momentum=0.998,
            weight_u=1.0,
            use_bbox_reg=False,
            semantic_reweight=False,
            pseudo_num=np.zeros(1),
            pseudo_num_tp=np.zeros(1),
            _assert_strict_empty_targets=(
                UnbiasedTeacher._assert_strict_empty_targets),
            update_ema_model=mock.Mock(),
            inference_unlabeled=mock.Mock(return_value=[[]]),
            create_pseudo_results=mock.Mock(return_value=(
                [torch.zeros(0, 5)], [torch.zeros(0, dtype=torch.long)])),
            analysis=mock.Mock(),
            forward_train=mock.Mock(return_value={
                'loss_cls': torch.tensor(2.0),
                'loss_bbox': torch.tensor(3.0),
            }),
            parse_loss=lambda losses: losses,
        )
        return teacher

    def test_strict_flow_skips_source_forward_and_hides_target_gt(self):
        teacher = self._fake_teacher()
        weak = torch.zeros(1, 3, 4, 4)
        strong = torch.ones(1, 3, 4, 4)
        empty_boxes = [torch.zeros(0, 5)]
        empty_labels = [torch.zeros(0, dtype=torch.long)]
        weak_metas = [dict(scale_factor=1.0)]
        strong_metas = [dict(scale_factor=1.0)]

        with mock.patch.dict(os.environ, {'CGA_FILTER_MODE': ''}):
            losses = UnbiasedTeacher.forward_train_semi(
                teacher,
                weak,
                weak_metas,
                empty_boxes,
                empty_labels,
                weak,
                weak_metas,
                empty_boxes,
                empty_labels,
                strong,
                strong_metas,
                empty_boxes,
                empty_labels,
            )

        teacher.forward_train.assert_called_once()
        self.assertIs(teacher.forward_train.call_args.args[0], strong)
        pseudo_call = teacher.create_pseudo_results.call_args.args
        self.assertEqual(pseudo_call[4:7], (None, None, None))
        self.assertNotIn('pseudo_num(acc)', losses)
        self.assertEqual(losses['loss_cls_unlabeled'].item(), 2.0)
        self.assertEqual(losses['loss_bbox_unlabeled'].item(), 0.0)

    def test_strict_flow_rejects_nonempty_target_ground_truth(self):
        teacher = self._fake_teacher()
        image = torch.zeros(1, 3, 4, 4)
        empty_boxes = [torch.zeros(0, 5)]
        empty_labels = [torch.zeros(0, dtype=torch.long)]
        nonempty_labels = [torch.ones(1, dtype=torch.long)]

        with self.assertRaisesRegex(
                ValueError, 'rejects nonempty target ground truth'):
            UnbiasedTeacher.forward_train_semi(
                teacher,
                image,
                [{}],
                empty_boxes,
                nonempty_labels,
                image,
                [{}],
                empty_boxes,
                nonempty_labels,
                image,
                [{}],
                empty_boxes,
                empty_labels,
            )
        teacher.update_ema_model.assert_not_called()
        teacher.forward_train.assert_not_called()


if __name__ == '__main__':
    unittest.main()
