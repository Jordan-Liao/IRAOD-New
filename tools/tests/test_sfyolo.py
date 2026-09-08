"""CPU checks for frozen TAM admission, image geometry and the SSM causal boundary."""

from pathlib import Path
import tempfile
import unittest

from mmcv import Config
from mmcv.runner import EpochBasedRunner, Hook
import torch
from torch import nn

from experiments.comparison.tam_artifacts import (
    BGR_MEAN, FORMAL_STEPS, SCHEMA, checkpoint_payload, load_completed_tam)
from sfod.extensions.proposal_teacher import AfterOptimizerTeacherHook
from sfod.extensions.sfyolo import SFYOLOOBB, StudentStabilizationHook, augment_detector_batch


ROOT = Path(__file__).resolve().parents[2]


class TinySFYOLO(SFYOLOOBB):
    def __init__(self, beta=.5):
        nn.Module.__init__(self)
        self.weight = nn.Parameter(torch.tensor([0.], dtype=torch.double))
        self.register_buffer('fixed_float', torch.tensor([3.]))
        self.register_buffer('counter', torch.tensor(7))
        teacher = nn.Module()
        teacher.weight = nn.Parameter(torch.tensor([0.], dtype=torch.double), requires_grad=False)
        teacher.register_buffer('fixed_float', torch.tensor([5.]))
        teacher.register_buffer('counter', torch.tensor(9))
        self.ema_model = teacher
        self.ssm_teacher_fraction = beta


class SFYOLOTest(unittest.TestCase):
    def test_centered_bgr_roundtrip_preserves_shapes_padding_and_metadata(self):
        class IdentityTAM:
            def encoder(self, x):
                return (x,)

            def transform(self, content, style, alpha):
                self.style = style.clone()
                self.alpha = alpha
                return content

            def decoder(self, x):
                return x

        tam = IdentityTAM()
        images = torch.randn(2, 3, 8, 10)
        metas = [dict(img_shape=(5, 7, 3), img_norm_cfg=dict(
            mean=[123., 117., 104.], std=[58., 57., 56.], to_rgb=True)) for _ in range(2)]
        before = images.clone()
        result = augment_detector_batch(tam, images, metas, style_index=1)
        torch.testing.assert_close(result, before)
        torch.testing.assert_close(images, before)
        raw_rgb = images[1, :, :5, :7] * torch.tensor([58., 57., 56.])[:, None, None]
        raw_rgb += torch.tensor([123., 117., 104.])[:, None, None]
        expected_style = raw_rgb.flip(0) - torch.tensor(BGR_MEAN)[:, None, None]
        torch.testing.assert_close(tam.style[0], expected_style)
        self.assertEqual(tam.alpha, .4)

    def test_parameter_ema_and_ssm_never_blend_fixed_or_integer_buffers(self):
        model = TinySFYOLO()
        with torch.no_grad():
            model.weight.fill_(2)
        model.update_ema_model(.999)
        self.assertAlmostEqual(model.ema_model.weight.item(), .002)
        model.stabilize_student()
        self.assertAlmostEqual(model.weight.item(), 1.001)
        self.assertEqual(model.fixed_float.item(), 3)
        self.assertEqual(model.counter.item(), 7)
        self.assertEqual(model.ema_model.fixed_float.item(), 5)
        self.assertEqual(model.ema_model.counter.item(), 9)

    def test_ssm_cannot_change_one_epoch_teacher_but_affects_later_epochs(self):
        class Step(Hook):
            def after_train_iter(self, runner):
                with torch.no_grad():
                    runner.model.weight.add_(1)

        def run(epochs, beta):
            runner = object.__new__(EpochBasedRunner)
            runner.model, runner._hooks = TinySFYOLO(beta), []
            runner.register_hook(StudentStabilizationHook(), priority=30)
            runner.register_hook(Step(), priority=40)
            runner.register_hook(AfterOptimizerTeacherHook(), priority=45)
            for epoch in range(epochs):
                runner._epoch = epoch
                runner.call_hook('before_train_epoch')
                runner.call_hook('after_train_iter')
            return runner.model.ema_model.weight.item()

        self.assertEqual(run(1, .5), run(1, 0))
        self.assertNotEqual(run(2, .5), run(2, 0))

    def test_detector_rejects_smoke_or_wrong_identity_tam_before_loading_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'tam.pth'
            identity = dict(dataset='RSAR', domain='chaff', seed=42)
            for status, steps, saved_identity in [
                    ('smoke_not_formal', 1, identity),
                    ('complete', FORMAL_STEPS, {**identity, 'seed': 43})]:
                torch.save(dict(schema=SCHEMA, status=status,
                                training_steps=steps, identity=saved_identity), path)
                with self.assertRaisesRegex(ValueError, 'completed TAM'):
                    load_completed_tam(path, identity, 'cpu')

    def test_payload_excludes_encoder_and_rejects_nonfinite_weights(self):
        class Parts(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(2, 2)
                self.decoder = nn.Linear(2, 2)
                self.F1, self.F2 = nn.Linear(2, 2), nn.Linear(2, 2)

        model = Parts()
        payload = checkpoint_payload(model, {}, '/external/vgg.pth', 1, 'code', '/images.json')
        self.assertEqual(set(payload['components']), {'decoder', 'F1', 'F2'})
        self.assertEqual(payload['status'], 'smoke_not_formal')
        with torch.no_grad():
            model.F1.weight.fill_(float('nan'))
        with self.assertRaisesRegex(FloatingPointError, 'nonfinite'):
            checkpoint_payload(model, {}, '/external/vgg.pth', FORMAL_STEPS, 'code', '/images.json')

    def test_configs_require_tam_and_keep_one_epoch_without_extra_teacher_update(self):
        for dataset in ('rsar', 'dior'):
            cfg = Config.fromfile(
                ROOT / f'configs/unbiased_teacher/sfod/extensions/sfyolo_{dataset}.py',
                import_custom_modules=False)
            self.assertEqual(cfg.model.type, 'SFYOLOOBB')
            self.assertIsNone(cfg.model.cfg.tam_checkpoint)
            self.assertEqual(cfg.runner.max_epochs, 1)
            self.assertEqual(cfg.data.samples_per_gpu, 32)
            self.assertEqual(cfg.optimizer.lr, .02)
            names = [h['type'] for h in cfg.custom_hooks]
            self.assertIn('StudentStabilizationHook', names)
            self.assertIn('AfterOptimizerTeacherHook', names)
            self.assertNotIn('EpochFinalTeacherHook', names)


if __name__ == '__main__':
    unittest.main()
