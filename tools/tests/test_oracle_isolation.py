"""CPU oracle integration: real student/CGA methods, tiny detector/VLM seams."""

from copy import deepcopy
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import mmcv
import torch
from torch import nn
from mmrotate.models.detectors.two_stage import RotatedTwoStageDetector

from experiments.comparison.labels import CLASSES
from experiments.comparison import oracle_adapters
from sfod.cga import CGA
from sfod.extensions import oracle
from sfod.oriented_rcnn_cga import OrientedRCNN_CGA
from sfod.rotated_unbiased_teacher import UnbiasedTeacher
from sfod.unbiased_teacher_vlst import UnbiasedTeacherVLST


ROOT = Path(__file__).resolve().parents[2]
VARIANTS = (oracle.OracleCGAStudent, oracle.OracleCGAVLSTStudent)


def tiny_detector_init(self, backbone, rpn_head, roi_head, train_cfg,
                       test_cfg, **kwargs):
    nn.Module.__init__(self)
    self.backbone = nn.Linear(4, 4)
    self.roi_head = nn.Module()
    self.roi_head.bbox_head = nn.Linear(4, 4)
    self.roi_head.bbox_head.num_classes = roi_head['bbox_head']['num_classes']
    self.train_cfg, self.test_cfg = train_cfg, test_cfg


class OracleIsolationTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = str(Path(temporary.name) / 'sarclip.bin')
        Path(self.base).touch()
        self.env = {
            'CGA_SCORER': 'sarclip', 'CGA_BACKEND': 'sarclip',
            'CGA_STRICT': '1', 'SARCLIP_PRETRAINED': self.base,
            'VLST_BACKEND': 'sarclip', 'CGA_FILTER_MODE': 'veto_soft',
            'CGA_VETO_PRED_THR': '0.7', 'CGA_VETO_LABEL_THR': '0.1',
            'CGA_PROTECT_DET_SCORE': '0.9', 'CGA_BLEND_DET_WEIGHT': '0.7',
            'CGA_EXCLUDE_IDS': '1,3', 'CGA_TAU': '77',
        }
        self.start(patch.dict(os.environ, self.env, clear=True))
        self.start(patch.object(torch.cuda, 'is_available', return_value=False))
        self.start(patch.object(
            RotatedTwoStageDetector, '__init__', tiny_detector_init))
        self.load_source = self.start(patch(
            'sfod.rotated_semi_base.load_checkpoint', return_value={}))
        self.build = self.start(patch(
            'sfod.rotated_semi_base.build_detector', side_effect=self.build_teacher))
        self.metadata = Mock(side_effect=lambda path, dataset, base: {
            'classes': list(CLASSES[dataset]), 'dataset': dataset})
        self.start(patch.object(oracle, 'inspect_oracle_adapter', self.metadata))
        self.start(patch.object(oracle_adapters, 'inspect_oracle_adapter', self.metadata))
        self.load_adapter = self.start(patch.object(
            oracle_adapters, 'load_adapter_checkpoint', side_effect=self.load_visual))
        self.attach = self.start(patch.object(
            oracle, 'attach_oracle_adapter',
            wraps=oracle_adapters.attach_oracle_adapter))
        self.ambient_load = self.start(patch(
            'sarclip_adapter.load_adapter_checkpoint',
            side_effect=AssertionError('No implicit LoRA loading')))
        self.sarclip = SimpleNamespace(
            create_model_with_args=Mock(side_effect=lambda *a, **k: nn.Linear(4, 4)),
            get_tokenizer=Mock(return_value=object()),
            build_zero_shot_classifier=Mock(
                side_effect=lambda model, **k: torch.ones(4, len(k['classnames']))),
            get_model_preprocess_cfg=Mock(return_value={}),
            image_transform=Mock(return_value=object()),
        )
        self.start(patch('sfod.cga._ensure_sarclip_importable', return_value=self.sarclip))
        self.optical = self.start(patch(
            'sfod.cga._ensure_clip_importable',
            side_effect=AssertionError('No optical CLIP fallback')))

    def start(self, patcher):
        value = patcher.start()
        self.addCleanup(patcher.stop)
        return value

    @staticmethod
    def load_visual(model, path, map_location):
        # Loading LoRA can introduce trainable tensors; the real attach helper
        # must freeze these too, not merely the pre-adapter encoder.
        model.oracle_visual = nn.Parameter(torch.ones(4))
        model.train()
        return {'adapter_type': 'lora', 'missing_keys': []}

    @staticmethod
    def teacher_config(dataset='RSAR'):
        return dict(
            model=dict(
                type='OrientedRCNN_CGA',
                backbone=dict(type='OrthoNet', depth=50, frozen_stages=1),
                neck=dict(type='FPN', out_channels=256),
                rpn_head=dict(type='OrientedRPNHead', anchor_generator=dict(scales=[8])),
                roi_head=dict(type='OrientedStandardRoIHead',
                              bbox_head=dict(type='RotatedShared2FCBBoxHead',
                                             num_classes=len(CLASSES[dataset]))),
                train_cfg=dict(rpn=dict(assigner=dict(pos_iou_thr=0.7))),
                test_cfg=dict(rcnn=dict(score_thr=0.05))),
            unrelated=dict(keep=['exactly', 'unchanged']))

    def cfg(self, dataset='RSAR'):
        return dict(
            oracle_dataset=dataset, oracle_adapter='/explicit/final_lora.pth',
            oracle_base_weights=self.base,
            strict_source_free=True, weight_l=0., weight_u=1.,
            use_bbox_reg=False, vlst_enabled=True, vlst_strict=True,
            vlst_lora_path=None, vlst_pretrained=self.base,
            vlst_vlm_dim=4, vlst_detector_dim=4)

    @staticmethod
    def build_teacher(config):
        options = deepcopy(config)
        kind = options.pop('type')
        cls = getattr(oracle, kind) if kind == 'OrientedRCNNOracleCGA' else OrientedRCNN_CGA
        return cls(**options)

    def student(self, variant=oracle.OracleCGAStudent, dataset='RSAR',
                cfg=None, ema_config=None, **overrides):
        detector = deepcopy(self.teacher_config(dataset)['model'])
        detector.pop('type')
        return variant(
            **detector, cfg=self.cfg(dataset) if cfg is None else cfg,
            ema_config=self.teacher_config(dataset) if ema_config is None else ema_config,
            **dict(dict(ema_ckpt='/unchanged/source_detector.pth'), **overrides))

    def test_wrap_preserves_architecture_source_config_and_detector_state(self):
        for dataset in CLASSES:
            for variant in VARIANTS:
                with self.subTest(dataset=dataset, variant=variant.__name__):
                    config = mmcv.Config(self.teacher_config(dataset))
                    before = deepcopy(config)
                    env = dict(os.environ)
                    model = self.student(variant, dataset, ema_config=config)
                    built = deepcopy(self.build.call_args.args[0])
                    self.assertEqual(built.pop('type'), 'OrientedRCNNOracleCGA')
                    for key in ('oracle_dataset', 'oracle_adapter', 'oracle_base_weights'):
                        self.assertEqual(built.pop(key), self.cfg(dataset)[key])
                    expected = deepcopy(before['model'])
                    expected.pop('type')
                    self.assertEqual(built, expected)
                    self.assertEqual(config._cfg_dict, before._cfg_dict)
                    self.assertEqual(dict(os.environ), env)
                    self.load_source.assert_called_with(
                        model.ema_model, '/unchanged/source_detector.pth', map_location='cpu')
                    self.assertEqual(model.fairness_group, 'Target-supervised')
                    self.assertTrue(model.strict_source_free)
                    self.assertEqual(model.weight_l, 0)
                    self.assertFalse(model.use_bbox_reg)
                    self.assertIsInstance(model.ema_model, OrientedRCNN_CGA)
                    self.assertEqual(model.ema_model.cga.class_names, list(CLASSES[dataset]))
                    raw = self.build_teacher(before['model'])
                    self.assertEqual(set(raw.state_dict()), set(model.ema_model.state_dict()))
                    self.assertEqual(model.ema_model.exclude_ids, [1, 3])
                    self.assertEqual(model.ema_model.cga_filter_mode, 'veto_soft')
                    self.assertEqual(model.ema_model.cga_veto_pred_thr, .7)
                    self.assertEqual(model.ema_model.cga_veto_label_thr, .1)
                    self.assertEqual(model.ema_model.cga_protect_det_score, .9)
                    self.assertEqual(model.ema_model.cga_blend_detector_weight, .7)
        self.ambient_load.assert_not_called()
        self.optical.assert_not_called()

    def test_string_ema_config_uses_mmcv_without_replacing_source_architecture(self):
        path = str(ROOT / 'configs/baseline/ema_config/'
                   'baseline_oriented_rcnn_ema_rsar_cga_orthonet.py')
        # CPU metadata loading must not import the production OrthoNet/timm.
        original_fromfile = mmcv.Config.fromfile
        with patch.object(mmcv.Config, 'fromfile', side_effect=lambda p:
                          original_fromfile(p, import_custom_modules=False)) as read:
            self.student(ema_config=path)
        read.assert_called_once_with(path)
        original = original_fromfile(path, import_custom_modules=False)['model']
        built = deepcopy(self.build.call_args.args[0])
        built['type'] = 'OrientedRCNN_CGA'
        for key in ('oracle_dataset', 'oracle_adapter', 'oracle_base_weights'):
            built.pop(key)
        self.assertEqual(built, original)

    def test_independent_frozen_vlst_and_cga_encoders_and_default_prompts(self):
        for dataset in CLASSES:
            with self.subTest(dataset=dataset):
                model = self.student(oracle.OracleCGAVLSTStudent, dataset)
                env = dict(os.environ)
                model._init_vlst_text_prototypes()
                teacher_vlm, vlst_vlm = model.ema_model.cga, model._vlst_vlm
                self.assertIsNot(teacher_vlm, vlst_vlm)
                self.assertIsNot(teacher_vlm.clip, vlst_vlm.clip)
                for scorer in (teacher_vlm, vlst_vlm):
                    self.assertTrue(scorer.strict)
                    self.assertEqual(scorer.backend, 'sarclip')
                    self.assertEqual(scorer.class_names, list(CLASSES[dataset]))
                    self.assertFalse(scorer.clip.training)
                    self.assertTrue(all(not p.requires_grad for p in scorer.clip.parameters()))
                # Existing inference tau, not the adapter's train-time scale.
                self.assertEqual(teacher_vlm.tau, 77.)
                self.assertEqual(vlst_vlm.tau, 100.)
                self.assertEqual(tuple(model.vlst_teacher.text_prototypes.shape),
                                 (len(CLASSES[dataset]), 4))
                self.assertEqual(dict(os.environ), env)
                calls = self.sarclip.build_zero_shot_classifier.call_args_list[-2:]
                for call in calls:
                    self.assertEqual([t('ship') for t in call.kwargs['templates']],
                                     ['A SAR image of a ship', 'This SAR patch shows a ship'])
                count = self.attach.call_count
                model._init_vlst_text_prototypes()
                self.assertEqual(self.attach.call_count, count)
                self.assertFalse(any('_vlst_vlm' in key or 'oracle_visual' in key
                                     for key in model.state_dict()))
        self.ambient_load.assert_not_called()

    def test_explicit_inference_templates_are_preserved(self):
        with patch.dict(os.environ, {'CGA_TEMPLATES': 'frozen {};other {}'}):
            model = self.student(oracle.OracleCGAVLSTStudent)
            model._init_vlst_text_prototypes()
        for call in self.sarclip.build_zero_shot_classifier.call_args_list:
            self.assertEqual([t('ship') for t in call.kwargs['templates']],
                             ['frozen ship', 'other ship'])

    def test_normal_models_never_attach_or_read_oracle_adapter(self):
        for variant in (UnbiasedTeacher, UnbiasedTeacherVLST):
            cfg = {k: v for k, v in self.cfg().items() if not k.startswith('oracle_')}
            model = self.student(variant, cfg=cfg)
            model.ema_model.cga, _ = model.ema_model._build_cga(6)
            if isinstance(model, UnbiasedTeacherVLST):
                model._init_vlst_text_prototypes()
        self.attach.assert_not_called()
        self.metadata.assert_not_called()
        self.load_adapter.assert_not_called()
        self.ambient_load.assert_not_called()

    def test_strict_environment_rejections_do_not_mutate_environment_or_load_vlm(self):
        bad_environments = [
            {'CGA_SCORER': ''}, {'CGA_SCORER': 'raw'}, {'CGA_BACKEND': 'clip'},
            {'CGA_STRICT': '0'}, {'SARCLIP_PRETRAINED': ''},
            {'SARCLIP_PRETRAINED': '/other/base'}, {'SARCLIP_LORA': '/ambient.pth'},
            {'SARCLIP_MODEL': 'ViT-L-14'},
        ]
        for variant in VARIANTS:
            for changes in bad_environments:
                with self.subTest(variant=variant.__name__, changes=changes):
                    with patch.dict(os.environ, changes):
                        before = dict(os.environ)
                        with self.assertRaises((ValueError, RuntimeError)):
                            self.student(variant)
                        self.assertEqual(dict(os.environ), before)
        self.sarclip.create_model_with_args.assert_not_called()
        self.attach.assert_not_called()

    def test_original_strict_cga_guards_remain_enforced(self):
        with patch.dict(os.environ, {'SARCLIP_LORA': '/ambient.pth'}):
            with self.assertRaisesRegex(RuntimeError, 'forbids SARCLIP_LORA'):
                CGA(CLASSES['RSAR'], backend='sarclip', strict=True, pretrained=self.base)
            with self.assertRaisesRegex(RuntimeError, 'forbids SARCLIP_LORA'):
                self.build_teacher(self.teacher_config()['model'])._build_cga(6)
        for path in ('', '/missing/base.pth'):
            with self.assertRaises((ValueError, FileNotFoundError)):
                CGA(CLASSES['RSAR'], backend='sarclip', strict=True, pretrained=path)
        self.sarclip.create_model_with_args.assert_not_called()
        self.ambient_load.assert_not_called()

    def test_required_recipe_fields_and_teacher_model_are_not_silently_replaced(self):
        for variant in VARIANTS:
            for key in ('oracle_dataset', 'oracle_adapter', 'oracle_base_weights'):
                cfg = self.cfg()
                cfg.pop(key)
                with self.subTest(variant=variant.__name__, missing=key):
                    with self.assertRaises(ValueError):
                        self.student(variant, cfg=cfg)
            for change in ({'strict_source_free': False}, {'weight_l': 1},
                           {'use_bbox_reg': True}, {'weight_u': 0}):
                cfg = dict(self.cfg(), **change)
                with self.assertRaises(ValueError):
                    self.student(variant, cfg=cfg)
            bad = self.teacher_config()
            bad['model']['type'] = 'OrientedRCNN'
            with self.assertRaisesRegex(ValueError, 'original OrientedRCNN_CGA'):
                self.student(variant, ema_config=bad)
            with self.assertRaisesRegex(ValueError, 'class counts'):
                self.student(variant, ema_config=self.teacher_config('DIOR'))
            with self.assertRaisesRegex(ValueError, 'source teacher'):
                self.student(variant, ema_ckpt=None)

    def test_vlst_cannot_disable_strictness_or_enable_another_adapter(self):
        for change in (
                {'vlst_enabled': False}, {'vlst_strict': False},
                {'vlst_lora_path': '/alternate.pth'}, {'vlst_lora_path': ''},
                {'vlst_pretrained': '/alternate/base.pth'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.student(oracle.OracleCGAVLSTStudent, cfg=dict(self.cfg(), **change))
        with patch.dict(os.environ, {'VLST_BACKEND': 'clip'}):
            with self.assertRaises(ValueError):
                self.student(oracle.OracleCGAVLSTStudent)

    def test_existing_adapter_validator_and_source_load_errors_propagate(self):
        for variant in VARIANTS:
            # Admission policy belongs to the existing validator, not this wrapper.
            for reason in ('wrong dataset', 'wrong base model', 'incomplete factors'):
                with patch.object(oracle, 'inspect_oracle_adapter',
                                  side_effect=ValueError(reason)):
                    with self.assertRaisesRegex(ValueError, reason):
                        self.student(variant)
            with patch('sfod.rotated_semi_base.load_checkpoint',
                       side_effect=ValueError('wrong source checkpoint')):
                with self.assertRaisesRegex(ValueError, 'wrong source checkpoint'):
                    self.student(variant)

    def test_adapter_and_actual_base_load_failures_abort_even_before_detections(self):
        for variant in VARIANTS:
            with patch.object(oracle_adapters, 'load_adapter_checkpoint',
                              side_effect=RuntimeError('adapter load failed')):
                with self.assertRaisesRegex(RuntimeError, 'adapter load failed'):
                    self.student(variant)
            with patch.object(self.sarclip, 'create_model_with_args',
                              side_effect=RuntimeError('base load failed')):
                with self.assertRaisesRegex(RuntimeError, 'base load failed'):
                    self.student(variant)
        self.optical.assert_not_called()

    def test_attach_rejects_class_order_and_unfilled_adapter(self):
        config = self.teacher_config()['model']
        raw = self.build_teacher(config)
        scorer, _ = raw._build_cga(6)
        scorer.class_names.reverse()
        with self.assertRaisesRegex(ValueError, 'class order'):
            oracle_adapters.attach_oracle_adapter(
                scorer, '/explicit.pth', 'RSAR', self.base)
        for info in ({'adapter_type': 'visual_proj', 'missing_keys': []},
                     {'adapter_type': 'lora',
                      'missing_keys': ['visual.layer.lora_up.weight']}):
            with patch.object(oracle_adapters, 'load_adapter_checkpoint', return_value=info):
                with self.assertRaisesRegex(ValueError, 'required LoRA factors'):
                    self.student()

    def test_inference_has_no_raw_fallback_and_preserves_all_return_shapes(self):
        for variant in VARIANTS:
            model = self.student(variant)
            for wrapped in (False, True):
                teacher = model.ema_model
                if wrapped:
                    model.ema_model = SimpleNamespace(module=teacher)
                for feat in (False, True):
                    for meta in (False, True):
                        expected = ('boxes', 'meta') if meta else 'boxes'

                        def infer(*args, **kwargs):
                            self.assertFalse(torch.is_grad_enabled())
                            self.assertEqual(kwargs, dict(
                                with_cga=True, rescale=False, return_cga_meta=meta))
                            return expected

                        with patch.object(teacher, 'simple_test', side_effect=infer):
                            self.assertEqual(model.inference_unlabeled(
                                'image', ['meta'], rescale=False,
                                return_feat=feat, return_cga_meta=meta), expected)
                with patch.dict(os.environ, {'CGA_STRICT': '0', 'CGA_SCORER': 'raw'}):
                    with patch.object(teacher, 'simple_test',
                                      side_effect=RuntimeError('scorer failed')) as run:
                        with self.assertRaisesRegex(RuntimeError, 'scorer failed'):
                            model.inference_unlabeled('image', ['meta'])
                        self.assertEqual(run.call_count, 1)

    def test_vlst_loading_failures_propagate_without_sharing_teacher_encoder(self):
        for failure in ('base', 'adapter', 'prototypes'):
            model = self.student(oracle.OracleCGAVLSTStudent)
            target = {
                'base': (self.sarclip, 'create_model_with_args'),
                'adapter': (oracle_adapters, 'load_adapter_checkpoint'),
                'prototypes': (CGA, 'text_prototype_matrix'),
            }[failure]
            with patch.object(*target, side_effect=RuntimeError(failure)):
                with self.assertRaisesRegex(RuntimeError, failure):
                    model._init_vlst_text_prototypes()
            self.assertIsNone(model.vlst_teacher.text_prototypes)
            self.assertIsNot(model._vlst_vlm, model.ema_model.cga)


if __name__ == '__main__':
    unittest.main()
