"""DDP log_vars key alignment for VLST diagnostics on empty vs nonempty ranks."""
from __future__ import annotations

import ast
import importlib.util
import sys
import types
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
VLST_PATH = REPO_ROOT / 'sfod' / 'unbiased_teacher_vlst.py'
PROTOTYPE_PATH = REPO_ROOT / 'sfod' / 'semantic_teacher' / 'prototype_teacher.py'

VLST_DIAG_KEYS = (
    'vlst_avg',
    'vlst_samples',
    'vlst_margin',
    'vlst_pos_cos',
    'vlst_proto_cls',
)

# Observed F DDP2 first-step split: rank0 13 keys, rank1 18 keys.
F_BASE_LOG_KEYS = (
    'loss_rpn_cls',
    'loss_rpn_bbox',
    'loss_cls',
    'acc',
    'loss_bbox',
    'loss_rpn_cls_unlabeled',
    'loss_rpn_bbox_unlabeled',
    'loss_cls_unlabeled',
    'acc_unlabeled',
    'loss_bbox_unlabeled',
    'loss_vlst_proto_unlabeled',
    'pseudo_num',
    'pseudo_num(acc)',
)

C_BASE_LOG_KEYS = (
    'loss_rpn_cls_unlabeled',
    'loss_rpn_bbox_unlabeled',
    'loss_cls_unlabeled',
    'acc_unlabeled',
    'loss_bbox_unlabeled',
    'pseudo_num',
)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROTOTYPE_MODULE = _load_module(
    'iraod_prototype_teacher_under_test', PROTOTYPE_PATH)


def _load_vlst_module():
    fake_sfod = types.ModuleType('sfod')
    fake_sfod.__path__ = [str(REPO_ROOT / 'sfod')]
    fake_semantic = types.ModuleType('sfod.semantic_teacher')
    fake_semantic.__path__ = [str(REPO_ROOT / 'sfod' / 'semantic_teacher')]
    fake_rotated = types.ModuleType('sfod.rotated_unbiased_teacher')
    fake_rotated.UnbiasedTeacher = type('UnbiasedTeacher', (), {})

    fake_builder = types.ModuleType('mmdet.models.builder')

    class _Registry:
        @staticmethod
        def register_module():
            return lambda cls: cls

    fake_builder.DETECTORS = _Registry()
    fake_models = types.ModuleType('mmdet.models')
    fake_models.builder = fake_builder
    fake_mmdet = types.ModuleType('mmdet')
    fake_mmdet.models = fake_models

    modules = {
        'sfod': fake_sfod,
        'sfod.rotated_unbiased_teacher': fake_rotated,
        'sfod.semantic_teacher': fake_semantic,
        'sfod.semantic_teacher.prototype_teacher': PROTOTYPE_MODULE,
        'mmdet': fake_mmdet,
        'mmdet.models': fake_models,
        'mmdet.models.builder': fake_builder,
    }
    with patch.dict(sys.modules, modules):
        vlst_module = _load_module('sfod.unbiased_teacher_vlst', VLST_PATH)
    return vlst_module


def _scalar_log_vars(keys, extra=None):
    log_vars = OrderedDict((key, torch.tensor(0.0)) for key in keys)
    if extra:
        log_vars.update(extra)
    return log_vars


def _mmdet_assert_log_var_keys_aligned(rank_log_vars):
    """Replica of mmdet BaseDetector._parse_losses DDP key-length assert."""
    lengths = [len(log_vars) for log_vars in rank_log_vars]
    reduced = sum(lengths)
    world = len(lengths)
    for rank, length in enumerate(lengths):
        if reduced != length * world:
            raise AssertionError(
                'loss log variables are different across GPUs: '
                f'rank{rank} {length} keys')


def _stub_vlst(loss_count, loss_sum=0.0, sample_sum=0.0, visual_counts=None):
    vlst_module = _load_vlst_module()
    model = object.__new__(vlst_module.UnbiasedTeacherVLST)
    model.vlst_enabled = True
    model.vlst_loss_count = loss_count
    model.vlst_loss_sum = loss_sum
    model.vlst_sample_sum = sample_sum
    teacher = PROTOTYPE_MODULE.SemanticPrototypeTeacher(
        num_classes=2,
        vlm_dim=4,
        detector_dim=4,
        projection_hidden=3,
        score_threshold=0.0,
    )
    if visual_counts is not None:
        teacher.visual_counts = visual_counts
        teacher.margin_sum = 0.4
        teacher.pos_cos_sum = 0.8
        teacher.margin_count = 2
    model.vlst_teacher = teacher
    return model


class TestVLSTDDPLogVars(unittest.TestCase):
    def test_consumer_rejects_observed_13_vs_18_split(self):
        rank0 = _scalar_log_vars(F_BASE_LOG_KEYS)
        rank1 = _scalar_log_vars(
            F_BASE_LOG_KEYS,
            extra={key: torch.tensor(1.0) for key in VLST_DIAG_KEYS})
        self.assertEqual(len(rank0), 13)
        self.assertEqual(len(rank1), 18)
        with self.assertRaisesRegex(
                AssertionError,
                'loss log variables are different across GPUs'):
            _mmdet_assert_log_var_keys_aligned([rank0, rank1])

    def test_empty_and_nonempty_pseudo_ranks_emit_the_same_vlst_keys(self):
        empty = _stub_vlst(loss_count=0)
        nonempty = _stub_vlst(
            loss_count=2,
            loss_sum=0.4,
            sample_sum=6.0,
            visual_counts=np.array([1, 0], dtype=np.int64),
        )
        device = torch.device('cpu')
        empty_vars = empty._vlst_diagnostic_log_vars(device)
        nonempty_vars = nonempty._vlst_diagnostic_log_vars(device)
        self.assertEqual(set(empty_vars), set(nonempty_vars))
        self.assertEqual(set(empty_vars), set(VLST_DIAG_KEYS))
        for key in VLST_DIAG_KEYS:
            self.assertNotIn('loss', key)
            self.assertTrue(torch.isfinite(empty_vars[key]).all())
            self.assertTrue(torch.isfinite(nonempty_vars[key]).all())
        self.assertEqual(empty_vars['vlst_avg'].item(), 0.0)
        self.assertEqual(empty_vars['vlst_samples'].item(), 0.0)
        self.assertAlmostEqual(nonempty_vars['vlst_avg'].item(), 0.2)
        self.assertAlmostEqual(nonempty_vars['vlst_samples'].item(), 3.0)
        self.assertEqual(nonempty_vars['vlst_proto_cls'].item(), 1.0)

        rank0 = _scalar_log_vars(F_BASE_LOG_KEYS, extra=empty_vars)
        rank1 = _scalar_log_vars(F_BASE_LOG_KEYS, extra=nonempty_vars)
        _mmdet_assert_log_var_keys_aligned([rank0, rank1])
        self.assertEqual(len(rank0), len(rank1))

    def test_forward_train_semi_merges_diagnostics_unconditionally(self):
        tree = ast.parse(VLST_PATH.read_text(encoding='utf-8'))
        class_node = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == 'UnbiasedTeacherVLST')
        forward = next(
            node for node in class_node.body
            if isinstance(node, ast.FunctionDef)
            and node.name == 'forward_train_semi')
        updates = [
            node for node in ast.walk(forward)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'update'
            and node.args
            and isinstance(node.args[0], ast.Call)
            and isinstance(node.args[0].func, ast.Attribute)
            and node.args[0].func.attr == '_vlst_diagnostic_log_vars']
        self.assertEqual(len(updates), 1)
        for parent in ast.walk(forward):
            if not isinstance(parent, ast.If):
                continue
            if ast.dump(parent.test).find('vlst_loss_count') != -1:
                self.fail('diagnostic merge is gated on vlst_loss_count')

    def test_strict_c_path_stays_aligned_without_vlst_diagnostics(self):
        rank0 = _scalar_log_vars(C_BASE_LOG_KEYS)
        rank1 = _scalar_log_vars(C_BASE_LOG_KEYS)
        _mmdet_assert_log_var_keys_aligned([rank0, rank1])
        for key in VLST_DIAG_KEYS:
            self.assertNotIn(key, rank0)
            self.assertNotIn(key, rank1)


if __name__ == '__main__':
    unittest.main()
