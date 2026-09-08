import ast
import importlib.util
import os
import runpy
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[3]
CGA_PATH = REPO_ROOT / "sfod" / "cga.py"
VLST_PATH = REPO_ROOT / "sfod" / "unbiased_teacher_vlst.py"
PROTOTYPE_PATH = (
    REPO_ROOT / "sfod" / "semantic_teacher" / "prototype_teacher.py"
)
STRICT_CONFIGS = (
    REPO_ROOT / "configs" / "unbiased_teacher" / "sfod"
    / "unbiased_teacher_oriented_rcnn_selftraining_cga_rsar_orthonet_arm_b.py",
    REPO_ROOT / "configs" / "unbiased_teacher" / "sfod"
    / "unbiased_teacher_oriented_rcnn_selftraining_vlst_rsar_orthonet.py",
    REPO_ROOT / "configs" / "unbiased_teacher" / "sfod"
    / "unbiased_teacher_oriented_rcnn_selftraining_vlst_cga_rsar_orthonet.py",
    REPO_ROOT / "configs" / "unbiased_teacher" / "sfod"
    / "unbiased_teacher_oriented_rcnn_selftraining_cga_rsar_orthonet_arm_b_strict.py",
    REPO_ROOT / "configs" / "unbiased_teacher" / "sfod"
    / "unbiased_teacher_oriented_rcnn_selftraining_vlst_rsar_orthonet_strict.py",
    REPO_ROOT / "configs" / "unbiased_teacher" / "sfod"
    / "unbiased_teacher_oriented_rcnn_selftraining_vlst_cga_rsar_orthonet_strict.py",
)
CLIP_STRICT_CONFIG = (
    REPO_ROOT / "configs" / "unbiased_teacher" / "sfod"
    / "unbiased_teacher_oriented_rcnn_selftraining_clip_cga_rsar_orthonet_strict.py"
)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CGA_MODULE = _load_module("iraod_strict_cga_under_test", CGA_PATH)
PROTOTYPE_MODULE = _load_module(
    "iraod_prototype_teacher_under_test", PROTOTYPE_PATH
)


class _FakeSarclipModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.Linear(4, 4)

    def forward(self, image=None, **_kwargs):
        return {"image_features": self.visual(image)}


def _fake_sarclip_module():
    module = types.ModuleType("sar_clip")
    module.__file__ = "<fake-sar-clip>"
    module.created_models = []

    def create_model_with_args(_model, device, **_kwargs):
        model = _FakeSarclipModel().to(device)
        module.created_models.append(model)
        return model

    def build_zero_shot_classifier(
            _model, classnames, device, **_kwargs):
        classifier = torch.eye(4, len(classnames), device=device)
        return classifier / classifier.norm(dim=0, keepdim=True)

    module.create_model_with_args = create_model_with_args
    module.get_tokenizer = lambda *_args, **_kwargs: object()
    module.build_zero_shot_classifier = build_zero_shot_classifier
    module.get_model_preprocess_cfg = lambda _model: {"size": 4}
    module.image_transform = lambda *_args, **_kwargs: (
        lambda _image: torch.ones(4)
    )
    return module


def _fake_clip_module():
    module = types.ModuleType("clip")
    module.__file__ = "<fake-clip>"

    class _FakeClipModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = nn.Linear(4, 4)

        def encode_text(self, tokens):
            return torch.ones(tokens.shape[0], 4)

    def load(_model, device="cpu"):
        return _FakeClipModel().to(device), (lambda image: image)

    module.load = load
    module.tokenize = lambda texts: torch.ones(len(texts), 4, dtype=torch.long)
    return module


def _build_strict_clip():
    fake_clip = _fake_clip_module()
    with patch.dict(sys.modules, {"clip": fake_clip}):
        return CGA_MODULE.CGA(
            class_names=["ship", "aircraft"],
            model="RN50x64",
            templates=["an aerial image of a {}"],
            backend="clip",
            strict=True,
        )


def _build_strict_cga(checkpoint):
    fake_sarclip = _fake_sarclip_module()
    with patch.dict(sys.modules, {"sar_clip": fake_sarclip}):
        with patch.dict(
                os.environ, {"SARCLIP_PRETRAINED": str(checkpoint)}, clear=True):
            return CGA_MODULE.CGA(
                class_names=["ship", "aircraft"],
                model="ViT-B-32",
                pretrained=str(checkpoint),
                cache_dir=str(checkpoint.parent),
                backend="sarclip",
                strict=True,
            )


def _load_vlst_module():
    fake_sfod = types.ModuleType("sfod")
    fake_sfod.__path__ = [str(REPO_ROOT / "sfod")]
    fake_semantic = types.ModuleType("sfod.semantic_teacher")
    fake_semantic.__path__ = [
        str(REPO_ROOT / "sfod" / "semantic_teacher")
    ]
    fake_rotated = types.ModuleType("sfod.rotated_unbiased_teacher")
    fake_rotated.UnbiasedTeacher = type("UnbiasedTeacher", (), {})

    fake_builder = types.ModuleType("mmdet.models.builder")

    class _Registry:
        @staticmethod
        def register_module():
            return lambda cls: cls

    fake_builder.DETECTORS = _Registry()
    fake_models = types.ModuleType("mmdet.models")
    fake_models.builder = fake_builder
    fake_mmdet = types.ModuleType("mmdet")
    fake_mmdet.models = fake_models

    modules = {
        "sfod": fake_sfod,
        "sfod.rotated_unbiased_teacher": fake_rotated,
        "sfod.semantic_teacher": fake_semantic,
        "sfod.semantic_teacher.prototype_teacher": PROTOTYPE_MODULE,
        "mmdet": fake_mmdet,
        "mmdet.models": fake_models,
        "mmdet.models.builder": fake_builder,
    }
    with patch.dict(sys.modules, modules):
        vlst_module = _load_module("sfod.unbiased_teacher_vlst", VLST_PATH)
    return vlst_module, fake_sfod


class TestStrictVLMFreeze(unittest.TestCase):
    def test_strict_cga_freezes_independent_sarclip_encoders(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "vit_b_32_model.safetensors"
            checkpoint.write_bytes(b"checkpoint")
            cga = _build_strict_cga(checkpoint)
            vlst = _build_strict_cga(checkpoint)

        for scorer in (cga, vlst):
            self.assertFalse(scorer.clip.training)
            self.assertEqual(
                [name for name, param in scorer.clip.named_parameters()
                 if param.requires_grad],
                [],
            )
        self.assertIsNot(cga.clip, vlst.clip)
        self.assertNotEqual(
            next(cga.clip.parameters()).data_ptr(),
            next(vlst.clip.parameters()).data_ptr(),
        )

    def test_strict_clip_encoder_is_frozen(self):
        scorer = _build_strict_clip()
        self.assertFalse(scorer.clip.training)
        self.assertEqual(
            [name for name, param in scorer.clip.named_parameters()
             if param.requires_grad],
            [],
        )
        self.assertIn("strict=self.cga_strict", CGA_PATH.read_text(encoding="utf-8"))

    def test_backward_keeps_vlm_grad_free_and_trains_student_projection(self):
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "vit_b_32_model.safetensors"
            checkpoint.write_bytes(b"checkpoint")
            scorer = _build_strict_cga(checkpoint)

        teacher = PROTOTYPE_MODULE.SemanticPrototypeTeacher(
            num_classes=2,
            vlm_dim=4,
            detector_dim=4,
            projection_hidden=3,
            score_threshold=0.0,
        )
        teacher.set_text_prototypes(scorer.text_prototype_matrix())
        student = nn.Linear(4, 4)
        pseudo_features = student(torch.randn(4, 4))
        pseudo_labels = torch.tensor([0, 1, 0, 1])
        pseudo_scores = torch.ones(4)

        loss, num_samples = teacher.compute_prototype_loss(
            pseudo_features, pseudo_labels, pseudo_scores)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(num_samples, 4)
        self.assertTrue(any(
            param.grad is not None and param.grad.abs().sum() > 0
            for param in student.parameters()
        ))
        self.assertTrue(any(
            param.grad is not None and param.grad.abs().sum() > 0
            for param in teacher.projection_head.parameters()
        ))
        self.assertTrue(all(
            param.grad is None for param in scorer.clip.parameters()
        ))

    def test_strict_cga_rejects_lora_and_missing_base_checkpoint(self):
        missing = REPO_ROOT / "missing-strict-sarclip.safetensors"
        with patch.dict(
                os.environ, {"SARCLIP_LORA": "/tmp/adapter.pth"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "forbids SARCLIP_LORA"):
                CGA_MODULE.CGA(
                    class_names=["ship"],
                    pretrained=str(missing),
                    backend="sarclip",
                    strict=True,
                )

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(
                    FileNotFoundError, "Strict SARCLIP base checkpoint"):
                CGA_MODULE.CGA(
                    class_names=["ship"],
                    pretrained=str(missing),
                    backend="sarclip",
                    strict=True,
                )

    def test_strict_vlst_builds_its_own_configured_encoder(self):
        vlst_module, fake_sfod = _load_vlst_module()

        class _ExistingCGA:
            def text_prototype_matrix(self):
                raise AssertionError("strict VLST reused the CGA encoder")

        class _VLSTTeacher:
            text_prototypes = None

            def set_text_prototypes(self, prototypes):
                self.text_prototypes = prototypes

        class _ConfiguredCGA:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            @staticmethod
            def text_prototype_matrix():
                return np.eye(2, 4, dtype=np.float32)

        fake_cga = types.ModuleType("sfod.cga")
        fake_cga.CGA = _ConfiguredCGA
        fake_cga.RSAR_CLASSES = ("ship", "aircraft")
        fake_cga.DIOR_CLASSES = tuple(f"class-{index}" for index in range(20))

        model = object.__new__(vlst_module.UnbiasedTeacherVLST)
        model.vlst_teacher = _VLSTTeacher()
        model.vlst_strict = True
        model.vlst_pretrained = "/weights/base.safetensors"
        model.vlst_cache_dir = "/weights"
        model.vlst_lora_path = None
        model._vlst_vlm = None
        model.num_classes = 2
        existing_cga = _ExistingCGA()
        model.ema_model = types.SimpleNamespace(
            module=types.SimpleNamespace(
                cga=existing_cga, CLASSES=("ship", "aircraft"))
        )

        with patch.dict(
                sys.modules, {"sfod": fake_sfod, "sfod.cga": fake_cga}):
            with patch.dict(
                    os.environ, {"VLST_BACKEND": "sarclip"}, clear=True):
                model._init_vlst_text_prototypes()

        self.assertIsInstance(model._vlst_vlm, _ConfiguredCGA)
        self.assertIsNot(model._vlst_vlm, existing_cga)
        self.assertTrue(model._vlst_vlm.kwargs["strict"])
        self.assertEqual(
            model._vlst_vlm.kwargs["pretrained"],
            "/weights/base.safetensors",
        )
        self.assertIsNotNone(model.vlst_teacher.text_prototypes)

    def test_prototype_path_uses_teacher_pseudo_outputs_not_target_gt(self):
        tree = ast.parse(VLST_PATH.read_text(encoding="utf-8"))
        class_node = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "UnbiasedTeacherVLST"
        )
        forward = next(
            node for node in class_node.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "forward_train_semi"
        )
        calls = {
            node.func.attr: [ast.unparse(arg) for arg in node.args]
            for node in ast.walk(forward)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {
                "_vlst_update_prototypes",
                "_forward_train_with_vlst",
            }
        }

        self.assertEqual(
            calls["_vlst_update_prototypes"],
            ["img_metas_unlabeled", "bbox_results"],
        )
        self.assertIn("gt_bboxes_pred", calls["_forward_train_with_vlst"])
        self.assertIn("gt_labels_pred", calls["_forward_train_with_vlst"])
        self.assertNotIn(
            "gt_bboxes_unlabeled", calls["_forward_train_with_vlst"])
        self.assertNotIn(
            "gt_labels_unlabeled", calls["_forward_train_with_vlst"])

    def test_strict_configs_require_base_and_never_enable_lora(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "vit_b_32_model.safetensors"
            checkpoint.write_bytes(b"checkpoint")
            for config_path in STRICT_CONFIGS:
                with self.subTest(config=config_path.name):
                    with patch.dict(
                            os.environ,
                            {"SARCLIP_PRETRAINED": str(checkpoint)},
                            clear=True):
                        config = runpy.run_path(str(config_path))
                        self.assertEqual(os.environ["CGA_STRICT"], "1")
                        self.assertNotIn("SARCLIP_LORA", os.environ)
                    source = config_path.read_text(encoding="utf-8")
                    self.assertNotIn(
                        "os.environ['SARCLIP_LORA'] =", source)
                    model = config.get("model")
                    if model and model.get("type") == "UnbiasedTeacherVLST":
                        self.assertTrue(model["cfg"]["vlst_strict"])
                        self.assertEqual(
                            model["cfg"]["vlst_pretrained"],
                            str(checkpoint),
                        )
                        self.assertIsNone(model["cfg"]["vlst_lora_path"])

    def test_clip_cga_strict_config_has_no_sarclip_or_vlst(self):
        with patch.dict(os.environ, {"SARCLIP_LORA": "/tmp/adapter.pth"}, clear=True):
            config = runpy.run_path(str(CLIP_STRICT_CONFIG))
            self.assertEqual(os.environ["CGA_SCORER"], "clip")
            self.assertEqual(os.environ["CGA_BACKEND"], "clip")
            self.assertEqual(os.environ["CGA_CLIP_MODEL"], "RN50x64")
            self.assertEqual(os.environ["CGA_FILTER_MODE"], "legacy")
            self.assertNotIn("SARCLIP_LORA", os.environ)
            self.assertNotIn("SARCLIP_PRETRAINED", os.environ)
            self.assertNotIn("VLST_BACKEND", os.environ)
            self.assertIsNone(config.get("model"))

    def test_strict_config_missing_base_failure_is_actionable(self):
        missing = REPO_ROOT / "missing-strict-config.safetensors"
        with patch.dict(
                os.environ, {"SARCLIP_PRETRAINED": str(missing)}, clear=True):
            with self.assertRaisesRegex(
                    FileNotFoundError, "base checkpoint does not exist"):
                runpy.run_path(str(STRICT_CONFIGS[0]))


if __name__ == "__main__":
    unittest.main()
