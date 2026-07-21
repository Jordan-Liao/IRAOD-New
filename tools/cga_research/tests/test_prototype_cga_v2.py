"""Calibration and integration tests for prototype_legacy_v2."""
import copy
import os
import sys

import numpy as np
import pytest
import torch
from PIL import Image


REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import sfod.cga as cga_module  # noqa: E402
import sfod.prototype_cga as prototype_module  # noqa: E402
from sfod.cga import CGA, TestMixins as _TestMixins  # noqa: E402
from sfod.prototype_cga import (  # noqa: E402
    VisualPrototypeBank,
    legacy_score_blend,
    mixed_prototype_matrix,
    prototype_legacy_v2_probabilities,
    softmax,
)
from sfod.rotated_semi_two_stage import SemiTwoStageDetector  # noqa: E402
from sfod.rotated_unbiased_teacher import UnbiasedTeacher  # noqa: E402


RTOL = 1e-6
ATOL = 1e-7


def _normalized(rng, shape):
    value = rng.normal(size=shape)
    return value / np.linalg.norm(value, axis=-1, keepdims=True)


def _fusion_fixture(seed=11, n=7, c=6, d=12, tau=100.0):
    rng = np.random.default_rng(seed)
    embedding = _normalized(rng, (n, d)).astype(np.float32)
    text = _normalized(rng, (c, d)).astype(np.float32)
    visual = _normalized(rng, (c, d)).astype(np.float32)
    text_sim = embedding @ text.T
    text_prob = softmax(tau * text_sim)
    return embedding, text, visual, text_sim, text_prob


def test_all_inactive_complete_probability_matrix_equals_legacy():
    embedding, text, visual, text_sim, text_prob = _fusion_fixture()
    active = np.zeros(len(text), dtype=bool)
    _, fused_sim, fused_logits, fused_prob = (
        prototype_legacy_v2_probabilities(
            embedding, text, visual, active, tau=100.0, beta=0.50,
            text_sim=text_sim, text_prob=text_prob))

    np.testing.assert_allclose(
        fused_sim, text_sim, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        fused_logits, 100.0 * text_sim, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        fused_prob, text_prob, rtol=RTOL, atol=ATOL)


def test_all_inactive_final_scores_equal_legacy():
    embedding, text, visual, text_sim, text_prob = _fusion_fixture()
    active = np.zeros(len(text), dtype=bool)
    _, _, _, fused_prob = prototype_legacy_v2_probabilities(
        embedding, text, visual, active, tau=100.0, beta=0.50,
        text_sim=text_sim, text_prob=text_prob)
    detector_scores = np.array(
        [0.99, 0.95, 0.93, 0.91, 0.89, 0.72, 0.51], dtype=np.float64)
    detector_labels = np.array([0, 1, 2, 3, 4, 5, 0], dtype=np.int64)
    legacy_scores, _, _ = legacy_score_blend(
        text_prob, detector_scores, detector_labels, detector_weight=0.70)
    v2_scores, _, _ = legacy_score_blend(
        fused_prob, detector_scores, detector_labels, detector_weight=0.70)

    np.testing.assert_allclose(
        v2_scores, legacy_scores, rtol=RTOL, atol=ATOL)


class _FakeClip:
    def __init__(self):
        self.calls = 0

    def __call__(self, image):
        self.calls += 1
        mean = image.mean(dim=(1, 2, 3))
        features = torch.stack(
            [mean + 0.5, mean * 0.5 + 1.0,
             mean * 0.25 + 1.5, mean * 0.125 + 2.0], dim=1)
        return {"image_features": features}


def _preprocess_for_crop(image):
    array = np.asarray(image.resize((8, 8)), dtype=np.float32).copy()
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    return torch.from_numpy(array).permute(2, 0, 1) / 255.0


def test_v2_aabb_forward_reuses_exact_legacy_crop(tmp_path, monkeypatch):
    image = np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3)
    image_path = tmp_path / "crop.png"
    Image.fromarray(image).save(image_path)

    cga = CGA.__new__(CGA)
    cga.backend = "sarclip"
    cga.class_names = ["a", "b", "c"]
    cga.device = torch.device("cpu")
    cga.expand_ratio = 0.4
    cga.force_grayscale = False
    cga.preprocess = _preprocess_for_crop
    cga.clip = _FakeClip()
    cga.tau = 100.0
    cga._first_call_logged = True
    classifier = torch.tensor(
        [[1.0, 0.0, 1.0],
         [0.0, 1.0, 1.0],
         [1.0, 1.0, 0.0],
         [0.5, 0.5, 1.0]], dtype=torch.float32)
    cga.classifier = classifier / classifier.norm(dim=0, keepdim=True)

    boxes = np.array([[2.2, 3.4, 18.8, 20.1],
                      [30.5, 28.2, 62.9, 60.7]], dtype=np.float32)
    scores = np.array([0.98, 0.91], dtype=np.float32)
    labels = np.array([0, 2], dtype=np.int64)
    crop_batches = []
    original_crop = cga._crop_patches

    def record_crop(*args, **kwargs):
        tensors, originals = original_crop(*args, **kwargs)
        crop_batches.append(tensors.detach().clone())
        return tensors, originals

    monkeypatch.setattr(cga, "_crop_patches", record_crop)
    legacy_prob, _ = cga(str(image_path), boxes, scores, labels)
    _, _, v2_text_prob = cga.forward_aabb_embed(
        str(image_path), boxes, scores, labels)

    assert cga.clip.calls == 2  # one batched encoder call per public forward
    assert len(crop_batches) == 2
    assert torch.equal(crop_batches[0], crop_batches[1])
    np.testing.assert_allclose(
        v2_text_prob, legacy_prob, rtol=RTOL, atol=ATOL)


class _FakeCGA:
    def __init__(self, num_classes=6, embed_dim=8):
        rng = np.random.default_rng(21)
        self.text = _normalized(rng, (num_classes, embed_dim)).astype(np.float32)
        self.tau = 100.0
        self.forward_boxes = None
        self.last_probability = None
        self.rotated_calls = 0

    def _values(self, num_proposals):
        rng = np.random.default_rng(22)
        embedding = _normalized(
            rng, (num_proposals, self.text.shape[1])).astype(np.float32)
        text_sim = embedding @ self.text.T
        text_prob = softmax(self.tau * text_sim)
        return embedding, text_sim, text_prob

    def __call__(self, filename, boxes, scores, labels):
        self.forward_boxes = np.asarray(boxes).copy()
        _, _, text_prob = self._values(len(boxes))
        self.last_probability = text_prob.copy()
        return text_prob, []

    def forward_aabb_embed(self, filename, boxes, scores, labels):
        self.forward_boxes = np.asarray(boxes).copy()
        embedding, text_sim, text_prob = self._values(len(boxes))
        self.last_probability = text_prob.copy()
        return embedding, text_sim, text_prob

    def forward_rotated_embed(self, *args, **kwargs):
        self.rotated_calls += 1
        raise AssertionError("v2 must not call rotated crop")

    def text_prototype_matrix(self):
        return self.text.copy()


def _image_results():
    return [
        np.array([[20.0, 20.0, 12.0, 6.0, 0.3, 0.98]], dtype=np.float32),
        np.empty((0, 6), dtype=np.float32),
        np.array([[42.0, 30.0, 8.0, 5.0, -0.2, 0.94]], dtype=np.float32),
        np.empty((0, 6), dtype=np.float32),
        np.empty((0, 6), dtype=np.float32),
        np.empty((0, 6), dtype=np.float32),
    ]


def _v2_host(strict=True):
    host = _TestMixins()
    host.cga_filter_mode = "prototype_legacy_v2"
    host.cga = _FakeCGA()
    host.exclude_ids = []
    host.cga_proto_beta = 0.50
    host.cga_proto_momentum = 0.95
    host.cga_proto_min_count = 20
    host.cga_blend_detector_weight = 0.70
    host.cga_strict = strict
    host.reset_proto_pending()
    return host


def _flat_scores(image_results):
    return np.concatenate([
        result[:, -1] for result in image_results if len(result)
    ])


def test_production_paths_inactive_v2_exactly_degrade_to_plain_legacy():
    """Exercise the two real score-write paths, not just fusion helpers."""
    source = _image_results()

    legacy = _TestMixins()
    legacy.cga_filter_mode = "legacy"
    legacy.cga = _FakeCGA()
    legacy.exclude_ids = []
    legacy.cga_blend_detector_weight = 0.70
    legacy_output = legacy._refine_single(
        copy.deepcopy(source), {"filename": "same.png"})

    v2 = _v2_host(strict=True)
    v2_output = v2._refine_single(
        copy.deepcopy(source), {"filename": "same.png"})
    pending = v2._proto_pending[0]

    # Full production probability matrix, scaled logits and predictions.
    np.testing.assert_allclose(
        pending["text_prob"], legacy.cga.last_probability,
        rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        pending["fused_prob"], legacy.cga.last_probability,
        rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        v2.cga.tau * pending["fused_sim"],
        v2.cga.tau * pending["text_sim"],
        rtol=RTOL, atol=ATOL)
    np.testing.assert_array_equal(
        pending["fused_prob"].argmax(axis=1),
        legacy.cga.last_probability.argmax(axis=1))

    # Scores after the actual per-class image_results write-back.
    np.testing.assert_allclose(
        _flat_scores(v2_output), _flat_scores(legacy_output),
        rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        pending["v2_score"], pending["legacy_score"],
        rtol=RTOL, atol=ATOL)


def test_first_v2_dispatch_uses_environment_before_cga_build(monkeypatch):
    """Regression: the first production weak image must not run as legacy."""
    monkeypatch.setenv("CGA_FILTER_MODE", "prototype_legacy_v2")
    host = _TestMixins()
    host.cga_strict = True
    host.cga_proto_beta = 0.50
    host.cga_proto_momentum = 0.95
    host.cga_proto_min_count = 20
    host.cga_blend_detector_weight = 0.70
    fake = _FakeCGA()
    build_calls = []

    def build(num_classes):
        build_calls.append(num_classes)
        host.cga_filter_mode = "prototype_legacy_v2"
        return fake, []

    host._build_cga = build
    empty = [np.empty((0, 6), dtype=np.float32) for _ in range(6)]
    host._refine_single(empty, {"filename": "empty-first.png"})
    host._refine_single(_image_results(), {"filename": "first-boxes.png"})

    assert build_calls == [6]
    assert len(host._proto_pending) == 2
    assert host._proto_pending[0] is None
    assert host._proto_pending[1]["filename"] == "first-boxes.png"


def test_v2_dispatch_uses_legacy_aabb_order_and_never_rotated_or_zscore(
        monkeypatch):
    host = _v2_host()
    original = _image_results()
    expected_boxes, _, _ = host._flatten_cga_inputs(copy.deepcopy(original))

    def forbidden(*args, **kwargs):
        raise AssertionError("v2 must not call z-score/rotated helpers")

    monkeypatch.setattr(cga_module, "_zscore_np", forbidden)
    monkeypatch.setattr(prototype_module, "_zscore", forbidden)
    monkeypatch.setattr(prototype_module, "rotated_align_crop", forbidden)
    host._refine_single(copy.deepcopy(original), {"filename": "unused.png"})

    np.testing.assert_array_equal(host.cga.forward_boxes, expected_boxes)
    assert host.cga.rotated_calls == 0
    pending = host._proto_pending[0]
    np.testing.assert_allclose(
        pending["fused_prob"], pending["text_prob"],
        rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        pending["v2_score"], pending["legacy_score"],
        rtol=RTOL, atol=ATOL)


def test_active_mixed_prototype_is_normalized_and_inactive_is_exact_text():
    _, text, visual, _, _ = _fusion_fixture()
    active = np.array([True, False, True, False, False, True])
    mixed = mixed_prototype_matrix(text, visual, active, beta=0.50)

    np.testing.assert_allclose(
        np.linalg.norm(mixed, axis=1), np.ones(len(text)),
        rtol=RTOL, atol=ATOL)
    np.testing.assert_array_equal(mixed[~active], text[~active])
    expected = text[0] * 0.50 + visual[0] * 0.50
    expected = expected / np.linalg.norm(expected)
    np.testing.assert_allclose(mixed[0], expected, rtol=RTOL, atol=ATOL)


def test_beta_zero_is_strict_legacy_degradation():
    embedding, text, visual, text_sim, text_prob = _fusion_fixture()
    active = np.ones(len(text), dtype=bool)
    mixed, fused_sim, fused_logits, fused_prob = (
        prototype_legacy_v2_probabilities(
            embedding, text, visual, active, tau=100.0, beta=0.0,
            text_sim=text_sim, text_prob=text_prob))

    np.testing.assert_array_equal(mixed, text)
    np.testing.assert_allclose(fused_sim, text_sim, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        fused_logits, 100.0 * text_sim, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(fused_prob, text_prob, rtol=RTOL, atol=ATOL)


def test_visual_equal_text_is_strict_legacy_degradation():
    embedding, text, _, text_sim, text_prob = _fusion_fixture()
    active = np.ones(len(text), dtype=bool)
    mixed, fused_sim, _, fused_prob = prototype_legacy_v2_probabilities(
        embedding, text, text.copy(), active, tau=100.0, beta=0.50,
        text_sim=text_sim, text_prob=text_prob)

    np.testing.assert_array_equal(mixed, text)
    np.testing.assert_allclose(fused_sim, text_sim, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(fused_prob, text_prob, rtol=RTOL, atol=ATOL)


def test_strict_mode_propagates_aabb_forward_exception():
    host = _v2_host(strict=True)

    def fail(*args, **kwargs):
        raise RuntimeError("forced crop failure")

    host.cga.forward_aabb_embed = fail
    with pytest.raises(RuntimeError, match="forced crop failure"):
        host._refine_single(_image_results(), {"filename": "unused.png"})


def test_strict_mode_is_not_swallowed_by_teacher_fallback(monkeypatch):
    class FailingEMA:
        def __init__(self):
            self.calls = 0

        def simple_test(self, *args, **kwargs):
            self.calls += 1
            raise RuntimeError("strict weak failure")

    class Harness:
        def __init__(self):
            self.ema_model = FailingEMA()

    monkeypatch.setenv("CGA_SCORER", "sarclip")
    monkeypatch.setenv("CGA_FILTER_MODE", "prototype_legacy_v2")
    monkeypatch.setenv("CGA_STRICT", "1")
    harness = Harness()
    with pytest.raises(RuntimeError, match="strict weak failure"):
        SemiTwoStageDetector.inference_unlabeled(
            harness, torch.zeros((1, 3, 8, 8)), [{}], rescale=True)
    assert harness.ema_model.calls == 1


def test_nonfinite_input_is_rejected_and_finite_output_has_no_nan_inf():
    embedding, text, visual, text_sim, text_prob = _fusion_fixture()
    active = np.array([True, False, False, False, False, False])
    _, _, logits, probability = prototype_legacy_v2_probabilities(
        embedding, text, visual, active, tau=100.0, beta=0.50,
        text_sim=text_sim, text_prob=text_prob)
    assert np.all(np.isfinite(logits))
    assert np.all(np.isfinite(probability))

    bad_embedding = embedding.copy()
    bad_embedding[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN/Inf"):
        prototype_legacy_v2_probabilities(
            bad_embedding, text, visual, active, tau=100.0,
            text_sim=text_sim, text_prob=text_prob)


def test_visual_bank_ema_min_count_and_no_self_inclusion_still_hold():
    rng = np.random.default_rng(31)
    bank = VisualPrototypeBank(
        num_classes=6, embed_dim=8, momentum=0.95, min_count=20)
    first = _normalized(rng, (10, 8))
    second = _normalized(rng, (10, 8))
    bank.update({0: first}, cur_iter=1)
    assert not bank.is_active(0)
    pre_update = bank.matrix().copy()
    bank.snapshot_previous()
    bank.update({0: second}, cur_iter=2)
    assert bank.is_active(0)
    assert bank.prototype_count[0] == 20
    assert bank.prototype_update_count[0] == 2
    assert bank.first_active_iteration[0] == 2
    np.testing.assert_allclose(
        bank.previous_prototype[0], pre_update[0], rtol=RTOL, atol=ATOL)


def test_gt_hit_rate_diagnostic_cannot_change_prototype_bank():
    host = _v2_host(strict=True)
    host._refine_single(_image_results(), {"filename": "same.png"})
    bank = host._proto_bank
    item = host._proto_pending[0]
    before_counts = bank.prototype_count.copy()
    before_updates = bank.prototype_update_count.copy()
    before_prototypes = copy.deepcopy(bank.prototype)

    class DiagnosticHarness:
        score_thr = 0.0
        dynamic_threshold_enabled = False

        def _pseudo_score_threshold(self, class_id):
            return self.score_thr

    gt_boxes = [torch.as_tensor(item["obb"][:1], dtype=torch.float32)]
    gt_labels = [torch.as_tensor(item["label"][:1], dtype=torch.long)]
    metas = [{"scale_factor": np.ones(4, dtype=np.float32)}]
    UnbiasedTeacher._accumulate_prototype_v2_paired_diagnostics(
        DiagnosticHarness(), host, gt_boxes, gt_labels, metas)

    np.testing.assert_array_equal(bank.prototype_count, before_counts)
    np.testing.assert_array_equal(bank.prototype_update_count, before_updates)
    assert bank.prototype == before_prototypes
    assert host._proto_diag["legacy_hits"] == 1
    assert host._proto_diag["v2_hits"] == 1


class _StrongTeacher:
    def __init__(self, events):
        self.events = events
        self.training = True

    def eval(self):
        self.training = False
        self.events.append("ema_eval")
        return self

    def reset_proto_pending(self):
        self.events.append("pending_reset")

    def simple_test(self, img, img_metas, rescale=True):
        self.events.append(
            ("strong", torch.is_grad_enabled(), self.training))
        return [[np.empty((0, 6), dtype=np.float32) for _ in range(6)]
                for _ in img_metas]


class _ForwardTrainHarness:
    forward_train_semi = UnbiasedTeacher.forward_train_semi

    def __init__(self):
        self.events = []
        self.ema_model = _StrongTeacher(self.events)
        self.image_num = 0
        self.momentum = 0.998
        self.weight_l = 1.0
        self.weight_u = 0.3
        self.use_bbox_reg = False
        self.semantic_reweight = False
        self.pseudo_num = np.ones(6, dtype=np.float64)
        self.pseudo_num_tp = np.ones(6, dtype=np.float64)
        self.cur_iter = 0

    def update_ema_model(self, momentum):
        self.events.append("ema_update")

    def forward_train(self, *args, **kwargs):
        self.events.append("student_forward")
        return {"loss_cls": torch.tensor(1.0)}

    @staticmethod
    def parse_loss(losses):
        return losses

    def inference_unlabeled(self, img, img_metas, rescale=True):
        self.events.append("weak")
        return [[np.empty((0, 6), dtype=np.float32) for _ in range(6)]
                for _ in img_metas]

    def create_pseudo_results(self, img, results, transform, device,
                              *args, **kwargs):
        count = len(results)
        return ([torch.empty((0, 5), device=device) for _ in range(count)],
                [torch.empty((0,), dtype=torch.long, device=device)
                 for _ in range(count)])

    def analysis(self):
        self.events.append("analysis")

    def _prototype_bank_update(self, ema_host, strong_results, **kwargs):
        self.events.append("bank_update")

    def _accumulate_prototype_v2_paired_diagnostics(self, *args, **kwargs):
        self.events.append("paired_diagnostics")


def test_v2_strong_pass_is_no_grad_eval_and_update_is_iteration_end(
        monkeypatch):
    monkeypatch.setenv("CGA_FILTER_MODE", "prototype_legacy_v2")
    harness = _ForwardTrainHarness()
    image = torch.zeros((1, 3, 8, 8))
    metas = [{"scale_factor": np.ones(4, dtype=np.float32)}]
    empty_boxes = [torch.empty((0, 5))]
    empty_labels = [torch.empty((0,), dtype=torch.long)]
    harness.forward_train_semi(
        image, metas, empty_boxes, empty_labels,
        image, metas, empty_boxes, empty_labels,
        image, metas, empty_boxes, empty_labels)

    strong_event = next(
        event for event in harness.events
        if isinstance(event, tuple) and event[0] == "strong")
    assert strong_event == ("strong", False, False)
    student_positions = [
        index for index, event in enumerate(harness.events)
        if event == "student_forward"]
    assert harness.events.index("bank_update") > max(student_positions)


def test_empty_weak_image_keeps_batch_alignment_slot():
    host = _v2_host(strict=True)
    empty = [np.empty((0, 6), dtype=np.float32) for _ in range(6)]
    host._refine_single(empty, {"filename": "empty.png"})
    host._refine_single(_image_results(), {"filename": "nonempty.png"})
    assert len(host._proto_pending) == 2
    assert host._proto_pending[0] is None
    assert host._proto_pending[1]["filename"] == "nonempty.png"
