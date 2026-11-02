"""CGA crops the unflipped source file, so teacher boxes predicted on a
flipped image must be restored before every crop, while pseudo-labels stay
in the teacher (flipped) frame."""
import os
import sys

import numpy as np
import pytest
import torch
from PIL import Image

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from mmrotate.datasets.pipelines import RRandomFlip  # noqa: E402
from sfod.cga import CGA, TestMixins, obb2xyxy  # noqa: E402
from sfod.prototype_cga import rotated_align_crop  # noqa: E402

H, W = 48, 64
# Rotated (le90) source box; its AABB edges sit far from integer pixels.
SOURCE_OBB = np.array([[14.0, 12.0, 12.0, 6.0, 0.3]])
DIRECTIONS = [None, "horizontal", "vertical", "diagonal"]
MIRROR_AXES = {"horizontal": (1,), "vertical": (0,), "diagonal": (0, 1)}


def _source_image(tmp_path):
    image = np.zeros((H, W, 3), dtype=np.uint8)
    image[7:18, 7:22] = (200, 30, 30)    # the object
    image[7:18, 7:11, 1] = 220           # left-edge marker: not h-symmetric
    image[7:10, 7:22, 2] = 250           # top-edge marker: not v-symmetric
    path = tmp_path / "source.png"
    Image.fromarray(image).save(path)
    return str(path), image


def _meta(path, direction):
    meta = {"filename": path, "ori_shape": (H, W, 3), "flip": bool(direction)}
    if direction:
        meta["flip_direction"] = direction
    return meta


def _teacher_results(direction):
    """Detections as the teacher emits them on the (maybe flipped) image."""
    obb = SOURCE_OBB.copy()
    if direction:
        obb = RRandomFlip(version="le90").bbox_flip(obb, (H, W), direction)
    detections = np.hstack([obb, [[0.9]]]).astype(np.float32)
    return [detections, np.empty((0, 6), dtype=np.float32)]


def _expected_aabb_crop(image):
    x1, y1, x2, y2 = obb2xyxy(SOURCE_OBB)[0]
    return np.asarray(Image.fromarray(image).crop(
        (int(x1), int(y1), int(x2), int(y2))))


class _RecordingCGA(CGA):
    """Real ``_crop_patches`` with a constant scorer; records the crops."""

    def __init__(self):
        self.class_names = ["a", "b"]
        self.expand_ratio = 0.0
        self.force_grayscale = False
        self.device = torch.device("cpu")
        self.preprocess = lambda image: torch.zeros(1)
        self.crops = []

    def __call__(self, img_path, boxes, scores, labels):
        _, crops = self._crop_patches(img_path, boxes, scores, labels)
        self.crops.extend(np.asarray(crop) for crop in crops)
        return np.tile([[0.2, 0.8]], (len(boxes), 1)), crops


class _Host(TestMixins):
    def __init__(self, cga, mode):
        super().__init__()
        self.cga = cga
        self.exclude_ids = []
        self.cga_filter_mode = mode
        self.cga_blend_detector_weight = 0.7
        self.cga_shuffle_seed = 0
        self.cga_filter_log_every = 0
        self.cga_strict = True
        self._cga_filter_calls = 1
        self.reset_proto_pending()


@pytest.mark.parametrize("mode", ["legacy", "shuffled_legacy"])
@pytest.mark.parametrize("direction", DIRECTIONS)
def test_aabb_crop_reads_the_object_from_the_source_image(
        tmp_path, direction, mode):
    path, image = _source_image(tmp_path)
    expected = _expected_aabb_crop(image)
    results = [_teacher_results(direction)]
    teacher_boxes = results[0][0][:, :5].copy()
    cga = _RecordingCGA()

    _Host(cga, mode).refine_test(results, [_meta(path, direction)])

    assert len(cga.crops) == 1
    np.testing.assert_array_equal(cga.crops[0], expected)
    if direction:
        mirror = np.flip(expected, MIRROR_AXES[direction])
        assert not np.array_equal(mirror, expected)
        assert not np.array_equal(cga.crops[0], mirror)
    # Pseudo-labels stay in the teacher frame; only the score is rewritten.
    np.testing.assert_array_equal(results[0][0][:, :5], teacher_boxes)
    assert results[0][0][0, -1] == pytest.approx(0.7 * 0.9 + 0.3 * 0.2)


class _EmbedRecorder:
    """Prototype-path scorer: records crops / boxes, returns valid embeds."""

    def __init__(self, image, ctx_image):
        self.image = image
        self.ctx_image = ctx_image
        self.tau = 100.0
        self.text = np.eye(2, 4, dtype=np.float32)
        self.patches = []
        self.aabb = []

    def _values(self, n):
        embed = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], np.float32), (n, 1))
        text_sim = embed @ self.text.T
        return embed, text_sim, np.tile([[0.5, 0.5]], (n, 1))

    def forward_rotated_embed(self, img_path, obboxes, context_ratio=0.15):
        for row in obboxes:
            self.patches.append(rotated_align_crop(
                self.ctx_image, *map(float, row[:5]),
                context_ratio=context_ratio))
        return self._values(len(obboxes))

    def forward_aabb_embed(self, img_path, boxes, scores, labels):
        self.aabb.append(np.asarray(boxes).copy())
        return self._values(len(boxes))

    def text_prototype_matrix(self):
        return self.text.copy()


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_rotated_prototype_crop_reads_the_object_from_the_source_image(
        tmp_path, direction):
    path, image = _source_image(tmp_path)
    expected = rotated_align_crop(
        image, *map(float, SOURCE_OBB[0]), context_ratio=0.15)
    scorer = _EmbedRecorder(image, image)
    host = _Host(scorer, "prototype_legacy")
    results = _teacher_results(direction)
    teacher_obb = results[0][:, :5].copy()

    host._refine_single(results, _meta(path, direction))

    assert len(scorer.patches) == 1
    np.testing.assert_array_equal(scorer.patches[0], expected)
    if direction:
        assert not np.array_equal(
            scorer.patches[0], np.flip(expected, MIRROR_AXES[direction]))
    np.testing.assert_array_equal(results[0][:, :5], teacher_obb)
    np.testing.assert_array_equal(
        host._proto_pending[-1]["obb"], teacher_obb.astype(np.float64))


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_prototype_v2_crops_source_boxes_and_stashes_teacher_boxes(
        tmp_path, direction):
    path, image = _source_image(tmp_path)
    scorer = _EmbedRecorder(image, image)
    host = _Host(scorer, "prototype_legacy_v2")
    host.cga_proto_beta = 0.5
    results = _teacher_results(direction)
    teacher_obb = results[0][:, :5].copy()

    host._refine_single(results, _meta(path, direction))

    np.testing.assert_allclose(
        scorer.aabb[0], obb2xyxy(SOURCE_OBB), atol=1e-4)
    np.testing.assert_array_equal(results[0][:, :5], teacher_obb)
    pending = host._proto_pending[-1]
    np.testing.assert_array_equal(pending["obb"], teacher_obb)
    np.testing.assert_allclose(
        pending["boxes"], obb2xyxy(teacher_obb), atol=1e-4)
