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


RED, GREEN = (220, 0, 0), (0, 220, 0)
# (class, source obb, colour, detector score); axis-aligned so crops are pure.
MULTI = [
    (0, [10.5, 10.5, 8.0, 6.0, 0.0], RED, 0.9),
    (0, [40.5, 12.5, 8.0, 6.0, 0.0], GREEN, 0.8),
    (1, [25.5, 34.5, 10.0, 6.0, 0.0], GREEN, 0.7),
]


def _multi_image(tmp_path):
    image = np.zeros((H, W, 3), dtype=np.uint8)
    for _, (cx, cy, w, h, _), colour, _ in MULTI:
        image[int(cy - h / 2):int(cy + h / 2) + 1,
              int(cx - w / 2):int(cx + w / 2) + 1] = colour
    path = tmp_path / "multi.png"
    Image.fromarray(image).save(path)
    return str(path)


def _multi_results(direction):
    flip = RRandomFlip(version="le90")
    results = [[], []]
    for label, obb, _, score in MULTI:
        obb = np.asarray([obb])
        if direction:
            obb = flip.bbox_flip(obb, (H, W), direction)
        results[label].append(np.hstack([obb, [[score]]])[0])
    return [np.asarray(rows, dtype=np.float32) for rows in results]


class _ColourCGA(_RecordingCGA):
    """Red crop -> class 0, green crop -> class 1; records every crop."""

    def __init__(self):
        super().__init__()
        self._first_call_logged = True
        self.view_crops = []

    @staticmethod
    def _probs(crops):
        return np.asarray([
            [0.9, 0.1] if crop[..., 0].max() > crop[..., 1].max()
            else [0.1, 0.9] for crop in crops])

    def __call__(self, img_path, boxes, scores, labels):
        _, crops = self._crop_patches(img_path, boxes, scores, labels)
        crops = [np.asarray(crop) for crop in crops]
        self.crops.extend(crops)
        return self._probs(crops), crops

    def _crop_patches(self, img_path, boxes, scores, labels,
                      expand_ratio=None, image=None):
        tensors, crops = super()._crop_patches(
            img_path, boxes, scores, labels, expand_ratio, image)
        if image is not None:  # forward_views passes the opened image
            self.view_crops.append((expand_ratio, [np.asarray(c) for c in crops]))
        return tensors, crops

    def _score_images(self, images):
        return self._probs(self.view_crops[-1][1])


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_multi_class_multi_box_scores_follow_their_own_crop(
        tmp_path, direction):
    path = _multi_image(tmp_path)
    results = [_multi_results(direction)]
    teacher = [r.copy() for r in results[0]]
    cga = _ColourCGA()

    _Host(cga, "legacy").refine_test(results, [_meta(path, direction)])

    # Flat order is class 0 boxes then class 1 boxes, matching MULTI.
    colours = [crop.reshape(-1, 3).max(axis=0) for crop in cga.crops]
    assert [tuple(c) for c in colours] == [m[2] for m in MULTI]
    # Only the green box labelled class 0 disagrees and is blended.
    np.testing.assert_allclose(
        results[0][0][:, -1], [0.9, 0.7 * 0.8 + 0.3 * 0.1], rtol=1e-6)
    np.testing.assert_allclose(results[0][1][:, -1], [0.7], rtol=1e-6)
    for got, want in zip(results[0], teacher):
        np.testing.assert_array_equal(got[:, :5], want[:, :5])


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_forward_views_crop_the_source_object_at_every_scale(
        tmp_path, direction):
    path, image = _source_image(tmp_path)
    expected = _expected_aabb_crop(image)
    results = [_teacher_results(direction)]
    cga = _ColourCGA()

    _Host(cga, "reliability_gated_mv").refine_test(
        results, [_meta(path, direction)])

    ratios = [ratio for ratio, _ in cga.view_crops]
    assert ratios == [0.0, 0.25, 0.5]
    np.testing.assert_array_equal(cga.view_crops[0][1][0], expected)
    x1, y1, x2, y2 = obb2xyxy(SOURCE_OBB)[0]
    for ratio, crops in cga.view_crops[1:]:
        w, h = x2 - x1, y2 - y1
        left = int(max(0, x1 - w * ratio))
        top = int(max(0, y1 - h * ratio))
        right, bottom = int(x2 + w * ratio), int(y2 + h * ratio)
        want = np.asarray(Image.fromarray(image).crop((left, top, right, bottom)))
        np.testing.assert_array_equal(crops[0], want)


# Source-frame centres of 10x6 boxes partly / wholly outside each image side.
# DETR heads emit such boxes (e.g. y scaled by width on non-square images);
# after a flip restore, a box past the bottom/right lands above/left.
OUTSIDE = {
    ("left", "partly"): (2.25, 24.25), ("left", "fully"): (-10.25, 24.25),
    ("right", "partly"): (W - 2.25, 24.25), ("right", "fully"): (W + 10.25, 24.25),
    ("top", "partly"): (30.25, 1.25), ("top", "fully"): (30.25, -10.25),
    ("bottom", "partly"): (30.25, H - 1.25), ("bottom", "fully"): (30.25, H + 10.25),
}


def _pre_fix_crop(image, xyxy):
    """The crop the pre-flip-fix code takes (raises where it raised)."""
    x1, y1, x2, y2 = xyxy
    return np.asarray(Image.fromarray(image).crop(
        (int(max(0, x1)), int(max(0, y1)), int(x2), int(y2))))


@pytest.mark.parametrize("side_extent", sorted(OUTSIDE),
                         ids=["-".join(key) for key in sorted(OUTSIDE)])
@pytest.mark.parametrize("direction", DIRECTIONS)
def test_boxes_outside_the_image_keep_pre_fix_crops(
        tmp_path, direction, side_extent):
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    # Every pixel distinct and non-zero, so a misplaced crop cannot pass.
    image = np.stack([xs * 4, ys * 5, np.full_like(xs, 100)], -1).astype(np.uint8)
    path = str(tmp_path / "gradient.png")
    Image.fromarray(image).save(path)
    cx, cy = OUTSIDE[side_extent]
    source = np.array([[cx, cy, 10.0, 6.0, 0.0]])
    obb = source if direction is None else RRandomFlip(
        version="le90").bbox_flip(source.copy(), (H, W), direction)
    results = [[np.hstack([obb, [[0.9]]]).astype(np.float32),
                np.empty((0, 6), dtype=np.float32)]]
    cga = _RecordingCGA()

    _Host(cga, "legacy").refine_test(results, [_meta(path, direction)])

    assert len(cga.crops) == 1  # never dropped
    crop = cga.crops[0]
    xyxy = obb2xyxy(source)[0]
    side, extent = side_extent
    if extent == "fully" and side in ("left", "top"):
        # Pre-fix code raised here; crop black padding like its mirror box
        # past the right/bottom edge, which pre-fix code cropped.
        assert crop.size and not crop.any()
        mirror = {"left": "right", "top": "bottom"}[side]
        mx, my = OUTSIDE[(mirror, "fully")]
        twin = _pre_fix_crop(image, obb2xyxy(
            np.array([[mx, my, 10.0, 6.0, 0.0]]))[0])
        assert not twin.any()
        assert abs(crop.shape[0] - twin.shape[0]) <= 1
        assert abs(crop.shape[1] - twin.shape[1]) <= 1
    else:
        np.testing.assert_array_equal(crop, _pre_fix_crop(image, xyxy))
