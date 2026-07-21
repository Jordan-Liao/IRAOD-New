"""Unit tests for Target-Adaptive Prototype-Guided CGA (prototype_legacy).

Covers geometry (rotated crop), weak/strong candidate matching, EMA bank
behavior, text/visual fusion + degradation to legacy, and score handling. Pure
numpy/PIL where possible so it runs without CUDA / SARCLIP weights.
"""
import os
import sys
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sfod.prototype_cga import (  # noqa: E402
    rotated_align_crop, fuse_logits, softmax, greedy_match_weak_strong,
    VisualPrototypeBank, _zscore,
)


def _rot_iou(a, b):
    import torch
    from mmrotate.core.bbox import rbbox_overlaps
    return rbbox_overlaps(
        torch.tensor(np.asarray(a), dtype=torch.float32),
        torch.tensor(np.asarray(b), dtype=torch.float32)).cpu().numpy()


def test_rotated_crop_angle_orientation():
    """Authoritative le90 sign check: an OBB rendered via the project's own
    obb2poly(version='le90') at angle +35deg must be STRAIGHTENED (wide & thin,
    high fill) when cropped with the SAME angle passed as-is, and destroyed when
    the angle is negated. Verifies rotated_align_crop uses the detector's le90
    angle directly."""
    import cv2
    import torch
    from mmrotate.core.bbox.transforms import obb2poly
    cx, cy, w, h, ang = 120., 120., 90., 14., np.radians(35.0)
    obb = torch.tensor([[cx, cy, w, h, ang]], dtype=torch.float32)
    poly = obb2poly(obb, version='le90').numpy().reshape(-1, 2).astype(np.int32)
    img = np.zeros((240, 240, 3), dtype=np.uint8)
    cv2.fillPoly(img, [poly], (255, 255, 255))

    patch = rotated_align_crop(img, cx, cy, w, h, ang, context_ratio=0.15)
    ph, pw = patch.shape[:2]
    xs = np.where(patch[:, :, 0] > 128)[1]
    fill = len(np.where(patch[:, :, 0] > 128)[0]) / (pw * ph + 1e-6)
    assert pw > ph, (ph, pw)                 # straightened -> wider than tall
    assert fill > 0.5, fill                  # bar fills the crop

    bad = rotated_align_crop(img, cx, cy, w, h, -ang, context_ratio=0.15)
    fill_bad = len(np.where(bad[:, :, 0] > 128)[0]) / (bad.shape[0] * bad.shape[1] + 1e-6)
    assert fill_bad < fill, (fill_bad, fill)  # wrong sign is worse
    print(f'[PASS] rotated crop le90 sign correct (fill={fill:.2f} vs wrong {fill_bad:.2f})')


def test_rotated_crop_angle_zero_reasonable():
    img = np.arange(200 * 200 * 3, dtype=np.uint8).reshape(200, 200, 3)
    patch = rotated_align_crop(img, 100, 100, 40, 20, 0.0, context_ratio=0.15)
    # 40*1.15=46, 20*1.15=23
    assert abs(patch.shape[1] - 46) <= 2 and abs(patch.shape[0] - 23) <= 2, patch.shape
    print('[PASS] rotated crop angle=0 size reasonable')


def test_rotated_crop_boundary_nonempty():
    img = np.full((100, 100, 3), 127, dtype=np.uint8)
    # box near corner, partly out of image -> clipped but non-empty
    patch = rotated_align_crop(img, 5, 5, 40, 40, 0.3, context_ratio=0.15)
    assert patch.size > 0 and patch.shape[0] >= 1 and patch.shape[1] >= 1
    print('[PASS] rotated crop boundary non-empty')


def test_rotated_crop_tiny_and_degenerate():
    img = np.full((100, 100, 3), 127, dtype=np.uint8)
    patch = rotated_align_crop(img, 50, 50, 2, 2, 0.5, context_ratio=0.15)
    assert patch.size > 0
    for bad in [(0.0, 10.0), (10.0, -1.0), (float('nan'), 10.0)]:
        try:
            rotated_align_crop(img, 50, 50, bad[0], bad[1], 0.0)
            assert False, 'degenerate box should raise'
        except ValueError:
            pass
    print('[PASS] rotated crop tiny ok / degenerate raises')


def test_greedy_match_same_class_and_iou():
    # weak: two ships (label 0), one car (label 2)
    weak_obb = np.array([[50, 50, 20, 10, 0.0],
                         [120, 120, 20, 10, 0.0],
                         [80, 80, 20, 10, 0.0]], dtype=np.float32)
    weak_label = np.array([0, 0, 2])
    weak_score = np.array([0.99, 0.98, 0.99])
    # strong: ship overlapping #0, ship far from #1 (low IoU), car overlapping #2
    strong_obb = np.array([[50, 50, 20, 10, 0.0],
                           [200, 200, 20, 10, 0.0],
                           [80, 80, 20, 10, 0.0]], dtype=np.float32)
    strong_label = np.array([0, 0, 2])
    q = greedy_match_weak_strong(weak_obb, weak_label, weak_score,
                                 strong_obb, strong_label, _rot_iou,
                                 score_thr=0.97, iou_thr=0.70)
    assert q[0] and q[2] and not q[1], q
    print('[PASS] greedy match: same-class + IoU>=0.70 only')


def test_match_requires_same_label():
    weak_obb = np.array([[50, 50, 20, 10, 0.0]], dtype=np.float32)
    strong_obb = np.array([[50, 50, 20, 10, 0.0]], dtype=np.float32)
    # same box, different label -> no match
    q = greedy_match_weak_strong(weak_obb, np.array([0]), np.array([0.99]),
                                 strong_obb, np.array([1]), _rot_iou)
    assert not q.any(), q
    print('[PASS] match requires same label')


def test_match_score_and_iou_gates():
    weak_obb = np.array([[50, 50, 20, 10, 0.0]], dtype=np.float32)
    strong_obb = np.array([[50, 50, 20, 10, 0.0]], dtype=np.float32)
    # score below 0.97 -> no update
    q = greedy_match_weak_strong(weak_obb, np.array([0]), np.array([0.96]),
                                 strong_obb, np.array([0]), _rot_iou)
    assert not q.any(), 'score<0.97 must not qualify'
    # low IoU -> no update
    strong_far = np.array([[300, 300, 20, 10, 0.0]], dtype=np.float32)
    q2 = greedy_match_weak_strong(weak_obb, np.array([0]), np.array([0.99]),
                                  strong_far, np.array([0]), _rot_iou)
    assert not q2.any(), 'IoU<0.70 must not qualify'
    print('[PASS] score>=0.97 and IoU>=0.70 gates enforced')


def test_ema_update_and_normalization():
    D = 8
    bank = VisualPrototypeBank(6, embed_dim=D, momentum=0.95, min_count=20)
    rng = np.random.default_rng(0)
    e1 = rng.normal(size=(5, D)); e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    bank.update({0: e1}, cur_iter=1)
    assert bank.prototype_initialized[0]
    assert abs(np.linalg.norm(bank.prototype[0]) - 1.0) < 1e-9  # normalized
    p_before = bank.prototype[0].copy()
    e2 = rng.normal(size=(7, D)); e2 /= np.linalg.norm(e2, axis=1, keepdims=True)
    bank.update({0: e2}, cur_iter=2)
    assert abs(np.linalg.norm(bank.prototype[0]) - 1.0) < 1e-9
    assert not np.allclose(bank.prototype[0], p_before)  # moved
    assert bank.prototype_count[0] == 12
    assert bank.prototype_update_count[0] == 2
    print('[PASS] EMA update correct + prototype stays normalized')


def test_min_count_gates_activation():
    D = 8
    bank = VisualPrototypeBank(6, embed_dim=D, momentum=0.95, min_count=20)
    rng = np.random.default_rng(1)
    e = rng.normal(size=(10, D)); e /= np.linalg.norm(e, axis=1, keepdims=True)
    bank.update({0: e}, cur_iter=1)
    assert not bank.is_active(0)  # count 10 < 20
    assert bank.first_active_iteration[0] == -1
    bank.update({0: e}, cur_iter=2)
    assert bank.is_active(0)  # count 20 >= 20
    assert bank.first_active_iteration[0] == 2
    print('[PASS] min_count=20 gates activation + first_active_iteration')


def test_fusion_degrades_to_text_when_inactive():
    N, C = 4, 6
    rng = np.random.default_rng(2)
    sim_text = rng.normal(size=(N, C))
    sim_visual = rng.normal(size=(N, C))
    inactive = np.zeros(C, dtype=bool)
    fused = fuse_logits(sim_text, sim_visual, inactive, beta=0.5)
    # all inactive -> fused == z-scored text exactly
    assert np.allclose(fused, _zscore(sim_text)), 'must degrade to text-only'
    # one active class changes only that column
    active = inactive.copy(); active[2] = True
    fused2 = fuse_logits(sim_text, sim_visual, active, beta=0.5)
    for c in range(C):
        if c == 2:
            assert not np.allclose(fused2[:, c], _zscore(sim_text)[:, c])
        else:
            assert np.allclose(fused2[:, c], _zscore(sim_text)[:, c])
    print('[PASS] fusion: inactive=text-only, active class fused only')


def test_fusion_beta_blend_values():
    N, C = 3, 6
    rng = np.random.default_rng(3)
    sim_text = rng.normal(size=(N, C))
    sim_visual = rng.normal(size=(N, C))
    active = np.zeros(C, dtype=bool); active[0] = True
    beta = 0.5
    fused = fuse_logits(sim_text, sim_visual, active, beta)
    zt = _zscore(sim_text); zv = _zscore(sim_visual)
    expected = (1 - beta) * zt[:, 0] + beta * zv[:, 0]
    assert np.allclose(fused[:, 0], expected), (fused[:, 0], expected)
    print('[PASS] fusion beta blend matches (1-b)*zt + b*zv')


def test_no_self_inclusion_snapshot_semantics():
    """The bank the fused blend uses is the pre-update snapshot: updating after
    scoring must not change the matrix used for the just-scored batch."""
    D = 4
    bank = VisualPrototypeBank(6, embed_dim=D, momentum=0.95, min_count=1)
    rng = np.random.default_rng(4)
    e = rng.normal(size=(3, D)); e /= np.linalg.norm(e, axis=1, keepdims=True)
    bank.update({0: e}, cur_iter=1)
    mat_used = bank.matrix().copy()          # snapshot scoring would use
    # a later update must not retroactively change mat_used
    e2 = rng.normal(size=(3, D)); e2 /= np.linalg.norm(e2, axis=1, keepdims=True)
    bank.snapshot_previous()
    bank.update({0: e2}, cur_iter=2)
    assert np.allclose(mat_used[0], bank.previous_prototype[0]), \
        'scoring snapshot must equal pre-update prototype'
    assert not np.allclose(bank.matrix()[0], mat_used[0])
    print('[PASS] no self-inclusion: scoring uses pre-update snapshot')


def test_zero_norm_update_raises():
    D = 4
    bank = VisualPrototypeBank(6, embed_dim=D, momentum=0.95, min_count=1)
    z = np.zeros((2, D))  # mean is zero -> zero norm batch proto
    try:
        bank.update({0: z}, cur_iter=1)
        assert False, 'zero-norm batch proto should raise'
    except ValueError:
        pass
    print('[PASS] zero-norm prototype update raises (strict-safe)')


def test_softmax_finite_and_normalized():
    logits = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    p = softmax(logits, axis=-1)
    assert np.allclose(p.sum(axis=-1), 1.0)
    assert np.all(np.isfinite(p))
    print('[PASS] softmax finite + normalized')


if __name__ == '__main__':
    test_rotated_crop_angle_orientation()
    test_rotated_crop_angle_zero_reasonable()
    test_rotated_crop_boundary_nonempty()
    test_rotated_crop_tiny_and_degenerate()
    test_greedy_match_same_class_and_iou()
    test_match_requires_same_label()
    test_match_score_and_iou_gates()
    test_ema_update_and_normalization()
    test_min_count_gates_activation()
    test_fusion_degrades_to_text_when_inactive()
    test_fusion_beta_blend_values()
    test_no_self_inclusion_snapshot_semantics()
    test_zero_norm_update_raises()
    test_softmax_finite_and_normalized()
    print('\nALL PROTOTYPE CGA UNIT TESTS PASSED')
