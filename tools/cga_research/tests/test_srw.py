#!/usr/bin/env python3
"""Unit tests for SARCLIP Semantic Reliability Reweighting (SRW).

Run in the iraod env:

  LD_LIBRARY_PATH=$CONDA_PREFIX/lib \
  python tools/cga_research/tests/test_srw.py

Covers:
  1. semantic_weight formula in sfod/cga.py (agreement / disagreement ramp).
  2. return_cga_meta alignment for batch_size=2 (no cross-image mixup).
  3. ROI positive label_weights are scaled by pos_assigned_gt_inds mapping,
     negatives stay 1 (the core reweighting contract).
  4. backward compatibility: refine_test without return_cga_meta unchanged.
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class _FakeCGA:
    """Deterministic stand-in for the SARCLIP scorer.

    Returns a probability vector per box such that the argmax (SARCLIP top1)
    equals a scripted class, letting us control agreement/disagreement without
    loading the real model.
    """

    def __init__(self, class_names, scripted_top1):
        self.class_names = list(class_names)
        # scripted_top1: dict keyed by (round(score,4), label) -> top1 class
        self._scripted = scripted_top1

    def __call__(self, img_path, boxes, scores, labels):
        num_classes = len(self.class_names)
        logits = np.full((len(boxes), num_classes), 0.02, dtype=np.float64)
        for i, (score, label) in enumerate(zip(scores, labels)):
            top1 = self._scripted.get((round(float(score), 4), int(label)),
                                      int(label))
            logits[i, :] = 0.02
            logits[i, top1] = 0.9  # dominant, low entropy
            logits[i] = logits[i] / logits[i].sum()
        return logits, []


def _make_image_results(num_classes, per_class_boxes):
    """per_class_boxes: list over classes of list of (score,) rows.

    Returns rbbox result list (each class -> (N,6) obb+score array).
    """
    results = []
    for cls in range(num_classes):
        rows = per_class_boxes[cls]
        if not rows:
            results.append(np.zeros((0, 6), dtype=np.float32))
            continue
        arr = np.zeros((len(rows), 6), dtype=np.float32)
        for k, score in enumerate(rows):
            # cx, cy, w, h, angle, score  (values irrelevant except score)
            arr[k] = [10 + k, 10 + k, 8, 8, 0.0, score]
        results.append(arr)
    return results


def test_semantic_weight_formula():
    from sfod.cga import TestMixins

    class Host(TestMixins):
        CLASSES = ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor')

    host = Host()
    num_classes = 6
    # Build the CGA config attributes without touching SARCLIP.
    os.environ['CGA_FILTER_MODE'] = 'semantic_reweight'
    os.environ['CGA_SEM_LOW_THR'] = '0.90'
    os.environ['CGA_SEM_HIGH_THR'] = '0.95'
    os.environ['CGA_SEM_LAMBDA'] = '0.50'
    os.environ['CGA_SCORER'] = 'sarclip'
    os.environ['CGA_BACKEND'] = 'sarclip'

    # class 0 = ship. Boxes: scores 0.90/0.925/0.95 disagree (top1=car=2);
    # one agreement box at 0.90 (top1=ship=0).
    per_class = [[], [], [], [], [], []]
    per_class[0] = [0.90, 0.925, 0.95, 0.90]
    results = [_make_image_results(num_classes, per_class)]
    img_metas = [{'filename': 'fake.png'}]

    # scripted top1: first three ship boxes -> car (disagree), last -> ship.
    scripted = {
        (0.9, 0): 0,     # default; overridden per-row below via index trick
    }
    # We cannot key purely on (score,label) because two ship boxes share
    # score 0.90 with different desired outcomes. Instead patch by order.

    call = {'i': 0}
    desired_top1 = [2, 2, 2, 0]  # car,car,car,ship

    class OrderedFakeCGA(_FakeCGA):
        def __call__(self, img_path, boxes, scores, labels):
            logits = np.full((len(boxes), num_classes), 0.02, dtype=np.float64)
            for i in range(len(boxes)):
                t1 = desired_top1[call['i']]
                call['i'] += 1
                logits[i, :] = 0.02
                logits[i, t1] = 0.9
                logits[i] = logits[i] / logits[i].sum()
            return logits, []

    host.cga = OrderedFakeCGA(Host.CLASSES, {})
    host.exclude_ids = []
    # populate config attrs
    host.cga, host.exclude_ids = host.cga, []
    host._build_veto_groups(list(Host.CLASSES))
    host.cga_filter_mode = 'semantic_reweight'
    host.cga_sem_low_thr = 0.90
    host.cga_sem_high_thr = 0.95
    host.cga_sem_lambda = 0.50
    host.cga_filter_log_every = 0
    host.cga = OrderedFakeCGA(Host.CLASSES, {})

    refined, metas = host.refine_test(results, img_metas, return_cga_meta=True)
    w = metas[0]['semantic_weight'][0]  # class 0 weights, in box order
    # Expected: 0.90 disagree -> 1-0.5*1.0 = 0.50
    #           0.925 disagree -> g=(0.95-0.925)/0.05=0.5 -> 1-0.25=0.75
    #           0.95 disagree -> g=0 -> 1.0
    #           0.90 agree -> 1.0
    expected = np.array([0.50, 0.75, 1.00, 1.00])
    assert np.allclose(w, expected, atol=1e-6), (w, expected)
    # scores must be UNCHANGED (SRW never rescales). refined[0] = per-class
    # list; refined[0][0] = class-0 (ship) array.
    assert np.allclose(refined[0][0][:, -1],
                       np.array([0.90, 0.925, 0.95, 0.90]),
                       atol=1e-6), refined[0][0][:, -1]
    print('[PASS] semantic_weight formula + scores untouched')


def test_batch_alignment():
    from sfod.cga import TestMixins

    class Host(TestMixins):
        CLASSES = ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor')

    host = Host()
    num_classes = 6
    # image 0: one ship box (agree). image 1: one aircraft box (disagree->car).
    res0 = _make_image_results(num_classes, [[0.93], [], [], [], [], []])
    res1 = _make_image_results(num_classes, [[], [0.90], [], [], [], []])
    results = [res0, res1]
    img_metas = [{'filename': 'a.png'}, {'filename': 'b.png'}]

    seq = {'i': 0}
    # img0 ship -> ship (agree, w=1); img1 aircraft -> car (disagree, s=0.90 -> 0.5)
    desired = [0, 2]

    class OrderedFakeCGA:
        def __init__(self, class_names):
            self.class_names = list(class_names)

        def __call__(self, img_path, boxes, scores, labels):
            logits = np.full((len(boxes), num_classes), 0.02, dtype=np.float64)
            for i in range(len(boxes)):
                t1 = desired[seq['i']]
                seq['i'] += 1
                logits[i, :] = 0.02
                logits[i, t1] = 0.9
                logits[i] = logits[i] / logits[i].sum()
            return logits, []

    host.cga = OrderedFakeCGA(Host.CLASSES)
    host.exclude_ids = []
    host._build_veto_groups(list(Host.CLASSES))
    host.cga_filter_mode = 'semantic_reweight'
    host.cga_sem_low_thr = 0.90
    host.cga_sem_high_thr = 0.95
    host.cga_sem_lambda = 0.50
    host.cga_filter_log_every = 0

    _, metas = host.refine_test(results, img_metas, return_cga_meta=True)
    assert len(metas) == 2
    # image 0 ship weight = 1.0 (agree)
    assert np.allclose(metas[0]['semantic_weight'][0], [1.0]), metas[0]
    # image 0 aircraft class empty
    assert len(metas[0]['semantic_weight'][1]) == 0
    # image 1 aircraft weight = 0.5 (disagree at 0.90)
    assert np.allclose(metas[1]['semantic_weight'][1], [0.5]), metas[1]
    # image 1 ship class empty (no cross-image leakage)
    assert len(metas[1]['semantic_weight'][0]) == 0
    print('[PASS] batch_size=2 metadata alignment (no cross-image mixup)')


def test_roi_label_weight_scaling():
    """The core contract: positive ROI label_weights scaled by
    gt_semantic_weights[pos_assigned_gt_inds]; negatives stay 1."""
    from sfod.semantic_weighted_roi_head import (
        SemanticWeightedOrientedStandardRoIHead,
    )

    # Minimal fake bbox_head implementing get_targets(concat=False) + loss.
    class FakeBBoxHead:
        num_classes = 6

        def get_targets(self, sampling_results, gt_bboxes, gt_labels, cfg,
                        concat=True):
            labels_l, lw_l, bt_l, bw_l = [], [], [], []
            for res in sampling_results:
                num_pos = res.pos_bboxes.size(0)
                num_neg = res.neg_bboxes.size(0)
                n = num_pos + num_neg
                labels = torch.full((n,), self.num_classes, dtype=torch.long)
                lw = torch.zeros(n)
                if num_pos > 0:
                    labels[:num_pos] = res.pos_gt_labels
                    lw[:num_pos] = 1.0
                if num_neg > 0:
                    lw[-num_neg:] = 1.0
                labels_l.append(labels)
                lw_l.append(lw)
                bt_l.append(torch.zeros(n, 5))
                bw_l.append(torch.zeros(n, 5))
            if concat:
                return (torch.cat(labels_l), torch.cat(lw_l),
                        torch.cat(bt_l), torch.cat(bw_l))
            return labels_l, lw_l, bt_l, bw_l

        captured = {}

        def loss(self, cls_score, bbox_pred, rois, labels, label_weights,
                 bbox_targets, bbox_weights):
            FakeBBoxHead.captured['label_weights'] = label_weights.clone()
            FakeBBoxHead.captured['labels'] = labels.clone()
            return {'loss_cls': label_weights.sum() * 0.0}

    class FakeSamplingResult:
        def __init__(self, num_pos, num_neg, pos_assigned_gt_inds, pos_labels):
            self.pos_bboxes = torch.zeros(num_pos, 5)
            self.neg_bboxes = torch.zeros(num_neg, 5)
            self.bboxes = torch.zeros(num_pos + num_neg, 5)
            self.pos_assigned_gt_inds = torch.tensor(pos_assigned_gt_inds,
                                                     dtype=torch.long)
            self.pos_gt_labels = torch.tensor(pos_labels, dtype=torch.long)

    # Build a head instance without running its heavy __init__.
    head = SemanticWeightedOrientedStandardRoIHead.__new__(
        SemanticWeightedOrientedStandardRoIHead)
    head.bbox_head = FakeBBoxHead()
    head.train_cfg = None

    def fake_bbox_forward(x, rois):
        return {'cls_score': torch.zeros(rois.size(0), 7),
                'bbox_pred': torch.zeros(rois.size(0), 5),
                'bbox_feats': None}
    head._bbox_forward = fake_bbox_forward

    # 3 positives, 2 negatives. pos_assigned_gt_inds = [0,2,1].
    sr = FakeSamplingResult(num_pos=3, num_neg=2,
                            pos_assigned_gt_inds=[0, 2, 1],
                            pos_labels=[0, 0, 0])
    gt_semantic_weights = [torch.tensor([1.0, 0.5, 0.8])]

    import unittest.mock as mock
    with mock.patch(
            'sfod.semantic_weighted_roi_head.rbbox2roi',
            lambda lst: torch.zeros(sum(b.size(0) for b in lst), 6)):
        head._bbox_forward_train_semantic_weighted(
            x=[torch.zeros(1)], sampling_results=[sr],
            gt_bboxes=[torch.zeros(3, 5)], gt_labels=[torch.zeros(3)],
            img_metas=[{}], gt_semantic_weights=gt_semantic_weights)

    lw = FakeBBoxHead.captured['label_weights']
    # positives reordered by pos_assigned_gt_inds=[0,2,1] -> [1.0, 0.8, 0.5]
    expected = torch.tensor([1.0, 0.8, 0.5, 1.0, 1.0])
    assert torch.allclose(lw, expected, atol=1e-6), (lw, expected)
    print('[PASS] ROI positive label_weights=[1.0,0.8,0.5], negatives=1')


def test_backward_compat_legacy_unchanged():
    """refine_test without return_cga_meta returns just results (legacy)."""
    from sfod.cga import TestMixins

    class Host(TestMixins):
        CLASSES = ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor')

    host = Host()
    num_classes = 6
    res = _make_image_results(num_classes, [[0.93], [], [], [], [], []])
    results = [res]
    img_metas = [{'filename': 'a.png'}]

    class AgreeCGA:
        def __init__(self, class_names):
            self.class_names = list(class_names)

        def __call__(self, img_path, boxes, scores, labels):
            logits = np.full((len(boxes), num_classes), 0.02, dtype=np.float64)
            for i, label in enumerate(labels):
                logits[i, int(label)] = 0.9
                logits[i] = logits[i] / logits[i].sum()
            return logits, []

    host.cga = AgreeCGA(Host.CLASSES)
    host.exclude_ids = []
    host._build_veto_groups(list(Host.CLASSES))
    host.cga_filter_mode = 'legacy'
    host.cga_blend_detector_weight = 0.7
    host.cga_filter_log_every = 0

    out = host.refine_test(results, img_metas)  # no return_cga_meta
    assert isinstance(out, list) and not isinstance(out, tuple), type(out)
    # legacy agreement leaves score untouched. out[0][0] = class-0 array.
    assert np.allclose(out[0][0][:, -1], [0.93], atol=1e-6)
    print('[PASS] backward-compat: legacy refine_test returns list, score kept')


def _mk_host(mode, **attrs):
    from sfod.cga import TestMixins

    class Host(TestMixins):
        CLASSES = ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor')

    host = Host()
    host.exclude_ids = []
    host._build_veto_groups(list(Host.CLASSES))
    host.cga_filter_mode = mode
    host.cga_sem_low_thr = 0.90
    host.cga_sem_high_thr = 0.95
    host.cga_sem_lambda = 0.50
    host.cga_sem_gated_lambda = 0.80
    host.cga_sem_view_ratios = (0.0, 0.25, 0.5)
    host.cga_filter_log_every = 0
    for k, v in attrs.items():
        setattr(host, k, v)
    return host


def test_reliability_gated_single_view():
    """reliability_gated: w = 1 - λ_g · 1[disagree] · r · g(s),
    r = p_top1 · margin · (1 - H_norm). Score never modified."""
    from sfod.cga import _prob_entropy_normalized
    num_classes = 6

    # ship (label 0) box, det score 0.90 (g=1), SARCLIP top1 = car (2), confident.
    prob = np.array([0.02, 0.02, 0.90, 0.02, 0.02, 0.02], dtype=np.float64)
    prob = prob / prob.sum()
    p_top1 = float(np.sort(prob)[::-1][0])
    margin = float(np.sort(prob)[::-1][0] - np.sort(prob)[::-1][1])
    H = _prob_entropy_normalized(prob)
    r = p_top1 * margin * (1.0 - H)
    expected_w = 1.0 - 0.80 * r * 1.0  # g(0.90)=1

    results = [_make_image_results(num_classes, [[0.90], [], [], [], [], []])]
    img_metas = [{'filename': 'a.png'}]

    # A hesitant disagreement (high entropy, low margin) should be downweighted
    # much LESS than the confident one above -- the core discriminative property.
    prob_unsure = np.array([0.20, 0.18, 0.24, 0.14, 0.12, 0.12], dtype=np.float64)
    prob_unsure = prob_unsure / prob_unsure.sum()  # top1=car but barely
    results2 = [_make_image_results(num_classes, [[0.90], [], [], [], [], []])]

    seq = {'i': 0}
    rows = [prob, prob_unsure]

    class FakeCGA:
        class_names = ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor')

        def __call__(self, img_path, boxes, scores, labels):
            r = rows[seq['i']]; seq['i'] += 1
            return r.reshape(1, -1).copy(), []

    host = _mk_host('reliability_gated', cga=FakeCGA())
    refined, metas = host.refine_test(results, img_metas, return_cga_meta=True)
    w_conf = metas[0]['semantic_weight'][0][0]
    assert abs(w_conf - expected_w) < 1e-6, (w_conf, expected_w)
    # score untouched
    assert abs(refined[0][0][0, -1] - 0.90) < 1e-6

    _, metas2 = host.refine_test(results2, img_metas, return_cga_meta=True)
    w_unsure = metas2[0]['semantic_weight'][0][0]
    # hesitant opposition => weight much closer to 1 than the confident one
    assert w_unsure > w_conf, (w_unsure, w_conf)
    assert w_unsure > 0.95, w_unsure
    print(f'[PASS] reliability_gated: confident w={w_conf:.4f} (r={r:.3f}) vs '
          f'hesitant w={w_unsure:.4f} -- discriminates opposition strength')


def test_reliability_gated_mv_consistency():
    """The novel behavior: views that DISAGREE among themselves => r=0 =>
    weight stays 1.0 even though the (tight) view disagrees with detector.
    Consistent confident views => strong downweight."""
    num_classes = 6

    # Two boxes, both ship(label 0), both det score 0.90.
    results = [_make_image_results(num_classes, [[0.90, 0.90], [], [], [], [], []])]
    img_metas = [{'filename': 'a.png'}]

    # box0: all 3 views say car(2), confident -> reliable disagreement.
    # box1: views say car(2)/tank(3)/aircraft(1) -> inconsistent -> r=0.
    def onehot(c, conf=0.90):
        p = np.full(num_classes, (1 - conf) / (num_classes - 1))
        p[c] = conf
        return p

    view_logits = np.stack([
        np.stack([onehot(2), onehot(2)]),   # view0 (tight): box0->car, box1->car
        np.stack([onehot(2), onehot(3)]),   # view1: box0->car, box1->tank
        np.stack([onehot(2), onehot(1)]),   # view2: box0->car, box1->aircraft
    ], axis=0)  # (3, 2, 6)

    class FakeMVCGA:
        class_names = ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor')

        def __call__(self, img_path, boxes, scores, labels):
            # single-view path uses view 0
            return view_logits[0].copy(), []

        def forward_views(self, img_path, boxes, scores, labels, expand_ratios):
            return view_logits.copy()

    host = _mk_host('reliability_gated_mv', cga=FakeMVCGA())
    refined, metas = host.refine_test(results, img_metas, return_cga_meta=True)
    w = metas[0]['semantic_weight'][0]  # class ship, 2 boxes
    rel = metas[0]['reliability'][0]
    # box0: consistent confident car -> reliable -> downweighted below 1
    assert w[0] < 0.9, (w, rel)
    assert rel[0] > 0.0
    # box1: inconsistent views -> r=0 -> weight stays 1.0 (KEY test)
    assert abs(w[1] - 1.0) < 1e-9, (w, rel)
    assert rel[1] == 0.0, rel
    # scores untouched
    assert np.allclose(refined[0][0][:, -1], [0.90, 0.90], atol=1e-6)
    print(f'[PASS] reliability_gated_mv: consistent w={w[0]:.4f}(r={rel[0]:.3f}), '
          f'inconsistent w={w[1]:.4f}(r={rel[1]:.3f}) -> no downweight')


def test_reliability_gated_legacy():
    """Legacy score-blend fires ONLY on reliable disagreement below the 0.95
    cap; protected above cap; skipped when hesitant; tau=0 = cap-only."""
    num_classes = 6
    dw = 0.7  # det_weight

    def onehot_pair(det_label, clip_top1, p_top1, p_det):
        # prob vector: p_top1 on clip_top1, p_det on det_label, rest uniform
        p = np.full(num_classes, 0.0)
        p[clip_top1] = p_top1
        p[det_label] = p_det
        rem = 1.0 - p.sum()
        others = [c for c in range(num_classes) if c not in (clip_top1, det_label)]
        for c in others:
            p[c] = rem / len(others)
        return p

    # Box: ship(0) det, SARCLIP says car(2) confidently. p_top1=0.85, p_det=0.05.
    prob_reliable = onehot_pair(0, 2, 0.85, 0.05)
    # Hesitant: p_top1=0.30, p_det=0.22 -> opposition=(0.30-0.22)/0.5=0.16, high H
    prob_hesitant = onehot_pair(0, 2, 0.30, 0.22)

    def run(det_score, prob, tau):
        results = [_make_image_results(num_classes, [[det_score], [], [], [], [], []])]

        class FakeCGA:
            class_names = ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor')

            def __call__(self, img_path, boxes, scores, labels):
                return prob.reshape(1, -1).copy(), []

        host = _mk_host('reliability_gated_legacy', cga=FakeCGA(),
                        cga_blend_detector_weight=dw, cga_rel_legacy_tau=tau)
        out = host.refine_test(results, [{'filename': 'a.png'}])
        return out[0][0][0, -1]

    # 1) reliable disagreement, score 0.91 (<0.95): blended to 0.7*0.91+0.3*0.05
    s = run(0.91, prob_reliable, 0.10)
    expected = 0.91 * dw + 0.05 * (1 - dw)
    assert abs(s - expected) < 1e-6, (s, expected)
    assert s < 0.9, s  # would be removed by 0.9 admission
    # 2) reliable but score 0.96 (>=0.95): PROTECTED, unchanged
    s = run(0.96, prob_reliable, 0.10)
    assert abs(s - 0.96) < 1e-6, s
    # 2b) UNCAPPED (high_thr=2.0): the same 0.96 reliable disagreement now FIRES
    def run_uncap(det_score, prob, tau, high_thr):
        results = [_make_image_results(num_classes, [[det_score], [], [], [], [], []])]

        class FakeCGA:
            class_names = ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor')

            def __call__(self, img_path, boxes, scores, labels):
                return prob.reshape(1, -1).copy(), []

        host = _mk_host('reliability_gated_legacy', cga=FakeCGA(),
                        cga_blend_detector_weight=dw, cga_rel_legacy_tau=tau,
                        cga_sem_high_thr=high_thr)
        return host.refine_test(results, [{'filename': 'a.png'}])[0][0][0, -1]

    s = run_uncap(0.96, prob_reliable, 0.10, 2.0)
    expected = 0.96 * dw + 0.05 * (1 - dw)
    assert abs(s - expected) < 1e-6, (s, expected)  # cap removed => fires
    # 3) hesitant disagreement at 0.91: r<0.10, skipped, unchanged
    s = run(0.91, prob_hesitant, 0.10)
    assert abs(s - 0.91) < 1e-6, s
    # 4) tau=0 (cap-only): even hesitant disagreement below cap blends
    s = run(0.91, prob_hesitant, 0.0)
    expected = 0.91 * dw + prob_hesitant[0] * (1 - dw)
    assert abs(s - expected) < 1e-6, (s, expected)
    print('[PASS] reliability_gated_legacy: fires reliable<0.95, protects>=0.95, '
          'skips hesitant, tau=0=cap-only')


if __name__ == '__main__':
    test_semantic_weight_formula()
    test_batch_alignment()
    test_roi_label_weight_scaling()
    test_backward_compat_legacy_unchanged()
    test_reliability_gated_single_view()
    test_reliability_gated_mv_consistency()
    test_reliability_gated_legacy()
    print('\nALL SRW UNIT TESTS PASSED')
