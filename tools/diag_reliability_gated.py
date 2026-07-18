#!/usr/bin/env python
"""Zero-training diagnostic for reliability_gated / reliability_gated_mv SRW.

On chaff/val GT boxes (REAL targets), every SARCLIP disagreement is a SARCLIP
ERROR (the GT label is correct). So any semantic downweight applied to a GT box
is FALSE downweight on a real target. This tool compares three SRW weightings
at a fixed detector score (default 0.90, so g(s)=1 and only the reliability term
varies), reporting how much false downweight each applies to real targets:

  srw_linear         : w = 1 - lambda            (lambda=0.5)  on every disagree
  reliability_gated  : w = 1 - lambda_g * r      r=p_top1*margin*(1-H)
  reliability_gated_mv: w = 1 - lambda_g * r_mv  r_mv=1[views agree]*mean_p*(1-mean_H)

Key question the prior evidence_veto result raises: does gating / multi-view
concentrate strong downweight AWAY from correct boxes? Higher mean weight on
disagreements = safer (less collateral damage on real targets). This is
NECESSARY-not-sufficient: it can't measure the beneficial downweight on true
detector FPs (no detector here), same limitation as diag_evidence_veto.
"""
import argparse
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from iraod_runtime import ensure_iraod_runtime

ensure_iraod_runtime()

import numpy as np
from sfod.cga import CGA, _prob_entropy_normalized

CLASSES = ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor')
CLS_TO_ID = {c: i for i, c in enumerate(CLASSES)}


def find_image(img_dir, stem):
    for ext in ('.jpg', '.png', '.bmp', '.jpeg', '.tif', '.tiff'):
        p = os.path.join(img_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None


def load_boxes(ann_path):
    boxes, labels = [], []
    with open(ann_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9 or parts[8] not in CLS_TO_ID:
                continue
            try:
                coords = list(map(float, parts[:8]))
            except ValueError:
                continue
            xs, ys = coords[0::2], coords[1::2]
            boxes.append([min(xs), min(ys), max(xs), max(ys)])
            labels.append(CLS_TO_ID[parts[8]])
    return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img-dir', default='/home/storageSDA1/liaojr/dataset/RSAR/corruptions/chaff/val/images')
    ap.add_argument('--ann-dir', default='/home/storageSDA1/liaojr/dataset/RSAR/val/annfiles')
    ap.add_argument('--num-images', type=int, default=1200)
    ap.add_argument('--single-expand', type=float, default=0.4,
                    help='expand_ratio for linear/gated single-view (training default)')
    ap.add_argument('--view-ratios', default='0.0,0.25,0.5')
    ap.add_argument('--lambda-linear', type=float, default=0.5)
    ap.add_argument('--lambda-gated', type=float, default=0.8)
    ap.add_argument('--det-score', type=float, default=0.90,
                    help='fixed detector score; 0.90 => g(s)=1 isolates reliability')
    ap.add_argument('--low-thr', type=float, default=0.90)
    ap.add_argument('--high-thr', type=float, default=0.95)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    view_ratios = tuple(float(v) for v in args.view_ratios.split(','))
    os.environ.setdefault('CGA_SCORER', 'sarclip')
    os.environ.setdefault('CGA_BACKEND', 'sarclip')
    templates = (os.environ.get('CGA_TEMPLATES')
                 or 'A SAR image of a {};This SAR patch shows a {}').split(';')
    cga = CGA(list(CLASSES), model='ViT-B-32',
              pretrained='/home/storageSDA1/Dataset/SARCLIP/ViT-B-32/vit_b_32_model.safetensors',
              cache_dir='/home/storageSDA1/Dataset/SARCLIP/ViT-B-32', precision='fp32',
              templates=templates, tau=float(os.environ.get('CGA_TAU', '100.0')),
              expand_ratio=args.single_expand, backend='sarclip')
    cga._first_call_logged = True

    # g(s) at the chosen detector score.
    g = np.clip((args.high_thr - args.det_score) / (args.high_thr - args.low_thr), 0, 1)

    stems = sorted(Path(args.ann_dir).glob('*.txt'))
    random.seed(args.seed)
    random.shuffle(stems)

    sv_prob = []          # single-view prob (expand=single-expand)
    mv_agree = []         # multi-view: all views top1 agree
    mv_mean_top1 = []     # multi-view mean top1 prob
    mv_mean_ent = []      # multi-view mean entropy
    mv_top1 = []          # tight-view top1 class
    labels_all = []
    n_img = 0
    for ann in stems:
        img = find_image(args.img_dir, ann.stem)
        if img is None:
            continue
        boxes, labels = load_boxes(str(ann))
        if len(boxes) == 0:
            continue
        scores = np.full(len(boxes), args.det_score, dtype=np.float32)
        sv_logits, _ = cga(img, boxes, scores, labels)
        vlog = cga.forward_views(img, boxes, scores, labels, view_ratios)  # (V,N,C)
        vt1 = np.argmax(vlog, axis=2)               # (V,N)
        vt1p = np.max(vlog, axis=2)                 # (V,N)
        for i in range(len(boxes)):
            sv_prob.append(np.asarray(sv_logits[i], dtype=np.float64))
            mv_agree.append(bool(np.all(vt1[:, i] == vt1[0, i])))
            mv_mean_top1.append(float(vt1p[:, i].mean()))
            mv_mean_ent.append(float(np.mean([_prob_entropy_normalized(vlog[v, i])
                                              for v in range(vlog.shape[0])])))
            mv_top1.append(int(vt1[0, i]))
            labels_all.append(int(labels[i]))
        n_img += 1
        if n_img >= args.num_images:
            break

    sv_prob = np.array(sv_prob); lab = np.array(labels_all)
    mv_agree = np.array(mv_agree); mv_mean_top1 = np.array(mv_mean_top1)
    mv_mean_ent = np.array(mv_mean_ent); mv_top1 = np.array(mv_top1)
    N = len(lab)

    # single-view stats
    sv_pred = sv_prob.argmax(1)
    sorted_p = np.sort(sv_prob, 1)[:, ::-1]
    sv_margin = sorted_p[:, 0] - sorted_p[:, 1]
    sv_top1p = sorted_p[:, 0]
    sv_ent = np.array([_prob_entropy_normalized(p) for p in sv_prob])
    sv_disagree = sv_pred != lab
    mv_disagree = mv_top1 != lab

    print(f'[diag] {n_img} images, {N} real-target GT boxes, det_score={args.det_score} (g={g:.2f})')
    print(f'[diag] single-view (expand={args.single_expand}) disagreement = '
          f'{sv_disagree.mean()*100:.2f}%  ({sv_disagree.sum()}/{N})')
    print(f'[diag] multi-view tight-view disagreement = {mv_disagree.mean()*100:.2f}%')
    print(f'[diag] multi-view CROSS-SCALE AGREEMENT overall = {mv_agree.mean()*100:.2f}%')
    if mv_disagree.sum():
        print(f'[diag] among tight-view DISAGREEMENTS (SARCLIP errors), all-3-views-agree = '
              f'{mv_agree[mv_disagree].mean()*100:.2f}%  '
              f'(the rest get r_mv=0 => NO downweight)')

    # weights (fixed det_score => g constant)
    r_gated = np.clip(sv_top1p * sv_margin * (1 - sv_ent), 0, 1)
    w_linear = np.where(sv_disagree, 1 - args.lambda_linear * g, 1.0)
    w_gated = np.where(sv_disagree, 1 - args.lambda_gated * r_gated * g, 1.0)
    w_gated = np.clip(w_gated, 0, 1)
    r_mv = np.clip(mv_agree.astype(float) * mv_mean_top1 * (1 - mv_mean_ent), 0, 1)
    w_mv = np.where(mv_disagree, 1 - args.lambda_gated * r_mv * g, 1.0)
    w_mv = np.clip(w_mv, 0, 1)

    def report(name, w, dis):
        d = w[dis]
        print(f'  {name:20s} disagree_n={dis.sum():5d} '
              f'mean_w_on_disagree={d.mean() if len(d) else 1.0:.4f} '
              f'frac_w<0.9={np.mean(d < 0.9)*100 if len(d) else 0:.1f}% '
              f'frac_w<0.7={np.mean(d < 0.7)*100 if len(d) else 0:.1f}% '
              f'total_false_downweight={np.sum(1 - w):.1f}')

    print('\n=== FALSE downweight on real targets (higher mean_w = safer) ===')
    print('  (total_false_downweight = sum(1-w) over ALL real targets; lower = less damage)')
    report('srw_linear',          w_linear, sv_disagree)
    report('reliability_gated',   w_gated,  sv_disagree)
    report('reliability_gated_mv', w_mv,    mv_disagree)

    # per-class mv agreement among disagreements (where mv protects real targets)
    print('\n=== per-class: multi-view protection of real targets ===')
    print('  (of tight-view SARCLIP errors, how many get r_mv=0 => protected)')
    for cid in range(len(CLASSES)):
        m = (lab == cid) & mv_disagree
        if m.sum() == 0:
            continue
        protected = (~mv_agree[m]).mean() * 100
        print(f'  {CLASSES[cid]:8s} sarclip_errors={m.sum():4d} '
              f'protected(r_mv=0)={protected:.1f}% '
              f'mean_w_mv={w_mv[m].mean():.4f}')

    # summary verdict
    print('\n=== SUMMARY ===')
    ml = w_linear[sv_disagree].mean() if sv_disagree.sum() else 1.0
    mg = w_gated[sv_disagree].mean() if sv_disagree.sum() else 1.0
    mm = w_mv[mv_disagree].mean() if mv_disagree.sum() else 1.0
    print(f'  mean weight on real-target disagreements: '
          f'linear={ml:.4f}  gated={mg:.4f}  mv={mm:.4f}')
    print(f'  total false downweight on real targets: '
          f'linear={np.sum(1-w_linear):.1f}  gated={np.sum(1-w_gated):.1f}  '
          f'mv={np.sum(1-w_mv):.1f}  (of N={N})')


if __name__ == '__main__':
    main()
