#!/usr/bin/env python
"""Diagnostic for CGA_FILTER_MODE=evidence_veto.

Runs SARCLIP+LoRA once on chaff/val GT boxes (REAL targets) and reports, per
threshold setting, the FALSE-VETO RATE: fraction of real targets the gate would
wrongly downweight. The evidence_veto design only fires on reliable, low-
uncertainty opposition; this measures its collateral damage on real targets
BEFORE spending a training run. Lower false-veto rate = safer.

Gate (fires only if ALL hold, and only on disagreement pred!=label):
  pred_prob >= pred_hi, label_prob <= label_lo, margin >= margin_thr,
  norm_entropy <= entropy_thr, pred/label not in same confusion group.
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
from sfod.cga import CGA, RSAR_CONFUSION_GROUPS, _prob_entropy_normalized

CLASSES = ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor')
CLS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
GROUP_OF = {}
for gid, g in enumerate(RSAR_CONFUSION_GROUPS):
    for n in g:
        if n in CLS_TO_ID:
            GROUP_OF[CLS_TO_ID[n]] = gid


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
            if len(parts) < 9:
                continue
            try:
                coords = list(map(float, parts[:8]))
            except ValueError:
                continue
            if parts[8] not in CLS_TO_ID:
                continue
            xs, ys = coords[0::2], coords[1::2]
            boxes.append([min(xs), min(ys), max(xs), max(ys)])
            labels.append(CLS_TO_ID[parts[8]])
    return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img-dir', default='/home/storageSDA1/liaojr/dataset/RSAR/corruptions/chaff/val/images')
    ap.add_argument('--ann-dir', default='/home/storageSDA1/liaojr/dataset/RSAR/val/annfiles')
    ap.add_argument('--num-images', type=int, default=1500)
    ap.add_argument('--expand-ratio', type=float, default=0.4)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    os.environ.setdefault('CGA_SCORER', 'sarclip')
    os.environ.setdefault('CGA_BACKEND', 'sarclip')
    templates = (os.environ.get('CGA_TEMPLATES')
                 or 'A SAR image of a {};This SAR patch shows a {}').split(';')
    cga = CGA(list(CLASSES), model='ViT-B-32',
              pretrained='/home/storageSDA1/Dataset/SARCLIP/ViT-B-32/vit_b_32_model.safetensors',
              cache_dir='/home/storageSDA1/Dataset/SARCLIP/ViT-B-32', precision='fp32',
              templates=templates, tau=float(os.environ.get('CGA_TAU', '100.0')),
              expand_ratio=args.expand_ratio, backend='sarclip')
    cga._first_call_logged = True

    stems = sorted(Path(args.ann_dir).glob('*.txt'))
    random.seed(args.seed)
    random.shuffle(stems)
    # Collect all SARCLIP prob vectors for real-target GT boxes.
    all_prob, all_label = [], []
    n_img = 0
    for ann in stems:
        img = find_image(args.img_dir, ann.stem)
        if img is None:
            continue
        boxes, labels = load_boxes(str(ann))
        if len(boxes) == 0:
            continue
        scores = np.full(len(boxes), 0.9, dtype=np.float32)
        logits, _ = cga(img, boxes, scores, labels)
        for i, prob in enumerate(logits):
            all_prob.append(np.asarray(prob, dtype=np.float64))
            all_label.append(int(labels[i]))
        n_img += 1
        if n_img >= args.num_images:
            break
    all_prob = np.array(all_prob)
    all_label = np.array(all_label)
    N = len(all_label)
    print(f'[diag] {n_img} images, {N} real-target GT boxes')

    # Precompute per-box stats.
    pred = all_prob.argmax(1)
    pred_prob = all_prob[np.arange(N), pred]
    label_prob = all_prob[np.arange(N), all_label]
    sorted_p = np.sort(all_prob, 1)[:, ::-1]
    margin = sorted_p[:, 0] - sorted_p[:, 1]
    ent = np.array([_prob_entropy_normalized(p) for p in all_prob])
    disagree = pred != all_label
    same_group = np.array([
        (p in GROUP_OF and l in GROUP_OF and GROUP_OF[p] == GROUP_OF[l])
        for p, l in zip(pred, all_label)
    ])
    print(f'[diag] baseline: disagreement on real targets = {disagree.mean()*100:.2f}% '
          f'({disagree.sum()}/{N}); of those same-confusion-group = '
          f'{same_group[disagree].mean()*100:.1f}%')

    def false_veto(pred_hi, label_lo, margin_thr, ent_thr, use_group=True):
        fire = disagree & (pred_prob >= pred_hi) & (label_prob <= label_lo) \
            & (margin >= margin_thr) & (ent <= ent_thr)
        if use_group:
            fire = fire & (~same_group)
        return fire

    print('\n=== FALSE-VETO RATE on real targets (want LOW) ===')
    print(f'{"pred_hi":>8}{"label_lo":>9}{"margin":>8}{"entropy":>8}{"group":>7}'
          f'{"fired":>8}{"rate%":>8}')
    configs = [
        (0.90, 0.05, 0.60, 0.35, True),
        (0.90, 0.05, 0.60, 0.35, False),
        (0.85, 0.10, 0.50, 0.45, True),
        (0.95, 0.02, 0.70, 0.25, True),
        (0.80, 0.15, 0.40, 0.55, True),
        (0.99, 0.01, 0.90, 0.15, True),
    ]
    for ph, ll, mg, et, ug in configs:
        fire = false_veto(ph, ll, mg, et, ug)
        print(f'{ph:>8.2f}{ll:>9.2f}{mg:>8.2f}{et:>8.2f}{str(ug):>7}'
              f'{fire.sum():>8}{fire.mean()*100:>8.3f}')

    # Per-class false-veto at the default config.
    fire = false_veto(0.90, 0.05, 0.60, 0.35, True)
    print('\n=== per-class false-veto at default (0.90,0.05,0.60,0.35,group) ===')
    for cid in range(len(CLASSES)):
        m = all_label == cid
        if m.sum() == 0:
            continue
        print(f'  {CLASSES[cid]:8s} n={m.sum():5d} fired={fire[m].sum():4d} '
              f'rate={fire[m].mean()*100:.3f}%')

    # For fired boxes, show what class SARCLIP flipped them to.
    if fire.sum() > 0:
        print('\n=== flips among fired (true->SARCLIP) at default ===')
        flip = defaultdict(int)
        for l, p in zip(all_label[fire], pred[fire]):
            flip[(CLASSES[l], CLASSES[p])] += 1
        for k, v in sorted(flip.items(), key=lambda x: -x[1]):
            print(f'  {k[0]} -> {k[1]}: {v}')


if __name__ == '__main__':
    main()
