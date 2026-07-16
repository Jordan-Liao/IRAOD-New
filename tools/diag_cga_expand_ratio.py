#!/usr/bin/env python
"""Zero-training diagnostic: does tightening CGA crop (expand_ratio) improve the
SARCLIP label_prob signal that legacy CGA depends on, under chaff drift?

Replicates the EXACT CGA instance used in self-training (same SARCLIP + LoRA +
templates + preprocess), but instead of detector boxes it feeds the clean val
GT boxes cropped from the *chaff-corrupted* val images (= the unlabeled set of
self-training: img_prefix_u=corruptions/chaff/val, GT=val/annfiles).

For each expand_ratio it reports, over real targets:
  - mean label_prob  : SARCLIP softmax prob on the TRUE class
  - top1 acc         : fraction where argmax == true class (== legacy "agree",
                       i.e. boxes legacy leaves UNTOUCHED / does not penalize)
Higher on both = cleaner signal = legacy penalizes fewer real targets.
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

from sfod.cga import CGA

CLASSES = ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor')
CLS_TO_ID = {c: i for i, c in enumerate(CLASSES)}


def find_image(img_dir, stem):
    for ext in ('.jpg', '.png', '.bmp', '.jpeg', '.tif', '.tiff'):
        p = os.path.join(img_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None


def load_boxes(ann_path):
    """DOTA poly -> AABB xyxy + label id (matches CGA's obb2xyxy AABB usage)."""
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
            cls = parts[8]
            if cls not in CLS_TO_ID:
                continue
            xs, ys = coords[0::2], coords[1::2]
            boxes.append([min(xs), min(ys), max(xs), max(ys)])
            labels.append(CLS_TO_ID[cls])
    return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img-dir',
                    default='/home/storageSDA1/liaojr/dataset/RSAR/corruptions/chaff/val/images')
    ap.add_argument('--ann-dir',
                    default='/home/storageSDA1/liaojr/dataset/RSAR/val/annfiles')
    ap.add_argument('--num-images', type=int, default=800)
    ap.add_argument('--expand-ratios', default='0.4,0.1,0.0')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    ratios = [float(x) for x in args.expand_ratios.split(',')]

    # Build CGA exactly like self-training would (env vars drive backend/LoRA).
    os.environ.setdefault('CGA_SCORER', 'sarclip')
    os.environ.setdefault('CGA_BACKEND', 'sarclip')
    tau = float(os.environ.get('CGA_TAU', '100.0'))
    templates = (os.environ.get('CGA_TEMPLATES')
                 or 'A SAR image of a {};This SAR patch shows a {}').split(';')
    model = os.environ.get('SARCLIP_MODEL', 'ViT-B-32')
    pretrained = os.environ.get(
        'SARCLIP_PRETRAINED',
        '/home/storageSDA1/Dataset/SARCLIP/ViT-B-32/vit_b_32_model.safetensors')
    cache_dir = os.environ.get('SARCLIP_CACHE_DIR',
                               '/home/storageSDA1/Dataset/SARCLIP/ViT-B-32')
    precision = os.environ.get('SARCLIP_PRECISION', 'fp32')
    force_grayscale = os.environ.get('CGA_FORCE_GRAYSCALE', '0').lower() in ('1', 'true', 'yes')

    print(f'[diag] building CGA sarclip lora={os.environ.get("SARCLIP_LORA","<unset>")} '
          f'templates={templates} tau={tau}')
    cga = CGA(list(CLASSES), model=model, pretrained=pretrained, cache_dir=cache_dir,
              precision=precision, templates=templates, tau=tau,
              expand_ratio=ratios[0], force_grayscale=force_grayscale, backend='sarclip')

    # Collect a fixed sample of (img_path, boxes, labels).
    stems = sorted(Path(args.ann_dir).glob('*.txt'))
    random.seed(args.seed)
    random.shuffle(stems)
    samples = []
    total_boxes = 0
    for ann in stems:
        img = find_image(args.img_dir, ann.stem)
        if img is None:
            continue
        boxes, labels = load_boxes(str(ann))
        if len(boxes) == 0:
            continue
        samples.append((img, boxes, labels))
        total_boxes += len(boxes)
        if len(samples) >= args.num_images:
            break
    print(f'[diag] sampled {len(samples)} images, {total_boxes} GT boxes')

    results = {}
    for r in ratios:
        cga.expand_ratio = float(r)
        cga._first_call_logged = True  # silence per-call log
        sum_label_prob = 0.0
        n = 0
        top1 = 0
        per_cls_prob = defaultdict(float)
        per_cls_n = defaultdict(int)
        per_cls_top1 = defaultdict(int)
        for img, boxes, labels in samples:
            scores = np.full(len(boxes), 0.9, dtype=np.float32)
            logits, _ = cga(img, boxes, scores, labels)
            if len(logits) == 0:
                continue
            for i, prob in enumerate(logits):
                lab = int(labels[i])
                lp = float(prob[lab])
                pred = int(np.argmax(prob))
                sum_label_prob += lp
                n += 1
                per_cls_prob[lab] += lp
                per_cls_n[lab] += 1
                if pred == lab:
                    top1 += 1
                    per_cls_top1[lab] += 1
        mean_lp = sum_label_prob / max(n, 1)
        acc = top1 / max(n, 1)
        results[r] = (mean_lp, acc, n)
        print(f'\n[diag] expand_ratio={r:.2f}  boxes={n}  '
              f'mean_label_prob={mean_lp:.4f}  top1_acc(agree)={acc:.4f}')
        for cid in range(len(CLASSES)):
            if per_cls_n[cid] == 0:
                continue
            print(f'    {CLASSES[cid]:8s} n={per_cls_n[cid]:5d} '
                  f'mean_lp={per_cls_prob[cid]/per_cls_n[cid]:.4f} '
                  f'top1={per_cls_top1[cid]/per_cls_n[cid]:.4f}')

    print('\n==== SUMMARY (chaff/val GT boxes, real targets) ====')
    print(f'{"expand_ratio":>14} {"mean_label_prob":>16} {"top1_acc(agree)":>16} {"n":>8}')
    for r in ratios:
        mean_lp, acc, n = results[r]
        print(f'{r:>14.2f} {mean_lp:>16.4f} {acc:>16.4f} {n:>8d}')
    print('\nHigher mean_label_prob AND top1 = cleaner signal = legacy penalizes '
          'fewer REAL targets under chaff. If tightening crop raises both, '
          'expand_ratio was poisoning the p_clip signal.')


if __name__ == '__main__':
    main()
