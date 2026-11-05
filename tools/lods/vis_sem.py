"""Side-by-side original vs SEM-enhanced grids for LODS visual checks.

Annotations are read only to choose and outline illustrative targets in the
figure; they never reach SEM training or detector adaptation.

    python tools/lods/vis_sem.py --sem SEM.pth --img-dir DIR --ann-dir DIR \
        --fmt rsar|dior --classes ship --mode small|slender|dense --out grid.jpg
"""
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import random
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import torch

sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
from sfod.lods_sem import PIXEL_MEANS_BGR, StyleEnhancementModule  # noqa: E402

PM = np.array(PIXEL_MEANS_BGR, np.float32)


def read_ann(path, fmt):
    """-> list of (class, 4x2 polygon)."""
    out = []
    if fmt == 'rsar':
        for line in open(path):
            p = line.split()
            if len(p) >= 9:
                out.append((p[8], np.array(p[:8], np.float32).reshape(4, 2)))
    else:
        for obj in ET.parse(path).getroot().iter('object'):
            rb = obj.find('robndbox')
            k = ['x_left_top', 'y_left_top', 'x_right_top', 'y_right_top',
                 'x_right_bottom', 'y_right_bottom', 'x_left_bottom', 'y_left_bottom']
            out.append((obj.find('name').text.lower(),
                        np.array([float(rb.find(x).text) for x in k], np.float32).reshape(4, 2)))
    return out


def poly_stats(poly):
    rect = cv2.minAreaRect(poly)
    w, h = rect[1]
    return w * h, max(w, h) / max(min(w, h), 1e-3)


def select(img_dir, ann_dir, fmt, classes, mode, n, seed):
    rng = random.Random(seed)
    names = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.bmp', '.tif')))
    ext = '.txt' if fmt == 'rsar' else '.xml'
    scored = []
    for f in names:
        ap = osp.join(ann_dir, osp.splitext(f)[0] + ext)
        if not osp.exists(ap):
            continue
        objs = [(c, p) for c, p in read_ann(ap, fmt) if c in classes]
        if not objs:
            continue
        stats = [poly_stats(p) for _, p in objs]
        if mode == 'small':
            score = sum(a < 20 * 20 for a, _ in stats)
        elif mode == 'slender':
            score = max(r for _, r in stats) if max(r for _, r in stats) >= 3 else 0
        else:  # dense
            score = len(objs)
        if score > 0:
            scored.append((score + rng.random() * 1e-3, f, objs))
    scored.sort(reverse=True)
    return scored[:n]


def detector_scale(img, scale=(800, 800)):
    h, w = img.shape[:2]
    r = min(max(scale) / max(h, w), min(scale) / min(h, w))
    return cv2.resize(img, (int(w * r + 0.5), int(h * r + 0.5))), r


@torch.no_grad()
def enhance(sem, img, style, alpha, device):
    c = torch.from_numpy((img.astype(np.float32) - PM).transpose(2, 0, 1))[None].to(device)
    s = cv2.resize(style.astype(np.float32) - PM, (img.shape[1], img.shape[0]))
    s = torch.from_numpy(s.transpose(2, 0, 1))[None].to(device)
    y = sem.add_style(c, s, alpha)[0].cpu().numpy().transpose(1, 2, 0) + PM
    return y.clip(0, 255).astype(np.uint8)


def target_metrics(orig, enh, polys):
    """Gradient correlation inside targets; target-vs-ring contrast before/after."""
    g = lambda im: cv2.magnitude(*[cv2.Sobel(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32), cv2.CV_32F, *d) for d in ((1, 0), (0, 1))])  # noqa: E731
    go, ge = g(orig), g(enh)
    m = np.zeros(orig.shape[:2], np.uint8)
    for p in polys:
        cv2.fillPoly(m, [p.astype(np.int32)], 1)
    ring = cv2.dilate(m, np.ones((9, 9), np.uint8)) - m
    gray = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)  # noqa: E731
    def contrast(im):
        return float(abs(gray(im)[m > 0].mean() - gray(im)[ring > 0].mean()) / (gray(im).std() + 1e-3))
    corr = float(np.corrcoef(go[m > 0], ge[m > 0])[0, 1]) if m.sum() > 4 else float('nan')
    return dict(grad_corr_in_targets=corr, contrast_orig=contrast(orig), contrast_enh=contrast(enh))


def zoom_box(polys, shape, size=160):
    c = np.concatenate(polys).mean(0)
    x0 = int(np.clip(c[0] - size / 2, 0, max(shape[1] - size, 0)))
    y0 = int(np.clip(c[1] - size / 2, 0, max(shape[0] - size, 0)))
    return x0, y0, size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sem', required=True)
    ap.add_argument('--img-dir', required=True)
    ap.add_argument('--ann-dir', required=True)
    ap.add_argument('--fmt', choices=['rsar', 'dior'], required=True)
    ap.add_argument('--classes', nargs='+', required=True)
    ap.add_argument('--mode', choices=['small', 'slender', 'dense'], required=True)
    ap.add_argument('--n', type=int, default=4)
    ap.add_argument('--alpha', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sem = StyleEnhancementModule()
    sem.load_state_dict(torch.load(args.sem, map_location='cpu'))
    sem.eval().to(device)
    names = sorted(os.listdir(args.img_dir))
    rng = random.Random(args.seed)
    rows, report = [], []
    for _, f, objs in select(args.img_dir, args.ann_dir, args.fmt, set(args.classes), args.mode, args.n, args.seed):
        img, r = detector_scale(cv2.imread(osp.join(args.img_dir, f), cv2.IMREAD_COLOR))
        style = cv2.imread(osp.join(args.img_dir, names[rng.randrange(len(names))]), cv2.IMREAD_COLOR)
        enh = enhance(sem, img, style, args.alpha, device)
        polys = [p * r for _, p in objs]
        report.append(dict(image=f, n_targets=len(polys), **target_metrics(img, enh, polys)))
        x0, y0, s = zoom_box(polys, img.shape)
        zo, ze = img[y0:y0 + s, x0:x0 + s].copy(), enh[y0:y0 + s, x0:x0 + s].copy()
        for p in polys:
            cv2.polylines(zo, [(p - [x0, y0]).astype(np.int32)], True, (0, 255, 0), 1)
        full = lambda im: cv2.resize(im, (400, int(400 * im.shape[0] / im.shape[1])))  # noqa: E731
        zoom = lambda im: cv2.resize(im, (400, 400), interpolation=cv2.INTER_NEAREST)  # noqa: E731
        fo, fe = full(img), full(enh)
        h = max(fo.shape[0], 400)
        pad = lambda im: cv2.copyMakeBorder(im, 0, h - im.shape[0], 0, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255))  # noqa: E731
        rows.append(np.hstack([pad(fo), pad(fe), pad(zoom(zo)), pad(zoom(ze))]))
    if not rows:
        raise SystemExit('no matching images')
    w = max(r.shape[1] for r in rows)
    grid = np.vstack([cv2.copyMakeBorder(r, 0, 4, 0, w - r.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for r in rows])
    os.makedirs(osp.dirname(osp.abspath(args.out)), exist_ok=True)
    cv2.imwrite(args.out, grid)
    json.dump(report, open(osp.splitext(args.out)[0] + '.json', 'w'), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
