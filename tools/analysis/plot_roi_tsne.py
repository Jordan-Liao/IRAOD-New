#!/usr/bin/env python
"""Plot Fig.7-style t-SNE panels from exported ROI features.

The exporter stores a fixed ROI matrix per image and the corresponding
prediction records in index.json.  Prediction rows are aligned to the first
N ROI rows for each image, where N is the number of recorded predictions.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


METHOD_LABELS = {
    'A_source': 'Source only',
    'B_ema': 'Mean Teacher',
    'B_student': 'Mean Teacher Student',
    'C_ema': 'CLIP-CGA',
    'D_ema': 'SARCLIP-CGA',
    'E_ema': 'VLST',
    'F_ema': 'CGA+VLST',
}
CLASS_NAMES = ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor')
COLORS = ('#377eb8', '#e41a1c', '#4daf4a', '#984ea3', '#ff7f00', '#a65628')


def load_method(root, method, score_thr):
    method_dir = root / method
    with (method_dir / 'index.json').open() as f:
        index = json.load(f)
    features, labels, scores = [], [], []
    skipped = 0
    for record in index['records']:
        image_id = record['image_id']
        feature_path = method_dir / (Path(image_id).stem + '.npy')
        if not feature_path.exists():
            skipped += 1
            continue
        matrix = np.load(feature_path, mmap_mode='r')
        preds = record.get('preds', [])
        n = min(len(preds), matrix.shape[0])
        for row, pred in enumerate(preds[:n]):
            label, score = int(pred[0]), float(pred[1])
            if score >= score_thr and 0 <= label < len(CLASS_NAMES):
                features.append(np.asarray(matrix[row], dtype=np.float32))
                labels.append(label)
                scores.append(score)
    if skipped:
        print(f'[{method}] skipped missing feature files: {skipped}')
    if not features:
        raise RuntimeError(f'{method}: no ROI rows remain at score threshold {score_thr}')
    return np.stack(features), np.asarray(labels), np.asarray(scores)


def balanced_sample(x, y, score, max_points, seed):
    if len(x) <= max_points:
        return x, y, score
    rng = np.random.default_rng(seed)
    chosen = []
    per_class = max(1, max_points // len(CLASS_NAMES))
    for cls in range(len(CLASS_NAMES)):
        ids = np.flatnonzero(y == cls)
        if len(ids):
            chosen.extend(rng.choice(ids, min(per_class, len(ids)), replace=False))
    remaining = np.setdiff1d(np.arange(len(x)), np.asarray(chosen, dtype=int))
    need = max_points - len(chosen)
    if need > 0 and len(remaining):
        chosen.extend(rng.choice(remaining, min(need, len(remaining)), replace=False))
    ids = np.asarray(chosen[:max_points], dtype=int)
    return x[ids], y[ids], score[ids]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True,
                        help='.../roi_features/RSAR/<corruption> directory')
    parser.add_argument('--methods', nargs='+',
                        default=['A_source', 'B_ema', 'B_student', 'C_ema', 'D_ema', 'E_ema', 'F_ema'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--score-thr', type=float, default=0.05)
    parser.add_argument('--max-points', type=int, default=5000)
    parser.add_argument('--perplexity', type=float, default=30.0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    panels = []
    for method in args.methods:
        x, y, score = load_method(args.root, method, args.score_thr)
        x, y, score = balanced_sample(x, y, score, args.max_points, args.seed)
        print(f'{method}: {len(x)} points, mean score={score.mean():.3f}')
        x = StandardScaler().fit_transform(x)
        panels.append((method, x, y))

    all_x = np.concatenate([x for _, x, _ in panels], axis=0)
    embedding = TSNE(
        n_components=2, perplexity=args.perplexity, init='pca',
        learning_rate='auto', random_state=args.seed,
    ).fit_transform(all_x)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4),
                             squeeze=False)
    offset = 0
    for col, (method, x, y) in enumerate(panels):
        ax = axes[0, col]
        z = embedding[offset:offset + len(x)]
        offset += len(x)
        for cls, name in enumerate(CLASS_NAMES):
            ids = y == cls
            if ids.any():
                ax.scatter(z[ids, 0], z[ids, 1], s=7, alpha=0.58,
                           c=COLORS[cls], label=name, linewidths=0)
        ax.set_title(METHOD_LABELS.get(method, method), fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
    handles, labels = axes[0, -1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(CLASS_NAMES),
               frameon=False, bbox_to_anchor=(0.5, -0.015))
    fig.suptitle(f'RSAR chaff ROI features: t-SNE (score >= {args.score_thr:g})',
                 y=1.01, fontsize=13)
    fig.tight_layout()
    fig.savefig(args.out, dpi=250, bbox_inches='tight')
    print(f'written: {args.out}')


if __name__ == '__main__':
    main()
