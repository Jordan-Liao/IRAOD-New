"""One joint predicted-label embedding per dataset/domain/role comparison."""

import argparse
import csv
from pathlib import Path

from experiments.comparison.result_completion import (
    load_export, read_json, write_json, SCHEMA)


def joint_tsne(plan, dataset, domain, comparison, out_dir, cap=1000, perplexity=30):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import sklearn
    from sklearn.manifold import TSNE

    if plan["schema"] != SCHEMA or comparison not in ("ema", "student") or cap < 1:
        raise ValueError("Expected a v2 plan, ema/student comparison and positive cap")
    selected = sorted(
        (r for r in plan["runs"] if r["dataset"] == dataset and r["domain"] == domain
         and (r["role"] == comparison or (r["method"], r["role"]) == ("A", "source"))),
        key=lambda r: r["method"])
    if [r["method"] for r in selected] != list("ABCDEF"):
        raise ValueError("A/source plus B-F of the requested role are required")
    if len({r["checkpoint_domain"] for r in selected if r["method"] != "A"}) != 1:
        raise ValueError("B-F must share one adaptation domain within a comparison")
    pools, provenance, stats = [], [], []
    reference = None
    for run in selected:
        index, records = load_export(run)
        identity = (index["classes"], index["test_cfg"], index["rescale"],
                    index["feature_point"], run["image_ids"], run["show_score_thr"],
                    run["seed"])
        if reference is None:
            reference = identity
        elif identity != reference:
            raise ValueError("Comparison class order, NMS, selection or thresholds differ")
        feats, points = [], []
        for record, arrays in records:
            for row in np.flatnonzero(arrays["scores"] > run["show_score_thr"]):
                feats.append(arrays["features"][row])
                points.append({
                    **{k: run[k] for k in ("dataset", "domain", "method", "role", "seed",
                                           "checkpoint", "config", "checkpoint_domain")},
                    "image_id": record["image_id"],
                    "feature_file": str(Path(run["out_dir"]) / record["feature_file"]),
                    "feature_row": int(row),
                    "proposal_index": int(arrays["proposal_indices"][row]),
                    "flat_index": int(arrays["flat_indices"][row]),
                    "detection_index": int(arrays["detection_indices"][row]),
                    "predicted_label": int(arrays["labels"][row]),
                    "predicted_class": index["classes"][int(arrays["labels"][row])],
                    "score": float(arrays["scores"][row]),
                    **dict(zip(("cx", "cy", "w", "h", "angle"),
                               map(float, arrays["boxes"][row]))),
                })
        if not feats:
            raise ValueError(f"No detections above the fixed threshold: {run['run_id']}")
        pool = np.asarray(feats, dtype=np.float64)
        pools.append(pool)
        provenance.append(points)
        stats.append({
            "run_id": run["run_id"], "eligible_points": len(pool),
            "feature_dimension": pool.shape[1],
            "raw_mean": float(pool.mean()), "raw_std": float(pool.std()),
            "raw_l2_mean": float(np.linalg.norm(pool, axis=1).mean()),
            "code_commit": index["code_commit"],
        })
    count = min(cap, *(len(pool) for pool in pools))
    if count * len(pools) <= perplexity or perplexity <= 0:
        raise ValueError("Fixed perplexity must be positive and below joint sample size")
    if len({pool.shape[1] for pool in pools}) != 1:
        raise ValueError("Feature dimensions differ across methods")
    # One common transform fitted only to A/source's eligible predicted instances.
    mean, std = pools[0].mean(axis=0), pools[0].std(axis=0)
    scale = np.where(std == 0, 1.0, std)
    sampled, points = [], []
    for pool, metadata in zip(pools, provenance):
        chosen = np.sort(np.random.default_rng(42).choice(len(pool), count, replace=False))
        sampled.append((pool[chosen] - mean) / scale)
        points.extend(metadata[int(row)] for row in chosen)
    features = np.concatenate(sampled)
    estimator = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                     init="random", learning_rate=200.0, metric="euclidean",
                     method="barnes_hut", angle=0.5, n_jobs=1)
    coords = estimator.fit_transform(features)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "embedding.npz", coordinates=coords,
                        normalized_features=features, source_mean=mean,
                        source_std=std, source_scale=scale)
    for i, point in enumerate(points):
        point.update(point_index=i, x=float(coords[i, 0]), y=float(coords[i, 1]))
    with (out / "points.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)
    write_json(out / "protocol.json", {
        "schema": SCHEMA, "dataset": dataset, "domain": domain, "comparison": comparison,
        "sampling_seed": 42, "sample_cap": cap, "points_per_method": count,
        "sampling": "uniform without replacement; equal count; no class/GT selection",
        "normalization": "shared z-score fitted to all eligible A/source points in this domain",
        "normalization_reference": selected[0],
        "zero_std_channels": int((std == 0).sum()),
        "source_stats_count": len(pools[0]), "feature_stats": stats,
        "tsne_parameters": estimator.get_params(), "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__, "kl_divergence": float(estimator.kl_divergence_),
        "runs": selected, "classes": reference[0],
    })
    limits = np.column_stack((coords.min(axis=0), coords.max(axis=0)))
    padding = np.maximum((limits[:, 1] - limits[:, 0]) * 0.05, 1e-3)
    palette = plt.get_cmap("tab20", len(reference[0]))
    for i, run in enumerate(selected):
        start, end = i * count, (i + 1) * count
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.scatter(coords[start:end, 0], coords[start:end, 1], s=5, linewidths=0,
                   c=[p["predicted_label"] for p in points[start:end]],
                   cmap=palette, vmin=-0.5, vmax=len(reference[0]) - 0.5)
        ax.set(xlim=(limits[0, 0] - padding[0], limits[0, 1] + padding[0]),
               ylim=(limits[1, 0] - padding[1], limits[1, 1] + padding[1]))
        ax.set_aspect("equal")
        ax.set_axis_off()
        fig.savefig(out / f"{dataset}_{domain}_{comparison}_{run['method']}_{run['role']}.pdf",
                    bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    # Class/color correspondence is separate so panels remain title-free.
    fig, ax = plt.subplots(figsize=(6, 1 + len(reference[0]) * 0.15))
    for label, name in enumerate(reference[0]):
        ax.scatter([], [], color=palette(label), s=15, label=name)
    ax.legend(frameon=False, ncol=3, loc="center")
    ax.set_axis_off()
    fig.savefig(out / "predicted_class_legend.pdf", bbox_inches="tight")
    plt.close(fig)
    return coords, points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--dataset", required=True, choices=("RSAR", "DIOR"))
    parser.add_argument("--domain", required=True)
    parser.add_argument("--comparison", required=True, choices=("ema", "student"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cap", type=int, default=1000)
    parser.add_argument("--perplexity", type=float, default=30)
    args = parser.parse_args()
    joint_tsne(read_json(args.plan), args.dataset, args.domain, args.comparison,
               args.out_dir, args.cap, args.perplexity)


if __name__ == "__main__":
    main()
