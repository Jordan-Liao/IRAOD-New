"""One joint predicted-label embedding per dataset/domain/role comparison."""

import argparse
import csv
import heapq
from pathlib import Path

from experiments.comparison.result_completion import (
    load_export, read_json, write_json, SCHEMA, comparison_runs, plan_methods, FEATURE_VERSION)


def joint_tsne(plan, dataset, domain, comparison, out_dir, cap=1000, perplexity=30):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import sklearn
    from sklearn.manifold import TSNE

    if plan["schema"] != SCHEMA or comparison not in ("ema", "student") or cap < 1:
        raise ValueError("Expected a v3 full-test plan, ema/student comparison and positive cap")
    selected = comparison_runs(plan, dataset, domain, comparison)
    adaptation_seed = plan.get("adaptation_seed", 42)
    if len({r["checkpoint_domain"] for r in selected if r["method"] != "A"}) != 1:
        raise ValueError("Methods must share one adaptation domain within a comparison")
    out = Path(out_dir)
    if out.exists():
        raise FileExistsError(out)
    reservoirs, stats = [], []
    source_mean = source_std = None
    reference = None
    for run in selected:
        index, records = load_export(run)
        identity = (index["classes"], index["test_cfg"], index["rescale"],
                    index["feature_point"], run["image_ids"], run["show_score_thr"])
        if reference is None:
            reference = identity
        elif identity != reference:
            raise ValueError("Comparison class order, NMS, selection or thresholds differ")
        heap = []
        rng = np.random.default_rng(42)
        total, norm_sum = 0, 0.0
        mean = m2 = None
        for record, arrays in records:
            rows = np.flatnonzero(arrays["scores"] > run["show_score_thr"])
            if not len(rows):
                continue
            batch = arrays["features"][rows].astype(np.float64)
            batch_mean = batch.mean(axis=0)
            batch_m2 = ((batch - batch_mean) ** 2).sum(axis=0)
            if mean is None:
                mean, m2 = batch_mean, batch_m2
            else:
                if mean.shape != batch_mean.shape:
                    raise ValueError("Feature dimension changed within an export")
                delta = batch_mean - mean
                m2 += batch_m2 + delta ** 2 * total * len(batch) / (total + len(batch))
                mean += delta * len(batch) / (total + len(batch))
            norm_sum += float(np.linalg.norm(batch, axis=1).sum())
            priorities = rng.random(len(rows))
            # Bottom-k random priorities are a uniform bounded reservoir.
            candidates = np.argpartition(priorities, min(cap, len(rows)) - 1)[:cap]
            for offset in candidates:
                priority = float(priorities[offset])
                if len(heap) == cap and priority >= -heap[0][0]:
                    continue
                row = rows[offset]
                point = {
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
                }
                item = (-priority, -(total + int(offset)), batch[offset].copy(), point)
                if len(heap) < cap:
                    heapq.heappush(heap, item)
                else:
                    heapq.heapreplace(heap, item)
            total += len(rows)
        if not total:
            raise ValueError(f"No detections above the fixed threshold: {run['run_id']}")
        reservoirs.append(heap)
        if run["method"] == "A":
            source_mean, source_std = mean.copy(), np.sqrt(m2 / total)
        raw_mean = float(mean.mean())
        stats.append({
            "run_id": run["run_id"], "eligible_points": total,
            "reservoir_points": len(heap), "feature_dimension": len(mean),
            "raw_mean": raw_mean,
            "raw_std": float(np.sqrt(np.mean(m2 / total + (mean - raw_mean) ** 2))),
            "raw_l2_mean": norm_sum / total,
            "code_commit": index["code_commit"],
        })
    count = min(len(heap) for heap in reservoirs)
    if count * len(reservoirs) <= perplexity or perplexity <= 0:
        raise ValueError("Fixed perplexity must be positive and below joint sample size")
    if len({s["feature_dimension"] for s in stats}) != 1:
        raise ValueError("Feature dimensions differ across methods")
    # One common transform fitted only to A/source's eligible predicted instances.
    mean, std = source_mean, source_std
    scale = np.where(std == 0, 1.0, std)
    sampled, points = [], []
    for heap in reservoirs:
        chosen = sorted(sorted(heap, key=lambda entry: -entry[0])[:count],
                        key=lambda entry: -entry[1])
        sampled.append((np.stack([entry[2] for entry in chosen]) - mean) / scale)
        points.extend(entry[3] for entry in chosen)
    features = np.concatenate(sampled)
    estimator = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                     init="random", learning_rate=200.0, metric="euclidean",
                     method="barnes_hut", angle=0.5, n_jobs=1)
    coords = estimator.fit_transform(features)
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
        "adaptation_seed": adaptation_seed,
        "methods": plan_methods(plan), "feature_version": FEATURE_VERSION,
        "budget_labels": {r["method"]: {
            "budget_group": r.get("budget_group", "source" if r["method"] == "A"
                                 else "common_one_epoch"),
            "final_checkpoint_iteration": r.get("final_checkpoint_iteration"),
            "rank_with_common_one_epoch": r.get("rank_with_common_one_epoch", r["method"] != "A"),
        } for r in selected},
        "interpretation": "Qualitative geometry, not statistical ranking across budgets; "
                          "SFYOLO uses two detector epochs plus TAM.",
        "sampling_seed": 42, "sample_cap": cap, "points_per_method": count,
        "roi_scope": "full_test",
        "sampling": "streamed bottom-k random-priority reservoir; equal count; no class/GT selection",
        "memory_bound": "one image NPZ plus at most cap feature vectors per method",
        "normalization": "shared z-score fitted to all eligible A/source points in this domain",
        "normalization_reference": selected[0],
        "zero_std_channels": int((std == 0).sum()),
        "source_stats_count": stats[0]["eligible_points"], "feature_stats": stats,
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
