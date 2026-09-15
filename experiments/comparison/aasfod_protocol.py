"""Frozen, pre-result choices for the approved post-hoc-dropout AASFOD port."""

import math


TSD_CHOICE = {
    "reference": "ChuQiaosong/AASFOD@c383cacdc050b804c95a6127d994cb1eba0596df",
    "passes": 20, "sigma": 0.8, "similar": "highest_variance_floor_N_over_5",
    "variance": "sum_proposals(sum_class_population_variance * sum_delta_population_variance)",
    "box_space": "unchanged_five_regression_deltas",
    "dropout": "post_hoc_p0.5_after_each_shared_FC_ReLU_not_source_trained",
}


def budget(target_images):
    """Common padded global32 epoch; retain author30:20 stages within that budget."""
    total = math.ceil(target_images / 32)
    alignment = total * 3 // 5
    # Author README: batch1, 30+20 epochs of10000 steps, EMA each2500.
    # Compress the interval by the same total-duration ratio; nearest integer,
    # minimum one completed optimizer update. No step0 or final flush.
    cadence = max(1, math.floor(2500 * total / 500000 + 0.5))
    return dict(total_updates=total, alignment_updates=alignment,
                fns_updates=total - alignment, ema_interval=cadence, ema_momentum=0.99,
                global_original_images=32, alignment_pairs=16, fns_canvases=8,
                final_checkpoint_iteration=total + 1)


def validate_split(split, cell):
    """A real TSD result must correspond to this source, seed and VAL-only cell."""
    expected = {key: cell[key] for key in
                ("dataset", "domain", "seed", "source_checkpoint", "target_val")}
    if split["identity"] != expected or split["choice"] != TSD_CHOICE:
        raise ValueError("TSD identity/algorithm differs from the frozen detector cell")
    if split["status"] != "complete":
        raise ValueError("Smoke/partial TSD cannot drive formal adaptation")
    a, b = split["similar"], split["dissimilar"]
    scores = split["scores"]
    if (len(scores) != cell["unlabeled_epoch_size"]
            or len(a) != len(scores) // 5 or not a or not b
            or set(a) & set(b) or set(a + b) != set(scores)):
        raise ValueError("TSD must partition the exact target VAL images")
    if not all(math.isfinite(v) and v >= 0 for v in scores.values()):
        raise ValueError("TSD contains invalid variances")
    ordered = sorted(scores, key=lambda name: (scores[name], name))
    if max(scores.values()) == min(scores.values()):
        raise ValueError("All tied variances are not a valid stochastic TSD")
    if a != ordered[-len(a):] or b != ordered[:-len(a)]:
        raise ValueError("TSD similar subset must be the highest-variance20%")
