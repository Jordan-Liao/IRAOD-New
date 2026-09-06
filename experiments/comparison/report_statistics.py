"""Seed-block statistics, without the unrelated same-GPU pairing policy."""

import math

import numpy as np

from experiments.comparison.result_completion import DOMAINS
from tools.cga_research.statistics import (
    bootstrap_mean_ci, holm_adjust, paired_t_test, sign_flip_test)


SEEDS = (42, 43, 44)
METRICS = ("mPC", "delta_A", "rPC_percent", "method_clean_normalized_percent",
           "clean_mAP50", "clean_delta_A")


def describe(values):
    """Only a complete three-seed block receives inferential summaries."""
    result = {"n": len(values), "mean": None, "sample_std": None,
              "bootstrap_ci95_low": None, "bootstrap_ci95_high": None}
    if len(values) != len(SEEDS):
        return result
    result["mean"] = float(np.mean(values))
    result["sample_std"] = float(np.std(values, ddof=1))
    low, high = bootstrap_mean_ci(values, seed=42)
    result.update(bootstrap_ci95_low=low, bootstrap_ci95_high=high)
    return result


def summarize(rows, roles=("ema",)):
    lookup = {(r["dataset"], r["domain"], r["method"], r["seed"], r["role"]): r
              for r in rows if r["status"] == "complete"}

    def value(dataset, domain, method, seed, role):
        row = lookup.get((dataset, domain, method, seed, role))
        return None if row is None else row["mAP50"]

    per_seed, per_domain, summary, tests = [], [], [], []
    for dataset, domains in DOMAINS.items():
        for domain in domains:
            baseline = value(dataset, domain, "A", 42, "source")
            per_domain.append({
                "dataset": dataset, "domain": domain, "method": "A", "role": "source",
                "status": "fixed_source" if baseline is not None else "incomplete",
                "n": int(baseline is not None), "mean": baseline, "sample_std": None,
                "bootstrap_ci95_low": None, "bootstrap_ci95_high": None,
            })
            for role in roles:
                for method in "BCDEF":
                    values = [value(dataset, domain, method, seed, role) for seed in SEEDS]
                    available = [x for x in values if x is not None]
                    per_domain.append({
                        "dataset": dataset, "domain": domain, "method": method, "role": role,
                        "status": "complete_low_power" if len(available) == 3 else "incomplete",
                        **describe(available),
                    })
        source = [value(dataset, d, "A", 42, "source") for d in domains[1:]]
        source_clean = value(dataset, "clean", "A", 42, "source")
        source_mpc = float(np.mean(source)) if all(x is not None for x in source) else None
        source_values = {
            "mPC": source_mpc, "delta_A": 0.0 if source_mpc is not None else None,
            "rPC_percent": (100 * source_mpc / source_clean
                            if source_mpc is not None and source_clean else None),
            "method_clean_normalized_percent": None,
            "clean_mAP50": source_clean,
            "clean_delta_A": 0.0 if source_clean is not None else None,
        }
        for metric in METRICS:
            measured = source_values[metric]
            summary.append({
                "dataset": dataset, "method": "A", "role": "source", "metric": metric,
                "status": "fixed_source" if measured is not None else "incomplete",
                "n": int(measured is not None), "mean": measured, "sample_std": None,
                "bootstrap_ci95_low": None, "bootstrap_ci95_high": None,
                "seeds": "42 (one fixed source measurement)",
            })
        for role in roles:
            for method in "BCDEF":
                blocks = []
                for seed in SEEDS:
                    corrupted = [value(dataset, d, method, seed, role) for d in domains[1:]]
                    clean = value(dataset, "clean", method, seed, role)
                    mpc = (float(np.mean(corrupted))
                           if all(x is not None for x in corrupted) else None)
                    paired = (float(np.mean([m - a for m, a in zip(corrupted, source)]))
                              if mpc is not None and source_mpc is not None else None)
                    block = {
                        "dataset": dataset, "method": method, "role": role, "seed": seed,
                        "corruptions_observed": sum(x is not None for x in corrupted),
                        "corruptions_expected": len(domains) - 1,
                        "status": "complete_corruption_block" if mpc is not None else "incomplete",
                        "mPC": mpc, "delta_A": paired,
                        "rPC_percent": (100 * mpc / source_clean
                                        if mpc is not None and source_clean else None),
                        "method_clean_normalized_percent": (
                            100 * mpc / clean if mpc is not None and clean else None),
                        "clean_mAP50": clean,
                        "clean_delta_A": (clean - source_clean
                                          if clean is not None and source_clean is not None
                                          else None),
                    }
                    blocks.append(block)
                    per_seed.append(block)
                for metric in METRICS:
                    vals = [b[metric] for b in blocks if b[metric] is not None]
                    summary.append({
                        "dataset": dataset, "method": method, "role": role, "metric": metric,
                        "status": "complete_low_power" if len(vals) == 3 else "incomplete",
                        "seeds": ",".join(str(b["seed"]) for b in blocks if b[metric] is not None),
                        **describe(vals),
                    })
                for metric in ("delta_A", "clean_delta_A"):
                    differences = [b[metric] for b in blocks if b[metric] is not None]
                    record = {
                        "dataset": dataset, "method": method, "role": role, "metric": metric,
                        "family": f"{dataset}/{role}/{metric}/B-F",
                        "n": len(differences), "status": "incomplete",
                        "differences": differences, "t": None, "df": None, "t_p": None,
                        "t_na_reason": "requires seeds 42,43,44",
                        "sign_flip_p": None, "sign_assignments": None,
                        "t_p_holm": None, "sign_flip_p_holm": None,
                        "holm_status": "family_incomplete",
                    }
                    if len(differences) == 3:
                        record["status"] = "complete_low_power"
                        perm = sign_flip_test(differences, seed=42)
                        record.update(sign_flip_p=perm["p"], sign_assignments=perm["assignments"])
                        if math.isclose(float(np.std(differences, ddof=1)), 0, abs_tol=1e-15):
                            record["t_na_reason"] = "zero variance; t-test undefined"
                        else:
                            t = paired_t_test(differences)
                            record.update(t=t["t"], df=t["df"], t_p=t["p"], t_na_reason=None)
                    tests.append(record)
    # Do not shrink the planned B-F family to whichever methods finished first.
    for family in sorted({t["family"] for t in tests}):
        members = [t for t in tests if t["family"] == family]
        if any(t["n"] != 3 for t in members):
            continue
        for raw, adjusted in (("t_p", "t_p_holm"), ("sign_flip_p", "sign_flip_p_holm")):
            # Undefined tests occupy their planned slot conservatively with p=1.
            values = {t["method"]: t[raw] if t[raw] is not None else 1.0 for t in members}
            corrected = holm_adjust(values)
            for t in members:
                t[adjusted] = corrected[t["method"]] if t[raw] is not None else None
                t["holm_status"] = "complete_family"
    return {"per_seed": per_seed, "per_domain": per_domain,
            "summary": summary, "paired_statistics": tests,
            "statistical_protocol": {
                "unit": "adaptation seed; corruptions averaged within seed first",
                "pairing": "same dataset/domain/seed; GPU topology is provenance, not a matching key",
                "source": "one fixed A; no repeated independent source measurements",
                "uncertainty": "conditional on fixed source; adaptation randomness only",
                "n": 3, "warning": "low power; exact two-sided sign-flip minimum p is 0.25",
                "ci": "95% percentile bootstrap, 100000 seed-block resamples, RNG seed42; fragile at n=3",
                "t_test": "two-sided one-sample t on seed-level paired deltas; assumes normal differences",
                "holm": "planned five B-F tests per dataset/role/metric; no adjustment until family complete",
                "rPC": "100 * seed mPC / fixed A clean TEST",
                "method_clean_normalized": "separate: 100 * seed mPC / same-method same-seed clean",
            }}
