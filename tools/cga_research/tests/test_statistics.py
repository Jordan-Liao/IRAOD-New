from __future__ import annotations

import contextlib
import io
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scipy import stats

from tools.cga_research.statistics import (
    align_pairs,
    analyze_comparisons,
    audit_experiment_rows,
    bootstrap_mean_ci,
    holm_adjust,
    main as statistics_main,
    render_markdown,
    sign_flip_test,
    summarize_comparison,
)


def experiment(
    method: str,
    seed: int,
    final_map: float,
    gpu_uuid: str = "GPU-a",
    work_suffix: str = "",
) -> dict[str, object]:
    return {
        "method": method,
        "actual_seed": seed,
        "final_map": final_map,
        "gpu_uuid": gpu_uuid,
        "work_dir": f"/tmp/{method}/seed_{seed}{work_suffix}",
        "status": "completed",
        "complete": True,
    }


def write_success_sentinel(
    row: dict[str, object],
    *,
    overrides: dict[str, object] | None = None,
) -> None:
    work_dir = Path(str(row["work_dir"]))
    work_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "status": "completed",
        "success": True,
        "full_run": True,
        "failure_kind": None,
        "actual_seed": row["actual_seed"],
        "method": row["method"],
        "gpu_uuid": row["gpu_uuid"],
    }
    payload.update(overrides or {})
    (work_dir / "run_result.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


class StatisticsTests(unittest.TestCase):
    def paired_rows(self) -> list[dict[str, object]]:
        return [
            experiment("no_cga", 1, 0.50),
            experiment("candidate", 1, 0.60),
            experiment("no_cga", 2, 0.50),
            experiment("candidate", 2, 0.70),
            experiment("no_cga", 3, 0.50),
            experiment("candidate", 3, 0.65),
        ]

    def test_paired_statistics_have_exact_direction_df_and_p_values(self) -> None:
        paired = align_pairs(
            self.paired_rows(),
            "candidate",
            "no_cga",
            require_success_sentinel=False,
        )
        summary = summarize_comparison(
            paired,
            bootstrap_resamples=5_000,
            bootstrap_seed=123,
        )

        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["N"], 3)
        self.assertEqual([pair["seed"] for pair in summary["pairs"]], [1, 2, 3])
        self.assertAlmostEqual(summary["mean_difference"], 0.15)
        self.assertAlmostEqual(summary["sample_std_difference"], 0.05)
        self.assertAlmostEqual(summary["standard_error"], 0.05 / math.sqrt(3))
        self.assertEqual(summary["paired_t"]["df"], 2)

        expected = stats.ttest_1samp([0.1, 0.2, 0.15], popmean=0.0)
        self.assertAlmostEqual(summary["paired_t"]["t"], expected.statistic)
        self.assertAlmostEqual(summary["paired_t"]["p"], expected.pvalue)
        self.assertEqual(summary["permutation_test"]["mode"], "exact")
        self.assertAlmostEqual(summary["permutation_test"]["p"], 0.25)
        self.assertAlmostEqual(summary["cohens_dz"], 3.0)
        self.assertEqual(summary["paired_seeds"], [1, 2, 3])
        self.assertAlmostEqual(summary["baseline_descriptive"]["mean"], 0.5)
        self.assertAlmostEqual(
            summary["baseline_descriptive"]["sample_sd"], 0.0
        )
        self.assertAlmostEqual(summary["candidate_descriptive"]["mean"], 0.65)
        self.assertAlmostEqual(
            summary["candidate_descriptive"]["sample_sd"], 0.05
        )
        self.assertLessEqual(
            summary["bootstrap_ci_95"][0], summary["mean_difference"]
        )
        self.assertGreaterEqual(
            summary["bootstrap_ci_95"][1], summary["mean_difference"]
        )

    def test_bootstrap_and_monte_carlo_permutation_are_deterministic(self) -> None:
        differences = [index / 100.0 for index in range(1, 22)]
        self.assertEqual(
            bootstrap_mean_ci(differences, resamples=1_000, seed=9),
            bootstrap_mean_ci(differences, resamples=1_000, seed=9),
        )
        first = sign_flip_test(
            differences, seed=9, monte_carlo_samples=2_000
        )
        second = sign_flip_test(
            differences, seed=9, monte_carlo_samples=2_000
        )
        self.assertEqual(first, second)
        self.assertEqual(first["mode"], "monte_carlo")

    def test_exact_sign_flip_resolution_for_three_five_and_six_pairs(self) -> None:
        for n, expected_p in ((3, 0.25), (5, 0.0625), (6, 0.03125)):
            with self.subTest(n=n):
                result = sign_flip_test([0.1] * n)
                self.assertEqual(result["mode"], "exact")
                self.assertEqual(result["assignments"], 2**n)
                self.assertEqual(result["extreme_assignments"], 2)
                self.assertAlmostEqual(result["p"], expected_p)

    def test_insufficient_n_and_zero_variance_are_reported_as_na(self) -> None:
        one_pair = align_pairs(
            [experiment("base", 1, 0.5), experiment("candidate", 1, 0.6)],
            "candidate",
            "base",
            require_success_sentinel=False,
        )
        insufficient = summarize_comparison(one_pair)
        self.assertEqual(insufficient["status"], "insufficient_n")
        self.assertIsNone(insufficient["paired_t"])
        self.assertIsNone(insufficient["bootstrap_ci_95"])
        self.assertIsNone(insufficient["cohens_dz"])

        rows = [
            experiment("base", 1, 0.5),
            experiment("candidate", 1, 0.6),
            experiment("base", 2, 0.4),
            experiment("candidate", 2, 0.5),
        ]
        zero_variance = summarize_comparison(
            align_pairs(
                rows,
                "candidate",
                "base",
                require_success_sentinel=False,
            ),
            bootstrap_resamples=1_000,
        )
        self.assertEqual(zero_variance["status"], "zero_variance")
        self.assertEqual(zero_variance["sample_std_difference"], 0.0)
        self.assertAlmostEqual(zero_variance["bootstrap_ci_95"][0], 0.1)
        self.assertAlmostEqual(zero_variance["bootstrap_ci_95"][1], 0.1)
        self.assertIsNone(zero_variance["paired_t"])
        self.assertIn("zero sample variance", zero_variance["paired_t_na_reason"])
        self.assertEqual(zero_variance["permutation_test"]["mode"], "exact")
        self.assertAlmostEqual(zero_variance["permutation_test"]["p"], 0.5)
        self.assertIsNone(zero_variance["cohens_dz"])
        self.assertIn("zero sample variance", zero_variance["cohens_dz_na_reason"])

        zero_report = analyze_comparisons(
            rows,
            baseline_method="base",
            candidate_methods=["candidate"],
            require_success_sentinel=False,
            bootstrap_resamples=1_000,
        )
        zero_comparison = zero_report["comparisons"][0]
        self.assertIsNone(zero_comparison["paired_t_p_holm"])
        self.assertAlmostEqual(zero_comparison["permutation_p_holm"], 0.5)
        zero_markdown = render_markdown(zero_report)
        self.assertIn("Paired t: NA —", zero_markdown)
        self.assertIn("Cohen's dz: NA —", zero_markdown)
        self.assertIn("method (paired seeds only)", zero_markdown)

    def test_duplicate_seed_and_gpu_mismatch_are_excluded_not_replicated(self) -> None:
        candidate_seed_one = experiment("candidate", 1, 0.6)
        rows = [
            experiment("base", 1, 0.5),
            candidate_seed_one,
            dict(candidate_seed_one),  # exact duplicate collapses
            experiment("base", 2, 0.5),
            experiment("candidate", 2, 0.6, work_suffix="_a"),
            experiment("candidate", 2, 0.61, work_suffix="_b"),
            experiment("base", 3, 0.5, gpu_uuid="GPU-a"),
            experiment("candidate", 3, 0.6, gpu_uuid="GPU-b"),
        ]
        paired = align_pairs(
            rows,
            "candidate",
            "base",
            require_success_sentinel=False,
        )
        self.assertEqual(len(paired.pairs), 1)
        self.assertEqual(paired.pairs[0].seed, 1)
        self.assertEqual(paired.duplicate_candidate_seeds, (2,))
        self.assertEqual(paired.gpu_mismatch_seeds, (3,))
        self.assertNotIn(2, paired.missing_candidate_seeds)

    def test_failed_or_incomplete_rows_are_not_eligible(self) -> None:
        failed = experiment("candidate", 1, 0.9)
        failed["status"] = "failed"
        incomplete = experiment("candidate", 2, 0.9)
        incomplete["complete"] = False
        rows = [
            experiment("base", 1, 0.5),
            failed,
            experiment("base", 2, 0.5),
            incomplete,
        ]
        paired = align_pairs(
            rows,
            "candidate",
            "base",
            require_success_sentinel=False,
        )
        self.assertEqual(len(paired.pairs), 0)
        self.assertEqual(paired.missing_candidate_seeds, (1, 2))

    def test_missing_seed_and_gpu_mismatch_are_reported_separately(self) -> None:
        rows = [
            experiment("base", 1, 0.5),
            experiment("candidate", 1, 0.6),
            experiment("base", 2, 0.5),
            experiment("base", 3, 0.5, gpu_uuid="GPU-a"),
            experiment("candidate", 3, 0.6, gpu_uuid="GPU-b"),
        ]
        paired = align_pairs(
            rows,
            "candidate",
            "base",
            require_success_sentinel=False,
        )
        self.assertEqual([pair.seed for pair in paired.pairs], [1])
        self.assertEqual(paired.missing_candidate_seeds, (2,))
        self.assertEqual(paired.gpu_mismatch_seeds, (3,))

    def test_default_admission_requires_matching_success_sentinel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            admitted_row = experiment("base", 1, 0.5)
            admitted_row["work_dir"] = str(root / "admitted")
            write_success_sentinel(admitted_row)

            missing_sentinel = experiment("candidate", 1, 0.6)
            missing_sentinel["work_dir"] = str(root / "missing")
            mismatched_gpu = experiment("candidate", 2, 0.7, gpu_uuid="GPU-a")
            mismatched_gpu["work_dir"] = str(root / "gpu_mismatch")
            write_success_sentinel(
                mismatched_gpu,
                overrides={"gpu_uuid": "GPU-other"},
            )

            admitted, audit = audit_experiment_rows(
                [admitted_row, missing_sentinel, mismatched_gpu]
            )
            self.assertEqual(admitted, [admitted_row])
            self.assertTrue(audit["success_sentinel_required"])
            self.assertEqual(audit["admitted_rows"], 1)
            self.assertEqual(audit["rejected_rows"], 2)
            self.assertEqual(
                audit["rejection_reason_counts"]["missing_success_sentinel"], 1
            )
            self.assertEqual(
                audit["rejection_reason_counts"]["sentinel_gpu_uuid_mismatch"],
                1,
            )

            legacy_rows, legacy_audit = audit_experiment_rows(
                [missing_sentinel], require_success_sentinel=False
            )
            self.assertEqual(legacy_rows, [missing_sentinel])
            self.assertFalse(legacy_audit["success_sentinel_required"])

    def test_holm_adjustment_and_multi_candidate_report(self) -> None:
        adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
        self.assertAlmostEqual(adjusted["a"], 0.03)
        self.assertAlmostEqual(adjusted["b"], 0.06)
        self.assertAlmostEqual(adjusted["c"], 0.06)

        rows = self.paired_rows()
        rows.extend(
            [
                experiment("candidate_2", 1, 0.55),
                experiment("candidate_2", 2, 0.58),
                experiment("candidate_2", 3, 0.52),
            ]
        )
        report = analyze_comparisons(
            rows,
            baseline_method="no_cga",
            candidate_methods=["candidate", "candidate_2", "candidate"],
            require_success_sentinel=False,
            bootstrap_resamples=2_000,
        )
        self.assertEqual(report["candidate_methods"], ["candidate", "candidate_2"])
        self.assertEqual(len(report["comparisons"]), 2)
        for comparison in report["comparisons"]:
            self.assertIsNotNone(comparison["paired_t_p_holm"])
            self.assertIsNotNone(comparison["permutation_p_holm"])
        self.assertIn("this analyze_comparisons call", report["settings"]["holm_family"])

    def test_cli_writes_markdown_and_optional_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiments_path = root / "experiments.jsonl"
            rows = self.paired_rows()
            for index, row in enumerate(rows):
                row["work_dir"] = str(root / f"run_{index}")
                write_success_sentinel(row)
            experiments_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            markdown_path = root / "statistical_report.md"
            json_path = root / "statistical_report.json"
            with contextlib.redirect_stdout(io.StringIO()):
                result = statistics_main(
                    [
                        str(experiments_path),
                        "--baseline",
                        "no_cga",
                        "--candidate",
                        "candidate",
                        "--output",
                        str(markdown_path),
                        "--json-output",
                        str(json_path),
                        "--bootstrap-resamples",
                        "1000",
                    ]
                )
            self.assertEqual(result, 0)
            self.assertIn("Cohen's dz", markdown_path.read_text(encoding="utf-8"))
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["admission_audit"]["rejected_rows"], 0)
            self.assertEqual(payload["comparisons"][0]["N"], 3)
            self.assertEqual(payload["comparisons"][0]["paired_t"]["df"], 2)

    def test_cli_executes_via_real_script_path_without_name_shadowing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = self.paired_rows()
            for index, row in enumerate(rows):
                row["work_dir"] = str(root / f"run_{index}")
                write_success_sentinel(row)
            experiments_path = root / "experiments.jsonl"
            experiments_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            markdown_path = root / "subprocess_report.md"
            json_path = root / "subprocess_report.json"
            script_path = Path(__file__).resolve().parents[1] / "statistics.py"

            completed = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    str(experiments_path),
                    "--baseline",
                    "no_cga",
                    "--candidate",
                    "candidate",
                    "--output",
                    str(markdown_path),
                    "--json-output",
                    str(json_path),
                    "--bootstrap-resamples",
                    "1000",
                ],
                cwd=root,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=f"stdout={completed.stdout}\nstderr={completed.stderr}",
            )
            self.assertTrue(markdown_path.is_file())
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["comparisons"][0]["N"], 3)
            self.assertAlmostEqual(
                payload["comparisons"][0]["candidate_descriptive"]["mean"],
                0.65,
            )


if __name__ == "__main__":
    unittest.main()
