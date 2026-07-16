from __future__ import annotations

import csv
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from tools.cga_research.parse_results import (
    CLASSES,
    EXPERIMENT_FIELDS,
    IncompleteRunError,
    ResultParseError,
    discover_log_pairs,
    discover_run_dirs,
    main as parse_main,
    parse_cga_windows,
    parse_work_dir,
    upsert_experiments,
)


def class_table(values: list[float]) -> str:
    return "\n".join(
        f"| {class_name} | 1 | 1 | 1 | {value:.4f} |"
        for class_name, value in zip(CLASSES, values)
    )


class ParseResultsTests(unittest.TestCase):
    def make_completed_run(self, root: Path) -> tuple[Path, dict[str, str]]:
        work_dir = (
            root
            / "runs"
            / "chaff"
            / "legacy"
            / "seed_41"
            / "attempt_1_gpu_abc"
        )
        log_dir = work_dir / "nested_logs"
        log_dir.mkdir(parents=True)

        old_log = log_dir / "20260714_100000.log"
        old_log.write_text(
            "2026-07-14 10:00:00,000 - Set random seed to 41, deterministic: True\n"
            + class_table([0.1] * 6)
            + "\n2026-07-14 10:01:00,000 - Epoch(val) [1][10] mAP: 0.1000\n",
            encoding="utf-8",
        )
        (Path(f"{old_log}.json")).write_text(
            json.dumps({"mode": "val", "epoch": 1, "mAP": 0.10002}) + "\n",
            encoding="utf-8",
        )

        new_log = log_dir / "20260714_110000.log"
        diagnostic_one = (
            "[CGA] diag_window calls=2, total=10, agree=6, dropped=0, "
            "blended=4, boosted=0, multiplied=0, penalized=0, "
            "threshold_dropped=0, shuffled=0, moved=0, unmoved=0, "
            "real_agree=6, operative_agree=6, mean_label_prob=0.2000, "
            "label_prob_pct=min=0.0,p25=0.1,p50=0.2,p75=0.3,max=0.4, "
            "argmax=ship:6,car:4, "
            "detector=ship:n=6,agree=4,drop=0;car:n=4,agree=2,drop=0"
        )
        diagnostic_two = (
            "[CGA] diag_window calls=3, total=30, agree=20, dropped=0, "
            "blended=10, boosted=0, multiplied=0, penalized=0, "
            "threshold_dropped=0, shuffled=0, moved=0, unmoved=0, "
            "real_agree=20, operative_agree=20, mean_label_prob=0.8000, "
            "label_prob_pct=min=0.1,p25=0.2,p50=0.3,p75=0.4,max=0.9, "
            "argmax=ship:10,bridge:20, "
            "detector=ship:n=12,agree=9,drop=0;bridge:n=18,agree=11,drop=0"
        )
        new_log.write_text(
            "2026-07-14 11:00:00,000 - Set random seed to 41, deterministic: True\n"
            + class_table([0.2] * 6)
            + "\n2026-07-14 11:01:00,000 - Epoch(val) [1][20] mAP: 0.6000\n"
            + diagnostic_one
            + "\n"
            + class_table([0.61, 0.62, 0.63, 0.64, 0.65, 0.66])
            + "\n"
            + diagnostic_two
            + "\n2026-07-14 11:02:00,000 - Epoch(val) [1][4234] mAP: 0.7000\n",
            encoding="utf-8",
        )
        Path(f"{new_log}.json").write_text(
            "\n".join(
                [
                    json.dumps({"mode": "train", "epoch": 1, "loss": 1.0}),
                    json.dumps(
                        {"mode": "val", "epoch": 1, "iter": 20, "mAP": 0.60003}
                    ),
                    json.dumps(
                        {
                            "mode": "val",
                            "epoch": 1,
                            "iter": 4234,
                            "mAP": 0.70004,
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        # Selection follows mtime, not a hard-coded filename ordering rule.
        os.utime(old_log, ns=(1_000_000_000, 1_000_000_000))
        os.utime(new_log, ns=(2_000_000_000, 2_000_000_000))

        pseudo_lines = [
            (
                f"Epoch [1][{iteration}/7] pseudo_num: {float(iteration):.1f}, "
                f"pseudo_num(acc): {iteration / 10.0:.1f}"
            )
            for iteration in range(1, 8)
        ]
        (work_dir / "run_train.log").write_text(
            "\n".join(line for line in pseudo_lines for _ in range(2)),
            encoding="utf-8",
        )
        (work_dir / "run_result.json").write_text(
            json.dumps(
                {
                    "method": "legacy",
                    "seed": 41,
                    "actual_seed": 41,
                    "status": "completed",
                    "success": True,
                    "exit_code": 0,
                    "final_val_epoch": 1,
                    "final_val_iteration": 4234,
                    "final_val_log": str(new_log.resolve()),
                    "progress": {"epoch": 1, "iteration": 4230, "total": 4234},
                    "gpu_uuid": "GPU-abc",
                    "gpu_index": 2,
                    "started_at": 100.0,
                    "ended_at": 160.0,
                    "gpu_seconds": 60.0,
                    "environment": {"CGA_FILTER_MODE": "legacy"},
                }
            ),
            encoding="utf-8",
        )
        (work_dir / "iter_4235.pth").touch()
        (work_dir / "iter_4235_ema.pth").touch()
        (work_dir / "launch_environment.json").write_text(
            json.dumps({"CGA_SCORER": "sarclip"}), encoding="utf-8"
        )
        (work_dir / "process_environment.txt").write_text(
            "CUDA_VISIBLE_DEVICES=2\nSARCLIP_LORA=/tmp/model.pth\n",
            encoding="utf-8",
        )
        registry = {
            "requested_seed": "41",
            "method": "legacy",
            "gpu_uuid": "GPU-abc",
            "physical_gpu_index": "2",
            "gpu_name": "Synthetic GPU",
        }
        return work_dir, registry

    def test_parse_completed_run_uses_last_ema_and_high_precision_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work_dir, registry = self.make_completed_run(root)

            pairs = discover_log_pairs(work_dir)
            self.assertEqual(len(pairs), 2)
            self.assertTrue(pairs[-1].text_path.name.startswith("20260714_110000"))
            row = parse_work_dir(work_dir, registry)

            self.assertEqual(row["status"], "completed")
            self.assertTrue(row["complete"])
            self.assertAlmostEqual(row["final_map"], 0.70004)
            self.assertEqual(row["text_final_map"], 0.7)
            self.assertEqual(row["final_eval_source"], "jsonl")
            self.assertEqual(row["final_iteration"], 4234)
            self.assertEqual(row["actual_seed"], 41)
            self.assertTrue(row["seed_verified"])
            self.assertEqual(row["gpu_uuid"], "GPU-abc")
            self.assertEqual(row["method"], "legacy")
            self.assertEqual(row["training_seconds"], 60.0)

            for offset, class_name in enumerate(CLASSES):
                self.assertAlmostEqual(row[f"ap_{class_name}"], 0.61 + offset * 0.01)

            # Seven records map to zero-based indices 1, 3, and 5.
            self.assertEqual(row["pseudo_record_count"], 7)
            self.assertEqual(row["pseudo_early"], 2.0)
            self.assertEqual(row["pseudo_middle"], 4.0)
            self.assertEqual(row["pseudo_late"], 6.0)
            self.assertEqual(row["pseudo_acc_late"], 0.6)

            self.assertEqual(row["cga_window_count"], 2)
            self.assertEqual(row["cga_calls"], 5)
            self.assertEqual(row["cga_last_logged_call"], 5)
            self.assertEqual(row["cga_coverage_scope"], "logged_prefix")
            self.assertEqual(row["cga_total"], 40)
            self.assertEqual(row["cga_agree"], 26)
            self.assertEqual(row["cga_dropped"], 0)
            self.assertEqual(row["cga_blended"], 14)
            self.assertEqual(row["cga_moved"], 0)
            self.assertEqual(row["cga_real_agree"], 26)
            self.assertEqual(row["cga_operative_agree"], 26)
            self.assertAlmostEqual(row["cga_mean_label_prob"], 0.65)
            self.assertEqual(
                json.loads(row["cga_argmax_json"]),
                {"bridge": 20, "car": 4, "ship": 16},
            )
            per_class = json.loads(row["cga_per_class_json"])
            self.assertEqual(per_class["ship"]["total"], 18)
            self.assertEqual(per_class["ship"]["disagreement_candidates"], 5)
            self.assertEqual(per_class["ship"]["hard_drop_count"], 0)
            self.assertEqual(per_class["ship"]["actual_operation_count"], 5)
            self.assertEqual(per_class["ship"]["operation_kind"], "blend")
            self.assertFalse(per_class["ship"]["excluded_from_cga"])
            self.assertTrue(
                per_class["ship"]["actual_operation_recoverable"]
            )
            environment = json.loads(row["environment_json"])
            self.assertEqual(environment["CGA_FILTER_MODE"], "legacy")
            self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "2")

            # A nested timestamp directory still belongs to the run root.
            self.assertEqual(discover_run_dirs(root), [work_dir.resolve()])

    def test_seed_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            work_dir, registry = self.make_completed_run(Path(directory))
            registry["requested_seed"] = "42"
            with self.assertRaisesRegex(ResultParseError, "seed mismatch"):
                parse_work_dir(work_dir, registry)

    def test_completion_requires_final_ema_and_complete_class_table(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing_ema = root / "missing_ema"
            missing_ema.mkdir()
            (missing_ema / "20260714_120000.log").write_text(
                "Set random seed to 1, deterministic: True\n" + class_table([0.1] * 6),
                encoding="utf-8",
            )
            with self.assertRaises(IncompleteRunError):
                parse_work_dir(missing_ema, {"requested_seed": 1})

            # An earlier timestamp log has a complete table, but the selected
            # final timestamp log does not.  The parser must not borrow it.
            missing_classes, registry = self.make_completed_run(root)
            selected_log = missing_classes / "nested_logs" / "20260714_110000.log"
            selected_log.write_text(
                "2026-07-14 11:00:00,000 - Set random seed to 41, "
                "deterministic: True\n"
                "2026-07-14 11:02:00,000 - Epoch(val) [1][4234] "
                "mAP: 0.7000\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ResultParseError, "six-class"):
                parse_work_dir(missing_classes, registry)

    def test_completion_sentinel_and_both_checkpoints_are_mandatory(self) -> None:
        mutations = {
            "status": lambda payload: payload.update(status="running"),
            "success": lambda payload: payload.update(success=False),
            "exit_code": lambda payload: payload.update(exit_code=1),
            "actual_seed": lambda payload: payload.update(actual_seed=99),
            "final_val_iteration": lambda payload: payload.update(
                final_val_iteration=4233
            ),
            "progress_total": lambda payload: payload["progress"].update(total=4233),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                work_dir, registry = self.make_completed_run(Path(directory))
                result_path = work_dir / "run_result.json"
                payload = json.loads(result_path.read_text(encoding="utf-8"))
                mutate(payload)
                result_path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(ResultParseError):
                    parse_work_dir(work_dir, registry)

        for checkpoint in ("iter_4235.pth", "iter_4235_ema.pth"):
            with self.subTest(checkpoint=checkpoint), tempfile.TemporaryDirectory() as directory:
                work_dir, registry = self.make_completed_run(Path(directory))
                (work_dir / checkpoint).unlink()
                with self.assertRaises(IncompleteRunError):
                    parse_work_dir(work_dir, registry)

        with tempfile.TemporaryDirectory() as directory:
            work_dir, registry = self.make_completed_run(Path(directory))
            (work_dir / "run_result.json").unlink()
            with self.assertRaises(IncompleteRunError):
                parse_work_dir(work_dir, registry)

        with tempfile.TemporaryDirectory() as directory:
            work_dir, registry = self.make_completed_run(Path(directory))
            log_path = work_dir / "nested_logs" / "20260714_110000.log"
            log_path.write_text(
                log_path.read_text(encoding="utf-8").replace(
                    "Epoch(val) [1][4234]", "Epoch(val) [1][4233]"
                ),
                encoding="utf-8",
            )
            json_path = Path(f"{log_path}.json")
            json_path.write_text(
                json_path.read_text(encoding="utf-8").replace(
                    '"iter": 4234', '"iter": 4233'
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ResultParseError, "selected final validation"
            ):
                parse_work_dir(work_dir, registry)

    def test_cga_per_class_semantics_exclusions_and_aggregate_invariants(self) -> None:
        adaptive_line = (
            "[CGA] diag_window calls=5, total=10, agree=6, dropped=0, "
            "blended=2, boosted=0, multiplied=0, penalized=0, "
            "threshold_dropped=0, shuffled=0, moved=0, unmoved=0, "
            "real_agree=6, operative_agree=6, mean_label_prob=0.5, "
            "label_prob_pct=min=0,p25=0.2,p50=0.5,p75=0.8,max=1, "
            "argmax=ship:5,car:5, "
            "detector=ship:n=5,agree=3,drop=0;car:n=5,agree=3,drop=0"
        )
        parsed = parse_cga_windows(
            adaptive_line,
            {
                "CGA_FILTER_MODE": "adaptive_blend",
                "CGA_EXCLUDE_IDS": "2",
            },
        )
        ship = parsed["per_class"]["ship"]
        car = parsed["per_class"]["car"]
        self.assertEqual(ship["disagreement_candidates"], 2)
        self.assertEqual(ship["actual_operation_count"], 2)
        self.assertFalse(ship["excluded_from_cga"])
        self.assertEqual(car["disagreement_candidates"], 2)
        self.assertEqual(car["actual_operation_count"], 0)
        self.assertEqual(car["operation_kind"], "excluded")
        self.assertTrue(car["excluded_from_cga"])

        invalid = adaptive_line.replace("blended=2", "blended=3")
        with self.assertRaisesRegex(ResultParseError, "operation invariant"):
            parse_cga_windows(
                invalid,
                {
                    "CGA_FILTER_MODE": "adaptive_blend",
                    "CGA_EXCLUDE_IDS": "2",
                },
            )

        threshold_line = (
            "[CGA] diag_window calls=5, total=10, agree=6, dropped=2, "
            "blended=0, boosted=0, multiplied=0, penalized=0, "
            "threshold_dropped=2, shuffled=0, moved=0, unmoved=0, "
            "real_agree=6, operative_agree=6, mean_label_prob=0.5, "
            "label_prob_pct=min=0,p25=0.2,p50=0.5,p75=0.8,max=1, "
            "argmax=ship:5,car:5, "
            "detector=ship:n=5,agree=3,drop=2;car:n=5,agree=3,drop=0"
        )
        threshold = parse_cga_windows(
            threshold_line,
            {"CGA_FILTER_MODE": "disagreement_threshold"},
        )["per_class"]
        self.assertEqual(threshold["ship"]["disagreement_candidates"], 2)
        self.assertEqual(threshold["ship"]["hard_drop_count"], 2)
        self.assertEqual(threshold["ship"]["actual_operation_count"], 2)
        self.assertEqual(threshold["car"]["actual_operation_count"], 0)
        self.assertEqual(
            threshold["ship"]["operation_kind"],
            "hard_drop_below_disagreement_threshold",
        )

    def test_shuffled_global_diagnostics_are_extracted_and_validated(self) -> None:
        line = (
            "[CGA] diag_window calls=5, total=12, agree=7, dropped=0, "
            "blended=5, boosted=0, multiplied=0, penalized=0, "
            "threshold_dropped=0, shuffled=5, moved=10, unmoved=2, "
            "real_agree=7, operative_agree=7, mean_label_prob=0.5, "
            "label_prob_pct=min=0,p25=0.2,p50=0.5,p75=0.8,max=1, "
            "argmax=ship:8,car:4, "
            "detector=ship:n=8,agree=5,drop=0;car:n=4,agree=2,drop=0"
        )
        text = "[CGA] filter mode=shuffled_legacy, calls=5, total=12\n" + line
        parsed = parse_cga_windows(
            text, {"CGA_FILTER_MODE": "shuffled_legacy"}
        )
        self.assertEqual(parsed["moved"], 10)
        self.assertEqual(parsed["unmoved"], 2)
        self.assertEqual(parsed["real_agree"], 7)
        self.assertEqual(parsed["operative_agree"], 7)
        self.assertEqual(parsed["last_logged_call"], 5)
        self.assertEqual(parsed["coverage_scope"], "logged_prefix")

        invalid = line.replace("operative_agree=7", "operative_agree=6")
        with self.assertRaisesRegex(ResultParseError, "operative agreement"):
            parse_cga_windows(
                invalid, {"CGA_FILTER_MODE": "shuffled_legacy"}
            )

        with self.assertRaisesRegex(ResultParseError, "logged-prefix call"):
            parse_cga_windows(
                "[CGA] filter mode=shuffled_legacy, calls=6, total=12\n"
                + line,
                {"CGA_FILTER_MODE": "shuffled_legacy"},
            )

    def test_atomic_upsert_is_idempotent_and_has_stable_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work_dir, registry = self.make_completed_run(root)
            row = parse_work_dir(work_dir, registry)
            output = root / "results"
            self.assertEqual(upsert_experiments(output, [row]), 1)
            self.assertEqual(upsert_experiments(output, [row]), 1)

            with (output / "experiments.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                reader = csv.DictReader(handle)
                csv_rows = list(reader)
                self.assertEqual(reader.fieldnames, EXPERIMENT_FIELDS)
            self.assertEqual(len(csv_rows), 1)
            jsonl_rows = [
                json.loads(line)
                for line in (output / "experiments.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line
            ]
            self.assertEqual(len(jsonl_rows), 1)
            self.assertEqual(jsonl_rows[0]["work_dir"], str(work_dir.resolve()))

    def test_cli_supports_single_run_and_recursive_root_scan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work_dir, _ = self.make_completed_run(root)
            smoke_dir = root / "smoke_training" / "legacy_seed41"
            smoke_dir.mkdir(parents=True)
            (smoke_dir / "run_result.json").write_text("{}", encoding="utf-8")
            (smoke_dir / "20260714_090000.log").write_text(
                "smoke must not be discovered\n", encoding="utf-8"
            )
            self.assertEqual(discover_run_dirs(root), [work_dir.resolve()])
            single_output = root / "single_results"
            with contextlib.redirect_stdout(io.StringIO()):
                result = parse_main(
                    [
                        "--work-dir",
                        str(work_dir),
                        "--output-root",
                        str(single_output),
                    ]
                )
            self.assertEqual(result, 0)
            self.assertTrue((single_output / "experiments.csv").is_file())

            scan_output = root / "scan_results"
            upsert_experiments(
                scan_output,
                [{"work_dir": str(smoke_dir), "method": "smoke"}],
            )
            with contextlib.redirect_stdout(io.StringIO()):
                result = parse_main(
                    [
                        "--research-root",
                        str(root),
                        "--output-root",
                        str(scan_output),
                        "--strict",
                    ]
                )
            self.assertEqual(result, 0)
            rows = [
                json.loads(line)
                for line in (scan_output / "experiments.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line
            ]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["work_dir"], str(work_dir.resolve()))


if __name__ == "__main__":
    unittest.main()
