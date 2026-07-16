from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import torch

from tools.cga_research.final_test_manifest import (
    FROZEN_STATE,
    ManifestError,
    canonical_json_bytes,
    create_manifest,
    load_json,
    main,
    sha256_file,
    sidecar_path,
    validate_manifest,
    verify_manifest,
)


class FinalTestManifestTests(unittest.TestCase):
    METHODS = ("no_cga", "ship_adaptive")
    SEEDS = (41, 42)

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()

        self.evidence = self._write("reports/statistics.json", b'{"selected":true}\n')
        self.config = self._write("configs/final.py", b"model = dict()\n")
        self.runtime = self._write("state/runtime.json", b'{"python":"iraod"}\n')
        self.data_manifest = self._write(
            "state/data_manifest.json", b'{"test":"sealed"}\n'
        )
        self.payload = {
            "schema_version": 2,
            "state": FROZEN_STATE,
            "project_root": str(self.root),
            "arms": [self._arm(method) for method in self.METHODS],
            "aggregation": {
                "required_seed_set": list(self.SEEDS),
                "paired_comparator": "no_cga",
                "exclude_failed_seed": False,
                "mean": True,
                "sample_std_ddof": 1,
                "bootstrap_samples": 10000,
                "bootstrap_seed": 20260715,
                "paired_t_test": True,
                "exact_sign_flip_permutation": True,
                "cohens_dz": True,
                "holm_correction": True,
            },
            "selection": {
                "evidence": [self._artifact(self.evidence)],
                "config": self._artifact(self.config),
                "runtime_files": [self._artifact(self.runtime)],
                "data_manifest": self._artifact(self.data_manifest),
            },
        }

    def _write(self, relative_path: str, content: bytes) -> Path:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def _artifact(self, path: Path) -> dict[str, object]:
        return {
            "path": str(path.relative_to(self.root)),
            "sha256": sha256_file(path),
        }

    def _run(self, method: str, seed: int) -> dict[str, object]:
        checkpoint = self.root / "runs" / method / f"seed_{seed}" / "iter_4235_ema.pth"
        checkpoint.parent.mkdir(parents=True)
        torch.save(
            {
                "meta": {"iter": 4235, "epoch": 1, "seed": seed},
                "state_dict": {},
            },
            checkpoint,
        )
        return {
            "seed": seed,
            "training_fingerprint": hashlib.sha256(
                f"{method}:{seed}".encode("utf-8")
            ).hexdigest(),
            "checkpoint": {
                "path": str(checkpoint.relative_to(self.root)),
                "size_bytes": checkpoint.stat().st_size,
                "sha256": sha256_file(checkpoint),
                "meta": {"iter": 4235, "epoch": 1, "seed": seed},
            },
            "output_dir": f"final_test_outputs/{method}/seed_{seed}",
        }

    def _arm(self, method: str) -> dict[str, object]:
        return {
            "method": method,
            "method_environment": (
                {"CGA_SCORER": "none"}
                if method == "no_cga"
                else {
                    "CGA_SCORER": "sarclip",
                    "CGA_FILTER_MODE": "adaptive_blend",
                }
            ),
            "hyperparameters": {
                "corrupt": "chaff",
                "model.cfg.score_thr": 0.9,
            },
            "runs": [self._run(method, seed) for seed in self.SEEDS],
        }

    def _write_draft(self, payload=None) -> Path:
        draft = self.root / "draft.json"
        draft.write_text(
            json.dumps(self.payload if payload is None else payload, indent=3),
            encoding="utf-8",
        )
        return draft

    def test_create_freezes_whole_bundle_and_detached_hash(self) -> None:
        output = self.root / "frozen" / "final_test_manifest.json"
        digest = create_manifest(self._write_draft(), output)

        content = output.read_bytes()
        self.assertEqual(content, canonical_json_bytes(self.payload))
        self.assertEqual(digest, hashlib.sha256(content).hexdigest())
        self.assertEqual(
            sidecar_path(output).read_text(encoding="utf-8"),
            f"{digest}  {output.name}\n",
        )
        self.assertEqual(verify_manifest(output), digest)

    def test_create_is_write_once_for_both_bundle_paths(self) -> None:
        draft = self._write_draft()
        output = self.root / "frozen" / "final_test_manifest.json"
        create_manifest(draft, output)
        original_manifest = output.read_bytes()
        original_sidecar = sidecar_path(output).read_bytes()

        with self.assertRaisesRegex(ManifestError, "write-once"):
            create_manifest(draft, output)
        self.assertEqual(output.read_bytes(), original_manifest)
        self.assertEqual(sidecar_path(output).read_bytes(), original_sidecar)

        sidecar_claimed = self.root / "sidecar_claimed.json"
        sidecar_path(sidecar_claimed).write_bytes(b"competing sidecar")
        with self.assertRaisesRegex(ManifestError, "sidecar.*write-once"):
            create_manifest(draft, sidecar_claimed)
        self.assertFalse(sidecar_claimed.exists())
        self.assertEqual(
            sidecar_path(sidecar_claimed).read_bytes(), b"competing sidecar"
        )

        manifest_claimed = self.root / "manifest_claimed.json"
        manifest_claimed.write_bytes(b"competing manifest")
        with self.assertRaisesRegex(ManifestError, "manifest.*write-once"):
            create_manifest(draft, manifest_claimed)
        self.assertEqual(manifest_claimed.read_bytes(), b"competing manifest")
        self.assertFalse(sidecar_path(manifest_claimed).exists())

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks are unavailable")
    def test_create_rejects_manifest_or_sidecar_symlink(self) -> None:
        draft = self._write_draft()
        target = self._write("outside/target", b"outside")
        manifest_link = self.root / "manifest_link.json"
        manifest_link.symlink_to(target)
        with self.assertRaisesRegex(ManifestError, "symlink"):
            create_manifest(draft, manifest_link)
        self.assertEqual(target.read_bytes(), b"outside")

        sidecar_linked = self.root / "sidecar_linked.json"
        sidecar_path(sidecar_linked).symlink_to(target)
        with self.assertRaisesRegex(ManifestError, "sidecar.*symlink"):
            create_manifest(draft, sidecar_linked)
        self.assertFalse(sidecar_linked.exists())
        self.assertEqual(target.read_bytes(), b"outside")

    def test_concurrent_creators_have_exactly_one_winner(self) -> None:
        draft = self._write_draft()
        output = self.root / "frozen" / "concurrent.json"

        def create_once():
            try:
                return ("created", create_manifest(draft, output))
            except ManifestError as error:
                return ("rejected", str(error))

        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = list(executor.map(lambda _: create_once(), range(2)))

        self.assertEqual([status for status, _ in outcomes].count("created"), 1)
        self.assertEqual([status for status, _ in outcomes].count("rejected"), 1)
        self.assertEqual(verify_manifest(output), sha256_file(output))

    def test_half_publish_failure_rolls_back_owned_sidecar(self) -> None:
        draft = self._write_draft()
        output = self.root / "frozen" / "half_publish.json"
        real_link = os.link
        calls = 0

        def fail_manifest_link(source, destination):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("synthetic second-link failure")
            return real_link(source, destination)

        with patch(
            "tools.cga_research.final_test_manifest.os.link",
            side_effect=fail_manifest_link,
        ), self.assertRaisesRegex(ManifestError, "second-link failure"):
            create_manifest(draft, output)
        self.assertFalse(output.exists())
        self.assertFalse(sidecar_path(output).exists())

    def test_post_publish_verification_failure_rolls_back_bundle(self) -> None:
        draft = self._write_draft()
        output = self.root / "frozen" / "verify_failure.json"
        with patch(
            "tools.cga_research.final_test_manifest.verify_manifest",
            side_effect=ManifestError("synthetic verification failure"),
        ), self.assertRaisesRegex(ManifestError, "verification failure"):
            create_manifest(draft, output)
        self.assertFalse(output.exists())
        self.assertFalse(sidecar_path(output).exists())

    def test_schema_and_aggregation_protocol_are_fail_closed(self) -> None:
        mutations = []

        wrong_state = copy.deepcopy(self.payload)
        wrong_state["state"] = "evaluated"
        mutations.append(wrong_state)

        unknown_nested = copy.deepcopy(self.payload)
        unknown_nested["arms"][0]["runs"][0]["metric"] = 0.9
        mutations.append(unknown_nested)

        empty_runtime = copy.deepcopy(self.payload)
        empty_runtime["selection"]["runtime_files"] = []
        mutations.append(empty_runtime)

        invalid_environment = copy.deepcopy(self.payload)
        invalid_environment["arms"][0]["method_environment"] = {"BAD-KEY": "x"}
        mutations.append(invalid_environment)

        wrong_ddof = copy.deepcopy(self.payload)
        wrong_ddof["aggregation"]["sample_std_ddof"] = 0
        mutations.append(wrong_ddof)

        disabled_pairing = copy.deepcopy(self.payload)
        disabled_pairing["aggregation"]["paired_t_test"] = False
        mutations.append(disabled_pairing)

        for payload in mutations:
            with self.subTest(payload=payload), self.assertRaises(ManifestError):
                validate_manifest(payload)

    def test_every_arm_must_exactly_cover_required_seed_set(self) -> None:
        missing = copy.deepcopy(self.payload)
        missing["arms"][1]["runs"].pop()
        with self.assertRaisesRegex(ManifestError, r"missing=\[42\]"):
            validate_manifest(missing)

        extra = copy.deepcopy(self.payload)
        extra["aggregation"]["required_seed_set"] = [41]
        with self.assertRaisesRegex(ManifestError, r"extra=\[42\]"):
            validate_manifest(extra)

    def test_methods_run_seeds_checkpoints_and_outputs_must_be_unique(self) -> None:
        duplicate_method = copy.deepcopy(self.payload)
        duplicate_method["arms"][1]["method"] = "no_cga"

        duplicate_seed = copy.deepcopy(self.payload)
        duplicate_seed["arms"][0]["runs"][1]["seed"] = 41

        duplicate_checkpoint = copy.deepcopy(self.payload)
        duplicate_checkpoint["arms"][1]["runs"][0]["checkpoint"] = copy.deepcopy(
            duplicate_checkpoint["arms"][0]["runs"][0]["checkpoint"]
        )

        duplicate_output = copy.deepcopy(self.payload)
        duplicate_output["arms"][1]["runs"][0]["output_dir"] = duplicate_output["arms"][
            0
        ]["runs"][0]["output_dir"]

        duplicate_required_seed = copy.deepcopy(self.payload)
        duplicate_required_seed["aggregation"]["required_seed_set"] = [41, 41]

        cases = {
            "method": duplicate_method,
            "run_seed": duplicate_seed,
            "checkpoint": duplicate_checkpoint,
            "output": duplicate_output,
            "required_seed": duplicate_required_seed,
        }
        for name, payload in cases.items():
            with self.subTest(name=name), self.assertRaisesRegex(
                ManifestError, "duplicate|reused"
            ):
                validate_manifest(payload)

    def test_paired_comparator_must_name_a_declared_arm(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["aggregation"]["paired_comparator"] = "missing_baseline"
        with self.assertRaisesRegex(ManifestError, "does not name an arm"):
            validate_manifest(payload)

    def test_checkpoint_identity_and_actual_metadata_are_mandatory(self) -> None:
        base = self.payload["arms"][0]["runs"][0]
        mutations = []

        wrong_basename = copy.deepcopy(self.payload)
        wrong_basename["arms"][0]["runs"][0]["checkpoint"][
            "path"
        ] = "runs/no_cga/seed_41/final.pth"
        mutations.append(wrong_basename)

        wrong_size = copy.deepcopy(self.payload)
        wrong_size["arms"][0]["runs"][0]["checkpoint"]["size_bytes"] += 1
        mutations.append(wrong_size)

        wrong_hash = copy.deepcopy(self.payload)
        wrong_hash["arms"][0]["runs"][0]["checkpoint"]["sha256"] = "0" * 64
        mutations.append(wrong_hash)

        wrong_iter = copy.deepcopy(self.payload)
        wrong_iter["arms"][0]["runs"][0]["checkpoint"]["meta"]["iter"] = 4234
        mutations.append(wrong_iter)

        for payload in mutations:
            with self.subTest(payload=payload), self.assertRaises(ManifestError):
                validate_manifest(payload)

        other_checkpoint = self.root / "other" / "iter_4235_ema.pth"
        other_checkpoint.parent.mkdir()
        torch.save(
            {"meta": {"iter": 4235, "epoch": 1, "seed": 42}},
            other_checkpoint,
        )
        actual_meta_mismatch = copy.deepcopy(self.payload)
        replacement = actual_meta_mismatch["arms"][0]["runs"][0]["checkpoint"]
        replacement.update(
            {
                "path": str(other_checkpoint.relative_to(self.root)),
                "size_bytes": other_checkpoint.stat().st_size,
                "sha256": sha256_file(other_checkpoint),
            }
        )
        self.assertEqual(base["seed"], 41)
        with self.assertRaisesRegex(ManifestError, "metadata 'seed' mismatch"):
            validate_manifest(actual_meta_mismatch)

    def test_nonempty_or_conflicting_output_directories_are_rejected(self) -> None:
        nonempty = self.root / "already_has_results"
        self._write("already_has_results/occupied.sentinel", b"occupied\n")
        payload = copy.deepcopy(self.payload)
        payload["arms"][0]["runs"][0]["output_dir"] = str(nonempty)
        with self.assertRaisesRegex(ManifestError, "already contain results"):
            validate_manifest(payload)

        draft = self._write_draft()
        planned = self.root / self.payload["arms"][0]["runs"][0]["output_dir"]
        with self.assertRaisesRegex(ManifestError, "planned test output"):
            create_manifest(draft, planned / "manifest.json")

    def test_verify_allows_outputs_after_pristine_freeze(self) -> None:
        manifest = self.root / "frozen" / "final_test_manifest.json"
        create_manifest(self._write_draft(), manifest)

        selected_output = self.root / self.payload["arms"][0]["runs"][0]["output_dir"]
        selected_output.mkdir(parents=True)
        (selected_output / "predictions.pkl").write_bytes(b"generated later")

        # Direct validation remains the create-time pristine-output gate.
        with self.assertRaisesRegex(ManifestError, "already contain results"):
            validate_manifest(self.payload)
        # Integrity verification is phase-aware and still validates every
        # frozen artifact while allowing generated results.
        self.assertEqual(
            verify_manifest(manifest), hashlib.sha256(manifest.read_bytes()).hexdigest()
        )

    def test_verify_rejects_selection_artifact_drift(self) -> None:
        output = self.root / "manifest.json"
        create_manifest(self._write_draft(), output)
        self.runtime.write_bytes(b"changed after freeze\n")

        with self.assertRaisesRegex(ManifestError, "selection.runtime_files.*mismatch"):
            verify_manifest(output)

    def test_verify_rejects_manifest_sidecar_and_noncanonical_drift(self) -> None:
        output = self.root / "manifest.json"
        create_manifest(self._write_draft(), output)

        original = output.read_bytes()
        output.write_bytes(original + b" ")
        with self.assertRaisesRegex(ManifestError, "sidecar mismatch"):
            verify_manifest(output)

        output.write_bytes(original)
        sidecar_path(output).write_text("0" * 64 + f"  {output.name}\n")
        with self.assertRaisesRegex(ManifestError, "sidecar mismatch"):
            verify_manifest(output)

        pretty = (
            json.dumps(self.payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        )
        output.write_bytes(pretty)
        digest = hashlib.sha256(pretty).hexdigest()
        sidecar_path(output).write_text(f"{digest}  {output.name}\n", encoding="utf-8")
        with self.assertRaisesRegex(ManifestError, "not canonical JSON"):
            verify_manifest(output)

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks are unavailable")
    def test_verify_rejects_symlink_replacement_after_create(self) -> None:
        output = self.root / "frozen" / "final_test_manifest.json"
        create_manifest(self._write_draft(), output)
        manifest_copy = self._write("outside/manifest.json", output.read_bytes())
        sidecar_copy = self._write(
            "outside/manifest.sha256", sidecar_path(output).read_bytes()
        )

        output.unlink()
        output.symlink_to(manifest_copy)
        with self.assertRaisesRegex(ManifestError, "regular non-symlink"):
            verify_manifest(output)

        output.unlink()
        output.write_bytes(manifest_copy.read_bytes())
        sidecar_path(output).unlink()
        sidecar_path(output).symlink_to(sidecar_copy)
        with self.assertRaisesRegex(ManifestError, "regular non-symlink"):
            verify_manifest(output)

    def test_duplicate_json_keys_and_cli_failure_are_rejected(self) -> None:
        duplicate = self.root / "duplicate.json"
        duplicate.write_text(
            '{"schema_version":2,"schema_version":2}\n', encoding="utf-8"
        )
        with self.assertRaisesRegex(ManifestError, "duplicate JSON key"):
            load_json(duplicate)

        output = self.root / "must_not_exist.json"
        with redirect_stderr(io.StringIO()):
            self.assertEqual(
                main(
                    [
                        "create",
                        "--draft",
                        str(duplicate),
                        "--output",
                        str(output),
                    ]
                ),
                1,
            )
        self.assertFalse(output.exists())

    def test_create_refuses_to_overwrite_any_frozen_input(self) -> None:
        draft = self._write_draft()
        checkpoint = Path(self.payload["arms"][0]["runs"][0]["checkpoint"]["path"])
        for output in (draft, self.root / checkpoint, self.runtime):
            with self.subTest(output=output), self.assertRaisesRegex(
                ManifestError, "would overwrite a frozen input"
            ):
                create_manifest(draft, output)


if __name__ == "__main__":
    unittest.main()
