"""Exact auxiliary admission, new provenance, and task-scoped GPU selection."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

from experiments.comparison.mdp_checkpoint_admission import (
    auxiliary_shapes, detector_state_for_mdp, load_mdp_checkpoint,
)
from experiments.comparison import mdp_joint_inputs as declarations
from experiments.comparison import roi_rollout_bind as binder
from experiments.comparison import roi_rollout_native as producer


class MDPJointValidationTest(unittest.TestCase):
    def core(self):
        return {
            "backbone.layer1.0.bn3.weight": torch.ones(16),
            "backbone.layer3.0.bn3.weight": torch.ones(16),
            "roi_head.bbox_head.fc_cls.weight": torch.ones(3, 16),
        }

    def student(self, core):
        return {**core, **{
            key: torch.zeros(shape, dtype=(
                torch.int64 if key.endswith("num_batches_tracked") else torch.float32))
            for key, shape in auxiliary_shapes(core).items()
        }}

    def test_only_exact_auxiliaries_are_removed_and_core_tensors_preserved(self):
        core = self.core()
        student = self.student(core)
        result, proof = detector_state_for_mdp(student, core, "student")
        self.assertEqual(len(proof["auxiliary_keys"]), 11)
        self.assertEqual(set(result), set(core))
        self.assertEqual(len(student), len(core) + 11)
        for key in core:
            self.assertIs(result[key], student[key])
        result, proof = detector_state_for_mdp(core, core, "ema")
        self.assertEqual(proof["auxiliary_keys"], [])
        with self.assertRaisesRegex(ValueError, "key mismatch"):
            detector_state_for_mdp(student, core, "ema")

    def test_missing_extra_rogue_extra_core_shape_and_aux_shape_are_rejected(self):
        core = self.core()
        student = self.student(core)
        variants = []
        missing_aux = dict(student)
        missing_aux.pop("mdp.transform.0.bias")
        variants.append(missing_aux)
        variants.append({**student, "mdp.unrecognized.weight": torch.ones(1)})
        missing_core = dict(student)
        missing_core.pop(next(iter(core)))
        variants.append(missing_core)
        variants.append({**student, "roi_head.bbox_head.fc_cls.weight": torch.ones(2, 16)})
        variants.append({**student, "mdp.history.previous_student": torch.zeros(2, 16)})
        variants.append({**student, "mdp.transform.0.bias": torch.zeros(16, dtype=torch.float64)})
        for state in variants:
            with self.subTest(keys=list(state)), self.assertRaises(ValueError):
                detector_state_for_mdp(state, core, "student")

    def test_nonfinite_auxiliary_cannot_be_hidden_by_inference_filtering(self):
        core = self.core()
        student = self.student(core)
        student["mdp.transform.0.bias"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite MDP training auxiliary"):
            detector_state_for_mdp(student, core, "student")

    def test_loader_must_actually_apply_strict_detector_state(self):
        from mmcv.runner import checkpoint as checkpoint_module

        core = self.core()
        raw = self.student(core)
        model = types.SimpleNamespace(state_dict=lambda: core)
        calls = []

        def original_state_loader(module, state, strict=False, logger=None):
            calls.append((module, state, strict))

        def loader(module, filename):
            checkpoint_module.load_state_dict(module, raw, strict=False)
            return {"state_dict": raw, "meta": {"iter": 266}}

        with patch.object(checkpoint_module, "load_state_dict", original_state_loader):
            checkpoint, proof = load_mdp_checkpoint(loader, "student", model, "fixture.pth")
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0][2])
        self.assertEqual(set(calls[0][1]), set(core))
        self.assertIs(checkpoint["state_dict"], raw)
        self.assertTrue(proof["strict_detector_load"])
        with self.assertRaisesRegex(RuntimeError, "did not execute"):
            load_mdp_checkpoint(lambda *_args: {}, "student", model, "fixture.pth")

    def plan(self, root):
        checkpoint = root / "iter_266.pth"
        checkpoint.write_bytes(b"fixture-not-a-model")
        config = root / "config.py"
        config.write_text("fixture = True\n")
        selection = root / "selection.json"
        ids = [str(i) for i in range(8538)]
        selection.write_text(json.dumps({"image_ids": ids[:32]}))
        proof_file = root / "producer.json"
        proof = {
            "status": "FINITE_NATIVE_FINAL_PAIR_VALIDATED",
            "cell": "RSAR/chaff/42/MDP", "algorithm_code_sha": "actual-training",
            "valid_finite_diagnostic_pair": True, "completed_updates": 265,
            "original_nan_fixed": False, "lineage": {"seed": 42},
            "student": {"path": "/source/iter_266.pth", "bytes": checkpoint.stat().st_size,
                        "finite": True, "epoch": 1, "iteration": 266},
        }
        proof_file.write_text(json.dumps(proof))
        run = {
            "run_id": "RSAR/chaff/seed_42/MDP/student", "canonical_key": "RSAR/chaff/42/MDP/student",
            "dataset": "RSAR", "domain": "chaff", "seed": 42, "method": "MDP",
            "role": "student", "scope": "full_test", "checkpoint": str(checkpoint),
            "config": str(config), "checkpoint_proof": str(proof_file),
            "training_code_sha": "actual-training", "source_id": "source-id",
            "evaluation_code_sha": "actual-evaluation", "image_ids": ids,
            "visualization_image_ids": ids[:32], "visualization_selection_evidence": str(selection),
            "declaration_kind": "NEW_MDP_RUNTIME_ROW_NOT_HISTORICAL_REUSE",
            "ann_file": "/staged/test/annfiles", "img_prefix": "/staged/chaff/test/images",
            "allowed_gpus": [1], "out_dir": "/future/roi",
        }
        transfer = {
            "role": "student", "source_path": proof["student"]["path"],
            "destination_path": str(checkpoint), "source": declarations.identity(checkpoint),
            "destination": declarations.identity(checkpoint), "match": True,
        }
        plan = {
            "schema": declarations.SCHEMA, "runs": [run],
            "input_receipts": {
                "checkpoint_identities": [transfer],
                "config_identities": {
                    "producer_proof": declarations.identity(proof_file),
                    "evaluator_config": declarations.identity(config),
                    "frozen32": declarations.identity(selection),
                },
            },
        }
        return plan, run

    def test_new_mdp_binding_uses_real_producer_not_fabricated_bf_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, original = self.plan(root)
            run, declaration = declarations.declared_run(plan, original["run_id"])
            source_plan = root / "declared.json"
            source_plan.write_text(json.dumps(plan))
            native = root / "native"
            native.mkdir()
            (native / "execution.json").write_text("{}")
            (native / "predictions.pkl.image_ids.json").write_text("{}")
            cell = {name: run[name] for name in (
                "dataset", "domain", "seed", "method", "role", "checkpoint", "config",
                "ann_file", "img_prefix", "training_code_sha", "source_id")}
            cell["eval_dir"] = str(native)
            (root / "native_pass.json").write_text(json.dumps({
                "status": "complete", "kind": "mdp", "profile": {"host": 134},
                "cell": cell, "physical_gpu": 1, "canonical_key": run["canonical_key"],
                "evaluation_code_sha": "actual-evaluation",
                "original_input": {"run": original, "source_plan": str(source_plan),
                                   "declaration": declaration},
            }))
            with patch.object(binder, "load_host", return_value=types.SimpleNamespace(
                    map_path=str, map_data=copy.deepcopy)):
                with self.assertRaises(FileNotFoundError):
                    binder.bind_plan(root, export_code_sha="new-exporter")
                (root / "mdp_checkpoint_load.json").write_text(json.dumps({
                    "method": "MDP", "role": "student", "checkpoint": run["checkpoint"],
                    "strict_detector_load": True, "detector_keys": 396,
                    "auxiliary_keys": list(auxiliary_shapes(self.core())),
                }))
                _, bound, _ = binder.bind_plan(root, export_code_sha="new-exporter")
            self.assertEqual(bound["allowed_gpus"], [1])
            self.assertNotIn("historical_input_provenance", bound["native_prediction"])
            self.assertEqual(bound["native_prediction"]["declared_training_provenance"], declaration)
            self.assertEqual(json.loads(source_plan.read_text()), plan)

    def test_declaration_rejects_historical_reuse_or_changed_visual_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, run = self.plan(Path(directory))
            plan["runs"][0]["native_prediction"] = {"eval_dir": "/old/bf"}
            with self.assertRaisesRegex(ValueError, "not historical"):
                declarations.declared_run(plan, run["run_id"])
            del plan["runs"][0]["native_prediction"]
            plan["runs"][0]["visualization_image_ids"] = run["image_ids"][1:33]
            with self.assertRaisesRegex(ValueError, "frozen selection"):
                declarations.declared_run(plan, run["run_id"])

    def test_mdp_producer_resolves_new_helper_outside_the_frozen_writer_namespace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, run = self.plan(root)
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps(plan))
            case = root / "case"
            case.mkdir()
            profile = {"writer_code": str(root), "python": sys.executable,
                       "evaluation_code": "/frozen-evaluator", "allowed_gpus": [1, 2]}
            spec_file = root / "spec.json"
            spec_file.write_text(json.dumps({
                "kind": "mdp", "run_id": run["run_id"], "source_plan": str(plan_path),
                "canonical_key": run["canonical_key"], "case_root": str(case),
                "profile": profile,
            }))
            host = types.SimpleNamespace(
                approved_gpus=lambda: (1, 2), map_data=copy.deepcopy)
            calls = []
            shadow = types.ModuleType("experiments.comparison.mdp_joint_inputs")
            shadow.declared_run = lambda *_args: self.fail("Loaded the writer's wrong helper")
            cwd = Path.cwd()
            try:
                with patch.object(producer, "load_profile_producer", return_value=(
                        host, lambda cell, runtime, gpu: calls.append((cell, runtime, gpu)))), \
                        patch.object(producer.subprocess, "check_output", return_value="actual-evaluation\n"), \
                        patch.object(sys, "argv", ["native", str(spec_file), "1"]), \
                        patch.object(sys, "path", list(sys.path)), \
                        patch.dict(os.environ, {"IRAOD_RUNTIME_READY": "1"}), \
                        patch.dict(sys.modules, {shadow.__name__: shadow}):
                    producer.main()
            finally:
                os.chdir(cwd)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][0]["method"], "MDP")
            self.assertEqual(calls[0][2], 1)
            passed = json.loads((case / "native_pass.json").read_text())
            self.assertEqual(passed["kind"], "mdp")
            self.assertEqual(passed["original_input"]["declaration"]["kind"],
                             "new-trained-method-declaration")

    def test_task_resolver_preserves_base_file_and_wraps_itself_on_gpu1_2(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.py"
            text = 'TARGET_HOST_134 = "host134"\ndef approved_gpus():\n    return (4, 5, 6, 7)\n'
            base.write_text(text)
            output = root / "task_binding.py"
            declarations.write_resolver(output, base)
            spec = importlib.util.spec_from_file_location("mdp_resolver_test", output)
            resolver = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(resolver)
            with patch.object(resolver.socket, "gethostname", return_value="host134"):
                self.assertEqual(resolver.approved_gpus(), (1, 2))
                self.assertEqual(resolver.base.approved_gpus(), (1, 2))
            with patch.object(resolver.socket, "gethostname", return_value="other"):
                with self.assertRaises(ValueError):
                    resolver.approved_gpus()
            command = ["python", "/native/test.py", "config.py"]
            expected = ["python", str(output), "native", "/native/test.py", "config.py"]
            self.assertEqual(resolver.native_command(command, "/native/test.py"), expected)
            self.assertEqual(resolver.base.native_command(command, "/native/test.py"), expected)
            self.assertEqual(base.read_text(), text)


if __name__ == "__main__":
    unittest.main()
