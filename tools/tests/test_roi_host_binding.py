"""CPU-only regression for frozen ROI metadata on the selected migrated host."""

from contextlib import ExitStack
from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from experiments.comparison import host_binding as host
from experiments.comparison import result_completion as completion
from experiments.comparison import roi_host_binding as overlay
from experiments.comparison.report_qualitative import reuse_completed_evidence, inspect_embedding
from experiments.comparison.joint_tsne import joint_tsne
from tools.tests import test_generalized_qualitative as fixtures


class ROIHostBindingTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.GeneralizedQualitativeTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        queue = self.fixture.prepare(("IRG",))
        plan = completion.read_json(queue / "qualitative_seed42.json")
        self.plan = plan
        self.run = next(run for run in plan["runs"]
                        if (run["dataset"], run["domain"], run["method"], run["role"])
                        == ("DIOR", "brightness", "IRG", "ema"))
        self.old_root = "/mnt/shared/zechuan/iraod_artifacts"
        self.physical_root = str(self.fixture.root)
        self.run = deepcopy(self.run)
        for key in ("config", "ann_file", "img_prefix"):
            self.run[key] = self.physical_root + self.run[key]
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(
            host.socket, "gethostname", return_value=host.TARGET_HOST))
        self.stack.enter_context(patch.object(host, "PREFIXES", (
            (self.old_root, self.physical_root),
            ("/mnt/SSD2_8TB/zechuan/iraod_artifacts", self.physical_root),
            ("/mnt/HDD_14TB/zechuan/iraod_artifacts", self.physical_root),
            ("/home/zechuan/iraod_artifacts", self.physical_root),
        )))

    def original_paths(self, value):
        if isinstance(value, dict):
            return {key: self.original_paths(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.original_paths(item) for item in value]
        if isinstance(value, str) and value.startswith(self.physical_root + "/"):
            return self.old_root + value[len(self.physical_root):]
        return value

    def test_native_binding_reads_migrated_files_without_changing_logical_plan(self):
        self.fixture.native_fixture(self.run)
        run = self.original_paths(self.run)
        before = deepcopy(run)
        evidence = completion.native_binding_evidence(run)
        self.assertEqual(evidence["status"], "complete")
        self.assertEqual(run, before)
        self.assertFalse(Path(run["checkpoint"]).exists())

    def test_collector_reuses_migrated_export_with_equivalent_path_identity(self):
        self.fixture.export_fixture(self.run)
        run = self.original_paths(self.run)
        plan = {"schema": completion.SCHEMA, "runs": [run]}
        plan_file = self.fixture.root / "preserved-plan.json"
        content = json.dumps(plan).encode()
        plan_file.write_bytes(content)
        loaded = completion.load_run(self.original_paths(str(plan_file)), run["run_id"])
        self.assertEqual(loaded, run)
        index, records = completion.load_export(loaded)
        self.assertEqual(len(list(records)), len(run["image_ids"]))
        self.assertEqual(index["code_commit"], run["export_code_sha"])
        rows = list(completion.collect(plan))
        self.assertTrue(all(row["roi_status"] == "complete" for row in rows))
        self.assertEqual(plan_file.read_bytes(), content)

    def test_strict_native_mismatches_and_non_target_paths_remain_rejected(self):
        self.fixture.native_fixture(self.run)
        logical = self.original_paths(self.run)
        root = Path(self.run["native_prediction"]["eval_dir"])
        filename = root / "predictions.pkl.image_ids.json"
        original = filename.read_bytes()
        for field, value in (
                ("checkpoint", logical["checkpoint"] + ".different"),
                ("config", "/different/config.py"),
                ("training_code_sha", "wrong-training-code"),
                ("evaluation_code_sha", "wrong-eval-code"),
                ("image_ids", list(reversed(logical["image_ids"][:-1])))):
            with self.subTest(field=field):
                data = json.loads(original)
                data[field] = value
                filename.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    completion.native_binding_evidence(logical)
        filename.write_bytes(original)
        data = json.loads(original)
        data["cfg_options"]["data.test.img_prefix"] += "/different"
        filename.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "TEST split/domain"):
            completion.native_binding_evidence(logical)
        filename.write_bytes(original)
        with patch.object(host, "is_target_host", return_value=False):
            self.assertEqual(completion.native_binding_evidence(logical)["status"], "pending")
            self.assertFalse(host.same_data(self.run, logical))
            self.assertEqual(completion.native_binding_evidence(self.run)["status"], "complete")

    def test_native_metadata_accepts_only_equivalent_home_hdd_and_old_aliases(self):
        self.fixture.native_fixture(self.run)
        root = Path(self.run["native_prediction"]["eval_dir"])
        files = [root / name for name in ("execution.json", "predictions.pkl.image_ids.json")]
        before = {path: path.read_text() for path in files}
        logical = self.original_paths(self.run)
        for prefix in (self.old_root, "/mnt/SSD2_8TB/zechuan/iraod_artifacts",
                       "/mnt/HDD_14TB/zechuan/iraod_artifacts", "/home/zechuan/iraod_artifacts"):
            with self.subTest(prefix=prefix):
                for path, content in before.items():
                    path.write_text(content.replace(self.physical_root, prefix))
                self.assertEqual(completion.native_binding_evidence(logical)["status"], "complete")
        execution = json.loads(files[0].read_text())
        execution["checkpoint"] += ".different"
        completion.write_json(files[0], execution)
        with self.assertRaisesRegex(ValueError, "execution identity"):
            completion.native_binding_evidence(logical)

    def test_target_gpu_mapping_requires_lock_and_rejects_old_physical_devices(self):
        run = {"allowed_gpus": [4, 5, 6]}
        with patch.dict(os.environ, {"IRAOD_GPU_LOCKED": "1"}):
            for gpu in range(5):
                with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": str(gpu)}):
                    overlay.require_owned_gpu(run)
            for device in ("5", "6", "7", "0,1", ""):
                with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": device}):
                    with self.assertRaisesRegex(ValueError, "bound physical"):
                        overlay.require_owned_gpu(run)
        with patch.dict(os.environ, {"IRAOD_GPU_LOCKED": "0", "CUDA_VISIBLE_DEVICES": "0"}):
            with self.assertRaisesRegex(RuntimeError, "actual GPU lock"):
                overlay.require_owned_gpu(run)
        self.assertEqual(run, {"allowed_gpus": [4, 5, 6]})

    def frozen_export(self, sha, port):
        import torch

        code = self.fixture.root / ("frozen-port" if port else "frozen-bf")
        subprocess.run(["git", "clone", "--quiet", "--shared", "--no-checkout",
                        str(overlay.ROOT), str(code)], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(code), "checkout", "--quiet", "--detach", sha],
                       check=True, capture_output=True)
        run = deepcopy(self.run)
        if not port:
            for field in ("native_prediction", "export_code_sha", "allowed_gpus"):
                run.pop(field)
            run.update(method="B", seed=43, run_id="DIOR/brightness/B/ema")
        else:
            run.update(export_code_sha=sha, allowed_gpus=[4, 5, 6])
        run["out_dir"] = str(self.fixture.root / ("port-export" if port else "bf-export"))
        images = self.fixture.root / "input-images"
        images.mkdir(exist_ok=True)
        run["img_prefix"] = str(images)
        run["ann_file"] = str(self.fixture.root / "test.txt")
        Path(run["ann_file"]).write_text("\n".join(run["image_ids"]))
        run["config"] = str(self.fixture.root / "frozen-config.py")
        Path(run["config"]).write_text(
            "model=dict(pretrained=None,train_cfg=None)\n"
            "data=dict(test=dict(test_mode=True))\n")
        for image_id in run["image_ids"]:
            (images / f"{image_id}.png").write_bytes(b"synthetic image fixture")
        arrays = self.fixture.arrays(run["image_ids"][0])
        per_class = [np.column_stack((arrays["boxes"][arrays["labels"] == label],
                                      arrays["scores"][arrays["labels"] == label]))
                     for label in range(2)]
        if port:
            self.fixture.native_fixture(run)
            native = Path(run["native_prediction"]["eval_dir"])
            (native / "predictions.pkl").write_bytes(pickle.dumps([per_class] * len(run["image_ids"])))
        else:
            Path(run["checkpoint"]).parent.mkdir(parents=True, exist_ok=True)
            Path(run["checkpoint"]).write_bytes(b"synthetic checkpoint; not model weights")
        logical = self.original_paths(run)
        plan_file = self.fixture.root / ("port-plan.json" if port else "bf-plan.json")
        completion.write_json(plan_file, {"schema": completion.SCHEMA, "runs": [logical]})
        argv = [sys.executable, "-m", overlay.EXPORT_MODULE, "--plan",
                self.original_paths(str(plan_file)), "--run-id", run["run_id"]]
        job = dict(seed=run["seed"], run_id=run["run_id"], plan=argv[4],
                   checkpoint=logical["checkpoint"], out_dir=logical["out_dir"],
                   export_code_sha=sha, export_argv=argv)
        job["cwd" if port else "export_code"] = self.original_paths(str(code))
        jobs_file = self.fixture.root / ("port-jobs.json" if port else "bf-jobs.json")
        completion.write_json(jobs_file, [job])
        before = {path: path.read_bytes() for path in (
            plan_file, jobs_file, Path(run["config"]),
            code / "experiments/comparison/dior_recovery/extract_roi_pre_fc_cls.py",
            code / "experiments/comparison/aligned_roi.py")}

        class Dataset:
            CLASSES = ("class0", "class1")
            data_infos = [{"filename": str(images / f"{image_id}.png")}
                          for image_id in run["image_ids"]]

            def __len__(self):
                return len(self.data_infos)

        def build_dataset(cfg):
            self.assertEqual(cfg.ann_file, run["ann_file"])
            self.assertEqual(cfg.img_prefix, run["img_prefix"])
            return Dataset()

        def loader(subset, **kwargs):
            self.assertEqual(kwargs, dict(samples_per_gpu=1, workers_per_gpu=2,
                                          dist=False, shuffle=False))
            for i in subset.indices:
                filename = subset.dataset.data_infos[i]["filename"]
                yield {"img_metas": [SimpleNamespace(data=[[{
                    "filename": filename, "ori_filename": filename}]])]}

        dets = torch.tensor(np.column_stack((arrays["boxes"], arrays["scores"])))
        labels = torch.tensor(arrays["labels"])
        nms = SimpleNamespace(multiclass_nms_rotated=lambda *a, **kw: (
            dets, labels, torch.tensor(arrays["flat_indices"])))

        class Model:
            roi_head = SimpleNamespace(
                bbox_head=SimpleNamespace(fc_cls=torch.nn.Identity()),
                test_cfg={"score_thr": .05})

            def eval(self):
                pass

            def __call__(self, **kwargs):
                self.roi_head.bbox_head.fc_cls(torch.tensor(arrays["features"]))
                boxes, classes = nms.multiclass_nms_rotated(
                    torch.zeros((3, 5)), torch.zeros((3, 3)))
                return [[boxes[classes == label].numpy() for label in range(2)]]

        class Parallel:
            def __init__(self, model, device_ids):
                assert device_ids == [0]
                self.module = model

            def eval(self):
                self.module.eval()

            def __call__(self, **kwargs):
                return self.module(**kwargs)

        def checkpoint(model, filename, **kwargs):
            self.assertEqual(filename, run["checkpoint"])
            self.assertTrue(Path(filename).is_file())
            self.assertEqual(sys.path[0], str(code))
            self.assertEqual(Path(sys.modules["experiments.comparison.aligned_roi"].__file__),
                             code / "experiments/comparison/aligned_roi.py")
            return {"meta": {"CLASSES": Dataset.CLASSES}}

        modules = {
            "iraod_runtime": SimpleNamespace(ensure_iraod_runtime=lambda: None),
            "mmcv.parallel": SimpleNamespace(MMDataParallel=Parallel),
            "mmcv.runner": SimpleNamespace(load_checkpoint=checkpoint, wrap_fp16_model=lambda m: None),
            "mmdet.datasets": SimpleNamespace(build_dataloader=loader),
            "mmrotate.datasets": SimpleNamespace(build_dataset=build_dataset),
            "mmrotate.models": SimpleNamespace(build_detector=lambda *a, **kw: Model()),
            "mmrotate.models.roi_heads.bbox_heads": SimpleNamespace(rotated_bbox_head=nms),
            "mmrotate.utils": SimpleNamespace(compat_cfg=lambda c: c, setup_multi_processes=lambda c: None),
            "sfod.utils": SimpleNamespace(patch_config=lambda c: c),
        }
        previous_cwd, previous_path = Path.cwd(), list(sys.path)
        with ExitStack() as stack:
            stack.callback(os.chdir, previous_cwd)
            stack.callback(lambda: sys.path.__setitem__(slice(None), previous_path))
            stack.enter_context(patch.object(sys, "argv", []))
            stack.enter_context(patch.dict(sys.modules, modules))
            stack.enter_context(patch.dict(os.environ, {
                "IRAOD_GPU_LOCKED": "1", "CUDA_VISIBLE_DEVICES": "0"}))
            completion.write_json(jobs_file, [{**job, "export_code_sha": "0" * 40}])
            with self.assertRaisesRegex(ValueError, "recorded export code SHA"):
                overlay.export(self.original_paths(str(jobs_file)), run["seed"], run["run_id"])
            jobs_file.write_bytes(before[jobs_file])
            if port:
                completion.write_json(plan_file, {
                    "schema": completion.SCHEMA,
                    "runs": [{**logical, "export_code_sha": "0" * 40}]})
                with self.assertRaisesRegex(ValueError, "Export checkout differs"):
                    overlay.export(self.original_paths(str(jobs_file)), run["seed"], run["run_id"])
                plan_file.write_bytes(before[plan_file])
            self.assertFalse(Path(run["out_dir"]).exists())
            overlay.export(self.original_paths(str(jobs_file)), run["seed"], run["run_id"])
        index, records = completion.load_export(logical)
        for _, actual in records:
            for field, values in arrays.items():
                if field != "image_ids":
                    np.testing.assert_array_equal(actual[field], values)
        self.assertEqual(index["run"], logical)
        self.assertEqual(index["code_commit"], sha)
        self.assertEqual(index["host_binding"]["export_code_sha"], sha)
        self.assertEqual(index["host_binding"]["wrapper_code_sha"], subprocess.check_output(
            ["git", "-C", str(overlay.ROOT), "rev-parse", "HEAD"], text=True).strip())
        self.assertEqual(index["host_binding"]["export_argv"], argv)
        self.assertEqual(index["host_binding"]["physical_gpu"], 0)
        self.assertIsInstance(index["host_binding"]["wrapper_worktree_modified"], bool)
        self.assertTrue(all(path.read_bytes() == content for path, content in before.items()))
        def render(filename, boxes, labels, **kwargs):
            self.assertTrue(Path(filename).is_file())
            Path(kwargs["out_file"]).write_bytes(b"synthetic rendered view")

        with patch.dict(sys.modules, {"mmrotate.core": SimpleNamespace(imshow_det_rbboxes=render)}):
            completion.visualize(logical)
        rows = list(completion.collect({"schema": completion.SCHEMA, "runs": [logical]}))
        self.assertTrue(all(row["roi_status"] == "complete" for row in rows))
        self.assertEqual(sum(row["vis_status"] == "complete" for row in rows), 16)
        return logical

    def test_actual_frozen_bf_and_port_exporters_use_mapped_io_and_frozen_arithmetic(self):
        for sha, port in (
                ("331d2131b84651f0a2930a3d53faeefad8701531", False),
                ("b87f34ef06eb85589cbc7dd4d385666c37696083", True)):
            with self.subTest(port=port):
                self.frozen_export(sha, port)

    def test_migrated_joint_embedding_output_and_point_identity(self):
        for run in completion.comparison_runs(self.plan, "DIOR", "brightness", "ema"):
            self.fixture.export_fixture(run)
        logical = self.original_paths(self.plan)

        class Estimator:
            def __init__(self, **params):
                self.params, self.kl_divergence_ = params, 0.

            def get_params(self):
                return self.params

            def fit_transform(self, features):
                return features[:, :2].copy()

        figure, axes, plt = MagicMock(), MagicMock(), MagicMock()
        figure.savefig.side_effect = lambda path, **kw: Path(path).write_bytes(b"synthetic panel")
        plt.subplots.return_value = figure, axes
        modules = {
            "matplotlib": SimpleNamespace(use=lambda backend: None, pyplot=plt),
            "matplotlib.pyplot": plt,
            "sklearn": SimpleNamespace(__version__="synthetic-test-double"),
            "sklearn.manifold": SimpleNamespace(TSNE=Estimator),
        }
        directory = self.original_paths(str(self.fixture.root / "embedding"))
        with patch.dict(sys.modules, modules):
            coords, points = joint_tsne(
                logical, "DIOR", "brightness", "ema", directory, cap=3, perplexity=2)
        self.assertEqual(coords.shape, (21, 2))
        entry = dict(dataset="DIOR", domain="brightness", comparison="ema", directory=directory)
        self.assertEqual(inspect_embedding(entry, self.plan)["status"], "complete")
        points_file = host.read_path(directory) / "points.csv"
        points_file.write_text(points_file.read_text().replace(
            points[0]["checkpoint"], points[0]["checkpoint"] + ".different"))
        with self.assertRaisesRegex(ValueError, "point provenance"):
            inspect_embedding(entry, logical)

    def test_completed_collector_reuse_maps_compact_indices_without_reopening_npzs(self):
        # Synthetic compact evidence represents a prior audit; no experiment
        # result or feature NPZ is created or inferred by this reuse test.
        root = self.fixture.root / "accepted-evidence"
        root.mkdir()
        plan_file = self.fixture.root / "accepted-plan.json"
        logical = self.original_paths(self.plan)
        completion.write_json(plan_file, logical)
        groups = []
        for run in self.plan["runs"]:
            out = Path(run["out_dir"])
            (out / "visualizations").mkdir(parents=True)
            records = [dict(image_id=image_id, feature_file=image_id + ".npz", n_detections=3)
                       for image_id in run["image_ids"]]
            index = dict(schema=completion.SCHEMA, status="complete", run=run, records=records,
                         feature_point=completion.FEATURE_POINT, classes=["class0", "class1"],
                         code_commit=run.get("export_code_sha", "accepted-core"))
            if "native_prediction" in run:
                self.fixture.native_fixture(run)
                index.update(native_prediction=completion.native_binding_evidence(run),
                             feature_version=completion.FEATURE_VERSION)
            completion.write_json(out / "index.json", index)
            completion.write_json(out / "visualizations/index.json", dict(
                schema=completion.SCHEMA, status="complete", run=run,
                images=[dict(image_id=i, file=i + ".png") for i in run["visualization_image_ids"]]))
            groups.append(dict(
                run_id=run["run_id"], scope="full_test", status="complete", inspected=True,
                out_dir=run["out_dir"], expected_images=len(run["image_ids"]),
                validated_images=len(run["image_ids"]), detection_rows=3 * len(run["image_ids"]),
                visualizations_complete=len(run["visualization_image_ids"]),
                budget_group=run.get("budget_group", "source" if run["method"] == "A"
                                     else "common_one_epoch"),
                **{k: run[k] for k in ("dataset", "domain", "seed", "method", "role", "checkpoint")}))
        embeddings = []
        for ds, domains in completion.DOMAINS.items():
            for domain in domains:
                for role in ("ema", "student"):
                    out = root / "embeddings" / ds / domain / role
                    out.mkdir(parents=True)
                    runs = completion.comparison_runs(self.plan, ds, domain, role)
                    completion.write_json(out / "protocol.json", dict(
                        schema=completion.SCHEMA, runs=runs, sampling_seed=42,
                        tsne_parameters=dict(random_state=42), points_per_method=1, sample_cap=1000))
                    embeddings.append(dict(
                        dataset=ds, domain=domain, comparison=role, directory=str(out),
                        status="complete", n_points=len(runs)))
        for filename, rows in (("roi_group_summary.csv", groups), ("embedding_index.csv", embeddings)):
            with (root / filename).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        (root / "roi_vis_coverage.csv").write_text("synthetic prior streaming audit\n")
        fragment = dict(qualitative_plan=self.original_paths(str(plan_file)),
                        embeddings=self.original_paths(embeddings))
        completion.write_json(root / "report_input_fragment.json", fragment)
        images = sum(g["validated_images"] for g in groups)
        views = sum(g["visualizations_complete"] for g in groups)
        summary = dict(
            schema="iraod-qualitative-completion-summary-v1", qualitative_status="complete",
            full_test_roi="complete", plan=str(plan_file),
            roi_expected_image_roles=images, roi_identified_image_roles=images, roi_complete=images,
            roi_detection_rows=3 * images, roi_inspected_groups=len(groups), roi_complete_groups=len(groups),
            vis_expected_image_roles=views, vis_complete=views, embeddings_expected=24, embeddings_complete=24,
            embedding_points=sum(e["n_points"] for e in embeddings))
        summary.update({field: str(root / name) for field, name in (
            ("roi_image_manifest", "roi_vis_coverage.csv"), ("roi_group_summary", "roi_group_summary.csv"),
            ("embedding_index", "embedding_index.csv"), ("report_input_fragment", "report_input_fragment.json"))})
        completion.write_json(root / "qualitative_summary.json", self.original_paths(summary))
        with patch.object(np, "load", side_effect=AssertionError("Reuse must not reopen NPZs")):
            _, _, coverage = reuse_completed_evidence(dict(
                qualitative_plan=self.original_paths(str(plan_file)),
                qualitative_evidence=self.original_paths(str(root))))
        self.assertEqual(coverage["full_test_roi"], "complete")
        self.assertEqual(coverage["roi_complete_groups"], len(self.plan["runs"]))
        self.assertTrue(Path(coverage["roi_image_manifest"]).is_file())


if __name__ == "__main__":
    unittest.main()
