"""CPU regressions using the real bbox head, rotated kernel and export consumer."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from mmcv import ConfigDict
from mmrotate.models.roi_heads.bbox_heads import rotated_bbox_head
from mmrotate.models.roi_heads.bbox_heads.convfc_rbbox_head import (
    RotatedShared2FCBBoxHead)

from experiments.comparison.aligned_roi import AlignedRoICapture, SCHEMA
from experiments.comparison.result_completion import (
    DOMAINS, EXPECTED_TEST_IMAGES, build_plan, collect, load_export, visualize, write_json)
from experiments.comparison.joint_tsne import joint_tsne


# Load the real compatibility module without importing unrelated training plugins.
spec = importlib.util.spec_from_file_location(
    "iraod_nms_compat", Path(__file__).resolve().parents[2] / "sfod/compat.py")
compat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compat)
compat.patch_mmrotate_multiclass_nms_rotated()


class AlignedMappingTest(unittest.TestCase):
    def setUp(self):
        self.head = RotatedShared2FCBBoxHead(
            in_channels=4, fc_out_channels=8, roi_feat_size=1,
            num_classes=2, reg_class_agnostic=False)
        with torch.no_grad():
            self.head.fc_cls.weight.zero_()
            self.head.fc_cls.weight[:, :3] = torch.eye(3)
            self.head.fc_cls.bias.zero_()
        self.features = torch.arange(32, dtype=torch.float32).reshape(4, 8)
        self.features[:, :3] = torch.tensor([
            [.05, .8, .15], [.9, .08, .02], [.7, .25, .05],
            [.6, .35, .05]]).log()
        self.rois = torch.tensor([
            [0., 100, 100, 20, 10, .4], [0, 20, 20, 20, 10, .4],
            [0, 20, 20, 20, 10, .4], [0, 200, 200, 20, 10, .4]])
        self.delta = torch.zeros((4, 10))
        # A class-specific decoded box must not be replaced by its proposal box.
        self.delta[3, 5] = 1.0
        self.cfg = ConfigDict(
            score_thr=.2, nms=dict(iou_thr=.1), max_per_img=4)

    def infer(self, features, rois, delta, rescale=True):
        logits = self.head.fc_cls(features)
        return self.head.get_bboxes(
            rois, logits, delta, (400, 400, 3),
            np.array([2., 2., 2., 2.]), rescale=rescale, cfg=self.cfg)

    def capture(self, features, rois, delta, rescale=True):
        with AlignedRoICapture(self.head, rotated_bbox_head) as capture:
            dets, labels = self.infer(features, rois, delta, rescale)
            per_class = [dets[labels == i].detach().numpy() for i in range(2)]
            arrays = capture.aligned("same_image", per_class)
        return arrays, dets, labels

    def test_real_multiclass_nms_export_matches_unobserved_inference(self):
        for rescale in (True, False):
            with self.subTest(rescale=rescale):
                plain_boxes, plain_labels = self.infer(
                    self.features, self.rois, self.delta, rescale)
                arrays, boxes, labels = self.capture(
                    self.features, self.rois, self.delta, rescale)
                torch.testing.assert_close(boxes, plain_boxes, rtol=0, atol=0)
                torch.testing.assert_close(labels, plain_labels, rtol=0, atol=0)
                np.testing.assert_array_equal(arrays["flat_indices"], [2, 1, 6, 7])
                np.testing.assert_array_equal(arrays["proposal_indices"], [1, 0, 3, 3])
                np.testing.assert_array_equal(arrays["labels"], [0, 1, 0, 1])
                np.testing.assert_array_equal(
                    arrays["features"], self.features.detach().numpy()[[1, 0, 3, 3]])
                np.testing.assert_array_equal(arrays["boxes"], boxes[:, :5].detach().numpy())
                np.testing.assert_array_equal(arrays["scores"], boxes[:, 5].detach().numpy())
                self.assertNotEqual(float(boxes[2, 0]), float(boxes[3, 0]))
                self.assertIs(rotated_bbox_head.multiclass_nms_rotated,
                              compat._multiclass_nms_rotated_device_safe)

    def test_no_proposals_and_all_below_threshold(self):
        low = self.features.clone()
        low[:, :3] = torch.tensor([.01, .01, .98]).log()
        for features, rois, delta in (
                (self.features[:0], self.rois[:0], self.delta[:0]),
                (low, self.rois, self.delta)):
            with self.subTest(proposals=len(features)):
                arrays, boxes, labels = self.capture(features, rois, delta)
                self.assertEqual(arrays["features"].shape, (0, 8))
                self.assertEqual(arrays["boxes"].shape, (0, 5))
                self.assertEqual(boxes.shape, (0, 6))
                self.assertEqual(labels.shape, (0,))

    def test_score_factors_filter_before_multiply_and_original_indices(self):
        scores = torch.tensor([[.1, .8, .1], [.9, .05, .05]])
        boxes = self.rois[:2, 1:].clone()
        factors = torch.tensor([.1, 1.])
        dets, labels, flat = compat._multiclass_nms_rotated_device_safe(
            boxes, scores, .2, ConfigDict(iou_thr=.1), score_factors=factors,
            return_inds=True)
        np.testing.assert_array_equal(flat.numpy(), [2, 1])
        np.testing.assert_array_equal(labels.numpy(), [0, 1])
        np.testing.assert_allclose(dets[:, 5].numpy(), [.9, .08])

    def test_export_rejects_changed_prediction_order(self):
        with AlignedRoICapture(self.head, rotated_bbox_head) as capture:
            boxes, labels = self.infer(self.features, self.rois, self.delta)
            per_class = [boxes[labels == i].detach().numpy()[::-1] for i in range(2)]
            with self.assertRaisesRegex(ValueError, "returned predictions"):
                capture.aligned("same_image", per_class)


def bindings(root, full_size=False):
    root.mkdir(parents=True, exist_ok=True)
    datasets = {}
    for dataset, domains in DOMAINS.items():
        count = ({"RSAR": 8538, "DIOR": 11738} if full_size else
                 {"RSAR": 35, "DIOR": 18})[dataset]
        ids = ([f"rsar_{i:05}" for i in range(count)] if dataset == "RSAR"
               else [str(i) for i in range(11726, 11726 + count)])
        selection = root / f"{dataset}_selection.json"
        write_json(selection, {"image_ids": ids[:32 if dataset == "RSAR" else 16]})
        images = root / dataset / "images"
        images.mkdir(parents=True, exist_ok=True)
        for image_id in ids:
            (images / (image_id + ".png")).touch()
        split = root / dataset / "test.txt"
        if dataset == "RSAR":
            split = root / dataset / "annfiles"
            split.mkdir(exist_ok=True)
            for image_id in ids:
                (split / (image_id + ".txt")).touch()
        else:
            split.write_text("\n".join(ids) + "\n")
        datasets[dataset] = {
            "config": "/remote/source_inference.py",
            "source_checkpoint": f"/remote/{dataset}/source/epoch_100.pth",
            "test_split": str(split),
            "visualization_selection_evidence": str(selection),
            "domains": {
                domain: {"ann_file": str(split),
                         "img_prefix": str(images),
                         "checkpoint_domain": domain}
                for domain in domains},
            "checkpoints": {
                domain: {
                    method: {role: f"/remote/{dataset}/{domain}/{method}/final_{role}.pth"
                             for role in ("ema", "student")} for method in "BCDEF"}
                for domain in domains},
        }
    return {"output_root": str(root), "datasets": datasets}


class CompletionTest(unittest.TestCase):
    def setUp(self):
        sizes = patch.dict(EXPECTED_TEST_IMAGES, {"RSAR": 35, "DIOR": 18})
        sizes.start()
        self.addCleanup(sizes.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.plan = build_plan(bindings(self.root))

    def test_exact_coverage_and_no_invented_student_or_completion(self):
        self.assertEqual(len(self.plan["runs"]), 132)
        rows = list(collect(self.plan))
        self.assertEqual(len(rows), 3872)
        self.assertTrue(all(row["roi_status"] == "not_started" for row in rows))
        self.assertEqual({r["role"] for r in rows if r["method"] == "A"}, {"source"})
        self.assertEqual(len([r for r in rows if r["dataset"] == "RSAR"]), 3080)
        bad = bindings(self.root)
        visual = Path(bad["datasets"]["DIOR"]["visualization_selection_evidence"])
        ids = json.loads(visual.read_text())["image_ids"]
        ids[0], ids[1] = ids[1], ids[0]
        write_json(visual, {"image_ids": ids})
        with self.assertRaisesRegex(ValueError, "11726-11741"):
            build_plan(bad)
        bad = bindings(self.root)
        visual = Path(bad["datasets"]["RSAR"]["visualization_selection_evidence"])
        ids = json.loads(visual.read_text())["image_ids"]
        ids[0] = "not_in_test"
        write_json(visual, {"image_ids": ids})
        with self.assertRaisesRegex(ValueError, "visualization selection"):
            build_plan(bad)

    def test_cpu_cli_plan_collect_and_export_help(self):
        binding_file = self.root / "bindings.json"
        plan_file = self.root / "plan.json"
        write_json(binding_file, bindings(self.root / "actual_size", full_size=True))
        subprocess.run([
            sys.executable, "-m", "experiments.comparison.result_completion",
            "plan", "--bindings", str(binding_file), "--out", str(plan_file)],
            check=True, capture_output=True, text=True)
        full = json.loads(plan_file.read_text())
        self.assertEqual(full["roi_image_roles"], 1267816)
        self.assertEqual(full["visualization_image_roles"], 3520)
        self.assertEqual(len(full["runs"]), 132)
        self.assertEqual(len(full["runs"][0]["image_ids"]), 8538)
        self.assertEqual(len(full["runs"][-1]["image_ids"]), 11738)
        # The CLI check streams one full-TEST run, not a million-row temporary CSV.
        full["runs"] = full["runs"][:1]
        write_json(plan_file, full)
        subprocess.run([
            sys.executable, "-m", "experiments.comparison.result_completion",
            "collect", "--plan", str(plan_file)],
            check=True, capture_output=True, text=True)
        coverage = self.root / "actual_size" / SCHEMA / "coverage.csv"
        self.assertEqual(len(coverage.read_text().splitlines()), 8539)
        self.assertNotIn(",complete,", coverage.read_text())
        for module in (
                "experiments.comparison.dior_recovery.extract_roi_pre_fc_cls",
                "experiments.comparison.joint_tsne"):
            subprocess.run([sys.executable, "-m", module, "--help"],
                           check=True, capture_output=True, text=True)
        scripts = Path(__file__).resolve().parents[2] / "experiments/comparison/dior_recovery"
        for name in ("run_roi.sh", "run_vis.sh"):
            result = subprocess.run([str(scripts / name)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("usage:", result.stderr)

    def test_each_domain_uses_its_own_final_checkpoint(self):
        expected = bindings(self.root)["datasets"]
        for run in self.plan["runs"]:
            with self.subTest(run=run["run_id"]):
                spec = expected[run["dataset"]]
                if run["method"] == "A":
                    self.assertEqual(run["checkpoint_domain"], "source")
                    self.assertEqual(run["checkpoint"], spec["source_checkpoint"])
                else:
                    self.assertEqual(run["checkpoint_domain"], run["domain"])
                    self.assertEqual(
                        run["checkpoint"],
                        spec["checkpoints"][run["domain"]][run["method"]][run["role"]])

    def test_clean_rejects_corruption_checkpoint_binding(self):
        for dataset, domains in DOMAINS.items():
            for corruption in domains[1:]:
                with self.subTest(dataset=dataset, checkpoint_domain=corruption):
                    bad = bindings(self.root)
                    bad["datasets"][dataset]["domains"]["clean"]["checkpoint_domain"] = corruption
                    with self.assertRaisesRegex(ValueError, "clean must use its own"):
                        build_plan(bad)

    def export_fixture(self, run, arrays):
        import cv2

        out = Path(run["out_dir"])
        out.mkdir(parents=True)
        records = []
        for image_id in run["image_ids"]:
            filename = image_id + ".npz"
            copy = {k: v.copy() for k, v in arrays.items()}
            copy["image_ids"] = np.full(len(copy["labels"]), image_id)
            np.savez_compressed(out / filename, **copy)
            image = self.root / (image_id + ".png")
            cv2.imwrite(str(image), np.zeros((240, 240, 3), dtype=np.uint8))
            records.append({"image_id": image_id, "image_path": str(image),
                            "feature_file": filename, "n_roi": 4,
                            "n_detections": len(copy["labels"])})
        write_json(out / "index.json", {
            "schema": SCHEMA, "status": "complete", "run": run,
            "code_commit": "test-fixture", "classes": ["class0", "class1"],
            "rescale": True, "test_cfg": {"score_thr": .2, "nms": {"iou_thr": .1}},
            "feature_point": "roi_head.bbox_head.fc_cls.input", "records": records})

    def test_real_consumer_to_render_manifest_and_joint_tsne(self):
        fixture = AlignedMappingTest()
        fixture.setUp()
        arrays, _, _ = fixture.capture(fixture.features, fixture.rois, fixture.delta)
        runs = [r for r in self.plan["runs"]
                if r["dataset"] == "DIOR" and r["domain"] == "clean"
                and r["role"] in ("source", "ema")]
        for run in runs:
            self.export_fixture(run, arrays)
        visualize(runs[0])
        rows = list(collect(self.plan))
        self.assertEqual(sum(r["roi_status"] == "complete" for r in rows), 108)
        self.assertEqual(sum(r["vis_status"] == "complete" for r in rows), 16)
        coordinates, points = joint_tsne(
            self.plan, "DIOR", "clean", "ema", self.root / "tsne", cap=5, perplexity=2)
        self.assertEqual(coordinates.shape, (30, 2))
        self.assertEqual(len(list((self.root / "tsne").glob("*.pdf"))), 7)
        for point in points:
            with np.load(point["feature_file"]) as source:
                row = point["feature_row"]
                self.assertEqual(point["proposal_index"], source["proposal_indices"][row])
                self.assertEqual(point["predicted_label"], source["labels"][row])
        protocol = json.loads((self.root / "tsne/protocol.json").read_text())
        self.assertEqual(protocol["source_stats_count"], 72)
        self.assertTrue(all(s["reservoir_points"] == 5 for s in protocol["feature_stats"]))
        with np.load(self.root / "tsne/embedding.npz") as saved:
            population = arrays["features"].astype(np.float64)
            np.testing.assert_allclose(saved["source_mean"], population.mean(axis=0))
            np.testing.assert_allclose(saved["source_std"], population.std(axis=0))
        self.assertEqual(protocol["normalization_reference"]["role"], "source")
        second, metadata = joint_tsne(
            self.plan, "DIOR", "clean", "ema", self.root / "tsne-repeat",
            cap=5, perplexity=2)
        np.testing.assert_array_equal(coordinates, second)
        self.assertEqual(points, metadata)
        index_path = Path(runs[0]["out_dir"]) / "index.json"
        index = json.loads(index_path.read_text())
        index["records"].pop()
        write_json(index_path, index)
        with self.assertRaisesRegex(ValueError, "same-image selection"):
            load_export(runs[0])

    def test_streaming_full_roi_empty_rows_and_visual_subset(self):
        fixture = AlignedMappingTest()
        fixture.setUp()
        empty, _, _ = fixture.capture(fixture.features[:0], fixture.rois[:0], fixture.delta[:0])
        run = self.plan["runs"][0]
        self.export_fixture(run, empty)
        _, records = load_export(run)
        self.assertIs(iter(records), records)
        self.assertEqual(next(records)[1]["features"].shape, (0, 8))
        rows = list(collect({"schema": SCHEMA, "runs": [run]}))
        self.assertEqual(len(rows), 35)
        self.assertTrue(all(r["roi_status"] == "complete" and r["n_detections"] == 0 for r in rows))
        self.assertEqual(sum(r["vis_status"] == "not_selected" for r in rows), 3)
        self.assertEqual(sum(r["vis_status"] == "not_started" for r in rows), 32)
        Path(run["out_dir"], run["image_ids"][-1] + ".npz").unlink()
        with self.assertRaises(FileNotFoundError):
            list(collect({"schema": SCHEMA, "runs": [run]}))

    def test_missing_test_image_and_subset_schema_are_rejected(self):
        spec = bindings(self.root)
        image = next(Path(spec["datasets"]["RSAR"]["domains"]["clean"]["img_prefix"]).glob("*.png"))
        image.unlink()
        with self.assertRaisesRegex(ValueError, "complete TEST split"):
            build_plan(spec)
        with self.assertRaisesRegex(ValueError, "v3 full-test"):
            list(collect({"schema": "iraod-aligned-roi-v2", "runs": self.plan["runs"]}))

    def test_streaming_npz_rejects_non_aligned_row_fields(self):
        fixture = AlignedMappingTest()
        fixture.setUp()
        arrays, _, _ = fixture.capture(fixture.features, fixture.rois, fixture.delta)
        run = self.plan["runs"][0]
        self.export_fixture(run, arrays)
        path = Path(run["out_dir"]) / (run["image_ids"][0] + ".npz")
        with np.load(path) as saved:
            broken = {key: saved[key] for key in saved.files}
        broken["features"] = broken["features"][:-1]
        np.savez_compressed(path, **broken)
        _, records = load_export(run)
        with self.assertRaisesRegex(ValueError, "row count/shape mismatch"):
            next(records)


if __name__ == "__main__":
    unittest.main()
