"""Exercise the real inference iterators and paired prediction/ID writer on CPU."""

from functools import partial
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import mmcv
from mmcv.parallel import DataContainer, collate
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from tools.prediction_export import save_predictions_with_ids


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "actual_evaluation_api", ROOT / "mmdet_extension/apis/test.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class OrderedFixture(Dataset):
    data_infos = [{"filename": name + ".png"} for name in ("zeta", "alpha", "mu")]

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, index):
        name = self.data_infos[index]["filename"]
        return {
            "img": [torch.tensor([index], dtype=torch.float32)],
            "img_metas": [DataContainer({
                "ori_filename": name, "filename": "/fixture/images/" + name,
            }, cpu_only=True)],
        }


class PredictionFixture(torch.nn.Module):
    def forward(self, img, img_metas, return_loss, rescale):
        assert not return_loss and rescale
        return [[np.zeros((0, 6), np.float32) if int(row[0]) == 1 else
                 np.array([[10 + int(row[0]), 20, 3, 4, .1, .2]], np.float32)]
                for row in img[0]]


class PredictionImageOrderTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = OrderedFixture()
        self.metadata = {
            "dataset_size": len(self.dataset), "dataset_type": type(self.dataset).__name__,
            "config": "/fixture/inference.py", "checkpoint": "/fixture/iter_266_ema.pth",
            "training_code_sha": "0f98a48",
        }

    def loader(self, padded=False):
        # Deliberately not data_infos order: prove no separate dataset scan is used.
        sampler = [2, 0, 1, 2] if padded else [2, 0, 1]
        return DataLoader(self.dataset, batch_size=2, sampler=sampler,
                          collate_fn=partial(collate, samples_per_gpu=2))

    def assert_bundle(self, plain, paired, name):
        outputs, records = paired
        for expected, actual in zip(plain, outputs):
            np.testing.assert_array_equal(expected[0], actual[0])
        self.assertEqual([r["image_id"] for r in records], ["mu", "zeta", "alpha"])
        self.assertNotEqual([r["ori_filename"] for r in records],
                            [r["filename"] for r in self.dataset.data_infos])
        out = self.root / name
        sidecar = save_predictions_with_ids(out, outputs, records, self.metadata)
        saved = mmcv.load(str(out))
        order = json.loads(sidecar.read_text())
        self.assertEqual(order["image_ids"], ["mu", "zeta", "alpha"])
        self.assertEqual(order["training_code_sha"], "0f98a48")
        self.assertNotEqual(order["evaluation_code_sha"], order["training_code_sha"])
        self.assertEqual(order["dataset_size"], 3)
        for i, (detection, record) in enumerate(zip(saved, order["records"])):
            self.assertEqual(record["prediction_index"], i)
            self.assertEqual(record["image_id"], order["image_ids"][i])
            np.testing.assert_array_equal(detection[0], plain[i][0])
            if record["image_id"] == "alpha":
                self.assertEqual(detection[0].shape, (0, 6))
            else:
                self.assertEqual(float(detection[0][0, 0]),
                                 {"mu": 12., "zeta": 10.}[record["image_id"]])
        return out, records

    def test_actual_single_inference_and_saved_ids_match_item_by_item(self):
        model = PredictionFixture()
        plain = evaluation.single_gpu_test(model, self.loader())
        paired = evaluation.single_gpu_test(model, self.loader(), return_image_ids=True)
        self.assert_bundle(plain, paired, "single.pkl")

    def test_actual_cpu_collect_removes_padding_from_predictions_and_ids_together(self):
        torch.distributed.init_process_group(
            "gloo", init_method=f"file://{self.root / 'gloo-init'}", rank=0, world_size=1)
        self.addCleanup(torch.distributed.destroy_process_group)
        model = PredictionFixture()
        plain = evaluation.multi_gpu_test(
            model, self.loader(padded=True), tmpdir=str(self.root / "plain-parts"))
        paired = evaluation.multi_gpu_test(
            model, self.loader(padded=True), tmpdir=str(self.root / "paired-parts"),
            return_image_ids=True)
        self.assertEqual(len(plain), len(self.dataset))
        self.assert_bundle(plain, paired, "collected.pkl")

    def test_existing_predictions_cannot_receive_backfilled_verified_ids(self):
        paired = evaluation.single_gpu_test(
            PredictionFixture(), self.loader(), return_image_ids=True)
        outputs, records = paired
        old = self.root / "legacy.pkl"
        mmcv.dump(outputs, str(old))
        before = old.read_bytes()
        with self.assertRaisesRegex(FileExistsError, "cannot overwrite or backfill"):
            save_predictions_with_ids(old, outputs, records, self.metadata)
        self.assertFalse(Path(str(old) + ".image_ids.json").exists())
        self.assertEqual(before, old.read_bytes())

    def test_missing_image_record_prevents_output_and_training_sha_is_not_inferred(self):
        outputs, records = evaluation.single_gpu_test(
            PredictionFixture(), self.loader(), return_image_ids=True)
        out = self.root / "mismatch.pkl"
        with self.assertRaisesRegex(ValueError, "same dataset"):
            save_predictions_with_ids(out, outputs, records[:-1], self.metadata)
        self.assertFalse(out.exists())
        metadata = {k: v for k, v in self.metadata.items() if k != "training_code_sha"}
        sidecar = save_predictions_with_ids(self.root / "unknown-training.pkl",
                                            outputs, records, metadata)
        order = json.loads(sidecar.read_text())
        self.assertIsNone(order["training_code_sha"])
        self.assertTrue(order["evaluation_code_sha"])


if __name__ == "__main__":
    unittest.main()
