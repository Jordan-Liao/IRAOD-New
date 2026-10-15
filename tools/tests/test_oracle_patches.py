"""CPU-only acceptance tests; all labels and pixels live in temporary directories."""

import csv
import io
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from experiments.comparison.labels import CLASSES
from tools import build_oracle_patches as oracle
from tools import build_rsar_sarclip_patches as legacy


POLY = (10, 12, 20, 10, 22, 20, 12, 22)
DIOR_CLASSES = (
    "airplane", "airport", "baseballfield", "basketballcourt", "bridge",
    "chimney", "expressway-service-area", "expressway-toll-station", "dam",
    "golffield", "groundtrackfield", "harbor", "overpass", "ship", "stadium",
    "storagetank", "tenniscourt", "trainstation", "vehicle", "windmill",
)


class OraclePatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"

    def image(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with Image.new("L", (40, 40), color=128) as image:
            image.save(path)

    def rsar(self, ids=("one",), rows=None):
        ann_root = self.data / "train" / "annfiles"
        ann_root.mkdir(parents=True)
        if rows is None:
            rows = "imagesource: fixture\ngsd: 1\n" + " ".join(map(str, POLY)) + " ship 1\n"
        for image_id in ids:
            (ann_root / f"{image_id}.txt").write_text(rows, encoding="utf-8")
            for domain in oracle.CORRUPTIONS["RSAR"]:
                self.image(self.data / "corruptions" / domain / "train" / "images" / f"{image_id}.png")
        return ann_root

    def xml(self, path, names, coords=POLY):
        path.parent.mkdir(parents=True, exist_ok=True)
        root = ET.Element("annotation")
        for name in names:
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = name
            box = ET.SubElement(obj, "robndbox")
            for key, value in zip(oracle.DIOR_COORDINATES, coords):
                ET.SubElement(box, key).text = str(value)
        ET.ElementTree(root).write(path, encoding="utf-8")

    def dior(self, ids=("one",), names=("Ship",)):
        source = self.data / "ImageSets" / "train.txt"
        source.parent.mkdir(parents=True)
        source.write_text("\n".join(ids) + "\n", encoding="utf-8")
        ann_root = self.data / "Annotations" / "Oriented Bounding Boxes"
        for image_id in ids:
            self.xml(ann_root / f"{image_id}.xml", names)
            for domain in oracle.CORRUPTIONS["DIOR"]:
                self.image(self.data / "Corruption" / f"JPEGImages-{domain}" / f"{image_id}.jpg")
        return ann_root

    def rows(self, out):
        with (out / "metadata.csv").open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            self.assertEqual(reader.fieldnames, ["dataset", *legacy.METADATA_FIELDS])
            return list(reader)

    def test_shared_geometry_and_exact_per_side_expansion(self):
        self.assertIs(oracle.crop_aabb, legacy.crop_aabb)
        self.assertIs(oracle.expanded_aabb, legacy.expanded_aabb)
        self.assertIs(oracle.METADATA_FIELDS, legacy.METADATA_FIELDS)
        poly = np.array(((10, 10), (20, 10), (20, 20), (10, 20)), dtype=np.float32)
        with Image.new("RGB", (40, 40)) as image:
            patch, box = oracle.crop_aabb(image, poly, oracle.EXPAND)
            with patch:
                self.assertEqual(box, (6, 6, 24, 24))
                self.assertEqual(patch.size, (18, 18))
        clipped = oracle.expanded_aabb(poly - 10, oracle.EXPAND, 12, 12)
        self.assertEqual(clipped, (0, 0, 12, 12))

    def test_dataset_class_and_corruption_mapping(self):
        self.assertIs(oracle.CLASSES, CLASSES)
        self.assertEqual(CLASSES["RSAR"], ("ship", "aircraft", "car", "tank", "bridge", "harbor"))
        self.assertEqual(CLASSES["DIOR"], DIOR_CLASSES)
        self.assertEqual(oracle.CORRUPTIONS["RSAR"], (
            "chaff", "gaussian_white_noise", "point_target", "noise_suppression",
            "am_noise_horizontal", "smart_suppression", "am_noise_vertical",
        ))
        self.assertEqual(oracle.CORRUPTIONS["DIOR"], ("brightness", "cloudy", "contrast"))

    def test_rsar_all_classes_difficulty_and_legacy_naming(self):
        text = "imagesource: fixture\ngsd: 1\n" + "\n".join(
            " ".join(map(str, POLY)) + f" {name} {index % 2}"
            for index, name in enumerate(CLASSES["RSAR"])
        )
        self.rsar(rows=text)
        out = self.root / "rsar"
        with mock.patch.object(oracle, "crop_aabb", wraps=legacy.crop_aabb) as crop:
            summary = oracle.build("RSAR", self.data, out)
        self.assertEqual(crop.call_count, 42)
        rows = self.rows(out)
        self.assertEqual(len(rows), 42)
        self.assertEqual(summary["valid_crop_count"], 42)
        self.assertEqual(summary["rejections"]["total"], 0)
        for row in rows:
            self.assertEqual(row["dataset"], "RSAR")
            self.assertEqual(row["split"], "train")
            self.assertEqual(row["crop_mode"], "aabb")
            self.assertEqual(float(row["crop_expand"]), 0.4)
            self.assertEqual(int(row["class_id"]), CLASSES["RSAR"].index(row["class_name"]))
            path = Path(row["patch_path"])
            self.assertEqual(path.relative_to(out).parts[:2], ("RSAR", "aabb"))
            with Image.open(path) as patch:
                self.assertEqual(patch.mode, "RGB")
                self.assertEqual(patch.size, (22, 22))
        self.assertEqual(Path(rows[0]["patch_path"]).name, "chaff_train_one_2_e0.4.png")
        self.assertEqual(summary["per_class"], dict.fromkeys(CLASSES["RSAR"], 7))
        self.assertFalse((out / "metadata.csv.partial").exists())
        self.assertEqual(json.loads((out / "summary.json").read_text()), summary)

    def test_dior_parser_all_twenty_classes_and_original_object_order(self):
        ann_root = self.dior(names=[name.upper() for name in DIOR_CLASSES])
        objects = list(oracle.read_dior_objects(ann_root / "one.xml"))
        self.assertEqual([name for _, _, name in objects], list(DIOR_CLASSES))
        self.assertEqual([index for index, _, _ in objects], list(range(20)))
        for _, poly, _ in objects:
            np.testing.assert_array_equal(poly.reshape(-1), POLY)
        out = self.root / "dior"
        summary = oracle.build("DIOR", self.data, out)
        rows = self.rows(out)
        self.assertEqual(len(rows), 60)
        self.assertEqual(summary["per_class"], dict.fromkeys(DIOR_CLASSES, 3))
        self.assertEqual([int(row["class_id"]) for row in rows[:20]], list(range(20)))
        for row in rows:
            self.assertEqual(row["dataset"], "DIOR")
            self.assertEqual(int(row["class_id"]), DIOR_CLASSES.index(row["class_name"]))
            coords = [float(row[f"poly_{axis}{i}"]) for i in range(1, 5) for axis in ("x", "y")]
            self.assertEqual(coords, list(POLY))

    def test_dior_reads_only_explicit_train_despite_other_split_files(self):
        ann_root = self.dior(ids=("train_b", "train_a"))
        for split in ("val", "test"):
            (self.data / "ImageSets" / f"{split}.txt").write_text(split + "\n")
            (ann_root / f"{split}.xml").write_text("invalid XML must never be read")
            for domain in oracle.CORRUPTIONS["DIOR"]:
                path = self.data / "Corruption" / f"JPEGImages-{domain}" / f"{split}.jpg"
                path.write_bytes(b"invalid pixels must never be read")
        out = self.root / "selected"
        summary = oracle.build("DIOR", self.data, out)
        self.assertEqual(summary["source_train"]["ids"], ["train_b", "train_a"])
        self.assertEqual(summary["valid_crop_count"], 6)
        self.assertEqual({row["image_name"] for row in self.rows(out)}, {"train_b.jpg", "train_a.jpg"})

    def test_rsar_never_reads_val_or_test_annotations(self):
        self.rsar()
        for split in ("val", "test"):
            path = self.data / split / "annfiles"
            path.mkdir(parents=True)
            (path / "unselected.txt").write_text("invalid")
        self.assertEqual(oracle.build("RSAR", self.data, self.root / "out")["valid_crop_count"], 7)

    def test_streamed_rows_count_only_and_tail_budget_agree(self):
        # 22 objects * 3 corruptions = 66; the tail batch must count.
        ann_root = self.dior(names=("ship",) * 22)
        tree = ET.parse(ann_root / "one.xml")
        obj = ET.SubElement(tree.getroot(), "object")
        ET.SubElement(obj, "name").text = "ship"
        box = ET.SubElement(obj, "robndbox")
        for key, value in zip(oracle.DIOR_COORDINATES, (100, 100, 110, 100, 110, 110, 100, 110)):
            ET.SubElement(box, key).text = str(value)
        tree.write(ann_root / "one.xml")
        real_writer = csv.DictWriter
        real_save = Image.Image.save
        written = []
        saved = []

        def save(image, path, *args, **kwargs):
            # A row for the prior crop must already have been streamed before the next save.
            self.assertEqual(len(written), len(saved))
            result = real_save(image, path, *args, **kwargs)
            saved.append(path)
            return result

        def writer(*args, **kwargs):
            instance = real_writer(*args, **kwargs)
            original = instance.writerow

            def row(value):
                if value["dataset"] != "dataset":  # Ignore DictWriter.writeheader().
                    self.assertTrue(Path(value["patch_path"]).is_file())
                    self.assertTrue((self.root / "full" / "metadata.csv.partial").exists())
                    self.assertFalse((self.root / "full" / "metadata.csv").exists())
                    written.append(value["patch_path"])
                return original(value)

            instance.writerow = row
            return instance

        with mock.patch.object(oracle.csv, "DictWriter", side_effect=writer), \
                mock.patch.object(Image.Image, "save", autospec=True, side_effect=save):
            full = oracle.build("DIOR", self.data, self.root / "full")
        with mock.patch.object(Image.Image, "load", side_effect=AssertionError("pixel decode")), \
                mock.patch.object(Image.Image, "convert", side_effect=AssertionError("conversion")), \
                mock.patch.object(oracle, "crop_aabb", side_effect=AssertionError("crop")), \
                mock.patch.object(Image.Image, "save", side_effect=AssertionError("save")):
            count = oracle.build("DIOR", self.data, self.root / "count", count_only=True)
        for key in ("valid_crop_count", "per_class", "per_corruption", "per_corruption_class",
                    "rejections", "canonical_budget", "source_train"):
            self.assertEqual(full[key], count[key])
        self.assertEqual(count["valid_crop_count"], len(saved))
        self.assertEqual(count["valid_crop_count"], 66)
        self.assertEqual(count["rejections"]["total"], 3)
        self.assertEqual(count["rejections"]["per_class"]["ship"], 3)
        self.assertEqual(count["rejections"]["per_corruption"], dict.fromkeys(oracle.CORRUPTIONS["DIOR"], 1))
        self.assertEqual(count["canonical_budget"], {
            "epochs": 10, "batch_size": 64, "updates": 20,
            "sampler": "WeightedRandomSampler", "draws_per_epoch": 66,
            "replacement": True, "drop_last": False,
        })
        self.assertEqual(count["status"], "count_only_no_patch_artifacts")
        self.assertEqual(list((self.root / "count").iterdir()), [self.root / "count" / "summary.json"])
        self.assertEqual(len(self.rows(self.root / "full")), len(saved))

    def test_count_only_uses_legacy_degenerate_crop_semantics(self):
        self.rsar(rows="10 10 10 10 10 10 10 10 ship 0\n")
        summary = oracle.build("RSAR", self.data, self.root / "count", count_only=True)
        # The shared recipe uses a minimum box dimension of 1 before expansion.
        self.assertEqual(summary["valid_crop_count"], 7)
        self.assertEqual(summary["rejections"]["total"], 0)

    def test_missing_train_images_reports_exact_counts_without_success_artifacts(self):
        self.dior(ids=("one", "two"))
        for domain in oracle.CORRUPTIONS["DIOR"]:
            image_root = self.data / "Corruption" / f"JPEGImages-{domain}"
            for image_id in ("one", "two"):
                (image_root / f"{image_id}.jpg").unlink()
            # Clean TRAIN and unrelated split images must not be used as substitutes.
            self.image(self.data / "JPEGImages" / "one.jpg")
            self.image(image_root / "val.jpg")
        for count_only in (False, True):
            out = self.root / f"missing-{count_only}"
            with self.assertRaises(FileNotFoundError) as raised:
                oracle.build("DIOR", self.data, out, count_only=count_only)
            message = str(raised.exception)
            for domain in oracle.CORRUPTIONS["DIOR"]:
                self.assertIn(f"{domain}: missing 2/2 required TRAIN images", message)
            self.assertEqual(message.count("examples: one, two"), 3)
            self.assertFalse(out.exists())

    def test_missing_rsar_image_and_default_root_fail_explicitly(self):
        self.rsar()
        image = self.data / "corruptions" / "chaff" / "train" / "images" / "one.png"
        image.unlink()
        image.parent.rmdir()
        out = self.root / "missing"
        with self.assertRaisesRegex(FileNotFoundError, "chaff: missing 1/1 required TRAIN images"):
            oracle.build("RSAR", self.data, out)
        self.assertFalse(out.exists())

    def test_image_root_mapping_is_exact_and_used(self):
        self.dior()
        roots = {}
        for domain in oracle.CORRUPTIONS["DIOR"]:
            alternate = self.root / "known-train" / domain
            alternate.mkdir(parents=True)
            original = self.data / "Corruption" / f"JPEGImages-{domain}" / "one.jpg"
            original.rename(alternate / "one.jpg")
            roots[domain] = str(alternate)
        for mapping in ({}, {**roots, "clean": str(self.root)}, {**roots, "cloudy": str(self.root / "missing")}):
            with self.subTest(mapping=mapping), self.assertRaises((ValueError, FileNotFoundError)):
                oracle.build("DIOR", self.data, self.root / "invalid", image_roots=mapping)
        with redirect_stdout(io.StringIO()):
            oracle.main(["--dataset", "DIOR", "--data-root", str(self.data),
                         "--out", str(self.root / "alternate"), "--image-roots", json.dumps(roots),
                         "--count-only"])
        summary = json.loads((self.root / "alternate" / "summary.json").read_text())
        self.assertEqual(summary["source_train"]["image_roots"], roots)
        self.assertEqual(summary["valid_crop_count"], 3)

    def test_duplicate_train_ids_and_missing_train_annotation_fail(self):
        self.dior()
        source = self.data / "ImageSets" / "train.txt"
        source.write_text("one\none\n")
        with self.assertRaisesRegex(ValueError, "duplicate TRAIN ID 'one'"):
            oracle.build("DIOR", self.data, self.root / "duplicate")
        source.write_text("one\nmissing\n")
        with self.assertRaisesRegex(FileNotFoundError, "Missing TRAIN annotation: .*missing.xml"):
            oracle.build("DIOR", self.data, self.root / "missing")

    def test_malformed_rsar_rows_fail_with_file_and_line(self):
        ann_root = self.rsar()
        path = ann_root / "one.txt"
        good = " ".join(map(str, POLY))
        for row in ("broken", good + " unknown 0", good + " ship broken",
                    "nan 1 2 3 4 5 6 7 ship 0", "x 1 2 3 4 5 6 7 ship 0",
                    good + " ship 0 unexpected"):
            with self.subTest(row=row):
                path.write_text("imagesource: fixture\n" + row + "\n")
                with self.assertRaisesRegex(ValueError, r"one.txt:2:"):
                    list(oracle.read_rsar_objects(path))

    def test_malformed_dior_annotations_fail_not_hbb_substituted(self):
        ann_root = self.dior()
        path = ann_root / "one.xml"
        cases = (
            "<annotation><object><name>unknown</name></object></annotation>",
            "<annotation><object><name>ship</name><bndbox/></object></annotation>",
            "<annotation><object><name>ship</name><robndbox/></object></annotation>",
            "<wrong/>",
            "<annotation>",
        )
        for text in cases:
            with self.subTest(text=text):
                path.write_text(text)
                with self.assertRaisesRegex(ValueError, r"one.xml:"):
                    list(oracle.read_dior_objects(path))
        self.xml(path, ("ship",), coords=("nan", *POLY[1:]))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            list(oracle.read_dior_objects(path))

    def test_annotation_failure_leaves_only_partial_metadata(self):
        ann_root = self.rsar(ids=("a", "b"))
        (ann_root / "b.txt").write_text("malformed\n")
        for count_only in (False, True):
            out = self.root / f"failed-{count_only}"
            with self.assertRaisesRegex(ValueError, r"b.txt:1:"):
                oracle.build("RSAR", self.data, out, count_only=count_only)
            self.assertFalse((out / "metadata.csv").exists())
            self.assertFalse((out / "summary.json").exists())
            if count_only:
                self.assertEqual(list(out.iterdir()), [])
            else:
                self.assertTrue((out / "metadata.csv.partial").exists())
                self.assertEqual(len(list(out.rglob("*.png"))), 7)

    def test_save_failure_never_publishes_success(self):
        self.rsar()
        out = self.root / "failed-save"
        with mock.patch.object(Image.Image, "save", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                oracle.build("RSAR", self.data, out)
        self.assertFalse((out / "metadata.csv").exists())
        self.assertFalse((out / "summary.json").exists())
        self.assertTrue((out / "metadata.csv.partial").exists())

    def test_existing_output_is_not_overwritten(self):
        self.rsar()
        out = self.root / "existing"
        out.mkdir()
        marker = out / "marker"
        marker.write_text("keep")
        with self.assertRaises(FileExistsError):
            oracle.build("RSAR", self.data, out)
        self.assertEqual(marker.read_text(), "keep")
        self.assertEqual(list(out.iterdir()), [marker])

    def test_cli_rejects_split_and_recipe_overrides(self):
        for extra in (["--split", "test"], ["--crop-expand", "0.2"], ["--max-per-class", "1"]):
            with self.subTest(extra=extra), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    oracle.main(["--dataset", "RSAR", "--data-root", str(self.data),
                                 "--out", str(self.root / "out"), *extra])
                self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
