"""Real CPU image/Collect/TSD chain, without importing detector CUDA extensions.

The installed pipeline files and repository dataset/mechanism files execute
unchanged. Only unused native-op/annotation imports fail closed. The detector
is a tiny CPU fixture; this is not full-detector or GPU execution evidence.
"""

from contextlib import contextmanager
import copy
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import mmcv
from mmcv import Config
from mmcv.parallel import scatter
from mmcv.utils import Registry
import numpy as np
import torch
from torch import nn

from experiments.comparison import aasfod_tsd
from experiments.comparison.aasfod_protocol import validate_split


ROOT = Path(__file__).resolve().parents[2]


def forbidden(*args, **kwargs):
    raise AssertionError("Native ops, annotations and augmentation are outside this CPU chain")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def module_scope(modules):
    # Restore only our import namespaces, not lazily loaded torch registrations.
    prefixes = tuple(name + "." for name in modules)

    def owned(name):
        return name in modules or name.startswith(prefixes)

    original = {name: module for name, module in sys.modules.items() if owned(name)}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name in list(sys.modules):
            if owned(name):
                del sys.modules[name]
        sys.modules.update(original)


@contextmanager
def cpu_pipeline_modules():
    """Minimal registries, real complete pipeline modules; no fake Collect."""
    mmdet = Path(importlib.util.find_spec("mmdet").origin).parent
    mmrotate = Path(importlib.util.find_spec("mmrotate").origin).parent
    modules = {}

    def package(name, **attributes):
        module = ModuleType(name)
        module.__path__ = []
        module.__dict__.update(attributes)
        modules[name] = module
        return module

    pipelines = Registry("pipeline")
    datasets = Registry("dataset")
    package("mmdet")
    package("mmdet.datasets")
    package("mmdet.datasets.builder", PIPELINES=pipelines)
    package("mmdet.datasets.pipelines")
    package("mmdet.core", BitmapMasks=forbidden, PolygonMasks=forbidden,
            find_inside_bboxes=forbidden)
    package("mmdet.core.evaluation")
    package("mmdet.core.evaluation.bbox_overlaps", bbox_overlaps=forbidden)
    package("mmcv.ops", box_iou_rotated=forbidden)
    package("mmrotate")
    package("mmrotate.core", norm_angle=forbidden, obb2poly_np=forbidden,
            poly2obb_np=forbidden)

    class AnnotationDataset:
        __init__ = forbidden
        load_annotations = forbidden

    package("mmrotate.datasets", DOTADataset=AnnotationDataset)
    package("mmrotate.datasets.builder", ROTATED_DATASETS=datasets,
            ROTATED_PIPELINES=pipelines)
    package("mmrotate.datasets.pipelines")
    package("sfod")
    package("sfod.extensions")
    with module_scope(modules):
        load_module("mmdet.utils", mmdet / "utils/logger.py")
        for name in ("loading", "transforms", "formatting"):
            load_module(f"mmdet.datasets.pipelines.{name}",
                        mmdet / f"datasets/pipelines/{name}.py")
        load_module("mmrotate.datasets.pipelines.transforms",
                    mmrotate / "datasets/pipelines/transforms.py")
        data = load_module("sfod.semi_dota_dataset", ROOT / "sfod/semi_dota_dataset.py")
        mechanisms = load_module("sfod.extensions.aasfod_mechanisms",
                                 ROOT / "sfod/extensions/aasfod_mechanisms.py")
        yield data, mechanisms


class TinyROI(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.bbox_head = nn.Module()
        self.bbox_head.shared_fcs = nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 8)])
        self.cls = nn.Linear(8, classes + 1)
        self.box = nn.Linear(8, 5)
        self.outputs, self.rois = [], []

    def _bbox_forward(self, features, rois):
        self.rois.append(rois)
        for fc in self.bbox_head.shared_fcs:
            features = fc(features).relu()
        output = dict(cls_score=self.cls(features), bbox_pred=self.box(features))
        self.outputs.append(output)
        return output


class TinyDetector(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.bn = nn.BatchNorm1d(8)
        self.roi_head = TinyROI(classes)
        self.rpn_head = SimpleNamespace(simple_test_rpn=self.rpn)
        self.images, self.metas = [], []
        self.cuda_calls = 0

    def cuda(self):
        self.cuda_calls += 1
        return self

    def extract_feat(self, images):
        if self.training or any(p.requires_grad for p in self.parameters()):
            raise AssertionError("TSD source must remain frozen/eval")
        self.images.append(images.clone())
        features = images.mean((-2, -1)).repeat(1, 3)[:, :8]
        return self.bn(features)

    def rpn(self, features, metas):
        self.metas.extend(metas)
        return [features.new_tensor([[16., 8., 10., 4., .2, .9]])]


def cpu_seed(seed, deterministic):
    assert deterministic
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class AASFODTSDPipelineTest(unittest.TestCase):
    def setUp(self):
        cwd, path = os.getcwd(), sys.path[:]
        states = random.getstate(), np.random.get_state(), torch.get_rng_state()
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(os.chdir, cwd)
        self.addCleanup(sys.path.__setitem__, slice(None), path)
        self.addCleanup(random.setstate, states[0])
        self.addCleanup(np.random.set_state, states[1])
        self.addCleanup(torch.set_rng_state, states[2])
        self.addCleanup(torch.set_num_threads, threads)
        self.native, self.mechanisms = self.enterContext(cpu_pipeline_modules())
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def test_run_real_weak_collect_and_twenty_roi_passes_both_families(self):
        for family, classes in (("rsar", 6), ("dior", 20)):
            with self.subTest(family=family):
                self.check_family(family, classes)

    def check_family(self, family, classes):
        config_path = ROOT / f"configs/unbiased_teacher/sfod/extensions/aasfod_{family}.py"
        fromfile = Config.fromfile
        config = fromfile(str(config_path), import_custom_modules=False)
        original_data = copy.deepcopy(config.data.train)
        original_data_repr = repr(config.data.train)
        images = self.directory / family
        images.mkdir()
        pixels = {}
        for i in (3, 1, 4, 0, 2):
            image = np.zeros((16, 32, 3), dtype=np.uint8)
            image[:, :11] = [12 + i, 61 + i, 190 + i]
            image[3:12, 17:] = [211 - i, 23 + i, 47 + i]
            name = f"target-{i}.png"
            self.assertTrue(mmcv.imwrite(image, str(images / name)))
            pixels[name] = image
        checkpoint = self.directory / f"{family}-source.pth"
        torch.manual_seed(7)
        source = TinyDetector(classes)
        torch.save(dict(state_dict=source.state_dict()), checkpoint)
        detector = TinyDetector(classes)
        cell = dict(dataset=family.upper(), domain="clean", seed=42,
                    training_code=str(ROOT), config=str(config_path),
                    tsd_split=str(self.directory / f"{family}-tsd.json"),
                    source_checkpoint=str(checkpoint), target_val=str(images),
                    unlabeled_epoch_size=5)
        targets = []

        def build_target(**kwargs):
            self.assertNotIn("ann_file", kwargs)
            self.assertEqual(kwargs["pipeline_weak"], original_data.pipeline_weak)
            self.assertEqual(kwargs["pipeline_strong"], [])
            self.assertEqual(kwargs["classes"], original_data.classes)
            target = dataset_class(**kwargs)
            targets.append(target)
            return target

        def read_config(path):
            if Path(path) == config_path:
                return config
            return fromfile(str(ROOT / path), import_custom_modules=False)

        def cpu_scatter(batch, devices):
            self.assertEqual(devices, [0])
            self.assertEqual(set(batch), {"img", "img_metas"})
            return scatter(batch, [-1])

        dataset_class = self.native.StrictSourceFreeDOTADataset
        core = sys.modules["mmrotate.core"]
        core.rbbox2roi = lambda boxes: torch.cat([
            torch.cat((box.new_full((len(box), 1), i), box), dim=1)
            for i, box in enumerate(boxes)])
        apis, models = ModuleType("mmdet.apis"), ModuleType("mmrotate.models")
        apis.set_random_seed = Mock(side_effect=cpu_seed)
        models.build_detector = Mock(return_value=detector)
        reads = []
        get = mmcv.FileClient.get

        def read_image(client, path):
            self.assertIn(Path(path).name, pixels)
            self.assertEqual(Path(path).parent, images)
            reads.append(Path(path).name)
            return get(client, path)

        with module_scope({"mmdet.apis": apis, "mmrotate.models": models}), \
                patch.dict(os.environ, {"IRAOD_GPU_LOCKED": "1"}), \
                patch.object(aasfod_tsd, "load_cell", return_value=(
                    dict(training_code=str(ROOT)), cell)) as load_cell, \
                patch.object(Config, "fromfile", side_effect=read_config), \
                patch.object(self.native, "StrictSourceFreeDOTADataset", side_effect=build_target), \
                patch("mmcv.parallel.scatter", side_effect=cpu_scatter), \
                patch.object(mmcv.FileClient, "get", read_image), \
                patch.object(mmcv, "imflip", side_effect=forbidden), \
                patch.object(np.random, "choice", side_effect=forbidden):
            output = aasfod_tsd.run("unchanged-queue", family.upper(), "clean", 42)
            target = targets[0]
            self.assertEqual(target.indices, random.Random(42).sample(range(5), 5))
            names = [target.filenames[i] for i in target.indices]
            self.assertEqual(reads, names)
            self.assertEqual([Path(meta["filename"]).name for meta in detector.metas], names)
            for index, name in enumerate(names):
                state = torch.get_rng_state(), np.random.get_state(), random.getstate()
                first, second = target[index], target[index]
                torch.testing.assert_close(torch.get_rng_state(), state[0], rtol=0, atol=0)
                np.testing.assert_equal(np.random.get_state(), state[1])
                self.assertEqual(random.getstate(), state[2])
                torch.testing.assert_close(first["img"].data, second["img"].data, rtol=0, atol=0)
                np.testing.assert_equal(first["img_metas"].data, second["img_metas"].data)
                self.assertEqual(first["gt_bboxes"].data.shape, (0, 5))
                self.assertEqual(first["gt_labels"].data.shape, (0,))
                meta = first["img_metas"].data
                self.assertEqual(set(meta), set(original_data.pipeline_weak[-1]["meta_keys"]))
                self.assertIs(meta["flip"], False)
                self.assertIsNone(meta["flip_direction"])
                self.assertEqual(meta["ori_shape"], (16, 32, 3))
                self.assertEqual(meta["img_shape"], (400, 800, 3))
                self.assertEqual(meta["pad_shape"], (416, 800, 3))
                np.testing.assert_equal(meta["scale_factor"], np.full(4, 25.))
                self.assertEqual(meta["ori_filename"], name)
                expected = mmcv.imrescale(pixels[name], (800, 800))
                expected = mmcv.imnormalize(expected, **meta["img_norm_cfg"])
                expected = mmcv.impad_to_multiple(expected, 32)
                expected = torch.from_numpy(expected.transpose(2, 0, 1).copy())
                self.assertTrue(torch.isfinite(first["img"].data).all())
                torch.testing.assert_close(first["img"].data, expected, rtol=0, atol=0)
                torch.testing.assert_close(detector.images[index][0], expected, rtol=0, atol=0)

        self.assertEqual(repr(config.data.train), original_data_repr)
        load_cell.assert_called_once_with("unchanged-queue", family.upper(), "clean", 42, "AASFOD")
        apis.set_random_seed.assert_called_once_with(42, deterministic=True)
        self.assertEqual(models.build_detector.call_args.args[0].roi_head.bbox_head.num_classes,
                         classes)
        self.assertEqual(detector.cuda_calls, 1)
        for key, value in detector.state_dict().items():
            torch.testing.assert_close(value, source.state_dict()[key], rtol=0, atol=0)
        self.assertEqual(len(detector.images), 5)
        self.assertEqual(len(detector.roi_head.outputs), 100)
        self.assertTrue(all(not fc._forward_hooks for fc in detector.roi_head.bbox_head.shared_fcs))
        result = json.loads(output.read_text())
        validate_split(result, cell)
        self.assertEqual(result["images"], 5)
        self.assertEqual(result["stochastic_roi_passes"], 100)
        self.assertEqual(list(result["scores"]), names)
        for index, name in enumerate(names):
            start = index * 20
            outputs = detector.roi_head.outputs[start:start + 20]
            probabilities = torch.stack([row["cls_score"].softmax(-1) for row in outputs])
            deltas = torch.stack([row["bbox_pred"] for row in outputs])
            self.assertEqual(probabilities.shape, (20, 1, classes + 1))
            self.assertEqual(deltas.shape, (20, 1, 5))
            expected = (probabilities.var(0, unbiased=False).sum(-1)
                        * deltas.var(0, unbiased=False).sum(-1)).sum()
            self.assertEqual(result["scores"][name], float(expected))
            rois = detector.roi_head.rois[start:start + 20]
            self.assertTrue(all(roi is rois[0] for roi in rois))


if __name__ == "__main__":
    unittest.main()
