"""Native CPU dataflow seams; run in the existing mmrotate environment, no GPU."""

import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn


NATIVE = importlib.util.find_spec("mmdet") is not None and importlib.util.find_spec("mmrotate") is not None
if NATIVE:
    from mmcv import Config, ConfigDict
    from sfod.extensions.aasfod import AASFODOBB, AASFODTargetDataset, AASFODTeacherHook
    from sfod.extensions.aasfod_mechanisms import TargetDiscriminators


@unittest.skipUnless(NATIVE, "requires the existing native mmdet/mmrotate CPU environment")
class AASFODNativeTest(unittest.TestCase):
    def setUp(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)

    def test_native_configs_register_and_keep_source_detector(self):
        root = Path(__file__).resolve().parents[2]
        for dataset, count in (("dior", 20), ("rsar", 6)):
            config = Config.fromfile(str(root / "configs/unbiased_teacher/sfod/extensions"
                                        / f"aasfod_{dataset}.py"))
            self.assertEqual(config.model.type, "AASFODOBB")
            self.assertEqual(config.model.roi_head.bbox_head.num_classes, count)
            self.assertEqual(config.model.backbone.frozen_stages, 1)
            self.assertEqual(config.model.roi_head.bbox_head.type, "RotatedShared2FCBBoxHead")
            self.assertTrue(config.model.cfg.use_bbox_reg)
            self.assertEqual(config.custom_hooks, [dict(type="AASFODTeacherHook", priority=45)])
            self.assertEqual(config.runner.type, "SemiIterBasedRunner")

    def test_target_dataset_consumes_split_no_annotations_and_fns_uses_all(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = [f"{i}.png" for i in range(10)]
            for name in names:
                (root / name).touch()  # discovery only; callable pipeline doesn't read pixels
            split = root / "tsd.json"
            split.write_text(json.dumps(dict(similar=names[8:], dissimilar=names[:8])))

            def share(data):
                self.assertEqual(data["gt_bboxes"].shape, (0, 5))
                self.assertEqual(data["gt_labels"].shape, (0,))
                return dict(img=data["img_info"]["filename"], img_metas={},
                            gt_bboxes=data["gt_bboxes"], gt_labels=data["gt_labels"])

            args = dict(img_prefix=directory, pipeline_share=[share], pipeline_weak=[],
                        pipeline_strong=[], tsd_split=str(split), unlabeled_epoch_size=10)
            data = AASFODTargetDataset(stage="alignment", **args)
            for index in range(10):
                batch = data[index]
                self.assertIn(batch["img"], names[8:])
                self.assertIn(batch["img_dissimilar"], names[:8])
                self.assertEqual(batch["img_unlabeled_1"], batch["img"])
            data = AASFODTargetDataset(stage="fns", **args)
            self.assertEqual({data[index]["img"] for index in range(10)}, set(names))

    def test_forward_alignment_and_fns_reach_real_losses_with_pre_mosaic_teacher(self):
        class RPN(nn.Module):
            def simple_test_rpn(self, features, metas):
                return [features[0].new_tensor([[16., 16., 10., 8., .3, .9]]) for _ in metas]

            def forward_train(self, features, metas, boxes, **kwargs):
                self.detection_batches.append(len(metas))
                value = features[0].square().mean()
                return dict(loss_rpn_cls=value, loss_rpn_bbox=value), self.simple_test_rpn(features, metas)

        class ROI(nn.Module):
            def simple_test(self, features, proposals, metas, rescale):
                self.teacher_batches.append(len(metas))
                return [None] * len(metas)

            def forward_train(self, features, metas, proposals, boxes, labels, **kwargs):
                self.last_targets = boxes
                value = features[0].square().mean()
                return dict(loss_cls=value, loss_bbox=value)

        class Backbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 4, 1)
                self.calls = 0

            def forward(self, images):
                self.calls += 1
                value = self.conv(images)
                return value, torch.cat([value, value], dim=1)

        class Teacher(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone, self.rpn_head, self.roi_head = Backbone(), RPN(), ROI()
                self.roi_head.teacher_batches = []

            def extract_feat(self, images):
                return self.backbone(images)

        class Port(AASFODOBB):
            def __init__(self, stage):
                nn.Module.__init__(self)
                self.aasfod_stage = stage
                self.backbone, self.rpn_head, self.roi_head = Backbone(), RPN(), ROI()
                self.domain_heads = TargetDiscriminators(4, 8)
                self.rpn_head.detection_batches = []
                self.ema_model = Teacher().requires_grad_(False).eval()
                self.cur_iter = self.image_num = 0
                self.train_cfg = ConfigDict(rpn_proposal={})
                self.test_cfg = ConfigDict(rpn={})

            def create_pseudo_results(self, images, detections, transforms, device):
                return ([images.new_tensor([[16., 16., 16., 8., .4]]) for _ in detections],
                        [torch.tensor([1], device=device) for _ in detections])

            def extract_feat(self, images):
                return self.backbone(images)

        weak, strong = torch.rand(8, 3, 32, 32), torch.rand(8, 3, 32, 32)
        metas = [dict(img_shape=(32, 32, 3), scale_factor=1.) for _ in range(8)]
        empty_boxes, empty_labels = [torch.zeros(0, 5)] * 8, [torch.zeros(0, dtype=torch.long)] * 8
        for stage in ("alignment", "fns"):
            model = Port(stage)
            extras = dict(img_dissimilar=torch.rand_like(weak)) if stage == "alignment" else {}
            losses = model(
                weak, metas, gt_bboxes=empty_boxes, gt_labels=empty_labels,
                img_unlabeled_1=strong, img_metas_unlabeled_1=metas,
                gt_bboxes_unlabeled_1=empty_boxes, gt_labels_unlabeled_1=empty_labels, **extras)
            expected = {"loss_rpn_cls", "loss_rpn_bbox", "loss_cls", "loss_bbox"}
            if stage == "alignment":
                expected |= {"loss_aasfod_local", "loss_aasfod_global"}
            self.assertEqual(set(losses), expected)
            self.assertEqual(model.ema_model.roi_head.teacher_batches, [8])
            self.assertEqual(model.rpn_head.detection_batches, [8 if stage == "alignment" else 2])
            self.assertEqual(model.backbone.calls, 2 if stage == "alignment" else 1)
            sum(losses.values()).backward()
            self.assertTrue(all(p.grad is None for p in model.ema_model.parameters()))
            if stage == "fns":
                self.assertTrue(all(p.grad is None for p in model.domain_heads.parameters()))
            with self.assertRaisesRegex(ValueError, "ground truth"):
                model(weak, metas, gt_bboxes=[torch.ones(1, 5)] * 8, gt_labels=empty_labels,
                      img_unlabeled_1=strong, img_metas_unlabeled_1=metas,
                      gt_bboxes_unlabeled_1=empty_boxes, gt_labels_unlabeled_1=empty_labels, **extras)

    def test_hook_cadence_occurs_after_optimizer_not_step_zero(self):
        class Port(nn.Linear):
            def __init__(self):
                super().__init__(2, 2)
                self.ema_model = nn.Linear(2, 2)
                self.ema_interval = 2

            def set_epoch(self, epoch):
                self.epoch = epoch

        model = Port()
        # Teacher normally is unregistered on SemiBaseDetector; remove it here too.
        teacher = model._modules.pop("ema_model")
        object.__setattr__(model, "ema_model", teacher)
        runner = SimpleNamespace(model=model, iter=0)
        hook = AASFODTeacherHook()
        hook.before_run(runner)
        initial = teacher.weight.clone()
        with torch.no_grad():
            model.weight.add_(1)
        hook.after_train_iter(runner)
        torch.testing.assert_close(teacher.weight, initial)
        runner.iter = 1
        hook.after_train_iter(runner)
        torch.testing.assert_close(teacher.weight, initial + .01)


if __name__ == "__main__":
    unittest.main()
