"""AASFOD target-to-target alignment followed by detection-only OBB FNS."""

import json
import random

import torch
from torch.nn import functional as F
from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS, Hook
from mmdet.models.builder import DETECTORS
from mmrotate.datasets.builder import ROTATED_DATASETS

from sfod.semi_dota_dataset import StrictSourceFreeDOTADataset
from .aasfod_mechanisms import TargetDiscriminators, copy_detector, ema_detector
from .aasfod_mosaic import four_image_mosaic
from .proposal_teacher import ProposalAlignedTeacher, view_scale


@ROTATED_DATASETS.register_module()
class AASFODTargetDataset(StrictSourceFreeDOTADataset):
    """Read only target images; paired A/B in alignment, all targets in FNS."""

    def __init__(self, *args, tsd_split, stage, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage = stage
        with open(tsd_split) as stream:
            split = json.load(stream)
        selected = {self.filenames[index]: position for position, index in enumerate(self.indices)}
        self.similar = [selected[name] for name in split["similar"]]
        self.dissimilar = [selected[name] for name in split["dissimilar"]]
        if set(selected) != set(split["similar"] + split["dissimilar"]):
            raise ValueError("AASFOD data must use exactly the prepared TSD image set")

    def __getitem__(self, index):
        if self.stage == "fns":
            return super().__getitem__(index)
        sample = super().__getitem__(random.choice(self.similar))
        other = super().__getitem__(random.choice(self.dissimilar))
        # Author target-dissimilar branch uses the weak view without detector loss.
        sample["img_dissimilar"] = other["img"]
        sample["img_metas_dissimilar"] = other["img_metas"]
        return sample


@DETECTORS.register_module()
class AASFODOBB(ProposalAlignedTeacher):
    def __init__(self, *args, cfg, **kwargs):
        super().__init__(*args, cfg=cfg, **kwargs)
        self.aasfod_stage = cfg["aasfod_stage"]
        self.ema_interval = cfg["aasfod_ema_interval"]
        if self.aasfod_stage == "alignment":
            self.domain_heads = TargetDiscriminators(256, 2048)
        elif self.aasfod_stage != "fns":
            raise ValueError("AASFOD stage must be alignment or fns")

    def forward(self, img, img_metas, return_loss=True, **kwargs):
        if not return_loss:
            return super().forward(img, img_metas, return_loss=False, **kwargs)
        dissimilar = kwargs.pop("img_dissimilar", None)
        kwargs.pop("img_metas_dissimilar", None)
        self._assert_strict_empty_targets(**{
            key: kwargs[key] for key in (
                "gt_bboxes", "gt_labels", "gt_bboxes_unlabeled_1", "gt_labels_unlabeled_1")})
        return self.target_losses(
            img, img_metas, kwargs["img_unlabeled_1"],
            kwargs["img_metas_unlabeled_1"], dissimilar)

    @torch.no_grad()
    def teacher_labels(self, weak, weak_metas, strong, strong_metas):
        teacher = getattr(self.ema_model, "module", self.ema_model)
        teacher.eval()
        features = teacher.extract_feat(weak)
        proposals = teacher.rpn_head.simple_test_rpn(features, weak_metas)
        detections = teacher.roi_head.simple_test(features, proposals, weak_metas, rescale=True)
        boxes, labels = self.create_pseudo_results(strong, detections, [], weak.device)
        # Shared flips are already reflected by ROI simple_test; rescale=True
        # removes only resize. Both views use the same pre-view geometric flip.
        for target, meta in zip(boxes, strong_metas):
            target[:, :4] *= view_scale(meta, target)
        return boxes, labels

    def detection_losses(self, features, metas, boxes, labels):
        proposal_cfg = self.train_cfg.get("rpn_proposal", self.test_cfg.rpn)
        rpn, proposals = self.rpn_head.forward_train(
            features, metas, boxes, gt_labels=None, gt_bboxes_ignore=None,
            proposal_cfg=proposal_cfg)
        roi = self.roi_head.forward_train(
            features, metas, proposals, boxes, labels, gt_bboxes_ignore=None, gt_masks=None)
        return self.parse_loss({**rpn, **roi})

    def target_losses(self, weak, weak_metas, strong, strong_metas, dissimilar=None):
        self.cur_iter += 1
        self.image_num += len(weak_metas)
        # Always predict ORIGINAL weak images, never the composed canvas.
        boxes, labels = self.teacher_labels(weak, weak_metas, strong, strong_metas)
        if self.aasfod_stage == "fns":
            if len(strong) % 4:
                raise ValueError("FNS batch must contain groups of four original images")
            mosaics = [four_image_mosaic(
                strong[i:i + 4], strong_metas[i:i + 4], boxes[i:i + 4], labels[i:i + 4])
                for i in range(0, len(strong), 4)]
            images, metas, boxes, labels = zip(*mosaics)
            height = max(image.shape[-2] for image in images)
            width = max(image.shape[-1] for image in images)
            # Native collation pads right/bottom, retaining each canvas's
            # pad_shape for anchor validity rather than the batch extent.
            batch = torch.stack([F.pad(
                image, (0, width - image.shape[-1], 0, height - image.shape[-2]))
                for image in images])
            features = self.extract_feat(batch)
            losses = self.detection_losses(features, list(metas), list(boxes), list(labels))
        else:
            # Existing OrthoNet layer1/layer4, not FPN outputs or a new encoder.
            backbone_a = self.backbone(strong)
            features = self.neck(backbone_a) if self.with_neck else backbone_a
            losses = self.detection_losses(features, strong_metas, boxes, labels)
            backbone_b = self.backbone(dissimilar)
            a = self.domain_heads(backbone_a[0], backbone_a[-1], 0)
            b = self.domain_heads(backbone_b[0], backbone_b[-1], 1)
            losses.update(loss_aasfod_local=a["local"] + b["local"],
                          loss_aasfod_global=a["global"] + b["global"])
        return losses


@HOOKS.register_module()
class AASFODTeacherHook(Hook):
    """Priority45: initialization after source load; EMA after optimizer40."""

    def before_run(self, runner):
        model = runner.model.module if is_module_wrapper(runner.model) else runner.model
        teacher = getattr(model.ema_model, "module", model.ema_model)
        copy_detector(model, teacher)
        model.set_epoch(0)

    def after_train_iter(self, runner):
        model = runner.model.module if is_module_wrapper(runner.model) else runner.model
        if (runner.iter + 1) % model.ema_interval == 0:
            ema_detector(model, getattr(model.ema_model, "module", model.ema_model), 0.99)
