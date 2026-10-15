"""Independent TAM+MT+SSM OBB mechanism under the separate two-epoch budget."""

from pathlib import Path

import torch
from torch.nn import functional as F
from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS, Hook
from mmdet.models.builder import DETECTORS

from experiments.comparison.tam_artifacts import BGR_MEAN, FIT_SEED, load_completed_tam
from .sfut import SFUTOBB


@torch.no_grad()
def augment_detector_batch(tam, images, metas, style_index):
    """Use centered BGR for both inputs, then restore detector normalization/geometry."""
    contents = []
    for image, meta in zip(images, metas):
        height, width = meta["img_shape"][:2]
        norm = meta["img_norm_cfg"]
        mean = image.new_tensor(norm["mean"])[:, None, None]
        std = image.new_tensor(norm["std"])[:, None, None]
        pixels = image[:, :height, :width] * std + mean
        if norm.get("to_rgb", True):
            pixels = pixels.flip(0)
        contents.append((pixels - image.new_tensor(BGR_MEAN)[:, None, None]).unsqueeze(0))
    style = tam.encoder(contents[style_index])[-1]
    output = images.clone()
    for index, (content, meta) in enumerate(zip(contents, metas)):
        features = tam.encoder(content)[-1]
        generated = tam.decoder(tam.transform(features, style, alpha=0.4))
        height, width = content.shape[-2:]
        generated = F.interpolate(generated, size=(height, width),
                                  mode="bilinear", align_corners=False)[0]
        pixels = (generated + generated.new_tensor(BGR_MEAN)[:, None, None]).clamp(0, 255)
        norm = meta["img_norm_cfg"]
        if norm.get("to_rgb", True):
            pixels = pixels.flip(0)
        mean = pixels.new_tensor(norm["mean"])[:, None, None]
        std = pixels.new_tensor(norm["std"])[:, None, None]
        output[index, :, :height, :width] = (pixels - mean) / std
    return output


@DETECTORS.register_module()
class SFYOLOOBB(SFUTOBB):
    iteration_teacher_momentum = 0.999
    ssm_teacher_fraction = 0.5

    def __init__(self, *args, cfg, **kwargs):
        super().__init__(*args, cfg=cfg, **kwargs)
        self.tam_checkpoint = cfg.get("tam_checkpoint")
        if not self.tam_checkpoint:
            raise ValueError("SF-YOLO requires an actual completed target-domain TAM checkpoint")
        if not Path(self.tam_checkpoint).is_file():
            raise FileNotFoundError(f"Missing trained TAM checkpoint: {self.tam_checkpoint}")
        self.tam_identity = {
            "dataset": cfg["tam_dataset"], "domain": cfg["tam_domain"], "seed": int(cfg["tam_seed"])}
        if self.tam_identity["seed"] != FIT_SEED:
            raise ValueError("SF-YOLO reuses the domain TAM fit seed42 for every detector seed")
        object.__setattr__(self, "_tam", None)

    def forward_train_semi(
            self, img, img_metas, gt_bboxes, gt_labels,
            img_unlabeled, img_metas_unlabeled, gt_bboxes_unlabeled, gt_labels_unlabeled,
            img_unlabeled_1, img_metas_unlabeled_1, gt_bboxes_unlabeled_1, gt_labels_unlabeled_1):
        self._assert_strict_empty_targets(
            gt_bboxes=gt_bboxes, gt_labels=gt_labels,
            gt_bboxes_unlabeled=gt_bboxes_unlabeled, gt_labels_unlabeled=gt_labels_unlabeled,
            gt_bboxes_unlabeled_1=gt_bboxes_unlabeled_1, gt_labels_unlabeled_1=gt_labels_unlabeled_1)
        if self._tam is None:
            object.__setattr__(self, "_tam", load_completed_tam(
                self.tam_checkpoint, self.tam_identity, img_unlabeled_1.device))
        style_index = int(torch.randint(len(img_metas_unlabeled_1), ()).item())
        augmented = augment_detector_batch(
            self._tam, img_unlabeled_1, img_metas_unlabeled_1, style_index)
        return super().forward_train_semi(
            img, img_metas, gt_bboxes, gt_labels,
            img_unlabeled, img_metas_unlabeled, gt_bboxes_unlabeled, gt_labels_unlabeled,
            augmented, img_metas_unlabeled_1, gt_bboxes_unlabeled_1, gt_labels_unlabeled_1)

    @torch.no_grad()
    def update_ema_model(self, momentum=0.999):
        teacher = getattr(self.ema_model, "module", self.ema_model)
        student = dict(self.named_parameters())
        for name, parameter in teacher.named_parameters():
            parameter.mul_(momentum).add_(student[name], alpha=1 - momentum)

    @torch.no_grad()
    def stabilize_student(self):
        teacher = getattr(self.ema_model, "module", self.ema_model)
        teacher_parameters = dict(teacher.named_parameters())
        for name, parameter in self.named_parameters():
            parameter.mul_(1 - self.ssm_teacher_fraction).add_(
                teacher_parameters[name], alpha=self.ssm_teacher_fraction)


@HOOKS.register_module()
class StudentStabilizationHook(Hook):
    """Code-defined next-epoch SSM; never fabricate a within-epoch/final-teacher update."""

    def before_train_epoch(self, runner):
        if runner.epoch > 0:
            model = runner.model.module if is_module_wrapper(runner.model) else runner.model
            model.stabilize_student()
