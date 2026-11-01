"""Strict source-free teacher-student wrappers for ARS-DETR (DETR group).

Same dataflow as the Oriented R-CNN group's strict UnbiasedTeacher: target
images only (weak view -> EMA teacher, strong view -> student), weight_l = 0,
pseudo labels = teacher detections with score >= score_thr. DETR-specific
choices and their reasons are in experiments/comparison/detr_portability_audit.md §10.
"""
from __future__ import annotations

import copy
import os

import numpy as np
import torch
from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS, Hook
from mmrotate.models.builder import ROTATED_DETECTORS

from ars_detr.models.detectors.ars_detr import ARSDETR
from experiments.comparison import tam_artifacts
from .cga import DIOR_CLASSES, RSAR_CLASSES, CGA, TestMixins
from .extensions.sfut import SFUTOBB
from .extensions.sfyolo import SFYOLOOBB, augment_detector_batch
from .extensions.tam import TargetAugmentationModule
from .rotated_pets import PETSUnbiasedTeacher
from .rotated_semi_base import SemiBaseDetector
from .semantic_teacher import SemanticPrototypeTeacher

# Every ARS-DETR loss except classification is a box/angle regression term:
# loss_bbox (L1), loss_iou / loss_piou (GIoU), loss_angle (AR-CSL), for the
# last layer (no prefix), d{i}. layers and the enc_ proposals.
REGRESSION_LOSS_SUFFIXES = ('loss_bbox', 'loss_iou', 'loss_piou', 'loss_angle')
# CGA modes whose rule never raises a detector score (see port notes), so
# dropping teacher boxes below score_thr before CGA leaves the pseudo labels
# unchanged.
SCORE_NON_INCREASING_CGA_MODES = ('legacy', 'veto_soft')


def weight_pseudo_label_losses(losses, weight_u):
    """Classification-only training on pseudo labels: regression terms * 0."""
    out = {}
    for key, value in losses.items():
        if 'loss' not in key:
            out[key] = value
        elif key.endswith(REGRESSION_LOSS_SUFFIXES):
            out[key] = 0 * value
        else:
            out[key] = weight_u * value
    return out


@ROTATED_DETECTORS.register_module()
class ARSDETR_CGA(ARSDETR, TestMixins):
    """EMA-teacher type for CGA arms: ARS-DETR plus CGA's refine_test."""


@ROTATED_DETECTORS.register_module()
class DETRMeanTeacher(SemiBaseDetector, ARSDETR):
    """Method B on ARS-DETR (UnbiasedTeacher-style, final EMA evaluated)."""

    def __init__(self, backbone, bbox_head, neck=None, train_cfg=None,
                 test_cfg=None, pretrained=None, init_cfg=None,
                 ema_config=None, ema_ckpt=None, cfg=dict()):
        SemiBaseDetector.__init__(self, ema_config=ema_config,
                                  ema_ckpt=ema_ckpt)
        ARSDETR.__init__(self, backbone=backbone, neck=neck,
                         bbox_head=bbox_head, train_cfg=train_cfg,
                         test_cfg=test_cfg, pretrained=pretrained,
                         init_cfg=init_cfg)
        if float(cfg.get('weight_l', 0.0)) != 0.0:
            raise ValueError('strict source-free DETR requires weight_l=0')
        self.score_thr = float(cfg.get('score_thr', 0.7))
        self.weight_u = float(cfg.get('weight_u', 1.0))
        self.momentum = float(cfg.get('momentum', 0.998))
        self.num_classes = self.bbox_head.num_classes
        self.cur_iter = 0
        self.image_num = 0
        self.pseudo_num = 0

    def forward(self, img, img_metas, return_loss=True, **kwargs):
        if not return_loss:
            return self.forward_test(img, img_metas, **kwargs)
        expected = {'gt_bboxes', 'gt_labels', 'img_unlabeled_1',
                    'img_metas_unlabeled_1', 'gt_bboxes_unlabeled_1',
                    'gt_labels_unlabeled_1'}
        if set(kwargs) != expected:
            raise ValueError('strict source-free batch must carry exactly '
                             f'{sorted(expected)}, got {sorted(kwargs)}')
        for name in ('gt_bboxes', 'gt_labels', 'gt_bboxes_unlabeled_1',
                     'gt_labels_unlabeled_1'):
            if any(t.numel() for t in kwargs[name]):
                raise ValueError(
                    f'strict source-free rejects target ground truth: {name}')
        return self.forward_train_semi(
            img, img_metas, kwargs['img_unlabeled_1'],
            kwargs['img_metas_unlabeled_1'])

    def forward_train_semi(self, img_weak, metas_weak, img_strong,
                           metas_strong):
        # The DETR head reads batch_input_shape, which BaseDetector.forward_*
        # would set; these views bypass it.
        for img, metas in ((img_weak, metas_weak), (img_strong, metas_strong)):
            for meta in metas:
                meta['batch_input_shape'] = tuple(img.shape[-2:])
        self.before_student_step()
        self.cur_iter += 1
        self.image_num += len(metas_weak)
        results = self.inference_unlabeled(img_weak, metas_weak)
        gt_bboxes, gt_labels = self.pseudo_labels(results, img_strong.device)
        self.pseudo_num += sum(len(b) for b in gt_bboxes)
        losses = self.student_losses(img_strong, metas_strong, gt_bboxes,
                                     gt_labels, results)
        losses = self.weight_losses(losses)
        losses = {f'{key}_unlabeled': val for key, val in losses.items()}
        losses['pseudo_num'] = torch.tensor(
            [self.pseudo_num / self.image_num], device=img_strong.device)
        return losses

    def before_student_step(self):
        self.update_ema_model(self.momentum)

    def weight_losses(self, losses):
        return weight_pseudo_label_losses(losses, self.weight_u)

    @torch.no_grad()
    def inference_unlabeled(self, img, img_metas, rescale=True):
        """EMA teacher on the weak view; label-level CGA when CGA_SCORER is set.

        CGA failures raise: there is no raw-teacher fallback in this group.
        """
        teacher = getattr(self.ema_model, 'module', self.ema_model)
        results = teacher.simple_test(img, img_metas, rescale=rescale)
        scorer = os.environ.get('CGA_SCORER', '').strip().lower()
        if scorer in ('', 'none'):
            return results
        mode = os.environ.get('CGA_FILTER_MODE', 'legacy').strip().lower()
        if mode not in SCORE_NON_INCREASING_CGA_MODES:
            raise ValueError(f'CGA_FILTER_MODE={mode} may raise scores; the '
                             'DETR pre-filter is only exact for '
                             f'{SCORE_NON_INCREASING_CGA_MODES}')
        results = [[r[r[:, -1] >= self.score_thr] for r in image]
                   for image in results]
        return teacher.refine_test(results, img_metas)

    def pseudo_labels(self, results, device):
        """Teacher boxes with score >= score_thr and valid geometry."""
        gt_bboxes, gt_labels = [], []
        for image in results:
            rows = np.concatenate([r[:, :6] for r in image]).astype(np.float32)
            labels = np.concatenate(
                [np.full(len(r), c, dtype=np.int64) for c, r in enumerate(image)])
            keep = ((rows[:, 5] >= self.score_thr)
                    & np.isfinite(rows[:, :5]).all(axis=1)
                    & (rows[:, 2] > 0) & (rows[:, 3] > 0))
            gt_bboxes.append(torch.from_numpy(rows[keep, :5]).to(device))
            gt_labels.append(torch.from_numpy(labels[keep]).to(device))
        return gt_bboxes, gt_labels

    def student_losses(self, img, img_metas, gt_bboxes, gt_labels,
                       teacher_results):
        outs = self.bbox_head(self.extract_feat(img), img_metas)
        return self.bbox_head.loss(*outs, gt_bboxes, gt_labels, img_metas)


@ROTATED_DETECTORS.register_module()
class DETRPETSTeacher(DETRMeanTeacher):
    """PETS on ARS-DETR: static/dynamic teachers, exchange, rotated WBF.

    Same mechanics and official constants as PETSUnbiasedTeacher (R-CNN
    port); only the detector differs.
    """

    def __init__(self, *args, cfg=dict(), **kwargs):
        super().__init__(*args, cfg=cfg, **kwargs)
        object.__setattr__(self, 'static_teacher',
                           copy.deepcopy(self.ema_model))
        self.pets_epochs = int(cfg.get('pets_epochs', 10))
        self.pets_warmup_epochs = int(cfg.get('pets_warmup_epochs', 2))
        self.pets_period = int(cfg.get('pets_period', 1))
        self.pets_keep_rate = float(cfg.get('pets_keep_rate', 0.999))
        self.pets_pair_iou_thr = float(cfg.get('pets_pair_iou_thr', 0.5))
        self.pets_wbf_iou_thr = float(cfg.get('pets_wbf_iou_thr', 0.55))
        self.pets_skip_box_thr = float(cfg.get('pets_skip_box_thr', 0.01))
        self.momentum = float(cfg.get('pets_warmup_keep_rate', 0.9996))
        self.pets_total_iters = None
        self.pets_exchange_iters = None
        self.pets_num_exchanges = 0

    # Detector-agnostic PETS steps, shared verbatim with the R-CNN port.
    set_pets_schedule = PETSUnbiasedTeacher.set_pets_schedule
    _pets_before_iter = PETSUnbiasedTeacher._pets_before_iter
    inference_unlabeled = PETSUnbiasedTeacher.inference_unlabeled

    def cuda(self, device=None):
        self.static_teacher.cuda(device=device)
        return super().cuda(device=device)

    def forward_train_semi(self, img_weak, *args):
        self._pets_before_iter()
        losses = super().forward_train_semi(img_weak, *args)
        losses['pets_exchanges'] = torch.tensor(
            [float(self.pets_num_exchanges)], device=img_weak.device)
        return losses


@ROTATED_DETECTORS.register_module()
class DETRVLSTTeacher(DETRMeanTeacher):
    """VLST on ARS-DETR (method E; with CGA env = method F).

    Adaptation: the student feature is the last decoder layer's query
    feature (inter_states[-1], 256-d) instead of the 1024-d RoI fc feature;
    positives are the queries Hungarian-matched to pseudo GT in that layer.
    """

    def __init__(self, *args, cfg=dict(), **kwargs):
        super().__init__(*args, cfg=cfg, **kwargs)
        if cfg.get('vlst_lora_path') or os.environ.get('SARCLIP_LORA', '').strip():
            raise ValueError('strict VLST forbids a SARCLIP LoRA')
        self.vlst_loss_weight = float(cfg.get('vlst_loss_weight', 0.1))
        self.vlst_score_thr = float(cfg.get('vlst_score_thr') or self.score_thr)
        self.vlst_pretrained = cfg.get('vlst_pretrained')
        self.vlst_cache_dir = cfg.get('vlst_cache_dir')
        self.vlst_teacher = SemanticPrototypeTeacher(
            num_classes=self.num_classes,
            vlm_dim=int(cfg.get('vlst_vlm_dim', 512)),
            detector_dim=self.bbox_head.embed_dims,
            projection_hidden=int(cfg.get('vlst_projection_hidden', 256)),
            text_visual_alpha=float(cfg.get('vlst_text_visual_alpha', 0.5)),
            prototype_momentum=float(cfg.get('vlst_prototype_momentum', 0.9)),
            temperature=float(cfg.get('vlst_temperature', 0.07)),
            score_threshold=self.vlst_score_thr,
        )
        # Frozen SARCLIP, kept off the module tree (no optimizer, no EMA).
        object.__setattr__(self, '_vlst_vlm', None)

    def _init_vlst(self):
        if self.vlst_teacher.text_prototypes is not None:
            return
        if not self.vlst_pretrained or not os.path.isfile(self.vlst_pretrained):
            raise FileNotFoundError(
                f'strict VLST needs the base SARCLIP checkpoint: {self.vlst_pretrained}')
        names = {len(RSAR_CLASSES): RSAR_CLASSES, len(DIOR_CLASSES): DIOR_CLASSES}
        vlm = CGA(class_names=list(names[self.num_classes]), backend='sarclip',
                  model='ViT-B-32', pretrained=self.vlst_pretrained,
                  cache_dir=self.vlst_cache_dir
                  or os.path.dirname(self.vlst_pretrained), strict=True)
        object.__setattr__(self, '_vlst_vlm', vlm)
        self.vlst_teacher.set_text_prototypes(
            torch.from_numpy(vlm.text_prototype_matrix()).float())

    def forward_train_semi(self, *args):
        self._init_vlst()
        return super().forward_train_semi(*args)

    @torch.no_grad()
    def _update_visual_prototypes(self, results, img_metas):
        """EMA visual prototypes from frozen-SARCLIP crops of teacher boxes."""
        for image, meta in zip(results, img_metas):
            # Rows below vlst_score_thr never update a prototype; drop them
            # before cropping instead of after.
            image = [r[r[:, -1] >= self.vlst_score_thr] for r in image]
            inputs = TestMixins._flatten_cga_inputs(image)
            if inputs is None:
                continue
            boxes, scores, labels = inputs
            boxes = restore_unflipped_aabb(boxes, meta)
            valid = ((boxes[:, 2] - boxes[:, 0]) >= 1.0) & \
                ((boxes[:, 3] - boxes[:, 1]) >= 1.0)
            if not valid.any():
                continue
            feats, _, _ = self._vlst_vlm.forward_aabb_embed(
                meta['filename'], boxes[valid], scores[valid], labels[valid])
            if feats.shape[0] == 0:
                continue
            device = self.vlst_teacher.text_prototypes.device
            self.vlst_teacher.update_visual_prototypes(
                torch.from_numpy(feats).float().to(device),
                torch.from_numpy(labels[valid]).long().to(device),
                torch.from_numpy(scores[valid]).float().to(device),
                score_thr=self.vlst_score_thr)

    def student_losses(self, img, img_metas, gt_bboxes, gt_labels,
                       teacher_results):
        if self._vlst_vlm is not None:
            self._update_visual_prototypes(teacher_results, img_metas)
        head = self.bbox_head
        num_layers = head.transformer.decoder.num_layers
        captured = []
        # Input of the last decoder layer's classifier = inter_states[-1].
        handle = head.cls_branches[num_layers - 1].register_forward_hook(
            lambda module, inputs, output: captured.append(inputs[0]))
        try:
            outs = head(self.extract_feat(img), img_metas)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError(f'expected one last-layer query capture, got {len(captured)}')
        losses = head.loss(*outs, gt_bboxes, gt_labels, img_metas)
        pos_feats, pos_labels = last_layer_positive_queries(
            head, outs, captured[0], gt_bboxes, gt_labels, img_metas)
        if len(pos_labels) == 0:
            losses['loss_vlst_proto'] = captured[0].sum() * 0.0
            return losses
        # As in the R-CNN port, positives carry the admission threshold as
        # their score, so every positive enters the loss.
        loss, _ = self.vlst_teacher.compute_prototype_loss(
            pos_feats, pos_labels,
            torch.full_like(pos_labels, self.vlst_score_thr, dtype=torch.float32),
            score_thr=self.vlst_score_thr)
        losses['loss_vlst_proto'] = self.vlst_loss_weight * loss
        return losses


@ROTATED_DETECTORS.register_module()
class DETRSFUTTeacher(DETRMeanTeacher):
    """SF-UT (arXiv 2407.07586) on ARS-DETR, same mechanics as SFUTOBB.

    Pseudo labels >= score_thr, every pseudo loss kept (classification and
    box/angle regression), teacher frozen during the step and EMA-updated
    after each optimizer step (DETRAfterOptimizerTeacherHook).
    """
    iteration_teacher_momentum = SFUTOBB.iteration_teacher_momentum

    def before_student_step(self):
        pass  # the teacher moves after the optimizer step, not before the forward

    def weight_losses(self, losses):
        return {key: self.weight_u * value if 'loss' in key else value
                for key, value in losses.items()}


@ROTATED_DETECTORS.register_module()
class DETRSFYOLOTeacher(DETRSFUTTeacher):
    """SF-YOLO (TAM + Mean Teacher + SSM) on ARS-DETR, same mechanics as SFYOLOOBB:
    TAM-stylised strong view, parameter EMA 0.999 after each optimizer step,
    student stabilisation at each later epoch start (StudentStabilizationHook).
    """
    iteration_teacher_momentum = SFYOLOOBB.iteration_teacher_momentum
    ssm_teacher_fraction = SFYOLOOBB.ssm_teacher_fraction
    update_ema_model = SFYOLOOBB.update_ema_model
    stabilize_student = SFYOLOOBB.stabilize_student

    def __init__(self, *args, cfg=dict(), **kwargs):
        super().__init__(*args, cfg=cfg, **kwargs)
        self.tam_checkpoint = cfg['tam_checkpoint']
        self.tam_encoder_weights = cfg['tam_encoder_weights']
        self.tam_identity = dict(dataset=cfg['tam_dataset'],
                                 domain=cfg['tam_domain'], seed=int(cfg['tam_seed']))
        object.__setattr__(self, '_tam', None)

    def forward_train_semi(self, img_weak, metas_weak, img_strong, metas_strong):
        if self._tam is None:
            object.__setattr__(self, '_tam', load_tam(
                self.tam_checkpoint, self.tam_identity,
                self.tam_encoder_weights, img_strong.device))
        style_index = int(torch.randint(len(metas_strong), ()).item())
        img_strong = augment_detector_batch(
            self._tam, img_strong, metas_strong, style_index)
        return super().forward_train_semi(img_weak, metas_weak, img_strong,
                                          metas_strong)


def load_tam(path, identity, encoder_weights, device):
    """tam_artifacts.load_completed_tam, but the frozen VGG encoder is read from
    this host's copy (the payload records the fitting host's path)."""
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if (payload['schema'] != tam_artifacts.SCHEMA
            or payload['status'] != 'complete'
            or payload['training_steps'] != tam_artifacts.FORMAL_STEPS
            or payload['identity'] != identity
            or identity['seed'] != tam_artifacts.FIT_SEED
            or payload['normalization'] != tam_artifacts.NORMALIZATION
            or os.path.basename(encoder_weights)
            != os.path.basename(payload['encoder_weights'])):
        raise ValueError(f'{path} is not the completed seed-42 TAM for {identity}')
    model = TargetAugmentationModule(weights_path=encoder_weights)
    for name in ('decoder', 'F1', 'F2'):
        getattr(model, name).load_state_dict(payload['components'][name], strict=True)
    return model.to(device).freeze_for_inference()


@HOOKS.register_module()
class DETRAfterOptimizerTeacherHook(Hook):
    """Teacher EMA after the optimizer step (priority 45: optimizer 40, checkpoint 50)."""

    def after_train_iter(self, runner):
        model = runner.model.module if is_module_wrapper(runner.model) else runner.model
        model.update_ema_model(model.iteration_teacher_momentum)


def last_layer_positive_queries(head, outs, queries, gt_bboxes, gt_labels,
                                img_metas):
    """Queries the last decoder layer Hungarian-matches to pseudo GT.

    Re-runs the loss's own assignment for that layer (same inputs as
    ``ARSDeformableDETRHead.loss``), so the positives are exactly the ones
    whose classification target is a pseudo label.
    """
    all_cls, all_bbox, all_angle, all_angle_cls = outs[:4]
    last = len(all_cls) - 1
    num_imgs = len(img_metas)
    with torch.no_grad():
        labels_list = head.get_targets(
            [all_cls[last][i] for i in range(num_imgs)],
            [all_bbox[last][i] for i in range(num_imgs)],
            [all_angle[last][i] for i in range(num_imgs)],
            [all_angle_cls[last][i] for i in range(num_imgs)],
            head.angle_coders[last], gt_bboxes, gt_labels, img_metas)[0]
    labels = torch.stack(labels_list)            # (B, Q), num_classes = bg
    pos = labels < head.num_classes
    return queries[pos], labels[pos]


def restore_unflipped_aabb(boxes, img_meta):
    """Undo the shared RRandomFlip on xyxy boxes before cropping the file.

    Copied from the R-CNN group's UnbiasedTeacherVLST: teacher boxes are
    rescaled to the original size but stay in the flipped frame.
    """
    if not bool(img_meta.get('flip', False)):
        return boxes
    height, width = (int(v) for v in img_meta['ori_shape'][:2])
    direction = img_meta.get('flip_direction', 'horizontal')
    if direction not in ('horizontal', 'vertical', 'diagonal'):
        raise ValueError(f'Unsupported VLST flip direction: {direction}')
    restored = boxes.copy()
    if direction in ('horizontal', 'diagonal'):
        restored[:, 0] = width - boxes[:, 2] - 1
        restored[:, 2] = width - boxes[:, 0] - 1
    if direction in ('vertical', 'diagonal'):
        restored[:, 1] = height - boxes[:, 3] - 1
        restored[:, 3] = height - boxes[:, 1] - 1
    restored[:, 0::2] = restored[:, 0::2].clip(0, width - 1)
    restored[:, 1::2] = restored[:, 1::2].clip(0, height - 1)
    return restored
