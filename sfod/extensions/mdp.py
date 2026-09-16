"""Preprint-guided MDP-OBB; source detector and ordinary evaluation are unchanged."""

from contextlib import contextmanager

import torch
from mmcv.ops import RoIAlignRotated
from mmcv.runner import load_checkpoint
from mmdet.models.builder import DETECTORS

from .mdp_losses import MDPModules, mix_target_pairs, rotated_class_prototypes
from .proposal_teacher import ProposalAlignedTeacher, map_shared_geometry


@contextmanager
def capture_roi_predictions(roi_head):
    captured = {'rois': [], 'logits': []}
    roi_hook = roi_head.bbox_roi_extractor.register_forward_pre_hook(
        lambda _module, args: captured['rois'].append(args[1]))
    head_hook = roi_head.bbox_head.register_forward_hook(
        lambda _module, _args, output: captured['logits'].append(output[0]))
    try:
        yield captured
    finally:
        roi_hook.remove()
        head_hook.remove()


def captured_roi_tensors(captured, reference, num_classes, empty):
    if not captured['rois'] and not captured['logits'] and empty:
        return reference.new_empty((0, 6)), reference.new_empty((0, num_classes))
    if len(captured['rois']) != 1 or len(captured['logits']) != 1:
        raise RuntimeError('MDP requires one actual ROI/classifier pass with aligned rows')
    return captured['rois'][0], captured['logits'][0]


@DETECTORS.register_module()
class MDPOBB(ProposalAlignedTeacher):
    # The preprint experiment settings and official script both use epoch EMA.
    epoch_teacher_momentum = .9

    def __init__(self, *args, cfg, ema_ckpt=None, **kwargs):
        super().__init__(*args, cfg=cfg, ema_ckpt=None, **kwargs)
        self._mdp_teacher_loaded = False
        self._mdp_weights_loaded = False
        if ema_ckpt is not None:
            load_checkpoint(self.ema_model, ema_ckpt, map_location='cpu', strict=True)
            self._mdp_teacher_loaded = True
        self._initialize_mdp()

    def _initialize_mdp(self):
        self.mdp_num_classes = self.num_classes + 1
        c2_channels = self.backbone._tap_channels(0)
        c4_channels = self.backbone._tap_channels(2)
        self.mdp = MDPModules(c2_channels, c4_channels, self.mdp_num_classes)
        self.mdp.pool = RoIAlignRotated(7, 1.0 / 16.0, 2, True, True)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        expected = self.state_dict()
        supplied = {key[len(prefix):]: value for key, value in state_dict.items()
                    if key.startswith(prefix)}
        resumed = any(key.startswith('mdp.') for key in supplied)
        core = {key for key in expected if not key.startswith('mdp.')}
        required = set(expected) if resumed else core
        missing = required - supplied.keys()
        extra = supplied.keys() - expected.keys()
        wrong_shape = [
            key for key in supplied.keys() & expected.keys()
            if supplied[key].shape != expected[key].shape
        ]
        if missing or extra or wrong_shape:
            raise RuntimeError(
                f'MDP checkpoint mismatch: missing={sorted(missing)}, '
                f'unexpected={sorted(extra)}, shape={wrong_shape}')
        teacher = getattr(self.ema_model, 'module', self.ema_model)
        if not resumed:
            teacher_state = teacher.state_dict()
            if set(teacher_state) != core:
                raise RuntimeError('MDP teacher and Student original detector keys differ')
            source = {key: supplied[key] for key in core}
            if self._mdp_teacher_loaded:
                unequal = [
                    key for key, value in teacher_state.items()
                    if not torch.equal(source[key].to(value), value)
                ]
                if unequal:
                    raise RuntimeError(f'MDP Student and teacher source weights differ: {unequal}')
            else:
                teacher.load_state_dict(source, strict=True)
                self._mdp_teacher_loaded = True
            teacher.requires_grad_(False)
            teacher.eval()
        self._mdp_weights_loaded = self._mdp_teacher_loaded
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def _features(self, detector, images, adversarial=False):
        c4 = []
        backbone = detector.backbone
        hooks = []
        if adversarial and detector.training:
            hooks.append(getattr(backbone, backbone.res_layers[0]).register_forward_hook(
                lambda _module, _args, output: self.mdp.afsp(output)))
        hooks.append(getattr(backbone, backbone.res_layers[2]).register_forward_hook(
            lambda _module, _args, output: c4.append(output)))
        try:
            features = detector.extract_feat(images)
        finally:
            for hook in hooks:
                hook.remove()
        if len(c4) != 1:
            raise RuntimeError('MDP requires exactly one original stride16 C4 output')
        return features, c4[0]

    def _detection_losses(self, features, metas, boxes, labels, capture=False):
        proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
        rpn_losses, proposals = self.rpn_head.forward_train(
            features, metas, boxes, gt_labels=None, gt_bboxes_ignore=None,
            proposal_cfg=proposal_cfg)
        if capture:
            with capture_roi_predictions(self.roi_head) as captured:
                roi_losses = self.roi_head.forward_train(
                    features, metas, proposals, boxes, labels,
                    gt_bboxes_ignore=None, gt_masks=None)
            rois, logits = captured_roi_tensors(
                captured, features[0], self.mdp_num_classes,
                empty=not any(len(value) for value in proposals + boxes))
        else:
            roi_losses = self.roi_head.forward_train(
                features, metas, proposals, boxes, labels,
                gt_bboxes_ignore=None, gt_masks=None)
            rois = logits = None
        return self.parse_loss({**rpn_losses, **roi_losses}), rois, logits

    def forward_train_semi(
            self, img, img_metas, gt_bboxes, gt_labels,
            img_unlabeled, img_metas_unlabeled, gt_bboxes_unlabeled, gt_labels_unlabeled,
            img_unlabeled_1, img_metas_unlabeled_1, gt_bboxes_unlabeled_1, gt_labels_unlabeled_1):
        if not self._mdp_weights_loaded:
            raise RuntimeError('MDP needs complete source initialization or a paired EMA for resume')
        self._assert_strict_empty_targets(
            gt_bboxes=gt_bboxes, gt_labels=gt_labels,
            gt_bboxes_unlabeled=gt_bboxes_unlabeled, gt_labels_unlabeled=gt_labels_unlabeled,
            gt_bboxes_unlabeled_1=gt_bboxes_unlabeled_1, gt_labels_unlabeled_1=gt_labels_unlabeled_1)
        batch = len(img_unlabeled)
        if batch < 2 or batch % 2 or len(img_unlabeled_1) != batch:
            raise ValueError('MDP requires aligned weak/strong batches with an even local size')
        self.cur_iter += 1
        self.image_num += batch
        teacher = getattr(self.ema_model, 'module', self.ema_model)
        teacher.eval()
        with torch.no_grad():
            teacher_features, teacher_c4 = self._features(teacher, img_unlabeled)
            proposals = teacher.rpn_head.simple_test_rpn(teacher_features, img_metas_unlabeled)
            with capture_roi_predictions(teacher.roi_head) as captured:
                detections = teacher.roi_head.simple_test(
                    teacher_features, proposals, img_metas_unlabeled, rescale=False)
            teacher_rois, teacher_logits = captured_roi_tensors(
                captured, teacher_c4, self.mdp_num_classes,
                empty=not any(len(value) for value in proposals))
            teacher_local, teacher_present = rotated_class_prototypes(
                teacher_c4, teacher_rois, teacher_logits, self.mdp_num_classes, self.mdp.pool)
        boxes, labels = self.create_pseudo_results(
            img_unlabeled_1, detections, [], img_unlabeled.device)
        boxes = [
            map_shared_geometry(value, weak, strong)
            for value, weak, strong in zip(boxes, img_metas_unlabeled, img_metas_unlabeled_1)
        ]
        self.analysis()
        mixed, mixed_boxes, mixed_labels, mixed_metas = mix_target_pairs(
            img_unlabeled_1, boxes, labels, img_metas_unlabeled_1)
        mixed_features, student_c4 = self._features(self, mixed)
        mixed_losses, student_rois, student_logits = self._detection_losses(
            mixed_features, mixed_metas, mixed_boxes, mixed_labels, capture=True)
        student_local, student_present = rotated_class_prototypes(
            self.mdp.transform(student_c4), student_rois, student_logits,
            self.mdp_num_classes, self.mdp.pool)
        pfd_loss, present = self.mdp.history(
            teacher_local, student_local, teacher_present, student_present)
        half = batch // 2
        perturbed_features, _ = self._features(self, img_unlabeled_1[:half], adversarial=True)
        afsp_losses, _, _ = self._detection_losses(
            perturbed_features, img_metas_unlabeled_1[:half], boxes[:half], labels[:half])
        losses = {
            f'mdp_{branch}_{name}': value * self.weight_u if 'loss' in name else value
            for branch, values in (('msp', mixed_losses), ('afsp', afsp_losses))
            for name, value in values.items()
        }
        losses.update(
            loss_mdp_pfd=.5 * self.weight_u * pfd_loss,
            mdp_pfd_classes=present.to(img_unlabeled.dtype),
            pseudo_num=img_unlabeled.new_tensor(self.pseudo_num.sum() / self.image_num),
        )
        return losses
