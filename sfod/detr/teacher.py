"""EMA self-training with CGA pseudo labels and query-level VLST."""
import copy
import os
import numpy as np
import torch
from mmcv.runner import load_checkpoint
from mmrotate.models.builder import ROTATED_DETECTORS
from sfod.rotated_semi_base import SemiBaseDetector
from sfod.cga import obb2xyxy
from .detector import OrthoRotatedDETR
from .geometry import from_original
from .semantic import QuerySemanticPrototypeTeacher


@ROTATED_DETECTORS.register_module()
class UnbiasedTeacherDETRVLST(SemiBaseDetector, OrthoRotatedDETR):
    """Keep detector state keys compatible with its supervised source model.

EMA is a plain object, as required by IRAOD's training API. It is excluded
from the optimizer. Both teacher tensors and semantic buffers are included
in student checkpoints, allowing exact restoration of adaptation state.
"""
    def __init__(self, *args, cfg=None, source_checkpoint=None,
                 allow_random_init=False, **kwargs):
        OrthoRotatedDETR.__init__(self, *args, **kwargs)
        cfg = dict(cfg or {})
        self.num_classes = self.bbox_head.num_classes
        self.weight_l = float(cfg.get('weight_l', 0.))
        self.weight_u = float(cfg.get('weight_u', 1.))
        self.use_bbox_reg = bool(cfg.get('use_bbox_reg', False))
        self.score_thr = float(cfg.get('score_thr', .7))
        self.momentum = float(cfg.get('momentum', .998))
        self.use_cga = bool(cfg.get('cga_enabled', True))
        self.vlst_enabled = bool(cfg.get('vlst_enabled', True))
        self.vlst_loss_weight = float(cfg.get('vlst_loss_weight', .1))
        self.vlst_score_thr = cfg.get('vlst_score_thr')
        if self.vlst_score_thr is None:
            self.vlst_score_thr = self.score_thr
        self.source_checkpoint = source_checkpoint
        self.allow_random_init = bool(allow_random_init)
        if not 0 <= self.momentum < 1 or not 0 <= self.score_thr <= 1:
            raise ValueError('Invalid EMA momentum or pseudo score threshold')
        if not allow_random_init and not source_checkpoint:
            raise ValueError('Train a DETR source detector and set source_checkpoint first')
        if source_checkpoint and not os.path.isfile(source_checkpoint):
            raise FileNotFoundError('DETR source checkpoint missing: ' + source_checkpoint)
        teacher_kwargs = copy.deepcopy(kwargs)
        # Source initialization is loaded only once, in init_weights below.
        teacher_kwargs['backbone']['init_cfg'] = None
        self.ema_model = OrthoRotatedDETR(*copy.deepcopy(args), **teacher_kwargs)
        self.ema_model.eval()
        self.register_buffer('_ema_ready', torch.tensor(False))
        self.register_buffer('_adaptation_steps', torch.tensor(0, dtype=torch.long))
        self.vlst_teacher = None
        if self.vlst_enabled:
            self.vlst_teacher = QuerySemanticPrototypeTeacher(
                num_classes=self.num_classes,
                detector_dim=self.bbox_head.embed_dims,
                vlm_dim=int(cfg.get('vlst_vlm_dim', 512)),
                projection_hidden=int(cfg.get('vlst_projection_hidden', 256)),
                text_visual_alpha=float(cfg.get('vlst_text_visual_alpha', .5)),
                prototype_momentum=float(cfg.get('vlst_prototype_momentum', .9)),
                temperature=float(cfg.get('vlst_temperature', .07)),
                score_threshold=float(self.vlst_score_thr))
        self._vlst_vlm = None

    def _ema_host(self):
        return getattr(self.ema_model, 'module', self.ema_model)

    def init_weights(self):
        OrthoRotatedDETR.init_weights(self)
        if self.source_checkpoint:
            # Strictly validate the whole detector. An R-CNN checkpoint must
            # never silently initialize random transformer/angle heads.
            checkpoint = load_checkpoint(self._ema_host(), self.source_checkpoint,
                                         map_location='cpu', strict=True)
            state = checkpoint.get('state_dict', checkpoint)
            state = {k.removeprefix('module.'): v for k, v in state.items()}
            incompatible = self.load_state_dict(state, strict=False)
            unexpected = incompatible.unexpected_keys
            missing = [k for k in incompatible.missing_keys
                       if not k.startswith('vlst_teacher.') and
                       k not in ('_ema_ready', '_adaptation_steps')]
            if missing or unexpected:
                raise RuntimeError(f'Invalid DETR source state: missing={missing}, extra={unexpected}')
        self.update_ema_model(0.)

    def _apply(self, fn):
        super()._apply(fn)
        if getattr(self, 'ema_model', None) is not None:
            self.ema_model._apply(fn)
        return self

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, 'ema_model', None) is not None:
            self.ema_model.eval()
        return self

    @torch.no_grad()
    def update_ema_model(self, momentum=None):
        teacher = self._ema_host()
        # Use Module's state traversal, without the serialized EMA copy.
        own = dict(self.named_parameters())
        own.update(dict(self.named_buffers()))
        momentum = self.momentum if momentum is None else momentum
        if not self._ema_ready:
            momentum = 0.
        for key, value in teacher.state_dict().items():
            source = own[key].detach()
            if value.is_floating_point():
                value.mul_(momentum).add_(source, alpha=1 - momentum)
            else:
                value.copy_(source)
        self._ema_ready.fill_(True)
        teacher.eval()

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        for key, value in self._ema_host().state_dict(keep_vars=keep_vars).items():
            destination[prefix + 'ema_teacher.' + key] = value

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        marker = prefix + 'ema_teacher.'
        keys = [key for key in state_dict if key.startswith(marker)]
        if keys:
            state = {key[len(marker):]: state_dict.pop(key) for key in keys}
            self._ema_host().load_state_dict(state, strict=True)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _init_semantics(self):
        if self._vlst_vlm is None:
            self._vlst_vlm, _ = self._ema_host()._build_cga(self.num_classes)
            if not self.vlst_teacher.text_ready:
                self.vlst_teacher.set_text_prototypes(self._vlst_vlm.text_prototype_matrix())

    @torch.no_grad()
    def _pseudo_targets(self, results, weak_metas, strong_metas, device):
        boxes, labels, scores, original = [], [], [], []
        for per_class, weak, strong in zip(results, weak_metas, strong_metas):
            if weak['filename'] != strong['filename']:
                raise ValueError('Weak/strong view image identities differ')
            b, l, s = [], [], []
            for cls, rows in enumerate(per_class):
                keep = np.isfinite(rows).all(axis=1) & (rows[:, 5] >= self.score_thr)
                keep &= (rows[:, 2] >= 1.) & (rows[:, 3] >= 1.)
                rows = rows[keep]
                b.append(torch.as_tensor(rows[:, :5], device=device))
                s.append(torch.as_tensor(rows[:, 5], device=device))
                l.append(torch.full((len(rows),), cls, device=device, dtype=torch.long))
            obb = torch.cat(b).float()
            original.append(obb)
            boxes.append(from_original(obb, strong))
            labels.append(torch.cat(l))
            scores.append(torch.cat(s).float())
        return boxes, labels, scores, original

    @torch.no_grad()
    def _update_prototypes(self, original, labels, scores, metas):
        features, selected_labels, selected_scores = [], [], []
        device = self.vlst_teacher.prototype_counts.device
        for boxes, cls, conf, meta in zip(original, labels, scores, metas):
            if not len(boxes):
                continue
            aabb = obb2xyxy(boxes.detach().cpu().numpy())
            valid = ((aabb[:, 2] - aabb[:, 0]) >= 1) & ((aabb[:, 3] - aabb[:, 1]) >= 1)
            ids = torch.as_tensor(np.flatnonzero(valid), device=device)
            if not len(ids):
                continue
            f, _, _ = self._vlst_vlm.forward_aabb_embed(
                meta['filename'], aabb[valid], conf[ids].cpu().numpy(), cls[ids].cpu().numpy())
            if len(f) != len(ids):
                raise ValueError('VLM crop features lost pseudo-label alignment')
            features.append(torch.as_tensor(f, device=device, dtype=torch.float32))
            selected_labels.append(cls[ids])
            selected_scores.append(conf[ids])
        features = torch.cat(features) if features else torch.empty(
            (0, self.vlst_teacher.vlm_dim), device=device)
        cls = torch.cat(selected_labels) if selected_labels else torch.empty((0,), device=device, dtype=torch.long)
        conf = torch.cat(selected_scores) if selected_scores else torch.empty((0,), device=device)
        self.vlst_teacher.update_visual_prototypes(features, cls, conf)

    def forward_train_semi(self, img, img_metas, gt_bboxes, gt_labels,
                           img_unlabeled, img_metas_unlabeled,
                           gt_bboxes_unlabeled=None, gt_labels_unlabeled=None,
                           img_unlabeled_1=None, img_metas_unlabeled_1=None,
                           gt_bboxes_unlabeled_1=None, gt_labels_unlabeled_1=None):
        # Unlabeled GT arguments exist only for the legacy dataset interface;
        # they never enter matching, pseudo admission, losses or prototypes.
        self.update_ema_model()
        self._adaptation_steps.add_(1)
        losses = {}
        if self.weight_l:
            supervised = self.forward_train(img, img_metas, gt_bboxes, gt_labels)
            losses.update({k: v * self.weight_l for k, v in supervised.items()})
        with torch.no_grad():
            result = self._ema_host().predict_original(
                img_unlabeled, img_metas_unlabeled, with_cga=self.use_cga)
            boxes, labels, scores, original = self._pseudo_targets(
                result, img_metas_unlabeled, img_metas_unlabeled_1, img_unlabeled_1.device)
        if self.vlst_enabled:
            self._init_semantics()
            self._update_prototypes(original, labels, scores, img_metas_unlabeled)
        features = self.extract_feat_with_meta(img_unlabeled_1, img_metas_unlabeled_1)
        outputs, queries = self.bbox_head.forward_with_queries(features, img_metas_unlabeled_1)
        unlabeled, pairs = self.bbox_head.loss_with_matches(
            *outputs, boxes, labels, img_metas_unlabeled_1, regression=self.use_bbox_reg)
        count = 0
        if self.vlst_enabled:
            q = torch.cat([queries[-1, i, rows] for i, (rows, _) in enumerate(pairs)])
            cls = torch.cat([labels[i][cols] for i, (_, cols) in enumerate(pairs)])
            conf = torch.cat([scores[i][cols] for i, (_, cols) in enumerate(pairs)])
            proto, count = self.vlst_teacher.compute_prototype_loss(q, cls, conf)
            unlabeled['loss_vlst_proto'] = self.vlst_loss_weight * proto
        losses.update({k + '_unlabeled': self.weight_u * v for k, v in unlabeled.items()})
        losses['pseudo_num'] = queries.new_tensor(sum(len(b) for b in boxes) / len(boxes))
        losses['vlst_samples'] = queries.new_tensor(float(count))
        return losses
