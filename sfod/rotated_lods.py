"""LODS-OBB: "Source-Free Object Detection by Learning to Overlook Domain Style"
(Li et al., CVPR 2022) on the shared Oriented R-CNN / OrthoNet-50 source.

Port of ``Overlook/train_target.py`` + ``Overlook/gromovWasserstein.py`` +
``lib/model/faster_rcnn/faster_rcnn_style.py`` (official repo @ 2dc11b1).
Every deviation is listed in experiments/comparison/lods_obb_port.md.

Per iteration (train_target.py:587-655):
  teacher (EMA, original image)  -> rotated RPN proposals rois_t, RoI feats, cls
  student (SEM-enhanced image)   -> RoI feats, cls on the *same* rois_t
  loss = label_weight * CE(student, confident teacher labels incl. background)
       + gw_ins_weight * GW(instance graph, teacher relation prior T)
       + gw_global_weight * GW(backbone C4 pixel graph)
  teacher <- alpha * teacher + (1 - alpha) * student
"""
from __future__ import annotations

import os.path as osp
import random

import cv2
import mmcv
import torch
import torch.nn.functional as F
from mmdet.models.builder import DETECTORS
from mmrotate.core import rbbox2roi

from .lods_sem import PIXEL_MEANS_BGR, StyleEnhancementModule
from .rotated_semi_two_stage import SemiTwoStageDetector
from .semi_dota_dataset import _discover_image_filenames


def cosine_affinity(feat):
    """gromovWasserstein.affinity_matrix, affinity_type='cosine' (no eps, as official)."""
    energy = feat.pow(2).sum(1, keepdim=True).sqrt()
    return feat.mm(feat.t()) / energy.mm(energy.t())


def gromov_wasserstein_loss(feat_stu, feat_tea, t=None, beta=0.5):
    """gromovWasserstein.forward with l_type='KL' (gromovWasserstein.py:15-68)."""
    a_s = cosine_affinity(feat_stu)
    a_t = cosine_affinity(feat_tea)
    n = feat_stu.size(0)
    T = torch.eye(n, device=feat_stu.device, dtype=feat_stu.dtype)
    if t is not None:
        T = beta * t + T
    T = T / T.sum()
    ones = torch.ones(n, 1, device=feat_stu.device, dtype=feat_stu.dtype)
    p = T.mm(ones)
    q = T.t().mm(ones)
    f1_st = (a_s * torch.log(a_s + 1e-7) - a_s).mm(p).mm(ones.t())
    f2_st = ones.mm(q.t().mm(a_t.t()))
    cost = f1_st + f2_st - a_s.mm(T).mm(torch.log(a_t + 1e-7).t())
    return (cost * T).sum()


def instance_relation_prior(cls_score_t, labels_t, conf_t, thresh, bg_index):
    """train_target.py:623-633: same-confident-foreground mask * cosine(teacher logits)."""
    labels = labels_t.clone()
    labels[conf_t < thresh] = bg_index
    unit = torch.eye(cls_score_t.size(1), device=cls_score_t.device)
    unit[bg_index, bg_index] = 0  # ignore background
    onehot = unit[labels]
    mask = onehot.mm(onehot.t())
    norm = cls_score_t / cls_score_t.pow(2).sum(1, keepdim=True).sqrt()
    return (mask * norm.mm(norm.t())).detach()


def class_weighted_feat(cls_prob, pooled_feat):
    """train_target.py:620-621: flattened outer product prob (x) feature."""
    return torch.bmm(cls_prob.unsqueeze(2), pooled_feat.unsqueeze(1)).flatten(1)


def roi_cls_and_feat(roi_head, x, rois):
    """Rotated RoIAlign + shared FCs of RotatedShared2FCBBoxHead -> (cls_score, fc7)."""
    feats = roi_head.bbox_roi_extractor(x[:roi_head.bbox_roi_extractor.num_inputs], rois)
    head = roi_head.bbox_head
    h = feats.flatten(1)
    for fc in head.shared_fcs:
        h = head.relu(fc(h))
    return head.fc_cls(h), h


def valid_rotated_boxes(boxes):
    """Same rule as UnbiasedTeacher.create_pseudo_results: finite, w > 0, h > 0."""
    return torch.isfinite(boxes[:, :5]).all(1) & (boxes[:, 2] > 0) & (boxes[:, 3] > 0)


@DETECTORS.register_module()
class LODSTeacher(SemiTwoStageDetector):
    def __init__(self, backbone, rpn_head, roi_head, train_cfg, test_cfg,
                 neck=None, pretrained=None, ema_config=None, ema_ckpt=None,
                 cfg=dict()):
        super().__init__(backbone=backbone, rpn_head=rpn_head, roi_head=roi_head,
                         train_cfg=train_cfg, test_cfg=test_cfg, neck=neck,
                         pretrained=pretrained, ema_config=ema_config,
                         ema_ckpt=ema_ckpt)
        if not cfg.get('strict_source_free', False) or float(cfg.get('weight_l', 0)) != 0:
            raise ValueError('LODS-OBB runs only strict source-free (weight_l=0)')
        if cfg.get('use_bbox_reg', False):
            raise ValueError('official LODS never trains box regression')
        self.weight_u = float(cfg.get('weight_u', 1.0))
        self.momentum = float(cfg.get('momentum', 0.999))          # --alpha
        self.thresh = float(cfg.get('thresh', 0.8))                # --thresh
        self.gw_ins_weight = float(cfg.get('gw_ins_weight', 0.1))
        self.gw_global_weight = float(cfg.get('gw_global_weight', 0.1))
        self.label_weight = float(cfg.get('label_weight', 1.0))
        self.gw_add_rate = float(cfg.get('gw_add_rate', 0.5))
        self.style_add_alpha = float(cfg.get('style_add_alpha', 1.0))
        self.gw_global_level = int(cfg.get('gw_global_level', 2))  # C4 = res layer3
        self.num_classes = self.roi_head.bbox_head.num_classes
        self.cur_iter = 0
        self.sem = StyleEnhancementModule()
        # SEM trained beforehand on unlabeled target images (tools/lods/train_sem.py)
        self.sem.load_state_dict(torch.load(cfg['sem_ckpt'], map_location='cpu'))
        self.sem.requires_grad_(False).eval()
        self._style_files = {}

    def set_epoch(self, epoch):
        self.cur_epoch = epoch

    def train(self, mode=True):
        super().train(mode)
        self.sem.eval()
        return self

    def forward(self, img, img_metas, return_loss=True, **kwargs):
        if not return_loss:
            return self.forward_test(img, img_metas, **kwargs)
        # StrictSourceFreeDOTADataset: weak view = the original target image;
        # the *_unlabeled_1 view is not used by LODS. Target GT must be empty.
        for name in ('gt_bboxes', 'gt_labels'):
            if any(t.numel() for t in kwargs.get(name, [])):
                raise ValueError(f'strict source-free batch carries target {name}')
        return self.forward_train_lods(img, img_metas)

    # ----------------------------------------------------------- style
    def _random_style(self, img_meta, h, w, device):
        """enhance_base.load_style_img(random_style=True): a random target image."""
        root = osp.dirname(img_meta['filename'])
        if root not in self._style_files:
            self._style_files[root] = _discover_image_filenames(root)
        files = self._style_files[root]
        im = mmcv.imread(osp.join(root, files[random.randint(0, len(files) - 1)]))
        im = (im - PIXEL_MEANS_BGR).astype('float32')
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        return torch.from_numpy(im).permute(2, 0, 1)[None].to(device)

    def enhance(self, img, img_metas):
        """Student input: SEM(original) on the unpadded region, same size/geometry."""
        out = img.clone()
        for i, meta in enumerate(img_metas):
            h, w = meta['img_shape'][:2]
            norm = meta['img_norm_cfg']
            mean = img.new_tensor(norm['mean']).view(1, 3, 1, 1)
            std = img.new_tensor(norm['std']).view(1, 3, 1, 1)
            pix = img[i:i + 1, :, :h, :w] * std + mean
            bgr = pix.flip(1) if norm['to_rgb'] else pix
            means = img.new_tensor(PIXEL_MEANS_BGR).view(1, 3, 1, 1)
            style = self._random_style(meta, h, w, img.device)
            y = self.sem.add_style(bgr - means, style, self.style_add_alpha) + means
            y = y.flip(1) if norm['to_rgb'] else y
            out[i:i + 1, :, :h, :w] = (y - mean) / std
        return out

    # ----------------------------------------------------------- train
    def forward_train_lods(self, img, img_metas):
        self.update_ema_model(self.momentum)
        self.cur_iter += 1
        teacher = getattr(self.ema_model, 'module', self.ema_model)
        with torch.no_grad():
            c_t = teacher.backbone(img)
            x_t = teacher.neck(c_t)
            # official: teacher RCNN_rpn.eval() -> test-mode proposals
            proposals = teacher.rpn_head.simple_test_rpn(x_t, img_metas)
            proposals = [p[valid_rotated_boxes(p)] for p in proposals]
            rois = rbbox2roi([p[:, :5] for p in proposals])
            cls_t, feat_t = roi_cls_and_feat(teacher.roi_head, x_t, rois)

        img_s = self.enhance(img, img_metas)
        c_s = self.backbone(img_s)
        x_s = self.neck(c_s)
        cls_s, feat_s = roi_cls_and_feat(self.roi_head, x_s, rois)  # rois = rois_t

        prob_t = F.softmax(cls_t, 1)
        conf_t, labels_t = prob_t.max(1)
        keep = conf_t >= self.thresh
        if keep.any():
            label_loss = F.cross_entropy(cls_s[keep], labels_t[keep])
        else:
            label_loss = cls_s.sum() * 0  # official: except -> zeros(1)

        prob_s = F.softmax(cls_s, 1)
        fs_all = class_weighted_feat(prob_s, feat_s)
        ft_all = class_weighted_feat(prob_t, feat_t)
        gw_ins, gw_glob = [], []
        for b, meta in enumerate(img_metas):
            idx = rois[:, 0] == b
            if idx.any():
                t = instance_relation_prior(cls_t[idx], labels_t[idx], conf_t[idx],
                                            self.thresh, self.num_classes)
                gw_ins.append(gromov_wasserstein_loss(fs_all[idx], ft_all[idx], t,
                                                      self.gw_add_rate))
            # pixel graph on the unpadded part of the stride-16 backbone map
            fmap_s = c_s[self.gw_global_level][b]
            fmap_t = c_t[self.gw_global_level][b]
            stride = img.size(-1) / fmap_s.size(-1)
            hh = int(-(-meta['img_shape'][0] // stride))
            ww = int(-(-meta['img_shape'][1] // stride))
            ns = fmap_s[:, :hh, :ww].flatten(1).t()
            nt = fmap_t[:, :hh, :ww].flatten(1).t()
            gw_glob.append(gromov_wasserstein_loss(ns, nt, None, self.gw_add_rate))
        gw_ins_loss = torch.stack(gw_ins).mean() if gw_ins else cls_s.sum() * 0
        gw_global_loss = torch.stack(gw_glob).mean()

        w = self.weight_u
        return {
            'loss_lods_label': w * self.label_weight * label_loss,
            'loss_lods_gw_ins': w * self.gw_ins_weight * gw_ins_loss,
            'loss_lods_gw_global': w * self.gw_global_weight * gw_global_loss,
            'lods_rois_per_img': rois.new_tensor([rois.size(0) / len(img_metas)]),
            'lods_conf_frac': keep.float().mean().reshape(1),
            'lods_conf_fg_frac': (keep & (labels_t != self.num_classes)).float().mean().reshape(1),
        }
