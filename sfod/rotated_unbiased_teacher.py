# Copyright (c) Hangzhou Hikvision Digital Technology Co., Ltd. All rights reserved.
# Modified from https://github.com/open-mmlab/mmdetection
"""
Re-implementation: Unbiased teacher for semi-supervised object detection

There are several differences with official implementation:
1. we only use the strong-augmentation version of labeled data rather than \
the strong-augmentation and weak-augmentation version of labeled data.
"""
from collections import deque
import os

import numpy as np
import torch

import cv2
import mmcv
from mmcv.runner.dist_utils import get_dist_info

from mmdet.utils import get_root_logger
from mmdet.models.builder import DETECTORS
from mmrotate.core.bbox import rbbox_overlaps

from .rotated_semi_two_stage import SemiTwoStageDetector
from mmrotate.core.visualization import imshow_det_rbboxes

@DETECTORS.register_module()
class UnbiasedTeacher(SemiTwoStageDetector):
    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 # ema model
                 ema_config=None,
                 ema_ckpt=None,
                 # ut config
                 cfg=dict(),
                 ):
        super().__init__(backbone=backbone, rpn_head=rpn_head, roi_head=roi_head, train_cfg=train_cfg,
                         test_cfg=test_cfg, neck=neck, pretrained=pretrained,
                         ema_config=ema_config, ema_ckpt=ema_ckpt)
        self.debug = cfg.get('debug', False)
        self.vis_dir = cfg.get('vis_dir', None)
        self.num_classes = self.roi_head.bbox_head.num_classes
        self.cur_iter = 0
        
        # hyper-parameter
        self.score_thr = cfg.get('score_thr', 0.7)
        self.weight_u = cfg.get('weight_u', 1.0)
        self.weight_l = cfg.get('weight_l', 0.0)
        self.use_bbox_reg = cfg.get('use_bbox_reg', False)
        self.momentum = cfg.get('momentum', 0.998)

        # SARCLIP Semantic Reliability Reweighting (SRW).  When enabled, pseudo
        # admission still uses the RAW detector score (score_thr); SARCLIP only
        # supplies a per-pseudo-GT semantic weight that scales the ROI positive
        # classification loss (see SemanticWeightedOrientedStandardRoIHead).
        # The weights themselves are produced by CGA_FILTER_MODE=semantic_reweight
        # in sfod/cga.py; these cfg values make the config self-documenting and
        # seed the CGA env vars when they are not explicitly set.
        self.semantic_reweight = bool(cfg.get('semantic_reweight', False))
        self.semantic_low_thr = float(cfg.get('semantic_low_thr', 0.90))
        self.semantic_high_thr = float(cfg.get('semantic_high_thr', 0.95))
        self.semantic_lambda = float(cfg.get('semantic_lambda', 0.50))
        if self.semantic_reweight:
            if self.semantic_high_thr <= self.semantic_low_thr:
                raise ValueError(
                    'semantic_high_thr must be strictly greater than '
                    'semantic_low_thr')
            os.environ.setdefault('CGA_SEM_LOW_THR', str(self.semantic_low_thr))
            os.environ.setdefault(
                'CGA_SEM_HIGH_THR', str(self.semantic_high_thr))
            os.environ.setdefault('CGA_SEM_LAMBDA', str(self.semantic_lambda))

        # Stable class-wise thresholding inspired by ConsistentTeacher's
        # adaptive-threshold idea.  This intentionally uses an EMA quantile,
        # not the paper's two-component GMM, because the latter is not
        # justified unless the online score distribution is demonstrably
        # bimodal.  The feature is opt-in so historical configurations retain
        # their exact fixed-threshold behaviour.
        self.dynamic_threshold_enabled = bool(
            cfg.get('dynamic_threshold', False))
        self.dynamic_threshold_quantile = float(
            cfg.get('dynamic_threshold_quantile', 0.10))
        self.dynamic_threshold_momentum = float(
            cfg.get('dynamic_threshold_momentum', 0.90))
        self.dynamic_threshold_min = float(
            cfg.get('dynamic_threshold_min', self.score_thr))
        self.dynamic_threshold_max = float(
            cfg.get('dynamic_threshold_max', max(self.score_thr, 0.95)))
        self.dynamic_threshold_queue_size = int(
            cfg.get('dynamic_threshold_queue_size', 100))
        self.dynamic_threshold_min_samples = int(
            cfg.get('dynamic_threshold_min_samples', 20))
        if self.dynamic_threshold_enabled:
            self._validate_dynamic_threshold_config()
        self.dynamic_score_thresholds = np.full(
            self.num_classes, self.score_thr, dtype=np.float64)
        self.dynamic_threshold_targets = np.full(
            self.num_classes, self.score_thr, dtype=np.float64)
        self.dynamic_threshold_new_samples = np.zeros(
            self.num_classes, dtype=np.int64)
        self.dynamic_score_queues = [
            deque(maxlen=self.dynamic_threshold_queue_size)
            for _ in range(self.num_classes)
        ]

        # analysis
        self.image_num = 0
        self.pseudo_num = np.zeros(self.num_classes)
        self.pseudo_num_tp = np.zeros(self.num_classes)
        self.pseudo_num_gt = np.zeros(self.num_classes)
        # SRW analysis: per-class sum of semantic weights over admitted pseudo
        # boxes (effective pseudo count), plus disagreement bookkeeping.
        self.pseudo_sem_weight = np.zeros(self.num_classes)
        self.srw_disagree = 0
        self.srw_disagree_low = 0   # admitted, disagree, det_score in [low,high)
        self.srw_disagree_high = 0  # admitted, disagree, det_score >= high

    def _validate_dynamic_threshold_config(self):
        if not 0.0 <= self.dynamic_threshold_quantile <= 1.0:
            raise ValueError('dynamic_threshold_quantile must be in [0, 1]')
        if not 0.0 <= self.dynamic_threshold_momentum < 1.0:
            raise ValueError('dynamic_threshold_momentum must be in [0, 1)')
        if not self.score_thr <= self.dynamic_threshold_min <= 1.0:
            raise ValueError(
                'dynamic_threshold_min must be in [score_thr, 1]')
        if not self.dynamic_threshold_min <= self.dynamic_threshold_max <= 1.0:
            raise ValueError(
                'dynamic_threshold_max must be in '
                '[dynamic_threshold_min, 1]')
        if self.dynamic_threshold_queue_size <= 0:
            raise ValueError('dynamic_threshold_queue_size must be positive')
        if not 1 <= self.dynamic_threshold_min_samples <= \
                self.dynamic_threshold_queue_size:
            raise ValueError(
                'dynamic_threshold_min_samples must be in [1, queue_size]')

    def _update_dynamic_score_thresholds(self, bbox_results):
        """Update per-class EMA quantile thresholds from teacher predictions."""
        if not getattr(self, 'dynamic_threshold_enabled', False):
            return

        new_samples = np.zeros(self.num_classes, dtype=np.int64)
        for image_results in bbox_results:
            if len(image_results) != self.num_classes:
                raise ValueError(
                    'teacher result class count does not match num_classes')
            for cls, result in enumerate(image_results):
                if len(result) == 0:
                    continue
                scores = np.asarray(result)[:, -1].astype(
                    np.float64, copy=False)
                valid = scores[np.isfinite(scores) & (scores >= self.score_thr)]
                self.dynamic_score_queues[cls].extend(valid.tolist())
                new_samples[cls] += len(valid)

        self.dynamic_threshold_new_samples = new_samples

        for cls, score_queue in enumerate(self.dynamic_score_queues):
            if (new_samples[cls] == 0
                    or len(score_queue) < self.dynamic_threshold_min_samples):
                continue
            values = np.fromiter(score_queue, dtype=np.float64)
            target = float(np.quantile(
                values, self.dynamic_threshold_quantile))
            target = float(np.clip(
                target,
                self.dynamic_threshold_min,
                self.dynamic_threshold_max,
            ))
            self.dynamic_threshold_targets[cls] = target
            old = float(self.dynamic_score_thresholds[cls])
            momentum = self.dynamic_threshold_momentum
            updated = momentum * old + (1.0 - momentum) * target
            self.dynamic_score_thresholds[cls] = float(np.clip(
                updated,
                self.dynamic_threshold_min,
                self.dynamic_threshold_max,
            ))

    def _pseudo_score_threshold(self, cls):
        if getattr(self, 'dynamic_threshold_enabled', False):
            return float(self.dynamic_score_thresholds[cls])
        return float(self.score_thr)

    def set_epoch(self, epoch): 
        self.roi_head.cur_epoch = epoch 
        self.roi_head.bbox_head.cur_epoch = epoch
        self.cur_epoch = epoch
        
    def forward_train_semi(
            self, img, img_metas, gt_bboxes, gt_labels,
            img_unlabeled, img_metas_unlabeled, gt_bboxes_unlabeled, gt_labels_unlabeled,
            img_unlabeled_1, img_metas_unlabeled_1, gt_bboxes_unlabeled_1, gt_labels_unlabeled_1,
    ):
        device = img.device
        self.image_num += len(img_metas_unlabeled)
        self.update_ema_model(self.momentum)
        self.cur_iter += 1
        # # ---------------------label data---------------------
        losses = self.forward_train(img, img_metas, gt_bboxes, gt_labels)
        losses = self.parse_loss(losses)
        for key, val in losses.items():
            if key.find('loss') == -1:
                continue
            else:
                losses[key] = self.weight_l * val
        # # -------------------unlabeled data-------------------
        bbox_transform = []
        if self.semantic_reweight:
            bbox_results, cga_meta = self.inference_unlabeled(
                img_unlabeled, img_metas_unlabeled, rescale=True,
                return_cga_meta=True
            )
            gt_bboxes_pred, gt_labels_pred, gt_semantic_weights_pred = \
                self.create_pseudo_results(
                    img_unlabeled_1, bbox_results, bbox_transform, device,
                    gt_bboxes_unlabeled, gt_labels_unlabeled,
                    img_metas_unlabeled,  # for analysis
                    cga_meta=cga_meta, return_semantic_weights=True
                )
            self.analysis()
            losses_unlabeled = self.forward_train_semantic_weighted(
                img_unlabeled_1, img_metas_unlabeled_1,
                gt_bboxes_pred, gt_labels_pred, gt_semantic_weights_pred)
        else:
            bbox_results = self.inference_unlabeled(
                img_unlabeled, img_metas_unlabeled, rescale=True
            )
            gt_bboxes_pred, gt_labels_pred = self.create_pseudo_results(
                img_unlabeled_1, bbox_results, bbox_transform, device,
                gt_bboxes_unlabeled, gt_labels_unlabeled, img_metas_unlabeled  # for analysis
            )
            self.analysis()
            losses_unlabeled = self.forward_train(img_unlabeled_1, img_metas_unlabeled_1,
                                                  gt_bboxes_pred, gt_labels_pred)
        losses_unlabeled = self.parse_loss(losses_unlabeled)
        for key, val in losses_unlabeled.items():
            if key.find('loss') == -1:
                continue
            if key.find('bbox') != -1:
                losses_unlabeled[key] = self.weight_u * val if self.use_bbox_reg else 0 * val
            else:
                losses_unlabeled[key] = self.weight_u * val
        losses.update({f'{key}_unlabeled': val for key, val in losses_unlabeled.items()})
        extra_info = {
            'pseudo_num': torch.Tensor([self.pseudo_num.sum() / self.image_num]).to(device),
            'pseudo_num(acc)': torch.Tensor([self.pseudo_num_tp.sum() / self.pseudo_num.sum()]).to(device)
        }
        if self.semantic_reweight:
            # effective pseudo count = sum of semantic weights / image_num
            extra_info['pseudo_effective_num'] = torch.Tensor(
                [self.pseudo_sem_weight.sum() / self.image_num]).to(device)
        losses.update(extra_info)
        return losses

    def forward_train_semantic_weighted(
            self, img, img_metas, gt_bboxes, gt_labels, gt_semantic_weights):
        """Unlabeled forward for SRW.

        Runs the standard two-stage pipeline explicitly so that the semantic
        weights reach ONLY the ROI head.  RPN classification / bbox regression
        never see the semantic weights.
        """
        x = self.extract_feat(img)
        losses = dict()

        # RPN forward and loss -- NO semantic weights passed here.
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=None,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            raise RuntimeError(
                'forward_train_semantic_weighted requires an RPN head')

        # ROI forward and loss -- semantic weights applied to positive ROIs.
        roi_losses = self.roi_head.forward_train(
            x, img_metas, proposal_list, gt_bboxes, gt_labels,
            gt_bboxes_ignore=None, gt_masks=None,
            gt_semantic_weights=gt_semantic_weights)
        losses.update(roi_losses)
        return losses
    
    def create_pseudo_results(self, img, bbox_results, box_transform, device,
                              gt_bboxes=None, gt_labels=None, img_metas=None,
                              cga_meta=None, return_semantic_weights=False):
        """using dynamic score to create pseudo results.

        When ``return_semantic_weights`` is True, also returns per-image, per-GT
        semantic weights aligned 1:1 with the admitted pseudo boxes/labels.  The
        RAW detector score (``r[:, -1]``) always decides admission; SARCLIP
        never rescales it here.  Missing/None ``cga_meta`` -> all weights 1.0.
        """
        gt_bboxes_pred, gt_labels_pred = [], []
        gt_semantic_weights_pred = []
        _, _, h, w = img.shape
        use_gt = gt_bboxes is not None
        self._update_dynamic_score_thresholds(bbox_results)
        for b, result in enumerate(bbox_results):
            bboxes, labels, sem_weights = [], [], []
            image_meta = cga_meta[b] if (cga_meta is not None
                                         and b < len(cga_meta)
                                         and cga_meta[b] is not None) else None
            if use_gt:
                gt_bbox, gt_label = gt_bboxes[b].cpu().numpy(), gt_labels[b].cpu().numpy()
                scale_factor = img_metas[b]['scale_factor']
                gt_bbox_scale = gt_bbox.copy()
                gt_bbox_scale[:,:4] = gt_bbox[:,:4] / scale_factor
            for cls, r in enumerate(result):
                label = cls * np.ones_like(r[:, 0], dtype=np.uint8)
                flag = r[:, -1] >= self._pseudo_score_threshold(cls)
                # print(flag)
                bboxes.append(r[flag][:, :-1])
                labels.append(label[flag])
                # Semantic weight per admitted box (same row selection as flag).
                if image_meta is not None and len(image_meta["semantic_weight"][cls]) == len(r):
                    cls_weights = np.asarray(
                        image_meta["semantic_weight"][cls], dtype=np.float64)[flag]
                    cls_disagree = ~np.asarray(
                        image_meta["agreement"][cls], dtype=bool)[flag]
                    cls_det_score = np.asarray(
                        image_meta["det_score"][cls], dtype=np.float64)[flag]
                    # SRW disagreement bookkeeping over ADMITTED boxes only.
                    self.srw_disagree += int(cls_disagree.sum())
                    high = self.semantic_high_thr
                    low = self.semantic_low_thr
                    in_low = cls_disagree & (cls_det_score >= low) & (cls_det_score < high)
                    in_high = cls_disagree & (cls_det_score >= high)
                    self.srw_disagree_low += int(in_low.sum())
                    self.srw_disagree_high += int(in_high.sum())
                else:
                    cls_weights = np.ones(int(flag.sum()), dtype=np.float64)
                sem_weights.append(cls_weights)
                self.pseudo_sem_weight[cls] += float(cls_weights.sum())
                if use_gt and (gt_label == cls).sum() > 0 and len(bboxes[-1]) > 0:
                    overlap = rbbox_overlaps(torch.tensor(bboxes[-1]), torch.tensor(gt_bbox_scale[gt_label == cls]))
                    self.pseudo_num_tp[cls] += (torch.max(overlap,dim=1)[0] > 0.5).sum()
                if use_gt:
                    self.pseudo_num_gt[cls] += (gt_label == cls).sum()
                self.pseudo_num[cls] += len(bboxes[-1])
            bboxes = np.concatenate(bboxes)
            labels = np.concatenate(labels)
            sem_weights = np.concatenate(sem_weights) if sem_weights else np.ones(0, dtype=np.float64)
            gt_bboxes_pred.append(torch.from_numpy(bboxes).float().to(device))
            gt_labels_pred.append(torch.from_numpy(labels).long().to(device))
            gt_semantic_weights_pred.append(
                torch.from_numpy(sem_weights).float().to(device))
        if return_semantic_weights:
            return gt_bboxes_pred, gt_labels_pred, gt_semantic_weights_pred
        return gt_bboxes_pred, gt_labels_pred

    def analysis(self):
        if self.cur_iter % 500 == 0 and get_dist_info()[0] == 0:
            logger = get_root_logger()
            info = ' '.join([f'{b / (a + 1e-10):.2f}({a}-{cls})' for cls, a, b
                             in zip(self.CLASSES, self.pseudo_num, self.pseudo_num_tp)])
            info_gt = ' '.join([f'{a}' for a in self.pseudo_num_gt])
            logger.info(f'pseudo pos: {info}')
            logger.info(f'pseudo gt: {info_gt}')
            if getattr(self, 'semantic_reweight', False):
                admitted = float(self.pseudo_num.sum())
                eff = float(self.pseudo_sem_weight.sum())
                mean_w = eff / admitted if admitted > 0 else 0.0
                logger.info(
                    'SRW: admitted_pseudo=%d effective_pseudo=%.2f '
                    'mean_pos_weight=%.4f disagree=%d '
                    'disagree[%.2f,%.2f)=%d disagree>=%.2f=%d'
                    % (int(admitted), eff, mean_w, int(self.srw_disagree),
                       self.semantic_low_thr, self.semantic_high_thr,
                       int(self.srw_disagree_low), self.semantic_high_thr,
                       int(self.srw_disagree_high)))
                per_class = ' '.join(
                    '%s:n=%d,w=%.2f,mw=%.3f' % (
                        cls, int(n), sw, (sw / n if n > 0 else 0.0))
                    for cls, n, sw in zip(
                        self.CLASSES, self.pseudo_num, self.pseudo_sem_weight))
                logger.info(f'SRW per-class: {per_class}')
            if getattr(self, 'dynamic_threshold_enabled', False):
                thresholds = ' '.join(
                    f'{cls}={threshold:.5f}'
                    for cls, threshold in zip(
                        self.CLASSES, self.dynamic_score_thresholds))
                queue_counts = ' '.join(
                    f'{cls}={len(queue)}'
                    for cls, queue in zip(
                        self.CLASSES, self.dynamic_score_queues))
                targets = ' '.join(
                    f'{cls}={target:.5f}'
                    for cls, target in zip(
                        self.CLASSES, self.dynamic_threshold_targets))
                new_samples = ' '.join(
                    f'{cls}={count}'
                    for cls, count in zip(
                        self.CLASSES, self.dynamic_threshold_new_samples))
                logger.info(
                    f'dynamic score thresholds: {thresholds}')
                logger.info(
                    f'dynamic score targets: {targets}')
                logger.info(
                    f'dynamic score queue counts: {queue_counts}')
                logger.info(
                    f'dynamic score new samples: {new_samples}')
            
    def show_result(self,
                img,
                result,
                score_thr=0.3,
                bbox_color=(72, 101, 241),
                text_color=(72, 101, 241),
                mask_color=None,
                thickness=4,
                font_size=13,
                win_name='',
                show=False,
                wait_time=0,
                out_file=None):

        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        # draw segmentation masks
        segms = None
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            if isinstance(segms[0], torch.Tensor):
                segms = torch.stack(segms, dim=0).detach().cpu().numpy()
            else:
                segms = np.stack(segms, axis=0)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False

        PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
               (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
               (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
               (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
               (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157)]
        imshow_det_rbboxes(
            img,
            bboxes,
            labels,
            class_names=self.CLASSES,
            # class_names=None,
            score_thr=score_thr,
            show=show,
            wait_time=wait_time,
            out_file=out_file,
            thickness=4,
            font_size=20,
            bbox_color=PALETTE,
            text_color=(200, 200, 200))

        if not (show or out_file):
            return img
