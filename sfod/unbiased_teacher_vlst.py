"""UnbiasedTeacher with Vision-Language Semantic Teacher integration."""
import os
import torch

from mmdet.models.builder import DETECTORS
from .rotated_unbiased_teacher import UnbiasedTeacher
from .semantic_teacher.prototype_teacher import SemanticPrototypeTeacher


@DETECTORS.register_module()
class UnbiasedTeacherVLST(UnbiasedTeacher):
    """UnbiasedTeacher with Vision-Language Semantic Teacher.

    Adds feature-level semantic guidance via CLIP/SARCLIP prototypes on top
    of the existing label-level CGA guidance.
    """

    def __init__(self, *args, cfg=dict(), **kwargs):
        super().__init__(*args, cfg=cfg, **kwargs)

        # VLST config
        self.vlst_enabled = bool(cfg.get('vlst_enabled', False))
        if not self.vlst_enabled:
            self.vlst_teacher = None
            return

        self.vlst_loss_weight = float(cfg.get('vlst_loss_weight', 0.1))
        self.vlst_temperature = float(cfg.get('vlst_temperature', 0.07))
        self.vlst_prototype_momentum = float(cfg.get('vlst_prototype_momentum', 0.9))
        self.vlst_text_visual_alpha = float(cfg.get('vlst_text_visual_alpha', 0.5))
        self.vlst_score_thr = cfg.get('vlst_score_thr', None)
        if self.vlst_score_thr is None:
            self.vlst_score_thr = self.score_thr

        # Detector ROI feature dimension (from Oriented R-CNN bbox head)
        # RotatedShared2FCBBoxHead: fc_out_channels=1024 (default in configs)
        vlst_detector_dim = int(cfg.get('vlst_detector_dim', 1024))
        vlst_vlm_dim = int(cfg.get('vlst_vlm_dim', 512))  # ViT-B-32 SARCLIP
        vlst_projection_hidden = int(cfg.get('vlst_projection_hidden', 256))

        # Initialize Semantic Prototype Teacher
        self.vlst_teacher = SemanticPrototypeTeacher(
            num_classes=self.num_classes,
            vlm_dim=vlst_vlm_dim,
            detector_dim=vlst_detector_dim,
            projection_hidden=vlst_projection_hidden,
            text_visual_alpha=self.vlst_text_visual_alpha,
            prototype_momentum=self.vlst_prototype_momentum,
            temperature=self.vlst_temperature,
            score_threshold=self.vlst_score_thr,
        )

        # VLM used by VLST for text prototypes and instance embeddings. Kept
        # separate from the CGA branch so arm C (CGA off) still gets visual
        # prototypes. Plain attribute, not a submodule: it is frozen and must
        # not enter the optimizer or the EMA state dict.
        self._vlst_vlm = None

        # Diagnostics
        self.vlst_loss_sum = 0.0
        self.vlst_loss_count = 0
        self.vlst_sample_sum = 0

    def _init_vlst_text_prototypes(self):
        """Lazy initialization of text prototypes from VLM directly."""
        if self.vlst_teacher is None or self.vlst_teacher.text_prototypes is not None:
            return

        # Try to get from CGA first (if available)
        ema_host = getattr(self.ema_model, 'module', self.ema_model)

        if hasattr(ema_host, 'cga') and ema_host.cga is not None:
            try:
                text_proto = ema_host.cga.text_prototype_matrix()  # (C, D)
                self.vlst_teacher.set_text_prototypes(text_proto)
                print(f"[VLST] Initialized text prototypes from CGA: shape={text_proto.shape}")
                return
            except Exception as e:
                print(f"[VLST] Warning: Failed to get text prototypes from CGA: {e}")

        # Fall back to building text prototypes directly from VLM
        try:
            import os
            from sfod.cga import CGA, DIOR_CLASSES, RSAR_CLASSES

            # Resolve class names the same way CGA does, so the text prompts
            # match the dataset. Never hardcode a class list here: a wrong list
            # silently builds prototypes for the wrong semantics.
            class_names = getattr(ema_host, 'CLASSES', None)
            if class_names is None or len(class_names) != self.num_classes:
                if self.num_classes == len(RSAR_CLASSES):
                    class_names = list(RSAR_CLASSES)
                elif self.num_classes == len(DIOR_CLASSES):
                    class_names = list(DIOR_CLASSES)
                else:
                    print(f"[VLST] Warning: Cannot infer class names for {self.num_classes} classes")
                    return
            class_names = list(class_names)

            # Pick the VLM backend. CGA_SCORER=none only disables the
            # label-level CGA branch — the feature-level prototype teacher is
            # an independent arm (ST+Proto) and must still get its VLM.
            vlm_type = os.environ.get('VLST_BACKEND', '').strip().lower()
            if not vlm_type:
                vlm_type = os.environ.get('CGA_SCORER', '').strip().lower()
            if vlm_type in ('', 'none', 'false', '0', 'raw'):
                vlm_type = 'sarclip'

            # Build CGA instance directly
            vlm = CGA(
                class_names=class_names,
                backend=vlm_type,
                model="ViT-B-32",
                pretrained="/myfile/pretrain/SARCLIP/ViT-B-32/vit_b_32_model.safetensors",
                cache_dir="/myfile/pretrain/SARCLIP/ViT-B-32",
            )

            # Keep it for instance-feature extraction (visual prototypes).
            self._vlst_vlm = vlm

            # Extract text prototypes
            text_proto = vlm.text_prototype_matrix()  # (C, D)
            text_proto_tensor = torch.from_numpy(text_proto).float()

            self.vlst_teacher.set_text_prototypes(text_proto_tensor)
            print(f"[VLST] Initialized text prototypes directly from {vlm_type}: "
                  f"shape={text_proto_tensor.shape}, classes={class_names}")

        except Exception as e:
            print(f"[VLST] Warning: Failed to build text prototypes directly: {e}")
            import traceback
            traceback.print_exc()

    def forward_train_semi(
            self, img, img_metas, gt_bboxes, gt_labels,
            img_unlabeled, img_metas_unlabeled, gt_bboxes_unlabeled, gt_labels_unlabeled,
            img_unlabeled_1, img_metas_unlabeled_1, gt_bboxes_unlabeled_1, gt_labels_unlabeled_1,
    ):
        """Override to add VLST prototype loss."""
        if not hasattr(self, '_vlst_forward_count'):
            self._vlst_forward_count = 0
        self._vlst_forward_count += 1

        if not self.vlst_enabled:
            if self._vlst_forward_count == 1:
                print(f"[VLST] VLST is disabled, falling back to parent forward_train_semi")
            return super().forward_train_semi(
                img, img_metas, gt_bboxes, gt_labels,
                img_unlabeled, img_metas_unlabeled, gt_bboxes_unlabeled, gt_labels_unlabeled,
                img_unlabeled_1, img_metas_unlabeled_1, gt_bboxes_unlabeled_1, gt_labels_unlabeled_1,
            )

        if self._vlst_forward_count == 1:
            print(f"[VLST] VLST is enabled, using VLST forward path")

        device = img.device
        self.image_num += len(img_metas_unlabeled)
        self.update_ema_model(self.momentum)
        self.cur_iter += 1

        # Initialize text prototypes once CGA is available
        self._init_vlst_text_prototypes()

        if self._vlst_forward_count <= 3:
            text_ready = self.vlst_teacher is not None and self.vlst_teacher.text_prototypes is not None
            print(f"[VLST] Iter {self._vlst_forward_count}: text_prototypes ready = {text_ready}")

        # Labeled data loss
        losses = self.forward_train(img, img_metas, gt_bboxes, gt_labels)
        losses = self.parse_loss(losses)
        for key, val in losses.items():
            if key.find('loss') == -1:
                continue
            else:
                losses[key] = self.weight_l * val

        # Teacher inference on weak view
        bbox_results = self.inference_unlabeled(
            img_unlabeled, img_metas_unlabeled, rescale=True)

        # Create pseudo labels
        gt_bboxes_pred, gt_labels_pred = self.create_pseudo_results(
            img_unlabeled_1, bbox_results, [], device,
            gt_bboxes_unlabeled, gt_labels_unlabeled, img_metas_unlabeled)
        self.analysis()

        # VLST: Extract VLM features and update prototypes
        if self.vlst_teacher is not None and self.vlst_teacher.text_prototypes is not None:
            self._vlst_update_prototypes(
                img_metas_unlabeled, bbox_results)

        # Student forward with prototype loss
        losses_unlabeled = self._forward_train_with_vlst(
            img_unlabeled_1, img_metas_unlabeled_1,
            gt_bboxes_pred, gt_labels_pred, bbox_results)

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
        if self.vlst_loss_count > 0:
            # NOTE: key must NOT contain "loss" — mmdet parse_losses sums every
            # key that matches, so a diagnostic average would be double-counted.
            extra_info['vlst_avg'] = torch.Tensor(
                [self.vlst_loss_sum / self.vlst_loss_count]).to(device)
            extra_info['vlst_samples'] = torch.Tensor(
                [self.vlst_sample_sum / self.vlst_loss_count]).to(device)
            diag = self.vlst_teacher.get_diagnostics()
            # Separability of the student ROI embedding w.r.t. the prototypes.
            # margin > 0 and rising is the signal that feature-level guidance
            # is actually taking hold.
            extra_info['vlst_margin'] = torch.Tensor([diag['margin']]).to(device)
            extra_info['vlst_pos_cos'] = torch.Tensor([diag['pos_cos']]).to(device)
            extra_info['vlst_proto_cls'] = torch.Tensor(
                [float((diag['visual_counts'] > 0).sum())]).to(device)

        losses.update(extra_info)
        return losses

    def _vlst_update_prototypes(self, img_metas, bbox_results):
        """Extract VLM features and update visual prototypes."""
        ema_host = getattr(self.ema_model, 'module', self.ema_model)

        # Always use VLST's own frozen VLM, never the CGA branch's instance.
        # CGA may carry a LoRA-tuned SARCLIP whose image embeddings live in a
        # slightly different space; mixing it with base-SARCLIP text prototypes
        # would corrupt the contrastive targets. Keeping VLST self-contained
        # also makes arms C and D differ ONLY in CGA on/off.
        vlm = self._vlst_vlm
        if vlm is None:
            return

        for img_meta, result in zip(img_metas, bbox_results):
            # Flatten per-image detections
            inputs = ema_host._flatten_cga_inputs(result)
            if inputs is None:
                continue
            boxes, scores, labels = inputs

            # Drop degenerate AABBs: PIL crop raises (and the try/except would
            # abort prototype updates for the WHOLE image) when a rotated box
            # converts to a zero-width/height AABB.
            widths = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]
            valid = (widths >= 1.0) & (heights >= 1.0)
            if not valid.any():
                continue
            boxes = boxes[valid]
            scores = scores[valid]
            labels = labels[valid]

            # Frozen VLM instance embeddings (no grad, spec: VLM stays frozen)
            try:
                vlm_features, _, _ = vlm.forward_aabb_embed(
                    img_meta['filename'], boxes, scores, labels)
                if vlm_features.shape[0] == 0:
                    continue

                # Convert to tensors
                vlm_features_t = torch.from_numpy(vlm_features).float().to(scores.device if hasattr(scores, 'device') else 'cuda')
                labels_t = torch.from_numpy(labels).long().to(vlm_features_t.device)
                scores_t = torch.from_numpy(scores).float().to(vlm_features_t.device)

                # Update prototypes
                with torch.no_grad():
                    self.vlst_teacher.update_visual_prototypes(
                        vlm_features_t, labels_t, scores_t, score_thr=self.vlst_score_thr)
            except Exception as e:
                print(f"[VLST] Warning: prototype update failed: {e}")
                continue

    def _forward_train_with_vlst(
            self, img, img_metas, gt_bboxes, gt_labels, teacher_results):
        """Student forward + VLST prototype loss."""
        x = self.extract_feat(img)
        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x, img_metas, gt_bboxes, gt_labels=None,
                gt_bboxes_ignore=None, proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            raise RuntimeError('VLST requires RPN head')

        # ROI forward and loss + VLST prototype loss
        roi_losses, vlst_loss = self._roi_forward_with_vlst(
            x, img_metas, proposal_list, gt_bboxes, gt_labels, teacher_results)
        losses.update(roi_losses)
        if vlst_loss is not None:
            losses['loss_vlst_proto'] = vlst_loss
            if self._vlst_forward_count <= 3:
                print(f"[VLST] Iter {self._vlst_forward_count}: loss_vlst_proto = {vlst_loss.item():.4f}")
        else:
            if self._vlst_forward_count <= 3:
                print(f"[VLST] Iter {self._vlst_forward_count}: vlst_loss is None")

        return losses

    def _roi_forward_with_vlst(
            self, x, img_metas, proposal_list, gt_bboxes, gt_labels, teacher_results):
        """ROI forward with VLST prototype loss extraction."""
        from mmrotate.core import rbbox2roi

        # Standard ROI head assignment and sampling
        if gt_bboxes[0].numel() == 0:
            gt_bboxes_ignore = [None] * len(img_metas)
        else:
            gt_bboxes_ignore = [None] * len(img_metas)

        sampling_results = []
        for i in range(len(img_metas)):
            assign_result = self.roi_head.bbox_assigner.assign(
                proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i])
            sampling_result = self.roi_head.bbox_sampler.sample(
                assign_result, proposal_list[i], gt_bboxes[i], gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])

            if gt_bboxes[i].numel() == 0:
                sampling_result.pos_gt_bboxes = gt_bboxes[i].new(
                    (0, gt_bboxes[0].size(-1))).zero_()
            else:
                sampling_result.pos_gt_bboxes = gt_bboxes[i][sampling_result.pos_assigned_gt_inds, :]
            sampling_results.append(sampling_result)

        # ROI feature extraction
        rois = rbbox2roi([res.bboxes for res in sampling_results])
        bbox_feats = self.roi_head.bbox_roi_extractor(
            x[:self.roi_head.bbox_roi_extractor.num_inputs], rois)
        if self.roi_head.with_shared_head:
            bbox_feats = self.roi_head.shared_head(bbox_feats)

        # Extract Student ROI features through bbox_head fc layers
        # We need the fc output (1024-dim) not the raw bbox_feats
        bbox_head = self.roi_head.bbox_head

        # Forward through bbox_head to get both predictions and fc features
        if bbox_feats.dim() == 4:
            # Flatten spatial dimensions if needed
            bbox_feats_flat = bbox_feats.flatten(1)
        else:
            bbox_feats_flat = bbox_feats

        # Get fc features for VLST. This path is intentionally NOT detached:
        # the method requires grad L_proto -> Student ROI feature -> Student
        # detector. Detaching would confine the loss to the projection head,
        # making the arm a no-op. (The earlier pseudo-label collapse traced to
        # text prototypes built from the wrong class names, not to this path.)
        x_fc = bbox_feats_flat
        if hasattr(bbox_head, 'shared_fcs') and bbox_head.shared_fcs is not None:
            for fc in bbox_head.shared_fcs:
                x_fc = bbox_head.relu(fc(x_fc))

        # Get predictions
        cls_score, bbox_pred = bbox_head(bbox_feats)

        # Compute standard detection loss
        bbox_targets = bbox_head.get_targets(
            sampling_results, gt_bboxes, gt_labels, self.train_cfg.rcnn)
        loss_bbox = bbox_head.loss(
            cls_score, bbox_pred, rois, *bbox_targets)

        # VLST prototype loss on fc features (not raw bbox_feats)
        vlst_loss = None
        if (self.vlst_teacher is not None
                and self.vlst_teacher.text_prototypes is not None):
            vlst_loss = self._compute_vlst_loss(
                x_fc, sampling_results, gt_labels, teacher_results)

        return loss_bbox, vlst_loss

    def _compute_vlst_loss(
            self, bbox_feats, sampling_results, gt_labels, teacher_results):
        """Compute VLST prototype contrastive loss on positive ROIs."""
        # Extract positive ROI features and their pseudo labels
        pos_features = []
        pos_labels = []
        pos_scores = []

        for i, sampling_result in enumerate(sampling_results):
            num_pos = sampling_result.pos_bboxes.size(0)
            if num_pos == 0:
                continue

            # Get pseudo GT indices for positive ROIs
            pos_assigned_gt_inds = sampling_result.pos_assigned_gt_inds
            pos_gt_labels = gt_labels[i][pos_assigned_gt_inds]

            # Get detector scores for these pseudo GTs (from teacher_results)
            # Note: This is an approximation; ideally we'd track exact scores
            # For now, use the pseudo-label threshold as a proxy
            pos_gt_scores = torch.full(
                (num_pos,), float(self.vlst_score_thr),
                dtype=torch.float32, device=pos_gt_labels.device)

            # Collect positive ROI features (first num_pos rows of bbox_feats)
            offset = sum(len(sr.bboxes) for sr in sampling_results[:i])
            pos_roi_feats = bbox_feats[offset:offset+num_pos]

            pos_features.append(pos_roi_feats)
            pos_labels.append(pos_gt_labels)
            pos_scores.append(pos_gt_scores)

        if not pos_features:
            return bbox_feats.sum() * 0.0  # Zero loss

        pos_features = torch.cat(pos_features, dim=0)
        pos_labels = torch.cat(pos_labels, dim=0)
        pos_scores = torch.cat(pos_scores, dim=0)

        # Compute prototype loss
        loss, num_samples = self.vlst_teacher.compute_prototype_loss(
            pos_features, pos_labels, pos_scores, score_thr=self.vlst_score_thr)

        # Diagnostics
        self.vlst_loss_sum += loss.item()
        self.vlst_loss_count += 1
        self.vlst_sample_sum += num_samples

        return self.vlst_loss_weight * loss
