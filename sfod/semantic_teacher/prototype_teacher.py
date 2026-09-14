"""Semantic Prototype Teacher: VLM feature-level guidance for SFOD.

Maintains per-class semantic prototypes fusing text and visual features from
CLIP/SARCLIP, guiding Student ROI features via contrastive loss.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticPrototypeTeacher(nn.Module):
    """Vision-Language Semantic Prototype Teacher.

    Core idea: CLIP/SARCLIP transitions from label-level pseudo-label scorer
    (CGA) to feature-level semantic teacher, constraining Student ROI
    representations via class semantic prototypes.

    Args:
        num_classes (int): Number of object classes.
        vlm_dim (int): VLM embedding dimension (512 for ViT-B-32 SARCLIP).
        detector_dim (int): Student ROI feature dimension.
        projection_hidden (int): Hidden dimension for projection MLP.
        text_visual_alpha (float): Text prototype weight in [0,1].
        prototype_momentum (float): EMA momentum for visual prototypes.
        temperature (float): Contrastive loss temperature.
        score_threshold (float): Min detector score for prototype update.
            If None, inherits from the SFOD pseudo-label threshold.
    """

    def __init__(
        self,
        num_classes,
        vlm_dim=512,
        detector_dim=1024,
        projection_hidden=256,
        text_visual_alpha=0.5,
        prototype_momentum=0.9,
        temperature=0.07,
        score_threshold=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.vlm_dim = vlm_dim
        self.detector_dim = detector_dim
        self.projection_hidden = int(projection_hidden)
        if self.projection_hidden <= 0:
            raise ValueError('projection_hidden must be positive')
        self.text_visual_alpha = float(text_visual_alpha)
        self.prototype_momentum = float(prototype_momentum)
        self.temperature = float(temperature)
        self.score_threshold = score_threshold

        # How many accepted instances have contributed to each visual prototype
        self.visual_counts = np.zeros(num_classes, dtype=np.int64)

        # Projection head: Student ROI feature -> VLM embedding space.
        # Built eagerly in __init__ so that its parameters are registered on
        # the detector before the runner builds the optimizer. A lazily built
        # head is invisible to the optimizer and stays at random init, which
        # forces the backbone to absorb the whole contrastive loss.
        self.projection_head = nn.Sequential(
            nn.Linear(detector_dim, self.projection_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.projection_hidden, vlm_dim),
        )
        for module in self.projection_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Prototypes are buffers, not parameters: they receive no gradient and
        # are updated only by EMA from VLM features.
        self.register_buffer(
            'text_prototypes', None, persistent=False)
        self.register_buffer(
            'visual_prototypes', None, persistent=False)

        # Diagnostics
        self.total_updates = 0
        self.class_updates = np.zeros(num_classes, dtype=np.int64)
        self.loss_sum = 0.0
        self.loss_count = 0
        self.pos_cos_sum = 0.0
        self.neg_cos_sum = 0.0
        self.margin_sum = 0.0
        self.margin_count = 0

    def build_projection_head(self, device):
        """Kept for call-site compatibility; the head is built in __init__.

        The module moves with the detector via ``.cuda()`` / ``.to()``, so
        nothing device-specific is needed here.
        """
        return

    def set_text_prototypes(self, text_prototypes):
        """Set frozen text prototypes from VLM text encoder.

        Args:
            text_prototypes (Tensor or ndarray): (C, D) L2-normalized.
        """
        if isinstance(text_prototypes, np.ndarray):
            text_prototypes = torch.from_numpy(text_prototypes).float()
        if text_prototypes.shape[0] != self.num_classes:
            raise ValueError(
                f'text_prototypes shape {text_prototypes.shape} does not '
                f'match num_classes={self.num_classes}')
        if text_prototypes.shape[1] != self.vlm_dim:
            raise ValueError(
                f'text_prototypes dim {text_prototypes.shape[1]} does not '
                f'match vlm_dim={self.vlm_dim}')
        device = next(self.projection_head.parameters()).device
        # detach(): the VLM stays frozen, so no gradient may flow to prototypes.
        self.text_prototypes = F.normalize(
            text_prototypes.detach().float().to(device), p=2, dim=1)

    def get_fused_prototypes(self, device):
        """Return fused text+visual prototypes.

        Returns:
            prototypes (Tensor): (C, D) L2-normalized on device.
        """
        if self.text_prototypes is None:
            raise RuntimeError('text_prototypes not initialized')

        text_p = self.text_prototypes.to(device)

        # If no visual prototypes yet, fall back to text-only
        if self.visual_prototypes is None:
            return text_p

        visual_p = self.visual_prototypes.to(device)
        alpha = self.text_visual_alpha

        # Fuse: alpha * text + (1-alpha) * visual, per-class
        # Classes without visual prototype use text-only
        fused = []
        for c in range(self.num_classes):
            if self.visual_counts[c] == 0:
                fused.append(text_p[c])
            else:
                fused.append(alpha * text_p[c] + (1.0 - alpha) * visual_p[c])
        fused = torch.stack(fused, dim=0)
        return F.normalize(fused, p=2, dim=1)

    def update_visual_prototypes(
        self, vlm_features, labels, scores, score_thr=None
    ):
        """EMA update visual prototypes from high-confidence pseudo boxes.

        Args:
            vlm_features (Tensor): (N, D) L2-normalized VLM embeddings.
            labels (Tensor): (N,) pseudo-label class IDs.
            scores (Tensor): (N,) detector confidence scores.
            score_thr (float): Threshold for prototype update. If None, uses
                self.score_threshold (which may also be None = accept all).
        """
        if score_thr is None:
            score_thr = self.score_threshold
        if score_thr is None:
            score_thr = 0.0  # Accept all by default

        vlm_features = F.normalize(vlm_features, p=2, dim=1)

        # Filter by score threshold
        mask = scores >= score_thr
        if mask.sum() == 0:
            return

        filtered_features = vlm_features[mask]
        filtered_labels = labels[mask]

        # Lazy init visual prototypes
        if self.visual_prototypes is None:
            self.visual_prototypes = torch.zeros(
                self.num_classes, self.vlm_dim,
                dtype=vlm_features.dtype, device=vlm_features.device)

        visual_p = self.visual_prototypes.to(vlm_features.device)

        # Per-class EMA update
        for c in range(self.num_classes):
            class_mask = filtered_labels == c
            if class_mask.sum() == 0:
                continue

            class_features = filtered_features[class_mask]
            class_mean = class_features.mean(dim=0)
            class_mean = F.normalize(class_mean.unsqueeze(0), p=2, dim=1).squeeze(0)

            if self.visual_counts[c] == 0:
                # First update: direct assignment
                visual_p[c] = class_mean
            else:
                # EMA update
                momentum = self.prototype_momentum
                visual_p[c] = momentum * visual_p[c] + (1.0 - momentum) * class_mean
                visual_p[c] = F.normalize(visual_p[c].unsqueeze(0), p=2, dim=1).squeeze(0)

            self.visual_counts[c] += class_mask.sum().item()
            self.class_updates[c] += 1

        self.visual_prototypes = visual_p
        self.total_updates += 1

    def compute_prototype_loss(
        self, student_roi_features, pseudo_labels, pseudo_scores, score_thr=None
    ):
        """Compute prototype contrastive loss.

        Args:
            student_roi_features (Tensor): (N, detector_dim) from ROI head.
            pseudo_labels (Tensor): (N,) pseudo-label class IDs.
            pseudo_scores (Tensor): (N,) detector confidence scores.
            score_thr (float): Only compute loss for boxes >= this threshold.

        Returns:
            loss (Tensor): Scalar prototype contrastive loss.
            num_samples (int): Number of samples used in loss.
        """
        if score_thr is None:
            score_thr = self.score_threshold
        if score_thr is None:
            score_thr = 0.0

        # Filter by score threshold (only reliable pseudo boxes)
        mask = pseudo_scores >= score_thr
        if mask.sum() == 0:
            return student_roi_features.sum() * 0.0, 0

        filtered_features = student_roi_features[mask]
        filtered_labels = pseudo_labels[mask]

        device = filtered_features.device
        self.build_projection_head(device)

        # Project Student ROI features to VLM space
        projected = self.projection_head(filtered_features)
        projected = F.normalize(projected, p=2, dim=1)  # (M, D)

        # Get fused prototypes
        prototypes = self.get_fused_prototypes(device)  # (C, D)

        # Compute cosine similarity logits
        logits = projected @ prototypes.T / self.temperature  # (M, C)

        # Cross-entropy loss
        loss = F.cross_entropy(logits, filtered_labels)

        # Diagnostics
        self.loss_sum += loss.item()
        self.loss_count += 1

        # Separability diagnostics (spec: positive cosine, hardest-negative
        # cosine, and their margin M = sim(z, p_y) - max_{c!=y} sim(z, p_c)).
        with torch.no_grad():
            cos = projected @ prototypes.T                       # (M, C)
            pos = cos.gather(1, filtered_labels[:, None]).squeeze(1)
            neg = cos.clone()
            neg.scatter_(1, filtered_labels[:, None], float('-inf'))
            hard_neg = neg.max(dim=1).values
            self.pos_cos_sum += pos.mean().item()
            self.neg_cos_sum += hard_neg.mean().item()
            self.margin_sum += (pos - hard_neg).mean().item()
            self.margin_count += 1

        return loss, int(mask.sum())

    def get_diagnostics(self):
        """Return diagnostic statistics."""
        n = max(self.margin_count, 1)
        proto_norm = (float(self.visual_prototypes.norm(dim=1).mean())
                      if self.visual_prototypes is not None else 0.0)
        return {
            'total_updates': self.total_updates,
            'class_updates': self.class_updates.copy(),
            'visual_counts': self.visual_counts.copy(),
            'mean_loss': (self.loss_sum / self.loss_count
                         if self.loss_count > 0 else 0.0),
            'pos_cos': self.pos_cos_sum / n,
            'neg_cos': self.neg_cos_sum / n,
            'margin': self.margin_sum / n,
            'proto_norm': proto_norm,
        }

    def reset_diagnostics(self):
        """Reset diagnostic counters."""
        self.loss_sum = 0.0
        self.loss_count = 0
        self.pos_cos_sum = 0.0
        self.neg_cos_sum = 0.0
        self.margin_sum = 0.0
        self.margin_count = 0
