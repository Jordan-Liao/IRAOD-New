"""Checkpointable query adapter for the existing semantic prototype teacher."""
import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F
from sfod.semantic_teacher.prototype_teacher import SemanticPrototypeTeacher


class QuerySemanticPrototypeTeacher(SemanticPrototypeTeacher):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_prototypes = torch.zeros(self.num_classes, self.vlm_dim)
        self.visual_prototypes = torch.zeros(self.num_classes, self.vlm_dim)
        self._non_persistent_buffers_set.discard('text_prototypes')
        self._non_persistent_buffers_set.discard('visual_prototypes')
        self.register_buffer('prototype_counts', torch.zeros(self.num_classes, dtype=torch.long))
        self.register_buffer('text_ready', torch.tensor(False))

    def set_text_prototypes(self, prototypes):
        super().set_text_prototypes(prototypes)
        self.text_ready.fill_(True)

    def get_fused_prototypes(self, device):
        self.visual_counts = self.prototype_counts.detach().cpu().numpy().copy()
        return super().get_fused_prototypes(device)

    @torch.no_grad()
    def update_visual_prototypes(self, features, labels, scores, score_thr=None):
        threshold = self.score_threshold if score_thr is None else score_thr
        mask = scores >= (0. if threshold is None else threshold)
        features = F.normalize(features[mask].float(), dim=-1)
        labels = labels[mask]
        sums = self.visual_prototypes.new_zeros(self.visual_prototypes.shape)
        counts = self.prototype_counts.new_zeros(self.prototype_counts.shape)
        sums.index_add_(0, labels, features)
        counts.index_add_(0, labels, torch.ones_like(labels))
        # Every rank calls this once per batch, even if it has no pseudo boxes.
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(sums)
            dist.all_reduce(counts)
        for cls in range(self.num_classes):
            if counts[cls] == 0:
                continue
            mean = F.normalize(sums[cls], dim=0)
            if self.prototype_counts[cls] == 0:
                self.visual_prototypes[cls] = mean
            else:
                m = self.prototype_momentum
                self.visual_prototypes[cls] = F.normalize(
                    m * self.visual_prototypes[cls] + (1 - m) * mean, dim=0)
        self.prototype_counts.add_(counts)
        self.visual_counts = self.prototype_counts.detach().cpu().numpy().copy()
        self.total_updates += 1

    def compute_prototype_loss(self, features, labels, scores, score_thr=None):
        threshold = self.score_threshold if score_thr is None else score_thr
        threshold = 0. if threshold is None else threshold
        keep = scores >= threshold
        # Include projection parameters in the empty-batch graph for DDP.
        if not keep.any():
            return self.projection_head(features[:0]).sum() + features.sum() * 0., 0
        return super().compute_prototype_loss(features, labels, scores, score_thr=threshold)
