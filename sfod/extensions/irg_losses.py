"""Independent implementation of the specified IRG train-time tensor mechanisms.

Mathematical reference: IRG commit 82c84894c938b895a6f96f67bdb084f44cbfe8e2,
meta_arch/GCN.py:67-103, student_sfda_rcnn.py:145-150,211-227, losses.py:12-63.
Implemented from the supplied algorithm description, not upstream source.

Every call handles matched proposals from ONE image; the detector owns image
averaging, EMA and inference. Instantiate IRGLosses before optimizer creation.
"""

from typing import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class IRGGraph(nn.Module):
    """Shared D -> 512 -> 512 -> D graph; return (representation, raw S)."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.q = nn.Linear(feature_dim, feature_dim)
        self.k = nn.Linear(feature_dim, feature_dim)
        self.layers = nn.ModuleList([
            nn.Linear(feature_dim, 512),
            nn.Linear(512, 512),
            nn.Linear(512, feature_dim),
        ])
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight, gain=0.02)
            nn.init.zeros_(layer.bias)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        raw_affinities = self.q(features) @ self.k(features).T
        adjacency = F.normalize(raw_affinities.square(), p=1, dim=-1)
        representation = features
        for layer in self.layers:
            representation = F.relu(layer(adjacency @ representation))
        return representation, raw_affinities


def contrastive_mask(raw_affinities: Tensor) -> Tensor:
    """Detached row-minmax S > .5 positives, with the diagonal always positive.

    Use raw signed S, not squared/normalized adjacency. No epsilon is specified
    for minmax: a constant row yields no threshold positives (NaN > .5 is false)
    before the diagonal is set.
    """
    scores = raw_affinities.detach()
    row_min = scores.amin(dim=-1, keepdim=True)
    row_max = scores.amax(dim=-1, keepdim=True)
    mask = (scores - row_min) / (row_max - row_min) > 0.5
    mask.fill_diagonal_(True)
    return mask


class IRGLosses(nn.Module):
    """Train-only graph and two projection heads, registered at construction.

    ``forward`` accepts [N,D] matched student/teacher ROI features and [N,C]
    direct logits, where C includes background. Pass the current student and
    frozen teacher classifiers into each call; neither is stored as a submodule.
    Both classifiers remain differentiable with respect to graph inputs.
    All four returned scalar losses have coefficient one.
    """

    def __init__(self, feature_dim: int = 1024):
        super().__init__()
        self.graph = IRGGraph(feature_dim)
        self.head1 = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.head2 = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )

    def contrastive_loss(self, student_features: Tensor, raw_affinities: Tensor) -> Tensor:
        """Student-only, asymmetric two-head contrast, including self columns."""
        first = F.normalize(self.head1(student_features), p=2, dim=-1)
        second = F.normalize(self.head2(student_features), p=2, dim=-1)
        logits = first @ second.T / 0.07
        positives = contrastive_mask(raw_affinities)
        log_probabilities = F.log_softmax(logits, dim=-1)
        per_row = (log_probabilities * positives).sum(dim=-1) / positives.sum(dim=-1)
        return -0.5 * per_row.mean()

    def forward(
        self,
        student_features: Tensor,
        teacher_features: Tensor,
        student_logits: Tensor,
        teacher_logits: Tensor,
        student_classifier: Callable[[Tensor], Tensor],
        teacher_classifier: Callable[[Tensor], Tensor],
    ) -> dict[str, Tensor]:
        student_graph, student_affinities = self.graph(student_features)
        teacher_graph, _ = self.graph(teacher_features.detach())
        # Student graph also trains the student head; teacher graph retains
        # input gradients through the frozen teacher head into shared G.
        student_graph_logits = student_classifier(student_graph)
        teacher_graph_logits = teacher_classifier(teacher_graph)
        teacher_target = F.softmax(teacher_logits.detach(), dim=-1)
        student_target = F.softmax(student_logits.detach(), dim=-1)
        return {
            "loss_irg_direct": F.kl_div(
                F.log_softmax(student_logits, dim=-1), teacher_target,
                reduction="batchmean",
            ),
            "loss_irg_student_graph": F.kl_div(
                F.log_softmax(student_graph_logits, dim=-1), student_target,
                reduction="batchmean",
            ),
            "loss_irg_teacher_graph": F.kl_div(
                F.log_softmax(teacher_graph_logits, dim=-1), teacher_target,
                reduction="batchmean",
            ),
            "loss_irg_contrast": self.contrastive_loss(student_features, student_affinities),
        }
