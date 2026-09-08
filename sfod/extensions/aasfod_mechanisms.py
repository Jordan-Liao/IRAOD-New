"""Independent tensor implementation of published AASFOD mechanisms (no author code)."""

from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F


def tsd_variance(probabilities, deltas):
    """Code-defined sum of proposal-wise products; exactly20 population samples."""
    if probabilities.shape[0] != 20 or deltas.shape[0] != 20:
        raise ValueError("TSD requires exactly20 aligned stochastic predictions")
    return (probabilities.var(dim=0, unbiased=False).sum(-1)
            * deltas.var(dim=0, unbiased=False).sum(-1)).sum()


def high_variance_split(scores):
    ordered = sorted(scores, key=lambda name: (scores[name], name))
    count = len(ordered) // 5
    split = len(ordered) - count
    return ordered[split:], ordered[:split]


@contextmanager
def posthoc_roi_dropout(head):
    """Stateless hooks preserve checkpoint keys and deterministic ordinary inference.

    RotatedShared2FCBBoxHead applies ReLU immediately after each shared FC. Apply
    ReLU+dropout at that output; the head's following ReLU is idempotent. This is
    explicitly NOT dropout learned during source training.
    """
    handles = [fc.register_forward_hook(
        lambda _module, _args, output: F.dropout(F.relu(output), p=0.5, training=True))
        for fc in head.shared_fcs]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def aligned_tsd(detector, image, metas, to_rois):
    """One deterministic backbone/RPN; twenty heads on the exact same proposals."""
    detector.eval()
    features = detector.extract_feat(image)
    proposals = detector.rpn_head.simple_test_rpn(features, metas)
    rois = to_rois([p[:, :5] for p in proposals])
    probabilities, deltas = [], []
    with posthoc_roi_dropout(detector.roi_head.bbox_head):
        for _ in range(20):
            result = detector.roi_head._bbox_forward(features, rois)
            probabilities.append(result["cls_score"].softmax(-1))
            deltas.append(result["bbox_pred"])
    return tsd_variance(torch.stack(probabilities), torch.stack(deltas))


class _Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient):
        return -gradient


def reverse_gradient(value):
    return _Reverse.apply(value)


def focal_domain_loss(logits, domain):
    log_p = logits.log_softmax(-1)[:, domain]
    return (-((1 - log_p.exp()) ** 3) * log_p).mean()


class TargetDiscriminators(nn.Module):
    """Shallow local / deep global taps before FPN; GRL coefficient1, lambda1."""

    def __init__(self, shallow_channels=256, deep_channels=2048):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(shallow_channels, 256, 1, bias=False), nn.ReLU(),
            nn.Conv2d(256, 128, 1, bias=False), nn.ReLU(),
            nn.Conv2d(128, 1, 1, bias=False), nn.Sigmoid())
        layers = []
        for in_c, out_c in ((deep_channels, 512), (512, 128), (128, 128)):
            layers.extend((nn.Conv2d(in_c, out_c, 3, stride=2, padding=1, bias=False),
                           nn.BatchNorm2d(out_c), nn.ReLU(), nn.Dropout(0.5)))
        self.global_features = nn.Sequential(*layers)
        self.global_classifier = nn.Linear(128, 2)
        for layer in self.local:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)

    def forward(self, shallow, deep, domain):
        local = self.local(reverse_gradient(shallow))
        global_map = self.global_features(reverse_gradient(deep))
        logits = self.global_classifier(global_map.mean(dim=(-2, -1)))
        return {"local": 0.5 * (local - domain).square().mean(),
                "global": 0.5 * focal_domain_loss(logits, domain)}


@torch.no_grad()
def copy_detector(student, teacher):
    """Copy complete detector state (including frozen tensors and buffers), no heads."""
    state = student.state_dict()
    teacher.load_state_dict({key: state[key] for key in teacher.state_dict()}, strict=True)
    teacher.requires_grad_(False).eval()


@torch.no_grad()
def ema_detector(student, teacher, momentum=0.99):
    """Author parameter EMA, preserving frozen parameters and source buffers."""
    parameters = dict(student.named_parameters())
    for name, target in teacher.named_parameters():
        source = parameters[name]
        if source.requires_grad:
            target.mul_(momentum).add_(source, alpha=1 - momentum)
    teacher.eval()
