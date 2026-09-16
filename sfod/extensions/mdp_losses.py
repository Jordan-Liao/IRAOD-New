"""MDP tensor operations from arXiv:2401.17916v1, Eqs. (5), (8)-(19).

AFSP architecture reference: weix-liu/AFSP; experiments/comparison/AFSP_LICENSE.txt.
"""

import torch
from torch import nn


class _ReverseGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient):
        return -gradient


class AdversarialFeatureStyle(nn.Module):
    """C2 style network; dual GRL reverses its parameters, not upstream features."""

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.style = nn.Sequential(
            nn.Conv2d(2 * channels, channels // 16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // 16, 2 * channels, 1, bias=False),
        )

    def perturb(self, features):
        mean = features.mean(dim=(-2, -1), keepdim=True)
        std = features.std(dim=(-2, -1), unbiased=False, keepdim=True) + 1e-6
        predicted = self.style(torch.cat((mean, std), dim=1))
        mean_adv, std_adv = predicted.split(self.channels, dim=1)
        mixed_mean = .5 * mean + .5 * mean_adv
        mixed_std = .5 * std + .5 * std_adv
        perturbed = mixed_std * (features - mean) / std + mixed_mean
        original_norm = features.abs().sum(dim=(1, 2, 3), keepdim=True)
        new_norm = perturbed.abs().sum(dim=(1, 2, 3), keepdim=True)
        return perturbed * (original_norm / (new_norm + 1e-8))

    def forward(self, features):
        return _ReverseGradient.apply(self.perturb(_ReverseGradient.apply(features)))


def mix_target_pairs(images, boxes, labels, metas):
    """Pair the two halves of a batch without losing a sample or changing angles."""
    count = len(images)
    if count < 2 or count % 2:
        raise ValueError('MDP requires an even local image batch for two N/2 groups')
    if not (len(boxes) == len(labels) == len(metas) == count):
        raise ValueError('MDP images, pseudo boxes, labels and metadata must align')
    half = count // 2
    mixed = .5 * images[:half] + .5 * images[half:]
    mixed_boxes, mixed_labels, mixed_metas = [], [], []
    for index in range(half):
        other = index + half
        mixed_boxes.append(torch.cat((boxes[index], boxes[other])))
        mixed_labels.append(torch.cat((labels[index], labels[other])))
        first, second = metas[index], metas[other]
        height = max(first['img_shape'][0], second['img_shape'][0])
        width = max(first['img_shape'][1], second['img_shape'][1])
        padded_height = max(first['pad_shape'][0], second['pad_shape'][0])
        padded_width = max(first['pad_shape'][1], second['pad_shape'][1])
        name = f"mdp_mix({first['ori_filename']}|{second['ori_filename']})"
        mixed_metas.append({
            **first, 'filename': name, 'ori_filename': name,
            'ori_shape': (height, width, images.shape[1]),
            'img_shape': (height, width, images.shape[1]),
            'pad_shape': (padded_height, padded_width, images.shape[1]),
            'scale_factor': 1.0, 'flip': False, 'flip_direction': None,
            'mdp_sources': (first['ori_filename'], second['ori_filename']),
        })
    return mixed, mixed_boxes, mixed_labels, mixed_metas


def global_class_means(sums, counts):
    """Preserve global-batch prototype semantics under the existing DDP path."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        from torch.distributed.nn.functional import all_reduce
        sums = all_reduce(sums)
        counts = counts.clone()
        torch.distributed.all_reduce(counts)
    return sums / counts.clamp_min(1).to(sums.dtype).unsqueeze(1), counts > 0


def rotated_class_prototypes(features, rois, logits, num_classes, roi_align, chunk_size=256):
    """Average the actual classification RoIs on C4 without materializing all pools."""
    if len(rois) != len(logits) or logits.shape[1] != num_classes:
        raise ValueError('MDP prototype RoIs and class logits do not align')
    sums = features.new_zeros((num_classes, features.shape[1])) + features.sum() * 0
    counts = torch.zeros(num_classes, device=features.device, dtype=torch.long)
    labels = logits.detach().argmax(dim=1)
    for start in range(0, len(rois), chunk_size):
        selected = slice(start, start + chunk_size)
        pooled = roi_align(features, rois[selected].detach()).mean(dim=(-2, -1))
        sums = sums.index_add(0, labels[selected], pooled)
        counts += torch.bincount(labels[selected], minlength=num_classes)
    return global_class_means(sums, counts)


class PrototypeHistory(nn.Module):
    """Eq. (16) uses previous local prototypes, not overwritten aliases or EMA."""

    def __init__(self, num_classes, channels):
        super().__init__()
        self.register_buffer('previous_teacher', torch.zeros(num_classes, channels))
        self.register_buffer('previous_student', torch.zeros(num_classes, channels))

    def forward(self, teacher, student, teacher_present, student_present):
        previous_teacher = self.previous_teacher.detach().clone()
        previous_student = self.previous_student.detach().clone()
        global_teacher = .7 * teacher.detach() + .3 * previous_teacher
        global_student = .7 * student + .3 * previous_student
        present = teacher_present & student_present
        loss = torch.linalg.vector_norm(
            global_student[present] - global_teacher[present], dim=1).sum()
        with torch.no_grad():
            self.previous_teacher.copy_(teacher.detach())
            self.previous_student.copy_(student.detach())
        return loss, present.sum()


class MDPModules(nn.Module):
    def __init__(self, c2_channels, c4_channels, num_classes):
        super().__init__()
        self.afsp = AdversarialFeatureStyle(c2_channels)
        # Keep C4 coordinates unchanged for rotated RoIAlign.
        self.transform = nn.Sequential(
            nn.Conv2d(c4_channels, c4_channels, 3, padding=1),
            nn.BatchNorm2d(c4_channels), nn.ReLU(),
        )
        self.history = PrototypeHistory(num_classes, c4_channels)
