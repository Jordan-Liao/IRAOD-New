"""le90 geometry and the flip-before-resize convention of IRAOD pipelines."""
import math
import numpy as np
import torch


def canonicalize(boxes):
    """Equivalent long-edge boxes, angle in [-pi/2, pi/2)."""
    swap = boxes[..., 2] < boxes[..., 3]
    width = torch.where(swap, boxes[..., 3], boxes[..., 2])
    height = torch.where(swap, boxes[..., 2], boxes[..., 3])
    angle = boxes[..., 4] + swap.to(boxes.dtype) * (math.pi / 2)
    angle = torch.remainder(angle + math.pi / 2, math.pi) - math.pi / 2
    return torch.stack((boxes[..., 0], boxes[..., 1], width, height, angle), -1)


def scale_factors(meta, boxes):
    scale = np.asarray(meta.get('scale_factor', 1.), dtype=np.float64).reshape(-1)
    sx, sy = float(scale[0]), float(scale[1] if len(scale) > 1 else scale[0])
    if sx <= 0 or sy <= 0:
        raise ValueError('Invalid image scale')
    # Exactly the MMRotate 0.3.4 RResize convention for OBB side lengths.
    side = math.sqrt(sx * sy)
    return boxes.new_tensor([sx, sy, side, side, 1.])


def flip_boxes(boxes, meta):
    if not meta.get('flip', False):
        return boxes
    height, width = meta['ori_shape'][:2]
    direction = meta.get('flip_direction', 'horizontal')
    if direction not in ('horizontal', 'vertical', 'diagonal'):
        raise ValueError('Unsupported flip: ' + str(direction))
    out = boxes.clone()
    if direction in ('horizontal', 'diagonal'):
        out[..., 0] = width - 1 - boxes[..., 0]
    if direction in ('vertical', 'diagonal'):
        out[..., 1] = height - 1 - boxes[..., 1]
    if direction != 'diagonal':
        out[..., 4] = -boxes[..., 4]
    return canonicalize(out)


def to_original(boxes, meta):
    """Undo resize, then shared flip; original file coordinates for CGA."""
    if meta.get('rotate', False) or meta.get('crop_offset') is not None:
        raise ValueError('DETR adapter supports shared flips and resize, not crop/rotate')
    return flip_boxes(boxes / scale_factors(meta, boxes), meta)


def from_original(boxes, meta):
    """Apply shared flip, then resize; strong-view pseudo GT coordinates."""
    return canonicalize(flip_boxes(boxes, meta) * scale_factors(meta, boxes))
