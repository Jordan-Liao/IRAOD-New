"""Four-image FNS: clip transformed OBB polygons, then project to le90 rectangles."""

import math
import random

import cv2
import numpy as np
import torch
from torch.nn import functional as F


def corners(box):
    cx, cy, width, height, angle = box
    points = np.array([[-width, -height], [width, -height],
                       [width, height], [-width, height]], dtype=np.float32) / 2
    c, s = math.cos(angle), math.sin(angle)
    return points @ np.array([[c, s], [-s, c]], dtype=np.float32) + [cx, cy]


def clip_polygon(points, rect):
    """Sutherland-Hodgman clipping; retain rotated polygons, not HBB envelopes."""
    points = [np.asarray(p, dtype=np.float64) for p in points]
    for axis, bound, sign in ((0, rect[0], 1), (0, rect[2], -1),
                              (1, rect[1], 1), (1, rect[3], -1)):
        result = []
        if not points:
            break
        previous = points[-1]
        prev_inside = sign * (previous[axis] - bound) >= 0
        for current in points:
            inside = sign * (current[axis] - bound) >= 0
            if inside != prev_inside:
                t = (bound - previous[axis]) / (current[axis] - previous[axis])
                result.append(previous + t * (current - previous))
            if inside:
                result.append(current)
            previous, prev_inside = current, inside
        points = result
    return np.asarray(points, dtype=np.float32).reshape(-1, 2)


def le90_rectangle(points):
    if len(points) < 3 or abs(cv2.contourArea(points)) <= 0:
        return None
    (x, y), (width, height), degrees = cv2.minAreaRect(points)
    if width <= 0 or height <= 0:
        return None
    angle = math.radians(degrees)
    if width < height:
        width, height, angle = height, width, angle + math.pi / 2
    angle = (angle + math.pi / 2) % math.pi - math.pi / 2
    return [x, y, width, height, angle]


def transform_boxes(boxes, labels, crop, output_size, source_window, destination):
    """Crop, anisotropic resize, quadrant clip, le90 min-area projection."""
    left, top, right, bottom = crop
    height, width = output_size
    sx, sy = width / (right - left), height / (bottom - top)
    wx, wy, wr, wb = source_window
    dx, dy = destination
    output, selected = [], []
    for box, label in zip(boxes.detach().cpu().numpy(), labels.detach().cpu().tolist()):
        polygon = clip_polygon(corners(box), crop)
        polygon = (polygon - [left, top]) * [sx, sy]
        polygon = clip_polygon(polygon, (wx, wy, wr, wb))
        polygon += np.array([dx - wx, dy - wy], dtype=np.float32)
        rectangle = le90_rectangle(polygon)
        if rectangle is not None:
            output.append(rectangle)
            selected.append(label)
    return (boxes.new_tensor(output).reshape(-1, 5),
            labels.new_tensor(selected).reshape(-1))


def four_image_mosaic(images, metas, boxes, labels, rng=random):
    """Teacher boxes must already be mapped to the four original strong views.

    Resize each randomly padded/cropped source back to canvas size, then select
    the matching quadrant. Padding is the source image's channel mean. There is
    deliberately no teacher inference here.
    """
    if len(images) != 4:
        raise ValueError("FNS requires four original images per canvas")
    height, width = metas[0]["img_shape"][:2]
    cut_x = rng.randint(int(width * 0.2), int(width * 0.8))
    cut_y = rng.randint(int(height * 0.2), int(height * 0.8))
    output = images.new_zeros((images.shape[1], height, width))
    all_boxes, all_labels = [], []
    for i, (image, meta, targets, classes) in enumerate(zip(images, metas, boxes, labels)):
        h, w = meta["img_shape"][:2]
        image = image[:, :h, :w]
        left, right = (rng.randint(-int(w * 0.2), int(w * 0.2)) for _ in range(2))
        top, bottom = (rng.randint(-int(h * 0.2), int(h * 0.2)) for _ in range(2))
        cw, ch = w - left - right, h - top - bottom
        cropped = image.mean(dim=(-2, -1), keepdim=True).expand(-1, ch, cw).clone()
        x0, y0, x1, y1 = max(left, 0), max(top, 0), min(w - right, w), min(h - bottom, h)
        cropped[:, y0 - top:y1 - top, x0 - left:x1 - left] = image[:, y0:y1, x0:x1]
        resized = F.interpolate(cropped[None], (height, width), mode="bilinear",
                                align_corners=False)[0]
        left_shift = min(int(max(0, -left * width / cw)), cut_x, width - cut_x)
        right_shift = min(int(max(0, -right * width / cw)), width - cut_x, cut_x)
        top_shift = min(int(max(0, -top * height / ch)), cut_y, height - cut_y)
        bottom_shift = min(int(max(0, -bottom * height / ch)), height - cut_y, cut_y)
        dx, dy = (0 if i % 2 == 0 else cut_x), (0 if i < 2 else cut_y)
        qw, qh = (cut_x if i % 2 == 0 else width - cut_x), (cut_y if i < 2 else height - cut_y)
        wx = left_shift if i % 2 == 0 else cut_x - right_shift
        wy = top_shift if i < 2 else cut_y - bottom_shift
        output[:, dy:dy + qh, dx:dx + qw] = resized[:, wy:wy + qh, wx:wx + qw]
        mapped, kept = transform_boxes(
            targets, classes, (left, top, w - right, h - bottom), (height, width),
            (wx, wy, wx + qw, wy + qh), (dx, dy))
        all_boxes.append(mapped)
        all_labels.append(kept)
    # Keep the detector's divisor32 padding convention.
    padded = F.pad(output, (0, (-width) % 32, 0, (-height) % 32))
    meta = dict(metas[0], img_shape=(height, width, 3),
                pad_shape=(*padded.shape[-2:], 3), scale_factor=np.ones(4, np.float32),
                flip=False, flip_direction=None)
    return padded, meta, torch.cat(all_boxes), torch.cat(all_labels)
