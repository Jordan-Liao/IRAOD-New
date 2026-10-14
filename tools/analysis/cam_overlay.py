"""EigenCAM helpers. Numpy only so CPU tests do not import mmrotate."""

from __future__ import annotations

import numpy as np


def eigen_cam(feat: np.ndarray) -> np.ndarray:
    """First-component EigenCAM. `feat` is C,H,W; result is H,W in [0, 1]."""
    if feat.ndim != 3:
        raise ValueError(f"expected C,H,W, got {feat.shape}")
    channels, height, width = feat.shape
    flat = feat.reshape(channels, -1).astype(np.float64, copy=False)
    if flat.size == 0:
        return np.zeros((height, width), dtype=np.float32)
    _u, _s, vh = np.linalg.svd(flat, full_matrices=False)
    cam = vh[0].reshape(height, width)
    if cam.mean() < 0:
        cam = -cam
    cam = np.clip(cam, 0, None)
    peak = float(cam.max())
    if peak <= 0:
        return np.zeros((height, width), dtype=np.float32)
    return (cam / peak).astype(np.float32)


def eigen_cam_torch(feat) -> np.ndarray:
    """GPU/CPU torch EigenCAM. `feat` is C,H,W; SVD stays on the tensor device."""
    import torch

    if not torch.is_tensor(feat) or feat.ndim != 3:
        raise ValueError(f"expected C,H,W tensor, got {type(feat)} {getattr(feat, 'shape', None)}")
    channels, height, width = feat.shape
    flat = feat.reshape(channels, -1).float()
    if flat.numel() == 0:
        return np.zeros((height, width), dtype=np.float32)
    vh = torch.linalg.svd(flat, full_matrices=False).Vh
    cam = vh[0].reshape(height, width)
    if cam.mean() < 0:
        cam = -cam
    cam = cam.clamp(min=0)
    peak = cam.max()
    if float(peak) <= 0:
        return np.zeros((height, width), dtype=np.float32)
    return (cam / peak).detach().cpu().numpy().astype(np.float32)


def overlay_cam(image_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a JET heatmap onto an RGB uint8 image."""
    import cv2

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB, got {image_rgb.shape}")
    heat = cv2.applyColorMap((np.clip(cam, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    heat = cv2.resize(heat, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    blended = (1.0 - alpha) * image_rgb.astype(np.float32) + alpha * heat.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)
