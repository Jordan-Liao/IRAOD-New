from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


# NOTE:
# - This file is intentionally self-contained so it can be reused by multiple
#   dataset-prep scripts (RSAR corruptions, LoRA pair generation, etc.).
# - Interference names include both legacy ones (used by prepare_rsar_interference.py)
#   and the 7 "compliant" corruption names requested for RSAR.


LEGACY_TYPES = {
    "awgn",
    "noise_jamming",
    "corner_reflector",
    "chaff",
    "smart_noise_jamming",
    "noise_am_jamming",
}

RSAR_COMPLIANT_TYPES = {
    # 7 required names (directory names under dataset/RSAR/corruptions/)
    "chaff",
    "gaussian_white_noise",
    "point_target",
    "noise_suppression",
    "am_noise_horizontal",
    "smart_suppression",
    "am_noise_vertical",
}

SUPPORTED_TYPES = LEGACY_TYPES | RSAR_COMPLIANT_TYPES

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def stable_int_seed(*parts: str) -> int:
    h = hashlib.sha256("::".join(parts).encode("utf-8")).digest()
    return int.from_bytes(h[:4], "little", signed=False)


def _split_alpha(img: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    if img.ndim == 2:
        return img.astype(np.float32)[..., None], None
    if img.ndim != 3:
        raise ValueError(f"unsupported image shape: {img.shape}")

    _h, _w, c = img.shape
    if c == 4:
        base = img[..., :3].astype(np.float32)
        alpha = img[..., 3:].astype(np.float32)
        return base, alpha
    if c == 3:
        return img.astype(np.float32), None
    if c == 1:
        return img.astype(np.float32), None

    base = img[..., :3].astype(np.float32)
    alpha = img[..., 3:4].astype(np.float32) if c > 3 else None
    return base, alpha


def _merge_alpha(base: np.ndarray, alpha: np.ndarray | None) -> np.ndarray:
    if base.ndim == 2:
        base = base[..., None]
    if alpha is None:
        return base
    if base.shape[-1] == 1:
        base = np.repeat(base, 3, axis=2)
    return np.concatenate([base, alpha], axis=2)


def _apply_scalar_field(base: np.ndarray, field_2d: np.ndarray) -> np.ndarray:
    if base.ndim != 3:
        raise ValueError("base must be HxWxC")
    h, w, c = base.shape
    if field_2d.shape != (h, w):
        raise ValueError(f"field_2d must be shape (H,W), got {field_2d.shape} vs {(h, w)}")
    if c == 1:
        return (base[..., 0] + field_2d)[..., None]
    return base + field_2d[..., None]


def _clip_to_dtype(arr: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(arr, info.min, info.max).astype(dtype)
    return np.clip(arr, 0.0, 255.0).astype(dtype)


def _to_locations(value: Any) -> list[tuple[float, float]]:
    locs: list[tuple[float, float]] = []
    if not value:
        return locs
    if not isinstance(value, list):
        raise ValueError("locations must be a list of [row_frac, col_frac]")
    for it in value:
        if not isinstance(it, (list, tuple)) or len(it) != 2:
            raise ValueError(f"invalid location entry: {it!r}")
        locs.append((float(it[0]), float(it[1])))
    return locs


def _gaussian_mask(h: int, w: int, *, sigma_r: float, sigma_c: float) -> np.ndarray:
    sigma_r = max(float(sigma_r), 1.0)
    sigma_c = max(float(sigma_c), 1.0)
    rr, cc = np.mgrid[-h // 2 : h - h // 2, -w // 2 : w - w // 2]
    g = np.exp(-0.5 * ((rr / sigma_r) ** 2 + (cc / sigma_c) ** 2)).astype(np.float32)
    m = float(g.max())
    return (g / m) if m > 0 else np.zeros((h, w), dtype=np.float32)


def _apply_chaff(
    base: np.ndarray,
    *,
    rng: np.random.RandomState,
    locations: list[tuple[float, float]],
    cloud_size: tuple[float, float],
    density_sigma_factor: float,
    noise_sigma: float,
) -> np.ndarray:
    h, w, _c = base.shape
    ch = max(1, int(float(cloud_size[0]) * h))
    cw = max(1, int(float(cloud_size[1]) * w))
    sr = max(ch * float(density_sigma_factor), 1.0)
    sc = max(cw * float(density_sigma_factor), 1.0)
    density = _gaussian_mask(ch, cw, sigma_r=sr, sigma_c=sc)

    out = base.copy()
    for r_frac, c_frac in locations:
        r_center = int(r_frac * h)
        c_center = int(c_frac * w)

        rs = max(0, r_center - ch // 2)
        cs = max(0, c_center - cw // 2)
        re = min(h, rs + ch)
        ce = min(w, cs + cw)
        ah, aw = re - rs, ce - cs
        if ah <= 0 or aw <= 0:
            continue

        mrs = max(0, ch // 2 - (r_center - rs))
        mcs = max(0, cw // 2 - (c_center - cs))
        m_re = mrs + ah
        m_ce = mcs + aw
        mask = density[mrs:m_re, mcs:m_ce]

        noise = rng.normal(0.0, float(noise_sigma), (ah, aw)).astype(np.float32) * mask
        roi = out[rs:re, cs:ce]
        out[rs:re, cs:ce] = _apply_scalar_field(roi, noise)

    return out


def _apply_smart_noise(
    base: np.ndarray,
    *,
    rng: np.random.RandomState,
    locations: list[tuple[float, float]],
    noise_size: tuple[float, float],
    noise_sigma: float,
) -> np.ndarray:
    h, w, _c = base.shape
    nh = max(1, int(float(noise_size[0]) * h))
    nw = max(1, int(float(noise_size[1]) * w))

    out = base.copy()
    for r_frac, c_frac in locations:
        r_center = int(r_frac * h)
        c_center = int(c_frac * w)
        rs = max(0, r_center - nh // 2)
        cs = max(0, c_center - nw // 2)
        re = min(h, rs + nh)
        ce = min(w, cs + nw)
        ah, aw = re - rs, ce - cs
        if ah <= 0 or aw <= 0:
            continue

        noise = rng.normal(0.0, float(noise_sigma), (ah, aw)).astype(np.float32)
        roi = out[rs:re, cs:ce]
        out[rs:re, cs:ce] = _apply_scalar_field(roi, noise)

    return out


def _apply_noise_am_lines(
    base: np.ndarray,
    *,
    rng: np.random.RandomState,
    line_frequency: float,
    base_intensity: float,
    noise_sigma: float,
    line_width: int,
    direction: str,
    blend_factor: float,
) -> np.ndarray:
    if float(line_frequency) <= 0:
        raise ValueError("lineFrequency must be > 0")
    if direction not in ("horizontal", "vertical"):
        raise ValueError("direction must be horizontal|vertical")
    if not (0.0 <= float(blend_factor) <= 1.0):
        raise ValueError("blendFactor must be within [0,1]")

    h, w, _c = base.shape
    out = base.copy()

    spacing = int(1.0 / float(line_frequency))
    hw = max(1, int(line_width) // 2)

    if direction == "horizontal":
        for r in range(0, h, spacing):
            rs = max(0, r - hw)
            re = min(h, r + hw)
            if re <= rs:
                continue
            patch_h = re - rs
            noise = rng.normal(float(base_intensity), float(noise_sigma), (patch_h, w)).astype(np.float32)
            roi = out[rs:re, :, :]
            roi_mix = roi * (1.0 - float(blend_factor))
            out[rs:re, :, :] = _apply_scalar_field(roi_mix, noise * float(blend_factor))
    else:
        for c in range(0, w, spacing):
            cs = max(0, c - hw)
            ce = min(w, c + hw)
            if ce <= cs:
                continue
            patch_w = ce - cs
            noise = rng.normal(float(base_intensity), float(noise_sigma), (h, patch_w)).astype(np.float32)
            roi = out[:, cs:ce, :]
            roi_mix = roi * (1.0 - float(blend_factor))
            out[:, cs:ce, :] = _apply_scalar_field(roi_mix, noise * float(blend_factor))

    return out


def _apply_point_targets(
    base: np.ndarray,
    *,
    locations: list[tuple[float, float]],
    intensity: float,
    sigma_frac: float,
) -> np.ndarray:
    h, w, _c = base.shape
    sigma_px = max(1.0, float(sigma_frac) * float(min(h, w)))
    # Build a reusable Gaussian spot template (square) with radius ~3 sigma.
    radius = int(max(2.0, 3.0 * sigma_px))
    size = 2 * radius + 1
    rr, cc = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    spot = np.exp(-0.5 * (rr**2 + cc**2) / (sigma_px**2)).astype(np.float32)
    spot = spot / max(float(spot.max()), 1e-8) * float(intensity)

    out = base.copy()
    for r_frac, c_frac in locations:
        r_center = int(r_frac * h)
        c_center = int(c_frac * w)
        if not (0 <= r_center < h and 0 <= c_center < w):
            continue

        rs = max(0, r_center - radius)
        re = min(h, r_center + radius + 1)
        cs = max(0, c_center - radius)
        ce = min(w, c_center + radius + 1)

        trs = radius - (r_center - rs)
        tcs = radius - (c_center - cs)
        tre = trs + (re - rs)
        tce = tcs + (ce - cs)

        patch = spot[trs:tre, tcs:tce]
        roi = out[rs:re, cs:ce]
        out[rs:re, cs:ce] = _apply_scalar_field(roi, patch)

    return out


def _blur_like_suppression(base: np.ndarray, *, method: str, ksize: int) -> np.ndarray:
    ksize = int(ksize)
    if ksize < 3:
        ksize = 3
    if ksize % 2 == 0:
        ksize += 1

    # OpenCV expects uint8; keep dynamic range by clipping.
    tmp = np.clip(base, 0.0, 255.0).astype(np.uint8)
    if method == "gaussian":
        out = cv2.GaussianBlur(tmp, ksize=(ksize, ksize), sigmaX=0.0)
    elif method == "median":
        out = cv2.medianBlur(tmp, ksize=ksize)
    else:
        raise ValueError(f"unknown suppression blur method: {method}")
    return out.astype(np.float32)


def add_interference(img: np.ndarray, *, itype: str, params: dict[str, Any] | None = None, seed: int = 0) -> np.ndarray:
    """Apply one interference/corruption to an image array.

    Args:
        img: np.ndarray read by cv2/PIL -> np array. Supports gray or RGB(A).
        itype: one of SUPPORTED_TYPES.
        params: optional parameters (type-dependent). Unknown keys are ignored.
        seed: deterministic seed.
    """
    itype = str(itype or "").strip().lower()
    if itype not in SUPPORTED_TYPES:
        raise ValueError(f"unknown itype: {itype} (supported: {sorted(SUPPORTED_TYPES)})")
    params = dict(params or {})

    # Normalize "compliant" names to internal legacy operations when possible.
    if itype == "gaussian_white_noise":
        itype = "awgn"
        params.setdefault("noiseVariance", 25.0)
    elif itype == "point_target":
        # Implemented as synthetic point targets (no external template file).
        itype = "point_target"
        params.setdefault("locations", [[0.5, 0.5]])
        params.setdefault("intensity", 200.0)
        params.setdefault("sigmaFrac", 0.01)
    elif itype == "am_noise_horizontal":
        itype = "noise_am_jamming"
        params.setdefault("direction", "horizontal")
    elif itype == "am_noise_vertical":
        itype = "noise_am_jamming"
        params.setdefault("direction", "vertical")
    elif itype == "noise_suppression":
        itype = "noise_suppression"
        params.setdefault("noiseVariance", 50.0)
        params.setdefault("blurKsize", 5)
    elif itype == "smart_suppression":
        itype = "smart_suppression"
        params.setdefault("noiseSigma", 200.0)
        params.setdefault("noiseSize", [0.25, 0.25])
        params.setdefault("locations", [[0.5, 0.5]])
        params.setdefault("blurKsize", 5)

    dtype = img.dtype
    base, alpha = _split_alpha(img)
    h, w, _c = base.shape
    rng = np.random.RandomState(int(seed))

    out_base = base
    if itype == "awgn":
        variance = float(params.get("noiseVariance", params.get("variance", 25.0)))
        if variance < 0:
            raise ValueError("noiseVariance must be >= 0")
        if variance > 0:
            sigma = math.sqrt(variance)
            noise = rng.normal(0.0, sigma, (h, w)).astype(np.float32)
            out_base = _apply_scalar_field(base, noise)
    elif itype == "noise_jamming":
        # Simple sinusoidal stripe field (legacy).
        js_ratio_db = float(params.get("jsRatio", 10.0))
        stripe_frequency = float(params.get("stripeFreq", 0.01))
        stripe_amplitude = float(params.get("stripeAmplitude", 50.0))
        js_ratio_linear = 10 ** (float(js_ratio_db) / 10.0)
        amplitude = float(stripe_amplitude) * js_ratio_linear
        y = np.arange(h, dtype=np.float32)
        field_1d = amplitude * (np.sin(2.0 * np.pi * float(stripe_frequency) * y) + 1.0) / 2.0
        field_2d = np.tile(field_1d[:, None], (1, w)).astype(np.float32)
        out_base = _apply_scalar_field(base, field_2d)
    elif itype == "chaff":
        locs = _to_locations(params.get("locations", [[0.5, 0.5]]))
        cloud_size = params.get("cloudSize", [0.2, 0.3])
        if not isinstance(cloud_size, (list, tuple)) or len(cloud_size) != 2:
            raise ValueError("cloudSize must be [height_frac, width_frac]")
        out_base = _apply_chaff(
            base,
            rng=rng,
            locations=locs,
            cloud_size=(float(cloud_size[0]), float(cloud_size[1])),
            density_sigma_factor=float(params.get("densitySigmaFactor", 0.25)),
            noise_sigma=float(params.get("noiseSigma", 300.0)),
        )
    elif itype == "smart_noise_jamming":
        locs = _to_locations(params.get("locations", [[0.5, 0.5]]))
        noise_size = params.get("noiseSize", [0.2, 0.2])
        if not isinstance(noise_size, (list, tuple)) or len(noise_size) != 2:
            raise ValueError("noiseSize must be [height_frac, width_frac]")
        out_base = _apply_smart_noise(
            base,
            rng=rng,
            locations=locs,
            noise_size=(float(noise_size[0]), float(noise_size[1])),
            noise_sigma=float(params.get("noiseSigma", 200.0)),
        )
    elif itype == "noise_am_jamming":
        out_base = _apply_noise_am_lines(
            base,
            rng=rng,
            line_frequency=float(params.get("lineFrequency", 0.05)),
            base_intensity=float(params.get("baseIntensity", 150.0)),
            noise_sigma=float(params.get("noiseSigma", 200.0)),
            line_width=int(params.get("lineWidth", 10)),
            direction=str(params.get("direction", "vertical")).strip().lower(),
            blend_factor=float(params.get("blendFactor", 0.3)),
        )
    elif itype == "corner_reflector":
        # Keep the legacy name but implement with synthetic point targets to
        # avoid requiring a binary template asset in-repo.
        locs = _to_locations(params.get("locations", [[0.5, 0.5]]))
        out_base = _apply_point_targets(
            base,
            locations=locs,
            intensity=float(params.get("intensity", 200.0)),
            sigma_frac=float(params.get("sigmaFrac", 0.01)),
        )
    elif itype == "point_target":
        locs = _to_locations(params.get("locations", [[0.5, 0.5]]))
        out_base = _apply_point_targets(
            base,
            locations=locs,
            intensity=float(params.get("intensity", 200.0)),
            sigma_frac=float(params.get("sigmaFrac", 0.01)),
        )
    elif itype == "noise_suppression":
        # A simple "suppression-like" artifact: add noise then denoise/blur.
        variance = float(params.get("noiseVariance", 50.0))
        if variance < 0:
            raise ValueError("noiseVariance must be >= 0")
        sigma = math.sqrt(max(variance, 0.0))
        noise = rng.normal(0.0, sigma, (h, w)).astype(np.float32)
        tmp = _apply_scalar_field(base, noise)
        out_base = _blur_like_suppression(tmp, method="gaussian", ksize=int(params.get("blurKsize", 5)))
    elif itype == "smart_suppression":
        # Add localized noise then suppress via median blur (artifact differs from pure blur).
        locs = _to_locations(params.get("locations", [[0.5, 0.5]]))
        noise_size = params.get("noiseSize", [0.25, 0.25])
        if not isinstance(noise_size, (list, tuple)) or len(noise_size) != 2:
            raise ValueError("noiseSize must be [height_frac, width_frac]")
        tmp = _apply_smart_noise(
            base,
            rng=rng,
            locations=locs,
            noise_size=(float(noise_size[0]), float(noise_size[1])),
            noise_sigma=float(params.get("noiseSigma", 200.0)),
        )
        out_base = _blur_like_suppression(tmp, method="median", ksize=int(params.get("blurKsize", 5)))
    else:
        raise AssertionError(f"unhandled itype: {itype}")

    out_base_u8 = _clip_to_dtype(out_base, dtype)
    alpha_u8 = _clip_to_dtype(alpha, dtype) if alpha is not None else None
    out = _merge_alpha(out_base_u8, alpha_u8)

    if img.ndim == 2:
        return out[..., 0]
    if img.ndim == 3 and img.shape[2] == 1 and out.ndim == 3 and out.shape[2] == 1:
        return out
    if out.ndim == 3 and out.shape[2] == 1:
        return out[..., 0]
    return out


@dataclass(frozen=True)
class RsarCorruptionSpec:
    name: str
    itype: str
    params: dict[str, Any]


def default_rsar_corruptions() -> list[RsarCorruptionSpec]:
    """Default 7 RSAR corruption specs (names align with required directory names)."""
    return [
        RsarCorruptionSpec(name="chaff", itype="chaff", params={"locations": [[0.5, 0.5]], "cloudSize": [0.25, 0.35]}),
        RsarCorruptionSpec(name="gaussian_white_noise", itype="gaussian_white_noise", params={"noiseVariance": 25.0}),
        RsarCorruptionSpec(name="point_target", itype="point_target", params={"locations": [[0.5, 0.5]], "intensity": 200.0, "sigmaFrac": 0.01}),
        RsarCorruptionSpec(name="noise_suppression", itype="noise_suppression", params={"noiseVariance": 50.0, "blurKsize": 5}),
        RsarCorruptionSpec(name="am_noise_horizontal", itype="am_noise_horizontal", params={"direction": "horizontal", "lineFrequency": 0.05}),
        RsarCorruptionSpec(name="smart_suppression", itype="smart_suppression", params={"locations": [[0.5, 0.5]], "noiseSigma": 200.0, "noiseSize": [0.25, 0.25], "blurKsize": 5}),
        RsarCorruptionSpec(name="am_noise_vertical", itype="am_noise_vertical", params={"direction": "vertical", "lineFrequency": 0.05}),
    ]

