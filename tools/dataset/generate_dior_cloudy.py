"""Independently reproduce DOTA-C's stateful cloudy algorithm on DIOR TRAIN."""

import argparse
from functools import lru_cache
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image

from tools.build_oracle_patches import required_images, train_annotations


AUTHOR_COMMIT = "c9ce98fad9b2fbd7218346d8b4bdba6974323ac1"
AUTHOR_SOURCE = (
    "https://github.com/hehaodong530/DOTA-C/blob/"
    + AUTHOR_COMMIT + "/clouds/Cloudy_Image_Arithmetic.m")
CLOUD_ARCHIVE_SHA256 = "66b15135711c62236d68ecdecbb1e45062d4b1a84363de32a9e356e80f588963"
TRAIN_COUNT = 5862
ROOT = Path(__file__).resolve().parents[2]


def uint8_round(values):
    """MATLAB's saturating uint8 conversion: nearest integer, positive ties up."""
    return np.floor(np.clip(values, 0, 255) + .5).astype(np.uint8)


def cubic(distance):
    """Keys cubic convolution with a=-0.5, the imresize bicubic kernel."""
    x = np.abs(distance)
    return np.where(x <= 1, 1.5 * x**3 - 2.5 * x**2 + 1,
                    np.where(x <= 2, -.5 * x**3 + 2.5 * x**2 - 4 * x + 2, 0))


def resize_weights(source_size, target_size):
    scale = target_size / source_size
    kernel_scale = min(scale, 1.0)
    width = 4 / kernel_scale
    # Pixel-center mapping, symmetric endpoint extension, downsample antialiasing.
    centers = (np.arange(target_size, dtype=np.float64) + .5) / scale - .5
    left = np.floor(centers - width / 2).astype(np.int64)
    positions = left[:, None] + np.arange(int(np.ceil(width)) + 2)
    weights = kernel_scale * cubic(kernel_scale * (centers[:, None] - positions))
    weights /= weights.sum(axis=1, keepdims=True)
    periodic = positions % (2 * source_size)
    indices = np.where(periodic < source_size, periodic, 2 * source_size - 1 - periodic)
    return indices, weights


def matlab_bicubic_uint8(image, shape):
    """Separable antialiased bicubic resize of uint8 RGB with double accumulation.

    Resize first along the more-downsampled axis, as MATLAB imresize does.
    Preserve uint8 between spatial passes, as imresizemex does; do not use
    Pillow's different kernel or resize a converted floating image instead.
    """
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Cloud resize requires uint8 RGB")
    output = image.astype(np.float64)
    scales = np.asarray(shape) / np.asarray(image.shape[:2])
    for axis in np.argsort(scales, kind="stable"):
        indices, weights = resize_weights(output.shape[axis], shape[axis])
        moved = np.moveaxis(output, axis, 0)
        # Accumulate taps rather than allocating H*W*channels*kernel_width.
        resized = np.zeros((shape[axis], *moved.shape[1:]), dtype=np.float64)
        for tap in range(indices.shape[1]):
            resized += moved[indices[:, tap]] * weights[:, tap, None, None]
        output = uint8_round(np.moveaxis(resized, 0, axis)).astype(np.float64)
    return uint8_round(output)


def cloud_intensity(cloud, shape):
    scene = matlab_bicubic_uint8(cloud, shape).astype(np.float64)
    gamma = scene.min(axis=(0, 1))  # Author Threshold=0.
    difference = scene - gamma
    denominator = difference.sum(axis=(0, 1))
    if np.any(denominator == 0):
        raise ValueError("Author cloudy lambda is undefined for a constant texture channel")
    numerator = np.where(scene > gamma, scene, 0).sum(axis=(0, 1))
    return difference * (numerator / denominator)


def synthesize(original, intensity, atmospheric_light):
    """A is shared across R/G/B AND subsequent images, exactly as in the script."""
    if original.dtype != np.uint8 or original.shape != intensity.shape:
        raise ValueError("Cloud intensity must align with the uint8 RGB original")
    result = np.empty_like(original)
    a = float(atmospheric_light)
    for channel in range(3):
        cloud = intensity[:, :, channel]
        land = (255 - cloud) / 255 * original[:, :, channel]
        mixed = land + a * cloud
        # MATLAB find returns column-major ties; use its FIRST global maximum.
        flat = int(np.argmax(mixed.ravel(order="F")))
        row, col = np.unravel_index(flat, mixed.shape, order="F")
        if cloud[row, col] == 0:
            # Author0/0 gives NaN; its comparison to A is false.
            adjusted = np.nan
        else:
            adjusted = (255 - land[row, col]) / cloud[row, col]
        if adjusted < a:
            a = float(adjusted)
        result[:, :, channel] = uint8_round(land + a * cloud)
    return result, a


def generate(data_root, clouds_root, out_dir, resume=False):
    data_root, clouds_root, out = (
        Path(path).resolve() for path in (data_root, clouds_root, out_dir))
    split_file, ids, _ = train_annotations("DIOR", data_root)
    if len(ids) != TRAIN_COUNT:
        raise ValueError(f"Expected exactly{TRAIN_COUNT} approved DIOR TRAIN IDs")
    ids = sorted(ids)  # Author dir('*.png') ordering, restricted to TRAIN.
    images = required_images(ids, {"clean": data_root / "JPEGImages"})["clean"]
    cloud_names = sorted(f"{i}.png" for i in range(1, 31))
    order = json.loads((clouds_root / "order.json").read_text())
    if (order["zip_sha256"] != CLOUD_ARCHIVE_SHA256
            or order["pinned_matlab_compatible_order"] != cloud_names):
        raise ValueError("Cloud provenance/order differs from the staged author archive")
    if not (clouds_root / "NOTICE.md").is_file():
        raise FileNotFoundError("The cloud texture attribution NOTICE.md is required")
    for name in cloud_names:
        if not (clouds_root / "png" / name).is_file():
            raise FileNotFoundError(clouds_root / "png" / name)
    recipe = {
        "algorithm": AUTHOR_SOURCE, "source_commit": AUTHOR_COMMIT,
        "implementation": "independent Python translation; no author source vendored",
        "split": "train", "split_file": str(split_file), "image_ids": ids,
        "clean_root": str(data_root / "JPEGImages"),
        "cloud_root": str(clouds_root), "cloud_archive_sha256": CLOUD_ARCHIVE_SHA256,
        "cloud_order": cloud_names, "threshold": 0, "beta": 255,
        "initial_atmospheric_light": .95,
        "state": "continuous across RGB channels and lexicographic TRAIN images",
        "resize": "Keys a=-0.5 bicubic; downsample antialias; symmetric; uint8 round each axis",
        "image_io": "Pillow JPEG decode to RGB; lossless PNG output; alpha ignored as MATLAB imread",
        "png_compression_level": 1,
    }
    if out.exists():
        if not resume:
            raise FileExistsError(f"Existing output needs explicit --resume: {out}")
        if json.loads((out / "recipe.json").read_text()) != recipe:
            raise ValueError("Cannot adopt cloudy outputs with a different recipe")
        extras = {p.stem for p in out.glob("*.png")} - set(ids)
        if extras:
            raise ValueError(f"Output contains non-TRAIN images: {sorted(extras)[:5]}")
    else:
        out.mkdir(parents=True)
        (out / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")

    @lru_cache(maxsize=30)
    def intensity(name, shape):
        with Image.open(clouds_root / "png" / name) as cloud:
            return cloud_intensity(np.asarray(cloud.convert("RGB")), shape)

    atmospheric_light = .95
    records = []
    for index, image_id in enumerate(ids):
        with Image.open(images[image_id]) as image:
            original = np.asarray(image.convert("RGB"))
        name = cloud_names[index % len(cloud_names)]
        before = atmospheric_light
        pixels, atmospheric_light = synthesize(
            original, intensity(name, original.shape[:2]), atmospheric_light)
        target = out / f"{image_id}.png"
        if target.exists():
            with Image.open(target) as existing:
                if not np.array_equal(np.asarray(existing.convert("RGB")), pixels):
                    raise ValueError(f"Existing cloudy image differs from author-state replay: {target}")
        else:
            partial = target.with_suffix(".png.partial")
            Image.fromarray(pixels).save(partial, format="PNG", compress_level=1)
            partial.replace(target)
        records.append(dict(image_id=image_id, cloud=name, source=str(images[image_id]),
                            output=str(target), shape=list(original.shape),
                            atmospheric_light_before=before,
                            atmospheric_light_after=atmospheric_light))
    summary = {
        "status": "complete", "scope": "DIOR TRAIN cloudy only", "images": len(records),
        "final_atmospheric_light": atmospheric_light,
        "code_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "recipe": str(out / "recipe.json"), "records": records,
        "resume_policy": "Replay from A=.95, compare existing pixels exactly, never reset A per image",
        "bitwise_matlab_runtime_comparison": "not_run; no MATLAB/Octave runtime available",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "clouds-root", "out-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--resume", action="store_true")
    summary = generate(**vars(parser.parse_args()))
    print(json.dumps({k: summary[k] for k in ("status", "images", "final_atmospheric_light")}))


if __name__ == "__main__":
    main()
