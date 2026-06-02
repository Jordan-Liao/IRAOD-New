#!/usr/bin/env python
import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from iraod_runtime import ensure_iraod_runtime

os.environ.setdefault("IRAOD_CONDA_PREFIX", "/home/liaojr/anaconda3/envs/cliptorch")
ensure_iraod_runtime()

import numpy as np
from PIL import Image


CLASSES = ["ship", "aircraft", "car", "tank", "bridge", "harbor"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASSES)}
DEFAULT_CORRUPTIONS = [
    "chaff",
    "gaussian_white_noise",
    "point_target",
    "noise_suppression",
    "am_noise_horizontal",
    "smart_suppression",
    "am_noise_vertical",
]
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
METADATA_FIELDS = [
    "patch_path",
    "image_name",
    "ann_file",
    "split",
    "corruption",
    "class_name",
    "class_id",
    "crop_mode",
    "crop_expand",
    "x1",
    "y1",
    "x2",
    "y2",
    "poly_x1",
    "poly_y1",
    "poly_x2",
    "poly_y2",
    "poly_x3",
    "poly_y3",
    "poly_x4",
    "poly_y4",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Build RSAR SARCLIP object patches from DOTA annfiles.")
    parser.add_argument("--data-root", default="/home/storageSDA1/liaojr/dataset/RSAR")
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument("--use-corruptions", type=int, choices=[0, 1], default=1)
    parser.add_argument("--corruptions", nargs="+", default=DEFAULT_CORRUPTIONS)
    parser.add_argument("--out", required=True)
    parser.add_argument("--crop-modes", nargs="+", choices=["aabb", "rotated"], default=["aabb"])
    parser.add_argument("--crop-expands", nargs="+", type=float, default=[0.2, 0.4, 0.8, 1.2])
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--force-rgb", type=int, choices=[0, 1], default=1)
    return parser.parse_args()


def find_image(image_dir, stem):
    for suffix in IMAGE_SUFFIXES:
        path = image_dir / f"{stem}{suffix}"
        if path.exists():
            return path
    lower_stem = stem.lower()
    for path in image_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and path.stem.lower() == lower_stem:
            return path
    return None


def read_ann_file(path):
    objects = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) < 10:
                continue
            class_name = parts[8]
            if class_name not in CLASS_TO_ID:
                continue
            try:
                coords = [float(value) for value in parts[:8]]
            except ValueError:
                continue
            poly = np.array(coords, dtype=np.float32).reshape(4, 2)
            objects.append((line_idx, poly, class_name))
    return objects


def expanded_aabb(poly, expand, width, height):
    x1 = float(poly[:, 0].min())
    y1 = float(poly[:, 1].min())
    x2 = float(poly[:, 0].max())
    y2 = float(poly[:, 1].max())
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    x1 = max(0.0, x1 - box_w * expand)
    y1 = max(0.0, y1 - box_h * expand)
    x2 = min(float(width), x2 + box_w * expand)
    y2 = min(float(height), y2 + box_h * expand)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def order_poly(poly):
    sums = poly.sum(axis=1)
    diffs = poly[:, 0] - poly[:, 1]
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = poly[np.argmin(sums)]
    ordered[2] = poly[np.argmax(sums)]
    ordered[1] = poly[np.argmax(diffs)]
    ordered[3] = poly[np.argmin(diffs)]
    return ordered


def crop_aabb(image, poly, expand):
    box = expanded_aabb(poly, expand, image.width, image.height)
    if box is None:
        return None, None
    x1, y1, x2, y2 = box
    patch = image.crop((int(np.floor(x1)), int(np.floor(y1)), int(np.ceil(x2)), int(np.ceil(y2))))
    return patch, box


def crop_rotated(image, poly, expand):
    ordered = order_poly(poly)
    center = ordered.mean(axis=0, keepdims=True)
    source = center + (ordered - center) * (1.0 + float(expand))
    top_w = np.linalg.norm(source[1] - source[0])
    bottom_w = np.linalg.norm(source[2] - source[3])
    left_h = np.linalg.norm(source[3] - source[0])
    right_h = np.linalg.norm(source[2] - source[1])
    out_w = max(1, int(round(max(top_w, bottom_w))))
    out_h = max(1, int(round(max(left_h, right_h))))
    resample = getattr(Image, "Resampling", Image).BICUBIC
    patch = image.transform(
        (out_w, out_h),
        Image.QUAD,
        data=tuple(float(v) for v in source.reshape(-1)),
        resample=resample,
    )
    x1, y1 = source[:, 0].min(), source[:, 1].min()
    x2, y2 = source[:, 0].max(), source[:, 1].max()
    return patch, (float(x1), float(y1), float(x2), float(y2))


def safe_name(value):
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value)


def reached_max_per_class(counts, max_per_class):
    if max_per_class is None:
        return False
    return all(counts[class_name] >= max_per_class for class_name in CLASSES)


def build_for_image(image_path, ann_file, split, corruption, args, out_root, counts):
    image = Image.open(image_path)
    if args.force_rgb:
        image = image.convert("RGB")

    rows = []
    objects = read_ann_file(ann_file)
    for line_idx, poly, class_name in objects:
        if args.max_per_class is not None and counts[class_name] >= args.max_per_class:
            continue
        counts[class_name] += 1
        class_id = CLASS_TO_ID[class_name]

        for crop_mode in args.crop_modes:
            for crop_expand in args.crop_expands:
                if crop_mode == "aabb":
                    patch, box = crop_aabb(image, poly, crop_expand)
                else:
                    patch, box = crop_rotated(image, poly, crop_expand)
                if patch is None:
                    continue

                patch_dir = out_root / crop_mode / class_name
                patch_dir.mkdir(parents=True, exist_ok=True)
                file_name = safe_name(
                    f"{corruption}_{split}_{image_path.stem}_{line_idx}_"
                    f"e{crop_expand:g}.png"
                )
                patch_path = patch_dir / file_name
                patch.save(patch_path)

                x1, y1, x2, y2 = box
                row = {
                    "patch_path": str(patch_path.resolve()),
                    "image_name": image_path.name,
                    "ann_file": str(ann_file.resolve()),
                    "split": split,
                    "corruption": corruption,
                    "class_name": class_name,
                    "class_id": class_id,
                    "crop_mode": crop_mode,
                    "crop_expand": crop_expand,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
                for idx, (px, py) in enumerate(poly, start=1):
                    row[f"poly_x{idx}"] = float(px)
                    row[f"poly_y{idx}"] = float(py)
                rows.append(row)
    return rows


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    ann_dir = data_root / args.split / "annfiles"
    if not ann_dir.exists():
        raise FileNotFoundError(f"Annotation directory not found: {ann_dir}")

    corruptions = args.corruptions if args.use_corruptions else ["clean"]
    counts = defaultdict(int)
    rows = []
    ann_files = sorted(ann_dir.glob("*.txt"))
    for corruption in corruptions:
        if reached_max_per_class(counts, args.max_per_class):
            break
        if args.use_corruptions:
            image_dir = data_root / "corruptions" / corruption / args.split / "images"
        else:
            image_dir = data_root / args.split / "images"
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        for ann_file in ann_files:
            image_path = find_image(image_dir, ann_file.stem)
            if image_path is None:
                continue
            rows.extend(build_for_image(image_path, ann_file, args.split, corruption, args, out_root, counts))
            if reached_max_per_class(counts, args.max_per_class):
                break

    metadata_path = out_root / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[build_rsar_sarclip_patches] wrote {len(rows)} patches")
    print(f"[build_rsar_sarclip_patches] metadata: {metadata_path}")


if __name__ == "__main__":
    main()
