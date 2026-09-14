#!/usr/bin/env python
"""Stream labeled-TRAIN oracle AABB patches, or count them without decoding pixels."""

import argparse
import csv
import json
import math
import sys
import xml.etree.ElementTree as ET
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.comparison.labels import CLASSES
from tools.build_rsar_sarclip_patches import (
    DEFAULT_CORRUPTIONS,
    IMAGE_SUFFIXES,
    METADATA_FIELDS,
    crop_aabb,
    expanded_aabb,
    safe_name,
)

CORRUPTIONS = {
    "RSAR": tuple(DEFAULT_CORRUPTIONS),
    "DIOR": ("brightness", "cloudy", "contrast"),
}
FIELDS = ["dataset", *METADATA_FIELDS]
EXPAND = 0.4
DIOR_COORDINATES = (
    "x_left_top", "y_left_top",
    "x_right_top", "y_right_top",
    "x_right_bottom", "y_right_bottom",
    "x_left_bottom", "y_left_bottom",
)


def polygon(values, location):
    """Keep the legacy float32 geometry, rejecting malformed annotations."""
    try:
        poly = np.array([float(value) for value in values], dtype=np.float32).reshape(4, 2)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{location}: invalid polygon coordinates") from exc
    if not np.isfinite(poly).all():
        raise ValueError(f"{location}: non-finite polygon coordinates")
    return poly


def read_rsar_objects(path):
    """Yield (original zero-based line index, polygon, class); retain difficulty."""
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            text = line.strip()
            if not text or text.startswith(("imagesource:", "gsd:")):
                continue
            location = f"{path}:{index + 1}"
            parts = text.split()
            if len(parts) != 10:
                raise ValueError(f"{location}: expected 8 coordinates, class, difficulty")
            if parts[8] not in CLASSES["RSAR"]:
                raise ValueError(f"{location}: unknown RSAR class {parts[8]!r}")
            try:
                int(parts[9])
            except ValueError as exc:
                raise ValueError(f"{location}: invalid difficulty {parts[9]!r}") from exc
            yield index, polygon(parts[:8], location), parts[8]


def read_dior_objects(path):
    """Read original oriented XML coordinates in object order, never fit an HBB."""
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"{path}: malformed DIOR XML: {exc}") from exc
    if root.tag != "annotation":
        raise ValueError(f"{path}: expected DIOR annotation root")
    for index, obj in enumerate(root.findall("object")):
        location = f"{path}:object {index + 1}"
        name = (obj.findtext("name") or "").strip().lower()
        if name not in CLASSES["DIOR"]:
            raise ValueError(f"{location}: unknown DIOR class {name!r}")
        box = obj.find("robndbox")
        if box is None:
            raise ValueError(f"{location}: missing oriented robndbox")
        yield index, polygon([box.findtext(key) for key in DIOR_COORDINATES], location), name


def train_annotations(dataset, data_root):
    if dataset == "RSAR":
        source = data_root / "train" / "annfiles"
        if not source.is_dir():
            raise FileNotFoundError(f"Missing TRAIN annotation directory: {source}")
        annotations = {path.stem: path for path in sorted(source.glob("*.txt"))}
        ids = list(annotations)
    else:
        source = data_root / "ImageSets" / "train.txt"
        ids = [line.strip() for line in source.read_text(encoding="utf-8").splitlines()
               if line.strip()]
        annotation_root = data_root / "Annotations" / "Oriented Bounding Boxes"
        annotations = {image_id: annotation_root / f"{image_id}.xml" for image_id in ids}
    if not ids:
        raise ValueError(f"{source}: no TRAIN IDs")
    seen = set()
    for image_id in ids:
        # The legacy image lookup is case-insensitive, so IDs must be unique there too.
        key = image_id.lower()
        if key in seen:
            raise ValueError(f"{source}: duplicate TRAIN ID {image_id!r}")
        seen.add(key)
        if not annotations[image_id].is_file():
            raise FileNotFoundError(f"Missing TRAIN annotation: {annotations[image_id]}")
    return source, ids, annotations


def resolve_image_roots(dataset, data_root, image_roots):
    if image_roots is not None:
        if not isinstance(image_roots, dict) or set(image_roots) != set(CORRUPTIONS[dataset]):
            raise ValueError(f"--image-roots must map exactly {list(CORRUPTIONS[dataset])}")
        roots = {domain: Path(image_roots[domain]).resolve() for domain in CORRUPTIONS[dataset]}
        for domain, root in roots.items():
            if not root.is_dir():
                raise FileNotFoundError(f"Missing supplied TRAIN image directory for {domain}: {root}")
        return roots
    if dataset == "RSAR":
        return {domain: data_root / "corruptions" / domain / "train" / "images"
                for domain in CORRUPTIONS[dataset]}
    return {domain: data_root / "Corruption" / f"JPEGImages-{domain}"
            for domain in CORRUPTIONS[dataset]}


def required_images(ids, roots):
    """Index filenames, not pixels; report all missing TRAIN IDs across domains."""
    selected = {}
    failures = []
    required = {image_id.lower() for image_id in ids}
    for domain, root in roots.items():
        candidates = {}
        if root.is_dir():
            for path in root.iterdir():
                key = path.stem.lower()
                if key in required and path.suffix.lower() in IMAGE_SUFFIXES and path.is_file():
                    candidates.setdefault(key, []).append(path)
        # Preserve the legacy extension preference without rescanning a directory per ID.
        images = {}
        for image_id in ids:
            matches = candidates.get(image_id.lower(), [])
            if matches:
                images[image_id] = min(
                    matches,
                    key=lambda p: (
                        not (p.stem == image_id and p.suffix in IMAGE_SUFFIXES),
                        IMAGE_SUFFIXES.index(p.suffix.lower()), p.name,
                    ),
                )
        missing = [image_id for image_id in ids if image_id not in images]
        if missing:
            failures.append(
                f"{domain}: missing {len(missing)}/{len(ids)} required TRAIN images "
                f"in {root}; examples: {', '.join(missing[:10])}"
            )
        selected[domain] = images
    if failures:
        raise FileNotFoundError("Incomplete corrupt TRAIN data:\n" + "\n".join(failures))
    return selected


def metadata_row(dataset, path, image, annotation, domain, name, poly, box):
    row = {
        "dataset": dataset,
        "patch_path": str(path),
        "image_name": image.name,
        "ann_file": str(annotation),
        "split": "train",
        "corruption": domain,
        "class_name": name,
        "class_id": CLASSES[dataset].index(name),
        "crop_mode": "aabb",
        "crop_expand": EXPAND,
        **dict(zip(("x1", "y1", "x2", "y2"), box)),
    }
    for index, (x, y) in enumerate(poly, start=1):
        row[f"poly_x{index}"] = float(x)
        row[f"poly_y{index}"] = float(y)
    return row


def build(dataset, data_root, out, *, image_roots=None, count_only=False):
    """Build into a new root. Failure leaves only partial artifacts, never success CSV."""
    data_root, out = Path(data_root).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError(f"Output root must be new: {out}")
    source, ids, annotations = train_annotations(dataset, data_root)
    roots = resolve_image_roots(dataset, data_root, image_roots)
    images = required_images(ids, roots)
    out.mkdir(parents=True)
    classes = CLASSES[dataset]
    per_class = dict.fromkeys(classes, 0)
    per_corruption = dict.fromkeys(roots, 0)
    per_corruption_class = {domain: dict.fromkeys(classes, 0) for domain in roots}
    rejected_class = dict.fromkeys(classes, 0)
    rejected_corruption = dict.fromkeys(roots, 0)
    reader = read_rsar_objects if dataset == "RSAR" else read_dior_objects
    partial = out / "metadata.csv.partial"
    count = rejected = 0
    with ExitStack() as stack:
        if not count_only:
            stream = stack.enter_context(partial.open("x", newline="", encoding="utf-8"))
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
        for image_id in ids:
            annotation = annotations[image_id]
            # Only one image's objects are resident; never accumulate metadata rows.
            objects = list(reader(annotation))
            for domain in roots:
                image_path = images[domain][image_id]
                with Image.open(image_path) as header:
                    # PIL reads size from headers; count-only never loads/crops pixels.
                    with ExitStack() as image_stack:
                        image = header if count_only else image_stack.enter_context(header.convert("RGB"))
                        for index, poly, name in objects:
                            if count_only:
                                box = expanded_aabb(poly, EXPAND, image.width, image.height)
                            else:
                                patch, box = crop_aabb(image, poly, EXPAND)
                            if box is None:
                                rejected += 1
                                rejected_class[name] += 1
                                rejected_corruption[domain] += 1
                                continue
                            if not count_only:
                                directory = out / dataset / "aabb" / name
                                directory.mkdir(parents=True, exist_ok=True)
                                path = directory / safe_name(
                                    f"{domain}_train_{image_path.stem}_{index}_e{EXPAND:g}.png"
                                )
                                with patch:
                                    patch.save(path)
                                writer.writerow(metadata_row(
                                    dataset, path, image_path, annotation, domain, name, poly, box,
                                ))
                            count += 1
                            per_class[name] += 1
                            per_corruption[domain] += 1
                            per_corruption_class[domain][name] += 1
    summary = {
        "dataset": dataset,
        "split": "train",
        "status": "count_only_no_patch_artifacts" if count_only else "complete",
        "crop_mode": "aabb",
        "crop_expand": EXPAND,
        "force_rgb": True,
        "classes": list(classes),
        "valid_crop_count": count,
        "per_class": per_class,
        "per_corruption": per_corruption,
        "per_corruption_class": per_corruption_class,
        "rejections": {
            "reason": "empty_expanded_aabb_after_clipping",
            "total": rejected,
            "per_class": rejected_class,
            "per_corruption": rejected_corruption,
        },
        "source_train": {
            "data_root": str(data_root),
            "ids_source": str(source),
            "ids": ids,
            "annotation_root": str(annotations[ids[0]].parent),
            "image_roots": {domain: str(root) for domain, root in roots.items()},
        },
        "canonical_budget": {
            "epochs": 10,
            "batch_size": 64,
            "updates": 10 * math.ceil(count / 64),
            "sampler": "WeightedRandomSampler",
            "draws_per_epoch": count,
            "replacement": True,
            "drop_last": False,
        },
    }
    # Both are terminal artifacts; prepare JSON before publishing either of them.
    summary_partial = out / "summary.json.partial"
    summary_partial.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if not count_only:
        partial.rename(out / "metadata.csv")
    summary_partial.rename(out / "summary.json")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=tuple(CLASSES))
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--image-roots", type=json.loads,
                        help="JSON object mapping every dataset corruption to a known TRAIN image directory")
    parser.add_argument("--count-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        summary = build(**vars(args))
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
