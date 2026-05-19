#!/usr/bin/env python3
"""Generate DIOR-C corruption image folders for SFOD-RS.

The SFOD-RS paper uses 19 ImageNet-C style corruptions at severity 3 on the
original DIOR val/test images. The resulting folders are consumed by configs as:

    <DIOR_ROOT>/Corruption/JPEGImages-${corrupt}

Example:
    python tools/dataset/generate_dior_corruptions.py \
        --dior-root /home/storageSDA1/Dataset/DIOR \
        --splits val test \
        --severity 3 \
        --corruptions all
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DIOR_C_CORRUPTIONS = (
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "speckle_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "gaussian_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "spatter",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
    "saturate",
)

ALIASES = {
    # README has a typo in one example folder name; configs should use brightness.
    "brigtness": "brightness",
}


@dataclass(frozen=True)
class ImageItem:
    src: Path
    name: str


@dataclass(frozen=True)
class Task:
    src: Path
    dst: Path
    corruption: str
    severity: int
    seed: int
    overwrite: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate DIOR-C folders under DIOR/Corruption."
    )
    parser.add_argument(
        "--dior-root",
        type=Path,
        default=Path("/home/storageSDA1/Dataset/DIOR"),
        help="DIOR root directory.",
    )
    parser.add_argument(
        "--src-dir",
        default="JPEGImages",
        help="Clean source image directory relative to DIOR root.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Output root. Defaults to <DIOR_ROOT>/Corruption.",
    )
    parser.add_argument(
        "--split-dir",
        default="ImageSets",
        help="Split directory relative to DIOR root.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["val", "test"],
        help='Split names to corrupt, e.g. "val test". Use "all" for all src images.',
    )
    parser.add_argument(
        "--suffix",
        default=".jpg",
        help="Image suffix appended to split ids that do not include an extension.",
    )
    parser.add_argument(
        "--corruptions",
        nargs="+",
        default=["all"],
        help="Corruption names, comma-separated names, or all.",
    )
    parser.add_argument(
        "--severity",
        type=int,
        default=3,
        choices=(1, 2, 3, 4, 5),
        help="ImageNet-C severity level. SFOD-RS uses 3.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
        help="Parallel workers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed for stochastic corruptions.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit images per corruption for smoke tests. Do not use for reproduction.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate existing output images.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the plan; do not import imagecorruptions or write files.",
    )
    return parser.parse_args()


def normalize_corruptions(values: Iterable[str]) -> list[str]:
    names: list[str] = []
    for value in values:
        for raw_name in value.split(","):
            name = raw_name.strip()
            if not name:
                continue
            name = ALIASES.get(name, name)
            if name in {"all", "dior-c", "dior_c"}:
                names.extend(DIOR_C_CORRUPTIONS)
            else:
                names.append(name)

    unique_names: list[str] = []
    seen = set()
    for name in names:
        if name in seen:
            continue
        if name not in DIOR_C_CORRUPTIONS:
            allowed = ", ".join(DIOR_C_CORRUPTIONS)
            raise ValueError(f"Unsupported corruption '{name}'. Allowed: {allowed}")
        seen.add(name)
        unique_names.append(name)
    return unique_names


def resolve_source_image(src_dir: Path, image_id: str, suffix: str) -> ImageItem:
    image_id = image_id.strip().split()[0]
    rel_name = image_id if Path(image_id).suffix else f"{image_id}{suffix}"
    src = src_dir / rel_name
    if src.exists():
        return ImageItem(src=src, name=rel_name)

    candidates = sorted(src_dir.glob(f"{image_id}.*"))
    if candidates:
        return ImageItem(src=candidates[0], name=candidates[0].name)

    raise FileNotFoundError(f"Image id '{image_id}' not found in {src_dir}")


def collect_images(
    dior_root: Path, src_dir_name: str, split_dir_name: str, splits: list[str], suffix: str
) -> list[ImageItem]:
    src_dir = dior_root / src_dir_name
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Source image directory not found: {src_dir}")

    if len(splits) == 1 and splits[0].lower() == "all":
        return [ImageItem(src=p, name=p.name) for p in sorted(src_dir.iterdir()) if p.is_file()]

    split_dir = dior_root / split_dir_name
    items_by_name: dict[str, ImageItem] = {}
    for split in splits:
        split_file = split_dir / f"{split}.txt"
        if not split_file.is_file():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        for line in split_file.read_text().splitlines():
            image_id = line.strip()
            if not image_id:
                continue
            item = resolve_source_image(src_dir, image_id, suffix)
            items_by_name[item.name] = item
    return list(items_by_name.values())


def stable_seed(base_seed: int, corruption: str, image_name: str) -> int:
    digest = hashlib.blake2b(
        f"{base_seed}:{corruption}:{image_name}".encode("utf-8"), digest_size=4
    ).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def patch_skimage_for_imagecorruptions() -> None:
    """Make old imagecorruptions releases work with newer scikit-image.

    imagecorruptions calls skimage.filters.gaussian(..., multichannel=True).
    scikit-image >= 0.20 removed that argument in favor of channel_axis.
    """
    import inspect

    import skimage.filters

    gaussian = skimage.filters.gaussian
    if "multichannel" in inspect.signature(gaussian).parameters:
        return

    def gaussian_compat(
        image,
        sigma=1,
        output=None,
        mode="nearest",
        cval=0,
        preserve_range=False,
        truncate=4.0,
        *,
        multichannel=None,
        channel_axis=None,
    ):
        if multichannel is not None and channel_axis is None:
            channel_axis = -1 if multichannel else None
        return gaussian(
            image,
            sigma=sigma,
            output=output,
            mode=mode,
            cval=cval,
            preserve_range=preserve_range,
            truncate=truncate,
            channel_axis=channel_axis,
        )

    skimage.filters.gaussian = gaussian_compat

    # If imagecorruptions was imported before forking worker processes, update
    # the function reference captured inside its corruptions module as well.
    try:
        import imagecorruptions.corruptions as corruptions_mod

        corruptions_mod.gaussian = gaussian_compat
    except ImportError:
        pass


def corrupt_one(task: Task) -> str:
    if task.dst.exists() and not task.overwrite:
        return "skipped"

    import numpy as np
    from PIL import Image

    patch_skimage_for_imagecorruptions()
    from imagecorruptions import corrupt

    np.random.seed(task.seed)
    image = Image.open(task.src).convert("RGB")
    image_np = np.asarray(image)
    corrupted = corrupt(
        image_np, severity=task.severity, corruption_name=task.corruption
    )
    corrupted = np.asarray(corrupted).clip(0, 255).astype(np.uint8)
    task.dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(corrupted).save(task.dst, quality=95)
    return "written"


def ensure_backend_available() -> None:
    try:
        patch_skimage_for_imagecorruptions()
        import imagecorruptions  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: imagecorruptions. Install it first, e.g. "
            "`pip install imagecorruptions` in the SFOD-RS environment."
        ) from exc


def run_corruption(
    corruption: str,
    images: list[ImageItem],
    out_root: Path,
    severity: int,
    workers: int,
    seed: int,
    overwrite: bool,
) -> tuple[int, int]:
    out_dir = out_root / f"JPEGImages-{corruption}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        Task(
            src=item.src,
            dst=out_dir / item.name,
            corruption=corruption,
            severity=severity,
            seed=stable_seed(seed, corruption, item.name),
            overwrite=overwrite,
        )
        for item in images
    ]

    written = 0
    skipped = 0
    done = 0
    total = len(tasks)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(corrupt_one, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            written += int(result == "written")
            skipped += int(result == "skipped")
            done += 1
            if done == total or done % 500 == 0:
                print(
                    f"[{corruption}] {done}/{total} "
                    f"(written={written}, skipped={skipped})",
                    flush=True,
                )
    return written, skipped


def main() -> int:
    args = parse_args()
    dior_root = args.dior_root.resolve()
    out_root = (args.out_root or (dior_root / "Corruption")).resolve()
    corruptions = normalize_corruptions(args.corruptions)
    splits = [split.lower() for split in args.splits]
    images = collect_images(dior_root, args.src_dir, args.split_dir, splits, args.suffix)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be a positive integer")
        images = images[: args.limit]

    print(f"DIOR root: {dior_root}")
    print(f"Source images: {dior_root / args.src_dir}")
    print(f"Output root: {out_root}")
    print(f"Splits: {' '.join(splits)}")
    print(f"Images: {len(images)}")
    print(f"Severity: {args.severity}")
    print(f"Corruptions: {', '.join(corruptions)}")
    print(f"Workers: {args.workers}")

    if args.dry_run:
        for corruption in corruptions:
            print(f"Would create: {out_root / f'JPEGImages-{corruption}'}")
        return 0

    ensure_backend_available()

    total_written = 0
    total_skipped = 0
    for corruption in corruptions:
        written, skipped = run_corruption(
            corruption=corruption,
            images=images,
            out_root=out_root,
            severity=args.severity,
            workers=args.workers,
            seed=args.seed,
            overwrite=args.overwrite,
        )
        total_written += written
        total_skipped += skipped

    print(f"Done. written={total_written}, skipped={total_skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
