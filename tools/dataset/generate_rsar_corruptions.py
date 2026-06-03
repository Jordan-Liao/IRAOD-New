from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

# Migrated from /home/storageSDA1/liaojr/IRAOD/tools/prepare_rsar_corruption.py.
# Keep the helper next to this script so new servers do not depend on the old
# /home/storageSDA1/liaojr/IRAOD checkout.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from rsar_interference_generator import SUPPORTED_EXTS, add_interference, default_rsar_corruptions, stable_int_seed  # noqa: E402


def _log(msg: str) -> None:
    print(f"[generate_rsar_corruptions] {msg}")


def _read_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    return img


def _write_image(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp" + path.suffix)
    ok = cv2.imwrite(str(tmp), img)
    if not ok:
        raise IOError(f"failed to write: {tmp}")
    tmp.replace(path)


def _iter_images(img_dir: Path) -> Iterable[Path]:
    for root, _dirs, files in os.walk(img_dir):
        for fn in files:
            p = Path(root) / fn
            if p.suffix.lower() in SUPPORTED_EXTS:
                yield p


def _ensure_symlink(dst: Path, src: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        if dst.is_dir() and not dst.is_symlink():
            # If a real directory already exists, keep it (user may have generated it manually).
            return
        dst.unlink()
    try:
        rel = os.path.relpath(str(src), str(dst.parent))
        dst.symlink_to(rel)
    except Exception:
        dst.symlink_to(src)


@dataclass(frozen=True)
class Job:
    src: str
    dst: str
    seed: int
    itype: str
    params: dict
    overwrite: bool = False


def _process_one(job: Job) -> tuple[str, str]:
    try:
        src = Path(job.src)
        dst = Path(job.dst)
        if dst.is_file() and not job.overwrite:
            return "skip", str(dst)
        img = _read_image(src)
        out = add_interference(img, itype=str(job.itype), params=dict(job.params), seed=int(job.seed))
        _write_image(dst, out)
        return "ok", str(dst)
    except Exception as e:
        return "fail", f"{job.src} -> {job.dst} ({e})"


def _check_diff(samples: list[Path], *, clean_root: Path, corrupt_root: Path) -> None:
    checked = 0
    identical = 0
    for src in samples:
        rel = src.relative_to(clean_root)
        dst = corrupt_root / rel
        if not dst.is_file():
            continue
        a = _read_image(src)
        b = _read_image(dst)
        if a.shape != b.shape:
            continue
        checked += 1
        if np.array_equal(a, b):
            identical += 1
    if checked == 0:
        raise SystemExit("diff check: no sample pairs checked (unexpected)")
    if identical == checked:
        raise SystemExit(f"diff check failed: all {checked} samples are byte-identical (no corruption applied?)")
    _log(f"diff check: checked={checked} identical={identical}")


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the seven RSAR corruption folders under "
            "RSAR/corruptions/<corruption>/<split>/images."
        )
    )
    parser.add_argument("--data-root", default="dataset/RSAR", help="RSAR root containing train/val/test")
    parser.add_argument(
        "--corruptions",
        default="",
        help="Comma-separated corruption names to generate (default: all 7).",
    )
    parser.add_argument("--splits", default="train,val,test", help="comma-separated splits")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="If >0, limit number of images per split (useful for smoke).",
    )
    parser.add_argument(
        "--diff-samples",
        type=int,
        default=64,
        help="Sample N images and ensure clean != corrupt (0 disables).",
    )
    parser.add_argument(
        "--no-link-legacy",
        action="store_true",
        help="Do not create legacy symlinks: dataset/RSAR/<split>/images-<corrupt> -> dataset/RSAR/corruptions/<corrupt>/<split>/images",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    splits = _split_csv(str(args.splits))
    if not splits:
        raise SystemExit("--splits must not be empty")

    all_specs = {s.name: s for s in default_rsar_corruptions()}
    if args.corruptions.strip():
        names = _split_csv(str(args.corruptions))
        if any(name in {"all", "rsar-c", "rsar_c"} for name in names):
            names = list(all_specs.keys())
    else:
        names = list(all_specs.keys())

    unknown = [n for n in names if n not in all_specs]
    if unknown:
        raise SystemExit(f"unknown corruptions: {unknown}. supported={sorted(all_specs)}")

    out_root = data_root / "corruptions"
    out_root.mkdir(parents=True, exist_ok=True)

    total_ok = 0
    total_skip = 0
    total_fail = 0

    for name in names:
        spec = all_specs[name]
        _log(f"=== corruption={name} (itype={spec.itype}) ===")

        _overwrite = bool(args.overwrite)
        jobs: list[Job] = []
        for split in splits:
            clean_dir = data_root / split / "images"
            if not clean_dir.is_dir():
                raise FileNotFoundError(clean_dir)
            corrupt_dir = out_root / name / split / "images"
            corrupt_dir.mkdir(parents=True, exist_ok=True)

            images = list(_iter_images(clean_dir))
            if args.max_images and int(args.max_images) > 0:
                images = images[: int(args.max_images)]

            for src in images:
                rel = src.relative_to(clean_dir)
                dst = corrupt_dir / rel
                seed = int(args.seed) ^ stable_int_seed(str(src), name, str(args.seed))
                jobs.append(Job(src=str(src), dst=str(dst), seed=seed, itype=spec.itype, params=spec.params, overwrite=_overwrite))

        if not jobs:
            _log("nothing to do (no images found?)")
            continue

        _log(f"generate: jobs={len(jobs)} workers={args.workers} overwrite={bool(args.overwrite)}")
        ok = 0
        skipped = 0
        failed = 0

        chunksize = max(1, len(jobs) // (int(args.workers) * 4))
        with ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
            for i, (status, msg) in enumerate(ex.map(_process_one, jobs, chunksize=chunksize)):
                if status == "ok":
                    ok += 1
                elif status == "skip":
                    skipped += 1
                else:
                    failed += 1
                    _log(f"ERROR: {msg}")
                if (i + 1) % 5000 == 0:
                    _log(f"progress: {i+1}/{len(jobs)} ok={ok} skip={skipped} fail={failed}")

        _log(f"done: ok={ok} skipped={skipped} failed={failed}")
        total_ok += ok
        total_skip += skipped
        total_fail += failed

        if int(args.diff_samples) > 0:
            # Diff-check only on the first split for speed.
            split0 = splits[0]
            clean0 = data_root / split0 / "images"
            corrupt0 = out_root / name / split0 / "images"
            samples = list(_iter_images(clean0))[: int(args.diff_samples)]
            _check_diff(samples, clean_root=clean0, corrupt_root=corrupt0)

        if not bool(args.no_link_legacy):
            for split in splits:
                legacy = data_root / split / f"images-{name}"
                target = (out_root / name / split / "images").resolve()
                _ensure_symlink(legacy, target)

    _log(f"summary: ok={total_ok} skipped={total_skip} failed={total_fail}")
    return 0 if total_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
