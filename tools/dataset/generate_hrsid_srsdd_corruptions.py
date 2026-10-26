from __future__ import annotations

import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from generate_rsar_corruptions import Job, _check_diff, _iter_images, _process_one
from rsar_interference_generator import default_rsar_corruptions, stable_int_seed


DATASETS = {
    "hrsid": {
        "source": Path("/myfile/dataset/HRSID_JPG"),
        "output": Path("/myfile/dataset/HRSID_interference"),
        "image_dirs": ("JPEGImages",),
        "annotation_paths": ("annotations",),
    },
    "srsdd": {
        "source": Path("/myfile/dataset/srsdd"),
        "output": Path("/myfile/dataset/srsdd_interference"),
        "image_dirs": ("train/images", "test/images"),
        "annotation_paths": (
            "train/labels", "train/train.json",
            "test/labels", "test/test.json",
        ),
    },
}


def copy_annotations(source: Path, output: Path, paths: tuple[str, ...]) -> None:
    for relative in paths:
        src = source / relative
        dst = output / relative
        if not src.exists():
            raise FileNotFoundError(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate the seven RSAR interference types for HRSID or SRSdd."
    )
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--types", default="", help="Comma-separated interference names; default all seven")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=0, help="Limit each image directory for a smoke test")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.workers < 1 or args.max_images < 0:
        parser.error("--workers must be >= 1 and --max-images must be >= 0")

    config = DATASETS[args.dataset]
    source = (args.source_root or config["source"]).resolve()
    output = (args.output_root or config["output"]).resolve()
    if source == output or source in output.parents or output in source.parents:
        parser.error("source and output roots must be separate")
    specs = {spec.name: spec for spec in default_rsar_corruptions()}
    names = [name.strip() for name in args.types.split(",") if name.strip()] if args.types else list(specs)
    if not names or any(name not in specs for name in names):
        parser.error(f"unknown/empty --types; supported: {', '.join(specs)}")

    failures = 0
    for name in names:
        spec = specs[name]
        variant_root = output / name
        variant_failures = 0
        for relative in config["image_dirs"]:
            clean_dir = source / relative
            if not clean_dir.is_dir():
                raise FileNotFoundError(clean_dir)
            corrupt_dir = variant_root / relative
            images = sorted(_iter_images(clean_dir))
            if args.max_images:
                images = images[:args.max_images]
            if not images:
                raise RuntimeError(f"No images in {clean_dir}")
            jobs = [
                Job(
                    src=str(src),
                    dst=str(corrupt_dir / src.relative_to(clean_dir)),
                    seed=args.seed ^ stable_int_seed(str(src), name, str(args.seed)),
                    itype=spec.itype,
                    params=spec.params,
                    overwrite=args.overwrite,
                )
                for src in images
            ]
            corrupt_dir.mkdir(parents=True, exist_ok=True)
            print(f"[{args.dataset}/{name}/{relative}] images={len(jobs)} workers={args.workers}", flush=True)
            ok = skipped = failed = 0
            chunksize = max(1, min(64, len(jobs) // (args.workers * 16)))
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for index, (status, message) in enumerate(pool.map(_process_one, jobs, chunksize=chunksize), 1):
                    if status == "ok":
                        ok += 1
                    elif status == "skip":
                        skipped += 1
                    else:
                        failed += 1
                        print(f"ERROR: {message}", flush=True)
                    if index % 500 == 0:
                        print(f"  progress={index}/{len(jobs)} ok={ok} skip={skipped} fail={failed}", flush=True)
            print(f"  done ok={ok} skip={skipped} fail={failed}", flush=True)
            failures += failed
            variant_failures += failed
            if not failed:
                _check_diff(images[:min(8, len(images))], clean_root=clean_dir, corrupt_root=corrupt_dir)
        if variant_failures == 0 and args.max_images == 0:
            copy_annotations(source, variant_root, config["annotation_paths"])
            print(f"  annotations copied to {variant_root}", flush=True)

    print(f"complete: dataset={args.dataset} types={len(names)} failures={failures}", flush=True)
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
