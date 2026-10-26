from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from generate_rsar_corruptions import Job, _check_diff, _process_one
from rsar_interference_generator import default_rsar_corruptions, stable_int_seed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply the seven RSAR interference types to SIMDATA_ship PNG images only."
    )
    parser.add_argument("--source-root", type=Path, default=Path("/myfile/dataset/SIMDATA_ship"))
    parser.add_argument("--output-root", type=Path, default=Path("/myfile/dataset/SIMDATA_ship_interference"))
    parser.add_argument("--types", default="", help="Comma-separated interference names; default all seven")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=0, help="Limit each class for a smoke test")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.workers < 1 or args.max_images < 0:
        parser.error("--workers must be >= 1 and --max-images must be >= 0")
    source = args.source_root.resolve()
    output = args.output_root.resolve()
    if source == output or source in output.parents or output in source.parents:
        parser.error("source and output roots must be separate")
    if not source.is_dir():
        raise FileNotFoundError(source)

    specs = {spec.name: spec for spec in default_rsar_corruptions()}
    names = [name.strip() for name in args.types.split(",") if name.strip()] if args.types else list(specs)
    if not names or any(name not in specs for name in names):
        parser.error(f"unknown/empty --types; supported: {', '.join(specs)}")

    classes = sorted(path for path in source.iterdir() if path.is_dir() and (path / "png").is_dir())
    if not classes:
        raise RuntimeError(f"No class/png directories found in {source}")
    images_by_class = {}
    for class_dir in classes:
        images = sorted(path for path in (class_dir / "png").iterdir() if path.suffix.lower() == ".png")
        if args.max_images:
            images = images[:args.max_images]
        if not images:
            raise RuntimeError(f"No PNG images in {class_dir / 'png'}")
        images_by_class[class_dir.name] = images

    failures = 0
    for name in names:
        spec = specs[name]
        jobs = []
        for class_name, images in images_by_class.items():
            destination = output / name / class_name / "png"
            destination.mkdir(parents=True, exist_ok=True)
            for src in images:
                jobs.append(
                    Job(
                        src=str(src),
                        dst=str(destination / src.name),
                        seed=args.seed ^ stable_int_seed(str(src), name, str(args.seed)),
                        itype=spec.itype,
                        params=spec.params,
                        overwrite=args.overwrite,
                    )
                )
        print(f"[SIMDATA_ship/{name}] PNG jobs={len(jobs)} classes={len(classes)} workers={args.workers}", flush=True)
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
        if failed == 0:
            for class_name, images in images_by_class.items():
                _check_diff(
                    images[:min(8, len(images))],
                    clean_root=source / class_name / "png",
                    corrupt_root=output / name / class_name / "png",
                )

    print(f"complete: types={len(names)} PNG images={sum(map(len, images_by_class.values())) * len(names)} failures={failures}", flush=True)
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
