"""Finite, target-VAL-only TAM fitting; GPU execution belongs to the compute owner.

Launch with PYTHONNOUSERSITE=1 at interpreter startup and IRAOD_GPU_LOCKED=1:
    python -m experiments.comparison.train_tam --help

Each dataset/domain gets one seed42 fit shared by detector seeds42/43/44. Deterministic shuffled
streams fix author sampling nondeterminism; they are not bitwise author replay.
Smoke outputs are NON_RESULT and cannot certify a formal TAM.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from experiments.comparison.extension_training import TARGET_VAL_SIZE, target_val
from experiments.comparison.report_qualitative import validate_plan
from experiments.comparison.result_completion import DOMAINS
from experiments.comparison.tam_artifacts import (
    BGR_MEAN, FIT_SEED, FORMAL_STEPS, NORMALIZATION, SCHEMA, checkpoint_payload,
)
from experiments.comparison import host_binding as host
from experiments.comparison.numerical_integrity import optimizer_boundary


ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_SFYOLO_SHA = "c84dfad79a889b5f172d7192ee0379edfd738835"
BATCH_SIZE = 8
SEEDS = (FIT_SEED,)
GPUS = host.approved_gpus()
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def reproducibility(seed, workers):
    return {
        "sampling": "independent infinite shuffled permutations; no wrap reseeding",
        "content_seed": seed, "style_seed": seed + 1,
        "global_rngs": ["torch", "random", "numpy"],
        "worker_rngs": "torch initial_seed -> torch/random/numpy",
        "workers_per_loader": workers, "drop_last": True,
        "cudnn_deterministic": True, "cudnn_benchmark": False,
        "replay": "Fixes author sampling nondeterminism; not bitwise author replay",
        "preprocessing": {
            "decode": "PIL RGB", "resize_wh": [800, 600], "resize": "bilinear",
            "random_crop_wh": [128, 128], "horizontal_flip_probability": 0.5,
            "output": "float32 CHW BGR, 0..255 minus BGR mean",
            "bgr_mean": list(BGR_MEAN),
        },
    }


def discover_target_images(base_plan, dataset, domain, seed, workers=16):
    """Resolve only the exact A/source run's target VAL; never read annotations."""
    if dataset not in DOMAINS or domain not in DOMAINS[dataset]:
        raise ValueError("Unsupported dataset/domain")
    if seed not in SEEDS:
        raise ValueError("TAM fit seed must be 42; detector seeds reuse this fit")
    plan = host.read_json(base_plan)
    validate_plan(plan)
    if plan.get("adaptation_seed", 42) != 42:
        raise ValueError("The accepted base plan must use adaptation seed42")
    source = next(run for run in plan["runs"]
                  if (run["dataset"], run["domain"], run["method"], run["role"])
                  == (dataset, domain, "A", "source"))
    root = Path(target_val(dataset, domain, source["img_prefix"])).resolve()
    paths = sorted((path.resolve() for path in root.rglob("*")
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
                   key=lambda path: (path.stem, str(path)))
    ids = [path.stem for path in paths]
    if len(ids) != TARGET_VAL_SIZE[dataset] or len(set(ids)) != len(ids):
        raise ValueError(
            f"{dataset} VAL requires {TARGET_VAL_SIZE[dataset]} images with unique stems; "
            f"found {len(ids)} images, {len(set(ids))} unique stems")
    return {
        "dataset": dataset, "domain": domain, "seed": seed, "split": "val",
        "root": host.map_path(root), "image_ids": ids,
        "image_paths": [host.map_path(path) for path in paths],
        "selection": "All image files in target VAL; no labels or GT-presence filtering",
        "reproducibility": reproducibility(seed, workers),
    }


class InfinitePermutationSampler(Sampler):
    """Every permutation visits all IDs; a local RNG continues across wraps."""

    def __init__(self, size, seed):
        self.size = size
        self.seed = seed

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed)
        while True:
            yield from torch.randperm(self.size, generator=generator).tolist()


def preprocess_image(image):
    image = image.convert("RGB").resize((800, 600), Image.Resampling.BILINEAR)
    left, top = random.randint(0, 800 - 128), random.randint(0, 600 - 128)
    image = image.crop((left, top, left + 128, top + 128))
    if random.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    bgr = np.asarray(image, dtype=np.float32)[:, :, ::-1].copy()
    bgr -= np.asarray(BGR_MEAN, dtype=np.float32)
    return torch.from_numpy(bgr.transpose(2, 0, 1).copy())


class TargetImages(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return preprocess_image(image)


def seed_worker(_worker_id):
    seed = torch.initial_seed()
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def make_loaders(manifest, workers):
    dataset = TargetImages(manifest["image_paths"])
    return tuple(
        DataLoader(
            dataset, batch_size=BATCH_SIZE,
            sampler=InfinitePermutationSampler(len(dataset), seed),
            drop_last=True, num_workers=workers, worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(seed),
        )
        for seed in (manifest["seed"], manifest["seed"] + 1)
    )


def train_loop(module, content_loader, style_loader, steps, device, log_file):
    """Run exactly steps outer loops, each with decoder then joint F1/F2 updates."""
    if not 0 < steps <= FORMAL_STEPS:
        raise ValueError("Steps must be positive and at most 160000")
    optimizer_d, optimizer_f = module.make_optimizers()
    content, style = iter(content_loader), iter(style_loader)
    log = csv.DictWriter(log_file, fieldnames=("iteration", "decoder", "moments", "lr"))
    log.writeheader()
    log_file.flush()
    # TAM already checks both losses before backward and its final payload before
    # save. Add only the missing actual-step boundary, including bound TAM code.
    with optimizer_boundary():
        for iteration in range(steps):
            losses = module.alternating_step(
                next(content).to(device), next(style).to(device),
                optimizer_d, optimizer_f, iteration)
            log.writerow({"iteration": iteration, **{
                key: float(losses[key]) for key in ("decoder", "moments", "lr")}})
            if (iteration + 1) % 100 == 0:
                log_file.flush()
    log_file.flush()
    return steps


def _write_json(path, payload):
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(payload, indent=2) + "\n")
    partial.replace(path)


def run_training(*, base_plan, dataset, domain, seed, vgg_weights, out_dir,
                 steps=FORMAL_STEPS, workers=16, device="cuda:0", module_factory=None):
    """Own a NEW output directory through terminal state; no resume or CPU fallback.

    Tests explicitly inject a tiny module and CPU device. Production main checks
    GPU ownership/environment before calling this function.
    """
    if not 0 < steps <= FORMAL_STEPS or workers < 0:
        raise ValueError("Invalid steps or workers")
    out = Path(host.map_path(out_dir)).resolve()
    out.mkdir(parents=True, exist_ok=False)
    terminal = {"schema": SCHEMA, "status": "failed", "requested_steps": steps,
                "identity": {"dataset": dataset, "domain": domain, "seed": seed},
                "loop_completed": False}
    checkpoint = out / "tam.pth"
    partial = out / "tam.pth.partial"
    try:
        encoder = host.read_path(vgg_weights).resolve()
        if not encoder.is_file() or not encoder.stat().st_size:
            raise ValueError("VGG weights must be an explicit existing nonempty external file")
        manifest = discover_target_images(base_plan, dataset, domain, seed, workers)
        code_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        config = {
            "schema": SCHEMA, "identity": terminal["identity"], "split": "val",
            "base_plan": host.map_path(Path(host.map_path(base_plan)).resolve()),
            "training_code_sha": code_sha,
            "official_sfyolo_commit": OFFICIAL_SFYOLO_SHA,
            "encoder_weights": host.map_path(encoder), "encoder_bytes": encoder.stat().st_size,
            "steps": steps, "formal_steps": FORMAL_STEPS, "batch_size": BATCH_SIZE,
            "optimizer_updates": 2 * steps, "detector_seeds": [42, 43, 44],
            "result_scope": "formal" if steps == FORMAL_STEPS else "NON_RESULT",
            "optimizers": "decoder Adam and F1/F2 Adam; module defaults",
            "lr_schedule": "1e-4 / (1 + 5e-5 * iteration_zero_based)",
            "objective": "generic_aerial_code_formula; decoder_then_F1_F2; alpha_train1",
            "normalization": NORMALIZATION,
            "device": str(device), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "reproducibility": manifest["reproducibility"],
        }
        _write_json(out / "training_config.json", config)
        _write_json(out / "image_manifest.json", manifest)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if module_factory is None:
            from sfod.extensions.tam import TargetAugmentationModule
            module_factory = TargetAugmentationModule
        module = module_factory(weights_path=str(encoder)).to(device)
        module.train()
        content, style = make_loaders(manifest, workers)
        with (out / "train_log.csv").open("w", newline="") as log_file:
            completed_steps = train_loop(module, content, style, steps, device, log_file)
        terminal["loop_completed"] = True
        payload = checkpoint_payload(
            module, terminal["identity"], encoder, completed_steps,
            code_sha, out / "image_manifest.json")
        torch.save(payload, partial)
        partial.replace(checkpoint)
        terminal.update(status=payload["status"], training_steps=completed_steps,
                        training_code_sha=code_sha, checkpoint=str(checkpoint))
    finally:
        # Exceptions propagate to the finite job owner; no completed artifact on failure.
        if terminal["status"] == "failed":
            partial.unlink(missing_ok=True)
            checkpoint.unlink(missing_ok=True)
        _write_json(out / "terminal.json", terminal)
    return terminal


def configure_gpu(gpu):
    """Validate ownership before any CUDA/model call, then set physical GPU visibility."""
    if gpu not in GPUS:
        raise ValueError(f"GPU must be one of {GPUS}")
    if os.environ.get("IRAOD_GPU_LOCKED") != "1":
        raise RuntimeError("Compute owner must hold the GPU lock: IRAOD_GPU_LOCKED=1")
    if os.environ.get("PYTHONNOUSERSITE") != "1" or not sys.flags.no_user_site:
        raise RuntimeError("Start the interpreter with PYTHONNOUSERSITE=1")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-plan", type=Path, required=True)
    parser.add_argument("--dataset", choices=tuple(DOMAINS), required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--vgg-weights", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, choices=GPUS, required=True)
    parser.add_argument("--smoke-steps", type=int, help="NON_RESULT: 1..159999 only")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    if args.domain not in DOMAINS[args.dataset]:
        parser.error("--domain must belong to the selected dataset")
    if args.smoke_steps is not None and not 0 < args.smoke_steps < FORMAL_STEPS:
        parser.error("--smoke-steps must be positive and smaller than 160000")
    if args.workers < 0:
        parser.error("--workers must be nonnegative")
    return args


def main(argv=None):
    args = parse_args(argv)
    configure_gpu(args.gpu)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; TAM training has no CPU fallback")
    return run_training(
        base_plan=args.base_plan, dataset=args.dataset, domain=args.domain,
        seed=args.seed, vgg_weights=args.vgg_weights, out_dir=args.out_dir,
        steps=FORMAL_STEPS if args.smoke_steps is None else args.smoke_steps,
        workers=args.workers)


if __name__ == "__main__":
    main()
