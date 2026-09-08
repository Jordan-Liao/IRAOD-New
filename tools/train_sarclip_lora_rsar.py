#!/usr/bin/env python
import argparse
import csv
import json
import os
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from iraod_runtime import ensure_iraod_runtime

if __name__ == "__main__":
    os.environ.setdefault("IRAOD_CONDA_PREFIX", "/home/liaojr/anaconda3/envs/cliptorch")
    ensure_iraod_runtime()

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from experiments.comparison.labels import CLASSES as DATASET_CLASSES
from sarclip_adapter import (
    ADAPTER_FORMAT,
    inject_lora,
    mark_lora_trainable,
    mark_visual_proj_trainable,
    trainable_state_dict,
)


CLASSES = list(DATASET_CLASSES["RSAR"])
DEFAULT_TEMPLATE = "A SAR image of a {}"
DEFAULT_TEMPLATES = {"RSAR": DEFAULT_TEMPLATE, "DIOR": "an aerial image of a {}"}


def force_math_sdpa():
    """Route attention through the math backend.

    On sm_90 (H100) with torch 2.0.1+cu118 the flash and mem-efficient SDPA
    backends raise "CUDA error: an illegal instruction was encountered" in the
    *backward* pass. Forward succeeds, which is why the failure first appears at
    ``optimizer.step()`` -- CUDA reports asynchronously. It only bites once LoRA
    is injected across every block, because autograd must then traverse each
    attention layer instead of pruning the graph.

    The math backend is slower but numerically identical, and attention is not
    the bottleneck here: only the LoRA factors carry gradients.
    """
    if not torch.cuda.is_available():
        return
    capability = torch.cuda.get_device_capability()
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    print(f"[train_sarclip_lora] sm_{capability[0]}{capability[1]}: "
          "forced math SDPA backend (flash/mem-efficient backward is broken here)")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Fine-tune SARCLIP adapters on labeled TRAIN patches.")
    parser.add_argument("--dataset", choices=["RSAR", "DIOR"], default="RSAR")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--crop-mode", choices=["aabb", "rotated", "all"], default="aabb")
    parser.add_argument("--sarclip-dir", default="/home/storageSDA1/liaojr/SARCLIP")
    parser.add_argument("--sarclip-pretrained", required=True)
    parser.add_argument("--sarclip-cache-dir", default=None)
    parser.add_argument("--sarclip-model", default="ViT-B-32")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--precision", default="fp32")
    parser.add_argument("--templates", default=None)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    # Decoding and resizing the patches is the bottleneck, not the 737k LoRA
    # factors: with num_workers=0 the main process saturated 28 cores on PIL work
    # while the GPU sat at 0%.
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-patches", type=int, default=None,
                        help="Legacy stratified cap; formal oracle recipe leaves this unset")
    parser.add_argument("--train-visual-proj-only", action="store_true")
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("--epochs must be positive for final-epoch selection")
    if args.max_patches is not None and args.max_patches < 1:
        parser.error("--max-patches must be positive")
    if args.templates is None:
        args.templates = DEFAULT_TEMPLATES[args.dataset]
    return args


def split_templates(value):
    if ";" in value:
        return [item.strip() for item in value.split(";") if item.strip()]
    return [value.strip()]


def import_sarclip(sarclip_dir):
    if sarclip_dir:
        sys.path.insert(0, sarclip_dir)
    import sar_clip
    return sar_clip


def load_metadata(path, crop_mode, dataset="RSAR"):
    classes = DATASET_CLASSES[dataset]
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for line, row in enumerate(reader, start=2):
            context = f"{path}:{line}"
            if row.get("split") != "train":
                raise ValueError(f"{context}: only split=train is permitted")
            row_dataset = row.get("dataset")
            if row_dataset != dataset and not (dataset == "RSAR" and not row_dataset):
                raise ValueError(f"{context}: dataset must be {dataset}")
            try:
                class_id = int(row["class_id"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{context}: invalid class_id") from exc
            if not 0 <= class_id < len(classes) or row.get("class_name") != classes[class_id]:
                raise ValueError(f"{context}: class_name/class_id do not match {dataset}")
            if crop_mode != "all" and row.get("crop_mode") != crop_mode:
                continue
            if not row.get("patch_path") or not Path(row["patch_path"]).is_file():
                raise FileNotFoundError(f"{context}: patch file not found: {row.get('patch_path')}")
            row["class_id"] = class_id
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No metadata rows found for crop_mode={crop_mode}: {path}")
    return rows


class PatchDataset(Dataset):
    def __init__(self, rows, preprocess):
        self.rows = rows
        self.preprocess = preprocess

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        with Image.open(row["patch_path"]) as image:
            return self.preprocess(image.convert("RGB")), row["class_id"]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_balanced_sampler(rows, seed=42):
    counts = Counter(row["class_id"] for row in rows)
    weights = [1.0 / counts[row["class_id"]] for row in rows]
    return WeightedRandomSampler(
        weights, num_samples=len(weights), replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )


def build_loader(rows, preprocess, args):
    return DataLoader(
        PatchDataset(rows, preprocess),
        batch_size=args.batch_size,
        sampler=build_balanced_sampler(rows, args.seed),
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(args.seed),
    )


def cap_metadata(rows, maximum, seed):
    if maximum is None or len(rows) <= maximum:
        return rows
    by_class = {}
    for row in rows:
        by_class.setdefault(row["class_id"], []).append(row)
    per_class = max(1, maximum // len(by_class))
    rng = random.Random(seed)
    return [row for class_id in sorted(by_class)
            for row in (by_class[class_id] if len(by_class[class_id]) <= per_class
                        else rng.sample(by_class[class_id], per_class))]


def build_model(args, device):
    pretrained = Path(args.sarclip_pretrained)
    if not pretrained.exists():
        raise FileNotFoundError(f"SARCLIP pretrained file not found: {pretrained}")
    cache_dir = args.sarclip_cache_dir or str(pretrained.parent)
    sar_clip = import_sarclip(args.sarclip_dir)
    model = sar_clip.create_model_with_args(
        args.sarclip_model,
        pretrained=str(pretrained),
        precision=args.precision,
        device=str(device),
        cache_dir=cache_dir,
        output_dict=True,
    )

    tokenizer = sar_clip.get_tokenizer(args.sarclip_model, cache_dir=cache_dir)
    templates = split_templates(args.templates)
    model.requires_grad_(False)
    with torch.no_grad():
        classifier = sar_clip.build_zero_shot_classifier(
            model,
            tokenizer=tokenizer,
            classnames=list(DATASET_CLASSES[args.dataset]),
            templates=templates,
            num_classes_per_batch=None,
            device=device,
            use_tqdm=False,
        )
        classifier = (classifier / classifier.norm(dim=0, keepdim=True)).detach()

    preprocess_cfg = sar_clip.get_model_preprocess_cfg(model)
    preprocess = sar_clip.image_transform(
        preprocess_cfg.get("size", 224),
        is_train=True,
        mean=preprocess_cfg.get("mean"),
        std=preprocess_cfg.get("std"),
        interpolation=preprocess_cfg.get("interpolation"),
        resize_mode=preprocess_cfg.get("resize_mode"),
        fill_color=preprocess_cfg.get("fill_color", 0),
    )
    return model, classifier, preprocess


def configure_trainable_params(model, args):
    if args.train_visual_proj_only:
        trainable_names = mark_visual_proj_trainable(model, train_logit_scale=True)
        if not any(name.startswith("visual.") for name in trainable_names):
            raise RuntimeError("No visual projection trainable parameters found")
        return "visual_proj", trainable_names

    inject_lora(
        model,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_prefixes=("visual",),
    )
    trainable_names = mark_lora_trainable(model, train_logit_scale=True)
    actual_names = [name for name, param in model.named_parameters() if param.requires_grad]
    if not all(any(f".{factor}." in name for name in actual_names)
               for factor in ("lora_down", "lora_up")):
        raise RuntimeError("LoRA injection produced no trainable LoRA factor pair")
    return "lora", trainable_names


def batch_entropy(probs):
    return float(-(probs * probs.clamp_min(1e-12).log()).sum(dim=1).mean().item())


def train_one_epoch(model, classifier, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_prob_gt = 0.0
    total_entropy = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(image=images)
        image_features = out["image_features"] if isinstance(out, dict) else out[0]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = model.logit_scale.exp() if hasattr(model, "logit_scale") else torch.tensor(1 / 0.07, device=device)
        logits = logit_scale * (image_features @ classifier)
        loss = F.cross_entropy(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite labeled-TRAIN LoRA loss")
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            probs = logits.softmax(dim=-1)
            preds = probs.argmax(dim=1)
            batch_size = labels.numel()
            total_loss += float(loss.item()) * batch_size
            total_correct += int((preds == labels).sum().item())
            total_samples += batch_size
            total_prob_gt += float(probs[torch.arange(batch_size, device=device), labels].sum().item())
            total_entropy += batch_entropy(probs) * batch_size

    return {
        "loss": total_loss / total_samples,
        "top1_acc": total_correct / total_samples,
        "mean_prob_gt": total_prob_gt / total_samples,
        "mean_entropy": total_entropy / total_samples,
    }


def prepare_output(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    existing = [path for path in (output / "config.json", output / "train_log.csv")
                if path.exists()]
    existing.extend(output.glob("*.pth"))
    existing.extend(output.glob("*.pth.partial"))
    if existing:
        raise FileExistsError(f"Refusing to overwrite training artifacts: {existing}")
    return output


def build_config(args, rows, model, adapter_type, trainable_names):
    classes = list(DATASET_CLASSES[args.dataset])
    counts = Counter(row["class_name"] for row in rows)
    return {
        "format": ADAPTER_FORMAT,
        "adapter_type": adapter_type,
        "dataset": args.dataset,
        "seed": args.seed,
        "classes": classes,
        "train_only": True,
        "split": "train",
        "status": "running",
        "epochs": args.epochs,
        "epochs_requested": args.epochs,
        "epochs_completed": 0,
        "selection": "final_epoch",
        "selected_epoch": None,
        "batch_size": args.batch_size,
        "optimizer": "AdamW",
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "precision": args.precision,
        "num_workers": args.num_workers,
        "sampler": {
            "type": "inverse_class_weighted",
            "replacement": True,
            "num_samples": len(rows),
            "seed": args.seed,
        },
        "drop_last": False,
        "force_rgb": True,
        "metadata": str(Path(args.metadata).resolve()),
        # These statistics describe accepted rows after the requested crop filter.
        "metadata_row_count": len(rows),
        "max_patches": args.max_patches,
        "class_counts": {name: counts[name] for name in classes},
        "crop_mode": args.crop_mode,
        "crop_modes": dict(Counter(row["crop_mode"] for row in rows)),
        "crop_expansions": dict(Counter(row.get("crop_expand", "") for row in rows)),
        "sarclip_dir": args.sarclip_dir,
        "sarclip_pretrained": str(Path(args.sarclip_pretrained).resolve()),
        "sarclip_cache_dir": args.sarclip_cache_dir,
        "sarclip_model": args.sarclip_model,
        "templates": split_templates(args.templates),
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_prefixes": ["visual"],
        },
        "trainable_names": trainable_names,
        "trainable_numel": sum(param.numel() for param in model.parameters()
                               if param.requires_grad),
        "training_git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
        ).strip(),
    }


def save_final_adapter(output, model, config, epochs_completed):
    if epochs_completed != config["epochs_requested"] or epochs_completed < 1:
        raise RuntimeError("Only the fully completed requested final epoch can be saved")
    final_config = {
        **config, "status": "complete", "epochs_completed": epochs_completed,
        "selected_epoch": epochs_completed,
    }
    destination = output / f"{config['adapter_type']}_{config['dataset'].lower()}.pth"
    partial = destination.with_suffix(".pth.partial")
    if destination.exists() or partial.exists():
        raise FileExistsError(f"Refusing to overwrite adapter: {destination}")
    state = trainable_state_dict(model)
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise FloatingPointError("Cannot publish nonfinite LoRA parameters")
    torch.save({**final_config, "state_dict": state}, partial)
    os.replace(partial, destination)
    with (output / "config.json").open("w", encoding="utf-8") as f:
        json.dump(final_config, f, indent=2)
    return destination


def main(argv=None):
    args = parse_args(argv)
    output = prepare_output(args.output)
    seed_everything(args.seed)
    force_math_sdpa()
    rows = load_metadata(args.metadata, args.crop_mode, args.dataset)
    rows = cap_metadata(rows, args.max_patches, args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classifier, preprocess = build_model(args, device)
    adapter_type, trainable_names = configure_trainable_params(model, args)
    print(f"[train_sarclip_lora] adapter_type={adapter_type}")
    print(f"[train_sarclip_lora] trainable_params={len(trainable_names)}")
    config = build_config(args, rows, model, adapter_type, trainable_names)
    with (output / "config.json").open("x", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    loader = build_loader(rows, preprocess, args)
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    epochs_completed = 0
    with (output / "train_log.csv").open("x", newline="", encoding="utf-8") as f:
        fieldnames = ["dataset", "seed", "epoch", "loss", "top1_acc", "mean_prob_gt", "mean_entropy"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            metrics = train_one_epoch(model, classifier, loader, optimizer, device)
            epochs_completed = epoch
            writer.writerow({"dataset": args.dataset, "seed": args.seed, "epoch": epoch, **metrics})
            f.flush()
            print(
                f"[train_sarclip_lora] epoch={epoch} "
                f"loss={metrics['loss']:.4f} acc={metrics['top1_acc']:.4f} "
                f"prob_gt={metrics['mean_prob_gt']:.4f} entropy={metrics['mean_entropy']:.4f}"
            )

    destination = save_final_adapter(output, model, config, epochs_completed)
    print(f"[train_sarclip_lora] saved {destination}")


if __name__ == "__main__":
    main()
