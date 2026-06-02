#!/usr/bin/env python
import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from iraod_runtime import ensure_iraod_runtime

os.environ.setdefault("IRAOD_CONDA_PREFIX", "/home/liaojr/anaconda3/envs/cliptorch")
ensure_iraod_runtime()

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from sarclip_adapter import (
    ADAPTER_FORMAT,
    inject_lora,
    mark_lora_trainable,
    mark_visual_proj_trainable,
    trainable_state_dict,
)


CLASSES = ["ship", "aircraft", "car", "tank", "bridge", "harbor"]
DEFAULT_TEMPLATE = "A SAR image of a {}"


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune SARCLIP adapters on RSAR patches.")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--crop-mode", choices=["aabb", "rotated", "all"], default="aabb")
    parser.add_argument("--sarclip-dir", default="/home/storageSDA1/liaojr/SARCLIP")
    parser.add_argument("--sarclip-pretrained", required=True)
    parser.add_argument("--sarclip-cache-dir", default=None)
    parser.add_argument("--sarclip-model", default="ViT-B-32")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--precision", default="fp32")
    parser.add_argument("--templates", default=DEFAULT_TEMPLATE)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--train-visual-proj-only", action="store_true")
    return parser.parse_args()


def split_templates(value):
    if ";" in value:
        return [item.strip() for item in value.split(";") if item.strip()]
    return [value.strip()]


def import_sarclip(sarclip_dir):
    if sarclip_dir:
        sys.path.insert(0, sarclip_dir)
    import sar_clip
    return sar_clip


def load_metadata(path, crop_mode):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if crop_mode != "all" and row.get("crop_mode") != crop_mode:
                continue
            row["class_id"] = int(row["class_id"])
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
        image = Image.open(row["patch_path"]).convert("RGB")
        return self.preprocess(image), row["class_id"]


def build_balanced_sampler(rows):
    counts = Counter(row["class_id"] for row in rows)
    weights = [1.0 / counts[row["class_id"]] for row in rows]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


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
    classifier = sar_clip.build_zero_shot_classifier(
        model,
        tokenizer=tokenizer,
        classnames=CLASSES,
        templates=templates,
        num_classes_per_batch=None,
        device=device,
        use_tqdm=False,
    )
    classifier = classifier / classifier.norm(dim=0, keepdim=True)

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
    fallback_reason = None
    if not args.train_visual_proj_only:
        try:
            inject_lora(
                model,
                r=args.lora_r,
                alpha=args.lora_alpha,
                dropout=args.lora_dropout,
                target_prefixes=("visual",),
            )
            trainable_names = mark_lora_trainable(model, train_logit_scale=True)
            if trainable_names:
                return "lora", trainable_names, fallback_reason
            fallback_reason = "LoRA injection produced no trainable parameters"
        except Exception as exc:
            fallback_reason = repr(exc)

    trainable_names = mark_visual_proj_trainable(model, train_logit_scale=True)
    if not trainable_names:
        raise RuntimeError(f"No fallback trainable parameters found. LoRA failure: {fallback_reason}")
    return "visual_proj", trainable_names, fallback_reason


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


def main():
    args = parse_args()
    rows = load_metadata(args.metadata, args.crop_mode)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classifier, preprocess = build_model(args, device)
    adapter_type, trainable_names, fallback_reason = configure_trainable_params(model, args)
    print(f"[train_sarclip_lora] adapter_type={adapter_type}")
    print(f"[train_sarclip_lora] trainable_params={len(trainable_names)}")
    if fallback_reason:
        print(f"[train_sarclip_lora] LoRA fallback reason: {fallback_reason}")

    dataset = PatchDataset(rows, preprocess)
    sampler = build_balanced_sampler(rows)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    log_rows = []
    for epoch in range(1, args.epochs + 1):
        metrics = train_one_epoch(model, classifier, loader, optimizer, device)
        row = {"epoch": epoch, **metrics}
        log_rows.append(row)
        print(
            f"[train_sarclip_lora] epoch={epoch} "
            f"loss={metrics['loss']:.4f} acc={metrics['top1_acc']:.4f} "
            f"prob_gt={metrics['mean_prob_gt']:.4f} entropy={metrics['mean_entropy']:.4f}"
        )

    with (output / "train_log.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["epoch", "loss", "top1_acc", "mean_prob_gt", "mean_entropy"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log_rows)

    config = {
        "format": ADAPTER_FORMAT,
        "adapter_type": adapter_type,
        "classes": CLASSES,
        "metadata": args.metadata,
        "crop_mode": args.crop_mode,
        "sarclip_dir": args.sarclip_dir,
        "sarclip_pretrained": args.sarclip_pretrained,
        "sarclip_cache_dir": args.sarclip_cache_dir,
        "sarclip_model": args.sarclip_model,
        "templates": split_templates(args.templates),
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_prefixes": ["visual"],
        },
        "fallback_reason": fallback_reason,
        "trainable_names": trainable_names,
    }
    with (output / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    checkpoint = {
        **config,
        "state_dict": trainable_state_dict(model),
    }
    torch.save(checkpoint, output / "lora_rsar.pth")
    print(f"[train_sarclip_lora] saved {output / 'lora_rsar.pth'}")


if __name__ == "__main__":
    main()
