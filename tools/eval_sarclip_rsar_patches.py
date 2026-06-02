#!/usr/bin/env python
import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from iraod_runtime import ensure_iraod_runtime

os.environ.setdefault("IRAOD_CONDA_PREFIX", "/home/liaojr/anaconda3/envs/cliptorch")
ensure_iraod_runtime()

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from sarclip_adapter import load_adapter_checkpoint


CLASSES = ["ship", "aircraft", "car", "tank", "bridge", "harbor"]
DEFAULT_TEMPLATE = "A SAR image of a {}"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SARCLIP on RSAR object patches.")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--crop-mode", choices=["aabb", "rotated", "all"], default="aabb")
    parser.add_argument("--sarclip-dir", default="/home/storageSDA1/liaojr/SARCLIP")
    parser.add_argument("--sarclip-pretrained", required=True)
    parser.add_argument("--sarclip-cache-dir", default=None)
    parser.add_argument("--sarclip-model", default="ViT-B-32")
    parser.add_argument("--precision", default="fp32")
    parser.add_argument("--templates", default=DEFAULT_TEMPLATE)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lora", default=None)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


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
        return self.preprocess(image), row["class_id"], idx


def split_templates(value):
    if ";" in value:
        return [item.strip() for item in value.split(";") if item.strip()]
    return [value.strip()]


def import_sarclip(sarclip_dir):
    if sarclip_dir:
        sys.path.insert(0, sarclip_dir)
    import sar_clip
    return sar_clip


def build_model_and_classifier(args, device):
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
    if args.lora:
        info = load_adapter_checkpoint(model, args.lora, map_location=device)
        print(f"[eval_sarclip] loaded adapter: {args.lora} ({info['adapter_type']})")
    model.eval()

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
        is_train=False,
        mean=preprocess_cfg.get("mean"),
        std=preprocess_cfg.get("std"),
        interpolation=preprocess_cfg.get("interpolation"),
        resize_mode=preprocess_cfg.get("resize_mode"),
        fill_color=preprocess_cfg.get("fill_color", 0),
    )
    return model, classifier, preprocess


def entropy(prob):
    return float(-(prob * (prob.clamp_min(1e-12).log())).sum().item())


def summarize(predictions, rows):
    n = len(predictions)
    correct = sum(1 for pred in predictions if pred["correct"])
    confusion = [[0 for _ in CLASSES] for _ in CLASSES]
    top1_distribution = Counter()
    per_class = {name: [] for name in CLASSES}

    for pred in predictions:
        gt = pred["class_id"]
        top1 = pred["pred_id"]
        confusion[gt][top1] += 1
        top1_distribution[CLASSES[top1]] += 1
        per_class[CLASSES[gt]].append(pred)

    metrics = {
        "overall_acc": correct / n if n else 0.0,
        "per_class_acc": {},
        "per_class_mean_prob_gt": {},
        "per_class_mean_max_prob": {},
        "per_class_mean_entropy": {},
        "top1_distribution": dict(top1_distribution),
        "confusion_matrix": confusion,
    }
    for class_name, items in per_class.items():
        if not items:
            metrics["per_class_acc"][class_name] = None
            metrics["per_class_mean_prob_gt"][class_name] = None
            metrics["per_class_mean_max_prob"][class_name] = None
            metrics["per_class_mean_entropy"][class_name] = None
            continue
        metrics["per_class_acc"][class_name] = sum(item["correct"] for item in items) / len(items)
        metrics["per_class_mean_prob_gt"][class_name] = sum(item["prob_gt"] for item in items) / len(items)
        metrics["per_class_mean_max_prob"][class_name] = sum(item["max_prob"] for item in items) / len(items)
        metrics["per_class_mean_entropy"][class_name] = sum(item["entropy"] for item in items) / len(items)

    if any(row.get("corruption") for row in rows):
        grouped = defaultdict(list)
        for pred in predictions:
            grouped[pred["corruption"]].append(pred)
        metrics["per_corruption_metrics"] = {}
        metrics["per_corruption_per_class_metrics"] = {}
        for corruption, items in grouped.items():
            metrics["per_corruption_metrics"][corruption] = {
                "overall_acc": sum(item["correct"] for item in items) / len(items),
                "num_samples": len(items),
            }
            by_class = defaultdict(list)
            for item in items:
                by_class[CLASSES[item["class_id"]]].append(item)
            metrics["per_corruption_per_class_metrics"][corruption] = {
                class_name: {
                    "acc": sum(item["correct"] for item in class_items) / len(class_items),
                    "num_samples": len(class_items),
                    "mean_prob_gt": sum(item["prob_gt"] for item in class_items) / len(class_items),
                }
                for class_name, class_items in by_class.items()
            }
    return metrics


def write_confusion(path, matrix):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["gt\\pred"] + CLASSES)
        for class_name, row in zip(CLASSES, matrix):
            writer.writerow([class_name] + row)


def main():
    args = parse_args()
    rows = load_metadata(args.metadata, args.crop_mode)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classifier, preprocess = build_model_and_classifier(args, device)

    dataset = PatchDataset(rows, preprocess)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    logit_scale = getattr(model, "logit_scale", torch.tensor(math.log(1 / 0.07), device=device)).exp()

    predictions = []
    with torch.no_grad():
        for images, labels, indices in loader:
            images = images.to(device)
            labels = labels.to(device)
            out = model(image=images)
            image_features = out["image_features"] if isinstance(out, dict) else out[0]
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            probs = (logit_scale * (image_features @ classifier)).softmax(dim=-1)
            max_probs, pred_ids = probs.max(dim=1)
            for batch_i, row_idx in enumerate(indices.tolist()):
                row = rows[row_idx]
                label = int(labels[batch_i].item())
                prob = probs[batch_i].detach().cpu()
                pred_id = int(pred_ids[batch_i].item())
                predictions.append({
                    "patch_path": row["patch_path"],
                    "corruption": row.get("corruption", ""),
                    "crop_mode": row.get("crop_mode", ""),
                    "class_name": CLASSES[label],
                    "class_id": label,
                    "pred_name": CLASSES[pred_id],
                    "pred_id": pred_id,
                    "correct": int(pred_id == label),
                    "prob_gt": float(prob[label].item()),
                    "max_prob": float(max_probs[batch_i].item()),
                    "entropy": entropy(prob),
                    **{f"prob_{name}": float(prob[i].item()) for i, name in enumerate(CLASSES)},
                })

    metrics = summarize(predictions, rows)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with (out_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(predictions[0].keys()))
        writer.writeheader()
        writer.writerows(predictions)
    write_confusion(out_dir / "confusion_matrix.csv", metrics["confusion_matrix"])
    print(f"[eval_sarclip] samples={len(predictions)} overall_acc={metrics['overall_acc']:.4f}")
    print(f"[eval_sarclip] wrote {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
