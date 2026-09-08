"""Explicit Target-supervised adapter admission; never an ambient strict-baseline fallback."""

from pathlib import Path

import torch

from experiments.comparison.labels import CLASSES, ORACLE_TRAINING_TEMPLATES
from experiments.comparison.result_completion import DOMAINS
from experiments.comparison import host_binding as host
from sarclip_adapter import ADAPTER_FORMAT, freeze_for_inference, load_adapter_checkpoint


def inspect_oracle_adapter(path, dataset, base_weights):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "format": ADAPTER_FORMAT, "adapter_type": "lora", "dataset": dataset,
        "seed": 42, "train_only": True, "split": "train", "status": "complete",
        "epochs": 10, "epochs_requested": 10, "epochs_completed": 10,
        "selection": "final_epoch", "selected_epoch": 10, "batch_size": 64,
        "optimizer": "AdamW", "lr": 1e-4, "weight_decay": 1e-4,
        "precision": "fp32", "crop_mode": "aabb", "force_rgb": True,
        "drop_last": False, "sarclip_model": "ViT-B-32",
        "templates": [ORACLE_TRAINING_TEMPLATES[dataset]],
    }
    if any(payload.get(key) != value for key, value in required.items()):
        raise ValueError("Oracle adapter does not match the frozen dataset/TRAIN/final-epoch LoRA recipe")
    if payload["classes"] != list(CLASSES[dataset]):
        raise ValueError("Oracle adapter class order differs from the detector dataset")
    if (Path(host.map_path(payload["sarclip_pretrained"])).expanduser().resolve()
            != Path(host.map_path(base_weights)).expanduser().resolve()):
        raise ValueError("Oracle adapter does not bind the declared base SARCLIP initialization")
    if payload.get("max_patches") is not None or not payload.get("training_git_sha"):
        raise ValueError("Oracle requires the uncapped recipe and actual training-code provenance")
    count = payload["metadata_row_count"]
    if (count <= 0 or set(payload["class_counts"]) != set(CLASSES[dataset])
            or sum(payload["class_counts"].values()) != count
            or payload["crop_modes"] != {"aabb": count}
            or payload["crop_expansions"] != {"0.4": count}
            or set(payload["corruptions"]) != set(DOMAINS[dataset][1:])
            or sum(payload["corruptions"].values()) != count
            or any(n <= 0 for n in payload["corruptions"].values())):
        raise ValueError("Oracle crop coverage differs from all declared corrupted TRAIN domains")
    sampler = payload["sampler"]
    if sampler != {
            "type": "inverse_class_weighted", "replacement": True,
            "num_samples": count, "seed": 42}:
        raise ValueError("Oracle sampling differs from the frozen labeled-TRAIN recipe")
    if payload["lora"] != {
            "r": 8, "alpha": 16.0, "dropout": 0.0, "target_prefixes": ["visual"]}:
        raise ValueError("Oracle must use the existing visual LoRA recipe")
    state = payload["state_dict"]
    down = {k[:-len(".lora_down.weight")] for k in state if k.endswith(".lora_down.weight")}
    up = {k[:-len(".lora_up.weight")] for k in state if k.endswith(".lora_up.weight")}
    if (not down or down != up or "logit_scale" not in state
            or set(state) != set(payload["trainable_names"])
            or any(k != "logit_scale" and not (
                k.startswith("visual.") and k.endswith((".lora_down.weight", ".lora_up.weight")))
                   for k in state)
            or sum(v.numel() for v in state.values()) != payload["trainable_numel"]
            or any(not torch.isfinite(v).all() for v in state.values())):
        raise ValueError("Oracle adapter contains incomplete or invalid trained LoRA factors")
    return {k: v for k, v in payload.items() if k != "state_dict"}


def attach_oracle_adapter(scorer, path, dataset, base_weights):
    """Attach only through explicit oracle classes, after the normal strict base build."""
    if scorer.backend != "sarclip" or not scorer.strict or scorer.class_names != list(CLASSES[dataset]):
        raise ValueError("Oracle requires the explicit strict SARCLIP base and matching class order")
    evidence = inspect_oracle_adapter(path, dataset, base_weights)
    info = load_adapter_checkpoint(scorer.clip, path, map_location=scorer.device)
    if info["adapter_type"] != "lora" or any(
            ".lora_down." in key or ".lora_up." in key for key in info["missing_keys"]):
        raise ValueError("Oracle adapter did not fill the base model's required LoRA factors")
    freeze_for_inference(scorer.clip)
    scorer.oracle_evidence = evidence
    return scorer
