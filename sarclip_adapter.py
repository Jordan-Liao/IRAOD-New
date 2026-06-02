import math
from collections import OrderedDict

import torch
from torch import nn


ADAPTER_FORMAT = "sarclip_rsar_adapter_v1"


class LoRALinear(nn.Module):
    def __init__(self, base, r=8, alpha=16.0, dropout=0.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base)!r}")
        if r <= 0:
            raise ValueError("LoRA rank must be positive")

        self.base = base
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.lora_down = nn.Linear(base.in_features, self.r, bias=False)
        self.lora_up = nn.Linear(self.r, base.out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        self.lora_down.to(device=base.weight.device, dtype=base.weight.dtype)
        self.lora_up.to(device=base.weight.device, dtype=base.weight.dtype)
        for param in self.base.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.base(x) + self.lora_up(self.lora_down(self.dropout(x))) * self.scaling


def _get_parent_module(root, module_name):
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def iter_lora_target_names(model, target_prefixes=("visual",)):
    for name, module in model.named_modules():
        if not name or not isinstance(module, nn.Linear):
            continue
        if any(name == prefix or name.startswith(prefix + ".") for prefix in target_prefixes):
            parent, _ = _get_parent_module(model, name)
            if isinstance(parent, nn.MultiheadAttention):
                continue
            yield name


def inject_lora(model, r=8, alpha=16.0, dropout=0.0, target_prefixes=("visual",)):
    target_names = list(iter_lora_target_names(model, target_prefixes=target_prefixes))
    if not target_names:
        raise RuntimeError(f"No nn.Linear LoRA targets found for prefixes={target_prefixes}")

    for name in target_names:
        parent, child = _get_parent_module(model, name)
        base = getattr(parent, child)
        if isinstance(base, LoRALinear):
            continue
        setattr(parent, child, LoRALinear(base, r=r, alpha=alpha, dropout=dropout))
    return target_names


def freeze_all(model):
    for param in model.parameters():
        param.requires_grad = False


def mark_lora_trainable(model, train_logit_scale=True):
    freeze_all(model)
    trainable_names = []
    for name, param in model.named_parameters():
        is_lora = ".lora_down." in name or ".lora_up." in name
        is_logit_scale = train_logit_scale and name == "logit_scale"
        if is_lora or is_logit_scale:
            param.requires_grad = True
            trainable_names.append(name)
    return trainable_names


def mark_visual_proj_trainable(model, train_logit_scale=True):
    freeze_all(model)
    trainable_names = []
    keywords = ("proj", "projection", "ln_post", "final", "head", "norm")
    for name, param in model.named_parameters():
        is_logit_scale = train_logit_scale and name == "logit_scale"
        is_visual_head = name.startswith("visual.") and any(key in name.lower() for key in keywords)
        if is_logit_scale or is_visual_head:
            param.requires_grad = True
            trainable_names.append(name)
    return trainable_names


def trainable_state_dict(model):
    return OrderedDict(
        (name, param.detach().cpu())
        for name, param in model.named_parameters()
        if param.requires_grad
    )


def load_adapter_checkpoint(model, adapter_path, map_location="cpu"):
    checkpoint = torch.load(adapter_path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Unsupported SARCLIP adapter checkpoint type: {type(checkpoint)!r}")

    if checkpoint.get("format") != ADAPTER_FORMAT:
        state_dict = checkpoint.get("state_dict", checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected keys while loading SARCLIP adapter: {unexpected}")
        return {
            "adapter_type": "raw_state_dict",
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
        }

    adapter_type = checkpoint.get("adapter_type")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise RuntimeError("SARCLIP adapter checkpoint is missing state_dict")

    if adapter_type == "lora":
        lora_cfg = checkpoint.get("lora", {})
        inject_lora(
            model,
            r=int(lora_cfg.get("r", 8)),
            alpha=float(lora_cfg.get("alpha", 16.0)),
            dropout=float(lora_cfg.get("dropout", 0.0)),
            target_prefixes=tuple(lora_cfg.get("target_prefixes", ("visual",))),
        )
    elif adapter_type != "visual_proj":
        raise RuntimeError(f"Unsupported SARCLIP adapter_type: {adapter_type}")

    model_keys = set(model.state_dict().keys())
    unexpected = [key for key in state_dict if key not in model_keys]
    if unexpected:
        raise RuntimeError(f"Unexpected keys while loading SARCLIP adapter: {unexpected}")

    missing, load_unexpected = model.load_state_dict(state_dict, strict=False)
    if load_unexpected:
        raise RuntimeError(f"Unexpected keys while loading SARCLIP adapter: {load_unexpected}")

    return {
        "adapter_type": adapter_type,
        "missing_keys": list(missing),
        "unexpected_keys": list(load_unexpected),
        "trainable_names": list(checkpoint.get("trainable_names", [])),
    }
