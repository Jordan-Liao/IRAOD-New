"""Executable single-coordinate deletions of the frozen strict F baseline.

The caller owns the 36 cells per method, metadata writes and frozen train.py
launch. This module only reads source configs/weights and uses owned temporary
files to verify resolved config execution; it never imports custom model modules.
"""

import ast
from contextlib import contextmanager
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import tempfile

from mmcv import Config

from experiments.comparison.b_regression import (
    COMMON_LAUNCH_PATHS,
    TRAINING_CODE_SHA,
    config_diff,
)


ALPHA_PATH = "model.cfg.vlst_text_visual_alpha"
BLEND_KEY = "CGA_BLEND_DET_WEIGHT"
MODEL_PREFIXES = ("CGA_", "SARCLIP_", "VLST_")
F_COMMON_LAUNCH_PATHS = COMMON_LAUNCH_PATHS | {"find_unused_parameters"}
CLIP_CACHE = "/mnt/shared/zechuan/iraod_weights/clip"


def _model_environment():
    return {
        key: value for key, value in os.environ.items()
        if key.startswith(MODEL_PREFIXES)
    }


@contextmanager
def _f_environment(base_weights):
    """Reproduce the original DDP wrapper, restoring caller state on failure too."""
    previous = dict(os.environ)
    try:
        for key in _model_environment():
            del os.environ[key]
        os.environ.update({
            "SARCLIP_PRETRAINED": str(base_weights),
            "HOME": "/tmp/iraod_clip_home",
            "CLIP_DOWNLOAD_ROOT": CLIP_CACHE,
            "CGA_CLIP_CACHE": CLIP_CACHE,
        })
        yield
    finally:
        os.environ.clear()
        os.environ.update(previous)


def _validate_original(config, environment, base_weights):
    if config.model.type != "UnbiasedTeacherVLST":
        raise ValueError("F requires the original UnbiasedTeacherVLST baseline")
    cfg = config.model.cfg
    for key, expected in {
        "vlst_enabled": True,
        "vlst_strict": True,
        "strict_source_free": True,
        "use_bbox_reg": False,
    }.items():
        if cfg.get(key) is not expected:
            raise ValueError(f"Original F model.cfg.{key} must be {expected}")
    if cfg.get("vlst_text_visual_alpha") != 0.5:
        raise ValueError("Original F vlst_text_visual_alpha must be 0.5")
    if cfg.get("vlst_lora_path") or "SARCLIP_LORA" in environment:
        raise ValueError("F deletions forbid SARCLIP_LORA and vlst_lora_path")
    if (cfg.get("vlst_pretrained") != str(base_weights)
            or environment.get("SARCLIP_PRETRAINED") != str(base_weights)):
        raise ValueError("F CGA and VLST must use the explicit base SARCLIP path")
    for key, expected in {
        "CGA_SCORER": "sarclip",
        "CGA_BACKEND": "sarclip",
        "CGA_STRICT": "1",
        "CGA_FILTER_MODE": "veto_soft",
        "CGA_DROP_SCORE": "0.0",
        "CGA_VETO_PRED_THR": "0.7",
        "CGA_VETO_LABEL_THR": "0.1",
        "CGA_PROTECT_DET_SCORE": "0.9",
        BLEND_KEY: "0.7",
        "VLST_BACKEND": "sarclip",
    }.items():
        if environment.get(key) != expected:
            raise ValueError(f"Original F {key} must be {expected!r}")


def _resolved_text(config, environment):
    text = config.pretty_text
    # The actual F pretty_text has only dict and ColorJitter constructors.
    # Unknown repr constructors must fail here, not at an expensive launch.
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in {
                "dict", "ColorJitter"
            }:
                raise ValueError("Unsupported constructor in resolved F config")
    return (
        "from torchvision.transforms import ColorJitter\n"
        + text
        + "\ndel ColorJitter\n"
        # An inherited config would execute the original unconditional .7
        # assignment AFTER the child's 1.0 assignment. No inheritance remains.
        + "\nimport os as _f_os\n"
        + "for _f_key in list(_f_os.environ):\n"
        + f"    if _f_key.startswith({MODEL_PREFIXES!r}):\n"
        + "        del _f_os.environ[_f_key]\n"
        + "del _f_key\n"
        + "".join(
            f"_f_os.environ[{key!r}] = {value!r}\n"
            for key, value in sorted(environment.items())
        )
        + "del _f_os\n"
    )


def build_f_deletion_spec(paths, dataset, method, base_weights, overrides):
    """Return a standalone config and exact semantic config/environment diffs.

    ``paths`` exposes the original core rsar_cfg('F') / dior_cfg('F').
    ``base_weights`` must be the caller's verified existing base SARCLIP file.
    Common MMCV CLI overrides are applied identically to baseline and candidate
    and embedded in the resolved config. They may not alter algorithm controls.
    Launch with frozen ``training_code/train.py`` at DDP 2 x 16, global 32,
    LR .02, one epoch. The parent preserves the original wrapper HOME/cache.
    Caller model-environment poison (including LoRA) is cleared, never inherited.
    """
    if dataset not in ("RSAR", "DIOR"):
        raise ValueError(f"Unsupported F dataset: {dataset}")
    if method not in ("F_text_only", "F_veto_only"):
        raise ValueError(f"Unsupported F deletion: {method}")
    base_weights = Path(base_weights).expanduser().resolve()
    if not base_weights.is_file():
        raise ValueError(f"Base SARCLIP checkpoint does not exist: {base_weights}")
    rsar_config = Path(paths.rsar_cfg("F")).resolve()
    reference = rsar_config if dataset == "RSAR" else Path(paths.dior_cfg("F")).resolve()
    training_code = rsar_config.parents[3]
    if not (training_code / "train.py").is_file():
        raise ValueError(f"Frozen F training checkout lacks train.py: {training_code}")
    sha = subprocess.check_output(
        ["git", "-C", str(training_code), "rev-parse", "HEAD"], text=True).strip()
    if sha != TRAINING_CODE_SHA:
        raise ValueError(f"F deletions require training code {TRAINING_CODE_SHA}, got {sha}")

    with _f_environment(base_weights):
        original = Config.fromfile(str(reference), import_custom_modules=False)
        original_environment = _model_environment()
    _validate_original(original, original_environment, base_weights)
    baseline = deepcopy(original)
    baseline.merge_from_dict(deepcopy(overrides))
    unexpected = [
        change["path"] for change in config_diff(original, baseline)
        if change["path"] not in F_COMMON_LAUNCH_PATHS
    ]
    if unexpected:
        raise ValueError(f"Overrides change F algorithm fields: {unexpected}")
    if (baseline.data.samples_per_gpu != 16 or baseline.optimizer.lr != 0.02
            or baseline.runner.max_epochs != 1):
        raise ValueError("F requires DDP 2x16, optimizer.lr=.02, one epoch")

    candidate = deepcopy(baseline)
    environment = deepcopy(original_environment)
    expected_config_diff = []
    expected_environment_diff = []
    if method == "F_text_only":
        candidate.model.cfg.vlst_text_visual_alpha = 1.0
        expected_config_diff = [{"path": ALPHA_PATH, "before": 0.5, "after": 1.0}]
    else:
        environment[BLEND_KEY] = "1.0"
        expected_environment_diff = [{"path": BLEND_KEY, "before": "0.7", "after": "1.0"}]
    differences = config_diff(baseline, candidate)
    environment_differences = config_diff(original_environment, environment)
    if (differences != expected_config_diff
            or environment_differences != expected_environment_diff):
        raise ValueError("F deletion must change exactly its authorized coordinate")

    text = _resolved_text(candidate, environment)
    with tempfile.TemporaryDirectory(prefix="f-deletion-") as directory:
        generated = Path(directory) / "resolved.py"
        generated.write_text(text)
        with _f_environment(base_weights):
            # Deliberately poison the reload to prove the executable config,
            # not an outer env override, owns the final effective setting.
            os.environ[BLEND_KEY] = "0.123"
            os.environ["SARCLIP_LORA"] = "must-not-survive"
            reloaded = Config.fromfile(str(generated), import_custom_modules=False)
            effective_environment = _model_environment()
    if config_diff(candidate, reloaded):
        raise ValueError("Resolved F config failed semantic round-trip")
    if config_diff(environment, effective_environment):
        raise ValueError("Resolved F config failed effective environment round-trip")
    return {
        "method": method,
        "reference_config": str(reference),
        "training_code": str(training_code),
        "training_code_sha": sha,
        "config_text": text,
        "config_diff": differences,
        "environment_diff": environment_differences,
        "effective_model_environment": effective_environment,
        "wrapper_environment": {
            "HOME": "/tmp/iraod_clip_home", "CLIP_DOWNLOAD_ROOT": CLIP_CACHE,
        },
        "topology": {"world_size": 2, "samples_per_gpu": 16, "global_batch_size": 32},
    }
