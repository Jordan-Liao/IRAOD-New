"""Read-only B_REG specification: frozen B with pseudo-box regression enabled.

B_REG is not SFUT: B keeps its default .998 iteration-start EMA and original
hooks. The caller owns the 12-domain x 3-seed (42/43/44) training/EMA cells,
writes the overlay into new metadata, and launches the returned training_code.
No source training, F grid, artifact writes, or model imports happen here.
"""

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
import subprocess

from mmcv import Config


TRAINING_CODE_SHA = "0f98a48b539f9260055fc305784ffbc924f50451"
REGRESSION_PATH = "model.cfg.use_bbox_reg"

# Launch options shared by B and B_REG, not alternative algorithm settings.
COMMON_LAUNCH_PATHS = frozenset({
    "corrupt",
    "data.img_prefix",
    "data.train.img_prefix",
    "data.train.unlabeled_epoch_size",
    "data.samples_per_gpu",
    "data.workers_per_gpu",
    "load_from",
    "model.ema_ckpt",
    "optimizer.lr",
    "runner.max_epochs",
    "seed",
    "work_dir",
    "checkpoint_config.max_keep_ckpts",
    "checkpoint_config.save_last",
})


def _semantic(value):
    """Compare config containers by value and transform instances by repr."""
    if isinstance(value, Config):
        value = value._cfg_dict
    if isinstance(value, Mapping):
        return {key: _semantic(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_semantic(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_semantic(item) for item in value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    # Config pipelines contain e.g. ColorJitter instances with identity equality
    # but deterministic repr. Never compare deep-copied transforms by identity.
    return repr(value)


def config_diff(before, after):
    """Return leaf differences, using dotted keys and numeric sequence indices."""
    differences = []

    def visit(left, right, path):
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(left.keys() | right.keys()):
                child = f"{path}.{key}" if path else str(key)
                if key not in left or key not in right:
                    differences.append({
                        "path": child, "before": left.get(key), "after": right.get(key)})
                else:
                    visit(left[key], right[key], child)
        elif type(left) is type(right) and isinstance(left, (list, tuple)):
            if len(left) != len(right):
                differences.append({"path": path, "before": left, "after": right})
            else:
                for index, (old, new) in enumerate(zip(left, right)):
                    visit(old, new, f"{path}.{index}")
        elif isinstance(left, bool) != isinstance(right, bool) or left != right:
            differences.append({"path": path, "before": left, "after": right})

    visit(_semantic(before), _semantic(after), "")
    return differences


def _reject_regression_override(overrides):
    for key, value in overrides.items():
        if "use_bbox_reg" in key.split("."):
            raise ValueError("Common overrides must not contain use_bbox_reg")
        if isinstance(value, Mapping):
            _reject_regression_override(value)


def build_b_regression_spec(paths, dataset, overrides):
    """Audit a common-options B against B_REG without writing either config.

    ``paths`` is the loaded core resolver exposing rsar_cfg/dior_cfg. Overrides
    use MMCV's merge_from_dict format and must be applied unchanged to both
    launch commands; they are deliberately NOT embedded in overlay_text.
    Existing algorithm options may be restated, but not changed. The batch
    setting is 32 per GPU; the caller owns the single-GPU/global-32 launch.
    Operational seed/work_dir values can differ from historical B; the audit
    compares B and B_REG with the SAME supplied values, not historical runs.
    """
    if dataset not in ("RSAR", "DIOR"):
        raise ValueError(f"Unsupported B dataset: {dataset}")
    _reject_regression_override(overrides)
    rsar_config = Path(paths.rsar_cfg("B")).resolve()
    reference = rsar_config if dataset == "RSAR" else Path(paths.dior_cfg("B")).resolve()
    training_code = rsar_config.parents[3]
    if not (training_code / "train.py").is_file():
        raise ValueError(f"Frozen B training checkout lacks train.py: {training_code}")
    sha = subprocess.check_output(
        ["git", "-C", str(training_code), "rev-parse", "HEAD"], text=True).strip()
    if sha != TRAINING_CODE_SHA:
        raise ValueError(f"B_REG requires training code {TRAINING_CODE_SHA}, got {sha}")

    original = Config.fromfile(str(reference), import_custom_modules=False)
    if original.model.type != "UnbiasedTeacher":
        raise ValueError("B_REG requires the original UnbiasedTeacher B baseline")
    if original.model.cfg.use_bbox_reg is not False:
        raise ValueError("Original B model.cfg.use_bbox_reg must be False")
    baseline = deepcopy(original)
    baseline.merge_from_dict(deepcopy(overrides))
    unexpected = [
        change["path"] for change in config_diff(original, baseline)
        if change["path"] not in COMMON_LAUNCH_PATHS
    ]
    if unexpected:
        raise ValueError(f"Overrides change B algorithm fields: {unexpected}")
    if (baseline.data.samples_per_gpu != 32 or baseline.optimizer.lr != 0.02
            or baseline.runner.max_epochs != 1):
        raise ValueError("B_REG requires samples_per_gpu=32, optimizer.lr=.02, one epoch")

    candidate = deepcopy(baseline)
    candidate.model.cfg.use_bbox_reg = True
    differences = config_diff(baseline, candidate)
    expected = [{"path": REGRESSION_PATH, "before": False, "after": True}]
    if differences != expected:
        raise ValueError(f"B_REG must differ only in {REGRESSION_PATH}: {differences}")
    return {
        "method": "B_REG",
        "reference_config": str(reference),
        "training_code": str(training_code),
        "training_code_sha": sha,
        "overlay_text": (
            f"_base_ = {str(reference)!r}\n"
            "model = dict(cfg=dict(use_bbox_reg=True))\n"
        ),
        "config_diff": differences,
        "operational_note": (
            "Common launch overrides apply equally to B and B_REG. Seed/work_dir "
            "may differ from historical B; no algorithm fields change except "
            "model.cfg.use_bbox_reg. Launch frozen training_code at global batch 32."
        ),
    }
