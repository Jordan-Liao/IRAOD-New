"""CPU preparation of one dataset's approved oracle grid: DIOR24 or RSAR48."""

import argparse
from contextlib import contextmanager
from copy import deepcopy
import os
from pathlib import Path

from mmcv import Config

from experiments.comparison import extension_training as training
from experiments.comparison import host_binding as host
from experiments.comparison.f_deletion import MODEL_PREFIXES, _resolved_text
from experiments.comparison.labels import CLASSES
from experiments.comparison.oracle_adapters import inspect_oracle_adapter


@contextmanager
def model_environment(base):
    """Resolve the original recipes without inheriting an ambient adapter."""
    previous = dict(os.environ)
    try:
        for key in list(os.environ):
            if key.startswith(MODEL_PREFIXES):
                del os.environ[key]
        os.environ.update(
            SARCLIP_PRETRAINED=str(base), SARCLIP_CACHE_DIR=str(base.parent))
        with host.native_config_paths():
            yield
    finally:
        os.environ.clear()
        os.environ.update(previous)


def build_specs(paths, adapter, base_weights, dataset="DIOR"):
    """Read the selected D/F recipes; retain their detector architecture."""
    adapter = training.require_file(adapter)
    base = training.require_file(base_weights)
    evidence = inspect_oracle_adapter(adapter, dataset, base)
    classes = CLASSES[dataset]
    source_config = paths.rsar_cfg if dataset == "RSAR" else paths.dior_cfg
    specs = {}
    for method, baseline, model_type in (
            ("LoRA-CGA", "D", "OracleCGAStudent"),
            ("LoRA-CGA+VLST", "F", "OracleCGAVLSTStudent")):
        reference = training.require_file(source_config(baseline))
        with model_environment(base):
            original = Config.fromfile(str(reference), import_custom_modules=False)
            environment = {
                key: host.map_path(value) for key, value in os.environ.items()
                if key.startswith(MODEL_PREFIXES)}
            teacher_path = Path(host.map_path(original.model.ema_config))
            if not teacher_path.is_absolute():
                teacher_path = training.ROOT / teacher_path
            teacher = Config.fromfile(
                str(training.require_file(teacher_path)), import_custom_modules=False)
        cfg = original.model.cfg
        expected_type = "UnbiasedTeacher" if baseline == "D" else "UnbiasedTeacherVLST"
        if (original.model.type != expected_type
                or cfg.strict_source_free is not True or cfg.weight_l != 0
                or cfg.weight_u != 1 or cfg.use_bbox_reg is not False
                or cfg.get("momentum", .998) != .998
                or original.runner.max_epochs != 1
                or original.data.train.type != "StrictSourceFreeDOTADataset"
                or list(original.data.train.classes) != list(classes)
                or original.model.roi_head.bbox_head.num_classes != len(classes)):
            raise ValueError(f"Oracle requires the original image-only {dataset} D/F one-epoch recipe")
        if (teacher.model.type not in ("OrientedRCNN", "OrientedRCNN_CGA")
                or teacher.model.roi_head.bbox_head.num_classes != len(classes)):
            raise ValueError(
                f"Oracle requires the matching {len(classes)}-class {dataset} OrientedRCNN source teacher")
        if (baseline == "F" and (
                cfg.vlst_enabled is not True or cfg.vlst_strict is not True
                or cfg.vlst_lora_path is not None)):
            raise ValueError("Oracle requires strict adapter-free original F VLST")
        candidate = deepcopy(original)
        # CGA changes inference postprocessing only. Do not replace backbone,
        # heads, normalization, source initialization, or any teacher settings.
        teacher.model.type = "OrientedRCNN_CGA"
        candidate.model.ema_config = teacher._cfg_dict
        candidate.model.type = model_type
        imports = candidate.get("custom_imports", {}).get("imports", [])
        candidate.custom_imports = dict(
            imports=list(dict.fromkeys([*imports, "sfod.extensions.oracle"])),
            allow_failed_imports=False)
        candidate.model.cfg.update(
            oracle_dataset=dataset, oracle_adapter=str(adapter),
            oracle_base_weights=str(base))
        if baseline == "F":
            candidate.model.cfg.vlst_pretrained = str(base)
            candidate.model.cfg.vlst_cache_dir = str(base.parent)
        # Resolved configs contain no inherited environment assignments or stale
        # source config paths. Runtime common overrides bind source A and VAL.
        environment["SARCLIP_PRETRAINED"] = str(base)
        environment["SARCLIP_CACHE_DIR"] = str(base.parent)
        specs[method] = dict(
            config_text=_resolved_text(candidate, environment),
            model_environment=environment,
            oracle_training_git_sha=evidence["training_git_sha"],
            oracle_adapter=str(adapter), oracle_base_weights=str(base))
    return specs


def prepare(base_plan, core_report, core_paths, out_dir, artifact_root,
            eval_code, python, adapter, sarclip_base, dataset="DIOR"):
    """Publish metadata only after the selected dataset's payload passes admission."""
    return training.prepare(
        **{key: host.map_path(value) for key, value in dict(
            base_plan=base_plan, core_report=core_report, core_paths=core_paths,
            out_dir=out_dir, artifact_root=artifact_root, eval_code=eval_code,
            python=python, oracle_adapter=adapter, sarclip_base=sarclip_base).items()},
        methods=training.ORACLE_METHODS, oracle_dataset=dataset)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("DIOR", "RSAR"), default="DIOR")
    for name in ("base-plan", "core-report", "core-paths", "out-dir",
                 "artifact-root", "eval-code", "python", "adapter", "sarclip-base"):
        parser.add_argument("--" + name, required=True)
    print(prepare(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
