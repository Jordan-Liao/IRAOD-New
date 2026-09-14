"""Stream one-forward features, then publish only after real native completion."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
from unittest.mock import patch

import numpy as np

from experiments.comparison.aligned_roi import AlignedRoICapture, validate_native_predictions
# Resolve exporter-only helpers before selecting the frozen model checkout.
from experiments.comparison.dior_recovery.extract_roi_pre_fc_cls import runtime_provenance
from experiments.comparison.result_completion import (
    EXPECTED_TEST_IMAGES, FEATURE_POINT, FEATURE_VERSION, SCHEMA,
    iter_export_records, load_run, native_binding_evidence, read_json, validate_run,
)


ROOT = Path(__file__).resolve().parents[2]


def observe_native_loop(native_loop, model, loader, *, pending, dataset_name,
                        nms_module, provenance, checkpoint_meta, loop_args, loop_kwargs):
    from tools.prediction_export import capture_image_records

    if (loader.batch_size != 1 or len(loader.dataset) != EXPECTED_TEST_IMAGES[dataset_name]
            or loop_args[:2] not in ((), (False, None))
            or not loop_kwargs.get("return_image_ids")):
        raise ValueError("Joint export requires the full native TEST, batch1 and same-inference IDs")
    if tuple(model.module.CLASSES) != tuple(loader.dataset.CLASSES):
        raise ValueError("Checkpoint and dataset class order differ")
    pending = Path(pending)
    pending.mkdir(parents=True, exist_ok=False)
    records = []
    original_forward = model.forward
    with AlignedRoICapture(model.module.roi_head.bbox_head, nms_module) as capture:
        def forward(*args, **kwargs):
            capture.reset()
            result = original_forward(*args, **kwargs)
            if len(result) != 1:
                raise ValueError("Joint export requires one returned image per forward")
            image = capture_image_records(kwargs, 1)[0]
            image_id = image["image_id"]
            arrays = capture.aligned(image_id, result[0])
            validate_native_predictions(arrays, result[0])
            if any(arrays[key].dtype != np.float32 for key in ("features", "boxes", "scores")):
                raise ValueError("Joint export preserves FP32; mixed precision is not admitted")
            name = image_id + ".npz"
            with (pending / name).open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            records.append({
                "image_id": image_id, "image_path": image["filename"],
                "feature_file": name, "n_roi": len(capture.features[0]),
                "n_detections": len(arrays["labels"]),
            })
            capture.reset()
            return result

        with patch.object(model, "forward", forward):
            outputs = native_loop(model, loader, *loop_args, **loop_kwargs)
    predictions, image_records = outputs
    ids = [record["image_id"] for record in records]
    if (len(records) != len(loader.dataset) or len(predictions) != len(records)
            or len(set(ids)) != len(ids)
            or ids != [record["image_id"] for record in image_records]):
        raise ValueError("Captured ROI and native loop do not cover the same full TEST")
    payload = {
        "status": "captured_not_published", "records": records,
        "classes": list(model.module.CLASSES), "test_cfg": dict(model.module.roi_head.test_cfg),
        "checkpoint_meta": checkpoint_meta, "execution_runtime": provenance,
        "native_loop_file": str(Path(sys.modules[native_loop.__module__].__file__).resolve()),
    }
    (pending / "capture.json").write_text(json.dumps(payload, indent=2) + "\n")
    return outputs


def capture_native():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-root", required=True)
    parser.add_argument("--dataset", choices=tuple(EXPECTED_TEST_IMAGES), required=True)
    parser.add_argument("--physical-gpu", required=True, type=int)
    parser.add_argument("native_arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.native_arguments
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    entry = Path(arguments[0]).resolve()
    root = Path(args.case_root).resolve()
    if (entry.name != "test.py" or os.environ.get("IRAOD_GPU_LOCKED") != "1"
            or os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu)):
        raise ValueError("Use the assigned single-GPU native TEST entry and owner lock")
    os.chdir(entry.parent)
    sys.path.insert(0, str(entry.parent))
    sys.argv = [str(entry), *arguments[1:]]
    from iraod_runtime import ensure_iraod_runtime
    ensure_iraod_runtime()
    import torch
    import mmcv
    import mmdet
    from mmrotate.models.roi_heads.bbox_heads import rotated_bbox_head

    module_spec = importlib.util.spec_from_file_location("joint_actual_native_test", entry)
    native = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(native)
    original_loop, original_load = native.single_gpu_test, native.load_checkpoint
    checkpoint_meta = {}

    def load_checkpoint(*positional, **keywords):
        checkpoint = original_load(*positional, **keywords)
        checkpoint_meta.update({key: checkpoint.get("meta", {}).get(key) for key in ("epoch", "iter")})
        return checkpoint

    def fp16_not_admitted(*_args, **_kwargs):
        raise ValueError("Joint export requires unchanged FP32, not an fp16 configuration")

    def single_gpu_test(model, loader, *positional, **keywords):
        return observe_native_loop(
            original_loop, model, loader, pending=root / "roi.unverified",
            dataset_name=args.dataset, nms_module=rotated_bbox_head,
            provenance=runtime_provenance(np, torch, mmcv, mmdet, args.physical_gpu),
            checkpoint_meta=checkpoint_meta, loop_args=positional, loop_kwargs=keywords)

    with patch.object(native, "single_gpu_test", single_gpu_test), \
            patch.object(native, "load_checkpoint", load_checkpoint), \
            patch.object(native, "wrap_fp16_model", fp16_not_admitted):
        native.main()


def completed_predictions(run):
    evidence = native_binding_evidence(run)
    if evidence["status"] != "complete":
        raise ValueError(f"Joint ROI pending genuine native completion: {evidence['missing']}")
    root = Path(run["native_prediction"]["eval_dir"])
    order = read_json(root / "predictions.pkl.image_ids.json")
    with (root / "predictions.pkl").open("rb") as stream:
        predictions = pickle.load(stream)
    if len(predictions) != len(order["image_ids"]):
        raise ValueError("Native predictions differ from their same-inference ID sidecar")
    return evidence, dict(zip(order["image_ids"], predictions)), order["image_ids"]


def exact_baseline_predictions(candidate, baseline):
    if len(candidate) != len(baseline):
        raise ValueError("Joint/native baseline class count differs")
    for actual, reference in zip(candidate, baseline):
        if (actual.dtype != reference.dtype or actual.shape != reference.shape
                or not np.array_equal(actual, reference)):
            raise ValueError("Joint predictions differ exactly from ordinary native baseline")


def publish(root, baseline_root=None):
    from experiments.comparison.report_inputs import class_table
    from experiments.comparison.roi_rollout_bind import bind_plan

    root = Path(root).resolve()
    code_sha = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    passed, run, plan = bind_plan(root, export_code_sha=code_sha)
    validate_run(run)
    if passed["cell"]["joint_exporter"]["code_commit"] != code_sha:
        raise ValueError("Capture and publication exporter revisions differ")
    evidence, predictions, native_ids = completed_predictions(run)
    ap = class_table(Path(run["native_prediction"]["eval_dir"]) / "class_ap.txt", run["dataset"])
    baseline_predictions = None
    if baseline_root is not None:
        baseline_root = Path(baseline_root)
        baseline = load_run(baseline_root / "roi_plan.json", run["run_id"])
        fields = ("dataset", "domain", "seed", "method", "role", "checkpoint", "config",
                  "ann_file", "img_prefix", "image_ids")
        if any(baseline[key] != run[key] for key in fields):
            raise ValueError("Baseline ROI reference selects different scientific inputs")
        _, baseline_predictions, baseline_ids = completed_predictions(baseline)
        if native_ids != baseline_ids:
            raise ValueError("Candidate and ordinary native inference image order differ")
        if ap != class_table(baseline_root / "native/class_ap.txt", run["dataset"]):
            raise ValueError("Candidate and ordinary native class AP tables differ")
    pending = root / "roi.unverified"
    captured = read_json(pending / "capture.json")
    if (captured["status"] != "captured_not_published"
            or [record["image_id"] for record in captured["records"]] != native_ids):
        raise ValueError("ROI capture differs from authentic same-forward native image order")
    by_id = {record["image_id"]: record for record in captured["records"]}
    index = {
        "schema": SCHEMA, "status": "complete", "scope": "full_test", "run": run,
        "n_images": len(native_ids),
        "n_post_nms_detections": sum(record["n_detections"] for record in captured["records"]),
        "code_commit": code_sha, "classes": captured["classes"], "rescale": True,
        "test_cfg": captured["test_cfg"], "checkpoint_meta": captured["checkpoint_meta"],
        "execution_runtime": captured["execution_runtime"],
        "physical_gpu": passed["physical_gpu"], "native_prediction": evidence,
        "feature_point": FEATURE_POINT, "feature_version": FEATURE_VERSION,
        "records": [by_id[image_id] for image_id in run["image_ids"]],
        "execution_mode": "single-forward-native-roi",
        "ordinary_native_baseline": str(baseline_root) if baseline_root is not None else None,
    }
    for record, arrays in iter_export_records({**run, "out_dir": str(pending)}, index):
        per_class = predictions[record["image_id"]]
        if len(per_class) != len(index["classes"]):
            raise ValueError("Native and ROI class counts differ")
        validate_native_predictions(arrays, per_class)
        if baseline_predictions is not None:
            exact_baseline_predictions(per_class, baseline_predictions[record["image_id"]])
            validate_native_predictions(arrays, baseline_predictions[record["image_id"]])
    destination = Path(run["out_dir"])
    if destination.exists():
        raise FileExistsError(destination)
    with (pending / "index.json").open("x") as stream:
        json.dump(index, stream, indent=2)
    pending.rename(destination)
    print(json.dumps({
        "status": "JOINT_NATIVE_ROI_COMPLETE", "plan": str(plan), "run_id": run["run_id"],
        "n_images": len(native_ids), "ordinary_native_exact_parity": baseline_predictions is not None,
    }))
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-root", required=True)
    parser.add_argument("--baseline-root")
    args = parser.parse_args()
    publish(args.case_root, args.baseline_root)
