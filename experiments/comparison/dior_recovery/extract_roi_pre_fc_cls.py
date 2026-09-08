"""Versioned, exactly post-NMS-aligned fc_cls input export (compute owner only)."""

import argparse
import json
import os
from pathlib import Path
import subprocess

from experiments.comparison.result_completion import (
    load_run, native_binding_evidence, read_json, FEATURE_POINT, FEATURE_VERSION)


def require_owned_gpu(run):
    if os.environ.get("IRAOD_GPU_LOCKED") != "1":
        raise RuntimeError("Port ROI extraction requires the existing owner's actual GPU lock")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in tuple(map(str, run["allowed_gpus"])):
        raise ValueError("Port ROI extraction requires one bound physical GPU within4-7")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    run = load_run(args.plan, args.run_id)
    code_commit = subprocess.check_output(
        ["git", "-C", str(Path(__file__).resolve().parents[3]), "rev-parse", "HEAD"],
        text=True).strip()
    native = predictions = prediction_indices = None
    if "native_prediction" in run:
        require_owned_gpu(run)
        native = native_binding_evidence(run)
        if native["status"] != "complete":
            raise ValueError(f"ROI pending native inputs: {native['missing']}")
        if code_commit != run["export_code_sha"]:
            raise ValueError("Export checkout differs from the prepared binding")
        import pickle
        root = Path(run["native_prediction"]["eval_dir"])
        order = read_json(root / "predictions.pkl.image_ids.json")
        with (root / "predictions.pkl").open("rb") as stream:
            predictions = pickle.load(stream)
        if len(predictions) != len(order["image_ids"]):
            raise ValueError("Native prediction list differs from its same-inference ID binding")
        prediction_indices = {image_id: i for i, image_id in enumerate(order["image_ids"])}

    from iraod_runtime import ensure_iraod_runtime
    ensure_iraod_runtime()
    import numpy as np
    import torch
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import load_checkpoint, wrap_fp16_model
    from mmdet.datasets import build_dataloader
    from mmrotate.datasets import build_dataset
    from mmrotate.models import build_detector
    from mmrotate.models.roi_heads.bbox_heads import rotated_bbox_head
    from mmrotate.utils import compat_cfg, setup_multi_processes
    from sfod.utils import patch_config
    from experiments.comparison.aligned_roi import (
        AlignedRoICapture, SCHEMA, validate_native_predictions)

    out = Path(run["out_dir"])
    out.mkdir(parents=True, exist_ok=False)
    cfg = patch_config(compat_cfg(Config.fromfile(run["config"])))
    setup_multi_processes(cfg)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.data.test.test_mode = True
    cfg.data.test.ann_file = run["ann_file"]
    cfg.data.test.img_prefix = run["img_prefix"]
    dataset = build_dataset(cfg.data.test)
    by_id = {Path(info["filename"]).stem: i
             for i, info in enumerate(dataset.data_infos)}
    if len(by_id) != len(dataset) or set(by_id) != set(run["image_ids"]):
        raise ValueError("Evaluation dataset does not match every full TEST image ID")
    indices = [by_id[image_id] for image_id in run["image_ids"]]
    loader = build_dataloader(
        torch.utils.data.Subset(dataset, indices), samples_per_gpu=1,
        workers_per_gpu=2, dist=False, shuffle=False)
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    if cfg.get("fp16") is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, run["checkpoint"], map_location="cpu")
    model.CLASSES = checkpoint.get("meta", {}).get("CLASSES", dataset.CLASSES)
    if tuple(model.CLASSES) != tuple(dataset.CLASSES):
        raise ValueError("Checkpoint and dataset class order differ")
    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    records = []
    with AlignedRoICapture(model.module.roi_head.bbox_head,
                          rotated_bbox_head) as capture, torch.no_grad():
        for image_id, data in zip(run["image_ids"], loader):
            capture.reset()
            result = model(return_loss=False, rescale=True, **data)
            meta = data["img_metas"][0].data[0][0]
            if Path(meta["ori_filename"]).stem != image_id:
                raise ValueError("Loader image order differs from fixed selection")
            arrays = capture.aligned(image_id, result[0])
            if predictions is not None:
                validate_native_predictions(arrays, predictions[prediction_indices[image_id]])
            name = image_id + ".npz"
            np.savez_compressed(out / name, **arrays)
            records.append({
                "image_id": image_id, "image_path": meta["filename"],
                "feature_file": name, "n_roi": len(capture.features[0]),
                "n_detections": len(arrays["labels"]),
            })
    if len(records) != len(run["image_ids"]):
        raise ValueError("Extraction stopped before every full TEST image was exported")
    index = {
        "schema": SCHEMA, "status": "complete", "scope": "full_test", "run": run,
        "n_images": len(records),
        "n_post_nms_detections": sum(r["n_detections"] for r in records),
        "code_commit": code_commit,
        "classes": list(model.module.CLASSES), "rescale": True,
        "test_cfg": dict(model.module.roi_head.test_cfg),
        "feature_point": FEATURE_POINT, "feature_version": FEATURE_VERSION,
        "checkpoint_meta": {key: checkpoint.get("meta", {}).get(key)
                            for key in ("epoch", "iter")},
        "records": records,
    }
    if native is not None:
        index["native_prediction"] = native
    (out / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"Completed {run['run_id']}: {len(records)} images -> {out}")


if __name__ == "__main__":
    main()
