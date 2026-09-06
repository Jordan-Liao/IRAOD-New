"""Versioned, exactly post-NMS-aligned fc_cls input export (compute owner only)."""

import argparse
import json
from pathlib import Path
import subprocess

from experiments.comparison.result_completion import load_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    run = load_run(args.plan, args.run_id)

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
    from experiments.comparison.aligned_roi import AlignedRoICapture, SCHEMA

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
            name = image_id + ".npz"
            np.savez_compressed(out / name, **arrays)
            records.append({
                "image_id": image_id, "image_path": meta["filename"],
                "feature_file": name, "n_roi": len(capture.features[0]),
                "n_detections": len(arrays["labels"]),
            })
    index = {
        "schema": SCHEMA, "status": "complete", "run": run,
        "code_commit": subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[3]),
             "rev-parse", "HEAD"], text=True).strip(),
        "classes": list(model.module.CLASSES), "rescale": True,
        "test_cfg": dict(model.module.roi_head.test_cfg),
        "feature_point": "roi_head.bbox_head.fc_cls.input",
        "checkpoint_meta": {key: checkpoint.get("meta", {}).get(key)
                            for key in ("epoch", "iter")},
        "records": records,
    }
    (out / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"Completed {run['run_id']}: {len(records)} images -> {out}")


if __name__ == "__main__":
    main()
