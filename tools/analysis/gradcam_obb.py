"""EigenCAM heatmaps for Oriented R-CNN TEST images.

CLI matches `test.py` vis: CONFIG CKPT --show-dir --show-score-thr --cfg-options.
Does not compute mAP. Overlay is EigenCAM on the highest-resolution FPN map
(or backbone output), then the detector `show_result` boxes.
"""

from iraod_runtime import ensure_iraod_runtime

ensure_iraod_runtime()

import argparse
import os
import os.path as osp
import warnings

import mmcv
import numpy as np
import torch
from mmcv import Config, DictAction
from mmcv.image import tensor2imgs
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.datasets import build_dataloader, replace_ImageToTensor
from mmrotate.datasets import build_dataset
from mmrotate.models import build_detector
from mmrotate.utils import compat_cfg, setup_multi_processes
from sfod.utils import patch_config

from tools.analysis.cam_overlay import eigen_cam_torch, overlay_cam


def parse_args():
    parser = argparse.ArgumentParser(description="Oriented R-CNN EigenCAM over TEST")
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--show-dir", required=True)
    parser.add_argument("--show-score-thr", type=float, default=0.3)
    parser.add_argument("--target-layer", choices=("neck", "backbone"), default="neck")
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--max-images", type=int, default=0, help="0 = full TEST")
    parser.add_argument("--skip-done", action="store_true")
    parser.add_argument("--no-boxes", action="store_true")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    parser.add_argument("--gpu-ids", type=int, nargs="+")
    return parser.parse_args()


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


def _hook_module(model, name):
    inner = _unwrap(model)
    target = inner.neck if name == "neck" and hasattr(inner, "neck") else inner.backbone
    captured = []

    def _hook(_module, _inputs, output):
        captured.clear()
        if isinstance(output, (list, tuple)):
            captured.extend(output)
        else:
            captured.append(output)

    handle = target.register_forward_hook(_hook)
    return captured, handle


def _largest_map(tensors):
    maps = [item for item in tensors if torch.is_tensor(item) and item.ndim == 4]
    if not maps:
        raise RuntimeError("CAM hook captured no NCHW feature maps")
    return max(maps, key=lambda item: item.shape[-1] * item.shape[-2])


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg = compat_cfg(cfg)
    setup_multi_processes(cfg)
    cfg.model.pretrained = None
    cfg.gpu_ids = args.gpu_ids if args.gpu_ids is not None else range(1)
    if len(cfg.gpu_ids) > 1:
        warnings.warn("non-distributed CAM uses one GPU; keeping gpu-ids[0]")
        cfg.gpu_ids = cfg.gpu_ids[0:1]
    loader_args = dict(samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False)
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if "samples_per_gpu" in cfg.data.test:
            loader_args["samples_per_gpu"] = cfg.data.test.pop("samples_per_gpu")
        if loader_args["samples_per_gpu"] > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    cfg = patch_config(cfg)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **{**cfg.data.get("test_dataloader", {}), **loader_args})
    mmcv.mkdir_or_exist(osp.abspath(args.show_dir))
    if args.skip_done:
        existing = sum(1 for name in os.listdir(args.show_dir) if name.lower().endswith((".png", ".jpg")))
        expected = len(dataset) if args.max_images <= 0 else min(args.max_images, len(dataset))
        if existing >= expected:
            print(f"skip_done {existing}/{expected} {args.show_dir}")
            return
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.CLASSES = checkpoint.get("meta", {}).get("CLASSES", dataset.CLASSES)
    model = MMDataParallel(model, device_ids=cfg.gpu_ids)
    model.eval()
    captured, handle = _hook_module(model, args.target_layer)
    prog = mmcv.ProgressBar(len(dataset) if args.max_images <= 0 else min(args.max_images, len(dataset)))
    written = 0
    try:
        for data in data_loader:
            img_metas = data["img_metas"][0].data[0]
            out_file = osp.join(args.show_dir, img_metas[0]["ori_filename"])
            if args.skip_done and osp.isfile(out_file):
                written += 1
                prog.update()
                if args.max_images > 0 and written >= args.max_images:
                    return
                continue
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)
            feat = _largest_map(captured)[0].detach()
            cam = eigen_cam_torch(feat[0] if feat.ndim == 4 else feat)
            img_tensor = data["img"][0] if isinstance(data["img"][0], torch.Tensor) else data["img"][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]["img_norm_cfg"])
            for img, img_meta, det in zip(imgs, img_metas, result):
                height, width, _ = img_meta["img_shape"]
                vis = img[:height, :width, :]
                ori_h, ori_w = img_meta["ori_shape"][:-1]
                vis = mmcv.imresize(vis, (ori_w, ori_h))
                vis = overlay_cam(vis, cam, alpha=args.alpha)
                out_file = osp.join(args.show_dir, img_meta["ori_filename"])
                if args.no_boxes:
                    mmcv.imwrite(vis, out_file)
                else:
                    _unwrap(model).show_result(
                        vis, det, show=False, out_file=out_file, score_thr=args.show_score_thr)
                written += 1
                prog.update()
                if args.max_images > 0 and written >= args.max_images:
                    return
    finally:
        handle.remove()


if __name__ == "__main__":
    main()
