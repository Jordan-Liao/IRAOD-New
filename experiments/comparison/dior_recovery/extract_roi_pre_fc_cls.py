from iraod_runtime import ensure_iraod_runtime
ensure_iraod_runtime()
import argparse, json, os
import numpy as np, torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet.datasets import build_dataloader
from mmrotate.datasets import build_dataset
from mmrotate.models import build_detector
from mmrotate.utils import compat_cfg, setup_multi_processes
from sfod.utils import patch_config

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config"); ap.add_argument("checkpoint")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ann-file", default="")
    ap.add_argument("--img-prefix", default="")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = patch_config(compat_cfg(Config.fromfile(args.config)))
    setup_multi_processes(cfg)
    cfg.model.train_cfg = None
    cfg.data.test.test_mode = True
    if args.ann_file: cfg.data.test.ann_file = args.ann_file
    if args.img_prefix: cfg.data.test.img_prefix = args.img_prefix
    dataset = build_dataset(cfg.data.test)
    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False)
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model = MMDataParallel(model, device_ids=[0]); model.eval()
    feats=[]
    def hook(_m, inp, _out):
        feats.append(inp[0].detach().float().cpu().numpy())
    handle = model.module.roi_head.bbox_head.fc_cls.register_forward_hook(hook)
    recs=[]; n=0
    with torch.no_grad():
        for data in loader:
            feats.clear()
            result = model(return_loss=False, rescale=True, **data)
            img_id = data["img_metas"][0].data[0][0]["ori_filename"]
            per_cls = result[0] if isinstance(result, list) and result and isinstance(result[0], list) else result
            boxes=[]
            for cls_i, arr in enumerate(per_cls):
                if arr is None or len(arr)==0: continue
                for row in np.asarray(arr):
                    boxes.append((int(cls_i), float(row[-1]), row[:5].astype(float).tolist()))
            feat = feats[-1] if feats else np.zeros((0,1), np.float32)
            recs.append({"image_id": img_id, "n_roi": int(feat.shape[0]), "preds": boxes})
            np.save(os.path.join(args.out_dir, os.path.splitext(os.path.basename(img_id))[0]+".npy"), feat)
            n += 1
    handle.remove()
    with open(os.path.join(args.out_dir, "index.json"), "w") as f:
        json.dump({"checkpoint": args.checkpoint, "n_images": n, "records": recs}, f)
    print("wrote", args.out_dir, "n", n)
if __name__ == "__main__":
    main()
