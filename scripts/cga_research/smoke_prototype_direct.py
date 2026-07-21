"""Fast direct smoke for prototype_legacy: real teacher + real SARCLIP+LoRA on a
handful of real corrupted-val images, bypassing the slow 8467-file dataset build.

Exercises the NEW code end-to-end: rotated-align crop, ONE SARCLIP encode per
proposal (embedding + text prob share it), text/visual fusion + legacy blend,
weak/strong greedy matching, EMA bank update, strict mode. No GT is used to
build prototypes (proposals come from the detector, matched weak vs strong).
"""
import os
import sys
import glob
import numpy as np
import torch

REPO = "/home/storageSDA1/liaojr/IRAOD-New"
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

import mmcv                                    # noqa: E402
from mmcv.parallel import collate, scatter     # noqa: E402
from mmcv.runner import load_checkpoint        # noqa: E402
from mmdet.datasets.pipelines import Compose   # noqa: E402
from mmrotate.models import build_detector     # noqa: E402
import sfod                                     # noqa: E402  (registers modules)
from sfod.prototype_cga import greedy_match_weak_strong  # noqa: E402
from mmrotate.core.bbox import rbbox_overlaps   # noqa: E402


def build_test_data(cfg, img_paths):
    pipeline = Compose([
        dict(type='LoadImageFromFile'),
        dict(type='MultiScaleFlipAug', img_scale=(800, 800), flip=False,
             transforms=[
                 dict(type='RResize'),
                 dict(type='Normalize',
                      mean=[123.675, 116.28, 103.53],
                      std=[58.395, 57.12, 57.375], to_rgb=True),
                 dict(type='Pad', size_divisor=32),
                 dict(type='DefaultFormatBundle'),
                 dict(type='Collect', keys=['img']),
             ])])
    datas = []
    for p in img_paths:
        d = pipeline(dict(img_info=dict(filename=p), img_prefix=None,
                          filename=p, ori_filename=p))
        datas.append(d)
    return datas


def flatten_obb(host, results_per_img):
    return host._flatten_cga_obb(results_per_img)


def main():
    corrupt = os.environ.get("SMOKE_CORRUPT", "chaff")
    val_dir = (f"/home/storageSDA1/liaojr/dataset/RSAR/corruptions/"
               f"{corrupt}/val/images/")
    imgs = sorted(glob.glob(os.path.join(val_dir, "*")))[:6]
    assert imgs, f"no images under {val_dir}"
    print(f"[smoke] {len(imgs)} corrupted-val images from {val_dir}")

    ema_cfg = mmcv.Config.fromfile(
        "configs/baseline/ema_config/"
        "baseline_oriented_rcnn_ema_rsar_cga_orthonet.py")
    model = build_detector(ema_cfg.model, test_cfg=ema_cfg.get("test_cfg"))
    load_checkpoint(model, "work_dirs/oriented_rcnn_orthonet_rsar/epoch_100.pth",
                    map_location="cpu")
    model.CLASSES = ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor')
    model.cuda().eval()
    print("[smoke] teacher (OrientedRCNN_CGA) built + epoch_100 loaded")

    datas = build_test_data(ema_cfg, imgs)
    device = next(model.parameters()).device

    # weak = prototype_legacy CGA path (stashes weak proposals + embeddings);
    # strong = raw pass (reuse same tensor: geometry identical, exercises match).
    if hasattr(model, "reset_proto_pending"):
        model.reset_proto_pending()

    encode_ok = True
    n_props_total = 0
    for it, d in enumerate(datas, 1):
        batch = scatter(collate([d], samples_per_gpu=1), [device.index])[0]
        img = batch["img"][0]
        metas = batch["img_metas"][0]
        with torch.no_grad():
            weak = model.simple_test(img, metas, with_cga=True, rescale=True)
            strong = model.simple_test(img, metas, rescale=True)

        # verify one encode per proposal via the stash
        pend = model._proto_pending[-1] if model._proto_pending else None
        if pend is not None:
            n = len(pend["obb"])
            n_props_total += n
            if not (pend["embed"].shape[0] == n == len(pend["label"])):
                encode_ok = False
            if n and not np.all(np.isfinite(pend["embed"])):
                raise RuntimeError("NaN/Inf in embeddings")

        # weak/strong greedy match -> EMA bank update (mirrors teacher hook)
        s_inp = flatten_obb(model, strong[0])
        if pend is not None and s_inp is not None:
            s_obb, s_score, s_label = s_inp

            def iou_fn(a, b):
                return rbbox_overlaps(
                    torch.tensor(np.asarray(a), dtype=torch.float32),
                    torch.tensor(np.asarray(b), dtype=torch.float32)
                ).cpu().numpy()

            qual = greedy_match_weak_strong(
                pend["obb"], pend["label"], pend["raw_score"],
                s_obb, s_label, iou_fn,
                score_thr=float(getattr(model, "cga_proto_score_thr", 0.97)),
                iou_thr=float(getattr(model, "cga_proto_iou_thr", 0.70)))
            c2e = {}
            for i in np.where(qual)[0]:
                c = int(pend["label"][i])
                c2e.setdefault(c, []).append(pend["embed"][i])
            stacked = {c: np.stack(v, 0) for c, v in c2e.items()}
            model._proto_bank.snapshot_previous()
            model._proto_bank.update(stacked, it)
        bank = model._proto_bank
        print(f"[smoke] iter{it}: proposals={0 if pend is None else len(pend['obb'])} "
              f"proto_count={list(bank.prototype_count)} "
              f"active={int(bank.active_mask().sum())} "
              f"fallback={model._proto_diag['fallback_count']} "
              f"nan_inf={model._proto_diag['nan_inf_count']} "
              f"align_err={model._proto_diag['alignment_error_count']}")

    bank = model._proto_bank
    diag = model._proto_diag
    print("\n[smoke] === CHECKS ===")
    checks = {
        "strict_mode_on": bool(getattr(model, "cga_strict", False)),
        "encode_once_per_proposal": encode_ok,
        "proposals_seen>0": n_props_total > 0,
        "prototype_count_nonzero": int(bank.prototype_count.sum()) > 0,
        "at_least_one_initialized": bool(bank.prototype_initialized.any()),
        "no_nan_inf": diag["nan_inf_count"] == 0,
        "fallback_zero": diag["fallback_count"] == 0,
        "align_err_zero": diag["alignment_error_count"] == 0,
        "proto_update_err_zero": diag["prototype_update_error_count"] == 0,
    }
    for k, v in checks.items():
        print(f"  {'OK ' if v else 'FAIL'} {k} = {v}")
    print(f"[smoke] final proto_count={list(bank.prototype_count)} "
          f"initialized={list(bank.prototype_initialized)} "
          f"active={int(bank.active_mask().sum())}")

    # strict-mode negative check: degenerate OBB must raise
    raised = False
    try:
        model.cga.forward_rotated_embed(imgs[0], np.array([[10, 10, -1, 5, 0.0]]))
    except Exception:
        raised = True
    print(f"  {'OK ' if raised else 'FAIL'} strict_degenerate_raises = {raised}")
    checks["strict_degenerate_raises"] = raised

    if all(checks.values()):
        print("\nSMOKE PASSED")
        return 0
    print("\nSMOKE FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
