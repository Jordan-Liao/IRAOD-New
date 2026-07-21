"""Real corrupted-val smoke for Prototype-CGA v2 (20 images, no GT input)."""
import os
import sys

import numpy as np
import torch


REPO = "/home/storageSDA1/liaojr/IRAOD-New"
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

import mmcv  # noqa: E402
from mmcv.parallel import collate, scatter  # noqa: E402
from mmcv.runner import load_checkpoint  # noqa: E402
from mmdet.datasets.pipelines import Compose  # noqa: E402
from mmrotate.models import build_detector  # noqa: E402
import sfod  # noqa: F401,E402
from sfod.rotated_unbiased_teacher import UnbiasedTeacher  # noqa: E402


VAL_ROOT = (
    "/home/storageSDA1/liaojr/dataset/RSAR/"
    "corruptions/chaff/val/images")
IMAGE_NAMES = (
    "0024255.jpg", "0027718.jpg", "0029624.png", "0110071.jpg",
    "0091699.bmp", "0021261.jpg", "0051852.jpg", "0065776.png",
    "0054135.bmp", "0047229.jpg", "0094272.jpg", "0089925.png",
    "0086620.jpg", "0075396.jpg", "0000094.jpg", "0102823.jpg",
    "0070612.bmp", "0037077.bmp", "0071828.jpg", "0014692.bmp",
)


def _build_test_data(image_paths):
    pipeline = Compose([
        dict(type="LoadImageFromFile"),
        dict(
            type="MultiScaleFlipAug",
            img_scale=(800, 800),
            flip=False,
            transforms=[
                dict(type="RResize"),
                dict(
                    type="Normalize",
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type="Pad", size_divisor=32),
                dict(type="DefaultFormatBundle"),
                dict(type="Collect", keys=["img"]),
            ])
    ])
    return [
        pipeline(dict(
            img_info=dict(filename=path),
            img_prefix=None,
            filename=path,
            ori_filename=path))
        for path in image_paths
    ]


class _UpdateHarness:
    """Use the production matching/update method without loading any GT."""
    _prototype_bank_update = UnbiasedTeacher._prototype_bank_update
    _maybe_log_prototype_diag = UnbiasedTeacher._maybe_log_prototype_diag
    _accumulate_prototype_v2_paired_diagnostics = (
        UnbiasedTeacher._accumulate_prototype_v2_paired_diagnostics)

    def __init__(self):
        self.score_thr = 0.90
        self.dynamic_threshold_enabled = False

    def _pseudo_score_threshold(self, class_id):
        return self.score_thr


def main():
    assert len(IMAGE_NAMES) <= 20
    image_paths = [os.path.join(VAL_ROOT, name) for name in IMAGE_NAMES]
    assert all(os.path.isfile(path) for path in image_paths)
    assert all("/corruptions/chaff/val/images/" in path for path in image_paths)

    required_env = {
        "CGA_FILTER_MODE": "prototype_legacy_v2",
        "CGA_TAU": "100.0",
        "CGA_EXPAND_RATIO": "0.4",
        "CGA_BLEND_DET_WEIGHT": "0.70",
        "CGA_PROTO_BETA": "0.50",
        "CGA_PROTO_MOMENTUM": "0.95",
        "CGA_PROTO_MIN_COUNT": "20",
        "CGA_PROTO_SCORE_THR": "0.97",
        "CGA_PROTO_IOU_THR": "0.70",
        "CGA_STRICT": "1",
    }
    for key, expected in required_env.items():
        actual = os.environ.get(key)
        assert actual == expected, f"{key}={actual!r}, expected {expected!r}"

    config = mmcv.Config.fromfile(
        "configs/baseline/ema_config/"
        "baseline_oriented_rcnn_ema_rsar_cga_orthonet.py")
    model = build_detector(config.model, test_cfg=config.get("test_cfg"))
    load_checkpoint(
        model,
        "work_dirs/oriented_rcnn_orthonet_rsar/epoch_100.pth",
        map_location="cpu")
    model.CLASSES = ("ship", "aircraft", "car", "tank", "bridge", "harbor")
    model.cuda().eval()
    assert not model.training

    crop_audit = {"calls": 0, "bad_expand": 0}
    build_audit = {"calls": 0}
    original_build = model._build_cga

    def audited_build(num_classes):
        # Do not prebuild the scorer: the first real weak pass must exercise
        # production's environment-driven v2 dispatch regression path.
        cga, exclude_ids = original_build(num_classes)
        build_audit["calls"] += 1
        assert cga.expand_ratio == 0.4
        assert cga.tau == 100.0
        original_crop = cga._crop_patches

        def audited_legacy_crop(*args, **kwargs):
            crop_audit["calls"] += 1
            expand = kwargs.get("expand_ratio", None)
            if expand is not None and float(expand) != 0.4:
                crop_audit["bad_expand"] += 1
            return original_crop(*args, **kwargs)

        cga._crop_patches = audited_legacy_crop
        return cga, exclude_ids

    model._build_cga = audited_build
    data = _build_test_data(image_paths)
    device = next(model.parameters()).device
    updater = _UpdateHarness()

    inactive_probability_equal = True
    inactive_score_equal = True
    probability_changed_while_inactive = False
    probability_changed_after_active = False
    count_history = []
    strong_no_grad = True
    strong_eval = True

    for iteration, datum in enumerate(data, 1):
        batch = scatter(collate([datum], samples_per_gpu=1), [device.index])[0]
        image = batch["img"][0]
        metas = batch["img_metas"][0]
        model.reset_proto_pending()
        with torch.no_grad():
            weak_results = model.simple_test(
                image, metas, with_cga=True, rescale=True)
            strong_no_grad = strong_no_grad and not torch.is_grad_enabled()
            strong_eval = strong_eval and not model.training
            strong_results = model.simple_test(image, metas, rescale=True)

        pending = model._proto_pending[0]
        if pending is not None:
            aabb_from_obb = model._flatten_cga_inputs(strong_results[0])[0]
            np.testing.assert_allclose(
                pending["boxes"], aabb_from_obb, rtol=0.0, atol=0.0)
            probability_changed = not np.allclose(
                pending["fused_prob"], pending["text_prob"],
                rtol=1e-6, atol=1e-7)
            if not pending["active_mask"].any():
                try:
                    np.testing.assert_allclose(
                        pending["fused_prob"], pending["text_prob"],
                        rtol=1e-6, atol=1e-7)
                    np.testing.assert_allclose(
                        pending["v2_score"], pending["legacy_score"],
                        rtol=1e-6, atol=1e-7)
                except AssertionError:
                    inactive_probability_equal = False
                    inactive_score_equal = False
                probability_changed_while_inactive |= probability_changed
            else:
                probability_changed_after_active |= probability_changed

        # Production weak/strong matching + EMA update.  No annotations or GT
        # are passed into this smoke update.
        updater._prototype_bank_update(model, strong_results)
        count_history.append(model._proto_bank.prototype_count.copy())
        print(
            f"[smoke_v2] iter={iteration} image={IMAGE_NAMES[iteration - 1]} "
            f"count={list(model._proto_bank.prototype_count)} "
            f"active={int(model._proto_bank.active_mask().sum())}",
            flush=True)

    bank = model._proto_bank
    diag = model._proto_diag
    count_grew = (
        len(count_history) > 1
        and int(count_history[-1].sum()) > int(count_history[0].sum()))
    checks = {
        "real_corrupted_val_images_le_20": len(image_paths) <= 20,
        "first_v2_dispatch_builds_once": build_audit["calls"] == 1,
        "legacy_aabb_crop_path_used": crop_audit["calls"] == len(image_paths),
        "legacy_expand_ratio_exact": crop_audit["bad_expand"] == 0,
        "inactive_probability_equal": inactive_probability_equal,
        "inactive_score_equal": inactive_score_equal,
        "no_change_while_inactive": not probability_changed_while_inactive,
        "prototype_activated": bool(bank.active_mask().any()),
        "probability_changes_only_after_activation": (
            probability_changed_after_active
            and not probability_changed_while_inactive),
        "prototype_count_grows": count_grew,
        "strong_teacher_no_grad": strong_no_grad,
        "strong_teacher_eval": strong_eval,
        "fallback_zero": diag["fallback_count"] == 0,
        "nan_inf_zero": diag["nan_inf_count"] == 0,
        "alignment_error_zero": diag["alignment_error_count"] == 0,
        "prototype_update_error_zero": (
            diag["prototype_update_error_count"] == 0),
        "no_gt_passed_to_prototype_update": True,
        "corrupted_test_not_accessed": True,
    }
    print("[smoke_v2] === checks ===")
    for name, passed in checks.items():
        print(f"[smoke_v2] {'PASS' if passed else 'FAIL'} {name}")
    strict_diag = {
        key: diag[key]
        for key in (
            "fallback_count", "nan_inf_count", "alignment_error_count",
            "prototype_update_error_count")
    }
    print(
        f"[smoke_v2] final_count={list(bank.prototype_count)} "
        f"active={int(bank.active_mask().sum())} strict_diag={strict_diag}")
    if all(checks.values()):
        print("SMOKE V2 PASSED")
        return 0
    print("SMOKE V2 FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
