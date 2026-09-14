"""CPU check of retained native ROI regression code and failure-only operand capture."""

import argparse
import ast
import functools
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

import mmcv
import numpy as np
import torch
from mmcv.runner import force_fp32

from experiments.comparison.nonfinite_a39_epoch import load_finite_helper
from experiments.comparison.nonfinite_bbox_operands import BBoxLossProbe


ROOT = Path(__file__).resolve().parents[2]
NATIVE = {}


class CoderFixture:
    means = (0.,) * 5
    stds = (.1, .1, .2, .2, .1)
    angle_range = "le90"
    norm_factor = None
    edge_swap = proj_xy = True

    def encode(self, proposals, gt):
        return NATIVE["bbox2delta"](
            proposals, gt, self.means, self.stds, self.angle_range,
            self.norm_factor, self.edge_swap, self.proj_xy)


class HeadFixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_classes = 6
        self.reg_class_agnostic = True
        self.reg_decoded_bbox = False
        self.fp16_enabled = False
        self.bbox_coder = CoderFixture()
        self.loss_bbox = NATIVE["SmoothL1Loss"](beta=1.)

    def _get_target_single(self, *args, **kwargs):
        return NATIVE["_get_target_single"](self, *args, **kwargs)

    def loss(self, *args, **kwargs):
        return NATIVE["loss"](self, *args, **kwargs)


class RunnerFixture:
    def __init__(self):
        self.iter = 0
        self.head = HeadFixture()
        self.model = types.SimpleNamespace(roi_head=types.SimpleNamespace(bbox_head=self.head))

    def call_hook(self, name):
        return name


class OperandTest(unittest.TestCase):
    def test_preservation_and_native_red_operand_boundaries(self):
        checks = load_finite_helper(ROOT / "experiments/comparison/numerical_integrity.py")
        original_save = torch.save
        original_reduce = NATIVE["utils"].weight_reduce_loss
        source_boxes = torch.tensor([[20., 30, 10, 8, 0], [40, 50, 6, 12, 0]])
        with tempfile.TemporaryDirectory() as temporary:
            reference = None
            for mode in ("baseline", "finite", "target_nan", "pred_nan", "reduction_inf"):
                with self.subTest(mode=mode):
                    torch.manual_seed(44)
                    before_rng = torch.get_rng_state().clone()
                    runner = RunnerFixture()
                    pred = torch.nn.Parameter(torch.full((4, 5), .2))
                    optimizer = torch.optim.SGD([pred], lr=.02, momentum=.9)
                    updates, losses = 0, []
                    failure = Path(temporary) / mode
                    probe = BBoxLossProbe(failure, checks.require_finite, original_save, NATIVE["utils"])

                    def run():
                        nonlocal updates
                        runner.call_hook("before_run")
                        for i in range(265):
                            runner.iter = i
                            runner.call_hook("before_train_iter")
                            proposals = source_boxes.clone()
                            gt = source_boxes.clone()
                            gt[:, 0] += .5
                            if i == 2 and mode == "target_nan":
                                proposals[0, 2] = gt[0, 2] = 0
                                gt[0, 0] = proposals[0, 0]
                            empty = runner.head._get_target_single(
                                source_boxes[:0], source_boxes[:1], torch.empty(0),
                                torch.empty(0, dtype=torch.long), types.SimpleNamespace(pos_weight=-1))
                            positive = runner.head._get_target_single(
                                proposals, source_boxes[:1], gt, torch.tensor([0, 1]),
                                types.SimpleNamespace(pos_weight=-1))
                            labels, label_weights, targets, weights = [
                                torch.cat((left, right)) for left, right in zip(empty, positive)]
                            current_pred = pred
                            if i == 2 and mode == "pred_nan":
                                current_pred = pred * torch.tensor(-1.).sqrt()
                            elif i == 2 and mode == "reduction_inf":
                                current_pred = pred * 0 + torch.tensor(1e38)
                            result = runner.head.loss(
                                None, current_pred, torch.zeros(4, 6), labels,
                                label_weights, targets, weights)["loss_bbox"]
                            checks.require_finite(result, "fixture loss before backward")
                            losses.append(result.item())
                            optimizer.zero_grad()
                            result.backward()
                            optimizer.step()
                            updates += 1
                            runner.call_hook("after_train_iter")

                    if mode in ("baseline", "finite"):
                        with checks.optimizer_boundary():
                            if mode == "finite":
                                with probe.install(RunnerFixture):
                                    run()
                            else:
                                run()
                        self.assertEqual(updates, 265)
                        self.assertFalse(failure.exists())
                        self.assertGreater(losses[0], 0)
                        self.assertFalse(torch.equal(pred, torch.full((4, 5), .2)))
                        if mode == "baseline":
                            reference = (pred.detach().clone(), optimizer.state_dict(), losses)
                        else:
                            self.assertTrue(torch.equal(pred, reference[0]))
                            self.assertTrue(torch.equal(optimizer.state_dict()["state"][0]["momentum_buffer"],
                                                        reference[1]["state"][0]["momentum_buffer"]))
                            self.assertEqual(losses, reference[2])
                    else:
                        with probe.install(RunnerFixture), checks.optimizer_boundary(), \
                                self.assertRaises(FloatingPointError):
                            run()
                        self.assertEqual(updates, 2)
                        metadata = json.loads((failure / "failure.json").read_text())
                        self.assertEqual(metadata["capture_status"], "complete")
                        self.assertEqual(metadata["update_one_based"], 3)
                        self.assertEqual(metadata["positive_count"], 2)
                        self.assertEqual(metadata["geometry_positive_count"], 2)
                        self.assertEqual(metadata["avg_factor_repr"], "4")
                        self.assertTrue(metadata["avg_factor_isfinite"])
                        self.assertIsNotNone(metadata["per_element_loss"])
                        self.assertEqual(metadata["stage"], "bbox_reduction_output"
                                         if mode == "reduction_inf" else "before_bbox_weight_reduction")
                        payload = torch.load(failure / "bbox_loss_operands.pt",
                                             map_location="cpu", weights_only=False)
                        self.assertEqual(payload["pred"].shape, (2, 5))
                        self.assertEqual(len(payload["positive_geometry"]), 2)
                        self.assertEqual(payload["positive_geometry"][0]["gt_boxes"].shape, (0,))
                        self.assertEqual(payload["positive_geometry"][0]["encoded_targets"].shape, (0, 5))
                        self.assertIsNone(metadata["geometry"][0]["gt_nonpositive_wh"])
                        self.assertFalse(metadata["geometry"][0]["encode_called"])
                        self.assertTrue(metadata["geometry"][1]["encode_called"])
                        self.assertTrue(torch.equal(
                            torch.isnan(payload["positive_geometry"][1]["encoded_targets"]),
                            torch.isnan(payload["target"])))
                        self.assertFalse({"model", "optimizer", "features", "cls_score", "batch"} & payload.keys())
                        self.assertEqual(set(p.name for p in failure.iterdir()),
                                         {"failure.json", "bbox_loss_operands.pt"})
                        if mode == "target_nan":
                            self.assertGreater(metadata["encoded_targets"]["nan_count"], 0)
                            self.assertGreater(metadata["geometry"][1]["proposal_nonpositive_wh"], 0)
                        elif mode == "pred_nan":
                            self.assertGreater(metadata["pred"]["nan_count"], 0)
                        else:
                            self.assertEqual(metadata["per_element_loss"]["finite_count"], 10)
                            self.assertGreater(metadata["output"]["inf_count"], 0)
                    self.assertTrue(torch.equal(torch.get_rng_state(), before_rng))
                    self.assertIs(NATIVE["utils"].weight_reduce_loss, original_reduce)
                    self.assertFalse(torch.cuda.is_initialized())
                    self.assertNotIn("forward", runner.head.loss_bbox.__dict__)
                    self.assertNotIn("_get_target_single", runner.head.__dict__)
            # Native empty-positive behavior is preserved, not reclassified as a loss failure.
            runner = RunnerFixture()
            probe = BBoxLossProbe(Path(temporary) / "empty", checks.require_finite,
                                  original_save, NATIVE["utils"])
            with probe.install(RunnerFixture):
                runner.call_hook("before_run")
                runner.call_hook("before_train_iter")
                loss = runner.head.loss(None, torch.full((1, 5), float("nan")), torch.zeros(1, 6),
                                        torch.tensor([6]), torch.ones(1),
                                        torch.zeros(1, 5), torch.zeros(1, 5))["loss_bbox"]
                self.assertEqual(loss.item(), 0)
                # The global reduction function is unchanged outside the selected ROI call.
                self.assertEqual(NATIVE["utils"].weight_reduce_loss(torch.tensor([2., 4.])).item(), 3)
            self.assertFalse((Path(temporary) / "empty").exists())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-packet", required=True)
    args = parser.parse_args()
    packet = Path(args.source_packet)
    utils = types.ModuleType("native_loss_utils_fixture")
    utils.__dict__.update(torch=torch, F=torch.nn.functional, mmcv=mmcv, functools=functools)
    tree = ast.parse((packet / "mmdet_loss_utils.py").read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])),
                 str(packet / "mmdet_loss_utils.py"), "exec"), utils.__dict__)
    NATIVE["utils"] = utils
    namespace = {"torch": torch, "nn": torch.nn, "mmcv": mmcv, "weighted_loss": utils.weighted_loss}
    tree = ast.parse((packet / "mmdet_smooth_l1_loss.py").read_text())
    definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    for node in definitions:
        if isinstance(node, ast.ClassDef):
            node.decorator_list = []
    exec(compile(ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[])),
                 str(packet / "mmdet_smooth_l1_loss.py"), "exec"), namespace)
    NATIVE["SmoothL1Loss"] = namespace["SmoothL1Loss"]
    namespace = {"torch": torch, "force_fp32": force_fp32}
    tree = ast.parse((packet / "mmrotate_rotated_bbox_head.py").read_text())
    head = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RotatedBBoxHead")
    methods = [node for node in head.body if isinstance(node, ast.FunctionDef)
               and node.name in ("_get_target_single", "loss")]
    exec(compile(ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[])),
                 str(packet / "mmrotate_rotated_bbox_head.py"), "exec"), namespace)
    NATIVE.update({key: namespace[key] for key in ("_get_target_single", "loss")})
    tree = ast.parse((packet / "mmrotate_delta_xywha_rbbox_coder.py").read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "bbox2delta")
    namespace = {"torch": torch, "np": np, "mmcv": mmcv,
                 "norm_angle": lambda angle, version: (angle + np.pi / 2) % np.pi - np.pi / 2}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 str(packet / "mmrotate_delta_xywha_rbbox_coder.py"), "exec"), namespace)
    NATIVE["bbox2delta"] = namespace["bbox2delta"]
    unittest.main(argv=[sys.argv[0]], verbosity=2)
