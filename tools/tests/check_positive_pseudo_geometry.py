"""Replay captured degenerate pseudo geometry through the real admission/coder/loss."""

import argparse
import ast
import functools
from pathlib import Path
import subprocess
import sys
import types
import unittest

import mmcv
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
METHOD_FILE = "sfod/rotated_unbiased_teacher.py"
A39 = "a39c83221c73e71d273b96ef95c65f6d8d1e84f9"
D668 = "d668a7719833ea59c6320380b6bc4f80a59b5399"
FUNCTIONS = []
NATIVE = {}
OPERANDS = None


def git_source(ref, path):
    return subprocess.check_output(["git", "-C", str(ROOT), "show", ref + ":" + path], text=True)


def admission(source, filename):
    tree = ast.parse(source)
    model = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "UnbiasedTeacher")
    method = next(node for node in model.body
                  if isinstance(node, ast.FunctionDef) and node.name == "create_pseudo_results")

    def no_gt(*args, **kwargs):
        raise AssertionError("Image-only pseudo admission must not inspect test GT")

    namespace = {"np": np, "torch": torch, "rbbox_overlaps": no_gt}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                 filename, "exec"), namespace)
    return namespace["create_pseudo_results"], ast.dump(method, include_attributes=False)


def host():
    model = types.SimpleNamespace(
        pseudo_num=np.zeros(6), pseudo_sem_weight=np.zeros(6),
        pseudo_num_tp=np.zeros(6), pseudo_num_gt=np.zeros(6),
        srw_disagree=0, srw_disagree_low=0, srw_disagree_high=0,
        semantic_low_thr=.7, semantic_high_thr=.97, threshold_calls=[])
    model._pseudo_score_threshold = lambda cls: .7
    model._update_dynamic_score_thresholds = model.threshold_calls.append
    return model


def encode(boxes):
    cfg = OPERANDS["configuration"]
    return NATIVE["bbox2delta"](
        boxes, boxes, cfg["means"], cfg["stds"], cfg["angle_range"],
        cfg["norm_factor"], cfg["edge_swap"], cfg["proj_xy"])


class PositiveGeometryTest(unittest.TestCase):
    def test_actual_zero_height_admission_and_native_loss_regression(self):
        geometry = OPERANDS["positive_geometry"]
        bad = next(row for row in geometry if row["image_index"] == 26)
        proposal = bad["proposals"][2:3]
        assigned = bad["gt_boxes"][2:3]
        self.assertTrue(torch.equal(proposal, assigned))
        self.assertEqual(proposal[0, 3].item(), 0.)
        self.assertTrue(torch.isfinite(proposal).all())
        original_target = encode(proposal)
        self.assertEqual(torch.isnan(original_target).nonzero().tolist(), [[0, 1], [0, 3]])
        self.assertTrue(torch.equal(torch.isnan(original_target), torch.isnan(OPERANDS["target"][379:380])))
        criterion = NATIVE["SmoothL1Loss"](beta=1., loss_weight=1.)
        pred = OPERANDS["pred"][379:380].clone()
        self.assertTrue(torch.isfinite(pred).all())
        self.assertTrue(torch.isnan(criterion(pred, original_target, torch.ones_like(pred),
                                             avg_factor=OPERANDS["avg_factor"])))
        valid = torch.cat([row["gt_boxes"] for row in geometry
                           if row["gt_boxes"].ndim == 2 and row["gt_boxes"].numel()])
        valid = valid[torch.isfinite(valid).all(dim=1) & (valid[:, 2] > 0) & (valid[:, 3] > 0)][:3]
        self.assertEqual(len(valid), 3)
        # Scores are controlled above-threshold fixtures; the captured geometry
        # and native encoder/loss operands, not an invented historical score, define RED.
        result = [np.empty((0, 6), dtype=np.float32) for _ in range(6)]
        result[1] = np.column_stack((valid[:1].numpy(), np.array([.85], dtype=np.float32)))
        result[5] = np.column_stack((
            torch.cat((valid[1:2], proposal, valid[2:3], valid[1:2])).numpy(),
            np.array([.90, .95, .99, .1], dtype=np.float32)))
        saved_inputs = [row.copy() for row in result]
        meta = {"semantic_weight": [np.ones(len(row)) for row in result],
                "agreement": [np.ones(len(row), dtype=bool) for row in result],
                "det_score": [row[:, -1].copy() for row in result]}
        meta["semantic_weight"][1] = np.array([1.2])
        meta["semantic_weight"][5] = np.array([.11, .22, .33, .44])
        meta["agreement"][5] = np.array([False, False, False, False])
        original, original_ast = admission(git_source(A39, METHOD_FILE), "original_a39")
        _, d668_ast = admission(git_source(D668, METHOD_FILE), "original_d668")
        self.assertEqual(original_ast, d668_ast)
        old_boxes, _ = original(host(), torch.zeros(1, 3, 8, 8), [result], [], "cpu")
        self.assertFalse(torch.isfinite(encode(old_boxes[0])).all())

        expected_boxes = torch.cat((valid[:1], valid[1:2], valid[2:3]))
        expected_labels = torch.tensor([1, 5, 5])
        for label, method in FUNCTIONS:
            with self.subTest(source=label):
                model = host()
                boxes, labels, weights = method(
                    model, torch.zeros(1, 3, 8, 8), [result], [], "cpu",
                    cga_meta=[meta], return_semantic_weights=True)
                targets = encode(boxes[0])
                prediction = pred.repeat(len(boxes[0]), 1).requires_grad_()
                loss = criterion(prediction, targets, torch.ones_like(targets),
                                 avg_factor=OPERANDS["avg_factor"])
                self.assertTrue(torch.isfinite(targets).all(), "Degenerate pseudo GT reached the native coder")
                self.assertTrue(torch.isfinite(loss), "Admitted geometry still produces native NaN loss")
                loss.backward()
                self.assertGreater(loss.item(), 0)
                self.assertTrue(torch.isfinite(prediction.grad).all())
                self.assertGreater(prediction.grad.abs().sum().item(), 0)
                self.assertTrue(torch.equal(boxes[0], expected_boxes))
                self.assertTrue(torch.equal(labels[0], expected_labels))
                torch.testing.assert_close(weights[0], torch.tensor([1.2, .11, .33]), rtol=0, atol=0)
                self.assertEqual(model.srw_disagree, 2)
                self.assertEqual(model.srw_disagree_low, 1)
                self.assertEqual(model.srw_disagree_high, 1)
                self.assertEqual(model.pseudo_num.tolist(), [0, 1, 0, 0, 0, 2])
                self.assertEqual(model.pseudo_num_gt.sum(), 0)
                self.assertIs(model.threshold_calls[0][0], result)
                for current, before in zip(result, saved_inputs):
                    np.testing.assert_array_equal(current, before)

                clean = [row.copy() for row in result]
                clean[5] = clean[5][[0, 2, 3]]
                old = original(host(), torch.zeros(1, 3, 8, 8), [clean], [], "cpu")
                new = method(host(), torch.zeros(1, 3, 8, 8), [clean], [], "cpu")
                for old_fields, new_fields in zip(old, new):
                    self.assertTrue(torch.equal(old_fields[0], new_fields[0]))
                invalid = [np.empty((0, 6), dtype=np.float32) for _ in range(6)]
                invalid[5] = np.repeat(result[5][1:2], 3, axis=0)
                invalid[5][1, 2] = -1.
                invalid[5][2, 4] = np.nan
                empty = method(host(), torch.zeros(1, 3, 8, 8), [invalid], [], "cpu",
                               return_semantic_weights=True)
                self.assertEqual(len(empty[0]), 1)
                self.assertEqual(empty[0][0].shape, (0, 5))
                self.assertEqual(empty[1][0].shape, (0,))
                self.assertEqual(empty[2][0].shape, (0,))
                empty_pred = torch.empty(0, 5, requires_grad=True)
                empty_loss = criterion(empty_pred, encode(empty[0][0]), avg_factor=1)
                self.assertEqual(empty_loss.item(), 0.)
                empty_loss.backward()

        for ref in (A39, D668):
            shared = git_source(ref, "sfod/extensions/proposal_teacher.py")
            self.assertLess(shared.index("self.create_pseudo_results("),
                            shared.index("self.rpn_head.forward_train("))
            self.assertLess(shared.index("self.create_pseudo_results("),
                            shared.index("self.roi_head.forward_train("))
            for method in ("irg", "lpld", "sfut"):
                self.assertIn("(ProposalAlignedTeacher)", git_source(ref, f"sfod/extensions/{method}.py"))
        sfyolo = git_source(D668, "sfod/extensions/sfyolo.py")
        self.assertIn("class SFYOLOOBB(SFUTOBB)", sfyolo)
        self.assertIn("super().forward_train_semi(", sfyolo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operand-packet", required=True)
    parser.add_argument("--source-packet", required=True)
    parser.add_argument("--angle-source", required=True)
    parser.add_argument("--other-patched-ref")
    args = parser.parse_args()
    OPERANDS = torch.load(args.operand_packet, map_location="cpu", weights_only=True)
    source = Path(args.source_packet)
    namespace = {"torch": torch, "np": np, "mmcv": mmcv, "functools": functools,
                 "nn": torch.nn, "F": torch.nn.functional}
    for filename, wanted in (
            (Path(args.angle_source), {"norm_angle"}),
            (source / "mmdet_loss_utils.py", {"reduce_loss", "weight_reduce_loss", "weighted_loss"}),
            (source / "mmrotate_delta_xywha_rbbox_coder.py", {"bbox2delta"}),
            (source / "mmdet_smooth_l1_loss.py", {"smooth_l1_loss", "SmoothL1Loss"})):
        tree = ast.parse(filename.read_text())
        nodes = [node for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted]
        for node in nodes:
            if isinstance(node, ast.ClassDef):
                node.decorator_list = []
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                     str(filename), "exec"), namespace)
    NATIVE.update({name: namespace[name] for name in ("bbox2delta", "SmoothL1Loss")})
    current, current_ast = admission((ROOT / METHOD_FILE).read_text(), str(ROOT / METHOD_FILE))
    FUNCTIONS.append(("working_tree", current))
    if args.other_patched_ref:
        other, other_ast = admission(git_source(args.other_patched_ref, METHOD_FILE), args.other_patched_ref)
        if current_ast != other_ast:
            raise ValueError("a39/d668 patched pseudo-admission bodies differ")
        FUNCTIONS.append((args.other_patched_ref, other))
    unittest.main(argv=[sys.argv[0]], verbosity=2)
