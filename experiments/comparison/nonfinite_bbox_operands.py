"""Observe only the selected ROI bbox loss/reduction; save small operands on failure."""

import argparse
from contextlib import contextmanager, ExitStack
import importlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import sys
from unittest.mock import patch


def tensor_info(value):
    import torch

    if value is None:
        return None
    tensor = value.detach()
    finite = torch.isfinite(tensor)
    count = int(finite.sum().item())
    values = tensor[finite]
    return {
        "dtype": str(tensor.dtype), "shape": list(tensor.shape), "numel": tensor.numel(),
        "finite_count": count, "nan_count": int(torch.isnan(tensor).sum().item()),
        "inf_count": int(torch.isinf(tensor).sum().item()),
        "finite_min_repr": repr(values.min().item()) if count else None,
        "finite_max_repr": repr(values.max().item()) if count else None,
    }


def cpu_tensor(value):
    return value.detach().cpu() if value is not None else None


def nonpositive_wh(value):
    if value.ndim < 2 or value.shape[-1] < 4:
        return None
    return int((value[..., 2:4] <= 0).sum().item())


class BBoxLossProbe:
    def __init__(self, failure_dir, require_finite, save_data, loss_utils=None):
        self.failure_dir = Path(failure_dir)
        self.require_finite, self.save_data = require_finite, save_data
        self.loss_utils = loss_utils
        self.active = False
        self.iteration = None
        self.geometry = []
        self.operands = None
        self.reduction = None
        self.failed = False

    def capture(self, stage, trigger, output=None):
        import torch

        if self.failed:
            raise RuntimeError("The first ROI bbox failure was already captured")
        self.failed = True
        pred, target, weight = (self.operands[key] for key in ("pred", "target", "weight"))
        reduction = self.reduction
        per_element = reduction["per_element"] if reduction is not None else None
        avg_factor = (reduction["avg_factor"] if reduction is not None
                      else self.operands["avg_factor"])
        avg_scalar = avg_factor.item() if isinstance(avg_factor, torch.Tensor) else avg_factor
        effective_reduction = (reduction["reduction"] if reduction is not None
                               else self.operands["reduction_override"] or self.config["reduction"])
        info = {
            "schema": "iraod-roi-bbox-loss-operands-v1", "diagnostic_only": True,
            "stage": stage, "trigger": trigger,
            "runner_iter_zero_based": self.iteration, "update_one_based": self.iteration + 1,
            "positive_count": pred.shape[0], "pred": tensor_info(pred),
            "encoded_targets": tensor_info(target), "weights": tensor_info(weight),
            "avg_factor_repr": repr(avg_scalar),
            "avg_factor_isfinite": math.isfinite(float(avg_scalar)) if avg_scalar is not None else None,
            "reduction": effective_reduction,
            "denominator_expression": "avg_factor + torch.finfo(torch.float32).eps when supplied and mean",
            "denominator_epsilon": torch.finfo(torch.float32).eps,
            "per_element_loss": tensor_info(per_element), "output": tensor_info(output),
            "geometry_positive_count": sum(row["proposals"].shape[0] for row in self.geometry),
            "geometry": [{
                "image_index": i, "proposals": tensor_info(row["proposals"]),
                "gt_boxes": tensor_info(row["gt_boxes"]),
                "proposal_nonpositive_wh": nonpositive_wh(row["proposals"]),
                "gt_nonpositive_wh": nonpositive_wh(row["gt_boxes"]),
                "encode_called": row["encode_called"],
                "native_encoded_targets": tensor_info(row["encoded_targets"]),
            } for i, row in enumerate(self.geometry)],
            "configuration": self.config, "sources": self.sources,
            "capture_status": "pending", "operand_file": "bbox_loss_operands.pt",
            "not_a_model_or_resume_checkpoint": True,
        }
        self.failure_dir.mkdir(parents=True, exist_ok=False)
        metadata = self.failure_dir / "failure.json"
        metadata.write_text(json.dumps(info, indent=2, allow_nan=False) + "\n")
        temporary = self.failure_dir / "bbox_loss_operands.pt.partial"
        self.save_data({
            "schema": "iraod-roi-bbox-loss-operands-v1", "diagnostic_only": True,
            "not_a_model_or_resume_checkpoint": True,
            "pred": cpu_tensor(pred), "target": cpu_tensor(target), "weight": cpu_tensor(weight),
            "avg_factor": cpu_tensor(avg_factor) if isinstance(avg_factor, torch.Tensor) else avg_factor,
            "reduction": effective_reduction, "configuration": self.config,
            "per_element_loss": cpu_tensor(per_element), "output": cpu_tensor(output),
            "positive_geometry": [{
                "image_index": i, "proposals": cpu_tensor(row["proposals"]),
                "gt_boxes": cpu_tensor(row["gt_boxes"]), "gt_labels": cpu_tensor(row["gt_labels"]),
                "encoded_targets": cpu_tensor(row["encoded_targets"]),
                "encode_called": row["encode_called"],
            } for i, row in enumerate(self.geometry)],
        }, temporary)
        temporary.replace(self.failure_dir / "bbox_loss_operands.pt")
        info["capture_status"] = "complete"
        metadata.write_text(json.dumps(info, indent=2, allow_nan=False) + "\n")
        print("NONFINITE_BBOX_OPERANDS " + json.dumps(info, allow_nan=False), flush=True)

    @contextmanager
    def install(self, runner_class):
        from mmcv.parallel import is_module_wrapper

        original_hook = runner_class.call_hook
        with ExitStack() as stack:
            def attach(runner):
                model = runner.model.module if is_module_wrapper(runner.model) else runner.model
                head = model.roi_head.bbox_head
                criterion = head.loss_bbox
                utils = self.loss_utils or importlib.import_module("mmdet.models.losses.utils")
                original_target = head._get_target_single
                original_loss = criterion.forward
                original_reduce = utils.weight_reduce_loss
                signature = inspect.signature(original_loss)
                coder = head.bbox_coder
                self.config = {
                    "criterion": type(criterion).__name__, "beta": criterion.beta,
                    "loss_weight": criterion.loss_weight, "reduction": criterion.reduction,
                    "reg_class_agnostic": head.reg_class_agnostic,
                    "reg_decoded_bbox": head.reg_decoded_bbox,
                    "coder": type(coder).__name__, "means": list(coder.means), "stds": list(coder.stds),
                    "angle_range": coder.angle_range, "norm_factor": coder.norm_factor,
                    "edge_swap": coder.edge_swap, "proj_xy": coder.proj_xy,
                }
                self.sources = {
                    "target": inspect.getsourcefile(original_target),
                    "criterion": inspect.getsourcefile(original_loss),
                    "reduction": inspect.getsourcefile(original_reduce),
                }

                def targets(pos_bboxes, neg_bboxes, pos_gt_bboxes, pos_gt_labels, cfg):
                    result = original_target(pos_bboxes, neg_bboxes, pos_gt_bboxes, pos_gt_labels, cfg)
                    self.geometry.append({
                        "proposals": pos_bboxes, "gt_boxes": pos_gt_bboxes, "gt_labels": pos_gt_labels,
                        "encoded_targets": result[2][:pos_bboxes.shape[0]],
                        "encode_called": bool(pos_bboxes.shape[0]) and not head.reg_decoded_bbox,
                    })
                    return result

                def reduce(loss, weight=None, reduction="mean", avg_factor=None):
                    if not self.active:
                        return original_reduce(loss, weight, reduction, avg_factor)
                    self.reduction = {"per_element": loss, "reduction": reduction, "avg_factor": avg_factor}
                    tensors = {"pred": self.operands["pred"], "target": self.operands["target"],
                               "weight": weight, "per_element_loss": loss}
                    try:
                        self.require_finite(tensors, "ROI bbox before weight/reduction")
                    except FloatingPointError as error:
                        self.capture("before_bbox_weight_reduction", str(error))
                        raise
                    result = original_reduce(loss, weight, reduction, avg_factor)
                    try:
                        self.require_finite(result, "ROI bbox reduction output")
                    except FloatingPointError as error:
                        self.capture("bbox_reduction_output", str(error), result)
                        raise
                    return result

                def loss(*args, **kwargs):
                    bound = signature.bind(*args, **kwargs)
                    bound.apply_defaults()
                    self.operands = bound.arguments
                    self.reduction = None
                    self.active = True
                    try:
                        result = original_loss(*args, **kwargs)
                        try:
                            self.require_finite(result, "ROI bbox criterion output")
                        except FloatingPointError as error:
                            self.capture("bbox_criterion_output", str(error), result)
                            raise
                        return result
                    finally:
                        self.active = False
                        self.operands = self.reduction = None

                stack.enter_context(patch.object(head, "_get_target_single", targets))
                stack.enter_context(patch.object(criterion, "forward", loss))
                stack.enter_context(patch.object(utils, "weight_reduce_loss", reduce))

            def call_hook(runner, name):
                result = original_hook(runner, name)
                if name == "before_run":
                    attach(runner)
                elif name == "before_train_iter":
                    self.iteration = runner.iter
                    self.geometry = []
                elif name == "after_train_iter":
                    self.geometry = []
                return result

            stack.enter_context(patch.object(runner_class, "call_hook", call_hook))
            try:
                yield self
            finally:
                self.geometry = []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("code-root", "finite-helper", "epoch-adapter", "failure-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("train_arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    code, helper = Path(args.code_root).resolve(), Path(args.finite_helper).resolve()
    epoch_path, failure = Path(args.epoch_adapter).resolve(), Path(args.failure_dir).resolve()
    arguments = args.train_arguments
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    sys.argv = [str(Path(__file__).resolve()), "--code-root", str(code), "--finite-helper", str(helper),
                "--epoch-adapter", str(epoch_path), "--failure-dir", str(failure), "--", *arguments]
    os.chdir(code)
    sys.path.insert(0, str(code))
    os.environ["PYTHONPATH"] = str(code)
    from iraod_runtime import ensure_iraod_runtime
    ensure_iraod_runtime()
    if failure.exists():
        raise FileExistsError(failure)
    spec = importlib.util.spec_from_file_location("_attempt8_epoch_adapter", epoch_path)
    epoch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(epoch)
    checks = epoch.load_finite_helper(helper)
    import torch

    probe = BBoxLossProbe(failure, checks.require_finite, torch.save)
    original_observer = epoch.observe_natural_epoch

    @contextmanager
    def observe_bbox(runner_class, emit):
        with original_observer(runner_class, emit) as state, probe.install(runner_class):
            yield state

    with patch.object(epoch, "load_finite_helper", return_value=checks), \
            patch.object(epoch, "observe_natural_epoch", observe_bbox), \
            patch.object(sys, "argv", [str(epoch_path), "--code-root", str(code),
                                      "--finite-helper", str(helper), "--", *arguments]):
        epoch.main()


if __name__ == "__main__":
    main()
