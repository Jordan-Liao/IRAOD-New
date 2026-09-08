"""Read-only finite checks at future native loss/step/save/eval boundaries.

These checks diagnose the boundary, not the first arithmetic cause. They never
repair tensors, skip updates, flush EMA, or alter optimizer/config state.
"""

from collections.abc import Mapping
from contextlib import contextmanager
from functools import wraps

import torch
from torch.optim.optimizer import register_optimizer_step_pre_hook


def _tensors(value, path):
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _tensors(item, f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _tensors(item, f"{path}[{index}]")


def _require_finite(named_tensors, boundary):
    # One host synchronization per device on the finite path, not per parameter.
    checks = {}
    for name, tensor in named_tensors:
        if tensor.is_floating_point() or tensor.is_complex():
            checks.setdefault(tensor.device, []).append(
                (name, torch.isfinite(tensor.detach()).all()))
    for device_checks in checks.values():
        if not torch.stack([finite for _, finite in device_checks]).all().item():
            name = next(name for name, finite in device_checks if not finite.item())
            raise FloatingPointError(f"Nonfinite {boundary}: {name}")


def require_finite(value, boundary):
    """Inspect floating tensors in the actual payload without copying/mutating it."""
    _require_finite(_tensors(value, boundary), boundary)


def _before_step(optimizer, args, kwargs):
    _require_finite(
        ((f"param_groups[{group_index}].params[{index}].grad", parameter.grad)
         for group_index, group in enumerate(optimizer.param_groups)
         for index, parameter in enumerate(group["params"])
         if parameter.grad is not None),
        "gradient before optimizer.step")


@contextmanager
def optimizer_boundary():
    """Observe real SGD/Adam steps, including steps wrapped by the smoke observer."""
    handle = register_optimizer_step_pre_hook(_before_step)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def native_boundaries(*, evaluate=False):
    """Scope MMCV instrumentation to one original native entrypoint execution.

    Training uses the existing OptimizerHook implementation unchanged. MMCV
    serializes the assembled student/EMA + optimizer payload into BytesIO before
    FileClient.put; checking torch.save therefore precedes artifact publication.
    Evaluation checks the model *after* MMCV's actual load/key translation.
    Install before runpy imports native entrypoint aliases, not into snapshots.
    """
    from mmcv.runner import OptimizerHook
    from mmcv.runner import checkpoint

    if evaluate:
        original_load = checkpoint.load_state_dict

        @wraps(original_load)
        def load_state_dict(module, *args, **kwargs):
            result = original_load(module, *args, **kwargs)
            _require_finite(
                (*module.named_parameters(), *module.named_buffers()),
                "loaded model before evaluation")
            return result

        checkpoint.load_state_dict = load_state_dict
        try:
            yield
        finally:
            checkpoint.load_state_dict = original_load
        return

    original_iter = OptimizerHook.after_train_iter
    original_save = torch.save

    @wraps(original_iter)
    def after_train_iter(hook, runner):
        require_finite(runner.outputs["loss"],
                       f"loss before backward (iteration {runner.iter})")
        return original_iter(hook, runner)

    @wraps(original_save)
    def save(payload, *args, **kwargs):
        require_finite(payload, "checkpoint before publication")
        return original_save(payload, *args, **kwargs)

    OptimizerHook.after_train_iter = after_train_iter
    torch.save = save
    try:
        with optimizer_boundary():
            yield
    finally:
        torch.save = original_save
        OptimizerHook.after_train_iter = original_iter
