"""Admit MDP's training-only auxiliaries without weakening detector loading."""

from unittest.mock import patch


def auxiliary_shapes(detector_state):
    c2 = detector_state["backbone.layer1.0.bn3.weight"].numel()
    c4 = detector_state["backbone.layer3.0.bn3.weight"].numel()
    classes = detector_state["roi_head.bbox_head.fc_cls.weight"].shape[0]
    shapes = {
        "mdp.afsp.style.0.weight": (c2 // 16, 2 * c2, 1, 1),
        "mdp.afsp.style.2.weight": (2 * c2, c2 // 16, 1, 1),
        "mdp.transform.0.weight": (c4, c4, 3, 3),
        "mdp.transform.0.bias": (c4,),
        "mdp.transform.1.weight": (c4,),
        "mdp.transform.1.bias": (c4,),
        "mdp.transform.1.running_mean": (c4,),
        "mdp.transform.1.running_var": (c4,),
        "mdp.transform.1.num_batches_tracked": (),
        "mdp.history.previous_teacher": (classes, c4),
        "mdp.history.previous_student": (classes, c4),
    }
    return shapes


def detector_state_for_mdp(state, detector_state, role):
    import torch

    if role not in ("student", "ema"):
        raise ValueError("MDP checkpoint role must be explicit Student or EMA")
    expected_aux = auxiliary_shapes(detector_state) if role == "student" else {}
    missing = detector_state.keys() - state.keys()
    extra = state.keys() - detector_state.keys()
    if missing or extra != expected_aux.keys():
        raise ValueError(
            f"MDP checkpoint key mismatch: missing_core={sorted(missing)}, "
            f"actual_extra={sorted(extra)}, expected_extra={sorted(expected_aux)}")
    for key, reference in detector_state.items():
        value = state[key]
        if (not torch.is_tensor(value) or value.shape != reference.shape
                or value.dtype != reference.dtype):
            raise ValueError(f"MDP detector checkpoint shape/dtype mismatch: {key}")
    feature_dtype = detector_state["backbone.layer3.0.bn3.weight"].dtype
    for key, shape in expected_aux.items():
        value = state[key]
        dtype = torch.int64 if key.endswith("num_batches_tracked") else feature_dtype
        if (not torch.is_tensor(value) or tuple(value.shape) != shape or value.dtype != dtype):
            raise ValueError(f"MDP auxiliary checkpoint shape/dtype mismatch: {key}")
        if not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite MDP training auxiliary: {key}")
    core = {key: state[key] for key in detector_state}
    proof = {
        "method": "MDP", "role": role, "strict_detector_load": True,
        "detector_keys": len(core), "auxiliary_keys": sorted(expected_aux),
        "auxiliary_reason": "MDPOBB ordinary evaluation bypasses all mdp training auxiliaries",
    }
    return core, proof


def load_mdp_checkpoint(loader, role, *positional, **keywords):
    from mmcv.runner import checkpoint as checkpoint_module

    original_state_load = checkpoint_module.load_state_dict
    admission = {}

    def strict_load(module, state, strict=False, logger=None):
        core, proof = detector_state_for_mdp(state, module.state_dict(), role)
        admission.update(proof)
        return original_state_load(module, core, strict=True, logger=logger)

    with patch.object(checkpoint_module, "load_state_dict", strict_load):
        checkpoint = loader(*positional, **keywords)
    if not admission:
        raise RuntimeError("Native loader did not execute the strict MDP state-load boundary")
    return checkpoint, admission
