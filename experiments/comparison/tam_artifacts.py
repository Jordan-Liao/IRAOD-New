"""TAM completion/identity boundary; encoder weights remain an external resource."""

from pathlib import Path

import torch


SCHEMA = "iraod-target-augmentation-v1"
FORMAL_STEPS = 160000
BGR_MEAN = (102.9801, 115.9465, 122.7717)


def checkpoint_payload(module, identity, encoder_weights, steps, code_sha, image_manifest):
    components = {
        name: {key: value.detach().cpu() for key, value in getattr(module, name).state_dict().items()}
        for name in ("decoder", "F1", "F2")
    }
    if any(not torch.isfinite(value).all() for state in components.values() for value in state.values()):
        raise FloatingPointError("Cannot publish nonfinite trained TAM parameters")
    return {
        "schema": SCHEMA,
        "status": "complete" if steps == FORMAL_STEPS else "smoke_not_formal",
        "training_steps": steps,
        "identity": identity,
        "training_code_sha": code_sha,
        "encoder_weights": str(Path(encoder_weights).resolve()),
        "image_manifest": str(Path(image_manifest).resolve()),
        "normalization": {"order": "BGR", "mean": list(BGR_MEAN), "input_range": "0..255"},
        "objective": "generic_aerial_code_formula; decoder_then_F1_F2; alpha_train1",
        "components": components,
    }


def load_completed_tam(path, identity, device):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (payload["schema"] != SCHEMA or payload["status"] != "complete"
            or payload["training_steps"] != FORMAL_STEPS or payload["identity"] != identity):
        raise ValueError("Detector adaptation requires a completed TAM for this exact dataset/domain/seed")
    expected = {"order": "BGR", "mean": list(BGR_MEAN), "input_range": "0..255"}
    if payload["normalization"] != expected:
        raise ValueError("TAM checkpoint preprocessing differs from the declared BGR convention")
    from sfod.extensions.tam import TargetAugmentationModule

    model = TargetAugmentationModule(weights_path=payload["encoder_weights"])
    for name in ("decoder", "F1", "F2"):
        getattr(model, name).load_state_dict(payload["components"][name], strict=True)
    return model.to(device).freeze_for_inference()
