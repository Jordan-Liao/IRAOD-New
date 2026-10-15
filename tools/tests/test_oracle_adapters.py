"""Formal oracle metadata admission never upgrades a partial or wrong-dataset adapter."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch
from torch import nn

from experiments.comparison.labels import CLASSES, ORACLE_TRAINING_TEMPLATES
from experiments.comparison.oracle_adapters import inspect_oracle_adapter, attach_oracle_adapter
from experiments.comparison.result_completion import DOMAINS
from sarclip_adapter import ADAPTER_FORMAT, inject_lora, mark_lora_trainable, trainable_state_dict


class TinyClip(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.Sequential(nn.Linear(4, 4))
        self.text = nn.Linear(4, 4)
        self.logit_scale = nn.Parameter(torch.tensor(0.))


class OracleAdapterTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / "base.safetensors"
        self.base.touch()
        self.path = self.root / "lora.pth"
        model = TinyClip()
        inject_lora(model)
        names = mark_lora_trainable(model)
        state = trainable_state_dict(model)
        count = len(DOMAINS["RSAR"]) - 1
        self.payload = {
            "format": ADAPTER_FORMAT, "adapter_type": "lora", "dataset": "RSAR", "seed": 42,
            "train_only": True, "split": "train", "status": "complete",
            "epochs": 10, "epochs_requested": 10, "epochs_completed": 10,
            "selection": "final_epoch", "selected_epoch": 10, "batch_size": 64,
            "optimizer": "AdamW", "lr": 1e-4, "weight_decay": 1e-4, "precision": "fp32",
            "crop_mode": "aabb", "force_rgb": True, "drop_last": False,
            "sarclip_model": "ViT-B-32", "classes": list(CLASSES["RSAR"]),
            "templates": [ORACLE_TRAINING_TEMPLATES["RSAR"]],
            "sarclip_pretrained": str(self.base), "max_patches": None,
            "training_git_sha": "fixture-producer", "metadata_row_count": count,
            "class_counts": {name: count if i == 0 else 0 for i, name in enumerate(CLASSES["RSAR"])},
            "crop_modes": {"aabb": count}, "crop_expansions": {"0.4": count},
            "corruptions": {name: 1 for name in DOMAINS["RSAR"][1:]},
            "sampler": {"type": "inverse_class_weighted", "replacement": True,
                        "num_samples": count, "seed": 42},
            "lora": {"r": 8, "alpha": 16., "dropout": 0., "target_prefixes": ["visual"]},
            "trainable_names": names, "state_dict": state,
            "trainable_numel": sum(p.numel() for p in state.values()),
        }

    def save(self, payload=None):
        torch.save(self.payload if payload is None else payload, self.path)

    def test_completed_matching_adapter_attaches_and_freezes_without_text_change(self):
        self.save()
        clip = TinyClip().requires_grad_(False).eval()
        before = {name: value.clone() for name, value in clip.text.state_dict().items()}
        scorer = SimpleNamespace(backend="sarclip", strict=True, class_names=list(CLASSES["RSAR"]),
                                 clip=clip, device=torch.device("cpu"))
        self.assertIs(attach_oracle_adapter(scorer, self.path, "RSAR", self.base), scorer)
        self.assertTrue(all(not p.requires_grad for p in scorer.clip.parameters()))
        self.assertFalse(scorer.clip.training)
        for name, value in scorer.clip.text.state_dict().items():
            torch.testing.assert_close(value, before[name])
        self.assertEqual(scorer.oracle_evidence["selected_epoch"], 10)

    def test_wrong_dataset_partial_visual_projection_or_modified_recipe_are_rejected(self):
        for key, value in (
                ("dataset", "DIOR"), ("status", "running"), ("epochs_completed", 9),
                ("adapter_type", "visual_proj"), ("split", "test"), ("seed", 43),
                ("max_patches", 100), ("templates", ["different {}"]),
                ("corruptions", {"clean": 7})):
            with self.subTest(key=key):
                self.save({**self.payload, key: value})
                with self.assertRaises(ValueError):
                    inspect_oracle_adapter(self.path, "RSAR", self.base)

    def test_incomplete_or_nonfinite_factor_state_cannot_count_as_lora(self):
        incomplete = deepcopy(self.payload)
        incomplete["state_dict"].pop(next(k for k in incomplete["state_dict"] if ".lora_up." in k))
        self.save(incomplete)
        with self.assertRaisesRegex(ValueError, "LoRA factors"):
            inspect_oracle_adapter(self.path, "RSAR", self.base)
        invalid = deepcopy(self.payload)
        next(v for k, v in invalid["state_dict"].items() if ".lora_up." in k).fill_(float("nan"))
        self.save(invalid)
        with self.assertRaisesRegex(ValueError, "LoRA factors"):
            inspect_oracle_adapter(self.path, "RSAR", self.base)


if __name__ == "__main__":
    unittest.main()
