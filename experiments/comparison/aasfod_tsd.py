"""Explicit GPU prerequisite for an already CPU-prepared AASFOD queue; no GT IO."""

import argparse
import os
from pathlib import Path
import sys

from experiments.comparison.aasfod_protocol import TSD_CHOICE, validate_split
from experiments.comparison.extension_training import load_cell
from experiments.comparison.result_completion import write_json
from experiments.comparison import host_binding as host


def _no_flip_metadata(results):
    """Supply Collect's geometry metadata without flipping or consuming RNG."""
    results.update(flip=False, flip_direction=None)
    return results


def run(queue, dataset, domain, seed, smoke_images=None):
    if os.environ.get("IRAOD_GPU_LOCKED") != "1":
        raise RuntimeError("TSD must run under the existing owner's shared GPU lock")
    runtime, cell = load_cell(queue, dataset, domain, seed, "AASFOD")
    code = Path(cell.get("training_code", runtime["training_code"]))
    output = Path(cell["tsd_split"])
    if smoke_images is not None:
        if smoke_images < 5:
            raise ValueError("TSD smoke needs at least five images for the20% split")
        output = output.with_name("tsd_smoke.json")
    if output.exists():
        raise FileExistsError(output)
    sys.path.insert(0, str(code))
    os.chdir(code)
    # Native dependencies are deliberately lazy: queue preparation needs none.
    import torch
    from mmcv import Config
    from mmcv.parallel import collate, scatter
    from mmcv.runner import load_checkpoint
    from mmdet.apis import set_random_seed
    from mmrotate.core import rbbox2roi
    from mmrotate.models import build_detector
    from sfod.semi_dota_dataset import StrictSourceFreeDOTADataset
    from sfod.extensions.aasfod_mechanisms import aligned_tsd, high_variance_split

    set_random_seed(seed, deterministic=True)
    with host.native_config_paths():
        config = Config.fromfile(cell["config"])
        # Unmodified architecture for the common source; no adaptation discriminators.
        teacher_cfg = Config.fromfile(config.model.ema_config)
    detector = build_detector(teacher_cfg.model)
    load_checkpoint(detector, cell["source_checkpoint"], map_location="cpu")
    detector.cuda().requires_grad_(False).eval()
    data = dict(config.data.train)
    for key in ("type", "stage", "tsd_split"):
        data.pop(key)
    data.update(img_prefix=cell["target_val"],
                unlabeled_epoch_size=cell["unlabeled_epoch_size"],
                pipeline_share=[dict(type="LoadImageFromFile"), _no_flip_metadata],
                pipeline_strong=[])
    target = StrictSourceFreeDOTADataset(**data)
    scores = {}
    limit = len(target) if smoke_images is None else min(smoke_images, len(target))
    for index in range(limit):
        sample = target[index]
        # Only deterministic weak view is consumed; no randomized flip/BN noise.
        sample = {key: sample[key] for key in ("img", "img_metas")}
        batch = scatter(collate([sample], samples_per_gpu=1), [0])[0]
        score = aligned_tsd(detector, batch["img"], batch["img_metas"], rbbox2roi)
        scores[target.filenames[target.indices[index]]] = float(score.cpu())
    similar, dissimilar = high_variance_split(scores)
    result = {
        "identity": {key: cell[key] for key in
                     ("dataset", "domain", "seed", "source_checkpoint", "target_val")},
        "choice": TSD_CHOICE, "scores": scores, "similar": similar, "dissimilar": dissimilar,
        "status": "complete" if smoke_images is None else "smoke_not_formal",
        "images": limit, "stochastic_roi_passes": 20 * limit,
    }
    if smoke_images is None:
        validate_split(result, cell)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("queue", "dataset", "domain"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--seed", required=True, type=int, choices=(42, 43, 44))
    parser.add_argument("--smoke-images", type=int)
    print(run(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
