"""Save predictions and image order captured from the same inference batches."""

from pathlib import Path
import subprocess

IMAGE_ORDER_SCHEMA = "iraod-prediction-image-order-v1"
IMAGE_ORDER_ORIGIN = "inference_batch_img_metas"


def capture_image_records(data, batch_size):
    metas = [meta for group in data["img_metas"][0].data for meta in group]
    if len(metas) != batch_size:
        raise ValueError("Prediction batch size differs from its inference image metadata")
    return [{
        "image_id": Path(meta["ori_filename"]).stem,
        "ori_filename": meta["ori_filename"],
        "filename": meta["filename"],
    } for meta in metas]


def new_prediction_paths(out):
    path = Path(out)
    sidecar = Path(str(path) + ".image_ids.json")
    temporary = Path(str(sidecar) + ".partial")
    for artifact in (path, sidecar, temporary):
        if artifact.exists():
            raise FileExistsError(
                f"Use a fresh prediction output; cannot overwrite or backfill identity: {artifact}")
    return path, sidecar, temporary


def save_predictions_with_ids(out, outputs, image_records, metadata):
    import mmcv

    path, sidecar, temporary = new_prediction_paths(out)
    if len(outputs) != len(image_records) or len(outputs) != metadata["dataset_size"]:
        raise ValueError("Predictions and captured image order do not cover the same dataset")
    ids = [record["image_id"] for record in image_records]
    if any(not image_id for image_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("Captured prediction image IDs must be unique and nonempty")
    payload = {
        **metadata,
        "schema": IMAGE_ORDER_SCHEMA, "origin": IMAGE_ORDER_ORIGIN,
        "status": "complete", "predictions_file": path.name,
        "n_images": len(outputs), "image_ids": ids,
        "records": [{**record, "prediction_index": i}
                    for i, record in enumerate(image_records)],
        "evaluation_code_sha": subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[1]),
             "rev-parse", "HEAD"], text=True).strip(),
        "training_code_sha": metadata.get("training_code_sha"),
    }
    # Preserve the original prediction object/serialization; publish IDs only after it is saved.
    mmcv.dump(outputs, str(path))
    mmcv.dump(payload, str(temporary), file_format="json")
    temporary.replace(sidecar)
    return sidecar
