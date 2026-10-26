"""Side-by-side ground-truth and detection visualization helpers."""

import os.path as osp

import cv2
import mmcv
import numpy as np

from mmrotate.core import obb2poly_np


def _as_array(value):
    if value is None:
        return None
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _bbox_to_polygons(bboxes, version='le90'):
    bboxes = _as_array(bboxes)
    if bboxes is None or bboxes.size == 0:
        return np.zeros((0, 8), dtype=np.float32)
    bboxes = bboxes.reshape(-1, bboxes.shape[-1])
    if bboxes.shape[1] >= 5:
        # mmrotate's numpy converter expects a score column even when the
        # caller is converting score-less ground-truth OBBs.
        scores = np.zeros((bboxes.shape[0], 1), dtype=bboxes.dtype)
        return obb2poly_np(np.concatenate((bboxes[:, :5], scores), axis=1),
                           version=version)[:, :8]
    if bboxes.shape[1] >= 4:
        x1, y1, x2, y2 = bboxes[:, :4].T
        return np.stack((x1, y1, x2, y1, x2, y2, x1, y2), axis=1)
    raise ValueError(f'Unsupported bbox shape: {bboxes.shape}')


def _annotation_items(annotation, version='le90'):
    annotation = annotation or {}
    polygons = annotation.get('polygons')
    if polygons is None:
        polygons = _bbox_to_polygons(annotation.get('bboxes'), version)
    polygons = _as_array(polygons)
    if polygons is None:
        polygons = np.zeros((0, 8), dtype=np.float32)
    polygons = polygons.reshape(-1, 8)
    labels = _as_array(annotation.get('labels'))
    if labels is None:
        labels = np.zeros((0,), dtype=np.int64)
    return polygons, labels.reshape(-1)


def _prediction_items(result, version='le90', score_thr=0.3):
    if isinstance(result, tuple):
        result = result[0]
    if isinstance(result, dict):
        return
    if result is None:
        return
    for label, class_bboxes in enumerate(result):
        boxes = _as_array(class_bboxes)
        if boxes is None or boxes.size == 0:
            continue
        boxes = boxes.reshape(-1, boxes.shape[-1])
        if boxes.shape[1] >= 6:
            boxes = boxes[boxes[:, 5] >= score_thr]
            scores = boxes[:, 5]
            polygons = _bbox_to_polygons(boxes[:, :5], version)
        elif boxes.shape[1] == 5:
            boxes = boxes[boxes[:, 4] >= score_thr]
            scores = boxes[:, 4]
            polygons = _bbox_to_polygons(boxes[:, :4], version)
        elif boxes.shape[1] >= 4:
            scores = np.ones((boxes.shape[0],), dtype=np.float32)
            polygons = _bbox_to_polygons(boxes[:, :4], version)
        else:
            continue
        for polygon, score in zip(polygons, scores):
            yield polygon, label, float(score)


def _draw_polygon(image, polygon, color, thickness):
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] < 4 or not np.isfinite(points).all():
        return
    points = np.round(points).astype(np.int32)
    cv2.polylines(image, [points], isClosed=True, color=color,
                  thickness=thickness, lineType=cv2.LINE_AA)


def _draw_text(image, text, polygon, color):
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    if points.size == 0 or not np.isfinite(points).all():
        return
    x = max(0, int(round(points[:, 0].min())))
    y = max(14, int(round(points[:, 1].min())))
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                color, 1, cv2.LINE_AA)


def imshow_gt_pred_side_by_side(img,
                                annotation,
                                result,
                                class_names=None,
                                score_thr=0.3,
                                version='le90',
                                thickness=4,
                                show=False,
                                out_file=None):
    """Draw GT on the left and predictions on the right.

    Both panels use the same original image, so rotated boxes can be compared
    without the overlap that occurs when drawing GT and predictions together.
    """
    image = mmcv.imread(img) if isinstance(img, str) else np.asarray(img)
    if image is None:
        raise FileNotFoundError(f'Cannot read image: {img}')
    image = np.ascontiguousarray(image.copy())
    gt_panel = image.copy()
    pred_panel = image.copy()
    class_names = class_names or ()

    gt_polygons, gt_labels = _annotation_items(annotation, version)
    for polygon, label in zip(gt_polygons, gt_labels):
        _draw_polygon(gt_panel, polygon, (0, 210, 0), thickness)
        label = int(label)
        if 0 <= label < len(class_names):
            _draw_text(gt_panel, str(class_names[label]), polygon,
                       (0, 150, 0))

    for polygon, label, score in _prediction_items(result, version, score_thr):
        _draw_polygon(pred_panel, polygon, (0, 0, 255), thickness)
        text = (str(class_names[label])
                if 0 <= label < len(class_names) else str(label))
        if score < 1.0:
            text = f'{text}:{score:.2f}'
        _draw_text(pred_panel, text, polygon, (0, 0, 220))

    combined = np.concatenate((gt_panel, pred_panel), axis=1)
    if show:
        cv2.imshow('ground truth | prediction', combined)
        cv2.waitKey(0)
    if out_file is not None:
        mmcv.mkdir_or_exist(osp.dirname(out_file))
        mmcv.imwrite(combined, out_file)
    return combined
