"""Visualize DOTA ground truth and rotated detector predictions side by side.

The regular ``test.py --show-dir`` path only receives model predictions and
therefore cannot draw ground truth.  This script keeps the same test dataset
and checkpoint loading path, then places two panels next to each other:

* left panel: ground-truth polygons in green
* right panel: predictions above ``--score-thr`` in red
"""

import os.path as osp
import sys

_REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from iraod_runtime import ensure_iraod_runtime

ensure_iraod_runtime()

import argparse

import cv2
import mmcv
import numpy as np
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.datasets import build_dataloader, replace_ImageToTensor
from mmrotate.core import obb2poly_np
from mmrotate.datasets import build_dataset
from mmrotate.models import build_detector
from mmrotate.utils import compat_cfg, setup_multi_processes

from sfod.utils import patch_config


def parse_args():
    parser = argparse.ArgumentParser(
        description='Render DOTA GT and rotated detections side by side.')
    parser.add_argument('config', help='test config file')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out-dir', required=True,
                        help='directory for side-by-side images')
    parser.add_argument('--score-thr', type=float, default=0.3,
                        help='minimum prediction score to draw')
    parser.add_argument('--line-width', type=int, default=4,
                        help='bbox line width; 4 matches test.py')
    parser.add_argument('--max-images', type=int, default=0,
                        help='maximum images to render; 0 means all')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                        help='config overrides in key=value form')
    return parser.parse_args()


def _unwrap(value):
    """Unwrap the first DataContainer/list layer used by MMDetection."""
    while True:
        if hasattr(value, 'data'):
            value = value.data
            continue
        if isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
            continue
        break
    return value


def _draw_polygon(image, polygon, color, thickness=2):
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] < 4 or not np.isfinite(points).all():
        return
    points = np.round(points).astype(np.int32)
    cv2.polylines(image, [points], isClosed=True, color=color,
                  thickness=thickness, lineType=cv2.LINE_AA)


def _draw_label(image, text, polygon, color):
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    x, y = np.round(points[0]).astype(np.int32)
    x = max(0, int(x))
    y = max(14, int(y))
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                color, 1, cv2.LINE_AA)


def _prediction_items(result, class_names, score_thr):
    """Yield (polygon, label, score) from MMRotate per-class results."""
    if isinstance(result, tuple):
        result = result[0]
    for label, boxes in enumerate(result):
        boxes = np.asarray(boxes)
        if boxes.size == 0:
            continue
        boxes = boxes.reshape(-1, boxes.shape[-1])
        if boxes.shape[1] < 6:
            continue
        boxes = boxes[boxes[:, 5] >= score_thr]
        if boxes.shape[0] == 0:
            continue
        polygons = obb2poly_np(boxes[:, :6], version='le90')[:, :8]
        for polygon, box in zip(polygons, boxes):
            yield polygon, class_names[label], float(box[5])


def main():
    args = parse_args()
    if args.score_thr < 0 or args.score_thr > 1:
        raise ValueError('--score-thr must be in [0, 1]')
    if args.line_width <= 0:
        raise ValueError('--line-width must be positive')
    if args.max_images < 0:
        raise ValueError('--max-images must be non-negative')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg = compat_cfg(cfg)
    setup_multi_processes(cfg)
    cfg.gpu_ids = range(1)
    cfg = patch_config(cfg)

    if not isinstance(cfg.data.test, dict):
        raise TypeError('This visualization script expects data.test to be a dict')
    cfg.data.test.test_mode = True
    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
    if samples_per_gpu > 1:
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    dataset = build_dataset(cfg.data.test)
    # Keep one image per batch so annotation indices and image metadata align.
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.get('workers_per_gpu', 2),
        dist=False,
        shuffle=False)

    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16', None) is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    model = MMDataParallel(model, device_ids=cfg.gpu_ids)
    model.eval()

    mmcv.mkdir_or_exist(args.out_dir)
    total = len(dataset)
    limit = total if args.max_images == 0 else min(args.max_images, total)
    progress = mmcv.ProgressBar(limit)
    for index, data in enumerate(data_loader):
        if index >= limit:
            break
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)[0]

        img_metas = _unwrap(data['img_metas'])
        img_meta = img_metas[0] if isinstance(img_metas, (list, tuple)) else img_metas
        filename = img_meta.get('filename')
        if not filename or not osp.isfile(filename):
            filename = osp.join(dataset.img_prefix, dataset.data_infos[index]['filename'])
        image = mmcv.imread(filename)
        if image is None:
            raise FileNotFoundError(f'Cannot read image: {filename}')

        ann = dataset.get_ann_info(index)
        gt_polygons = ann.get('polygons', np.zeros((0, 8), dtype=np.float32))
        gt_labels = ann.get('labels', np.zeros((0,), dtype=np.int64))
        gt_panel = image.copy()
        prediction_panel = image.copy()

        for polygon, label in zip(gt_polygons, gt_labels):
            _draw_polygon(gt_panel, polygon, (0, 210, 0),
                          thickness=args.line_width)
            _draw_label(gt_panel, dataset.CLASSES[int(label)], polygon,
                        (0, 150, 0))

        for polygon, label, score in _prediction_items(
                result, dataset.CLASSES, args.score_thr):
            _draw_polygon(prediction_panel, polygon, (0, 0, 255),
                          thickness=args.line_width)
            _draw_label(prediction_panel, f'{label}:{score:.2f}', polygon,
                        (0, 0, 220))

        image = np.concatenate([gt_panel, prediction_panel], axis=1)

        out_name = osp.basename(img_meta.get(
            'ori_filename', dataset.data_infos[index]['filename']))
        out_file = osp.join(args.out_dir, out_name)
        mmcv.imwrite(image, out_file)
        progress.update()

    print(f'Wrote {limit} GT+prediction visualizations to {args.out_dir}')


if __name__ == '__main__':
    main()
