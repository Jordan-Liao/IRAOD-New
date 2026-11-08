"""Detector and CGA boundary; no change to legacy Oriented R-CNN classes."""
import os
from contextlib import contextmanager
import torch
from mmrotate.core import rbbox2result
from mmrotate.models.builder import ROTATED_DETECTORS
from mmrotate.models.detectors.single_stage import RotatedSingleStageDetector
from sfod.cga import TestMixins
from .geometry import to_original, from_original, scale_factors


@contextmanager
def scorer_environment(values):
    previous = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@ROTATED_DETECTORS.register_module()
class OrthoRotatedDETR(RotatedSingleStageDetector, TestMixins):
    def __init__(self, *args, classes=None, cga_env=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.CLASSES = tuple(classes) if classes is not None else None
        self.cga_env = dict(cga_env or {})
        if classes is not None and len(classes) != self.bbox_head.num_classes:
            raise ValueError('Class names do not match DETR head')

    def _build_cga(self, num_classes):
        with scorer_environment(self.cga_env):
            return TestMixins._build_cga(self, num_classes)

    def set_epoch(self, epoch):
        self.cur_epoch = epoch

    def _predictions(self, img, img_metas):
        for meta in img_metas:
            meta['batch_input_shape'] = tuple(img.shape[-2:])
        features = self.extract_feat(img)
        return self.bbox_head.get_bboxes(*self.bbox_head(features, img_metas),
                                         img_metas, rescale=False)

    @torch.no_grad()
    def predict_original(self, img, img_metas, with_cga=False):
        results = []
        for (boxes, labels), meta in zip(self._predictions(img, img_metas), img_metas):
            boxes = torch.cat((to_original(boxes[:, :5], meta), boxes[:, 5:]), -1)
            results.append(rbbox2result(boxes, labels, self.bbox_head.num_classes))
        if with_cga:
            with scorer_environment(self.cga_env):
                self.refine_test(results, img_metas)
        return results

    def simple_test(self, img, img_metas, rescale=False, with_cga=False, **kwargs):
        if with_cga:
            # CGA always sees the original image coordinates, including when
            # inference was performed on a resized/flipped training view.
            results = self.predict_original(img, img_metas, with_cga=True)
            for image_results, meta in zip(results, img_metas):
                for result in image_results:
                    boxes = torch.as_tensor(result[:, :5])
                    boxes = from_original(boxes, meta)
                    if rescale:
                        boxes = boxes / scale_factors(meta, boxes)
                    result[:, :5] = boxes.numpy()
            return results
        return [rbbox2result(boxes.detach(), labels, self.bbox_head.num_classes)
                for boxes, labels in self.bbox_head.get_bboxes(
                    *self.bbox_head(self.extract_feat_with_meta(img, img_metas), img_metas),
                    img_metas, rescale=rescale)]

    def extract_feat_with_meta(self, img, img_metas):
        for meta in img_metas:
            meta['batch_input_shape'] = tuple(img.shape[-2:])
        return self.extract_feat(img)
