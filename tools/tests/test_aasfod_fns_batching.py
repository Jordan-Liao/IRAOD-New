"""Actual FNS method/mosaic on CPU; compiled detector ops must never execute."""

import copy
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from mmcv.parallel import DataContainer, collate
from mmcv.utils import ext_loader


def unavailable_op(*args, **kwargs):
    raise AssertionError("Compiled native ops are outside this CPU batching test")


def unused_ops(name, functions):
    return SimpleNamespace(**{function: unavailable_op for function in functions})


# Only unused compiled kernels are replaced, not Python model/mosaic/collation.
with patch.object(ext_loader, "load_ext", side_effect=unused_ops):
    from mmdet.core.anchor import AnchorGenerator
    from mmdet.models.dense_heads import AnchorHead
    from sfod.extensions.aasfod import AASFODOBB
    from sfod.extensions.aasfod_mosaic import four_image_mosaic


class Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.rpn_head = SimpleNamespace(simple_test_rpn=lambda features, metas: [])
        self.roi_head = SimpleNamespace(simple_test=self.predict)

    def extract_feat(self, images):
        self.originals = images
        return images

    def predict(self, features, proposals, metas, rescale):
        assert rescale
        self.metas = metas
        return [None] * len(metas)


class Port(AASFODOBB):
    def __init__(self):
        nn.Module.__init__(self)
        self.aasfod_stage = "fns"
        self.cur_iter = self.image_num = 0
        self.scale = nn.Parameter(torch.tensor(0.75))
        self.ema_model = Teacher()

    def create_pseudo_results(self, images, detections, transforms, device):
        return (
            [images.new_tensor([[12., 10., 18., 12., .4]]) for _ in detections],
            [torch.tensor([index], device=device) for index in range(len(detections))])

    def extract_feat(self, images):
        self.batch = images
        return images * self.scale

    def detection_losses(self, features, metas, boxes, labels):
        self.targets = metas, boxes, labels
        return {"loss": features.square().sum()}


class AASFODFNSBatchingTest(unittest.TestCase):
    def setUp(self):
        states = random.getstate(), np.random.get_state(), torch.get_rng_state()
        threads = torch.get_num_threads()
        self.addCleanup(random.setstate, states[0])
        self.addCleanup(np.random.set_state, states[1])
        self.addCleanup(torch.set_rng_state, states[2])
        self.addCleanup(torch.set_num_threads, threads)
        torch.set_num_threads(1)
        torch.manual_seed(11)
        random.seed(23)

    def check_batch(self, shapes):
        # 32 already-collated originals, but native metas retain individual sizes.
        metas = [dict(img_shape=(*shape, 3), pad_shape=(
            (shape[0] + 31) // 32 * 32, (shape[1] + 31) // 32 * 32, 3),
            scale_factor=np.ones(4, np.float32), filename=f"original-{i}")
            for i, shape in enumerate(shapes)]
        saved_metas = copy.deepcopy(metas)
        weak = torch.rand(32, 3, 96, 96)
        strong = torch.rand_like(weak).requires_grad_()
        reference = strong.detach().clone().requires_grad_()
        model = Port()
        boxes, labels = model.teacher_labels(weak, metas, reference, metas)
        initial_rng = random.getstate()
        mosaics = [four_image_mosaic(
            reference[i:i + 4], metas[i:i + 4], boxes[i:i + 4], labels[i:i + 4])
            for i in range(0, 32, 4)]
        expected_rng = random.getstate()
        images, expected_metas, expected_boxes, expected_labels = zip(*mosaics)
        if len(set(shapes)) == 1:
            expected = torch.stack(images)  # Exact old equal-size stack.
        else:
            expected = collate([DataContainer(image, stack=True) for image in images],
                               samples_per_gpu=8).data[0]
        scale = model.scale.detach().clone().requires_grad_()
        (expected * scale).square().sum().backward()
        random.setstate(initial_rng)
        torch_state, numpy_state = torch.get_rng_state(), np.random.get_state()
        losses = model.target_losses(weak, metas, strong, metas)
        self.assertEqual(random.getstate(), expected_rng)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
        np.testing.assert_equal(np.random.get_state(), numpy_state)
        self.assertIs(model.ema_model.originals, weak)
        self.assertIs(model.ema_model.metas, metas)
        self.assertEqual((model.cur_iter, model.image_num), (1, 32))
        self.assertEqual(len(model.batch), 8)
        self.assertTrue(torch.equal(model.batch, expected))
        sum(losses.values()).backward()
        self.assertTrue(torch.equal(strong.grad, reference.grad))
        self.assertTrue(torch.equal(model.scale.grad, scale.grad))
        self.assertGreater(strong.grad.abs().sum().item(), 0)
        got_metas, got_boxes, got_labels = model.targets
        for i, image in enumerate(images):
            height, width = image.shape[-2:]
            self.assertTrue(torch.equal(model.batch[i, :, :height, :width], image))
            self.assertEqual(torch.count_nonzero(model.batch[i, :, height:, :]).item(), 0)
            self.assertEqual(torch.count_nonzero(model.batch[i, :, :, width:]).item(), 0)
            np.testing.assert_equal(got_metas[i], expected_metas[i])
            self.assertTrue(torch.equal(got_boxes[i], expected_boxes[i]))
            self.assertTrue(torch.equal(got_labels[i], expected_labels[i]))
        np.testing.assert_equal(metas, saved_metas)
        self.assertGreater(sum(len(target) for target in got_boxes), 0)
        return model

    def test_real_target_losses_mixed_canvases_match_native_padding(self):
        model = self.check_batch([(17, 29)] * 4 + [(65, 49)] * 4
                                 + [(33, 70)] * 4 + [(29, 17)] * 20)
        self.assertEqual(tuple(model.batch.shape), (8, 3, 96, 96))
        self.assertEqual(model.targets[0][0]["pad_shape"], (32, 32, 3))
        self.assertEqual(model.targets[0][1]["pad_shape"], (96, 64, 3))

    def test_equal_size_is_bitwise_old_stack_including_gradients(self):
        self.check_batch([(27, 23)] * 32)

    def test_mixed_original_sizes_within_each_mosaic_keep_first_canvas_extent(self):
        small, tall, wide, short = (17, 29), (65, 49), (33, 70), (29, 17)
        model = self.check_batch([small, tall, wide, short, tall, small, short, wide] * 4)
        self.assertEqual([meta["img_shape"][:2] for meta in model.targets[0]],
                         [small, tall] * 4)

    def test_native_anchor_validity_uses_per_canvas_not_batch_padding(self):
        generator = AnchorGenerator(strides=[8, 16, 32], ratios=[1.], scales=[1.])
        head = SimpleNamespace(prior_generator=generator)
        sizes = [(12, 12), (6, 6), (3, 3)]
        metas = [dict(img_shape=(17, 29, 3), pad_shape=(32, 32, 3)),
                 dict(img_shape=(65, 49, 3), pad_shape=(96, 64, 3))]
        _, flags = AnchorHead.get_anchors(head, sizes, metas, device="cpu")
        self.assertEqual([int(level.sum()) for level in flags[0]], [16, 4, 1])
        self.assertEqual([int(level.sum()) for level in flags[1]], [96, 24, 6])
        _, common_flags = AnchorHead.get_anchors(
            head, sizes, [dict(pad_shape=(96, 96, 3))], device="cpu")
        self.assertTrue(all(int(common.sum()) > int(local.sum())
                            for common, local in zip(common_flags[0], flags[0])))


if __name__ == "__main__":
    unittest.main()
