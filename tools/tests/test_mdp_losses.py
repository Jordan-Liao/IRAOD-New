"""CPU checks for the real MDP mixing, adversarial style and prototype operators."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

_path = Path(__file__).resolve().parents[2] / 'sfod/extensions/mdp_losses.py'
_spec = importlib.util.spec_from_file_location('mdp_losses_cpu', _path)
mdp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mdp)


def distributed_prototypes(rank, directory):
    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        'gloo', init_method='file://' + str(Path(directory) / 'init'),
        rank=rank, world_size=2)
    try:
        sums = torch.tensor(
            [[2., 4.], [0., 0.]] if rank == 0 else [[4., 8.], [9., 3.]],
            requires_grad=True)
        counts = torch.tensor([1, 0] if rank == 0 else [2, 1])
        means, present = mdp.global_class_means(sums, counts)
        means.square().sum().backward()
        (Path(directory) / f'{rank}.json').write_text(json.dumps({
            'means': means.tolist(), 'present': present.tolist(), 'gradient': sums.grad.tolist(),
        }))
    finally:
        torch.distributed.destroy_process_group()


class MDPLossesTest(unittest.TestCase):
    def setUp(self):
        state = torch.get_rng_state()
        threads = torch.get_num_threads()
        torch.manual_seed(71)
        torch.set_num_threads(1)
        self.addCleanup(torch.set_rng_state, state)
        self.addCleanup(torch.set_num_threads, threads)

    def test_msp_two_halves_preserve_all_labels_and_obb_angles(self):
        images = torch.stack([torch.full((3, 4, 6), value) for value in (2., 4., 6., 8.)])
        boxes = [torch.tensor([[float(i), 2., 3., 1., -.3 + i * .1]]) for i in range(4)]
        labels = [torch.tensor([i]) for i in range(4)]
        metas = [dict(ori_filename=f'{i}.png', img_shape=(3 + i % 2, 5 + i % 2, 3),
                      pad_shape=(4, 6, 3), scale_factor=.5, flip=True) for i in range(4)]
        mixed, targets, target_labels, metadata = mdp.mix_target_pairs(images, boxes, labels, metas)
        torch.testing.assert_close(mixed, torch.stack([torch.full((3, 4, 6), 4.),
                                                     torch.full((3, 4, 6), 6.)]))
        self.assertFalse(torch.allclose(mixed, .5 * images[:2] + .25 * images[2:]))
        for i in range(2):
            torch.testing.assert_close(targets[i], torch.cat((boxes[i], boxes[i + 2])))
            torch.testing.assert_close(target_labels[i], torch.tensor([i, i + 2]))
            self.assertEqual(metadata[i]['scale_factor'], 1.)
            self.assertFalse(metadata[i]['flip'])
        targets[0][0, 4] = 99
        self.assertAlmostEqual(boxes[0][0, 4].item(), -.3, places=6)
        with self.assertRaisesRegex(ValueError, 'even local'):
            mdp.mix_target_pairs(images[:3], boxes[:3], labels[:3], metas[:3])

    def test_msp_keeps_pair_valid_padding_not_the_largest_collated_image(self):
        images = torch.zeros(4, 3, 64, 64)
        boxes = [torch.empty(0, 5) for _ in images]
        labels = [torch.empty(0, dtype=torch.long) for _ in images]
        metas = [
            dict(ori_filename=f'{i}.png', img_shape=(size, size, 3),
                 pad_shape=(size, size, 3))
            for i, size in enumerate((32, 64, 32, 64))
        ]
        mixed, _, _, metadata = mdp.mix_target_pairs(images, boxes, labels, metas)
        self.assertEqual(tuple(mixed.shape), (2, 3, 64, 64))
        self.assertEqual(metadata[0]['img_shape'], (32, 32, 3))
        self.assertEqual(metadata[0]['pad_shape'], (32, 32, 3))
        self.assertEqual(metadata[1]['pad_shape'], (64, 64, 3))

    def test_afsp_forward_per_image_norm_and_dual_grl_gradient_signs(self):
        style = mdp.AdversarialFeatureStyle(16).double()
        with torch.no_grad():
            style.style[0].weight.fill_(.02)
            style.style[2].weight.copy_(torch.linspace(.01, .2, 32).reshape(32, 1, 1, 1))
        x = (torch.randn(2, 16, 4, 5, dtype=torch.double) + .6).requires_grad_()
        weights = torch.randn_like(x)
        plain = style.perturb(x)
        original_grad = torch.autograd.grad((plain * weights).sum(), (x, *style.parameters()))
        reversed_result = style(x)
        reversed_grad = torch.autograd.grad(
            (reversed_result * weights).sum(), (x, *style.parameters()))
        torch.testing.assert_close(reversed_result, plain)
        torch.testing.assert_close(reversed_grad[0], original_grad[0])
        for expected, actual in zip(original_grad[1:], reversed_grad[1:]):
            self.assertGreater(actual.abs().sum().item(), 0)
            torch.testing.assert_close(actual, -expected)
        torch.testing.assert_close(
            reversed_result.abs().sum((1, 2, 3)), x.abs().sum((1, 2, 3)),
            rtol=1e-8, atol=1e-8)
        torch.testing.assert_close(reversed_result, torch.cat([style(row[None]) for row in x]))
        self.assertFalse(torch.allclose(reversed_result, x))

    def test_afsp_constant_and_single_spatial_values_remain_finite(self):
        for value in (0., 1.):
            style = mdp.AdversarialFeatureStyle(16)
            x = torch.full((2, 16, 1, 1), value, requires_grad=True)
            output = style(x)
            output.square().sum().backward()
            self.assertTrue(torch.isfinite(output).all())
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in style.parameters()))

    def test_prototypes_use_previous_local_snapshots_and_paper_l2_sum(self):
        history = mdp.PrototypeHistory(2, 2).double()
        teacher1 = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.double, requires_grad=True)
        student1 = torch.tensor([[2., 4.], [3., 1.]], dtype=torch.double, requires_grad=True)
        present = torch.tensor([True, True])
        loss1, count = history(teacher1, student1, present, present)
        expected1 = .7 * torch.linalg.vector_norm(student1 - teacher1, dim=1).sum()
        torch.testing.assert_close(loss1, expected1)
        self.assertEqual(count.item(), 2)
        self.assertFalse(torch.allclose(loss1, ((.7 * student1 - .7 * teacher1) ** 2).mean()))
        loss1.backward()
        self.assertIsNone(teacher1.grad)
        self.assertGreater(student1.grad.abs().sum().item(), 0)
        teacher2 = torch.tensor([[5., 1.], [7., 2.]], dtype=torch.double)
        student2 = torch.tensor([[4., 2.], [2., 6.]], dtype=torch.double, requires_grad=True)
        expected2 = torch.linalg.vector_norm(
            .7 * student2 + .3 * student1.detach()
            - .7 * teacher2 - .3 * teacher1.detach(), dim=1).sum()
        loss2, _ = history(teacher2, student2, present, present)
        torch.testing.assert_close(loss2, expected2)
        loss2.backward()
        self.assertGreater(student2.grad.abs().sum().item(), 0)
        self.assertNotEqual(history.previous_student.data_ptr(), student2.data_ptr())
        self.assertFalse(history.previous_student.requires_grad)
        restored = mdp.PrototypeHistory(2, 2).double()
        restored.load_state_dict(history.state_dict())
        torch.testing.assert_close(restored.previous_student, student2.detach())

    def test_absent_class_has_no_invented_alignment_and_connected_empty_loss(self):
        history = mdp.PrototypeHistory(2, 3)
        student = torch.randn(2, 3, requires_grad=True)
        teacher = torch.randn(2, 3)
        loss, count = history(
            teacher, student, torch.tensor([True, False]), torch.tensor([False, True]))
        self.assertEqual(loss.item(), 0)
        self.assertEqual(count.item(), 0)
        loss.backward()
        torch.testing.assert_close(student.grad, torch.zeros_like(student))

    def test_class_means_preserve_global_ddp_gradient_semantics(self):
        with tempfile.TemporaryDirectory(prefix='mdp-prototype-cpu-') as directory:
            torch.multiprocessing.spawn(distributed_prototypes, args=(directory,), nprocs=2)
            for rank in range(2):
                result = json.loads((Path(directory) / f'{rank}.json').read_text())
                torch.testing.assert_close(torch.tensor(result['means']), torch.tensor([[2., 4.], [9., 3.]]))
                self.assertEqual(result['present'], [True, True])
                # Differentiable SUM is followed by DDP's parameter-gradient average.
                torch.testing.assert_close(torch.tensor(result['gradient']),
                                           torch.tensor([[8 / 3, 16 / 3], [36., 12.]]))


if __name__ == '__main__':
    unittest.main()
