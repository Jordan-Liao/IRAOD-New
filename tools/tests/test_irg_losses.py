"""Focused CPU checks for independently specified IRG tensor mathematics."""

import importlib.util
from pathlib import Path
import unittest

import torch
from torch import nn
from torch.nn import functional as F


# Load only the tensor module: sfod.__init__ imports the detector/GPU stack.
_path = Path(__file__).resolve().parents[2] / "sfod/extensions/irg_losses.py"
_spec = importlib.util.spec_from_file_location("irg_losses_cpu", _path)
irg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(irg)


class IRGLossesTest(unittest.TestCase):
    def setUp(self):
        self.rng_state = torch.get_rng_state()
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        torch.manual_seed(23)
        self.addCleanup(torch.set_rng_state, self.rng_state)
        self.addCleanup(torch.set_num_threads, self.threads)

    def test_graph_squared_row_adjacency_and_three_convolutions(self):
        graph = irg.IRGGraph(2).double()
        with torch.no_grad():
            graph.q.weight.copy_(torch.eye(2))
            graph.k.weight.copy_(torch.eye(2))
            graph.q.bias.zero_()
            graph.k.bias.zero_()
            for layer in graph.layers:
                layer.weight.fill_(0.01)
                layer.bias.fill_(0.03)
        x = torch.tensor([[1., 0.], [-2., 1.], [0., 0.]], dtype=torch.double)
        output, raw = graph(x)
        expected_raw = torch.tensor([[1., -2., 0.], [-2., 5., 0.], [0., 0., 0.]],
                                    dtype=torch.double)
        adjacency = torch.tensor([[1 / 5, 4 / 5, 0.], [4 / 29, 25 / 29, 0.],
                                  [0., 0., 0.]], dtype=torch.double)
        torch.testing.assert_close(raw, expected_raw)
        torch.testing.assert_close(F.normalize(raw.square(), p=1, dim=-1), adjacency)
        self.assertFalse(torch.allclose(adjacency, raw.softmax(dim=-1)))
        expected, softmax_result = x, x
        for layer in graph.layers:
            expected = (adjacency @ expected @ layer.weight.T + layer.bias).clamp_min(0)
            softmax_result = (raw.softmax(-1) @ softmax_result @ layer.weight.T
                              + layer.bias).clamp_min(0)
        torch.testing.assert_close(output, expected)
        self.assertFalse(torch.allclose(output, softmax_result))

    def test_exact_initialization_and_registered_projection_heads(self):
        graph = irg.IRGGraph(3)
        torch.manual_seed(23)
        q, k = nn.Linear(3, 3), nn.Linear(3, 3)
        layers = [nn.Linear(3, 512), nn.Linear(512, 512), nn.Linear(512, 3)]
        for layer in layers:
            nn.init.xavier_normal_(layer.weight, gain=0.02)
            nn.init.zeros_(layer.bias)
        for actual, expected in zip([graph.q, graph.k, *graph.layers], [q, k, *layers]):
            torch.testing.assert_close(actual.weight, expected.weight, rtol=0, atol=0)
            torch.testing.assert_close(actual.bias, expected.bias, rtol=0, atol=0)
        torch.manual_seed(37)
        losses = irg.IRGLosses(3)
        torch.manual_seed(37)
        irg.IRGGraph(3)
        for head in (losses.head1, losses.head2):
            expected = nn.Sequential(nn.Linear(3, 3), nn.ReLU(), nn.Linear(3, 3))
            self.assertIsInstance(head[1], nn.ReLU)
            for actual_param, expected_param in zip(head.parameters(), expected.parameters()):
                torch.testing.assert_close(actual_param, expected_param, rtol=0, atol=0)
        parameters = dict(losses.named_parameters())
        self.assertEqual(len(parameters), 18)
        self.assertTrue(all(p.requires_grad for p in parameters.values()))
        self.assertIsNot(losses.head1[0].weight, losses.head2[0].weight)

    def test_mask_raw_signed_row_minmax_strict_threshold_and_detach(self):
        raw = torch.tensor([[-4., -2., 0.], [3., -1., 1.], [-5., -4., -3.]],
                           requires_grad=True)
        expected = torch.tensor([[True, False, True], [True, True, False],
                                 [False, False, True]])
        mask = irg.contrastive_mask(raw)
        torch.testing.assert_close(mask, expected)
        self.assertFalse(mask.requires_grad)
        self.assertIsNone(mask.grad_fn)
        self.assertFalse(torch.equal(mask, irg.contrastive_mask(
            F.normalize(raw.square(), p=1, dim=-1))))

    def test_contrastive_numerical_all_columns_diagonal_and_gradients(self):
        losses = irg.IRGLosses(2).double()
        with torch.no_grad():
            for head in (losses.head1, losses.head2):
                for layer in (head[0], head[2]):
                    layer.weight.copy_(torch.eye(2))
                    layer.bias.zero_()
            losses.head2[2].weight.copy_(torch.tensor([[1., 0.3], [0.2, 1.]]))
        x = torch.tensor([[1., 0.2], [0.3, 1.], [1., 1.]], dtype=torch.double,
                         requires_grad=True)
        raw = torch.tensor([[0., 2., 1.], [-2., 1., 3.], [3., -1., 0.]],
                           dtype=torch.double, requires_grad=True)
        actual = losses.contrastive_loss(x, raw)
        first = x / x.norm(dim=1, keepdim=True)
        projected = x @ losses.head2[2].weight.T
        second = projected / projected.norm(dim=1, keepdim=True)
        logits = first @ second.T / 0.07
        # Independent fixed positives include each diagonal, even below threshold.
        positive_columns = ([0, 1], [1, 2], [0, 2])
        terms = []
        diagonal_excluded_terms = []
        for row, columns in enumerate(positive_columns):
            denominator = logits[row].exp().sum().log()
            terms.append(torch.stack([denominator - logits[row, col]
                                      for col in columns]).mean())
            without_self = torch.cat((logits[row, :row], logits[row, row + 1:]))
            diagonal_excluded_terms.append(
                torch.stack([without_self.exp().sum().log() - logits[row, col]
                             for col in columns]).mean())
        expected = 0.5 * torch.stack(terms).mean()
        torch.testing.assert_close(actual, expected)
        self.assertFalse(torch.allclose(actual, 0.5 * torch.stack(diagonal_excluded_terms).mean()))
        actual.backward()
        self.assertIsNone(raw.grad)
        self.assertGreater(x.grad.abs().sum().item(), 0)
        for head in (losses.head1, losses.head2):
            for parameter in head.parameters():
                self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_kl_values_detached_targets_and_both_graph_branch_gradients(self):
        losses = irg.IRGLosses(2).double()
        # Positive graph paths prevent a dead ReLU from obscuring gradient checks.
        with torch.no_grad():
            for layer in losses.graph.layers:
                layer.weight.fill_(0.002)
                layer.bias.fill_(0.01)
        classifier = nn.Linear(2, 3).double().requires_grad_(False)
        student_classifier = nn.Linear(2, 3).double()
        with torch.no_grad():
            classifier.weight.copy_(torch.tensor([[1., 0.2], [-0.3, 0.4], [0.1, -0.6]]))
            classifier.bias.copy_(torch.tensor([0.1, -0.2, 0.3]))
            student_classifier.weight.copy_(torch.tensor([[0.4, -0.7], [-0.2, 0.1], [0.9, 0.2]]))
            student_classifier.bias.copy_(torch.tensor([0.2, 0.3, -0.1]))
        xs = torch.tensor([[1., 0.2], [0.3, 1.]], dtype=torch.double, requires_grad=True)
        xt = torch.tensor([[0.5, 1.], [1.2, 0.4]], dtype=torch.double, requires_grad=True)
        sl = torch.tensor([[0.2, 0.4, -0.1], [0.6, -0.2, 0.3]],
                          dtype=torch.double, requires_grad=True)
        tl = torch.tensor([[-0.5, 0.3, 1.4], [0.3, 0.8, 1.2]],
                          dtype=torch.double, requires_grad=True)
        before = tuple(id(p) for p in losses.parameters())
        result = losses(xs, xt, sl, tl, student_classifier, classifier)
        self.assertEqual(before, tuple(id(p) for p in losses.parameters()))
        self.assertFalse(any(p is c for p in losses.parameters() for c in classifier.parameters()))
        sg, _ = losses.graph(xs)
        tg, _ = losses.graph(xt.detach())
        pairs = {
            "loss_irg_direct": (sl, tl.detach()),
            "loss_irg_student_graph": (student_classifier(sg), sl.detach()),
            "loss_irg_teacher_graph": (classifier(tg), tl.detach()),
        }
        self.assertEqual(set(result), {*pairs, "loss_irg_contrast"})
        for name, (prediction, target) in pairs.items():
            probabilities = target.softmax(-1)
            expected = (probabilities * (target.log_softmax(-1)
                                         - prediction.log_softmax(-1))).sum() / len(xs)
            torch.testing.assert_close(result[name], expected)
        graph_parameters = tuple(losses.graph.parameters())
        for name in ("loss_irg_student_graph", "loss_irg_teacher_graph"):
            grads = torch.autograd.grad(result[name], (*graph_parameters, xs, xt, sl, tl),
                                        retain_graph=True, allow_unused=True)
            for grad in grads[:len(graph_parameters)]:
                self.assertIsNotNone(grad)
                self.assertGreater(grad.abs().sum().item(), 0)
            gx, gt, gsl, gtl = grads[len(graph_parameters):]
            if name == "loss_irg_student_graph":
                self.assertGreater(gx.abs().sum().item(), 0)
            else:
                self.assertIsNone(gx)
            self.assertIsNone(gt)
            self.assertIsNone(gsl)
            self.assertIsNone(gtl)
        result["loss_irg_direct"].backward(retain_graph=True)
        self.assertGreater(sl.grad.abs().sum().item(), 0)
        self.assertIsNone(tl.grad)
        sum(result.values()).backward()
        self.assertGreater(xs.grad.abs().sum().item(), 0)
        self.assertIsNone(xt.grad)
        self.assertIsNone(tl.grad)
        self.assertTrue(all(p.grad is None for p in classifier.parameters()))
        self.assertTrue(all(p.grad is not None and p.grad.abs().sum() > 0
                            for p in student_classifier.parameters()))

    def test_1024_api_per_image_callable_classifier(self):
        losses = irg.IRGLosses()
        classifier = nn.Linear(1024, 4).requires_grad_(False)
        student_classifier = nn.Linear(1024, 4)
        registered = tuple(id(p) for p in losses.parameters())
        # Separate images have different proposal counts; no batched graph edges.
        for count in (2, 3):
            xs, xt = torch.randn(count, 1024), torch.randn(count, 1024)
            representation, raw = losses.graph(xs)
            self.assertEqual(representation.shape, (count, 1024))
            self.assertEqual(raw.shape, (count, count))
            result = losses(xs, xt, student_classifier(xs), classifier(xt),
                            student_classifier, lambda features: classifier(features))
            self.assertTrue(all(value.shape == () and torch.isfinite(value)
                                for value in result.values()))
        self.assertEqual(registered, tuple(id(p) for p in losses.parameters()))


if __name__ == "__main__":
    unittest.main()
