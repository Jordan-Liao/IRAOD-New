"""Deterministic CPU tests of the mathematical TAM contract; no downloads."""

import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn
from torch.nn import functional as F


# Avoid sfod.__init__, which imports the detector/GPU stack.
_path = Path(__file__).resolve().parents[2] / "sfod/extensions/tam.py"
_spec = importlib.util.spec_from_file_location("tam_cpu", _path)
tam = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tam)


def reference_moments(x):
    flat = x.flatten(2)
    mean = flat.sum(-1, keepdim=True) / flat.shape[-1]
    variance = (flat - mean).square().sum(-1, keepdim=True) / (flat.shape[-1] - 1)
    return mean.unsqueeze(-1), (variance + 1e-5).sqrt().unsqueeze(-1)


def reference_moment_loss(x, y):
    xm, xs = reference_moments(x)
    ym, ys = reference_moments(y)
    return (xm - ym).square().mean() + (xs - ys).square().mean()


class TinyEncoder(nn.Module):
    """Explicit test double: cheap differentiable taps with the real channels."""

    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.75))

    def forward(self, images):
        base = images.mean(1, keepdim=True) * self.gain
        taps = []
        for index, channels in enumerate((64, 128, 256, 512)):
            if index:
                base = F.avg_pool2d(base, 2)
            channel_scale = torch.linspace(0.5, 1.5, channels, device=base.device,
                                           dtype=base.dtype).view(1, channels, 1, 1)
            taps.append(base * channel_scale)
        return tuple(taps)


def synthetic_full_vgg_state():
    """Synthetic test state only; constructed independently of the loader."""
    layers = []
    source = 3
    for width, count in ((64, 2), (128, 2), (256, 3), (512, 3), (512, 3)):
        for _ in range(count):
            layers.extend([nn.Conv2d(source, width, 3, padding=1), nn.ReLU()])
            source = width
        layers.append(nn.MaxPool2d(2, 2, ceil_mode=True))
    return nn.Sequential(*layers).state_dict()


class TAMTest(unittest.TestCase):
    def setUp(self):
        rng_state = torch.get_rng_state()
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        torch.manual_seed(41)
        self.addCleanup(torch.set_rng_state, rng_state)
        self.addCleanup(torch.set_num_threads, threads)

    def images(self):
        content = torch.randn(1, 3, 16, 24) * 2 + 1
        style = torch.randn(1, 3, 16, 24) * 3 + 4
        return content, style

    def test_channel_moments_use_sample_variance_and_epsilon(self):
        x = torch.tensor([[[[1., 3.], [5., 7.]], [[9., 9.], [9., 9.]]]],
                         dtype=torch.double, requires_grad=True)
        mean, std = tam.channel_moments(x)
        torch.testing.assert_close(mean, torch.tensor([4., 9.], dtype=x.dtype).view(1, 2, 1, 1))
        expected_std = torch.tensor([20 / 3 + 1e-5, 1e-5], dtype=x.dtype).sqrt()
        torch.testing.assert_close(std, expected_std.view(1, 2, 1, 1))
        self.assertFalse(torch.allclose(std[:, :1], (x[:, :1].var(unbiased=False) + 1e-5).sqrt()))
        (mean.sum() + std.sum()).backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        torch.testing.assert_close(tam.moment_mse(x, x * 2 + 3),
                                   reference_moment_loss(x, x * 2 + 3))

    def test_style_first_learned_formula_with_negative_scale_and_alpha(self):
        module = tam.TargetAugmentationModule(encoder=TinyEncoder()).double()
        for mlp in (module.F1, module.F2):
            self.assertEqual(len(mlp), 3)
            self.assertIsInstance(mlp[1], nn.ReLU)
            self.assertEqual((mlp[0].in_features, mlp[0].out_features), (1024, 512))
            self.assertEqual((mlp[2].in_features, mlp[2].out_features), (512, 512))
            self.assertIsNotNone(mlp[0].bias)
            self.assertIsNotNone(mlp[2].bias)
        diagonal = torch.arange(512)
        with torch.no_grad():
            for mlp in (module.F1, module.F2):
                mlp[0].weight.zero_()
                mlp[2].weight.zero_()
            module.F1[0].weight[diagonal, diagonal] = 1
            module.F1[0].weight[diagonal, diagonal + 512] = 2
            module.F1[0].bias.fill_(0.3)
            module.F1[2].weight[diagonal, diagonal] = 0.5
            module.F1[2].bias.fill_(-2)
            module.F2[0].weight[diagonal, diagonal] = 3
            module.F2[0].weight[diagonal, diagonal + 512] = 0.5
            module.F2[0].bias.fill_(0.1)
            module.F2[2].weight[diagonal, diagonal] = -1
            module.F2[2].bias.fill_(-0.2)
        content = torch.arange(2 * 512 * 6, dtype=torch.double).reshape(2, 512, 2, 3) / 1000
        style = content * 4 + 7
        cm, cs = reference_moments(content)
        sm, ss = reference_moments(style)
        shift = 0.5 * (sm + 2 * cm + 0.3).relu() - 2
        scale = -(3 * ss + 0.5 * cs + 0.1).relu() - 0.2
        self.assertTrue((scale < 0).all())
        expected = (content - cm) / cs * scale + shift
        torch.testing.assert_close(module.transform(content, style), expected)
        torch.testing.assert_close(module.transform(content, style, 0.4), 0.4 * expected + 0.6 * content)
        torch.testing.assert_close(module.transform(content, style, 0), content)
        reversed_shift = 0.5 * (cm + 2 * sm + 0.3).relu() - 2
        self.assertFalse(torch.allclose(shift, reversed_shift))

    def test_decoder_layers_cumulative_stages_and_unclipped_output(self):
        decoder = tam.TAMDecoder()
        convs = [layer for layer in decoder.layers if isinstance(layer, nn.Conv2d)]
        self.assertEqual([(layer.in_channels, layer.out_channels) for layer in convs],
                         [(512, 256), (256, 256), (256, 256), (256, 128),
                          (128, 128), (128, 64), (64, 64), (64, 3)])
        for conv in convs:
            self.assertEqual(conv.kernel_size, (3, 3))
            self.assertEqual(conv.padding, (1, 1))
            self.assertIsNotNone(conv.bias)
        self.assertEqual([i for i, layer in enumerate(decoder.layers)
                          if isinstance(layer, nn.ReLU)], [1, 4, 6, 8, 11, 13, 16])
        self.assertEqual([i for i, layer in enumerate(decoder.layers)
                          if isinstance(layer, nn.Upsample)], [2, 9, 14])
        for index in (2, 9, 14):
            self.assertEqual(decoder.layers[index].mode, "nearest")
            self.assertEqual(decoder.layers[index].scale_factor, 2)
        z = torch.randn(1, 512, 2, 3)
        stages = decoder.forward_stages(z)
        self.assertEqual([tuple(x.shape) for x in stages],
                         [(1, 256, 4, 6), (1, 128, 8, 12),
                          (1, 64, 16, 24), (1, 3, 16, 24)])
        expected = z
        for stage, (start, end) in zip(stages, ((0, 7), (7, 12), (12, 17), (17, 18))):
            expected = decoder.layers[start:end](expected)
            torch.testing.assert_close(stage, expected)
        torch.testing.assert_close(decoder(z), stages[-1])
        with torch.no_grad():
            convs[-1].weight.zero_()
            convs[-1].bias.fill_(-17)
        torch.testing.assert_close(decoder(z), torch.full_like(stages[-1], -17))

    def test_full_vgg_strict_loading_taps_freezing_and_input_gradient(self):
        state = synthetic_full_vgg_state()
        self.assertEqual(len(state), 26)
        path = Path("explicit-external-test-vgg.pth")
        with mock.patch.object(tam.torch, "load", return_value={"model": state}) as load:
            encoder = tam.FrozenVGGEncoder(path)
        load.assert_called_once_with(path, weights_only=True, map_location="cpu")
        self.assertEqual(len(encoder.features), 19)
        self.assertEqual(set(encoder.state_dict()),
                         {f"features.{key}" for key in state if int(key.split(".")[0]) < 19})
        for key, value in encoder.features.state_dict().items():
            torch.testing.assert_close(value, state[key], rtol=0, atol=0)
        pools = [layer for layer in encoder.features if isinstance(layer, nn.MaxPool2d)]
        self.assertEqual(len(pools), 3)
        self.assertTrue(all(layer.ceil_mode for layer in pools))
        self.assertFalse(hasattr(encoder, "classifier"))
        encoder.train()
        self.assertFalse(encoder.training)
        self.assertTrue(all(not layer.training for layer in encoder.modules()))
        self.assertTrue(all(not p.requires_grad for p in encoder.parameters()))
        images = torch.randn(1, 3, 17, 25, requires_grad=True)
        taps = encoder(images)
        self.assertEqual([tuple(x.shape) for x in taps],
                         [(1, 64, 17, 25), (1, 128, 9, 13),
                          (1, 256, 5, 7), (1, 512, 3, 4)])
        expected = images
        for tap, (start, end) in zip(taps, ((0, 2), (2, 7), (7, 12), (12, 19))):
            expected = encoder.features[start:end](expected)
            torch.testing.assert_close(tap, expected)
        taps[-1].square().mean().backward()
        self.assertGreater(images.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(images.grad).all())
        self.assertTrue(all(p.grad is None for p in encoder.parameters()))

    def test_missing_weights_and_full_state_mismatch_fail_visibly(self):
        with self.assertRaisesRegex(ValueError, "explicit pretrained"):
            tam.FrozenVGGEncoder()
        with self.assertRaisesRegex(ValueError, "explicit pretrained"):
            tam.TargetAugmentationModule()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                tam.FrozenVGGEncoder(Path(directory) / "absent.pth")
        state = synthetic_full_vgg_state()
        # A bad layer beyond the retained taps must still fail full-state loading.
        state["28.bias"] = torch.zeros(511)
        with mock.patch.object(tam.torch, "load", return_value={"model": state}):
            with self.assertRaisesRegex(RuntimeError, "size mismatch for 28.bias"):
                tam.FrozenVGGEncoder("explicit-external-test-vgg.pth")
        del state["28.bias"]
        with mock.patch.object(tam.torch, "load", return_value={"model": state}):
            with self.assertRaisesRegex(RuntimeError, "Missing key"):
                tam.FrozenVGGEncoder("explicit-external-test-vgg.pth")
        with mock.patch.object(tam.torch, "load", return_value={}):
            with self.assertRaises(KeyError):
                tam.FrozenVGGEncoder("explicit-external-test-vgg.pth")

    def test_decoder_objective_exact_five_terms_and_stopgrad_z(self):
        module = tam.TargetAugmentationModule(encoder=TinyEncoder())
        content, style = self.images()
        ct, st = module.encoder(content), module.encoder(style)
        z = module.transform(ct[-1], st[-1])
        stages = module.decoder.forward_stages(ct[-1])
        latent_term = (module.encoder(module.decoder(z))[-1] - z.detach()).square().mean()
        expected = latent_term + sum((target - output).square().mean() for target, output in (
            (ct[0], stages[2]), (ct[1], stages[1]), (ct[2], stages[0]), (content, stages[3]),
        ))
        with mock.patch.object(module, "transform", return_value=z):
            loss = module.decoder_objective(content, style)
        torch.testing.assert_close(loss, expected)
        z.retain_grad()
        loss.backward()
        self.assertIsNone(z.grad)
        self.assertTrue(all(p.grad is None for p in module.F1.parameters()))
        self.assertTrue(all(p.grad is None for p in module.F2.parameters()))
        self.assertTrue(all(p.grad is None for p in module.encoder.parameters()))
        self.assertGreater(sum(p.grad.abs().sum().item() for p in module.decoder.parameters()), 0)
        # Isolate the latent term so reconstruction gradients cannot hide an E no_grad bug.
        module.zero_grad(set_to_none=True)
        encoded = module.encoder(module.decoder(z.detach()))[-1]
        (encoded - z.detach()).square().mean().backward()
        self.assertGreater(sum(p.grad.abs().sum().item() for p in module.decoder.parameters()), 0)

    def test_moment_objective_all_taps_weights_and_fixed_decoder_gradient(self):
        module = tam.TargetAugmentationModule(encoder=TinyEncoder())
        content, style = self.images()
        module.decoder.requires_grad_(False)
        ct, st = module.encoder(content), module.encoder(style)
        generated = module.encoder(module.decoder(module.transform(ct[-1], st[-1])))
        style_terms = [reference_moment_loss(g, s) for g, s in zip(generated, st)]
        content_terms = [reference_moment_loss(g, c) for g, c in zip(generated, ct)]
        expected = 50 * sum(style_terms) + sum(content_terms)
        loss = module.moment_objective(content, style)
        torch.testing.assert_close(loss, expected)
        self.assertFalse(torch.allclose(loss, 50 * style_terms[-1]))
        self.assertFalse(torch.allclose(loss, 50 * sum(style_terms)))
        loss.backward()
        for head in (module.F1, module.F2):
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in head.parameters()))
            self.assertGreater(sum(p.grad.abs().sum().item() for p in head.parameters()), 0)
        self.assertTrue(all(p.grad is None for p in module.decoder.parameters()))
        self.assertTrue(all(p.grad is None for p in module.encoder.parameters()))

    def test_alternating_adam_ownership_order_updated_decoder_and_learning_rate(self):
        module = tam.TargetAugmentationModule(encoder=TinyEncoder())
        reference = copy.deepcopy(module)
        content, style = self.images()
        decoder_opt, moment_opt = module.make_optimizers()
        self.assertIsInstance(decoder_opt, torch.optim.Adam)
        self.assertIsInstance(moment_opt, torch.optim.Adam)
        owned = lambda opt: {id(p) for group in opt.param_groups for p in group["params"]}
        decoder_ids, moment_ids = owned(decoder_opt), owned(moment_opt)
        self.assertEqual(decoder_ids, {id(p) for p in module.decoder.parameters()})
        self.assertEqual(moment_ids, {id(p) for h in (module.F1, module.F2) for p in h.parameters()})
        self.assertFalse(decoder_ids & moment_ids)
        self.assertFalse((decoder_ids | moment_ids) & {id(p) for p in module.encoder.parameters()})
        self.assertEqual(decoder_opt.param_groups[0]["lr"], 1e-4)
        self.assertEqual(moment_opt.param_groups[0]["lr"], 1e-4)

        before = {key: value.clone() for key, value in module.state_dict().items()}
        ref_decoder_opt, ref_moment_opt = reference.make_optimizers()
        iteration = 123
        lr = 1e-4 / (1 + 5e-5 * iteration)
        for opt in (ref_decoder_opt, ref_moment_opt):
            opt.param_groups[0]["lr"] = lr
        ref_decoder_loss = reference.decoder_objective(content, style)
        ref_decoder_loss.backward()
        ref_decoder_opt.step()
        ref_decoder_opt.zero_grad(set_to_none=True)
        reference.decoder.requires_grad_(False)
        ref_moment_loss = reference.moment_objective(content, style)
        ref_moment_loss.backward()
        ref_moment_opt.step()

        events = []
        actual_decoder_step, actual_moment_step = decoder_opt.step, moment_opt.step
        original_moment_objective = module.moment_objective

        def decoder_step():
            events.append("decoder")
            self.assertTrue(all(p.grad is None for h in (module.F1, module.F2) for p in h.parameters()))
            actual_decoder_step()
            for name, p in module.named_parameters():
                if not name.startswith("decoder."):
                    torch.testing.assert_close(p, before[name], rtol=0, atol=0)

        def moment_objective(c, s):
            events.append("recompute")
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in module.decoder.parameters()))
            for p, expected in zip(module.decoder.parameters(), reference.decoder.parameters()):
                torch.testing.assert_close(p, expected, rtol=0, atol=0)
            self.assertTrue(any(not torch.equal(value, before[key])
                                for key, value in module.state_dict().items() if key.startswith("decoder.")))
            return original_moment_objective(c, s)

        def moment_step():
            events.append("moments")
            actual_moment_step()

        with mock.patch.object(decoder_opt, "step", side_effect=decoder_step), \
                mock.patch.object(moment_opt, "step", side_effect=moment_step), \
                mock.patch.object(module, "moment_objective", side_effect=moment_objective):
            result = module.alternating_step(content, style, decoder_opt, moment_opt, iteration)
        self.assertEqual(events, ["decoder", "recompute", "moments"])
        self.assertEqual(result["lr"], lr)
        self.assertEqual(decoder_opt.param_groups[0]["lr"], lr)
        self.assertEqual(moment_opt.param_groups[0]["lr"], lr)
        torch.testing.assert_close(result["decoder"], ref_decoder_loss.detach())
        torch.testing.assert_close(result["moments"], ref_moment_loss.detach())
        self.assertFalse(result["decoder"].requires_grad)
        self.assertFalse(result["moments"].requires_grad)
        for key, actual in module.state_dict().items():
            torch.testing.assert_close(actual, reference.state_dict()[key], rtol=0, atol=0)
        for prefix in ("F1.", "F2.", "decoder."):
            self.assertTrue(any(not torch.equal(value, before[key]) for key, value in
                                module.state_dict().items() if key.startswith(prefix)))
        torch.testing.assert_close(module.encoder.gain, before["encoder.gain"], rtol=0, atol=0)
        self.assertTrue(all(p.requires_grad and p.grad is None for p in module.decoder.parameters()))
        self.assertTrue(all(p.grad is None for p in module.encoder.parameters()))

    def test_nonfinite_decoder_loss_stops_before_any_optimizer_step(self):
        module = tam.TargetAugmentationModule(encoder=TinyEncoder())
        decoder_opt, moment_opt = module.make_optimizers()
        content, style = self.images()
        with mock.patch.object(module, "decoder_objective", return_value=torch.tensor(float("nan"))), \
                mock.patch.object(decoder_opt, "step") as decoder_step, \
                mock.patch.object(moment_opt, "step") as moment_step, \
                self.assertRaisesRegex(FloatingPointError, "Nonfinite TAM decoder"):
            module.alternating_step(content, style, decoder_opt, moment_opt, 0)
        decoder_step.assert_not_called()
        moment_step.assert_not_called()

    def test_augmentation_defaults_and_inference_freeze(self):
        module = tam.TargetAugmentationModule(encoder=TinyEncoder())
        content, style = self.images()
        module.train()
        self.assertFalse(module.encoder.training)
        self.assertTrue(all(not p.requires_grad for p in module.encoder.parameters()))
        with mock.patch.object(module, "transform", wraps=module.transform) as transform:
            training_image = module(content, style)
        self.assertEqual(transform.call_args.args[2], 1.0)
        ct, st = module.encoder(content)[-1], module.encoder(style)[-1]
        torch.testing.assert_close(training_image, module.decoder(module.transform(ct, st, 1)))
        self.assertIs(module.freeze_for_inference(), module)
        self.assertTrue(all(not p.requires_grad for p in module.parameters()))
        self.assertTrue(all(not child.training for child in module.modules()))
        with mock.patch.object(module, "transform", wraps=module.transform) as transform:
            output = module.augment(content, style)
        self.assertEqual(transform.call_args.args[2], 0.4)
        self.assertFalse(output.requires_grad)
        torch.testing.assert_close(output, module.decoder(module.transform(ct, st, 0.4)))
        with mock.patch.object(module, "transform", wraps=module.transform) as transform:
            module(content, style, alpha=0.7)
        self.assertEqual(transform.call_args.args[2], 0.7)
        content.requires_grad_(True)
        module(content, style).square().mean().backward()
        self.assertGreater(content.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in module.parameters()))


if __name__ == "__main__":
    unittest.main()
