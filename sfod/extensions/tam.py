"""SF-YOLO Target Augmentation Module, independently implemented from math.

Algorithm reference: vs-cv/sf-yolo@c84dfad79a889b5f172d7192ee0379edfd738835,
TargetAugment_train/{net.py:6-140,function.py:4-29,train.py:136-165}.
No upstream implementation is included. Inputs are centered BGR float tensors;
normalization, sampling, checkpoint admission and detector training belong to
the caller. Reconstruction training uses spatial sizes divisible by eight.

VGG weights are an explicit external resource, never downloaded here. Tests
may explicitly inject an encoder returning the four feature taps; there is no
random-encoder fallback for production.
"""

from itertools import chain
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from experiments.comparison.tam_artifacts import ENCODER_OFFSET


def channel_moments(features: Tensor) -> tuple[Tensor, Tensor]:
    """Per-image/channel HW mean and sqrt(sample variance + 1e-5)."""
    mean = features.mean(dim=(-2, -1), keepdim=True)
    std = (features.var(dim=(-2, -1), unbiased=True, keepdim=True) + 1e-5).sqrt()
    return mean, std


def moment_mse(first: Tensor, second: Tensor) -> Tensor:
    """Mean MSE plus standard-deviation MSE, each with mean reduction."""
    first_mean, first_std = channel_moments(first)
    second_mean, second_std = channel_moments(second)
    return F.mse_loss(first_mean, second_mean) + F.mse_loss(first_std, second_std)


class FrozenVGGEncoder(nn.Module):
    """Strictly load Oxford VGG16 conv1_1..conv4_1 (16 flat-key tensors).

    ``weights_path`` must point to the externally supplied feature checkpoint
    with numeric Sequential keys (0.weight, etc.), without a model wrapper.
    Inputs stay in TAM working BGR space; the offset honors Oxford's mean.
    Frozen parameters do NOT disable differentiation with respect to images.
    """

    def __init__(self, weights_path: str | Path | None = None):
        super().__init__()
        if weights_path is None:
            raise ValueError("An explicit pretrained VGG feature weights_path is required")
        layers = []
        in_channels = 3
        for channels in (
            64, 64, "pool", 128, 128, "pool", 256, 256, 256, "pool",
            512,
        ):
            if channels == "pool":
                layers.append(nn.MaxPool2d(2, 2, ceil_mode=True))
            else:
                layers.extend([
                    nn.Conv2d(in_channels, channels, 3, padding=1, bias=True),
                    nn.ReLU(),
                ])
                in_channels = channels
        features = nn.Sequential(*layers)
        checkpoint = torch.load(weights_path, weights_only=True, map_location="cpu")
        features.load_state_dict(checkpoint, strict=True)
        self.features = features
        self.register_buffer(
            "input_offset", torch.tensor(ENCODER_OFFSET).view(1, 3, 1, 1),
            persistent=False)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        super().train(False)
        return self

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        images = images + self.input_offset
        taps = []
        for index, layer in enumerate(self.features):
            images = layer(images)
            if index in (1, 6, 11, 18):
                taps.append(images)
        return tuple(taps)


class TAMDecoder(nn.Module):
    """Learned 8-convolution RGB decoder with three nearest-neighbor doublings."""

    def __init__(self):
        super().__init__()
        channels = (512, 256, 256, 256, 128, 128, 64, 64, 3)
        layers = []
        for index, (source, target) in enumerate(zip(channels, channels[1:])):
            layers.append(nn.Conv2d(source, target, 3, padding=1, bias=True))
            if index != 7:
                layers.append(nn.ReLU())
            if index in (0, 3, 5):
                layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
        self.layers = nn.Sequential(*layers)

    def forward_stages(self, features: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return cumulative G1, G2, G3, G4 outputs, not independent heads."""
        stages = []
        for index, layer in enumerate(self.layers):
            features = layer(features)
            if index in (6, 11, 16, 17):
                stages.append(features)
        return tuple(stages)

    def forward(self, features: Tensor) -> Tensor:
        return self.forward_stages(features)[-1]


class TargetAugmentationModule(nn.Module):
    """Learned style-first moment transform and generic aerial TAM objectives.

    Use ``weights_path=...`` for pretrained construction, or explicitly pass a
    test ``encoder`` with 64/128/256/512-channel taps at 1/1, 1/2, 1/4, 1/8 scale.
    ``F1`` predicts the shift; ``F2`` predicts an unconstrained signed scale.
    """

    def __init__(
        self, weights_path: str | Path | None = None, *, encoder: nn.Module | None = None,
    ):
        super().__init__()
        self.encoder = FrozenVGGEncoder(weights_path) if encoder is None else encoder
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.F1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 512))
        self.F2 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 512))
        self.decoder = TAMDecoder()

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def transform(self, content: Tensor, style: Tensor, alpha: float = 1.0) -> Tensor:
        """Transform deepest features; style moments precede content moments."""
        content_mean, content_std = channel_moments(content)
        style_mean, style_std = channel_moments(style)
        shift = self.F1(torch.cat((style_mean, content_mean), dim=1).flatten(1))
        scale = self.F2(torch.cat((style_std, content_std), dim=1).flatten(1))
        z = (content - content_mean) / content_std * scale[:, :, None, None]
        z = z + shift[:, :, None, None]
        return alpha * z + (1 - alpha) * content

    def augment(self, content: Tensor, style: Tensor, alpha: float | None = None) -> Tensor:
        """Generate centered BGR images; default alpha is 1 in train, .4 in eval."""
        if alpha is None:
            alpha = 1.0 if self.training else 0.4
        z = self.transform(self.encoder(content)[-1], self.encoder(style)[-1], alpha)
        return self.decoder(z)

    def forward(self, content: Tensor, style: Tensor, alpha: float | None = None) -> Tensor:
        return self.augment(content, style, alpha)

    def decoder_objective(self, content: Tensor, style: Tensor) -> Tensor:
        """Unit-weight latent consistency plus four content reconstructions.

        Detaching z isolates this update to G. The generated-image encoding
        stays differentiable through the frozen encoder into G.
        """
        content_taps = self.encoder(content)
        z = self.transform(content_taps[-1], self.encoder(style)[-1]).detach()
        generated_deepest = self.encoder(self.decoder(z))[-1]
        g1, g2, g3, g4 = self.decoder.forward_stages(content_taps[-1])
        return (
            F.mse_loss(generated_deepest, z)
            + F.mse_loss(content_taps[0], g3)
            + F.mse_loss(content_taps[1], g2)
            + F.mse_loss(content_taps[2], g1)
            + F.mse_loss(content, g4)
        )

    def moment_objective(self, content: Tensor, style: Tensor) -> Tensor:
        """50 * all-tap style moment loss + all-tap content moment loss.

        This is the generic aerial objective, not the City2Foggy special case.
        The step helper fixes G's parameters, but preserves its input gradient.
        """
        content_taps = self.encoder(content)
        style_taps = self.encoder(style)
        z = self.transform(content_taps[-1], style_taps[-1])
        generated_taps = self.encoder(self.decoder(z))
        style_loss = sum(moment_mse(g, s) for g, s in zip(generated_taps, style_taps))
        content_loss = sum(moment_mse(g, c) for g, c in zip(generated_taps, content_taps))
        return 50 * style_loss + content_loss

    def make_optimizers(self) -> tuple[torch.optim.Adam, torch.optim.Adam]:
        """Return the decoder and F1/F2 Adam optimizers, with disjoint ownership."""
        return (
            torch.optim.Adam(self.decoder.parameters(), lr=1e-4),
            torch.optim.Adam(chain(self.F1.parameters(), self.F2.parameters()), lr=1e-4),
        )

    def alternating_step(
        self,
        content: Tensor,
        style: Tensor,
        decoder_optimizer: torch.optim.Adam,
        moment_optimizer: torch.optim.Adam,
        iteration_zero_based: int,
    ) -> dict[str, Tensor | float]:
        """One decoder-then-F update using the pair from ``make_optimizers``.

        Recompute the F objective with UPDATED G, without stopping gradients
        through G or E to F. The caller owns batch size and iteration count.
        """
        lr = 1e-4 / (1 + 5e-5 * iteration_zero_based)
        for optimizer in (decoder_optimizer, moment_optimizer):
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)

        decoder_loss = self.decoder_objective(content, style)
        if not torch.isfinite(decoder_loss).all():
            raise FloatingPointError("Nonfinite TAM decoder objective")
        decoder_loss.backward()
        decoder_optimizer.step()
        decoder_optimizer.zero_grad(set_to_none=True)

        decoder_parameters = list(self.decoder.parameters())
        trainable = [parameter.requires_grad for parameter in decoder_parameters]
        self.decoder.requires_grad_(False)
        try:
            moment_loss = self.moment_objective(content, style)
            if not torch.isfinite(moment_loss).all():
                raise FloatingPointError("Nonfinite TAM moment objective")
            moment_loss.backward()
            moment_optimizer.step()
        finally:
            for parameter, requires_grad in zip(decoder_parameters, trainable):
                parameter.requires_grad_(requires_grad)
        return {"decoder": decoder_loss.detach(), "moments": moment_loss.detach(), "lr": lr}

    def freeze_for_inference(self):
        """Freeze E, G, F1 and F2 for later detector adaptation; use alpha .4."""
        self.requires_grad_(False)
        self.eval()
        return self
