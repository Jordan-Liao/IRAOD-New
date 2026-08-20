"""Separable low-rank projection (SLRP): one unrolled robust-PCA step.

Motivation
----------
Coherent SAR interference is low-rank and separable in the range-azimuth plane:
narrowband RFI and AM jamming are stripes, which are exactly rank-one outer
products of a line profile with a constant vector, while barrage suppression and
chaff are smooth low-rank blobs. Targets are the opposite: spatially compact and
locally full-rank. Writing the feature energy as

    E = L (low-rank, interference) + S (sparse, target) + N

is the robust-PCA model, and this module is one unrolled step of it: ``L`` is
estimated by a local separable projection, ``S`` is what survives a soft
threshold on the residual, and the two energy shares plus the two directional
profile ratios form a per-pixel interference descriptor that drives a bounded
gate on the feature.

The decomposition is local rather than global for two reasons: test-time zero
padding would otherwise inject spurious separable structure into the row and
column profiles, and ``chaff``/``smart_suppression`` are localized phenomena
that a whole-image profile averages away.

The gate is ``1 + gamma * tanh(...)`` with a zero-initialized output layer, so a
freshly constructed module is exactly the identity and a pretrained backbone
checkpoint keeps its behaviour until the module is trained.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


DESCRIPTOR_CHANNELS = 6


def _replicate_pad(x, pad_h, pad_w):
    if pad_h == 0 and pad_w == 0:
        return x
    # Replicate rather than reflect: reflect requires pad < size, which fails on
    # the 8x8 C5 maps of small RSAR crops.
    return F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode='replicate')


def _band_mean(x, kernel_h, kernel_w):
    """Local mean over a ``kernel_h x kernel_w`` window, stride one."""
    padded = _replicate_pad(x, kernel_h // 2, kernel_w // 2)
    return F.avg_pool2d(padded, (kernel_h, kernel_w), stride=1)


def local_separable_decomposition(energy, window, eps=1e-6):
    """Split an energy map into local separable and residual parts.

    ``energy`` is ``[B, 1, H, W]``. The row band averages along the width and the
    column band along the height, so a horizontal stripe keeps its row band high
    while its column band regresses to the local mean; their normalised product
    is the local rank-one (separable) model of the window.
    """
    if energy.ndim != 4 or energy.shape[1] != 1:
        raise ValueError(
            f'energy must be [B, 1, H, W]; got {tuple(energy.shape)}')
    if window < 1 or window % 2 == 0:
        raise ValueError(f'window must be a positive odd integer; got {window}')
    height, width = energy.shape[2:]
    # An even-sized window would shift the output by half a pixel, so clamp to
    # the largest odd window the map can host.
    span_h = min(window, height if height % 2 == 1 else height - 1)
    span_w = min(window, width if width % 2 == 1 else width - 1)
    span_h = max(1, span_h)
    span_w = max(1, span_w)

    row_band = _band_mean(energy, 1, span_w)
    col_band = _band_mean(energy, span_h, 1)
    total = _band_mean(energy, span_h, span_w).clamp_min(eps)
    separable = row_band * col_band / total
    residual = energy - separable
    return separable, residual, row_band, col_band, total


def _local_cv(energy, span_h, span_w, eps):
    """Coefficient of variation of the energy inside a local window.

    Fully developed speckle has a fixed CV, so this is a physically anchored
    homogeneity measure: barrage-style suppression lifts the noise floor and
    drives it down, a coherent spike drives it up.
    """
    mean = _band_mean(energy, span_h, span_w)
    mean_sq = _band_mean(energy.square(), span_h, span_w)
    variance = (mean_sq - mean.square()).clamp_min(0.0)
    return variance.sqrt() / mean.clamp_min(eps)


def interference_descriptor(feature, window, sparse_multiplier=1.0, eps=1e-6):
    """Per-pixel, gain-invariant descriptor of separable interference structure.

    Returns ``[B, 6, H, W]``:

    0. separable energy share -- how much of the local energy the rank-one model
       explains, which is the low-rank interference term;
    1. soft-thresholded residual share -- the sparse target term;
    2-3. log ratios of the row and column bands to the local total, whose sign
       distinguishes ``am_noise_horizontal`` from ``am_noise_vertical``;
    4. local CV -- absolute homogeneity, anchored on speckle statistics;
    5. log ratio of the local CV to the scene's own median CV -- within-scene
       homogeneity contrast in the CFAR sense, which is what a localized
       ``chaff`` or ``smart_suppression`` patch perturbs while leaving the scene
       mean intact.

    Channels 4 and 5 were added after a descriptor-versus-diagnostic comparison
    (``work_dirs/slrp_descriptor_400``) showed the first four trailing badly on
    exactly those two area corruptions: 0.744 vs 0.978 on ``chaff`` and 0.720 vs
    0.840 on ``smart_suppression``. Channel 5 references the scene median rather
    than a wider window because a ``smart_suppression`` patch is far wider than
    any fixed window at this tap, so both windows land inside it and their ratio
    collapses to one everywhere except the patch edge.
    """
    if feature.ndim != 4:
        raise ValueError(
            f'feature must be [B, C, H, W]; got {tuple(feature.shape)}')
    energy = feature.square().mean(dim=1, keepdim=True)
    separable, residual, row_band, col_band, total = \
        local_separable_decomposition(energy, window, eps=eps)

    denominator = energy.clamp_min(eps)
    separable_share = (separable / denominator).clamp(0.0, 4.0)
    threshold = sparse_multiplier * total
    sparse = (residual.abs() - threshold).clamp_min(0.0)
    sparse_share = (sparse / denominator).clamp(0.0, 4.0)
    row_ratio = torch.log((row_band / total).clamp_min(eps))
    col_ratio = torch.log((col_band / total).clamp_min(eps))

    height, width = energy.shape[2:]
    span_h = max(1, min(window, height if height % 2 == 1 else height - 1))
    span_w = max(1, min(window, width if width % 2 == 1 else width - 1))
    cv_local = _local_cv(energy, span_h, span_w, eps)
    reference = _scene_median_cv(cv_local, energy, eps)
    cv_contrast = torch.log((cv_local / reference).clamp_min(eps))
    return torch.cat(
        (separable_share, sparse_share, row_ratio, col_ratio,
         cv_local, cv_contrast), dim=1)


def _scene_median_cv(cv_local, energy, eps):
    """Per-sample median CV over the image content, excluding test-time padding.

    Padding is near-zero energy and therefore near-zero CV, which would drag the
    median down and inflate the contrast of every padded image. Positions below a
    small fraction of the sample's mean energy are masked out of the median.
    """
    batch = cv_local.shape[0]
    flat_cv = cv_local.reshape(batch, -1)
    flat_energy = energy.reshape(batch, -1)
    floor = 1e-3 * flat_energy.mean(dim=1, keepdim=True)
    masked = torch.where(
        flat_energy > floor, flat_cv, torch.full_like(flat_cv, float('nan')))
    median = torch.nanmedian(masked, dim=1, keepdim=True).values
    # A fully masked sample (an all-zero feature map) falls back to the
    # unmasked median so the descriptor stays finite.
    fallback = flat_cv.median(dim=1, keepdim=True).values
    median = torch.where(torch.isnan(median), fallback, median)
    return median.clamp_min(eps).reshape(batch, 1, 1, 1)


class SeparableLowRankProjection(nn.Module):
    """Gate a feature map by its local separable-interference descriptor.

    ``gamma`` bounds the multiplicative correction to ``[1 - gamma, 1 + gamma]``,
    and the output layer starts at zero so the module is initially an exact
    identity.
    """

    def __init__(self,
                 channels,
                 window=7,
                 hidden_channels=16,
                 gamma=0.5,
                 sparse_multiplier=1.0,
                 eps=1e-6):
        super().__init__()
        if channels <= 0:
            raise ValueError('channels must be positive')
        if window < 1 or window % 2 == 0:
            raise ValueError('window must be a positive odd integer')
        if hidden_channels <= 0:
            raise ValueError('hidden_channels must be positive')
        if not 0.0 <= gamma <= 1.0:
            raise ValueError('gamma must be in [0, 1]')
        if sparse_multiplier < 0.0:
            raise ValueError('sparse_multiplier must be non-negative')
        if eps <= 0.0:
            raise ValueError('eps must be positive')

        self.channels = int(channels)
        self.window = int(window)
        self.hidden_channels = int(hidden_channels)
        self.gamma = float(gamma)
        self.sparse_multiplier = float(sparse_multiplier)
        self.eps = float(eps)

        self.gate = nn.Sequential(
            nn.Conv2d(DESCRIPTOR_CHANNELS, self.hidden_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_channels, self.channels, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self._last_diagnostics = {}
        self._last_gate_deviation = None

    def last_gate_deviation(self):
        """Differentiable ``factor - 1`` from the most recent forward pass."""
        if self._last_gate_deviation is None:
            raise RuntimeError('SLRP has not run a forward pass yet')
        return self._last_gate_deviation

    def get_slrp_diagnostics(self):
        return {
            name: value.detach().clone()
            for name, value in self._last_diagnostics.items()
        }

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(
                'SLRP expects [B, C, H, W] with '
                f'C={self.channels}; got {tuple(x.shape)}')
        descriptor = interference_descriptor(
            x, self.window,
            sparse_multiplier=self.sparse_multiplier, eps=self.eps)
        correction = torch.tanh(self.gate(descriptor.to(dtype=x.dtype)))
        factor = 1.0 + self.gamma * correction
        # Kept differentiable and un-detached so a calibration objective can
        # penalise the gate directly -- constraining "do not touch clean input"
        # at the gate is far sharper than measuring it as a feature distance,
        # which a frozen detection head turns out to be very sensitive to.
        self._last_gate_deviation = factor - 1.0
        # Channel-averaged suppression map, kept for interference localization
        # and for gating downstream pseudo-label confidence.
        suppression = (1.0 - factor).mean(dim=1, keepdim=True)
        self._last_diagnostics = {
            'separable_share_mean': descriptor[:, 0].mean(dim=(1, 2)).detach(),
            'sparse_share_mean': descriptor[:, 1].mean(dim=(1, 2)).detach(),
            'gate_magnitude': (factor - 1.0).abs().mean(dim=(1, 2, 3)).detach(),
            'suppression_map': suppression.detach(),
        }
        return x * factor
