import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


SOGC_SOURCE_STAT_NAMES = frozenset({
    'source_mean', 'source_var', 'source_count', 'source_m2', 'stats_ready'
})


def is_sogc_source_stat_key(key):
    """Return whether a state-dict key is an immutable SOGC source buffer."""
    return '.sogc.' in key and key.rsplit('.', 1)[-1] in \
        SOGC_SOURCE_STAT_NAMES


class SOGCCalibrator(nn.Module):
    """Source-anchored residual calibration for one channel attention.

    Source statistics are accumulated over the batch/sample dimension. They
    are kept in float64 buffers for stable collection and are never optimizer
    parameters.
    """

    VALID_MODES = frozenset({'off', 'collect', 'calibrate'})

    def __init__(self,
                 channels,
                 mode='off',
                 gamma=0.10,
                 reduction=16,
                 eps=1e-5,
                 z_clip=5.0,
                 strict_stats=True):
        super().__init__()
        if channels <= 0:
            raise ValueError('channels must be positive')
        if not 0.0 <= gamma <= 1.0:
            raise ValueError('gamma must be in [0, 1]')
        if reduction <= 0:
            raise ValueError('reduction must be positive')
        if eps <= 0.0:
            raise ValueError('eps must be positive')
        if z_clip <= 0.0:
            raise ValueError('z_clip must be positive')

        self.channels = int(channels)
        self.gamma = float(gamma)
        self.reduction = int(reduction)
        self.eps = float(eps)
        self.z_clip = float(z_clip)
        self.strict_stats = bool(strict_stats)

        hidden_channels = max(self.channels // self.reduction, 1)
        self.mlp = nn.Sequential(
            nn.Linear(3 * self.channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, self.channels),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        shape = (1, self.channels, 1, 1)
        self.register_buffer(
            'source_mean', torch.zeros(shape, dtype=torch.float64))
        self.register_buffer(
            'source_var', torch.full(shape, self.eps, dtype=torch.float64))
        self.register_buffer(
            'source_count', torch.zeros((), dtype=torch.long))
        self.register_buffer(
            'source_m2', torch.zeros(shape, dtype=torch.float64))
        self.register_buffer(
            'stats_ready', torch.zeros((), dtype=torch.bool))

        self._warned_missing_stats = False
        self._last_diagnostics = {}
        self.set_sogc_mode(mode)

    def _apply(self, fn):
        super()._apply(fn)
        # model.half() must not lower the precision of Welford accumulators.
        self.source_mean = self.source_mean.to(dtype=torch.float64)
        self.source_var = self.source_var.to(dtype=torch.float64)
        self.source_m2 = self.source_m2.to(dtype=torch.float64)
        self.source_count = self.source_count.to(dtype=torch.long)
        self.stats_ready = self.stats_ready.to(dtype=torch.bool)
        return self

    def reset_source_stats(self):
        self.source_mean.zero_()
        self.source_var.fill_(self.eps)
        self.source_count.zero_()
        self.source_m2.zero_()
        self.stats_ready.zero_()
        self._warned_missing_stats = False
        self._last_diagnostics = {}

    @torch.no_grad()
    def update_source_stats(self, r):
        self._validate_response(r)
        values = r.detach().to(dtype=torch.float64)
        if not torch.isfinite(values).all():
            raise FloatingPointError(
                'SOGC source response contains non-finite values')

        batch_count = values.shape[0]
        if batch_count == 0:
            return
        batch_mean = values.mean(dim=0, keepdim=True)
        centered = values - batch_mean
        batch_m2 = (centered * centered).sum(dim=0, keepdim=True)

        old_count = int(self.source_count.item())
        new_count = old_count + batch_count
        if old_count == 0:
            self.source_mean.copy_(batch_mean)
            self.source_m2.copy_(batch_m2)
        else:
            delta = batch_mean - self.source_mean
            self.source_mean.add_(delta * (batch_count / new_count))
            cross = delta.square() * (old_count * batch_count / new_count)
            self.source_m2.add_(batch_m2 + cross)
        self.source_count.fill_(new_count)
        self.stats_ready.zero_()

    @torch.no_grad()
    def finalize_source_stats(self):
        count = int(self.source_count.item())
        if count <= 1:
            raise RuntimeError(
                'SOGC source statistics require at least two samples; '
                f'got source_count={count}')
        variance = self.source_m2 / count
        self.source_var.copy_(variance.clamp_min(self.eps))
        if not torch.isfinite(self.source_mean).all():
            raise FloatingPointError('SOGC source_mean is non-finite')
        if not torch.isfinite(self.source_var).all():
            raise FloatingPointError('SOGC source_var is non-finite')
        self.stats_ready.fill_(True)

    def set_sogc_mode(self, mode):
        if mode not in self.VALID_MODES:
            choices = ', '.join(sorted(self.VALID_MODES))
            raise ValueError(f'invalid SOGC mode {mode!r}; expected {choices}')
        self.mode = mode

    def has_valid_source_stats(self):
        if not bool(self.stats_ready.item()):
            return False
        if int(self.source_count.item()) <= 1:
            return False
        if not bool(torch.isfinite(self.source_mean).all().item()):
            return False
        if not bool(torch.isfinite(self.source_var).all().item()):
            return False
        return bool((self.source_var >= self.eps).all().item())

    def get_sogc_diagnostics(self):
        return {
            name: value.detach().clone()
            for name, value in self._last_diagnostics.items()
        }

    def _validate_response(self, r):
        expected_tail = (self.channels, 1, 1)
        if r.ndim != 4 or tuple(r.shape[1:]) != expected_tail:
            raise ValueError(
                'SOGC response must have shape [B, C, 1, 1] with '
                f'C={self.channels}; got {tuple(r.shape)}')

    def _identity_diagnostics(self, r):
        batch = r.shape[0]
        nan = torch.full((batch,), float('nan'), device=r.device)
        zero = torch.zeros((batch,), device=r.device)
        one = torch.ones((batch,), device=r.device)
        self._last_diagnostics = {
            'mean_abs_z': nan.detach(),
            'mean_sq_z': nan.detach(),
            'max_abs_z': nan.detach(),
            'z_clip_fraction': nan.detach(),
            'calibration_magnitude': zero.detach(),
            'calibration_factor_min': one.detach(),
            'calibration_factor_max': one.detach(),
        }

    def forward(self, r, a0):
        self._validate_response(r)
        if a0.shape != r.shape:
            raise ValueError(
                f'a0 shape {tuple(a0.shape)} does not match r '
                f'{tuple(r.shape)}')

        if self.mode == 'off':
            self._identity_diagnostics(r)
            return a0
        if self.mode == 'collect':
            self.update_source_stats(r)
            self._identity_diagnostics(r)
            return a0

        if not self.has_valid_source_stats():
            message = (
                'SOGC calibrate mode requires finalized source statistics '
                f'(source_count={int(self.source_count.item())}, '
                f'stats_ready={bool(self.stats_ready.item())}). Run '
                'tools/analysis/collect_sogc_source_stats.py first.')
            if self.strict_stats:
                raise RuntimeError(message)
            if not self._warned_missing_stats:
                warnings.warn(message + ' Falling back to original attention.',
                              RuntimeWarning)
                self._warned_missing_stats = True
            self._identity_diagnostics(r)
            return a0

        mean = self.source_mean.to(dtype=r.dtype)
        variance = self.source_var.to(dtype=r.dtype)
        z = (r - mean) / torch.sqrt(variance + self.eps)
        z = z.clamp(min=-self.z_clip, max=self.z_clip)
        normalized = F.layer_norm(r.flatten(1), (self.channels,))
        q = torch.cat((normalized, z.flatten(1), z.abs().flatten(1)), dim=1)
        delta = torch.tanh(self.mlp(q)).view_as(r)
        factor = 1.0 + self.gamma * delta
        attention = a0 * factor

        flat_z = z.flatten(1)
        flat_factor = factor.flatten(1)
        # Share of channels pinned at the clamp boundary. A large value means
        # the response is dominated by z_clip and variance-floor channels
        # rather than by a graded source deviation.
        saturated = (flat_z.abs() >= self.z_clip - self.eps).to(flat_z.dtype)
        self._last_diagnostics = {
            'mean_abs_z': flat_z.abs().mean(dim=1).detach(),
            'mean_sq_z': flat_z.square().mean(dim=1).detach(),
            'max_abs_z': flat_z.abs().amax(dim=1).detach(),
            'z_clip_fraction': saturated.mean(dim=1).detach(),
            'calibration_magnitude':
                (flat_factor - 1.0).abs().mean(dim=1).detach(),
            'calibration_factor_min': flat_factor.amin(dim=1).detach(),
            'calibration_factor_max': flat_factor.amax(dim=1).detach(),
        }
        return attention
