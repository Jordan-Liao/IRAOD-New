"""Local coefficient-of-variation observables on SAR amplitude.

Fully developed speckle has a fixed CV (``sqrt(4/pi - 1) ~= 0.523`` for Rayleigh
amplitude), so the local CV field is a physically anchored homogeneity measure
rather than a learned statistic. Interference moves it in predictable
directions: a raised noise floor homogenises the scene and lowers CV, while a
coherent spike raises it locally.

These observables are deliberately computed on the input amplitude and not on
deep features, because that theoretical anchor only holds in the amplitude
domain. Every observable except ``amplitude_mean`` is invariant to a global gain
``x -> a * x``; ``amplitude_mean`` is kept as a trivial brightness baseline.

RSAR is 8-bit and JPEG-compressed, so the 0.523 anchor is a motivation rather
than an exact expectation. ``cv_median`` is reported so the empirical
source-domain clutter level can be used as the reference instead.
"""

import numpy as np

OBSERVABLE_NAMES = (
    'cv_median',
    'cv_p10',
    'cv_contrast',
    'cv_below_half_fraction',
    'amplitude_peak_to_median',
    'amplitude_mean',
)

RAYLEIGH_AMPLITUDE_CV = float(np.sqrt(4.0 / np.pi - 1.0))


def _box_mean(image, window):
    """Stride-one box mean via a summed-area table, with edge replication."""
    if window < 1 or window % 2 == 0:
        raise ValueError(f'window must be a positive odd integer; got {window}')
    radius = window // 2
    padded = np.pad(image, radius, mode='edge')
    integral = padded.cumsum(axis=0).cumsum(axis=1)
    integral = np.pad(integral, ((1, 0), (1, 0)), mode='constant')
    height, width = image.shape
    total = (integral[window:window + height, window:window + width]
             - integral[0:height, window:window + width]
             - integral[window:window + height, 0:width]
             + integral[0:height, 0:width])
    return total / float(window * window)


def local_cv_field(amplitude, window, eps=1e-9):
    """Coefficient of variation of the amplitude inside each local window."""
    if amplitude.ndim != 2:
        raise ValueError(
            f'amplitude must be 2-D; got shape {amplitude.shape}')
    if min(amplitude.shape) < window:
        window = max(1, min(amplitude.shape) | 1)
        if window > min(amplitude.shape):
            window -= 2
        window = max(1, window)
    values = amplitude.astype(np.float64)
    mean = _box_mean(values, window)
    mean_sq = _box_mean(values * values, window)
    variance = np.maximum(mean_sq - mean * mean, 0.0)
    return np.sqrt(variance) / np.maximum(mean, eps)


def compute_observables(amplitude, window=5, eps=1e-9):
    """Return the amplitude-domain CV observables of one 2-D amplitude image."""
    field = local_cv_field(amplitude, window, eps=eps)
    flat = field.reshape(-1)
    median = float(np.median(flat))
    values = amplitude.astype(np.float64).reshape(-1)
    amplitude_median = float(np.median(values))
    return {
        # Absolute homogeneity level: a lifted noise floor drives this down.
        'cv_median': median,
        # Lower tail: the most homogeneous windows in the scene, which is where
        # barrage suppression shows first.
        'cv_p10': float(np.percentile(flat, 10.0)),
        # Within-scene contrast, referenced to the scene's own median so it
        # stays comparable across scenes of different brightness.
        'cv_contrast': float(np.percentile(flat, 90.0)) / max(median, eps),
        # Share of the scene that has gone substantially smoother than its own
        # clutter level.
        'cv_below_half_fraction': float((flat < 0.5 * median).mean()),
        # Peak-to-background ratio, the sparse point-target detector.
        'amplitude_peak_to_median':
            float(values.max()) / max(amplitude_median, eps),
        'amplitude_mean': float(values.mean()),
    }
