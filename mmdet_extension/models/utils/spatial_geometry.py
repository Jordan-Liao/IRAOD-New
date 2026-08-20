"""Spatially resolved geometry observables for interference diagnosis.

Rationale
---------
``SOGCCalibrator`` derives its statistic from the globally pooled channel
response, so it is invariant to where an interference sits in the image. Every
RSAR corruption in ``tools/dataset/rsar_interference_generator.py`` is however
spatially structured: ``chaff`` is a Gaussian blob, ``point_target`` a spike,
``am_noise_*`` periodic stripes and ``smart_suppression`` a local patch. The
observables below keep the spatial axes and measure how much feature energy
lies in a separable (rank-one) subspace, which is the structure those
corruptions inject.

All observables except ``mean_energy`` are invariant to a global feature gain
``x -> a * x``; ``mean_energy`` is retained as a deliberately trivial baseline
so that a gate result can be attributed to structure rather than brightness.
"""

import torch


def valid_feature_extent(feature_shape, img_shape, input_shape):
    """Return the ``(h, w)`` of a feature map that is not test-time padding.

    ``input_shape`` is the padded tensor size fed to the backbone and
    ``img_shape`` the unpadded content size recorded by the pipeline. The
    backbone stride is inferred from the padded pair, which is exact because
    mmdet pads to a multiple of the network stride.
    """
    feat_h, feat_w = int(feature_shape[0]), int(feature_shape[1])
    pad_h, pad_w = int(input_shape[0]), int(input_shape[1])
    content_h, content_w = int(img_shape[0]), int(img_shape[1])
    for name, value in (('feature', min(feat_h, feat_w)),
                        ('padded', min(pad_h, pad_w)),
                        ('content', min(content_h, content_w))):
        if value <= 0:
            raise ValueError(f'{name} extent must be positive')
    if content_h > pad_h or content_w > pad_w:
        raise ValueError(
            f'content shape {(content_h, content_w)} exceeds padded shape '
            f'{(pad_h, pad_w)}')
    # -(-a // b) is ceil division; mmdet feature maps round up on odd sizes.
    stride_h = -(-pad_h // feat_h)
    stride_w = -(-pad_w // feat_w)
    keep_h = min(feat_h, max(1, -(-content_h // stride_h)))
    keep_w = min(feat_w, max(1, -(-content_w // stride_w)))
    return keep_h, keep_w


def energy_map(feature, eps=1e-12):
    """Channel-averaged energy map ``E[h, w]`` of a ``[C, H, W]`` feature."""
    if feature.ndim != 3:
        raise ValueError(
            f'feature must be [C, H, W]; got {tuple(feature.shape)}')
    if min(feature.shape) <= 0:
        raise ValueError('feature must not be empty')
    energy = feature.to(dtype=torch.float64).square().mean(dim=0)
    if not torch.isfinite(energy).all():
        raise FloatingPointError('feature energy contains non-finite values')
    return energy.clamp_min(eps)


def separable_decomposition(energy, eps=1e-12):
    """Split an energy map into its separable (rank-one) part and residual.

    For a non-negative map the outer product of the row and column marginals,
    normalised by the total mass, is the rank-one map that reproduces both
    marginals exactly. It is the independence model of a contingency table, and
    it is what an ideal stripe or a separable Gaussian blob collapses onto.
    """
    if energy.ndim != 2:
        raise ValueError(
            f'energy must be [H, W]; got {tuple(energy.shape)}')
    row_profile = energy.mean(dim=1)
    col_profile = energy.mean(dim=0)
    total = energy.mean()
    separable = torch.outer(row_profile, col_profile) / total.clamp_min(eps)
    return separable, energy - separable, row_profile, col_profile


def _coefficient_of_variation(profile, eps=1e-12):
    mean = profile.mean()
    return (profile.std(unbiased=False) / mean.clamp_min(eps)).item()


def _gini(values, eps=1e-12):
    flat = values.flatten()
    count = flat.numel()
    if count < 2:
        return 0.0
    ordered = torch.sort(flat).values
    index = torch.arange(
        1, count + 1, device=ordered.device, dtype=ordered.dtype)
    weighted = ((2.0 * index - count - 1.0) * ordered).sum()
    return (weighted / (count * ordered.sum().clamp_min(eps))).item()


def _dominant_spectral_peak(profile, eps=1e-12):
    """Share of a profile's AC power held by its strongest non-DC frequency.

    Periodic stripes concentrate their power in a single bin, so this is near
    one for ``am_noise_*`` and near ``2 / len(profile)`` for broadband texture.
    """
    if profile.numel() < 4:
        return float('nan')
    centered = profile - profile.mean()
    power = torch.fft.rfft(centered).abs().square()[1:]
    if power.numel() == 0:
        return float('nan')
    return (power.amax() / power.sum().clamp_min(eps)).item()


def _participation_ratio(feature, eps=1e-12):
    """Effective-rank ratio of the feature matrix, in ``(0, 1]``.

    ``tr(G)^2 / ||G||_F^2`` is the participation ratio of the eigenvalues of
    the Gram matrix, computed without an eigendecomposition. Interference that
    injects one dominant response direction drives it down. The smaller of the
    channel and spatial Gram matrices is used since both share a spectrum.
    """
    channels = feature.shape[0]
    matrix = feature.to(dtype=torch.float64).reshape(channels, -1)
    positions = matrix.shape[1]
    if min(channels, positions) < 2:
        return float('nan')
    if channels <= positions:
        gram = matrix @ matrix.t()
    else:
        gram = matrix.t() @ matrix
    trace = torch.diagonal(gram).sum()
    frobenius_sq = gram.square().sum()
    ratio = trace.square() / frobenius_sq.clamp_min(eps)
    return (ratio / min(channels, positions)).item()


def _sparse_energy_fraction(residual, multiplier, eps=1e-12):
    """Energy share surviving a scale-free soft threshold on the residual.

    This is the ``S`` term of one unrolled robust-PCA step: what remains after
    the separable component and a median-scaled threshold are removed.
    """
    magnitude = residual.abs()
    threshold = multiplier * magnitude.median()
    survived = (magnitude - threshold).clamp_min(0.0)
    return (survived.sum() / magnitude.sum().clamp_min(eps)).item()


def _local_mean(energy, span_h, span_w):
    """Stride-one local mean with replicate padding, on a ``[H, W]`` map."""
    padded = torch.nn.functional.pad(
        energy[None, None], (span_w // 2, span_w // 2,
                             span_h // 2, span_h // 2), mode='replicate')
    return torch.nn.functional.avg_pool2d(
        padded, (span_h, span_w), stride=1)[0, 0]


def _clamped_odd_span(size, window):
    span = min(window, size if size % 2 == 1 else size - 1)
    return max(1, span)


def local_cv_field(energy, window, eps=1e-12):
    """Coefficient of variation of the energy inside each local window.

    Fully developed speckle has a fixed CV, so this is a physically anchored
    homogeneity measure: interference that lifts the noise floor drives the
    scene toward homogeneity and lowers CV, while coherent spikes raise it.
    """
    span_h = _clamped_odd_span(energy.shape[0], window)
    span_w = _clamped_odd_span(energy.shape[1], window)
    mean = _local_mean(energy, span_h, span_w)
    mean_sq = _local_mean(energy.square(), span_h, span_w)
    variance = (mean_sq - mean.square()).clamp_min(0.0)
    return variance.sqrt() / mean.clamp_min(eps)


def _quantile_ratio(field, upper=0.9, eps=1e-12):
    """Upper-quantile-to-median ratio: within-scene contrast, scene-normalised.

    A global statistic dilutes a localized interference across the whole image;
    referencing the tail against the scene's own median is the CFAR view and
    keeps the observable comparable across scenes of different brightness.
    """
    flat = field.flatten()
    if flat.numel() < 4:
        return float('nan')
    median = flat.median()
    tail = torch.quantile(flat, upper)
    return (tail / median.clamp_min(eps)).item()


def _residual_peak_to_median(residual, eps=1e-12):
    """Peak-to-background ratio of the residual, the sparse-target detector."""
    magnitude = residual.abs().flatten()
    if magnitude.numel() < 4:
        return float('nan')
    return (magnitude.amax() / magnitude.median().clamp_min(eps)).item()


OBSERVABLE_NAMES = (
    'separable_residual_fraction',
    'row_profile_cv',
    'col_profile_cv',
    'profile_anisotropy',
    'row_spectral_peak',
    'col_spectral_peak',
    'energy_gini',
    'sparse_energy_fraction',
    'participation_ratio',
    'local_cv_median',
    'local_cv_contrast',
    'local_cv_median_wide',
    'local_cv_contrast_wide',
    'residual_peak_to_median',
    'mean_energy',
)


def compute_observables(feature,
                        sparse_multiplier=1.0,
                        window=3,
                        wide_window=9,
                        eps=1e-12):
    """Return the spatial geometry observables of one ``[C, H, W]`` feature.

    ``feature`` must already be cropped to its unpadded extent; see
    ``valid_feature_extent``.
    """
    energy = energy_map(feature, eps=eps)
    separable, residual, row_profile, col_profile = \
        separable_decomposition(energy, eps=eps)
    row_cv = _coefficient_of_variation(row_profile, eps=eps)
    col_cv = _coefficient_of_variation(col_profile, eps=eps)
    residual_fraction = (
        residual.abs().sum() / energy.sum().clamp_min(eps)).item()
    cv_narrow = local_cv_field(energy, window, eps=eps)
    cv_wide = local_cv_field(energy, wide_window, eps=eps)
    return {
        'local_cv_median': cv_narrow.median().item(),
        'local_cv_contrast': _quantile_ratio(cv_narrow, eps=eps),
        'local_cv_median_wide': cv_wide.median().item(),
        'local_cv_contrast_wide': _quantile_ratio(cv_wide, eps=eps),
        'residual_peak_to_median': _residual_peak_to_median(
            residual, eps=eps),
        'separable_residual_fraction': residual_fraction,
        'row_profile_cv': row_cv,
        'col_profile_cv': col_cv,
        # Signed, so horizontal and vertical stripes separate in opposite
        # directions instead of cancelling in a magnitude-only statistic.
        'profile_anisotropy': float(
            torch.log(torch.tensor(max(row_cv, eps) / max(col_cv, eps)))),
        'row_spectral_peak': _dominant_spectral_peak(row_profile, eps=eps),
        'col_spectral_peak': _dominant_spectral_peak(col_profile, eps=eps),
        'energy_gini': _gini(energy, eps=eps),
        'sparse_energy_fraction': _sparse_energy_fraction(
            residual, sparse_multiplier, eps=eps),
        'participation_ratio': _participation_ratio(feature, eps=eps),
        'mean_energy': energy.mean().item(),
    }
