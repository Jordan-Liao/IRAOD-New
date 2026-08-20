"""Randomized interference family for SLRP calibration.

Deliberately disjoint from the seven fixed evaluation specs in
``default_rsar_corruptions()``: every geometric and radiometric parameter is
drawn from a range that excludes, or only marginally overlaps, the evaluation
value. The physical mechanism is shared -- stripes are still rank-one, blobs are
still smooth and low-rank -- but the parameters are not, so a module calibrated
on this family and evaluated on the fixed set is making a generalization claim
rather than memorizing the test distribution.

Evaluation values held out here:

- ``chaff``: centred at (0.5, 0.5) with ``cloudSize=[0.25, 0.35]``
- ``smart_suppression``: centred, ``noiseSize=[0.25, 0.25]``, ``noiseSigma=200``
- ``am_noise_*``: ``lineFrequency=0.05``, ``lineWidth=10``, ``noiseSigma=200``
- ``gaussian_white_noise``: ``noiseVariance=25``
- ``noise_suppression``: ``noiseVariance=50``, ``blurKsize=5``
- ``point_target``: centred, ``intensity=200``, ``sigmaFrac=0.01``
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.dataset.rsar_interference_generator import (  # noqa: E402
    add_interference)

TRAIN_FAMILIES = (
    'chaff', 'gaussian_white_noise', 'point_target', 'noise_suppression',
    'am_noise_horizontal', 'am_noise_vertical', 'smart_suppression')


def _offset_location(rng):
    """Off-centre placement; the evaluation set always uses the exact centre."""
    return [[float(rng.uniform(0.2, 0.8)), float(rng.uniform(0.2, 0.8))]]


def sample_interference(rng):
    """Draw one ``(itype, params)`` pair from the calibration family."""
    itype = str(rng.choice(TRAIN_FAMILIES))
    if itype == 'chaff':
        params = dict(
            locations=_offset_location(rng),
            cloudSize=[float(rng.uniform(0.10, 0.22)),
                       float(rng.uniform(0.10, 0.22))],
            noiseSigma=float(rng.uniform(150.0, 450.0)),
            densitySigmaFactor=float(rng.uniform(0.15, 0.40)))
    elif itype == 'gaussian_white_noise':
        # Straddles the evaluation variance of 25 without containing it as a
        # mode; the calibration signal must not depend on hitting it exactly.
        params = dict(noiseVariance=float(rng.uniform(8.0, 60.0)))
    elif itype == 'point_target':
        count = int(rng.integers(1, 4))
        params = dict(
            locations=[[float(rng.uniform(0.15, 0.85)),
                        float(rng.uniform(0.15, 0.85))]
                       for _ in range(count)],
            intensity=float(rng.uniform(120.0, 320.0)),
            sigmaFrac=float(rng.uniform(0.004, 0.030)))
    elif itype == 'noise_suppression':
        params = dict(
            noiseVariance=float(rng.uniform(20.0, 90.0)),
            blurKsize=int(rng.choice([3, 7, 9])))
    elif itype in ('am_noise_horizontal', 'am_noise_vertical'):
        params = dict(
            lineFrequency=float(rng.uniform(0.02, 0.12)),
            # Excludes the evaluation width of 10 exactly.
            lineWidth=int(rng.choice([4, 5, 6, 7, 8, 9, 11, 12, 14, 16, 17])),
            baseIntensity=float(rng.uniform(90.0, 210.0)),
            noiseSigma=float(rng.uniform(120.0, 300.0)),
            blendFactor=float(rng.uniform(0.4, 0.9)))
    elif itype == 'smart_suppression':
        params = dict(
            locations=_offset_location(rng),
            noiseSize=[float(rng.uniform(0.12, 0.22)),
                       float(rng.uniform(0.12, 0.22))],
            noiseSigma=float(rng.uniform(120.0, 320.0)),
            blurKsize=int(rng.choice([3, 7, 9])))
    else:
        raise ValueError(f'unhandled calibration family member: {itype}')
    return itype, params


def apply_random_interference(image, rng):
    """Apply one randomly drawn interference to an ``uint8`` image array."""
    itype, params = sample_interference(rng)
    seed = int(rng.integers(0, 2 ** 31 - 1))
    return add_interference(
        image, itype=itype, params=params, seed=seed), itype
