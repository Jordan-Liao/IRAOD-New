"""Pipeline transform applying a randomly drawn SAR interference.

Used to calibrate SLRP against the real detection loss on labelled source data:
the interference is synthetic and its parameters are held out from the seven
fixed evaluation specs, so no target-domain data is touched.

Placed directly after ``LoadImageFromFile`` so the corruption acts in the 8-bit
amplitude domain, before resizing and normalization, which is where it
physically occurs.
"""

import numpy as np
from mmdet.datasets.builder import PIPELINES

from tools.dataset.random_interference import apply_random_interference


@PIPELINES.register_module()
class RandomSARInterference:
    """Apply one interference from the calibration family with probability ``p``.

    Args:
        p (float): probability of corrupting a given image. Values below one keep
            clean samples in the batch, which is what stops the gate from
            learning to fire unconditionally.
        seed (int): base seed; combined with a per-call counter so the stream is
            reproducible without repeating the same corruption every epoch.
    """

    def __init__(self, p=0.5, seed=0):
        if not 0.0 <= p <= 1.0:
            raise ValueError(f'p must be in [0, 1]; got {p}')
        self.p = float(p)
        self.seed = int(seed)
        self._counter = 0

    def __call__(self, results):
        image = results.get('img')
        if image is None:
            raise KeyError('RandomSARInterference requires "img"; it must run '
                           'after LoadImageFromFile')
        self._counter += 1
        rng = np.random.default_rng(self.seed + self._counter)
        results['sar_interference'] = 'none'
        if rng.random() >= self.p:
            return results
        original_dtype = image.dtype
        corrupted, itype = apply_random_interference(image, rng)
        results['img'] = np.clip(corrupted, 0, 255).astype(original_dtype)
        results['sar_interference'] = itype
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(p={self.p}, seed={self.seed})'
