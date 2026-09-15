"""OrientedRCNN variant that adds a clean-input gate quiescence penalty.

Calibrating SLRP against the detection loss alone recovered 10-15 mAP on the
suppression and stripe corruptions but still cost 2.5 points of clean recall. The
cause is an asymmetry in the gradient, not a bug: the source split is clean and
already pretrained for 100 epochs, so clean samples sit near a loss minimum and
contribute almost nothing, while corrupted samples dominate and pull the gate
towards firing everywhere.

This detector fixes it directly. ``RandomSARInterference`` tags each sample with
``sar_interference``, so within one batch the clean samples are known. Their gate
deviation is penalised explicitly, which states "be the identity where there is
no interference" instead of hoping the sampling ratio implies it. No second
forward pass is needed.
"""

import torch
from mmrotate.models.builder import ROTATED_DETECTORS
from mmrotate.models.detectors import OrientedRCNN


@ROTATED_DETECTORS.register_module()
class SLRPCalibrationRCNN(OrientedRCNN):
    """Args:
        quiescence_weight (float): weight of the clean-sample gate penalty.
        clean_tag (str): value of ``sar_interference`` marking an uncorrupted
            sample.
        require_tag (bool): if True, raise when the tag is absent, which catches
            a pipeline that forgot to collect it and would otherwise train with
            the penalty silently disabled.
    """

    def __init__(self,
                 *args,
                 quiescence_weight=1.0,
                 clean_tag='none',
                 require_tag=True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if quiescence_weight < 0.0:
            raise ValueError('quiescence_weight must be non-negative')
        self.quiescence_weight = float(quiescence_weight)
        self.clean_tag = str(clean_tag)
        self.require_tag = bool(require_tag)

    def _clean_mask(self, img_metas, device):
        tags = []
        for meta in img_metas:
            if 'sar_interference' not in meta:
                if self.require_tag:
                    raise KeyError(
                        "img_metas is missing 'sar_interference'; add it to the "
                        'Collect meta_keys and keep RandomSARInterference in the '
                        'train pipeline')
                return None
            tags.append(meta['sar_interference'] == self.clean_tag)
        return torch.tensor(tags, dtype=torch.bool, device=device)

    def _quiescence_loss(self, mask):
        """Mean squared gate deviation over the clean samples of the batch."""
        modules = [module for _, module in self.backbone.iter_slrp_modules()]
        if not modules:
            raise RuntimeError('backbone exposes no SLRP modules to calibrate')
        total = None
        for module in modules:
            deviation = module.last_gate_deviation()
            if deviation.shape[0] != mask.shape[0]:
                raise RuntimeError(
                    f'gate deviation batch {deviation.shape[0]} does not match '
                    f'img_metas batch {mask.shape[0]}')
            selected = deviation[mask]
            term = selected.square().mean() if selected.numel() > 0 \
                else deviation.sum() * 0.0
            total = term if total is None else total + term
        return total / len(modules)

    def forward_train(self, img, img_metas, *args, **kwargs):
        losses = super().forward_train(img, img_metas, *args, **kwargs)
        if self.quiescence_weight == 0.0:
            return losses
        mask = self._clean_mask(img_metas, img.device)
        if mask is None:
            return losses
        # A batch with no clean sample contributes a true zero that still
        # carries a graph, so DDP gradient reduction stays consistent.
        losses['loss_slrp_quiescence'] = (
            self.quiescence_weight * self._quiescence_loss(mask))
        return losses
