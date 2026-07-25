"""Checkpoint-compatible OrthoNet with optional SOGC calibration."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import constant_init
from mmdet.models.backbones.resnet import (BasicBlock, Bottleneck, ResLayer,
                                           ResNet)
from mmdet.models.builder import BACKBONES

from ..utils import SOGCCalibrator


class OrthogonalChannelAttention(nn.Module):
    """Channel attention driven by a learned orthogonal response.

    The original RSAR checkpoint establishes this module's public state-dict
    contract: ``ortho_vector`` is a learned vector and ``fc`` is a bias-free
    two-layer excitation MLP. The response is the pooled channel vector with
    its component along the normalized learned vector removed.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        if channels <= 0:
            raise ValueError('channels must be positive')
        if reduction <= 0:
            raise ValueError('reduction must be positive')
        self.channels = int(channels)
        self.reduction = int(reduction)
        hidden_channels = max(self.channels // self.reduction, 1)
        self.ortho_vector = nn.Parameter(torch.randn(self.channels))
        self.fc = nn.Sequential(
            nn.Linear(self.channels, hidden_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, self.channels, bias=False),
            nn.Sigmoid(),
        )

    def orthogonal_response(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(
                'orthogonal attention expects [B, C, H, W] with '
                f'C={self.channels}; got {tuple(x.shape)}')
        pooled = F.adaptive_avg_pool2d(x, 1).flatten(1)
        direction = F.normalize(
            self.ortho_vector, p=2, dim=0,
            eps=torch.finfo(self.ortho_vector.dtype).eps)
        component = (pooled * direction).sum(dim=1, keepdim=True)
        response = pooled - component * direction.unsqueeze(0)
        return response.view(x.shape[0], self.channels, 1, 1)

    def enable_sogc(self, sogc_cfg):
        if hasattr(self, 'sogc'):
            raise RuntimeError('SOGC is already enabled for this attention')
        self.sogc = SOGCCalibrator(channels=self.channels, **sogc_cfg)

    def forward(self, x):
        response = self.orthogonal_response(x)
        attention = self.fc(response.flatten(1)).view_as(response)
        if hasattr(self, 'sogc'):
            attention = self.sogc(response, attention)
        return x * attention


class OrthoBasicBlock(BasicBlock):
    """MMDetection BasicBlock with checkpoint-compatible Ortho attention."""

    def __init__(self, *args, reduction=16, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn = OrthogonalChannelAttention(
            self.norm2.num_features, reduction=reduction)

    def forward(self, x):
        def _inner_forward(xx):
            identity = xx
            out = self.conv1(xx)
            out = self.norm1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.norm2(out)
            out = self.attn(out)
            if self.downsample is not None:
                identity = self.downsample(xx)
            return out + identity

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)
        return self.relu(out)


class OrthoBottleneck(Bottleneck):
    """MMDetection Bottleneck with attention after the third norm."""

    def __init__(self, *args, reduction=16, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn = OrthogonalChannelAttention(
            self.planes * self.expansion, reduction=reduction)

    def forward(self, x):
        def _inner_forward(xx):
            identity = xx
            out = self.conv1(xx)
            out = self.norm1(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(
                    out, self.after_conv1_plugin_names)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(
                    out, self.after_conv2_plugin_names)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.with_plugins:
                out = self.forward_plugin(
                    out, self.after_conv3_plugin_names)

            out = self.attn(out)
            if self.downsample is not None:
                identity = self.downsample(xx)
            return out + identity

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)
        return self.relu(out)


@BACKBONES.register_module()
class OrthoNet(ResNet):
    """ResNet-layout OrthoNet matching the supplied RSAR checkpoint.

    Stage indices are zero-based: 0/1/2/3 correspond to C2/C3/C4/C5.
    ``sogc_cfg=None`` or ``enabled=False`` creates no SOGC module or state
    keys, preserving the legacy checkpoint surface.
    """

    arch_settings = {
        18: (OrthoBasicBlock, (2, 2, 2, 2)),
        34: (OrthoBasicBlock, (3, 4, 6, 3)),
        50: (OrthoBottleneck, (3, 4, 6, 3)),
        101: (OrthoBottleneck, (3, 4, 23, 3)),
        152: (OrthoBottleneck, (3, 8, 36, 3)),
    }

    def __init__(self,
                 depth,
                 reduction=16,
                 sogc_cfg=None,
                 num_stages=4,
                 **kwargs):
        if reduction <= 0:
            raise ValueError('reduction must be positive')
        self.reduction = int(reduction)
        self._sogc_cfg = self._validate_sogc_cfg(
            sogc_cfg, num_stages=num_stages)
        super().__init__(
            depth=depth, num_stages=num_stages, **kwargs)

    @staticmethod
    def _validate_sogc_cfg(sogc_cfg, num_stages):
        if sogc_cfg is None or not sogc_cfg.get('enabled', True):
            return None
        cfg = dict(sogc_cfg)
        cfg.pop('enabled', None)
        stages = tuple(cfg.pop('stages', (2, 3)))
        selector = cfg.pop('block_selector', 'last')
        if selector != 'last':
            raise ValueError(
                "phase-one SOGC supports block_selector='last' only")
        if len(set(stages)) != len(stages):
            raise ValueError('SOGC stages must be unique')
        if any(stage < 0 or stage >= num_stages for stage in stages):
            raise ValueError(
                f'SOGC stages {stages} are invalid for '
                f'num_stages={num_stages}')
        allowed = {
            'mode', 'gamma', 'reduction', 'eps', 'z_clip', 'strict_stats'
        }
        unknown = set(cfg) - allowed
        if unknown:
            raise ValueError(f'unknown SOGC options: {sorted(unknown)}')
        return dict(
            stages=stages,
            block_selector=selector,
            module_cfg=cfg)

    def make_res_layer(self, **kwargs):
        stage_index = len(self.res_layers)
        res_layer = ResLayer(reduction=self.reduction, **kwargs)
        if (self._sogc_cfg is not None and
                stage_index in self._sogc_cfg['stages']):
            res_layer[-1].attn.enable_sogc(
                dict(self._sogc_cfg['module_cfg']))
        return res_layer

    def init_weights(self):
        already_initialized = self._is_init
        super().init_weights()
        if already_initialized or not self.zero_init_residual:
            return
        if isinstance(self.init_cfg, dict) and \
                self.init_cfg.get('type') == 'Pretrained':
            return
        # ResNet only installs this override for its exact block classes.
        for module in self.modules():
            if isinstance(module, OrthoBottleneck):
                constant_init(module.norm3, 0)
            elif isinstance(module, OrthoBasicBlock):
                constant_init(module.norm2, 0)

    def iter_sogc_calibrators(self):
        for stage_index, layer_name in enumerate(self.res_layers):
            layer = getattr(self, layer_name)
            for block_index, block in enumerate(layer):
                if hasattr(block.attn, 'sogc'):
                    yield stage_index, block_index, block.attn.sogc

    def set_sogc_mode(self, mode):
        for _, _, calibrator in self.iter_sogc_calibrators():
            calibrator.set_sogc_mode(mode)

    def reset_source_stats(self):
        for _, _, calibrator in self.iter_sogc_calibrators():
            calibrator.reset_source_stats()

    def finalize_source_stats(self):
        for _, _, calibrator in self.iter_sogc_calibrators():
            calibrator.finalize_source_stats()

    def has_valid_source_stats(self):
        calibrators = list(self.iter_sogc_calibrators())
        return bool(calibrators) and all(
            calibrator.has_valid_source_stats()
            for _, _, calibrator in calibrators)

    def get_sogc_diagnostics(self):
        by_stage = OrderedDict()
        for stage_index, _, calibrator in self.iter_sogc_calibrators():
            by_stage[f'C{stage_index + 2}'] = \
                calibrator.get_sogc_diagnostics()
        return by_stage

    def get_sogc_metadata(self):
        active = []
        for stage_index, block_index, calibrator in \
                self.iter_sogc_calibrators():
            active.append({
                'stage_index': stage_index,
                'feature_stage': f'C{stage_index + 2}',
                'block_index': block_index,
                'channels': calibrator.channels,
                'mode': calibrator.mode,
                'gamma': calibrator.gamma,
                'reduction': calibrator.reduction,
                'eps': calibrator.eps,
                'z_clip': calibrator.z_clip,
                'strict_stats': calibrator.strict_stats,
            })
        return active

    def train(self, mode=True):
        super().train(mode)
        return self
