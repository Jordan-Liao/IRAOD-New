# Copyright (c) 2022 MCG-NKU.
#
# Adapted from the official Strip R-CNN implementation:
# https://github.com/YXB-NKU/Strip-R-CNN/blob/main/mmrotate/models/backbones/stripnet.py
# The upstream repository is distributed under CC BY-NC 4.0; see
# THIRD_PARTY_NOTICES.md.  This adaptation changes the registry/imports for the
# MMDetection/MMRotate versions used in IRAOD, reformats the implementation,
# adds argument validation, and removes unused classification-only helpers.
# The StripNet detection compute graph and parameter names are unchanged.

import math
import warnings
from functools import partial

import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmcv.cnn.utils.weight_init import (constant_init, normal_init,
                                        trunc_normal_init)
from mmcv.runner import BaseModule
from mmdet.models.builder import BACKBONES
from timm.models.layers import DropPath
from torch.nn.modules.utils import _pair as to_2tuple


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, bias=True,
            groups=dim)

    def forward(self, x):
        return self.dwconv(x)


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class StripBlock(nn.Module):
    def __init__(self, dim, k1, k2):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial1 = nn.Conv2d(
            dim,
            dim,
            kernel_size=(k1, k2),
            stride=1,
            padding=(k1 // 2, k2 // 2),
            groups=dim)
        self.conv_spatial2 = nn.Conv2d(
            dim,
            dim,
            kernel_size=(k2, k1),
            stride=1,
            padding=(k2 // 2, k1 // 2),
            groups=dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        attn = self.conv0(x)
        attn = self.conv_spatial1(attn)
        attn = self.conv_spatial2(attn)
        attn = self.conv1(attn)
        return x * attn


class Attention(nn.Module):
    def __init__(self, d_model, k1, k2):
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = StripBlock(d_model, k1, k2)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        return x + shortcut


class Block(nn.Module):
    def __init__(self,
                 dim,
                 mlp_ratio=4.0,
                 k1=1,
                 k2=19,
                 drop=0.0,
                 drop_path=0.0,
                 act_layer=nn.GELU,
                 norm_cfg=None):
        super().__init__()
        if norm_cfg:
            self.norm1 = build_norm_layer(norm_cfg, dim)[1]
            self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        else:
            self.norm1 = nn.BatchNorm2d(dim)
            self.norm2 = nn.BatchNorm2d(dim)
        self.attn = Attention(dim, k1, k2)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones(dim), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(
            self.layer_scale_1[:, None, None] * self.attn(self.norm1(x)))
        x = x + self.drop_path(
            self.layer_scale_2[:, None, None] * self.mlp(self.norm2(x)))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=7,
                 stride=4,
                 in_chans=3,
                 embed_dim=768,
                 norm_cfg=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2))
        if norm_cfg:
            self.norm = build_norm_layer(norm_cfg, embed_dim)[1]
        else:
            self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, height, width = x.shape
        return self.norm(x), height, width


@BACKBONES.register_module()
class StripNet(BaseModule):
    """StripNet backbone from Strip R-CNN.

    The S variant used in this experiment has output channels
    ``[64, 128, 320, 512]`` and output strides ``[4, 8, 16, 32]``.
    """

    def __init__(self,
                 img_size=224,
                 in_chans=3,
                 embed_dims=(64, 128, 256, 512),
                 mlp_ratios=(8, 8, 4, 4),
                 k1s=(1, 1, 1, 1),
                 k2s=(19, 19, 19, 19),
                 drop_rate=0.0,
                 drop_path_rate=0.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=(3, 4, 6, 3),
                 num_stages=4,
                 pretrained=None,
                 init_cfg=None,
                 norm_cfg=None):
        super().__init__(init_cfg=init_cfg)
        if init_cfg and pretrained:
            raise ValueError(
                'init_cfg and pretrained cannot be set at the same time')
        if isinstance(pretrained, str):
            warnings.warn(
                'pretrained is deprecated; use init_cfg instead',
                DeprecationWarning)
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')
        if not (len(embed_dims) == len(mlp_ratios) == len(k1s) ==
                len(k2s) == len(depths) == num_stages):
            raise ValueError('all per-stage settings must match num_stages')

        self.depths = tuple(depths)
        self.num_stages = num_stages
        dpr = [
            value.item()
            for value in torch.linspace(0, drop_path_rate, sum(depths))
        ]
        cursor = 0

        for stage in range(num_stages):
            patch_embed = OverlapPatchEmbed(
                img_size=(img_size if stage == 0 else
                          img_size // (2 ** (stage + 1))),
                patch_size=7 if stage == 0 else 3,
                stride=4 if stage == 0 else 2,
                in_chans=in_chans if stage == 0 else embed_dims[stage - 1],
                embed_dim=embed_dims[stage],
                norm_cfg=norm_cfg)
            blocks = nn.ModuleList([
                Block(
                    dim=embed_dims[stage],
                    mlp_ratio=mlp_ratios[stage],
                    k1=k1s[stage],
                    k2=k2s[stage],
                    drop=drop_rate,
                    drop_path=dpr[cursor + block_index],
                    norm_cfg=norm_cfg)
                for block_index in range(depths[stage])
            ])
            norm = norm_layer(embed_dims[stage])
            cursor += depths[stage]
            setattr(self, f'patch_embed{stage + 1}', patch_embed)
            setattr(self, f'block{stage + 1}', blocks)
            setattr(self, f'norm{stage + 1}', norm)

    def init_weights(self):
        if self.init_cfg is not None:
            super().init_weights()
            return
        for module in self.modules():
            if isinstance(module, nn.Linear):
                trunc_normal_init(module, std=0.02, bias=0.0)
            elif isinstance(module, nn.LayerNorm):
                constant_init(module, val=1.0, bias=0.0)
            elif isinstance(module, nn.Conv2d):
                fan_out = (module.kernel_size[0] * module.kernel_size[1] *
                           module.out_channels)
                fan_out //= module.groups
                normal_init(
                    module,
                    mean=0,
                    std=math.sqrt(2.0 / fan_out),
                    bias=0)

    def forward_features(self, x):
        batch_size = x.shape[0]
        outputs = []
        for stage in range(self.num_stages):
            patch_embed = getattr(self, f'patch_embed{stage + 1}')
            blocks = getattr(self, f'block{stage + 1}')
            norm = getattr(self, f'norm{stage + 1}')
            x, height, width = patch_embed(x)
            for block in blocks:
                x = block(x)
            x = x.flatten(2).transpose(1, 2)
            x = norm(x)
            x = x.reshape(batch_size, height, width, -1)
            x = x.permute(0, 3, 1, 2).contiguous()
            outputs.append(x)
        return outputs

    def forward(self, x):
        return self.forward_features(x)
