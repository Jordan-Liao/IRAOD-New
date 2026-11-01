"""LODS Style Enhancement Module (SEM), ported from the official code.

Official sources (Flashkong/Source-Free-Object-Detection-by-Learning-to-Overlook-Domain-Style
@ 2dc11b1): ``Enhance/net.py`` (architecture, training losses),
``Enhance/function.py`` (fc-modulated AdaIN) and
``Overlook/enhance/enhance_base.py`` (inference ``style_transfer`` / ``add_style``).

Images enter and leave this module in the official working space: BGR,
0-255, minus ``PIXEL_MEANS`` (py-faster-rcnn caffe convention).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Overlook/lib/model/utils/config.py:257 (also Enhance/train.py:41).
PIXEL_MEANS_BGR = (102.9801, 115.9465, 122.7717)


def build_vgg16_encoder() -> nn.Sequential:
    """VGG16 features[:19] (conv1_1..relu4_1), ceil_mode pools (net.py:29-34)."""
    layers = []
    in_ch = 3
    for v in (64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512):
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
        else:
            layers += [nn.Conv2d(in_ch, v, 3, padding=1), nn.ReLU(inplace=True)]
            in_ch = v
    return nn.Sequential(*layers)


def build_decoder() -> nn.Sequential:
    """Official decoder (net.py:6-27)."""
    def conv(i, o):
        return nn.Conv2d(i, o, kernel_size=3, stride=1, padding=1)
    up = lambda: nn.Upsample(scale_factor=2, mode='nearest')  # noqa: E731
    relu = lambda: nn.ReLU(inplace=True)  # noqa: E731
    return nn.Sequential(
        conv(512, 256), relu(), up(),
        conv(256, 256), relu(), conv(256, 256), relu(), conv(256, 128), relu(), up(),
        conv(128, 128), relu(), conv(128, 64), relu(), up(),
        conv(64, 64), relu(), conv(64, 3))


def build_fc() -> nn.Sequential:
    """fc1 / fc2 (net.py:35-36)."""
    return nn.Sequential(nn.Linear(1024, 512), nn.ReLU(inplace=True), nn.Linear(512, 512))


def calc_mean_std(feat, eps=1e-5):
    n, c = feat.shape[:2]
    var = feat.reshape(n, c, -1).var(dim=2) + eps
    return feat.reshape(n, c, -1).mean(dim=2).view(n, c, 1, 1), var.sqrt().view(n, c, 1, 1)


def adain_fc(content_feat, style_feat, fc1, fc2):
    """function.py:15-29: AdaIN whose target stats are fc(style ++ content)."""
    s_mean, s_std = calc_mean_std(style_feat)
    c_mean, c_std = calc_mean_std(content_feat)
    normalized = (content_feat - c_mean) / c_std
    new_mean = fc1(torch.cat((s_mean, c_mean), 1).flatten(1))[..., None, None]
    new_std = fc2(torch.cat((s_std, c_std), 1).flatten(1))[..., None, None]
    return normalized * new_std + new_mean


class StyleEnhancementModule(nn.Module):
    """Encoder (frozen VGG16) + decoder + fc1/fc2, official split points."""

    def __init__(self):
        super().__init__()
        enc = list(build_vgg16_encoder().children())
        self.enc_1 = nn.Sequential(*enc[:2])     # relu1_1
        self.enc_2 = nn.Sequential(*enc[2:7])    # relu2_1
        self.enc_3 = nn.Sequential(*enc[7:12])   # relu3_1
        self.enc_4 = nn.Sequential(*enc[12:])    # relu4_1
        dec = list(build_decoder().children())
        self.dec_1 = nn.Sequential(*dec[:7])
        self.dec_2 = nn.Sequential(*dec[7:12])
        self.dec_3 = nn.Sequential(*dec[12:17])
        self.dec_4 = nn.Sequential(*dec[17:])
        self.fc1 = build_fc()
        self.fc2 = build_fc()
        for i in range(1, 5):
            for p in getattr(self, f'enc_{i}').parameters():
                p.requires_grad = False

    def encode_with_intermediate(self, x):
        out = []
        for i in range(1, 5):
            x = getattr(self, f'enc_{i}')(x)
            out.append(x)
        return out

    def decode_with_intermediate(self, x):
        out = []
        for i in range(1, 5):
            x = getattr(self, f'dec_{i}')(x)
            out.append(x)
        return out

    @staticmethod
    def _style_loss(a, b):
        am, astd = calc_mean_std(a)
        bm, bstd = calc_mean_std(b)
        return F.mse_loss(am, bm) + F.mse_loss(astd, bstd)

    def loss_decoder(self, content, style, alpha=1.0):
        """net.py:99-119 (flag=0, generic branch): loss_c, loss_const."""
        style_feats = self.encode_with_intermediate(style)
        content_feats = self.encode_with_intermediate(content)
        t = adain_fc(content_feats[-1], style_feats[-1], self.fc1, self.fc2)
        t = alpha * t + (1 - alpha) * content_feats[-1]
        g_t_feats = self.encode_with_intermediate(self.decode_with_intermediate(t)[-1])
        loss_c = F.mse_loss(g_t_feats[-1], t.detach())
        deco = self.decode_with_intermediate(content_feats[-1])
        loss_const = F.mse_loss(deco[-2], content_feats[0])
        for i in range(1, 3):
            loss_const = loss_const + F.mse_loss(deco[-(i + 2)], content_feats[i])
        loss_const = loss_const + F.mse_loss(deco[-1], content)
        return loss_c, loss_const

    def loss_fcs(self, content, style, alpha=1.0):
        """net.py:120-140 (flag=1, generic branch): loss_s_1, loss_s_2."""
        style_feats = self.encode_with_intermediate(style)
        content_feats = self.encode_with_intermediate(content)
        t = adain_fc(content_feats[-1], style_feats[-1], self.fc1, self.fc2)
        t = alpha * t + (1 - alpha) * content_feats[-1]
        g_t_feats = self.encode_with_intermediate(self.decode_with_intermediate(t)[-1])
        loss_s_1 = sum(self._style_loss(g_t_feats[i], style_feats[i]) for i in range(4))
        loss_s_2 = sum(self._style_loss(g_t_feats[i], content_feats[i]) for i in range(4))
        return loss_s_1, loss_s_2

    @torch.no_grad()
    def style_transfer(self, content, style, alpha):
        """enhance_base.py:83-101 for one (1,3,H,W) content / style pair."""
        size = content.shape
        c = self.enc_4(self.enc_3(self.enc_2(self.enc_1(content))))
        s = self.enc_4(self.enc_3(self.enc_2(self.enc_1(style))))
        feat = adain_fc(c, s, self.fc1, self.fc2)
        feat = feat * alpha + c * (1 - alpha)
        feat = self.dec_4(self.dec_3(self.dec_2(self.dec_1(feat))))
        if feat.shape != size:
            # official: cv2.resize(..., INTER_LINEAR) back to the content size
            feat = F.interpolate(feat, size=size[2:], mode='bilinear', align_corners=False)
        return feat

    @torch.no_grad()
    def add_style(self, content, style, alpha):
        """enhance_base.py:14-40: transfer, then clip to [0, 255] pixel range."""
        out = self.style_transfer(content, style, alpha)
        means = out.new_tensor(PIXEL_MEANS_BGR).view(1, 3, 1, 1)
        return ((out + means).clamp(0, 255) - means).detach()

    def load_encoder(self, path):
        """Load conv1_1..conv4_1 keyed ``enc_{i}.{j}.*`` (Oxford caffe VGG16, BGR)."""
        state = torch.load(path, map_location='cpu')
        missing, unexpected = self.load_state_dict(state, strict=False)
        enc_missing = [k for k in missing if k.startswith('enc_')]
        if enc_missing or unexpected:
            raise RuntimeError(f'encoder load mismatch: missing={enc_missing} unexpected={unexpected}')
