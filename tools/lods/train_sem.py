"""Train the LODS Style Enhancement Module on unlabeled target images.

Port of the official ``Enhance/train.py`` (generic scenario branch: content and
style images both come from the target training images). No labels are read.

    python tools/lods/train_sem.py --target-dir DIR --encoder ENC.pt \
        --work-dir OUT --seed 42 [--max-iter 160000]
"""
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.utils.data as data

sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
from sfod.lods_sem import PIXEL_MEANS_BGR, StyleEnhancementModule  # noqa: E402

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


class TargetCrops(data.Dataset):
    """train.py:22-54: resize to the detector input scale, RandomCrop 128,
    BGR, random h-flip, minus PIXEL_MEANS."""

    def __init__(self, root, scale=(800, 800), crop=128):
        self.paths = sorted(osp.join(r, f) for r, _, fs in os.walk(root)
                            for f in fs if f.lower().endswith(IMG_EXTS))
        if not self.paths:
            raise FileNotFoundError(f'no images under {root}')
        self.scale, self.crop = scale, crop

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = cv2.imread(self.paths[i], cv2.IMREAD_COLOR)  # BGR
        h, w = img.shape[:2]
        # mmrotate RResize(img_scale=(800, 800), keep_ratio=True) rescale rule
        r = min(max(self.scale) / max(h, w), min(self.scale) / min(h, w))
        img = cv2.resize(img, (int(w * r + 0.5), int(h * r + 0.5)),
                         interpolation=cv2.INTER_LINEAR)
        h, w = img.shape[:2]
        # torch RNG: DataLoader reseeds it per worker (numpy's is forked as-is)
        y = int(torch.randint(0, h - self.crop + 1, ()))
        x = int(torch.randint(0, w - self.crop + 1, ()))
        img = img[y:y + self.crop, x:x + self.crop]
        if torch.rand(()) >= 0.5:
            img = img[:, ::-1]
        img = img.astype(np.float32) - np.array(PIXEL_MEANS_BGR, np.float32)
        return torch.from_numpy(img.transpose(2, 0, 1).copy())


def infinite(loader):
    while True:
        yield from loader


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--target-dir', required=True)
    p.add_argument('--encoder', required=True, help='enc_{i}.* VGG16 conv1_1..conv4_1')
    p.add_argument('--work-dir', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--workers', type=int, default=16)
    # official defaults, Enhance/train.py:82-94
    p.add_argument('--lr-decay', type=float, default=5e-5)
    p.add_argument('--max-iter', type=int, default=160000)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr-decoder', type=float, default=1e-4)
    p.add_argument('--lr-fcs', type=float, default=1e-4)
    p.add_argument('--style-weight', type=float, default=50)
    p.add_argument('--content-weight', type=float, default=1)
    p.add_argument('--content-style-weight', type=float, default=1)
    p.add_argument('--constrain-weight', type=float, default=1)
    p.add_argument('--before-fcs-steps', type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.work_dir, exist_ok=True)
    device = torch.device('cuda')

    net = StyleEnhancementModule()
    net.load_encoder(args.encoder)
    net.train().to(device)
    ds = TargetCrops(args.target_dir)

    def loader():
        # official: two independent InfiniteSamplerWrapper streams
        return infinite(data.DataLoader(
            ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
            num_workers=args.workers, persistent_workers=True))
    content_iter, style_iter = loader(), loader()

    dec_params = [q for i in range(1, 5) for q in getattr(net, f'dec_{i}').parameters()]
    fc_params = list(net.fc1.parameters()) + list(net.fc2.parameters())
    opt1 = torch.optim.Adam(dec_params, lr=args.lr_decoder)
    opt2 = torch.optim.Adam(fc_params, lr=args.lr_fcs)

    log = open(osp.join(args.work_dir, 'train_sem.log'), 'a')
    t0 = time.time()
    for i in range(args.max_iter):
        for g in opt1.param_groups:
            g['lr'] = args.lr_decoder / (1.0 + args.lr_decay * i)
        content = next(content_iter).to(device, non_blocking=True)
        style = next(style_iter).to(device, non_blocking=True)

        loss_c, loss_const = net.loss_decoder(content, style)
        loss1 = args.content_weight * loss_c + args.constrain_weight * loss_const
        opt1.zero_grad(); opt2.zero_grad()
        loss1.backward()
        opt1.step()

        loss2 = torch.zeros(())
        if i >= args.before_fcs_steps:
            for g in opt2.param_groups:
                g['lr'] = args.lr_fcs / (1.0 + args.lr_decay * (i - args.before_fcs_steps))
            loss_s_1, loss_s_2 = net.loss_fcs(content, style)
            loss2 = args.style_weight * loss_s_1 + args.content_style_weight * loss_s_2
            opt1.zero_grad(); opt2.zero_grad()
            loss2.backward()
            opt2.step()

        if not (torch.isfinite(loss1) and torch.isfinite(loss2)):
            raise FloatingPointError(f'non-finite SEM loss at iter {i + 1}')
        if (i + 1) % 1000 == 0 or i == 0:
            rec = dict(iter=i + 1, loss_c=loss_c.item(), loss_const=loss_const.item(),
                       loss_style=float(loss2), sec=round(time.time() - t0, 1))
            log.write(json.dumps(rec) + '\n'); log.flush()
            print(rec, flush=True)

    torch.save(net.state_dict(), osp.join(args.work_dir, f'sem_iter_{args.max_iter}.pth'))
    print('SEM_DONE', flush=True)


if __name__ == '__main__':
    main()
