#!/usr/bin/env python
"""Render DIOR fog images with the SFOD weak and strong views side by side.

The shared flip, weak resize, and strong photometric transforms mirror
``unbiased_teacher_oriented_rcnn_selftraining_cga.py``.  The saved images are
display images: model normalization and padding are intentionally not shown.
"""

import argparse
import copy
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms

# Allow direct execution as ``python tools/visualize_...py`` from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mmrotate.datasets.pipelines import RRandomFlip, RResize
from sfod.dense_teacher_rand_aug import (
    DTGaussianBlur,
    DTRandomApply,
    DTRandomGrayscale,
    DTToNumpy,
    DTToPILImage,
)


DEFAULT_MEAN = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
DEFAULT_STD = np.asarray([58.395, 57.12, 57.375], dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize SFOD weak/strong views for DIOR images.')
    parser.add_argument('--data-root', default='/myfile/dataset/DIOR')
    parser.add_argument('--split', choices=('train', 'val', 'test'),
                        default='test')
    parser.add_argument('--corruption', default='fog')
    parser.add_argument('--out', required=True)
    parser.add_argument(
        '--image-ids', nargs='+', default=None,
        help='specific image stems, e.g. 11743; bypasses the split list')
    parser.add_argument(
        '--max-images', type=int, default=None,
        help='process only the first N images; default is the whole split')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log-interval', type=int, default=500)
    parser.add_argument('--jpeg-quality', type=int, choices=range(1, 101),
                        default=95)
    return parser.parse_args()


def validate_args(args):
    if args.max_images is not None and args.max_images <= 0:
        raise ValueError('--max-images must be positive')
    if args.log_interval < 0:
        raise ValueError('--log-interval must be non-negative')


def load_ids(split_file, max_images):
    ids = [line.strip() for line in split_file.read_text().splitlines()
           if line.strip()]
    return ids if max_images is None else ids[:max_images]


def make_shared_pipeline():
    return RRandomFlip(
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version='le90')


def make_weak_pipeline():
    return RResize(img_scale=(800, 800))


def make_strong_pipeline():
    # These are the exact operations and probabilities from the SFOD config.
    return [
        DTToPILImage(),
        DTRandomApply(
            operations=[transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
            p=0.8),
        DTRandomGrayscale(p=0.2),
        DTRandomApply(
            operations=[DTGaussianBlur(rad_range=[0.1, 2.0])]),
        DTToNumpy(),
    ]


def apply_shared(image, transform):
    results = {
        'img': image.copy(),
        'img_shape': image.shape,
        'img_fields': ['img'],
        'bbox_fields': [],
    }
    return transform(results)['img']


def apply_weak(image, transform):
    results = {
        'img': image.copy(),
        'img_shape': image.shape,
        'img_fields': ['img'],
        'bbox_fields': [],
    }
    return transform(results)['img']


def apply_strong(image, transforms_):
    results = {
        'img': image.copy(),
        'img_fields': ['img'],
    }
    for transform in transforms_:
        results = transform(results)
    return results['img']


def display_image(image):
    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.shape[2] == 4:
        image = image[:, :, :3]
    return Image.fromarray(image, mode='RGB')


def make_montage(original, weak, strong):
    panels = [display_image(panel) for panel in (original, weak, strong)]
    width = max(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    panels = [panel.resize((width, height), Image.Resampling.BILINEAR)
              if panel.size != (width, height) else panel for panel in panels]
    label_height = 36
    canvas = Image.new('RGB', (width * 3, height + label_height), 'white')
    labels = ('fog input', 'weak view', 'strong view')
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 20)
    except OSError:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(canvas)
    for index, (label, panel) in enumerate(zip(labels, panels)):
        x = index * width
        canvas.paste(panel, (x, label_height))
        draw.text((x + 8, 7), label, fill='black', font=font)
    return canvas


def main():
    args = parse_args()
    validate_args(args)
    data_root = Path(args.data_root).resolve()
    image_dir = data_root / 'Corruption' / f'JPEGImages-{args.corruption}'
    split_file = data_root / 'ImageSets' / 'Main' / f'{args.split}.txt'
    out_root = Path(args.out).resolve()
    if not image_dir.is_dir():
        raise FileNotFoundError(f'Image directory not found: {image_dir}')
    if args.image_ids is None and not split_file.is_file():
        raise FileNotFoundError(f'Split file not found: {split_file}')
    if out_root.exists() and any(out_root.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {out_root}')
    out_root.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    shared = make_shared_pipeline()
    weak = make_weak_pipeline()
    strong = make_strong_pipeline()
    image_ids = (list(dict.fromkeys(args.image_ids))
                 if args.image_ids is not None
                 else load_ids(split_file, args.max_images))
    if args.max_images is not None and args.image_ids is not None:
        image_ids = image_ids[:args.max_images]
    manifest_path = out_root / 'manifest.csv'
    missing = []
    written = 0

    with manifest_path.open('w', newline='', encoding='utf-8') as manifest:
        writer = csv.DictWriter(
            manifest, fieldnames=('image_id', 'input_path', 'output_path'))
        writer.writeheader()
        for index, image_id in enumerate(image_ids, start=1):
            image_path = image_dir / f'{image_id}.jpg'
            if not image_path.is_file():
                # DIOR normally uses .jpg, but tolerate another image suffix.
                candidates = list(image_dir.glob(f'{image_id}.*'))
                image_path = candidates[0] if candidates else image_path
            if not image_path.is_file():
                missing.append(image_id)
                continue
            with Image.open(image_path) as source:
                original = np.asarray(source.convert('RGB'))
            shared_image = apply_shared(original, shared)
            weak_image = apply_weak(shared_image, weak)
            strong_image = apply_strong(copy.deepcopy(shared_image), strong)
            montage = make_montage(original, weak_image, strong_image)
            output_path = out_root / f'{image_id}.jpg'
            montage.save(output_path, quality=args.jpeg_quality, optimize=True)
            writer.writerow({
                'image_id': image_id,
                'input_path': str(image_path),
                'output_path': str(output_path),
            })
            written += 1
            if args.log_interval and index % args.log_interval == 0:
                print(f'[visualize_dior_weak_strong_aug] '
                      f'{index}/{len(image_ids)} images, {written} written')

    print(f'[visualize_dior_weak_strong_aug] requested images: {len(image_ids)}')
    print(f'[visualize_dior_weak_strong_aug] images written: {written}')
    print(f'[visualize_dior_weak_strong_aug] missing images: {len(missing)}')
    if missing:
        print(f'[visualize_dior_weak_strong_aug] missing ids: {missing[:20]}')
    print(f'[visualize_dior_weak_strong_aug] output: {out_root}')
    print(f'[visualize_dior_weak_strong_aug] manifest: {manifest_path}')


if __name__ == '__main__':
    main()
