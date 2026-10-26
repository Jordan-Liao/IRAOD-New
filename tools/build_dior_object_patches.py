#!/usr/bin/env python
"""Crop DIOR objects from XML oriented-box annotations."""

import argparse
import csv
import math
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


CLASSES = (
    'airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
    'chimney', 'dam', 'expressway-service-area',
    'expressway-toll-station', 'golffield', 'groundtrackfield', 'harbor',
    'overpass', 'ship', 'stadium', 'storagetank', 'tenniscourt',
    'trainstation', 'vehicle', 'windmill')
CLASS_TO_ID = {name: index for index, name in enumerate(CLASSES)}
IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
METADATA_FIELDS = (
    'patch_path', 'image_name', 'ann_file', 'split', 'corruption', 'class_name',
    'class_id', 'object_index', 'crop_mode', 'crop_expand', 'x1', 'y1', 'x2',
    'y2', 'poly_x1', 'poly_y1', 'poly_x2', 'poly_y2', 'poly_x3', 'poly_y3',
    'poly_x4', 'poly_y4')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build class-organized DIOR object patches from XML GT.')
    parser.add_argument('--data-root', default='/myfile/dataset/DIOR')
    parser.add_argument('--split', choices=('train', 'val', 'test'),
                        required=True)
    parser.add_argument(
        '--corruptions', nargs='*', default=None,
        help=('corruption names without the JPEGImages- prefix; omit this '
              'option to crop clean images'))
    parser.add_argument('--out', required=True)
    parser.add_argument('--crop-modes', nargs='+', choices=('aabb', 'rotated'),
                        default=('aabb',))
    parser.add_argument('--crop-expands', nargs='+', type=float, default=(0.2,))
    parser.add_argument(
        '--classes', nargs='+', choices=CLASSES, default=None,
        help='classes to crop; defaults to all 20 DIOR classes')
    parser.add_argument(
        '--max-per-class', type=int, default=None,
        help='maximum source objects per class across all selected domains')
    parser.add_argument(
        '--log-interval', type=int, default=500,
        help='print progress every N source images; use 0 to disable')
    parser.add_argument('--force-rgb', type=int, choices=(0, 1), default=1)
    return parser.parse_args()


def validate_args(args):
    if args.max_per_class is not None and args.max_per_class <= 0:
        raise ValueError('--max-per-class must be positive')
    if args.log_interval < 0:
        raise ValueError('--log-interval must be non-negative')
    if not args.crop_expands:
        raise ValueError('--crop-expands cannot be empty')
    if any(not math.isfinite(value) or value < 0
           for value in args.crop_expands):
        raise ValueError('--crop-expands values must be finite and non-negative')


def find_image(image_dir, stem):
    for suffix in IMAGE_SUFFIXES:
        path = image_dir / f'{stem}{suffix}'
        if path.is_file():
            return path
    return None


def read_objects(xml_path):
    root = ET.parse(xml_path).getroot()
    objects = []
    for object_index, obj in enumerate(root.findall('object')):
        class_name = obj.findtext('name', '').strip().lower()
        if class_name not in CLASS_TO_ID:
            continue
        box = obj.find('robndbox')
        if box is None:
            continue
        field_names = (
            'x_left_top', 'y_left_top', 'x_right_top', 'y_right_top',
            'x_right_bottom', 'y_right_bottom', 'x_left_bottom',
            'y_left_bottom')
        try:
            polygon = np.asarray(
                [float(box.findtext(name)) for name in field_names],
                dtype=np.float32).reshape(4, 2)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(polygon).all():
            continue
        objects.append((object_index, polygon, class_name))
    return objects


def reached_limit(counts, selected_classes, max_per_class):
    return (max_per_class is not None
            and all(counts[name] >= max_per_class
                    for name in selected_classes))


def expanded_polygon(polygon, expand):
    center = polygon.mean(axis=0, keepdims=True)
    return center + (polygon - center) * (1.0 + 2.0 * expand)


def crop_aabb(image, polygon, expand):
    polygon = expanded_polygon(polygon, expand)
    x1 = max(0, int(math.floor(float(polygon[:, 0].min()))))
    y1 = max(0, int(math.floor(float(polygon[:, 1].min()))))
    x2 = min(image.width, int(math.ceil(float(polygon[:, 0].max()))))
    y2 = min(image.height, int(math.ceil(float(polygon[:, 1].max()))))
    if x2 <= x1 or y2 <= y1:
        return None, None
    return image.crop((x1, y1, x2, y2)), (x1, y1, x2, y2)


def order_polygon(polygon):
    center = polygon.mean(axis=0)
    angles = np.arctan2(polygon[:, 1] - center[1],
                        polygon[:, 0] - center[0])
    ordered = polygon[np.argsort(angles)]
    start = int(np.argmin(ordered.sum(axis=1)))
    return np.roll(ordered, -start, axis=0)


def crop_rotated(image, polygon, expand):
    source = order_polygon(expanded_polygon(polygon, expand))
    width = max(np.linalg.norm(source[1] - source[0]),
                np.linalg.norm(source[2] - source[3]))
    height = max(np.linalg.norm(source[3] - source[0]),
                 np.linalg.norm(source[2] - source[1]))
    out_width = max(1, int(round(float(width))))
    out_height = max(1, int(round(float(height))))
    resample = getattr(Image, 'Resampling', Image).BICUBIC
    patch = image.transform(
        (out_width, out_height), Image.QUAD,
        data=tuple(float(value) for value in source.reshape(-1)),
        resample=resample)
    bounds = (float(source[:, 0].min()), float(source[:, 1].min()),
              float(source[:, 0].max()), float(source[:, 1].max()))
    return patch, bounds


def image_domains(data_root, split, corruptions):
    if corruptions:
        return [
            (name, data_root / 'Corruption' / f'JPEGImages-{name}')
            for name in corruptions
        ]
    clean_dir = ('JPEGImages-test' if split == 'test'
                 else 'JPEGImages-trainval')
    return [('clean', data_root / clean_dir)]


def output_is_empty(out_root):
    return not out_root.exists() or not any(out_root.iterdir())


def main():
    args = parse_args()
    validate_args(args)
    data_root = Path(args.data_root).resolve()
    out_root = Path(args.out).resolve()
    split_file = data_root / 'ImageSets' / 'Main' / f'{args.split}.txt'
    ann_dir = data_root / 'Annotations' / 'Oriented Bounding Boxes'
    if not split_file.is_file():
        raise FileNotFoundError(f'Split file not found: {split_file}')
    if not ann_dir.is_dir():
        raise FileNotFoundError(f'Annotation directory not found: {ann_dir}')
    if not output_is_empty(out_root):
        raise FileExistsError(f'Output directory is not empty: {out_root}')

    domains = image_domains(data_root, args.split, args.corruptions)
    for domain, image_dir in domains:
        if not image_dir.is_dir():
            raise FileNotFoundError(
                f'Image directory for {domain} not found: {image_dir}')

    out_root.mkdir(parents=True, exist_ok=True)
    selected_classes = tuple(args.classes or CLASSES)
    image_ids = [line.strip() for line in split_file.read_text().splitlines()
                 if line.strip()]
    object_counts = Counter()
    patch_counts = Counter()
    missing_images = Counter()
    skipped_objects = 0
    metadata_path = out_root / 'metadata.csv'

    with metadata_path.open('w', newline='', encoding='utf-8') as metadata:
        writer = csv.DictWriter(metadata, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        for domain, image_dir in domains:
            if reached_limit(
                    object_counts, selected_classes, args.max_per_class):
                break
            for image_index, image_id in enumerate(image_ids, start=1):
                if reached_limit(
                        object_counts, selected_classes, args.max_per_class):
                    break
                if (args.log_interval
                        and image_index % args.log_interval == 0):
                    print(f'[build_dior_object_patches] {domain}: '
                          f'{image_index}/{len(image_ids)} images, '
                          f'{sum(patch_counts.values())} patches')
                image_path = find_image(image_dir, image_id)
                if image_path is None:
                    missing_images[domain] += 1
                    continue
                xml_path = ann_dir / f'{image_id}.xml'
                if not xml_path.is_file():
                    raise FileNotFoundError(f'Annotation not found: {xml_path}')
                objects = read_objects(xml_path)
                with Image.open(image_path) as source_image:
                    image = (source_image.convert('RGB') if args.force_rgb
                             else source_image.copy())
                for object_index, polygon, class_name in objects:
                    if class_name not in selected_classes:
                        continue
                    if (args.max_per_class is not None
                            and object_counts[class_name]
                            >= args.max_per_class):
                        continue
                    object_counts[class_name] += 1
                    for crop_mode in args.crop_modes:
                        for crop_expand in args.crop_expands:
                            if crop_mode == 'aabb':
                                patch, bounds = crop_aabb(
                                    image, polygon, crop_expand)
                            else:
                                patch, bounds = crop_rotated(
                                    image, polygon, crop_expand)
                            if patch is None:
                                skipped_objects += 1
                                continue
                            class_dir = out_root / crop_mode / class_name
                            class_dir.mkdir(parents=True, exist_ok=True)
                            filename = (
                                f'{domain}_{args.split}_{image_id}_'
                                f'{object_index}_e{crop_expand:g}.png')
                            patch_path = class_dir / filename
                            patch.save(patch_path)
                            patch_counts[class_name] += 1
                            row = {
                                'patch_path': str(patch_path),
                                'image_name': image_path.name,
                                'ann_file': str(xml_path),
                                'split': args.split,
                                'corruption': domain,
                                'class_name': class_name,
                                'class_id': CLASS_TO_ID[class_name],
                                'object_index': object_index,
                                'crop_mode': crop_mode,
                                'crop_expand': crop_expand,
                                'x1': bounds[0], 'y1': bounds[1],
                                'x2': bounds[2], 'y2': bounds[3],
                            }
                            for point_index, (x, y) in enumerate(
                                    polygon, start=1):
                                row[f'poly_x{point_index}'] = float(x)
                                row[f'poly_y{point_index}'] = float(y)
                            writer.writerow(row)

    print(f'[build_dior_object_patches] images in split: {len(image_ids)}')
    print(f'[build_dior_object_patches] source objects: '
          f'{sum(object_counts.values())}')
    print(f'[build_dior_object_patches] patches written: '
          f'{sum(patch_counts.values())}')
    print(f'[build_dior_object_patches] per class: '
          f'{dict(sorted(patch_counts.items()))}')
    print(f'[build_dior_object_patches] missing images: '
          f'{dict(sorted(missing_images.items()))}')
    print(f'[build_dior_object_patches] degenerate crops skipped: '
          f'{skipped_objects}')
    print(f'[build_dior_object_patches] metadata: {metadata_path}')


if __name__ == '__main__':
    main()
