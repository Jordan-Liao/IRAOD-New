# sfod/semi_dota_dataset.py
import os
import os.path as osp
import copy
import random
import collections

import numpy as np
from torch.utils.data import Dataset
from mmcv.utils import build_from_cfg
from mmrotate.datasets.builder import ROTATED_DATASETS, ROTATED_PIPELINES

try:
    from mmrotate.datasets import DOTADataset as _MMRotateDOTADataset
except Exception:
    from mmrotate.datasets.dota import DOTADataset as _MMRotateDOTADataset


class Compose:
    def __init__(self, transforms):
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        for t in transforms:
            if isinstance(t, dict):
                t = build_from_cfg(t, ROTATED_PIPELINES)
            elif not callable(t):
                raise TypeError('transform must be callable or a dict')
            self.transforms.append(t)

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        return f'{self.__class__.__name__}({self.transforms})'


IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
IMG_EXT_PRIORITY = {ext: i for i, ext in enumerate(IMG_EXTS)}


def _image_sort_key(filename):
    ext = osp.splitext(filename)[1].lower()
    return IMG_EXT_PRIORITY.get(ext, len(IMG_EXT_PRIORITY)), filename.lower()


def _build_stem_to_filename(img_dir):
    stem_to_files = collections.defaultdict(list)
    if not img_dir or not osp.isdir(img_dir):
        return {}

    for root, _, files in os.walk(img_dir):
        for filename in files:
            ext = osp.splitext(filename)[1].lower()
            if ext not in IMG_EXTS:
                continue
            full_path = osp.join(root, filename)
            rel_path = osp.relpath(full_path, img_dir)
            stem = osp.splitext(osp.basename(filename))[0]
            stem_to_files[stem].append(rel_path)

    return {
        stem: sorted(filenames, key=_image_sort_key)[0]
        for stem, filenames in stem_to_files.items()
    }


def _resolve_real_filename(filename, stem_to_filename):
    if not filename or not isinstance(filename, str):
        return None

    stem = osp.splitext(osp.basename(filename))[0]
    return stem_to_filename.get(stem)


def _discover_image_filenames(img_dir):
    if not img_dir or not osp.isdir(img_dir):
        raise FileNotFoundError(f'image directory does not exist: {img_dir}')

    filenames = []
    for root, _, files in os.walk(img_dir):
        for filename in files:
            if osp.splitext(filename)[1].lower() not in IMG_EXTS:
                continue
            filenames.append(osp.relpath(osp.join(root, filename), img_dir))
    if not filenames:
        raise ValueError(f'image directory contains no supported images: {img_dir}')
    return sorted(filenames, key=_image_sort_key)


@ROTATED_DATASETS.register_module(name='DOTADataset', force=True)
class DOTADataset(_MMRotateDOTADataset):
    """Project-local DOTADataset with automatic image suffix matching."""

    def load_annotations(self, ann_file):
        data_infos = super().load_annotations(ann_file)
        stem_to_filename = _build_stem_to_filename(self.img_prefix)

        for info in data_infos:
            real_filename = _resolve_real_filename(
                info.get('filename'), stem_to_filename)
            if real_filename is not None:
                info['filename'] = real_filename

        self.img_ids = [
            osp.splitext(osp.basename(info['filename']))[0]
            for info in data_infos
        ]
        return data_infos


@ROTATED_DATASETS.register_module()
class StrictSourceFreeDOTADataset(Dataset):
    """Target-image-only weak/strong dataset for strict source-free training."""

    def __init__(self,
                 img_prefix,
                 pipeline_share,
                 pipeline_weak,
                 pipeline_strong,
                 classes=None,
                 unlabeled_epoch_size=None,
                 unlabeled_subset_seed=0):
        super().__init__()
        if unlabeled_epoch_size is not None:
            if (not isinstance(unlabeled_epoch_size, int)
                    or isinstance(unlabeled_epoch_size, bool)):
                raise TypeError('unlabeled_epoch_size must be an integer')
            if unlabeled_epoch_size <= 0:
                raise ValueError('unlabeled_epoch_size must be positive')

        self.img_prefix = img_prefix
        self.filenames = _discover_image_filenames(img_prefix)
        self.pipeline_share = Compose(pipeline_share or [])
        self.pipeline_weak = Compose(pipeline_weak or [])
        self.pipeline_strong = Compose(pipeline_strong or [])
        self.CLASSES = classes

        indices = list(range(len(self.filenames)))
        if unlabeled_epoch_size is not None:
            if unlabeled_epoch_size > len(indices):
                raise ValueError(
                    'unlabeled_epoch_size cannot exceed the target image '
                    'dataset length')
            indices = random.Random(unlabeled_subset_seed).sample(
                indices, unlabeled_epoch_size)
        self.indices = indices
        self.flag = np.zeros(len(self.indices), dtype=np.uint8)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        filename = self.filenames[self.indices[idx]]
        shared = self.pipeline_share(dict(
            img_info=dict(filename=filename),
            img_prefix=self.img_prefix,
            bbox_fields=['gt_bboxes'],
            mask_fields=[],
            seg_fields=[],
            gt_bboxes=np.zeros((0, 5), dtype=np.float32),
            gt_labels=np.zeros((0,), dtype=np.int64),
        ))
        if shared is None:
            raise RuntimeError(
                f'shared target pipeline rejected image: {filename}')

        strong = self.pipeline_strong(copy.deepcopy(shared))
        weak = self.pipeline_weak(shared)
        if weak is None or strong is None:
            raise RuntimeError(
                f'weak/strong target pipeline rejected image: {filename}')
        weak.update({
            f'{key}_unlabeled_1': value
            for key, value in strong.items()
        })
        return weak


@ROTATED_DATASETS.register_module()
class SemiDOTADataset(Dataset):
    """基于 DOTA 的半监督封装。"""

    def __init__(self,
                 ann_file,
                 ann_file_u,
                 ann_subdir=None,          # 兼容保留，无实际作用
                 pipeline=None,
                 pipeline_u_share=None,
                 pipeline_u=None,
                 pipeline_u_1=None,
                 data_root=None,
                 img_prefix='',            # /.../train/images/
                 seg_prefix=None,
                 proposal_file=None,
                 data_root_u=None,
                 img_prefix_u='',          # /.../val/images 或无标注目录
                 seg_prefix_u=None,
                 proposal_file_u=None,
                 # 兼容旧 cfg，保留 img_suffix 参数，但这里不再使用
                 img_suffix='.jpg',
                 img_suffix_u=None,
                 classes=None,
                 filter_empty_gt=True,
                 unlabeled_epoch_size=None,
                 unlabeled_subset_seed=0):
        super().__init__()

        if unlabeled_epoch_size is not None:
            if (not isinstance(unlabeled_epoch_size, int)
                    or isinstance(unlabeled_epoch_size, bool)):
                raise TypeError('unlabeled_epoch_size must be an integer')
            if unlabeled_epoch_size <= 0:
                raise ValueError('unlabeled_epoch_size must be positive')

        # 标注集
        self.dota_labeled = ROTATED_DATASETS.build(dict(
            type='DOTADataset',
            ann_file=ann_file,
            pipeline=pipeline,
            data_root=data_root,
            img_prefix=img_prefix,
            test_mode=False,
            filter_empty_gt=filter_empty_gt,
            classes=classes,
        ))

        # 未标注集（共享弱几何增强）
        self.dota_unlabeled = ROTATED_DATASETS.build(dict(
            type='DOTADataset',
            ann_file=ann_file_u,
            pipeline=pipeline_u_share,
            data_root=data_root_u,
            img_prefix=img_prefix_u,
            test_mode=False,
            filter_empty_gt=False,
            classes=classes,
        ))

        self.CLASSES = classes
        self.pipeline_u = Compose(pipeline_u or [])
        self.pipeline_u_1 = Compose(pipeline_u_1) if pipeline_u_1 else None

        self.unlabeled_epoch_size = unlabeled_epoch_size
        self.unlabeled_subset_seed = unlabeled_subset_seed
        self.unlabeled_indices = None
        if unlabeled_epoch_size is not None:
            if unlabeled_epoch_size > len(self.dota_unlabeled):
                raise ValueError(
                    'unlabeled_epoch_size cannot exceed the unlabeled '
                    'dataset length')
            rng = random.Random(unlabeled_subset_seed)
            self.unlabeled_indices = rng.sample(
                range(len(self.dota_unlabeled)), unlabeled_epoch_size)

        unlabeled_flag = getattr(self.dota_unlabeled, 'flag', None)
        if self.unlabeled_indices is None or unlabeled_flag is None:
            self.flag = unlabeled_flag
        else:
            self.flag = unlabeled_flag[self.unlabeled_indices]

    def __len__(self):
        if self.unlabeled_epoch_size is not None:
            return self.unlabeled_epoch_size
        return len(self.dota_labeled)

    def __getitem__(self, idx):
        # 有标注随机采一个，未标注按 idx 配对（防越界）
        idx_label = random.randint(0, len(self.dota_labeled) - 1)
        results = self.dota_labeled[idx_label]

        if self.unlabeled_indices is None:
            u_idx = idx % len(self.dota_unlabeled)
        else:
            u_idx = self.unlabeled_indices[idx % len(self.unlabeled_indices)]
        results_u = self.dota_unlabeled[u_idx]

        # student 强增强分支（如果有）
        if self.pipeline_u_1:
            results_u_1 = copy.deepcopy(results_u)
            results_u_1 = self.pipeline_u_1(results_u_1)
            results.update({f'{k}_unlabeled_1': v
                            for k, v in results_u_1.items()})

        # teacher 弱增强分支
        results_u = self.pipeline_u(results_u)
        results.update({f'{k}_unlabeled': v for k, v in results_u.items()})
        return results
