"""Shared runtime helpers for SOGC analysis scripts."""

import copy
import json
import os
import random
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from mmcv.parallel import MMDataParallel
from mmrotate.datasets import build_dataset


RANDOM_PIPELINE_MARKERS = (
    'Random', 'RandAug', 'AutoAug', 'ColorJitter', 'GaussianBlur')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_text):
    device = torch.device(device_text)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError(
                f'CUDA device {device_text!r} was requested but CUDA is '
                'not available')
        index = device.index if device.index is not None else 0
        torch.cuda.set_device(index)
        return torch.device('cuda', index), [index]
    if device.type != 'cpu':
        raise ValueError(f'unsupported device type: {device.type}')
    return device, []


def detector_backbone(model):
    host = getattr(model, 'module', model)
    backbone = getattr(host, 'backbone', None)
    if backbone is None:
        raise TypeError('model does not expose a backbone')
    required = ('set_sogc_mode', 'iter_sogc_calibrators')
    if any(not hasattr(backbone, name) for name in required):
        raise TypeError('configured backbone is not SOGC-capable OrthoNet')
    if not list(backbone.iter_sogc_calibrators()):
        raise RuntimeError('configured OrthoNet has no active SOGC modules')
    return backbone


def _checkpoint_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError('checkpoint must contain a dictionary')
    state_dict = checkpoint.get('state_dict', checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError('checkpoint state_dict must be a dictionary')
    revised = OrderedDict()
    for key, value in state_dict.items():
        revised[re.sub(r'^module\.', '', key)] = value
    return revised


# Modules that are zero-initialized and therefore legitimately absent from a
# legacy checkpoint. Anything else missing is a real weight-loading fault.
OPTIONAL_MODULE_MARKERS = ('.sogc.', '.slrp.')


def load_checkpoint_sogc_compatible(model,
                                    checkpoint_path,
                                    allow_missing_sogc):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'checkpoint not found: {checkpoint_path}')
    checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
    state_dict = _checkpoint_state_dict(checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    invalid_missing = [
        key for key in missing
        if not any(marker in key for marker in OPTIONAL_MODULE_MARKERS)]
    if invalid_missing:
        preview = ', '.join(invalid_missing[:20])
        raise RuntimeError(
            'checkpoint is missing pre-existing detector weights; '
            f'first keys: {preview}')
    if missing and not allow_missing_sogc:
        preview = ', '.join(missing[:20])
        raise RuntimeError(
            f'checkpoint is missing SOGC keys: {preview}')
    if unexpected:
        preview = ', '.join(unexpected[:20])
        raise RuntimeError(f'checkpoint has unexpected keys: {preview}')
    return checkpoint, {
        'missing_keys': missing,
        'unexpected_keys': unexpected,
        'loaded_key_count': len(state_dict),
    }


def _walk_pipeline(steps):
    for step in steps:
        if not isinstance(step, dict):
            continue
        yield step
        transforms = step.get('transforms')
        if isinstance(transforms, (list, tuple)):
            yield from _walk_pipeline(transforms)
        operations = step.get('operations')
        if isinstance(operations, (list, tuple)):
            yield from _walk_pipeline(operations)


def validate_deterministic_pipeline(pipeline):
    for step in _walk_pipeline(pipeline):
        step_type = str(step.get('type', ''))
        if step_type == 'MultiScaleFlipAug':
            if step.get('flip', False):
                raise ValueError(
                    'deterministic pipeline cannot enable MultiScaleFlipAug '
                    'flipping')
            continue
        if step_type in ('RRandomFlip', 'RandomFlip'):
            raise ValueError(
                f'deterministic pipeline contains {step_type}')
        if any(marker in step_type for marker in RANDOM_PIPELINE_MARKERS):
            raise ValueError(
                f'deterministic pipeline contains stochastic step '
                f'{step_type}')


def clean_source_dataset_cfg(cfg):
    dataset_cfg = copy.deepcopy(cfg.data.train)
    if not isinstance(dataset_cfg, dict):
        raise TypeError('cfg.data.train must be a dataset dictionary')
    if dataset_cfg.get('type', '').startswith('Semi'):
        raise ValueError(
            'source collection requires a supervised clean-source config, '
            'not a semi-supervised dataset')
    reference = cfg.data.get('val', cfg.data.get('test'))
    if reference is None or 'pipeline' not in reference:
        raise KeyError('config has no val/test pipeline for deterministic use')
    pipeline = copy.deepcopy(reference.pipeline)
    validate_deterministic_pipeline(pipeline)
    dataset_cfg['pipeline'] = pipeline
    dataset_cfg['test_mode'] = True
    return dataset_cfg


def evaluation_dataset_cfg(cfg):
    dataset_cfg = copy.deepcopy(cfg.data.test)
    if isinstance(dataset_cfg, (list, tuple)):
        if len(dataset_cfg) != 1:
            raise ValueError('analysis supports exactly one test dataset')
        dataset_cfg = dataset_cfg[0]
    pipeline = copy.deepcopy(dataset_cfg['pipeline'])
    validate_deterministic_pipeline(pipeline)
    dataset_cfg['pipeline'] = pipeline
    dataset_cfg['test_mode'] = True
    return dataset_cfg


class BackboneCollector(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, img, **kwargs):
        if isinstance(img, (list, tuple)):
            if len(img) != 1:
                raise ValueError('SOGC collection does not support test-time '
                                 'multi-augmentation')
            img = img[0]
        return self.backbone(img)


def make_backbone_runner(backbone, device, device_ids):
    collector = BackboneCollector(backbone).to(device)
    return MMDataParallel(collector, device_ids=device_ids)


def make_detector_runner(model, device, device_ids):
    model = model.to(device)
    return MMDataParallel(model, device_ids=device_ids)


def build_dataset_checked(dataset_cfg):
    dataset = build_dataset(dataset_cfg)
    if len(dataset) == 0:
        raise RuntimeError(
            'dataset is empty; check ann_file and img_prefix in the config')
    return dataset


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def dump_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(json_ready(payload), handle, indent=2, sort_keys=True)
        handle.write('\n')
