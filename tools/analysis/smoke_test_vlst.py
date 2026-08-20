#!/usr/bin/env python
"""Smoke test for VLST implementation.

Verifies:
1. Config loads without errors
2. Model builds successfully
3. Forward pass completes
4. Prototype loss is finite
5. Prototype updates work
6. Gradients flow to projection head and Student
"""
import os
import sys
import torch
import numpy as np

# Add repo to path
sys.path.insert(0, '/myfile/mycode/IRAOD-New')

from mmcv import Config
from mmdet.models import build_detector


def smoke_test_vlst():
    print("="*80)
    print("VLST Smoke Test")
    print("="*80)

    # 1. Load config
    print("\n[1/6] Loading config...")
    config_path = 'configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_vlst_rsar_orthonet.py'
    cfg = Config.fromfile(config_path)
    print(f"✓ Config loaded from {config_path}")
    print(f"  Model type: {cfg.model.type}")
    print(f"  VLST enabled: {cfg.model.cfg.get('vlst_enabled', False)}")
    print(f"  VLST loss weight: {cfg.model.cfg.get('vlst_loss_weight', 0.0)}")

    # 2. Build model
    print("\n[2/6] Building model...")
    cfg.model.pretrained = None
    cfg.model.ema_ckpt = None  # Don't load checkpoint for smoke test
    model = build_detector(cfg.model)
    model.eval()
    print(f"✓ Model built: {type(model).__name__}")
    print(f"  VLST teacher: {type(model.vlst_teacher).__name__ if model.vlst_teacher else 'None'}")

    if model.vlst_teacher is None:
        print("✗ VLST teacher not initialized!")
        return False

    # 3. Initialize text prototypes (simulate CGA scorer)
    print("\n[3/6] Initializing text prototypes...")
    num_classes = 6  # RSAR
    vlm_dim = 512
    text_prototypes = torch.randn(num_classes, vlm_dim)
    text_prototypes = torch.nn.functional.normalize(text_prototypes, p=2, dim=1)
    model.vlst_teacher.set_text_prototypes(text_prototypes)
    print(f"✓ Text prototypes initialized: shape={text_prototypes.shape}")

    # 4. Test prototype update
    print("\n[4/6] Testing prototype update...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    N = 20  # Number of pseudo boxes
    vlm_features = torch.randn(N, vlm_dim).to(device)
    vlm_features = torch.nn.functional.normalize(vlm_features, p=2, dim=1)
    labels = torch.randint(0, num_classes, (N,)).to(device)
    scores = torch.rand(N).to(device) * 0.3 + 0.7  # [0.7, 1.0]

    model.vlst_teacher.update_visual_prototypes(vlm_features, labels, scores, score_thr=0.7)
    print(f"✓ Visual prototypes updated")
    print(f"  Visual counts: {model.vlst_teacher.visual_counts}")
    print(f"  Total updates: {model.vlst_teacher.total_updates}")

    # 5. Test prototype loss computation
    print("\n[5/6] Testing prototype loss computation...")
    detector_dim = 1024
    student_features = torch.randn(N, detector_dim).to(device)
    student_features.requires_grad = True

    loss, num_samples = model.vlst_teacher.compute_prototype_loss(
        student_features, labels, scores, score_thr=0.7)

    print(f"✓ Prototype loss computed")
    print(f"  Loss value: {loss.item():.4f}")
    print(f"  Num samples: {num_samples}")
    print(f"  Loss finite: {torch.isfinite(loss).item()}")

    if not torch.isfinite(loss):
        print("✗ Loss is NaN or Inf!")
        return False

    if num_samples == 0:
        print("✗ No samples used in loss!")
        return False

    # 6. Test gradient flow
    print("\n[6/6] Testing gradient flow...")
    loss.backward()

    # Check projection head gradients
    proj_has_grad = False
    if model.vlst_teacher.projection_head is not None:
        for param in model.vlst_teacher.projection_head.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                proj_has_grad = True
                break

    # Check student feature gradients
    student_has_grad = student_features.grad is not None and student_features.grad.abs().sum() > 0

    print(f"✓ Gradients computed")
    print(f"  Projection head has gradients: {proj_has_grad}")
    print(f"  Student features have gradients: {student_has_grad}")

    if not proj_has_grad:
        print("✗ Projection head has no gradients!")
        return False

    if not student_has_grad:
        print("✗ Student features have no gradients!")
        return False

    # Success
    print("\n" + "="*80)
    print("✓ All smoke tests PASSED")
    print("="*80)
    return True


if __name__ == '__main__':
    os.chdir('/myfile/mycode/IRAOD-New')
    success = smoke_test_vlst()
    sys.exit(0 if success else 1)
