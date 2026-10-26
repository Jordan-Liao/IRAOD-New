"""Fully supervised RSAR source training with the OrthoRSNet backbone.

Same six-class RSAR protocol as ``oriented_rcnn_orthonet_rsar.py``.  Two
deliberate differences:

``load_from = None``
    The supplied ``oriented_rcnn_orthonet_rsar_epoch_100.pth`` checkpoint was
    trained under the legacy orthogonal attention, whose ``ortho_vector`` and
    ``fc`` keys do not describe the OSP gate.  Loading it would leave the new
    modules at their identity while silently keeping a dead attention, so this
    run starts from ImageNet weights like the DIOR source run.  To warm-start
    from the legacy checkpoint instead, pass
    ``--cfg-options load_from=/myfile/pretrain/oriented_rcnn_orthonet_rsar_epoch_100.pth``:
    every convolution key matches, and the legacy attention keys are simply
    reported as unexpected.

``init_cfg``
    torchvision ResNet-50.  OrthoRSNet's non-attention parameters are
    key-for-key identical to ResNet-50, and all three new modules are exact
    identities at initialization, so step zero is a plain ImageNet ResNet-50.
"""

_base_ = './oriented_rcnn_orthonet_rsar.py'

load_from = None

model = dict(
    backbone=dict(
        _delete_=True,
        type='OrthoRSNet',
        depth=50,
        reduction=16,
        in_channels=3,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        radiometric_cfg=dict(
            enabled=True,
            window=5,
            eps=1e-4,
            gamma=0.5,
            hidden_channels=16,
            learnable_affine=True,
            # SAR intensity is multiplicative; the log transform makes the
            # multiplicative speckle additive before the local statistics,
            # which is the homomorphic form of speckle filtering.  Kept off
            # for the first run so the DIOR and RSAR arms share one front end;
            # flip to True as its own ablation.
            log_input=False,
            use_variance=True),
        stem_cfg=dict(
            enabled=True,
            orientations=(0, 45, 90, 135),
            dir_kernel_size=7,
            dir_channels=8,
            renormalize=True),
        attn_cfg=dict(
            enabled=True,
            basis_rank=32,
            strip_pool=True,
            strip_kernel=7,
            identity_init=True,
            basis_ridge=1e-3),
        frozen_stages=1,
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='torchvision://resnet50')))

optimizer = dict(
    paramwise_cfg=dict(
        custom_keys={
            'ortho_vector': dict(decay_mult=0.0),
            'bg_basis': dict(decay_mult=0.0),
        }))

work_dir = 'work_dirs/oriented_rcnn_orthorsnet_rsar'
