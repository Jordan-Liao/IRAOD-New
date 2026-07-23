from pathlib import Path

from mmcv import Config


SOURCE_CFG = 'configs/baseline/oriented_rcnn_stripnet_rsar.py'
EMA_CFG = (
    'configs/baseline/ema_config/'
    'baseline_oriented_rcnn_ema_rsar_stripnet.py')
SFOD_CFG = (
    'configs/unbiased_teacher/sfod/'
    'unbiased_teacher_oriented_rcnn_stripnet_rsar.py')
LAUNCHER = Path('scripts/run_stripnet_rsar_seed42.sh')


def test_stripnet_stage_contract():
    import torch

    import mmdet_extension  # noqa: F401
    from mmdet.models.builder import BACKBONES

    backbone = BACKBONES.build(dict(
        type='StripNet',
        embed_dims=[64, 128, 320, 512],
        k1s=[1, 1, 1, 1],
        k2s=[19, 19, 19, 19],
        depths=[2, 2, 4, 2],
        drop_rate=0.1,
        drop_path_rate=0.15,
        init_cfg=None))
    backbone.eval()
    with torch.no_grad():
        outputs = backbone(torch.randn(1, 3, 128, 128))
    assert [tuple(output.shape) for output in outputs] == [
        (1, 64, 32, 32),
        (1, 128, 16, 16),
        (1, 320, 8, 8),
        (1, 512, 4, 4),
    ]


def test_source_config_only_uses_stripnet_backbone_with_standard_head():
    cfg = Config.fromfile(SOURCE_CFG)
    assert cfg.model.type == 'OrientedRCNN'
    assert cfg.model.backbone.type == 'StripNet'
    assert cfg.model.backbone.init_cfg is None
    assert list(cfg.model.neck.in_channels) == [64, 128, 320, 512]
    assert cfg.model.rpn_head.type == 'OrientedRPNHead'
    assert cfg.model.roi_head.type == 'OrientedStandardRoIHead'
    assert cfg.model.roi_head.bbox_head.type == 'RotatedShared2FCBBoxHead'
    assert cfg.model.roi_head.bbox_head.num_classes == 6
    assert cfg.angle_version == 'le90'
    assert tuple(cfg.image_size) == (800, 800)
    assert cfg.load_from is None


def test_student_and_ema_structure_match():
    student_cfg = Config.fromfile(SFOD_CFG)
    ema_cfg = Config.fromfile(EMA_CFG)
    for model_cfg in (student_cfg.model, ema_cfg.model):
        assert model_cfg.backbone.type == 'StripNet'
        assert model_cfg.backbone.init_cfg is None
        assert list(model_cfg.neck.in_channels) == [64, 128, 320, 512]
        assert model_cfg.roi_head.type == 'OrientedStandardRoIHead'
        assert model_cfg.roi_head.bbox_head.type == \
            'RotatedShared2FCBBoxHead'
    assert student_cfg.model.ema_ckpt == student_cfg.load_from
    assert student_cfg.model.cfg.weight_l == 0.0
    assert student_cfg.model.cfg.weight_u == 1.0
    assert student_cfg.model.cfg.semantic_reweight is False
    assert student_cfg.model.cfg.dynamic_threshold is False
    assert student_cfg.model.cfg.use_bbox_reg is False


def test_source_free_data_split_and_no_strip_head():
    cfg = Config.fromfile(SFOD_CFG)
    assert '/corruptions/${corrupt}/val/images/' in \
        cfg.data.train.img_prefix_u
    assert cfg.data.train.unlabeled_epoch_size == 8467
    assert cfg.data.train.unlabeled_subset_seed == 42
    assert '/corruptions/${corrupt}/test/images/' in cfg.data.test.img_prefix
    assert 'StripHead' not in cfg.pretty_text
    assert 'SemanticWeightedOrientedStandardRoIHead' not in cfg.pretty_text


def test_launcher_disables_cga_and_covers_clean_chaff_matrix():
    launcher = LAUNCHER.read_text(encoding='utf-8')
    assert 'CGA_SCORER="none"' in launcher
    assert 'CGA_BACKEND="none"' in launcher
    assert 'CGA_FILTER_MODE="none"' in launcher
    assert '--no-validate' in launcher
    for stage in (
            'source_clean_test',
            'source_chaff_test',
            'adapted_clean_test',
            'adapted_chaff_test'):
        assert f'STAGE {stage} START' in launcher
