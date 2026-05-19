from mmcv import Config
from mmdet.datasets import build_dataset
from torch.utils.data import DataLoader
from mmcv.parallel import collate  # mmdet 的数据是 DataContainer，需要这个

cfg = Config.fromfile('configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga_rsar.py')
cfg.data.workers_per_gpu = 0

dataset = build_dataset(cfg.data.train)
print('len(train)=', len(dataset))

loader = DataLoader(
    dataset,
    batch_size=cfg.data.samples_per_gpu,
    shuffle=True,
    num_workers=0,
    collate_fn=collate
)

batch = next(iter(loader))
print('got one batch keys:', batch.keys())


