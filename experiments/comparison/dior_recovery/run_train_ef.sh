#!/usr/bin/env bash
set -euo pipefail
GPU_PAIR=$1
METHOD=$2
CORRUPT=$3
PORT=$4
CODE=/mnt/SSD2_8TB/zechuan/IRAOD-New-strict-af
PY=/home/zechuan/miniforge3/envs/iraod/bin/python
SOURCE=/mnt/shared/zechuan/iraod_artifacts/dior_source/seed42-orthonet-ddp4-gpu4567-spg16-gbs64/train/epoch_100.pth
Q=/mnt/shared/zechuan/iraod_artifacts/comparison/dior/queue_0f98a48
SPLITS=/mnt/shared/zechuan/iraod_artifacts/dior_splits
ROOT=/mnt/shared/zechuan/iraod_artifacts/comparison/dior/${CORRUPT}/seed_42/0f98a48
WORK=$ROOT/methods/${METHOD}/ddp2/work
mkdir -p "$WORK" "$ROOT/methods/${METHOD}/ddp2/logs"
if ls "$WORK"/iter_*_ema.pth >/dev/null 2>&1; then
  echo "skip_done_train $METHOD $CORRUPT"; exit 0
fi
CFG=$Q/configs/dior_${METHOD}_strict.py
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU_PAIR
export PYTHONPATH=$CODE PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 IRAOD_RUNTIME_READY=1
export CONDA_PREFIX=/home/zechuan/miniforge3/envs/iraod
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MASTER_PORT=$PORT
export SARCLIP_PRETRAINED=/mnt/shared/zechuan/iraod_weights/sarclip/ViT-B-32/vit_b_32_model.safetensors
for k in $(env | awk -F= '/^(CGA_|SARCLIP_|VLST_)/{print $1}'); do unset "$k"; done
export SARCLIP_PRETRAINED=/mnt/shared/zechuan/iraod_weights/sarclip/ViT-B-32/vit_b_32_model.safetensors
if [ "$METHOD" = F ]; then
  export HOME=/tmp/iraod_clip_home
  export CLIP_DOWNLOAD_ROOT=/mnt/shared/zechuan/iraod_weights/clip
  export CGA_CLIP_CACHE=/mnt/shared/zechuan/iraod_weights/clip
fi
cd "$CODE"
set +e
"$PY" -m torch.distributed.launch --nproc_per_node=2 --master_port=$PORT \
  train.py "$CFG" \
  --launcher pytorch --work-dir "$WORK" --seed 42 --deterministic --no-validate \
  --cfg-options checkpoint_config.max_keep_ckpts=2 checkpoint_config.save_last=True \
    data.samples_per_gpu=16 optimizer.lr=0.02 find_unused_parameters=True \
    model.cfg.strict_source_free=True model.cfg.weight_l=0 model.cfg.weight_u=1 model.cfg.use_bbox_reg=False model.roi_head.bbox_head.num_classes=20 \
    data.train.img_prefix=$SPLITS/${CORRUPT}/val \
    load_from=$SOURCE model.ema_ckpt=$SOURCE corrupt=$CORRUPT \
  > "$ROOT/methods/${METHOD}/ddp2/logs/train.log" 2>&1
ec=$?
set -e
echo "tmux_wrap_exit=$ec $(date --iso-8601=seconds)" | tee "$ROOT/methods/${METHOD}/ddp2/terminal_status"
exit $ec
