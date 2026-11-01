#!/usr/bin/env bash
# RSAR LODS cell on gpu-67: SEM on the corruption's val images (no labels) ->
# LODS 1 epoch 1x32 lr 0.02 (iter_266) -> final-EMA TEST eval (AP50 + mAP50-95).
# Same paths/overrides as rsar_dru_seed.sh; only the code dir, config and SEM differ.
# Usage: rsar_lods_seed.sh GPU SEED DOMAIN...
source /mnt/shared/zechuan/iraod_artifacts/comparison/queue_dru20260919/common.sh
set +e
CODE=/mnt/SSD2_8TB/zechuan/IRAOD-New-lods-20261101
export PYTHONPATH=$CODE
# official SEM stage 160000 iters for a 3-epoch detector stage (README:59,162) -> x1/3 for 1 epoch
SEM_ITERS=53334
ENC=/mnt/shared/zechuan/iraod_artifacts/third_party/vgg16_oxford/encoder_split_enc_keys.pt
GPU=$1; SEED=$2; shift 2
export CUDA_VISIBLE_DEVICES=$GPU
prep_env
cd "$CODE"
for CORRUPT in "$@"; do
  CELL=$ART/rsar/${CORRUPT}/seed_${SEED}/lods_e1
  SEM=$CELL/sem/sem_iter_${SEM_ITERS}.pth
  if [ ! -f "$SEM" ]; then
    mkdir -p "$CELL/sem"
    "$PY" tools/lods/train_sem.py --target-dir "$RSAR/corruptions/${CORRUPT}/val/images" \
      --encoder "$ENC" --work-dir "$CELL/sem" --seed "$SEED" --workers 32 --max-iter "$SEM_ITERS" > "$CELL/sem/stdout.log" 2>&1
    [ -f "$SEM" ] || { echo "SEM_FAIL LODS $CORRUPT seed=$SEED"; continue; }
  fi
  ROOT=$CELL/methods/LODS
  WORK=$ROOT/work; LOG=$ROOT/logs; OUT=$ROOT/eval_full_${CORRUPT}_ids_v1
  mkdir -p "$LOG"
  if [ ! -f "$ROOT/train_done" ]; then
    rm -rf "$WORK"; mkdir -p "$WORK"
    echo "lods_rsar_train_start gpu=$GPU seed=$SEED corrupt=$CORRUPT sem=$SEM $(date --iso-8601=seconds)" | tee "$LOG/train.log"
    "$PY" train.py configs/unbiased_teacher/sfod/lods_oriented_rcnn_rsar_orthonet_strict.py \
      --work-dir "$WORK" --gpus 1 --seed "$SEED" --deterministic --no-validate \
      --cfg-options checkpoint_config.max_keep_ckpts=2 checkpoint_config.save_last=True data.samples_per_gpu=32 optimizer.lr=0.02 \
        model.cfg.strict_source_free=True model.cfg.weight_l=0 model.cfg.weight_u=1 \
        data.train.type=StrictSourceFreeDOTADataset \
        data.train.img_prefix=$RSAR/corruptions/${CORRUPT}/val/images corrupt=${CORRUPT} \
        load_from=$RSAR_SRC model.ema_ckpt=$RSAR_SRC model.cfg.sem_ckpt=$SEM \
      >> "$LOG/train.log" 2>&1
    ec=$?
    EMA=$(ls "$WORK"/iter_*_ema.pth 2>/dev/null | sed -E 's/.*iter_([0-9]+)_ema\.pth/\1 &/' | sort -n | awk 'END{print $2}')
    if [ "$ec" -ne 0 ] || [ -z "$EMA" ]; then echo "TRAIN_FAIL LODS $CORRUPT seed=$SEED ec=$ec"; continue; fi
    echo "ema=$EMA $(date --iso-8601=seconds)" > "$ROOT/train_done"
  fi
  EMA=$(sed -E 's/^ema=([^ ]+).*/\1/' "$ROOT/train_done")
  grep -qs eval_exit=0 "$OUT/eval_status" && continue
  mkdir -p "$OUT"
  "$PY" test.py configs/baseline/oriented_rcnn_orthonet_rsar.py "$EMA" --eval mAP --map50-95 \
    --out "$OUT/predictions.pkl" --work-dir "$OUT" \
    --cfg-options data.samples_per_gpu=16 data.workers_per_gpu=4 \
      data.test.ann_file=$RSAR/test/annfiles/ \
      data.test.img_prefix=$RSAR/corruptions/${CORRUPT}/test/images/ \
    > "$OUT/eval.log" 2>&1
  echo "eval_exit=$? name=LODS domain=$CORRUPT seed=$SEED $(date --iso-8601=seconds) ema=$EMA" | tee -a "$OUT/eval_status"
done
echo "RSAR_LODS_SEED_DONE seed=$SEED $(date --iso-8601=seconds)"
