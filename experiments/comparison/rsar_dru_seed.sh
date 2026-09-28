#!/usr/bin/env bash
# RSAR DRU extra seed = seed-42 queue_dru20260919 train+eval commands with only --seed changed.
# Usage: rsar_dru_seed.sh GPU SEED DOMAIN...   (1 GPU x 32, 1 epoch -> iter_266)
source /mnt/shared/zechuan/iraod_artifacts/comparison/queue_dru20260919/common.sh
set +e
GPU=$1; SEED=$2; shift 2
export CUDA_VISIBLE_DEVICES=$GPU
prep_env
cd "$CODE"
for CORRUPT in "$@"; do
  ROOT=$ART/rsar/${CORRUPT}/seed_${SEED}/dru_e1/methods/DRU
  WORK=$ROOT/work; LOG=$ROOT/logs; OUT=$ROOT/eval_full_${CORRUPT}_ids_v1
  mkdir -p "$LOG"
  if [ ! -f "$ROOT/train_done" ]; then
    rm -rf "$WORK"; mkdir -p "$WORK"
    echo "dru_rsar_train_start gpu=$GPU seed=$SEED corrupt=$CORRUPT $(date --iso-8601=seconds)" | tee "$LOG/train.log"
    "$PY" train.py configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_dru_rsar_orthonet_strict.py \
      --work-dir "$WORK" --gpus 1 --seed "$SEED" --deterministic --no-validate \
      --cfg-options checkpoint_config.max_keep_ckpts=2 checkpoint_config.save_last=True data.samples_per_gpu=32 optimizer.lr=0.02 \
        model.cfg.strict_source_free=True model.cfg.weight_l=0 model.cfg.weight_u=1 \
        data.train.type=StrictSourceFreeDOTADataset \
        data.train.img_prefix=$RSAR/corruptions/${CORRUPT}/val/images corrupt=${CORRUPT} \
        load_from=$RSAR_SRC model.ema_ckpt=$RSAR_SRC \
      >> "$LOG/train.log" 2>&1
    ec=$?
    EMA=$(ls "$WORK"/iter_*_ema.pth 2>/dev/null | sed -E 's/.*iter_([0-9]+)_ema\.pth/\1 &/' | sort -n | awk 'END{print $2}')
    if [ "$ec" -ne 0 ] || [ -z "$EMA" ]; then echo "TRAIN_FAIL DRU $CORRUPT seed=$SEED ec=$ec"; continue; fi
    echo "ema=$EMA $(date --iso-8601=seconds)" > "$ROOT/train_done"
  fi
  EMA=$(sed -E 's/^ema=([^ ]+).*/\1/' "$ROOT/train_done")
  grep -qs eval_exit=0 "$OUT/eval_status" && continue
  mkdir -p "$OUT"
  "$PY" test.py configs/baseline/oriented_rcnn_orthonet_rsar.py "$EMA" --eval mAP --work-dir "$OUT" \
    --cfg-options data.samples_per_gpu=16 data.workers_per_gpu=4 \
      data.test.ann_file=$RSAR/test/annfiles/ \
      data.test.img_prefix=$RSAR/corruptions/${CORRUPT}/test/images/ \
    > "$OUT/eval.log" 2>&1
  echo "eval_exit=$? name=DRU domain=$CORRUPT seed=$SEED $(date --iso-8601=seconds) ema=$EMA" | tee -a "$OUT/eval_status"
done
echo "RSAR_DRU_SEED_DONE seed=$SEED $(date --iso-8601=seconds)"
