#!/usr/bin/env bash
# DIOR extra-seed cell: train + final-EMA TEST eval, matching the seed-42 DIOR protocol.
# Usage: GPUS METHOD DOMAIN SEED [PORT]
#   GPUS "a"   -> 1 GPU x samples_per_gpu 32
#   GPUS "a,b" -> DDP2  x samples_per_gpu 16   (required for E/F: 1x32 OOMs)
# Protocol (= seed 42): global batch 32, lr 0.02, runner.max_epochs=1 (~185 iters),
# unlabeled_epoch_size 5863, strict_source_free, weight_l=0, final-iteration EMA.
# Outputs go to ext_runs_e1/ so the old 2-epoch / batch-16 runs in ext_runs/ are never reused.
set -u
GPUS=$1
METHOD=$2
DOMAIN=$3
SEED=$4
PORT=${5:-$((29500 + RANDOM % 1000))}
CONDA_PREFIX=/home/zechuan/miniconda3/envs/iraod
PY=$CONDA_PREFIX/bin/python
SPLITS=/home/zechuan/iraod_jobs/dior_splits
SRC_DIR=/home/zechuan/iraod_artifacts/dior_source/seed42-orthonet-ddp4-gpu4567-spg16-gbs64
DIOR_SRC=$SRC_DIR/train/epoch_100.pth
RESOLVED=$SRC_DIR/reproducibility/resolved_config.py
LOCAL=/home/zechuan/iraod_jobs/ext_runs_e1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 IRAOD_RUNTIME_READY=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID NCCL_IB_DISABLE=1 OMP_NUM_THREADS=1
export CONDA_PREFIX
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# Same env whitelist as the seed-42 DIOR workers (dior_bf/worker_94.sh).
for k in $(env | awk -F= '/^(CGA_|SARCLIP_|VLST_)/{print $1}'); do unset "$k"; done
export SARCLIP_PRETRAINED=/home/zechuan/iraod_weights/sarclip/ViT-B-32/vit_b_32_model.safetensors
export HOME=/tmp/iraod_clip_home
CLIP_DIR=/home/zechuan/iraod_jobs/weights/clip
[[ -f $CLIP_DIR/RN50x64.pt ]] || CLIP_DIR=/mnt/shared/zechuan/iraod_weights/clip
export CLIP_DOWNLOAD_ROOT=$CLIP_DIR CGA_CLIP_CACHE=$CLIP_DIR
mkdir -p /tmp/iraod_clip_home/.cache/clip
[[ -e /tmp/iraod_clip_home/.cache/clip/RN50x64.pt ]] || ln -sf $CLIP_DIR/RN50x64.pt /tmp/iraod_clip_home/.cache/clip/RN50x64.pt

EXTRA=()
case "$METHOD" in
  B|C|D|E|F)
    CODE=/home/zechuan/IRAOD-New-strict-af
    CFG=/home/zechuan/iraod_jobs/dior_bf/configs/dior_${METHOD}_strict.py
    EXTRA+=(model.cfg.use_bbox_reg=False)
    ;;
  DRU)
    CODE=/home/zechuan/IRAOD-New-dru-20260919
    CFG=configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_dru_dior_orthonet_strict.py
    EXTRA+=(data.train.type=StrictSourceFreeDOTADataset)
    ;;
  IRG|LPLD|SFUT)
    CODE=/home/zechuan/iraod_jobs/ext/repo_code_a39c832
    CFG=configs/unbiased_teacher/sfod/extensions/${METHOD,,}_dior.py
    EXTRA+=(data.train.type=StrictSourceFreeDOTADataset)
    ;;
  AASFOD)
    # Seeds disagree: TSD split + spg16 (110/159 iters, 09-22 reruns) vs no TSD (185 iters, DIOR-17 s42).
    echo "AASFOD protocol undecided (TSD vs non-TSD); choose one before rerunning"; exit 2
    ;;
  *) echo "bad method $METHOD"; exit 2 ;;
esac
case "$METHOD" in
  B) export CGA_SCORER=none CGA_BACKEND=none CGA_FILTER_MODE=none ;;
  C) export CGA_SCORER=clip CGA_BACKEND=clip CGA_CLIP_MODEL=RN50x64 CGA_FILTER_MODE=legacy ;;
esac
export PYTHONPATH=$CODE

if [[ "$GPUS" == *,* ]]; then
  NGPU=2; SPG=16
  LAUNCH=(-m torch.distributed.launch --nproc_per_node=2 --master_port="$PORT" train.py "$CFG" --launcher pytorch)
  EXTRA+=(find_unused_parameters=True)
else
  NGPU=1; SPG=32
  LAUNCH=(train.py "$CFG" --gpus 1)
fi
if [[ ( "$METHOD" == E || "$METHOD" == F ) && "$NGPU" -ne 2 ]]; then
  echo "E/F need two GPUs (DDP2 x16); got GPUS=$GPUS"; exit 2
fi
export CUDA_VISIBLE_DEVICES=$GPUS

root=$LOCAL/${METHOD}_d17_s${SEED}/$DOMAIN
work=$root/work
logd=$root/logs
done_file=$root/train_done
mkdir -p "$logd"

# Skip only a run that finished; a partial work dir is discarded, never evaluated.
if [[ -f "$done_file" ]]; then
  echo "skip_train_done $METHOD $DOMAIN $SEED"
else
  rm -rf "$work"; mkdir -p "$work"
  echo "train_start $METHOD $DOMAIN seed=$SEED gpus=$GPUS spg=$SPG $(date --iso-8601=seconds)" | tee "$logd/train.log"
  cd "$CODE"
  set +e
  "$PY" "${LAUNCH[@]}" --work-dir "$work" --seed "$SEED" --deterministic --no-validate \
    --cfg-options checkpoint_config.max_keep_ckpts=2 checkpoint_config.save_last=True \
      data.samples_per_gpu=$SPG data.workers_per_gpu=2 optimizer.lr=0.02 \
      runner.max_epochs=1 \
      model.cfg.strict_source_free=True model.cfg.weight_l=0 model.cfg.weight_u=1 \
      model.roi_head.bbox_head.num_classes=20 model.ema_config=$RESOLVED \
      data.train.img_prefix=$SPLITS/$DOMAIN/val \
      data.train.unlabeled_epoch_size=5863 \
      load_from=$DIOR_SRC model.ema_ckpt=$DIOR_SRC corrupt=${DOMAIN} \
      "${EXTRA[@]}" \
    >> "$logd/train.log" 2>&1
  ec=$?
  set -u
  # Final EMA = highest iteration number, not newest mtime.
  final=$(ls "$work"/iter_*_ema.pth 2>/dev/null | sed -E 's/.*iter_([0-9]+)_ema\.pth/\1 &/' | sort -n | awk 'END{print $2}')
  if [[ "$ec" -ne 0 || -z "$final" ]]; then
    echo "TRAIN_FAIL $METHOD $DOMAIN seed=$SEED ec=$ec"
    exit 1
  fi
  echo "ema=$final $(date --iso-8601=seconds)" > "$done_file"
fi
ema=$(sed -E 's/^ema=([^ ]+).*/\1/' "$done_file")
echo "TRAIN_DONE $METHOD $DOMAIN seed=$SEED ema=$ema"

EVAL=/home/zechuan/iraod_jobs/IRAOD-New-rc331d213
[[ -f "$EVAL/test.py" ]] || EVAL=/home/zechuan/iraod_jobs/ext/integration_code_36c2053
DIOR=/home/zechuan/iraod_data/DIOR
[[ -f "$DIOR/ImageSets/test.txt" ]] || DIOR=/mnt/HDD2_4TB/zechuan/iraod_data/DIOR
[[ -f "$DIOR/ImageSets/test.txt" ]] || DIOR=/mnt/HDD1_3TB/zechuan/iraod_data/DIOR
[[ -f "$DIOR/ImageSets/test.txt" ]] || DIOR=/mnt/shared/zechuan/iraod_data/DIOR
out=$root/eval_full_${DOMAIN}_ids_v1
if ! grep -q eval_exit=0 "$out/eval_status" 2>/dev/null; then
  rm -rf "$out"
  mkdir -p "$out"
  echo "eval_start $METHOD $DOMAIN seed=$SEED gpu=${GPUS%%,*} $(date --iso-8601=seconds)" | tee "$out/eval.log"
  cd "$EVAL"
  set +e
  CUDA_VISIBLE_DEVICES=${GPUS%%,*} PYTHONPATH=$EVAL:$CODE "$PY" "$EVAL/test.py" "$RESOLVED" "$ema" --eval mAP --out "$out/pred_new.pkl" --work-dir "$out" \
    --cfg-options data.samples_per_gpu=16 data.workers_per_gpu=4 \
      data.test.ann_file="$DIOR/ImageSets/test.txt" \
      data.test.img_prefix="$SPLITS/$DOMAIN/test/" \
      data.test.ann_subdir="$DIOR/Annotations/Oriented Bounding Boxes/" \
    >> "$out/eval.log" 2>&1
  rec=$?
  echo "eval_exit=$rec name=$METHOD corrupt=$DOMAIN seed=$SEED $(date --iso-8601=seconds) ema=$ema" | tee -a "$out/eval_status" "$out/eval.log"
  set -u
  [[ "$rec" -eq 0 ]] || { echo "EVAL_FAIL $METHOD $DOMAIN seed=$SEED"; exit 1; }
fi
echo "CELL_DONE $METHOD $DOMAIN seed=$SEED"
exit 0
