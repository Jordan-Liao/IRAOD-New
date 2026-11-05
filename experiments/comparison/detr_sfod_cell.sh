#!/usr/bin/env bash
# DETR-group (ARS-DETR) strict source-free cell: train + final-EMA TEST eval.
# Usage: detr_sfod_cell.sh METHOD DATASET DOMAIN SEED GPU
#   METHOD  A (source only, eval) | B | C | D | E | F | PETS | SFUT | SFYOLO (2 epochs)
#   DATASET rsar | dior;  DOMAIN = corruption name (e.g. chaff);  GPU = one id
# Protocol (fixed for every method, see detr_portability_audit.md §10): 1 GPU x 8 images,
# 1 epoch of target val images, AdamW 1e-4, weight_l=0, final-iteration EMA
# evaluated on corruption TEST with the shared rotated VOC AP50 evaluator.
# Env overrides: CODE (deployed code dir), PY, DATA (RSAR root), DIOR (DIOR
# root), SPLITS (DIOR corruption splits), ART (output root).
set -u
METHOD=$1; DATASET=$2; DOMAIN=$3; SEED=$4; GPU=$5
CODE=${CODE:-/mnt/SSD2_8TB/zechuan/IRAOD-New-detr-sfod-20261101}
CONDA_PREFIX=${CONDA_PREFIX:-/home/zechuan/miniforge3/envs/iraod}
PY=${PY:-$CONDA_PREFIX/bin/python}
DATA=${DATA:-/mnt/shared/zechuan/iraod_data/RSAR}
DIOR=${DIOR:-/mnt/shared/zechuan/iraod_data/DIOR}
SPLITS=${SPLITS:-/home/zechuan/iraod_jobs/dior_splits}
ART=${ART:-/mnt/shared/zechuan/iraod_artifacts/ars_detr/sfod}
export PYTHONPATH=$CODE PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 IRAOD_RUNTIME_READY=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=1
export CONDA_PREFIX LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# Same env whitelist as the R-CNN group's workers; method configs set CGA_*.
for k in $(env | awk -F= '/^(CGA_|SARCLIP_|VLST_)/{print $1}'); do unset "$k"; done
export SARCLIP_PRETRAINED=${SARCLIP_PRETRAINED_PATH:-/home/zechuan/iraod_weights/sarclip/ViT-B-32/vit_b_32_model.safetensors}
CLIP_DIR=${CLIP_DIR:-/mnt/shared/zechuan/iraod_weights/clip}
export HOME=/tmp/iraod_clip_home CLIP_DOWNLOAD_ROOT=$CLIP_DIR CGA_CLIP_CACHE=$CLIP_DIR
mkdir -p $HOME/.cache/clip
[[ -e $HOME/.cache/clip/RN50x64.pt ]] || ln -sf $CLIP_DIR/RN50x64.pt $HOME/.cache/clip/RN50x64.pt

case "$METHOD" in
  A) NAME= ;; B) NAME=st ;; C) NAME=clip_cga ;; D) NAME=sarclip_cga ;;
  E) NAME=vlst ;; F) NAME=vlst_cga ;; PETS) NAME=pets ;; SFUT) NAME=sfut ;; SFYOLO) NAME=sfyolo ;;
  *) echo "bad method $METHOD"; exit 2 ;;
esac
case "$DATASET" in
  rsar)
    SRC_CFG=configs/ars_detr/arcsl_detr_r50_rsar.py
    SRC=/mnt/shared/zechuan/iraod_artifacts/ars_detr/official_rsar/arcsl_detr_r50_rsar_epoch_36.pth
    TRAIN_OPTS=(data.train.img_prefix=$DATA/corruptions/$DOMAIN/val/images/)
    TEST_OPTS=(data.test.ann_file=$DATA/test/annfiles/
               data.test.img_prefix=$DATA/corruptions/$DOMAIN/test/images/) ;;
  dior)
    SRC_CFG=configs/ars_detr/arcsl_detr_r50_dior.py
    SRC=/mnt/shared/zechuan/iraod_artifacts/ars_detr/dior_source/seed42-r50-1gpu-spg8/epoch_72.pth
    TRAIN_OPTS=(data.train.img_prefix=$SPLITS/$DOMAIN/val/)
    TEST_OPTS=(data.test.ann_file=$DIOR/ImageSets/test.txt
               data.test.img_prefix=$SPLITS/$DOMAIN/test/
               "data.test.ann_subdir=$DIOR/Annotations/Oriented Bounding Boxes/") ;;
  *) echo "bad dataset $DATASET"; exit 2 ;;
esac

# SFYOLO reuses the R-CNN group's completed seed-42 TAM fit for this domain.
TAM=${TAM:-/mnt/shared/zechuan/iraod_artifacts/comparison/xaf_student_quant_20260908/tam_runs_84a1aeb/$DATASET/$DOMAIN/seed_42/tam.pth}
[[ "$METHOD" == SFYOLO ]] && TRAIN_OPTS+=(model.cfg.tam_checkpoint=$TAM)

root=$ART/$DATASET/${METHOD}_s${SEED}/$DOMAIN
work=$root/work
mkdir -p "$root"
cd "$CODE"
if [[ "$METHOD" == A ]]; then
  ema=$SRC
elif [[ -f $root/train_done ]]; then
  ema=$(sed -E 's/^ema=([^ ]+).*/\1/' "$root/train_done")
else
  rm -rf "$work"; mkdir -p "$work"
  echo "train_start $METHOD $DATASET $DOMAIN seed=$SEED gpu=$GPU code=$(cat .deploy_commit 2>/dev/null) $(date -Is)" | tee "$root/train.log"
  "$PY" train.py "configs/ars_detr/sfod/detr_${NAME}_${DATASET}_strict.py" \
    --work-dir "$work" --seed "$SEED" --deterministic --no-validate \
    --cfg-options corrupt=$DOMAIN load_from=$SRC model.ema_ckpt=$SRC \
      checkpoint_config.max_keep_ckpts=1 "${TRAIN_OPTS[@]}" \
    >> "$root/train.log" 2>&1
  ec=$?
  # Final EMA = highest iteration number.
  ema=$(ls "$work"/iter_*_ema.pth 2>/dev/null | sed -E 's/.*iter_([0-9]+)_ema\.pth/\1 &/' | sort -n | awk 'END{print $2}')
  if [[ $ec -ne 0 || -z "$ema" ]]; then echo "TRAIN_FAIL $METHOD $DATASET $DOMAIN seed=$SEED ec=$ec" | tee -a "$root/train.log"; exit 1; fi
  echo "ema=$ema $(date -Is)" > "$root/train_done"
fi

out=$root/eval_test
if ! grep -q eval_exit=0 "$out/eval_status" 2>/dev/null; then
  rm -rf "$out"; mkdir -p "$out"
  CGA_SCORER=none "$PY" test.py "$SRC_CFG" "$ema" --eval mAP --out "$out/predictions.pkl" --work-dir "$out" \
    --cfg-options data.samples_per_gpu=8 data.workers_per_gpu=4 "${TEST_OPTS[@]}" > "$out/eval.log" 2>&1
  rec=$?
  echo "eval_exit=$rec method=$METHOD dataset=$DATASET domain=$DOMAIN seed=$SEED ema=$ema $(date -Is)" | tee -a "$out/eval_status"
  [[ $rec -eq 0 ]] || exit 1
fi
echo "CELL_DONE $METHOD $DATASET $DOMAIN seed=$SEED"
