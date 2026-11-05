#!/usr/bin/env bash
# Canonical LODS SEM store: one seed-42, 160000-iter fit per (dataset, domain),
# trained on gpu-67 from the unlabeled val images and kept with its sha256 at
#   /mnt/shared/zechuan/iraod_artifacts/comparison/<dataset>/<domain>/lods_sem_seed42/sem_iter_160000.pth{,.sha256}
# Every seed (42/43/44) of that domain uses this one file.
#
#   on gpu-67:          lods_sem.sh train rsar|dior DOMAIN GPU
#   on the controller:  lods_sem.sh push  rsar|dior DOMAIN HOST DEST_DIR   (copy + sha256 check on HOST)
set -euo pipefail
CANON_ROOT=/mnt/shared/zechuan/iraod_artifacts/comparison
NAME=sem_iter_160000.pth
ENC=/mnt/shared/zechuan/iraod_artifacts/third_party/vgg16_oxford/encoder_split_enc_keys.pt
ENC_SHA=5d5e4b9416d37ace4144beaafff7ec9acc1977c3281d504fefed670395156a8f
ACTION=$1; DATASET=$2; DOMAIN=$3
CANON=$CANON_ROOT/$DATASET/$DOMAIN/lods_sem_seed42

case "$ACTION" in
  train)
    GPU=$4
    CODE=/mnt/SSD2_8TB/zechuan/IRAOD-New-lods-20261101
    PY=/home/zechuan/miniforge3/envs/iraod/bin/python
    case "$DATASET" in
      rsar) TARGET=/mnt/shared/zechuan/iraod_data/RSAR/corruptions/$DOMAIN/val/images ;;
      dior) TARGET=/mnt/shared/zechuan/iraod_artifacts/dior_splits/$DOMAIN/val ;;
      *) echo "bad dataset $DATASET"; exit 2 ;;
    esac
    if [ -f "$CANON/$NAME.sha256" ]; then echo "SEM exists: $CANON/$NAME"; exit 0; fi
    echo "$ENC_SHA  $ENC" | sha256sum -c --quiet
    mkdir -p "$CANON"
    ( cd "$CODE" && CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$CODE "$PY" tools/lods/train_sem.py \
        --target-dir "$TARGET" --encoder "$ENC" --work-dir "$CANON" --seed 42 \
        --workers 32 --max-iter 160000 > "$CANON/stdout.log" 2>&1 )
    ( cd "$CANON" && sha256sum "$NAME" > "$NAME.sha256" )
    echo "SEM_DONE $CANON/$NAME $(cut -c1-16 "$CANON/$NAME.sha256")"
    ;;
  push)
    HOST=$4; DEST=$5
    ssh zechuan@"$HOST" "mkdir -p '$DEST'"
    scp -3 "zechuan@gpu-67:$CANON/$NAME" "zechuan@gpu-67:$CANON/$NAME.sha256" "zechuan@$HOST:$DEST/"
    ssh zechuan@"$HOST" "cd '$DEST' && sha256sum -c '$NAME.sha256'"
    ;;
  *) echo "usage: lods_sem.sh train|push ..."; exit 2 ;;
esac
