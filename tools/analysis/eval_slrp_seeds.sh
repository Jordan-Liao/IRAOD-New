#!/usr/bin/env bash
# Evaluate SLRP calibration checkpoints across corruptions, one row per seed.
#
# The point is seed-to-seed spread of the *delta* against the zero-init baseline.
# A single calibration run cannot tell you whether +14.5 is the method or the draw,
# and the SFOD noise floor (~7 mAP) does not transfer here: this is static inference
# on a fixed checkpoint, so the evaluation itself is deterministic and any spread
# comes from calibration training alone.
set -u

PY=/opt/conda/envs/iraod/bin/python
CFG=configs/baseline/oriented_rcnn_slrp_orthonet_rsar.py
ROOT=/myfile/dataset/RSAR
CORRUPTIONS="${CORRUPTIONS:-noise_suppression am_noise_horizontal smart_suppression chaff}"
CKPTS="${CKPTS:-}"

if [ -z "$CKPTS" ]; then
  echo "usage: CKPTS='label=path label=path' $0" >&2
  exit 2
fi

for entry in $CKPTS; do
  label="${entry%%=*}"
  ckpt="${entry#*=}"
  if [ ! -f "$ckpt" ]; then
    echo "MISSING $label -> $ckpt" >&2
    continue
  fi
  for c in $CORRUPTIONS; do
    printf '%-14s %-22s ' "$label" "$c"
    timeout 18000 "$PY" test.py "$CFG" "$ckpt" --eval mAP \
      --cfg-options "data.test.img_prefix=$ROOT/corruptions/$c/test/images/" \
      2>/dev/null | grep -ao "{'mAP': [0-9.]*}" | tail -1
  done
done
echo EVALDONE
