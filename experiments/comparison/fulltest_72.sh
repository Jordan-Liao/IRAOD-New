#!/usr/bin/env bash
# 72 seed-42 A–F full-TEST cells from results/paper_comparison/checkpoints.json
# MODE=vis|gradcam  DRY_RUN=1  SKIP_DONE=1  SHARD=i/n  GPUS=4,5,6,7
set -euo pipefail

MODE=${MODE:-vis}
DRY_RUN=${DRY_RUN:-1}
SKIP_DONE=${SKIP_DONE:-1}
SHARD=${SHARD:-1/1}
GPUS=${GPUS:-0}
MAX_IMAGES=${MAX_IMAGES:-0}
ONLY=${ONLY:-all}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
CODE=${CODE:-$(cd "$SCRIPT_DIR/../.." && pwd)}

if [[ -x /home/zechuan/anaconda3/envs/iraod/bin/python ]]; then
  PY=${PY:-/home/zechuan/anaconda3/envs/iraod/bin/python}
elif [[ -x /home/zechuan/miniforge3/envs/iraod/bin/python ]]; then
  PY=${PY:-/home/zechuan/miniforge3/envs/iraod/bin/python}
else
  PY=${PY:-python}
fi

if [[ -d /home/zechuan/iraod_artifacts && ! -d /mnt/shared/zechuan/iraod_artifacts ]]; then
  ART=${ART:-/home/zechuan/iraod_artifacts}
  RSAR_DATA=${RSAR_DATA:-/home/zechuan/iraod_data/RSAR}
  DIOR_DATA=${DIOR_DATA:-/home/zechuan/iraod_data/DIOR}
else
  ART=${ART:-/mnt/shared/zechuan/iraod_artifacts}
  RSAR_DATA=${RSAR_DATA:-/mnt/shared/zechuan/iraod_data/RSAR}
  DIOR_DATA=${DIOR_DATA:-/mnt/shared/zechuan/iraod_data/DIOR}
fi

DIOR_SPLITS=${DIOR_SPLITS:-$ART/dior_splits}
if [[ $DRY_RUN == 0 && ! -d $ART ]]; then
  echo "missing ART=$ART" >&2
  exit 2
fi
if [[ $MODE == vis && $MAX_IMAGES -gt 0 ]]; then
  echo "MAX_IMAGES ignored for vis; test.py has no image cap" >&2
fi
if [[ $MODE == vis ]]; then
  SHOW_ROOT=${SHOW_ROOT:-$ART/comparison/show_dir_fulltest}
else
  SHOW_ROOT=${SHOW_ROOT:-$ART/comparison/gradcam_fulltest}
fi

RSAR_CFG=${RSAR_CFG:-$CODE/configs/baseline/oriented_rcnn_orthonet_rsar.py}
DIOR_CFG=${DIOR_CFG:-$ART/dior_source/seed42-orthonet-ddp4-gpu4567-spg16-gbs64/reproducibility/resolved_config.py}
RSAR_A=$ART/rsar_source/seed42-orthonet-ddp4-gpu4567-spg16-gbs64/train/epoch_100.pth
DIOR_A=$ART/dior_source/seed42-orthonet-ddp4-gpu4567-spg16-gbs64/train/epoch_100.pth
GRADCAM_PY=${GRADCAM_PY:-$CODE/tools/analysis/gradcam_obb.py}

RSAR_DOMAINS=(clean chaff gaussian_white_noise point_target noise_suppression am_noise_horizontal smart_suppression am_noise_vertical)
DIOR_DOMAINS=(clean brightness cloudy contrast)
METHODS=(A B C D E F)
EXPECTED_RSAR=8538
EXPECTED_DIOR=11738

IFS=/ read -r SHARD_I SHARD_N <<<"$SHARD"
IFS=, read -r -a GPU_ARR <<<"$GPUS"

ckpt_path() {
  local ds=$1 domain=$2 m=$3
  if [[ $m == A ]]; then
    if [[ $ds == RSAR ]]; then echo "$RSAR_A"; else echo "$DIOR_A"; fi
    return
  fi
  local code=0f98a48 work=work iter
  if [[ $ds == RSAR ]]; then
    iter=iter_266_ema.pth
    [[ $domain == chaff ]] && code=f2145b4
  else
    iter=iter_185_ema.pth
  fi
  case $m in
    E) work=ddp2/work ;;
    F) if [[ $ds == RSAR && $domain == chaff ]]; then work=preflight_ddp2/work; else work=ddp2/work; fi ;;
  esac
  local root=$ART/comparison/rsar
  [[ $ds == DIOR ]] && root=$ART/comparison/dior
  echo "$root/$domain/seed_42/$code/methods/$m/$work/$iter"
}

test_split() {
  local ds=$1 domain=$2
  if [[ $ds == RSAR ]]; then
    echo "$RSAR_DATA/test/annfiles"
    if [[ $domain == clean ]]; then echo "$RSAR_DATA/test/images"
    else echo "$RSAR_DATA/corruptions/$domain/test/images"; fi
  else
    echo "$DIOR_DATA/ImageSets/test.txt"
    if [[ $domain == clean ]]; then echo "$DIOR_DATA/JPEGImages/"
    else echo "$DIOR_SPLITS/$domain/test/"; fi
  fi
}

png_count() {
  local dir=$1
  [[ -d $dir ]] || { echo 0; return; }
  find "$dir" -maxdepth 1 -type f \( -name '*.png' -o -name '*.jpg' -o -name '*.bmp' \) | wc -l
}

n=0
queued=0
run_cell() {
  local ds=$1 domain=$2 m=$3
  n=$((n + 1))
  if (( (n - 1) % SHARD_N != SHARD_I - 1 )); then return; fi
  local cfg ckpt show ann img gpu expected
  if [[ $ds == RSAR ]]; then cfg=$RSAR_CFG; expected=$EXPECTED_RSAR; else cfg=$DIOR_CFG; expected=$EXPECTED_DIOR; fi
  ckpt=$(ckpt_path "$ds" "$domain" "$m")
  show=$SHOW_ROOT/$ds/$domain/$m
  { read -r ann; read -r img; } < <(test_split "$ds" "$domain")
  gpu=${GPU_ARR[$((queued % ${#GPU_ARR[@]}))]}
  queued=$((queued + 1))
  local extra=()
  if [[ $MODE == vis ]]; then
    extra+=(--eval mAP --show-dir "$show" --show-score-thr 0.3)
    local entry=$CODE/test.py
  else
    extra+=(--show-dir "$show" --show-score-thr 0.3)
    [[ $SKIP_DONE == 1 ]] && extra+=(--skip-done)
    [[ $MAX_IMAGES -gt 0 ]] && extra+=(--max-images "$MAX_IMAGES")
    local entry=$GRADCAM_PY
  fi
  local cfgopt=(--cfg-options "data.test.ann_file=$ann" "data.test.img_prefix=$img")
  if [[ $ds == DIOR ]]; then
    cfgopt+=("data.test.ann_subdir=$DIOR_DATA/Annotations/obb/")
  fi
  echo "# $n $ds $domain $m gpu=$gpu"
  echo "RSAR_ROOT=$RSAR_DATA CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$CODE PYTHONNOUSERSITE=1 IRAOD_RUNTIME_READY=1 \\"
  echo "  $PY $entry $cfg $ckpt \\"
  echo "  ${extra[*]} \\"
  echo "  ${cfgopt[*]}"
  echo
  if [[ $DRY_RUN != 0 ]]; then return; fi
  if [[ $SKIP_DONE == 1 && $MODE == vis ]]; then
    local have
    have=$(png_count "$show")
    if [[ $MAX_IMAGES -gt 0 ]]; then expected=$MAX_IMAGES; fi
    if (( have >= expected )); then echo "skip_done $have/$expected $show"; return; fi
  fi
  mkdir -p "$show"
  wait_slot
  (
    export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu
    export PYTHONPATH=$CODE PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 IRAOD_RUNTIME_READY=1
    export RSAR_ROOT=$RSAR_DATA
    cd "$CODE"
    "$PY" "$entry" "$cfg" "$ckpt" "${extra[@]}" "${cfgopt[@]}"
  ) >"$show/run.log" 2>&1 &
}

wait_slot() {
  local max=${#GPU_ARR[@]}
  while (( $(jobs -pr | wc -l) >= max )); do
    wait -n || true
  done
}

if [[ $ONLY != DIOR ]]; then
  for domain in "${RSAR_DOMAINS[@]}"; do
    for m in "${METHODS[@]}"; do run_cell RSAR "$domain" "$m"; done
  done
fi
if [[ $ONLY != RSAR ]]; then
  for domain in "${DIOR_DOMAINS[@]}"; do
    for m in "${METHODS[@]}"; do run_cell DIOR "$domain" "$m"; done
  done
fi
if [[ $DRY_RUN == 0 ]]; then wait || true; fi
echo "# listed=$n shard=$SHARD launched=$queued mode=$MODE dry=$DRY_RUN"
