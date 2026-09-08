#!/usr/bin/env bash
# Run on gpu-67, or stream with: ssh gpu-67 'bash -s -- matrix 4 4,5' < this-file
set -euo pipefail
Q=/mnt/shared/zechuan/iraod_artifacts/comparison/dior/queue_0f98a48

cell() {
  local method=$1 corruption=$2 gpu=$3 pair=${4:-4,5}
  case "$corruption" in
    brightness|cloudy|contrast) ;;
    *) echo "Unsupported corruption: $corruption" >&2; return 2 ;;
  esac
  case "$method" in
    A) bash "$Q/run_A_eval.sh" "$gpu" "$corruption" ;;
    B|C|D)
      bash "$Q/run_train_bcd.sh" "$gpu" "$method" "$corruption"
      bash "$Q/run_eval_ema.sh" "$gpu" "$method" "$corruption"
      ;;
    E|F)
      bash "$Q/run_train_ef.sh" "$pair" "$method" "$corruption" 29711
      bash "$Q/run_eval_ema.sh" "$gpu" "$method" "$corruption"
      ;;
    *) echo "Unsupported method: $method" >&2; return 2 ;;
  esac
}

case "${1:-help}" in
  source)
    bash "$Q/run_source_eval.sh" "${2:-4}" val
    bash "$Q/run_source_eval.sh" "${2:-4}" test
    ;;
  cell)
    cell "${2:?method required}" "${3:?corruption required}" "${4:-4}" "${5:-4,5}"
    ;;
  matrix)
    gpu=${2:-4}
    pair=${3:-4,5}
    bash "$Q/run_source_eval.sh" "$gpu" val
    bash "$Q/run_source_eval.sh" "$gpu" test
    for corruption in brightness cloudy contrast; do
      for method in A B C D E F; do
        cell "$method" "$corruption" "$gpu" "$pair"
      done
    done
    ;;
  *)
    echo "Usage: bash $0 source [gpu] | cell METHOD CORRUPTION [gpu] [gpu_pair] | matrix [gpu] [gpu_pair]" >&2
    exit 2
    ;;
esac
