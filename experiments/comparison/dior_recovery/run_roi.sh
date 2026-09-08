#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 3 ]; then
  echo "usage: IRAOD_PYTHON=/path/to/python $0 GPU PLAN RUN_ID" >&2
  exit 2
fi
: "${IRAOD_PYTHON:?Set the detector environment Python explicitly}"
CODE=$(cd "$(dirname "$0")/../../.." && pwd)
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$1"
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
exec "$IRAOD_PYTHON" -m experiments.comparison.dior_recovery.extract_roi_pre_fc_cls \
  --plan "$2" --run-id "$3"
