#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 2 ]; then
  echo "usage: IRAOD_PYTHON=/path/to/python $0 PLAN RUN_ID (CPU render)" >&2
  exit 2
fi
: "${IRAOD_PYTHON:?Set the detector environment Python explicitly}"
CODE=$(cd "$(dirname "$0")/../../.." && pwd)
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}" PYTHONNOUSERSITE=1
exec "$IRAOD_PYTHON" -m experiments.comparison.result_completion visualize \
  --plan "$1" --run-id "$2"
