#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

if [[ -z "${IRAOD_PYTHON:-}" ]]; then
  echo "IRAOD_PYTHON is required" >&2
  exit 2
fi

if [[ -z "${AUTO_RESEARCH_ROOT:-}" ]]; then
  echo "AUTO_RESEARCH_ROOT is required" >&2
  exit 2
fi

if [[ ! -x "$IRAOD_PYTHON" ]]; then
  echo "IRAOD_PYTHON is not executable: $IRAOD_PYTHON" >&2
  exit 2
fi

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

cd -- "$PROJECT_ROOT"
exec "$IRAOD_PYTHON" "$PROJECT_ROOT/tools/cga_research/gpu_scheduler.py" \
  --project-root "$PROJECT_ROOT" \
  --research-root "$AUTO_RESEARCH_ROOT" \
  --python "$IRAOD_PYTHON" \
  "$@"
