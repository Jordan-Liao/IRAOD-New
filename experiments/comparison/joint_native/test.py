"""Keep the host wrapper's native TEST finite-check dispatch unchanged."""

from pathlib import Path
import sys

from iraod_runtime import ensure_iraod_runtime

ensure_iraod_runtime()

root = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, root)
from experiments.comparison.joint_roi import capture_native
sys.path.remove(root)


if __name__ == "__main__":
    capture_native()
