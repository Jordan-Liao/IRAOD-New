import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[2]
HOST_FILE = ROOT / "experiments/comparison/host_binding.py"
BOOTSTRAP = """
from contextlib import nullcontext
import sys
from experiments.comparison import host_binding

host_binding.is_target_host = lambda: True
host_binding.native_config_paths = nullcontext
host_binding.install_native_artifact_reader = lambda: None
sys.argv = sys.argv[1:]
host_binding.main()
"""
FROZEN_ENTRY = """
import argparse
import json
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--local-rank", "--local_rank", type=int, default=0)
parser.add_argument("config")
parser.add_argument("--launcher")
args = parser.parse_args()
print(json.dumps({
    "rank": args.local_rank,
    "argv": sys.argv,
    "config": args.config,
    "launcher": args.launcher,
    "env": {key: os.environ.get(key)
            for key in ("LOCAL_RANK", "RANK", "WORLD_SIZE")},
}))
"""


class HostBindingRankTest(unittest.TestCase):
    def run_bridge(self, rank, injected_flag=None):
        with tempfile.TemporaryDirectory() as directory:
            entry = Path(directory) / "fake_frozen_train.py"
            entry.write_text(textwrap.dedent(FROZEN_ENTRY))
            arguments = ["frozen_config.py", "--launcher", "pytorch"]
            command = [sys.executable, "-S", "-c", BOOTSTRAP, str(HOST_FILE)]
            if injected_flag is not None:
                command.append(injected_flag)
            command.extend(["native", str(entry), *arguments])
            result = subprocess.run(
                command,
                cwd=ROOT,
                env={
                    **os.environ,
                    "LOCAL_RANK": str(rank),
                    "RANK": str(rank),
                    "WORLD_SIZE": "2",
                    "CUDA_VISIBLE_DEVICES": "",
                },
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["config"], "frozen_config.py")
            self.assertEqual(payload["launcher"], "pytorch")
            self.assertEqual(
                payload["env"],
                {"LOCAL_RANK": str(rank), "RANK": str(rank), "WORLD_SIZE": "2"},
            )
            return payload, [str(entry), *arguments]

    def test_injected_rank_aliases(self):
        for alias in ("--local-rank", "--local_rank"):
            for rank in (0, 1):
                with self.subTest(alias=alias, rank=rank):
                    payload, argv = self.run_bridge(rank, f"{alias}={rank}")
                    self.assertEqual(payload["rank"], rank)
                    self.assertEqual(
                        payload["argv"],
                        [argv[0], f"--local-rank={rank}", *argv[1:]],
                    )

    def test_without_injected_rank_preserves_argv(self):
        payload, argv = self.run_bridge(1)
        self.assertEqual(payload["rank"], 0)
        self.assertEqual(payload["argv"], argv)


if __name__ == "__main__":
    unittest.main()
