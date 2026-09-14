"""Replay the real joint child across candidate and frozen331d namespaces on CPU."""

import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import types
import unittest


ROOT = Path(__file__).resolve().parents[2]
RELATIVE_EXPORTER = "experiments/comparison/dior_recovery/extract_roi_pre_fc_cls.py"


def _entry_child(frozen):
    # Import-only fixtures: any detector/checkpoint/forward entry remains unreachable.
    for name in ("torch", "mmcv", "mmdet", "mmrotate", "mmrotate.models",
                 "mmrotate.models.roi_heads", "mmrotate.models.roi_heads.bbox_heads",
                 "mmrotate.models.roi_heads.bbox_heads.rotated_bbox_head"):
        module = types.ModuleType(name)
        module.__path__ = []
        module.__version__ = "CPU_IMPORT_FIXTURE"
        sys.modules[name] = module
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(sys.modules[parent], child, module)
    sys.modules["torch"].version = types.SimpleNamespace(cuda=None)
    guard = types.ModuleType("joint_import_fixture_guard")
    guard.reached = False
    sys.modules[guard.__name__] = guard
    frozen = Path(frozen)
    wrapper = ROOT / "experiments/comparison/joint_native/test.py"
    sys.argv = [
        str(wrapper), "--case-root", str(frozen / "candidate"), "--dataset", "DIOR",
        "--physical-gpu", "8", "--", str(frozen / "test.py"), "fixture-config", "fixture-checkpoint",
    ]
    os.environ.update(IRAOD_RUNTIME_READY="1", IRAOD_GPU_LOCKED="1", CUDA_VISIBLE_DEVICES="8")
    runpy.run_path(str(wrapper), run_name="__main__")
    from experiments.comparison import joint_roi
    import inspect
    helper_file = Path(inspect.getsourcefile(joint_roi.runtime_provenance)).resolve()
    assert helper_file == ROOT / RELATIVE_EXPORTER
    assert guard.reached
    assert guard.argv == [str(frozen / "test.py"), "fixture-config", "fixture-checkpoint"]
    assert not (frozen / "candidate").exists()
    metadata = joint_roi.runtime_provenance(
        types.SimpleNamespace(__version__="CPU_IMPORT_FIXTURE"),
        sys.modules["torch"], sys.modules["mmcv"], sys.modules["mmdet"], 8)
    print(json.dumps({
        "status": "OWNED_HELPER_AND_FROZEN_NATIVE_ENTRY_PASS",
        "helper_file": str(helper_file), "native_entry": guard.argv[0],
        "cuda": metadata["cuda"], "model_checkpoint_forward_calls": 0,
    }))


class JointImportTest(unittest.TestCase):
    def test_runtime_helper_stays_with_exporter_across_native_path_switch(self):
        with tempfile.TemporaryDirectory() as temporary:
            frozen = Path(temporary) / "frozen331d"
            old_exporter = frozen / RELATIVE_EXPORTER
            old_exporter.parent.mkdir(parents=True)
            old_exporter.write_bytes(subprocess.check_output([
                "git", "-C", str(ROOT), "show", "331d213:" + RELATIVE_EXPORTER]))
            (frozen / "test.py").write_text(
                "def single_gpu_test(*args, **kwargs):\n"
                "    raise AssertionError('No forward in import regression')\n"
                "def load_checkpoint(*args, **kwargs):\n"
                "    raise AssertionError('No checkpoint in import regression')\n"
                "def wrap_fp16_model(*args, **kwargs):\n"
                "    raise AssertionError('No model in import regression')\n"
                "def main():\n"
                "    import sys\n"
                "    import joint_import_fixture_guard as guard\n"
                "    guard.reached = True\n"
                "    guard.argv = list(sys.argv)\n")
            command = [
                sys.executable, "-c",
                "import runpy,sys; runpy.run_path(sys.argv[1])['_entry_child'](sys.argv[2])",
                str(Path(__file__).resolve()), str(frozen),
            ]
            result = subprocess.run(command, cwd=frozen, env={
                **os.environ, "PYTHONPATH": str(ROOT), "CUDA_VISIBLE_DEVICES": "",
                "PYTHONDONTWRITEBYTECODE": "1",
            }, capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            evidence = json.loads(result.stdout)
            self.assertEqual(evidence["status"], "OWNED_HELPER_AND_FROZEN_NATIVE_ENTRY_PASS")
            self.assertEqual(evidence["model_checkpoint_forward_calls"], 0)
            self.assertIsNone(evidence["cuda"])


if __name__ == "__main__":
    unittest.main()
