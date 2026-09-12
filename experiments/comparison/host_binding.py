"""Operational bindings for the selected five-A6000 host, not a scheduler.

Importing this module requires only stdlib (including from standalone LoRA).
Copied metadata and model checkouts are never rewritten.
"""

import argparse
import ast
from contextlib import contextmanager
import io
import importlib.util
import json
import os
from pathlib import Path
import runpy
import socket
import sys
import tempfile
import tokenize
from types import ModuleType


TARGET_HOST = "73F3-5xA6000-221"
PREFIXES = (
    ("/mnt/shared/zechuan/iraod_artifacts", "/home/zechuan/iraod_artifacts"),
    ("/mnt/shared/zechuan/iraod_data", "/home/zechuan/iraod_data"),
    ("/mnt/shared/zechuan/iraod_weights", "/home/zechuan/iraod_weights"),
    ("/mnt/SSD2_8TB/zechuan", "/home/zechuan"),
    # The owner's selected artifact root is a symlink to this same storage.
    ("/mnt/HDD_14TB/zechuan/iraod_artifacts", "/home/zechuan/iraod_artifacts"),
)
SHARED_LOCK_ROOT = "/home/zechuan/iraod_artifacts/comparison/xaf_s424344/gpu_locks"
PYTHON_PREFIX = "/home/zechuan/miniforge3/envs/iraod"


def is_target_host():
    return socket.gethostname() == TARGET_HOST


def map_path(value):
    """Map one path to the selected lexical home prefix; never resolve a venv."""
    value = str(value)
    if is_target_host():
        for source, target in PREFIXES:
            if value == source or value.startswith(source + "/"):
                return target + value[len(source):]
    return value


def map_data(value):
    """Copy containers, changing only path strings, not keys or scientific data."""
    if isinstance(value, str):
        return map_path(value)
    if isinstance(value, dict):
        return {key: map_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [map_data(item) for item in value]
    if isinstance(value, tuple):
        return tuple(map_data(item) for item in value)
    return value


def same_path(first, second):
    """Compare selected-host artifact identities without resolving interpreter paths."""
    if not is_target_host():
        return first == second
    return Path(map_path(first)).resolve() == Path(map_path(second)).resolve()


def approved_gpus():
    return (0, 1, 2, 3, 4) if is_target_host() else (4, 5, 6, 7)


def pair_ports():
    return ({(0, 1): 29804, (1, 2): 29805, (2, 3): 29806} if is_target_host()
            else {(4, 5): 29804, (6, 7): 29806})


def read_path(value):
    """Follow copied absolute symlinks after mapping each target, not before."""
    path = Path(map_path(value))
    if not is_target_host():
        return path
    while path.is_symlink():
        target = Path(map_path(os.readlink(path)))
        path = target if target.is_absolute() else path.parent / target
    return path


def read_json(value):
    return map_data(json.loads(read_path(value).read_text()))


def path_literals(text, source):
    """Change Python string path literals only; keep imports/scalars verbatim.

    Relative _base_ references must stay relative to the original config, not
    the temporary file MMCV loads. Other relative strings retain native cwd.
    """
    tree = ast.parse(text)
    bases = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "_base_" for t in node.targets):
            bases.update((n.lineno, n.col_offset) for n in ast.walk(node.value)
                         if isinstance(n, ast.Constant) and isinstance(n.value, str))
    tokens = []
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type == getattr(tokenize, "FSTRING_MIDDLE", -1):
            token = token._replace(string=map_path(token.string))
        if token.type == tokenize.STRING:
            expression = ast.parse(token.string, mode="eval").body
            if isinstance(expression, ast.JoinedStr):
                # Resolvers use f-strings for cell-relative suffixes.
                prefix = next((part.value for part in expression.values
                               if isinstance(part, ast.Constant)), "")
                mapped = map_path(prefix)
                if mapped != prefix:
                    token = token._replace(string=token.string.replace(prefix, mapped, 1))
                tokens.append(token)
                continue
            value = ast.literal_eval(expression)
            if isinstance(value, str):
                mapped = map_path(value)
                if token.start in bases and not Path(mapped).is_absolute():
                    mapped = str(Path(source).parent / mapped)
                if mapped != value:
                    token = token._replace(string=repr(mapped))
        tokens.append(token)
    return tokenize.untokenize(tokens)


def load_paths(queue):
    """Execute the copied resolver with path literals bound before its reads."""
    filename = read_path(Path(map_path(queue)) / "paths.py")
    module = ModuleType("bound_xaf_paths")
    module.__file__ = str(filename)
    exec(compile(path_literals(filename.read_text(), filename), str(filename), "exec"),
         module.__dict__)
    for key, value in list(vars(module).items()):
        if callable(value) and getattr(value, "__module__", None) == module.__name__:
            def bound(*args, _function=value, **kwargs):
                return map_data(_function(*args, **kwargs))
            setattr(module, key, bound)
        elif isinstance(value, (str, dict, list, tuple)):
            setattr(module, key, map_data(value))
    return module


@contextmanager
def native_config_paths():
    """Load literal-only temporary config views, including recursive _base_.

    Native imports and model implementation still come from the bound checkout.
    Config.fromfile retains the original filename for native provenance. Temporary
    views live outside all frozen snapshots and are removed after each load.
    """
    if not is_target_host():
        yield
        return
    from mmcv import Config

    original = Config._file2dict

    def file2dict(filename, use_predefined_variables=True):
        source = read_path(filename).absolute()
        with tempfile.TemporaryDirectory(prefix="iraod-bound-config-") as directory:
            view = Path(directory) / source.name
            if use_predefined_variables:
                Config._substitute_predefined_vars(str(source), str(view))
                text = view.read_text()
            else:
                text = source.read_text()
            view.write_text(path_literals(text, source))
            data, text = original(str(view), False)
            return map_data(data), text

    Config._file2dict = staticmethod(file2dict)
    try:
        yield
    finally:
        Config._file2dict = staticmethod(original)


def native_command(command, entry):
    """Insert the path/finite-boundary shim around native entries on target only."""
    command = list(command)
    if is_target_host():
        index = command.index(str(entry))
        command[index:index] = [str(Path(__file__).absolute()), "native"]
    return command


def native_environment():
    """Keep frozen entrypoints from re-execing away the active path-only overlay."""
    env = {**os.environ, "IRAOD_RUNTIME_READY": "1", "IRAOD_CONDA_PREFIX": PYTHON_PREFIX,
           "CONDA_PREFIX": PYTHON_PREFIX, "PYTHONNOUSERSITE": "1",
           "PYTHONDONTWRITEBYTECODE": "1"}
    for key, first in (("PATH", PYTHON_PREFIX + "/bin"),
                       ("LD_LIBRARY_PATH", PYTHON_PREFIX + "/lib")):
        rest = [part for part in env.get(key, "").split(os.pathsep)
                if part and part != first]
        env[key] = os.pathsep.join([first, *rest])
    return env


def install_native_artifact_reader():
    """Use the mapped TAM artifact reader with the unchanged native TAM model.

    Only this operational reader is overlaid. sfod/mmdet model imports continue
    to resolve from the bound code checkout, including its original TAM class.
    """
    sys.modules["experiments.comparison.host_binding"] = sys.modules[__name__]
    name = "experiments.comparison.tam_artifacts"
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("tam_artifacts.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module


def native_integrity(*, evaluate=False):
    """Load current instrumentation without redirecting frozen model imports."""
    spec = importlib.util.spec_from_file_location(
        "iraod_native_integrity", Path(__file__).with_name("numerical_integrity.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.native_boundaries(evaluate=evaluate)


def locked_command(gpus, command):
    """Compatible outer lock entry for LoRA/TAM/TSD; smoke locks itself."""
    from experiments.comparison.finite_resumer import idle_devices, take_lock

    if not is_target_host():
        raise ValueError("This lock entry is only for the selected target host")
    if (not gpus or len(set(gpus)) != len(gpus)
            or not set(gpus).issubset(approved_gpus())
            or len(gpus) > 1 and tuple(gpus) not in pair_ports()):
        raise ValueError("GPU assignment is not an approved host single/pair")
    locks = []
    try:
        for gpu in sorted(gpus):
            lock = take_lock(Path(SHARED_LOCK_ROOT) / f"gpu{gpu}.lock")
            if lock is None:
                raise RuntimeError(f"Shared GPU lock busy: {gpu}")
            locks.append(lock)
        if not set(gpus).issubset(idle_devices(gpus)):
            raise RuntimeError("Assigned GPU occupied; no command invoked")
        import subprocess
        return subprocess.call(command, env={
            **native_environment(), "IRAOD_GPU_LOCKED": "1",
            "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus)),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID"})
    finally:
        for lock in reversed(locks):
            lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-rank", "--local_rank", type=int)
    commands = parser.add_subparsers(dest="action", required=True)
    native = commands.add_parser("native")
    native.add_argument("entry")
    native.add_argument("arguments", nargs=argparse.REMAINDER)
    locked = commands.add_parser("lock")
    locked.add_argument("gpus")
    locked.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.action == "lock":
        command = args.command
        if command[:1] == ["--"]:
            command = command[1:]
        if not command:
            parser.error("lock requires a command")
        raise SystemExit(locked_command(tuple(map(int, args.gpus.split(","))), command))
    entry, arguments = args.entry, args.arguments
    if args.local_rank is not None:
        arguments = [f"--local-rank={args.local_rank}", *arguments]
    sys.path.insert(0, str(Path(entry).parent))
    sys.argv = [entry, *arguments]
    if is_target_host():
        os.environ.update(native_environment())
        install_native_artifact_reader()
    with native_config_paths(), native_integrity(evaluate=Path(entry).name == "test.py"):
        runpy.run_path(entry, run_name="__main__")


if __name__ == "__main__":
    main()
