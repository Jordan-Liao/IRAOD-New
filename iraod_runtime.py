import os
import sys
from pathlib import Path


_READY_ENV = "IRAOD_RUNTIME_READY"
_DEFAULT_PREFIX = "/home/liaojr/anaconda3/envs/iraod"


def _prepend_env_path(env, key, value):
    current = env.get(key, "")
    parts = [part for part in current.split(os.pathsep) if part]
    if value in parts:
        parts.remove(value)
    env[key] = os.pathsep.join([value] + parts)


def ensure_iraod_runtime():
    if os.environ.get(_READY_ENV) == "1":
        return

    prefix = Path(os.environ.get("IRAOD_CONDA_PREFIX", _DEFAULT_PREFIX)).expanduser()
    python_bin = prefix / "bin" / "python"
    lib_dir = prefix / "lib"

    env = os.environ.copy()
    env[_READY_ENV] = "1"
    env["CONDA_PREFIX"] = str(prefix)
    env.setdefault("MPLCONFIGDIR", "/tmp/iraod_mplconfig")
    env.setdefault("XDG_CACHE_HOME", "/tmp/iraod_cache")

    _prepend_env_path(env, "PATH", str(prefix / "bin"))
    _prepend_env_path(env, "LD_LIBRARY_PATH", str(lib_dir))

    os.makedirs(env["MPLCONFIGDIR"], exist_ok=True)
    os.makedirs(env["XDG_CACHE_HOME"], exist_ok=True)

    executable = str(python_bin if python_bin.exists() else Path(sys.executable))
    os.execvpe(executable, [executable] + sys.argv, env)
