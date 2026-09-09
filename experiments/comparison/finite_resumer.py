"""Finite xaf resumption: canonical tmux jobs, shared locks and pidfd exit events."""

import argparse
from contextlib import contextmanager
from copy import deepcopy
import csv
import ctypes
from dataclasses import asdict, dataclass
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import platform
import re
import selectors
import shlex
import signal
import subprocess
import sys
import time

from experiments.comparison.result_completion import DOMAINS
from experiments.comparison import host_binding as host


APPROVED = host.approved_gpus()
PAIR_PORTS = host.pair_ports()
EVAL_SHA = "331d2131b84651f0a2930a3d53faeefad8701531"
FORMAL_PORT_METHODS = ("IRG", "LPLD", "SFUT", "AASFOD", "SFYOLO",
                       "B_REG", "F_text_only", "F_veto_only", "LoRA-CGA", "LoRA-CGA+VLST")
STUDENT_METHODS = (*tuple("BCDEF"), "IRG", "LPLD", "SFUT", "AASFOD", "SFYOLO")
PREREQUISITE_METHODS = ("AASFOD", "SFYOLO", "LoRA-CGA", "LoRA-CGA+VLST")
SCRIPT = Path(__file__).resolve()
LIBC = ctypes.CDLL(None, use_errno=True)
LIBC.syscall.restype = ctypes.c_long


class Blocked(RuntimeError):
    pass


def open_pidfd(pid):
    # The actual x86_64 host supports pidfds, but its Conda Python omits os.pidfd_open.
    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise Blocked("This runner requires the deployed Linux x86_64 pidfd interface")
    fd = LIBC.syscall(ctypes.c_long(434), ctypes.c_int(pid), ctypes.c_uint(0))
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return int(fd)


@dataclass(frozen=True)
class Cell:
    dataset: str
    domain: str
    seed: int
    method: str
    role: str = "ema"

    @property
    def model(self):
        """EMA and Student are two evaluations of the same trained model."""
        return Cell(self.dataset, self.domain, self.seed, self.method)

    @property
    def key(self):
        suffix = "/student" if self.role == "student" else ""
        return f"{self.dataset}/{self.domain}/{self.seed}/{self.method}{suffix}"

    def session(self, phase):
        if self.role == "student" and phase == "eval":
            return f"xafS-{self.dataset}-{self.domain}-{self.seed}-{self.method}"
        suffix = "-student" if self.role == "student" else ""
        return f"{'xaf' if phase == 'train' else 'xafE'}-{self.dataset}-{self.domain}-{self.seed}-{self.method}{suffix}"

    @property
    def width(self):
        return 2 if self.role == "ema" and self.method in (
            "E", "F", "F_text_only", "F_veto_only", "LoRA-CGA+VLST") else 1


def load_cells(train_files, eval_files):
    cells = {}
    for training, files in ((True, train_files), (False, eval_files)):
        for path in files:
            for line in host.read_path(path).read_text().splitlines():
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                fields = line.split()
                if len(fields) not in (4, 5):
                    raise ValueError("Finite rows require DS DOMAIN SEED METHOD [ema|student]")
                ds, domain, seed, method = fields[:4]
                cell = Cell(ds, domain, int(seed), method, fields[4] if len(fields) == 5 else "ema")
                if (ds not in DOMAINS or domain not in DOMAINS[ds] or cell.seed not in (42, 43, 44)
                        or method not in (*tuple("ABCDEF"), *FORMAL_PORT_METHODS)
                        or (method in ("LoRA-CGA", "LoRA-CGA+VLST") and ds != "DIOR")
                        or (training and method == "A")
                        or (method == "A" and cell.seed != 42)
                        or cell.role not in ("ema", "student")
                        or (cell.role == "student" and (training or method not in STUDENT_METHODS))):
                    raise ValueError(f"Cell outside the approved finite protocol: {line}")
                cells[cell] = cells.get(cell, False) or training
    if not cells:
        raise ValueError("Supply a nonempty finite train/eval list")
    return cells


def load_work(args):
    cells = load_cells(args.train_list, args.eval_list + args.external_train_list)
    external = set(load_cells(args.external_train_list, [])) if args.external_train_list else set()
    for cell in external:
        cells[cell] = False
    owners = {}
    for value in args.external_owner_session:
        name, spec = value.split("=", 1)
        gpus = tuple(int(g) for g in spec.split(","))
        if not external or not name or not gpus or len(set(gpus)) != len(gpus) or not set(gpus).issubset(APPROVED):
            raise ValueError("External owner sessions require reserved training cells and approved GPUs")
        if name in owners:
            raise ValueError("Duplicate external owner session")
        owners[name] = gpus
    return cells, external, owners


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def lock_root(queue):
    if host.is_target_host():
        return Path(host.SHARED_LOCK_ROOT)
    for line in (Path(queue) / "with_gpu_lock.sh").read_text().splitlines():
        if line.startswith("LOCKDIR="):
            values = shlex.split(line.split("=", 1)[1])
            if len(values) == 1 and Path(values[0]).is_absolute() and "$" not in values[0]:
                return Path(values[0])
    raise Blocked("Cannot resolve the actual shared GPU lock directory")


def take_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+")
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        return None
    return stream


def last_exit(path, names):
    path = Path(path)
    if not path.is_file():
        return None
    matches = re.findall(r"\b(?:" + "|".join(names) + r")=(-?\d+)\b", path.read_text())
    return int(matches[-1]) if matches else None


def load_paths(queue):
    if host.is_target_host():
        return host.load_paths(queue)
    spec = importlib.util.spec_from_file_location("finite_xaf_paths", Path(queue) / "paths.py")
    module = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old
    return module


def train_state(queue, paths, cell, check_wrap=True):
    cell = cell.model
    args = (cell.dataset, cell.domain, str(cell.seed), cell.method)
    ema = Path(paths.ema_path(*args))
    if cell.method == "A":
        return "complete" if ema.is_file() and ema.stat().st_size else "pending"
    student = Path(paths.student_path(*args))
    finals = all(p.is_file() and p.stat().st_size > 0 for p in (ema, student))
    if not finals:
        failed = (ema.parent.exists()
                  or last_exit(ema.parent.parent / "terminal_status",
                               ("tmux_wrap_exit", "launcher_exit")) is not None
                  or check_wrap and not wrappers_complete(queue, paths, cell, "train"))
        return "blocked" if ema.exists() or student.exists() or failed else "pending"
    terminal = ema.parent.parent / "terminal_status"
    rc = last_exit(terminal, ("tmux_wrap_exit", "launcher_exit"))
    if rc == 0 and (not check_wrap or wrappers_complete(queue, paths, cell, "train")):
        return "complete"
    # Existing final files with conflicting terminal evidence must not be retrained.
    return "blocked"


def prerequisite_state(paths, cell):
    if cell.role == "student" or cell.method not in PREREQUISITE_METHODS:
        return "ready", ""
    from experiments.comparison.extension_training import require_prerequisites

    try:
        require_prerequisites(paths.binding(cell.dataset, cell.domain, str(cell.seed), cell.method))
    except (OSError, ValueError, KeyError, RuntimeError, EOFError, pickle.UnpicklingError) as error:
        return "waiting", f"{cell.method} prerequisite: {error}"
    return "ready", ""


def input_state(queue, paths, cell, phase):
    """Read prepared inputs only; never load a detector or change its binding."""
    if not host.is_target_host() and not getattr(paths, "DATA", {}).get("schema"):
        return prerequisite_state(paths, cell) if phase == "train" else ("ready", "")
    from experiments.comparison.extension_training import load_cell, require_file, require_prerequisites

    try:
        _, binding = load_cell(queue, cell.dataset, cell.domain, cell.seed, cell.method, cell.role)
        if phase == "train":
            require_file(binding["source_checkpoint"])
            images = host.read_path(binding["target_val"])
            if not images.is_dir():
                raise ValueError(f"Missing target image directory: {images}")
            # Match StrictSourceFreeDOTADataset's recursive image discovery;
            # no annotations, sampling or image decoding belong in admission.
            count = sum(p.is_file() and p.suffix.lower() in
                        (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
                        for p in images.rglob("*"))
            if count < binding["unlabeled_epoch_size"]:
                raise ValueError(f"Insufficient target images: {images}: "
                                 f"{count} < {binding['unlabeled_epoch_size']}")
            require_prerequisites(binding)
        else:
            require_file(binding["checkpoint"])
            annotations = host.read_path(binding["ann_file"])
            if cell.dataset == "DIOR":
                require_file(annotations)
            elif not annotations.is_dir():
                raise ValueError(f"Missing TEST annotation directory: {annotations}")
            if not host.read_path(binding["img_prefix"]).is_dir():
                raise ValueError(f"Missing TEST image directory: {binding['img_prefix']}")
    except (OSError, ValueError, KeyError, RuntimeError, EOFError, pickle.UnpicklingError) as error:
        return "waiting", f"{phase} input: {error}"
    return "ready", ""


def allowed_gpus(paths):
    # Historical core queues intentionally retain their original four-card API.
    if host.is_target_host():
        return host.approved_gpus()
    return tuple(g for g in getattr(paths, "DATA", {}).get("allowed_gpus", APPROVED)
                 if g in APPROVED)


def wrappers_complete(queue, paths, cell, phase):
    origin = getattr(paths, "DATA", {}).get("source_queues", {}).get(cell.key, queue)
    return all(last_exit(Path(q) / f"wrap_{cell.session(phase)}.status", ("wrap_exit",)) in (None, 0)
               for q in {str(queue), str(origin)})


def eval_state(queue, paths, cell, check_wrap=True):
    resolve = paths.eval_student_dir if cell.role == "student" else paths.eval_full_dir
    out = Path(resolve(cell.dataset, cell.domain, str(cell.seed), cell.method))
    pred = out / "predictions.pkl"
    sidecar = out / "predictions.pkl.image_ids.json"
    if not pred.exists() and not sidecar.exists():
        return ("blocked" if out.exists()
                or check_wrap and not wrappers_complete(queue, paths, cell, "eval") else "pending")
    required = (pred, sidecar, out / "class_ap.txt", out / "pred_count.txt")
    if not all(p.is_file() and p.stat().st_size > 0 for p in required):
        return "blocked"
    status_file = out / "eval_status"
    if not status_file.is_file():
        return "blocked"
    lines = [line for line in status_file.read_text().splitlines() if line.startswith("eval_exit=")]
    fields = (dict(token.split("=", 1) for token in shlex.split(lines[-1]) if "=" in token)
              if lines else {})
    expected_status = {"eval_exit": "0", "name": cell.method,
                       "domain": cell.domain, "seed": str(cell.seed)}
    if cell.role == "student":
        expected_status["role"] = "student"
    if any(fields.get(key) != value for key, value in expected_status.items()):
        return "blocked"
    if check_wrap and not wrappers_complete(queue, paths, cell, "eval"):
        return "blocked"
    try:
        order = json.loads(sidecar.read_text())
        metrics = list(out.glob("eval_*.json"))
        if len(metrics) != 1 or "MISSING_CLASS_TABLE" in (out / "class_ap.txt").read_text():
            return "blocked"
        metric = json.loads(metrics[0].read_text())
        value = float(metric["metric"]["mAP"])
        if (not math.isfinite(value) or not 0 <= value <= 1
                or not host.same_path(metric["config"], order["config"])):
            return "blocked"
        n = paths.EXPECT_PRED[cell.dataset]
        ids = order["image_ids"]
        records = order["records"]
        checkpoint = paths.student_path if cell.role == "student" else paths.ema_path
        expected_checkpoint = checkpoint(cell.dataset, cell.domain, str(cell.seed), cell.method)
        if (order["schema"] != "iraod-prediction-image-order-v1"
                or order["origin"] != "inference_batch_img_metas" or order["status"] != "complete"
                or not host.same_path(order["checkpoint"], expected_checkpoint)
                or order["evaluation_code_sha"] != getattr(paths, "EVALUATION_CODE_SHA", EVAL_SHA)
                or order["predictions_file"] != pred.name
                or order["n_images"] != n or order["dataset_size"] != n
                or len(ids) != n or len(set(ids)) != n or len(records) != n
                or int((out / "pred_count.txt").read_text().split()[0]) != n):
            return "blocked"
        if any(r["prediction_index"] != i or r["image_id"] != ids[i]
               or Path(r["ori_filename"]).stem != ids[i] for i, r in enumerate(records)):
            return "blocked"
        # Trusted owner-produced pickle: validate actual rows, not just a count sentinel.
        with pred.open("rb") as stream:
            predictions = pickle.load(stream)
        return "complete" if isinstance(predictions, list) and len(predictions) == n else "blocked"
    except (OSError, ValueError, KeyError, TypeError, EOFError, pickle.UnpicklingError):
        return "blocked"


def runner_command(queue, cell, phase, gpus):
    if host.is_target_host():
        if (not set(gpus).issubset(host.approved_gpus())
                or len(gpus) != (cell.width if phase == "train" else 1)):
            raise ValueError("Invalid target-host GPU assignment")
        if cell.role != "ema" and phase == "train":
            raise ValueError("Student quantitative extensions are evaluation-only")
        # Never execute the copied Bash wrappers (old absolute paths/GPU IDs).
        from experiments.comparison.extension_training import load_cell
        runtime, _ = load_cell(queue, cell.dataset, cell.domain, cell.seed, cell.method, cell.role)
        action = "evaluate" if phase == "eval" else "train-ddp" if len(gpus) == 2 else "train"
        command = [runtime["python"], str(SCRIPT.with_name("extension_training.py")), action,
                   "--queue", host.map_path(queue), "--dataset", cell.dataset,
                   "--domain", cell.domain, "--seed", str(cell.seed), "--method", cell.method]
        if len(gpus) == 2:
            command += ["--gpu-pair", ",".join(map(str, gpus)),
                        "--port", str(host.pair_ports()[tuple(gpus)])]
        else:
            command += ["--gpu", str(gpus[0])]
        if phase == "eval":
            command += ["--role", cell.role]
        return command
    args = [cell.dataset, cell.domain, str(cell.seed), cell.method]
    if phase == "eval":
        role = [cell.role] if cell.role == "student" else []
        return ["bash", str(Path(queue) / "run_eval_full.sh"), str(gpus[0]), *args, *role]
    if cell.role != "ema":
        raise ValueError("Student quantitative extensions are evaluation-only")
    if cell.width == 1:
        return ["bash", str(Path(queue) / "run_train_1gpu.sh"), str(gpus[0]), *args]
    pair = tuple(gpus)
    if pair not in PAIR_PORTS:
        raise ValueError("The actual two-GPU API only supports 4,5 and 6,7")
    return ["bash", str(Path(queue) / "run_train_2gpu.sh"), ",".join(map(str, pair)),
            str(PAIR_PORTS[pair]), *args]


def idle_devices(approved):
    ids = ",".join(map(str, approved))
    gpu = subprocess.check_output([
        "nvidia-smi", "-i", ids, "--query-gpu=index,uuid,memory.used",
        "--format=csv,noheader,nounits"], text=True)
    apps = subprocess.check_output([
        "nvidia-smi", "-i", ids, "--query-compute-apps=gpu_uuid,pid",
        "--format=csv,noheader,nounits"], text=True)
    busy_uuids = {r[0].strip() for r in csv.reader(apps.splitlines()) if r}
    observed, free = set(), set()
    for row in csv.reader(gpu.splitlines()):
        number, uuid, memory = int(row[0]), row[1].strip(), float(row[2])
        observed.add(number)
        if number in approved and uuid not in busy_uuids and memory < 500:
            free.add(number)
    if not set(approved).issubset(observed):
        raise Blocked("Incomplete GPU availability response")
    return free


def old_producers(queue):
    script = str(Path(queue) / "resume_empty_gpu.sh")
    found = []
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            if path.stat().st_uid != os.getuid():
                continue
            argv = path.read_bytes().decode().strip("\0").split("\0")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if argv and Path(argv[0]).name in ("bash", "sh") and (
                script in argv or any(script in a and "bash " + script in a for a in argv[1:])):
            found.append(int(path.parent.name))
    return found


def pane_job(name, pane, queue):
    """Read the canonical job's actual command; never infer GPUs from its name."""
    lexer = shlex.shlex(pane["command"], posix=True, punctuation_chars=";&|")
    lexer.whitespace_split = True
    tokens = list(lexer)
    for i, token in enumerate(tokens):
        if Path(token).name == SCRIPT.name and i + 2 < len(tokens) and tokens[i + 1] == "worker":
            job = json.loads(Path(tokens[i + 2]).read_text())
            cell = Cell(**job["cell"])
            if cell.session(job["phase"]) != name:
                raise Blocked(f"Canonical job identity mismatch: {name}")
            return cell, job["phase"], tuple(job["gpus"]), Path(tokens[i + 2])
        filename = Path(token).name
        if filename in ("run_train_1gpu.sh", "run_train_2gpu.sh",
                        "run_eval_full.sh", "run_eval_student.sh"):
            start = i + (3 if filename == "run_train_2gpu.sh" else 2)
            ds, domain, seed, method = tokens[start:start + 4]
            phase = "eval" if filename in ("run_eval_full.sh", "run_eval_student.sh") else "train"
            role = ("student" if filename == "run_eval_student.sh"
                    or phase == "eval" and tokens[start + 4:start + 5] == ["student"]
                    else "ema")
            cell = Cell(ds, domain, int(seed), method, role)
            parent = Path(token).parent
            allowed_parent = parent == Path(queue)
            if not allowed_parent:
                origin = getattr(load_paths(queue), "DATA", {}).get("source_queues", {}).get(cell.key)
                allowed_parent = origin is not None and parent == Path(origin)
            if not allowed_parent and filename == "run_eval_student.sh":
                legacy = getattr(load_paths(queue), "LEGACY_STUDENT_QUEUE", None)
                allowed_parent = legacy is not None and parent == Path(legacy)
            if cell.session(phase) != name or not allowed_parent:
                raise Blocked(f"Canonical runner identity mismatch: {name}")
            return cell, phase, tuple(int(g) for g in tokens[i + 1].split(",")), None
    # A bash -c wrapper is one token containing its inner command.
    for token in tokens:
        if ("run_train_" in token or "run_eval_full.sh" in token
                or "run_eval_student.sh" in token or SCRIPT.name in token):
            if token != pane["command"] and " " in token:
                return pane_job(name, {**pane, "command": token}, queue)
    raise Blocked(f"Cannot identify the actual runner/GPU assignment for {name}")


class TmuxBackend:
    def __init__(self, queue, run_dir, gpus=APPROVED, tmux=("tmux",)):
        self.queue, self.run_dir = Path(host.map_path(queue)), Path(host.map_path(run_dir))
        self.gpus, self.tmux = tuple(gpus), tuple(tmux)
        self.lock_dir = lock_root(queue)
        self.paths = load_paths(queue)
        if not set(self.gpus).issubset(allowed_gpus(self.paths)):
            raise Blocked("GPU assignment exceeds this prepared queue's allowed_gpus")
        self.selector = selectors.DefaultSelector()
        self.handles = {}

    def tmux_call(self, *args, check=True):
        env = dict(os.environ)
        env.pop("TMUX", None)
        return subprocess.run([*self.tmux, *args], env=env, text=True,
                              capture_output=True, check=check)

    @contextmanager
    def producer(self):
        lock = take_lock(self.lock_dir.parent / "finite_resumer.producer.lock")
        if lock is None:
            raise Blocked("Another finite producer owns this queue")
        try:
            previous = old_producers(self.queue)
            if previous:
                raise Blocked(f"Old producer still active: PIDs {previous}; stop producers only, not GPU tmux")
            self.run_dir.mkdir(parents=True, exist_ok=False)
            yield
        finally:
            for handle in list(self.handles.values()):
                self.close(handle)
            self.selector.close()
            lock.close()

    def panes(self):
        result = self.tmux_call("list-panes", "-a", "-F",
                                "#{session_name}\t#{pane_pid}\t#{pane_dead}\t#{pane_start_command}", check=False)
        if result.returncode:
            if "no server running" in result.stderr or "error connecting" in result.stderr:
                return {}
            raise Blocked(result.stderr.strip())
        panes = {}
        for line in result.stdout.splitlines():
            name, pid, dead, command = line.split("\t", 3)
            if name in panes:
                panes[name]["multiple"] = True
            else:
                panes[name] = {"pid": int(pid), "dead": dead == "1", "command": command}
        return panes

    def observe(self, cell, phase, pane, gpus, spec=None):
        if pane.get("multiple") or not set(gpus).issubset(APPROVED):
            raise Blocked(f"Unsupported canonical pane/GPU identity: {cell.session(phase)}")
        if pane["dead"]:
            return None
        try:
            fd = open_pidfd(pane["pid"])
        except ProcessLookupError:
            return None
        # Old canonical jobs do not hold cell locks; pin their cell while adopted.
        cell_lock = (take_lock(self.lock_dir.parent / "cell_locks" / (cell.model.session("train") + ".lock"))
                     if spec is None else None)
        handle = {"cell": cell, "phase": phase, "gpus": tuple(gpus), "fd": fd,
                  "pid": pane["pid"], "spec": spec, "adoption_lock": cell_lock}
        self.handles[fd] = handle
        self.selector.register(fd, selectors.EVENT_READ, handle)
        return handle

    def discover(self, cells, active):
        panes = self.panes()
        found = []
        # Eval-only Students must also adopt the existing canonical model jobs.
        for cell in dict.fromkeys([*cells, *(c.model for c in cells)]):
            for phase in (("eval",) if cell.role == "student" else ("train", "eval")):
                name = cell.session(phase)
                if name not in panes or name in active or panes[name]["dead"]:
                    continue
                actual, actual_phase, gpus, spec = pane_job(name, panes[name], self.queue)
                if actual != cell or actual_phase != phase:
                    raise Blocked(f"Canonical task identity differs from requested cell: {name}")
                handle = self.observe(cell, phase, panes[name], gpus, spec)
                if handle:
                    found.append(handle)
        return found

    def discover_external_owners(self, owners, active):
        if not owners:
            return []
        panes = self.panes()
        found = []
        for name, gpus in owners.items():
            key = "external-owner:" + name
            if key in active or name not in panes or panes[name]["dead"]:
                continue
            pane = panes[name]
            if pane.get("multiple"):
                raise Blocked(f"External owner must have one identifiable pane: {name}")
            try:
                fd = open_pidfd(pane["pid"])
            except ProcessLookupError:
                continue
            handle = {"cell": None, "phase": "external_owner", "owner_key": key,
                      "gpus": tuple(gpus), "fd": fd, "pid": pane["pid"],
                      "spec": None, "adoption_lock": None}
            self.handles[fd] = handle
            self.selector.register(fd, selectors.EVENT_READ, handle)
            found.append(handle)
        return found

    def available(self):
        free = idle_devices(self.gpus)
        available = set()
        for gpu in free:
            lock = take_lock(self.lock_dir / f"gpu{gpu}.lock")
            if lock is not None:
                available.add(gpu)
                lock.close()
        return available

    def evidence(self, cell, phase, check_wrap=True):
        return (train_state if phase == "train" else eval_state)(
            self.queue, self.paths, cell, check_wrap=check_wrap)

    def admission(self, cell, phase):
        return input_state(self.queue, self.paths, cell, phase)

    def archive_retry(self, cell, phase):
        """Free only the selected failed native destination, retaining every byte."""
        args = (cell.dataset, cell.domain, str(cell.seed), cell.method)
        if self.evidence(cell, phase) == "complete":
            raise Blocked(f"Cannot retry completed {cell.key}:{phase}")
        if phase == "train":
            if cell.method == "AASFOD":
                raise Blocked("AASFOD staged training and its retained TSD split require owner-specific "
                              "recovery; no automatic artifact relocation")
            if any(Path(resolve(*args)).exists()
                   for resolve in (self.paths.ema_path, self.paths.student_path)):
                raise Blocked(f"Conflicting final checkpoints require owner resolution: {cell.key}")
            destinations = [Path(self.paths.eval_full_dir(*args))]
            # Mixed resolvers expose this function even for EMA-only bindings.
            if hasattr(self.paths, "eval_student_dir") and (
                    not hasattr(self.paths, "DATA")
                    or cell.model.key in self.paths.DATA["student_cells"]):
                destinations.append(Path(self.paths.eval_student_dir(*args)))
            if any(p.exists() for p in destinations):
                raise Blocked(f"Training retry would affect retained evaluations: {cell.key}")
            output = Path(self.paths.method_dir(*args))
        else:
            resolve = self.paths.eval_student_dir if cell.role == "student" else self.paths.eval_full_dir
            output = Path(resolve(*args))
        origin = getattr(self.paths, "DATA", {}).get("source_queues", {}).get(cell.key, self.queue)
        candidates = [output, *(Path(host.map_path(q)) / f"wrap_{cell.session(phase)}.status"
                                for q in dict.fromkeys((str(self.queue), str(origin))))]
        moves = []
        for source in candidates:
            if source.exists():
                # Adjacent rename remains on the artifact filesystem, even when
                # run metadata is elsewhere. Never overwrite an earlier archive.
                destination = source.with_name(source.name + ".finite-retry-" + self.run_dir.name)
                if destination.exists():
                    raise Blocked(f"Retry archive already exists: {destination}")
                moves.append({"source": str(source), "archive": str(destination)})
        record = self.run_dir / "recovery" / (cell.session(phase) + ".json")
        write_json(record, {"cell": asdict(cell), "phase": phase, "moves": moves, "status": "planned"})
        for move in moves:
            Path(move["source"]).rename(move["archive"])
        write_json(record, {"cell": asdict(cell), "phase": phase, "moves": moves, "status": "archived"})
        return moves

    def start(self, cell, phase, gpus):
        name = cell.session(phase)
        job_file = self.run_dir / "jobs" / (name + ".json")
        receipt = self.run_dir / "results" / (name + ".json")
        write_json(job_file, {
            "cell": asdict(cell), "phase": phase, "gpus": list(gpus),
            "queue": str(self.queue), "receipt": str(receipt), "tmux": list(self.tmux),
        })
        logs = self.run_dir / "logs"
        logs.mkdir(exist_ok=True)
        command = shlex.join(["env", f"PYTHONPATH={SCRIPT.parents[2]}",
                             sys.executable, "-u", str(SCRIPT), "worker", str(job_file)])
        command += " >" + shlex.quote(str(logs / (name + ".log"))) + " 2>&1"
        result = self.tmux_call("new-session", "-d", "-s", name, command, check=False)
        pane = self.panes().get(name)
        if result.returncode and pane is None:
            raise Blocked(f"Could not create canonical task {name}: {result.stderr.strip()}")
        if pane is not None:
            actual, actual_phase, actual_gpus, actual_spec = pane_job(name, pane, self.queue)
            if actual != cell or actual_phase != phase:
                raise Blocked(f"Canonical task collision: {name}")
            return self.observe(cell, phase, pane, actual_gpus, actual_spec)
        # A fast task may finish before pane discovery; its receipt must still prove success.
        return None

    def finish(self, cell, phase, spec=None):
        if spec is None:
            candidate = self.run_dir / "jobs" / (cell.session(phase) + ".json")
            spec = candidate if candidate.is_file() else None
        if spec is not None:
            job = json.loads(Path(spec).read_text())
            receipt = Path(job["receipt"])
            if not receipt.is_file():
                return "failed", "worker exited without a terminal receipt"
            result = json.loads(receipt.read_text())
            if Cell(**result["cell"]) != cell or result["phase"] != phase:
                return "failed", "terminal receipt identity mismatch"
            if result["status"] != "complete":
                return result["status"], result.get("reason", "worker failed")
        state = self.evidence(cell, phase)
        return ("complete", "") if state == "complete" else (
            "failed", f"terminal process exited without valid final {phase} evidence")

    def wait(self):
        return [event.data for event, _ in self.selector.select()]

    def close(self, handle):
        self.selector.unregister(handle["fd"])
        os.close(handle["fd"])
        self.handles.pop(handle["fd"], None)
        if handle["adoption_lock"]:
            handle["adoption_lock"].close()


def choose_training(ready, free):
    singles = sorted((c for c in ready if c.width == 1), key=lambda c: (c.method != "C", c.key))
    doubles = sorted((c for c in ready if c.width == 2), key=lambda c: c.key)
    pair = next((p for p in PAIR_PORTS if set(p).issubset(free)), None)
    # A long C plus a supported pair packs 1+2; the next single fills the fourth card.
    if singles and len(free) >= 3:
        if pair and doubles:
            single_gpu = next(g for g in sorted(free) if g not in pair)
            return singles[0], (single_gpu,)
        return singles[0], (min(free),)
    if doubles and pair:
        return doubles[0], pair
    if singles and free:
        return singles[0], (min(free),)
    return None


def run_finite(cells, backend, external=(), external_owners=None, *,
               previous_state=None, hold_cells=(), release_cells=(), retry_cells=()):
    external = set(external)
    external_owners = external_owners or {}
    state = {c.key: {"cell": asdict(c), "train_requested": requested,
                     "training_ownership": "external" if c in external else "producer" if requested else "eval_only",
                     "train": "pending", "eval": "pending", "reasons": [], "adopted": [],
                     "held": False, "attempts": []}
             for c, requested in cells.items()}
    controls = set(hold_cells) | set(release_cells)
    retries = set()
    for value in retry_cells:
        key, phase = value.rsplit(":", 1)
        if phase not in ("train", "eval") or key not in state:
            raise Blocked(f"Unknown retry cell/phase: {value}")
        retries.add((key, phase))
    if controls - state.keys() or set(hold_cells) & set(release_cells):
        raise Blocked("Hold/release requires distinct exact keys in the finite scope")
    if retries and not previous_state:
        raise Blocked("Selected retries require --previous-state; never reset a ledger")
    if previous_state:
        previous = json.loads(host.read_path(previous_state).read_text())
        if "queue" in previous and not host.same_path(previous["queue"], str(backend.queue)):
            raise Blocked("Previous state belongs to a different queue")
        rows = {Cell(**r["cell"]).key: r for r in previous["cells"]}
        if rows.keys() != state.keys():
            raise Blocked("Previous state must have exactly the same finite scope")
        for key, row in rows.items():
            if any(row[field] != state[key][field]
                   for field in ("train_requested", "training_ownership")):
                raise Blocked(f"Previous ownership differs for {key}")
            state[key].update(deepcopy(row))
            state[key].setdefault("attempts", [])
        for key, phase in retries:
            row = state[key]
            if row[phase] not in ("failed", "blocked"):
                raise Blocked(f"Retry requires a failed/blocked phase: {key}:{phase}")
            if phase == "train" and (not row["train_requested"] or row["cell"].get("role", "ema") != "ema"):
                raise Blocked(f"Training retry is not producer-owned: {key}")
            if phase == "eval" and row["train"] != "complete":
                raise Blocked(f"Evaluation retry requires completed training: {key}")
    for key in hold_cells:
        state[key]["held"] = True
    for key in release_cells:
        state[key]["held"] = False
    active = {}
    attempted = set()
    recovered = False

    def snapshot(status="running", reason=""):
        data = {"scope": "finite_input_only", "status": status, "reason": reason,
                "producer_pid": os.getpid(),
                "queue": str(backend.queue),
                "previous_state": str(previous_state) if previous_state else None,
                "cells": list(state.values()),
                "external_reservations": {name: list(gpus) for name, gpus in external_owners.items()},
                "active": [{"cell": asdict(h["cell"]) if h["cell"] is not None else None, "phase": h["phase"],
                            "gpus": h["gpus"], "pid": h["pid"],
                            "external_owner": h.get("owner_key")} for h in active.values()]}
        write_json(backend.run_dir / "state.json", data)
        return data

    def terminal(cell, phase, spec=None):
        status, reason = backend.finish(cell, phase, spec)
        if cell.key not in state:
            return  # Adopted model job; its Student dependency is checked after exit.
        state[cell.key][phase] = status
        state[cell.key]["attempts"].append({
            "phase": phase, "status": status, "reason": reason, "run_dir": str(backend.run_dir),
            "spec": str(spec) if spec else None})
        if reason:
            state[cell.key]["reasons"].append(reason)
        if phase == "train" and status != "complete":
            state[cell.key]["eval"] = "blocked"

    with backend.producer():
        try:
            while True:
                for handle in backend.discover_external_owners(external_owners, active):
                    active[handle["owner_key"]] = handle
                for handle in backend.discover(cells, active):
                    cell, phase = handle["cell"], handle["phase"]
                    if any(h["cell"] is not None and h["cell"].model == cell.model
                           for h in active.values()):
                        raise Blocked(f"Concurrent canonical jobs share model {cell.model.key}")
                    active[cell.session(phase)] = handle
                    if cell.key not in state:
                        continue
                    state[cell.key][phase] = "running"
                    state[cell.key]["adopted"].append({
                        "phase": phase, "pid": handle["pid"], "gpus": list(handle["gpus"])})
                    if phase == "eval":
                        state[cell.key]["train"] = backend.evidence(cell, "train")
                occupied = {h["cell"].model for h in active.values() if h["cell"] is not None}
                if not recovered:
                    if previous_state:
                        # Preserve the original ledger, including failures made by
                        # old code before per-attempt history was available.
                        write_json(backend.run_dir / "previous_state.json", previous)
                    for key, phase in sorted(retries):
                        cell = Cell(**state[key]["cell"])
                        if cell.model in occupied:
                            raise Blocked(f"Retry requires no live canonical model job: {key}")
                        lock = take_lock(backend.lock_dir.parent / "cell_locks"
                                         / (cell.model.session("train") + ".lock"))
                        if lock is None:
                            raise Blocked(f"Retry cell lock busy: {key}")
                        try:
                            archives = backend.archive_retry(cell, phase)
                        finally:
                            lock.close()
                        row = state[key]
                        row["attempts"].append({"phase": phase, "status": "retry_authorized",
                                                "previous_status": row[phase], "archives": archives})
                        row[phase] = "pending"
                        if phase == "train":
                            row["eval"] = "pending"
                    for cell in cells:
                        for phase in ("train", "eval"):
                            row = state[cell.key]
                            if row[phase] == "running" and cell.session(phase) not in active:
                                row[phase] = backend.evidence(cell, phase)
                                if row[phase] != "complete":
                                    row[phase] = "blocked"
                                    row["reasons"].append("previous running task has no live canonical job")
                    recovered = True
                for cell, requested in sorted(cells.items(), key=lambda item: item[0].role != "ema"):
                    row = state[cell.key]
                    if cell.model in occupied:
                        if (row["train"] != "complete"
                                and cell.session("eval") not in active and cell.session("train") not in active):
                            row["train"] = row["eval"] = "waiting"
                        continue
                    model_row = state.get(cell.model.key)
                    readonly_blocked = (
                        cell.role == "student" and not requested
                        and row["training_ownership"] == "eval_only"
                        and row["train"] == row["eval"] == "blocked"
                        and not cells.get(cell.model) and cell.model not in external
                        and not row["held"] and not (model_row and model_row["held"])
                        and not row["attempts"]
                        and not any(a["phase"] == "eval" for a in row["adopted"]))
                    if row["train"] in ("pending", "external", "waiting") or readonly_blocked:
                        evidence = backend.evidence(cell, "train")
                        if readonly_blocked:
                            if evidence != "complete":
                                continue
                            row["eval"] = "pending"
                        waiting_model = (cell.role == "student" and model_row is not None
                                         and (cells.get(cell.model) or cell.model in external)
                                         and model_row["train"] not in ("complete", "failed", "blocked"))
                        if evidence != "complete" and waiting_model:
                            row["train"] = row["eval"] = "waiting"
                            continue
                        if evidence == "pending" and cell in external:
                            row["train"] = "external"
                        else:
                            row["train"] = "ready" if evidence == "pending" and requested else evidence
                        if row["train"] == "pending":
                            row["train"] = "blocked"
                            row["reasons"].append("eval dependency is not trained; no training authorized")
                        if row["train"] == "blocked":
                            row["eval"] = "blocked"
                            row["reasons"].append("missing/conflicting final training evidence")
                    if row["train"] == "ready":
                        row["train"], row["train_input_reason"] = backend.admission(cell, "train")
                        row["eval"] = "waiting" if row["train"] == "waiting" else "pending"
                    if row["train"] == "complete" and row["eval"] in ("pending", "waiting"):
                        evidence = backend.evidence(cell, "eval")
                        row["eval"] = "ready" if evidence == "pending" else evidence
                        if evidence == "blocked":
                            row["reasons"].append("existing incomplete/conflicting native evaluation; preserve it")
                    if row["eval"] == "ready":
                        row["eval"], row["eval_input_reason"] = backend.admission(cell, "eval")
                while True:
                    occupied = {h["cell"].model for h in active.values() if h["cell"] is not None}
                    free = backend.available() - {g for h in active.values() for g in h["gpus"]}
                    if any(state[c.key]["train"] not in ("complete", "failed", "blocked") for c in external):
                        free -= {g for gpus in external_owners.values() for g in gpus}
                    trains = [c for c in cells if c.model not in occupied and state[c.key]["train"] == "ready"
                              and not state[c.key]["held"]
                              and cells[c] and c.role == "ema"
                              and (c, "train") not in attempted]
                    picked = choose_training(trains, free)
                    phase = "train"
                    if picked is None and free:
                        evaluations = [c for c in cells if c.model not in occupied
                                       and not state[c.key]["held"]
                                       and state[c.key]["train"] == "complete"
                                       and state[c.key]["eval"] == "ready" and (c, "eval") not in attempted]
                        if evaluations:
                            phase, picked = "eval", (evaluations[0], (min(free),))
                    if picked is None:
                        break
                    cell, gpus = picked
                    attempted.add((cell, phase))
                    state[cell.key][phase] = "running"
                    handle = backend.start(cell, phase, gpus)
                    if handle:
                        active[cell.session(phase)] = handle
                    else:
                        terminal(cell, phase)
                    snapshot()
                if not active:
                    statuses = [r[p] for r in state.values() for p in ("train", "eval")]
                    if all(s == "complete" for s in statuses):
                        return snapshot("complete")
                    if "failed" in statuses:
                        return snapshot("failed", "one or more finite tasks failed; no automatic retries")
                    if "external" in statuses:
                        return snapshot("blocked", "external training incomplete; no live canonical task or owner to await")
                    if "waiting" in statuses:
                        return snapshot("blocked", "required inputs waiting; ready independent work exhausted")
                    if any(r["held"] for r in state.values()):
                        return snapshot("blocked", "selected cells held; ready independent work exhausted")
                    return snapshot("blocked", "no allocatable approved GPU group and no tracked live task")
                snapshot()
                for handle in backend.wait():
                    cell, phase = handle["cell"], handle["phase"]
                    if phase == "external_owner":
                        active.pop(handle["owner_key"])
                    else:
                        terminal(cell, phase, handle["spec"])
                        active.pop(cell.session(phase))
                    backend.close(handle)
        except (Blocked, KeyboardInterrupt, OSError, ValueError, KeyError, TypeError,
                subprocess.SubprocessError) as error:
            return snapshot("blocked", str(error) or "producer interrupted; canonical jobs preserved")


def worker(job_file):
    job = host.read_json(job_file)
    cell, phase = Cell(**job["cell"]), job["phase"]
    queue, gpus = Path(job["queue"]), tuple(job["gpus"])
    root = lock_root(queue)
    locks = []
    status, reason, rc = "failed", "worker did not reach a terminal outcome", 1
    owns_cell = False
    try:
        if phase == "train" and cell.role != "ema":
            raise Blocked("Student quantitative extensions are evaluation-only")
        lock = take_lock(root.parent / "cell_locks" / (cell.model.session("train") + ".lock"))
        if lock is None:
            raise Blocked("cell lock busy; no runner invoked")
        locks.append(lock)
        owns_cell = True
        paths = load_paths(queue)
        state = (train_state if phase == "train" else eval_state)(queue, paths, cell)
        if state == "complete":
            status, reason, rc = "complete", "", 0
        elif state == "blocked":
            raise Blocked("existing contradictory/partial final evidence; no runner invoked")
        else:
            if not set(gpus).issubset(allowed_gpus(paths)) or len(gpus) != (cell.width if phase == "train" else 1):
                raise Blocked("invalid GPU assignment")
            admission, detail = input_state(queue, paths, cell, phase)
            if admission != "ready":
                raise Blocked(detail)
            for gpu in sorted(gpus):
                lock = take_lock(root / f"gpu{gpu}.lock")
                if lock is None:
                    raise Blocked("GPU lock busy; no runner invoked")
                locks.append(lock)
            if not set(gpus).issubset(idle_devices(gpus)):
                raise Blocked("GPU is occupied by another process; no runner invoked")
            env = {**os.environ, "IRAOD_GPU_LOCKED": "1",
                   "PYTHONPATH": str(SCRIPT.parents[2])}
            rc = subprocess.run(runner_command(queue, cell, phase, gpus), env=env).returncode
            valid = (train_state if phase == "train" else eval_state)(
                queue, paths, cell, check_wrap=False)
            if rc == 0 and valid == "complete":
                status, reason = "complete", ""
            else:
                reason = f"runner exit={rc}, final evidence={valid}"
                rc = rc or 1
    except Blocked as error:
        status, reason, rc = "blocked", str(error), 75
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        status, reason, rc = "failed", f"{type(error).__name__}: {error}", 1
    finally:
        write_json(job["receipt"], {
            "cell": asdict(cell), "phase": phase, "gpus": list(gpus),
            "status": status, "exit_code": rc, "reason": reason,
        })
        if owns_cell and not host.is_target_host():
            with (queue / f"wrap_{cell.session(phase)}.status").open("a") as stream:
                stream.write(f"wrap_exit={rc} {time.time()} cell={cell.key} phase={phase}\n")
        for lock in reversed(locks):
            lock.close()
        if owns_cell:
            subprocess.run([*job["tmux"], "wait-for", "-S", cell.session(phase) + "-done"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    return rc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("resume", "launch"):
        resume = commands.add_parser(name)
        resume.add_argument("--queue", required=True)
        resume.add_argument("--run-dir", required=True)
        resume.add_argument("--train-list", action="append", default=[])
        resume.add_argument("--eval-list", action="append", default=[])
        resume.add_argument("--external-train-list", action="append", default=[],
                            help="Finite externally owned cells: never submit their training, only observe/evaluate")
        resume.add_argument("--external-owner-session", action="append", default=[],
                            help="SESSION=GPU[,GPU]: reserve cards and observe this existing owner's exit")
        resume.add_argument("--gpus", default=",".join(map(str, APPROVED)))
        resume.add_argument("--previous-state",
                            help="Prior state.json; carry exact scope, ownership, failures and holds forward")
        resume.add_argument("--hold-cell", action="append", default=[],
                            help="Exact cell key: suppress new train/eval submissions; never stop jobs")
        resume.add_argument("--release-cell", action="append", default=[],
                            help="Exact cell key: release a persisted hold, not a failed attempt")
        resume.add_argument("--retry-cell", action="append", default=[],
                            help="KEY:train|eval: archive selected failure and permit one new attempt")
        resume.add_argument("--handoff-confirmed", action="store_true",
                            help="Competing primary producers stopped; declared external owners and GPU jobs preserved")
    run_worker = commands.add_parser("worker")
    run_worker.add_argument("job")
    args = parser.parse_args()
    if args.command == "worker":
        raise SystemExit(worker(args.job))
    if not args.handoff_confirmed:
        parser.error("Stop old producers only; preserve canonical GPU tmux, then use --handoff-confirmed")
    gpus = tuple(int(g) for g in args.gpus.split(","))
    if len(set(gpus)) != len(gpus) or not set(gpus).issubset(APPROVED) or not gpus:
        parser.error(f"Only host-approved GPUs {APPROVED} are allowed")
    try:
        probe = open_pidfd(os.getpid())
        os.close(probe)
    except (Blocked, OSError) as error:
        print(f"blocked: pidfd completion events unavailable: {error}; no polling fallback", file=sys.stderr)
        raise SystemExit(2)
    if args.command == "launch":
        # The producer pane also keeps the existing tmux server alive between jobs.
        previous = old_producers(args.queue)
        if previous:
            raise SystemExit(f"blocked: old producer PIDs {previous}; GPU job sessions must be preserved")
        load_work(args)
        log = Path(str(args.run_dir) + ".producer.log")
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("x"):
            pass
        command = shlex.join(["env", f"PYTHONPATH={SCRIPT.parents[2]}", sys.executable, "-u",
                             str(SCRIPT), "resume", *sys.argv[2:]])
        command += " >" + shlex.quote(str(log)) + " 2>&1"
        backend = TmuxBackend(args.queue, args.run_dir, gpus)
        result = backend.tmux_call("new-session", "-d", "-s", "xaf-finite-producer", command, check=False)
        backend.selector.close()
        if result.returncode:
            raise SystemExit(f"blocked: producer session not created: {result.stderr.strip()}")
        print(f"Producer launched; NOT completion. Read {args.run_dir}/state.json and {log}")
        return
    def stop_producer(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_producer)
    try:
        cells, external, owners = load_work(args)
        result = run_finite(cells, TmuxBackend(args.queue, args.run_dir, gpus), external, owners,
                            previous_state=args.previous_state, hold_cells=args.hold_cell,
                            release_cells=args.release_cell, retry_cells=args.retry_cell)
    except Blocked as error:
        print(f"blocked: {error}", file=sys.stderr)
        raise SystemExit(2)
    print(json.dumps({"status": result["status"], "reason": result["reason"]}))
    raise SystemExit(0 if result["status"] == "complete" else 1 if result["status"] == "failed" else 2)


if __name__ == "__main__":
    main()
