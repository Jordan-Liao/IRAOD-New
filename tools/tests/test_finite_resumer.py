"""Real tmux/flock/pidfd CPU tests; fixture runners never call a GPU."""

import csv
import json
import os
from pathlib import Path
import select
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from experiments.comparison import finite_resumer as finite
from experiments.comparison.finite_resumer import (
    Cell, EVAL_SHA, TmuxBackend, idle_devices, open_pidfd, pane_job, runner_command)


ROOT = Path(__file__).resolve().parents[2]

PATHS_SOURCE = '''
from pathlib import Path
ROOT = Path(__file__).parent / "artifacts"
EXPECT_PRED = {"RSAR": 2, "DIOR": 2}
def method_dir(ds, domain, seed, method):
    return ROOT / ds / domain / str(seed) / method
def work_dir(ds, domain, seed, method):
    p = method_dir(ds, domain, seed, method)
    return p / "ddp2/work" if method in ("E", "F") else p / "work"
def ema_path(ds, domain, seed, method):
    return str(work_dir(ds, domain, seed, method) / ("iter_266_ema.pth" if ds=="RSAR" else "iter_185_ema.pth"))
def student_path(ds, domain, seed, method):
    return str(work_dir(ds, domain, seed, method) / ("iter_266.pth" if ds=="RSAR" else "iter_185.pth"))
def eval_full_dir(ds, domain, seed, method):
    return str(method_dir(ds, domain, seed, method) / ("eval_full_"+domain+"_ids_v1"))
def eval_student_dir(ds, domain, seed, method):
    return str(method_dir(ds, domain, seed, method) / ("eval_full_"+domain+"_student_ids_v1"))
'''

RUNNER_SOURCE = '''
import fcntl,json,os,pathlib,pickle,sys
sys.path.insert(0,str(pathlib.Path(__file__).parent))
import paths
q=pathlib.Path(__file__).parent
kind,*args=sys.argv[1:]
role="student" if kind=="eval" and len(args)==6 and args[-1]=="student" else "ema"
if role=="student": args.pop()
if kind=="train2":
    assert len(args)==6,args
    gpu,port,ds,domain,seed,method=args
    assert (gpu,port) in (("4,5","29804"),("6,7","29806")),(gpu,port)
else:
    assert len(args)==5,args
    gpu,ds,domain,seed,method=args
gpus=[int(x) for x in gpu.split(",")]
phase="eval" if kind=="eval" else "train"
name=("xafE" if phase=="eval" else "xaf")+"-"+ds+"-"+domain+"-"+seed+"-"+method
if role=="student": name="xafS-"+ds+"-"+domain+"-"+seed+"-"+method
held=[]
for g in gpus:
    f=(q/"gpu_locks"/("gpu"+str(g)+".lock")).open("a+")
    if os.environ.get("IRAOD_GPU_LOCKED")=="1":
        try:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            f.close()
        else:
            raise AssertionError("worker failed to hold GPU lock")
    else:
        fcntl.flock(f,fcntl.LOCK_EX)
        held.append(f)
release=q/"release"/name
if not release.exists(): os.mkfifo(release)
def event(kind):
    with (q/"events").open("w") as f:
        f.write(json.dumps({"event":kind,"name":name,"phase":phase,"gpus":gpus})+"\\n")
event("start")
with release.open() as f: rc=int(f.read().strip())
if phase=="train":
    w=paths.work_dir(ds,domain,seed,method); w.mkdir(parents=True,exist_ok=True)
    (w.parent/"terminal_status").write_text("tmux_wrap_exit="+str(rc)+"\\n")
    if rc==0:
        pathlib.Path(paths.ema_path(ds,domain,seed,method)).write_bytes(b"CPU fixture final EMA")
        pathlib.Path(paths.student_path(ds,domain,seed,method)).write_bytes(b"CPU fixture final student")
        if (q/"runtime.json").exists():
            (paths.method_dir(ds,domain,seed,method)/"execution.json").write_text(json.dumps({
                "training_code":"/cpu-fixture/training","training_code_sha":"cpu-fixture-executed"}))
else:
    resolve=paths.eval_student_dir if role=="student" else paths.eval_full_dir
    out=pathlib.Path(resolve(ds,domain,seed,method)); out.mkdir(parents=True,exist_ok=True)
    (out/"eval_status").write_text("eval_exit="+str(rc)+" name="+method+" domain="+domain+" seed="+seed+" role="+role+"\\n")
    if rc==0:
        with (out/"predictions.pkl").open("wb") as f: pickle.dump([[],[]],f)
        (out/"pred_count.txt").write_text("2 expect=2\\n")
        (out/"class_ap.txt").write_text("| class | ap |\\n")
        (out/"eval_fixture.json").write_text(json.dumps({"config":"/fixture.py","metric":{"mAP":0.5}}))
        (out/"predictions.pkl.image_ids.json").write_text(json.dumps({
            "schema":"iraod-prediction-image-order-v1","origin":"inference_batch_img_metas",
            "status":"complete","checkpoint":(paths.student_path if role=="student" else paths.ema_path)(ds,domain,seed,method),
            "config":"/fixture.py",
            "evaluation_code_sha":EVAL_SHA,"predictions_file":"predictions.pkl",
            "n_images":2,"dataset_size":2,"image_ids":["image_z","image_a"],
            "records":[{"prediction_index":i,"image_id":v,"ori_filename":v+".png"}
                       for i,v in enumerate(["image_z","image_a"])]}))
event("finish")
sys.exit(rc)
'''


class FiniteResumerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.q = self.root / "q"
        self.q.mkdir()
        for directory in ("bin", "release", "gpu_locks"):
            (self.q / directory).mkdir()
        (self.q / "paths.py").write_text(PATHS_SOURCE)
        (self.q / "runner.py").write_text(f"EVAL_SHA={EVAL_SHA!r}\n" + RUNNER_SOURCE)
        for filename, kind in (("run_train_1gpu.sh", "train1"),
                               ("run_train_2gpu.sh", "train2"), ("run_eval_full.sh", "eval")):
            (self.q / filename).write_text(
                "#!/bin/sh\nexec " + shlex.join([sys.executable, str(self.q / "runner.py"), kind]) + ' "$@"\n')
        (self.q / "with_gpu_lock.sh").write_text(f"LOCKDIR={self.q / 'gpu_locks'}\n")
        self.gpu_state = self.q / "gpu_state.json"
        self.gpu_state.write_text("[]")
        smi = self.q / "bin/nvidia-smi"
        smi.write_text(f"""#!{sys.executable}
import json,sys
state=json.load(open({str(self.gpu_state)!r}))
busy=state.get("busy",[]) if isinstance(state,dict) else state
memory=state.get("memory",{{}}) if isinstance(state,dict) else {{}}
if any(a.startswith('--query-gpu=') for a in sys.argv):
    print('\\n'.join(str(g)+', GPU-fixture-'+str(g)+', '+str(memory.get(str(g),1000 if g in busy else 0)) for g in (4,5,6,7)))
elif any(a.startswith('--query-compute-apps=') for a in sys.argv):
    print('\\n'.join('GPU-fixture-'+str(g)+', 999' for g in busy))
else: raise SystemExit(2)
""")
        smi.chmod(0o755)
        os.mkfifo(self.q / "events")
        self.events_fd = os.open(self.q / "events", os.O_RDWR | os.O_NONBLOCK)
        self.buffer = b""
        self.events = []
        self.live = {}
        self.released = set()
        self.processes = []
        self.env = {**os.environ, "PATH": str(self.q / "bin") + os.pathsep + os.environ["PATH"],
                    "TMUX_TMPDIR": str(self.root), "PYTHONPATH": str(ROOT),
                    "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"}
        self.env.pop("TMUX", None)
        self.env.pop("IRAOD_GPU_LOCKED", None)
        self.addCleanup(self.cleanup)
        # Production uses the launch entry's producer pane. Keep that server
        # lifetime in this fixture while the controllable producer is a Popen.
        subprocess.run(["tmux", "new-session", "-d", "-s", "fixture-controller", "cat"],
                       env=self.env, check=True, capture_output=True)

    def cleanup(self):
        for process in self.processes:
            if process.poll() is None:
                process.terminate()  # Specific owned producer PID, not its independent jobs.
                process.wait(timeout=10)
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
        server = subprocess.run(["tmux", "display-message", "-p", "#{pid}"],
                                env=self.env, capture_output=True, text=True)
        if server.returncode == 0 and server.stdout.strip().isdigit():
            try:
                os.kill(int(server.stdout.strip()), signal.SIGTERM)
            except ProcessLookupError:
                pass
        os.close(self.events_fd)
        self.temp.cleanup()

    def start(self, cells, name="run", eval_cells=(), external=(), owners=(), controls=(), queue=None):
        train = self.q / (name + "-train.txt")
        evaluate = self.q / (name + "-eval.txt")
        train.write_text("".join(f"{c.dataset} {c.domain} {c.seed} {c.method}\n" for c in cells))
        evaluate.write_text("".join(
            f"{c.dataset} {c.domain} {c.seed} {c.method}"
            + (" student" if c.role == "student" else "") + "\n" for c in eval_cells))
        extra = []
        if external:
            reserved = self.q / (name + "-external.txt")
            reserved.write_text("".join(
                f"{c.dataset} {c.domain} {c.seed} {c.method}\n" for c in external))
            extra += ["--external-train-list", str(reserved)]
        for owner in owners:
            extra += ["--external-owner-session", owner]
        run_dir = self.root / name
        log = (self.root / (name + ".log")).open("w")
        self.addCleanup(log.close)
        process = subprocess.Popen([
            sys.executable, "-u", "-m", "experiments.comparison.finite_resumer", "resume",
            "--queue", str(queue or self.q), "--run-dir", str(run_dir),
            "--train-list", str(train), "--eval-list", str(evaluate), "--handoff-confirmed",
            *extra, *controls,
        ], env=self.env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        self.processes.append(process)
        return process, run_dir

    def event(self, timeout=8):
        end = time.monotonic() + timeout
        while b"\n" not in self.buffer:
            ready, _, _ = select.select([self.events_fd], [], [], max(0, end - time.monotonic()))
            if not ready:
                logs = "\n".join(p.read_text() for p in self.root.glob("*.log"))
                raise AssertionError(f"No CPU event; logs:\n{logs}")
            self.buffer += os.read(self.events_fd, 65536)
        line, self.buffer = self.buffer.split(b"\n", 1)
        event = json.loads(line)
        self.events.append(event)
        if event["event"] == "start":
            self.live[event["name"]] = event
        else:
            self.live.pop(event["name"], None)
        return event

    def next_start(self):
        while True:
            event = self.event()
            if event["event"] == "start":
                return event

    def release(self, name, rc=0):
        if name in self.released:
            return
        with (self.q / "release" / name).open("w") as stream:
            stream.write(str(rc))
        self.released.add(name)

    def finish(self, process, expected_starts):
        while len([e for e in self.events if e["event"] == "start"]) < expected_starts:
            for name in list(self.live):
                self.release(name)
            self.event()
        for name in list(self.live):
            self.release(name)
        rc = process.wait(timeout=15)
        self.assertEqual(rc, 0, "\n".join(p.read_text() for p in self.root.glob("*.log")))

    def test_mixed_1_1_2_and_immediate_training_refill_before_eval(self):
        cells = [Cell("RSAR", "clean", 43, "C"), Cell("DIOR", "clean", 43, "C"),
                 Cell("RSAR", "clean", 43, "E"), Cell("DIOR", "cloudy", 44, "D")]
        process, directory = self.start(cells)
        starts = [self.next_start() for _ in range(3)]
        self.assertEqual(sorted(len(e["gpus"]) for e in starts), [1, 1, 2])
        self.assertTrue(all(e["phase"] == "train" for e in starts))
        self.assertEqual({g for e in starts for g in e["gpus"]}, {4, 5, 6, 7})
        single = next(e for e in starts if len(e["gpus"]) == 1)
        self.release(single["name"])
        refill = self.next_start()
        self.assertEqual(refill["name"], cells[-1].session("train"))
        self.assertEqual(refill["gpus"], single["gpus"])
        self.assertEqual(len(self.live), 3)  # Unrelated C/EF jobs have not been released.
        self.finish(process, 8)
        state = json.loads((directory / "state.json").read_text())
        self.assertEqual(state["status"], "complete")
        self.assertEqual(len({e["name"] for e in self.events if e["event"] == "start"}), 8)

    def test_two_supported_pairs_run_concurrently_and_correct_api(self):
        cells = [Cell("RSAR", "clean", 43, "E"), Cell("DIOR", "clean", 43, "F")]
        process, _ = self.start(cells)
        starts = [self.next_start(), self.next_start()]
        self.assertEqual({tuple(e["gpus"]) for e in starts}, {(4, 5), (6, 7)})
        command = runner_command(self.q, cells[1], "train", (6, 7))
        self.assertEqual(command[2:], ["6,7", "29806", "DIOR", "clean", "43", "F"])
        self.finish(process, 4)

    def test_actual_six_argument_bridge_abi_and_canonical_adoption_identity(self):
        cell = Cell("RSAR", "clean", 43, "F")
        for pair, port in (((4, 5), "29804"), ((6, 7), "29806")):
            with self.subTest(pair=pair):
                command = runner_command(self.q, cell, "train", pair)
                self.assertEqual(command[2:], [
                    ",".join(map(str, pair)), port, "RSAR", "clean", "43", "F"])
                parsed = pane_job(cell.session("train"), {
                    "command": shlex.quote(shlex.join(command) + "; echo wrap_exit=$?"),
                }, self.q)
                self.assertEqual(parsed, (cell, "train", pair, None))
        with self.assertRaisesRegex(ValueError, "only supports 4,5 and 6,7"):
            runner_command(self.q, cell, "train", (4, 6))

    def test_replacement_producer_adopts_live_jobs_without_restarting_them(self):
        cells = [Cell("RSAR", "clean", 43, "C"), Cell("DIOR", "clean", 43, "B"),
                 Cell("RSAR", "clean", 43, "E")]
        first, _ = self.start(cells, "first")
        starts = [self.next_start() for _ in range(3)]
        first.terminate()
        self.assertEqual(first.wait(timeout=10), 2)
        replacement, directory = self.start(cells, "replacement")
        b = next(e for e in starts if e["name"] == cells[1].session("train"))
        self.release(b["name"])
        following = self.next_start()
        self.assertEqual(following["name"], cells[1].session("eval"))
        self.finish(replacement, 6)
        state = json.loads((directory / "state.json").read_text())
        self.assertTrue(any(r["adopted"] for r in state["cells"]))
        self.assertEqual(len([e for e in self.events if e["event"] == "start" and e["phase"] == "train"]), 3)

    def test_no_capacity_and_no_owned_task_returns_blocked_without_polling(self):
        self.gpu_state.write_text("[4,5,6,7]")
        process, directory = self.start([Cell("RSAR", "clean", 43, "C")])
        self.assertEqual(process.wait(timeout=8), 2)
        self.assertEqual(json.loads((directory / "state.json").read_text())["status"], "blocked")
        self.assertEqual(self.events, [])

    def test_external_gpu4_is_excluded_even_when_our_gpu_lock_is_empty(self):
        self.gpu_state.write_text(json.dumps({"busy": [4], "memory": {"4": 57662}}))
        (self.q / "gpu_locks/gpu4.lock").touch()
        cells = [Cell("RSAR", "clean", 43, "C"), Cell("RSAR", "chaff", 43, "E")]
        process, _ = self.start(cells)
        starts = [self.next_start(), self.next_start()]
        self.assertEqual({g for event in starts for g in event["gpus"]}, {5, 6, 7})
        self.assertEqual({tuple(event["gpus"]) for event in starts}, {(5,), (6, 7)})
        self.finish(process, 4)
        self.assertTrue(all(4 not in e["gpus"] for e in self.events if e["event"] == "start"))
        self.assertEqual(json.loads(self.gpu_state.read_text())["memory"]["4"], 57662)

    def test_worker_rechecks_external_occupancy_after_gpu_lock_acquisition(self):
        # Parent sees a free GPU; a foreign process appears before worker admission.
        with patch.dict(os.environ, self.env):
            backend = TmuxBackend(self.q, self.root / "probe")
            self.assertIn(4, backend.available())
            backend.selector.close()
        self.gpu_state.write_text(json.dumps({"busy": [4], "memory": {"4": 57662}}))
        cell = Cell("RSAR", "clean", 43, "C")
        receipt = self.root / "foreign-receipt.json"
        job = self.root / "foreign-job.json"
        job.write_text(json.dumps({
            "cell": cell.__dict__, "phase": "train", "gpus": [4], "queue": str(self.q),
            "receipt": str(receipt), "tmux": ["tmux"]}))
        process = subprocess.Popen([
            sys.executable, "-m", "experiments.comparison.finite_resumer", "worker", str(job)],
            cwd=ROOT, env=self.env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self.processes.append(process)
        self.assertEqual(process.wait(timeout=8), 75)
        result = json.loads(receipt.read_text())
        self.assertEqual(result["status"], "blocked")
        self.assertIn("occupied by another process", result["reason"])
        self.assertEqual(self.events, [])

    def test_compute_app_blocks_even_when_reported_memory_is_low(self):
        self.gpu_state.write_text(json.dumps({"busy": [4], "memory": {"4": 0}}))
        with patch.dict(os.environ, self.env):
            self.assertNotIn(4, idle_devices((4, 5, 6, 7)))

    def test_failure_does_not_launch_dependent_eval_or_become_success(self):
        cell = Cell("RSAR", "clean", 43, "C")
        process, directory = self.start([cell])
        self.assertEqual(self.next_start()["name"], cell.session("train"))
        self.release(cell.session("train"), rc=9)
        self.assertEqual(process.wait(timeout=10), 1)
        state = json.loads((directory / "state.json").read_text())
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["cells"][0]["eval"], "blocked")

    def prepared_inputs(self, cells):
        paths = finite.load_paths(self.q)
        bindings, origins = {}, {}
        for cell in cells:
            args = (cell.dataset, cell.domain, str(cell.seed), cell.method)
            inputs = self.q / "inputs" / cell.dataset
            inputs.mkdir(parents=True, exist_ok=True)
            source = inputs / "train/epoch_100.pth"
            source.parent.mkdir(exist_ok=True)
            source.write_bytes(b"frozen source")
            images = inputs / "val"
            images.mkdir(exist_ok=True)
            for index in range(2):
                (images / f"frozen-{index}.png").write_bytes(b"fixture image")
            annotations = inputs / "ImageSets/test.txt" if cell.dataset == "DIOR" else inputs / "annotations"
            annotations.parent.mkdir(exist_ok=True)
            if cell.dataset == "DIOR":
                annotations.write_text("frozen-0\nfrozen-1\n")
            else:
                annotations.mkdir(exist_ok=True)
            bindings[cell.key] = {
                **cell.__dict__, "source_checkpoint": str(source), "target_val": str(images),
                "unlabeled_epoch_size": 2, "checkpoint": paths.ema_path(*args),
                "student_checkpoint": paths.student_path(*args),
                "method_dir": str(paths.method_dir(*args)),
                "ann_file": str(annotations), "img_prefix": str(images),
            }
            origin = self.q / ("formal_ports_manifests_a39c832-" + cell.method)
            origin.mkdir(exist_ok=True)
            (origin / "runtime.json").write_text(json.dumps({"python": sys.executable}))
            origins[cell.key] = str(origin)
        runtime = {"schema": "iraod-mixed-detector-queue-v1", "cells": bindings,
                   "source_queues": origins,
                   "student_cells": {key: {**binding, "role": "student",
                                           "checkpoint": binding["student_checkpoint"]}
                                     for key, binding in bindings.items()}}
        (self.q / "runtime.json").write_text(json.dumps(runtime))
        with (self.q / "paths.py").open("a") as stream:
            stream.write("\nimport json\nDATA = json.loads((Path(__file__).parent / 'runtime.json').read_text())\n")
        return runtime

    def test_missing_shared_source_blocks_b_reg_and_both_f_deletions_and_f(self):
        cells = [Cell("RSAR", "clean", 43, method)
                 for method in ("B_REG", "F_text_only", "F_veto_only", "F")]
        runtime = self.prepared_inputs(cells)
        sources = {binding["source_checkpoint"] for binding in runtime["cells"].values()}
        self.assertEqual(len(sources), 1)
        source = Path(sources.pop())
        source.unlink()
        process, directory = self.start(cells)
        self.assertEqual(process.wait(timeout=8), 2)
        rows = json.loads((directory / "state.json").read_text())["cells"]
        self.assertEqual(len(rows), 4)
        for row in rows:
            with self.subTest(method=row["cell"]["method"]):
                self.assertEqual(row["train"], "waiting")
                self.assertIn(str(source), row["train_input_reason"])
                self.assertEqual(row["attempts"], [])
        self.assertFalse((directory / "jobs").exists())
        self.assertFalse((self.q / "artifacts").exists())

    def test_both_eval_roles_require_test_list_and_their_own_source_runtime(self):
        ema = Cell("DIOR", "clean", 43, "C")
        student = Cell("DIOR", "clean", 43, "C", "student")
        runtime = self.prepared_inputs([ema])
        student_origin = self.q / "student-source-queue"
        student_origin.mkdir()
        student_metadata = student_origin / "runtime.json"
        student_metadata.write_text(json.dumps({"python": sys.executable}))
        runtime["source_queues"][student.key] = str(student_origin)
        finite.write_json(self.q / "runtime.json", runtime)
        self.successful_train_files(ema)
        annotations = Path(runtime["cells"][ema.key]["ann_file"])
        ema_metadata = Path(runtime["source_queues"][ema.key]) / "runtime.json"
        cases = [
            ("test-list", annotations, (ema, student), ()),
            ("ema-runtime", ema_metadata, (ema,), ("--hold-cell", student.key)),
            ("student-runtime", student_metadata, (student,), ("--hold-cell", ema.key)),
        ]
        for name, missing, waiting, controls in cases:
            with self.subTest(missing=name):
                original = missing.read_bytes()
                missing.unlink()
                process, directory = self.start([], name, eval_cells=(ema, student), controls=controls)
                self.assertEqual(process.wait(timeout=8), 2)
                rows = {Cell(**r["cell"]): r
                        for r in json.loads((directory / "state.json").read_text())["cells"]}
                for cell in waiting:
                    self.assertEqual(rows[cell]["train"], "complete")
                    self.assertEqual(rows[cell]["eval"], "waiting")
                    self.assertIn(str(missing), rows[cell]["eval_input_reason"])
                    self.assertEqual(rows[cell]["attempts"], [])
                for cell in {ema, student} - set(waiting):
                    self.assertEqual(rows[cell]["eval"], "ready")
                self.assertFalse((directory / "jobs").exists())
                missing.write_bytes(original)

    def test_missing_rsar_source_waits_without_attempt_then_exit_event_admits_it(self):
        rsar = Cell("RSAR", "clean", 43, "B_REG")
        independent = Cell("DIOR", "clean", 43, "B")
        runtime = self.prepared_inputs([rsar, independent])
        source = Path(runtime["cells"][rsar.key]["source_checkpoint"])
        source.unlink()
        process, directory = self.start([rsar, independent])
        self.assertEqual(self.next_start()["name"], independent.session("train"))
        self.assertFalse((directory / "jobs" / (rsar.session("train") + ".json")).exists())
        source.write_bytes(b"frozen source")
        self.release(independent.session("train"))
        self.assertEqual(self.next_start()["name"], rsar.session("train"))
        self.finish(process, 4)
        state = json.loads((directory / "state.json").read_text())
        row = next(r for r in state["cells"] if r["cell"] == rsar.__dict__)
        self.assertEqual([r["status"] for r in row["attempts"]], ["complete", "complete"])

    def test_missing_dior_test_list_waits_without_eval_attempt_and_recovers_at_boundary(self):
        cell = Cell("DIOR", "clean", 43, "B_REG")
        runtime = self.prepared_inputs([cell])
        self.successful_train_files(cell)
        annotations = Path(runtime["cells"][cell.key]["ann_file"])
        annotations.unlink()
        process, directory = self.start([cell])
        self.assertEqual(process.wait(timeout=8), 2)
        state = json.loads((directory / "state.json").read_text())
        self.assertEqual(state["cells"][0]["eval"], "waiting")
        self.assertIn("ImageSets/test.txt", state["cells"][0]["eval_input_reason"])
        self.assertEqual(state["cells"][0]["attempts"], [])
        self.assertFalse((directory / "jobs").exists())
        annotations.write_text("frozen-0\nfrozen-1\n")
        next_process, _ = self.start([cell], "restored", controls=(
            "--previous-state", str(directory / "state.json")))
        self.assertEqual(self.next_start()["name"], cell.session("eval"))
        self.finish(next_process, 1)

    def test_missing_origin_runtime_and_insufficient_images_are_unattempted(self):
        cell = Cell("DIOR", "brightness", 43, "B_REG")
        runtime = self.prepared_inputs([cell])
        metadata = Path(runtime["source_queues"][cell.key]) / "runtime.json"
        original = metadata.read_bytes()
        metadata.unlink()
        process, directory = self.start([cell])
        self.assertEqual(process.wait(timeout=8), 2)
        row = json.loads((directory / "state.json").read_text())["cells"][0]
        self.assertEqual(row["train"], "waiting")
        self.assertIn(str(metadata), row["train_input_reason"])
        self.assertEqual(row["attempts"], [])
        metadata.write_bytes(original)
        (Path(runtime["cells"][cell.key]["target_val"]) / "frozen-1.png").unlink()
        process, directory = self.start([cell], "short-target")
        self.assertEqual(process.wait(timeout=8), 2)
        row = json.loads((directory / "state.json").read_text())["cells"][0]
        self.assertIn("Insufficient target images", row["train_input_reason"])
        self.assertEqual(row["attempts"], [])
        self.assertFalse((directory / "jobs").exists())

    def test_worker_rechecks_inputs_before_gpu_probe_or_runner(self):
        cell = Cell("RSAR", "clean", 43, "B_REG")
        runtime = self.prepared_inputs([cell])
        Path(runtime["cells"][cell.key]["source_checkpoint"]).unlink()
        job = self.root / "admission-job.json"
        receipt = self.root / "admission-receipt.json"
        finite.write_json(job, {"cell": cell.__dict__, "phase": "train", "gpus": [4],
                               "queue": str(self.q), "receipt": str(receipt), "tmux": ["tmux"]})
        with patch.object(finite, "idle_devices") as idle, patch.object(finite.subprocess, "run") as run:
            self.assertEqual(finite.worker(job), 75)
        idle.assert_not_called()
        self.assertEqual(run.call_count, 1)  # Completion notification, never a runner.
        self.assertIn("epoch_100.pth", json.loads(receipt.read_text())["reason"])
        self.assertFalse((self.q / "artifacts").exists())

    def test_selected_retry_and_hold_preserve_other_failures_completed_cells_and_history(self):
        selected = Cell("RSAR", "clean", 43, "B_REG")
        untouched = Cell("DIOR", "clean", 43, "B")
        completed = Cell("DIOR", "cloudy", 43, "D")
        held = Cell("DIOR", "contrast", 43, "C")
        cells = [selected, untouched, completed, held]
        process, directory = self.start(cells, controls=("--hold-cell", held.key))
        starts = [self.next_start() for _ in range(3)]
        self.assertNotIn(held.session("train"), {r["name"] for r in starts})
        for cell in (selected, untouched):
            self.release(cell.session("train"), rc=9)
        self.release(completed.session("train"))
        self.assertEqual(self.next_start()["name"], completed.session("eval"))
        self.release(completed.session("eval"))
        self.assertEqual(process.wait(timeout=8), 1)
        previous = (directory / "state.json").read_bytes()
        paths = finite.load_paths(self.q)
        selected_dir = Path(paths.method_dir("RSAR", "clean", "43", "B_REG"))
        failed_bytes = {p.relative_to(selected_dir): p.read_bytes()
                        for p in selected_dir.rglob("*") if p.is_file()}
        untouched_dir = Path(paths.method_dir("DIOR", "clean", "43", "B"))
        completed_dir = Path(paths.method_dir("DIOR", "cloudy", "43", "D"))
        preserved = {p: p.read_bytes() for root in (untouched_dir, completed_dir)
                     for p in root.rglob("*") if p.is_file()}
        self.released.remove(selected.session("train"))
        retry, retry_dir = self.start(cells, "selected-retry", controls=(
            "--previous-state", str(directory / "state.json"),
            "--retry-cell", selected.key + ":train"))
        self.assertEqual(self.next_start()["name"], selected.session("train"))
        self.release(selected.session("train"))
        self.assertEqual(self.next_start()["name"], selected.session("eval"))
        self.release(selected.session("eval"))
        self.assertEqual(retry.wait(timeout=8), 1)  # Unselected failure still counts.
        rows = {Cell(**r["cell"]).key: r for r in json.loads((retry_dir / "state.json").read_text())["cells"]}
        self.assertEqual((rows[selected.key]["train"], rows[selected.key]["eval"]), ("complete", "complete"))
        self.assertEqual(rows[untouched.key]["train"], "failed")
        self.assertTrue(rows[held.key]["held"])
        self.assertEqual([r["status"] for r in rows[selected.key]["attempts"]],
                         ["failed", "retry_authorized", "complete", "complete"])
        self.assertEqual((directory / "state.json").read_bytes(), previous)
        self.assertEqual(json.loads((retry_dir / "previous_state.json").read_text()), json.loads(previous))
        for path, content in preserved.items():
            self.assertEqual(path.read_bytes(), content)
        archive = selected_dir.with_name(selected_dir.name + ".finite-retry-selected-retry")
        for path, content in failed_bytes.items():
            self.assertEqual((archive / path).read_bytes(), content)
        self.assertIn("wrap_exit=9", (self.q / (
            f"wrap_{selected.session('train')}.status.finite-retry-selected-retry")).read_text())
        release, release_dir = self.start(cells, "release-held", controls=(
            "--previous-state", str(retry_dir / "state.json"), "--release-cell", held.key))
        self.assertEqual(self.next_start()["name"], held.session("train"))
        self.release(held.session("train"))
        self.assertEqual(self.next_start()["name"], held.session("eval"))
        self.release(held.session("eval"))
        self.assertEqual(release.wait(timeout=8), 1)
        self.assertEqual(len(list((release_dir / "jobs").glob("*.json"))), 2)

    def test_selected_eval_retry_archives_partial_predictions_without_retraining(self):
        cell = Cell("DIOR", "clean", 43, "B")
        runtime = self.prepared_inputs([cell])
        self.successful_train_files(cell)
        process, directory = self.start([cell])
        self.assertEqual(self.next_start()["name"], cell.session("eval"))
        self.release(cell.session("eval"), rc=9)
        self.assertEqual(process.wait(timeout=8), 1)
        out = self.q / "artifacts/DIOR/clean/43/B/eval_full_clean_ids_v1"
        (out / "predictions.pkl").write_bytes(b"retained incomplete prediction")
        origin_status = Path(runtime["source_queues"][cell.key]) / f"wrap_{cell.session('eval')}.status"
        origin_status.write_text("wrap_exit=9\n")
        original = {p: p.read_bytes() for p in out.parent.glob("work/*")}
        self.released.remove(cell.session("eval"))
        retry, retry_dir = self.start([cell], "eval-retry", controls=(
            "--previous-state", str(directory / "state.json"), "--retry-cell", cell.key + ":eval"))
        self.assertEqual(self.next_start()["name"], cell.session("eval"))
        self.release(cell.session("eval"))
        self.assertEqual(retry.wait(timeout=8), 0)
        self.assertEqual((out.with_name(out.name + ".finite-retry-eval-retry") / "predictions.pkl").read_bytes(),
                         b"retained incomplete prediction")
        for path, content in original.items():
            self.assertEqual(path.read_bytes(), content)
        self.assertEqual(len(list((retry_dir / "jobs").glob("*.json"))), 1)
        self.assertEqual(origin_status.with_name(origin_status.name + ".finite-retry-eval-retry").read_text(),
                         "wrap_exit=9\n")
        rejected, _ = self.start([cell], "reject-complete", controls=(
            "--previous-state", str(retry_dir / "state.json"), "--retry-cell", cell.key + ":eval"))
        self.assertEqual(rejected.wait(timeout=8), 2)
        self.assertIn("failed/blocked phase", (self.root / "reject-complete.log").read_text())

    def test_old_ledger_retry_retains_failure_and_rejects_a_live_model(self):
        cell = Cell("RSAR", "clean", 43, "B_REG")
        previous = {"scope": "finite_input_only", "status": "failed", "reason": "",
                    "producer_pid": 123, "active": [], "external_reservations": {},
                    "cells": [{"cell": cell.__dict__, "train_requested": True,
                               "training_ownership": "producer", "train": "failed",
                               "eval": "blocked", "reasons": ["missing epoch_100.pth"],
                               "adopted": []}]}
        old_state = self.root / "old-state.json"
        old_state.write_text(json.dumps(previous))
        command = shlex.join(["bash", str(self.q / "run_train_1gpu.sh"),
                             "4", "RSAR", "clean", "43", "B_REG"])
        subprocess.run(["tmux", "new-session", "-d", "-s", cell.session("train"), command],
                       env=self.env, check=True)
        self.assertEqual(self.next_start()["name"], cell.session("train"))
        process, directory = self.start([cell], "live-retry", controls=(
            "--previous-state", str(old_state), "--retry-cell", cell.key + ":train"))
        self.assertEqual(process.wait(timeout=8), 2)
        state = json.loads((directory / "state.json").read_text())
        self.assertIn("no live canonical model job", state["reason"])
        self.assertFalse((directory / "recovery").exists())
        self.assertEqual(json.loads(old_state.read_text()), previous)
        self.assertEqual(json.loads((directory / "previous_state.json").read_text()), previous)
        self.assertIn("missing epoch_100.pth", state["cells"][0]["reasons"])
        # The blocked retry leaves the actual old canonical job alive.
        self.assertEqual(subprocess.run(
            ["tmux", "has-session", "-t", cell.session("train")], env=self.env,
            capture_output=True).returncode, 0)
        self.release(cell.session("train"), rc=9)

    def test_boundary_retry_cannot_train_reference_students_or_erase_completed_training(self):
        completed = Cell("DIOR", "clean", 43, "B_REG")
        references = [Cell("DIOR", "clean", 43, method, "student")
                      for method in ("IRG", "LPLD", "SFUT")]
        work, terminal = self.successful_train_files(completed)
        preserved = {path: path.read_bytes() for path in (*work.iterdir(), terminal)}
        rows = [{"cell": cell.__dict__, "train_requested": cell == completed,
                 "training_ownership": "producer" if cell == completed else "eval_only",
                 "train": "complete" if cell == completed else "blocked",
                 "eval": "failed" if cell == completed else "blocked",
                 "reasons": ["original failure evidence"], "adopted": [],
                 "attempts": [{"phase": "eval", "status": "failed", "reason": "original attempt"}]}
                for cell in (completed, *references)]
        previous = self.root / "boundary-state.json"
        finite.write_json(previous, {"scope": "finite_input_only", "status": "failed", "cells": rows})
        old_bytes = previous.read_bytes()
        for index, cell in enumerate((*references, completed)):
            with self.subTest(cell=cell.key):
                name = f"reject-training-{index}"
                process, directory = self.start([completed], name, eval_cells=references, controls=(
                    "--previous-state", str(previous), "--retry-cell", cell.key + ":train"))
                self.assertEqual(process.wait(timeout=8), 2)
                reason = "failed/blocked phase" if cell == completed else "not producer-owned"
                self.assertIn(reason, (self.root / (name + ".log")).read_text())
                self.assertFalse(directory.exists())  # Rejected before producer/recovery/worker startup.
                self.assertEqual(previous.read_bytes(), old_bytes)
                for path, content in preserved.items():
                    self.assertEqual(path.read_bytes(), content)
        self.assertEqual(self.events, [])

    def generated_retry_queue(self, cell, student_cell=None):
        from experiments.comparison import mixed_queue

        models = list(dict.fromkeys([cell, *([student_cell.model] if student_cell else [])]))
        runtime = self.prepared_inputs(models)
        paths = finite.load_paths(self.q)
        for model in models:
            args = (model.dataset, model.domain, str(model.seed), model.method)
            runtime["cells"][model.key].update(
                method_dir=str(paths.method_dir(*args)), eval_dir=paths.eval_full_dir(*args))
        binding = runtime["cells"][cell.key]
        runtime.update(evaluation_code_sha=EVAL_SHA, python=sys.executable, student_cells={})
        student_row = ""
        if student_cell:
            args = (student_cell.dataset, student_cell.domain, str(student_cell.seed), student_cell.method)
            student_binding = runtime["cells"][student_cell.model.key]
            runtime["student_cells"][student_cell.model.key] = {
                **student_binding, "role": "student", "checkpoint": student_binding["student_checkpoint"],
                "eval_dir": paths.eval_student_dir(*args)}
            student_row = " ".join(args) + " student\n"
        finite.write_json(self.q / "runtime.json", runtime)
        for phase in ("train", "eval"):
            row = f"{cell.dataset} {cell.domain} {cell.seed} {cell.method}"
            (self.q / (phase + ".list")).write_text(
                row + "\n" + (student_row if phase == "eval" else ""))
        queue = mixed_queue.prepare([self.q], self.root / "generated-mixed-queue")
        # Retain the real generated resolver/runtime and worker launch/receipt
        # contract; replace only the GPU executables with this suite's CPU runner.
        for script in ("run_train_1gpu.sh", "run_train_2gpu.sh", "run_eval_full.sh"):
            (queue / script).write_bytes((self.q / script).read_bytes())
        return queue, binding

    def test_generated_worker_spec_retry_without_a_student_evaluation_binding(self):
        cell = Cell("DIOR", "brightness", 42, "B_REG")
        reference = Cell("DIOR", "brightness", 42, "IRG", "student")
        queue, binding = self.generated_retry_queue(cell, student_cell=reference)
        process, directory = self.start([cell], queue=queue, eval_cells=(reference,))
        self.assertEqual(self.next_start()["name"], cell.session("train"))
        job_file = directory / "jobs" / (cell.session("train") + ".json")
        job = json.loads(job_file.read_text())
        self.assertEqual(set(job), {"cell", "phase", "gpus", "queue", "receipt", "tmux"})
        self.assertEqual(job["queue"], str(queue))
        self.release(cell.session("train"), rc=9)
        self.assertEqual(self.event()["event"], "finish")
        self.assertEqual(process.wait(timeout=8), 1)
        old_state = (directory / "state.json").read_bytes()
        old_job = job_file.read_bytes()
        old_receipt = Path(job["receipt"]).read_bytes()
        self.assertEqual(json.loads(old_receipt)["status"], "failed")
        paths = finite.load_paths(job["queue"])
        self.assertEqual(set(paths.DATA["cells"]), {cell.key, reference.model.key})
        self.assertIn("student_checkpoint", paths.DATA["cells"][cell.key])
        self.assertEqual(set(paths.DATA["student_cells"]), {reference.model.key})
        self.assertEqual(set(paths.DATA["source_queues"]), {cell.key, reference.model.key, reference.key})
        with self.assertRaises(KeyError) as missing:
            paths.eval_student_dir(cell.dataset, cell.domain, str(cell.seed), cell.method)
        self.assertEqual(missing.exception.args, ("DIOR/brightness/42/B_REG",))

        self.released.remove(cell.session("train"))
        retry, retry_dir = self.start(
            [cell], "generated-retry", queue=queue, eval_cells=(reference,), controls=(
                "--previous-state", str(directory / "state.json"), "--retry-cell", cell.key + ":train"))
        fd = open_pidfd(retry.pid)
        try:
            ready, _, _ = select.select([self.events_fd, fd], [], [], 8)
            self.assertNotIn(fd, ready, (retry_dir / "state.json").read_text()
                             if (retry_dir / "state.json").exists() else "retry exited without state")
        finally:
            os.close(fd)
        self.assertEqual(self.next_start()["name"], cell.session("train"))
        self.release(cell.session("train"), rc=9)
        self.assertEqual(retry.wait(timeout=8), 1)  # Intentional second CPU runner failure, not archive failure.
        rows = {Cell(**r["cell"]): r for r in json.loads((retry_dir / "state.json").read_text())["cells"]}
        row = rows[cell]
        self.assertEqual([a["status"] for a in row["attempts"]], ["failed", "retry_authorized", "failed"])
        self.assertFalse(rows[reference]["train_requested"])
        self.assertEqual(rows[reference]["training_ownership"], "eval_only")
        self.assertEqual(rows[reference]["attempts"], [])
        for run in (directory, retry_dir):
            for phase in ("train", "eval"):
                self.assertFalse((run / "jobs" / (reference.session(phase) + ".json")).exists())
        self.assertEqual((directory / "state.json").read_bytes(), old_state)
        self.assertEqual(job_file.read_bytes(), old_job)
        self.assertEqual(Path(job["receipt"]).read_bytes(), old_receipt)
        archive = Path(binding["method_dir"] + ".finite-retry-generated-retry")
        self.assertEqual((archive / "terminal_status").read_text(), "tmux_wrap_exit=9\n")

    def test_generated_retry_preserves_student_outputs_and_requires_explicit_binding_metadata(self):
        cell = Cell("DIOR", "clean", 43, "IRG")
        queue, binding = self.generated_retry_queue(
            cell, student_cell=Cell("DIOR", "clean", 43, "IRG", "student"))
        backend = TmuxBackend(queue, self.root / "protected-retry")
        self.addCleanup(backend.selector.close)
        output = Path(backend.paths.eval_student_dir("DIOR", "clean", "43", "IRG"))
        output.mkdir(parents=True)
        retained = output / "predictions.pkl"
        retained.write_bytes(b"retained reference Student evaluation")
        with self.assertRaisesRegex(finite.Blocked, "retained evaluations"):
            backend.archive_retry(cell, "train")
        self.assertEqual(retained.read_bytes(), b"retained reference Student evaluation")
        runtime = json.loads((queue / "runtime.json").read_text())
        del runtime["student_cells"][cell.key]["eval_dir"]
        finite.write_json(queue / "runtime.json", runtime)
        backend.paths = finite.load_paths(queue)
        with self.assertRaisesRegex(KeyError, "eval_dir"):
            backend.archive_retry(cell, "train")
        self.assertEqual(retained.read_bytes(), b"retained reference Student evaluation")
        del runtime["student_cells"]
        finite.write_json(queue / "runtime.json", runtime)
        backend.paths = finite.load_paths(queue)
        with self.assertRaisesRegex(KeyError, "student_cells"):
            backend.archive_retry(cell, "train")
        self.assertEqual(retained.read_bytes(), b"retained reference Student evaluation")
        self.assertFalse(Path(binding["method_dir"] + ".finite-retry-protected-retry").exists())
        self.assertFalse(backend.run_dir.exists())

    def successful_train_files(self, cell):
        work = self.q / "artifacts" / cell.dataset / cell.domain / str(cell.seed) / cell.method
        work = work / ("ddp2/work" if cell.width == 2 else "work")
        work.mkdir(parents=True)
        iteration = 266 if cell.dataset == "RSAR" else 185
        (work / f"iter_{iteration}_ema.pth").write_bytes(b"existing final EMA")
        (work / f"iter_{iteration}.pth").write_bytes(b"existing final Student")
        terminal = work.parent / "terminal_status"
        terminal.write_text("tmux_wrap_exit=0\n")
        if (self.q / "runtime.json").exists():
            method_dir = Path(finite.load_paths(self.q).method_dir(
                cell.dataset, cell.domain, str(cell.seed), cell.method))
            finite.write_json(method_dir / "execution.json", {
                "training_code": "/cpu-fixture/training", "training_code_sha": "cpu-fixture-executed"})
        return work, terminal

    def blocked_student_dependency(self):
        cell = Cell("RSAR", "point_target", 43, "IRG", "student")
        runtime = self.prepared_inputs([cell.model])
        work, terminal = self.successful_train_files(cell.model)
        original = self.root / "original-training-terminal.status"
        terminal.rename(original)
        origin = Path(runtime["source_queues"][cell.model.key])
        wrapper = origin / f"wrap_{cell.model.session('train')}.status"
        wrapper.write_text(f"wrap_exit=0 cell={cell.model.key} phase=train\n")
        process, directory = self.start([], "missing-student-dependency", eval_cells=(cell,))
        self.assertEqual(process.wait(timeout=8), 2)
        row = json.loads((directory / "state.json").read_text())["cells"][0]
        self.assertFalse(row["train_requested"])
        self.assertEqual(row["training_ownership"], "eval_only")
        self.assertEqual((row["train"], row["eval"], row["attempts"]), ("blocked", "blocked", []))
        self.assertEqual(self.events, [])
        return cell, directory / "state.json", work, terminal, original, wrapper

    def test_restored_never_attempted_student_dependency_rechecks_without_retry_or_training(self):
        cell, previous, work, terminal, original, wrapper = self.blocked_student_dependency()
        previous_bytes = previous.read_bytes()
        terminal.write_bytes(original.read_bytes())
        self.assertEqual(finite.train_state(self.q, finite.load_paths(self.q), cell), "complete")
        artifacts = {path: path.read_bytes() for path in (*work.iterdir(), terminal, original, wrapper)}
        process, directory = self.start([], "restored-student-dependency", eval_cells=(cell,), controls=(
            "--previous-state", str(previous)))
        fd = open_pidfd(process.pid)
        try:
            ready, _, _ = select.select([self.events_fd, fd], [], [], 8)
            self.assertNotIn(fd, ready, (directory / "state.json").read_text()
                             if (directory / "state.json").exists() else "resumer exited without state")
        finally:
            os.close(fd)
        event = self.next_start()
        self.assertEqual((event["name"], event["phase"]), (cell.session("eval"), "eval"))
        self.finish(process, 1)
        row = json.loads((directory / "state.json").read_text())["cells"][0]
        self.assertEqual((row["train"], row["eval"]), ("complete", "complete"))
        self.assertFalse(row["train_requested"])
        self.assertEqual(row["training_ownership"], "eval_only")
        self.assertEqual([(a["phase"], a["status"]) for a in row["attempts"]], [("eval", "complete")])
        self.assertEqual(row["reasons"], json.loads(previous_bytes)["cells"][0]["reasons"])
        self.assertEqual(previous.read_bytes(), previous_bytes)
        self.assertEqual(json.loads((directory / "previous_state.json").read_text()), json.loads(previous_bytes))
        self.assertFalse((directory / "jobs" / (cell.session("train") + ".json")).exists())
        self.assertEqual({p.name for p in (directory / "jobs").glob("*.json")},
                         {cell.session("eval") + ".json"})
        self.assertFalse((directory / "recovery").exists())
        self.assertFalse(list(self.q.rglob("*.finite-retry-*")))
        self.assertTrue(all(path.read_bytes() == content for path, content in artifacts.items()))

    def test_restored_student_dependency_missing_conflicting_or_held_stays_blocked(self):
        cell, previous, work, terminal, original, wrapper = self.blocked_student_dependency()
        previous_bytes = previous.read_bytes()
        queue_wrapper = self.q / wrapper.name
        wrapper_bytes = wrapper.read_bytes()
        finals = {path: path.read_bytes() for path in work.iterdir()}
        for case in ("missing_terminal", "conflicting_terminal", "origin_wrapper_conflict",
                     "queue_wrapper_conflict", "held_reference"):
            with self.subTest(case=case):
                terminal.write_bytes(original.read_bytes())
                wrapper.write_bytes(wrapper_bytes)
                if queue_wrapper.exists():
                    queue_wrapper.unlink()
                if case == "missing_terminal":
                    terminal.unlink()
                elif case == "conflicting_terminal":
                    terminal.write_bytes(original.read_bytes() + b"tmux_wrap_exit=9\n")
                elif case == "origin_wrapper_conflict":
                    wrapper.write_bytes(wrapper_bytes + b"wrap_exit=9\n")
                elif case == "queue_wrapper_conflict":
                    queue_wrapper.write_text("wrap_exit=9\n")
                controls = ("--hold-cell", cell.key) if case == "held_reference" else ()
                evidence = finite.train_state(self.q, finite.load_paths(self.q), cell)
                self.assertEqual(evidence, "complete" if case == "held_reference" else "blocked")
                process, directory = self.start([], case, eval_cells=(cell,), controls=(
                    "--previous-state", str(previous), *controls))
                self.assertEqual(process.wait(timeout=8), 2)
                row = json.loads((directory / "state.json").read_text())["cells"][0]
                self.assertEqual((row["train"], row["eval"], row["attempts"]), ("blocked", "blocked", []))
                self.assertEqual(row["held"], case == "held_reference")
                self.assertFalse((directory / "jobs").exists())
                self.assertFalse((directory / "recovery").exists())
                self.assertEqual(previous.read_bytes(), previous_bytes)
                self.assertTrue(all(path.read_bytes() == data for path, data in finals.items()))
        self.assertEqual(self.events, [])

    def test_restored_student_dependency_does_not_reset_attempts_or_completed_models(self):
        cell, previous, work, terminal, original, wrapper = self.blocked_student_dependency()
        terminal.write_bytes(original.read_bytes())
        completed = Cell("DIOR", "clean", 43, "B_REG")
        completed_work, completed_terminal = self.successful_train_files(completed)
        preserved = {path: path.read_bytes() for path in (
            *work.iterdir(), terminal, original, wrapper, *completed_work.iterdir(), completed_terminal)}
        complete_row = {
            "cell": completed.__dict__, "train_requested": True, "training_ownership": "producer",
            "train": "complete", "eval": "complete", "reasons": ["preserved prior success"],
            "adopted": [], "held": False,
            "attempts": [{"phase": "train", "status": "complete"},
                         {"phase": "eval", "status": "complete"}]}
        initial = json.loads(previous.read_text())
        for case, updates in (
                ("failed_eval", {"eval": "failed", "attempts": [{"phase": "eval", "status": "failed"}]}),
                ("blocked_prior_attempt", {"attempts": [{"phase": "eval", "status": "failed"}]}),
                ("blocked_prior_adoption", {"adopted": [{"phase": "eval", "pid": 123, "gpus": [4]}]}),
                ("completed_reference", {"train": "complete", "eval": "complete",
                                         "attempts": [{"phase": "eval", "status": "complete"}]})):
            with self.subTest(case=case):
                old_row = {**initial["cells"][0], **updates}
                old = self.root / (case + "-previous.json")
                finite.write_json(old, {**initial, "cells": [old_row, complete_row]})
                old_bytes = old.read_bytes()
                process, directory = self.start([completed], case, eval_cells=(cell,), controls=(
                    "--previous-state", str(old)))
                expected = 0 if case == "completed_reference" else 1 if case == "failed_eval" else 2
                self.assertEqual(process.wait(timeout=8), expected)
                rows = {Cell(**r["cell"]): r for r in json.loads((directory / "state.json").read_text())["cells"]}
                self.assertEqual(rows[cell], old_row)
                self.assertEqual(rows[completed], complete_row)
                self.assertEqual(old.read_bytes(), old_bytes)
                self.assertFalse((directory / "jobs").exists())
                self.assertFalse((directory / "recovery").exists())
                self.assertTrue(all(path.read_bytes() == data for path, data in preserved.items()))
        self.assertEqual(self.events, [])

    def test_restored_student_dependency_preserves_partial_eval_and_checks_admission(self):
        cell, previous, _, terminal, original, _ = self.blocked_student_dependency()
        terminal.write_bytes(original.read_bytes())
        previous_bytes = previous.read_bytes()
        paths = finite.load_paths(self.q)
        out = Path(paths.eval_student_dir(cell.dataset, cell.domain, str(cell.seed), cell.method))
        out.mkdir(parents=True)
        partial = out / "predictions.pkl"
        partial.write_bytes(b"preserved interrupted native evaluation")
        process, directory = self.start([], "partial-student-eval", eval_cells=(cell,), controls=(
            "--previous-state", str(previous)))
        self.assertEqual(process.wait(timeout=8), 2)
        row = json.loads((directory / "state.json").read_text())["cells"][0]
        self.assertEqual((row["train"], row["eval"], row["attempts"]), ("complete", "blocked", []))
        self.assertEqual(partial.read_bytes(), b"preserved interrupted native evaluation")
        self.assertFalse((directory / "jobs").exists())
        self.assertFalse((directory / "recovery").exists())
        # A different empty-destination reference still needs original input admission.
        other = Cell("RSAR", "point_target", 44, "IRG", "student")
        self.successful_train_files(other.model)
        runtime = self.prepared_inputs([other.model])
        origin = Path(runtime["source_queues"][other.model.key])
        (origin / "runtime.json").unlink()
        old_row = {**json.loads(previous_bytes)["cells"][0], "cell": other.__dict__}
        old = self.root / "other-blocked-state.json"
        finite.write_json(old, {"queue": str(self.q), "cells": [old_row]})
        process, directory = self.start([], "missing-original-runtime", eval_cells=(other,), controls=(
            "--previous-state", str(old)))
        self.assertEqual(process.wait(timeout=8), 2)
        row = json.loads((directory / "state.json").read_text())["cells"][0]
        self.assertEqual((row["train"], row["eval"], row["attempts"]), ("complete", "waiting", []))
        self.assertIn("runtime.json", row["eval_input_reason"])
        self.assertFalse((directory / "jobs").exists())
        self.assertEqual(previous.read_bytes(), previous_bytes)
        self.assertEqual(self.events, [])

    def test_restored_student_dependency_adopts_completed_native_eval_without_attempt(self):
        cell, previous, _, terminal, original, _ = self.blocked_student_dependency()
        previous_bytes = previous.read_bytes()
        terminal.write_bytes(original.read_bytes())
        external, _ = self.start([], "original-student-eval", eval_cells=(cell,))
        self.assertEqual(self.next_start()["name"], cell.session("eval"))
        self.finish(external, 1)
        paths = finite.load_paths(self.q)
        self.assertEqual(finite.eval_state(self.q, paths, cell), "complete")
        out = Path(paths.eval_student_dir(cell.dataset, cell.domain, str(cell.seed), cell.method))
        native = {path: path.read_bytes() for path in out.iterdir()}
        process, directory = self.start([], "restored-existing-eval", eval_cells=(cell,), controls=(
            "--previous-state", str(previous)))
        self.assertEqual(process.wait(timeout=8), 0)
        row = json.loads((directory / "state.json").read_text())["cells"][0]
        self.assertEqual((row["train"], row["eval"], row["attempts"]), ("complete", "complete", []))
        self.assertFalse((directory / "jobs").exists())
        self.assertFalse((directory / "recovery").exists())
        self.assertEqual(previous.read_bytes(), previous_bytes)
        self.assertTrue(all(path.read_bytes() == data for path, data in native.items()))
        self.assertEqual([e["name"] for e in self.events if e["event"] == "start"], [cell.session("eval")])

    def test_existing_success_is_not_retrained_and_invalid_ids_are_blocked(self):
        cell = Cell("DIOR", "cloudy", 44, "C")
        self.successful_train_files(cell)
        process, _ = self.start([cell], "skip-train")
        self.assertEqual(self.next_start()["name"], cell.session("eval"))
        self.finish(process, 1)
        again, directory = self.start([cell], "already-done")
        self.assertEqual(again.wait(timeout=8), 0)
        self.assertEqual(json.loads((directory / "state.json").read_text())["status"], "complete")
        sidecar = (self.q / "artifacts/DIOR/cloudy/44/C/eval_full_cloudy_ids_v1"
                   / "predictions.pkl.image_ids.json")
        data = json.loads(sidecar.read_text())
        data["records"][0]["image_id"] = "wrong"
        sidecar.write_text(json.dumps(data))
        broken, directory = self.start([cell], "invalid-ids")
        self.assertEqual(broken.wait(timeout=8), 2)
        self.assertEqual(json.loads((directory / "state.json").read_text())["cells"][0]["eval"], "blocked")

    def test_last_failure_with_existing_final_files_is_not_success_or_retrained(self):
        cell = Cell("RSAR", "clean", 43, "C")
        _, terminal = self.successful_train_files(cell)
        with terminal.open("a") as stream:
            stream.write("tmux_wrap_exit=9\n")
        process, directory = self.start([cell])
        self.assertEqual(process.wait(timeout=8), 2)
        self.assertEqual(json.loads((directory / "state.json").read_text())["cells"][0]["train"], "blocked")
        self.assertEqual(self.events, [])

    def test_actual_legacy_gpu5_canonical_task_is_adopted(self):
        external = Cell("DIOR", "cloudy", 44, "C")
        name = external.session("train")
        command = shlex.join(["bash", str(self.q / "run_train_1gpu.sh"), "5",
                              "DIOR", "cloudy", "44", "C"])
        command += f"; echo wrap_exit=$? >> {self.q}/wrap_{name}.status"
        subprocess.run(["tmux", "new-session", "-d", "-s", name, command],
                       env=self.env, check=True)
        first = self.next_start()
        self.assertEqual(first["gpus"], [5])
        cells = [external, Cell("DIOR", "contrast", 44, "C"), Cell("RSAR", "chaff", 43, "E")]
        process, directory = self.start(cells)
        starts = [self.next_start(), self.next_start()]
        self.assertEqual({g for e in starts for g in e["gpus"]}, {4, 6, 7})
        self.finish(process, 6)
        state = json.loads((directory / "state.json").read_text())
        adopted = next(r for r in state["cells"] if r["cell"] == external.__dict__)
        self.assertEqual(adopted["adopted"][0]["gpus"], [5])
        self.assertEqual(sum(e["event"] == "start" and e["name"] == name for e in self.events), 1)

    def test_external_future_cell_is_never_submitted_and_owner_exit_unblocks_eval(self):
        cloudy = Cell("DIOR", "cloudy", 44, "C")
        contrast = Cell("DIOR", "contrast", 44, "C")
        own = Cell("RSAR", "clean", 43, "B")

        def external_train(cell):
            name = cell.session("train")
            command = shlex.join(["bash", str(self.q / "run_train_1gpu.sh"), "5",
                                  cell.dataset, cell.domain, str(cell.seed), cell.method])
            command += f"; echo wrap_exit=$? >> {self.q}/wrap_{name}.status"
            subprocess.run(["tmux", "new-session", "-d", "-s", name, command],
                           env=self.env, check=True)

        owner_name = "gpu5-xaf-pair-owner"
        owner_fifo = self.q / "release" / owner_name
        os.mkfifo(owner_fifo)
        subprocess.run(["tmux", "new-session", "-d", "-s", owner_name,
                        shlex.join(["cat", str(owner_fifo)])], env=self.env, check=True)
        external_train(cloudy)
        self.assertEqual(self.next_start()["name"], cloudy.session("train"))
        process, directory = self.start(
            [cloudy, contrast, own], external=(cloudy, contrast), owners=(owner_name + "=5",))
        self.assertEqual(self.next_start()["name"], own.session("train"))
        self.release(cloudy.session("train"))
        evaluation = self.next_start()
        self.assertEqual(evaluation["name"], cloudy.session("eval"))
        self.assertNotIn(5, evaluation["gpus"])  # Owner retains GPU5 across the gap.
        self.assertFalse((directory / "jobs" / (contrast.session("train") + ".json")).exists())

        self.release(own.session("train"))
        self.assertEqual(self.next_start()["name"], own.session("eval"))
        self.release(cloudy.session("eval"))
        self.release(own.session("eval"))
        # The auxiliary owner, not the new producer, creates the future canonical task.
        external_train(contrast)
        self.assertEqual(self.next_start()["name"], contrast.session("train"))
        self.release(contrast.session("train"))
        self.release(owner_name)
        last = self.next_start()
        self.assertEqual(last["name"], contrast.session("eval"))
        self.release(last["name"])
        self.assertEqual(process.wait(timeout=10), 0)
        state = json.loads((directory / "state.json").read_text())
        self.assertEqual(state["status"], "complete")
        for cell in (cloudy, contrast):
            row = next(r for r in state["cells"] if r["cell"] == cell.__dict__)
            self.assertFalse(row["train_requested"])
            self.assertEqual(row["training_ownership"], "external")
            self.assertFalse((directory / "jobs" / (cell.session("train") + ".json")).exists())
            self.assertEqual(sum(e["event"] == "start" and e["name"] == cell.session("train")
                                 for e in self.events), 1)

    def test_cell_lock_blocks_a_second_gpu_worker_without_overwriting_winner_status(self):
        cell = Cell("RSAR", "clean", 43, "C")
        def job(number, gpu):
            file = self.root / f"job-{number}.json"
            receipt = self.root / f"receipt-{number}.json"
            file.write_text(json.dumps({
                "cell": cell.__dict__, "phase": "train", "gpus": [gpu],
                "queue": str(self.q), "receipt": str(receipt), "tmux": ["tmux"]}))
            process = subprocess.Popen([
                sys.executable, "-m", "experiments.comparison.finite_resumer", "worker", str(file)],
                env=self.env, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            self.processes.append(process)
            return process, receipt
        winner, _ = job(1, 4)
        self.assertEqual(self.next_start()["gpus"], [4])
        try:
            loser, receipt = job(2, 5)
            self.assertEqual(loser.wait(timeout=8), 75)
            self.assertEqual(json.loads(receipt.read_text())["status"], "blocked")
            self.assertFalse((self.q / f"wrap_{cell.session('train')}.status").exists())
        finally:
            self.release(cell.session("train"))
            winner.wait(timeout=8)
        self.assertEqual(winner.returncode, 0)
        self.assertEqual(sum(e["event"] == "start" for e in self.events), 1)

    def test_old_producer_is_rejected_but_not_stopped(self):
        control = self.q / "old-control"
        os.mkfifo(control)
        script = self.q / "resume_empty_gpu.sh"
        script.write_text("read value < " + shlex.quote(str(control)) + "\n")
        old = subprocess.Popen(["bash", str(script)], env=self.env)
        self.processes.append(old)
        process, _ = self.start([Cell("RSAR", "clean", 43, "C")])
        self.assertEqual(process.wait(timeout=8), 2)
        self.assertIsNone(old.poll())
        self.assertIn("Old producer still active", (self.root / "run.log").read_text())

    def test_launch_entry_keeps_producer_pane_until_finite_completion(self):
        cell = Cell("RSAR", "clean", 43, "C")
        tasks = self.q / "launch-tasks.txt"
        tasks.write_text("RSAR clean 43 C\n")
        directory = self.root / "launched"
        process = subprocess.Popen([
            sys.executable, "-m", "experiments.comparison.finite_resumer", "launch",
            "--queue", str(self.q), "--run-dir", str(directory),
            "--train-list", str(tasks), "--handoff-confirmed",
        ], env=self.env, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.processes.append(process)
        self.assertEqual(process.wait(timeout=8), 0)
        self.assertEqual(self.next_start()["name"], cell.session("train"))
        pid = int(subprocess.check_output(
            ["tmux", "display-message", "-p", "-t", "xaf-finite-producer", "#{pane_pid}"],
            env=self.env, text=True).strip())
        fd = open_pidfd(pid)
        try:
            self.finish(process, 2)
            self.assertTrue(select.select([fd], [], [], 8)[0])
        finally:
            os.close(fd)
        self.assertEqual(json.loads((directory / "state.json").read_text())["status"], "complete")


if __name__ == "__main__":
    unittest.main()
