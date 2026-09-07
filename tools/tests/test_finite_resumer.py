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

from experiments.comparison.finite_resumer import Cell, EVAL_SHA, open_pidfd, pane_job, runner_command


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
'''

RUNNER_SOURCE = '''
import fcntl,json,os,pathlib,pickle,sys
sys.path.insert(0,str(pathlib.Path(__file__).parent))
import paths
q=pathlib.Path(__file__).parent
kind,*args=sys.argv[1:]
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
else:
    out=pathlib.Path(paths.eval_full_dir(ds,domain,seed,method)); out.mkdir(parents=True,exist_ok=True)
    (out/"eval_status").write_text("eval_exit="+str(rc)+" name="+method+" domain="+domain+" seed="+seed+"\\n")
    if rc==0:
        with (out/"predictions.pkl").open("wb") as f: pickle.dump([[],[]],f)
        (out/"pred_count.txt").write_text("2 expect=2\\n")
        (out/"class_ap.txt").write_text("| class | ap |\\n")
        (out/"eval_fixture.json").write_text(json.dumps({"config":"/fixture.py","metric":{"mAP":0.5}}))
        (out/"predictions.pkl.image_ids.json").write_text(json.dumps({
            "schema":"iraod-prediction-image-order-v1","origin":"inference_batch_img_metas",
            "status":"complete","checkpoint":paths.ema_path(ds,domain,seed,method),
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
busy=json.load(open({str(self.gpu_state)!r}))
if any(a.startswith('--query-gpu=') for a in sys.argv):
    print('\\n'.join(str(g)+', GPU-fixture-'+str(g)+', '+str(1000 if g in busy else 0) for g in (4,5,6,7)))
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

    def start(self, cells, name="run", eval_cells=(), external=(), owners=()):
        train = self.q / (name + "-train.txt")
        evaluate = self.q / (name + "-eval.txt")
        train.write_text("".join(f"{c.dataset} {c.domain} {c.seed} {c.method}\n" for c in cells))
        evaluate.write_text("".join(f"{c.dataset} {c.domain} {c.seed} {c.method}\n" for c in eval_cells))
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
            "--queue", str(self.q), "--run-dir", str(run_dir),
            "--train-list", str(train), "--eval-list", str(evaluate), "--handoff-confirmed", *extra,
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

    def test_failure_does_not_launch_dependent_eval_or_become_success(self):
        cell = Cell("RSAR", "clean", 43, "C")
        process, directory = self.start([cell])
        self.assertEqual(self.next_start()["name"], cell.session("train"))
        self.release(cell.session("train"), rc=9)
        self.assertEqual(process.wait(timeout=10), 1)
        state = json.loads((directory / "state.json").read_text())
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["cells"][0]["eval"], "blocked")

    def successful_train_files(self, cell):
        work = self.q / "artifacts" / cell.dataset / cell.domain / str(cell.seed) / cell.method
        work = work / ("ddp2/work" if cell.width == 2 else "work")
        work.mkdir(parents=True)
        iteration = 266 if cell.dataset == "RSAR" else 185
        (work / f"iter_{iteration}_ema.pth").write_bytes(b"existing final EMA")
        (work / f"iter_{iteration}.pth").write_bytes(b"existing final Student")
        terminal = work.parent / "terminal_status"
        terminal.write_text("tmux_wrap_exit=0\n")
        return work, terminal

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
