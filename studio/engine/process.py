"""The engine process: models load and run outside the window's process.

The window starts an Engine (client); it spawns `serve` in a separate Python process connected by a pipe.
The engine loads every model on parallel threads and reports each one as it is ready (status dots go green
one by one); a ready model can run while others are still loading. Runs are queued and executed one at a
time. If the engine process dies (a model crashed it), pending runs fail with EngineCrashed, the engine is
started again and the models reload; the window keeps running.

Messages (tuples) window -> engine:  ("load", specs, device) | ("run", job, model_id, args) | ("quit",)
                  engine -> window:  ("status", ModelState) | ("result", job, Result) | ("error", job, text)
"""
from __future__ import annotations

import multiprocessing as mp
import os
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

LOAD_THREADS = 4


@dataclass
class ModelState:
    """What the window knows about a model in the engine (the same fields as registry.LoadedModel)."""
    spec: object
    status: str
    error: str = ""
    device: str = "CPU"
    load_seconds: float = 0.0
    info: dict = field(default_factory=dict)


class EngineCrashed(RuntimeError):
    pass


# ---------------------------------------------------------------- engine side
def serve(conn):
    os.environ.setdefault("TQDM_DISABLE", "1")
    from . import load_model, run_model
    from .registry import LOADING, READY
    send_lock = threading.Lock()
    models = {}

    def send(msg):
        with send_lock:
            conn.send(msg)

    def load_one(spec, device):
        lm = load_model(spec, device)
        models[spec.id] = lm
        send(("status", ModelState(spec, lm.status, lm.error, lm.info.get("device", lm.device), lm.load_seconds,
                                   dict(lm.info))))

    loader = ThreadPoolExecutor(LOAD_THREADS, thread_name_prefix="load")
    runner = ThreadPoolExecutor(1, thread_name_prefix="run")

    def run_one(job, model_id, args):
        try:
            lm = models.get(model_id)
            if lm is None or lm.status != READY:
                raise RuntimeError(f"{model_id} is not ready")
            send(("result", job, run_model(lm, *args)))
        except Exception as e:  # noqa: BLE001 - reported to the window, the engine keeps going
            traceback.print_exc()
            send(("error", job, f"{type(e).__name__}: {e}"))

    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            break
        kind = msg[0]
        if kind == "load":
            _, specs, device = msg
            # fast models first, so the template and Zhu DCT are ready at once
            for s in sorted(specs, key=lambda s: (s.runtime != "numpy", s.runtime == "torch", s.order)):
                send(("status", ModelState(s, LOADING)))
                loader.submit(load_one, s, device)
        elif kind == "run":
            runner.submit(run_one, *msg[1:])
        elif kind == "_crash":                      # tests only: simulate a model taking the process down
            os._exit(3)
        elif kind == "quit":
            break
    os._exit(0)                                     # do not wait for model threads


# ---------------------------------------------------------------- window side
class Engine:
    """Client used by the window. on_status(ModelState) is called from a background thread."""

    def __init__(self, on_status=None, on_restart=None):
        self.on_status = on_status or (lambda st: None)
        self.on_restart = on_restart or (lambda: None)
        self._jobs: dict[int, Future] = {}
        self._next = 0
        self._lock = threading.Lock()
        self._specs, self._device = [], "auto"
        self._closing = False
        self.restarts = 0
        self._start()

    def _start(self):
        ctx = mp.get_context("spawn")
        self._conn, child = ctx.Pipe()
        self._proc = ctx.Process(target=serve, args=(child,), name="ppg2ecg-engine", daemon=True)
        self._proc.start()
        child.close()
        self._reader = threading.Thread(target=self._read, name="engine-reader", daemon=True)
        self._reader.start()

    def _send(self, msg):
        with self._lock:
            self._conn.send(msg)

    def load(self, specs, device="auto"):
        self._specs, self._device = list(specs), device
        self._send(("load", self._specs, device))

    def run(self, model_id, ppg128, peaks, ecg128=None, seed=0, src=None) -> Future:
        f = Future()
        with self._lock:
            self._next += 1
            job = self._next
            self._jobs[job] = f
            try:
                self._conn.send(("run", job, model_id, (ppg128, peaks, ecg128, seed, src)))
            except (OSError, EOFError, BrokenPipeError) as e:
                self._jobs.pop(job, None)
                f.set_exception(EngineCrashed(f"engine not reachable: {e}"))
        return f

    def _read(self):
        conn = self._conn
        while True:
            try:
                msg = conn.recv()
            except (EOFError, OSError):
                break
            kind = msg[0]
            if kind == "status":
                self.on_status(msg[1])
            elif kind in ("result", "error"):
                with self._lock:
                    f = self._jobs.pop(msg[1], None)
                if f is not None:
                    f.set_result(msg[2]) if kind == "result" else f.set_exception(RuntimeError(msg[2]))
        if self._closing:
            return
        self._crashed()

    def _crashed(self):
        code = self._proc.exitcode if self._proc is not None else None
        with self._lock:
            jobs, self._jobs = self._jobs, {}
        for f in jobs.values():
            f.set_exception(EngineCrashed(f"the engine stopped (exit code {code}); it was restarted"))
        self.restarts += 1
        self.on_restart()
        time.sleep(0.2)
        self._start()
        if self._specs:
            self.load(self._specs, self._device)

    def crash_for_test(self):
        self._send(("_crash",))

    def close(self):
        self._closing = True
        try:
            self._send(("quit",))
        except (OSError, EOFError, BrokenPipeError):
            pass
        self._proc.join(3)
        if self._proc.is_alive():
            self._proc.kill()
