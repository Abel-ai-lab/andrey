"""Sequential, killable library processes with bounded setup and fit deadlines."""

from __future__ import annotations

import atexit
import json
import logging
import os
import select
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

THREAD_ENV = {
    name: "1"
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMBA_NUM_THREADS",
        "ANDREY_NUM_WORKERS",
    )
}
THREAD_ENV.update(ANDREY_BACKEND="numpy", ANDREY_DEVICE="cpu", CUDA_VISIBLE_DEVICES="")
FIT_TIMEOUT = 60
SETUP_TIMEOUT = 120
PROTOCOL_FD = "ANDREY_DEMO_PROTOCOL_FD"


class FitError(Exception):
    """A request failed inside a worker that stays ready for the next one."""


class Worker:
    """Own a process and kill its entire process group on failure."""

    def __init__(self, package, command=None):
        self.package = package
        self.command = command or [
            sys.executable,
            str(Path(__file__).with_name("worker.py")),
            package,
        ]
        self.process = None
        self.channel = None
        self.buffer = b""
        self.version = ""

    def close(self):
        """Reap a worker and any children before another fit can start."""
        if self.process is not None:
            process, self.process = self.process, None
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
            process.stdin.close()
            os.close(self.channel)
            self.channel = None
            self.buffer = b""

    def receive(self, timeout):
        """Read one bounded protocol message, including partial-pipe reads."""
        deadline = time.monotonic() + timeout
        while b"\n" not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.channel], [], [], remaining)[0]:
                raise TimeoutError
            part = os.read(self.channel, 4096)
            if not part:
                raise RuntimeError("Worker stopped.")
            self.buffer += part
            if len(self.buffer) > 8192:
                raise RuntimeError("Worker protocol limit exceeded.")
        line, self.buffer = self.buffer.split(b"\n", 1)
        message = json.loads(line)
        if message.get("event") == "error":
            raise FitError(message["error"])
        return message

    def start(self):
        """Complete imports and warm-up before accepting any timed work."""
        if self.process is not None:
            return
        # Protocol messages use their own pipe, so library writes to stdout cannot corrupt them.
        self.channel, write = os.pipe()
        try:
            self.process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                pass_fds=(write,),
                start_new_session=True,
                env={**os.environ, **THREAD_ENV, PROTOCOL_FD: str(write)},
            )
        except BaseException:
            os.close(self.channel)
            self.channel = None
            raise
        finally:
            os.close(write)
        try:
            ready = self.receive(SETUP_TIMEOUT)
            if ready.get("event") != "ready":
                raise RuntimeError("Worker did not become ready.")
            self.version = ready["version"]
        except BaseException:
            self.close()
            raise

    def run(self, method, data, timeout=FIT_TIMEOUT):
        """Return fit seconds and marks; a failed fit keeps its worker, a timeout kills it."""
        phase = "setup"
        try:
            self.start()
            with tempfile.TemporaryDirectory(prefix="andrey-fit-") as directory:
                source, output = Path(directory) / "data.npy", Path(directory) / "graph.npy"
                np.save(source, data, allow_pickle=False)
                request = {"method": method, "data": str(source), "output": str(output)}
                self.process.stdin.write((json.dumps(request) + "\n").encode())
                self.process.stdin.flush()
                if self.receive(10).get("event") != "fit":
                    raise RuntimeError("Worker did not start a fit.")
                phase = "fit"
                result = self.receive(timeout)
                if result.get("event") != "result":
                    raise RuntimeError("Worker did not return a result.")
                marks = np.load(output, allow_pickle=False)
                return {
                    "status": "ok",
                    "seconds": result["seconds"],
                    "marks": marks,
                    "version": self.version,
                }
        except FitError as error:
            return {
                "status": "error",
                "seconds": None,
                "version": self.version,
                "error": str(error),
            }
        except TimeoutError:
            self.close()
            return {
                "status": "timeout" if phase == "fit" else "setup_timeout",
                "seconds": None,
                "version": self.version,
            }
        except Exception as error:
            self.close()
            return {
                "status": "error",
                "seconds": None,
                "version": self.version,
                "error": str(error),
            }
        except BaseException:
            self.close()
            raise


class Workers:
    """Serialize all work, including calls made outside the Gradio queue."""

    def __init__(self):
        self.lock = threading.Lock()
        self.workers = {}
        self.available = []
        atexit.register(self.close)

    def warm(self):
        """Warm each package sequentially and fail startup if a required one is absent."""
        with self.lock:
            self.available = []
            for package in ("Andrey", "causal-learn", "lingam", "gCastle"):
                worker = self.workers.setdefault(package, Worker(package))
                try:
                    worker.start()
                except Exception as error:
                    if package == "gCastle":
                        logging.warning("gCastle is unavailable: %s", error)
                        continue
                    raise
                self.available.append(package)

    def run(self, package, method, data, timeout=FIT_TIMEOUT):
        """Run one package under the global compute lock."""
        with self.lock:
            worker = self.workers.setdefault(package, Worker(package))
            return worker.run(method, data, timeout)

    def close(self):
        """Reap all workers at shutdown."""
        for worker in self.workers.values():
            worker.close()
