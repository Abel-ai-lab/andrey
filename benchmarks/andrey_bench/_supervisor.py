"""Kill the worker's process group when the driver or worker exits.

The driver holds a pipe open; EOF triggers a group kill, including the supervisor and worker pools.
``PR_SET_PDEATHSIG`` alone cannot clean up pools because it reaches only the immediate child.

Invoked as ``python _supervisor.py <death_read_fd> <exit_path> -- <worker cmd...>``. The worker
inherits stdio and the process group. Its exit code is saved before group cleanup so the driver can
classify the result. Uses only the standard library and runs by path in the solution's interpreter.
"""

from __future__ import annotations

import ctypes
import os
import signal
import subprocess
import sys
import threading
import traceback

#: ``prctl`` option: signal this process when the thread that created it exits.
_PR_SET_PDEATHSIG = 1

# Loaded here, in the supervisor, not in the forked child where a dlopen is unsafe.
_LIBC = ctypes.CDLL(None, use_errno=True) if sys.platform == "linux" else None
if _LIBC is not None:
    _LIBC.prctl.restype = ctypes.c_int
    _LIBC.prctl.argtypes = [ctypes.c_int] + [ctypes.c_ulong] * 4


def _pdeathsig() -> None:
    """Set the worker's parent-death signal while preserving its process group."""
    if _LIBC is None:
        return
    if _LIBC.prctl(_PR_SET_PDEATHSIG, int(signal.SIGKILL), 0, 0, 0) != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))


def _kill_group_on_driver_death(death_fd: int) -> None:
    """Wait for the driver's pipe to close, then SIGKILL this process group."""
    try:
        os.read(death_fd, 1)
    except OSError:
        pass
    try:
        os.killpg(os.getpgrp(), signal.SIGKILL)
    except OSError:
        pass


def main(argv: list[str]) -> int:
    death_fd = int(argv[1])
    exit_path = argv[2]
    if argv[3] != "--":
        raise SystemExit("usage: _supervisor.py <death_fd> <exit_path> -- <cmd...>")
    cmd = argv[4:]

    rc = 1
    try:
        # Spawn before starting the watch thread: ``preexec_fn`` requires a single-threaded parent.
        # The worker inherits stdio and the group so output and group kills reach its pool too.
        proc = subprocess.Popen(cmd, preexec_fn=_pdeathsig)

        watch = threading.Thread(target=_kill_group_on_driver_death, args=(death_fd,), daemon=True)
        watch.start()

        rc = proc.wait()
    except BaseException:
        traceback.print_exc()
    finally:
        try:
            # Publish the worker's status before SIGKILL also ends this supervisor. Closing the
            # file flushes it before the driver can observe the supervisor's exit.
            with open(exit_path, "w") as fh:
                fh.write(str(rc))
        finally:
            os.killpg(os.getpgrp(), signal.SIGKILL)
    return 0  # unreachable: the group kill includes this process


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
