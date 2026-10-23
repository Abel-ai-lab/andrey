"""Profile one supported method's fit on one dataset, with the profiler's own cost taken out.

The method's public call and the dataset come from the identity corpus (:mod:`qa.identity`): one
of the method's quality-gate datasets or its larger dataset, at one draw. Every fit runs on the
numpy backend with one worker and one BLAS thread. After the warm-up fits, the wall time is the
minimum over plain fits with no profiler attached, printed with the host: the CPU, the Python,
NumPy, and SciPy versions, the loaded BLAS, and the thread settings.

One more fit runs under cProfile. A profiled call costs the profiler's bookkeeping on top of its
own work, which inflates the functions that make many small calls. :func:`calibrate` measures that
cost per call on this host, from a loop of calls to a function that does nothing. Each function's
own time, a built-in function's included, loses its call count times the cost, floored at 0, and
its share is its corrected own time over the corrected total, so the shares sum to 1. A function's
cumulative share adds the corrected cumulative time of each function it calls, weighted by the part
of that function's time spent under this caller; a recursive call is counted once. Times stay on
screen; the table written by ``-o`` holds shares and call counts only.

    uv run python -m qa.profiler PC --list              # the dataset keys PC runs on
    uv run python -m qa.profiler PC linear_gauss_er_d20 # wall time and the top 25 functions
    uv run python -m qa.profiler BOSS linear_gauss_er_d100 --seed 1 --rows 40 -o boss.json
"""

from __future__ import annotations

import argparse
import cProfile
import gc
import json
import sys
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

import andrey
from qa import identity
from qa import quality_gate as qg

#: Calls per calibration loop, and loops per calibration; the fastest loop counts.
CALIBRATION_CALLS = 200_000
CALIBRATION_LOOPS = 5


def keys(method: str) -> tuple[str, ...]:
    """Return the dataset keys ``method`` runs on: its gate datasets, then its larger dataset."""
    return tuple(dict.fromkeys((*qg.SUPPORTED[method].datasets, identity.LARGER[method])))


def sample(dataset: str, seed: int) -> np.ndarray:
    """Return draw ``seed`` of ``dataset``, in C order as the identity corpus fits it."""
    return np.ascontiguousarray(identity.draws(dataset)[seed].data)


@contextmanager
def single_thread() -> Generator[None]:
    """Fit on the numpy backend with one worker and one BLAS thread."""
    with andrey.config(backend="numpy", num_workers=1), threadpool_limits(limits=1):
        yield


def wall_time(call: Callable[[np.ndarray], Any], x: np.ndarray, fits: int, warmup: int) -> float:
    """Return the fastest of ``fits`` plain fits, in seconds, after ``warmup`` fits."""
    for _ in range(warmup):
        call(x)
    times = []
    for _ in range(fits):
        start = time.perf_counter()
        call(x)
        times.append(time.perf_counter() - start)
    return min(times)


# ==================================================================================================
# The profiler's cost per call
# ==================================================================================================


def _nothing() -> None:
    pass


def _loop(n: int) -> None:
    for _ in range(n):
        _nothing()


def calibrate() -> float:
    """Return the profiler's cost per call on this host, in seconds: the profiled own time of a
    loop of calls to a function that does nothing, summed over every function, less the loop's
    plain time, over its calls. The fastest of :data:`CALIBRATION_LOOPS` loops counts."""
    plain, profiled = [], []
    for _ in range(CALIBRATION_LOOPS):
        start = time.perf_counter()
        _loop(CALIBRATION_CALLS)
        plain.append(time.perf_counter() - start)
        profiler = cProfile.Profile()
        profiler.runcall(_loop, CALIBRATION_CALLS)
        profiled.append(sum(e.inlinetime for e in profiler.getstats() if not _is_disable(e.code)))
    return max(0.0, (min(profiled) - min(plain)) / CALIBRATION_CALLS)


def _is_disable(code: Any) -> bool:
    return isinstance(code, str) and "_lsprof.Profiler" in code


# ==================================================================================================
# The profile
# ==================================================================================================


@dataclass(frozen=True)
class Row:
    """One function of a profile: its share of the corrected total, as its own time and with the
    functions it calls, and its call count."""

    function: str
    share: float
    cumulative_share: float
    calls: int


def profile(call: Callable[[np.ndarray], Any], x: np.ndarray, overhead: float) -> list[Row]:
    """Fit ``call`` on ``x`` under cProfile and return every function's row, largest share first.

    The garbage collector runs before the fit, so the call counts of two fits match.
    """
    gc.collect()
    profiler = cProfile.Profile()
    profiler.runcall(call, x)
    entries = {e.code: e for e in profiler.getstats() if not _is_disable(e.code)}
    own = {code: max(0.0, e.inlinetime - e.callcount * overhead) for code, e in entries.items()}
    total = sum(own.values())
    cumulative = _cumulative(entries, own)
    modules = _modules()
    rows = [
        Row(
            function=_label(code, modules),
            share=own[code] / total if total > 0 else 0.0,
            cumulative_share=min(1.0, cumulative[code] / total) if total > 0 else 0.0,
            calls=e.callcount,
        )
        for code, e in entries.items()
    ]
    return sorted(rows, key=lambda r: (-r.share, -r.cumulative_share, r.function))


def _cumulative(entries: dict[Any, Any], own: dict[Any, float]) -> dict[Any, float]:
    """Return each function's corrected cumulative time: its own, plus each callee's cumulative
    times the part of the callee's profiled time spent under this caller. A callee already on the
    path being summed, a recursive call, adds nothing."""
    out: dict[Any, float] = {}
    for root in entries:
        if root in out:
            continue
        path, stack = {root}, [(root, iter(entries[root].calls or ()))]
        while stack:
            code, callees = stack[-1]
            edge = next(callees, None)
            if edge is None:
                stack.pop()
                path.discard(code)
                out[code] = own[code] + sum(
                    out.get(e.code, 0.0) * _weight(e, entries)
                    for e in entries[code].calls or ()
                    if e.code in entries and e.code != code
                )
            elif edge.code in entries and edge.code not in out and edge.code not in path:
                path.add(edge.code)
                stack.append((edge.code, iter(entries[edge.code].calls or ())))
    return out


def _weight(edge: Any, entries: dict[Any, Any]) -> float:
    """The part of a callee's profiled cumulative time spent under one caller."""
    total = entries[edge.code].totaltime
    return min(1.0, edge.totaltime / total) if total > 0 else 0.0


def _modules() -> dict[str, str]:
    """Map each imported module's file to its name."""
    out = {}
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if path:
            out.setdefault(str(Path(path).resolve()), name)
    return out


def _label(code: Any, modules: dict[str, str]) -> str:
    """Name a function ``module:qualname:line``; a built-in function by the profiler's name."""
    if isinstance(code, str):
        return code
    path = str(Path(code.co_filename).resolve()) if code.co_filename[:1] != "<" else ""
    module = modules.get(path, code.co_filename)
    return f"{module}:{code.co_qualname}:{code.co_firstlineno}"


# ==================================================================================================
# Command line
# ==================================================================================================


def host() -> list[str]:
    """Describe the host as the identity manifest's header does, one line per part."""
    head = identity.header(identity.commit(), 0, 0, 1)
    blas = "; ".join(
        f"{p.get('internal_api')} {p.get('version')} ({p.get('threading_layer')}, "
        f"threads {p.get('num_threads')})"
        for p in head["blas"]
    )
    threads = ", ".join(f"{k} {v}" for k, v in head["threads"].items() if v is not None)
    return [
        f"cpu: {head['cpu']} ({head['machine']})",
        f"python {head['python']}, numpy {head['numpy']}, scipy {head['scipy']}, "
        f"andrey {head['andrey']} at {head['andrey_path']} ({head['andrey_commit'] or 'no git'})",
        f"blas: {blas or 'none loaded'}",
        f"threads: {threads}",
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Profile one method's fit on one dataset.")
    ap.add_argument("method", help="a supported method")
    ap.add_argument("dataset", nargs="?", help="a dataset key of the method (see --list)")
    ap.add_argument("--list", action="store_true", help="list the method's dataset keys")
    ap.add_argument("--seed", type=int, default=0, help="the draw of the dataset")
    ap.add_argument("--fits", type=int, default=3, help="timed plain fits; the fastest counts")
    ap.add_argument("--warmup", type=int, default=1, help="fits before the timed fits")
    ap.add_argument("--rows", type=int, default=25, help="functions to print")
    ap.add_argument("-o", "--out", type=Path, help="write every function's row to this JSON file")
    args = ap.parse_args(argv)

    if args.method not in identity.FITS:
        ap.error(f"unknown method {args.method!r}; known: {list(identity.FITS)}")
    valid = keys(args.method)
    if args.list:
        print("\n".join(valid))
        return 0
    if args.dataset not in valid:
        ap.error(f"dataset must be one of {list(valid)}")
    if args.fits < 1:
        ap.error("--fits must be at least 1")

    call, x = identity.FITS[args.method].call, sample(args.dataset, args.seed)
    with single_thread():
        seconds = wall_time(call, x, args.fits, args.warmup)
        overhead = calibrate()
        rows = profile(call, x, overhead)
        lines = host()

    print("\n".join(lines))
    print(
        f"{args.method} on {args.dataset} draw {args.seed}, shape {x.shape}: "
        f"{seconds:.3f} s, fastest of {args.fits} fits after {args.warmup} warm-up"
    )
    print(f"profiler cost per call: {overhead * 1e9:.0f} ns, taken out of each function's time\n")
    print(f"{'share':>7} {'cumul':>7} {'calls':>10}  function")
    for r in rows[: args.rows]:
        print(f"{r.share:7.2%} {r.cumulative_share:7.2%} {r.calls:10d}  {r.function}")
    if args.out:
        table = {
            "method": args.method,
            "dataset": args.dataset,
            "seed": args.seed,
            "rows": [asdict(r) for r in rows],
        }
        args.out.write_text(json.dumps(table, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
