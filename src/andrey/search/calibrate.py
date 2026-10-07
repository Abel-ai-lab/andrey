"""Measure the GES and HC worker-pool cutoffs on this machine (``andrey calibrate``).

A GES or HC pass runs on the worker pool only when its estimated work exceeds a cutoff:
``ANDREY_GES_PARALLEL_MIN_WORK`` (estimated candidate evaluations) and
``ANDREY_HC_PARALLEL_MIN_WORK`` (moves scanned). Where the pool starts to pay depends on the
machine, the worker count, and the numerical libraries, so the defaults are a starting point.
Nothing here runs unless called: fits never calibrate.

Each method is timed on whole fits of a sparse linear-Gaussian sample over ``d`` variables: a
serial fit against one that runs every pass on the pool, including starting it. :func:`bracket`
finds the two sizes around the crossover, extending below or above the probe grid when needed. The
cutoff is the first pass's work there, ``d * (d - 1)`` for both methods: a fit that large uses the
pool from its first pass, and a smaller fit only for its larger later passes. The process's first
pool start is paid before timing, so the values suit many fits in one process (:data:`NOTE`).
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

COMMAND = "andrey calibrate"
CUTOFFS = {"ges": "ANDREY_GES_PARALLEL_MIN_WORK", "hc": "ANDREY_HC_PARALLEL_MIN_WORK"}
NOTE = (
    "These values suit many fits in one process. Starting the first pool in a process costs more, "
    "so a single fit gains from the pool only at larger sizes; the defaults allow for it."
)
# A fit with more than one worker on a default cutoff warns once a pass's estimated work reaches
# this fraction of the cutoff. The cutoffs measured for many fits per process sit within this factor
# below the defaults, so a smaller fit would not use the pool under either value.
WARN_FRACTION = 0.1
# What each default suits: GES's allows for the first pool start of a one-off fit; HC's sits
# between the cutoffs for one fit and for many fits per process.
_DEFAULT_SUITS = {
    "ges": "which suits one fit per process",
    "hc": "which sits between the values for one fit and for many fits per process",
}

# Probe sizes (variables). Serial GES at d = 256 takes minutes, so the search stops there.
GRID = (16, 32, 64, 128)
_MIN_D, _MAX_D = 4, 256
_REFINE_STEPS = 3


@dataclass(frozen=True)
class Crossover:
    """Where the pooled pass starts to beat the serial one, as sizes ``d`` and work ``d(d - 1)``.

    ``below`` is the largest measured size where the serial fit was faster, ``above`` the smallest
    where the pooled pass was. ``None`` marks a side the search could not reach: the pool was faster
    down to the smallest size, or slower up to the largest. ``cutoff`` is the work to set.
    """

    below: int | None
    above: int | None
    timings: tuple[tuple[int, float, float], ...]

    @property
    def cutoff(self) -> int:
        if self.above is None:  # never faster: the largest size's first-pass work
            return _work(_MAX_D)
        if self.below is None:  # faster everywhere measured: use the pool from the smallest size
            return _work(self.above)
        return _round2(math.sqrt(_work(self.below) * _work(self.above)))


def cutoff_is_default(method: str) -> bool:
    """Whether ``method``'s cutoff is its built-in default: the variable is unset or blank."""
    from andrey.core import env

    return env.SETTINGS[CUTOFFS[method]].read() is None


def warn_default_cutoff(method: str, default: int) -> None:
    """Warn once that a ``method`` fit with more than one worker runs on the default cutoff."""
    from andrey.core.warning_policy import PerformanceWarning, warn_once

    warn_once(
        f"{method.upper()} decides when to use its worker pool with the default "
        f"{CUTOFFS[method]}={default}, {_DEFAULT_SUITS[method]}. If this process runs many fits, "
        f"`{COMMAND}` measures a value for this machine; any explicit value silences this warning.",
        PerformanceWarning,
    )


def bracket(
    serial: Callable[[int], float],
    pooled: Callable[[int], float],
    grid: Sequence[int] = GRID,
    *,
    refine: int = _REFINE_STEPS,
) -> Crossover:
    """Find the sizes around the serial/pooled crossover from timing functions of ``d``.

    Scans ``grid`` upward for the first size where the pooled pass is faster. If that is the first
    grid size, halves ``d`` until the serial pass wins again (or ``d`` reaches 4); if no grid size
    qualifies, doubles past the grid (up to 256). Then narrows the bracket by ``refine`` bisections
    of ``log d``.
    """
    timings: dict[int, tuple[float, float]] = {}

    def faster_pooled(d: int) -> bool:
        if d not in timings:
            timings[d] = (serial(d), pooled(d))
        s, p = timings[d]
        return p < s

    sizes = sorted(grid)
    below: int | None = None
    above: int | None = None
    for d in sizes:
        if faster_pooled(d):
            above = d
            break
        below = d
    if above is not None and below is None:  # the crossover lies below the grid
        d = above
        while d > _MIN_D:
            d = max(_MIN_D, d // 2)
            if not faster_pooled(d):
                below = d
                break
            above = d
    elif above is None:  # the crossover lies above the grid
        d = sizes[-1]
        while d < _MAX_D:
            d = min(_MAX_D, d * 2)
            if faster_pooled(d):
                above = d
                break
            below = d
    if below is not None and above is not None:
        for _ in range(refine):
            mid = round(math.sqrt(below * above))
            if mid in (below, above):
                break
            if faster_pooled(mid):
                above = mid
            else:
                below = mid
    rows = tuple((d, s, p) for d, (s, p) in sorted(timings.items()))
    return Crossover(below, above, rows)


def _work(d: int) -> int:
    return d * (d - 1)


def _round2(x: float) -> int:
    """``x`` rounded to two significant figures."""
    if x <= 0:
        return 0
    scale = 10 ** max(0, int(math.floor(math.log10(x))) - 1)
    return int(round(x / scale) * scale)


def _time(fn: Callable[[], object], repeats: int) -> float:
    """Median seconds of ``fn`` after one warm-up call; a single run once a call passes 1 s."""
    t0 = time.perf_counter()
    fn()
    if time.perf_counter() - t0 > 1.0:
        repeats = 1
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def sample(d: int, seed: int = 0) -> np.ndarray:
    """A linear-Gaussian SEM over ``d`` variables with average degree 2 and ``10 * d`` rows."""
    rng = np.random.default_rng(seed)
    edges = np.tril(rng.random((d, d)) < 2.0 / max(d - 1, 1), -1)
    weights = rng.uniform(0.4, 1.2, (d, d)) * rng.choice([-1.0, 1.0], (d, d)) * edges
    x = rng.standard_normal((10 * d, d))
    for j in range(d):
        x[:, j] += x @ weights[j]
    return x


def fit_timers(method: str, workers: int, repeats: int = 3) -> tuple[Callable[[int], float], ...]:
    """Serial and pooled timing functions for a whole ``method`` fit on :func:`sample` data.

    The pooled arm sets the method's cutoff to 0, so every pass runs on the pool and the time
    includes starting it; the process's first pool start is paid before timing.
    """
    import andrey

    fit = getattr(andrey, method)
    var = CUTOFFS[method]

    def run(d: int, n_workers: int, cutoff: str | None) -> float:
        x = sample(d)
        old = os.environ.pop(var, None)
        try:
            if cutoff is not None:
                os.environ[var] = cutoff
            with andrey.config(num_workers=n_workers):
                return _time(lambda: fit(x), repeats)
        finally:
            os.environ.pop(var, None)
            if old is not None:
                os.environ[var] = old

    return (lambda d: run(d, 1, None)), (lambda d: run(d, workers, "0"))


def calibrate(workers: int, methods: Sequence[str] = ("ges", "hc")) -> dict[str, Crossover]:
    """Measure each method's crossover with ``workers`` pool processes."""
    import warnings

    import andrey

    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", andrey.AndreyWarning)
        for m in methods:
            serial, pooled = fit_timers(m, workers)
            pooled(8)  # start the process's first pool (the forkserver) before timing
            out[m] = bracket(serial, pooled)
    return out
