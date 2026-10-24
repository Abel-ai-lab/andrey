"""Measure the worker-pool cutoffs on this machine (``andrey calibrate``).

A GES pass uses the pool when estimated candidate evaluations exceed
``ANDREY_GES_PARALLEL_MIN_WORK``, and a conditioning-size pass of the PC, FCI, GFCI, and CD-NOD
skeleton search when its tests reach ``ANDREY_SKELETON_PARALLEL_MIN_WORK``, or, with RCoT, when its
tests times the rows reach ``ANDREY_RCOT_PARALLEL_MIN_WORK``. The crossover depends on the machine,
worker count, and numerical libraries. Fits never calibrate automatically.

GES is timed on whole fits of sparse linear-Gaussian data over ``d`` variables: serial against every
pass pooled, pool start included. The skeleton search is timed on one pass over ``d`` variables, at
each conditioning size in :data:`SKELETON_SIZES`, the pass scanned in this process against the same
pass on a started pool, with the pool start included; the cutoff is the largest of the sizes'.
RCoT's skeleton search is timed the same way on one pass at conditioning size 2 over ``d`` variables
of :data:`RCOT_ROWS` rows (:func:`rcot_pass`).
:func:`bracket` finds sizes around the crossover, extending beyond the probe grid when needed. Each
size's work is the largest estimated work of a pass in its pooled run, as the gate counts it: a
cutoff below it sends that pass, and every pass after it, to the pool. The process's first pool
start occurs before timing, so the value suits many fits in one process (:data:`NOTE`).
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from andrey.core.ci import CITest

COMMAND = "andrey calibrate"


@dataclass(frozen=True)
class Method:
    """A worker-pool cutoff: its environment variable, how the default-cutoff warning names the
    search, and whether the gate sends a pass of exactly the cutoff's work to the pool."""

    variable: str
    search: str
    at_cutoff: bool


METHODS = {
    "ges": Method("ANDREY_GES_PARALLEL_MIN_WORK", "GES", at_cutoff=False),
    "skeleton": Method("ANDREY_SKELETON_PARALLEL_MIN_WORK", "The skeleton search", at_cutoff=True),
    "rcot": Method("ANDREY_RCOT_PARALLEL_MIN_WORK", "The RCoT skeleton search", at_cutoff=True),
}
NOTE = (
    "This value suits many fits in one process. Starting the first pool in a process costs more, "
    "so a single fit gains from the pool only at larger sizes; the default allows for it."
)
# A fit with more than one worker on a default cutoff warns once a pass's estimated work reaches
# this fraction of the cutoff. The cutoff measured for many fits per process sits within this factor
# below the default, so a smaller fit would not use the pool under either value.
WARN_FRACTION = 0.1

# Probe sizes (variables). Serial GES at d = 256 takes minutes, so the search stops there.
GRID = (16, 32, 64, 128)
_MIN_D, _MAX_D = 4, 256
_REFINE_STEPS = 3


@dataclass(frozen=True)
class Crossover:
    """Where the pooled run starts to beat the serial one, as sizes ``d`` and their pass work.

    ``below`` is the largest measured size where the serial run was faster, ``above`` the smallest
    where the pooled run was. ``None`` marks a side the search could not reach: the pool was faster
    down to the smallest size, or slower up to the largest. ``works`` holds each measured size's
    largest pass work (``d * (d - 1)``, the first pass's, where it is missing). ``cutoff`` is the
    work to set, for the ``method`` it was measured on. ``start_s``, where measured, is the seconds
    a pooled run over the smallest grid size takes, almost all of it starting and closing the pool.
    """

    below: int | None
    above: int | None
    timings: tuple[tuple[int, float, float], ...]
    works: tuple[tuple[int, int], ...] = ()
    method: str = "ges"
    start_s: float | None = None

    def work(self, d: int) -> int:
        """The largest pass work measured at size ``d``."""
        return dict(self.works).get(d, d * (d - 1))

    @property
    def cutoff(self) -> int:
        if self.above is None:  # never faster: keep the pool off up to the largest size measured
            slowest = self.work(self.below if self.below is not None else _MAX_D)
            return max(slowest + METHODS[self.method].at_cutoff, default_cutoff(self.method))
        if self.below is None:  # faster everywhere measured: use the pool from the smallest size
            return self.work(self.above)
        return _round2(math.sqrt(self.work(self.below) * self.work(self.above)))


def default_cutoff(method: str) -> int:
    """The built-in cutoff of ``method``."""
    if method == "skeleton":
        from andrey.constraint._parallel_skeleton import MIN_POOL_QUERIES

        return MIN_POOL_QUERIES
    if method == "rcot":
        from andrey.constraint._parallel_skeleton import MIN_POOL_RCOT_WORK

        return MIN_POOL_RCOT_WORK
    from andrey.search._parallel_ges import _DEFAULT_MIN_WORK

    return _DEFAULT_MIN_WORK


def cutoff_is_default(method: str) -> bool:
    """Whether ``method``'s cutoff is its built-in default: the variable is unset or blank."""
    from andrey.core import env

    return env.SETTINGS[METHODS[method].variable].read() is None


def warn_default_cutoff(method: str, default: int) -> None:
    """Warn once that a ``method`` fit with more than one worker runs on the default cutoff."""
    from andrey.core.warning_policy import PerformanceWarning, warn_once

    entry = METHODS[method]
    warn_once(
        f"{entry.search} decides when to use its worker pool with the default "
        f"{entry.variable}={default}, which suits one fit per process. If this process runs many "
        f"fits, `{COMMAND}` measures a value for this machine; any explicit value silences this "
        "warning.",
        PerformanceWarning,
    )


def bracket(
    serial: Callable[[int], float],
    pooled: Callable[[int], float],
    grid: Sequence[int] = GRID,
    *,
    refine: int = _REFINE_STEPS,
    work: Callable[[int], int] = lambda d: d * (d - 1),
    method: str = "ges",
) -> Crossover:
    """Find the sizes around the serial/pooled crossover from timing functions of ``d``.

    ``work(d)``, called after ``pooled(d)``, gives the largest pass work of the pooled run at ``d``.
    ``method`` names the cutoff the result is for.

    Scans ``grid`` upward for the first size where the pooled pass is faster. If that is the first
    grid size, halves ``d`` until the serial pass is faster (or ``d`` reaches 4); if no grid size
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
    return Crossover(below, above, rows, tuple((d, work(d)) for d in sorted(timings)), method)


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


def sample(d: int, seed: int = 0, rows: int | None = None) -> np.ndarray:
    """A linear-Gaussian SEM over ``d`` variables with average degree 2 and ``rows`` rows,
    ``10 * d`` when not given."""
    rng = np.random.default_rng(seed)
    edges = np.tril(rng.random((d, d)) < 2.0 / max(d - 1, 1), -1)
    weights = rng.uniform(0.4, 1.2, (d, d)) * rng.choice([-1.0, 1.0], (d, d)) * edges
    x = rng.standard_normal((10 * d if rows is None else rows, d))
    for j in range(d):
        x[:, j] += x @ weights[j]
    return x


@contextmanager
def _cutoff(var: str, value: str | None) -> Generator[None]:
    """Set the cutoff variable ``var`` to ``value`` (unset it for ``None``), then restore it."""
    old = os.environ.pop(var, None)
    try:
        if value is not None:
            os.environ[var] = value
        yield
    finally:
        os.environ.pop(var, None)
        if old is not None:
            os.environ[var] = old


def fit_timers(
    method: str, workers: int, repeats: int = 3
) -> tuple[Callable[[int], float], Callable[[int], float], Callable[[int], int]]:
    """Serial and pooled timing functions for a whole ``method`` fit on :func:`sample` data.

    Each time is the median of ``repeats`` fits after one warm-up fit, or a single fit once the
    warm-up takes more than a second. The pooled arm runs ``workers`` processes and sets the
    method's cutoff to 0, so every pass runs on the pool and the time includes starting it; the
    process's first pool start is paid before timing. The third function gives the largest pass
    work of the last pooled fit at ``d``.
    """
    import andrey
    from andrey.search import _parallel_ges

    fit = getattr(andrey, method)
    var = METHODS[method].variable
    works: dict[int, int] = {}

    def run(d: int, n_workers: int, cutoff: str | None) -> float:
        x = sample(d)
        with _cutoff(var, cutoff), andrey.config(num_workers=n_workers):
            seconds = _time(lambda: fit(x), repeats)
        if n_workers > 1:
            works[d] = _parallel_ges.largest_pass_work
        return seconds

    return (lambda d: run(d, 1, None)), (lambda d: run(d, workers, "0")), works.__getitem__


# Neighbors per variable in the skeleton probe at each conditioning size, which gives a pass one to
# two thousand tests per pair, so the largest probe holds a few million tests.
SKELETON_SIZES = {2: 40, 3: 20}
_ALPHA = 0.05


@dataclass(frozen=True)
class SkeletonPass:
    """One conditioning-size pass of the skeleton search: a test and the snapshot it scans.

    ``tests`` is the pass's CI-test count, the work the cutoff gate counts.
    """

    test: CITest
    adjacency: np.ndarray
    neighbors: list[np.ndarray]
    pairs: list[tuple[int, int]]
    size: int
    tests: int


def skeleton_pass(d: int, size: int) -> SkeletonPass:
    """A pass at conditioning ``size`` over ``d`` variables of :func:`sample` data.

    Each variable is adjacent to its ``SKELETON_SIZES[size]`` nearest neighbors on a ring (all the
    others when fewer), the start-of-size snapshot of a dense skeleton.
    """
    from andrey.constraint._skeleton import adjacent_pairs, pair_queries
    from andrey.core.ci import make_indep_test

    ring = np.abs(np.arange(d)[:, None] - np.arange(d)[None, :])
    gap = np.minimum(ring, d - ring)
    adjacency = (gap > 0) & (gap <= max(1, SKELETON_SIZES[size] // 2))
    neighbors = [np.flatnonzero(adjacency[i]) for i in range(d)]
    pairs = adjacent_pairs(adjacency)
    tests = sum(pair_queries(neighbors, x, y, size) for x, y in pairs)
    test = make_indep_test("fisherz", sample(d))
    return SkeletonPass(test, adjacency, neighbors, pairs, size, tests)


def skeleton_timers(
    workers: int, size: int, repeats: int = 3
) -> tuple[Callable[[int], float], Callable[[int], float], Callable[[int], int]]:
    """Serial and pooled timing functions for one skeleton pass at conditioning ``size``.

    The pass is :func:`skeleton_pass` over ``d`` variables. The serial arm scans every pair in this
    process as a fit does. The pooled arm starts a pool of ``workers`` processes with the cutoff set
    to 0 and scans the same pass on it, so its time includes starting and closing the pool. Times
    are as in :func:`fit_timers`. The third function gives the pass's tests, the work the gate
    counts.
    """
    from andrey.constraint._parallel_skeleton import SkeletonWorkerPool
    from andrey.constraint._skeleton import scan_pairs
    from andrey.core import backend
    from andrey.core.ci import pins_blas

    passes: dict[int, SkeletonPass] = {}

    def make(d: int) -> SkeletonPass:
        if d not in passes:
            passes[d] = skeleton_pass(d, size)
        return passes[d]

    def serial(d: int) -> float:
        p = make(d)

        def scan() -> None:
            with backend.one_blas_thread() if pins_blas(p.test) else nullcontext():
                scan_pairs(p.test, p.pairs, p.neighbors, size, _ALPHA)

        return _time(scan, repeats)

    def pooled(d: int) -> float:
        p = make(d)

        def scan() -> None:
            pool = SkeletonWorkerPool(p.test, workers)
            try:
                pool.scan(p.pairs, p.neighbors, p.adjacency, size, _ALPHA)
            finally:
                pool.close()

        with _cutoff(METHODS["skeleton"].variable, "0"):
            return _time(scan, repeats)

    return serial, pooled, lambda d: make(d).tests


# The RCoT probe's rows, and the neighbors per variable of its pass at conditioning size 2, which
# gives a pair about 20 tests, so the probe grid's passes hold one to eight thousand tests.
RCOT_ROWS = 2000
RCOT_NEIGHBORS = 6


def rcot_pass(d: int) -> SkeletonPass:
    """A pass at conditioning size 2 over ``d`` variables of :func:`sample` data with
    :data:`RCOT_ROWS` rows, each variable adjacent to its :data:`RCOT_NEIGHBORS` nearest neighbors
    on a ring, with an RCoT test. Its ``tests`` holds the pass's RCoT tests times the rows, the work
    the cutoff gate counts."""
    from andrey.constraint._skeleton import adjacent_pairs, pair_queries
    from andrey.core.ci_kernel import RCoT

    ring = np.abs(np.arange(d)[:, None] - np.arange(d)[None, :])
    gap = np.minimum(ring, d - ring)
    adjacency = (gap > 0) & (gap <= RCOT_NEIGHBORS // 2)
    neighbors = [np.flatnonzero(adjacency[i]) for i in range(d)]
    pairs = adjacent_pairs(adjacency)
    tests = sum(pair_queries(neighbors, x, y, 2) for x, y in pairs)
    test = RCoT(sample(d, rows=RCOT_ROWS))
    return SkeletonPass(test, adjacency, neighbors, pairs, 2, tests * RCOT_ROWS)


def rcot_timers(
    workers: int, repeats: int = 3
) -> tuple[Callable[[int], float], Callable[[int], float], Callable[[int], int]]:
    """Serial and pooled timing functions for one RCoT skeleton pass (:func:`rcot_pass`).

    Each timed scan takes a copy of the pass's test with empty caches, so no scan reuses another's
    features. The serial arm scans in this process on one BLAS thread, as a fit does; the pooled
    arm starts a pool of ``workers`` processes with the cutoff set to 0, so its time includes
    starting and closing the pool. Times are as in :func:`fit_timers`; the third function gives the
    pass's work.
    """
    from andrey.constraint._parallel_skeleton import RCoTWorkerPool
    from andrey.constraint._skeleton import scan_pairs
    from andrey.core import backend
    from andrey.core.ci_kernel import RCoT

    passes: dict[int, SkeletonPass] = {}

    def make(d: int) -> SkeletonPass:
        if d not in passes:
            passes[d] = rcot_pass(d)
        return passes[d]

    def fresh(p: SkeletonPass) -> RCoT:
        return RCoT._from_state(*p.test._state())  # ty: ignore[unresolved-attribute]  # an RCoT

    def serial(d: int) -> float:
        p = make(d)

        def scan() -> None:
            with backend.one_blas_thread():
                scan_pairs(fresh(p), p.pairs, p.neighbors, 2, _ALPHA)

        return _time(scan, repeats)

    def pooled(d: int) -> float:
        p = make(d)

        def scan() -> None:
            pool = RCoTWorkerPool(fresh(p), workers)
            try:
                pool.scan(p.pairs, p.neighbors, p.adjacency, 2, _ALPHA)
            finally:
                pool.close()

        with _cutoff(METHODS["rcot"].variable, "0"):
            return _time(scan, repeats)

    return serial, pooled, lambda d: make(d).tests


def calibrate(workers: int, methods: Sequence[str] = tuple(METHODS)) -> dict[str, Crossover]:
    """Measure each method's crossover with ``workers`` pool processes."""
    import dataclasses
    import warnings

    import andrey

    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", andrey.AndreyWarning)
        for m in methods:
            if m == "skeleton":
                timers = [skeleton_timers(workers, size) for size in SKELETON_SIZES]
            elif m == "rcot":
                timers = [rcot_timers(workers)]
            else:
                timers = [fit_timers(m, workers)]
            # A pooled run with next to no work times starting and closing the pool; its warm-up
            # call starts the process's first pool (the forkserver) before anything else is timed.
            start = timers[0][1](GRID[0])
            runs = [
                bracket(serial, pooled, GRID, work=work, method=m)
                for serial, pooled, work in timers
            ]
            widest = max(runs, key=lambda c: c.cutoff)
            out[m] = dataclasses.replace(widest, start_s=start)
    return out
