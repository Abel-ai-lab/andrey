"""Workers, partitioning, and deterministic reduction for parallel greedy search.

Each pass of a greedy search needs the previous graph, so passes run in order and the candidates
within a pass are scanned in parallel. :class:`StableWorkerSet` keeps each target on the same
worker across passes, so its score caches survive. :func:`even_chunks` and
:func:`balanced_chunks` split the work, :func:`reduce_best` picks the best move by a total-order
key, :func:`collect` raises worker failures, :func:`pick_context` chooses the start method, and
:func:`min_work` reads the crossover gate. Each learner supplies a picklable module-level worker
initializer and task function.

GES and HC share :data:`_EPS_IMPROVE`, :data:`_KEY_DP`, and :func:`stable_key`, so the chosen
move does not depend on rounding in the Schur delta-BIC. The final objective is recomputed from
the CPDAG.
"""

from __future__ import annotations

import multiprocessing as mp
import sys
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from typing import Any

import numpy as np

from andrey.core import env


def even_chunks(n: int, n_chunks: int) -> list[list[int]]:
    """Partition the unit axis ``[0, n)`` into up to ``n_chunks`` contiguous ranges (as lists)."""
    parts = np.array_split(np.arange(n, dtype=int), max(1, min(n_chunks, n)))
    return [part.tolist() for part in parts if len(part)]


def balanced_chunks(work_per_unit: Sequence[int], n_chunks: int) -> list[list[int]]:
    """Split units into up to ``n_chunks`` contiguous ranges of roughly equal estimated work.

    Split where cumulative work first exceeds each ``k / n_chunks`` share. The ranges are disjoint
    and cover every unit. Balancing work reduces idle time without changing the global-key result.
    """
    n = len(work_per_unit)
    k = max(1, min(n_chunks, n))
    cum = np.cumsum(np.asarray(work_per_unit, dtype=np.float64))
    if n == 0 or cum[-1] <= 0:
        # No work signal (for example, an empty graph): fall back to even ranges.
        return even_chunks(n, k)
    bounds = cum[-1] * np.arange(1, k) / k
    cuts = np.searchsorted(cum, bounds, side="right")
    edges = sorted({0, n, *(int(c) for c in cuts)})
    return [list(range(a, b)) for a, b in zip(edges, edges[1:]) if a < b]


# What a worker hands back for its chunk: a tuple whose leading entries form the total-order key
# (GES and HC both lead with ``(delta, x, y, index)``), or ``None`` when the chunk found no move.
ChunkBest = tuple[Any, ...]


def reduce_best(
    results: Sequence[ChunkBest | None], key: Callable[[ChunkBest], Any] | None = None
) -> ChunkBest | None:
    """Reduce chunk bests to the one global best by a strict total-order ``key`` (``None`` skipped).

    The default key ``r[:4]`` matches the ``(delta, x, y, index)`` prefix both GES and HC emit; a
    learner with a different arity passes its own. Strict ``<`` so the first chunk to reach a key
    wins ties, exactly as a single serial scan resolves them.
    """
    if key is None:
        key = lambda r: r[:4]  # noqa: E731 -- the shared (delta, src, tgt, op-index) prefix
    best: ChunkBest | None = None
    best_key: Any = None
    for r in results:
        if r is None:
            continue
        rk = key(r)
        if best is None or rk < best_key:
            best, best_key = r, rk
    return best


def collect(futures: list) -> list:
    """Collect every chunk result, propagating task exceptions.

    A killed worker (``BrokenProcessPool``) raises ``RuntimeError``. Dropping a failed chunk would
    change the argmin.
    """
    results = []
    try:
        for fut in as_completed(futures):
            results.append(fut.result())
    except BrokenProcessPool as exc:
        raise RuntimeError(
            "a parallel-search worker died mid-pass (e.g. OOM); rerun with ANDREY_NUM_WORKERS=1"
        ) from exc
    return results


def fork_safe() -> bool:
    """True when forking the current process is safe: torch unimported, or its CUDA uninitialized.

    Never imports torch -- it only inspects ``sys.modules``. Forking a process with a live CUDA
    context corrupts the child's, so the parent is forked only when no such context exists.
    """
    t = sys.modules.get("torch")
    if t is None:
        return True
    cuda = getattr(t, "cuda", None)
    return not (cuda is not None and cuda.is_initialized())


# Modules the forkserver template imports once, so every forked worker inherits them already loaded.
# A worker's first act is unpickling its initializer and task functions by module name, so without
# the preload each worker re-imports the whole chain (numpy, scipy, andrey.core, ...) from the
# filesystem -- slow on a network filesystem, serialized (each single-worker pool spawns inside its
# own first ``submit``) and repaid per search (the pools do not outlive a run). Every module that
# defines a worker initializer/task must be listed: the template is process-wide, so a worker only
# inherits what the template imported. The three below are the search workers (``_parallel_ges`` /
# ``_parallel_hc``) and the constraint-skeleton worker (``_parallel_skeleton``); score/backend/
# operators/CI/threadpoolctl come in transitively. Names must match the tree exactly: the forkserver
# imports the list in a ``try/except`` and *silently skips* a bad name, so a typo silently restores
# the tax -- test_forkserver_template_preloads_worker_modules resolves each name loudly.
_FORKSERVER_PRELOAD = (
    "andrey.search._parallel_ges",
    "andrey.search._parallel_hc",
    "andrey.constraint._parallel_skeleton",
)


def pick_context() -> mp.context.BaseContext:
    """Prefer ``forkserver``, then a safe ``fork``, then ``spawn``.

    The forkserver starts workers from a clean, single-threaded process, avoiding inherited CUDA
    state, threads, locks, and fork warnings from pool manager threads. ``fork`` is used only when
    forkserver is unavailable and :func:`fork_safe` permits it.

    Preload :data:`_FORKSERVER_PRELOAD` to share imports across workers. Keep ``__main__`` in the
    list: setting a preload replaces that stdlib default, and omitting it repeats parent-module
    imports in each child. Register the list before the forkserver starts; later calls have no
    effect. All Andrey pools use this function. Registration replaces another library's preload
    because the stdlib has no append operation.
    """
    available = mp.get_all_start_methods()
    if "forkserver" in available:
        ctx = mp.get_context("forkserver")
        ctx.set_forkserver_preload(["__main__", *_FORKSERVER_PRELOAD])
        return ctx
    if fork_safe() and "fork" in available:
        return mp.get_context("fork")
    return mp.get_context("spawn")


def min_work(setting: env.Setting[int], default: int) -> int:
    """The crossover gate in estimated candidate evals: ``setting`` override, else ``default``."""
    value = setting.read()
    return default if value is None else value


def _boot_task() -> None:
    """The no-op each fresh pool runs at construction, forcing its worker process into existence."""


class StableWorkerSet:
    """Single-process pools with fixed unit partitions for the whole search.

    Worker ``i`` always owns partition ``i``, retaining its per-unit score cache across passes.
    A global total-order reduction keeps the selected move independent of partitioning. Each worker
    has its own pool so :func:`collect` can detect its death through ``BrokenProcessPool``.
    The context selects the start method; the initializer applies any BLAS pinning.

    Workers start concurrently during construction. The first ``ProcessPoolExecutor.submit`` can
    block while writing pickled ``initargs``: a covariance of hundreds of KiB exceeds the 64 KiB
    pipe buffer. One boot thread per pool overlaps those writes and worker startup, costing roughly
    the slowest boot instead of their sum. Startup failures surface in the constructor.
    """

    def __init__(
        self,
        n_units: int,
        workers: int,
        initializer: Callable[..., None],
        initargs: tuple,
        context: mp.context.BaseContext | None = None,
    ) -> None:
        k = max(1, min(workers, n_units))
        parts = np.array_split(np.arange(n_units, dtype=int), k)
        self.parts = [part.tolist() for part in parts if len(part)]  # fixed, contiguous, disjoint
        ctx = context if context is not None else pick_context()
        self._pools = [
            ProcessPoolExecutor(
                max_workers=1,
                mp_context=ctx,
                initializer=initializer,
                initargs=initargs,
            )
            for _ in self.parts
        ]
        try:
            # Overlap the first submit's blocking initialization writes across pools.
            with ThreadPoolExecutor(max_workers=len(self._pools)) as boot:
                list(boot.map(lambda p: p.submit(_boot_task).result(), self._pools))
        # A failed start surfaces as any of these, by start method and initargs size.
        except (BrokenProcessPool, EOFError, ConnectionError) as exc:
            self.close()
            raise RuntimeError(
                "parallel-search workers failed to start. If a script calls this, guard its entry "
                'point with `if __name__ == "__main__":` (starting workers re-imports the script). '
                "Otherwise the likely cause is running out of memory. Set ANDREY_NUM_WORKERS=1 to "
                "run without workers."
            ) from exc
        except BaseException:
            self.close()  # never leak live worker processes from a failed constructor
            raise

    def dispatch(self, task_fn: Callable, *snapshot) -> list:
        """Send each worker ``task_fn(its_partition, *snapshot)``; collect the bests (fail-loud)."""
        futures = [
            pool.submit(task_fn, part, *snapshot) for pool, part in zip(self._pools, self.parts)
        ]
        return collect(futures)

    def close(self) -> None:
        for pool in self._pools:
            # cancel_futures so a mid-pass failure does not wait out already-queued chunk tasks.
            pool.shutdown(wait=True, cancel_futures=True)
        self._pools = []


# ---- determinism knobs shared by the GES and HC delta-search parallelizations -------------------
#
# The Schur delta-BIC is not bit-pure across evaluation paths (its float64 arithmetic depends on the
# einsum batch shape / the scorer's residual-cache history), so GES and HC must agree on two knobs
# for serial and parallel to select byte-identical moves: a strict-improvement floor, and a
# scale-invariant reduce tie-break key. Both live here so the two learners share one definition.

# Strict-improvement floor: a move counts as improving only when its raw delta is more than this
# below zero. It sits far above float64 Schur-delta noise and below the smallest real score
# improvement, so the accepted-move set is deterministic. Shared by hc.py and the GES chunk scans.
_EPS_IMPROVE = 1e-9

# Decimals kept when quantizing ``delta / n`` for the reduce tie-break key, where ``n`` is the
# caller's scale factor. The BIC delta grows with the score's sample count (``delta = n_samples *
# (log R_new - log R_old) +/- ...``), so its cross-path ULP noise does too; HC passes that sample
# count, so ``delta / n`` tracks the O(1) signal and a fixed ``_KEY_DP`` grid neither collapses
# distinct moves nor breaks ties badly across scales. GES passes the variable count instead: its
# per-target batch grouping already makes each delta bit-identical serial-vs-parallel, so the key is
# only a deterministic-remainder tie-break and never leans on the grid to absorb cross-path noise.
_KEY_DP = 9


def stable_key(delta: float, n: int) -> float:
    """Scale-invariant, chunk-invariant reduce key component: ``round(delta / n, _KEY_DP)``.

    ``n`` is the caller's scale factor (see :data:`_KEY_DP`): HC's sample count, GES's variable
    count. Two cross-path ULP-equal deltas quantize to one bucket, so the reduce breaks the
    resulting tie on the deterministic ``(x, y, index)`` remainder of the key rather than on
    unstable low-order bits. Serial and parallel therefore select byte-identical moves however the
    targets are chunked or the adds are batched into the einsum.
    """
    return round(delta / n, _KEY_DP)
