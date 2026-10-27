"""Bulk, reproducible, parallel generation of dataset ensembles.

Each :class:`Ensemble` task samples an SCM with a seed. ``SCM.sample`` derives its RNG streams
from that seed without global state, so a task regenerates from its config and seed. Results are
byte-identical across worker counts and scheduling orders.
"""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

from .dataset import CausalDataset
from .scm import SCM


@dataclass(frozen=True)
class Ensemble:
    """A grid of ``(SCM, seed)`` generation tasks sharing a sample count.

    Parameters
    ----------
    scms : iterable of SCM
        The scenarios.
    seeds : iterable of int
        Seeds crossed with every scenario.
    n : int
        Samples per dataset.
    """

    scms: tuple[SCM, ...]
    seeds: tuple[int, ...]
    n: int

    def __post_init__(self) -> None:
        """Coerce ``scms``, ``seeds``, and ``n`` to canonical immutable types after init."""
        object.__setattr__(self, "scms", tuple(self.scms))
        object.__setattr__(self, "seeds", tuple(int(s) for s in self.seeds))
        object.__setattr__(self, "n", int(self.n))

    @property
    def tasks(self) -> list[tuple[SCM, int]]:
        """The ``(scm, seed)`` tasks in a fixed order (scenario-major)."""
        return [(scm, seed) for scm in self.scms for seed in self.seeds]


def generate(
    ensemble: Ensemble,
    *,
    workers: int = 1,
    dtype: npt.DTypeLike = np.float64,
    scale: str = "standardize",
) -> list[CausalDataset]:
    """Sample every scenario/seed pair in ``ensemble``.

    Parameters
    ----------
    ensemble : Ensemble
        The scenario/seed grid.
    workers : int, default=1
        Worker processes; affects runtime only, preserving byte-identical results. They start with
        the interpreter's default start method. When that is ``forkserver`` (Linux from Python
        3.14), they start as the search workers do: if the temporary directory cannot hold the
        forkserver's socket, it goes in ``/dev/shm`` or ``/tmp``, else the workers start with
        ``fork`` or ``spawn``, with a warning.
    dtype : npt.DTypeLike, default=np.float64
        Output dtype of every dataset's ``data``; forwarded to :meth:`SCM.sample`.
    scale : {"standardize", "unitless", "rescale", "raw"}, default="standardize"
        How every dataset's values are scaled; forwarded to :meth:`SCM.sample`, which describes
        each mode.

    Returns
    -------
    list[CausalDataset]
        One dataset per task, in ``ensemble.tasks`` order.
    """
    tasks = [(scm, ensemble.n, seed, dtype, scale) for scm, seed in ensemble.tasks]
    if workers <= 1:
        return [_run(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=workers, mp_context=_context()) as pool:
        return list(pool.map(_run, tasks))


def _context() -> mp.context.BaseContext:
    """Return the default context, or ``pick_context()``'s choice when the default is forkserver."""
    ctx = mp.get_context()
    if ctx.get_start_method() != "forkserver":
        return ctx
    from andrey.search._parallel import pick_context

    return pick_context()


def _run(task: tuple[SCM, int, int, npt.DTypeLike, str]) -> CausalDataset:
    scm, n, seed, dtype, scale = task
    return scm.sample(n=n, seed=seed, dtype=dtype, scale=scale)
