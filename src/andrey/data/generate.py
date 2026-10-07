"""Bulk, reproducible, parallel generation of dataset ensembles.

Each :class:`Ensemble` task samples an SCM with a seed. ``SCM.sample`` derives its RNG streams
from that seed without global state, so a task regenerates from its config and seed. Results are
byte-identical across worker counts and scheduling orders.
"""

from __future__ import annotations

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
    ensemble: Ensemble, *, workers: int = 1, dtype: npt.DTypeLike = np.float64
) -> list[CausalDataset]:
    """Sample every scenario/seed pair in ``ensemble``.

    Parameters
    ----------
    ensemble : Ensemble
        The scenario/seed grid.
    workers : int, default=1
        Worker processes; affects runtime only, preserving byte-identical results.
    dtype : npt.DTypeLike, default=np.float64
        Output dtype of every dataset's ``data``; forwarded to :meth:`SCM.sample`.

    Returns
    -------
    list[CausalDataset]
        One dataset per task, in ``ensemble.tasks`` order.
    """
    tasks = [(scm, ensemble.n, seed, dtype) for scm, seed in ensemble.tasks]
    if workers <= 1:
        return [_run(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_run, tasks))


def _run(task: tuple[SCM, int, int, npt.DTypeLike]) -> CausalDataset:
    scm, n, seed, dtype = task
    return scm.sample(n=n, seed=seed, dtype=dtype)
