"""Test BIC penalty propagation, validation, and serial-parallel parity."""

from __future__ import annotations

import numpy as np
import pytest

import andrey
from andrey.core import backend
from andrey.core.score import BICScore
from andrey.search import _parallel_ges as pg
from andrey.search.boss import boss
from andrey.search.ges import ges
from andrey.search.gfci import gfci
from andrey.search.gies import gies
from andrey.search.grasp import grasp
from andrey.search.hc import hc


def _gaussian_sem(seed: int, n: int, d: int, p: float) -> np.ndarray:
    """A random strictly-lower-triangular linear-Gaussian SEM (edge density ``p``)."""
    rng = np.random.default_rng(seed)
    b = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            if rng.random() < p:
                b[j, i] = rng.uniform(0.4, 1.2) * rng.choice([-1.0, 1.0])
    e = rng.standard_normal((n, d))
    x = np.zeros((n, d))
    for j in range(d):
        x[:, j] = e[:, j] + x @ b[j]
    return x


def _marks(structure) -> int:
    return int(np.count_nonzero(structure.to_numpy()))


_SENSITIVE = _gaussian_sem(1, 400, 8, 0.4)
_LIGHT, _HEAVY = 0.5, 16.0
# GFCI's tests remove the edges a light penalty lets GES add, so its answer moves only once the
# penalty drops a true edge.
_GFCI_HEAVY = 32.0


def _cpdag(result):
    if isinstance(result, tuple):
        return result[0]
    return getattr(result, "structure", result)


@pytest.mark.parametrize(
    "name,run,weight",
    [
        ("ges", ges, _HEAVY),
        ("hc", hc, _HEAVY),
        ("boss", boss, _HEAVY),
        ("grasp", grasp, _HEAVY),
        ("gies", gies, _HEAVY),
        ("gfci", gfci, _GFCI_HEAVY),
        ("andrey.ges", andrey.ges, _HEAVY),
        ("andrey.hc", andrey.hc, _HEAVY),
        ("andrey.boss", andrey.boss, _HEAVY),
        ("andrey.grasp", andrey.grasp, _HEAVY),
        ("andrey.gies", andrey.gies, _HEAVY),
        ("andrey.gfci", andrey.gfci, _GFCI_HEAVY),
    ],
)
def test_penalty_moves_the_answer(name, run, weight):
    heavy = _marks(_cpdag(run(_SENSITIVE, lambda_value=weight)))
    light = _marks(_cpdag(run(_SENSITIVE, lambda_value=_LIGHT)))
    assert heavy < light, f"{name}: lambda_value did not change the graph ({light} -> {heavy})"


def _from_cov(data, *, lambda_value):
    return BICScore.from_cov(np.cov(data, rowvar=False), len(data), lambda_value=lambda_value)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), -1.0])
def test_a_weight_that_is_not_a_penalty_is_refused(bad):
    with pytest.raises(ValueError, match="lambda_value"):
        BICScore(_SENSITIVE, lambda_value=bad)


@pytest.mark.parametrize(
    "name,run",
    [
        ("from_cov", _from_cov),
        ("ges", ges),
        ("hc", hc),
        ("boss", boss),
        ("grasp", grasp),
        ("gies", gies),
        ("gfci", gfci),
        ("andrey.ges", andrey.ges),
        ("andrey.hc", andrey.hc),
        ("andrey.boss", andrey.boss),
    ],
)
def test_the_guard_is_reached_from_every_entry_point(name, run):
    with pytest.raises(ValueError, match="lambda_value"):
        run(_SENSITIVE, lambda_value=-1.0)


@pytest.mark.parametrize(
    "name,run",
    [
        ("andrey.ges", andrey.ges),
        ("andrey.gies", andrey.gies),
        ("andrey.xges", andrey.xges),
        ("andrey.hc", andrey.hc),
        ("andrey.tabu", andrey.tabu),
        ("andrey.boss", andrey.boss),
        ("andrey.grasp", andrey.grasp),
        ("andrey.exact_search", andrey.exact_search),
        ("andrey.gfci", andrey.gfci),
        ("andrey.mmhc", andrey.mmhc),
        ("andrey.rsmax2", andrey.rsmax2),
    ],
)
def test_the_ebic_guard_is_reached_from_every_entry_point(name, run):
    with pytest.raises(ValueError, match="ebic_gamma"):
        run(_SENSITIVE, ebic_gamma=-1.0)


_PARALLEL_GRID = (0.5, 1.0, 16.0)
_GES_PARALLEL_SEM = _gaussian_sem(0, 600, 12, 0.3)
_GES_WORKERS = 3


def _spy_pool(monkeypatch) -> list[int]:
    created: list[int] = []
    real = pg._create_pool

    def counting(*args, **kwargs):
        created.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(pg, "_create_pool", counting)
    return created


@pytest.mark.parametrize("lam", _PARALLEL_GRID)
def test_parallel_matches_serial_at_every_penalty(monkeypatch, lam):
    monkeypatch.setenv("ANDREY_GES_PARALLEL_MIN_WORK", "0")
    created = _spy_pool(monkeypatch)
    serial_structure, serial_score = ges(_GES_PARALLEL_SEM, lambda_value=lam)
    with backend.config(num_workers=_GES_WORKERS):
        parallel_structure, parallel_score = ges(_GES_PARALLEL_SEM, lambda_value=lam)
    assert sum(created) == 1, "the pool never opened -- the comparison would be serial vs serial"
    assert np.array_equal(serial_structure.to_numpy(), parallel_structure.to_numpy())
    assert serial_score == parallel_score


def test_parallel_matches_serial_under_ebic(monkeypatch):
    monkeypatch.setenv("ANDREY_GES_PARALLEL_MIN_WORK", "0")
    created = _spy_pool(monkeypatch)
    x = _gaussian_sem(0, 200, 12, 0.3)  # few enough rows that the term changes the graph
    serial_structure, serial_score = ges(x, ebic_gamma=1.0)
    with backend.config(num_workers=_GES_WORKERS):
        parallel_structure, parallel_score = ges(x, ebic_gamma=1.0)
    assert sum(created) == 1, "the pool never opened -- the comparison would be serial vs serial"
    assert np.array_equal(serial_structure.to_numpy(), parallel_structure.to_numpy())
    assert serial_score == parallel_score
    plain_structure, _ = ges(x, ebic_gamma=0.0)
    assert not np.array_equal(plain_structure.to_numpy(), serial_structure.to_numpy())


def test_parallel_arm_tracks_the_penalty(monkeypatch):
    monkeypatch.setenv("ANDREY_GES_PARALLEL_MIN_WORK", "0")
    created = _spy_pool(monkeypatch)
    with backend.config(num_workers=_GES_WORKERS):
        marks = {lam: _marks(ges(_GES_PARALLEL_SEM, lambda_value=lam)[0]) for lam in _PARALLEL_GRID}
    assert sum(created) == len(_PARALLEL_GRID), "a run stayed in-process instead of fanning out"
    assert len(set(marks.values())) == len(_PARALLEL_GRID), f"ges: pooled answers {marks}"
