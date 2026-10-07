"""Test BIC penalty propagation, validation, and serial-parallel parity."""

from __future__ import annotations

import numpy as np
import pytest

import andrey
from andrey.core import backend
from andrey.core.score import BICScore
from andrey.search import _parallel_ges as pg
from andrey.search import _parallel_hc as ph
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


def _cpdag(result):
    if isinstance(result, tuple):
        return result[0]
    return getattr(result, "structure", result)


@pytest.mark.parametrize(
    "name,run",
    [
        ("ges", ges),
        ("hc", hc),
        ("boss", boss),
        ("grasp", grasp),
        ("gies", gies),
        ("gfci", gfci),
        ("andrey.ges", andrey.ges),
        ("andrey.hc", andrey.hc),
        ("andrey.boss", andrey.boss),
        ("andrey.grasp", andrey.grasp),
        ("andrey.gies", andrey.gies),
        ("andrey.gfci", andrey.gfci),
    ],
)
def test_penalty_moves_the_answer(name, run):
    heavy = _marks(_cpdag(run(_SENSITIVE, lambda_value=_HEAVY)))
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


_PARALLEL_GRID = (0.5, 1.0, 16.0)
_GES_PARALLEL_SEM = _gaussian_sem(0, 600, 12, 0.3)
_HC_PARALLEL_SEM = _gaussian_sem(0, 600, 16, 0.2)


def _spy_pool(monkeypatch, module) -> list[int]:
    created: list[int] = []
    real = module._create_pool

    def counting(*args, **kwargs):
        created.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(module, "_create_pool", counting)
    return created


@pytest.mark.parametrize(
    "run,module,env,data,workers",
    [
        (ges, pg, "ANDREY_GES_PARALLEL_MIN_WORK", _GES_PARALLEL_SEM, 3),
        (hc, ph, "ANDREY_HC_PARALLEL_MIN_WORK", _HC_PARALLEL_SEM, 4),
    ],
    ids=["ges", "hc"],
)
@pytest.mark.parametrize("lam", _PARALLEL_GRID)
def test_parallel_matches_serial_at_every_penalty(
    monkeypatch, run, module, env, data, workers, lam
):
    monkeypatch.setenv(env, "0")
    created = _spy_pool(monkeypatch, module)
    serial_structure, serial_score = run(data, lambda_value=lam)
    with backend.config(num_workers=workers):
        parallel_structure, parallel_score = run(data, lambda_value=lam)
    assert sum(created) == 1, "the pool never opened -- the comparison would be serial vs serial"
    assert np.array_equal(serial_structure.to_numpy(), parallel_structure.to_numpy())
    assert serial_score == parallel_score


@pytest.mark.parametrize(
    "name,run,module,env,data,workers",
    [
        ("ges", ges, pg, "ANDREY_GES_PARALLEL_MIN_WORK", _GES_PARALLEL_SEM, 3),
        ("hc", hc, ph, "ANDREY_HC_PARALLEL_MIN_WORK", _HC_PARALLEL_SEM, 4),
    ],
)
def test_parallel_arm_tracks_the_penalty(monkeypatch, name, run, module, env, data, workers):
    monkeypatch.setenv(env, "0")
    created = _spy_pool(monkeypatch, module)
    with backend.config(num_workers=workers):
        marks = {lam: _marks(run(data, lambda_value=lam)[0]) for lam in _PARALLEL_GRID}
    assert sum(created) == len(_PARALLEL_GRID), "a run stayed in-process instead of fanning out"
    assert len(set(marks.values())) == len(_PARALLEL_GRID), f"{name}: pooled answers {marks}"
