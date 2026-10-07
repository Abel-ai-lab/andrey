"""Compare delta-BIC and full-score GES CPDAGs on well- and ill-conditioned data.

``andrey.ges`` updates local scores with a Schur step. The comparison allows two edges of total
structural difference across the cases because near-ties can resolve differently across CPUs.
"""

from __future__ import annotations

import numpy as np

import andrey
from andrey import data, metrics
from andrey.core import GraphStructure, stats
from andrey.search.ges import ges as _ges_internal

# A near-tie can resolve differently on another CPU; more than this many edges is a real divergence.
_MAX_EDGES_APART = 2


def _delta_ges_marks(x: np.ndarray) -> np.ndarray:
    return andrey.ges(x).structure.to_numpy()


def _full_ges_marks(x: np.ndarray) -> np.ndarray:
    return _ges_internal(x, _full_scores=True)[0].to_numpy()


def sample_dag(d: int, n: int, density: float, seed: int) -> data.CausalDataset:
    """Sample a standardized linear-Gaussian DAG at the requested edge density."""
    scm = data.SCM(
        graph=data.graphs.erdos_renyi(d, avg_degree=density * (d - 1)),
        functional=data.functional.linear(weight_range=(0.5, 1.5)),
        noise=data.noise.gaussian(),
    )
    return scm.sample(n=n, seed=seed, scale="standardize")


def _cond(x: np.ndarray) -> float:
    return float(np.linalg.cond(stats.cov(x, rowvar=False)))


def _well_conditioned_cases() -> list[np.ndarray]:
    return [sample_dag(30, 240, 0.05, 1000 + seed).data for seed in range(10)]


def _ill_conditioned_cases() -> list[tuple[np.ndarray, float]]:
    """Return data and covariance condition numbers for near-singular cases."""
    cases: list[tuple[np.ndarray, float]] = []
    # (1) near-collinear: vars 8 and 9 are near-copies of 0 and 1 (unit coefficient + tiny noise).
    for seed, eps in enumerate((3e-3, 1e-3, 3e-4, 3e-3)):
        rng = np.random.default_rng(5000 + seed)
        d, n = 10, 4000
        b = np.zeros((d, d))
        scale = np.ones(d)
        for i in range(6):  # a sparse well-conditioned base among the first six variables
            for j in range(i + 1, 6):
                if rng.random() < 0.3:
                    b[j, i] = rng.uniform(0.5, 1.0) * rng.choice([-1.0, 1.0])
        b[8, 0] = 1.0
        b[9, 1] = 1.0
        scale[8] = scale[9] = eps  # tiny innovation -> near-collinear -> cond ~ 1 / eps^2
        e = rng.standard_normal((n, d)) * scale
        x = np.zeros((n, d))
        for j in range(d):
            x[:, j] = e[:, j] + x @ b[j]
        cases.append((x, _cond(x)))
    # (2) n ~= d: sample count barely above the variable count -> near-singular covariance.
    for seed in range(3):
        d = 40
        ds = sample_dag(d, d + 6, 0.1, 6000 + seed)
        cases.append((ds.data, _cond(ds.data)))
    return cases


def test_ill_conditioned_cases_are_ill_conditioned():
    assert max(c for _, c in _ill_conditioned_cases()) >= 1e6


def test_delta_ges_matches_full_score_ges():
    cases = _well_conditioned_cases() + [x for x, _ in _ill_conditioned_cases()]
    apart = 0.0
    for x in cases:
        delta = GraphStructure.from_numpy(_delta_ges_marks(x), kind="cpdag")
        full = GraphStructure.from_numpy(_full_ges_marks(x), kind="cpdag")
        apart += metrics.shd(delta, full)
    assert apart <= _MAX_EDGES_APART, f"delta and full GES differ by {apart:.0f} edges in total"
