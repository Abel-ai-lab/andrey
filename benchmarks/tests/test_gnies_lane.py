"""The GnIES comparison's reading of the reference, and the score both packages compute.

The tests that run gnies 0.3.3 skip unless ``ANDREY_BENCH_GNIES_PYTHON`` names the interpreter
``benchmarks/start_gnies.sh`` builds. With it, Andrey's total at the standard BIC equals the
reference's ``GnIESScore`` run to convergence, converted to deviance units, on 20 draws of data,
DAG, and targets, and the reference's 0/1 matrix reads as the true class on a fixed chain.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
from gnies_cases import PYTHON_ENV_VAR, marks, reference

from andrey.core import ARROW, TAIL
from andrey.search.gnies import NoiseInterventionScore

needs_reference = pytest.mark.skipif(
    not os.environ.get(PYTHON_ENV_VAR), reason=f"{PYTHON_ENV_VAR} is not set"
)


def _draw(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Six variables, an observational and two other environments of 60-150 rows, each shifted in
    mean; every environment but the first changes the targets' noise variances."""
    rng = np.random.default_rng([6, seed])
    d = 6
    upper = np.triu(rng.random((d, d)) < 0.4, 1).astype(int)
    order = rng.permutation(d)
    dag = upper[np.ix_(order, order)]
    weights = dag * rng.uniform(0.5, 1.0, (d, d)) * rng.choice([-1, 1], (d, d))
    sizes = rng.integers(60, 151, 3)
    targets = sorted(rng.choice(d, size=int(rng.integers(0, 4)), replace=False).tolist())
    blocks, env = [], []
    for k, n in enumerate(sizes):
        sd = np.sqrt(rng.uniform(1, 2, d))
        if k > 0:
            sd[targets] = np.sqrt(rng.uniform(5, 10, len(targets)))
        noise = rng.standard_normal((n, d)) * sd
        blocks.append(noise @ np.linalg.inv(np.eye(d) - weights) + rng.normal(0, 3, d))
        env += [k] * int(n)
    return np.vstack(blocks), np.array(env), dag, targets


def test_marks_read_lone_entries_as_arrows_and_pairs_as_lines():
    icpdag = [[0, 1, 0], [0, 0, 1], [0, 1, 0]]
    m = marks(icpdag)
    assert (m[0, 1], m[1, 0]) == (TAIL, ARROW)
    assert (m[1, 2], m[2, 1]) == (TAIL, TAIL)
    assert m[0, 2] == m[2, 0] == 0


@needs_reference
@pytest.mark.parametrize("seed", range(20))
def test_total_equals_the_converged_reference_score(seed):
    X, env, dag, targets = _draw(seed)
    view = NoiseInterventionScore(X, env, ebic_gamma=0.0).view(targets)
    total = math.fsum(view.score(j, np.flatnonzero(dag[:, j])) for j in range(X.shape[1]))
    answer = reference(X, env, "score", A=dag, I=np.array(targets, dtype=int), iters=[1000])
    assert total == pytest.approx(answer["deviance"][0], abs=1e-6)


@needs_reference
def test_the_reference_matrix_reads_as_the_true_class():
    rng = np.random.default_rng(0)
    env = np.repeat([0, 1], 2000)
    x = rng.standard_normal(4000)
    y = 0.8 * x + rng.standard_normal(4000) * np.where(env == 1, 3.0, 1.0)
    z = 0.7 * y + rng.standard_normal(4000)
    X = np.column_stack([x, y, z])
    answer = reference(X, env, "known", I=np.array([1]))
    expected = np.zeros((3, 3), dtype=np.int8)
    expected[0, 1] = expected[1, 2] = TAIL
    expected[1, 0] = expected[2, 1] = ARROW
    assert np.array_equal(marks(answer["icpdag"]), expected)
