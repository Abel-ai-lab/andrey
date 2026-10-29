"""Bootstrap recovery: on a known graph with strong edges, the averaged graph is the graph."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import andrey
from andrey.core import ARROW, TAIL

# 0 -> 2 <- 1 is a collider, so PC orients it and 2 -> 3; 4 -> 5 stays undirected in a CPDAG.
EDGES = [(0, 2), (1, 2), (2, 3), (4, 5)]
PARENTS = {2: (0, 1), 3: (2,), 5: (4,)}
N_BOOT = 20


def simulate(noise, n=600, seed=0):
    """Six variables on ``EDGES``: a child is 0.9 times each parent plus half its own noise."""
    rng = np.random.default_rng(seed)
    x = noise(rng, (n, 6))
    for target, parents in PARENTS.items():  # the parents come before their children
        x[:, target] = 0.9 * sum(x[:, p] for p in parents) + 0.5 * x[:, target]
    return pd.DataFrame(x, columns=[f"x{i}" for i in range(6)])


def gaussian(rng, shape):
    return rng.standard_normal(shape)


def uniform(rng, shape):
    return rng.uniform(-1.7, 1.7, shape)


def adjacent(i, j):
    return any({i, j} == set(edge) for edge in EDGES)


def test_pc_bootstrap_recovers_the_cpdag():
    out = andrey.bootstrap("pc", simulate(gaussian), n_boot=N_BOOT, seed=0)
    graph = out.structure
    assert graph.kind == "cpdag" and graph.n_nodes == 6
    absent = out.frequency("absent")
    assert all(absent[i, j] <= 0.05 for i, j in EDGES)  # every true pair is adjacent in the fits
    for source, target in EDGES[:3]:
        # Some resamples keep a spurious edge between the collider's parents and leave the
        # orientation open.
        assert out.frequency("->")[source, target] >= 0.7
        assert graph.endpoints(source, target) == (TAIL, ARROW)
    assert out.frequency("--")[4, 5] >= 0.9 and graph.endpoints(4, 5) == (TAIL, TAIL)
    for i in range(6):
        for j in range(i):
            if not adjacent(i, j):
                assert absent[i, j] >= 0.8
    assert graph.to_numpy().astype(bool).sum() == 2 * len(EDGES)  # no edge beyond the truth


@pytest.mark.parametrize("method", ["ges", "boss"])
def test_score_and_permutation_searches_recover_the_cpdag(method):
    # BOSS keeps an edge between the collider's parents on some samples (seed 0 is one) in the fit
    # on the data itself; this sample is one where that fit is right, so the test is whether the
    # resamples keep it.
    data = simulate(gaussian, seed=3)
    truth = {tuple(sorted(edge)) for edge in EDGES}

    def adjacencies(graph):
        return {tuple(sorted(pair)) for pair in zip(*np.nonzero(graph.to_numpy()))}

    assert adjacencies(getattr(andrey, method)(data).structure) == truth
    out = andrey.bootstrap(method, data, n_boot=N_BOOT, seed=0)
    present = 1 - out.frequency("absent")
    assert all(present[i, j] >= 0.9 for i, j in EDGES)
    assert adjacencies(out.structure) == truth


def test_lingam_bootstrap_recovers_the_dag_and_its_directions():
    out = andrey.bootstrap("direct_lingam", simulate(uniform), n_boot=N_BOOT, seed=0)
    assert out.structure.kind == "dag"
    assert {(s, t) for s, t, _ in out.structure.oriented_edges(index=True)} == set(EDGES)
    assert all(out.frequency("->")[s, t] >= 0.9 for s, t in EDGES)
    assert all(out.frequency("<-")[s, t] <= 0.1 for s, t in EDGES)


def test_a_block_bootstrap_of_pcmci_recovers_the_lagged_links():
    rng = np.random.default_rng(0)
    A = np.zeros((3, 3))  # x(t) = A x(t-1) + noise: A[effect, cause]
    A[0, 0], A[1, 0], A[2, 1] = 0.5, 0.8, 0.8
    x = rng.standard_normal((800, 3))
    for t in range(1, 800):
        x[t] += A @ x[t - 1]
    out = andrey.bootstrap(
        "pcmci", pd.DataFrame(x), n_boot=N_BOOT, seed=0, block_length=40, tau_max=1
    )
    truth = {(0, 0), (0, 1), (1, 2)}  # (cause, effect)
    assert out.structure.lags == (1,)
    assert all(out.frequency("->", lag=1)[cause, effect] >= 0.9 for cause, effect in truth)
    links = {(s, t) for s, t, _ in out.structure.lag(1).oriented_edges(index=True)}
    assert links == truth
