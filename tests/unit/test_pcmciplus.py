"""PCMCI+ engine and adapter: the four phases on hand cases, and the mapping to a structure.

An oracle test reads the exact stationary covariance of a linear process
``x_t = B0 x_t + sum_k A_k x_{t-k} + e_t``, so every independence it reports holds in the
population and the expected graph follows from the process alone, with no sampling error.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import solve_discrete_lyapunov

import andrey.temporal.pcmciplus as pcmciplus_module
from andrey.api._temporal import _adapt_pcmciplus
from andrey.core import ARROW, NULL, TAIL, backend
from andrey.temporal.pcmciplus import (
    COLLIDER_RULES,
    PCMCIPlusResult,
    apply_orientations,
    collider_verdict,
    pcmciplus,
    run_phases,
)


class Oracle:
    """Partial correlations of the lagged design's population covariance; p is 0 or 1."""

    def __init__(self, b0: np.ndarray, a_stack: np.ndarray, tau_max: int) -> None:
        n, p = b0.shape[0], a_stack.shape[0]
        mix = np.linalg.inv(np.eye(n) - b0)
        companion = np.zeros((n * p, n * p))
        companion[:n] = np.hstack([mix @ a for a in a_stack])
        companion[n:, :-n] = np.eye(n * (p - 1))
        noise = np.zeros((n * p, n * p))
        noise[:n, :n] = mix @ mix.T
        state = solve_discrete_lyapunov(companion, noise)
        # autocovariance[h] = Cov(x_t, x_{t-h})
        autocovariance, power = [], np.eye(n * p)
        for _ in range(2 * tau_max + 1):
            autocovariance.append((power @ state)[:n, :n])
            power = companion @ power
        width = 2 * tau_max + 1
        self.cov = np.block(
            [
                [autocovariance[b - a] if b >= a else autocovariance[a - b].T for b in range(width)]
                for a in range(width)
            ]
        )

    def statistic_pvalues(self, x, y, conditions):
        statistics = []
        for index in np.column_stack([x, y, conditions]):
            precision = np.linalg.inv(self.cov[np.ix_(index, index)])
            statistics.append(-precision[0, 1] / np.sqrt(precision[0, 0] * precision[1, 1]))
        statistics = np.array(statistics)
        return statistics, np.where(np.abs(statistics) > 1e-8, 0.0, 1.0)


def _run(b0, a_stack, tau_max=1, collider_rule="majority") -> PCMCIPlusResult:
    n_vars = b0.shape[0]
    return run_phases(
        Oracle(b0, a_stack, tau_max), n_vars, tau_max, 0.01, collider_rule, range(n_vars)
    )


def _lagged(result: PCMCIPlusResult) -> set[tuple[int, int, int]]:
    """Every lagged ``(cause, lag, effect)`` link."""
    return {(int(i), int(lag), int(j)) for lag, i, j in np.argwhere(result.adjacency) if lag > 0}


def _contemporaneous(result: PCMCIPlusResult) -> set[tuple[int, int, str]]:
    """Every lag-0 link as ``(i, j, "->")`` for ``i -> j``, else ``(i < j, "--")`` or ``"x-x"``."""
    m = result.marks
    links = set()
    for i, j in zip(*np.nonzero(np.triu(m != NULL))):
        i, j = int(i), int(j)
        if m[i, j] == TAIL and m[j, i] == ARROW:
            links.add((i, j, "->"))
        elif m[i, j] == ARROW and m[j, i] == TAIL:
            links.add((j, i, "->"))
        else:
            links.add((i, j, "--" if m[i, j] == TAIL else "x-x"))
    return links


def _process(n: int, links: dict[tuple[int, int, int], float]) -> tuple[np.ndarray, np.ndarray]:
    """``B0`` and a one-lag ``A`` stack from ``{(cause, lag, effect): weight}``."""
    b0, a = np.zeros((n, n)), np.zeros((1, n, n))
    for (i, lag, j), weight in links.items():
        (b0 if lag == 0 else a[lag - 1])[j, i] = weight
    return b0, a


@pytest.mark.parametrize("collider_rule", COLLIDER_RULES)
def test_an_unlagged_chain_stays_unoriented(collider_rule):
    # X0 -> X1 -> X2 within each step, no memory: the chain's Markov equivalents all fit.
    result = _run(*_process(3, {(0, 0, 1): 0.8, (1, 0, 2): 0.8}), collider_rule=collider_rule)
    assert _contemporaneous(result) == {(0, 1, "--"), (1, 2, "--")}
    assert _lagged(result) == set()
    assert result.parents == ((), (), ())
    assert result.sepsets[(0, 0, 2)] == result.sepsets[(2, 0, 0)] == (1,)


@pytest.mark.parametrize("collider_rule", COLLIDER_RULES)
def test_an_unlagged_collider_is_oriented(collider_rule):
    result = _run(*_process(3, {(0, 0, 2): 0.8, (1, 0, 2): 0.8}), collider_rule=collider_rule)
    assert _contemporaneous(result) == {(0, 2, "->"), (1, 2, "->")}
    assert result.ambiguous_triples == ()


@pytest.mark.parametrize("collider_rule", COLLIDER_RULES)
def test_a_lagged_parent_orients_a_contemporaneous_chain(collider_rule):
    # X0 has memory; X0_t -> X1_t -> X2_t. R1 orients both lag-0 links from X0_{t-1} -> X0_t.
    links = {(0, 1, 0): 0.6, (0, 0, 1): 0.7, (1, 0, 2): 0.7}
    result = _run(*_process(3, links), collider_rule=collider_rule)
    assert _contemporaneous(result) == {(0, 1, "->"), (1, 2, "->")}
    assert _lagged(result) == {(0, 1, 0)}
    # Phase 1 keeps X0_{t-1} for X1 and X2 through their lag-0 ancestors; phase 2 removes it.
    assert (0, 1) in result.parents[1] and (0, 1) in result.parents[2]
    assert result.sepsets[(0, 1, 1)] == (0,)


@pytest.mark.parametrize("collider_rule", COLLIDER_RULES)
def test_a_lagged_cause_and_a_lag_0_cause_make_a_collider(collider_rule):
    # X0_{t-1} -> X1_t <- X2_t: only the lagged link can orient X2_t -> X1_t.
    result = _run(*_process(3, {(0, 1, 1): 0.7, (2, 0, 1): 0.7}), collider_rule=collider_rule)
    assert _contemporaneous(result) == {(2, 1, "->")}
    assert _lagged(result) == {(0, 1, 1)}


def test_a_mix_of_lagged_and_lag_0_links():
    # Memory in X0 and X3; X0_{t-1} -> X1_t; X1_t -> X2_t (R1 from the lagged link);
    # X2_t <- X3_t and X1_t -> X2_t make a lag-0 collider at X2; X2_{t-1} -> X3_t.
    links = {
        (0, 1, 0): 0.5,
        (3, 1, 3): 0.5,
        (0, 1, 1): 0.6,
        (1, 0, 2): 0.6,
        (3, 0, 2): 0.6,
        (2, 1, 3): 0.5,
    }
    result = _run(*_process(4, links))
    assert _lagged(result) == {(0, 1, 0), (3, 1, 3), (0, 1, 1), (2, 1, 3)}
    assert _contemporaneous(result) == {(1, 2, "->"), (3, 2, "->")}
    assert result.conflicts == ()


def test_p_values_are_the_largest_phase_2_test_of_each_link():
    result = _run(*_process(3, {(0, 0, 1): 0.8, (1, 0, 2): 0.8}))
    assert result.p_values[0, 0, 1] == result.p_values[0, 1, 0] == 0.0  # kept: every test p = 0
    assert result.p_values[0, 0, 2] == 1.0  # removed by the test given X1
    assert result.statistics[0, 0, 2] == pytest.approx(0.0, abs=1e-8)
    np.testing.assert_array_equal(result.adjacency[0], result.adjacency[0].T)
    assert np.all(np.diagonal(result.p_values[0]) == 1.0)


def test_opposite_requests_mark_a_conflict():
    marks = np.full((3, 3), TAIL, dtype=np.int8)
    np.fill_diagonal(marks, NULL)
    assert apply_orientations(marks, [(0, 1), (1, 0), (1, 2), (1, 2)])
    assert marks[0, 1] == marks[1, 0] == ARROW  # both ways: conflict
    assert (marks[1, 2], marks[2, 1]) == (TAIL, ARROW)
    assert not apply_orientations(marks, [])
    result = PCMCIPlusResult((), np.zeros((1, 3, 3), bool), marks, None, None, {}, ())
    assert result.conflicts == ((0, 1),)


def test_a_conflict_is_an_undirected_lag_0_link_listed_in_the_metadata():
    marks = np.zeros((3, 3), dtype=np.int8)
    marks[0, 1] = marks[1, 0] = ARROW  # conflict
    marks[1, 2], marks[2, 1] = TAIL, ARROW  # 1 -> 2
    adjacency = np.zeros((2, 3, 3), dtype=bool)
    adjacency[0] = marks != NULL
    adjacency[1, 2, 0] = True  # X2_{t-1} -> X0_t
    result = PCMCIPlusResult(
        ((), (), ((2, 1),)), adjacency, marks, np.ones((2, 3, 3)), np.zeros((2, 3, 3)), {}, ()
    )
    out = _adapt_pcmciplus(result)
    assert out.structure.lags == (0, 1)
    assert out.structure.lag(0).kind == "cpdag"
    assert out.structure.lag(0).oriented_edges() == [(0, 1, "undirected"), (1, 2, "directed")]
    assert out.structure.lag(1).oriented_edges() == [(2, 0, "directed")]
    assert out.metadata["conflicts"] == [[0, 1]]


@pytest.mark.parametrize(
    ("holding", "majority", "conservative"),
    [
        (0, "collider", "collider"),
        (1, "collider", "ambiguous"),  # 1 of 4
        (2, "ambiguous", "ambiguous"),  # exactly half
        (3, "non-collider", "ambiguous"),
        (4, "non-collider", "non-collider"),
    ],
)
def test_collider_rules_read_the_share_of_separating_subsets_holding_the_middle(
    holding, majority, conservative
):
    separating = [frozenset({7, s}) for s in range(holding)] + [
        frozenset({s}) for s in range(10, 14 - holding)
    ]
    assert collider_verdict(separating, 7, "majority") == majority
    assert collider_verdict(separating, 7, "conservative") == conservative
    assert collider_verdict([], 7, "majority") == "ambiguous"


def _chain_series(n_steps: int = 800, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((n_steps + 100, 3))
    x = np.zeros_like(e)
    for t in range(1, len(x)):
        x[t, 0] = 0.7 * x[t - 1, 0] + e[t, 0]
        x[t, 1] = 0.5 * x[t - 1, 1] + 0.6 * x[t, 0] + e[t, 1]
        x[t, 2] = 0.5 * x[t - 1, 2] + 0.6 * x[t, 1] + e[t, 2]
    return x[100:]


@pytest.mark.parametrize("indep_test", ["student_t", "fisherz"])
def test_pcmciplus_on_data_orients_the_contemporaneous_chain(indep_test):
    result = pcmciplus(_chain_series(), tau_max=2, indep_test=indep_test)
    assert _contemporaneous(result) == {(0, 1, "->"), (1, 2, "->")}
    assert _lagged(result) == {(0, 1, 0), (1, 1, 1), (2, 1, 2)}
    assert result.adjacency.shape == result.p_values.shape == (3, 3, 3)


def test_unknown_names_raise_naming_the_choices():
    with pytest.raises(ValueError, match=r"indep_test must be one of \['fisherz', 'student_t'\]"):
        pcmciplus(_chain_series(100), indep_test="spearman")
    with pytest.raises(ValueError, match=r"collider_rule must be one of \['majority', "):
        pcmciplus(_chain_series(100), collider_rule="vote")


def test_pcmciplus_runs_its_phases_on_one_blas_thread(monkeypatch):
    from threadpoolctl import threadpool_info, threadpool_limits

    if not threadpool_info():
        pytest.skip("no thread pool that threadpoolctl can control")
    counts = []

    def recording_run_phases(*args):
        counts.append({lib["num_threads"] for lib in threadpool_info()})
        return run_phases(*args)

    monkeypatch.setattr(pcmciplus_module, "run_phases", recording_run_phases)
    backend._thread_pools.cache_clear()  # pools loaded by earlier tests are found again
    with threadpool_limits(limits=2):  # a count other than 1, set after the pools were found
        pcmciplus(_chain_series(100))
    assert counts == [{1}]
