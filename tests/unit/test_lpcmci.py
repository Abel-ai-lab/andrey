"""LPCMCI engine and adapter: hand cases with hidden causes, and the mapping to a structure.

An oracle test reads the exact stationary covariance of a linear process
``x_t = B0 x_t + sum_k A_k x_{t-k} + e_t`` and keeps the observed variables only, so every
independence it reports holds in the population and the expected graph follows from the process.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import solve_discrete_lyapunov

from andrey.api._temporal import _adapt_lpcmci
from andrey.core import ARROW, CIRCLE, NULL, TAIL
from andrey.temporal.lpcmci import CONFLICT, LPCMCIResult, lpcmci, run

# A link ``(cause, lag, effect)`` and its weight.
Links = dict[tuple[int, int, int], float]


class Oracle:
    """Partial correlations of the observed lagged design's population covariance; p is 0 or 1."""

    def __init__(self, n: int, links: Links, tau_max: int, observed: list[int]) -> None:
        p = max([lag for _, lag, _ in links] + [1])
        b0, a_stack = np.zeros((n, n)), np.zeros((p, n, n))
        for (i, lag, j), weight in links.items():
            (b0 if lag == 0 else a_stack[lag - 1])[j, i] = weight
        mix = np.linalg.inv(np.eye(n) - b0)
        companion = np.zeros((n * p, n * p))
        companion[:n] = np.hstack([mix @ a for a in a_stack])
        companion[n:, :-n] = np.eye(n * (p - 1))
        noise = np.zeros((n * p, n * p))
        noise[:n, :n] = mix @ mix.T
        state = solve_discrete_lyapunov(companion, noise)
        width = 2 * tau_max + 1
        # autocovariance[h] = Cov(x_t, x_{t-h})
        autocovariance, power = [], np.eye(n * p)
        for _ in range(width):
            autocovariance.append((power @ state)[:n, :n])
            power = companion @ power
        cov = np.block(
            [
                [autocovariance[b - a] if b >= a else autocovariance[a - b].T for b in range(width)]
                for a in range(width)
            ]
        )
        keep = [lag * n + v for lag in range(width) for v in observed]
        self.cov = cov[np.ix_(keep, keep)]

    def statistic_pvalues(self, x, y, conditions):
        statistics = []
        for index in np.column_stack([x, y, conditions]):
            precision = np.linalg.inv(self.cov[np.ix_(index, index)])
            statistics.append(-precision[0, 1] / np.sqrt(precision[0, 0] * precision[1, 1]))
        statistics = np.array(statistics)
        return statistics, np.where(np.abs(statistics) > 1e-8, 0.0, 1.0)


def _links(result: LPCMCIResult) -> set[str]:
    """Every link as ``"i<marks>j@lag"``, a lag-0 link once with ``i < j``."""
    n_lags, n, _ = result.start_marks.shape
    return {
        f"{i}{result.link(i, j, lag)}{j}@{lag}"
        for lag in range(n_lags)
        for i in range(n)
        for j in range(n)
        if (lag > 0 or i < j) and result.link(i, j, lag)
    }


def _run(n: int, links: Links, tau_max: int, observed: list[int]) -> LPCMCIResult:
    return run(Oracle(n, links, tau_max, observed), len(observed), tau_max, 0.01, 1)


# Variable 2 is hidden white noise driving X0 and X1 within one step; X0 and X1 have memory.
SAME_STEP_CAUSE = {(0, 1, 0): 0.5, (1, 1, 1): 0.6, (2, 0, 0): 0.7, (2, 0, 1): 0.4}


@pytest.mark.parametrize("tau_max", [1, 2])
def test_a_hidden_common_cause_is_a_two_headed_link(tau_max):
    result = _run(3, SAME_STEP_CAUSE, tau_max, [0, 1])
    assert _links(result) == {"0<->1@0", "0-->0@1", "1-->1@1"}


def test_a_lagged_hidden_cause_is_a_two_headed_lagged_link():
    # The hidden cause drives X0 at once and X1 one step later.
    links = {(0, 1, 0): 0.5, (1, 1, 1): 0.6, (2, 0, 0): 0.7, (2, 1, 1): 0.4}
    assert _links(_run(3, links, 2, [0, 1])) == {"0<->1@1", "0-->0@1", "1-->1@1"}


def test_a_lagged_cause_needs_an_earlier_step_to_get_its_tail():
    # X0_{t-1} -> X1_t with memory in both: tau_max=1 cannot tell a hidden cause from a tail.
    links = {(0, 1, 0): 0.5, (1, 1, 1): 0.6, (0, 1, 1): 0.7}
    assert _links(_run(2, links, 1, [0, 1])) == {"0o->0@1", "0o->1@1", "1o->1@1"}
    assert _links(_run(2, links, 2, [0, 1])) == {"0-->0@1", "0-->1@1", "1-->1@1"}


def test_a_lagged_parent_orients_a_same_step_chain():
    # X0 has memory; X0_t -> X1_t -> X2_t. No variable is hidden.
    links = {(0, 1, 0): 0.6, (0, 0, 1): 0.7, (1, 0, 2): 0.7}
    assert _links(_run(3, links, 2, [0, 1, 2])) == {"0-->1@0", "1-->2@0", "0-->0@1"}


def test_an_unlagged_collider_keeps_its_circles():
    # X0 -> X2 <- X1 within one step and no memory: the tails stay open.
    links = {(0, 0, 2): 0.8, (1, 0, 2): 0.8}
    assert _links(_run(3, links, 1, [0, 1, 2])) == {"0o->2@0", "1o->2@0"}


def test_independent_white_noise_has_no_links():
    assert _links(_run(3, {}, 1, [0, 1, 2])) == set()


def test_a_hidden_autocorrelated_cause_leaves_lagged_links_open():
    # The hidden cause has memory, so it links every lag of X0 and X1.
    links = {(0, 1, 0): 0.5, (1, 1, 1): 0.6, (2, 1, 2): 0.7, (2, 0, 0): 0.7, (2, 0, 1): 0.4}
    found = _links(_run(3, links, 1, [0, 1]))
    assert found == {"0o-o1@0", "0o->0@1", "0o->1@1", "1o->0@1", "1o->1@1"}


def test_marks_are_consistent_at_both_orders_of_a_same_step_link():
    result = _run(3, SAME_STEP_CAUSE, 2, [0, 1])
    np.testing.assert_array_equal(result.start_marks[0], result.end_marks[0].T)
    assert (result.p_values[0] == result.p_values[0].T).all()
    assert result.link(1, 0, 0) == "<->"
    assert result.parents == (((0, 1),), ((1, 1),))


def _fixed_result() -> LPCMCIResult:
    """Three variables and ``tau_max=1``: ``0 x-> 1`` at lag 0; at lag 1, ``2 -> 2``, ``0 o-> 1``,
    and ``1 <-> 2``.
    """
    start = np.zeros((2, 3, 3), dtype=np.int8)
    end = np.zeros((2, 3, 3), dtype=np.int8)
    start[0, 0, 1], end[0, 0, 1] = CONFLICT, ARROW
    start[0, 1, 0], end[0, 1, 0] = ARROW, CONFLICT
    start[1, 2, 2], end[1, 2, 2] = TAIL, ARROW
    start[1, 0, 1], end[1, 0, 1] = CIRCLE, ARROW
    start[1, 1, 2], end[1, 1, 2] = ARROW, ARROW
    return LPCMCIResult(start, end, np.zeros((2, 3, 3)), np.ones((2, 3, 3)), {}, ((), (), ()))


def test_the_adapter_maps_a_conflict_to_a_circle_and_keeps_definite_lagged_causes():
    out = _adapt_lpcmci(_fixed_result())
    assert out.structure.lags == (0, 1)
    assert out.structure.lag(0).kind == "pag"
    assert out.structure.lag(0).oriented_edges() == [(0, 1, "partially_directed")]
    assert out.structure.lag(1).kind == "digraph"
    assert out.structure.lag(1).oriented_edges() == [(2, 2, "directed")]
    links = out.metadata["links"]
    assert (links[0][0][1], links[0][1][0]) == ("x->", "<-x")
    assert (links[1][0][1], links[1][1][2], links[1][2][2]) == ("o->", "<->", "-->")
    assert links[1][1][0] == ""
    assert out.metadata["conflicts"] == [[0, 0, 1]]
    assert out.metadata["algorithm"] == "lpcmci"


def test_an_empty_result_has_no_links():
    shape = (2, 2, 2)
    empty = LPCMCIResult(
        np.zeros(shape, np.int8),
        np.zeros(shape, np.int8),
        np.zeros(shape),
        np.ones(shape),
        {},
        ((), ()),
    )
    out = _adapt_lpcmci(empty)
    assert all(not out.structure.lag(k).oriented_edges() for k in (0, 1))
    assert out.metadata["conflicts"] == []
    assert (empty.adjacency == (empty.start_marks != NULL)).all()


def _hidden_cause_series(n_steps: int = 1000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    hidden = rng.standard_normal(n_steps)
    x = np.zeros((n_steps, 2))
    for t in range(1, n_steps):
        x[t, 0] = 0.5 * x[t - 1, 0] + 0.7 * hidden[t] + rng.standard_normal()
        x[t, 1] = 0.6 * x[t - 1, 1] + 0.6 * hidden[t] + rng.standard_normal()
    return x


@pytest.mark.parametrize("indep_test", ["student_t", "fisherz"])
def test_lpcmci_on_data_finds_the_hidden_cause(indep_test):
    result = lpcmci(_hidden_cause_series(), tau_max=2, indep_test=indep_test)
    assert _links(result) == {"0<->1@0", "0-->0@1", "1-->1@1"}


def test_no_preliminary_round_runs_the_final_round_alone():
    result = lpcmci(_hidden_cause_series(), tau_max=2, n_preliminary_iterations=0)
    assert result.parents == ((), ())
    assert "0<->1@0" in _links(result)


def test_an_unknown_test_name_is_refused():
    with pytest.raises(ValueError, match="indep_test"):
        lpcmci(_hidden_cause_series(), indep_test="kci")
