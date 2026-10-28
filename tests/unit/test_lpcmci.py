"""LPCMCI engine and adapter: hand cases with hidden causes, and the mapping to a structure.

An oracle test reads the exact stationary covariance of a linear process
``x_t = B0 x_t + sum_k A_k x_{t-k} + e_t`` and keeps the observed variables only, so every
independence it reports holds in the population and the expected graph follows from the process.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest
from scipy.linalg import solve_discrete_lyapunov

import andrey
import andrey.temporal.lpcmci as engine
from andrey.api._temporal import _adapt_lpcmci
from andrey.core import ARROW, CIRCLE, NULL, TAIL
from andrey.temporal.lpcmci import CONFLICT, LPCMCIResult, lpcmci, run
from andrey.temporal.pcmci import INDEP_TESTS, lagged_design

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
    oracle = Oracle(n, links, tau_max, observed)
    return run(oracle, len(observed), tau_max, 0.01, 1, range(len(observed)))


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


def _result_arrays(result: LPCMCIResult) -> tuple:
    return (
        result.start_marks.tolist(),
        result.end_marks.tolist(),
        result.statistics.tolist(),
        result.p_values.tolist(),
        result.sepsets,
        result.parents,
    )


def _related_series(n_vars: int = 6, n_steps: int = 600, seed: int = 0) -> np.ndarray:
    """Autocorrelated series with a chain of lagged links and two shared drivers."""
    rng = np.random.default_rng(seed)
    drivers = rng.standard_normal((n_steps, 2))
    x = np.zeros((n_steps, n_vars))
    for t in range(1, n_steps):
        x[t] = 0.5 * x[t - 1] + rng.standard_normal(n_vars)
        x[t, 1:] += 0.4 * x[t - 1, :-1]
        x[t, : n_vars // 2] += 0.6 * drivers[t, 0]
        x[t, n_vars // 3 :] += 0.5 * drivers[t, 1]
    return x


def test_a_depth_above_every_search_returns_the_uncapped_graph():
    data = _related_series()
    uncapped = lpcmci(data, tau_max=2, pc_alpha=0.01)
    assert _result_arrays(
        lpcmci(data, tau_max=2, pc_alpha=0.01, max_p_global=50)
    ) == _result_arrays(uncapped)


@pytest.mark.parametrize("max_p_global", [0, 1, 2])
def test_a_depth_cap_bounds_every_removal_search(monkeypatch, max_p_global):
    sizes = []
    searched = engine._Search.separating_subsets

    def recorded(self, a, b, search, sdef, p):
        sizes.append(p)
        return searched(self, a, b, search, sdef, p)

    monkeypatch.setattr(engine._Search, "separating_subsets", recorded)
    lpcmci(_related_series(), tau_max=1, pc_alpha=0.05, max_p_global=max_p_global)
    assert sizes and max(sizes) == max_p_global


@pytest.mark.parametrize("cached_depth", [-1, 100])
def test_tests_with_or_without_the_cache_give_the_same_graph(monkeypatch, cached_depth):
    data = _related_series(seed=2)
    expected = _result_arrays(lpcmci(data, tau_max=1, pc_alpha=0.05))
    monkeypatch.setattr(engine, "CACHED_DEPTH", cached_depth)
    assert _result_arrays(lpcmci(data, tau_max=1, pc_alpha=0.05)) == expected


def test_missing_values_count_each_query_once_at_any_cached_depth(monkeypatch):
    data = _related_series(n_steps=800)
    data[np.random.default_rng(1).random(data.shape) < 0.1] = np.nan
    expected = andrey.lpcmci(data, tau_max=2, pc_alpha=0.05).metadata
    monkeypatch.setattr(engine, "CACHED_DEPTH", -1)
    assert andrey.lpcmci(data, tau_max=2, pc_alpha=0.05).metadata == expected


# ---- bounded test calls, cache, and vote ------------------------------------------------------


class Counting:
    """Passes queries to ``test`` and records how many it receives per call."""

    def __init__(self, test) -> None:
        self.test, self.calls = test, []

    def statistic_pvalues(self, x, y, conditions):
        self.calls.append(len(x))
        return self.test.statistic_pvalues(x, y, conditions)


def _data_search(test_wrapper=lambda test: test) -> engine._Search:
    """A search over four variables of white noise, ``tau_max=1``, with the Student-t test."""
    data = np.random.default_rng(0).standard_normal((300, 4))
    test = INDEP_TESTS["student_t"](lagged_design(data, 1))
    return engine._Search(test_wrapper(test), 4, 1, 0.05, range(4))


# The pair X0_{t-1}, X0_t and the 15 two-node sets of the other window nodes.
PAIR = ((0, 1), (0, 0))
SETS = [frozenset(s) for s in combinations([(1, 0), (2, 0), (3, 0), (1, 1), (2, 1), (3, 1)], 2)]


def test_a_call_over_the_batch_limit_is_split_without_changing_results(monkeypatch):
    whole = _data_search().run_tests(*PAIR, SETS, rank=False, report=False)
    monkeypatch.setattr(engine, "BATCH", 4)
    counting = Counting(None)
    search = _data_search(lambda test: setattr(counting, "test", test) or counting)
    assert search.run_tests(*PAIR, SETS, rank=False, report=False) == whole
    assert counting.calls == [4, 4, 4, 3]


def test_evicting_cached_results_does_not_change_them(monkeypatch):
    calls = [SETS[:3], SETS[1:3] + SETS[5:9], SETS[:3]]  # read cached, add, then re-read evicted
    unbounded = [_data_search().run_tests(*PAIR, sets) for sets in calls]
    monkeypatch.setattr(engine, "CACHE_SIZE", 4)
    counting = Counting(None)
    bounded = _data_search(lambda test: setattr(counting, "test", test) or counting)
    for sets, expected in zip(calls, unbounded):
        tested = len(counting.calls)
        assert bounded.run_tests(*PAIR, sets) == expected
        assert len(bounded.cache) <= 4
    assert len(counting.calls) > tested  # the last call recomputed evicted results


class HashTest:
    """A test whose p-value is a fixed pseudo-random function of the queried design columns."""

    def statistic_pvalues(self, x, y, conditions):
        rows = np.column_stack([x, y, conditions]).astype(np.int64)
        weights = 7919 * np.arange(1, rows.shape[1] + 1)
        p_values = ((((rows + 3) ** 2) * weights).sum(axis=1) % 1009 + 0.5) / 1009
        return 1.0 - p_values, p_values


def test_a_split_search_finds_the_same_separating_set(monkeypatch):
    unbounded = engine._Search(HashTest(), 4, 1, 0.85, range(4))
    expected = unbounded.separating(*PAIR, iter(SETS))
    monkeypatch.setattr(engine, "BATCH", 2)
    split = engine._Search(HashTest(), 4, 1, 0.85, range(4))
    assert split.separating(*PAIR, iter(SETS)) == expected
    assert SETS.index(expected) >= 2  # found after the first batch
    assert (split.imin, split.best) == (unbounded.imin, unbounded.best)


def _listed_vote(search: engine._Search, key) -> list[frozenset]:
    """Every distinct separating set of the pair ``key`` for the vote, as one list."""
    i, j, lag = key
    start, end = (i, lag), (j, 0)
    sdef = (search.parents(start) | search.parents(end)) - {start, end}
    found = list(search.sepsets.get(key, []))
    for anchor, other in ((start, end), (end, start)):
        pool = [] if anchor == start and lag > 0 else sorted(search.apds(anchor, other) - sdef)
        for size in range(len(pool) + 1):
            for subset in combinations(pool, size):
                conditions = frozenset(subset) | sdef
                _, (p_value,) = search.run_tests(start, end, [conditions], rank=False)
                if p_value > search.alpha and conditions not in found:
                    found.append(conditions)
    return found


def _listed_share(search: engine._Search, a, b, c) -> float:
    key, _ = engine._key(a, c)
    found = _listed_vote(search, key)
    frame = min(a[1], c[1])
    return sum((b[0], b[1] - frame) in s for s in found) / len(found) if found else 0.5


# Outer pairs ``(a, c)``: a same-step pair in two frames and a lagged pair.
OUTER = [((0, 0), (1, 0)), ((0, 1), (1, 1)), ((0, 1), (2, 0))]


def _shares(search: engine._Search, listed: bool) -> list[float]:
    share = (lambda a, b, c: _listed_share(search, a, b, c)) if listed else search.share
    return [share(a, b, c) for a, c in OUTER for b in search.window if b not in (a, c)]


def test_the_counted_vote_matches_the_listed_vote_before_and_after_a_change():
    search = engine._Search(HashTest(), 4, 1, 0.5, range(4))
    search.initialize()
    for a, c in OUTER[::2]:
        search.remove(engine._key(a, c)[0])
    # Stored sets for the same-step pair: one the vote also finds from both ends, one it cannot.
    same_step = engine._key(*OUTER[0])[0]
    found = _listed_vote(search, same_step)
    shared = next(
        s for s in found if s <= search.apds((0, 0), (1, 0)) & search.apds((1, 0), (0, 0))
    )
    search.add_sepset(same_step, shared, 0)
    search.add_sepset(same_step, frozenset({(2, 0), (3, 0), (1, 1), (2, 1)}), 0)
    before = _shares(search, listed=False)
    assert before == _shares(search, listed=True)
    # X1_t -> X2_t makes X1_t a parent of X2_t and X2_t a non-ancestor of X1_t; X3_{t-1} -> X0_t.
    marks = {
        ((1, 2, 0), True): {TAIL},
        ((1, 2, 0), False): {ARROW},
        ((3, 0, 1), True): {TAIL},
    }
    assert search._apply(marks, {}, {}, lagged_only=False)
    after = _shares(search, listed=False)
    assert after == _shares(search, listed=True)
    assert after != before


# ---- separating sets the rules store do not depend on the variable order ----------------------


class TableTest:
    """P-values of ``X0_t`` and ``X1_t`` read from a table keyed by the conditioning variables.

    ``relabel`` maps a table variable to the variable the search sees; any query not in the
    table is a dependence. ``tau_max`` is 0, so a design column is a variable.
    """

    def __init__(self, table: dict[frozenset, float], relabel: dict[int, int]) -> None:
        self.table = {frozenset(relabel[v] for v in key): p for key, p in table.items()}

    def statistic_pvalues(self, x, y, conditions):
        p_values = np.array(
            [
                self.table.get(frozenset(row), 0.01) if {int(a), int(b)} == {0, 1} else 0.01
                for a, b, row in zip(x, y, np.asarray(conditions).tolist())
            ]
        )
        return 1.0 - p_values, p_values


def _relabeled(sets, relabel: dict[int, int]) -> set[frozenset]:
    return {frozenset((relabel[v], lag) for v, lag in s) for s in sets}


SWAP = {0: 0, 1: 1, 2: 3, 3: 2}  # exchanges X2 and X3


@pytest.mark.parametrize("relabel", [{v: v for v in range(4)}, SWAP])
def test_rule_r0a_stores_the_separating_set_of_every_triple(relabel):
    # X0 - X1 - X2 and X0 - X1 - X3 are unshielded; X0 and X1 are separated given the stored set
    # of either outer pair, {X3} for (X0, X2) and {X2} for (X0, X3), and not given nothing.
    table = {frozenset({3}): 0.5, frozenset({2}): 0.4}
    search = engine._Search(TableTest(table, relabel), 4, 0, 0.05, range(4))
    search.initialize()
    for outer, sepset in (((0, 2, 0), {(3, 0)}), ((0, 3, 0), {(2, 0)})):
        i, j, _ = outer
        key = engine._key((relabel[i], 0), (relabel[j], 0))[0]
        search.remove(key)
        search.add_sepset(key, frozenset(_relabeled([sepset], relabel).pop()), 0)
    assert search._apply(*search.rule_r0a(), lagged_only=False)
    stored = _relabeled(search.sepsets[(0, 1, 0)], relabel)
    assert stored == {frozenset({(2, 0)}), frozenset({(3, 0)})}


@pytest.mark.parametrize("relabel", [{v: v for v in range(4)}, SWAP])
def test_weakly_minimal_drops_the_node_that_leaves_the_largest_p_value(relabel):
    # {X2, X3} separates X0 and X1; dropping X2 leaves p = 0.6, dropping X3 leaves p = 0.3, and
    # dropping both leaves a dependence, so X2 goes whatever the variables are called.
    table = {frozenset({2, 3}): 0.9, frozenset({3}): 0.6, frozenset({2}): 0.3}
    search = engine._Search(TableTest(table, relabel), 4, 0, 0.05, range(4))
    search.initialize()
    found = search.weakly_minimal((0, 0), (1, 0), frozenset({(2, 0), (3, 0)}))
    assert _relabeled([found], relabel) == {frozenset({(3, 0)})}
