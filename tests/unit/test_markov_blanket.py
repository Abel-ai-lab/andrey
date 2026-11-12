"""The shared Markov-blanket engine: IAMB, Inter-IAMB, GS, Fast-IAMB, MMPC, HITON-PC, and the local
search.

Under a d-separation oracle each search returns the true set, each recorded separating set
separates, and the spouse step completes the parents and children into the Markov blanket. With a
Fisher-Z test, IAMB returns, bit for bit, the blankets of the Fisher-Z IAMB that CALM's moral graph
is built from. GS admits on a later pass what an earlier pass skipped and shrinks what its order
admitted early. Fast-IAMB adds a batch per round, stops a batch where the data hold too few rows per
cell, and ends a cycle of additions and removals. HITON-PC follows its paper's trace, orders by
the p-value and then the statistic, and skips a test with fewer than five rows per cell. HITON-MB
follows its paper's trace, admits a spouse only with five rows per cell of its test's table, and
skips a spouse test past ``max_depth``. A memoized
run is the run made afresh, and the local search gives the sets the methods return, whatever the
column order. A scan of many candidates gives each the bits of its own query, the ranks are read
only when keys tie, and the counts follow their definitions.
"""

from __future__ import annotations

from functools import partial
from itertools import combinations

import networkx as nx
import numpy as np
import pytest

import andrey
from andrey import GraphStructure, data
from andrey.constraint import blanket
from andrey.constraint._skeleton import scan_conditioning_sets
from andrey.constraint.blanket import (
    LEARNERS,
    Learner,
    LocalSearch,
    blanket_colliders,
    blanket_graph,
    local_search,
    markov_blanket,
    mmpc_sets,
    ordered,
    parents_children,
    reach,
    strongest,
    table_size,
)
from andrey.core.ci import ChiSquare, FisherZ, make_indep_test
from andrey.core.ci_missing import rows_tested
from andrey.core.orient import dag2cpdag
from andrey.core.ties import KEY_DP, tie_ranks
from andrey.data.latent import DSepOracle
from andrey.search.calm import moral_graph
from tests.recovery.helpers.oracle import DSeparation


def _fisherz_iamb(test: FisherZ, target: int, d: int, alpha: float, ranks: list[int]) -> list[int]:
    """Fisher-Z IAMB as CALM's moral graph is defined: grow by the largest absolute partial
    correlation among candidates with a p-value at or below ``alpha``, ties by rank, then shrink
    once in joining order."""
    blanket: list[int] = []
    while True:
        best_key, best_x = None, None
        for x in range(d):
            if x == target or x in blanket:
                continue
            r, p = test.statistic_pvalue(target, x, blanket)
            key = (-round(abs(r), KEY_DP), ranks[x])
            if p <= alpha and (best_key is None or key < best_key):
                best_key, best_x = key, x
        if best_x is None:
            break
        blanket.append(best_x)
    for x in list(blanket):
        rest = [z for z in blanket if z != x]
        if test(target, x, rest) > alpha:
            blanket.remove(x)
    return blanket


@pytest.mark.parametrize(
    ("d", "n", "density", "seed"),
    [(6, 300, 1.0, 0), (10, 500, 2.0, 1), (15, 1000, 3.0, 2), (20, 2000, 2.0, 3), (12, 80, 4.0, 4)],
)
@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.2])
def test_iamb_with_fisherz_is_calms_moral_graph_bit_for_bit(d, n, density, seed, alpha):
    x = data.sample_scm(graph="erdos_renyi", d=d, n=n, seed=seed, density=density).data
    test = FisherZ(x)
    ranks = tie_ranks(x).tolist()
    reference = np.zeros((d, d), dtype=bool)
    for i in range(d):
        blanket = markov_blanket(test, i, d, alpha, ranks)
        assert blanket == _fisherz_iamb(test, i, d, alpha, ranks)
        reference[i, blanket] = True
    want = (reference & reference.T).astype(np.float64)
    got = moral_graph(x, alpha=alpha)
    assert got.dtype == want.dtype and got.tobytes() == want.tobytes()


#: The settings that make a method's facade call the engine's defaults: no conditioning-set cap.
_UNCAPPED = {"hiton_pc": {"max_depth": None}, "hiton_mb": {"max_depth": None}}


def _oracle(seed: int, d: int = 12, density: float = 3.0) -> DSeparation:
    return DSeparation(
        data.sample_scm(graph="erdos_renyi", d=d, n=2, seed=seed, density=density).graph
    )


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("interleaved", [False, True], ids=["iamb", "inter_iamb"])
def test_iamb_finds_the_true_blanket_under_an_oracle(seed, interleaved):
    oracle = _oracle(seed)
    ranks = list(range(oracle.d))
    for v in range(oracle.d):
        got = markov_blanket(oracle, v, oracle.d, 0.5, ranks, interleaved=interleaved)
        assert set(got) == oracle.markov_blanket(v), v
    graph = blanket_graph(oracle, oracle.d, 0.5, ranks, interleaved=interleaved)
    np.testing.assert_array_equal(graph, oracle.moral_graph())


@pytest.mark.parametrize("seed", range(4))
def test_gs_finds_the_true_blanket_under_an_oracle(seed):
    oracle = _oracle(seed)
    search = LocalSearch(oracle, list(range(oracle.d)), 0.5)
    for v in range(oracle.d):
        assert set(search.run("gs", v)[0]) == oracle.markov_blanket(v), v
    np.testing.assert_array_equal(search.structure("gs", "moral") != 0, oracle.moral_graph())


def test_gs_admits_on_a_later_pass_a_spouse_an_earlier_member_hid():
    """0 <- 1 -> 2, 0 -> 3 <- 2: every other variable depends on 0, so the order is the ranks', 1,
    2, 3. The first pass adds 1, finds 2 independent given {1}, and adds 3; the second pass adds 2
    given {1, 3}, the collider opening the path. Nothing is shrunk."""
    oracle = DSepOracle(nx.DiGraph([(1, 0), (1, 2), (0, 3), (2, 3)]))
    search = LocalSearch(oracle, [0, 1, 2, 3], 0.5, trace=True)
    assert search.run("gs", 0) == ([1, 3, 2], {})
    # The order's three tests, 1 read from them, 2 and 3 given {1}, 2 given {1, 3}, and the
    # shrink's three tests.
    assert search.counts.issued == 9
    assert (0, 2, (1,)) in search.counts.queries and (0, 2, (1, 3)) in search.counts.queries


def test_gs_shrinks_a_member_its_order_admitted_early():
    """0 -> 1 -> 2 with 2 ranked first: both depend on 0, so 2 is visited first and joins, 1 joins
    given {2}, and the shrink, in the same order, drops 2 given {1}."""
    oracle = DSepOracle(nx.DiGraph([(0, 1), (1, 2)]))
    assert LocalSearch(oracle, [0, 2, 1], 0.5).run("gs", 0) == ([1], {2: (1,)})


@pytest.mark.parametrize("seed", range(4))
def test_fast_iamb_finds_the_true_blanket_under_an_oracle(seed):
    oracle = _oracle(seed)
    search = LocalSearch(oracle, list(range(oracle.d)), 0.5)
    for v in range(oracle.d):
        assert set(search.run("fast_iamb", v)[0]) == oracle.markov_blanket(v), v
    np.testing.assert_array_equal(search.structure("fast_iamb", "moral") != 0, oracle.moral_graph())


def test_fast_iamb_adds_a_batch_per_round():
    """0 -> 1, 0 -> 2, 0 -> 3 with no categories to count: one round adds all three, so the search
    asks the three tests of the round and the three of the shrink, where IAMB asks 3 + 2 + 1 and
    the shrink's three."""
    oracle = DSepOracle(nx.DiGraph([(0, 1), (0, 2), (0, 3)]))
    fast = LocalSearch(oracle, [0, 1, 2, 3], 0.5)
    assert fast.run("fast_iamb", 0) == ([1, 2, 3], {})
    plain = LocalSearch(oracle, [0, 1, 2, 3], 0.5)
    assert plain.run("iamb", 0)[0] == [1, 2, 3]
    assert (fast.counts.issued, plain.counts.issued) == (6, 9)


@pytest.mark.parametrize(("n", "want"), [(50, [1, 2]), (81, [1, 2, 3]), (80, [1, 2])])
def test_fast_iamb_stops_a_batch_where_a_table_has_five_rows_per_cell_or_fewer(n, want):
    """0 -> 1, 0 -> 2, 0 -> 3, each variable with two categories: adding the k-th member needs more
    than 5 * 2**(k + 1) rows, 20, 40, and 80. With fewer rows the batch stops, the shrink drops
    nothing, and the search ends, the variable left out with no separating set."""
    oracle = DSepOracle(nx.DiGraph([(0, 1), (0, 2), (0, 3)]))
    search = LocalSearch(oracle, [0, 1, 2, 3], 0.5, levels=np.full(4, 2), n=n)
    assert search.run("fast_iamb", 0) == (want, {})


class _Table:
    """A test of target 0 that finds each listed (variable, set) independent and the rest
    dependent."""

    def __init__(self, independent: set[tuple[int, tuple[int, ...]]]) -> None:
        self.independent = independent

    def __call__(self, x, y, cond=()):
        other = y if x == 0 else x
        return 1.0 if (other, tuple(sorted(cond))) in self.independent else 0.0


def test_fast_iamb_ends_a_cycle_of_additions_and_removals():
    """Round 1 adds 1 and 2 and the shrink keeps {2}; round 2 adds 3 and the shrink drops 2 and
    then 3; round 3 is round 1 again, and the search ends at the blanket it held after the first
    shrink."""
    independent = {(1, (2,)), (1, (3,)), (2, (1,)), (2, (1, 3)), (2, (3,)), (3, ())}
    independent |= {(3, (1,)), (3, (1, 2))}
    search = LocalSearch(_Table(independent), [0, 1, 2, 3], 0.5)
    assert search.run("fast_iamb", 0) == ([2], {1: (2,), 3: ()})


class _PValues:
    """A test of target 0 that answers each variable with its own p-value, whatever the set."""

    def __init__(self, pvalues: dict[int, float]) -> None:
        self.pvalues = pvalues

    def __call__(self, x, y, cond=()):
        return self.pvalues[y if x == 0 else x]


def test_the_strength_order_keeps_tiny_p_values_apart():
    """p-values below ``10**-KEY_DP`` keep the order of their sizes, the lowest rank first only
    among equal ones; partial correlations compare to ``KEY_DP`` decimals."""
    pvalues = {1: 1e-12, 2: 1e-70, 3: 1e-41, 4: 1e-70, 5: 0.5}
    search = LocalSearch(_PValues(pvalues), [0, 5, 4, 3, 2, 1], 0.05)
    strengths, _ = search.scan(0, [1, 2, 3, 4, 5])
    assert search.by_strength([1, 2, 3, 4, 5], strengths) == [4, 2, 3, 1, 5]
    x, test = _search_data("fisherz")
    search = LocalSearch(test, tie_ranks(x).tolist(), 0.05)
    xs = list(range(1, 15))
    strengths, _ = search.scan(0, xs)
    keys = {v: -round(s, KEY_DP) for v, s in zip(xs, strengths.tolist(), strict=True)}
    assert search.by_strength(xs, strengths) == ordered(keys, search.ranks)


def test_the_order_reads_ranks_only_when_keys_tie():
    keys = {3: 0.5, 1: 0.25, 7: 0.25, 2: 0.25, 5: 0.75}
    ranks = _Recorded([0, 9, 4, 8, 1, 2, 7, 3])
    assert ordered(keys, ranks) == [7, 2, 1, 3, 5]
    assert set(ranks.read) == {1, 7, 2}
    ranks = _Recorded(list(range(8)))
    assert ordered({4: 0.1, 2: 0.3, 6: 0.2}, ranks) == [4, 6, 2]
    assert ranks.read == []
    assert ordered({}, ranks) == []


@pytest.mark.parametrize("seed", range(4))
def test_mmpc_finds_the_true_parents_and_children_under_an_oracle(seed):
    oracle = _oracle(seed)
    ranks = list(range(oracle.d))
    sets = mmpc_sets(oracle, oracle.d, 0.5, ranks)
    np.testing.assert_array_equal(sets & sets.T, oracle.skeleton())
    for v in range(oracle.d):
        # Without the symmetry check, a set may also hold descendants it cannot separate.
        assert set(parents_children(oracle, v, oracle.d, 0.5, ranks)) >= oracle.parents_children(v)


def test_mmpc_symmetry_check_drops_a_descendant_the_target_cannot_separate():
    # t -> c <- b, b -> x, c -> x: only {c, b} separates t and x, and b, independent of t, never
    # joins t's set, so t's set keeps x; x's set separates t, so the skeleton drops the pair.
    t, c, b, x = range(4)
    marks = np.zeros((4, 4), dtype=np.int8)
    for i, j in [(t, c), (b, c), (b, x), (c, x)]:
        marks[i, j], marks[j, i] = 1, 2
    oracle = DSeparation(GraphStructure.from_numpy(marks, kind="dag"))
    ranks = list(range(4))
    assert set(parents_children(oracle, t, 4, 0.5, ranks)) == {c, x}
    sets = mmpc_sets(oracle, 4, 0.5, ranks)
    np.testing.assert_array_equal(sets & sets.T, oracle.skeleton())


@pytest.mark.parametrize("seed", range(4))
def test_hiton_pc_finds_the_true_parents_and_children_under_an_oracle(seed):
    oracle = _oracle(seed)
    search = LocalSearch(oracle, list(range(oracle.d)), 0.5)
    for v in range(oracle.d):
        # Without the symmetry check, a set may also hold descendants it cannot separate.
        assert set(search.run("hiton_pc", v)[0]) >= oracle.parents_children(v), v
        assert set(search.parents_children("hiton_pc", v)) == oracle.parents_children(v), v
    np.testing.assert_array_equal(search.structure("hiton_pc", "skeleton") != 0, oracle.skeleton())


class _Logged:
    """A test that answers as ``test`` does and logs each query in the order asked."""

    def __init__(self, test) -> None:
        self.test = test
        self.log: list[tuple[int, int, tuple[int, ...]]] = []

    def __call__(self, x, y, cond=()):
        self.log.append((x, y, tuple(cond)))
        return self.test(x, y, cond)


def test_hiton_pc_follows_the_papers_trace():
    """The network of Aliferis et al. (2010), Figure 7, and their trace at T (Table 2): with the
    order F, D, E, A, B, and G and C independent of T, each newcomer is tested alone, F leaves only
    in the pass after the last candidate, B, has joined, given {D, E}, and the symmetry check drops
    B, whose own set does not hold T (p. 197)."""
    T, A, B, C, D, E, F, G = range(8)
    edges = [(F, D), (F, E), (D, T), (E, T), (G, A), (T, A), (C, A), (C, B), (A, B)]
    oracle = DSepOracle(nx.DiGraph(edges))
    logged = _Logged(oracle)
    ranks = [5, 3, 4, 7, 1, 2, 0, 6]
    search = LocalSearch(logged, ranks, 0.5)
    assert search.run("hiton_pc", T) == ([D, E, A, B], {C: (), G: (), F: (D, E)})
    asked = [(other, cond) for t, other, cond in logged.log if t == T and cond]
    tested = [other for other, _ in asked]
    # Each newcomer is tested when it joins; F, which joined first against an empty set, is
    # tested only in the final pass, which it opens.
    assert set(tested[: tested.index(F)]) == {D, E, A, B}
    assert (F, (D, E)) in asked
    assert search.parents_children("hiton_pc", T) == [D, E, A]
    assert search.local("hiton_pc", T)[2][B] == (A, C)


class _Statistics:
    """A test of target 0 that answers each variable with its own statistic and p-value, whatever
    the set."""

    def __init__(self, answers: dict[int, tuple[float, float]]) -> None:
        self.answers = answers

    def __call__(self, x, y, cond=()):
        return self.answers[y if x == 0 else x][1]

    def statistic_pvalues(self, x, y, conditions):
        answers = np.array([self.answers[int(b if a == 0 else a)] for a, b in zip(x, y)])
        return answers[:, 0], answers[:, 1]


def test_hiton_pc_orders_by_the_p_value_then_by_the_statistic():
    """p-values below ``10**-KEY_DP`` keep the order of their sizes, and p-values that underflow
    to zero go by the larger statistic, so the ranks, here the reverse, are not read."""
    answers = {1: (500.0, 0.0), 2: (300.0, 1e-70), 3: (800.0, 0.0), 4: (50.0, 1e-12)}
    ranks = _Recorded([0, 4, 3, 2, 1])
    assert LocalSearch(_Statistics(answers), ranks, 0.05).run("hiton_pc", 0)[0] == [3, 1, 2, 4]
    assert ranks.read == []


@pytest.mark.parametrize(
    ("n", "want"),
    [(19, ([], {1: (), 2: (), 3: ()})), (79, ([1, 2, 3], {})), (80, ([1, 2], {3: (1, 2)}))],
)
def test_hiton_pc_runs_a_test_only_with_five_rows_per_cell(n, want):
    """3 -> 1 -> 0 and 3 -> 2 -> 0, every variable with two categories: a test of 0 given ``k``
    variables has ``2**(k + 2)`` cells, so it runs with at least 20, 40, and 80 rows. Below 20 every
    variable is taken as independent of 0 without a test; below 80, {1, 2}, the only set that
    separates 3, is not tested and 3 stays."""
    oracle = DSepOracle(nx.DiGraph([(3, 1), (1, 0), (3, 2), (2, 0)]))
    search = LocalSearch(oracle, [0, 1, 2, 3], 0.5, levels=np.full(4, 2), n=n)
    assert search.run("hiton_pc", 0) == want
    assert (search.counts.issued == 0) == (n < 20)


@pytest.mark.parametrize("seed", range(4))
def test_hiton_mb_finds_the_true_blanket_under_an_oracle(seed):
    oracle = _oracle(seed)
    search = LocalSearch(oracle, list(range(oracle.d)), 0.5)
    for v in range(oracle.d):
        pc, mb, sepsets = search.local("hiton_mb", v)
        assert pc is not None and mb is not None and mb[: len(pc)] == pc, v
        assert set(pc) == oracle.parents_children(v), v
        assert set(mb) == oracle.markov_blanket(v), v
        _check_sepsets(oracle, v, pc, sepsets)
    np.testing.assert_array_equal(search.structure("hiton_mb", "moral") != 0, oracle.moral_graph())


def test_hiton_mb_follows_the_papers_trace():
    """The network of Aliferis et al. (2010), Figure 7, and their trace of HITON-MB at T (pp.
    198-199): PC(T) = {D, E, A}; the candidate spouses are F, G, C, and B; G and C, separated from
    T by the empty set, are dependent on T given {A} and join; F, separated by {D, E}, and B, by
    {A, C}, are paired only with a child already in that set, so they are not tested again and stay
    out. The blanket is {D, E, A, G, C}."""
    T, A, B, C, D, E, F, G = range(8)
    edges = [(F, D), (F, E), (D, T), (E, T), (G, A), (T, A), (C, A), (C, B), (A, B)]
    logged = _Logged(DSepOracle(nx.DiGraph(edges)))
    search = LocalSearch(logged, [5, 3, 4, 7, 1, 2, 0, 6], 0.5)
    pc = search.parents_children("hiton_pc", T)
    for y in pc:
        search.parents_children("hiton_pc", y)
    before = len(logged.log)
    got, mb, sepsets = search.local("hiton_mb", T)
    assert got == pc == [D, E, A]
    assert mb is not None and set(mb[3:]) == {G, C} and mb[:3] == pc
    assert sepsets[F] == (D, E) and sepsets[B] == (A, C) and sepsets[G] == sepsets[C] == ()
    assert sorted(logged.log[before:]) == [(T, C, (A,)), (T, G, (A,))]


@pytest.mark.parametrize(("n", "want"), [(19, []), (39, [1]), (40, [1, 2])])
def test_hiton_mb_runs_a_spouse_test_only_with_five_rows_per_cell(n, want):
    """0 -> 1 <- 2, every variable with two categories: 2 is 0's spouse, separated from 0 by the
    empty set. The spouse test of 0 and 2 given {1} has 8 cells, so it runs with at least 40 rows;
    below 40, 2 does not join. Below 20, no unconditional test runs and the blanket is empty."""
    oracle = DSepOracle(nx.DiGraph([(0, 1), (2, 1)]))
    search = LocalSearch(oracle, [0, 1, 2], 0.5, levels=np.full(3, 2), n=n)
    assert search.local("hiton_mb", 0)[1] == want


@pytest.mark.parametrize(("max_depth", "joins"), [(1, False), (2, True), (None, True)])
def test_hiton_mb_does_not_test_a_spouse_past_max_depth(max_depth, joins):
    """W -> X, W -> T, X -> Y <- T: X is T's spouse, separated from T by {W}, so its test
    conditions on {W, Y}, which a cap of 1 does not allow: X joins only under a larger cap."""
    w, x, t, y = range(4)
    oracle = DSepOracle(nx.DiGraph([(w, x), (w, t), (x, y), (t, y)]))
    search = LocalSearch(oracle, [0, 1, 2, 3], 0.5, max_depth=max_depth)
    pc, mb, sepsets = search.local("hiton_mb", t)
    assert pc is not None and set(pc) == {w, y} and sepsets[x] == (w,)
    assert mb is not None and (x in mb) == joins


def test_independent_columns_have_empty_sets():
    x = np.random.default_rng(0).standard_normal((500, 6))
    test = FisherZ(x)
    ranks = tie_ranks(x).tolist()
    assert not blanket_graph(test, 6, 0.01, ranks).any()
    sets = mmpc_sets(test, 6, 0.01, ranks)
    assert not (sets & sets.T).any()


def _colliders_on_every_set(test, skeleton, blankets, alpha):
    """Each unshielded triple tested on every set of each size, by increasing size, up to the first
    size with a separating set: the separating sets and the collider ranks."""
    sepsets, ranks = {}, {}
    for b in range(skeleton.shape[0]):
        for a, c in combinations(np.flatnonzero(skeleton[b]).tolist(), 2):
            if skeleton[a, c]:
                continue
            sepsets.setdefault((a, c), ())
            sides = blanket._smaller(blankets, a, c, (b,))
            p, strength = -np.inf, np.inf
            for size in range(max(len(side) for side in sides) + 1):
                sets = list(blanket._collider_sets(sides, size, b))
                [(p_size, strength_size)] = blanket._weakest(test, a, [c], sets)
                p, strength = max(p, p_size), min(strength, strength_size)
                if p > alpha:
                    sepsets[a, c] = (*sepsets[a, c], b)
                    break
            else:
                ranks[a, b, c] = strength
    return sepsets, ranks


def _search_data(kind):
    """Data and a test of 15 variables: Gaussian with Fisher-z, discrete with the chi-square
    test, and discrete with a twentieth of the values missing, whose test deletes rows per test."""
    if kind == "fisherz":
        x = data.sample_scm(graph="erdos_renyi", d=15, n=1000, seed=2, density=4.0).data
        return x, FisherZ(x)
    x = data.sample_discrete(d=15, n=3000, seed=2, density=4.0).data
    if kind == "chisq":
        return x, ChiSquare(x)
    x = x.copy()
    x[np.random.default_rng(0).random(x.shape) < 0.05] = np.nan
    return x, make_indep_test("chisq", x)


@pytest.mark.parametrize("kind", ["fisherz", "chisq", "missing"])
@pytest.mark.parametrize(("first", "chunk"), [(1, 4096), (1, 2), (4, 8)])
def test_a_non_collider_stops_at_its_first_separating_chunk(monkeypatch, kind, first, chunk):
    """Every triple is decided, and every collider ranked, as testing every set of each size up to
    the first size that separates decides and ranks it, with fewer sets tested; a test that
    deletes rows per test tests the same sets, so the rows it reports are the same."""
    monkeypatch.setattr(blanket, "_FIRST_CHUNK", first)
    monkeypatch.setattr(blanket, "CHUNK", chunk)
    x, test = _search_data(kind)
    sets = mmpc_sets(test, 15, 0.05, tie_ranks(x).tolist())
    skeleton, blankets = sets & sets.T, reach(sets)
    tested = []
    weakest = blanket._weakest

    def counted(test, target, xs, conds):
        tested.append(len(conds))
        return weakest(test, target, xs, conds)

    monkeypatch.setattr(blanket, "_weakest", counted)
    with rows_tested() as rows:
        sepsets, ranks, ambiguous = blanket_colliders(test, skeleton, blankets, 0.05)
    found = sum(tested)
    tested.clear()
    with rows_tested() as want_rows:
        want_sepsets, want_ranks = _colliders_on_every_set(test, skeleton, blankets, 0.05)
    assert sepsets == want_sepsets
    assert {t: ranks[t] for t in want_ranks} == want_ranks
    assert not ambiguous
    assert rows == want_rows
    assert found == sum(tested) if kind == "missing" else found < sum(tested)


def _mmpc_on_every_set(test, target, d, alpha, ranks):
    """MMPC with every remaining candidate tested against every set at each addition, then the
    backward phase of :func:`parents_children`."""
    weakest = {x: blanket._weakest(test, target, [x], [()])[0] for x in range(d) if x != target}
    members: list[int] = []
    while open_ := {x: w for x, w in weakest.items() if w[0] <= alpha}:
        best = min(
            open_,
            key=lambda x: (round(open_[x][0], KEY_DP), -round(open_[x][1], KEY_DP), ranks[x]),
        )
        del open_[best]
        conds = [
            tuple(sorted((*rest, best)))
            for size in range(len(members) + 1)
            for rest in combinations(members, size)
        ]
        for x, (p, strength) in open_.items():
            [(p_x, strength_x)] = blanket._weakest(test, target, [x], conds)
            open_[x] = (max(p, p_x), min(strength, strength_x))
        members.append(best)
        weakest = open_
    for x in list(members):
        before = set(members[: members.index(x)])
        rest = sorted(z for z in members if z != x)
        for size in range(1, len(rest) + 1):
            if scan_conditioning_sets(test, target, x, rest, size, alpha, set(), before):
                members.remove(x)
                break
    return members


@pytest.mark.parametrize("kind", ["fisherz", "chisq", "missing"])
def test_mmpc_stops_testing_a_candidate_once_a_set_separates_it(kind):
    """Each variable's MMPC set, in joining order, is the one found by testing every remaining
    candidate against every set at each addition; a test that deletes rows per test tests the
    same sets, so the rows it reports are the same."""
    x, test = _search_data(kind)
    ranks = tie_ranks(x).tolist()
    for target in range(15):
        with rows_tested() as want_rows:
            want = _mmpc_on_every_set(test, target, 15, 0.05, ranks)
        with rows_tested() as rows:
            assert parents_children(test, target, 15, 0.05, ranks) == want, target
        assert rows == want_rows, target


def _fresh(method, test, target, d, alpha, ranks):
    """``target``'s run by ``method`` made anew: the set and the separating sets."""
    sepsets: dict[int, tuple[int, ...]] = {}
    if method in ("gs", "fast_iamb", "hiton_pc"):
        assert d == len(ranks)
        return LocalSearch(test, ranks, alpha).run(method, target)
    if method == "mmpc":
        found = parents_children(test, target, d, alpha, ranks, sepsets=sepsets)
    else:
        interleaved = method == "inter_iamb"
        found = markov_blanket(
            test, target, d, alpha, ranks, interleaved=interleaved, sepsets=sepsets
        )
    return found, sepsets


class _NoTests:
    """A test that fails on any use: a memoized answer must not reach it."""

    def __getattr__(self, name):
        raise AssertionError(f"test.{name} used")

    def __call__(self, *args):
        raise AssertionError("test called")


@pytest.mark.parametrize("kind", ["fisherz", "chisq", "missing"])
def test_a_memoized_run_is_the_run_made_afresh_and_tests_nothing_again(kind):
    """Each run of a local search, asked for every target in a shuffled order, is the run made
    afresh; the parents and children are the members whose own run holds the target; and asking
    again tests nothing."""
    x, test = _search_data(kind)
    ranks = tie_ranks(x).tolist()
    search = LocalSearch(test, ranks, 0.05)
    order = np.random.default_rng(0).permutation(15).tolist()
    want = {}
    for method in ("iamb", "inter_iamb", "gs", "fast_iamb", "mmpc", "hiton_pc"):
        for t in order:
            want[method, t] = search.local(method, t)
            assert search.run(method, t) == _fresh(method, test, t, 15, 0.05, ranks), (method, t)
        for t in order:
            members, _ = _fresh(method, test, t, 15, 0.05, ranks)
            if LEARNERS[method].kind == "pc":
                symmetric = [v for v in members if t in _fresh(method, test, v, 15, 0.05, ranks)[0]]
                assert want[method, t][0] == symmetric, t
            else:
                assert want[method, t][1] == members, t
    for method in ("hiton_mb", "mmmb"):
        for t in order:
            want[method, t] = search.local(method, t)
            assert want[method, t] == LocalSearch(test, ranks, 0.05).local(method, t), (method, t)
    search.test = _NoTests()
    for (method, t), local in want.items():
        assert search.local(method, t) == local


def _check_sepsets(oracle: DSeparation, t: int, learned: list[int], sepsets) -> None:
    """Every other variable outside ``learned`` has a recorded set, and each set separates."""
    assert set(sepsets) == set(range(oracle.d)) - set(learned) - {t}, t
    for v, given in sepsets.items():
        assert oracle(t, v, given) == 1.0, (t, v, given)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("method", ["iamb", "inter_iamb", "gs", "fast_iamb", "mmpc", "hiton_pc"])
def test_each_recorded_separating_set_separates_under_an_oracle(method, seed):
    oracle = _oracle(seed)
    search = LocalSearch(oracle, list(range(oracle.d)), 0.5)
    for t in range(oracle.d):
        pc, mb, sepsets = search.local(method, t)
        learned = mb if pc is None else pc
        assert learned is not None
        _check_sepsets(oracle, t, learned, sepsets)
        members, own = search.run(method, t)
        assert set(own) <= set(range(oracle.d)) - set(members) - {t}
        assert all(oracle(t, v, given) == 1.0 for v, given in own.items())


@pytest.mark.parametrize("kind", ["fisherz", "chisq"])
@pytest.mark.parametrize("method", ["iamb", "inter_iamb", "gs", "fast_iamb", "mmpc", "hiton_pc"])
def test_each_recorded_separating_set_is_a_test_that_separated(method, kind):
    """On data, the test given a recorded set has a p-value above alpha."""
    x, test = _search_data(kind)
    search = LocalSearch(test, tie_ranks(x).tolist(), 0.05)
    for t in range(15):
        for v, given in search.local(method, t)[2].items():
            assert test(t, v, given) > 0.05, (t, v, given)


@pytest.mark.parametrize("seed", range(4))
def test_mmmb_finds_the_true_blanket_under_an_oracle(seed):
    dag = data.sample_scm(graph="erdos_renyi", d=12, n=2, seed=seed, density=3.0).graph
    oracle = DSeparation(dag)
    search = LocalSearch(oracle, list(range(oracle.d)), 0.5)
    for t in range(oracle.d):
        pc, mb, sepsets = search.local("mmmb", t)
        assert pc is not None and mb is not None
        assert pc == search.parents_children("mmpc", t), t
        assert set(pc) == oracle.parents_children(t), t
        assert set(mb) == oracle.markov_blanket(t) and mb[: len(pc)] == pc, t
        _check_sepsets(oracle, t, pc, sepsets)
    np.testing.assert_array_equal(search.structure("mmmb", "moral") != 0, oracle.moral_graph())
    np.testing.assert_array_equal(search.structure("mmmb", "skeleton") != 0, oracle.skeleton())
    truth = dag2cpdag(dag.to_numpy())
    np.testing.assert_array_equal(search.structure("mmmb", "cpdag"), truth)
    search.test = _NoTests()
    assert search.local("mmmb", 0)[1] is not None


def _dag(edges: list[tuple[int, int]], d: int) -> DSeparation:
    marks = np.zeros((d, d), dtype=np.int8)
    for i, j in edges:
        marks[i, j], marks[j, i] = 1, 2
    return DSeparation(GraphStructure.from_numpy(marks, kind="dag"))


def test_mmmb_pairs_a_candidate_only_with_its_neighbor():
    """Pena et al. (2007), DAG (b): on P -> Q -> T -> S <- R <- P, T's blanket is {Q, S, R}. P is
    separated from T by {Q} and is dependent on T given {Q, S}, so the 2003 MMMB, which tests P
    with every member of PC(T) = {Q, S}, admits it. Q is P's only neighbor in PC(T) and lies in
    P's separating set, so P is never tested again; R, separated from T by {P}, joins through S."""
    p, q, t, s, r = range(5)
    oracle = _dag([(p, q), (q, t), (t, s), (r, s), (p, r)], 5)
    logged = _Logged(oracle)
    search = LocalSearch(logged, list(range(5)), 0.5)
    pc = search.parents_children("mmpc", t)
    for y in pc:
        search.parents_children("mmpc", y)
    before = len(logged.log)
    got, mb, sepsets = search.local("mmmb", t)
    assert got == pc and set(pc) == {q, s}
    assert mb is not None and mb[:2] == pc and mb[2:] == [r]
    assert sepsets[p] == (q,) and sepsets[r] == (p,)
    assert oracle(t, p, (q, s)) == 0.0
    assert logged.log[before:] == [(t, r, (p, s))]


def test_mmmb_runs_the_symmetry_check_of_mmpc():
    """Pena et al. (2007), DAG (a): on T -> Q <- P, P -> R, Q -> S <- R, T's blanket is {Q, P}.
    No subset of MMPC's set {Q, S} separates the grandchild S from T (only {Q, P} and {Q, R} do),
    so the 2003 MMMB, without the symmetry check, keeps S. S's own set does not hold T, so the
    check drops S, and its separating set comes from S's run."""
    t, q, p, r, s = range(5)
    oracle = _dag([(t, q), (p, q), (p, r), (q, s), (r, s)], 5)
    search = LocalSearch(oracle, list(range(5)), 0.5)
    assert set(search.run("mmpc", t)[0]) == {q, s}
    pc, mb, sepsets = search.local("mmmb", t)
    assert pc == [q] and mb == [q, p]
    assert oracle(t, s, sepsets[s]) == 1.0 and sepsets[s] == search.run("mmpc", s)[1][t]


_ALL = ["iamb", "inter_iamb", "gs", "fast_iamb", "mmpc", "hiton_pc", "hiton_mb", "mmmb"]


@pytest.mark.parametrize("kind", ["fisherz", "chisq"])
@pytest.mark.parametrize("method", _ALL)
def test_the_local_entry_gives_the_sets_the_method_returns(method, kind):
    """The parents and children and the Markov blanket, each the method learns, are the sets its
    target fit reports; a set it does not learn is ``None`` and absent from the metadata."""
    x, _ = _search_data(kind)
    indep_test = "fisherz" if kind == "fisherz" else "chisq"
    search = local_search(x, alpha=0.05, indep_test=indep_test)
    fit = partial(getattr(andrey, method), **_UNCAPPED.get(method, {}))
    for t in range(15):
        pc, mb, _ = search.local(method, t)
        got = fit(x, target=t, indep_test=indep_test).metadata
        assert (pc, mb) == (got.get("parents_and_children"), got.get("markov_blanket")), t
        assert (pc is None) == (LEARNERS[method].kind == "blanket"), t
        assert (mb is None) == (LEARNERS[method].kind == "pc"), t


@pytest.mark.parametrize("kind", ["fisherz", "chisq"])
@pytest.mark.parametrize("method", _ALL)
def test_a_column_shuffle_leaves_every_recorded_set(method, kind):
    x, _ = _search_data(kind)
    indep_test = "fisherz" if kind == "fisherz" else "chisq"
    perm = np.random.default_rng(1).permutation(15)
    search = local_search(x, alpha=0.05, indep_test=indep_test)
    shuffled = local_search(x[:, perm], alpha=0.05, indep_test=indep_test)
    where = np.argsort(perm)  # where[v]: the shuffled position of column v

    def back(found):
        return None if found is None else [int(perm[v]) for v in found]

    for t in range(15):
        pc, mb, sepsets = search.local(method, t)
        spc, smb, ssepsets = shuffled.local(method, int(where[t]))
        assert (back(spc), back(smb)) == (pc, mb), t
        moved = {int(perm[v]): tuple(sorted(int(perm[u]) for u in g)) for v, g in ssepsets.items()}
        assert moved == sepsets, t


#: The tests whose scan answers many candidates in one call, by the data each reads.
_BATCHED = {
    "fisherz": "gaussian",
    "student_t": "gaussian",
    "normal_scores": "gaussian",
    "fisherz-missing": "gaussian",
    "normal_scores-missing": "gaussian",
    "chisq": "discrete",
    "gsq": "discrete",
}


def _scan_data(kind: str, seed: int = 3) -> np.ndarray:
    """Twelve columns of the data ``kind`` reads, a twentieth of the values missing for a
    ``-missing`` kind."""
    if _BATCHED[kind] == "gaussian":
        x = data.sample_scm(graph="erdos_renyi", d=12, n=400, seed=seed, density=3.0).data
    else:
        x = data.sample_discrete(d=12, n=1500, seed=seed, density=3.0).data
    if kind.endswith("-missing"):
        x = x.copy()
        x[np.random.default_rng(seed).random(x.shape) < 0.05] = np.nan
    return x


@pytest.mark.parametrize("kind", list(_BATCHED))
def test_a_scan_is_its_candidates_asked_one_at_a_time_bit_for_bit(kind):
    """The scan of many candidates, answered in one call, gives each candidate the bits of the scan
    of that candidate alone and of the test's own answer, in the order of the candidates."""
    x = _scan_data(kind)
    test = make_indep_test(kind.removesuffix("-missing"), x)
    assert hasattr(test, "statistic_pvalues" if _BATCHED[kind] == "gaussian" else "pvalues_given")
    search = LocalSearch(test, tie_ranks(x).tolist(), 0.05)
    target = 4
    for cond in [(), (7,), (2, 9), (0, 3, 8, 11)]:
        xs = [v for v in range(12) if v != target and v not in cond]
        strengths, pvalues = search.scan(target, xs, cond)
        alone = [search.scan(target, [v], cond) for v in xs]
        assert strengths.tobytes() == np.concatenate([s for s, _ in alone]).tobytes(), cond
        assert pvalues.tobytes() == np.concatenate([p for _, p in alone]).tobytes(), cond
        assert pvalues.tolist() == [test(target, v, cond) for v in xs], cond
        backward = search.scan(target, xs[::-1], cond)
        assert backward[1].tobytes() == pvalues[::-1].tobytes(), cond


@pytest.mark.parametrize("kind", ["fisherz", "chisq"])
def test_a_scan_does_not_depend_on_the_column_order(kind):
    """On the columns shuffled, each candidate's strength and p-value agree to rounding and fall on
    the same side of alpha, and the strongest candidate is the same column."""
    x = _scan_data(kind)
    perm = np.random.default_rng(5).permutation(12)
    where = np.argsort(perm)  # where[v]: the shuffled position of column v
    search = local_search(x, alpha=0.05, indep_test=kind)
    shuffled = local_search(x[:, perm], alpha=0.05, indep_test=kind)
    target = 4
    for cond in [(), (7,), (2, 9), (0, 3, 8, 11)]:
        xs = [v for v in range(12) if v != target and v not in cond]
        strengths, pvalues = search.scan(target, xs, cond)
        moved = shuffled.scan(int(where[target]), where[xs].tolist(), where[list(cond)].tolist())
        np.testing.assert_allclose(moved[0], strengths, rtol=1e-9, atol=1e-300)
        np.testing.assert_allclose(moved[1], pvalues, rtol=1e-9, atol=1e-300)
        assert ((moved[1] <= 0.05) == (pvalues <= 0.05)).all(), cond
        keys = {v: -round(s, KEY_DP) for v, s in zip(xs, strengths.tolist(), strict=True)}
        keys_moved = {
            int(where[v]): -round(s, KEY_DP) for v, s in zip(xs, moved[0].tolist(), strict=True)
        }
        assert perm[shuffled.strongest(keys_moved)] == search.strongest(keys), cond


class _Recorded(list):
    """Ranks that record each position read."""

    def __init__(self, ranks):
        super().__init__(ranks)
        self.read: list[int] = []

    def __getitem__(self, i):
        self.read.append(i)
        return super().__getitem__(i)


@pytest.mark.parametrize("method", _ALL)
def test_ranks_are_read_only_when_keys_tie(method):
    """On continuous data no two strengths tie, so the search reads no rank and finds the sets it
    finds with the ranks read; under an oracle every dependent candidate ties, and the ranks read
    are those of tied candidates."""
    x, test = _search_data("fisherz")
    ranks = _Recorded(tie_ranks(x).tolist())
    search = LocalSearch(test, ranks, 0.05)
    plain = LocalSearch(test, list(ranks), 0.05)
    for t in range(15):
        assert search.local(method, t) == plain.local(method, t), t
    assert ranks.read == []
    keys = {3: 0.5, 1: 0.25, 7: 0.25, 2: 0.25, 5: 0.75}
    ranks = _Recorded([0, 9, 4, 8, 1, 2, 7, 3])
    assert strongest(keys, ranks) == 7
    assert set(ranks.read) <= {1, 7, 2}
    with pytest.raises(ValueError, match="no candidate"):
        strongest({}, ranks)
    oracle = _oracle(0)
    ranks = _Recorded(list(range(oracle.d)))
    LocalSearch(oracle, ranks, 0.5).local(method, 0)
    assert ranks.read


def test_a_search_counts_queries_issued_computed_and_distinct():
    """On the chain 0 -> 1 -> 2 under a memoizing oracle, IAMB at 0 asks (0, 1) and (0, 2) given
    nothing, then (0, 2) given 1, which separates them, and its shrink asks (0, 1) given nothing
    again, which the oracle answers from its memo: 4 issued, 3 distinct, 3 computed."""
    oracle = DSepOracle(nx.DiGraph([(0, 1), (1, 2)]))
    search = LocalSearch(oracle, [0, 1, 2], 0.5, trace=True)
    assert search.run("iamb", 0) == ([1], {2: (1,)})
    counts = search.counts
    assert (counts.issued, counts.distinct, counts.computed) == (4, 3, 3)
    assert counts.queries == {(0, 1, ()), (0, 2, ()), (0, 2, (1,))}
    assert 0.0 <= counts.test_seconds <= counts.search_seconds
    search.run("iamb", 0)
    assert (counts.issued, counts.computed) == (4, 3)


@pytest.mark.parametrize("method", _ALL)
def test_n_tests_is_the_count_of_tests_computed(method):
    """A target fit's ``n_tests`` is the engine's computed count; a test without kept answers
    computes every query it is asked, and a repeated query counts once among the distinct."""
    x, _ = _search_data("fisherz")
    for t in (0, 6):
        search = local_search(x, alpha=0.05, indep_test="fisherz", trace=True)
        search.local(method, t)
        counts = search.counts
        fit = getattr(andrey, method)(
            x, target=t, indep_test="fisherz", **_UNCAPPED.get(method, {})
        )
        assert fit.metadata["n_tests"] == counts.computed == counts.issued, t
        assert counts.distinct is not None and counts.distinct <= counts.issued
    x, _ = _search_data("chisq")
    search = local_search(x, alpha=0.05, indep_test="chisq")
    search.local(method, 0)
    fit = getattr(andrey, method)(x, target=0, indep_test="chisq", **_UNCAPPED.get(method, {}))
    assert fit.metadata["n_tests"] == search.counts.computed <= search.counts.issued
    assert search.counts.distinct is None


def test_table_size_counts_cells_and_degrees_of_freedom():
    levels = [2, 3, 4, 0]
    assert table_size(levels, 0, 1, [2]) == 24
    assert table_size(levels, 0, 1, [2], df=True) == 8
    assert table_size(levels, 1, 2, []) == 12
    assert table_size(levels, 0, 1, [3]) is None
    x = data.sample_discrete(d=4, n=500, seed=0).data
    want = [np.unique(c).size for c in x.T]
    assert local_search(x, alpha=0.05, indep_test="chisq").levels.tolist() == want
    assert local_search(x, alpha=0.05, indep_test="fisherz").levels is None


@pytest.mark.parametrize(
    "row",
    [
        {"kind": "pc_spouses"},
        {"kind": "pc_spouses", "pc": "mmpc", "search": blanket._mmpc},
        {"kind": "pc"},
        {"kind": "blanket", "search": blanket._mmpc, "pc": "mmpc"},
        {"kind": "pc", "search": blanket._mmpc, "cell_rows": 5},
    ],
)
def test_a_learner_row_names_one_search_or_one_pc_learner(row):
    with pytest.raises(ValueError, match="pc_spouses"):
        Learner(**row)
