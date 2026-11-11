"""The shared Markov-blanket engine: IAMB, Inter-IAMB, and MMPC, and the local search over them.

Under a d-separation oracle each search returns the true set, each recorded separating set
separates, and the spouse step completes the parents and children into the Markov blanket. With a
Fisher-Z test, IAMB returns, bit for bit, the blankets of the Fisher-Z IAMB that CALM's moral graph
is built from. A memoized run is the run made afresh, and the local search gives the sets the
methods return, whatever the column order.
"""

from __future__ import annotations

from itertools import combinations

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
    parents_children,
    reach,
    table_size,
)
from andrey.core.ci import ChiSquare, FisherZ, make_indep_test
from andrey.core.ci_missing import rows_tested
from andrey.core.orient import dag2cpdag
from andrey.core.ties import KEY_DP, tie_ranks
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
    for method in ("iamb", "inter_iamb", "mmpc"):
        for t in order:
            want[method, t] = search.local(method, t)
            assert search.run(method, t) == _fresh(method, test, t, 15, 0.05, ranks), (method, t)
        for t in order:
            members, _ = _fresh(method, test, t, 15, 0.05, ranks)
            if method == "mmpc":
                symmetric = [v for v in members if t in _fresh(method, test, v, 15, 0.05, ranks)[0]]
                assert want[method, t][0] == symmetric, t
            else:
                assert want[method, t][1] == members, t
    search.test = _NoTests()
    for (method, t), local in want.items():
        assert search.local(method, t) == local


def _check_sepsets(oracle: DSeparation, t: int, learned: list[int], sepsets) -> None:
    """Every other variable outside ``learned`` has a recorded set, and each set separates."""
    assert set(sepsets) == set(range(oracle.d)) - set(learned) - {t}, t
    for v, given in sepsets.items():
        assert oracle(t, v, given) == 1.0, (t, v, given)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("method", ["iamb", "inter_iamb", "mmpc"])
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
@pytest.mark.parametrize("method", ["iamb", "inter_iamb", "mmpc"])
def test_each_recorded_separating_set_is_a_test_that_separated(method, kind):
    """On data, the test given a recorded set has a p-value above alpha."""
    x, test = _search_data(kind)
    search = LocalSearch(test, tie_ranks(x).tolist(), 0.05)
    for t in range(15):
        for v, given in search.local(method, t)[2].items():
            assert test(t, v, given) > 0.05, (t, v, given)


@pytest.fixture
def mmpc_spouses(monkeypatch):
    """A parents-and-children-then-spouses learner over MMPC, registered for the test."""
    monkeypatch.setitem(LEARNERS, "mmpc_spouses", Learner("pc_spouses", pc="mmpc"))
    return "mmpc_spouses"


@pytest.mark.parametrize("seed", range(4))
def test_the_spouse_step_completes_the_blanket_under_an_oracle(mmpc_spouses, seed):
    dag = data.sample_scm(graph="erdos_renyi", d=12, n=2, seed=seed, density=3.0).graph
    oracle = DSeparation(dag)
    search = LocalSearch(oracle, list(range(oracle.d)), 0.5)
    for t in range(oracle.d):
        pc, mb, sepsets = search.local(mmpc_spouses, t)
        assert pc is not None and mb is not None
        assert set(pc) == oracle.parents_children(t), t
        assert set(mb) == oracle.markov_blanket(t) and mb[: len(pc)] == pc, t
        _check_sepsets(oracle, t, pc, sepsets)
    np.testing.assert_array_equal(
        search.structure(mmpc_spouses, "moral") != 0, oracle.moral_graph()
    )
    np.testing.assert_array_equal(
        search.structure(mmpc_spouses, "skeleton") != 0, oracle.skeleton()
    )
    truth = dag2cpdag(dag.to_numpy())
    np.testing.assert_array_equal(search.structure(mmpc_spouses, "cpdag"), truth)
    search.test = _NoTests()
    assert search.local(mmpc_spouses, 0)[1] is not None


def test_the_spouse_step_pairs_a_candidate_only_with_its_neighbor(mmpc_spouses):
    """On P -> Q -> T -> S <- R <- P, T's blanket is {Q, S, R}: P, separated from T by {Q}, is
    dependent on T given {Q, S}, but S is not adjacent to P, so that test is never run (Pena et
    al. 2007, DAG (b))."""
    p, q, t, s, r = range(5)
    marks = np.zeros((5, 5), dtype=np.int8)
    for i, j in [(p, q), (q, t), (t, s), (r, s), (p, r)]:
        marks[i, j], marks[j, i] = 1, 2
    oracle = DSeparation(GraphStructure.from_numpy(marks, kind="dag"))
    search = LocalSearch(oracle, list(range(5)), 0.5)
    pc, mb, sepsets = search.local(mmpc_spouses, t)
    assert pc == [q, s] or pc == [s, q]
    assert mb is not None and set(mb) == {q, s, r}
    assert oracle(t, p, (q, s)) == 0.0 and q in sepsets[p]


@pytest.mark.parametrize("kind", ["fisherz", "chisq"])
@pytest.mark.parametrize("method", ["iamb", "inter_iamb", "mmpc"])
def test_the_local_entry_gives_the_set_the_method_returns(method, kind):
    x, _ = _search_data(kind)
    indep_test = "fisherz" if kind == "fisherz" else "chisq"
    search = local_search(x, alpha=0.05, indep_test=indep_test)
    fit = getattr(andrey, method)
    key = "parents_and_children" if method == "mmpc" else "markov_blanket"
    for t in range(15):
        pc, mb, _ = search.local(method, t)
        assert (pc if method == "mmpc" else mb) == fit(x, target=t, indep_test=indep_test).metadata[
            key
        ], t
        assert (mb if method == "mmpc" else pc) is None


@pytest.mark.parametrize("kind", ["fisherz", "chisq"])
@pytest.mark.parametrize("method", ["iamb", "inter_iamb", "mmpc"])
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
    ],
)
def test_a_learner_row_names_one_search_or_one_pc_learner(row):
    with pytest.raises(ValueError, match="pc_spouses"):
        Learner(**row)
