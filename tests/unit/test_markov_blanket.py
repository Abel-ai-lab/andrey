"""The shared Markov-blanket engine: IAMB, Inter-IAMB, and MMPC.

Under a d-separation oracle each search returns the true set. With a Fisher-Z test, IAMB returns,
bit for bit, the blankets of the Fisher-Z IAMB that CALM's moral graph is built from.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from andrey import GraphStructure, data
from andrey.constraint import blanket
from andrey.constraint._skeleton import scan_conditioning_sets
from andrey.constraint.blanket import (
    blanket_colliders,
    blanket_graph,
    markov_blanket,
    mmpc_sets,
    mmpc_skeleton,
    parents_children,
    reach,
)
from andrey.core.ci import ChiSquare, FisherZ
from andrey.search._parallel import _KEY_DP
from andrey.search.calm import moral_graph
from andrey.search.hc import tie_ranks
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
            key = (-round(abs(r), _KEY_DP), ranks[x])
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
    skeleton = mmpc_skeleton(oracle, oracle.d, 0.5, ranks)
    np.testing.assert_array_equal(skeleton, oracle.skeleton())
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
    np.testing.assert_array_equal(mmpc_skeleton(oracle, 4, 0.5, ranks), oracle.skeleton())


def test_independent_columns_have_empty_sets():
    x = np.random.default_rng(0).standard_normal((500, 6))
    test = FisherZ(x)
    ranks = tie_ranks(x).tolist()
    assert not blanket_graph(test, 6, 0.01, ranks).any()
    assert not mmpc_skeleton(test, 6, 0.01, ranks).any()


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


@pytest.mark.parametrize("kind", ["fisherz", "chisq"])
@pytest.mark.parametrize(("first", "chunk"), [(1, 4096), (1, 2), (4, 8)])
def test_a_non_collider_stops_at_its_first_separating_chunk(monkeypatch, kind, first, chunk):
    """Every triple is decided, and every collider ranked, as testing every set of each size up to
    the first size that separates decides and ranks it, with fewer sets tested."""
    monkeypatch.setattr(blanket, "_FIRST_CHUNK", first)
    monkeypatch.setattr(blanket, "_CHUNK", chunk)
    if kind == "fisherz":
        x = data.sample_scm(graph="erdos_renyi", d=15, n=1000, seed=2, density=4.0).data
        test = FisherZ(x)
    else:
        x = data.sample_discrete(d=15, n=3000, seed=2, density=4.0).data
        test = ChiSquare(x)
    sets = mmpc_sets(test, 15, 0.05, tie_ranks(x).tolist())
    skeleton, blankets = sets & sets.T, reach(sets)
    tested = []
    weakest = blanket._weakest

    def counted(test, target, xs, conds):
        tested.append(len(conds))
        return weakest(test, target, xs, conds)

    monkeypatch.setattr(blanket, "_weakest", counted)
    sepsets, ranks, ambiguous = blanket_colliders(test, skeleton, blankets, 0.05)
    found = sum(tested)
    tested.clear()
    want_sepsets, want_ranks = _colliders_on_every_set(test, skeleton, blankets, 0.05)
    assert sepsets == want_sepsets
    assert {t: ranks[t] for t in want_ranks} == want_ranks
    assert not ambiguous
    assert found < sum(tested)


def _mmpc_on_every_set(test, target, d, alpha, ranks):
    """MMPC with every remaining candidate tested against every set at each addition, then the
    backward phase of :func:`parents_children`."""
    weakest = {x: blanket._weakest(test, target, [x], [()])[0] for x in range(d) if x != target}
    members: list[int] = []
    while open_ := {x: w for x, w in weakest.items() if w[0] <= alpha}:
        best = min(
            open_,
            key=lambda x: (round(open_[x][0], _KEY_DP), -round(open_[x][1], _KEY_DP), ranks[x]),
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


@pytest.mark.parametrize("kind", ["fisherz", "chisq"])
def test_mmpc_stops_testing_a_candidate_once_a_set_separates_it(kind):
    """Each variable's MMPC set, in joining order, is the one found by testing every remaining
    candidate against every set at each addition."""
    if kind == "fisherz":
        x = data.sample_scm(graph="erdos_renyi", d=15, n=1000, seed=2, density=4.0).data
        test = FisherZ(x)
    else:
        x = data.sample_discrete(d=15, n=3000, seed=2, density=4.0).data
        test = ChiSquare(x)
    ranks = tie_ranks(x).tolist()
    for target in range(15):
        want = _mmpc_on_every_set(test, target, 15, 0.05, ranks)
        assert parents_children(test, target, 15, 0.05, ranks) == want, target
