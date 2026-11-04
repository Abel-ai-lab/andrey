"""The permutation engine of ``andrey.search.igsp`` under oracle and stub tests."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from andrey.core import ARROW, TAIL
from andrey.core.knowledge import Knowledge, resolve
from andrey.core.structure import GraphStructure
from andrey.search import igsp
from andrey.search.gies import cut_pairs, replace_unprotected


def marks(parents: list[list[int]]) -> np.ndarray:
    """The mark matrix of the DAG with ``parents[j]`` the parents of ``j``."""
    n = len(parents)
    adj = np.zeros((n, n), dtype=np.int8)
    for j, pa in enumerate(parents):
        for i in pa:
            adj[i, j], adj[j, i] = TAIL, ARROW
    return adj


def essential(parents: list[list[int]], family: list[set[int]]) -> np.ndarray:
    """The interventional essential graph of the DAG under the observational setting and
    ``family``."""
    mask = np.zeros((len(family) + 1, len(parents)), dtype=bool)
    for k, t in enumerate(family, start=1):
        mask[k, sorted(t)] = True
    return replace_unprotected(marks(parents), cut_pairs(mask))


def as_lists(found: tuple[int, ...]) -> list[list[int]]:
    return [[i for i in range(len(found)) if pa >> i & 1] for pa in found]


def oracle_case(seed: int):
    """A random DAG on 4-8 variables, 1-4 settings of 1-3 targets each, some known.

    Returns the parents, each setting's targets and known targets, and the d-separation oracles
    of the interventional DAG: one context node per setting into its targets, all of them given
    in the observational distribution.
    """
    rng = np.random.default_rng(seed)
    p = int(rng.integers(4, 9))
    order = rng.permutation(p)
    parents: list[list[int]] = [[] for _ in range(p)]
    for a, b in itertools.combinations(range(p), 2):
        if rng.random() < 0.4:
            parents[order[b]].append(int(order[a]))
    targets, known = [], []
    for _ in range(int(rng.integers(1, 5))):
        t = {int(v) for v in rng.choice(p, size=int(rng.integers(1, 4)), replace=False)}
        targets.append(t)
        known.append(frozenset(v for v in t if rng.random() < 0.5))
    K = len(targets)
    augmented = [list(pa) for pa in parents] + [[] for _ in range(K)]
    for k, t in enumerate(targets):
        for v in t:
            augmented[v].append(p + k)
    graph = GraphStructure.from_numpy(marks(augmented), kind="dag")
    contexts = list(range(p, p + K))

    def independent(i, j, S):
        return graph.d_separated(i, j, [*S, *contexts])

    def invariant(j, S, k):
        return graph.d_separated(p + k, j, [*S, *(c for c in contexts if c != p + k)])

    return parents, targets, known, independent, invariant


def oracle_failures(depth: int) -> list[int]:
    failed = []
    for seed in range(50):
        parents, targets, known, independent, invariant = oracle_case(seed)
        rules, targets_of = igsp.ut_igsp_rules(invariant, known)
        found, _ = igsp.igsp_search(
            len(parents), independent, rules, n_starts=1, seed=0, depth=depth
        )
        learned = [set(t) for t in targets_of(found)]
        same = np.array_equal(essential(as_lists(found), learned), essential(parents, targets))
        if not (same and learned == targets):
            failed.append(seed)
    return failed


def test_the_oracle_search_returns_the_true_class_and_targets():
    """Consistency (Squires, Wang, and Uhler 2020, Theorem 1): with every test answered by
    d-separation, an unbounded search finds the true I-essential graph and every target."""
    assert oracle_failures(depth=10**6) == []


def test_the_oracle_search_at_depth_4_misses_no_recorded_case():
    """At the default depth the 50 oracle cases recorded none missed; this must not get worse."""
    assert oracle_failures(depth=igsp.DEPTH) == []


@pytest.mark.parametrize("seed", range(10))
def test_a_reversal_gives_the_minimal_i_map_of_the_swapped_order(seed):
    parents, _, _, independent, _ = oracle_case(seed)
    p = len(parents)
    test = igsp._memoized(independent)
    order = [int(v) for v in np.random.default_rng(seed + 100).permutation(p)]
    dag = igsp._imap(order, test, None)
    covered = [
        (i, j) for j in range(p) for i in range(p) if dag[j] >> i & 1 and dag[j] == dag[i] | 1 << i
    ]
    for i, j in covered:
        # j moves right after i: no variable between them is a parent of j, since j's parents
        # are i and i's parents, so the order stays a linear extension of the DAG.
        moved = [v for v in order if v != j]
        moved.insert(moved.index(i), j)  # then swap: j right before i
        assert igsp.reverse(dag, i, j, test) == igsp._imap(moved, test, None)


#: Five variables; the minimum-degree order of the complete graph, the start of every stub search.
N_STUB = 5
START = tuple(range(N_STUB))[::-1]


def swapped(order: tuple[int, ...], *positions: int) -> tuple[int, ...]:
    """``order`` with the neighbors at each of ``positions`` and the next one swapped in turn."""
    out = list(order)
    for k in positions:
        out[k], out[k + 1] = out[k + 1], out[k]
    return tuple(out)


def stub_rules(values: dict[tuple[int, ...], int]) -> igsp.Rules:
    """Rules over complete DAGs, where a reversal swaps two neighbors of the order: the objective
    of an order is ``values[order]``, 2 for any other, and children no higher than the current DAG
    join the search."""

    def objective(parents):
        return values.get(tuple(sorted(range(N_STUB), key=lambda v: parents[v].bit_count())), 2)

    return igsp.Rules(
        objective=objective,
        icovered=lambda parents, i, j: True,
        admits=lambda child, current: child <= current,
        restarts=lambda child, current: child < current,
    )


def search(values: dict[tuple[int, ...], int], depth: int = igsp.DEPTH):
    """The stub search; every I-MAP is complete under a test that finds nothing independent."""
    rules = stub_rules(values)
    return igsp.igsp_search(N_STUB, lambda i, j, S: False, rules, n_starts=1, seed=0, depth=depth)


@pytest.mark.parametrize(("distance", "found"), [(5, True), (6, False)])
def test_the_search_finds_an_improvement_at_most_depth_plus_one_reversals_away(distance, found):
    """A path of equal-objective orders leads from the start to a lower one ``distance`` swaps
    away: the children of the order ``depth`` swaps away are scored, never descended into."""
    path = [START]
    for k in (0, 1, 2, 3, 0, 1, 2):
        path.append(swapped(path[-1], k))
    values = {order: 1 for order in path[:distance]}
    values[path[distance]] = 0
    assert search(values)[1] == (0 if found else 1)


@pytest.mark.parametrize("target_branch", [0, 1])
def test_a_dead_end_returns_to_the_parent_and_its_next_child(target_branch):
    """Two branches of equal-objective orders leave the start; the lower order ends one of them,
    and the search finds it whichever branch it walks first."""
    branches = [
        [swapped(START, 0), swapped(START, 0, 2)],
        [swapped(START, 3), swapped(START, 3, 1)],
    ]
    values = {START: 1, **{order: 1 for branch in branches for order in branch}}
    values[swapped(branches[target_branch][-1], 0 if target_branch else 3)] = 0
    assert search(values, depth=2)[1] == 0


def test_a_known_target_whose_child_tests_invariant_blocks_the_reversal():
    variant = {(1, 0b001, 0)}  # x_1 given x_0 varies in setting 0, nothing else does
    rules, _ = igsp.ut_igsp_rules(
        lambda j, S, k: (j, sum(1 << s for s in S), k) not in variant, [frozenset({0})]
    )
    assert rules.icovered((0, 0b001, 0b011), 0, 1)  # 0 -> 1 with x_1 | x_0 varying
    assert rules.icovered((0, 0b001, 0b011), 1, 2)  # 1 is no known target: a covered arrow is
    blocked, _ = igsp.ut_igsp_rules(lambda j, S, k: True, [frozenset({0})])
    assert not blocked.icovered((0, 0b001, 0b011), 0, 1)
    free, _ = igsp.ut_igsp_rules(lambda j, S, k: True, [frozenset()])
    assert free.icovered((0, 0b001, 0b011), 0, 1)


def test_ut_igsp_rules_count_a_known_target_once_and_learn_the_variant_ones():
    variant = {(2, 0b011, 0)}
    rules, targets_of = igsp.ut_igsp_rules(
        lambda j, S, k: (j, sum(1 << s for s in S), k) not in variant, [frozenset({0})]
    )
    dag = (0, 0b001, 0b011)  # 0 -> 1, 0 -> 2, 1 -> 2
    assert targets_of(dag) == [frozenset({0, 2})]
    assert rules.objective(dag) == (3 + 2,)
    lexicographic, _ = igsp.ut_igsp_rules(
        lambda j, S, k: (j, sum(1 << s for s in S), k) not in variant,
        [frozenset({0})],
        objective="lexicographic",
    )
    assert lexicographic.objective(dag) == (3, 2)


def test_the_search_keeps_required_edges_and_never_adds_forbidden_ones():
    parents, targets, known, independent, invariant = oracle_case(3)
    p = len(parents)
    absent = next(
        (i, j)
        for i in range(p)
        for j in range(p)
        if i != j and i not in parents[j] and j not in parents[i]
    )
    present = next((i, j) for j in range(p) for i in parents[j])
    knowledge = Knowledge(required=[absent], forbidden=[present])
    constraints = resolve(knowledge, p, None, method="ut_igsp")
    for n_starts in (1, 3):
        rules, _ = igsp.ut_igsp_rules(invariant, known)
        found, _ = igsp.igsp_search(
            p, independent, rules, n_starts=n_starts, seed=0, constraints=constraints
        )
        assert found[absent[1]] >> absent[0] & 1
        assert not found[present[1]] >> present[0] & 1


def test_tiers_order_every_order_the_search_starts_from():
    rules = igsp.Rules(
        objective=lambda parents: (sum(pa.bit_count() for pa in parents),),
        icovered=lambda parents, i, j: True,
        admits=lambda child, current: child <= current,
        restarts=lambda child, current: child < current,
    )
    constraints = resolve(Knowledge(tiers=[[3], [0, 1, 2]]), 4, None, method="ut_igsp")
    found, _ = igsp.igsp_search(
        4, lambda i, j, S: False, rules, n_starts=4, seed=1, constraints=constraints
    )
    assert all(found[3] >> v & 1 == 0 for v in range(3))  # nothing into the first tier
    assert all(found[v] >> 3 & 1 for v in range(3))  # the complete DAG: 3 -> every other
