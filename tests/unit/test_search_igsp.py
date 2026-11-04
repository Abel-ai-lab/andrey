"""The permutation engine of ``andrey.search.igsp`` under oracle and stub tests."""

from __future__ import annotations

import functools
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


def oracles(parents: list[list[int]], family: list[set[int]]):
    """The d-separation oracles of the DAG ``parents`` under the settings ``family``: one context
    node per setting into its targets, every context given in the observational distribution."""
    p, K = len(parents), len(family)
    augmented = [list(pa) for pa in parents] + [[] for _ in range(K)]
    for k, t in enumerate(family):
        for v in t:
            augmented[v].append(p + k)
    graph = GraphStructure.from_numpy(marks(augmented), kind="dag")
    contexts = list(range(p, p + K))

    @functools.cache
    def independent(i, j, S):
        return graph.d_separated(i, j, [*S, *contexts])

    @functools.cache
    def invariant(j, S, k):
        return graph.d_separated(p + k, j, [*S, *(c for c in contexts if c != p + k)])

    return independent, invariant


@functools.cache
def igsp_case(seed: int):
    """A random DAG on 4-7 variables with mean degree 1.5 (even seeds) or 3 (odd ones), and 1-4
    settings of one target each, of 2-3 each, or of 1-3 each, in turn by ``seed // 2``."""
    rng = np.random.default_rng([seed, 655])
    p = int(rng.integers(4, 8))
    probability = min(1.0, (1.5, 3.0)[seed % 2] / (p - 1))
    order = rng.permutation(p)
    parents: list[list[int]] = [[] for _ in range(p)]
    for a, b in itertools.combinations(range(p), 2):
        if rng.random() < probability:
            parents[order[b]].append(int(order[a]))
    sizes = ((1, 2), (2, 4), (1, 4))[(seed // 2) % 3]
    family = [
        {int(v) for v in rng.choice(p, size=int(rng.integers(*sizes)), replace=False)}
        for _ in range(int(rng.integers(1, 5)))
    ]
    return parents, family, *oracles(parents, family)


def igsp_misses(depth: int) -> list[tuple[int, int]]:
    """The (case, start) pairs whose search from a random order misses the true class."""
    missed = []
    for seed in range(200):
        parents, family, independent, invariant = igsp_case(seed)
        truth = essential(parents, family)
        test = igsp._memoized(independent)
        for start in range(5):
            order = [int(v) for v in np.random.default_rng([start, seed]).permutation(len(parents))]
            rules = igsp.igsp_rules(invariant, [frozenset(t) for t in family])
            found, _ = igsp._run(igsp._imap(order, test, None), test, rules, depth, None, None)
            if not np.array_equal(essential(as_lists(found), family), truth):
                missed.append((seed, start))
    return missed


def test_the_igsp_oracle_search_returns_the_true_class_from_any_start():
    """Consistency (Yang, Katcoff, and Uhler 2018, Theorem 4.6): with every test answered by
    d-separation, an unbounded search from any order finds the true I-essential graph, for
    single-target, multi-target, and mixed families."""
    assert igsp_misses(depth=10**6) == []


#: The (case, start) pairs of :func:`igsp_misses` missed at depth 4: 8 of 1,000.
IGSP_MISSES_AT_DEPTH_4 = {(1, 0), (17, 4), (25, 0), (25, 1), (25, 3), (95, 2), (101, 0), (101, 2)}


def test_the_igsp_oracle_search_at_depth_4_misses_no_more_than_recorded():
    assert set(igsp_misses(depth=igsp.DEPTH)) <= IGSP_MISSES_AT_DEPTH_4


def test_the_igsp_oracle_search_from_its_own_starts_returns_the_true_class():
    for seed in range(200):
        parents, family, independent, invariant = igsp_case(seed)
        rules = igsp.igsp_rules(invariant, [frozenset(t) for t in family])
        found, _ = igsp.igsp_search(len(parents), independent, rules, n_starts=5, seed=0)
        assert np.array_equal(essential(as_lists(found), family), essential(parents, family))


def test_no_arc_of_the_true_dag_is_i_contradicting_under_the_oracle():
    for seed in range(200):
        parents, family, _, invariant = igsp_case(seed)
        rules = igsp.igsp_rules(invariant, [frozenset(t) for t in family])
        dag = tuple(sum(1 << i for i in pa) for pa in parents)
        assert rules.objective(dag) == (sum(map(len, parents)), 0), seed


def as_masks(parents: list[list[int]]) -> tuple[int, ...]:
    return tuple(sum(1 << i for i in pa) for pa in parents)


def table_rules(variant: set[tuple[int, tuple[int, ...], int]], family: list[set[int]]):
    """IGSP's rules under an invariance test that finds ``x_j`` given ``S`` variant in setting
    ``k`` exactly for the ``(j, S, k)`` in ``variant``."""
    return igsp.igsp_rules(lambda j, S, k: (j, S, k) not in variant, [frozenset(t) for t in family])


def test_a_covered_arc_is_i_covered_unless_its_head_varies_in_a_setting_of_its_tail_alone():
    dag = as_masks([[], [0], []])  # 0 -> 1; 2 alone
    # Two settings with the targets {0}: x_1 varies in one, so 0 -> 1 is not I-covered.
    assert not table_rules({(1, (), 1)}, [{0}, {0}]).icovered(dag, 0, 1)
    assert table_rules(set(), [{0}, {0}]).icovered(dag, 0, 1)
    # Only multi-target settings: every covered arc is I-covered.
    assert table_rules({(1, (), 0)}, [{0, 2}]).icovered(dag, 0, 1)


def test_the_single_target_and_the_general_definitions_differ_on_a_chain():
    """Truth 0 -> 1 -> 2 (3 and 4 alone) and the minimal I-MAP of the order (0, 2, 1): 0 -> 1,
    0 -> 2, 2 -> 1. Under the setting {0}, x_2's marginal changes, so the single-target
    definition does not mark 0 -> 2; the general one, which a two-target setting switches on,
    marks it, since x_2 given x_1 does not change."""
    truth = [[], [0], [1], [], []]
    imap = as_masks([[], [0, 2], [0], [], []])
    _, single = oracles(truth, [{0}])
    _, general = oracles(truth, [{0}, {3, 4}])
    assert igsp.igsp_rules(single, [frozenset({0})]).priority(imap, 0, 2) == 1
    assert igsp.igsp_rules(general, [frozenset({0}), frozenset({3, 4})]).priority(imap, 0, 2) == 0


def test_condition_2_asks_every_subset_for_some_varying_setting():
    """0 -> 1 with 2 -> 0, and two settings that target 1 and not 0: x_0 varies in setting 0
    given nothing and in setting 1 given x_2, so for every subset of 0's other neighbors some
    setting varies, although no one setting varies for all of them."""
    dag = as_masks([[2], [0], []])
    assert table_rules({(0, (), 0), (0, (2,), 1)}, [{1, 2}, {1}]).priority(dag, 0, 1) == 0
    assert table_rules({(0, (), 0)}, [{1, 2}, {1}]).priority(dag, 0, 1) == 1


def test_with_no_setting_holding_one_end_alone_condition_1_marks_nothing():
    """Every setting holds both 0 and 1, so neither condition applies to 0 -> 1, though x_1 is
    invariant everywhere."""
    rules = table_rules(set(), [{0, 1}])
    dag = as_masks([[], [0]])
    assert rules.priority(dag, 0, 1) == 1
    assert rules.objective(dag) == (1, 0)


def test_igsp_restarts_on_fewer_arcs_only_and_admits_every_child():
    rules = table_rules(set(), [{0}])
    assert rules.restarts((3, 5), (4, 0))
    assert not rules.restarts((4, 0), (4, 5))
    assert rules.admits((4, 9), (4, 0))


def test_the_reversals_of_i_contradicting_arcs_are_visited_first():
    """0 -> 1 and 2 -> 3 are covered; under the setting {2}, x_3 does not change, so 2 -> 3 is
    I-contradicting and its reversal comes first."""
    rules = table_rules(set(), [{2}])
    dag = as_masks([[], [0], [], [2]])
    assert igsp._moves(dag, rules, None, None) == [(2, 3), (0, 1)]


def test_a_restart_goes_to_the_lowest_of_the_children_that_pass_the_restart_test():
    """Two children of the start are lower; the search restarts from the lower, -1, though the
    first, 0, leads to -5."""
    first, lower = swapped(START, 0), swapped(START, 3)
    values = {START: 1, first: 0, lower: -1, swapped(first, 2): -5}
    assert search(values)[1] == -1
