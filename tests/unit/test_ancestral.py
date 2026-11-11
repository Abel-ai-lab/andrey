"""Ancestor and non-ancestor claims on an equivalence class, checked against every DAG of the class.

The oracle lists the DAGs of a small CPDAG's class, keeps those that meet the claims, and directs an
edge when every kept DAG directs it the same way; the MPDAG :func:`restrict` returns must be that
graph, and with its residual clauses it must keep exactly those DAGs, with no clause that follows
from the others or that keeps a head it can drop. A graph with no consistent extension stands for no
DAG to list: there the claims on undirected edges direct them, checked on fixed graphs and, on
random ones, against a reading of the rule that takes one claim at a time.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from andrey.core import ARROW, NULL, TAIL
from andrey.core.ancestral import ChainGraph, clause_mpdag, direct_claimed_edges, restrict
from andrey.core.orient import cpdag_of_pdag, dag2cpdag, meek

# Fang, Zhao, Liu, and He (2025), Figure 7a: A, B, C, and X form a clique of undirected edges, D is
# joined to A and X, and B -> Y <- C, Y <- E.
A, B, C, D, X, Y, E = range(7)
FIGURE_7 = [(A, B), (A, C), (A, X), (B, C), (X, B), (X, C), (D, A), (D, X)]
FIGURE_7_ARROWS = [(B, Y), (C, Y), (E, Y)]


def _graph(d: int, lines=(), arrows=()) -> np.ndarray:
    marks = np.full((d, d), NULL, dtype=np.int8)
    for i, j in lines:
        marks[i, j] = marks[j, i] = TAIL
    for i, j in arrows:
        marks[i, j], marks[j, i] = TAIL, ARROW
    return marks


def _directed(marks: np.ndarray) -> set[tuple[int, int]]:
    return {(i, j) for i, j in np.argwhere((marks == TAIL) & (marks.T == ARROW)).tolist()}


def _members(cpdag: np.ndarray) -> list[np.ndarray]:
    """Every DAG of the class of ``cpdag``: the orientations of its undirected edges that give it
    back as their CPDAG."""
    lines = [(i, j) for i, j in np.argwhere(np.triu((cpdag == TAIL) & (cpdag.T == TAIL))).tolist()]
    found = []
    for flips in itertools.product((False, True), repeat=len(lines)):
        dag = cpdag.copy()
        for (i, j), flip in zip(lines, flips):
            a, b = (j, i) if flip else (i, j)
            dag[a, b], dag[b, a] = TAIL, ARROW
        if _acyclic(dag) and np.array_equal(dag2cpdag(dag), cpdag):
            found.append(dag)
    return found


def _reach(dag: np.ndarray) -> np.ndarray:
    reach = (dag == TAIL) & (dag.T == ARROW)
    for k in range(len(dag)):
        reach |= reach[:, k, None] & reach[None, k, :]
    return reach


def _acyclic(dag: np.ndarray) -> bool:
    return not np.diagonal(_reach(dag)).any()


def _meets(dag: np.ndarray, required_paths: np.ndarray, forbidden_paths: np.ndarray) -> bool:
    reach = _reach(dag)
    return bool(not (required_paths & ~reach).any() and not (forbidden_paths & reach).any())


def _holds(dag: np.ndarray, clauses) -> bool:
    return all(any(dag[t, h] == TAIL and dag[h, t] == ARROW for h in heads) for t, heads in clauses)


def _expected(cpdag: np.ndarray, kept: list[np.ndarray]) -> np.ndarray:
    """The CPDAG with each undirected edge directed when every DAG of ``kept`` directs it alike."""
    out = cpdag.copy()
    for i, j in np.argwhere((cpdag == TAIL) & (cpdag.T == TAIL)).tolist():
        if all(dag[i, j] == TAIL and dag[j, i] == ARROW for dag in kept):
            out[j, i] = ARROW
    return out


def _random_dag(rng: np.random.Generator, d: int) -> np.ndarray:
    order = rng.permutation(d)
    p = rng.uniform(0.3, 0.8)
    arrows = [(order[a], order[b]) for a in range(d) for b in range(a + 1, d) if rng.random() < p]
    return _graph(d, arrows=arrows)


def _claims(rng: np.random.Generator, truth: np.ndarray, cpdag: np.ndarray, true: bool):
    """Ancestor and non-ancestor claims over random pairs: true of ``truth``, or drawn at random.

    Most claims start at a vertex with two undirected edges, where a claim can leave a clause.
    """
    d = len(truth)
    reach = _reach(truth)
    required_paths = np.zeros((d, d), dtype=bool)
    forbidden_paths = np.zeros((d, d), dtype=bool)
    starts = np.flatnonzero(((cpdag == TAIL) & (cpdag.T == TAIL)).sum(axis=1) >= 2)
    for _ in range(rng.integers(1, 4)):
        i, j = rng.choice(d, 2, replace=False)
        if len(starts) and rng.random() < 0.7:
            i = rng.choice(starts)
            j = rng.choice(np.delete(np.arange(d), i))
        is_ancestor = reach[i, j] if true else rng.random() < 0.5
        (required_paths if is_ancestor else forbidden_paths)[i, j] = True
    return required_paths, forbidden_paths


def _check(start: np.ndarray, members: list[np.ndarray], required_paths, forbidden_paths) -> bool:
    """Check :func:`restrict` on the graph ``start`` against the enumeration; ``members`` is the
    class of its CPDAG. The claims it keeps are those the enumeration keeps when it takes them
    in order, ``required_paths`` then ``forbidden_paths``, each kept when some DAG ``start``
    stands for meets it with those kept before it. Returns whether a claim was dropped."""
    d = len(start)
    required_paths = np.zeros((d, d), bool) if required_paths is None else required_paths
    forbidden_paths = np.zeros((d, d), bool) if forbidden_paths is None else forbidden_paths
    stands_for = [m for m in members if _directed(start) <= _directed(m)]
    kept_required, kept_forbidden = np.zeros((d, d), bool), np.zeros((d, d), bool)
    dropped = []
    for part, mask in (("required_paths", required_paths), ("forbidden_paths", forbidden_paths)):
        for i, j in np.argwhere(mask).tolist():
            trial = (kept_required.copy(), kept_forbidden.copy())
            trial[part == "forbidden_paths"][i, j] = True
            if any(_meets(m, *trial) for m in stands_for):
                kept_required, kept_forbidden = trial
            else:
                dropped.append((part, i, j))
    mpdag, clauses, unmet = restrict(start, required_paths, forbidden_paths)
    assert unmet == dropped, (start, required_paths, forbidden_paths)
    kept = [m for m in stands_for if _meets(m, kept_required, kept_forbidden)]
    cpdag = dag2cpdag(members[0])
    assert np.array_equal(mpdag, _expected(cpdag, kept)), (start, required_paths, forbidden_paths)
    assert np.array_equal(meek(mpdag, rule4=True), mpdag)
    assert all(len(heads) >= 2 for _, heads in clauses), clauses
    within = [m for m in stands_for if _directed(mpdag) <= _directed(m)]
    assert [_holds(m, clauses) for m in within] == [
        _meets(m, kept_required, kept_forbidden) for m in within
    ], (start, required_paths, forbidden_paths, clauses)
    _check_minimal(within, clauses)
    return bool(dropped)


def _check_minimal(within: list[np.ndarray], clauses) -> None:
    """Check that no clause follows from the MPDAG and the others, and that no clause keeps a head
    it can drop: ``within`` are the DAGs of the MPDAG."""
    for k, clause in enumerate(clauses):
        others = clauses[:k] + clauses[k + 1 :]
        assert any(_holds(m, others) and not _holds(m, [clause]) for m in within), (clause, clauses)
        t, heads = clause
        for h in heads:
            fewer = (t, heads - {h})
            assert any(_holds(m, clauses) and not _holds(m, [fewer]) for m in within), (h, clauses)


def test_every_restricted_class_matches_the_enumeration():
    rng = np.random.default_rng(0)
    checked = dropped = 0
    while checked < 300:
        d = int(rng.integers(4, 9))
        truth = _random_dag(rng, d)
        cpdag = dag2cpdag(truth)
        lines = np.argwhere(np.triu((cpdag == TAIL) & (cpdag.T == TAIL))).tolist()
        if len(lines) > 11:
            continue
        # Half the graphs carry a direction of the truth, as knowledge orients a learned graph.
        start = cpdag
        if lines and checked % 2:
            i, j = lines[rng.integers(len(lines))]
            if truth[i, j] == ARROW:  # j -> i in the truth
                i, j = j, i
            start = cpdag.copy()
            start[j, i] = ARROW
            start = meek(start, rule4=True)
        required_paths, forbidden_paths = _claims(rng, truth, cpdag, true=checked % 3 != 0)
        found = _check(start, _members(cpdag), required_paths, forbidden_paths)
        assert not (found and checked % 3), "a true claim is always met"
        dropped += found
        checked += 1
    assert dropped > 10  # the false claims reach the conflict


def _sink_dag(rng: np.random.Generator, d: int) -> np.ndarray:
    """A DAG whose first ``d - 2`` vertices form one chain component with no collider, and some
    of them and the last vertex point into vertex ``d - 2``, as in Figure 7a."""
    arrows: list[tuple[int, int]] = []
    parents: dict[int, list[int]] = {0: []}
    for v in range(1, d - 2):
        base = int(rng.integers(v))
        parents[v] = [base] + [u for u in parents[base] if rng.random() < 0.6]
        arrows += [(u, v) for u in parents[v]]
    into = [v for v in range(d - 2) if rng.random() < 0.4] or [0]
    return _graph(d, arrows=[*arrows, *((v, d - 2) for v in into), (d - 1, d - 2)])


def test_claims_about_a_sink_match_the_enumeration():
    # The claims that leave clauses: whether a vertex of the component causes the sink.
    rng = np.random.default_rng(1)
    clauses = 0
    for k in range(300):
        d = int(rng.integers(5, 10))
        truth = _sink_dag(rng, d)
        reach = _reach(truth)
        x = int(rng.integers(d - 2))
        required_paths = np.zeros((d, d), dtype=bool)
        forbidden_paths = np.zeros((d, d), dtype=bool)
        (required_paths if reach[x, d - 2] != (k % 4 == 0) else forbidden_paths)[x, d - 2] = True
        cpdag = dag2cpdag(truth)
        if not _check(cpdag, _members(cpdag), required_paths, forbidden_paths):
            clauses += bool(restrict(cpdag, required_paths, forbidden_paths)[1])
    assert clauses > 10


def test_a_forbidden_path_directs_its_critical_set_into_it():
    # Fang et al. (2025), Example 2: A is not a cause of Y, so B -> A and C -> A.
    cpdag = _graph(7, FIGURE_7, FIGURE_7_ARROWS)
    forbidden_paths = np.zeros((7, 7), dtype=bool)
    forbidden_paths[A, Y] = True
    mpdag, clauses, _ = restrict(cpdag, None, forbidden_paths)
    assert {(B, A), (C, A)} <= _directed(mpdag)
    assert clauses == []


def test_a_required_path_through_one_of_two_edges_stays_a_clause():
    # Fang et al. (2025), Example 2: D is a cause of Y exactly when D -> A or D -> X. Each way
    # directs A and X into B and C, and neither edge at D is the same in all three (Figure 7d-f).
    cpdag = _graph(7, FIGURE_7, FIGURE_7_ARROWS)
    required_paths = np.zeros((7, 7), dtype=bool)
    required_paths[D, Y] = True
    mpdag, clauses, _ = restrict(cpdag, required_paths, None)
    assert _directed(mpdag) - _directed(cpdag) == {(A, B), (A, C), (X, B), (X, C)}
    assert clauses == [(D, frozenset({A, X}))]


def test_a_clause_drops_a_head_another_claim_rules_out():
    # Every pair of 0, ..., 4 is joined but 0 and 4. 1 is an ancestor of 2, so 1 -> 2, and 0 of 4,
    # so 0 -> 1, 0 -> 2, or 0 -> 3. 0 -> 1 without 0 -> 2 closes 0 -> 1 -> 2 -> 0, so 1 is no head.
    cpdag = _graph(5, [pair for pair in itertools.combinations(range(5), 2) if pair != (0, 4)])
    required_paths = np.zeros((5, 5), dtype=bool)
    required_paths[0, 4] = required_paths[1, 2] = True
    none = np.zeros((5, 5), dtype=bool)
    assert not _check(cpdag, _members(cpdag), required_paths, none)
    assert restrict(cpdag, required_paths, none)[1] == [(0, frozenset({2, 3}))]


def test_claims_no_dag_meets_together_keep_the_first():
    # Fang et al. (2025), Example 3: the two claims of Example 2 hold together in no DAG; each holds
    # in some, so the required path, taken first, is kept, and the forbidden path is dropped.
    cpdag = _graph(7, FIGURE_7, FIGURE_7_ARROWS)
    required_paths = np.zeros((7, 7), dtype=bool)
    forbidden_paths = np.zeros((7, 7), dtype=bool)
    required_paths[D, Y] = forbidden_paths[A, Y] = True
    mpdag, clauses, unmet = restrict(cpdag, required_paths, forbidden_paths)
    assert unmet == [("forbidden_paths", A, Y)]
    alone = restrict(cpdag, required_paths, None)
    assert np.array_equal(mpdag, alone[0]) and clauses == alone[1] and alone[2] == []
    assert _check(cpdag, _members(cpdag), required_paths, forbidden_paths)


@pytest.mark.parametrize("part", ["required_paths", "forbidden_paths"])
def test_a_claim_the_class_cannot_meet_is_dropped(part):
    # a -> b <- d and b -> c, the class's one DAG: a is an ancestor of c, and c is not one of a.
    cpdag = dag2cpdag(_graph(4, arrows=[(0, 1), (3, 1), (1, 2)]))
    claim = np.zeros((4, 4), dtype=bool)
    pair = (2, 0) if part == "required_paths" else (0, 2)
    claim[pair] = True
    claims = (claim, None) if part == "required_paths" else (None, claim)
    mpdag, clauses, unmet = restrict(cpdag, *claims)
    assert unmet == [(part, *pair)] and clauses == []
    assert np.array_equal(mpdag, cpdag)


# The undirected 4-cycle a - b - c - d - a has no extension without a new collider, and neither has
# it with the edge d - e added.
SQUARE = [(0, 1), (1, 2), (2, 3), (3, 0)]


def _masks(d: int, required=(), forbidden=()) -> tuple[np.ndarray, np.ndarray]:
    required_paths, forbidden_paths = np.zeros((d, d), dtype=bool), np.zeros((d, d), dtype=bool)
    for pair in required:
        required_paths[pair] = True
    for pair in forbidden:
        forbidden_paths[pair] = True
    return required_paths, forbidden_paths


def _no_extension(pdag: np.ndarray) -> bool:
    try:
        cpdag_of_pdag(pdag)
    except ValueError:
        return True
    return False


def test_a_claim_on_a_pair_that_is_not_adjacent_is_dropped_without_an_extension():
    pdag = _graph(4, SQUARE)
    claim = np.zeros((4, 4), dtype=bool)
    claim[0, 2] = True
    applied: list = []
    mpdag, clauses, unmet = restrict(pdag, claim, claim.T, edge_claims=applied)
    assert np.array_equal(mpdag, pdag) and clauses == [] and applied == []
    assert unmet == [("required_paths", 0, 2), ("forbidden_paths", 2, 0)]


@pytest.mark.parametrize(
    "claims",
    [
        {"required": [(0, 1)]},
        {"forbidden": [(1, 0)]},
        {"required": [(0, 1)], "forbidden": [(1, 0)]},
    ],
)
def test_a_claim_directs_its_undirected_edge_without_an_extension(claims):
    # a is an ancestor of b, or b is none of a: a -> b. R1 then gives b -> c, c -> d, and d -> e;
    # d -> a would close a directed cycle, so d - a stays.
    pdag = _graph(5, [*SQUARE, (3, 4)])
    assert _no_extension(pdag)
    applied: list = []
    mpdag, clauses, unmet = restrict(pdag, *_masks(5, **claims), edge_claims=applied)
    assert np.array_equal(mpdag, _graph(5, [(3, 0)], [(0, 1), (1, 2), (2, 3), (3, 4)]))
    assert clauses == [] and unmet == []
    assert applied == [
        *(("required_paths", *pair) for pair in claims.get("required", [])),
        *(("forbidden_paths", *pair) for pair in claims.get("forbidden", [])),
    ]


def test_a_claim_against_a_directed_edge_is_dropped_and_one_with_it_is_met():
    # The square with a -> b directed: the marks stay whatever the claims say.
    pdag = _graph(4, SQUARE[1:], [(0, 1)])
    assert _no_extension(pdag)
    against = _masks(4, required=[(1, 0)], forbidden=[(0, 1)])
    applied: list = []
    mpdag, clauses, unmet = restrict(pdag, *against, edge_claims=applied)
    assert np.array_equal(mpdag, pdag) and clauses == [] and applied == []
    assert unmet == [("required_paths", 1, 0), ("forbidden_paths", 0, 1)]
    mpdag, clauses, unmet = restrict(
        pdag, *_masks(4, required=[(0, 1)], forbidden=[(1, 0)]), edge_claims=applied
    )
    assert np.array_equal(mpdag, pdag) and unmet == [] and applied == []


def test_a_claim_on_a_bidirected_edge_is_dropped():
    pdag = _graph(4, SQUARE[1:])
    pdag[0, 1] = pdag[1, 0] = ARROW
    assert _no_extension(pdag)
    mpdag, _, unmet = restrict(pdag, *_masks(4, required=[(0, 1)], forbidden=[(0, 1)]))
    assert np.array_equal(mpdag, pdag)
    assert unmet == [("required_paths", 0, 1), ("forbidden_paths", 0, 1)]


def test_meeks_rules_follow_the_claimed_edges():
    # The square a - b - c - d - a, which leaves the graph without an extension, beside the
    # triangle e - f - g with e - h, h adjacent to neither f nor g.
    pdag = _graph(8, [*SQUARE, (4, 5), (5, 6), (4, 6), (4, 7)])
    # R1: h -> e claimed, so e -> f and e -> g; f - g stays.
    mpdag, _, unmet = restrict(pdag, *_masks(8, required=[(7, 4)]))
    assert np.array_equal(mpdag, _graph(8, [*SQUARE, (5, 6)], [(7, 4), (4, 5), (4, 6)]))
    assert unmet == []
    # R2: e -> f and f -> g claimed, so e -> g.
    mpdag, _, unmet = restrict(pdag, *_masks(8, required=[(4, 5), (5, 6)]))
    assert _directed(mpdag) == {(4, 5), (5, 6), (4, 6)} and unmet == []
    # R3: b - a, b - c, and b - d with a -> d <- c claimed and a, c not adjacent, so b -> d.
    kite = _graph(8, [*SQUARE, (1, 3), (4, 5), (5, 6), (6, 7), (7, 4)])
    mpdag, _, unmet = restrict(kite, *_masks(8, required=[(0, 3), (2, 3)]))
    assert _directed(mpdag) == {(0, 3), (2, 3), (1, 3)} and unmet == []
    # R4: a - d and a - b with b -> c -> d claimed, a adjacent to c, and b, d not adjacent, so
    # a -> d.
    four = _graph(8, [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7), (7, 4)])
    mpdag, _, unmet = restrict(four, *_masks(8, required=[(1, 2), (2, 3)]))
    assert _directed(mpdag) == {(1, 2), (2, 3), (0, 3)} and unmet == []


def test_claims_for_both_directions_of_an_edge_are_all_dropped():
    pdag = _graph(5, [*SQUARE, (3, 4)])
    applied: list = []
    mpdag, _, unmet = restrict(
        pdag, *_masks(5, required=[(0, 1), (1, 0), (3, 4)], forbidden=[(0, 1)]), edge_claims=applied
    )
    # a - b stays undirected; d -> e is applied, and R1 directs nothing from it.
    assert np.array_equal(mpdag, _graph(5, SQUARE, [(3, 4)]))
    assert unmet == [("required_paths", 0, 1), ("required_paths", 1, 0), ("forbidden_paths", 0, 1)]
    assert applied == [("required_paths", 3, 4)]


def test_claims_that_close_a_directed_cycle_are_all_dropped():
    # Around the square, the four claims ask for a -> b -> c -> d -> a.
    pdag = _graph(5, [*SQUARE, (3, 4)])
    around = [(0, 1), (1, 2), (2, 3), (3, 0)]
    applied: list = []
    mpdag, _, unmet = restrict(pdag, *_masks(5, required=[*around, (3, 4)]), edge_claims=applied)
    assert np.array_equal(mpdag, _graph(5, SQUARE, [(3, 4)]))
    assert unmet == [("required_paths", *pair) for pair in around]
    assert applied == [("required_paths", 3, 4)]
    # With b -> c -> d directed by the data, the claims d -> a and a -> b close the cycle together;
    # each alone is applied.
    pdag = _graph(5, [(0, 1), (3, 0), (3, 4)], [(1, 2), (2, 3)])
    assert _no_extension(pdag)
    mpdag, _, unmet = restrict(pdag, *_masks(5, required=[(3, 0), (0, 1)]))
    assert np.array_equal(mpdag, pdag)
    assert unmet == [("required_paths", 0, 1), ("required_paths", 3, 0)]
    mpdag, _, unmet = restrict(pdag, *_masks(5, required=[(3, 0)]))
    assert (3, 0) in _directed(mpdag) and unmet == []


def _random_pdag(rng: np.random.Generator, d: int) -> np.ndarray:
    """A random mark matrix over ``d`` nodes: each pair is non-adjacent, undirected, or directed
    one way, and now and then bidirected."""
    marks = np.full((d, d), NULL, dtype=np.int8)
    for i, j in itertools.combinations(range(d), 2):
        kind = rng.choice(5, p=[0.5, 0.25, 0.11, 0.11, 0.03])
        if kind:
            marks[i, j] = ARROW if kind in (3, 4) else TAIL
            marks[j, i] = ARROW if kind in (2, 4) else TAIL
    return marks


def _one_at_a_time(pdag: np.ndarray, claims: list) -> tuple[set, set, set]:
    """The edges the claims direct and the claims applied and dropped, read off one claim at a
    time: a claim is dropped when its pair has no undirected edge directed its way or to direct,
    when another claim asks for the reverse, or when the edges every claim asks for, with the
    directed edges of ``pdag``, lead from its head back to its tail."""
    asks = {c: (c[1], c[2]) if c[0] == "required_paths" else (c[2], c[1]) for c in claims}
    line = {c for c, (a, b) in asks.items() if pdag[a, b] == TAIL and pdag[b, a] == TAIL}
    met = {c for c, (a, b) in asks.items() if pdag[a, b] == TAIL and pdag[b, a] == ARROW}
    reverse = {c for c in line if asks[c][::-1] in {asks[o] for o in line}}
    reach = (pdag == TAIL) & (pdag.T == ARROW)
    for c in line - reverse:
        reach[asks[c]] = True
    for k in range(len(pdag)):
        reach |= reach[:, k, None] & reach[None, k, :]
    cyclic = {c for c in line - reverse if reach[asks[c][::-1]]}
    applied = line - reverse - cyclic
    return {asks[c] for c in applied}, applied, set(claims) - applied - met


def test_claimed_edges_on_random_graphs_without_an_extension():
    rng = np.random.default_rng(0)
    checked = dropped_some = applied_some = conflicts = 0
    while checked < 300:
        d = int(rng.integers(4, 9))
        pdag = _random_pdag(rng, d)
        if not _no_extension(pdag):
            continue
        checked += 1
        pairs = rng.permutation([(i, j) for i in range(d) for j in range(d) if i != j])
        claims = [
            ("required_paths" if rng.random() < 0.5 else "forbidden_paths", int(i), int(j))
            for i, j in pairs[: rng.integers(1, 7)]
        ]
        edges, applied, dropped = _one_at_a_time(pdag, claims)
        out, was_applied, was_dropped = direct_claimed_edges(pdag, claims)
        assert was_applied == [c for c in claims if c in applied]
        assert was_dropped == [c for c in claims if c in dropped]
        # The skeleton and every mark the graph had stay; each applied claim's edge is directed.
        assert np.array_equal(out != NULL, pdag != NULL)
        lines = (pdag == TAIL) & (pdag.T == TAIL)
        assert np.array_equal(out[~lines], pdag[~lines])
        assert edges <= _directed(out)
        if edges:
            start = pdag.copy()
            for a, b in edges:
                start[a, b], start[b, a] = TAIL, ARROW
            assert np.array_equal(out, meek(start, rule4=True))
            assert np.array_equal(meek(out, rule4=True), out)
        else:
            assert np.array_equal(out, pdag)
        # Neither the order of the claims nor that of the columns changes the result.
        shuffled = [claims[k] for k in rng.permutation(len(claims))]
        again, again_applied, again_dropped = direct_claimed_edges(pdag, shuffled)
        assert np.array_equal(again, out)
        assert set(again_applied) == applied and set(again_dropped) == dropped
        perm = rng.permutation(d)  # column k of the relabeled graph is column perm[k]
        where = np.argsort(perm)
        moved = [(part, int(where[x]), int(where[y])) for part, x, y in claims]
        relabeled, moved_applied, moved_dropped = direct_claimed_edges(
            pdag[np.ix_(perm, perm)], moved
        )
        assert np.array_equal(relabeled, out[np.ix_(perm, perm)])
        assert {(p, int(perm[x]), int(perm[y])) for p, x, y in moved_applied} == applied
        assert {(p, int(perm[x]), int(perm[y])) for p, x, y in moved_dropped} == dropped
        # restrict takes this branch, with the claims in the order of its masks.
        required_paths, forbidden_paths = _masks(
            d,
            [c[1:] for c in claims if c[0] == "required_paths"],
            [c[1:] for c in claims if c[0] == "forbidden_paths"],
        )
        listed: list = []
        mpdag, clauses, unmet = restrict(pdag, required_paths, forbidden_paths, edge_claims=listed)
        assert np.array_equal(mpdag, out) and clauses == []
        assert set(unmet) == dropped and set(listed) == applied
        dropped_some += bool(dropped)
        applied_some += bool(applied)
        conflicts += len(applied) + len(dropped) < len(claims) or bool(
            {c for c in dropped if pdag[c[1], c[2]] == TAIL and pdag[c[2], c[1]] == TAIL}
        )
    assert min(dropped_some, applied_some, conflicts) >= 20, (dropped_some, applied_some, conflicts)


def test_a_graph_with_an_extension_lists_no_edge_claim():
    # Fang et al. (2025), Example 2 on Figure 7a: the published rule applies, as without the sink.
    cpdag = _graph(7, FIGURE_7, FIGURE_7_ARROWS)
    required_paths, forbidden_paths = _masks(7, required=[(D, Y)], forbidden=[(X, A)])
    listed: list = []
    with_sink = restrict(cpdag, required_paths, forbidden_paths, edge_claims=listed)
    without = restrict(cpdag, required_paths, forbidden_paths)
    assert listed == [] and np.array_equal(with_sink[0], without[0])
    assert with_sink[1:] == without[1:]
    assert _directed(with_sink[0]) > _directed(cpdag)


def test_the_orientation_components_give_the_mpdag_of_clauses():
    # Fang et al. (2025), Example 8 and Figure 11: three 4-cliques that share B, and four clauses.
    a, b, c, d, e, f, g, h = range(8)
    lines = [(b, c), (b, d), (b, e), (c, d), (c, e), (d, e)]  # {B, C, D, E}
    lines += [(a, b), (a, e), (a, f), (b, f), (e, f)]  # {A, B, E, F}
    lines += [(b, g), (b, h), (f, g), (f, h), (g, h)]  # {B, F, G, H}
    clauses = [
        (d, frozenset({e, b})),
        (e, frozenset({a, c})),
        (e, frozenset({b, f})),
        (g, frozenset({b, h})),
    ]
    mpdag = clause_mpdag(ChainGraph.of(_graph(8, lines)), clauses)
    assert _directed(mpdag) == {
        (e, a),
        (b, a),
        (e, f),
        (b, f),
        (f, g),
        (b, g),
        (b, h),
        (f, h),
        (g, h),
    }
