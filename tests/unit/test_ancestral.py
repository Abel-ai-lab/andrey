"""Ancestor and non-ancestor claims on an equivalence class, checked against every DAG of the class.

The oracle lists the DAGs of a small CPDAG's class, keeps those that meet the claims, and directs an
edge when every kept DAG directs it the same way; the MPDAG :func:`restrict` returns must be that
graph, and with its residual clauses it must keep exactly those DAGs, with no clause that follows
from the others or that keeps a head it can drop.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from andrey.core import ARROW, NULL, TAIL
from andrey.core.ancestral import ChainGraph, clause_mpdag, restrict
from andrey.core.orient import dag2cpdag, meek

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


def test_a_graph_that_stands_for_no_dag_drops_every_claim():
    # The undirected 4-cycle a - b - c - d - a has no extension without a new collider.
    pdag = _graph(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
    claim = np.zeros((4, 4), dtype=bool)
    claim[0, 2] = True
    mpdag, clauses, unmet = restrict(pdag, claim, claim.T)
    assert np.array_equal(mpdag, pdag) and clauses == []
    assert unmet == [("required_paths", 0, 2), ("forbidden_paths", 2, 0)]


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
