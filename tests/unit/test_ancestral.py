"""Ancestor and non-ancestor claims on an equivalence class, checked against every DAG of the class.

The oracle lists the DAGs of a small CPDAG's class, keeps those that meet the claims, and directs an
edge when every kept DAG directs it the same way; the MPDAG :func:`restrict` returns must be that
graph, and with its residual clauses it must keep exactly those DAGs.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from andrey.core import ARROW, NULL, TAIL
from andrey.core.ancestral import ChainGraph, clause_mpdag, restrict
from andrey.core.knowledge import KnowledgeError
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
    class of its CPDAG. Returns whether no DAG met the claims, so that it had to raise."""
    stands_for = [m for m in members if _directed(start) <= _directed(m)]
    kept = [m for m in stands_for if _meets(m, required_paths, forbidden_paths)]
    if not kept:
        with pytest.raises(KnowledgeError):
            restrict(start, required_paths, forbidden_paths)
        return True
    mpdag, clauses = restrict(start, required_paths, forbidden_paths)
    cpdag = dag2cpdag(members[0])
    assert np.array_equal(mpdag, _expected(cpdag, kept)), (start, required_paths, forbidden_paths)
    assert np.array_equal(meek(mpdag, rule4=True), mpdag)
    assert all(len(heads) >= 2 for _, heads in clauses), clauses
    within = [m for m in stands_for if _directed(mpdag) <= _directed(m)]
    assert [_holds(m, clauses) for m in within] == [
        _meets(m, required_paths, forbidden_paths) for m in within
    ], (start, required_paths, forbidden_paths, clauses)
    return False


def test_every_restricted_class_matches_the_enumeration():
    rng = np.random.default_rng(0)
    checked = raised = 0
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
            if truth[j, i] == ARROW:
                i, j = j, i
            start = cpdag.copy()
            start[j, i] = ARROW
            start = meek(start, rule4=True)
        required_paths, forbidden_paths = _claims(rng, truth, cpdag, true=checked % 3 != 0)
        raised += _check(start, _members(cpdag), required_paths, forbidden_paths)
        checked += 1
    assert raised > 10  # the false claims reach the conflict


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
    mpdag, clauses = restrict(cpdag, None, forbidden_paths)
    assert {(B, A), (C, A)} <= _directed(mpdag)
    assert clauses == []


def test_a_required_path_through_one_of_two_edges_stays_a_clause():
    # Fang et al. (2025), Example 2: D is a cause of Y exactly when D -> A or D -> X. Each way
    # directs A and X into B and C, and neither edge at D is the same in all three (Figure 7d-f).
    cpdag = _graph(7, FIGURE_7, FIGURE_7_ARROWS)
    required_paths = np.zeros((7, 7), dtype=bool)
    required_paths[D, Y] = True
    mpdag, clauses = restrict(cpdag, required_paths, None)
    assert _directed(mpdag) - _directed(cpdag) == {(A, B), (A, C), (X, B), (X, C)}
    assert clauses == [(D, frozenset({A, X}))]


def test_claims_no_dag_meets_together_raise():
    # Fang et al. (2025), Example 3: the two claims of Example 2 hold together in no DAG.
    cpdag = _graph(7, FIGURE_7, FIGURE_7_ARROWS)
    required_paths = np.zeros((7, 7), dtype=bool)
    forbidden_paths = np.zeros((7, 7), dtype=bool)
    required_paths[D, Y] = forbidden_paths[A, Y] = True
    with pytest.raises(KnowledgeError, match="together"):
        restrict(cpdag, required_paths, forbidden_paths)


@pytest.mark.parametrize(
    ("part", "message"), [("required_paths", "no DAG"), ("forbidden_paths", "every DAG")]
)
def test_a_claim_the_class_cannot_meet_names_itself(part, message):
    # a -> b <- d and b -> c, the class's one DAG: a is an ancestor of c, and c is not one of a.
    cpdag = dag2cpdag(_graph(4, arrows=[(0, 1), (3, 1), (1, 2)]))
    claim = np.zeros((4, 4), dtype=bool)
    claim[(2, 0) if part == "required_paths" else (0, 2)] = True
    claims = (claim, None) if part == "required_paths" else (None, claim)
    with pytest.raises(KnowledgeError, match=message):
        restrict(cpdag, *claims)


def test_a_graph_that_stands_for_no_dag_raises():
    # The undirected 4-cycle a - b - c - d - a has no extension without a new collider.
    pdag = _graph(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
    claim = np.zeros((4, 4), dtype=bool)
    claim[0, 2] = True
    with pytest.raises(KnowledgeError, match="stands for no DAG"):
        restrict(pdag, claim, None)


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
