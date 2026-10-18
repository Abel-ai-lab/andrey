"""The interventional essential graph GIES searches over, checked against its definition.

The essential graph of a DAG under a family of intervention targets is the union of the DAGs
equivalent to it: those with its skeleton and v-structures that orient alike every edge with exactly
one end in some target (Hauser and Buhlmann 2012, Theorem 10). Small DAGs are compared with that
union, enumerated by brute force.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from andrey.core import ARROW, TAIL
from andrey.core.orient import dag2cpdag
from andrey.search.gies import cut_pairs, replace_unprotected


def _random_dag(rng: np.random.Generator, d: int, p: float) -> np.ndarray:
    order = rng.permutation(d)
    adj = np.zeros((d, d), dtype=np.int8)
    for i, j in itertools.combinations(range(d), 2):
        if rng.random() < p:
            a, b = order[i], order[j]
            adj[a, b], adj[b, a] = TAIL, ARROW
    return adj


def _arrows(adj: np.ndarray) -> set[tuple[int, int]]:
    return {(int(a), int(b)) for a, b in zip(*np.nonzero((adj == TAIL) & (adj.T == ARROW)))}


def _v_structures(arrows: set[tuple[int, int]]) -> set[tuple[int, int, int]]:
    linked = {frozenset(e) for e in arrows}
    return {
        (a, v, c)
        for (a, v), (c, w) in itertools.permutations(arrows, 2)
        if v == w and a < c and frozenset((a, c)) not in linked
    }


def _acyclic(arrows: set[tuple[int, int]], d: int) -> bool:
    parents = {v: {a for a, b in arrows if b == v} for v in range(d)}
    placed: set[int] = set()
    while len(placed) < d:
        free = {v for v in range(d) if v not in placed and parents[v] <= placed}
        if not free:
            return False
        placed |= free
    return True


def _union_of_class(dag: np.ndarray, targets: list[set[int]]) -> np.ndarray:
    d = dag.shape[0]
    arrows = _arrows(dag)
    skeleton = sorted(tuple(sorted(e)) for e in arrows)
    union: set[tuple[int, int]] = set()
    for flips in itertools.product((False, True), repeat=len(skeleton)):
        other = {(b, a) if flip else (a, b) for (a, b), flip in zip(skeleton, flips)}
        if not _acyclic(other, d) or _v_structures(other) != _v_structures(arrows):
            continue
        cut_turned = any(
            (a in t) != (b in t) and (a, b) not in arrows for t in targets for a, b in other
        )
        if not cut_turned:
            union |= other
    out = np.zeros((d, d), dtype=np.int8)
    for a, b in union:
        out[a, b] = TAIL
        out[b, a] = TAIL if (b, a) in union else ARROW
    return out


@pytest.mark.parametrize("seed", range(4))
def test_replace_unprotected_gives_the_union_of_the_class(seed):
    rng = np.random.default_rng(seed)
    for _ in range(25):
        d = int(rng.integers(3, 6))
        dag = _random_dag(rng, d, rng.uniform(0.3, 0.8))
        targets = [set(rng.choice(d, size=int(rng.integers(1, 3)), replace=False).tolist())]
        targets = targets[: int(rng.integers(0, 2))] + [{int(rng.integers(d))}]
        mask = np.zeros((len(targets) + 1, d), dtype=bool)  # the first row: no intervention
        for k, target in enumerate(targets, start=1):
            mask[k, sorted(target)] = True
        expected = _union_of_class(dag, targets)
        assert np.array_equal(replace_unprotected(dag.copy(), cut_pairs(mask)), expected)


def test_without_interventions_it_is_the_cpdag():
    rng = np.random.default_rng(7)
    for _ in range(20):
        dag = _random_dag(rng, 7, 0.4)
        no_cut = np.zeros((7, 7), dtype=bool)
        assert np.array_equal(replace_unprotected(dag.copy(), no_cut), dag2cpdag(dag))
