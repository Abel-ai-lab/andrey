"""Adjustment criteria on a true graph, as far as the oracle answers of a pair need them.

Positions throughout; ``D[i, j]`` is the directed edge ``i -> j`` of a DAG, read from its marks.
The criteria: the causal nodes and forbidden set of a pair, the validity of one adjustment set in
a DAG (Shpitser et al. 2010; van der Zander et al. 2014), the optimal set (Henckel et al. 2022),
amenability of a CPDAG (Perkovic et al. 2018), and the local parent sets of a CPDAG (Maathuis et
al. 2009).
"""

from __future__ import annotations

from collections.abc import Collection, Iterable
from itertools import combinations

import numpy as np

from andrey.core._queries import separated
from andrey.core.structure import ARROW, TAIL, GraphStructure


def directed(marks: np.ndarray) -> np.ndarray:
    """``D[i, j]`` when the marks hold ``i -> j``: a tail at ``i`` and an arrowhead at ``j``."""
    return (marks == TAIL) & (marks.T == ARROW)


def reach(step: np.ndarray, start: Iterable[int]) -> set[int]:
    """The nodes ``start`` reaches along ``step`` (``step[i, j]`` an edge from ``i`` to ``j``),
    ``start`` included."""
    seen = set(start)
    stack = list(seen)
    while stack:
        for w in np.flatnonzero(step[stack.pop()]).tolist():
            if w not in seen:
                seen.add(w)
                stack.append(w)
    return seen


def causal_nodes(D: np.ndarray, x: int, y: int) -> set[int]:
    """The nodes other than ``x`` on a directed path from ``x`` to ``y``, ``y`` included when the
    path exists."""
    return (reach(D, [x]) - {x}) & reach(D.T, [y])


def forbidden(D: np.ndarray, x: int, y: int) -> set[int]:
    """``x`` and the descendants of every causal node of the pair: no adjustment set holds one."""
    return reach(D, causal_nodes(D, x, y)) | {x}


def is_valid(D: np.ndarray, x: int, y: int, z: Collection[int]) -> bool:
    """Whether ``z`` is a valid adjustment set for the total effect of ``x`` on ``y`` in the DAG.

    The adjustment criterion in its constructive form: ``z`` holds no forbidden node, and ``z``
    d-separates ``x`` and ``y`` in the proper back-door graph, the DAG without the first edge
    ``x -> c`` of each directed path from ``x`` to ``y``.
    """
    z = set(z)
    if z & (forbidden(D, x, y) | {y}):
        return False
    back_door = D.copy()
    for c in causal_nodes(D, x, y):
        back_door[x, c] = False
    return separated(dag_of(back_door), [x], [y], sorted(z))


def dag_of(D: np.ndarray) -> GraphStructure:
    """The DAG of a directed support."""
    marks = np.zeros(D.shape, dtype=np.int8)
    marks[D] = TAIL
    marks[D.T] = ARROW
    return GraphStructure.from_numpy(marks, kind="dag")


def canonical_set(D: np.ndarray, x: int, y: int, observed: Collection[int]) -> list[int]:
    """The observed ancestors of ``x`` and ``y`` that are not forbidden: valid exactly when some
    observed set is (van der Zander et al. 2014)."""
    keep = (reach(D.T, [x, y]) & set(observed)) - forbidden(D, x, y) - {y}
    return sorted(keep)


def optimal_set(D: np.ndarray, x: int, y: int) -> list[int]:
    """The optimal adjustment set of Henckel et al. (2022): the parents of the causal nodes that
    are not forbidden. Defined when ``x`` is an ancestor of ``y``."""
    parents = set()
    for c in causal_nodes(D, x, y):
        parents |= set(np.flatnonzero(D[:, c]).tolist())
    return sorted(parents - forbidden(D, x, y))


def cpdag_parts(marks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The directed support and the undirected support (symmetric) of a CPDAG's marks."""
    return directed(marks), (marks == TAIL) & (marks.T == TAIL)


def amenable(marks: np.ndarray, x: int, y: int) -> bool:
    """Whether the CPDAG is amenable for the pair: no possibly directed path from ``x`` to ``y``
    that avoids ``x`` after its start begins with an undirected edge at ``x``."""
    D, U = cpdag_parts(marks)
    step = D | U
    step[:, x] = False
    return not any(y in reach(step, [v]) for v in np.flatnonzero(U[x]).tolist())


def local_parent_sets(marks: np.ndarray, x: int) -> list[list[int]]:
    """The parent sets of ``x`` the CPDAG allows (Maathuis et al. 2009): its parents with each
    clique of its undirected neighbors, the empty clique included, ascending."""
    D, U = cpdag_parts(marks)
    parents = np.flatnonzero(D[:, x]).tolist()
    siblings = np.flatnonzero(U[x]).tolist()
    adjacent = D | D.T | U
    found = []
    for k in range(len(siblings) + 1):
        for chosen in combinations(siblings, k):
            if all(adjacent[a, b] for a, b in combinations(chosen, 2)):
                found.append(sorted(parents + list(chosen)))
    return sorted(found)
