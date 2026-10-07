"""Exact BIC-optimal DAG search, the engine behind :func:`andrey.exact_search`.

The algorithm, its settings, and its references are described on :func:`andrey.exact_search`.

A node's local score is ``n * log(resid / n) + len(parents) * log(n)``, where ``resid`` is the
residual sum of squares of the centered node on its centered parents, solved from the Gram matrix
(a pseudo-inverse when the parents' block is singular). Both searches run over the order graph,
whose nodes are variable subsets: the shortest path from the empty set to the full set gives each
variable its best parent set among the variables before it. The result is the optimal DAG itself,
not its CPDAG, and each search breaks score ties deterministically.
"""

from __future__ import annotations

import bisect
import heapq
import itertools
from collections.abc import Sequence

import numpy as np

from andrey.core import ARROW, TAIL, GraphStructure
from andrey.core.orient import to_structure
from andrey.core.score import Score

# One node of a variable's parent graph: its parent set, that set's local score, and the set as a
# bitmask (bit ``v`` set iff ``v`` is a parent) for O(1) subset tests.
_ParentEntry = tuple[tuple[int, ...], float, int]


def _structure_bits(structure: Sequence[int]) -> int:
    """Bitmask of a parent set: bit ``v`` set iff ``v`` is in ``structure``."""
    bits = 0
    for v in structure:
        bits |= 1 << int(v)
    return bits


def _best_subset(
    parent_graph: list[_ParentEntry], target_bits: int
) -> tuple[tuple[int, ...], float]:
    """Lowest-score parent set whose members all lie within ``target_bits``.

    ``parent_graph`` is sorted by ascending score, so the first entry that is a subset of the
    available variables is the optimal parent choice among them. The empty set is always present and
    is a subset of everything, so a match always exists.
    """
    for parents, score, bits in parent_graph:
        if bits & target_bits == bits:
            return parents, score
    return (), float("inf")


class _GramBICScore:
    """Linear-Gaussian BIC score on mean-centered data.

    Centers each column, then precomputes the Gram matrix ``S = X.T @ X`` and its diagonal once;
    :meth:`score` regresses a node on its parents through the normal equations on ``S`` to get the
    residual sum of squares, then applies the ``lambda=1`` BIC penalty. Without the centering, a
    nonzero mean would count as shared signal and link independent variables.
    """

    def __init__(self, data: np.ndarray) -> None:
        data = data - data.mean(axis=0)
        self._gram = data.T @ data
        self._diag = np.diag(self._gram)
        self._n = data.shape[0]
        self._log_n = float(np.log(self._n))

    def score(self, node: int, parents: Sequence[int]) -> float:
        """Return the negative local BIC score for ``node`` given ``parents`` (lower is better)."""
        if not parents:
            resid = self._diag[node]
        else:
            pa = list(parents)
            gram_pa = self._gram[np.ix_(pa, pa)]
            gram_node_pa = self._gram[node, pa]
            beta = _solve_normal_equations(gram_pa, gram_node_pa)
            resid = self._diag[node] - gram_node_pa @ beta
        return float(self._n * np.log(resid / self._n) + len(parents) * self._log_n)


def _solve_normal_equations(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve ``a @ x = b``, falling back to the pseudo-inverse when ``a`` is singular."""
    try:
        return np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(a) @ b


def _generate_parent_graph(score: Score, node: int, n_vars: int) -> list[_ParentEntry]:
    """Build a variable's parent graph: the maximal-candidate parent sets, sorted by score.

    Enumerates candidate parent sets by increasing size and keeps only *maximal candidates* -- a set
    is dropped when a subset scores at least as low, since it can then never be the optimal choice
    for any available-variable set. The retained entries stay sorted by ascending score so
    :func:`_best_subset` returns the optimum in one forward scan.
    """
    parent_graph: list[_ParentEntry] = []
    scores: list[float] = []

    def insert(parents: tuple[int, ...], value: float, bits: int) -> None:
        index = bisect.bisect_left(scores, value)
        parent_graph.insert(index, (parents, value, bits))
        scores.insert(index, value)

    insert((), score.score(node, ()), 0)
    candidates = [c for c in range(n_vars) if c != node]
    for size in range(1, len(candidates) + 1):
        for combo in itertools.combinations(candidates, size):
            bits = _structure_bits(combo)
            value = score.score(node, list(combo))
            maximal = True
            for v in combo:
                _, sub_value = _best_subset(parent_graph, bits & ~(1 << v))
                if sub_value < value:  # a subset already scores lower -- combo is dominated
                    maximal = False
                    break
            if maximal:
                insert(combo, value, bits)
    return parent_graph


def _astar_shortest_path(
    parent_graphs: list[list[_ParentEntry]], n_vars: int
) -> tuple[tuple[int, ...], ...]:
    """Shortest path over the order graph by A* with the sum-of-best-local-scores heuristic.

    A state is the set of already-placed variables (a bitmask); the edge that adds variable ``i``
    costs ``i``'s optimal local score given the placed variables. The heuristic sums each unplaced
    variable's globally best local score -- an admissible, consistent lower bound -- so the first
    time the full set is settled its path is optimal. Returns the optimal parent set per variable.
    """
    full = (1 << n_vars) - 1
    base = [pg[0][1] for pg in parent_graphs]  # each variable's globally best local score

    def heuristic(mask: int) -> float:
        return sum(base[i] for i in range(n_vars) if not mask & (1 << i))

    empty = tuple[int, ...]()
    start = tuple(empty for _ in range(n_vars))
    best_g: dict[int, float] = {0: 0.0}
    closed: set[int] = set()
    counter = 0
    heap: list[tuple[float, int, int, tuple[tuple[int, ...], ...]]] = [
        (heuristic(0), counter, 0, start)
    ]
    while heap:
        _, _, mask, structures = heapq.heappop(heap)
        if mask == full:
            return structures
        if mask in closed:
            continue
        closed.add(mask)
        g = best_g[mask]
        remaining = full & ~mask
        while remaining:
            lsb = remaining & -remaining
            i = lsb.bit_length() - 1
            remaining ^= lsb
            new_mask = mask | lsb
            if new_mask in closed:
                continue
            parents, value = _best_subset(parent_graphs[i], mask)
            new_g = g + value
            if new_mask not in best_g or new_g < best_g[new_mask]:
                best_g[new_mask] = new_g
                new_structures = structures[:i] + (parents,) + structures[i + 1 :]
                counter += 1
                entry = (new_g + heuristic(new_mask), counter, new_mask, new_structures)
                heapq.heappush(heap, entry)
    raise RuntimeError("order-graph search terminated without reaching the full variable set")


def _dp_shortest_path(
    parent_graphs: list[list[_ParentEntry]], n_vars: int
) -> tuple[tuple[int, ...], ...]:
    """Shortest path over the order graph by a subset-lattice dynamic program.

    Relaxes the lattice in order of increasing subset size: the optimal cost to settle a set of
    variables plus the best local score of one more variable given that set updates the larger set.
    Back-tracking the recorded predecessors yields the optimal parent set per variable.
    """
    full = (1 << n_vars) - 1
    distance: dict[int, float] = {0: 0.0}
    predecessor: dict[int, tuple[int, int, tuple[int, ...]]] = {}
    for mask in sorted(range(1 << n_vars), key=int.bit_count):
        if mask not in distance:
            continue
        settled = distance[mask]
        for i in range(n_vars):
            bit = 1 << i
            if mask & bit:
                continue
            parents, value = _best_subset(parent_graphs[i], mask)
            new_mask = mask | bit
            candidate = settled + value
            if new_mask not in distance or candidate < distance[new_mask]:
                distance[new_mask] = candidate
                predecessor[new_mask] = (mask, i, parents)
    structures: list[tuple[int, ...]] = [() for _ in range(n_vars)]
    mask = full
    while mask:
        prev_mask, i, parents = predecessor[mask]
        structures[i] = parents
        mask = prev_mask
    return tuple(structures)


def exact_search(data: np.ndarray, *, search_method: str = "astar") -> GraphStructure:
    """Return the globally BIC-optimal DAG for ``data`` as a ``dag`` structure.

    Scores candidates with the linear-Gaussian BIC (``lambda=1``) on mean-centered data.
    Finds the lowest-score DAG by shortest path over the order graph. ``search_method`` selects
    ``"astar"`` (default) or ``"dp"``; both reach the same optimal total score but may return
    different score-tied representatives of the same Markov equivalence class. Exponential in the
    number of variables, so intended for small problems.

    Parameters
    ----------
    data : ndarray, shape (n_samples, n_vars)
        Data matrix; rows are observations, columns are variables.
    search_method : str
        ``"astar"`` for A* search or ``"dp"`` for the subset-lattice dynamic program.

    Returns
    -------
    GraphStructure
        The BIC-optimal DAG, edge ``i -> j`` when ``i`` is an optimal parent of ``j``.
    """
    X = np.asarray(data, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"data must be a 2-D (n_samples, n_vars) array, got ndim={X.ndim}")
    if search_method not in ("astar", "dp"):
        raise ValueError(f"unknown search method {search_method!r}; use 'astar' or 'dp'")
    n_vars = X.shape[1]
    if n_vars <= 1:  # no candidate parents -- the optimum is the edgeless graph
        return to_structure(np.zeros((n_vars, n_vars), dtype=np.int8), kind="dag")

    score = _GramBICScore(X)
    parent_graphs = [_generate_parent_graph(score, i, n_vars) for i in range(n_vars)]

    if search_method == "astar":
        structures = _astar_shortest_path(parent_graphs, n_vars)
    else:
        structures = _dp_shortest_path(parent_graphs, n_vars)

    adj = np.zeros((n_vars, n_vars), dtype=np.int8)
    for child, parents in enumerate(structures):
        for parent in parents:
            adj[parent, child] = TAIL  # tail at the parent
            adj[child, parent] = ARROW  # arrowhead at the child
    return to_structure(adj, kind="dag")
