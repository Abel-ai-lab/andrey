"""Exact score-optimal DAG search, the engine behind :func:`andrey.exact_search`.

The algorithm, its settings, and its references are described on :func:`andrey.exact_search`.

Under the linear-Gaussian BIC, a node's local score is ``n * log(resid / n) + len(parents) *
log(n)``, where ``resid`` is the residual sum of squares of the centered node on its centered
parents, solved from the Gram matrix (a pseudo-inverse when the parents' block is singular). A
parent set of more than ``n - 2`` members, or one that leaves ``resid`` at most ``ZERO_RESIDUAL``
times the node's sum of squares, scores ``inf`` and is never chosen. The categorical scores are
:class:`~andrey.core.score.BDeuScore` and :class:`~andrey.core.score.CategoricalBICScore`, and the
mixed-data scores :class:`~andrey.core.score.DGScore` and :class:`~andrey.core.score.CGScore`, each
BIC with ``lambda_value=1``. Both searches, A* (``"astar"``) and the subset-lattice dynamic
program (``"dp"``), run over the order graph, whose nodes are variable subsets: the shortest
path from the empty set to the full set gives each variable its best parent set among the
variables before it. Every DAG of the optimal DAG's Markov equivalence class has the same score, so
the search's choice among them follows rounding and column order; :func:`_canonical_member`
replaces it by one member fixed by the CPDAG and :func:`~andrey.search.hc.tie_ranks`.
"""

from __future__ import annotations

import bisect
import heapq
import itertools
from collections.abc import Sequence

import numpy as np

from andrey.core import ARROW, TAIL, GraphStructure
from andrey.core.knowledge import Constraints, mpdag
from andrey.core.mixed import column_mask
from andrey.core.orient import dag2cpdag, pdag2dag, to_structure
from andrey.core.score import (
    DISCRETE_SCORES,
    MIXED_SCORES,
    ZERO_RESIDUAL,
    Score,
    check_columns,
    make_score,
)
from andrey.search.hc import tie_ranks

#: The local scores :func:`exact_search` accepts.
EXACT_SCORES: tuple[str, ...] = (
    "local_score_BIC",
    "local_score_BDeu",
    "local_score_BIC_categorical",
    "local_score_degen_gauss",
    "local_score_cond_gauss",
)

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
    available variables is the optimal parent choice among them. Without required parents the empty
    set is present and is a subset of everything, so a match always exists; with them, a target that
    lacks one has no match and gives ``((), inf)``.
    """
    for parents, score, bits in parent_graph:
        if bits & target_bits == bits:
            return parents, score
    return (), float("inf")


class _GramBICScore:
    """Linear-Gaussian BIC score on mean-centered data.

    Centers each column, then precomputes the Gram matrix ``S = X.T @ X`` and its diagonal once;
    :meth:`score_many` regresses a node on each parent set through the normal equations on ``S`` to
    get the residual sum of squares, then applies the ``lambda=1`` BIC penalty. Without the
    centering, a nonzero mean would count as shared signal and link independent variables. Raises
    ``ValueError`` for a column :func:`~andrey.core.score.check_columns` rejects.
    """

    def __init__(self, data: np.ndarray) -> None:
        data = data - data.mean(axis=0)
        self._gram = data.T @ data
        self._diag = np.diag(self._gram)
        self._n = data.shape[0]
        self._log_n = float(np.log(self._n))
        check_columns(self._gram / (self._n - 1), self._n)

    def score(self, node: int, parents: Sequence[int]) -> float:
        """Return the negative local BIC score for ``node`` given ``parents`` (lower is better)."""
        return self.score_many(node, [parents])[0]

    def score_many(self, node: int, parent_sets: Sequence[Sequence[int]]) -> list[float]:
        """Return the negative local BIC scores for parent sets of one size, in request order.

        Solves the sets' normal equations in one stacked call, or each with
        :func:`_solve_normal_equations` when one is singular; each score equals the one the set
        gets alone.
        """
        size = len(parent_sets[0])
        if size > self._n - 2:
            return [float("inf")] * len(parent_sets)
        if not size:
            resids = [self._diag[node]] * len(parent_sets)
        else:
            pa = np.array(parent_sets)
            gram_pa = self._gram[pa[:, :, None], pa[:, None, :]]
            gram_node_pa = self._gram[node, pa]
            try:
                betas = np.linalg.solve(gram_pa, gram_node_pa[..., None])[..., 0]
            except np.linalg.LinAlgError:  # one singular set fails the whole stacked call
                pairs = zip(gram_pa, gram_node_pa, strict=True)
                betas = [_solve_normal_equations(a, b) for a, b in pairs]
            resids = [
                self._diag[node] - b @ beta for b, beta in zip(gram_node_pa, betas, strict=True)
            ]
        values = []
        for resid in resids:
            if not resid > ZERO_RESIDUAL * self._diag[node]:
                values.append(float("inf"))
            else:
                values.append(float(self._n * np.log(resid / self._n) + size * self._log_n))
        return values


def _solve_normal_equations(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve ``a @ x = b``, falling back to the pseudo-inverse when ``a`` is singular."""
    try:
        return np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(a) @ b


def _generate_parent_graph(
    score: Score, node: int, n_vars: int, constraints: Constraints | None = None
) -> list[_ParentEntry]:
    """Build a variable's parent graph: the maximal-candidate parent sets, sorted by score.

    Enumerates candidate parent sets by increasing size and keeps only *maximal candidates* -- a set
    is dropped when a subset scores at least as low, since it can then never be the optimal choice
    for any available-variable set. The retained entries stay sorted by ascending score so
    :func:`_best_subset` returns the optimum in one forward scan.

    With ``constraints``, every set holds the node's required parents and adds parents that are
    neither forbidden nor required, and a set is compared only with its subsets that still hold
    the required parents. :func:`_best_subset` then finds no set until the required parents are
    available, and its ``inf`` score keeps the order searches off that path.
    """
    parent_graph: list[_ParentEntry] = []
    scores: list[float] = []

    def insert(parents: tuple[int, ...], value: float, bits: int) -> None:
        index = bisect.bisect_left(scores, value)
        parent_graph.insert(index, (parents, value, bits))
        scores.insert(index, value)

    if constraints is None:
        required: tuple[int, ...] = ()
        candidates = [c for c in range(n_vars) if c != node]
    else:
        required = tuple(np.flatnonzero(constraints.required[:, node]).tolist())
        legal = ~constraints.forbidden[:, node] & ~constraints.required[:, node]
        candidates = [c for c in np.flatnonzero(legal).tolist() if c != node]
    insert(required, score.score(node, required), _structure_bits(required))
    for size in range(1, len(candidates) + 1):
        extras = list(itertools.combinations(candidates, size))
        combos = [tuple(sorted(required + extra)) if required else extra for extra in extras]
        if isinstance(score, _GramBICScore):  # one stacked solve per size
            values = score.score_many(node, combos)
        else:
            values = [score.score(node, list(combo)) for combo in combos]
        for extra, combo, value in zip(extras, combos, values, strict=True):
            bits = _structure_bits(combo)
            maximal = True
            for v in extra:  # dropping a required parent leaves no legal subset
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


def _canonical_member(
    dag: np.ndarray, ranks: np.ndarray, constraints: Constraints | None = None
) -> np.ndarray:
    """The DAG of ``dag``'s Markov equivalence class fixed by its CPDAG and ``ranks``.

    Numbers the nodes by ascending rank and takes :func:`~andrey.core.orient.pdag2dag` of the
    CPDAG, which keeps the compelled edges and orients the reversible ones into the lowest-numbered
    node that can take them. Every DAG of one class gives the same member, and relabeling the
    columns relabels it, when the ranks move with the columns. With ``constraints``, which ``dag``
    satisfies, the CPDAG is first narrowed by them (:func:`~andrey.core.knowledge.mpdag`), so the
    member satisfies them too.
    """
    cpdag = dag2cpdag(dag)
    if constraints is not None:
        cpdag = mpdag(cpdag, constraints)
    order = np.argsort(ranks)
    member = np.empty_like(dag)
    member[np.ix_(order, order)] = pdag2dag(cpdag[np.ix_(order, order)])
    return member


def exact_search(
    data: np.ndarray,
    *,
    search_method: str = "astar",
    score_func: str = "local_score_BIC",
    categorical: np.ndarray | None = None,
    constraints: Constraints | None = None,
) -> GraphStructure:
    """Return the globally score-optimal DAG for ``data`` as a ``dag`` structure.

    Scores candidates with ``score_func``; the default, the linear-Gaussian BIC (``lambda=1``), on
    mean-centered data.
    Finds the lowest-score DAG by shortest path over the order graph. ``search_method`` selects
    ``"astar"`` (default) or ``"dp"``; both reach the same optimal total score. Exponential in the
    number of variables, so intended for small problems.

    Every DAG of the optimal Markov equivalence class scores the same, so the result is the member
    :func:`_canonical_member` fixes: the class's reversible edges that ``constraints`` leave open
    are oriented by the order of :func:`~andrey.search.hc.tie_ranks`, ranks computed from each
    column's own data. Both searches
    return this member, it does not depend on last-bit rounding, and reordering the columns
    relabels it.

    Parameters
    ----------
    data : ndarray, shape (n_samples, n_vars)
        Data matrix; rows are observations, columns are variables.
    search_method : str
        ``"astar"`` for A* search or ``"dp"`` for the subset-lattice dynamic program.
    score_func : str
        Local score, one of :data:`EXACT_SCORES`: the linear-Gaussian BIC, a categorical score, or
        a mixed-data BIC.
    categorical : ndarray of bool or None
        Marks the categorical columns for a mixed-data score.
    constraints : Constraints or None
        Background knowledge: each variable's parent sets hold its required parents and no
        forbidden one, so the DAG is the score-optimal one among those that satisfy the knowledge.

    Returns
    -------
    GraphStructure
        The score-optimal DAG, edge ``i -> j`` when ``i`` is a parent of ``j``.
    """
    X = np.asarray(data, dtype=np.float64)
    if search_method not in ("astar", "dp"):
        raise ValueError(f"unknown search method {search_method!r}; use 'astar' or 'dp'")
    if score_func not in EXACT_SCORES:
        raise NotImplementedError(
            f"unsupported score_func {score_func!r}; use one of {EXACT_SCORES}"
        )
    n_vars = X.shape[1]
    if n_vars <= 1:  # no candidate parents -- the optimum is the edgeless graph
        return to_structure(np.zeros((n_vars, n_vars), dtype=np.int8), kind="dag")

    score: Score = (
        _GramBICScore(X)
        if score_func == "local_score_BIC"
        else make_score(score_func, X, categorical=categorical)
    )
    parent_graphs = [_generate_parent_graph(score, i, n_vars, constraints) for i in range(n_vars)]

    if search_method == "astar":
        structures = _astar_shortest_path(parent_graphs, n_vars)
    else:
        structures = _dp_shortest_path(parent_graphs, n_vars)

    adj = np.zeros((n_vars, n_vars), dtype=np.int8)
    for child, parents in enumerate(structures):
        for parent in parents:
            adj[parent, child] = TAIL  # tail at the parent
            adj[child, parent] = ARROW  # arrowhead at the child
    mask = np.full(n_vars, score_func in DISCRETE_SCORES)
    if score_func in MIXED_SCORES:
        mask = column_mask(categorical, n_vars)
    return to_structure(_canonical_member(adj, tie_ranks(X, mask), constraints), kind="dag")
