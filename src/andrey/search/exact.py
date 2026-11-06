"""Exact score-optimal DAG search, the engine behind :func:`andrey.exact_search`.

The algorithm, its settings, and its references are described on :func:`andrey.exact_search`.

The local scores are those :func:`~andrey.core.score.make_score` builds: the linear-Gaussian BIC
:class:`~andrey.core.score.BICScore`, the categorical scores :class:`~andrey.core.score.BDeuScore`
and :class:`~andrey.core.score.CategoricalBICScore`, and the mixed-data scores
:class:`~andrey.core.score.DGScore` and :class:`~andrey.core.score.CGScore`, whose BIC penalties
``penalty_discount`` weighs; BDeu ignores it. Every parent set is scored through the score's
``score_many``, a batch of sets of one size per call, so the BIC solves the batch's normal
equations in one stacked call. A parent set of more than ``n - 2`` members, or one that leaves a
residual variance of zero up to rounding, scores ``inf`` under the BIC and is never chosen. Both
searches, A* (``"astar"``) and the
subset-lattice dynamic program (``"dp"``), run over the order graph, whose nodes are variable
subsets: the shortest path from the empty set to the full set gives each variable its best parent
set among the variables before it. Every DAG of the optimal DAG's Markov equivalence class has the
same score, so the search's choice among them follows rounding and column order;
:func:`_canonical_member` replaces it by one member fixed by the CPDAG and
:func:`~andrey.core.ties.tie_ranks`.
"""

from __future__ import annotations

import bisect
import heapq
import itertools

import numpy as np

from andrey.core import GraphStructure
from andrey.core._bitset import bitmask
from andrey.core.knowledge import Constraints, mpdag
from andrey.core.orient import dag2cpdag, pdag2dag, to_structure
from andrey.core.score import LocalScore, make_score
from andrey.core.structure import endpoint_marks
from andrey.core.ties import tie_mask, tie_ranks
from andrey.search.operators import check_parent_cap

# One node of a variable's parent graph: its parent set, that set's local score, and the set as a
# bitmask (bit ``v`` set iff ``v`` is a parent) for O(1) subset tests.
_ParentEntry = tuple[tuple[int, ...], float, int]

# Parent sets of one size scored per call, which bounds the memory of the BIC score's stacked
# normal equations.
_SCORE_BATCH = 4096

# The most bytes of subset lookups one fit builds: the first-entry tables take about
# ``n_vars * 2**n_vars`` bytes and the per-mask minima ``8 * 2**n_vars``. A fit over it scans the
# parent graphs instead (:func:`_best_subset`), which gives the same results.
_LOOKUP_BYTES = 1 << 28


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


def _subset_lookup(
    parent_graphs: list[list[_ParentEntry]], n_vars: int
) -> tuple[list[memoryview], list[list[tuple[tuple[int, ...], float]]]]:
    """Each variable's optimal parent choice within any set of variables, as ``(first, options)``.

    Variable ``i``'s choice within the set with bitmask ``mask`` is
    ``options[i][first[i][mask]]``, a parent set and its score. ``parent_graphs[i]`` is sorted by
    ascending score, so the choice is its first entry whose members all lie within the set.
    Without required parents the empty set is an entry and lies within every set; with them, a set
    that lacks one has no such entry and gets the last option, ``((), inf)``. Each ``first[i]``
    starts at the entries' own masks and takes, one variable at a time, the minimum over the masks
    with and without it, so each mask ends with the lowest index among its subsets.
    """
    first: list[memoryview] = []
    options: list[list[tuple[tuple[int, ...], float]]] = []
    for parent_graph in parent_graphs:
        count = len(parent_graph)
        table = np.full(1 << n_vars, count, dtype=np.min_scalar_type(count))
        table[[bits for _, _, bits in parent_graph]] = np.arange(count)
        for v in range(n_vars):
            pairs = table.reshape(-1, 2, 1 << v)  # axis 1: variable v absent, present
            np.minimum(pairs[:, 1], pairs[:, 0], out=pairs[:, 1])
        first.append(table.data)
        options.append([(parents, score) for parents, score, _ in parent_graph])
        options[-1].append(((), float("inf")))
    return first, options


def _generate_parent_graph(
    score: LocalScore,
    node: int,
    n_vars: int,
    constraints: Constraints | None = None,
    tables: bool = True,
    max_parents: int | None = None,
) -> list[_ParentEntry]:
    """Build a variable's parent graph: the maximal-candidate parent sets, sorted by score.

    Enumerates candidate parent sets by increasing size, up to ``max_parents`` members (required
    parents included) when it is set, and keeps only *maximal candidates* -- a set is dropped when
    a subset scores lower, since it can then never be the optimal choice for any available-variable
    set. The retained entries stay sorted by ascending score, so the first entry within an
    available-variable set is its optimum among the enumerated sets (:func:`_subset_lookup`). Each
    enumerated set records the lowest score of a retained set within it, so a set's subsets are
    looked up rather than searched for; with ``tables`` false, each subset is found by
    :func:`_best_subset` instead.

    With ``constraints``, every set holds the node's required parents and adds parents that are
    neither forbidden nor required, and a set is compared only with its subsets that still hold
    the required parents. :func:`_subset_lookup` then finds no set until the required parents
    are available, and its ``inf`` score keeps the order searches off that path.
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
    required_value = score.score_many(node, [required])[0]
    required_bits = bitmask(required)
    insert(required, required_value, required_bits)
    # Each enumerated set's bitmask -> the lowest score of a retained set within it.
    lowest = np.full(1 << n_vars if tables else 0, np.inf)
    if tables:
        lowest[required_bits] = required_value
    largest = len(candidates)
    if max_parents is not None:
        largest = min(largest, max_parents - len(required))
    for size in range(1, largest + 1):
        level = itertools.combinations(candidates, size)
        while extras := list(itertools.islice(level, _SCORE_BATCH)):
            combos = [tuple(sorted(required + extra)) if required else extra for extra in extras]
            values = score.score_many(node, combos)
            if not tables:
                for extra, combo, value in zip(extras, combos, values, strict=True):
                    bits = bitmask(combo)
                    maximal = True
                    for v in extra:  # dropping a required parent leaves no legal subset
                        _, sub_value = _best_subset(parent_graph, bits & ~(1 << v))
                        if sub_value < value:  # a subset already scores lower -- combo is dominated
                            maximal = False
                            break
                    if maximal:
                        insert(combo, value, bits)
                continue
            added = np.fromiter(itertools.chain.from_iterable(extras), np.int64, len(extras) * size)
            added_bits = 1 << added.reshape(len(extras), size)
            bits = np.bitwise_or.reduce(added_bits, axis=1) | required_bits
            # Each set less one added parent; dropping a required parent leaves no legal subset.
            subsets = bits[:, None] ^ added_bits
            sub_values = lowest[subsets].min(axis=1)
            dominated = sub_values < values  # a subset already scores lower
            lowest[bits] = np.where(dominated, sub_values, values)
            for k in np.flatnonzero(~dominated).tolist():
                insert(combos[k], values[k], int(bits[k]))
    return parent_graph


def _astar_shortest_path(
    parent_graphs: list[list[_ParentEntry]], n_vars: int, tables: bool = True
) -> tuple[tuple[int, ...], ...]:
    """Shortest path over the order graph by A* with the sum-of-best-local-scores heuristic.

    A state is the set of already-placed variables (a bitmask); the edge that adds variable ``i``
    costs ``i``'s optimal local score given the placed variables. The heuristic sums each unplaced
    variable's globally best local score -- an admissible, consistent lower bound -- so the first
    time the full set is settled its path is optimal. Returns the optimal parent set per variable.
    """
    full = (1 << n_vars) - 1
    base = [pg[0][1] for pg in parent_graphs]  # each variable's globally best local score
    first, options = _subset_lookup(parent_graphs if tables else [], n_vars)

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
            if tables:
                parents, value = options[i][first[i][mask]]
            else:
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
    parent_graphs: list[list[_ParentEntry]], n_vars: int, tables: bool = True
) -> tuple[tuple[int, ...], ...]:
    """Shortest path over the order graph by a subset-lattice dynamic program.

    Relaxes the lattice in order of increasing subset size: the optimal cost to settle a set of
    variables plus the best local score of one more variable given that set updates the larger set.
    Back-tracking the recorded predecessors yields the optimal parent set per variable.
    """
    full = (1 << n_vars) - 1
    distance: dict[int, float] = {0: 0.0}
    predecessor: dict[int, tuple[int, int, tuple[int, ...]]] = {}
    first, options = _subset_lookup(parent_graphs if tables else [], n_vars)
    for mask in sorted(range(1 << n_vars), key=int.bit_count):
        if mask not in distance:
            continue
        settled = distance[mask]
        for i in range(n_vars):
            bit = 1 << i
            if mask & bit:
                continue
            if tables:
                parents, value = options[i][first[i][mask]]
            else:
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
    score_func: str = "bic",
    penalty_discount: float = 1.0,
    ebic_gamma: float = 1.0,
    equivalent_sample_size: float = 1.0,
    max_parents: int | None = None,
    categorical: np.ndarray | None = None,
    constraints: Constraints | None = None,
) -> GraphStructure:
    """Return the globally score-optimal DAG for ``data`` as a ``dag`` structure.

    Scores candidates with ``score_func``; the default is the linear-Gaussian BIC.
    Finds the lowest-score DAG by shortest path over the order graph. ``search_method`` selects
    ``"astar"`` (default) or ``"dp"``; both reach the same optimal total score. Exponential in the
    number of variables, so intended for small problems.

    Every DAG of the optimal Markov equivalence class scores the same, so the result is the member
    :func:`_canonical_member` fixes: the class's reversible edges that ``constraints`` leave open
    are oriented by the order of :func:`~andrey.core.ties.tie_ranks`, ranks computed from the data.
    Both searches return this member, it does not depend on last-bit rounding, and reordering the
    columns relabels it.

    Parameters
    ----------
    data : ndarray, shape (n_samples, n_vars)
        Data matrix; rows are observations, columns are variables.
    search_method : str
        ``"astar"`` for A* search or ``"dp"`` for the subset-lattice dynamic program.
    score_func : str
        Local score, one of :data:`~andrey.core.score.SCORES`: the linear-Gaussian BIC
        (``"bic"``), a categorical score, or a mixed-data BIC.
    penalty_discount : float
        Multiplier on the BIC penalty of ``log(n)`` per parameter; BDeu ignores it.
    ebic_gamma : float
        Weight on the extended-BIC term of the linear-Gaussian BIC,
        ``2 * ebic_gamma * log C(n_vars - 1, n_parents)`` per node
        (:func:`~andrey.core.score.ebic_penalty`); ``0.0`` is the standard BIC. The other scores
        ignore it.
    equivalent_sample_size : float
        Prior weight of ``"bdeu"``, the equivalent sample size of its uniform Dirichlet prior
        (:class:`~andrey.core.score.BDeuScore`); positive. The other scores ignore it.
    max_parents : int or None
        Caps each node's parent set, required parents included: only parent sets of at most
        ``max_parents`` members are scored, so the DAG is the score-optimal one among those within
        the cap. ``None`` sets no cap. With a cap, ``constraints`` may not require more than
        ``max_parents`` parents of a node (``ValueError``).
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
    check_parent_cap(
        None if constraints is None else constraints.required, max_parents, in_class=False
    )
    n_vars = X.shape[1]
    if n_vars <= 1:  # no candidate parents -- the optimum is the edgeless graph
        return to_structure(np.zeros((n_vars, n_vars), dtype=np.int8), kind="dag")

    score = make_score(
        score_func,
        X,
        penalty_discount=penalty_discount,
        ebic_gamma=ebic_gamma,
        equivalent_sample_size=equivalent_sample_size,
        categorical=categorical,
    )
    tables = (n_vars + 8) << n_vars <= _LOOKUP_BYTES
    parent_graphs = [
        _generate_parent_graph(score, i, n_vars, constraints, tables, max_parents)
        for i in range(n_vars)
    ]

    if search_method == "astar":
        structures = _astar_shortest_path(parent_graphs, n_vars, tables)
    else:
        structures = _dp_shortest_path(parent_graphs, n_vars, tables)

    dag = np.zeros((n_vars, n_vars), dtype=bool)
    for child, parents in enumerate(structures):
        dag[list(parents), child] = True
    adj = endpoint_marks(dag)
    ranks = tie_ranks(X, tie_mask(score_func, categorical, n_vars))
    return to_structure(_canonical_member(adj, ranks, constraints), kind="dag")
