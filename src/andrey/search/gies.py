"""Greedy interventional equivalence search over a decomposable score, the engine behind
:func:`andrey.gies`.

The search state is the interventional essential graph: a chain graph whose arrows are the edges
that every DAG of the interventional Markov equivalence class orients alike. Each variable's local
score uses only the rows where it was not intervened on (:class:`InterventionalScore`), under the
linear-Gaussian BIC, the categorical BDeu and BIC, or a mixed-data BIC. The forward, backward, and
turning phases
repeat until none of them improves the score. A step orients the chain components it touches along
a lexicographic breadth-first search (LexBFS) ordering, adds, removes, or turns one edge, and
rebuilds the essential graph by turning every arrow that is not strongly protected into an
undirected edge. The algorithm, its settings, and its reference are described on
:func:`andrey.gies`.

References
----------
Hauser and Buhlmann (2012). Characterization and greedy learning of interventional Markov
equivalence classes of directed acyclic graphs. https://jmlr.org/papers/v13/hauser12a.html

Cooper and Yoo (1999). Causal discovery from a mixture of experimental and observational data.
Proceedings of the Fifteenth Conference on Uncertainty in Artificial Intelligence, 116-125.
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from andrey.core import ARROW, NULL, TAIL, GraphStructure, backend, score_delta, stats
from andrey.core.knowledge import Constraints, mpdag, orient, violates
from andrey.core.mixed import column_mask
from andrey.core.orient import pdag2dag, to_structure
from andrey.core.score import (
    DISCRETE_SCORES,
    MIXED_SCORES,
    SCORES,
    BDeuScore,
    BICScore,
    CategoricalBICScore,
    CGScore,
    DGScore,
    LocalScore,
    make_score,
)
from andrey.search import operators as ops
from andrey.search._parallel import _EPS_IMPROVE
from andrey.search._parallel_ges import move_key
from andrey.search.ges import _reachable, _recompute_objective, required_dag
from andrey.search.hc import tie_ranks

# The moves a scan skips, as ``(u, v, C)``: under background knowledge, those whose class holds no
# DAG that satisfies it on the current graph.
Excluded = set[tuple[int, int, frozenset[int]]]


def intervention_mask(interventions: npt.ArrayLike, data: np.ndarray) -> np.ndarray:
    """Return ``interventions`` as a boolean ``(n_samples, n_variables)`` matrix, checked.

    Raises ``ValueError`` when the shape differs from ``data``'s, a value is neither 0 nor 1, or a
    variable has fewer than two rows without an intervention, or the same value in all of them.
    """
    try:
        values = np.asarray(interventions, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"interventions must hold 0/1 or booleans: {error}") from error
    if values.shape != data.shape:
        raise ValueError(
            f"interventions must have the shape of data, {data.shape}, got {values.shape}"
        )
    if not np.isin(values, (0.0, 1.0)).all():
        raise ValueError("interventions must hold 0/1 or booleans")
    mask = values.astype(bool)
    for v in range(data.shape[1]):
        kept = data[~mask[:, v], v]
        if kept.size < 2 or np.ptp(kept) == 0:
            raise ValueError(
                f"variable {v} needs at least two rows without an intervention, with different "
                "values, to be scored"
            )
    return mask


def cut_pairs(mask: np.ndarray) -> np.ndarray:
    """``cut[a, b]`` is True when some row intervenes on exactly one of ``a`` and ``b``.

    Such an edge has the same orientation in every DAG of the interventional class.
    """
    d = mask.shape[1]
    cut = np.zeros((d, d), dtype=bool)
    for target in np.unique(mask, axis=0):
        cut |= target[:, None] != target[None, :]
    return cut


class InterventionalScore:
    """A local score with each variable scored on the rows where it was not intervened on.

    In a row where a variable was intervened on, its value was set from outside, so only the other
    rows show how its causes set it. ``score_func`` names the score, as for
    :func:`~andrey.core.score.make_score`. Variables left alone in the same rows share one score
    over those rows: the linear-Gaussian BIC over the rows' covariance; BDeu and the multinomial BIC
    over the rows' counts, with each column's categories those of every row
    (:meth:`~andrey.core.score.BDeuScore.on_rows`), which for BDeu is the score of Cooper and Yoo
    (1999) for a mix of experimental and observational data; and the mixed-data BICs over the rows
    alone, their categories and cells those of the rows. Every BIC charges its penalty with
    ``log(n)``, ``n`` counting every row, as the BIC of all the data does (Hauser and Buhlmann 2012,
    Sec. 5). With no interventions each score equals the one ``make_score`` builds over the data.

    Under ``"bic"``, a variable's local score is ``n_v * log(R) + penalty_discount * log(n) *
    len(parents)`` plus the extended-BIC term :func:`~andrey.core.score.ebic_penalty` of
    ``len(parents)`` among the ``d`` variables: ``n_v`` counts the rows without an intervention on
    the variable and ``R`` is the residual variance of its regression on its parents over those
    rows.

    Two graphs of one interventional class get the same total. They differ only in how the lines of
    the interventional essential graph are directed (Hauser and Buhlmann 2012, Prop. 16), and the
    two ends of a line are intervened on in the same rows (Def. 14), so the variables of a chain
    component share one score, which gives Markov equivalent graphs the same total.

    Parameters
    ----------
    data : ndarray, shape (n, d)
        Data matrix; rows are observations, columns are variables.
    mask : ndarray of bool, shape (n, d), or None
        True where the row's value of the variable was set by an intervention; None for none.
    score_func : str, default="bic"
        The local score, one of :data:`~andrey.core.score.SCORES`.
    penalty_discount : float, default=1.0
        Weight on the BIC complexity term of the deviance the search minimizes; BDeu ignores it.
    ebic_gamma : float, default=0.0
        Weight on the extended-BIC term of the linear-Gaussian BIC,
        ``2 * ebic_gamma * log C(d - 1, len(parents))`` per variable; ``0.0`` is the standard BIC.
        The other scores ignore it.
    categorical : ndarray of bool, shape (d,), or None
        Marks the categorical columns for a mixed-data score; the other scores ignore it.
    cache_size : int or None, default=1_000_000
        Memoization cache bound of each shared score; ``None`` is unbounded.

    Raises
    ------
    NotImplementedError
        If ``score_func`` is not one of :data:`~andrey.core.score.SCORES`.
    """

    def __init__(
        self,
        data: np.ndarray,
        mask: np.ndarray | None,
        *,
        score_func: str = "bic",
        penalty_discount: float = 1.0,
        ebic_gamma: float = 0.0,
        categorical: np.ndarray | None = None,
        cache_size: int | None = 1_000_000,
    ) -> None:
        if score_func not in SCORES:
            raise NotImplementedError(f"unsupported score_func {score_func!r}; use one of {SCORES}")
        X = np.asarray(data, dtype=np.float64)
        n, d = X.shape
        # Checks the settings and, for the BIC, the columns (check_columns).
        whole = make_score(
            score_func,
            X,
            penalty_discount=penalty_discount,
            ebic_gamma=ebic_gamma,
            categorical=categorical,
        )
        if mask is None:
            mask = np.zeros((n, d), dtype=bool)
        self.n = n
        self.d = d
        self.penalty_discount = float(penalty_discount)
        self.ebic_gamma = float(ebic_gamma)
        self.log_n = float(np.log(n))
        self._groups: list[LocalScore] = []
        self._group_of: list[int] = []
        index: dict[bytes, int] = {}
        for v in range(d):
            key = mask[:, v].tobytes()
            if key not in index:
                rows = ~mask[:, v]
                group: LocalScore
                if isinstance(whole, BICScore):
                    cov = np.atleast_2d(stats.cov(X[rows], rowvar=False))
                    group = BICScore.from_cov(
                        cov,
                        int(rows.sum()),
                        penalty_discount=penalty_discount,
                        ebic_gamma=ebic_gamma,
                        cache_size=cache_size,
                    )
                    group.log_n = self.log_n  # the penalty counts every row
                elif isinstance(whole, BDeuScore | CategoricalBICScore):
                    group = whole.on_rows(rows, cache_size)
                else:
                    kind = DGScore if isinstance(whole, DGScore) else CGScore
                    group = kind(
                        X[rows],
                        categorical,
                        penalty_discount=penalty_discount,
                        cache_size=cache_size,
                    )
                    group.log_n = self.log_n  # the penalty counts every row
                index[key] = len(self._groups)
                self._groups.append(group)
            self._group_of.append(index[key])

    def score(self, node: int, parents: Iterable[int]) -> float:
        """Return the memoized negative local score for ``node`` given ``parents``."""
        return self._groups[self._group_of[node]].score(node, tuple(parents))

    def add_deltas(self, node: int, parents: Iterable[int]) -> np.ndarray:
        """``score(node, parents + u) - score(node, parents)`` for every variable ``u``.

        Under the BIC, one rank-one update per ``u`` off a shared inverse
        (:func:`~andrey.core.score_delta.add_deltas`); under the other scores, two full scores
        (:func:`~andrey.core.score_delta.full_add_deltas`). ``inf`` for ``node`` and ``parents``.
        """
        group = self._groups[self._group_of[node]]
        if isinstance(group, BICScore):
            return score_delta.add_deltas(group, node, parents)
        return score_delta.full_add_deltas(group, node, parents, self.d)


# ---- the interventional essential graph ---------------------------------------------------------


def _neighborhoods(adj: np.ndarray) -> tuple[list[set[int]], list[set[int]], list[set[int]]]:
    """Each vertex's parents, line neighbors, and adjacent vertices, as sets."""
    d = adj.shape[0]
    out: tuple[list[set[int]], ...] = tuple([set() for _ in range(d)] for _ in range(3))
    for sets, marks in zip(
        out, (adj == ARROW, (adj == TAIL) & (adj.T == TAIL), adj != NULL), strict=True
    ):
        for a, b in zip(*np.nonzero(marks), strict=True):
            sets[a].add(int(b))
    pa, ne, ad = out
    return pa, ne, ad


def _strongly_protected(
    a: int,
    b: int,
    pa: list[set[int]],
    ch: list[set[int]],
    ne: list[set[int]],
    ad: list[set[int]],
    cut: np.ndarray,
) -> bool:
    """Whether the arrow ``a -> b`` is strongly protected (Hauser and Buhlmann 2012, Def. 14)."""
    if cut[a, b]:
        return True
    if any(c not in ad[b] for c in pa[a]):  # c -> a -> b, c and b non-adjacent
        return True
    if any(c != a and c not in ad[a] for c in pa[b]):  # a -> b <- c, a and c non-adjacent
        return True
    if ch[a] & pa[b]:  # a -> c -> b
        return True
    both = sorted(ne[a] & pa[b])  # a -- c1 -> b, a -- c2 -> b, c1 and c2 non-adjacent
    return any(c2 not in ad[c1] for i, c1 in enumerate(both) for c2 in both[i + 1 :])


def replace_unprotected(adj: np.ndarray, cut: np.ndarray) -> np.ndarray:
    """Turn the arrows of a partial essential graph that are not strongly protected into lines.

    ``adj`` holds a DAG, or a graph between a DAG and its interventional essential graph, as a mark
    matrix; it is changed in place and returned as that essential graph. Turning an arrow into a
    line can only unprotect arrows that share one of its endpoints, so only those are rechecked.
    """
    n = adj.shape[0]
    pa, ne, ad = _neighborhoods(adj)
    ch: list[set[int]] = [set() for _ in range(n)]
    for v in range(n):
        for p in pa[v]:
            ch[p].add(v)
    queue = deque(sorted((a, b) for b in range(n) for a in pa[b]))
    queued = set(queue)
    while queue:
        a, b = queue.popleft()
        queued.discard((a, b))
        if a not in pa[b] or _strongly_protected(a, b, pa, ch, ne, ad, cut):
            continue
        pa[b].discard(a)
        ch[a].discard(b)
        ne[a].add(b)
        ne[b].add(a)
        adj[a, b] = adj[b, a] = TAIL
        for x in (a, b):
            for arrow in [(p, x) for p in pa[x]] + [(x, c) for c in ch[x]]:
                if arrow not in queued:
                    queued.add(arrow)
                    queue.append(arrow)
    return adj


def admits(adj: np.ndarray, constraints: Constraints, cut: np.ndarray) -> bool:
    """Whether some DAG in the interventional class ``adj`` satisfies ``constraints``.

    Directs the lines the constraints decide and asks for a consistent extension (Dor and Tarsi)
    whose interventional essential graph is ``adj``. Once ``adj`` has every required adjacency,
    directs no forbidden edge, and joins no pair the constraints make non-adjacent, a line with a
    forbidden direction is one the constraints decide, so the extension uses no forbidden edge;
    when they decide no line, every DAG of the class satisfies them.
    """
    if violates(adj, constraints):
        return False
    oriented = orient(adj, constraints)
    if np.array_equal(oriented, adj):
        return True
    try:
        dag = pdag2dag(oriented)
    except ValueError:
        return False
    return bool(np.array_equal(replace_unprotected(dag, cut), adj))


def _lex_bfs(start: Sequence[int], ne: Mapping[int, set[int]]) -> list[int]:
    """LexBFS ordering of the vertices in ``start``; ties go to the vertex earlier in ``start``."""
    partition = [list(start)]
    order: list[int] = []
    while partition:
        a = partition[0].pop(0)
        if not partition[0]:
            partition.pop(0)
        order.append(a)
        refined = []
        for block in partition:
            inside = [b for b in block if b in ne[a]]
            outside = [b for b in block if b not in ne[a]]
            refined += [part for part in (inside, outside) if part]
        partition = refined
    return order


def _orient_component(adj: np.ndarray, first: Sequence[int]) -> None:
    """Orient the chain component of ``first[0]`` along a LexBFS ordering that begins at ``first``.

    The component's lines become arrows from the earlier to the later vertex, giving a DAG on it
    with no v-structure.
    """
    ne: dict[int, set[int]] = {}
    stack = [int(first[0])]
    while stack:
        v = stack.pop()
        if v not in ne:
            ne[v] = set(ops.undirected_neighbors(adj, v).tolist())
            stack += [w for w in ne[v] if w not in ne]
    rest = sorted(set(ne) - set(first))
    order = _lex_bfs([*first, *rest], ne)
    position = {v: k for k, v in enumerate(order)}
    for a in order:
        for b in ne[a]:
            if position[a] < position[b]:
                adj[a, b] = TAIL
                adj[b, a] = ARROW


# ---- the three phases ---------------------------------------------------------------------------
#
# A move's validity and score change for a target ``v`` depend on ``v``'s own marks, the lines among
# ``ne(v)``, and which of ``ne(v)`` each other vertex is adjacent to, except for the forward phase's
# path condition (Prop. 25(iii)) and the turning phase, which read the whole graph. The forward and
# backward candidates of a target are therefore kept between steps and enumerated again only for
# the targets a step touched (:func:`_touched_targets`); the path condition is checked when a
# candidate is selected. The selected move is the full scan's: the lowest key among valid moves.


def _changed_vertices(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """A boolean mask of the vertices with a mark that differs between the two graphs."""
    diff = before != after
    return diff.any(axis=0) | diff.any(axis=1)


def _touched_targets(after: np.ndarray, changed: np.ndarray) -> list[int]:
    """The targets whose forward and backward candidates a step can change.

    Those are the ``changed`` vertices and the vertices joined by a line to one of them: a line
    neighbor ``w`` of ``v`` whose adjacencies changed can change ``v``'s candidates.
    """
    lines = (after == TAIL) & (after.T == TAIL)
    return np.flatnonzero(changed | lines[changed].any(axis=0)).tolist()


class _Objective:
    """The score of an essential graph, through the DAG in its class that orients every chain
    component along a LexBFS ordering from its lowest vertex (Prop. 16 and Def. 17).

    It keeps one term per vertex; a step recomputes the terms of the chain components that hold a
    changed vertex. The total is a function of the graph alone, so a search whose every step lowers
    it never revisits a graph and stops.
    """

    def __init__(self, score: InterventionalScore, n: int) -> None:
        self.score = score
        self.terms = [score.score(v, ()) for v in range(n)]
        self.total = math.fsum(self.terms)

    def update(self, adj: np.ndarray, changed: np.ndarray) -> float:
        """Recompute the terms the change from the previous graph to ``adj`` can touch."""
        done: set[int] = set()
        for start in np.flatnonzero(changed).tolist():
            if start in done:
                continue
            ne: dict[int, set[int]] = {}
            stack = [start]
            while stack:
                v = stack.pop()
                if v not in ne:
                    ne[v] = set(np.flatnonzero((adj[v] == TAIL) & (adj[:, v] == TAIL)).tolist())
                    stack += [w for w in ne[v] if w not in ne]
            done |= ne.keys()
            earlier: set[int] = set()
            for v in _lex_bfs(sorted(ne), ne):
                parents = set(np.flatnonzero(adj[v] == ARROW).tolist()) | (ne[v] & earlier)
                self.terms[v] = self.score.score(v, parents)
                earlier.add(v)
        self.total = math.fsum(self.terms)
        return self.total


def _forward_candidates(
    v: int,
    adj: np.ndarray,
    score: InterventionalScore,
    n: int,
    ranks: Sequence[int],
    forbidden: np.ndarray | None = None,
) -> list[tuple[tuple, int, set[int]]]:
    """The improving insertions into ``v`` as ``(key, u, C)``, sorted by ``key``, path unchecked.

    ``u -> v`` with ``C`` the parents in ``v``'s component (Hauser and Buhlmann 2012, Prop. 25 (i)
    and (ii)); ``key`` is :func:`~andrey.search._parallel_ges.move_key` of the move with ``C``.
    Insertions that give ``v`` the same other parents are scored together
    (:meth:`InterventionalScore.add_deltas`). With ``forbidden`` (``[i, j]``: no edge
    ``i -> j``), neither ``u`` nor a member of ``C``'s free part is forbidden into ``v``.
    """
    row = adj[v]
    pa = np.flatnonzero(row == ARROW)
    ne = np.flatnonzero((row == TAIL) & (adj[:, v] == TAIL))
    sources = np.flatnonzero(row == NULL)
    sources = sources[sources != v]
    if forbidden is not None:
        sources = sources[~forbidden[sources, v]]
    out = []
    if ne.size == 0:
        deltas = score.add_deltas(v, pa)[sources]
        better = deltas < -_EPS_IMPROVE
        for u, delta in zip(sources[better].tolist(), deltas[better].tolist(), strict=True):
            out.append((move_key(delta, n, u, v, (), ranks), u, set()))
    else:
        pa_set = set(pa.tolist())
        ne_list = ne.tolist()
        linked = adj[np.ix_(sources, ne)] != NULL
        groups: dict[tuple[int, ...], list[tuple[int, set[int]]]] = {}
        for u, hits in zip(sources.tolist(), linked.tolist(), strict=True):
            na = {w for w, hit in zip(ne_list, hits, strict=True) if hit}
            t0 = [w for w, hit in zip(ne_list, hits, strict=True) if not hit]
            if forbidden is not None:
                t0 = [w for w in t0 if not forbidden[w, v]]
            for _index, T in ops.clique_extensions(adj, na, t0):
                C = na | set(T)
                groups.setdefault(tuple(sorted(pa_set | C)), []).append((u, C))
        for base, moves in groups.items():
            deltas = score.add_deltas(v, base)
            for u, C in moves:
                delta = float(deltas[u])
                if delta < -_EPS_IMPROVE:
                    out.append((move_key(delta, n, u, v, C, ranks), u, C))
    out.sort(key=lambda move: move[0])
    return out


def _backward_candidate(
    v: int,
    adj: np.ndarray,
    score: InterventionalScore,
    n: int,
    ranks: Sequence[int],
    constraints: Constraints | None = None,
    excluded: Excluded | None = None,
) -> tuple[tuple, int, set[int]] | None:
    """The best-improving deletion of an edge at ``v`` as ``(key, u, C)``, or None.

    ``C`` is ``v``'s neighbors kept as its parents (Prop. 28); ``key`` is
    :func:`~andrey.search._parallel_ges.move_key` of the move with ``C``. With ``constraints``, a
    required pair is never deleted, and a neighbor ``h`` whose orientation ``v -> h`` or ``u -> h``
    is forbidden is never dropped. A move in ``excluded`` is never returned.
    """
    row = adj[v]
    pa = set(np.flatnonzero(row == ARROW).tolist())
    lines = (row == TAIL) & (adj[:, v] == TAIL)
    ne = np.flatnonzero(lines)
    best: tuple[tuple, int, set[int]] | None = None
    for u in np.flatnonzero((row == ARROW) | lines).tolist():
        h0 = ne[adj[u, ne] != NULL].tolist()
        candidates = ops.clique_residuals(adj, h0)
        if constraints is not None:
            if constraints.required[u, v] or constraints.required[v, u]:
                continue
            f = constraints.forbidden
            pinned = {h for h in h0 if f[v, h] or f[u, h]}
            if pinned:
                candidates = ((i, H) for i, H in candidates if pinned.isdisjoint(H))
        for _index, H in candidates:
            C = set(h0) - set(H)
            kept = C | pa
            delta = score.score(v, kept - {u}) - score.score(v, kept | {u})
            if delta < -_EPS_IMPROVE:
                key = move_key(delta, n, u, v, C, ranks)
                if best is None or key < best[0]:
                    if excluded and (u, v, frozenset(C)) in excluded:
                        continue
                    best = (key, u, C)
    return best


def _reaches(succ: Sequence[Sequence[int]], v: int, u: int, blocked: set[int]) -> bool:
    """Whether a path ``v ~> u`` other than the edge ``v -> u`` avoids ``blocked``."""
    visited = {v}
    stack = [v]
    while stack:
        node = stack.pop()
        for c in succ[node]:
            if c in visited or c in blocked or (node == v and c == u):
                continue
            if c == u:
                return True
            visited.add(c)
            stack.append(c)
    return False


def _separated(
    ne: list[set[int]], within: set[int], a: set[int], b: set[int], by: set[int]
) -> bool:
    """Whether every path from ``a`` to ``b`` in the lines among ``within`` meets ``by``."""
    if not a or not b:
        return True
    seen = set(a)
    stack = list(a)
    while stack:
        node = stack.pop()
        for w in ne[node] & within:
            if w in b:
                return False
            if w not in seen and w not in by:
                seen.add(w)
                stack.append(w)
    return True


def _turning_best(
    adj: np.ndarray,
    score: InterventionalScore,
    n: int,
    ranks: Sequence[int],
    constraints: Constraints | None = None,
    excluded: Excluded | None = None,
) -> tuple[int, int, set[int], bool] | None:
    """The best-improving turn as ``(u, v, C, essential)``: ``v -> u`` (or ``v -- u``) becomes
    ``u -> v`` with ``C`` the parents of ``v`` in its component.

    Lines follow Prop. 31 and arrows Prop. 34; the score change is Corollary 32 or 36. Turns are
    compared by :func:`~andrey.search._parallel_ges.move_key` of the move with ``C``. With
    ``constraints``, a turn never makes a forbidden ``u -> v`` (the reverse of a required edge
    included) or a forbidden ``c -> v`` for a ``c`` of ``C`` not adjacent to ``u``, which the
    v-structure ``u -> v <- c`` compels. A move in ``excluded`` is never returned.
    """
    pa, nbr, adj_sets = _neighborhoods(adj)
    succ = ops.tail_successor_lists(adj)
    best_key: tuple | None = None
    best: tuple[int, int, set[int], bool] | None = None
    forbidden = None if constraints is None else constraints.forbidden

    def offer(delta: float, key: tuple, move: tuple[int, int, set[int], bool]) -> None:
        nonlocal best_key, best
        if delta < -_EPS_IMPROVE and (best_key is None or key < best_key):
            if excluded and (move[0], move[1], frozenset(move[2])) in excluded:
                return
            best_key, best = key, move

    for v in range(n):
        for u in sorted(nbr[v]):  # a line, turned in a DAG where it points v -> u
            if forbidden is not None and forbidden[u, v]:
                continue
            N = nbr[v] & adj_sets[u]
            for _index, T in ops.clique_extensions(adj, (), sorted(nbr[v] - {u})):
                C = set(T)
                if not C - N or not _separated(nbr, nbr[v], N - C, C - N, C & N):
                    continue
                if forbidden is not None and forbidden[list(C - N), v].any():
                    continue
                kept = pa[u] | (C & N)
                delta = (
                    score.score(v, pa[v] | C | {u})
                    + score.score(u, kept)
                    - score.score(v, pa[v] | C)
                    - score.score(u, kept | {v})
                )
                offer(delta, move_key(delta, n, u, v, C, ranks), (u, v, C, False))
        for u in np.nonzero(adj[:, v] == ARROW)[0].tolist():  # an arrow v -> u
            if forbidden is not None and forbidden[u, v]:
                continue
            N = nbr[v] & adj_sets[u]
            for _index, T in ops.clique_extensions(adj, N, sorted(nbr[v] - adj_sets[u])):
                C = N | set(T)
                if _reaches(succ, v, u, C | nbr[u]):
                    continue
                if forbidden is not None and forbidden[list(T), v].any():
                    continue
                delta = (
                    score.score(v, pa[v] | C | {u})
                    + score.score(u, pa[u] - {v})
                    - score.score(v, pa[v] | C)
                    - score.score(u, pa[u])
                )
                offer(delta, move_key(delta, n, u, v, C, ranks), (u, v, C, True))
    return best


def _insert(adj: np.ndarray, u: int, v: int, C: set[int], cut: np.ndarray) -> np.ndarray:
    _orient_component(adj, [*sorted(C), v])
    adj[u, v] = TAIL
    adj[v, u] = ARROW
    return replace_unprotected(adj, cut)


def _delete(adj: np.ndarray, u: int, v: int, C: set[int], cut: np.ndarray) -> np.ndarray:
    line = adj[u, v] == TAIL and adj[v, u] == TAIL
    _orient_component(adj, [*sorted(C), u, v] if line else [*sorted(C), v])
    adj[u, v] = adj[v, u] = NULL
    return replace_unprotected(adj, cut)


def _turn(
    adj: np.ndarray, u: int, v: int, C: set[int], essential: bool, cut: np.ndarray
) -> np.ndarray:
    if essential:
        _orient_component(adj, [u])
        _orient_component(adj, [*sorted(C), v])
    else:
        _orient_component(adj, [*sorted(C), v, u])
    adj[u, v] = TAIL
    adj[v, u] = ARROW
    return replace_unprotected(adj, cut)


class _Search:
    """The GIES search state: the essential graph, the cached candidates, and the objective.

    ``constraints`` filters the candidates, and ``start``, an essential graph, replaces the empty
    graph the search otherwise starts from.
    """

    def __init__(
        self,
        score: InterventionalScore,
        n: int,
        ranks: Sequence[int],
        constraints: Constraints | None = None,
        start: np.ndarray | None = None,
    ) -> None:
        self.score = score
        self.n = n
        self.ranks = ranks
        self.constraints = constraints
        self.forbidden = None if constraints is None else constraints.forbidden
        self.adj = np.zeros((n, n), dtype=np.int8) if start is None else start
        self.objective = _Objective(score, n)
        if start is not None:
            self.objective.update(start, np.ones(n, dtype=bool))
        self.forward: dict[int, list[tuple[tuple, int, set[int]]]] = {}
        self.backward: dict[int, tuple[tuple, int, set[int]] | None] = {}

    def forward_best(self, excluded: Excluded | None = None) -> tuple[int, int, set[int]] | None:
        """The best-improving insertion not in ``excluded`` as ``(u, v, C)`` (Prop. 25)."""
        adj, n = self.adj, self.n
        for v in range(n):
            if v not in self.forward:
                self.forward[v] = _forward_candidates(
                    v, adj, self.score, n, self.ranks, self.forbidden
                )
        heap = [(moves[0][0], v, 0) for v, moves in self.forward.items() if moves]
        if not heap:
            return None
        heapq.heapify(heap)
        reachable = _reachable(adj)
        while heap:
            _key, v, i = heapq.heappop(heap)
            moves = self.forward[v]
            _key, u, C = moves[i]
            if not reachable(v, u, C):  # Prop. 25(iii): every path from v to u meets C
                if not (excluded and (u, v, frozenset(C)) in excluded):
                    return u, v, C
            if i + 1 < len(moves):
                heapq.heappush(heap, (moves[i + 1][0], v, i + 1))
        return None

    def backward_best(self, excluded: Excluded | None = None) -> tuple[int, int, set[int]] | None:
        """The best-improving deletion not in ``excluded`` as ``(u, v, C)`` (Prop. 28).

        A target keeps one candidate; when it is excluded, the target is scanned again without
        the excluded moves.
        """
        adj, n = self.adj, self.n
        for v in range(n):
            if v not in self.backward:
                self.backward[v] = _backward_candidate(
                    v, adj, self.score, n, self.ranks, self.constraints
                )
        moves = []
        for v, move in self.backward.items():
            if move is not None and excluded and (move[1], v, frozenset(move[2])) in excluded:
                move = _backward_candidate(
                    v, adj, self.score, n, self.ranks, self.constraints, excluded
                )
            if move is not None:
                moves.append((move, v))
        best = min(moves, key=lambda found: found[0][0], default=None)
        if best is None:
            return None
        (_key, u, C), v = best
        return u, v, C

    def turning_best(
        self, excluded: Excluded | None = None
    ) -> tuple[int, int, set[int], bool] | None:
        """The best-improving turn not in ``excluded`` (:func:`_turning_best`)."""
        return _turning_best(self.adj, self.score, self.n, self.ranks, self.constraints, excluded)

    def step(self, best: Callable, move: Callable[..., np.ndarray], cut: np.ndarray) -> bool:
        """Apply the move ``best`` selects, built by ``move``; False when it selects none.

        Under knowledge, a move whose class holds no DAG that satisfies it is skipped, and ``best``
        selects again without it.
        """
        excluded: Excluded = set()
        while (found := best(excluded)) is not None:
            adj = move(self.adj.copy(), *found, cut)
            if self.constraints is None or admits(adj, self.constraints, cut):
                self.apply(adj)
                return True
            excluded.add((found[0], found[1], frozenset(found[2])))
        return False

    def apply(self, adj: np.ndarray) -> None:
        """Move to ``adj``, the graph after one step; raise unless it lowers the score."""
        changed = _changed_vertices(self.adj, adj)
        before = self.objective.total
        after = self.objective.update(adj, changed)
        if not after < before:
            raise RuntimeError(
                f"a GIES step did not lower the score ({before!r} -> {after!r}), so the search "
                "could repeat itself without end"
            )
        for v in _touched_targets(adj, changed):
            self.forward.pop(v, None)
            self.backward.pop(v, None)
        self.adj = adj


def gies(
    data: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    score_func: str = "bic",
    penalty_discount: float = 1.0,
    ebic_gamma: float = 1.0,
    categorical: np.ndarray | None = None,
    constraints: Constraints | None = None,
) -> tuple[GraphStructure, float]:
    """Discover an interventional essential graph by greedy interventional equivalence search.

    Starts from the empty graph and repeats the forward (insert), backward (delete), and turning
    phases, each until it finds no improving step, until a round of backward and turning phases
    changes nothing. Every step takes the move with the lowest score change divided by the
    variable count and rounded to nine decimal places; equal ones are ordered by the
    :func:`~andrey.search.hc.tie_ranks` of the move's two vertices, then of its other vertices.

    Parameters
    ----------
    data : ndarray, shape (n_samples, n_features)
        Data matrix; rows are observations, columns are variables.
    mask : ndarray of bool, shape (n_samples, n_features), or None
        True where the row's value of the variable was set by an intervention, as
        :func:`intervention_mask` returns; None for observational data.
    score_func : str
        Local score, scored on the rows without an intervention on each variable
        (:class:`InterventionalScore`): ``"bic"`` (linear-Gaussian BIC), ``"bdeu"`` or
        ``"discrete_bic"`` (BDeu or the multinomial BIC on category codes), or a mixed-data BIC,
        ``"degen_gauss"`` or ``"cond_gauss"``.
    penalty_discount : float
        Weight on the BIC complexity term of the deviance the search minimizes; BDeu ignores it.
    ebic_gamma : float
        Weight on the extended-BIC term of the linear-Gaussian BIC,
        ``2 * ebic_gamma * log C(n_features - 1, n_parents)`` per node
        (:func:`~andrey.core.score.ebic_penalty`); ``0.0`` is the standard BIC. The other scores
        ignore it.
    categorical : ndarray of bool or None
        Marks the categorical columns for a mixed-data score; the other scores ignore it.
    constraints : Constraints or None
        Background knowledge. No step adds a forbidden edge, removes or turns a required one, or
        reaches a class that holds no DAG that satisfies the knowledge (such a step is skipped
        for the next best one); the search starts from the class of the required edges. The
        result is the essential graph with the edges the knowledge decides directed, then Meek's
        rules with R4 (:func:`~andrey.core.knowledge.mpdag`).

    Returns
    -------
    tuple[GraphStructure, float]
        The essential graph (``kind="cpdag"``) and its total score (lower is better).

    Raises
    ------
    NotImplementedError
        If ``score_func`` is not one of :data:`~andrey.core.score.SCORES`.
    RuntimeError
        If a step does not lower the score, which a correct search never does.
    """
    X = np.asarray(data, dtype=np.float64)
    n = X.shape[1]
    score = InterventionalScore(
        X,
        mask,
        score_func=score_func,
        penalty_discount=penalty_discount,
        ebic_gamma=ebic_gamma,
        categorical=categorical,
    )
    cut = np.zeros((n, n), dtype=bool) if mask is None else cut_pairs(mask)
    columns = np.full(n, score_func in DISCRETE_SCORES)
    if score_func in MIXED_SCORES:
        columns = column_mask(categorical, n)
    ranks = tie_ranks(X, columns).tolist()
    start = None
    if constraints is not None and constraints.required.any():
        start = replace_unprotected(required_dag(constraints), cut)
    with backend.one_blas_thread():
        search = _Search(score, n, ranks, constraints, start)
        while True:
            while search.step(search.forward_best, _insert, cut):
                pass
            changed = False
            while search.step(search.backward_best, _delete, cut):
                changed = True
            while search.step(search.turning_best, _turn, cut):
                changed = True
            if not changed:
                break
        total = _recompute_objective(search.adj, score, n)
    adj = search.adj if constraints is None else mpdag(search.adj, constraints)
    return to_structure(adj, kind="cpdag"), total
