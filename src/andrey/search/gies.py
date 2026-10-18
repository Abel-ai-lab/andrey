"""Greedy interventional equivalence search over the linear-Gaussian BIC, the engine behind
:func:`andrey.gies`.

The search state is the interventional essential graph: a chain graph whose arrows are the edges
that every DAG of the interventional Markov equivalence class orients alike. Each variable's local
score uses only the rows where it was not intervened on. The forward, backward, and turning phases
repeat until none of them improves the score. A step orients the chain components it touches along
a LexBFS ordering, adds, removes, or turns one edge, and rebuilds the essential graph by turning
every arrow that is not strongly protected into an undirected edge. The algorithm, its settings,
and its reference are described on :func:`andrey.gies`.

References
----------
Hauser and Buhlmann (2012). Characterization and greedy learning of interventional Markov
equivalence classes of directed acyclic graphs. https://jmlr.org/papers/v13/hauser12a.html
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from andrey.core import ARROW, NULL, TAIL, GraphStructure, backend, score_delta, stats
from andrey.core.orient import to_structure
from andrey.core.score import BICScore, check_columns
from andrey.search import operators as ops
from andrey.search._parallel import _EPS_IMPROVE, stable_key
from andrey.search.ges import _reach_factory, _recompute_objective


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


class InterventionalBICScore:
    """Linear-Gaussian BIC with each variable scored on the rows where it was not intervened on.

    A variable's local score is ``n_v * log(R) + lambda_value * log(n) * len(parents)``: ``n_v``
    counts the rows without an intervention on the variable, ``R`` is the residual variance of its
    regression on its parents over those rows, and ``n`` counts every row. Variables left alone in
    the same rows share one :class:`~andrey.core.score.BICScore` over those rows' covariance, with
    the penalty of all ``n`` rows. With no interventions it equals
    :class:`~andrey.core.score.BICScore`.

    Parameters
    ----------
    data : ndarray, shape (n, d)
        Data matrix; rows are observations, columns are variables.
    mask : ndarray of bool, shape (n, d), or None
        True where the row's value of the variable was set by an intervention; None for none.
    lambda_value : float, default=1.0
        Weight on the BIC complexity term of the deviance the search minimizes.
    cache_size : int or None, default=1_000_000
        Memoization cache bound of each shared score; ``None`` is unbounded.
    """

    def __init__(
        self,
        data: np.ndarray,
        mask: np.ndarray | None,
        *,
        lambda_value: float = 1.0,
        cache_size: int | None = 1_000_000,
    ) -> None:
        lambda_value = float(lambda_value)
        if not np.isfinite(lambda_value) or lambda_value < 0.0:
            raise ValueError(f"lambda_value must be finite and non-negative, got {lambda_value}")
        X = np.asarray(data, dtype=np.float64)
        n, d = X.shape
        check_columns(np.atleast_2d(stats.cov(X, rowvar=False)), n)
        if mask is None:
            mask = np.zeros((n, d), dtype=bool)
        self.n = n
        self.lambda_value = lambda_value
        self.log_n = float(np.log(n))
        self._groups: list[BICScore] = []
        self._group_of: list[int] = []
        index: dict[bytes, int] = {}
        for v in range(d):
            key = mask[:, v].tobytes()
            if key not in index:
                rows = ~mask[:, v]
                cov = np.atleast_2d(stats.cov(X[rows], rowvar=False))
                group = BICScore.from_cov(
                    cov, int(rows.sum()), lambda_value=lambda_value, cache_size=cache_size
                )
                group.log_n = self.log_n  # the penalty counts every row
                index[key] = len(self._groups)
                self._groups.append(group)
            self._group_of.append(index[key])

    def score(self, node: int, parents: Iterable[int]) -> float:
        """Return the memoized negative local BIC score for ``node`` given ``parents``."""
        return self._groups[self._group_of[node]].score(node, tuple(parents))

    def add_deltas(self, node: int, parents: Iterable[int]) -> np.ndarray:
        """``score(node, parents + u) - score(node, parents)`` for every variable ``u``.

        One rank-one update per ``u`` off a shared inverse
        (:func:`~andrey.core.score_delta.add_deltas`); ``inf`` for ``node`` and ``parents``.
        """
        return score_delta.add_deltas(self._groups[self._group_of[node]], node, parents)


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

# The scan the forward and backward phases use. ``True`` enumerates every target at every step and
# checks every insertion's path condition before choosing; ``False`` keeps each target's candidates
# until a step touches it and checks the path condition in key order. Tests flip it to compare.
_FULL_RESCAN = False


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

    def __init__(self, score: InterventionalBICScore, n: int) -> None:
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
    v: int, adj: np.ndarray, score: InterventionalBICScore, n: int
) -> list[tuple[tuple, int, set[int]]]:
    """The improving insertions into ``v`` as ``(key, u, C)``, sorted by ``key``, path unchecked.

    ``u -> v`` with ``C`` the parents in ``v``'s component (Hauser and Buhlmann 2012, Prop. 25 (i)
    and (ii)); ``key`` is ``(stable_key(delta, n), u, v, index)``, ``index`` the position of
    ``C``'s free part among the subsets of ``ne(v)`` without ``u``'s neighbors. Insertions that
    give ``v`` the same other parents are scored together
    (:meth:`InterventionalBICScore.add_deltas`).
    """
    row = adj[v]
    pa = np.flatnonzero(row == ARROW)
    ne = np.flatnonzero((row == TAIL) & (adj[:, v] == TAIL))
    sources = np.flatnonzero(row == NULL)
    sources = sources[sources != v]
    out = []
    if ne.size == 0:
        deltas = score.add_deltas(v, pa)[sources]
        better = deltas < -_EPS_IMPROVE
        for u, delta in zip(sources[better].tolist(), deltas[better].tolist(), strict=True):
            out.append(((stable_key(delta, n), u, v, 0), u, set()))
    else:
        pa_set = set(pa.tolist())
        ne_list = ne.tolist()
        linked = adj[np.ix_(sources, ne)] != NULL
        groups: dict[tuple[int, ...], list[tuple[int, int, set[int]]]] = {}
        for u, hits in zip(sources.tolist(), linked.tolist(), strict=True):
            na = {w for w, hit in zip(ne_list, hits, strict=True) if hit}
            t0 = [w for w, hit in zip(ne_list, hits, strict=True) if not hit]
            for index, T in ops.clique_extensions(adj, na, t0):
                C = na | set(T)
                groups.setdefault(tuple(sorted(pa_set | C)), []).append((u, index, C))
        for base, moves in groups.items():
            deltas = score.add_deltas(v, base)
            for u, index, C in moves:
                delta = float(deltas[u])
                if delta < -_EPS_IMPROVE:
                    out.append(((stable_key(delta, n), u, v, index), u, C))
    out.sort(key=lambda move: move[0])
    return out


def _backward_candidate(
    v: int, adj: np.ndarray, score: InterventionalBICScore, n: int
) -> tuple[tuple, int, set[int]] | None:
    """The best-improving deletion of an edge at ``v`` as ``(key, u, C)``, or None.

    ``C`` is ``v``'s neighbors kept as its parents (Prop. 28); ``key`` is ``(stable_key(delta,
    n), u, v, index)``, ``index`` the position of the dropped neighbors among the subsets of
    ``ne(v)`` adjacent to ``u``.
    """
    row = adj[v]
    pa = set(np.flatnonzero(row == ARROW).tolist())
    lines = (row == TAIL) & (adj[:, v] == TAIL)
    ne = np.flatnonzero(lines)
    best: tuple[tuple, int, set[int]] | None = None
    for u in np.flatnonzero((row == ARROW) | lines).tolist():
        h0 = ne[adj[u, ne] != NULL].tolist()
        for index, H in ops.clique_residuals(adj, h0):
            C = set(h0) - set(H)
            kept = C | pa
            delta = score.score(v, kept - {u}) - score.score(v, kept | {u})
            if delta < -_EPS_IMPROVE:
                key = (stable_key(delta, n), u, v, index)
                if best is None or key < best[0]:
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
    adj: np.ndarray, score: InterventionalBICScore, n: int
) -> tuple[int, int, set[int], bool] | None:
    """The best-improving turn as ``(u, v, C, essential)``: ``v -> u`` (or ``v -- u``) becomes
    ``u -> v`` with ``C`` the parents of ``v`` in its component.

    Lines follow Prop. 31 and arrows Prop. 34; the score change is Corollary 32 or 36.
    """
    pa, nbr, adj_sets = _neighborhoods(adj)
    succ = ops.tail_successor_lists(adj)
    best_key: tuple | None = None
    best: tuple[int, int, set[int], bool] | None = None

    def offer(delta: float, key: tuple, move: tuple[int, int, set[int], bool]) -> None:
        nonlocal best_key, best
        if delta < -_EPS_IMPROVE and (best_key is None or key < best_key):
            best_key, best = key, move

    for v in range(n):
        for u in sorted(nbr[v]):  # a line, turned in a DAG where it points v -> u
            N = nbr[v] & adj_sets[u]
            for index, T in ops.clique_extensions(adj, (), sorted(nbr[v] - {u})):
                C = set(T)
                if not C - N or not _separated(nbr, nbr[v], N - C, C - N, C & N):
                    continue
                kept = pa[u] | (C & N)
                delta = (
                    score.score(v, pa[v] | C | {u})
                    + score.score(u, kept)
                    - score.score(v, pa[v] | C)
                    - score.score(u, kept | {v})
                )
                offer(delta, (stable_key(delta, n), u, v, 0, index), (u, v, C, False))
        for u in np.nonzero(adj[:, v] == ARROW)[0].tolist():  # an arrow v -> u
            N = nbr[v] & adj_sets[u]
            for index, T in ops.clique_extensions(adj, N, sorted(nbr[v] - adj_sets[u])):
                C = N | set(T)
                if _reaches(succ, v, u, C | nbr[u]):
                    continue
                delta = (
                    score.score(v, pa[v] | C | {u})
                    + score.score(u, pa[u] - {v})
                    - score.score(v, pa[v] | C)
                    - score.score(u, pa[u])
                )
                offer(delta, (stable_key(delta, n), u, v, 1, index), (u, v, C, True))
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
    """The GIES search state: the essential graph, the cached candidates, and the objective."""

    def __init__(self, score: InterventionalBICScore, n: int) -> None:
        self.score = score
        self.n = n
        self.build_state = _reach_factory(n)
        self.adj = np.zeros((n, n), dtype=np.int8)
        self.objective = _Objective(score, n)
        self.forward: dict[int, list[tuple[tuple, int, set[int]]]] = {}
        self.backward: dict[int, tuple[tuple, int, set[int]] | None] = {}

    def forward_best(self) -> tuple[int, int, set[int]] | None:
        """The best-improving insertion as ``(u, v, C)`` (Prop. 25)."""
        adj, n = self.adj, self.n
        if _FULL_RESCAN:
            reachable = self.build_state(adj)
            valid = (
                (key, u, C)
                for v in range(n)
                for key, u, C in _forward_candidates(v, adj, self.score, n)
                if not reachable(v, u, C)
            )
            best = min(valid, default=None)
            return None if best is None else (best[1], best[0][2], best[2])
        for v in range(n):
            if v not in self.forward:
                self.forward[v] = _forward_candidates(v, adj, self.score, n)
        heap = [(moves[0][0], v, 0) for v, moves in self.forward.items() if moves]
        if not heap:
            return None
        heapq.heapify(heap)
        reachable = self.build_state(adj)
        while heap:
            _key, v, i = heapq.heappop(heap)
            moves = self.forward[v]
            _key, u, C = moves[i]
            if not reachable(v, u, C):  # Prop. 25(iii): every path from v to u meets C
                return u, v, C
            if i + 1 < len(moves):
                heapq.heappush(heap, (moves[i + 1][0], v, i + 1))
        return None

    def backward_best(self) -> tuple[int, int, set[int]] | None:
        """The best-improving deletion as ``(u, v, C)`` (Prop. 28)."""
        adj, n = self.adj, self.n
        if _FULL_RESCAN:
            self.backward.clear()
        for v in range(n):
            if v not in self.backward:
                self.backward[v] = _backward_candidate(v, adj, self.score, n)
        best = min((move for move in self.backward.values() if move is not None), default=None)
        if best is None:
            return None
        key, u, C = best
        return u, key[2], C

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
    data: np.ndarray, mask: np.ndarray | None = None, *, lambda_value: float = 1.0
) -> tuple[GraphStructure, float]:
    """Discover an interventional essential graph by greedy interventional equivalence search.

    Starts from the empty graph and repeats the forward (insert), backward (delete), and turning
    phases, each until it finds no improving step, until a round of backward and turning phases
    changes nothing. Every step takes the move with the lowest score change, ties broken by the
    vertices and the move's position in the enumeration.

    Parameters
    ----------
    data : ndarray, shape (n_samples, n_features)
        Data matrix; rows are observations, columns are variables.
    mask : ndarray of bool, shape (n_samples, n_features), or None
        True where the row's value of the variable was set by an intervention, as
        :func:`intervention_mask` returns; None for observational data.
    lambda_value : float
        Weight on the BIC complexity term of the deviance the search minimizes.

    Returns
    -------
    tuple[GraphStructure, float]
        The essential graph (``kind="cpdag"``) and its total BIC score (lower is better).

    Raises
    ------
    RuntimeError
        If a step does not lower the score, which a correct search never does.
    """
    X = np.asarray(data, dtype=np.float64)
    n = X.shape[1]
    score = InterventionalBICScore(X, mask, lambda_value=lambda_value)
    cut = np.zeros((n, n), dtype=bool) if mask is None else cut_pairs(mask)
    with backend.one_blas_thread():
        search = _Search(score, n)
        while True:
            while (step := search.forward_best()) is not None:
                search.apply(_insert(search.adj.copy(), *step, cut))
            changed = False
            while (step := search.backward_best()) is not None:
                search.apply(_delete(search.adj.copy(), *step, cut))
                changed = True
            while (turn := _turning_best(search.adj, score, n)) is not None:
                search.apply(_turn(search.adj.copy(), *turn, cut))
                changed = True
            if not changed:
                break
        total = _recompute_objective(search.adj, score, n)
    return to_structure(search.adj, kind="cpdag"), total
