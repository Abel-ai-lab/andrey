"""The true time-series PAG of a window truth with unobserved series.

LPCMCI estimates a partial ancestral graph over the observed series at times ``t - tau_max .. t``.
Its target is ``P(G)^{tau_max}_{statAO}`` of Gerhardus and Runge (2020), Sect. 2.3, built here from
the generating time-series DAG ``G`` in three steps:

1. **The window MAG.** ``G`` is unrolled over ``t - tau_max - past .. t``. The observed series at
   times ``t - tau_max .. t`` are the window nodes; the unobserved series at every time and the
   observed series before ``t - tau_max`` are marginalized. Two window nodes are adjacent when
   their observed ancestors in the window do not d-separate them in ``G`` (the ancestral-set
   criterion of :func:`andrey.data.latent.marginal`). The mark at a node is a tail when it is an
   ancestor of the other end in ``G`` and an arrowhead otherwise. ``past`` grows until two
   successive steps leave the PAG unchanged, so the cut-off past does not change the answer.
2. **Repeating adjacencies.** A link ``X^i_{t-k-s} - X^j_{t-s}`` is kept only when every copy
   ``s = 0 .. tau_max - k`` inside the window is adjacent, so the window MAG holds every link at
   every shift or at none.
3. **Orientation.** All links start as circles. Time order puts an arrowhead at the later end of
   every lagged link. The unshielded colliders and Zhang's FCI rules (Andrey's FCI rules) then
   orient the rest, the rules' discriminating-path test answered by m-separation in the window
   MAG, and every mark is copied to all copies of its link; the two alternate until nothing
   changes. Every mark left is checked against the window MAG.

The result is a :class:`~andrey_bench.window.WindowGraph` over the observed series whose lag 0 is a
PAG. :class:`DesignOracle` answers LPCMCI's own tests by d-separation in the unrolled ``G``, so a
run with it shows which estimate an exact test would give.

References
----------
Gerhardus, A., and Runge, J. (2020). High-recall causal discovery for autocorrelated time series
with latent confounders. NeurIPS 33, 12615-12625. https://arxiv.org/abs/2007.01884

Zhang, J. (2008). On the completeness of orientation rules for causal discovery in the presence of
latent confounders and selection bias. Artificial Intelligence 172(16-17), 1873-1896.
"""

from __future__ import annotations

from itertools import combinations

import networkx as nx
import numpy as np

from andrey_bench.window import ARROW, CIRCLE, NULL, TAIL, WindowGraph

#: Rounds of unrolled past tried before giving up; each adds ``tau_max`` plus the largest true lag.
_MAX_ROUNDS = 12


def unrolled_dag(truth: WindowGraph, steps: int) -> nx.DiGraph:
    """The DAG ``G`` over ``(variable, lag)`` nodes, ``lag = 0 .. steps - 1`` steps before ``t``.

    ``truth`` covers every series, unobserved ones included, and its lag 0 is a DAG.
    """
    if truth.lag0 != "dag":
        raise ValueError("unrolled_dag needs a truth whose lag 0 is a DAG")
    n, marks = truth.n_nodes, truth.marks
    dag = nx.DiGraph()
    dag.add_nodes_from((i, s) for s in range(steps) for i in range(n))
    for k, i, j in zip(*np.nonzero(marks == TAIL), strict=True):
        if k == 0 and marks[0, j, i] != ARROW:
            raise ValueError(f"lag-0 truth link ({i}, {j}) is not directed")
        for s in range(steps - k):
            dag.add_edge((int(i), s + int(k)), (int(j), s))
    return dag


def _window_mag(truth: WindowGraph, tau_max: int, past: int):
    """The window nodes, the window MAG's marks with repeating adjacencies, and ``G``."""
    n_obs = truth.n_nodes - truth.latents
    dag = unrolled_dag(truth, tau_max + 1 + past)
    nodes = [(i, s) for s in range(tau_max + 1) for i in range(n_obs)]
    index = {v: a for a, v in enumerate(nodes)}
    window = set(nodes)
    anc = {v: nx.ancestors(dag, v) & window for v in nodes}
    adjacent = np.zeros((len(nodes), len(nodes)), dtype=bool)
    for a, b in combinations(nodes, 2):
        sep = (anc[a] | anc[b]) - {a, b}
        if not nx.is_d_separator(dag, {a}, {b}, sep):
            adjacent[index[a], index[b]] = adjacent[index[b], index[a]] = True
    # Repeating adjacencies: a link holds only when all its copies in the window do.
    for (i, sa), (j, sb) in combinations(nodes, 2):
        k, low = abs(sa - sb), min(sa, sb)
        if low != 0:
            continue
        early, late = ((i, sa), (j, sb)) if sa >= sb else ((j, sb), (i, sa))
        copies = [
            (index[(early[0], early[1] + s)], index[(late[0], late[1] + s)])
            for s in range(tau_max - k + 1)
        ]
        if not all(adjacent[x, y] for x, y in copies):
            for x, y in copies:
                adjacent[x, y] = adjacent[y, x] = False
    marks = np.zeros(adjacent.shape, dtype=np.int8)
    for a, b in zip(*np.nonzero(adjacent), strict=True):
        marks[a, b] = TAIL if nodes[a] in anc[nodes[b]] else ARROW
    return nodes, marks, anc


class _MagOracle:
    """``x _||_ y | z`` by m-separation in a MAG without undirected edges, as a p-value 1 or 0.

    Each bidirected edge becomes a hidden parent of both ends, so m-separation in the MAG is
    d-separation in that DAG.
    """

    def __init__(self, mag: np.ndarray) -> None:
        self._dag = nx.DiGraph()
        self._dag.add_nodes_from(range(len(mag)))
        for a, b in zip(*np.nonzero(mag), strict=True):
            if mag[a, b] == TAIL:
                self._dag.add_edge(int(a), int(b))
            elif a < b and mag[b, a] == ARROW:
                self._dag.add_edge(("bidirected", int(a), int(b)), int(a))
                self._dag.add_edge(("bidirected", int(a), int(b)), int(b))
        self._cache: dict = {}

    def __call__(self, x: int, y: int, condition_set=()) -> float:
        key = (int(x), int(y), frozenset(int(z) for z in condition_set))
        if key not in self._cache:
            separated = nx.is_d_separator(self._dag, {key[0]}, {key[1]}, set(key[2]))
            self._cache[key] = 1.0 if separated else 0.0
        return self._cache[key]


def _copies(nodes: list[tuple[int, int]], tau_max: int) -> list[list[tuple[int, int]]]:
    """Each window link's copies, as ``(a, b)`` index pairs with ``a`` the earlier node."""
    index = {v: a for a, v in enumerate(nodes)}
    n_obs = len(nodes) // (tau_max + 1)
    groups = []
    for k in range(tau_max + 1):
        for i in range(n_obs):
            for j in range(n_obs):
                if k == 0 and i >= j:
                    continue
                groups.append([(index[(i, k + s)], index[(j, s)]) for s in range(tau_max - k + 1)])
    return groups


def _repeat_marks(marks: np.ndarray, groups: list[list[tuple[int, int]]]) -> bool:
    """Copy every non-circle mark to all copies of its link; return whether a mark changed."""
    changed = False
    for group in groups:
        for end in (0, 1):
            seen = {int(marks[(a, b) if end == 0 else (b, a)]) for a, b in group}
            definite = seen - {NULL, CIRCLE}
            if len(definite) > 1:
                raise RuntimeError(f"copies of one link carry marks {sorted(definite)}")
            if definite and CIRCLE in seen:
                mark = definite.pop()
                for a, b in group:
                    where = (a, b) if end == 0 else (b, a)
                    marks[where] = mark
                changed = True
    return changed


def _orient(nodes, mag: np.ndarray, anc, tau_max: int) -> np.ndarray:
    """The PAG of the window MAG under time order and repeating orientations."""
    from andrey.constraint.fci import _circle_pag, _fci_orient, _orient_colliders

    marks = _circle_pag(mag != NULL)
    for a, b in zip(*np.nonzero(mag), strict=True):
        if nodes[a][1] < nodes[b][1]:  # a is later, so it cannot be an ancestor of b
            marks[a, b] = ARROW
    # A node is in the separating set of a non-adjacent pair when it is an ancestor of either end,
    # which is what the MAG's marks say about a node adjacent to both.
    sepsets = {}
    for a, c in combinations(range(len(nodes)), 2):
        if mag[a, c] == NULL:
            sep = tuple(
                sorted(x for x, v in enumerate(nodes) if v in anc[nodes[a]] | anc[nodes[c]])
            )
            sepsets[(a, c)] = sepsets[(c, a)] = tuple(x for x in sep if x not in (a, c))
    oracle = _MagOracle(mag)
    groups = _copies(nodes, tau_max)
    _orient_colliders(marks, sepsets)
    while True:
        before = marks.copy()
        _fci_orient(marks, oracle, 0.5, sepsets)
        _repeat_marks(marks, groups)
        if np.array_equal(before, marks):
            break
    definite = np.isin(marks, (TAIL, ARROW))
    if np.any(definite & (marks != mag)):
        raise RuntimeError("an oriented mark contradicts the window MAG")
    return marks


def _window_marks(nodes, marks: np.ndarray, n_obs: int, tau_max: int) -> np.ndarray:
    """Read the ``(tau_max + 1, n_obs, n_obs)`` window marks off the links ending at time ``t``."""
    index = {v: a for a, v in enumerate(nodes)}
    out = np.zeros((tau_max + 1, n_obs, n_obs), dtype=np.int8)
    for k in range(tau_max + 1):
        for i in range(n_obs):
            for j in range(n_obs):
                if k == 0 and i == j:
                    continue
                out[k, i, j] = marks[index[(i, k)], index[(j, 0)]]
    return out


def true_window_pag(truth: WindowGraph, tau_max: int) -> WindowGraph:
    """The true PAG over the observed series of ``truth``, lags ``0 .. tau_max``.

    ``truth`` is a window truth whose lag 0 is a DAG and whose last ``latents`` series are
    unobserved. Returns a window graph over the observed series with ``lag0="pag"``: lag 0 as
    endpoint marks and, at lag ``k >= 1``, the mark at the earlier end of each link, whose later end
    is an arrowhead. Raises ``RuntimeError`` when the unrolled past does not settle within
    ``_MAX_ROUNDS`` rounds.
    """
    n_obs = truth.n_nodes - truth.latents
    step = tau_max + truth.tau_max
    found: list[np.ndarray] = []
    for round_ in range(1, _MAX_ROUNDS + 1):
        nodes, mag, anc = _window_mag(truth, tau_max, round_ * step)
        found.append(_window_marks(nodes, _orient(nodes, mag, anc, tau_max), n_obs, tau_max))
        if len(found) >= 3 and all(np.array_equal(found[-1], f) for f in found[-3:-1]):
            return WindowGraph(found[-1], lag0="pag")
    raise RuntimeError(f"the true PAG did not settle within {_MAX_ROUNDS} rounds of unrolled past")


class DesignOracle:
    """LPCMCI's test over the lagged design, answered by d-separation in the unrolled DAG.

    Column ``lag * n_obs + i`` is observed series ``i`` at ``t - lag``, as in
    :func:`andrey.temporal.pcmci.lagged_design`. A d-separated query returns statistic 0 and
    p-value 1; a d-connected one statistic 1 and p-value 0.
    """

    def __init__(self, truth: WindowGraph, tau_max: int, past: int) -> None:
        self.n_obs = truth.n_nodes - truth.latents
        self.dag = unrolled_dag(truth, 2 * tau_max + 1 + past)
        self._cache: dict = {}

    def _node(self, column: int) -> tuple[int, int]:
        lag, i = divmod(int(column), self.n_obs)
        return (i, lag)

    def statistic_pvalues(
        self, x: np.ndarray, y: np.ndarray, conditions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Answer every row's query; see :class:`andrey.temporal.pcmci.LinkTest`."""
        separated = np.zeros(len(x), dtype=bool)
        for row, (a, b, cond) in enumerate(zip(x, y, conditions, strict=True)):
            key = (int(a), int(b), frozenset(int(c) for c in cond))
            if key not in self._cache:
                given = {self._node(c) for c in key[2]}
                self._cache[key] = nx.is_d_separator(
                    self.dag, {self._node(key[0])}, {self._node(key[1])}, given
                )
            separated[row] = self._cache[key]
        return (~separated).astype(float), separated.astype(float)


__all__ = ["DesignOracle", "true_window_pag", "unrolled_dag"]
