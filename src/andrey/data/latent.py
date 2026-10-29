"""Latent-confounder handling: designate hidden nodes, drop them, keep truth self-describing.

A latent SCM generates over the full node set, then hides a subset of nodes (dropping their columns
from the data) while keeping them in the ground-truth graph flagged via ``node_types`` LATENT.
Latents are chosen among nodes with at least two children -- common causes -- so hiding them
actually induces confounding, directly or transitively (a leaf or single-child latent would
project away to nothing). Observed nodes are re-indexed first so the data columns align 1:1 with
the observed truth nodes.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

from andrey.core.structure import ARROW, LATENT, OBSERVED, TAIL, GraphStructure

from .graphs import csr_from_endpoints

if TYPE_CHECKING:
    from andrey.core.knowledge import Constraints


def latent_order(edges: np.ndarray, d: int, n_latent: int, rng: np.random.Generator) -> np.ndarray:
    """Choose ``n_latent`` confounders; return the node order that puts them last.

    Parameters
    ----------
    edges : np.ndarray of shape (m, 2)
        Directed edges ``(parent, child)`` in the full node-label space.
    d : int
        Total node count.
    n_latent : int
        Number of latent confounders to hide.
    rng : np.random.Generator
        Stream for choosing which qualifying nodes become latent.

    Returns
    -------
    np.ndarray of shape (d,)
        ``order[j]`` is the label of the node re-indexed as ``j``: the observed labels ascending,
        then the latent ones ascending.

    Raises
    ------
    ValueError
        If fewer than ``n_latent`` nodes have >= 2 children (no genuine confounder to hide).
    """
    out_degree = np.bincount(edges[:, 0], minlength=d) if edges.shape[0] else np.zeros(d, dtype=int)
    candidates = np.flatnonzero(out_degree >= 2)  # >= 2 children (confounding, possibly transitive)
    if candidates.size < n_latent:
        raise ValueError(
            f"only {candidates.size} node(s) have >= 2 children; "
            f"cannot hide {n_latent} latent confounder(s)"
        )
    chosen = rng.choice(candidates, size=n_latent, replace=False)
    is_latent = np.zeros(d, dtype=bool)
    is_latent[chosen] = True
    return np.concatenate([np.flatnonzero(~is_latent), np.flatnonzero(is_latent)])


def marginal(graph: GraphStructure, target: str = "mag", *, max_nodes: int = 500) -> GraphStructure:
    """Project a latent-confounded DAG truth onto its observed nodes as a MAG or a PAG.

    Both share one skeleton -- observed ``a, b`` are adjacent iff they are not d-separated (in the
    full DAG) by their observed ancestors, the ancestor-set characterization. A moral-graph search
    from each observed node decides its pairs, and only the pairs that search joins through latent
    nodes need more than a lookup.

    - ``target="mag"`` returns the **maximal ancestral graph**: a single representative where a
      hidden common cause becomes a bidirected edge (``a -> b`` when ``a`` is an ancestor of
      ``b``, ``a <-> b`` when neither is). A MAG has one edge per pair, so it round-trips losslessly
      through the endpoint-mark CSR (a general ADMG, which could carry ``->`` *and* ``<->`` on one
      pair, does not).
    - ``target="pag"`` returns the **partial ancestral graph** -- the Markov-equivalence-class
      representative FCI/GFCI recover, the fair scoring target for those methods. It is built from
      the same skeleton and the true separating sets: unshielded colliders, then Zhang's complete
      FCI orientation rules to a fixed point, with the R4 discriminating-path collider answered by a
      d-separation **oracle** (:class:`DSepOracle`) on the true DAG -- exact at infinite sample,
      independent of any finite-sample search. No selection variables are present, so the PAG has no
      undirected (TAIL-TAIL) edges.

    Parameters
    ----------
    graph : GraphStructure
        A ``kind="dag"`` truth whose ``node_types`` flag LATENT confounders. With no latents,
        ``"mag"`` returns the DAG itself. ``"pag"`` differs from the CPDAG: a collider the
        CPDAG orients ``a -> c <- b`` is ``a o-> c <-o b`` in the PAG.
    target : {"mag", "pag"}, default="mag"
        Which observed-marginal graph to build.
    max_nodes : int, default=500
        Guard on the observed-node count (the all-pairs projection is quadratic); raises above it
        with the count rather than running an intractable projection silently.

    Returns
    -------
    GraphStructure
        The MAG over the observed nodes, ``kind="mag"``, or their PAG, ``kind="pag"``; with no
        latents, ``"mag"`` returns the DAG itself.

    Raises
    ------
    ValueError
        If ``target`` is unknown, ``n_obs > max_nodes``, or observed nodes are not indexed first.
    """
    if target not in ("mag", "pag"):
        raise ValueError(f"target must be 'mag' or 'pag', got {target!r}")
    types = graph.node_types
    has_latents = types is not None and bool(np.any(types == LATENT))
    if target == "mag" and not has_latents:
        return graph  # the MAG of a causally-sufficient DAG is the DAG itself
    n_obs, dag, anc = _observed_dag(graph, max_nodes)
    if target == "mag":
        return _mag_projection(dag, range(n_obs), anc)
    return _pag_projection(dag, range(n_obs), anc)


def _observed_dag(graph: GraphStructure, max_nodes: int):
    """Return the observed count, the full causal DiGraph, and each observed node's ancestors."""
    types = graph.node_types
    observed = np.arange(graph.n_nodes) if types is None else np.flatnonzero(types == OBSERVED)
    n_obs = int(observed.size)
    if not np.array_equal(observed, np.arange(n_obs)):
        raise ValueError(
            "marginal requires observed nodes indexed first (0 .. n_obs-1); "
            "SCM-generated truths satisfy this"
        )
    if n_obs > max_nodes:
        raise ValueError(
            f"marginal over {n_obs} observed nodes exceeds max_nodes={max_nodes} "
            f"(the projection is quadratic); pass max_nodes= to override"
        )
    dag = _causal_digraph(graph)
    return n_obs, dag, _observed_ancestors(dag, range(n_obs))


def _causal_digraph(graph: GraphStructure):
    """Return the ``networkx.DiGraph`` of a DAG, each edge from its tail to its arrowhead."""
    import networkx as nx

    typed = graph.to_edges()
    tail_is_i = typed["mark_i"] == TAIL
    parents = np.where(tail_is_i, typed["i"], typed["j"])
    children = np.where(tail_is_i, typed["j"], typed["i"])
    dag = nx.DiGraph()
    dag.add_nodes_from(range(graph.n_nodes))
    dag.add_edges_from(zip(parents.tolist(), children.tolist()))
    return dag


def _observed_ancestors(dag, observed: Iterable[int]) -> dict[int, set[int]]:
    """Return each observed node's observed ancestors in ``dag``, the node itself excluded."""
    import networkx as nx

    obs = list(observed)
    obs_set = set(obs)
    return {x: nx.ancestors(dag, x) & obs_set for x in obs}


def _mag_projection(
    dag,
    observed: Sequence[int],
    anc: Mapping[int, set[int]],
    selection: Collection[int] = (),
    anc_selection: Collection[int] = (),
) -> GraphStructure:
    """Build the MAG over ``observed``, node ``k`` being ``observed[k]``.

    Observed ``a, b`` are adjacent iff ``dag`` does not d-separate them given ``selection`` and
    their observed ancestors and those of ``selection`` (``anc_selection``), as
    :func:`_ancestral_adjacency` decides. The mark at ``a`` is a tail when ``a`` is an ancestor of
    ``b`` or of ``selection``, else an arrowhead.
    """
    up = set(anc_selection)
    support = _ancestral_adjacency(dag, observed, selection)
    rows, cols, marks = [], [], []
    for p, q in zip(*np.nonzero(np.triu(support)), strict=True):
        a, b = observed[p], observed[q]
        rows += [int(p), int(q)]
        cols += [int(q), int(p)]
        marks += [
            TAIL if a in anc[b] or a in up else ARROW,
            TAIL if b in anc[a] or b in up else ARROW,
        ]
    return csr_from_endpoints(rows, cols, marks, len(observed), kind="mag")


def _dag_index(dag) -> tuple[list, dict, list[list[int]], list[list[int]], list[int]]:
    """Return the nodes of ``dag``, their positions, parent and child lists, and ancestor masks.

    Bit ``u`` of ``masks[v]`` is set when node ``u`` is ``v`` or one of its ancestors. Raises
    ``networkx.NetworkXUnfeasible`` on a directed cycle.
    """
    import networkx as nx

    nodes = list(dag.nodes)
    index = {v: k for k, v in enumerate(nodes)}
    parents = [[index[u] for u in dag.predecessors(v)] for v in nodes]
    children = [[index[w] for w in dag.successors(v)] for v in nodes]
    masks = [0] * len(nodes)
    for v in nx.topological_sort(dag):
        k = index[v]
        mask = 1 << k
        for u in parents[k]:
            mask |= masks[u]
        masks[k] = mask
    return nodes, index, parents, children, masks


def _moral_path(
    parents: list[list[int]], children: list[list[int]], a: int, b: int, ancestral: int, inner: int
) -> bool:
    """Whether a path joins ``a`` and ``b`` in the moral graph of the ancestral set ``ancestral``.

    Every inner node of the path lies in ``inner``; both sets are bitmasks. A neighbor of ``x`` in
    that moral graph is a parent of ``x``, a child of ``x`` in the set, or another parent of such a
    child.
    """
    seen = {a}
    stack = [a]
    while stack:
        x = stack.pop()
        step = list(parents[x])
        for c in children[x]:
            if ancestral >> c & 1:
                step.append(c)
                step += parents[c]
        for y in step:
            if y == b:
                return True
            if y not in seen and inner >> y & 1:
                seen.add(y)
                stack.append(y)
    return False


def _ancestral_adjacency(dag, observed: Sequence[int], selection: Collection[int]) -> np.ndarray:
    """Return the MAG adjacency over ``observed`` as a symmetric boolean matrix.

    Observed ``a`` and ``b`` are adjacent iff ``dag`` does not d-separate them given ``selection``
    and the observed ancestors of ``a``, ``b`` and ``selection``. These nodes and ``a``, ``b`` form
    an ancestral set together with the hidden ancestors, so by the moralization criterion the pair
    is d-separated iff the moral graph of that set joins ``a`` and ``b`` by no path through hidden
    nodes alone. The moral graph of the whole DAG holds the moral graph of every ancestral set, so
    only the observed nodes it joins to ``a`` through hidden nodes are tested.

    References
    ----------
    Lauritzen, S. L., Dawid, A. P., Larsen, B. N., and Leimer, H.-G. (1990). Independence
    properties of directed Markov fields. Networks 20(5), 491-505.

    Richardson, T., and Spirtes, P. (2002). Ancestral graph Markov models. The Annals of
    Statistics 30(4), 962-1030.
    """
    _, index, parents, children, masks = _dag_index(dag)
    obs = [index[v] for v in observed]
    position = {k: p for p, k in enumerate(obs)}
    sel = [index[v] for v in selection]
    hidden = (1 << len(masks)) - 1
    for k in obs + sel:
        hidden &= ~(1 << k)
    up = 0
    for k in sel:
        up |= masks[k]
    support = np.zeros((len(obs), len(obs)), dtype=bool)
    for p, a in enumerate(obs):
        seen, stack, reached = {a}, [a], []
        while stack:  # the observed nodes the whole moral graph joins to a through hidden nodes
            x = stack.pop()
            for c in children[x]:
                for y in (c, *parents[c]):
                    if y not in seen:
                        seen.add(y)
                        (stack if hidden >> y & 1 else reached).append(y)
            for y in parents[x]:
                if y not in seen:
                    seen.add(y)
                    (stack if hidden >> y & 1 else reached).append(y)
        for b in reached:
            q = position.get(b)
            if q is not None and q > p:
                ancestral = masks[a] | masks[b] | up
                if _moral_path(parents, children, a, b, ancestral, hidden):
                    support[p, q] = support[q, p] = True
    return support


class DSepOracle:
    """A ``CITest`` answering ``X _||_ Y | Z`` by d-separation in a ground-truth DAG.

    Returns ``1.0`` when the query is d-separated (a "p-value" above any alpha -> independent) and
    ``0.0`` otherwise, so it drops into the FCI orientation rules in place of a data CI test; alpha
    is therefore irrelevant. Queries are memoized (FCI re-asks pairs constantly). Query index
    ``k`` names the DAG node ``nodes[k]``, or ``k`` itself when ``nodes`` is ``None``, and every
    query also conditions on the DAG nodes ``given``.

    Notes
    -----
    ``x`` and ``y`` are d-separated given ``Z`` iff the moral graph of the ancestral set of
    ``{x, y} | Z`` joins them by no path that avoids ``Z`` (Lauritzen et al. 1990).

    References
    ----------
    Lauritzen, S. L., Dawid, A. P., Larsen, B. N., and Leimer, H.-G. (1990). Independence
    properties of directed Markov fields. Networks 20(5), 491-505.
    """

    def __init__(
        self, dag, nodes: Sequence[int] | None = None, given: Collection[int] = ()
    ) -> None:
        _, self._index, self._parents, self._children, self._masks = _dag_index(dag)
        self._nodes = nodes
        self._given = frozenset(self._index[v] for v in given)
        self._cache: dict = {}

    def __call__(self, x: int, y: int, condition_set=()) -> float:
        """Return 1.0 if ``x`` and ``y`` are d-separated given ``condition_set``, else 0.0."""
        key = (int(x), int(y), frozenset(int(z) for z in condition_set))
        value = self._cache.get(key)
        if value is None:
            node = (lambda k: k) if self._nodes is None else self._nodes.__getitem__
            a, b = self._index[node(key[0])], self._index[node(key[1])]
            blocked, ancestral = 0, self._masks[a] | self._masks[b]
            for z in {self._index[node(z)] for z in key[2]} | self._given:
                blocked |= 1 << z
                ancestral |= self._masks[z]
            joined = _moral_path(
                self._parents, self._children, a, b, ancestral, ancestral & ~blocked
            )
            value = 0.0 if joined else 1.0
            self._cache[key] = value
        return value


class _AncestorSepsets(Mapping):
    """The separating set of each non-adjacent observed pair, built on lookup.

    The set of ``(a, c)`` holds the positions of the observed ancestors of ``a``, ``c`` and the
    selection variables, ``a`` and ``c`` excluded, in ascending order. Adjacent pairs have none.
    """

    def __init__(
        self,
        support: np.ndarray,
        observed: Sequence[int],
        anc: Mapping[int, set[int]],
        up: set[int],
    ) -> None:
        self._support = support
        self._observed = observed
        self._anc = anc
        self._up = up
        self._position = {v: k for k, v in enumerate(observed)}

    def __getitem__(self, key: tuple[int, int]) -> tuple[int, ...]:
        a, c = key
        if a == c or self._support[a, c]:
            raise KeyError(key)
        u, v = self._observed[a], self._observed[c]
        near = (self._anc[u] | self._anc[v] | self._up) - {u, v}
        return tuple(sorted(self._position[w] for w in near))

    def __iter__(self) -> Iterator[tuple[int, int]]:
        apart = ~self._support
        np.fill_diagonal(apart, False)
        return ((int(a), int(c)) for a, c in zip(*np.nonzero(apart), strict=True))

    def __len__(self) -> int:
        n = len(self._observed)
        return n * (n - 1) - int(np.count_nonzero(self._support))


def _pag_projection(
    dag,
    observed: Sequence[int],
    anc: Mapping[int, set[int]],
    selection: Collection[int] = (),
    anc_selection: Collection[int] = (),
    constraints: Constraints | None = None,
) -> GraphStructure:
    """Build the true PAG over ``observed`` via MAG-first orientation: oracle skeleton, FCI rules.

    The skeleton and separating sets are :func:`_mag_projection`'s; ``_orient_colliders`` fixes the
    unshielded colliders and ``_fci_orient`` runs Zhang's rules to a fixed point, its R4
    discriminating-path collider answered by the same d-separation oracle, which conditions on
    ``selection`` in every query. ``constraints``, over the positions of ``observed``, set their
    endpoints first, as ``fci`` sets them before its orientation rules (``_knowledge_marks``).
    """
    from andrey.constraint.fci import _circle_pag, _fci_orient, _knowledge_marks, _orient_colliders

    support = _ancestral_adjacency(dag, observed, selection)
    sepsets = _AncestorSepsets(support, observed, anc, set(anc_selection))
    marks = _circle_pag(support)
    _knowledge_marks(marks, constraints)
    _orient_colliders(marks, sepsets)
    _fci_orient(marks, DSepOracle(dag, observed, selection), 0.5, sepsets)
    return GraphStructure.from_numpy(marks, kind="pag")
