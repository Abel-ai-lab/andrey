"""Latent-confounder handling: designate hidden nodes, drop them, keep truth self-describing.

A latent SCM generates over the full node set, then hides a subset of nodes (dropping their columns
from the data) while keeping them in the ground-truth graph flagged via ``node_types`` LATENT.
Latents are chosen among nodes with at least two children -- common causes -- so hiding them
actually induces confounding, directly or transitively (a leaf or single-child latent would
project away to nothing). Observed nodes are re-indexed first so the data columns align 1:1 with
the observed truth nodes.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence

import numpy as np

from andrey.core.structure import ARROW, LATENT, OBSERVED, TAIL, GraphStructure

from .graphs import csr_from_endpoints


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


def with_latents(
    data: np.ndarray, edges: np.ndarray, d: int, n_latent: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hide ``n_latent`` confounders: drop their columns, re-index observed-first, flag them.

    Parameters
    ----------
    data : np.ndarray of shape (n, d)
        Full data over all nodes, node-label columns.
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
    data_observed : np.ndarray of shape (n, d - n_latent)
        Data over the observed nodes, columns in the re-indexed observed order (``0 .. n_obs-1``).
    new_edges : np.ndarray of shape (m, 2)
        Edges re-indexed so observed nodes are ``0 .. n_obs-1`` and latents are last.
    node_types : np.ndarray of shape (d,)
        OBSERVED / LATENT code per (re-indexed) node.

    Raises
    ------
    ValueError
        If fewer than ``n_latent`` nodes have >= 2 children (no genuine confounder to hide).
    """
    order = latent_order(edges, d, n_latent, rng)  # new index -> old label (observed first)
    n_obs = d - n_latent
    new_label = np.empty(d, dtype=np.int64)
    new_label[order] = np.arange(d)  # old label -> new index
    node_types = np.full(d, OBSERVED, dtype=np.int8)
    node_types[n_obs:] = LATENT
    return data[:, order[:n_obs]], new_label[edges], node_types


def marginal(graph: GraphStructure, target: str = "mag", *, max_nodes: int = 500) -> GraphStructure:
    """Project a latent-confounded DAG truth onto its observed nodes as a MAG or a PAG.

    Both share one skeleton -- observed ``a, b`` are adjacent iff they are not d-separated (in the
    full DAG) by their observed ancestors, the ancestor-set characterization -- so the cost is
    ``O(n_obs^2 * (V + E))`` d-separation queries, affordable for sparse-latent graphs.

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
        The MAG or PAG over the observed nodes, ``kind="pag"``.

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
            f"(the all-pairs projection is O(n_obs^2 * (V+E))); pass max_nodes= to override"
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
    their observed ancestors and those of ``selection`` (``anc_selection``). The mark at ``a`` is
    a tail when ``a`` is an ancestor of ``b`` or of ``selection``, else an arrowhead.
    """
    import networkx as nx

    given = set(selection)
    up = set(anc_selection)
    rows, cols, marks = [], [], []
    for p, a in enumerate(observed):
        for q in range(p + 1, len(observed)):
            b = observed[q]
            sep = (anc[a] | anc[b] | up) - {a, b}
            if not nx.is_d_separator(dag, {a}, {b}, sep | given):
                mark_a = TAIL if a in anc[b] or a in up else ARROW
                mark_b = TAIL if b in anc[a] or b in up else ARROW
                rows += [p, q]
                cols += [q, p]
                marks += [mark_a, mark_b]
    return csr_from_endpoints(rows, cols, marks, len(observed), kind="pag")


class DSepOracle:
    """A ``CITest`` answering ``X _||_ Y | Z`` by d-separation in a ground-truth DAG.

    Returns ``1.0`` when the query is d-separated (a "p-value" above any alpha -> independent) and
    ``0.0`` otherwise, so it drops into the FCI orientation rules in place of a data CI test; alpha
    is therefore irrelevant. Queries are memoized (FCI re-asks pairs constantly). Query index
    ``k`` names the DAG node ``nodes[k]``, or ``k`` itself when ``nodes`` is ``None``, and every
    query also conditions on the DAG nodes ``given``.
    """

    def __init__(
        self, dag, nodes: Sequence[int] | None = None, given: Collection[int] = ()
    ) -> None:
        self._dag = dag
        self._nodes = nodes
        self._given = frozenset(given)
        self._cache: dict = {}

    def __call__(self, x: int, y: int, condition_set=()) -> float:
        """Return 1.0 if ``x`` and ``y`` are d-separated given ``condition_set``, else 0.0."""
        key = (int(x), int(y), frozenset(int(z) for z in condition_set))
        value = self._cache.get(key)
        if value is None:
            import networkx as nx

            node = (lambda k: k) if self._nodes is None else self._nodes.__getitem__
            sep = {node(z) for z in key[2]} | self._given
            separated = nx.is_d_separator(self._dag, {node(key[0])}, {node(key[1])}, sep)
            value = 1.0 if separated else 0.0
            self._cache[key] = value
        return value


def _pag_projection(
    dag,
    observed: Sequence[int],
    anc: Mapping[int, set[int]],
    selection: Collection[int] = (),
    anc_selection: Collection[int] = (),
) -> GraphStructure:
    """Build the true PAG over ``observed`` via MAG-first orientation: oracle skeleton, FCI rules.

    The skeleton and separating sets are :func:`_mag_projection`'s; ``_orient_colliders`` fixes the
    unshielded colliders and ``_fci_orient`` runs Zhang's rules to a fixed point, its R4
    discriminating-path collider answered by the same d-separation oracle, which conditions on
    ``selection`` in every query.
    """
    from andrey.constraint.fci import _circle_pag, _fci_orient, _orient_colliders

    oracle = DSepOracle(dag, observed, selection)
    position = {v: k for k, v in enumerate(observed)}
    up = set(anc_selection)
    n_obs = len(observed)
    support = np.zeros((n_obs, n_obs), dtype=bool)
    sepsets: dict[tuple[int, int], tuple[int, ...]] = {}
    for a in range(n_obs):
        for c in range(a + 1, n_obs):
            near = (anc[observed[a]] | anc[observed[c]] | up) - {observed[a], observed[c]}
            sep = tuple(sorted(position[v] for v in near))
            if (
                oracle(a, c, sep) > 0.5
            ):  # d-separated -> non-adjacent; record the separator both ways
                sepsets[(a, c)] = sep
                sepsets[(c, a)] = sep
            else:
                support[a, c] = support[c, a] = True
    marks = _circle_pag(support)
    _orient_colliders(marks, sepsets)
    _fci_orient(marks, oracle, 0.5, sepsets)
    return GraphStructure.from_numpy(marks, kind="pag")
