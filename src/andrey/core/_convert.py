"""Conversions behind ``GraphStructure.to_dag``, ``to_mag``, and ``to_pag``.

The methods' docstrings describe the conversions and cite their algorithms.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Collection
from typing import TYPE_CHECKING, Literal

import numpy as np

from .orient import pdag2dag
from .structure import ARROW, CIRCLE, LATENT, TAIL, GraphStructure, is_ancestral_kind

if TYPE_CHECKING:
    import networkx

    from .knowledge import Constraints

_LISTED = 10  # undirected edges an error message lists before "and N more"


def dag_extension(
    structure: GraphStructure, tie_break: Literal["label"] | None, seed: int | None
) -> GraphStructure:
    """Return a DAG of ``structure``'s class, chosen by ``tie_break`` or ``seed``."""
    if structure.kind not in ("dag", "cpdag"):
        raise ValueError(f"to_dag takes a dag or a cpdag, not kind {structure.kind!r}")
    if tie_break not in ("label", None):
        raise ValueError(f'tie_break must be "label" or None, got {tie_break!r}')
    if tie_break is not None and seed is not None:
        raise ValueError("pass tie_break or seed, not both")
    structure.validate()
    marks = structure.to_numpy()
    first = pdag2dag(marks)  # raises when no extension exists
    undirected = [(i, j) for i, j, a, b in structure.to_edges().tolist() if a == b == TAIL]
    if not undirected:
        return GraphStructure.from_numpy(first, kind="dag", labels=structure.labels)
    if tie_break is None and seed is None:
        names = [" -- ".join(map(str, _names(structure, [u, v]))) for u, v in undirected[:_LISTED]]
        more = f" and {len(undirected) - _LISTED} more" if len(undirected) > _LISTED else ""
        raise ValueError(
            f"the {structure.kind} has {len(undirected)} undirected edge(s), "
            f'{", ".join(names)}{more}, so more than one DAG fits it; pass tie_break="label" '
            "to pick the DAG its labels decide, or seed= to pick one at random"
        )
    n = structure.n_nodes
    if seed is not None:
        rank = np.random.default_rng(seed).permutation(n)
    elif structure.labels is None:
        rank = np.arange(n)
    else:
        rank = np.argsort(np.array(structure.labels, dtype=object), kind="stable")
    return GraphStructure.from_numpy(
        extend_in_order(marks, rank), kind="dag", labels=structure.labels
    )


def extend_in_order(marks: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Return the consistent extension (Dor and Tarsi) of a PDAG's marks that ``order`` picks.

    ``order`` lists the nodes in the order the extension removes them when more than one is
    removable, so it decides the tied edges. Raises ``ValueError`` when no extension exists.
    """
    extension = pdag2dag(marks[np.ix_(order, order)])
    back = np.empty(len(order), dtype=np.intp)
    back[order] = np.arange(len(order))
    return extension[np.ix_(back, back)]


def mag_of(
    structure: GraphStructure,
    latent: Collection[str | int] | None,
    selection: Collection[str | int],
) -> GraphStructure:
    """Return the MAG of a ``dag`` over the nodes neither ``latent`` nor ``selection`` names."""
    if structure.kind != "dag":
        raise ValueError(f"to_mag takes a dag, not kind {structure.kind!r}")
    digraph = _causal_dag(structure)
    observed, selected = _split(structure, latent, selection)
    return _named(_project(digraph, observed, selected, "mag"), structure, observed)


def pag_of(
    structure: GraphStructure,
    latent: Collection[str | int] | None,
    selection: Collection[str | int],
    constraints: Constraints | None = None,
) -> GraphStructure:
    """Return the PAG of a ``dag`` or a MAG over the nodes ``latent`` and ``selection`` leave.

    ``constraints``, over those nodes, set their endpoints before the orientation rules run, as
    ``fci`` sets them; ``None`` sets none.
    """
    if structure.kind == "dag":
        digraph = _causal_dag(structure)
        observed, selected = _split(structure, latent, selection)
        return _named(
            _project(digraph, observed, selected, "pag", constraints), structure, observed
        )
    if not is_ancestral_kind(structure.kind):
        raise ValueError(f"to_pag takes a dag or a mag, not kind {structure.kind!r}")
    digraph, added_selection = _canonical_dag(structure)
    observed, selected = _split(structure, latent, selection)
    return _named(
        _project(digraph, observed, [*selected, *added_selection], "pag", constraints),
        structure,
        observed,
    )


def mag_of_pag(pag: np.ndarray) -> np.ndarray:
    """Return the MAG that Zhang's construction (2008, Theorem 2) builds from a PAG's mark matrix.

    A circle opposite an arrowhead becomes a tail (``o->`` to ``->``), and the ``o-o`` edges are
    oriented into a DAG with no unshielded collider, a consistent extension (Dor and Tarsi) of
    them as an undirected graph. For a PAG, the result is a MAG of its equivalence class. A circle
    opposite a tail, which a PAG without selection variables does not have, becomes an arrowhead,
    as pcalg's ``pag2magAM`` does. Raises ``ValueError`` when the ``o-o`` edges have no such
    orientation, which happens when they hold a chordless cycle of four or more nodes, so the input
    is not a PAG.
    """
    mag = pag.copy()
    circle = pag == CIRCLE
    both = circle & circle.T
    mag[circle & (pag.T == ARROW)] = TAIL
    mag[circle & (pag.T == TAIL)] = ARROW
    if both.any():
        dag = pdag2dag(np.where(both, TAIL, 0).astype(np.int8))
        mag[both] = dag[both]
    return mag


def _causal_dag(structure: GraphStructure) -> networkx.DiGraph:
    """Return the ``networkx.DiGraph`` of a ``dag``, each edge from tail to arrowhead.

    Raises ``ValueError`` on a directed cycle.
    """
    import networkx as nx

    from andrey.data.latent import _causal_digraph

    structure.validate()
    digraph = _causal_digraph(structure)
    if not nx.is_directed_acyclic_graph(digraph):
        cycle = " -> ".join(str(u) for u, _ in nx.find_cycle(digraph))
        raise ValueError(f"the dag has the directed cycle {cycle}")
    return digraph


def _canonical_dag(mag: GraphStructure) -> tuple[networkx.DiGraph, list[int]]:
    """Return the canonical DAG of a MAG and its added selection nodes, or raise for a non-MAG.

    Nodes ``0 .. n-1`` are the MAG's. Each ``a <-> b`` adds a hidden parent of ``a`` and ``b``, and
    each ``a -- b`` a selection child of both, numbered from ``n``.
    """
    import networkx as nx

    from andrey.data.latent import _mag_projection, _observed_ancestors

    mag.validate()
    marks = mag.to_numpy()
    if (marks == CIRCLE).any():
        raise ValueError("the pag has circles, so it is not a MAG; to_pag takes a dag or a mag")
    n = mag.n_nodes
    digraph = nx.DiGraph()
    digraph.add_nodes_from(range(n))
    added_selection: list[int] = []
    for row in mag.to_edges():
        i, j, mark_i, mark_j = (int(value) for value in row)
        if i == j:
            raise ValueError(f"a MAG has no self-loop, and the {mag.kind} has one at node {i}")
        if (mark_i, mark_j) == (TAIL, ARROW):
            digraph.add_edge(i, j)
        elif (mark_i, mark_j) == (ARROW, TAIL):
            digraph.add_edge(j, i)
        elif mark_i == mark_j == ARROW:
            hidden = digraph.number_of_nodes()
            digraph.add_edges_from([(hidden, i), (hidden, j)])
        else:
            selected = digraph.number_of_nodes()
            digraph.add_edges_from([(i, selected), (j, selected)])
            added_selection.append(selected)
    nodes = range(n)
    is_mag = nx.is_directed_acyclic_graph(digraph)
    if is_mag:
        anc = _observed_ancestors(digraph, nodes)
        up = _observed_ancestors_of(digraph, nodes, added_selection)
        projected = _mag_projection(digraph, nodes, anc, added_selection, up)
        is_mag = np.array_equal(projected.to_numpy(), marks)
    if not is_mag:
        raise ValueError(
            f"the {mag.kind} is not a MAG: it is not ancestral or not maximal, so no DAG projects "
            "onto it"
        )
    return digraph, added_selection


def _observed_ancestors_of(
    digraph: networkx.DiGraph, observed: Collection[int], nodes: Collection[int]
) -> set[int]:
    """Return the observed ancestors of ``nodes`` in ``digraph``."""
    import networkx as nx

    found: set[int] = set()
    for node in nodes:
        found |= nx.ancestors(digraph, node)
    return found & set(observed)


def _project(
    digraph: networkx.DiGraph,
    observed: list[int],
    selection: list[int],
    target: str,
    constraints: Constraints | None = None,
) -> GraphStructure:
    """Return the MAG or the PAG over ``observed``, node ``k`` being ``observed[k]``.

    Every d-separation query also conditions on ``selection``. A PAG gets the endpoints
    ``constraints`` set before its orientation rules.
    """
    from andrey.data.latent import _mag_projection, _observed_ancestors, _pag_projection

    anc = _observed_ancestors(digraph, observed)
    up = _observed_ancestors_of(digraph, observed, selection)
    if target == "mag":
        return _mag_projection(digraph, observed, anc, selection, up)
    return _pag_projection(digraph, observed, anc, selection, up, constraints)


def _named(
    projected: GraphStructure, structure: GraphStructure, observed: list[int]
) -> GraphStructure:
    """Return ``projected`` named by the labels of ``structure``'s ``observed`` nodes, if any."""
    if structure.labels is None:
        return projected
    return dataclasses.replace(projected, _labels=tuple(structure.labels[k] for k in observed))


def _split(
    structure: GraphStructure,
    latent: Collection[str | int] | None,
    selection: Collection[str | int],
) -> tuple[list[int], list[int]]:
    """Return the observed and the selection node positions of ``structure``."""
    if latent is None:
        types = structure.node_types
        hidden = [] if types is None else np.flatnonzero(types == LATENT).tolist()
    else:
        hidden = _positions(structure, latent, "latent")
    selected = _positions(structure, selection, "selection")
    shared = sorted(set(hidden) & set(selected))
    if shared:
        raise ValueError(f"latent and selection share the node(s) {_names(structure, shared)}")
    removed = set(hidden) | set(selected)
    observed = [k for k in range(structure.n_nodes) if k not in removed]
    return observed, selected


def _positions(structure: GraphStructure, nodes: Collection[str | int], what: str) -> list[int]:
    """Return the 0-based positions of ``nodes``: a string is a label, an integer a position."""
    if isinstance(nodes, str):
        raise TypeError(f"{what} takes a collection of nodes, such as [{nodes!r}], not a string")
    labels = structure.labels
    n = structure.n_nodes
    found: list[int] = []
    for node in nodes:
        if isinstance(node, str):
            if labels is None or node not in labels:
                have = f"its labels are {list(labels)}" if labels else "it has no labels"
                raise ValueError(f"{what} names the node {node!r}, which the graph lacks; {have}")
            found.append(labels.index(node))
        elif isinstance(node, (int, np.integer)) and not isinstance(node, bool):
            if not 0 <= node < n:
                raise ValueError(f"{what} lists the node {node}, but the graph has {n} nodes")
            found.append(int(node))
        else:
            raise TypeError(f"{what} takes labels or 0-based positions, got {node!r}")
    if len(set(found)) < len(found):
        raise ValueError(f"{what} names a node more than once")
    return found


def _names(structure: GraphStructure, positions: list[int]) -> list[str | int]:
    """Return the labels of ``positions``, or the positions when the graph has no labels."""
    labels = structure.labels
    return [labels[k] for k in positions] if labels is not None else list(positions)
