"""Conversions between graph kinds: a DAG from a CPDAG, and the MAG and PAG of a DAG.

:func:`to_dag` returns one DAG of a CPDAG's equivalence class, :func:`to_mag` projects a DAG with
hidden and selection variables onto its other nodes as a maximal ancestral graph (MAG), and
:func:`to_pag` returns the partial ancestral graph (PAG) of a DAG or a MAG, the graph FCI recovers
from data in the large-sample limit. :func:`andrey.metrics.to_cpdag` returns a DAG's CPDAG, and
``GraphStructure.to_dot`` writes any graph as Graphviz DOT text.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Collection
from typing import TYPE_CHECKING, Literal

import numpy as np

from andrey.core import ARROW, CIRCLE, LATENT, TAIL, GraphStructure, StructureOutput
from andrey.core.orient import pdag2dag

if TYPE_CHECKING:
    import networkx

__all__ = ["to_dag", "to_mag", "to_pag"]

_LISTED = 10  # undirected edges an error message lists before "and N more"


def to_dag(
    graph: GraphStructure | StructureOutput,
    *,
    tie_break: Literal["label"] | None = None,
    seed: int | None = None,
) -> GraphStructure:
    """Return a DAG from a CPDAG's Markov equivalence class.

    A CPDAG with an undirected edge stands for several DAGs, and the choice among them can change
    what a later step concludes, such as an effect estimate, so the caller picks how to choose:
    by label with ``tie_break="label"``, or at random with ``seed``.

    Parameters
    ----------
    graph : GraphStructure or StructureOutput
        A ``cpdag``, or a ``dag``, or a result that holds one.
    tie_break : {"label"} or None, default=None
        ``"label"`` picks the DAG the node labels decide, whatever the column order: the same DAG,
        in label terms, for every column order of a labeled graph. A graph without labels is
        named by position, so its DAG follows the column order.
    seed : int or None, default=None
        Picks a DAG of the class at random, the same one for the same seed. Every DAG of the class
        can come back, but not with equal probability.

    Returns
    -------
    GraphStructure
        A ``dag`` with the nodes and labels of ``graph``, its skeleton, its v-structures, and
        each of its directed edges, so a DAG of its class. A graph with no undirected edge has
        one such DAG, which comes back whatever ``tie_break`` and ``seed`` say.

    Raises
    ------
    TypeError
        If ``graph`` is not a ``GraphStructure`` or a result that holds one.
    ValueError
        If ``graph`` is not a ``dag`` or a ``cpdag``; if no DAG has its skeleton, its
        v-structures, and its directed edges, with a message that names a chordless undirected
        cycle of four or more nodes when there is one, since every orientation of such a cycle
        adds a v-structure or closes a directed cycle; if ``graph`` has an undirected edge and
        neither ``tie_break`` nor ``seed`` is given, with a message that names the undirected
        edges; if both are given; or if ``tie_break`` is not ``"label"`` or ``None``.

    Notes
    -----
    The algorithm of Dor and Tarsi [1]_ keeps every directed edge, then repeatedly removes the
    first node, in a ranking of the nodes, that has no outgoing directed edge and whose undirected
    neighbors are each adjacent to all its other neighbors, and directs its undirected edges into
    it. When no node qualifies before every node is removed, no extension exists.
    ``tie_break="label"`` ranks the nodes by label, and ``seed`` by a random permutation. A DAG of
    the class comes back from every ranking that lists its nodes from the last in its causal order
    to the first, so each DAG has a positive probability under ``seed``; DAGs that more rankings
    reach are more likely.

    References
    ----------
    .. [1] Dor, D., and Tarsi, M. (1992). A simple algorithm to construct a consistent extension
       of a partially oriented graph. Technical Report R-185, Cognitive Systems Laboratory, UCLA.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import GraphStructure
    >>> from andrey.graph import to_dag
    >>> M = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])  # a -- b -- c
    >>> cpdag = GraphStructure.from_numpy(M, kind="cpdag", labels=("a", "b", "c"))
    >>> dag = to_dag(cpdag, tie_break="label")
    >>> dag.kind, dag.oriented_edges()
    ('dag', [('b', 'a', 'directed'), ('c', 'b', 'directed')])
    """
    structure = _graph(graph)
    if structure.kind not in ("dag", "cpdag"):
        raise ValueError(f"to_dag takes a dag or a cpdag, not a {structure.kind}")
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
    extension = pdag2dag(marks[np.ix_(rank, rank)])
    back = np.empty(n, dtype=np.intp)
    back[rank] = np.arange(n)
    return GraphStructure.from_numpy(
        extension[np.ix_(back, back)], kind="dag", labels=structure.labels
    )


def to_mag(
    dag: GraphStructure | StructureOutput,
    *,
    latent: Collection[str | int] | None = None,
    selection: Collection[str | int] = (),
) -> GraphStructure:
    """Project a DAG with hidden and selection variables onto its other nodes as a MAG.

    Parameters
    ----------
    dag : GraphStructure or StructureOutput
        A ``dag``, or a result that holds one.
    latent : collection of str or int, or None, default=None
        The hidden variables, each a label or a 0-based position. ``None`` takes the nodes that
        ``node_types`` flags as latent, and none when it flags none.
    selection : collection of str or int, default=()
        The selection variables, each a label or a 0-based position: variables the data are
        conditioned on, such as one that decides which units enter the sample.

    Returns
    -------
    GraphStructure
        The MAG over the other nodes, in their order and with their labels, of kind ``"pag"``
        with no circle. Two nodes are adjacent when no set of the other observed nodes, with the
        selection variables, d-separates them. The mark at ``a`` on its edge with ``b`` is a tail
        when ``a`` is an ancestor of ``b`` or of a selection variable, and an arrowhead
        otherwise: ``a <-> b`` relates two nodes through hidden common causes, and ``a -- b`` two
        ancestors of the selection.

    Raises
    ------
    TypeError
        If ``dag`` is not a ``GraphStructure`` or a result that holds one, or ``latent`` or
        ``selection`` is a single string.
    ValueError
        If ``dag`` is not a ``dag`` or has a directed cycle, or ``latent`` or ``selection`` names
        a node the graph does not have, names a node twice, or shares a node with the other.

    Notes
    -----
    The construction is Richardson and Spirtes' [1]_. Observed ``a`` and ``b`` are adjacent
    exactly when the DAG does not d-separate them given the selection variables and the observed
    ancestors of ``a``, ``b``, and the selection variables, so one d-separation query decides
    each pair of observed nodes.

    References
    ----------
    .. [1] Richardson, T., and Spirtes, P. (2002). Ancestral graph Markov models. The Annals of
       Statistics 30(4), 962-1030.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import GraphStructure
    >>> from andrey.graph import to_mag
    >>> M = np.zeros((3, 3), dtype=int)
    >>> M[0, [1, 2]], M[[1, 2], 0] = 1, 2  # u -> a, u -> b
    >>> dag = GraphStructure.from_numpy(M, kind="dag", labels=("u", "a", "b"))
    >>> to_mag(dag, latent=["u"]).oriented_edges()
    [('a', 'b', 'bidirected')]
    """
    structure = _graph(dag)
    if structure.kind != "dag":
        raise ValueError(f"to_mag takes a dag, not a {structure.kind}")
    digraph = _causal_dag(structure)
    observed, selected = _split(structure, latent, selection)
    return _named(_project(digraph, observed, selected, "mag"), structure, observed)


def to_pag(
    graph: GraphStructure | StructureOutput,
    *,
    latent: Collection[str | int] | None = None,
    selection: Collection[str | int] = (),
) -> GraphStructure:
    """Return the PAG of a DAG or a MAG, after projecting out hidden and selection variables.

    Parameters
    ----------
    graph : GraphStructure or StructureOutput
        A ``dag``, or a MAG: a ``pag`` with no circle, such as :func:`to_mag` returns. Or a
        result that holds one.
    latent : collection of str or int, or None, default=None
        Nodes of ``graph`` to treat as hidden, each a label or a 0-based position. ``None``
        takes the nodes that ``node_types`` flags as latent, and none when it flags none.
    selection : collection of str or int, default=()
        Nodes of ``graph`` to treat as selection variables, each a label or a 0-based position.

    Returns
    -------
    GraphStructure
        The PAG over the other nodes, in their order and with their labels: the MAG of
        :func:`to_mag`, with each mark that some Markov-equivalent MAG does not share replaced by
        a circle. A tail or an arrowhead is a mark every MAG of the equivalence class has.

    Raises
    ------
    TypeError
        If ``graph`` is not a ``GraphStructure`` or a result that holds one, or ``latent`` or
        ``selection`` is a single string.
    ValueError
        If ``graph`` is not a ``dag`` or a ``pag``, a ``dag`` has a directed cycle, a ``pag`` has
        a circle or is not ancestral and maximal, or ``latent`` or ``selection`` names a node the
        graph does not have, names a node twice, or shares a node with the other.

    Notes
    -----
    A MAG is read through its canonical DAG [1]_: each ``a <-> b`` becomes a hidden common
    parent of ``a`` and ``b``, and each ``a -- b`` a selection variable that both ``a`` and
    ``b`` point into. The MAG of that DAG is the input exactly when the input is a MAG. The PAG
    then comes from the MAG's skeleton and separating sets: its unshielded colliders, then the
    orientation rules R1-R10 of Zhang [2]_, which are complete, applied to a fixed point. These
    are the rules :func:`andrey.fci` and :func:`andrey.gfci` apply, with the conditional
    independence test of R4 answered by d-separation in the DAG. R4 and R5 search a bounded
    number of paths, a bound small graphs do not reach.

    References
    ----------
    .. [1] Richardson, T., and Spirtes, P. (2002). Ancestral graph Markov models. The Annals of
       Statistics 30(4), 962-1030.
    .. [2] Zhang, J. (2008). On the completeness of orientation rules for causal discovery in the
       presence of latent confounders and selection bias. Artificial Intelligence 172(16-17),
       1873-1896.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import GraphStructure
    >>> from andrey.graph import to_pag
    >>> M = np.zeros((3, 3), dtype=int)
    >>> M[[0, 1], 2], M[2, [0, 1]] = 1, 2  # a -> c <- b
    >>> pag = to_pag(GraphStructure.from_numpy(M, kind="dag", labels=("a", "b", "c")))
    >>> pag.oriented_edges()
    [('a', 'c', 'partially_directed'), ('b', 'c', 'partially_directed')]
    """
    structure = _graph(graph)
    if structure.kind == "dag":
        digraph = _causal_dag(structure)
        observed, selected = _split(structure, latent, selection)
        return _named(_project(digraph, observed, selected, "pag"), structure, observed)
    if structure.kind != "pag":
        raise ValueError(
            f"to_pag takes a dag or a MAG, a pag without circles, not a {structure.kind}"
        )
    digraph, added_selection = _canonical_dag(structure)
    observed, selected = _split(structure, latent, selection)
    return _named(
        _project(digraph, observed, [*selected, *added_selection], "pag"), structure, observed
    )


def _graph(graph: GraphStructure | StructureOutput) -> GraphStructure:
    """Return the ``GraphStructure`` that ``graph`` is or holds."""
    structure = graph.structure if isinstance(graph, StructureOutput) else graph
    if not isinstance(structure, GraphStructure):
        raise TypeError(
            f"expected a GraphStructure or a result that holds one, got {type(structure).__name__}"
        )
    return structure


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
        raise ValueError("the pag has circles, so it is not a MAG; to_pag takes a dag or a MAG")
    n = mag.n_nodes
    digraph = nx.DiGraph()
    digraph.add_nodes_from(range(n))
    added_selection: list[int] = []
    for row in mag.to_edges():
        i, j, mark_i, mark_j = (int(value) for value in row)
        if i == j:
            raise ValueError(f"a MAG has no self-loop, and the pag has one at node {i}")
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
            "the pag is not a MAG: it is not ancestral or not maximal, so no DAG projects onto it"
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
    digraph: networkx.DiGraph, observed: list[int], selection: list[int], target: str
) -> GraphStructure:
    """Return the MAG or the PAG over ``observed``, node ``k`` being ``observed[k]``.

    Every d-separation query also conditions on ``selection``.
    """
    from andrey.data.latent import _mag_projection, _observed_ancestors, _pag_projection

    anc = _observed_ancestors(digraph, observed)
    up = _observed_ancestors_of(digraph, observed, selection)
    build = _mag_projection if target == "mag" else _pag_projection
    return build(digraph, observed, anc, selection, up)


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
