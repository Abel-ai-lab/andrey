"""Covariate adjustment for the total effect of a treatment on an outcome, read from a graph.

A set ``Z`` is an adjustment set for the total effect of ``x`` on ``y`` when the interventional
distribution is the observational one averaged over ``Z``,
``f(y | do(x)) = ∫ f(y | x, z) f(z) dz``, for every distribution the graph allows; for a CPDAG,
in every DAG of its class, and for a MAG or PAG, in every DAG with hidden variables it stands
for. The generalized adjustment criterion of Perković et al. (2018) characterizes these sets in a
DAG, a CPDAG, a MAG, and a PAG: the graph is amenable for the pair, ``Z`` holds no forbidden node,
and ``Z`` blocks every non-causal path from ``x`` to ``y`` that the graph can read. Functions here
take a :class:`~andrey.core.structure.GraphStructure` and nodes by 0-based position;
``GraphStructure.is_adjustment_set`` and ``optimal_adjustment_set`` name nodes by label.

A path is possibly directed (possibly causal) from its first node to its last when no edge on it
has an arrowhead at the end nearer the first node: ``a -> b``, ``a -- b``, ``a o-> b``, and
``a o-o b`` each lead on from ``a`` to ``b``. In a DAG every such path is directed. The possible
descendants of a node are the nodes such a path reaches from it, the node included.

References
----------
Henckel, L., Perković, E., and Maathuis, M. H. (2022). Graphical criteria for efficient total
effect estimation via adjustment in causal linear models. Journal of the Royal Statistical Society
Series B 84(2), 579-599.

Maathuis, M. H., Kalisch, M., and Bühlmann, P. (2009). Estimating high-dimensional intervention
effects from observational data. The Annals of Statistics 37(6A), 3133-3164.

Perković, E., Textor, J., Kalisch, M., and Maathuis, M. H. (2018). Complete graphical
characterization and construction of adjustment sets in Markov equivalence classes of ancestral
graphs. Journal of Machine Learning Research 18(220), 1-62.

Shpitser, I., VanderWeele, T., and Robins, J. M. (2010). On the validity of covariate adjustment
for estimating causal effects. Proceedings of the 26th Conference on Uncertainty in Artificial
Intelligence, AUAI Press, 527-536.

van der Zander, B., Liśkiewicz, M., and Textor, J. (2019). Separators and adjustment sets in causal
graphs: complete criteria and an algorithmic framework. Artificial Intelligence 270, 1-40.

Zhang, J. (2008). Causal reasoning with ancestral graphs. Journal of Machine Learning Research
9(47), 1437-1474.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable
from itertools import combinations

import numpy as np

from .structure import ARROW, NULL, TAIL, GraphStructure

#: The graph kinds the adjustment criterion is defined for here.
KINDS: tuple[str, ...] = ("dag", "cpdag", "mag", "pag")


def directed(marks: np.ndarray) -> np.ndarray:
    """``D[i, j]`` when the marks hold ``i -> j``: a tail at ``i`` and an arrowhead at ``j``."""
    return (marks == TAIL) & (marks.T == ARROW)


def undirected(marks: np.ndarray) -> np.ndarray:
    """``U[i, j]`` (symmetric) when the marks hold ``i -- j``."""
    return (marks == TAIL) & (marks.T == TAIL)


def reach(step: np.ndarray, start: Iterable[int]) -> set[int]:
    """The nodes ``start`` reaches along ``step`` (``step[i, j]`` an edge from ``i`` to ``j``),
    ``start`` included."""
    seen = set(start)
    stack = list(seen)
    while stack:
        for w in np.flatnonzero(step[stack.pop()]).tolist():
            if w not in seen:
                seen.add(w)
                stack.append(w)
    return seen


def checked(graph: GraphStructure, what: str, kinds: Collection[str] = KINDS) -> np.ndarray:
    """The marks of ``graph`` after checking that ``what`` is defined for it.

    Raises ``ValueError`` for a kind outside ``kinds``; for a ``dag`` with a directed cycle; for a
    ``cpdag`` that is not the CPDAG of its own class, such as an MPDAG that background knowledge
    oriented further, or that no DAG fits; and for a ``mag`` or ``pag`` that is not ancestral or
    that has an undirected edge, which marks selection bias.
    """
    from . import _queries
    from .orient import cpdag_of_pdag

    if graph.kind not in kinds:
        names = ", ".join(repr(k) for k in kinds)
        raise ValueError(f"{what} is defined for graphs of kind {names}, not {graph.kind!r}")
    graph.validate()
    marks = graph.to_numpy()
    if graph.kind == "cpdag":
        if not np.array_equal(cpdag_of_pdag(marks.astype(np.int8)), marks):
            raise ValueError(
                f"{what} reads a cpdag as a Markov equivalence class, and this graph has edges "
                "oriented beyond its class's CPDAG, as background knowledge orients an MPDAG"
            )
        return marks
    if graph.kind in ("mag", "pag") and undirected(marks).any():
        raise ValueError(
            f"{what} is defined for a {graph.kind} without selection bias, and this one has an "
            "undirected edge"
        )
    _queries.member(graph, what)  # raises for a directed cycle or a graph that is not ancestral
    return marks


def possible_steps(marks: np.ndarray) -> np.ndarray:
    """``S[a, b]`` when a possibly directed path can go from ``a`` to ``b``: the edge has no
    arrowhead at ``a``."""
    return (marks != NULL) & (marks != ARROW)


def possible_descendants(marks: np.ndarray, start: Iterable[int]) -> set[int]:
    """The possible descendants of ``start``, ``start`` included."""
    return reach(possible_steps(marks), start)


def possible_ancestors(marks: np.ndarray, start: Iterable[int]) -> set[int]:
    """The possible ancestors of ``start``, ``start`` included."""
    return reach(possible_steps(marks).T, start)


def _toward(marks: np.ndarray, x: int, y: int) -> tuple[np.ndarray, set[int]]:
    """The possibly directed steps that do not enter ``x``, and the nodes that reach ``y`` along
    them: the nodes a proper possibly directed path from ``x`` can continue from to ``y``."""
    step = possible_steps(marks)
    step[:, x] = False  # a proper path does not return to x
    return step, reach(step.T, [y]) - {x}


def _causal_children(marks: np.ndarray, x: int, y: int) -> list[int]:
    """The nodes ``c`` that begin a proper possibly directed path ``x, c, ..., y``."""
    step, onward = _toward(marks, x, y)
    return [c for c in np.flatnonzero(step[x]).tolist() if c in onward]


def _causal_nodes(marks: np.ndarray, x: int, y: int) -> set[int]:
    """The nodes other than ``x`` that a possibly directed path from ``x`` reaches without
    returning to ``x`` and that reach ``y`` the same way.

    Each lies on a proper possibly directed path from ``x`` to ``y``, or is a possible descendant
    of a node that does: the two paths meet first at such a node. So their possible descendants
    are the forbidden set exactly; in a ``dag`` they are the causal nodes themselves.
    """
    step, onward = _toward(marks, x, y)
    return reach(step, np.flatnonzero(step[x]).tolist()) & onward


def forbidden(graph: GraphStructure, x: int, y: int) -> set[int]:
    """The forbidden set of the pair: the possible descendants of every node other than ``x`` on
    a proper possibly directed path from ``x`` to ``y`` (Perković et al. 2018, Definition 3), empty
    when there is none. No adjustment set holds one of them."""
    marks = checked(graph, "the forbidden set")
    _pair(graph, x, y)
    return possible_descendants(marks, _causal_nodes(marks, x, y))


def visible(marks: np.ndarray, kind: str, a: int, b: int) -> bool:
    """Whether the edge ``a -> b`` is visible: in a ``dag`` or ``cpdag`` every directed edge is;
    in a ``mag`` or ``pag`` (Zhang 2008, Definition 8), some node ``c`` not adjacent to ``b`` has an
    edge into ``a``, or a collider path into ``a`` whose every inner node is a parent of ``b``."""
    if not (marks[a, b] == TAIL and marks[b, a] == ARROW):
        return False
    if kind in ("dag", "cpdag"):
        return True
    adjacent_b = (marks[b] != NULL) | (np.arange(len(marks)) == b)
    into = marks == ARROW  # into[v, w]: the edge v *-* w has an arrowhead at v
    parent_of_b = directed(marks)[:, b]
    both = into & into.T
    # The inner nodes of a collider path into a: joined by <-> edges, each a parent of b.
    first = [v for v in np.flatnonzero(both[a]).tolist() if parent_of_b[v]]
    inner = reach(both & parent_of_b[None, :], first) - {a}
    for v in [a, *inner]:
        for c in np.flatnonzero(into[v]).tolist():  # c *-> v
            if c != a and not adjacent_b[c] and c not in inner:
                return True
    return False


def optimal_set(graph: GraphStructure, x: int, y: int) -> list[int] | None:
    """The optimal adjustment set of Henckel et al. (2022), ascending, or ``None`` when it is not
    defined: no adjustment set exists, or ``x`` is not an ancestor of ``y``.

    In a ``dag``, the parents of the causal nodes (the nodes other than ``x`` on a causal path
    from ``x`` to ``y``) that are not forbidden and are not ``x``. In a ``cpdag`` for which an
    adjustment set exists, every DAG of the class has the same causal nodes, forbidden set, and
    optimal set, read here from one of them.
    """
    if graph.kind in ("mag", "pag"):
        raise ValueError(
            "the optimal adjustment set is defined here for a dag or a cpdag; with hidden "
            f"variables, as a {graph.kind} allows, a set best in every model the graph allows "
            "need not exist"
        )
    marks = checked(graph, "the optimal adjustment set", ("dag", "cpdag"))
    _pair(graph, x, y)
    if adjustment_set(graph, x, y) is None:
        return None
    if graph.kind == "cpdag":
        marks = graph.to_dag(tie_break="label").to_numpy()
    D = directed(marks)
    causal = _causal_nodes(marks, x, y)
    if not causal:
        return None
    parents = set(np.flatnonzero(D[:, sorted(causal)].any(axis=1)).tolist())
    return sorted(parents - reach(D, causal) - {x})


def local_parent_sets(graph: GraphStructure, x: int) -> list[list[int]]:
    """The parent sets of ``x`` in the DAGs of a ``cpdag``'s class, as local IDA finds them
    (Maathuis et al. 2009, Algorithm 3 and Lemma 3.1), ascending; one set, its parents, for a
    ``dag``.

    Each subset ``S`` of the undirected neighbors of ``x`` is directed into ``x``, the others out
    of it; the choice is kept when it makes no new v-structure with ``x`` as the collider, two
    parents of ``x`` that are not adjacent with one of them in ``S``. Exactly the kept choices
    give ``x`` its parents in some DAG of the class (Lemma 3.1), the parents of ``x`` with ``S``.
    The outcome of a pair can be among them; local IDA takes the effect there to be zero.
    """
    marks = checked(graph, "the local parent sets", ("dag", "cpdag"))
    if not 0 <= x < graph.n_nodes:
        raise ValueError(f"the treatment must be a node of the graph, got position {x}")
    parents = np.flatnonzero(directed(marks)[:, x]).tolist()
    siblings = np.flatnonzero(undirected(marks)[x]).tolist()
    found = []
    for k in range(len(siblings) + 1):
        for chosen in combinations(siblings, k):
            new = [(a, b) for a in chosen for b in parents + list(chosen) if a != b]
            if all(marks[a, b] != NULL for a, b in new):
                found.append(sorted(parents + list(chosen)))
    return sorted(found)


def _pair(graph: GraphStructure, x: int, y: int) -> None:
    n = graph.n_nodes
    if not (0 <= x < n and 0 <= y < n) or x == y:
        raise ValueError(
            f"the treatment and the outcome must be two nodes of the graph, got {x}, {y}"
        )


def amenable(graph: GraphStructure, x: int, y: int) -> bool:
    """Whether the graph is adjustment amenable for the pair (Perković et al. 2018, Definition 2):
    every proper possibly directed path from ``x`` to ``y`` starts with a visible edge out of
    ``x``. Always so in a ``dag``."""
    marks = checked(graph, "adjustment amenability")
    _pair(graph, x, y)
    return all(visible(marks, graph.kind, x, c) for c in _causal_children(marks, x, y))


def proper_back_door(marks: np.ndarray, x: int, y: int) -> np.ndarray:
    """The marks without the first edge of each proper possibly directed path from ``x`` to
    ``y``: the proper back-door graph."""
    cut = marks.copy()
    for c in _causal_children(marks, x, y):
        cut[x, c] = cut[c, x] = NULL
    return cut


def is_adjustment_set(graph: GraphStructure, x: int, y: int, z: Collection[int]) -> bool:
    """Whether ``z`` satisfies the generalized adjustment criterion for ``(x, y)``: the graph is
    amenable, ``z`` holds no forbidden node, and ``z`` blocks every proper non-causal path of
    definite status from ``x`` to ``y`` (Perković et al. 2018, Definition 4 and Theorem 5)."""
    marks = checked(graph, "the adjustment criterion")
    _pair(graph, x, y)
    given = set(z)
    if given & {x, y}:
        raise ValueError("an adjustment set holds neither the treatment nor the outcome")
    if not all(visible(marks, graph.kind, x, c) for c in _causal_children(marks, x, y)):
        return False
    if given & possible_descendants(marks, _causal_nodes(marks, x, y)):
        return False
    return _blocks(graph, marks, x, y, given)


def _blocks(graph: GraphStructure, marks: np.ndarray, x: int, y: int, given: set[int]) -> bool:
    """Whether ``given`` m-separates ``x`` and ``y`` in the proper back-door graph (Perković et
    al. 2018, Definition 6 and Theorem 7), for a graph that is amenable for the pair and a set
    that holds no forbidden node.

    In a ``cpdag`` or ``pag`` this is read in one member of the class, a DAG (``to_dag``) or the
    MAG of Zhang's construction (Perković et al. 2018, Lemma 43): under those two conditions the
    set blocks every proper non-causal path of definite status in the class's graph exactly when
    it does so in any one member (Lemma 10), and there when it separates the pair in the
    member's proper back-door graph (Theorem 7).
    """
    from ._convert import mag_of_pag
    from ._queries import separated

    if graph.kind == "cpdag":
        marks, kind = graph.to_dag(tie_break="label").to_numpy(), "dag"
    elif graph.kind == "pag":
        marks, kind = mag_of_pag(marks), "mag"
    else:
        kind = graph.kind
    cut = GraphStructure.from_numpy(proper_back_door(marks, x, y), kind=kind)
    return separated(cut, [x], [y], sorted(given))


def adjustment_set(
    graph: GraphStructure, x: int, y: int, *, among: Collection[int] | None = None
) -> list[int] | None:
    """An adjustment set for ``(x, y)`` when one exists, else ``None``: the possible ancestors of
    ``x`` and ``y`` that are neither forbidden nor ``x`` and ``y``, which is an adjustment set
    exactly when some set is (Perković et al. 2018, Definition 12 and Theorem 14).

    ``among`` (a ``dag`` only) keeps the set to those nodes, such as the observed ones of a DAG
    with hidden variables; the set is then an adjustment set exactly when some set of those
    nodes is (van der Zander et al. 2019, Theorem 2).
    """
    marks = checked(graph, "an adjustment set")
    _pair(graph, x, y)
    if among is not None and graph.kind != "dag":
        raise ValueError("among= restricts the set in a dag only")
    causal = _causal_nodes(marks, x, y)
    keep = possible_ancestors(marks, [x, y]) - possible_descendants(marks, causal) - {x, y}
    if among is not None:
        keep &= set(among)
    found = sorted(keep)
    return found if is_adjustment_set(graph, x, y, found) else None
