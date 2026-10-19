"""Background knowledge: edges a graph must have or must not have, the pairs it may connect, and
tiers that order the variables.

:class:`Knowledge` is the public value a search takes as ``knowledge=``. :func:`resolve` turns it
into :class:`Constraints`, two positional ``(d, d)`` boolean matrices the engines read, against the
data's column labels. :func:`orient` and :func:`admits` apply the constraints to an equivalence
class.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

from .structure import ARROW, NULL, TAIL

if TYPE_CHECKING:
    import numpy.typing as npt

    from .output import StructureOutput
    from .structure import GraphStructure

__all__ = ["Knowledge", "Constraints", "KnowledgePart", "KNOWLEDGE_PARTS", "resolve"]

#: A node, by column label or 0-based position.
Node = str | int
#: The parts of a :class:`Knowledge`, as a method's spec lists the ones it honors.
KnowledgePart = Literal["required", "forbidden", "allowed", "tiers"]
KNOWLEDGE_PARTS: tuple[KnowledgePart, ...] = ("required", "forbidden", "allowed", "tiers")


def _node(value: object, what: str) -> Node:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (str, int, np.integer)):
        raise TypeError(
            f"{what} names a node by column label or 0-based position, got {value!r} "
            f"({type(value).__name__})"
        )
    return value if isinstance(value, str) else int(value)


def _edges(values: Iterable | None, what: str) -> tuple[tuple[Node, Node], ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{what} must be a list of (a, b) pairs, got {values!r}")
    edges = []
    for pair in values:
        if isinstance(pair, (str, bytes)) or len(pair) != 2:
            raise TypeError(f"{what} must be a list of (a, b) pairs, got {pair!r}")
        a, b = _node(pair[0], what), _node(pair[1], what)
        if a == b:
            raise ValueError(f"{what} lists the self-loop ({a!r}, {b!r})")
        edges.append((a, b))
    return tuple(edges)


@dataclass(frozen=True, init=False)
class Knowledge:
    """Background knowledge a search honors: required and forbidden edges, the pairs that may be
    adjacent, and tiers that order the variables.

    A node is named by a column label (a DataFrame's column name, or the data file's header on the
    command line) or by its 0-based position. Labels stay attached to their columns, so the same
    knowledge fits a reordered table; positions follow the column order. Each method's ``knowledge``
    parameter states the parts it honors and raises ``NotImplementedError`` for another.

    Parameters
    ----------
    required : iterable of (cause, effect) pairs, optional
        Directed edges ``cause -> effect`` the result must have.
    forbidden : iterable of (cause, effect) pairs, optional
        Directed edges ``cause -> effect`` the result must not have. Forbid both directions to keep
        two variables non-adjacent.
    allowed : iterable of (a, b) pairs or None, optional
        The only pairs that may be adjacent, in either direction; ``None`` allows every pair. A
        small set restricts the search to it, which also makes the search faster. For the
        constraint-based methods it is the skeleton the search starts from.
    tiers : sequence of sequences of nodes, optional
        Variables grouped in time order: no edge points from a variable of a later tier into one of
        an earlier tier. A variable in no tier is unconstrained; a variable may be in one tier only.

    Raises
    ------
    TypeError
        If a part is not a list of pairs (``tiers``: of lists), or names a node by something other
        than a label or a position.
    ValueError
        If an edge is a self-loop or a node is in two tiers. A method raises ``ValueError`` when the
        knowledge names a column the data lacks, or contradicts itself: a required edge that is
        forbidden, directly, by the tiers, or by ``allowed``, or required edges forming a cycle.

    Notes
    -----
    The constraint-based methods start from the allowed pairs in place of the complete graph and
    never test a required pair, so neither is removed; a pair the knowledge keeps apart is tested
    for a separating set, so colliders over it are oriented as usual (when no set separates it, no
    rule orients through its triples), and knowledge that directs the edges the data keep into a
    cycle raises ``ValueError``. A method whose result is an equivalence
    class (a CPDAG) directs every edge the knowledge decides and then applies Meek's rules [1]_,
    including the fourth, which background knowledge needs; the result is the class of graphs that
    agree with both the data and the knowledge, still tagged ``cpdag``. In a PAG, a forbidden
    ``a -> b`` on an adjacent pair puts an arrowhead at ``a``, and a required ``a -> b`` is
    ``a -> b``. The methods that search over causal orders (``boss``, ``grasp``,
    ``direct_lingam``) also keep each variable of a tier after those of earlier tiers; with every
    variable in a tier, this is the same as forbidding the edges.

    References
    ----------
    .. [1] Meek, C. (1995). Causal inference and causal explanation with background knowledge.
       UAI, 403-410.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from andrey import Knowledge, pc
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal(500)
    >>> y = x + 0.5 * rng.standard_normal(500)
    >>> data = pd.DataFrame({"x": x, "y": y})
    >>> pc(data, knowledge=Knowledge(tiers=[["x"], ["y"]])).structure.oriented_edges()
    [('x', 'y', 'directed')]
    """

    required: tuple[tuple[Node, Node], ...] = field(default=())
    forbidden: tuple[tuple[Node, Node], ...] = field(default=())
    allowed: tuple[tuple[Node, Node], ...] | None = field(default=None)
    tiers: tuple[tuple[Node, ...], ...] = field(default=())

    def __init__(
        self,
        *,
        required: Iterable[Sequence[Node]] | None = None,
        forbidden: Iterable[Sequence[Node]] | None = None,
        allowed: Iterable[Sequence[Node]] | None = None,
        tiers: Iterable[Iterable[Node]] | None = None,
    ) -> None:
        object.__setattr__(self, "required", _edges(required, "required"))
        object.__setattr__(self, "forbidden", _edges(forbidden, "forbidden"))
        object.__setattr__(self, "allowed", None if allowed is None else _edges(allowed, "allowed"))
        if tiers is None:
            tiers = ()
        if isinstance(tiers, (str, bytes)):
            raise TypeError(f"tiers must be a list of lists of nodes, got {tiers!r}")
        grouped: list[tuple[Node, ...]] = []
        seen: set[Node] = set()
        for tier in tiers:
            if isinstance(tier, (str, bytes, int, np.integer)):
                raise TypeError(f"tiers must be a list of lists of nodes, got the tier {tier!r}")
            nodes = tuple(_node(v, "tiers") for v in tier)
            for v in nodes:
                if v in seen:
                    raise ValueError(f"tiers list {v!r} more than once")
                seen.add(v)
            grouped.append(nodes)
        object.__setattr__(self, "tiers", tuple(grouped))

    @classmethod
    def from_skeleton(
        cls,
        skeleton: StructureOutput | GraphStructure | npt.ArrayLike,
        *,
        required: Iterable[Sequence[Node]] | None = None,
        forbidden: Iterable[Sequence[Node]] | None = None,
        tiers: Iterable[Iterable[Node]] | None = None,
    ) -> Knowledge:
        """Knowledge whose allowed pairs are the adjacent pairs of ``skeleton``.

        Parameters
        ----------
        skeleton : StructureOutput, GraphStructure, or array-like of shape (d, d)
            A result, a graph, or an adjacency matrix whose nonzero entries mark adjacent pairs. A
            graph with labels names its pairs by label, otherwise by position.
        required, forbidden, tiers
            The other parts, as for :class:`Knowledge`.

        Returns
        -------
        Knowledge
            The knowledge, with ``allowed`` set to every adjacent pair of ``skeleton``.
        """
        graph = getattr(skeleton, "structure", skeleton)
        labels = getattr(graph, "labels", None)
        if hasattr(graph, "adjacency"):
            matrix = np.asarray(graph.adjacency) != 0
        else:
            matrix = np.asarray(skeleton) != 0
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError(f"skeleton must be a square matrix, got shape {matrix.shape}")
        xs, ys = np.nonzero(np.triu(matrix | matrix.T, 1))
        name = (lambda i: labels[i]) if labels is not None else int
        allowed = [(name(x), name(y)) for x, y in zip(xs.tolist(), ys.tolist())]
        return cls(required=required, forbidden=forbidden, allowed=allowed, tiers=tiers)

    def parts(self) -> tuple[KnowledgePart, ...]:
        """The parts this knowledge sets, in :data:`KNOWLEDGE_PARTS` order."""
        given = {
            "required": bool(self.required),
            "forbidden": bool(self.forbidden),
            "allowed": self.allowed is not None,
            "tiers": bool(self.tiers),
        }
        return tuple(part for part in KNOWLEDGE_PARTS if given[part])


@dataclass(frozen=True)
class Constraints:
    """Knowledge resolved against ``d`` columns: what the search engines read.

    ``forbidden[i, j]`` forbids the edge ``i -> j``: it holds the forbidden edges, the edges from a
    later tier into an earlier one, both directions of every pair outside the allowed set, and the
    reverse of every required edge. ``required[i, j]`` requires ``i -> j``. ``tier[i]`` is column
    ``i``'s tier, ``-1`` for none.
    """

    forbidden: np.ndarray
    required: np.ndarray
    tier: np.ndarray

    @property
    def d(self) -> int:
        """The number of columns."""
        return self.forbidden.shape[0]

    def support(self) -> np.ndarray:
        """The pairs that may be adjacent: not forbidden in both directions (diagonal ``False``)."""
        support = ~(self.forbidden & self.forbidden.T)
        np.fill_diagonal(support, False)
        return support

    def decided(self) -> np.ndarray:
        """``[i, j]``: the knowledge directs an adjacent ``i - j`` as ``i -> j``."""
        return self.forbidden.T & ~self.forbidden

    def precedence(self) -> np.ndarray:
        """``[i, j]``: ``i`` comes before ``j`` in every causal order an order search visits.

        Holds the required edges and every pair of tiered columns in tier order. Raises
        ``ValueError`` when they form a cycle, so that no order satisfies them.
        """
        tiered = self.tier >= 0
        before = self.required | (
            tiered[:, None] & tiered[None, :] & (self.tier[:, None] < self.tier[None, :])
        )
        if _has_cycle(before):
            raise ValueError(
                "the required edges and the tiers admit no causal order: following them leads "
                "from a column back to itself through an untiered column"
            )
        return before

    def padded(self, d: int) -> Constraints:
        """These constraints over ``d >= self.d`` columns, the added ones unconstrained."""
        forbidden = np.zeros((d, d), dtype=bool)
        required = np.zeros((d, d), dtype=bool)
        forbidden[: self.d, : self.d] = self.forbidden
        required[: self.d, : self.d] = self.required
        tier = np.full(d, -1, dtype=np.intp)
        tier[: self.d] = self.tier
        return Constraints(forbidden, required, tier)

    def permuted(self, order: npt.ArrayLike) -> Constraints:
        """These constraints with column ``order[k]`` as column ``k``."""
        idx = np.asarray(order, dtype=np.intp)
        return Constraints(
            self.forbidden[np.ix_(idx, idx)], self.required[np.ix_(idx, idx)], self.tier[idx]
        )


def _has_cycle(edges: np.ndarray) -> bool:
    """Whether the directed graph ``edges[i, j]`` (``i -> j``) has a cycle (Kahn's algorithm)."""
    indegree = edges.sum(axis=0)
    ready = list(np.flatnonzero(indegree == 0))
    seen = 0
    while ready:
        i = ready.pop()
        seen += 1
        for j in np.flatnonzero(edges[i]).tolist():
            indegree[j] -= 1
            if indegree[j] == 0:
                ready.append(j)
    return seen < edges.shape[0]


def resolve(
    knowledge: Knowledge | None,
    d: int,
    labels: Sequence[str] | None,
    *,
    method: str,
    honors: Sequence[KnowledgePart] = KNOWLEDGE_PARTS,
) -> Constraints | None:
    """Resolve ``knowledge`` against ``d`` columns named ``labels``; ``None`` when it sets nothing.

    ``method`` names the caller in messages; ``honors`` lists the parts it supports, and a part it
    does not list raises ``NotImplementedError``. Raises ``TypeError`` when ``knowledge`` is not a
    :class:`Knowledge`, and ``ValueError`` when it names a missing column or contradicts itself.
    """
    if knowledge is None:
        return None
    if not isinstance(knowledge, Knowledge):
        raise TypeError(f"knowledge must be an andrey.Knowledge, got {type(knowledge).__name__}")
    parts = knowledge.parts()
    if not parts:
        return None
    unsupported = [part for part in parts if part not in honors]
    if unsupported:
        raise NotImplementedError(
            f"{method} does not honor the knowledge part(s) {unsupported}; it honors {list(honors)}"
        )
    names = list(labels) if labels is not None else None

    def position(node: Node, what: str) -> int:
        if isinstance(node, str):
            if names is None:
                raise ValueError(
                    f"knowledge {what} names column {node!r}, but data has no column names; give a "
                    "DataFrame or the column's 0-based position"
                )
            if node not in names:
                raise ValueError(
                    f"knowledge {what} names column {node!r}, which data does not have; its "
                    f"columns are {names}"
                )
            return names.index(node)
        if not 0 <= node < d:
            raise ValueError(f"knowledge {what} lists column {node}, but data has {d} columns")
        return node

    def label(i: int) -> str:
        return repr(names[i]) if names is not None else str(i)

    forbidden = np.zeros((d, d), dtype=bool)
    required = np.zeros((d, d), dtype=bool)
    tier = np.full(d, -1, dtype=np.intp)
    for a, b in knowledge.forbidden:
        forbidden[position(a, "forbidden"), position(b, "forbidden")] = True
    for k, nodes in enumerate(knowledge.tiers):
        for v in nodes:
            i = position(v, "tiers")
            if tier[i] >= 0:
                raise ValueError(f"knowledge tiers list column {label(i)} more than once")
            tier[i] = k
    tiered = tier >= 0
    forbidden |= tiered[:, None] & tiered[None, :] & (tier[:, None] > tier[None, :])
    if knowledge.allowed is not None:
        allowed = np.zeros((d, d), dtype=bool)
        for a, b in knowledge.allowed:
            i, j = position(a, "allowed"), position(b, "allowed")
            allowed[i, j] = allowed[j, i] = True
        forbidden |= ~allowed
    np.fill_diagonal(forbidden, False)
    for a, b in knowledge.required:
        i, j = position(a, "required"), position(b, "required")
        if forbidden[i, j]:
            raise ValueError(
                f"knowledge requires the edge {label(i)} -> {label(j)}, which it also forbids "
                "(directly, by the tiers, or by leaving the pair out of allowed)"
            )
        required[i, j] = True
    if (required & required.T).any() or _has_cycle(required):
        raise ValueError("knowledge requires edges that form a directed cycle")
    forbidden |= required.T
    return Constraints(forbidden, required, tier)


# ---- applying constraints to an equivalence class -----------------------------------------------


def orient(pdag: np.ndarray, constraints: Constraints) -> np.ndarray:
    """Direct every undirected edge of ``pdag`` the constraints decide; returns a new matrix.

    ``pdag`` is a mark matrix in the convention of :mod:`andrey.core.orient` (``i -> j`` is
    ``pdag[i, j] == TAIL`` with ``pdag[j, i] == ARROW``). Directed edges are kept as they are.
    """
    out = pdag.copy()
    lines = (out == TAIL) & (out.T == TAIL)
    direct = lines & constraints.decided()
    out[direct.T] = ARROW  # i -> j: the mark at j on edge j-i is an arrowhead
    return out


def violates(pdag: np.ndarray, constraints: Constraints) -> bool:
    """Whether ``pdag`` directs an edge the constraints forbid, or joins a pair they keep apart."""
    directed = (pdag == TAIL) & (pdag.T == ARROW)
    adjacent = pdag != NULL
    kept_apart = adjacent & ~constraints.support()
    return bool((directed & constraints.forbidden).any() or kept_apart.any())


def mpdag(cpdag: np.ndarray, constraints: Constraints) -> np.ndarray:
    """The class of ``cpdag`` narrowed by the constraints: knowledge orientations, then Meek's rules
    with R4."""
    from .orient import meek

    return meek(orient(cpdag, constraints), rule4=True)


def admits(cpdag: np.ndarray, constraints: Constraints) -> bool:
    """Whether some DAG in the class ``cpdag`` satisfies the constraints.

    The class must direct no edge the constraints forbid and join no pair they keep apart. Every
    remaining undirected edge with a forbidden direction is one the constraints decide, so after
    directing those, a consistent extension (Dor and Tarsi) that keeps the class's skeleton and
    colliders satisfies them, and one must exist.
    """
    from .orient import _cpdag_of_pdag

    if violates(cpdag, constraints):
        return False
    oriented = orient(cpdag, constraints)
    if np.array_equal(oriented, cpdag):
        return True
    try:
        return bool(np.array_equal(_cpdag_of_pdag(oriented), cpdag))
    except ValueError:
        return False
