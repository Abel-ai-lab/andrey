"""First-class graph structures: the canonical CSR store and its derived views.

``Structure`` is the abstract base every algorithm's ``.structure`` is an instance of; the
concrete subclasses are ``GraphStructure`` (a single kind-tagged graph backed by one symmetric
endpoint-mark CSR) and ``TemporalStructure`` (a per-lag stack of graphs). One compact store per
graph; every dense / SciPy / NetworkX / edge-list view derives from it lazily and
uncached, so nothing is held twice.

Endpoint marks are unsigned: ``NULL`` (0, never stored) / ``TAIL`` (1) / ``ARROW`` (2) /
``CIRCLE`` (3). ``M[i][j]`` is the mark at node ``i`` on edge ``i-j``, so every edge contributes
two entries (``M[i][j]`` and ``M[j][i]``). A double arrowhead ``ARROW``/``ARROW`` on an
off-diagonal pair is kind-relative: a bidirected (latent-confounded) edge under ``pag`` but a
2-cycle (``i -> j`` and ``j -> i``, two directed edges) under ``digraph``, the directed graph kind
that also carries autoregressive self-loops.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal

import numpy as np

if TYPE_CHECKING:  # optional / derived-view deps, imported lazily in the methods
    import networkx
    import numpy.typing as npt
    import scipy.sparse

# Endpoint mark codes (unsigned; ``NULL`` is never stored in the CSR).
NULL = 0
TAIL = 1
ARROW = 2
CIRCLE = 3

# Node-type codes for ``GraphStructure._node_types``; ``None`` means every node is observed.
OBSERVED = 0
LATENT = 1

Kind = Literal["dag", "cpdag", "pag", "digraph"]

# Structured dtype for the derived edge-list view (one row per edge, ``i <= j``).
EDGE_DTYPE = np.dtype([("i", "i4"), ("j", "i4"), ("mark_i", "i1"), ("mark_j", "i1")])

# ``oriented_edges`` type by mark pair on edge ``i-j``, and whether ``(i, j)`` is already
# ``(source, target)``. The source is the tail or circle end; for ``-o`` it is the tail end.
_EDGE_TYPES: dict[tuple[int, int], tuple[str, bool]] = {
    (TAIL, ARROW): ("directed", True),
    (ARROW, TAIL): ("directed", False),
    (CIRCLE, ARROW): ("partially_directed", True),
    (ARROW, CIRCLE): ("partially_directed", False),
    (TAIL, CIRCLE): ("partially_undirected", True),
    (CIRCLE, TAIL): ("partially_undirected", False),
    (TAIL, TAIL): ("undirected", True),
    (ARROW, ARROW): ("bidirected", True),
    (CIRCLE, CIRCLE): ("circle", True),
}
# Printed symbol per edge type, read source end, line, target end.
EDGE_GLYPHS = {
    "directed": "->",
    "partially_directed": "o->",
    "partially_undirected": "-o",
    "undirected": "--",
    "bidirected": "<->",
    "circle": "o-o",
}
# Index into ``_TYPE_NAMES`` by mark-pair code ``mark_i * 4 + mark_j``; -1 for no type.
_TYPE_NAMES = tuple(EDGE_GLYPHS)
_TYPE_CODE = np.full(16, -1, dtype=np.int64)
_TYPE_CODE[[a * 4 + b for a, b in _EDGE_TYPES]] = [
    _TYPE_NAMES.index(t) for t, _ in _EDGE_TYPES.values()
]
SUMMARY_EDGES = 30  # edges a printed summary lists before "... and N more"


@dataclass(frozen=True, eq=False, slots=True, kw_only=True, repr=False)
class Structure:
    """Base class of the learned graphs, ``GraphStructure`` and ``TemporalStructure``.

    A method's result holds one in ``StructureOutput.structure``; ``Structure`` itself is
    abstract. The class attribute ``type`` is ``"graph"`` for a ``GraphStructure`` (including a
    ``SummaryGraph``) and ``"temporal"`` for a ``TemporalStructure``. Both have ``n_nodes`` and
    ``labels``.
    """

    type: ClassVar[Literal["graph", "temporal"]]  # set by each concrete subclass

    _n_nodes: int
    _labels: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if type(self) is Structure:
            raise TypeError("Structure is abstract; use GraphStructure or TemporalStructure")
        if self._n_nodes < 0:
            raise ValueError(f"n_nodes must be non-negative, got {self._n_nodes}")
        if self._labels is not None and len(self._labels) != self._n_nodes:
            raise ValueError(f"labels length {len(self._labels)} != n_nodes {self._n_nodes}")

    @property
    def n_nodes(self) -> int:
        """Number of nodes, including nodes without edges and any latent nodes the method adds."""
        return self._n_nodes

    @property
    def labels(self) -> tuple[str, ...] | None:
        """Node names (a DataFrame's column names), or ``None`` when nodes are numbered from 0."""
        return self._labels

    def __eq__(self, other: object) -> bool:
        """Shared value-equality; subclasses extend it with their own store fields.

        Returns ``NotImplemented`` for a different concrete type (so ``==`` falls back to
        identity), ``True`` when the shared fields match, else ``False``.
        """
        if type(self) is not type(other):
            return NotImplemented
        return self._n_nodes == other._n_nodes and self._labels == other._labels  # type: ignore[attr-defined]

    __hash__ = None  # type: ignore[assignment]  # frozen + custom __eq__ + unhashable ndarray fields

    def __repr__(self) -> str:
        """Return the structure's kind, size, and edges as a readable summary."""
        return "\n".join(self._summary())

    def _summary(
        self,
        *,
        name: str | None = None,
        weights: np.ndarray | scipy.sparse.csr_array | None = None,
        limit: int | None = SUMMARY_EDGES,
        more: str = "",
    ) -> list[str]:
        """Require concrete structures to provide summary lines.

        ``name`` leads the header. ``weights[source, target]`` follows a directed edge. ``limit``
        caps the listed edges (``None`` lists all), and ``more`` extends the omitted-edge count.
        """
        raise NotImplementedError(f"{type(self).__name__} does not define a summary")


@dataclass(frozen=True, eq=False, slots=True, kw_only=True, repr=False)
class GraphStructure(Structure):
    """A learned graph: its nodes, and a mark at each end of every edge.

    A method returns one as ``StructureOutput.structure``; ``from_numpy``, ``from_edges``, and
    ``from_networkx`` build one from an existing graph. ``kind`` says how to read the edges:

    - ``"dag"``: every edge is directed, with no directed cycles.
    - ``"cpdag"``: an equivalence class of DAGs; an undirected edge ``--`` can point either way.
    - ``"pag"``: an equivalence class that allows hidden variables; a circle ``o`` is an
      unresolved endpoint, and ``<->`` is a relation through hidden causes.
    - ``"digraph"``: directed edges that may form cycles and self-loops, such as the graph for a
      positive lag of a ``TemporalStructure``. Here the pair ``(2, 2)`` is two directed edges,
      ``i -> j`` and ``j -> i``.

    A cell ``M[i, j]`` of ``to_numpy()`` is the mark at node ``i`` on its edge with ``j``: ``0``
    absent, ``1`` tail, ``2`` arrowhead, ``3`` circle (``andrey.core.NULL``, ``TAIL``, ``ARROW``,
    ``CIRCLE``). ``endpoints(i, j)`` reads one edge, and ``oriented_edges()`` lists every edge as
    ``(source, target, type)``. ``oriented_edges()``, ``to_edges()``, and ``to_scipy_sparse()``
    scale to large graphs; ``to_numpy()`` and ``adjacency`` build a dense ``(n, n)`` array on each
    call. The graph is read-only.
    """

    type: ClassVar[Literal["graph"]] = "graph"

    _indptr: np.ndarray  # int64, (n_nodes + 1,)
    _indices: np.ndarray  # int32, (nnz,)
    _marks: np.ndarray  # int8, (nnz,)
    _kind: Kind = "dag"
    # Internal store layout: ``sym`` = the canonical symmetric endpoint-mark CSR; ``compact`` = a
    # halved dag/cpdag store (one entry per edge), reconstructed to the symmetric view on read.
    # A pure function of (kind, marks) set by ``_from_csr``; the public views are layout-agnostic.
    _layout: Literal["sym", "compact"] = "sym"
    _node_types: np.ndarray | None = None  # int8 (n_nodes,); OBSERVED/LATENT, None = all-observed

    # ---- construction ---------------------------------------------------------------------

    @classmethod
    def _from_csr(
        cls,
        n_nodes: int,
        indptr: np.ndarray,
        indices: np.ndarray,
        marks: np.ndarray,
        *,
        kind: Kind,
        labels: tuple[str, ...] | None = None,
        node_types: np.ndarray | None = None,
    ) -> GraphStructure:
        """Build directly from CSR arrays, casting to the canonical dtypes. Internal entry point."""
        n = int(n_nodes)
        types = None
        if node_types is not None:
            types = _frozen(node_types, np.int8)
            if types.shape != (n,):
                raise ValueError(f"node_types must have shape ({n},), got {types.shape}")
        # Opportunistically halve the store for a conforming dag/cpdag; a nonconforming graph stays
        # symmetric so ``validate()`` still reports it.
        layout: Literal["sym", "compact"] = "sym"
        compact = _try_compact(n, indptr, indices, marks, kind)
        if compact is not None:
            indptr, indices, marks = compact
            layout = "compact"
        return cls(
            _n_nodes=n,
            _labels=labels,
            _indptr=_frozen(indptr, np.int64),
            _indices=_frozen(indices, np.int32),
            _marks=_frozen(marks, np.int8),
            _kind=kind,
            _layout=layout,
            _node_types=types,
        )

    @classmethod
    def from_numpy(
        cls,
        M: np.ndarray,
        *,
        kind: Kind | None = None,
        labels: tuple[str, ...] | None = None,
        allow_self_loops: bool = False,
        node_types: np.ndarray | None = None,
    ) -> GraphStructure:
        """Build a graph from an ``(n, n)`` endpoint-mark matrix, the inverse of ``to_numpy``.

        Parameters
        ----------
        M : np.ndarray of shape (n, n)
            Endpoint marks: ``M[i, j]`` is the mark at node ``i`` on its edge with ``j``, ``0``
            absent, ``1`` tail, ``2`` arrowhead, ``3`` circle. ``i -> j`` is ``M[i, j] = 1`` and
            ``M[j, i] = 2``. Every edge needs a mark in both cells.
        kind : {"dag", "cpdag", "pag", "digraph"} or None, default=None
            Graph type. ``None`` infers ``"pag"`` from any circle or ``(2, 2)`` pair, else
            ``"cpdag"`` from any ``(1, 1)`` pair, else ``"dag"``; it never infers ``"digraph"``,
            so pass it for a graph with 2-cycles. The marks are not checked against ``kind``;
            call ``validate()``.
        labels : tuple[str, ...] or None, default=None
            Node names, one per node; ``None`` numbers the nodes from 0.
        allow_self_loops : bool, default=False
            Accept nonzero diagonal cells. A self-loop ``M[i, i] = 2`` is a variable's effect on
            its own future, in the graph for a positive lag of a ``TemporalStructure``.
        node_types : np.ndarray of shape (n,) or None, default=None
            ``0`` for an observed node and ``1`` for a latent one (``andrey.core.OBSERVED``,
            ``LATENT``); ``None`` when every node is observed.

        Returns
        -------
        GraphStructure
            The graph.

        Raises
        ------
        ValueError
            If ``M`` is not square, has a nonzero diagonal without ``allow_self_loops``, holds a
            value outside ``0..3``, has a mark in ``M[i, j]`` but not in ``M[j, i]``, ``labels``
            does not have ``n`` entries, or ``node_types`` is not shape ``(n,)``.
        """
        M = np.asarray(M)
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise ValueError(f"expected a square (n, n) matrix, got shape {M.shape}")
        n = M.shape[0]
        if not allow_self_loops and M.size and np.diagonal(M).any():
            raise ValueError("endpoint matrix must have a zero diagonal (no self-loops)")
        if (M < 0).any() or (M > CIRCLE).any():
            raise ValueError(f"endpoint marks must be in 0..{CIRCLE}, got other values")
        if not np.array_equal(M != 0, M.T != 0):
            raise ValueError("endpoint matrix support must be symmetric (every i-j has a j-i mark)")
        marks_i8 = M.astype(np.int8)
        resolved = kind if kind is not None else _infer_kind(marks_i8)
        indptr, indices, marks = _dense_to_csr(marks_i8)
        return cls._from_csr(
            n, indptr, indices, marks, kind=resolved, labels=labels, node_types=node_types
        )

    @classmethod
    def from_edges(
        cls,
        edges: np.ndarray,
        *,
        n_nodes: int,
        kind: Kind | None = None,
        labels: tuple[str, ...] | None = None,
    ) -> GraphStructure:
        """Build a graph from an edge list, the inverse of ``to_edges``.

        Parameters
        ----------
        edges : np.ndarray
            Structured array with fields ``i``, ``j``, ``mark_i``, ``mark_j`` (dtype
            ``andrey.core.EDGE_DTYPE``), one row per edge; ``mark_i`` is the mark at ``i`` and
            ``mark_j`` the mark at ``j``. A row with ``i == j`` is a self-loop.
        n_nodes : int
            Number of nodes, including nodes without edges.
        kind : {"dag", "cpdag", "pag", "digraph"} or None, default=None
            Graph type; ``None`` infers it as ``from_numpy`` does.
        labels : tuple[str, ...] or None, default=None
            Node names, one per node; ``None`` numbers the nodes from 0.

        Returns
        -------
        GraphStructure
            The graph.

        Raises
        ------
        ValueError
            If a mark is outside ``0..3``, a row has a mark at one end only, or ``labels`` does not
            have ``n_nodes`` entries.
        IndexError
            If ``edges`` is not a structured array with those fields, or names a node outside
            ``range(n_nodes)``.
        """
        edges = np.asarray(edges)
        M = np.zeros((n_nodes, n_nodes), dtype=np.int8)
        for row in edges:
            i, j, mi, mj = int(row["i"]), int(row["j"]), int(row["mark_i"]), int(row["mark_j"])
            if not (0 <= i < n_nodes and 0 <= j < n_nodes):
                raise IndexError(f"edge ({i}, {j}) names a node outside range({n_nodes})")
            M[i, j] = mi
            M[j, i] = mj
        # keep self-loops if the input carries any -- so a self-loop graph (a lag>=1 view) is a
        # faithful round-trip through these documented inverses of to_edges / to_networkx.
        self_loops = bool(M.size and np.diagonal(M).any())
        return cls.from_numpy(M, kind=kind, labels=labels, allow_self_loops=self_loops)

    @classmethod
    def from_networkx(cls, g: networkx.DiGraph, *, kind: Kind | None = None) -> GraphStructure:
        """Build a graph from a NetworkX graph, the inverse of ``to_networkx``.

        Each edge ``(u, v)`` is read from its ``endpoints=(mark at u, mark at v)`` attribute. An
        edge without the attribute is ``u -> v``. Two such edges ``u -> v`` and ``v -> u`` give the
        mark pair ``(2, 2)``, so pass ``kind="digraph"`` to read them as a 2-cycle rather than a
        ``pag`` bidirected edge. Self-loops are kept. Nodes are indexed in insertion order; unless
        they are exactly ``0, 1, ..., n - 1`` in that order, each node becomes a label, converted
        to ``str``.

        Parameters
        ----------
        g : networkx.DiGraph or networkx.MultiDiGraph
            The graph to read.
        kind : {"dag", "cpdag", "pag", "digraph"} or None, default=None
            Graph type; ``None`` infers it as ``from_numpy`` does.

        Returns
        -------
        GraphStructure
            The graph.

        Raises
        ------
        ValueError
            If an ``endpoints`` mark is outside ``0..3``, or an edge has a mark at one end only.
        """
        nodes = list(g.nodes())
        n = len(nodes)
        index = {node: k for k, node in enumerate(nodes)}
        integer_indexed = all(isinstance(node, (int, np.integer)) for node in nodes) and [
            int(node) for node in nodes
        ] == list(range(n))
        labels = None if integer_indexed else tuple(str(node) for node in nodes)
        M = np.zeros((n, n), dtype=np.int8)
        for u, v, data in g.edges(data=True):
            ep = data.get("endpoints")
            if ep is None:
                # A bare edge u -> v is an arrowhead at v; when the reciprocal bare edge (v, u) is
                # also present the pair is a 2-cycle (ARROW/ARROW), not the second write clobbering
                # the first into a lone directed edge.
                M[index[v], index[u]] = ARROW
                M[index[u], index[v]] = ARROW if g.has_edge(v, u) else TAIL
            else:
                M[index[u], index[v]] = int(ep[0])
                M[index[v], index[u]] = int(ep[1])
        # keep self-loops if the input carries any -- so a self-loop graph (a lag>=1 view) is a
        # faithful round-trip through these documented inverses of to_edges / to_networkx.
        self_loops = bool(M.size and np.diagonal(M).any())
        return cls.from_numpy(M, kind=kind, labels=labels, allow_self_loops=self_loops)

    # ---- store surface --------------------------------------------------------------------

    @property
    def kind(self) -> Kind:
        """Graph type: ``"dag"``, ``"cpdag"``, ``"pag"``, or ``"digraph"`` (cycles allowed)."""
        return self._kind

    @property
    def node_types(self) -> np.ndarray | None:
        """Per-node codes, ``0`` observed and ``1`` latent, or ``None`` when every node is observed.

        ``gin`` sets it for the latent nodes it adds (``andrey.core.OBSERVED``, ``LATENT``).
        """
        return self._node_types

    @property
    def adjacency(self) -> np.ndarray:
        """The endpoint-mark matrix, the same as ``to_numpy()``; see ``to_numpy`` for the codes.

        Each access builds a new dense ``(n, n)`` array; for a large graph use ``oriented_edges``,
        ``to_edges``, or ``to_scipy_sparse``.
        """
        return self.to_numpy()

    def to_numpy(self, *, dtype: npt.DTypeLike = np.int8) -> np.ndarray:
        """Return the endpoint marks as a dense ``(n, n)`` array.

        A cell ``M[i, j]`` is the mark at node ``i`` on its edge with ``j``: ``0`` absent, ``1``
        tail, ``2`` arrowhead, ``3`` circle. ``i -> j`` has ``M[i, j] = 1`` and ``M[j, i] = 2``.
        Each call builds a new array.

        Parameters
        ----------
        dtype : npt.DTypeLike, default=np.int8
            Data type of the returned array.

        Returns
        -------
        np.ndarray of shape (n, n)
            The endpoint-mark matrix.
        """
        n = self._n_nodes
        M = np.zeros((n, n), dtype=dtype)
        indptr, indices, marks = self._sym_csr()
        if indices.size:
            rows = np.repeat(np.arange(n, dtype=np.intp), np.diff(indptr))
            M[rows, indices] = marks
        return M

    def to_scipy_sparse(self) -> scipy.sparse.csr_array:
        """Return the endpoint marks as a ``scipy.sparse.csr_array``, without a dense array.

        Returns
        -------
        scipy.sparse.csr_array of shape (n, n)
            The ``to_numpy`` matrix in sparse form; memory grows with the number of edges.
        """
        import scipy.sparse

        indptr, indices, marks = self._sym_csr()
        return scipy.sparse.csr_array(
            (marks, indices, indptr), shape=(self._n_nodes, self._n_nodes)
        )

    def to_networkx(self, *, multigraph: bool = False) -> networkx.DiGraph | networkx.MultiDiGraph:
        """Return the graph as a NetworkX graph whose edges carry their marks.

        Each arc ``(u, v)`` has the attribute ``endpoints=(mark at u, mark at v)``. For a ``dag``,
        ``cpdag``, or ``pag``, each edge is one arc from the lower to the higher node index
        whatever its direction, so read ``endpoints`` or use ``oriented_edges()``. A ``digraph``
        runs each arc from tail to arrowhead and gives a 2-cycle both arcs, so NetworkX cycle
        functions see its cycles. A ``MultiDiGraph`` also adds the reverse arc for ``<->`` and
        ``o-o``. Nodes are named by ``labels`` when set, else numbered from 0. ``from_networkx``
        reads the result back.

        Parameters
        ----------
        multigraph : bool, default=False
            Return a ``MultiDiGraph`` for any ``kind``; a ``pag`` always gets one.

        Returns
        -------
        networkx.DiGraph or networkx.MultiDiGraph
            A ``MultiDiGraph`` for a ``pag`` or with ``multigraph=True``, else a ``DiGraph``.
        """
        import networkx as nx

        use_multi = multigraph or self._kind == "pag"
        # A multigraph adds the reciprocal for symmetric PAG marks so the pair does not collapse; a
        # digraph likewise emits both directed edges of a 2-cycle into a plain DiGraph.
        emit_reciprocal = use_multi or self._kind == "digraph"
        g: networkx.DiGraph | networkx.MultiDiGraph = (
            nx.MultiDiGraph() if use_multi else nx.DiGraph()
        )
        node_id = self._labels if self._labels is not None else range(self._n_nodes)
        g.add_nodes_from(node_id)
        ids = list(node_id)
        for i, j, mi, mj in self._iter_edges():
            # A digraph is topology-faithful: orient a plain directed edge tail -> arrowhead so a
            # cycle of any length is visible to NetworkX, not only a direction-symmetric 2-cycle.
            if self._kind == "digraph" and i != j and {int(mi), int(mj)} == {TAIL, ARROW}:
                src, dst, ep = (i, j, (mi, mj)) if mj == ARROW else (j, i, (mj, mi))
                g.add_edge(ids[src], ids[dst], endpoints=(int(ep[0]), int(ep[1])))
                continue
            g.add_edge(ids[i], ids[j], endpoints=(int(mi), int(mj)))
            # A 2-cycle (digraph) or a symmetric non-tail pair (i<->j, i o-o j) contributes the
            # reciprocal edge; a plain undirected i--j stays a single edge.
            if emit_reciprocal and i != j and mi == mj and mi in (ARROW, CIRCLE):
                g.add_edge(ids[j], ids[i], endpoints=(int(mj), int(mi)))
        return g

    def to_edges(self) -> np.ndarray:
        """Return the derived typed edge list, one row per edge ``i <= j``.

        Returns
        -------
        np.ndarray
            Structured array (dtype ``andrey.core.EDGE_DTYPE``) with fields ``i``, ``j``,
            ``mark_i``, ``mark_j``, one row per edge with ``i <= j``; ``mark_i`` is the mark at
            ``i``. A row with ``i == j`` is a self-loop.
        """
        rows = [(i, j, mi, mj) for i, j, mi, mj in self._iter_edges()]
        return np.array(rows, dtype=EDGE_DTYPE)

    def oriented_edges(self, *, index: bool = False) -> list[tuple[str | int, str | int, str]]:
        """Return one ``(source, target, type)`` row per edge, oriented by its marks.

        A directed edge runs from its tail to its arrowhead whatever the node order, and nodes are
        named by ``labels`` when the graph has them. ``andrey run`` prints the same rows.

        Parameters
        ----------
        index : bool, default=False
            Return node indices even when the graph has labels.

        Returns
        -------
        list[tuple[str | int, str | int, str]]
            Rows follow ``to_edges`` order. A ``digraph`` 2-cycle yields two ``directed`` rows,
            lower-to-higher node index first, then the reverse. ``type`` is one of:

            - ``directed``: ``source -> target``; a self-loop has ``source == target``.
            - ``partially_directed``: ``source o-> target``.
            - ``partially_undirected``: ``source -o target``, a tail at ``source`` and a circle at
              ``target``.
            - ``undirected``: ``source -- target``.
            - ``bidirected``: ``source <-> target`` in a ``pag``.
            - ``circle``: ``source o-o target``.

            The three symmetric types list the lower node index first.

        Examples
        --------
        >>> import numpy as np
        >>> from andrey import GraphStructure
        >>> M = np.array([[0, 2], [1, 0]])  # the arrowhead is at node 0
        >>> GraphStructure.from_numpy(M, labels=("wet", "rain")).oriented_edges()
        [('rain', 'wet', 'directed')]
        """
        names = range(self._n_nodes) if index or self._labels is None else self._labels
        return [(names[s], names[t], edge_type) for s, t, edge_type in self._oriented()]

    def neighbors(self, i: int) -> np.ndarray:
        """Return the nodes that share an edge with node ``i``, whatever the marks.

        For a ``dag`` or ``cpdag`` each call scans every edge; to visit every node, use
        ``to_scipy_sparse()``.

        Parameters
        ----------
        i : int
            Node index.

        Returns
        -------
        np.ndarray of int32
            The adjacent node indices, ascending and read-only; includes ``i`` if it has a
            self-loop.
        """
        if self._layout == "sym":
            return self._indices[self._indptr[i] : self._indptr[i + 1]]
        # compact: out-edges (row i) plus in-edges (i appearing as a column in another row).
        out_cols = self._indices[self._indptr[i] : self._indptr[i + 1]]
        in_pos = np.nonzero(self._indices == i)[0]
        in_rows = (np.searchsorted(self._indptr, in_pos, side="right") - 1).astype(np.int32)
        result = np.union1d(out_cols, in_rows).astype(np.int32)
        result.setflags(write=False)
        return result

    def endpoints(self, i: int, j: int) -> tuple[int, int]:
        """Return the endpoint marks on edge ``i-j``. ``O(log deg)``.

        Parameters
        ----------
        i : int
            First node index.
        j : int
            Second node index.

        Returns
        -------
        tuple[int, int]
            ``(mark at i, mark at j)``: ``1`` tail, ``2`` arrowhead, ``3`` circle; ``(0, 0)`` when
            there is no edge.
        """
        return int(self._mark_at(i, j)), int(self._mark_at(j, i))

    # ---- validation -----------------------------------------------------------------------

    def validate(self) -> None:
        """Check that every edge's marks are allowed for ``kind``.

        It does not check a ``dag`` for directed cycles, or that a ``cpdag`` or ``pag`` is a
        completed equivalence class.

        Raises
        ------
        ValueError
            If a circle appears outside a ``pag``, a ``dag`` or ``cpdag`` has an edge with two
            arrowheads, a ``dag`` or ``digraph`` has an undirected edge, or an edge has a mark at
            one end only.
        """
        indptr, indices, marks = self._sym_csr()
        if self._kind != "pag" and (marks == CIRCLE).any():
            raise ValueError(f"CIRCLE marks require kind='pag', not '{self._kind}'")
        for i, j, mi, mj in self._iter_edges():
            if i == j:
                continue  # a self-loop (autoregressive lag edge) is single-endpoint, not bidirected
            if self._kind in ("dag", "cpdag") and mi == ARROW and mj == ARROW:
                raise ValueError(
                    f"double arrowhead {i}<->{j} requires kind='pag' (bidirected) or "
                    f"kind='digraph' (2-cycle)"
                )
            if self._kind in ("dag", "digraph") and mi == TAIL and mj == TAIL:
                raise ValueError(f"undirected edge {i}--{j} requires kind='cpdag' or 'pag'")
        # Support symmetry: every stored (i, j) has a matching (j, i).
        n = self._n_nodes
        rows = np.repeat(np.arange(n, dtype=np.intp), np.diff(indptr))
        for i, j in zip(rows.tolist(), indices.tolist()):
            if _lookup(indptr, indices, marks, j, i) == NULL:
                raise ValueError(f"asymmetric half-edge: ({i}, {j}) stored but ({j}, {i}) missing")

    # ---- internals ------------------------------------------------------------------------

    def _sym_csr(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The symmetric endpoint-mark CSR (``indptr``, ``indices``, ``marks``) every view reads.

        Identity when the store is already symmetric; reconstructs the full pairs for a ``compact``
        dag/cpdag store, so no accessor observes the layout.
        """
        if self._layout == "sym":
            return self._indptr, self._indices, self._marks
        return _expand_compact(self._n_nodes, self._indptr, self._indices, self._marks)

    def _mark_at(self, i: int, j: int) -> int:
        if self._layout == "sym":
            return _lookup(self._indptr, self._indices, self._marks, i, j)
        # compact: the edge is stored once. An entry at (i, j) means i is the source (or an
        # undirected end) -- the mark AT i is TAIL, the diagonal is a self-loop; an entry at (j, i)
        # carries the far mark (ARROW if directed into i, TAIL if undirected).
        m = _lookup(self._indptr, self._indices, self._marks, i, j)
        if m != NULL:
            return m if i == j else TAIL
        return _lookup(self._indptr, self._indices, self._marks, j, i)

    def _mark_pairs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return mark arrays for edges with ``i <= j``, in ``to_edges`` order.

        The arrays hold ``(i, j, mark_i, mark_j)``. This vectorized form of ``_iter_edges`` avoids
        a search for each edge.
        """
        import scipy.sparse

        indptr, indices, marks = self._sym_csr()
        rows = np.repeat(np.arange(self._n_nodes, dtype=np.intp), np.diff(indptr))
        cols = indices.astype(np.intp)
        keep = rows <= cols
        rows, cols, near = rows[keep], cols[keep], marks[keep].astype(np.int64)
        if not rows.size:
            return rows, cols, near, near.copy()
        shape = (self._n_nodes, self._n_nodes)
        sym = scipy.sparse.csr_array((marks, indices, indptr), shape=shape)
        far = np.asarray(sym[cols, rows], dtype=np.int64).ravel()  # the mark at j on edge i-j
        return rows, cols, near, far

    def _oriented(
        self, pairs: tuple[np.ndarray, ...] | None = None
    ) -> Iterator[tuple[int, int, str]]:
        """Yield ``(source, target, type)`` by node index, in ``to_edges`` order.

        A ``digraph`` 2-cycle yields both directions. ``pairs`` accepts arrays already returned
        by ``_mark_pairs()``.
        """
        for i, j, mi, mj in zip(*(a.tolist() for a in pairs or self._mark_pairs())):
            if self._kind == "digraph" and i != j and mi == mj == ARROW:
                yield i, j, "directed"
                yield j, i, "directed"
                continue
            edge_type, forward = ("directed", True) if i == j else _EDGE_TYPES[(mi, mj)]
            yield (i, j, edge_type) if forward else (j, i, edge_type)

    def _type_counts(self) -> dict[str, int]:
        """Count edges by ``oriented_edges`` type, in order of first appearance."""
        i, j, near, far = self._mark_pairs()
        codes = np.where(i == j, 0, _TYPE_CODE[near * 4 + far])  # a self-loop is directed
        reciprocal_count = 0
        if self._kind == "digraph":
            reciprocal = (i != j) & (near == ARROW) & (far == ARROW)
            codes[reciprocal] = 0
            reciprocal_count = np.count_nonzero(reciprocal)
        if (codes < 0).any():
            raise ValueError("an edge carries a mark pair with no type (validate() says which)")
        _, first = np.unique(codes, return_index=True)
        counts = np.bincount(codes, minlength=len(_TYPE_NAMES))
        counts[0] += reciprocal_count
        return {_TYPE_NAMES[codes[k]]: int(counts[codes[k]]) for k in np.sort(first)}

    def _edge_count_text(self) -> str:
        """Format the edge count and type breakdown for a summary header."""
        counts = self._type_counts()
        types = ", ".join(f"{n} {t}" for t, n in counts.items()) or "no edges"
        return f"{sum(counts.values())} edges ({types})"

    def _edge_lines(
        self,
        *,
        weights: np.ndarray | scipy.sparse.csr_array | None = None,
        limit: int | None = SUMMARY_EDGES,
        more: str = "",
    ) -> list[str]:
        """Format up to ``limit`` edges and append the omitted-edge count, if any."""
        names = self._labels
        pairs = self._mark_pairs()
        lines = []
        for s, t, edge_type in itertools.islice(self._oriented(pairs), limit):
            directed = edge_type in ("directed", "partially_directed")
            weight = f"  ({float(weights[s, t]):+.3g})" if weights is not None and directed else ""
            source, target = (names[s], names[t]) if names else (str(s), str(t))
            lines.append(f"{source} {EDGE_GLYPHS[edge_type]} {target}{weight}")
        left = len(pairs[0]) - len(lines)
        if self._kind == "digraph":
            i, j, near, far = pairs
            left += np.count_nonzero((i != j) & (near == ARROW) & (far == ARROW))
        if left > 0:
            lines.append(f"... and {left} more{more}")
        return lines

    def _summary(
        self,
        *,
        name: str | None = None,
        weights: np.ndarray | scipy.sparse.csr_array | None = None,
        limit: int | None = SUMMARY_EDGES,
        more: str = "",
    ) -> list[str]:
        """Build the graph header, edge lines, and optional latent-node line."""
        head = f"{self._kind}  |  {self._n_nodes} nodes  |  {self._edge_count_text()}"
        lines = [head if name is None else f"{name}  {head}"]
        edges = self._edge_lines(weights=weights, limit=limit, more=more)
        if edges:
            lines += ["", *("  " + line for line in edges)]
        if self._node_types is not None and (self._node_types == LATENT).any():
            latent = np.nonzero(self._node_types == LATENT)[0].tolist()
            names = self._labels
            lines.append("  latent: " + ", ".join(names[i] if names else str(i) for i in latent))
        return lines

    def _iter_edges(self):
        """Yield ``(i, j, mark_i, mark_j)`` per edge, ``i <= j`` (self-loop when ``i == j``)."""
        indptr, indices, marks = self._sym_csr()
        n = self._n_nodes
        for i in range(n):
            lo, hi = int(indptr[i]), int(indptr[i + 1])
            for k in range(lo, hi):
                j = int(indices[k])
                if j > i:
                    yield i, j, int(marks[k]), _lookup(indptr, indices, marks, j, i)
                elif j == i:  # self-loop: single endpoint, both marks are the diagonal cell
                    yield i, i, int(marks[k]), int(marks[k])

    def __eq__(self, other: object) -> bool:
        base = Structure.__eq__(self, other)
        if base is not True:  # NotImplemented (other type) or False (shared fields differ)
            return base
        assert isinstance(other, GraphStructure)  # narrowed by the type check in the base
        if self._kind != other._kind or not _opt_array_equal(self._node_types, other._node_types):
            return False
        # Compare the symmetric view so equal graphs match regardless of store layout. Equal
        # constructor-built graphs always share a layout (a pure function of kind + marks), so the
        # same-layout raw compare is the fast path; differing layouts normalize through _sym_csr.
        if self._layout == other._layout:
            a = (self._indptr, self._indices, self._marks)
            b = (other._indptr, other._indices, other._marks)
        else:
            a, b = self._sym_csr(), other._sym_csr()
        return all(np.array_equal(x, y) for x, y in zip(a, b))

    __hash__ = None  # type: ignore[assignment]


@dataclass(frozen=True, eq=False, slots=True, kw_only=True, repr=False)
class TemporalStructure(Structure):
    """A time-series result: one ``GraphStructure`` per lag.

    ``varma_lingam`` and ``longitudinal_lingam`` return one as ``StructureOutput.structure``;
    ``from_lag_graphs`` and ``from_time_lag_graphs`` build one from existing graphs. An edge
    ``i -> j`` in ``lag(k)`` means ``X_i(t-k) -> X_j(t)``. Lag 0 holds the relations within one
    time step, and a self-loop at a positive lag is a variable's effect on its own future.
    ``lags`` lists the lag values, ``lag_weights`` holds the coefficients, and ``summary_graph()``
    merges the lags into one graph over the variables.

    A ``longitudinal_lingam`` result also has a time axis, one lag stack per occasion: ``times``
    lists the occasions, ``at(time, lag)`` selects a graph, and ``time_weights`` holds every
    occasion's coefficients. On such a result, ``lag(k)``, ``lag_weights``, and
    ``summary_graph()`` describe the final occasion only.
    """

    type: ClassVar[Literal["temporal"]] = "temporal"

    _lags: tuple[GraphStructure, ...] = ()
    _lag_indices: tuple[int, ...] = ()  # explicit lag value of each graph; () -> 0..L-1
    _lag_weights: np.ndarray | None = None  # AR weight stack, (n_lags, n, n) or None
    _lag_weights_ma: np.ndarray | None = None  # MA weight stack (VARMA), (q, n, n) or None
    # Optional (time, lag) axis (Longitudinal). Empty -> a lag-only stack. Each row is one
    # occasion's lag stack aligned 1:1 with ``_lag_indices``; ``_lag*`` project the final occasion.
    _time_graphs: tuple[tuple[GraphStructure, ...], ...] = ()
    _time_indices: tuple[int, ...] = ()  # explicit time value per row, strictly increasing
    _time_weights: np.ndarray | None = None  # (n_times, n_lags, n, n) float64; NaN = uncomputable

    def __post_init__(self) -> None:
        # Explicit parent call: zero-arg super() breaks under @dataclass(slots=True), whose
        # class rebuild leaves the method's __class__ cell pointing at the pre-slots class.
        Structure.__post_init__(self)
        for k, lag in enumerate(self._lags):
            if lag.n_nodes != self._n_nodes:
                raise ValueError(f"lag {k} has {lag.n_nodes} nodes, expected {self._n_nodes}")
        # Default the lag axis to a contiguous 0..L-1 stack when no indices are given;
        # ``from_lag_graphs`` and documents that carry lags pass explicit, possibly uneven, indices.
        if not self._lag_indices and self._lags:
            object.__setattr__(self, "_lag_indices", tuple(range(len(self._lags))))
        elif len(self._lag_indices) != len(self._lags):
            raise ValueError(
                f"lags length {len(self._lag_indices)} != number of graphs {len(self._lags)}"
            )
        self._validate_time_axis()

    def _validate_time_axis(self) -> None:
        """Validate the optional ``(time, lag)`` grid; a no-op for a lag-only stack."""
        if not self._time_graphs:
            if self._time_indices or self._time_weights is not None:
                raise ValueError("_time_indices / _time_weights require _time_graphs")
            return
        if not self._time_indices:
            object.__setattr__(self, "_time_indices", tuple(range(len(self._time_graphs))))
        elif len(self._time_indices) != len(self._time_graphs):
            raise ValueError(
                f"times length {len(self._time_indices)} != occasions {len(self._time_graphs)}"
            )
        if list(self._time_indices) != sorted(set(self._time_indices)):
            raise ValueError(f"times must be strictly increasing, got {self._time_indices}")
        n_lags = len(self._lag_indices)
        for t, row in enumerate(self._time_graphs):
            if len(row) != n_lags:
                raise ValueError(f"occasion {t} has {len(row)} lag graphs, expected {n_lags}")
            for g in row:
                if g.n_nodes != self._n_nodes:
                    raise ValueError(
                        f"occasion {t} graph has {g.n_nodes} nodes, expected {self._n_nodes}"
                    )
        if self._time_weights is not None:
            expect = (len(self._time_graphs), n_lags, self._n_nodes, self._n_nodes)
            if self._time_weights.shape != expect:
                raise ValueError(
                    f"time_weights must have shape {expect}, got {self._time_weights.shape}"
                )
        # The lag-only surface is documented to be a view of the final occasion; enforce it so a
        # corrupted / hand-edited doc cannot load with lag(k) != at(times[-1], k) (fail-fast).
        if self._lags != self._time_graphs[-1]:
            raise ValueError("the lag-only view must equal the final occasion (_lags != final row)")
        if (
            self._lag_weights is not None
            and self._time_weights is not None
            and not np.array_equal(self._lag_weights, np.nan_to_num(self._time_weights[-1]))
        ):
            raise ValueError("lag_weights must be the final occasion's time_weights, zero-filled")

    # ---- construction ---------------------------------------------------------------------

    @classmethod
    def from_lag_graphs(
        cls,
        graphs: Sequence[GraphStructure],
        *,
        lags: Sequence[int] | None = None,
        lag_weights: np.ndarray | None = None,
        lag_weights_ma: np.ndarray | None = None,
        labels: tuple[str, ...] | None = None,
    ) -> TemporalStructure:
        """Build a time-series result from one graph per lag.

        Parameters
        ----------
        graphs : Sequence[GraphStructure]
            One graph per lag, all over the same nodes. An edge ``i -> j`` in the graph for lag
            ``k`` means ``X_i(t-k) -> X_j(t)``; a variable's effect on its own future is a
            self-loop, ``M[i, i] = 2``, built with ``allow_self_loops=True``.
        lags : Sequence[int] or None, default=None
            The lag value of each graph, without repeats; gaps are allowed, for example,
            ``[0, 1, 12]``. ``None`` numbers the graphs ``0, 1, ..., len(graphs) - 1``. ``lag(k)``
            finds a graph by this value, not by position.
        lag_weights : np.ndarray of shape (n_lags, n, n) or None, default=None
            Coefficients, one block per graph: ``lag_weights[k, i, j]`` is the weight of
            ``i -> j`` in ``graphs[k]``. Stored as a read-only ``float64`` copy.
        lag_weights_ma : np.ndarray of shape (q, n, n) or None, default=None
            Moving-average coefficients of a VARMA model: ``lag_weights_ma[m, i, j]`` is the
            weight of ``i -> j`` at moving-average lag ``m + 1``. Stored as a read-only
            ``float64`` copy.
        labels : tuple[str, ...] or None, default=None
            Node names, one per node; ``None`` numbers the nodes from 0.

        Returns
        -------
        TemporalStructure
            The result, without a time axis.

        Raises
        ------
        ValueError
            If ``graphs`` is empty, ``lags`` does not have one value per graph or repeats a value,
            the graphs differ in node count, ``labels`` does not have one entry per node, or a
            weight array has the wrong shape.
        """
        graphs = tuple(graphs)
        if not graphs:
            raise ValueError("from_lag_graphs requires at least one lag GraphStructure")
        n = int(graphs[0].n_nodes)
        # per-graph node counts are re-checked by __post_init__ (which also guards direct
        # construction); the lags/graphs length must be checked here, since __post_init__ reads an
        # empty _lag_indices as "default to 0..L-1" and so cannot reject an explicit lags=[].
        if lags is None:
            indices = tuple(range(len(graphs)))
        else:
            indices = tuple(int(x) for x in lags)
            if len(indices) != len(graphs):
                raise ValueError(f"lags length {len(indices)} != number of graphs {len(graphs)}")
            if len(set(indices)) != len(indices):
                raise ValueError(f"lags must be unique, got {indices}")
        ar = _as_lag_stack(lag_weights, "lag_weights", n=n, n_lags=len(graphs))
        ma = _as_lag_stack(lag_weights_ma, "lag_weights_ma", n=n, n_lags=None)
        return cls(
            _n_nodes=n,
            _labels=labels,
            _lags=graphs,
            _lag_indices=indices,
            _lag_weights=ar,
            _lag_weights_ma=ma,
        )

    @classmethod
    def from_time_lag_graphs(
        cls,
        time_graphs: Sequence[Sequence[GraphStructure]],
        *,
        lags: Sequence[int] | None = None,
        times: Sequence[int] | None = None,
        time_weights: np.ndarray | None = None,
        labels: tuple[str, ...] | None = None,
    ) -> TemporalStructure:
        """Build a panel result: one lag stack per occasion.

        ``lag``, ``n_lags``, ``lag_weights``, and ``summary_graph()`` then describe the final
        occasion; ``at`` and ``time_weights`` reach every occasion.

        Parameters
        ----------
        time_graphs : Sequence[Sequence[GraphStructure]] of shape (n_times, n_lags)
            One row per occasion, each row a graph per lag in the order of ``lags``; every row has
            the same length. Use an empty graph for a block that cannot be computed.
        lags : Sequence[int] or None, default=None
            The lag value of each column, without repeats. ``None`` numbers the columns
            ``0, 1, ..., n_lags - 1``.
        times : Sequence[int] or None, default=None
            The occasion value of each row, strictly increasing; gaps are allowed. For example, a
            panel whose first occasion has no estimate starts at ``1``. ``None`` numbers the rows
            ``0, 1, ..., n_times - 1``.
        time_weights : np.ndarray of shape (n_times, n_lags, n, n) or None, default=None
            Coefficients: ``time_weights[t, k, i, j]`` is the weight of ``i -> j`` in
            ``time_graphs[t][k]``, with ``NaN`` for a block that cannot be computed.
            ``lag_weights`` is the last row with ``NaN`` replaced by ``0``.
        labels : tuple[str, ...] or None, default=None
            Node names, one per node; ``None`` numbers the nodes from 0.

        Returns
        -------
        TemporalStructure
            The result, with a time axis.

        Raises
        ------
        ValueError
            If the grid is empty or its rows differ in length, ``lags`` or ``times`` does not
            match the grid, ``lags`` repeats a value, ``times`` is not strictly increasing, the
            graphs differ in node count, ``labels`` does not have one entry per node, or
            ``time_weights`` has the wrong shape.
        """
        rows = tuple(tuple(row) for row in time_graphs)
        if not rows or not rows[0]:
            raise ValueError("from_time_lag_graphs requires a non-empty (time, lag) grid")
        n = int(rows[0][0].n_nodes)
        n_lags = len(rows[0])
        if lags is None:
            lag_indices = tuple(range(n_lags))
        else:
            lag_indices = tuple(int(x) for x in lags)
            if len(lag_indices) != n_lags:
                raise ValueError(f"lags length {len(lag_indices)} != row width {n_lags}")
            if len(set(lag_indices)) != len(lag_indices):
                raise ValueError(f"lags must be unique, got {lag_indices}")
        if times is None:
            time_indices = tuple(range(len(rows)))
        else:
            time_indices = tuple(int(t) for t in times)
            # __post_init__ reads an empty _time_indices as "default to 0..T-1", so an explicit
            # times=[] must be rejected here (mirrors the lags=[] guard above).
            if len(time_indices) != len(rows):
                raise ValueError(f"times length {len(time_indices)} != occasions {len(rows)}")
        tw = None
        if time_weights is not None:
            tw = np.asarray(time_weights, dtype=np.float64)
            expect = (len(rows), n_lags, n, n)
            if tw.shape != expect:
                raise ValueError(f"time_weights must have shape {expect}, got {tw.shape}")
            tw = _frozen(tw, np.float64)
        # The lag-only projection is the final occasion (fully-observed window, NaN zero-filled).
        proj_weights = None if tw is None else _frozen(np.nan_to_num(tw[-1]), np.float64)
        return cls(
            _n_nodes=n,
            _labels=labels,
            _lags=rows[-1],
            _lag_indices=lag_indices,
            _lag_weights=proj_weights,
            _lag_weights_ma=None,
            _time_graphs=rows,
            _time_indices=time_indices,
            _time_weights=tw,
        )

    # ---- surface --------------------------------------------------------------------------

    @property
    def n_lags(self) -> int:
        """Number of lags in the stack."""
        return len(self._lags)

    @property
    def lags(self) -> tuple[int, ...]:
        """The lag value of each graph, in stack order; ``lag(k)`` looks up by this value."""
        return self._lag_indices

    def lag(self, k: int) -> GraphStructure:
        """Select the graph at lag *value* ``k`` (not position).

        For a time-carrying structure this is the final occasion's lag graph (``at(times[-1], k)``);
        the full grid is reached via ``at`` / ``times``.

        Parameters
        ----------
        k : int
            Lag value to look up, matched against ``lags`` (not a positional index).

        Returns
        -------
        GraphStructure
            The graph stored at lag ``k``.

        Raises
        ------
        KeyError
            If no graph is stored at lag ``k``.
        """
        try:
            pos = self._lag_indices.index(k)
        except ValueError:
            raise KeyError(f"no lag {k}; present lags are {self._lag_indices}") from None
        return self._lags[pos]

    @property
    def lag_weights(self) -> np.ndarray | None:
        """Coefficients ``(n_lags, n, n)``, or ``None`` when the result has none.

        ``lag_weights[k, i, j]`` is the weight of ``i -> j`` at lag ``lags[k]``. On a panel result
        it is the final occasion with ``NaN`` replaced by ``0``; ``time_weights`` keeps every
        occasion.
        """
        return self._lag_weights

    @property
    def lag_weights_ma(self) -> np.ndarray | None:
        """Moving-average coefficients ``(q, n, n)`` from ``varma_lingam``, or ``None``.

        ``lag_weights_ma[m, i, j]`` is the weight of ``i -> j`` at moving-average lag ``m + 1``.
        """
        return self._lag_weights_ma

    @property
    def n_times(self) -> int:
        """Number of absolute occasions in the ``(time, lag)`` axis; ``0`` for a lag-only stack."""
        return len(self._time_indices)

    @property
    def times(self) -> tuple[int, ...]:
        """The explicit occasion index of each ``(time, lag)`` row; ``()`` when lag-only."""
        return self._time_indices

    def at(self, time: int, lag: int) -> GraphStructure:
        """Select the graph at occasion *value* ``time`` and lag *value* ``lag``.

        A lag-only stack has no time axis, so every ``at`` call raises there -- use ``lag`` instead.

        Parameters
        ----------
        time : int
            Occasion value to look up, matched against ``times`` (not a positional index).
        lag : int
            Lag value to look up, matched against ``lags`` (not a positional index).

        Returns
        -------
        GraphStructure
            The graph stored at occasion ``time`` and lag ``lag``.

        Raises
        ------
        KeyError
            If either ``time`` or ``lag`` is absent (including any ``at`` call on a lag-only stack).
        """
        try:
            ti = self._time_indices.index(time)
        except ValueError:
            raise KeyError(f"no time {time}; present times are {self._time_indices}") from None
        try:
            li = self._lag_indices.index(lag)
        except ValueError:
            raise KeyError(f"no lag {lag}; present lags are {self._lag_indices}") from None
        return self._time_graphs[ti][li]

    @property
    def time_weights(self) -> np.ndarray | None:
        """Coefficients ``(n_times, n_lags, n, n)``, or ``None`` without a time axis or weights.

        ``time_weights[t, k, i, j]`` is the weight of ``i -> j`` in ``at(times[t], lags[k])``;
        ``NaN`` marks a block that could not be computed.
        """
        return self._time_weights

    def _summary(
        self,
        *,
        name: str | None = None,
        weights: np.ndarray | scipy.sparse.csr_array | None = None,
        limit: int | None = SUMMARY_EDGES,
        more: str = "",
    ) -> list[str]:
        """Build the temporal header and one edge block per lag.

        ``weights`` is unused; each lag graph supplies its own edge lines.
        """
        head = f"temporal  |  {self._n_nodes} nodes  |  {self.n_lags} lags"
        lines = [head if name is None else f"{name}  {head}"]
        for lag, graph in zip(self._lag_indices, self._lags):
            lines += ["", f"  lag {lag}  |  {graph._edge_count_text()}"]
            lines += ["    " + line for line in graph._edge_lines(limit=limit, more=more)]
        return lines

    def summary_graph(self, *, include_lag0: bool = True) -> SummaryGraph:
        """Merge the lags into one directed graph over the variables.

        ``i -> j`` in the summary means ``i -> j`` in at least one lag graph; ``lags_of(i, j)``
        lists those lags. ``i -> j`` at one lag and ``j -> i`` at another form a 2-cycle, and a
        self-loop at a positive lag stays a self-loop. Only directed edges are kept; an undirected
        edge or one with a circle is left out. On a panel result it merges the final occasion's
        lags. Each call builds a new graph.

        Parameters
        ----------
        include_lag0 : bool, default=True
            Include lag 0, the relations within one time step. ``False`` summarizes the effects
            across time steps only.

        Returns
        -------
        SummaryGraph
            A ``kind="digraph"`` graph; ``lags_of(i, j)`` gives the lags of each edge.
        """
        n = self._n_nodes
        total = np.zeros((n, n), dtype=bool)
        lag_sets: dict[tuple[int, int], list[int]] = {}
        for lag, graph in zip(self._lag_indices, self._lags):
            if lag == 0 and not include_lag0:
                continue
            support = _directed_support(graph.to_numpy())
            total |= support
            for i, j in zip(*np.nonzero(support)):
                lag_sets.setdefault((int(i), int(j)), []).append(int(lag))
        # Rebuild endpoint marks from the directed union: a mutual pair (or the diagonal) is a
        # double arrowhead (2-cycle / self-loop), a one-way pair is tail -> arrow.
        marks = np.zeros((n, n), dtype=np.int8)
        one_way = total & ~total.T
        both = total & total.T
        marks[one_way] = TAIL
        marks[one_way.T] = ARROW
        marks[both] = ARROW
        indptr, indices, mark_data = _dense_to_csr(marks)
        edge_lags = tuple((i, j, tuple(sorted(lag_sets[(i, j)]))) for i, j in sorted(lag_sets))
        return SummaryGraph(
            _n_nodes=n,
            _labels=self._labels,
            _indptr=_frozen(indptr, np.int64),
            _indices=_frozen(indices, np.int32),
            _marks=_frozen(mark_data, np.int8),
            _kind="digraph",
            _edge_lags=edge_lags,
        )

    def __eq__(self, other: object) -> bool:
        base = Structure.__eq__(self, other)
        if base is not True:
            return base
        assert isinstance(other, TemporalStructure)  # narrowed by the base type check
        if self._lag_indices != other._lag_indices or self._time_indices != other._time_indices:
            return False
        if len(self._lags) != len(other._lags) or not all(
            a == b for a, b in zip(self._lags, other._lags)
        ):
            return False
        if len(self._time_graphs) != len(other._time_graphs) or not all(
            len(ra) == len(rb) and all(a == b for a, b in zip(ra, rb))
            for ra, rb in zip(self._time_graphs, other._time_graphs)
        ):
            return False
        return (
            _opt_array_equal(self._lag_weights, other._lag_weights, equal_nan=True)
            and _opt_array_equal(self._lag_weights_ma, other._lag_weights_ma, equal_nan=True)
            and _opt_array_equal(self._time_weights, other._time_weights, equal_nan=True)
        )

    __hash__ = None  # type: ignore[assignment]


@dataclass(frozen=True, eq=False, slots=True, kw_only=True, repr=False)
class SummaryGraph(GraphStructure):
    """The lags of a ``TemporalStructure`` merged into one directed graph over the variables.

    Get one from ``TemporalStructure.summary_graph()``. It is a ``GraphStructure`` with
    ``kind="digraph"``: ``i -> j`` means ``X_i`` affects ``X_j`` at one or more lags, so ``i -> j``
    at one lag and ``j -> i`` at another form a 2-cycle, and a variable's effect on its own future
    is a self-loop. ``lags_of(i, j)`` and ``edge_lags`` give the lags behind each edge. Every
    ``GraphStructure`` view applies. ``StructureOutput.save`` raises ``TypeError`` for a
    ``SummaryGraph``; save the ``TemporalStructure`` and call ``summary_graph()`` again after
    loading.
    """

    # Per directed edge ``(i, j, lags)``, sorted by ``(i, j)``; ``lags`` is the sorted, unique tuple
    # of lag values at which ``i -> j`` appears (a self-loop when ``i == j``).
    _edge_lags: tuple[tuple[int, int, tuple[int, ...]], ...] = ()

    def __post_init__(self) -> None:
        # Explicit parent call: zero-arg super() breaks under @dataclass(slots=True). GraphStructure
        # has no own __post_init__, so Structure's runs the shared node/label checks; the kind is
        # fixed for this derived type.
        Structure.__post_init__(self)
        if self._kind != "digraph":
            raise ValueError(f"SummaryGraph is always kind='digraph', got '{self._kind}'")

    @property
    def edge_lags(self) -> tuple[tuple[int, int, tuple[int, ...]], ...]:
        """Every edge with its lags, as ``(i, j, lags)`` triples sorted by ``(i, j)``."""
        return self._edge_lags

    def lags_of(self, i: int, j: int) -> tuple[int, ...]:
        """Return the sorted lag values at which the directed edge ``i -> j`` appears.

        Parameters
        ----------
        i : int
            Source node index of the directed edge.
        j : int
            Target node index of the directed edge.

        Returns
        -------
        tuple[int, ...]
            The sorted lag values at which ``i -> j`` appears; ``()`` if the edge is absent.
        """
        for a, b, lags in self._edge_lags:
            if a == i and b == j:
                return lags
        return ()

    def __eq__(self, other: object) -> bool:
        base = GraphStructure.__eq__(self, other)
        if base is not True:  # NotImplemented (other type) or False (marks / kind differ)
            return base
        assert isinstance(other, SummaryGraph)  # narrowed: same concrete type as self
        return self._edge_lags == other._edge_lags

    __hash__ = None  # type: ignore[assignment]


def _frozen(arr: np.ndarray, dtype: npt.DTypeLike) -> np.ndarray:
    """An owned, read-only copy in ``dtype`` -- keeps the frozen container's store immutable."""
    out = np.array(arr, dtype=dtype)
    out.setflags(write=False)
    return out


def relabel(structure: Structure, labels: tuple[str, ...]) -> Structure:
    """The structure with its nodes named ``labels``; a temporal stack renames every lag graph."""
    if not isinstance(structure, TemporalStructure):
        return dataclasses.replace(structure, _labels=labels)

    def named(graph: GraphStructure) -> GraphStructure:
        return dataclasses.replace(graph, _labels=labels)

    rows = tuple(tuple(named(g) for g in row) for row in structure._time_graphs)
    lags = rows[-1] if rows else tuple(named(g) for g in structure._lags)
    return dataclasses.replace(structure, _labels=labels, _lags=lags, _time_graphs=rows)


def _opt_array_equal(
    a: np.ndarray | None, b: np.ndarray | None, *, equal_nan: bool = False
) -> bool:
    """``np.array_equal`` for optionals: both ``None`` are equal; exactly one ``None`` differs."""
    if a is None or b is None:
        return a is None and b is None
    return bool(np.array_equal(a, b, equal_nan=equal_nan))


def _as_lag_stack(
    weights: np.ndarray | None, name: str, *, n: int, n_lags: int | None
) -> np.ndarray | None:
    """Validate a ``(*, n, n)`` weight stack and return an owned read-only ``float64`` copy.

    ``n_lags`` pins the leading axis (the AR stack is 1:1 with the lag graphs); pass ``None`` to
    leave it free (the MA stack carries its own order ``q``).
    """
    if weights is None:
        return None
    arr = np.asarray(weights, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1] != n or arr.shape[2] != n:
        expect = f"({'n_lags' if n_lags is None else n_lags}, {n}, {n})"
        raise ValueError(f"{name} must have shape {expect}, got {arr.shape}")
    if n_lags is not None and arr.shape[0] != n_lags:
        raise ValueError(f"{name} has {arr.shape[0]} lags, expected {n_lags} to match lags")
    return _frozen(arr, np.float64)


def _dense_to_csr(M: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Canonical CSR (sorted columns, no stored zeros) of the dense endpoint matrix."""
    import scipy.sparse

    sp = scipy.sparse.csr_array(M)
    sp.sum_duplicates()
    sp.sort_indices()
    return (
        sp.indptr.astype(np.int64),
        sp.indices.astype(np.int32),
        sp.data.astype(np.int8),
    )


def _lookup(indptr: np.ndarray, indices: np.ndarray, marks: np.ndarray, a: int, b: int) -> int:
    """Mark at cell ``(a, b)`` of an endpoint-mark CSR (``NULL`` if absent); ``O(log deg)``."""
    lo, hi = int(indptr[a]), int(indptr[a + 1])
    row = indices[lo:hi]
    pos = int(np.searchsorted(row, b))
    if pos < row.size and row[pos] == b:
        return int(marks[lo + pos])
    return NULL


def _try_compact(
    n: int, indptr: np.ndarray, indices: np.ndarray, marks: np.ndarray, kind: Kind
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Halved dag/cpdag store (one entry per edge) if the marks conform, else ``None`` (stay full).

    A directed edge ``i -> j`` keeps a single entry ``(i, j)`` carrying the far mark ``ARROW``; an
    undirected ``i -- j`` a single ``(min, max)`` entry with ``TAIL``; a self-loop the diagonal.
    Returns ``None`` for ``pag`` / ``digraph`` or any nonconforming / degenerate store (a bidirected
    pair, a circle, an undirected edge under ``dag``, an asymmetric half-edge, a non-canonical or
    ``NULL``-bearing CSR) so ``validate()`` still flags it. The public views reconstruct the
    symmetric matrix via ``_expand_compact``.
    """
    if kind not in ("dag", "cpdag"):
        return None
    import scipy.sparse

    n_int = int(n)
    indptr = np.asarray(indptr, dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int32)
    marks = np.asarray(marks, dtype=np.int8)
    # Only compact a canonical, non-degenerate store; anything else stays symmetric so
    # ``validate()`` reports it exactly as before (no silent canonicalization, no dropped NULL
    # half-edge). Stays ``O(nnz)`` -- never materializes a dense ``(n, n)`` matrix, so a large
    # sparse graph loads cheaply.
    if (marks == NULL).any() or (marks == CIRCLE).any():
        return None
    rows = np.repeat(np.arange(n_int, dtype=np.int64), np.diff(indptr))
    cols = indices.astype(np.int64)
    same_row = rows[1:] == rows[:-1]
    if bool(np.any(same_row & (np.diff(cols) <= 0))):
        return None  # unsorted or duplicate columns within a row (non-canonical CSR)
    sp = scipy.sparse.csr_array((marks, indices, indptr), shape=(n_int, n_int))
    spT = sp.T.tocsr()
    spT.sort_indices()
    if not (np.array_equal(sp.indptr, spT.indptr) and np.array_equal(sp.indices, spT.indices)):
        return None  # asymmetric support -> a half-edge; leave it for validate()
    partner = spT.data.astype(np.int8)  # partner[k] = mark at (cols[k], rows[k])
    off = rows != cols
    upper = off & (rows < cols)
    mi, mj = marks[upper], partner[upper]
    fwd = (mi == TAIL) & (mj == ARROW)  # r -> c
    bwd = (mi == ARROW) & (mj == TAIL)  # c -> r
    und = (mi == TAIL) & (mj == TAIL)  # r -- c (cpdag only)
    ok = fwd | bwd | (und if kind == "cpdag" else np.zeros_like(und))
    if not bool(ok.all()):
        return None  # a bidirected pair, or an undirected edge under dag
    r_up, c_up = rows[upper], cols[upper]
    er = np.concatenate([np.where(bwd, c_up, r_up), rows[~off]])  # + self-loops verbatim
    ec = np.concatenate([np.where(bwd, r_up, c_up), cols[~off]])
    em = np.concatenate([np.where(und, TAIL, ARROW).astype(np.int8), marks[~off]])
    order = np.lexsort((ec, er))
    er, ec, em = er[order], ec[order], em[order]
    new_indptr = np.zeros(n_int + 1, dtype=np.int64)
    np.add.at(new_indptr, er + 1, 1)
    np.cumsum(new_indptr, out=new_indptr)
    return new_indptr, ec.astype(np.int32), em.astype(np.int8)


def _expand_compact(
    n: int, indptr: np.ndarray, indices: np.ndarray, marks: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Symmetric endpoint-mark CSR from the halved dag/cpdag store (inverse of ``_try_compact``)."""
    if indices.size == 0:
        return indptr, indices, marks
    n_int = int(n)
    rows = np.repeat(np.arange(n_int, dtype=np.int64), np.diff(indptr))
    cols = indices.astype(np.int64)
    diag = rows == cols
    off = ~diag
    n_off = int(off.sum())
    r = np.concatenate([rows[diag], rows[off], cols[off]])
    c = np.concatenate([cols[diag], cols[off], rows[off]])
    m = np.concatenate(
        [
            marks[diag],  # self-loops verbatim
            np.full(n_off, TAIL, dtype=np.int8),  # (r, c): source / undirected end -> TAIL
            np.where(marks[off] == ARROW, ARROW, TAIL).astype(
                np.int8
            ),  # (c, r): the stored far mark
        ]
    )
    order = np.lexsort((c, r))
    r, c, m = r[order], c[order], m[order]
    new_indptr = np.zeros(n_int + 1, dtype=np.int64)
    np.add.at(new_indptr, r + 1, 1)
    np.cumsum(new_indptr, out=new_indptr)
    # Freeze so a compact-layout ``to_scipy_sparse`` hands scipy read-only buffers, identical to the
    # symmetric layout (which aliases the frozen store).
    return _frozen(new_indptr, np.int64), _frozen(c, np.int32), _frozen(m, np.int8)


def _directed_support(m: np.ndarray) -> np.ndarray:
    """Boolean ``(n, n)`` directed support of a mark matrix: ``D[i, j]`` is the edge ``i -> j``.

    A plain directed edge (arrowhead at ``j``) sets ``D[i, j]``; a 2-cycle / bidirected pair sets
    both directions; an autoregressive self-loop (``m[i, i] == ARROW``) sets ``D[i, i]``. Undirected
    and circle marks carry no direction and are ignored. Shared with ``metrics`` so the topological
    collapse cannot drift between the summary-graph projection and temporal scoring.
    """
    directed = (m == TAIL) & (m.T == ARROW)
    both_arrow = (m == ARROW) & (m.T == ARROW)
    return directed | both_arrow


def _infer_kind(M: np.ndarray) -> Kind:
    """Infer the mark domain from a dense endpoint matrix; used when ``kind`` is not given."""
    if (M == CIRCLE).any():
        return "pag"
    off_diag = np.arange(M.shape[0])[:, None] != np.arange(M.shape[0])
    both_arrow = (M == ARROW) & (M.T == ARROW) & off_diag  # off-diagonal: a self-loop is not pag
    if both_arrow.any():
        return "pag"
    both_tail = (M == TAIL) & (M.T == TAIL) & off_diag
    if both_tail.any():
        return "cpdag"
    return "dag"
