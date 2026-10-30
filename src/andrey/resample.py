"""Bootstrap refits of a method: edge frequencies by endpoint mark, and the averaged graph.

:func:`bootstrap` refits one method on resamples of the rows and counts, for every pair of
variables, how often each endpoint-mark pair appears. :class:`BootstrapOutput` holds those counts
and the averaged graph: the vote of the fits, made a valid graph of the method's kind.

Resample ``b`` draws from ``SeedSequence(seed, spawn_key=(b, 0))``, and a seeded method fits with
a seed from ``spawn_key=(b, 1)``, so a resample depends on ``seed`` and ``b`` only: the result is
the same at any worker count, and the first ``k`` resamples of a longer run are the shorter run's.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np

from andrey import api
from andrey.api._adapt import flag, int_in_range
from andrey.core import ARROW, CIRCLE, LATENT, TAIL, GraphStructure, StructureOutput
from andrey.core import backend as _backend
from andrey.core._convert import extend_in_order, mag_of_pag, pag_of
from andrey.core.knowledge import mpdag
from andrey.core.knowledge import orient as knowledge_orient
from andrey.core.orient import _cpdag_of_pdag
from andrey.core.structure import (
    _EDGE_TYPES,
    EDGE_GLYPHS,
    Kind,
    Structure,
    TemporalStructure,
    is_ancestral_kind,
)
from andrey.core.warning_policy import AndreyWarning, DeprecatedArgumentWarning, warn_once
from andrey.search._parallel import pick_context
from andrey.search.hc import tie_ranks

if TYPE_CHECKING:
    import numpy.typing as npt

    from andrey.core.knowledge import Constraints
    from andrey.core.output import JSONValue
    from andrey.spec.models import MethodSpec

#: The edges a pair of variables can read, as ``i edge j``, in the order the tables list them.
EDGES: tuple[str, ...] = ("->", "<-", "--", "<->", "o->", "<-o", "o-o", "-o", "o-")
_GLYPH_MARKS: dict[str, tuple[int, int]] = {
    "->": (TAIL, ARROW),
    "<-": (ARROW, TAIL),
    "--": (TAIL, TAIL),
    "<->": (ARROW, ARROW),
    "o->": (CIRCLE, ARROW),
    "<-o": (ARROW, CIRCLE),
    "o-o": (CIRCLE, CIRCLE),
    "-o": (TAIL, CIRCLE),
    "o-": (CIRCLE, TAIL),
}
_ABSENT = "absent"
_KIND_ORDER: dict[Kind, int] = {"dag": 0, "cpdag": 1, "pag": 2}
# The edge a pair takes when its most frequent edges tie: adjacent, with the least said about it.
_TIE_EDGE: dict[Kind, tuple[int, int]] = {
    "dag": (TAIL, TAIL),
    "cpdag": (TAIL, TAIL),
    "pag": (CIRCLE, CIRCLE),
    "undirected": (TAIL, TAIL),
}
_STATES = 16  # a pair's mark pairs, coded near * 4 + far
_MADE_VALID: tuple[Kind, ...] = ("dag", "cpdag", "pag")  # the kinds a vote is made a valid graph of


@dataclass(frozen=True, slots=True)
class _Slice:
    """The counts of one graph: a static graph, or one lag of a time-series result.

    A graph whose ``kind`` is not ``"digraph"`` counts mark pairs: ``codes`` holds
    ``(i * n + j) * 16 + near * 4 + far`` for ``i <= j``, and each fit adds at most one per pair. A
    ``"digraph"`` counts directed edges, ``i * n + j`` for ``i -> j``, so that a 2-cycle adds both.
    """

    kind: Kind
    codes: np.ndarray
    counts: np.ndarray


@dataclass(frozen=True, eq=False, slots=True, repr=False)
class BootstrapOutput:
    """The edge frequencies over bootstrap refits of one method, and the averaged graph.

    ``bootstrap`` returns one (``multi_group_direct_lingam`` returns one per group), and
    ``BootstrapOutput.from_fits`` builds one from fits made elsewhere. ``structure`` is the
    averaged graph. ``frequency`` reads the share of fits in which a pair has a given edge,
    ``table`` lists every edge that appeared, and ``average`` takes the averaged graph at another
    threshold without refitting. ``print(out)`` shows the averaged graph and its most frequent
    edges. The result is read-only.

    An edge is read as the pair ``(i, j)`` carries it: ``->`` is a tail at ``i`` and an arrowhead at
    ``j``, ``<-`` the reverse, ``--`` two tails, ``<->`` two arrowheads, and ``o`` a circle. A
    fit adds its edge to a pair's count once. A time-series result has its own frequencies for each
    lag. A lag graph that is a ``digraph`` (every lag above 0) counts each directed edge ``i -> j``
    on its own, so a 2-cycle adds to both ``->`` and ``<-``, and it holds no other edge.
    """

    _structure: Structure
    _n_resamples: int
    _threshold: float
    _lags: tuple[int, ...] | None
    _slices: tuple[_Slice, ...]
    _metadata: Mapping[str, JSONValue]
    _fits: tuple[StructureOutput, ...] | None = None
    _strict: bool = False
    _nearest: tuple[int, Structure] | None = None  # the fit nearest the others, by resample
    _constraints: Constraints | None = None
    _ranks: np.ndarray | None = None  # the nodes' tie ranks: the data's, from bootstrap

    # ---- construction ---------------------------------------------------------------------

    @classmethod
    def from_fits(
        cls,
        fits: Sequence[StructureOutput],
        *,
        threshold: float = 0.5,
        strict: bool = False,
        keep_fits: bool = False,
        metadata: Mapping[str, JSONValue] | None = None,
    ) -> BootstrapOutput:
        """Count the edges of fits and build the averaged graph.

        Use it to summarize fits made elsewhere, such as resamples drawn another way. Every fit
        must be over the same nodes and, for a time-series result, the same lags.

        Parameters
        ----------
        fits : Sequence[StructureOutput]
            The results to count, at least one.
        threshold : float, default=0.5
            A pair of the averaged graph must be adjacent in more than this share of the fits, or
            in all of them, in ``[0.5, 1]``.
        strict : bool, default=False
            Raise when the vote of the fits has no valid graph of their kind, instead of returning
            the fit nearest the others with a warning.
        keep_fits : bool, default=False
            Keep ``fits`` in the ``fits`` property.
        metadata : Mapping[str, JSONValue] or None, default=None
            Run values by name, with the types ``StructureOutput`` accepts.

        Returns
        -------
        BootstrapOutput
            The counts and the averaged graph. The vote gives a pair its most frequent edge when
            more fits have that edge than have no edge and the pair is adjacent in more than
            ``threshold`` of the fits, or in all of them; otherwise the pair has no edge. When two
            edges tie for most frequent, the pair takes the least committal one: ``--``, or
            ``o-o`` for a ``pag``. A ``digraph`` lag graph takes each directed edge that more than
            ``threshold`` of the fits have, or all of them. The vote is then made a valid graph of
            the fits' kind, the widest among them (``dag``, ``cpdag``, ``pag``, with a ``mag`` read
            as a ``pag``), as ``bootstrap``'s Notes describe. ``metadata["averaged_graph"]`` is
            ``"vote"``, or ``"nearest_resample"`` when the vote has no valid graph of that kind and
            the averaged graph is the fit nearest the others. ``from_fits`` has no data to rank
            the nodes by, so a ``dag`` vote with a tie edge becomes a ``cpdag`` that leaves the
            tie edge undirected; ``bootstrap`` points it by the ranks of its data.

        Raises
        ------
        TypeError
            If a fit is not a ``StructureOutput``.
        ValueError
            If ``fits`` is empty, ``threshold`` is not in ``[0.5, 1]``, the fits differ in graph
            type, node count, labels, or lags, a graph is a ``digraph`` or ``undirected`` in some
            fits and another kind in others, a fit has latent nodes, or ``strict`` is ``True`` and
            the vote has no valid graph of the fits' kind.

        Warns
        -----
        AndreyWarning
            When ``strict`` is ``False`` and the vote has no valid graph of the fits' kind.
        """
        return cls._count(fits, threshold, strict, keep_fits, metadata, None, None)

    @classmethod
    def _count(
        cls,
        fits: Sequence[StructureOutput],
        threshold: float,
        strict: bool,
        keep_fits: bool,
        metadata: Mapping[str, JSONValue] | None,
        constraints: Constraints | None,
        ranks: np.ndarray | None,
    ) -> BootstrapOutput:
        """``from_fits``, for fits made under the knowledge ``constraints`` (``None``: none).

        ``ranks`` holds each node's :func:`~andrey.search.hc.tie_ranks` rank on the data the fits
        ran on, which orients a ``dag`` vote's tied edges; ``None`` when there is no data.
        """
        fits = tuple(fits)
        if not fits:
            raise ValueError("fits is empty; at least one fit is needed")
        for fit in fits:
            if not isinstance(fit, StructureOutput):
                raise TypeError(f"fits must hold StructureOutput results, got {type(fit).__name__}")
        threshold = _check_threshold(threshold)
        strict = _check_strict(strict)
        first = fits[0].structure
        n, labels = first.n_nodes, first.labels
        lags = first.lags if isinstance(first, TemporalStructure) else None
        n_slices = 1 if lags is None else len(lags)
        coded: list[list[np.ndarray]] = [[] for _ in range(n_slices)]
        kinds: list[set[Kind]] = [set() for _ in range(n_slices)]
        for fit in fits:
            s = fit.structure
            same = type(s) is type(first) and s.n_nodes == n and s.labels == labels
            graphs = _graphs(s, lags) if same else None
            if graphs is None:
                raise ValueError(
                    "fits differ in graph type, node count, labels, or lags; "
                    "every fit needs the same nodes"
                )
            for k, graph in enumerate(graphs):
                if graph.node_types is not None and (graph.node_types == LATENT).any():
                    raise ValueError("a fit with latent nodes cannot be counted; its nodes vary")
                kinds[k].add(graph.kind)
                coded[k].append(
                    _directed_codes(graph) if graph.kind == "digraph" else _codes(graph)
                )
        slices = []
        for k in range(n_slices):
            codes, counts = np.unique(np.concatenate(coded[k]), return_counts=True)
            slices.append(_Slice(_widest(kinds[k]), codes, counts.astype(np.int64)))
        nearest = None
        if lags is None and slices[0].kind in _MADE_VALID:
            index = _nearest(slices[0], coded[0], len(fits))
            nearest = (index, fits[index].structure)
        structure, source = _average(
            slices, len(fits), threshold, lags, n, labels, nearest, strict, constraints, ranks
        )
        return cls(
            _structure=structure,
            _n_resamples=len(fits),
            _threshold=threshold,
            _lags=lags,
            _slices=tuple(slices),
            _metadata={**(metadata or {}), "averaged_graph": source},
            _fits=fits if keep_fits else None,
            _strict=strict,
            _nearest=nearest,
            _constraints=constraints,
            _ranks=ranks,
        )

    # ---- surface --------------------------------------------------------------------------

    @property
    def structure(self) -> Structure:
        """The averaged graph at ``threshold``: a ``GraphStructure``, or a ``TemporalStructure``.

        A time-series result has one graph per lag and no time axis or weights.
        """
        return self._structure

    @property
    def n_resamples(self) -> int:
        """The number of fits counted."""
        return self._n_resamples

    @property
    def n_boot(self) -> int:
        """The former name of ``n_resamples``, which stops working in the next release."""
        warn_once(
            "andrey.BootstrapOutput.n_boot is now n_resamples; the former name stops working in "
            "the next release",
            DeprecatedArgumentWarning,
        )
        return self._n_resamples

    @property
    def threshold(self) -> float:
        """The share of fits in which a pair of ``structure`` must be adjacent: more, or all."""
        return self._threshold

    @property
    def strict(self) -> bool:
        """Whether a vote with no valid graph of the fits' kind raises, rather than warns."""
        return self._strict

    @property
    def fits(self) -> tuple[StructureOutput, ...] | None:
        """The fits, in resample order, or ``None`` unless ``keep_fits=True``."""
        return self._fits

    @property
    def metadata(self) -> Mapping[str, JSONValue]:
        """Run values such as ``method``, ``seed``, and ``block_length``, as a read-only mapping."""
        return MappingProxyType(dict(self._metadata))

    def average(self, threshold: float = 0.5) -> Structure:
        """Return the averaged graph at ``threshold``, without refitting.

        Parameters
        ----------
        threshold : float, default=0.5
            A pair must be adjacent in more than this share of the fits, or in all of them, in
            ``[0.5, 1]``.

        Returns
        -------
        Structure
            The graph ``from_fits`` describes, at this threshold, with the ``strict`` this result
            was built with. When the vote has no valid graph of the fits' kind, it is the fit
            nearest the others, as ``structure`` would be.

        Raises
        ------
        ValueError
            If ``threshold`` is not in ``[0.5, 1]``, or the result was built with ``strict=True``
            and the vote at ``threshold`` has no valid graph of the fits' kind.

        Warns
        -----
        AndreyWarning
            When the result was built with ``strict=False`` and the vote at ``threshold`` has no
            valid graph of the fits' kind.
        """
        structure = self._structure
        graph, _ = _average(
            self._slices,
            self._n_resamples,
            _check_threshold(threshold),
            self._lags,
            structure.n_nodes,
            structure.labels,
            self._nearest,
            self._strict,
            self._constraints,
            self._ranks,
        )
        return graph

    def frequency(self, edge: str, *, lag: int | None = None) -> np.ndarray:
        """Return the share of fits in which each pair has ``edge``, as an ``(n, n)`` array.

        Parameters
        ----------
        edge : str
            One of ``"->"``, ``"<-"``, ``"--"``, ``"<->"``, ``"o->"``, ``"<-o"``, ``"o-o"``,
            ``"-o"``, ``"o-"``, or ``"absent"``. Cell ``[i, j]`` is the share of fits in which
            nodes ``i`` and ``j`` read ``i edge j``, so the array for ``"<-"`` is the transpose of
            the one for ``"->"``. ``"absent"`` is the share with no edge between ``i`` and ``j``
            (for a lag above 0, with no edge ``i -> j``).
        lag : int or None, default=None
            The lag of a time-series result; ``None`` for any other result.

        Returns
        -------
        np.ndarray of shape (n, n)
            The shares, from 0 to 1. A lag above 0 holds only ``->``, ``<-``, and ``absent``; any
            other edge there is 0, and so is an edge that no fit has.

        Raises
        ------
        ValueError
            If ``edge`` is not one of the names above, ``lag`` is missing for a time-series result,
            or ``lag`` is not one of its lags or is given for another result.
        """
        if edge != _ABSENT and edge not in _GLYPH_MARKS:
            raise ValueError(f"edge must be one of {(*EDGES, _ABSENT)}, got {edge!r}")
        s = self._slices[self._slice_of(lag)]
        n, b = self._structure.n_nodes, self._n_resamples
        out = np.zeros((n, n))
        if s.kind == "digraph":
            src, dst = np.divmod(s.codes, n)
            directed = np.zeros((n, n), dtype=np.int64)
            directed[src, dst] = s.counts
            if edge == _ABSENT:
                return (b - directed) / b
            if edge == "->":
                return directed / b
            return directed.T / b if edge == "<-" else out
        i, j, near, far = _decode(s.codes, n)
        if edge == _ABSENT:
            present = np.zeros((n, n), dtype=np.int64)
            np.add.at(present, (i, j), s.counts)
            np.add.at(present, (j, i), s.counts * (i != j))
            return (b - present) / b
        want = _GLYPH_MARKS[edge]
        here = (near == want[0]) & (far == want[1])
        mirror = (near == want[1]) & (far == want[0])
        out[i[here], j[here]] = s.counts[here] / b
        out[j[mirror], i[mirror]] = s.counts[mirror] / b
        return out

    def table(self, *, min_frequency: float = 0.0, index: bool = False) -> list[dict[str, Any]]:
        """List every edge that appeared in a fit, most frequent first.

        Parameters
        ----------
        min_frequency : float, default=0.0
            Leave out edges seen in a smaller share of the fits.
        index : bool, default=False
            Name the nodes by index even when the result has labels.

        Returns
        -------
        list[dict]
            One row per edge, most frequent first, then by lag and node index. A row has
            ``source``, ``target``, ``type`` (one of ``oriented_edges`` types: a directed edge runs
            from its tail to its arrowhead, and the symmetric types list the lower node index
            first), ``count``, and ``frequency``; a time-series result adds ``lag`` first. A pair
            can have several rows, one per edge it had in some fit.
        """
        labels = None if index else self._structure.labels
        n, b = self._structure.n_nodes, self._n_resamples
        rows = []
        for k, s in enumerate(self._slices):
            lag = None if self._lags is None else self._lags[k]
            if s.kind == "digraph":
                pairs = [(*divmod(int(c), n), "directed") for c in s.codes]
            else:
                i, j, near, far = _decode(s.codes, n)
                pairs = []
                for a, c, m, f in zip(i.tolist(), j.tolist(), near.tolist(), far.tolist()):
                    kind, forward = ("directed", True) if a == c else _EDGE_TYPES[(m, f)]
                    pairs.append((a, c, kind) if forward else (c, a, kind))
            for (source, target, kind), count in zip(pairs, s.counts.tolist()):
                if count / b < min_frequency:
                    continue
                row: dict[str, Any] = {} if lag is None else {"lag": lag}
                names = (source, target) if labels is None else (labels[source], labels[target])
                row |= {
                    "source": names[0],
                    "target": names[1],
                    "type": kind,
                    "count": count,
                    "frequency": count / b,
                }
                rows.append((-count, k, source, target, kind, row))
        rows.sort(key=lambda r: r[:5])
        return [r[5] for r in rows]

    def __repr__(self) -> str:
        """Return the averaged graph and the most frequent edges."""
        return "\n".join(self._summary())

    def _summary(
        self, *, name: str | None = None, limit: int | None = 10, more: str = ""
    ) -> list[str]:
        """Build the result summary as lines for ``repr`` and ``andrey bootstrap``.

        ``name`` defaults to ``metadata["method"]``. ``limit`` caps the listed edges, and ``more``
        extends the omitted-edge count.
        """
        method = self._metadata.get("method")
        name = name if name is not None else (str(method) if method else "bootstrap")
        head = (
            f"{name}  |  {self._n_resamples} fits  |  averaged graph at threshold {self._threshold}"
        )
        lines = [head, "", *self._structure._summary()]
        rows = self.table()
        if rows:
            lines += ["", "  most frequent edges:"]
        shown = rows if limit is None else rows[:limit]
        for row in shown:
            lag = f"lag {row['lag']}  " if "lag" in row else ""
            edge = f"{row['source']} {EDGE_GLYPHS[row['type']]} {row['target']}"
            lines.append(f"    {lag}{edge}  {row['frequency']:.2f}")
        if len(shown) < len(rows):
            lines.append(f"    ... and {len(rows) - len(shown)} more{more}")
        return lines

    # ---- internals ------------------------------------------------------------------------

    def _slice_of(self, lag: int | None) -> int:
        """Return the index of the slice for ``lag``."""
        if self._lags is None:
            if lag is not None:
                raise ValueError(
                    f"lag is for a time-series result; this one has no lags, got {lag}"
                )
            return 0
        if lag is None or lag not in self._lags:
            raise ValueError(f"lag must be one of {self._lags}, got {lag!r}")
        return self._lags.index(lag)


# ---- counting -----------------------------------------------------------------------------------


def _average(
    slices: Sequence[_Slice],
    n_resamples: int,
    threshold: float,
    lags: tuple[int, ...] | None,
    n: int,
    labels: tuple[str, ...] | None,
    nearest: tuple[int, Structure] | None,
    strict: bool,
    constraints: Constraints | None,
    ranks: np.ndarray | None,
) -> tuple[Structure, str]:
    """Build the averaged graph from the slices' counts: see ``BootstrapOutput.from_fits``.

    Returns the graph and its ``metadata["averaged_graph"]``: ``"vote"`` or ``"nearest_resample"``.
    """
    if lags is not None:
        graphs = [_vote_graph(s, n_resamples, threshold, n, labels) for s in slices]
        return TemporalStructure.from_lag_graphs(graphs, lags=lags, labels=labels), "vote"
    s = slices[0]
    if s.kind not in _MADE_VALID:
        return _vote_graph(s, n_resamples, threshold, n, labels), "vote"
    try:
        return _valid_graph(
            _vote(s, n_resamples, threshold, n), s.kind, labels, constraints, ranks
        ), "vote"
    except ValueError as exc:
        reason = f"the vote of the fits has no valid {s.kind}: {exc}"
    if strict:
        raise ValueError(f"{reason}. strict=False returns the fit nearest the others instead")
    assert nearest is not None  # computed for every kind made valid
    index, graph = nearest
    warn_once(
        f"{reason}. The averaged graph is resample {index}, the fit nearest the others; "
        "strict=True raises instead",
        AndreyWarning,
    )
    return graph, "nearest_resample"


def _vote(s: _Slice, n_resamples: int, threshold: float, n: int) -> np.ndarray:
    """Return the marks of the vote of a slice that counts mark pairs.

    A pair takes its most frequent edge when more fits have it than have no edge and the pair is
    adjacent in more than ``threshold`` of the fits, or in all of them; a tie between edges takes
    the kind's tie edge.
    """
    marks = np.zeros((n, n), dtype=np.int8)
    pair, state = np.divmod(s.codes, _STATES)
    breaks = (np.flatnonzero(np.diff(pair)) + 1).tolist()
    for lo, hi in zip([0, *breaks], [*breaks, len(pair)]) if len(pair) else ():
        counts = s.counts[lo:hi]
        adjacent = int(counts.sum())
        if counts.max() <= n_resamples - adjacent or not _often(adjacent, n_resamples, threshold):
            continue
        top = np.flatnonzero(counts == counts.max())
        near, far = divmod(int(state[lo + top[0]]), 4) if len(top) == 1 else _TIE_EDGE[s.kind]
        i, j = divmod(int(pair[lo]), n)
        marks[i, j], marks[j, i] = near, far
    return marks


def _often(count: int, n_resamples: int, threshold: float) -> bool:
    """Whether ``count`` of the fits is more than ``threshold`` of them, or all of them."""
    return count > threshold * n_resamples or count == n_resamples


def _vote_graph(
    s: _Slice, n_resamples: int, threshold: float, n: int, labels: tuple[str, ...] | None
) -> GraphStructure:
    """Return the vote of a slice as a graph, with no step to the slice's kind.

    A ``dag`` vote that holds a tie edge ``--`` is a ``cpdag``.
    """
    if s.kind == "digraph":
        src, dst = np.divmod(s.codes, n)
        keep = np.array([_often(int(c), n_resamples, threshold) for c in s.counts], dtype=bool)
        edge = np.zeros((n, n), dtype=bool)
        edge[src[keep], dst[keep]] = True
        marks = np.zeros((n, n), dtype=np.int8)
        marks[edge] = TAIL
        marks[edge.T] = ARROW  # a 2-cycle and a self-loop end as arrowheads
        return GraphStructure.from_numpy(
            marks, kind="digraph", labels=labels, allow_self_loops=True
        )
    marks = _vote(s, n_resamples, threshold, n)
    kind = s.kind
    if kind == "dag" and ((marks == TAIL) & (marks.T == TAIL)).any():
        kind = "cpdag"
    return GraphStructure.from_numpy(marks, kind=kind, labels=labels)


def _valid_graph(
    vote: np.ndarray,
    kind: Kind,
    labels: tuple[str, ...] | None,
    constraints: Constraints | None,
    ranks: np.ndarray | None,
) -> GraphStructure:
    """Return the valid graph of ``kind`` that the published steps make of ``vote``.

    A ``dag``: its consistent extension (Dor and Tarsi), which takes the nodes of ``ranks`` in
    their rank order, so a tie edge points as the data's ranks decide. Without ``ranks``, a
    ``dag`` vote with a tie edge has no such order and takes the ``cpdag`` step. A
    ``cpdag``: the CPDAG (Chickering) of its consistent extension; every extension gives the same
    CPDAG, so no order is needed. A ``pag``: the PAG of the MAG Zhang's construction builds. Under
    ``constraints``, the edges the knowledge decides are directed first, a ``cpdag`` becomes its
    MPDAG, and a ``pag`` is oriented as ``fci`` orients one under knowledge: the vote and the PAG
    get the knowledge's endpoints (``_knowledge_marks``) before the MAG step and before the
    orientation rules, which then run to a fixed point.
    Raises ``ValueError``, saying why, when no extension or no MAG exists.
    """
    if kind == "pag":
        if constraints is not None:
            from andrey.constraint.fci import _knowledge_marks

            vote = vote.copy()
            _knowledge_marks(vote, constraints)  # a tie edge o-o can drop a knowledge endpoint
        mag = GraphStructure.from_numpy(mag_of_pag(vote), kind="mag", labels=labels)
        return pag_of(mag, None, (), constraints)
    if ((vote == ARROW) & (vote.T == ARROW)).any():
        raise ValueError("it has a bidirected edge, which no DAG has")
    if constraints is not None:
        vote = knowledge_orient(vote, constraints)
    if kind == "dag" and (ranks is not None or not ((vote == TAIL) & (vote.T == TAIL)).any()):
        order = np.arange(len(vote)) if ranks is None else np.argsort(ranks, kind="stable")
        return GraphStructure.from_numpy(extend_in_order(vote, order), kind="dag", labels=labels)
    cpdag = _cpdag_of_pdag(vote)
    if constraints is not None:
        cpdag = mpdag(cpdag, constraints)
    return GraphStructure.from_numpy(cpdag, kind="cpdag", labels=labels)


def _nearest(s: _Slice, coded: list[np.ndarray], n_resamples: int) -> int:
    """Return the fit nearest the others: the first with the highest agreement (Tetrad's Median).

    A fit's agreement sums, over the pairs, the number of fits that give the pair the fit's own
    state, no edge included; the highest agreement is the least mean SHD to the fits. ``coded``
    holds each fit's codes, as ``_codes`` makes them.
    """
    pair = s.codes // _STATES
    pairs, where = np.unique(pair, return_inverse=True)
    adjacent = np.bincount(where, weights=s.counts).astype(np.int64)
    # A pair a fit leaves out agrees with the fits that leave it out too; summed over every pair,
    # that is a constant less, for each pair the fit has, the fits that leave that pair out.
    agreement = [
        int(
            (
                s.counts[np.searchsorted(s.codes, codes)]
                - (n_resamples - adjacent[np.searchsorted(pairs, codes // _STATES)])
            ).sum()
        )
        for codes in coded
    ]
    return int(np.argmax(agreement))


def _graphs(s: Structure, lags: tuple[int, ...] | None) -> list[GraphStructure] | None:
    """Return the graphs to count in ``s``: itself, or its graph for each of ``lags``.

    ``None`` when ``s`` is a time-series result with other lags than ``lags``.
    """
    if isinstance(s, TemporalStructure):
        return [s.lag(k) for k in lags] if lags == s.lags else None
    assert isinstance(s, GraphStructure)
    return [s]


def _check_threshold(threshold: float) -> float:
    """Return ``threshold`` as a ``float`` in ``[0.5, 1]``; raise as the method wrapper does."""
    from andrey.spec.concepts import BOOTSTRAP_PARAMS

    checked = BOOTSTRAP_PARAMS["threshold"].check(threshold, "bootstrap")
    assert isinstance(checked, float)
    return checked


def _check_strict(strict: bool) -> bool:
    """Return ``strict``; raise ``TypeError`` when it is not a ``bool``."""
    from andrey.spec.concepts import BOOTSTRAP_PARAMS

    checked = BOOTSTRAP_PARAMS["strict"].check(strict, "bootstrap")
    assert isinstance(checked, bool)
    return checked


def _check_resampling(resampling: str, sample_fraction: float) -> tuple[str, float]:
    """Return the checked ``(resampling, sample_fraction)``; raise as the method wrapper does."""
    from andrey.spec.concepts import BOOTSTRAP_PARAMS

    checked = BOOTSTRAP_PARAMS["resampling"].check(resampling, "bootstrap")
    fraction = BOOTSTRAP_PARAMS["sample_fraction"].check(sample_fraction, "bootstrap")
    assert isinstance(checked, str)
    assert isinstance(fraction, float)
    return checked, fraction


def _widest(kinds: set[Kind]) -> Kind:
    """Return the widest of ``kinds``: ``dag`` < ``cpdag`` < ``pag``, or a lone kind.

    ``digraph`` and ``undirected`` average only with fits of their own kind. A ``mag`` counts as
    a ``pag``: an averaged MAG can hold a tie, which leaves a circle.
    """
    kinds = {"pag" if is_ancestral_kind(kind) else kind for kind in kinds}
    for alone in ("digraph", "undirected"):
        if alone in kinds:
            if len(kinds) > 1:
                raise ValueError(
                    f"a graph is a {alone} in some fits and another kind in others: {kinds}"
                )
            return alone
    return max(kinds, key=_KIND_ORDER.__getitem__)


def _codes(graph: GraphStructure) -> np.ndarray:
    """Code each edge of ``graph`` as ``(i * n + j) * 16 + near * 4 + far``, for ``i <= j``."""
    i, j, near, far = graph._mark_pairs()
    return (i.astype(np.int64) * graph.n_nodes + j) * _STATES + near * 4 + far


def _directed_codes(graph: GraphStructure) -> np.ndarray:
    """Code each directed edge ``i -> j`` of a ``digraph`` as ``i * n + j``, a 2-cycle as two."""
    i, j, near, far = (a.astype(np.int64) for a in graph._mark_pairs())
    forward = ((near == TAIL) & (far == ARROW)) | ((near == ARROW) & (far == ARROW))
    backward = ((near == ARROW) & (far == TAIL)) | ((near == ARROW) & (far == ARROW) & (i != j))
    n = graph.n_nodes
    return np.concatenate([i[forward] * n + j[forward], j[backward] * n + i[backward]])


def _decode(codes: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(i, j, near, far)`` for mark-pair ``codes``."""
    pair, state = np.divmod(codes, _STATES)
    i, j = np.divmod(pair, n)
    return i, j, state // 4, state % 4


# ---- resampling and fitting ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Job:
    """Everything a resample needs, sent once to each worker."""

    name: str  # the facade, by name
    data: Any  # the first input: an array or table, or a list of them for a panel
    aligned: dict[str, Any]  # row-aligned inputs by keyword (cdnod's domain_index)
    kwargs: dict[str, Any]
    seed: int
    unit: str  # "rows", "blocks", "units" (one draw for every array), or "groups" (one each)
    block_length: int | None
    seeded: bool  # the method takes a seed, derived per resample
    backend: str
    tie_break: str
    resampling: str  # "subsample" or "bootstrap", resolved
    sample_fraction: float  # a subsample's share of the rows


def _take(x: Any, idx: np.ndarray) -> Any:
    """Return the rows ``idx`` of a table or array, keeping a table's columns and dtypes."""
    if hasattr(x, "iloc"):
        return x.iloc[idx].reset_index(drop=True)
    return np.asarray(x)[idx]


def _kept(n: int, sample_fraction: float) -> int:
    """Return the number of rows a subsample of ``n`` rows keeps."""
    return round(sample_fraction * n)


def _draw(job: _Job, b: int) -> list[np.ndarray]:
    """Draw resample ``b``'s row indices, one array per data array."""
    rng = np.random.default_rng(np.random.SeedSequence(job.seed, spawn_key=(b, 0)))
    sizes = [len(x) for x in job.data] if job.unit in ("units", "groups") else [len(job.data)]
    if job.resampling == "subsample":
        kept = [_kept(n, job.sample_fraction) for n in sizes]
        if job.unit == "blocks":  # one stretch of consecutive rows, as in Politis et al. (1999)
            start = int(rng.integers(0, sizes[0] - kept[0] + 1))
            return [np.arange(start, start + kept[0])]
        if job.unit == "units":
            return [np.sort(rng.choice(sizes[0], size=kept[0], replace=False))] * len(sizes)
        return [np.sort(rng.choice(n, size=m, replace=False)) for n, m in zip(sizes, kept)]
    if job.unit == "blocks":
        n, length = sizes[0], job.block_length
        assert length is not None
        starts = rng.integers(0, n - length + 1, size=math.ceil(n / length))
        return [(starts[:, None] + np.arange(length)).ravel()[:n]]
    if job.unit == "units":
        return [rng.integers(0, sizes[0], size=sizes[0])] * len(sizes)
    return [rng.integers(0, n, size=n) for n in sizes]


def _run_one(job: _Job, b: int) -> tuple[Any, list[tuple[type[Warning], str]]]:
    """Fit resample ``b``; return the fit and the warnings it raised."""
    idx = _draw(job, b)
    if job.unit in ("units", "groups"):
        data = [_take(x, ix) for x, ix in zip(job.data, idx)]
    else:
        data = _take(job.data, idx[0])
    aligned = {key: _take(x, idx[0]) for key, x in job.aligned.items()}
    kwargs = dict(job.kwargs)
    if job.seeded:
        sequence = np.random.SeedSequence(job.seed, spawn_key=(b, 1))
        kwargs["seed"] = int(sequence.generate_state(1)[0])
    fit = getattr(api, job.name)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = fit(data, **aligned, **kwargs)
    except Exception as exc:
        exc.add_note(
            f"raised in bootstrap resample {b} of seed {job.seed}; its rows are the draw's"
        )
        raise
    return out, [(w.category, str(w.message)) for w in caught]


_JOB: _Job | None = None
_BLAS: object = None  # a held threadpoolctl controller capping this worker's BLAS to one thread


def _init_worker(job: _Job) -> None:
    """Pin a worker to one fit at a time on one BLAS thread, with the parent's settings."""
    global _JOB, _BLAS
    from threadpoolctl import threadpool_limits

    _BLAS = threadpool_limits(limits=1)  # held for the worker's lifetime
    _backend.config.n_jobs = 1
    _backend.config.backend = job.backend
    _backend.config.tie_break = job.tie_break
    _JOB = job


def _task(b: int) -> tuple[Any, list[tuple[type[Warning], str]]]:
    """Worker side of one resample."""
    assert _JOB is not None  # _init_worker runs before any task
    return _run_one(_JOB, b)


def _fit_all(job: _Job, n_resamples: int, workers: int) -> list[tuple[Any, list]]:
    """Fit every resample, in order, in this process or on ``workers`` worker processes."""
    if workers <= 1:
        with _backend.one_blas_thread(), _backend.config(n_jobs=1):
            return [_run_one(job, b) for b in range(n_resamples)]
    pool = ProcessPoolExecutor(
        max_workers=workers, mp_context=pick_context(), initializer=_init_worker, initargs=(job,)
    )
    try:
        futures = [pool.submit(_task, b) for b in range(n_resamples)]
        return [future.result() for future in futures]
    except BrokenProcessPool as exc:
        raise RuntimeError(
            "a bootstrap worker died, most likely out of memory; rerun with andrey.config(n_jobs=1)"
        ) from exc
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


# ---- the public function ------------------------------------------------------------------------


_N_RESAMPLES_DEFAULT = 100


def bootstrap(
    method: str | Callable[..., Any],
    data: npt.ArrayLike | Sequence[npt.ArrayLike],
    *,
    n_resamples: int = _N_RESAMPLES_DEFAULT,
    seed: int | None = None,
    threshold: float = 0.5,
    strict: bool = False,
    resampling: str = "subsample",
    sample_fraction: float = 0.5,
    block_length: int | None = None,
    keep_fits: bool = False,
    **method_kwargs: Any,
) -> BootstrapOutput | list[BootstrapOutput]:
    """Refit a method on resamples of the rows; return each edge's frequency and an averaged graph.

    Each resample is a subsample, ``sample_fraction`` of the rows of ``data`` drawn without
    replacement, or a bootstrap resample, as many rows drawn with replacement (``resampling``), and
    ``method`` is fitted on it. The result counts, for every pair of variables, the fits that give
    each endpoint-mark pair (``->``, ``<-``, ``--``, ``<->``, ``o->``, ...) and builds the averaged
    graph: the vote of the fits, made a valid graph of the method's kind [1]_ [2]_ [3]_. A method
    with a row-aligned input
    (``cdnod``'s ``domain_index``, ``gies``'s ``interventions``) resamples that input with the
    rows. A time-series method on one series subsamples one stretch of consecutive rows (and
    bootstraps blocks of consecutive rows), ``longitudinal_lingam`` resamples whole units, and
    ``multi_group_direct_lingam`` resamples each dataset on its own.

    Parameters
    ----------
    method : str or callable
        A method's name, such as ``"pc"``, or the callable itself, such as ``andrey.pc``.
    data : array-like of shape (n_samples, n_variables), or a sequence of them
        The data, as the method takes it. ``multi_group_direct_lingam`` and
        ``longitudinal_lingam`` take a sequence of arrays.
    n_resamples : int, default=100
        The number of resamples and fits. ``n_boot``, the former name, works with a
        ``DeprecatedArgumentWarning`` for one more release.
    seed : int or None, default=None
        The seed of the resamples, a non-negative integer. ``None`` uses ``ANDREY_SEED``, or 0 when
        it is unset. Resample ``b`` depends on ``seed`` and ``b`` alone. A method that takes a
        ``seed`` fits resample ``b`` with a seed derived from this one and ``b``, so the method's
        own ``seed`` cannot be set here.
    threshold : float, default=0.5
        A pair of the averaged graph must be adjacent in more than this share of the fits, or in
        all of them, in ``[0.5, 1]``. Its most frequent edge must also be in more fits than no edge
        is, which needs more than half of them, so a value below 0.5 would change nothing.
    strict : bool, default=False
        What happens when the vote of the fits has no valid graph of the method's kind: ``False``
        returns the resample graph nearest the others with a warning, and ``True`` raises
        ``ValueError``.
    resampling : {"subsample", "bootstrap"}, default="subsample"
        How a resample is drawn. ``"subsample"`` draws ``round(sample_fraction * n_samples)``
        distinct rows, kept in their original order; for a time-series method on one series it is
        one stretch of consecutive rows at a random start [4]_ [5]_. ``"bootstrap"`` draws
        ``n_samples`` rows with replacement; for a time-series method on one series it draws blocks
        of ``block_length`` consecutive rows (a moving block bootstrap).
    sample_fraction : float, default=0.5
        The share of the rows in a subsample, in ``(0, 1]``, so that ``round(sample_fraction *
        n_samples)`` is at least 2; for ``longitudinal_lingam``, the share of the units, and for
        ``multi_group_direct_lingam``, the share of each dataset's rows. Not used by
        ``"bootstrap"``.
    block_length : int or None, default=None
        For ``resampling="bootstrap"`` of a time-series method on one series, the number of
        consecutive rows in a block, from 1 to ``n_samples``. ``None`` uses
        ``max(ceil(n_samples ** (1/3)), 10 * tau_max)`` for a method with a ``tau_max``, and
        ``ceil(n_samples ** (1/3))`` for the others, at most ``n_samples``. A block break puts rows
        that were not neighbors side by side, which weakens a link that many steps long or longer,
        so a block should be at least ten times the longest lag. Not for any other resampling.
    keep_fits : bool, default=False
        Keep every fit in the ``fits`` property.
    **method_kwargs
        The method's other arguments: its keyword parameters, and a row-aligned input such as
        ``domain_index=``.

    Returns
    -------
    BootstrapOutput or list[BootstrapOutput]
        The frequencies and the averaged graph; for ``multi_group_direct_lingam``, one per group.
        ``metadata`` records the ``resampling``, the ``sample_fraction`` of a subsample, the
        ``block_length`` of a block bootstrap, the ``tie_break`` rule in effect
        (:data:`andrey.config`), and ``averaged_graph``: ``"vote"``, or ``"nearest_resample"``
        when the averaged graph is the resample graph nearest the others.

    Raises
    ------
    ValueError
        If ``method`` is not a method of Andrey, an argument is out of range, a row-aligned input
        has a different number of rows from ``data``, ``block_length`` is given for a resampling
        that does not use blocks, ``strict=True`` and the vote has no valid graph of the method's
        kind, or a fit raises it: the error then names the resample in a note.
    NotImplementedError
        For ``gin``, whose number of latent nodes varies from fit to fit.
    RuntimeError
        If a worker process dies.

    Warns
    -----
    AndreyWarning
        Each distinct warning the fits raised, once; and, with ``strict=False``, when the vote has
        no valid graph of the method's kind, so the averaged graph is the resample graph nearest the
        others.

    Notes
    -----
    **Subsampling and the bootstrap.** A bootstrap resample repeats some rows and leaves out others,
    so a conditional-independence test or a score sees more evidence than the resample holds. A
    method built on such tests, such as ``pc``, keeps weak and false edges in more resamples than it
    does on ``data``, and an orientation that needs two variables to stay non-adjacent, a collider,
    is found in fewer. A subsample holds distinct rows only, so it has no such excess. Banerjee,
    Andrews, and Kummerfeld [3]_ compared both schemes with two score searches on linear Gaussian
    data: no scheme was best everywhere, the bootstrap at the full size performed poorly, and
    subsampling half of the rows performed about as well as a bootstrap scored at the effective
    sample size, about half the size. The default draws half of the rows, as stability selection
    does [4]_, and it changes only the draw, so it means the same for every method.

    On linear data with 20 and 50 variables and 1,000 and 5,000 rows (``pc``, ``ges``, and
    ``direct_lingam``; 10 datasets per cell), the averaged graph of half-size subsamples had no more
    errors (SHD) than the bootstrap's in any of the 10 cells, and fewer in 5; for ``ges`` with 50
    variables and 1,000 rows, 5.0 fewer on average (95% interval 2.7 to 7.9). For ``ges`` with 20
    variables and 1,000 rows, the averaged graph had 18 wrong edge types instead of 32, and 9 of
    the 10 averaged graphs were valid CPDAGs instead of 3. The adjacency frequencies had a lower
    calibration error in 9 of the 10 cells, and a fit took 0.56 to 0.95 of the time, since it has
    half the rows. The one cell where the point estimate was worse, ``pc`` with 20 variables and
    1,000 rows, was 0.43 errors above the bootstrap's over 30 datasets (interval -0.23 to 1.13). A
    larger ``sample_fraction`` gives fits closer to the fit on ``data``, so the resamples agree
    more and the frequencies say less about how much the fit varies.

    For a time-series method on one series, a subsample is one stretch of consecutive rows, which
    keeps the lags inside it [5]_. With ``pcmci``, ``pcmciplus``, ``lpcmci``, and ``varma_lingam``
    on simulated series with 5 and 10 variables and 500 and 1,000 rows (10 series per cell, 14
    cells; 30 series per cell, 12 cells, for the first three), a stretch of half of the series had
    no more errors (SHD) than the block bootstrap in any cell, and fewer, by an interval below 0, in
    9 of the 14 cells of 10 series. Those blocks were ``ceil(n_samples ** (1/3))`` rows, shorter
    than ten times the longest lag.

    **The averaged graph.** Each pair of variables takes its most frequent state in the fits, no
    edge counted as one of the states: the pair gets its most frequent edge when more fits have
    that edge than have no edge, which needs the pair adjacent in more than half of them, and
    ``threshold`` can ask for more. When two edges tie, the pair takes the least committal one,
    ``--``, or ``o-o`` for a ``pag``. This vote is the graph with the least mean structural Hamming
    distance to the fits. It is the ``Highest`` ensemble of Tetrad [6]_, with a tie between an edge
    and no edge going to no edge, as in tigramite's majority vote [7]_.

    A vote of CPDAGs or PAGs is often not a valid CPDAG or PAG: it can direct an edge that no
    v-structure compels. So the vote is then made a valid graph of the method's kind by the
    published steps. A ``dag`` becomes a consistent extension, a DAG with the vote's skeleton,
    v-structures, and directed edges [8]_: the vote itself when it has no directed cycle, with a
    tie edge pointing as the columns' :func:`~andrey.search.hc.tie_ranks` on ``data`` decide
    (``tie_break``, :data:`andrey.config`: ranks computed from each column's own data, or the
    column order under ``"position"``), so reordering the columns relabels the graph. A ``cpdag``
    becomes the CPDAG [9]_ of a consistent extension, so an edge that no v-structure compels is
    undirected. A ``pag`` becomes
    the PAG of the MAG that Zhang's construction builds from it [10]_. With ``knowledge``, the
    edges it directs stay directed: a ``cpdag`` becomes its maximally oriented PDAG, and a ``pag``
    is oriented as ``fci`` orients one under knowledge: the endpoints the knowledge sets come
    first, and Zhang's orientation rules then run from them until no endpoint changes, as they do
    in a fit [10]_, [11]_. A time-series result's lag graphs
    keep the vote, as tigramite's does [7]_: a lag-0 graph is oriented with the lagged edges, which
    the steps for one graph do not see.

    The vote has no consistent extension or MAG when, for example, its directed edges form a
    cycle or its undirected edges a chordless cycle of four or more nodes. With ``strict=False``,
    the averaged graph is then the resample graph nearest the others, a graph the method returned:
    the fit whose state of each pair the most fits share, summed over the pairs, which is the fit
    with the least mean structural Hamming distance to the others (Tetrad's ``Median`` [6]_). A
    warning says so, and ``metadata["averaged_graph"]`` is ``"nearest_resample"``; it is
    ``"vote"`` otherwise. ``strict=True`` raises instead, as bnlearn's ``cextend`` does by default
    [12]_. Either way, ``frequency`` and ``table`` keep the whole vote.

    On linear data with 20 and 50 variables and 500, 1,000, and 5,000 rows (``pc``, ``ges``,
    ``fci``, and ``direct_lingam``; 20 datasets per cell; 100 resamples with
    ``resampling="bootstrap"``), the averaged graph had fewer errors (SHD) than the rule of 0.11.0
    (a pair adjacent in at least half the fits takes its most frequent edge) in 11 of the 24
    cells, by an interval below 0, and more in none. It had fewer errors than the fit on all the
    rows in 14 cells and more in one, ``ges`` with 20 variables and 1,000 rows. The vote had a
    valid graph of the method's kind in 111 of the 120 datasets of ``pc``, 114 of ``fci``, and all
    of ``ges`` and ``direct_lingam``; the rule of 0.11.0 gave a valid CPDAG or PAG in 0 to 13 of
    the 20 datasets of a cell.

    A frequency is a ranking of the edges, not a probability, and no error bound comes with the
    averaged graph. Resampling can also make a column constant; a fit that raises stops the run,
    and its row numbers count the rows of the resample.

    **Workers.** The fits run on the worker processes ``andrey.config.n_jobs`` and
    ``ANDREY_N_JOBS`` set, as for every method: ``-1`` uses every usable core, and ``0`` or
    ``1`` fits in this process. Each fit runs on one worker and one BLAS thread, and the result is
    the same at any worker count.

    **Limitations.** Not benchmarked yet. Refitting is the cost: ``n_resamples`` times the method's
    own, shared across the workers. A moving block bootstrap weakens a link that spans a block
    break, and a stretch holds only ``sample_fraction`` of the series, so each understates what a
    very short series can show.

    References
    ----------
    .. [1] Efron, B. (1979). Bootstrap methods: another look at the jackknife. Annals of Statistics,
       7(1), 1-26.
    .. [2] Friedman, N., Goldszmidt, M., and Wyner, A. (1999). Data analysis with Bayesian
       networks: a bootstrap approach. Proceedings of the Fifteenth Conference on Uncertainty in
       Artificial Intelligence, 196-205.
    .. [3] Banerjee, R., Andrews, B., and Kummerfeld, E. (2025). An extensive simulation study
       evaluating the interaction of resampling techniques across multiple causal discovery
       contexts. arXiv:2503.15436.
    .. [4] Meinshausen, N., and Buhlmann, P. (2010). Stability selection. Journal of the Royal
       Statistical Society: Series B, 72(4), 417-473.
    .. [5] Politis, D. N., Romano, J. P., and Wolf, M. (1999). Subsampling. Springer.
    .. [6] Tetrad, ``edu.pitt.dbmi.algo.resampling.ResamplingEdgeEnsemble`` and
       ``edu.cmu.tetrad.util.GraphSampling``, github.com/cmu-phil/tetrad, development branch,
       2026-08-13. ``Median`` is not in release 7.6.10.
    .. [7] Debeire, K., Gerhardus, A., Runge, J., and Eyring, V. (2024). Bootstrap aggregation and
       confidence measures to improve time series causal discovery. Proceedings of the Third
       Conference on Causal Learning and Reasoning, PMLR 236.
    .. [8] Dor, D., and Tarsi, M. (1992). A simple algorithm to construct a consistent extension
       of a partially oriented graph. Technical Report R-185, Cognitive Systems Laboratory, UCLA.
    .. [9] Chickering, D. M. (2002). Learning equivalence classes of Bayesian-network structures.
       Journal of Machine Learning Research, 2, 445-498.
    .. [10] Zhang, J. (2008). On the completeness of orientation rules for causal discovery in the
       presence of latent confounders and selection bias. Artificial Intelligence, 172(16-17),
       1873-1896.
    .. [11] Andrews, B., Spirtes, P., and Cooper, G. F. (2020). On the completeness of causal
       discovery in the presence of latent confounding with tiered background knowledge.
       Proceedings of the 23rd International Conference on Artificial Intelligence and Statistics,
       PMLR 108, 4002-4011.
    .. [12] Scutari, M. (2010). Learning Bayesian networks with the bnlearn R package. Journal of
       Statistical Software, 35(3), 1-22.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import bootstrap
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal(300)
    >>> data = np.column_stack([x, x + rng.standard_normal(300), rng.standard_normal(300)])
    >>> out = bootstrap("pc", data, n_resamples=10, seed=0)
    >>> out.n_resamples
    10
    >>> bool(out.frequency("absent")[0, 1] < 0.5)
    True
    >>> out.metadata["resampling"], out.metadata["sample_fraction"]
    ('subsample', 0.5)
    """
    from andrey.core.seeding import resolve_seed

    if "n_boot" in method_kwargs:
        if n_resamples != _N_RESAMPLES_DEFAULT:
            raise ValueError(
                "bootstrap: n_boot is the former name of n_resamples; pass only n_resamples"
            )
        n_resamples = method_kwargs.pop("n_boot")
        warn_once(
            "andrey.bootstrap: n_boot is now n_resamples; the former name stops working in the "
            "next release",
            DeprecatedArgumentWarning,
        )
    spec = _method_spec(method)
    n_resamples = int_in_range(n_resamples, "n_resamples", 1)
    threshold = _check_threshold(threshold)
    strict = _check_strict(strict)
    resampling, sample_fraction = _check_resampling(resampling, sample_fraction)
    keep_fits = flag(keep_fits, "keep_fits")
    if spec.output.has_latents:
        raise NotImplementedError(
            f"bootstrap does not support {spec.name}: its latent nodes differ from fit to fit"
        )
    root = resolve_seed(seed)
    if root < 0:
        raise ValueError(f"seed must be a non-negative integer, got {root}")
    job = _job(spec, data, method_kwargs, block_length, root, resampling, sample_fraction)
    workers = min(_backend.worker_count(None), n_resamples)
    results = _fit_all(job, n_resamples, workers)
    ranks = _column_ranks(job.data, job.unit == "groups")
    for category, message in dict.fromkeys(w for _, caught in results for w in caught):
        warnings.warn(message, category, stacklevel=2)
    metadata: dict[str, JSONValue] = {
        "method": spec.name,
        "seed": root,
        "tie_break": job.tie_break,
        "resampling": job.resampling,
        "sample_fraction": job.sample_fraction if job.resampling == "subsample" else None,
        "block_length": job.block_length,
    }
    fits = [out for out, _ in results]
    groups = list(zip(*fits)) if spec.output.multi else [fits]
    outs = [
        BootstrapOutput._count(
            group,
            threshold,
            strict,
            keep_fits,
            metadata,
            _constraints(spec, method_kwargs.get("knowledge"), group[0].structure),
            ranks,
        )
        for group in groups
    ]
    return outs if spec.output.multi else outs[0]


def _column_ranks(data: Any, groups: bool) -> np.ndarray:
    """Return the columns' :func:`~andrey.search.hc.tie_ranks` on all of ``data``.

    ``data`` is a table or an array, or, for ``groups``, a list of them stacked as the methods
    over groups stack them. A column that does not hold numbers is numbered by its sorted values,
    as the fits read its categories; its pattern is then a function of the column alone.
    """
    stacked = np.vstack([_numbers(table) for table in (data if groups else [data])])
    return tie_ranks(stacked)


def _numbers(table: Any) -> np.ndarray:
    """Return ``table`` as a float matrix, numbering each column that does not hold numbers."""
    columns = []
    for column in np.asarray(table).T:
        try:
            columns.append(column.astype(np.float64))
        except (TypeError, ValueError):
            columns.append(np.unique(column.astype(str), return_inverse=True)[1].astype(np.float64))
    return np.column_stack(columns)


def _constraints(spec: MethodSpec, knowledge: Any, structure: Structure) -> Constraints | None:
    """Return the constraints ``knowledge`` sets on a static result's nodes, or ``None``.

    A time-series result's averaged graph takes no step that knowledge could change, so it gets
    ``None``. The fits ran with ``knowledge``, so it resolves.
    """
    from andrey.core.knowledge import resolve

    if knowledge is None or isinstance(structure, TemporalStructure):
        return None
    return resolve(knowledge, structure.n_nodes, structure.labels, method=spec.name)


def _method_spec(method: str | Callable[..., Any]) -> MethodSpec:
    """Return the ``MethodSpec`` of a method given by name or as its facade callable."""
    from andrey.spec import REGISTRY, get_spec

    name = method if isinstance(method, str) else getattr(method, "__name__", None)
    if name not in REGISTRY or (not isinstance(method, str) and getattr(api, name) is not method):
        raise ValueError(
            f"method must be one of {sorted(REGISTRY)} or such a callable, got {method!r}"
        )
    return get_spec(name)


def _default_block_length(spec: MethodSpec, kwargs: dict[str, Any], n: int) -> int:
    """Return ``max(ceil(n ** (1/3)), 10 * tau_max)``, at most ``n``.

    The lag is the call's ``tau_max``, else the method's default. A method without a ``tau_max``
    has the first term only.
    """
    length = math.ceil(n ** (1 / 3))
    lag = next((p for p in spec.params if p.concept == "tau_max"), None)
    if lag is not None:
        given = kwargs.get(lag.name, lag.default)
        if isinstance(given, (int, np.integer)) and not isinstance(given, bool):
            length = max(length, 10 * int(given))
    return min(length, n)


def _job(
    spec: MethodSpec,
    data: Any,
    kwargs: dict[str, Any],
    block_length: int | None,
    seed: int,
    resampling: str,
    sample_fraction: float,
) -> _Job:
    """Check the call against ``spec`` and bundle it for the workers."""
    kwargs = dict(kwargs)
    panel = spec.data.inputs[0].kind == "panel"
    temporal = spec.family == "temporal"
    if panel:
        data = list(data)
        sizes = [len(x) for x in data]
        if temporal and len(set(sizes)) > 1:
            raise ValueError(f"every dataset needs the same units, got {sizes} rows")
        unit = "units" if temporal else "groups"
    else:
        sizes = [len(data)]
        unit = "blocks" if temporal else "rows"
    if resampling == "subsample":
        too_few = [n for n in sizes if _kept(n, sample_fraction) < 2]
        if too_few:
            raise ValueError(
                f"sample_fraction {sample_fraction} of {too_few[0]} rows is "
                f"{_kept(too_few[0], sample_fraction)}; a subsample needs at least 2 rows"
            )
    if unit != "blocks" and block_length is not None:
        raise ValueError(f"block_length is for a time-series method on one series, not {spec.name}")
    if unit == "blocks" and resampling == "bootstrap":
        n = sizes[0]
        length = _default_block_length(spec, kwargs, n) if block_length is None else block_length
        block_length = int_in_range(length, "block_length", 1, n)
    elif block_length is not None:
        raise ValueError("block_length is for resampling='bootstrap', not resampling='subsample'")
    aligned = {}
    for entry in spec.data.inputs[1:]:
        value = kwargs.pop(entry.name, None)
        if value is None:
            if entry.required:
                raise ValueError(f"{spec.name} needs {entry.name}")
            continue
        if len(value) != sizes[0]:
            raise ValueError(f"{entry.name} has {len(value)} rows but data has {sizes[0]}")
        aligned[entry.name] = value
    return _Job(
        name=spec.name,
        data=data,
        aligned=aligned,
        kwargs=kwargs,
        seed=seed,
        unit=unit,
        block_length=block_length,
        seeded=spec.seeded,
        backend=_backend.config.backend,
        tie_break=_backend.config.tie_break,
        resampling=resampling,
        sample_fraction=sample_fraction,
    )
