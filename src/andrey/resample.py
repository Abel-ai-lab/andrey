"""Bootstrap refits of a method: edge frequencies by endpoint mark, and the averaged graph.

:func:`bootstrap` refits one method on resamples of the rows and counts, for every pair of
variables, how often each endpoint-mark pair appears. :class:`BootstrapOutput` holds those counts
and the graph that takes each pair's most frequent edge.

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
from andrey.core.structure import (
    _EDGE_TYPES,
    EDGE_GLYPHS,
    Kind,
    Structure,
    TemporalStructure,
    is_ancestral_kind,
)
from andrey.search._parallel import pick_context

if TYPE_CHECKING:
    import numpy.typing as npt

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
}
_STATES = 16  # a pair's mark pairs, coded near * 4 + far


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
    _n_boot: int
    _threshold: float
    _lags: tuple[int, ...] | None
    _slices: tuple[_Slice, ...]
    _metadata: Mapping[str, JSONValue]
    _fits: tuple[StructureOutput, ...] | None = None

    # ---- construction ---------------------------------------------------------------------

    @classmethod
    def from_fits(
        cls,
        fits: Sequence[StructureOutput],
        *,
        threshold: float = 0.5,
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
            The share of fits in which a pair must be adjacent for the averaged graph to hold it,
            in ``(0, 1]``.
        keep_fits : bool, default=False
            Keep ``fits`` in the ``fits`` property.
        metadata : Mapping[str, JSONValue] or None, default=None
            Run values by name, with the types ``StructureOutput`` accepts.

        Returns
        -------
        BootstrapOutput
            The counts and the averaged graph. A pair is adjacent in the averaged graph when it is
            adjacent in at least ``threshold`` of the fits; it then takes the edge it has most
            often, or, when two edges tie for most often, the least committal one: ``--``, or
            ``o-o`` for a ``pag``. A ``digraph`` lag graph takes each directed edge that appears
            in at least ``threshold`` of the fits. The graph's kind is the widest kind among the
            fits (``dag``, ``cpdag``, ``pag``, with a ``mag`` read as a ``pag``), widened to
            ``cpdag`` by a ``--``.

        Raises
        ------
        TypeError
            If a fit is not a ``StructureOutput``.
        ValueError
            If ``fits`` is empty, ``threshold`` is not in ``(0, 1]``, the fits differ in graph
            type, node count, labels, or lags, a graph is a ``digraph`` in some fits and another
            kind in others, or a fit has latent nodes.
        """
        fits = tuple(fits)
        if not fits:
            raise ValueError("fits is empty; at least one fit is needed")
        for fit in fits:
            if not isinstance(fit, StructureOutput):
                raise TypeError(f"fits must hold StructureOutput results, got {type(fit).__name__}")
        threshold = _check_threshold(threshold)
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
        return cls(
            _structure=_average(slices, len(fits), threshold, lags, n, labels),
            _n_boot=len(fits),
            _threshold=threshold,
            _lags=lags,
            _slices=tuple(slices),
            _metadata=dict(metadata) if metadata is not None else {},
            _fits=fits if keep_fits else None,
        )

    # ---- surface --------------------------------------------------------------------------

    @property
    def structure(self) -> Structure:
        """The averaged graph at ``threshold``: a ``GraphStructure``, or a ``TemporalStructure``.

        A time-series result has one graph per lag and no time axis or weights.
        """
        return self._structure

    @property
    def n_boot(self) -> int:
        """The number of fits counted."""
        return self._n_boot

    @property
    def threshold(self) -> float:
        """The share of fits in which a pair must be adjacent to be in ``structure``."""
        return self._threshold

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
            The share of fits in which a pair must be adjacent, in ``(0, 1]``.

        Returns
        -------
        Structure
            The graph ``from_fits`` describes, at this threshold.

        Raises
        ------
        ValueError
            If ``threshold`` is not in ``(0, 1]``.
        """
        structure = self._structure
        return _average(
            self._slices,
            self._n_boot,
            _check_threshold(threshold),
            self._lags,
            structure.n_nodes,
            structure.labels,
        )

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
        n, b = self._structure.n_nodes, self._n_boot
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
        n, b = self._structure.n_nodes, self._n_boot
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
        head = f"{name}  |  {self._n_boot} fits  |  averaged graph at threshold {self._threshold}"
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
    n_boot: int,
    threshold: float,
    lags: tuple[int, ...] | None,
    n: int,
    labels: tuple[str, ...] | None,
) -> Structure:
    """Build the averaged graph from the slices' counts: see ``BootstrapOutput.from_fits``."""
    graphs = [_average_slice(s, n_boot, threshold, n, labels) for s in slices]
    if lags is None:
        return graphs[0]
    return TemporalStructure.from_lag_graphs(graphs, lags=lags, labels=labels)


def _average_slice(
    s: _Slice, n_boot: int, threshold: float, n: int, labels: tuple[str, ...] | None
) -> GraphStructure:
    """Build one graph from one slice's counts."""
    marks = np.zeros((n, n), dtype=np.int8)
    if s.kind == "digraph":
        src, dst = np.divmod(s.codes, n)
        keep = s.counts / n_boot >= threshold
        edge = np.zeros((n, n), dtype=bool)
        edge[src[keep], dst[keep]] = True
        marks[edge] = TAIL
        marks[edge.T] = ARROW  # a 2-cycle and a self-loop end as arrowheads
        return GraphStructure.from_numpy(
            marks, kind="digraph", labels=labels, allow_self_loops=True
        )
    pair, state = np.divmod(s.codes, _STATES)
    breaks = (np.flatnonzero(np.diff(pair)) + 1).tolist()
    kind = s.kind
    for lo, hi in zip([0, *breaks], [*breaks, len(pair)]):
        counts = s.counts[lo:hi]
        if counts.sum() / n_boot < threshold:
            continue
        top = np.flatnonzero(counts == counts.max())
        near, far = divmod(int(state[lo + top[0]]), 4) if len(top) == 1 else _TIE_EDGE[kind]
        i, j = divmod(int(pair[lo]), n)
        marks[i, j], marks[j, i] = near, far
        if (near, far) == (TAIL, TAIL) and kind == "dag":
            kind = "cpdag"
    return GraphStructure.from_numpy(marks, kind=kind, labels=labels)


def _graphs(s: Structure, lags: tuple[int, ...] | None) -> list[GraphStructure] | None:
    """Return the graphs to count in ``s``: itself, or its graph for each of ``lags``.

    ``None`` when ``s`` is a time-series result with other lags than ``lags``.
    """
    if isinstance(s, TemporalStructure):
        return [s.lag(k) for k in lags] if lags == s.lags else None
    assert isinstance(s, GraphStructure)
    return [s]


def _check_threshold(threshold: float) -> float:
    """Return ``threshold`` as a ``float`` in ``(0, 1]``; raise ``ValueError`` otherwise."""
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float, np.number)):
        raise TypeError(f"threshold must be a number, got {type(threshold).__name__}")
    if not 0 < threshold <= 1:
        raise ValueError(f"threshold must lie in (0, 1], got {threshold}")
    return float(threshold)


def _widest(kinds: set[Kind]) -> Kind:
    """Return the widest of ``kinds``: ``dag`` < ``cpdag`` < ``pag``, or ``digraph`` alone.

    A ``mag`` counts as a ``pag``: an averaged MAG can hold a tie, which leaves a circle.
    """
    kinds = {"pag" if is_ancestral_kind(kind) else kind for kind in kinds}
    if "digraph" in kinds:
        if len(kinds) > 1:
            raise ValueError(
                f"a graph is a digraph in some fits and another kind in others: {kinds}"
            )
        return "digraph"
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


def _take(x: Any, idx: np.ndarray) -> Any:
    """Return the rows ``idx`` of a table or array, keeping a table's columns and dtypes."""
    if hasattr(x, "iloc"):
        return x.iloc[idx].reset_index(drop=True)
    return np.asarray(x)[idx]


def _draw(job: _Job, b: int) -> list[np.ndarray]:
    """Draw resample ``b``'s row indices, one array per data array."""
    rng = np.random.default_rng(np.random.SeedSequence(job.seed, spawn_key=(b, 0)))
    sizes = [len(x) for x in job.data] if job.unit in ("units", "groups") else [len(job.data)]
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
    _backend.config.num_workers = 1
    _backend.config.backend = job.backend
    _backend.config.tie_break = job.tie_break
    _JOB = job


def _task(b: int) -> tuple[Any, list[tuple[type[Warning], str]]]:
    """Worker side of one resample."""
    assert _JOB is not None  # _init_worker runs before any task
    return _run_one(_JOB, b)


def _fit_all(job: _Job, n_boot: int, workers: int) -> list[tuple[Any, list]]:
    """Fit every resample, in order, in this process or on ``workers`` worker processes."""
    if workers <= 1:
        with _backend.one_blas_thread(), _backend.config(num_workers=1):
            return [_run_one(job, b) for b in range(n_boot)]
    pool = ProcessPoolExecutor(
        max_workers=workers, mp_context=pick_context(), initializer=_init_worker, initargs=(job,)
    )
    try:
        futures = [pool.submit(_task, b) for b in range(n_boot)]
        return [future.result() for future in futures]
    except BrokenProcessPool as exc:
        raise RuntimeError(
            "a bootstrap worker died, most likely out of memory; rerun with "
            "andrey.config(num_workers=1)"
        ) from exc
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


# ---- the public function ------------------------------------------------------------------------


def bootstrap(
    method: str | Callable[..., Any],
    data: npt.ArrayLike | Sequence[npt.ArrayLike],
    *,
    n_boot: int = 100,
    seed: int | None = None,
    threshold: float = 0.5,
    block_length: int | None = None,
    keep_fits: bool = False,
    **method_kwargs: Any,
) -> BootstrapOutput | list[BootstrapOutput]:
    """Refit a method on bootstrap resamples; return each edge's frequency and an averaged graph.

    Each resample draws the rows of ``data`` with replacement, so it has as many rows as ``data``,
    and fits ``method`` on them. The result counts, for every pair of variables, the fits that give
    each endpoint-mark pair (``->``, ``<-``, ``--``, ``<->``, ``o->``, ...) and builds one graph
    from the pairs that are adjacent often enough [1]_ [2]_. A method with a row-aligned input
    (``cdnod``'s ``domain_index``, ``gies``'s ``interventions``) resamples that input with the
    rows. Time-series methods resample blocks of consecutive rows (a moving block bootstrap), and
    ``longitudinal_lingam`` resamples whole units.

    Parameters
    ----------
    method : str or callable
        A method's name, such as ``"pc"``, or the callable itself, such as ``andrey.pc``.
    data : array-like of shape (n_samples, n_variables), or a sequence of them
        The data, as the method takes it. ``multi_group_direct_lingam`` and
        ``longitudinal_lingam`` take a sequence of arrays.
    n_boot : int, default=100
        The number of resamples and fits.
    seed : int or None, default=None
        The seed of the resamples, a non-negative integer. ``None`` uses ``ANDREY_SEED``, or 0 when
        it is unset. Resample ``b`` depends on ``seed`` and ``b`` alone. A method that takes a
        ``seed`` fits resample ``b`` with a seed derived from this one and ``b``, so the method's
        own ``seed`` cannot be set here.
    threshold : float, default=0.5
        The share of fits in which a pair must be adjacent to be in the averaged graph, in
        ``(0, 1]``.
    block_length : int or None, default=None
        For a time-series method on one series, the number of consecutive rows in a block, from 1
        to ``n_samples``; ``None`` uses ``ceil(n_samples ** (1/3))``. A block break puts rows that
        were not neighbors side by side, which weakens a link of that many steps or more, so
        use a block of at least ten times the longest lag. Not for any other method.
    keep_fits : bool, default=False
        Keep every fit in the ``fits`` property.
    **method_kwargs
        The method's other arguments: its keyword parameters, and a row-aligned input such as
        ``domain_index=``.

    Returns
    -------
    BootstrapOutput or list[BootstrapOutput]
        The frequencies and the averaged graph; for ``multi_group_direct_lingam``, one per group.

    Raises
    ------
    ValueError
        If ``method`` is not a method of Andrey, an argument is out of range, a row-aligned input
        has a different number of rows from ``data``, ``block_length`` is given for a method that
        does not use blocks, or a fit raises it: the error then names the resample in a note.
    NotImplementedError
        For ``gin``, whose number of latent nodes varies from fit to fit.
    RuntimeError
        If a worker process dies.

    Warns
    -----
    AndreyWarning
        Each distinct warning the fits raised, once.

    Notes
    -----
    A resample repeats some rows and leaves out others, so a conditional-independence test sees
    more evidence than the resample holds. A method built on such tests, such as ``pc``, keeps
    weak and false edges in more resamples than it does on ``data``, and an orientation that
    needs two variables to stay non-adjacent, a collider, is found in fewer. Read a frequency as
    a ranking of the edges, not as a probability. Resampling can also make a column constant; a
    fit that raises stops the run, and its row numbers count the rows of the resample. The averaged
    graph can contain a directed cycle.

    **Workers.** The fits run on the worker processes ``andrey.config.num_workers`` and
    ``ANDREY_NUM_WORKERS`` set, as for every method: ``-1`` uses every usable core, and ``0`` or
    ``1`` fits in this process. Each fit runs on one worker and one BLAS thread, and the result is
    the same at any worker count.

    **Limitations.** Not benchmarked yet. Refitting is the cost: ``n_boot`` times the method's own,
    shared across the workers. A moving block bootstrap weakens a link that spans a block
    break and keeps the series' own order, so it understates what a very short series can show.

    References
    ----------
    .. [1] Efron, B. (1979). Bootstrap methods: another look at the jackknife. Annals of Statistics,
       7(1), 1-26.
    .. [2] Friedman, N., Goldszmidt, M., and Wyner, A. (1999). Data analysis with Bayesian
       networks: a bootstrap approach. Proceedings of the Fifteenth Conference on Uncertainty in
       Artificial Intelligence, 196-205.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import bootstrap
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal(300)
    >>> data = np.column_stack([x, x + rng.standard_normal(300), rng.standard_normal(300)])
    >>> out = bootstrap("pc", data, n_boot=10, seed=0)
    >>> out.n_boot
    10
    >>> bool(out.frequency("absent")[0, 1] < 0.5)
    True
    """
    from andrey.core.seeding import resolve_seed

    spec = _method_spec(method)
    n_boot = int_in_range(n_boot, "n_boot", 1)
    threshold = _check_threshold(threshold)
    keep_fits = flag(keep_fits, "keep_fits")
    if spec.output.has_latents:
        raise NotImplementedError(
            f"bootstrap does not support {spec.name}: its latent nodes differ from fit to fit"
        )
    root = resolve_seed(seed)
    if root < 0:
        raise ValueError(f"seed must be a non-negative integer, got {root}")
    job = _job(spec, data, method_kwargs, block_length, root)
    workers = min(_backend.worker_count(None), n_boot)
    results = _fit_all(job, n_boot, workers)
    for category, message in dict.fromkeys(w for _, caught in results for w in caught):
        warnings.warn(message, category, stacklevel=2)
    metadata: dict[str, JSONValue] = {
        "method": spec.name,
        "seed": root,
        "block_length": job.block_length,
    }
    fits = [out for out, _ in results]
    if spec.output.multi:
        return [
            BootstrapOutput.from_fits(
                group, threshold=threshold, keep_fits=keep_fits, metadata=metadata
            )
            for group in zip(*fits)
        ]
    return BootstrapOutput.from_fits(
        fits, threshold=threshold, keep_fits=keep_fits, metadata=metadata
    )


def _method_spec(method: str | Callable[..., Any]) -> MethodSpec:
    """Return the ``MethodSpec`` of a method given by name or as its facade callable."""
    from andrey.spec import REGISTRY, get_spec

    name = method if isinstance(method, str) else getattr(method, "__name__", None)
    if name not in REGISTRY or (not isinstance(method, str) and getattr(api, name) is not method):
        raise ValueError(
            f"method must be one of {sorted(REGISTRY)} or such a callable, got {method!r}"
        )
    return get_spec(name)


def _job(
    spec: MethodSpec, data: Any, kwargs: dict[str, Any], block_length: int | None, seed: int
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
    if unit == "blocks":
        n = sizes[0]
        length = math.ceil(n ** (1 / 3)) if block_length is None else block_length
        block_length = int_in_range(length, "block_length", 1, n)
    elif block_length is not None:
        raise ValueError(f"block_length is for a time-series method on one series, not {spec.name}")
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
    for param in spec.params:
        if param.concept == "seed" and set(param.aliases) & set(kwargs):
            raise ValueError(
                f"bootstrap derives each fit's seed from its own seed; {spec.name}'s "
                f"{sorted(set(param.aliases) & set(kwargs))[0]} cannot be set"
            )
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
    )
