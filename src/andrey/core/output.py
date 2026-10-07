"""``StructureOutput`` -- the one uniform return type of every algorithm.

A thin, immutable run envelope over a first-class ``Structure`` (``.structure``), plus the
run-level slots that hang off the run rather than the graph: the discovered ``ordering``, a sparse
``weighted_adjacency`` (its own asymmetric ``float64`` CSR), and JSON-safe ``metadata``. Every dense
view derives lazily and uncached. ``save`` / ``load`` serialize the whole envelope.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

import numpy as np

from .structure import (
    SUMMARY_EDGES,
    GraphStructure,
    Structure,
    SummaryGraph,
    TemporalStructure,
    _frozen,
)

if TYPE_CHECKING:
    import scipy.sparse

# A serialized (time, lag) axis requires format version 1; other files retain version 0.
_FORMAT_VERSION = 0
_TIME_FORMAT_VERSION = 1
_SUPPORTED_FORMAT_VERSIONS = frozenset({_FORMAT_VERSION, _TIME_FORMAT_VERSION})

JSONScalar = str | bool | int | float | None
JSONValue = JSONScalar | list | dict


@dataclass(frozen=True, eq=False, slots=True, repr=False)
class StructureOutput:
    """The result of every structure-learning method: the learned graph plus run details.

    Each method returns one (``multi_group_direct_lingam`` returns one per group). ``structure``
    holds the graph, a ``GraphStructure`` or ``TemporalStructure``. The other fields depend on the
    method, and each method's Returns section lists them: ``ordering`` is a causal order,
    ``weighted_adjacency`` holds edge weights, and ``metadata`` holds values such as the algorithm
    name and score. ``print(out)`` shows the graph kind, the edges, the order, and the metadata.
    ``save`` writes the result to a file and ``load`` reads it back. The result is read-only.
    """

    _structure: Structure
    _ordering: tuple[int, ...] | None = None
    _metadata: Mapping[str, JSONValue] = field(default_factory=dict)
    _w_indptr: np.ndarray | None = None  # int64
    _w_indices: np.ndarray | None = None  # int32
    _w_data: np.ndarray | None = None  # float64

    # ---- construction ---------------------------------------------------------------------

    @classmethod
    def new(
        cls,
        structure: Structure,
        *,
        ordering: tuple[int, ...] | list[int] | None = None,
        weighted_adjacency: np.ndarray | None = None,
        metadata: Mapping[str, JSONValue] | None = None,
    ) -> StructureOutput:
        """Build a result from a structure and optional run details.

        Use it to wrap a graph from custom code so that it prints, saves, and loads like a
        method's result.

        Parameters
        ----------
        structure : Structure
            The graph: a ``GraphStructure`` or ``TemporalStructure``.
        ordering : tuple[int, ...] or list[int] or None, default=None
            Causal order as node indices, causes first; a permutation of
            ``range(structure.n_nodes)``.
        weighted_adjacency : np.ndarray of shape (n_nodes, n_nodes) or None, default=None
            Edge weights, ``W[i, j]`` for ``i -> j``, stored sparsely. ``None`` gives no weights.
        metadata : Mapping[str, JSONValue] or None, default=None
            Run values by name. Keys are strings; values are ``str``, ``bool``, ``int``,
            ``float``, ``None``, or lists and dicts of them.

        Returns
        -------
        StructureOutput
            The result.

        Raises
        ------
        TypeError
            If ``metadata`` has a non-string key or a value of another type, such as a tuple, a
            set, or a NumPy array or integer.
        ValueError
            If ``weighted_adjacency`` is not ``(n_nodes, n_nodes)``, ``ordering`` is not a
            permutation of ``range(n_nodes)``, or a graph in ``structure`` fails ``validate()``.
        """
        meta = dict(metadata) if metadata is not None else {}
        _check_json_safe(meta, "metadata")
        w_indptr = w_indices = w_data = None
        if weighted_adjacency is not None:
            W = np.asarray(weighted_adjacency, dtype=np.float64)
            if W.ndim != 2 or W.shape[0] != W.shape[1] or W.shape[0] != structure.n_nodes:
                raise ValueError(
                    f"weighted_adjacency must be ({structure.n_nodes}, {structure.n_nodes}), "
                    f"got {W.shape}"
                )
            import scipy.sparse

            sp = scipy.sparse.csr_array(W)
            sp.sum_duplicates()
            sp.sort_indices()
            w_indptr = _frozen(sp.indptr, np.int64)
            w_indices = _frozen(sp.indices, np.int32)
            w_data = _frozen(sp.data, np.float64)
        out = cls(
            _structure=structure,
            _ordering=tuple(int(x) for x in ordering) if ordering is not None else None,
            _metadata=meta,
            _w_indptr=w_indptr,
            _w_indices=w_indices,
            _w_data=w_data,
        )
        out._validate()  # fail fast now, so new() never builds an envelope load() would reject
        return out

    # ---- surface --------------------------------------------------------------------------

    @property
    def structure(self) -> Structure:
        """The learned graph: a ``GraphStructure``, or a ``TemporalStructure`` for time series."""
        return self._structure

    @property
    def ordering(self) -> tuple[int, ...] | None:
        """Causal order as node indices, causes first; ``None`` unless the method finds one.

        Set by ``direct_lingam``, ``ica_lingam``, ``multi_group_direct_lingam``, and
        ``varma_lingam`` (the order within one time step).
        """
        return self._ordering

    @property
    def metadata(self) -> Mapping[str, JSONValue]:
        """Method-specific values such as ``algorithm`` and ``score``, as a read-only mapping."""
        return MappingProxyType(dict(self._metadata))

    @property
    def weighted_adjacency(self) -> np.ndarray | None:
        """Edge weights as a new dense ``(n_nodes, n_nodes)`` array, or ``None`` without weights.

        ``W[i, j]`` is the weight of ``i -> j``, so rows are causes: the transpose of ``B`` in
        ``x = Bx + e``. Set by ``calm``, ``direct_lingam``, ``ica_lingam``, and
        ``multi_group_direct_lingam``; a time-series result keeps its weights in
        ``structure.lag_weights``.
        """
        if self._w_indptr is None or self._w_indices is None or self._w_data is None:
            return None
        n = self._structure.n_nodes
        W = np.zeros((n, n), dtype=np.float64)
        if self._w_indices.size:
            rows = np.repeat(np.arange(n, dtype=np.intp), np.diff(self._w_indptr))
            W[rows, self._w_indices] = self._w_data
        return W

    def __repr__(self) -> str:
        """Return the structure summary with the algorithm, causal order, and metadata."""
        return "\n".join(self._summary())

    def _summary(
        self, *, name: str | None = None, limit: int | None = SUMMARY_EDGES, more: str = ""
    ) -> list[str]:
        """Build the result summary as lines for ``repr`` and ``andrey run``.

        ``name`` defaults to ``metadata["algorithm"]``. ``limit`` caps the listed edges, and
        ``more`` extends the omitted-edge count. Edge weights come from the sparse store.
        """
        algorithm = self._metadata.get("algorithm")
        if name is None and algorithm is not None:
            name = str(algorithm)
        weights = None if self._w_indptr is None else self.to_scipy_sparse()
        s = self._structure
        lines = s._summary(name=name, weights=weights, limit=limit, more=more)
        if self._ordering is not None:
            labels = s.labels
            order = " -> ".join(labels[i] if labels else str(i) for i in self._ordering)
            lines += ["", f"  order: {order}"]
        meta = [
            f"  {k}: {v:.4g}" if isinstance(v, float) else f"  {k}: {v}"
            for k, v in self._metadata.items()
            if k != "algorithm"
        ]
        if meta:
            lines += ["", *meta]
        return lines

    def to_scipy_sparse(self, *, weights: bool = True) -> scipy.sparse.csr_array:
        """Return the edge weights as a ``scipy.sparse.csr_array``, without a dense array.

        Parameters
        ----------
        weights : bool, default=True
            Must be ``True``. For the graph's endpoint marks, call ``structure.to_scipy_sparse()``.

        Returns
        -------
        scipy.sparse.csr_array of shape (n_nodes, n_nodes)
            The ``weighted_adjacency`` values: entry ``[i, j]`` is the weight of ``i -> j``.

        Raises
        ------
        ValueError
            If ``weights`` is ``False``, or the method gives no edge weights.
        """
        if not weights:
            raise ValueError("StructureOutput.to_scipy_sparse only exposes the weighted adjacency")
        if self._w_indptr is None:
            raise ValueError("this StructureOutput carries no edge weights")
        import scipy.sparse

        n = self._structure.n_nodes
        return scipy.sparse.csr_array((self._w_data, self._w_indices, self._w_indptr), shape=(n, n))

    # ---- serialization --------------------------------------------------------------------

    def save(self, path: str | os.PathLike, *, fmt: Literal["json", "npz"] = "json") -> None:
        """Write the result to ``path``; ``StructureOutput.load`` reads it back.

        Parameters
        ----------
        path : str or os.PathLike
            Destination file. With ``fmt="npz"`` the path must end in ``.npz``.
        fmt : {"json", "npz"}, default="json"
            File format. ``"json"`` holds a graph or time-series result; ``"npz"``, a NumPy
            archive, holds a ``GraphStructure`` result only.

        Raises
        ------
        ValueError
            If ``fmt`` is neither ``"json"`` nor ``"npz"``, or ``fmt="npz"`` and ``path`` does not
            end in ``.npz``.
        NotImplementedError
            If ``fmt="npz"`` and the structure is a ``TemporalStructure``.
        TypeError
            If the structure is a ``SummaryGraph``; save the ``TemporalStructure`` it came from.

        Notes
        -----
        The JSON file is one object with sorted keys. ``format_version`` is ``0``, or ``1`` for a
        panel result that keeps its occasions; ``load`` rejects any other version.
        ``structure_type`` is ``"graph"`` or ``"temporal"``. A graph is stored as its sparse
        arrays ``indptr``, ``indices``, and ``marks``, beside ``kind``, ``n_nodes``, ``labels``,
        ``node_types``, ``ordering``, ``weighted``, and ``metadata``; a temporal result stores one
        such graph per lag under ``lags``. The npz file holds the same arrays, with the other
        fields as one JSON string, and ``load`` reads it without pickle, so loading a file never
        runs code from it.
        """
        doc = self._to_doc()
        if fmt == "json":
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(doc, fh, indent=2, sort_keys=True)
        elif fmt == "npz":
            # np.savez silently appends '.npz'; load() dispatches on the suffix, so a path without
            # it would save to X.npz yet load(X) fails. Require the suffix up front.
            if not os.fspath(path).endswith(".npz"):
                raise ValueError("fmt='npz' requires a path ending in '.npz'")
            if doc["structure_type"] != "graph":
                raise NotImplementedError("npz is graph-only; use fmt='json' for temporal outputs")
            arrays: dict[str, np.ndarray] = {
                "indptr": np.asarray(doc["indptr"], dtype=np.int64),
                "indices": np.asarray(doc["indices"], dtype=np.int32),
                "marks": np.asarray(doc["marks"], dtype=np.int8),
            }
            weighted = doc["weighted"]
            if weighted is not None:
                arrays["w_indptr"] = np.asarray(weighted["indptr"], dtype=np.int64)
                arrays["w_indices"] = np.asarray(weighted["indices"], dtype=np.int32)
                arrays["w_data"] = np.asarray(weighted["data"], dtype=np.float64)
            # Non-numeric fields ride along as one JSON string in a 0-d <U array (pickle-free).
            blob = {
                k: doc[k]
                for k in (
                    "format_version",
                    "structure_type",
                    "kind",
                    "n_nodes",
                    "labels",
                    "node_types",
                    "ordering",
                    "metadata",
                    "has_weighted",
                )
            }
            arrays["_json"] = np.array(json.dumps(blob, sort_keys=True))
            np.savez(path, **arrays)  # ty: ignore[invalid-argument-type]  # numpy stub over-narrows savez kwargs
        else:
            raise ValueError(f"fmt must be 'json' or 'npz', got {fmt!r}")

    @classmethod
    def load(cls, path: str | os.PathLike) -> StructureOutput:
        """Read a result written by ``save``.

        After ``out.save(path)``, ``StructureOutput.load(path) == out``.

        Parameters
        ----------
        path : str or os.PathLike
            File to read. A ``.npz`` suffix reads the npz format; any other suffix reads JSON.

        Returns
        -------
        StructureOutput
            The saved result.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        ValueError
            If the file has an unsupported format version or an unknown structure type, or holds a
            result that ``new`` would reject.
        """
        path = os.fspath(path)
        if path.endswith(".npz"):
            out = cls._from_npz(path)
        else:
            with open(path, encoding="utf-8") as fh:
                out = cls._from_doc(json.load(fh))
        out._validate()
        return out

    def _to_doc(self) -> dict:
        s = self._structure
        weighted = None
        if self._w_indptr is not None and self._w_indices is not None and self._w_data is not None:
            weighted = {
                "indptr": self._w_indptr.tolist(),
                "indices": self._w_indices.tolist(),
                "data": self._w_data.tolist(),
            }
        _check_json_safe(dict(self._metadata), "metadata")
        doc: dict = {
            "format_version": _FORMAT_VERSION,
            "structure_type": s.type,
            "n_nodes": int(s.n_nodes),
            "labels": list(s.labels) if s.labels is not None else None,
            "ordering": list(self._ordering) if self._ordering is not None else None,
            "weighted": weighted,
            "has_weighted": weighted is not None,
            "metadata": dict(self._metadata),
            "dtypes": {
                "indptr": "int64",
                "indices": "int32",
                "marks": "int8",
                "w_indptr": "int64",
                "w_indices": "int32",
                "w_data": "float64",
            },
        }
        if s.type == "graph":
            doc.update(_graph_to_doc(s))  # ty: ignore[invalid-argument-type]  # s narrowed to GraphStructure by s.type
        elif s.type == "temporal":
            doc["lags"] = [_graph_to_doc(g) for g in s._lags]  # ty: ignore[unresolved-attribute]  # s is TemporalStructure here
            doc["lag_indices"] = list(s._lag_indices)  # ty: ignore[unresolved-attribute]
            doc["lag_weights"] = (
                s._lag_weights.tolist() if s._lag_weights is not None else None  # ty: ignore[unresolved-attribute]
            )
            if s._lag_weights_ma is not None:  # ty: ignore[unresolved-attribute]
                doc["lag_weights_ma"] = s._lag_weights_ma.tolist()  # ty: ignore[unresolved-attribute]
            if s._time_indices:  # ty: ignore[unresolved-attribute]  # (time, lag) axis -> v1
                doc["format_version"] = _TIME_FORMAT_VERSION
                doc["time_indices"] = list(s._time_indices)  # ty: ignore[unresolved-attribute]
                doc["time_graphs"] = [
                    [_graph_to_doc(g) for g in row]
                    for row in s._time_graphs  # ty: ignore[unresolved-attribute]
                ]
                tw = s._time_weights  # ty: ignore[unresolved-attribute]
                # NaN (uncomputable blocks) encodes as JSON null, not the non-standard bare ``NaN``
                # token; ``_structure_from_doc`` reads null back to NaN via ``np.asarray``.
                if tw is None:
                    doc["time_weights"] = None
                else:
                    masked = tw.astype(object)
                    masked[np.isnan(tw)] = None
                    doc["time_weights"] = masked.tolist()
        else:  # pragma: no cover - guarded by the type system
            raise TypeError(f"cannot serialize structure type {s.type!r}")
        return doc

    @classmethod
    def _from_doc(cls, doc: dict) -> StructureOutput:
        _check_format_version(doc.get("format_version"))
        structure = _structure_from_doc(doc)
        weighted = doc.get("weighted")
        w_indptr = w_indices = w_data = None
        if weighted is not None:
            w_indptr = _frozen(weighted["indptr"], np.int64)
            w_indices = _frozen(weighted["indices"], np.int32)
            w_data = _frozen(weighted["data"], np.float64)
        ordering = doc.get("ordering")
        return cls(
            _structure=structure,
            _ordering=tuple(int(x) for x in ordering) if ordering is not None else None,
            _metadata=dict(doc.get("metadata", {})),
            _w_indptr=w_indptr,
            _w_indices=w_indices,
            _w_data=w_data,
        )

    @classmethod
    def _from_npz(cls, path: str) -> StructureOutput:
        with np.load(path, allow_pickle=False) as npz:
            blob = json.loads(str(npz["_json"]))
            _check_format_version(blob.get("format_version"))
            if blob["structure_type"] != "graph":  # pragma: no cover - save writes only graph npz
                raise NotImplementedError("npz load supports graph structures only")
            labels = tuple(blob["labels"]) if blob["labels"] is not None else None
            node_types = (
                np.asarray(blob["node_types"], dtype=np.int8)
                if blob.get("node_types") is not None
                else None
            )
            structure = GraphStructure._from_csr(
                blob["n_nodes"],
                npz["indptr"],
                npz["indices"],
                npz["marks"],
                kind=blob["kind"],
                labels=labels,
                node_types=node_types,
            )
            w_indptr = w_indices = w_data = None
            if blob["has_weighted"]:
                w_indptr = _frozen(npz["w_indptr"], np.int64)
                w_indices = _frozen(npz["w_indices"], np.int32)
                w_data = _frozen(npz["w_data"], np.float64)
            ordering = blob["ordering"]
        return cls(
            _structure=structure,
            _ordering=tuple(int(x) for x in ordering) if ordering is not None else None,
            _metadata=dict(blob["metadata"]),
            _w_indptr=w_indptr,
            _w_indices=w_indices,
            _w_data=w_data,
        )

    def _validate(self) -> None:
        s = self._structure
        if isinstance(s, GraphStructure):
            s.validate()
        elif isinstance(s, TemporalStructure):
            for lag_graph in s._lags:  # traverse the stack directly; lag(k) by-value is for callers
                lag_graph.validate()
            for row in s._time_graphs:  # and the full (time, lag) grid when present
                for g in row:
                    g.validate()
        n = s.n_nodes
        if self._ordering is not None and sorted(self._ordering) != list(range(n)):
            raise ValueError("ordering must be a permutation of range(n_nodes)")
        if self._w_indptr is not None and self._w_indptr.shape[0] != n + 1:
            raise ValueError("weighted CSR indptr disagrees with n_nodes")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StructureOutput):
            return NotImplemented
        if not (
            self._structure == other._structure
            and self._ordering == other._ordering
            and dict(self._metadata) == dict(other._metadata)
        ):
            return False
        for a, b, equal_nan in (
            (self._w_indptr, other._w_indptr, False),
            (self._w_indices, other._w_indices, False),
            (self._w_data, other._w_data, True),  # float weights: NaN==NaN so a==a stays reflexive
        ):
            if (a is None) != (b is None):
                return False
            if a is not None and b is not None and not np.array_equal(a, b, equal_nan=equal_nan):
                return False
        return True

    __hash__ = None  # type: ignore[assignment]


def _graph_to_doc(g: GraphStructure) -> dict:
    if isinstance(g, SummaryGraph):
        # The single graph-serialization choke point (top-level structure and every temporal lag
        # graph, json and npz). A SummaryGraph is a derived projection: this path writes marks and
        # kind but not the lag-sets, so it would degrade to a bare digraph on load. Refuse rather
        # than lose data silently -- persist the source TemporalStructure, which re-derives it.
        raise TypeError(
            "a SummaryGraph is a derived projection and does not serialize (its lag-sets "
            "would be silently dropped); save the source TemporalStructure and re-derive via "
            "summary_graph(), or save the collapsed digraph explicitly with "
            "GraphStructure.from_numpy(g.to_numpy(), kind='digraph', allow_self_loops=True)"
        )
    # Self-contained: a per-lag graph carries its OWN n_nodes/labels so temporal stacks round-trip
    # even when a lag's labels differ from the envelope's. The store serializes symmetrically
    # (via _sym_csr) whatever the internal layout, so the on-disk format_version stays stable.
    indptr, indices, marks = g._sym_csr()
    return {
        "kind": g.kind,
        "n_nodes": int(g.n_nodes),
        "labels": list(g.labels) if g.labels is not None else None,
        "indptr": indptr.tolist(),
        "indices": indices.tolist(),
        "marks": marks.tolist(),
        "node_types": g._node_types.tolist() if g._node_types is not None else None,
    }


def _graph_from_doc(doc: dict) -> GraphStructure:
    labels = tuple(doc["labels"]) if doc.get("labels") is not None else None
    node_types = (
        np.asarray(doc["node_types"], dtype=np.int8) if doc.get("node_types") is not None else None
    )
    return GraphStructure._from_csr(
        doc["n_nodes"],
        doc["indptr"],
        doc["indices"],
        doc["marks"],
        kind=doc["kind"],
        labels=labels,
        node_types=node_types,
    )


def _structure_from_doc(doc: dict) -> Structure:
    kind = doc["structure_type"]
    if kind == "graph":
        return _graph_from_doc(doc)
    if kind == "temporal":
        labels = tuple(doc["labels"]) if doc.get("labels") is not None else None
        lags = tuple(_graph_from_doc(lag) for lag in doc["lags"])
        lag_weights = (
            _frozen(doc["lag_weights"], np.float64) if doc.get("lag_weights") is not None else None
        )
        lag_weights_ma = (
            _frozen(doc["lag_weights_ma"], np.float64)
            if doc.get("lag_weights_ma") is not None
            else None
        )
        lag_indices = tuple(doc["lag_indices"]) if doc.get("lag_indices") is not None else ()
        time_indices = tuple(doc["time_indices"]) if doc.get("time_indices") is not None else ()
        time_graphs = (
            tuple(tuple(_graph_from_doc(g) for g in row) for row in doc["time_graphs"])
            if doc.get("time_graphs") is not None
            else ()
        )
        time_weights = (
            _frozen(doc["time_weights"], np.float64)
            if doc.get("time_weights") is not None
            else None
        )
        return TemporalStructure(
            _n_nodes=doc["n_nodes"],
            _labels=labels,
            _lags=lags,
            _lag_indices=lag_indices,
            _lag_weights=lag_weights,
            _lag_weights_ma=lag_weights_ma,
            _time_graphs=time_graphs,
            _time_indices=time_indices,
            _time_weights=time_weights,
        )
    raise ValueError(f"unknown structure_type {kind!r}")


def _check_format_version(version: object) -> None:
    if version not in _SUPPORTED_FORMAT_VERSIONS:
        raise ValueError(
            f"unsupported format_version {version!r}; this build reads "
            f"{sorted(_SUPPORTED_FORMAT_VERSIONS)}"
        )


def _check_json_safe(obj: object, path: str) -> None:
    """Raise unless ``obj`` is JSON-safe (str/bool/int/float/None + list/dict thereof).

    Rejects numpy scalars (``np.int64``, ``np.float32``), ``np.ndarray``, ``set``, and ``tuple``
    so metadata cannot silently break ``json.dump`` or round-trip lossily.
    """
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return
    if isinstance(obj, list):
        for i, item in enumerate(obj):
            _check_json_safe(item, f"{path}[{i}]")
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            if not isinstance(key, str):
                raise TypeError(f"metadata keys must be str, got {type(key).__name__} at {path}")
            _check_json_safe(value, f"{path}.{key}")
        return
    raise TypeError(f"non-JSON-safe metadata value of type {type(obj).__name__} at {path}")
