"""Native on-disk storage for generated datasets (safetensors + JSON manifest).

A dataset serializes to one ``.safetensors`` file: the data matrix and the truth's **symmetric** CSR
as named tensors, with ``kind``, labels, provenance, the QA report, and the structural-param names
in the string-only metadata header (so a whole collection can be scanned without reading a data
buffer). safetensors is an optional dependency (the ``andrey[data]`` extra) and is imported
lazily, so importing ``andrey`` -- or ``andrey.data`` -- never requires it.

Collections write one file per dataset plus a ``collection.json`` manifest. The HF-datasets
porting surface (parquet, ``x_ref`` streaming) and X row-sharding are not implemented here; a single
file per dataset is the K=1 fast path that covers sparse graphs at benchmark ``n``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from andrey.core.structure import GraphStructure

from .dataset import CausalDataset, QAReport, SCMParams

_FORMAT = "andrey-dgp"
_VERSION = "1"


def _finite_or_none(value: float | None) -> float | None:
    """Map a NaN report field to None so the metadata JSON stays strict-valid (no bare ``NaN``)."""
    return None if value is not None and isinstance(value, float) and np.isnan(value) else value


def _require_safetensors():
    try:
        from safetensors import safe_open
        from safetensors.numpy import save_file
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "andrey.data.storage needs safetensors; install the data extra: "
            "pip install 'andrey-core[data]'"
        ) from exc
    return save_file, safe_open


def save(dataset: CausalDataset, path: str | Path) -> None:
    """Write ``dataset`` to a single ``.safetensors`` file at ``path``."""
    graph = dataset.graph
    # Type-check before requiring safetensors so an unsupported truth raises the accurate error even
    # without the [data] extra (otherwise the missing-dependency ImportError masks it).
    if not isinstance(graph, GraphStructure):
        raise NotImplementedError(
            "storage supports GraphStructure truth; temporal (TemporalStructure) datasets are not "
            "yet serializable here -- use them in memory, or persist via StructureOutput"
        )
    save_file, _ = _require_safetensors()
    indptr, indices, marks = graph._sym_csr()
    tensors = {
        "X": np.ascontiguousarray(dataset.data),
        "truth.indptr": np.ascontiguousarray(indptr, dtype=np.int64),
        "truth.indices": np.ascontiguousarray(indices, dtype=np.int32),
        "truth.marks": np.ascontiguousarray(marks, dtype=np.int8),
        "params.edges": np.ascontiguousarray(dataset.params.edges, dtype=np.int64),
        "params.weights": np.ascontiguousarray(dataset.params.weights, dtype=np.float64),
        "params.noise_scales": np.ascontiguousarray(dataset.params.noise_scales, dtype=np.float64),
    }
    node_types = graph.node_types
    if node_types is not None:  # absence, not a zeros tensor, so __eq__ round-trips
        tensors["truth.node_types"] = np.ascontiguousarray(node_types, dtype=np.int8)
    empty = [k for k, v in tensors.items() if v.size == 0]  # safetensors rejects zero-size tensors
    for k in empty:
        del tensors[k]
    report = dataset.report
    metadata = {
        "format": _FORMAT,
        "format_version": _VERSION,
        "kind": graph.kind,
        "n_nodes": str(graph.n_nodes),
        "x_shape": json.dumps(list(dataset.data.shape)),  # rebuild X if dropped as zero-size (n=0)
        "labels": json.dumps(list(graph.labels) if graph.labels is not None else None),
        "node_types_present": json.dumps(node_types is not None),
        "empty_tensors": json.dumps(empty),
        "provenance": json.dumps(dataset.provenance),
        "report": json.dumps(
            {
                "varsortability": _finite_or_none(report.varsortability),
                "r2sortability": _finite_or_none(report.r2sortability),
                "faithfulness_accept_rate": _finite_or_none(report.faithfulness_accept_rate),
                "scale": report.scale,
            }
        ),
        "params": json.dumps(
            {"functional": dataset.params.functional, "noise": dataset.params.noise}
        ),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata=metadata)


def load(path: str | Path) -> CausalDataset:
    """Load a :class:`~andrey.data.dataset.CausalDataset` written by :func:`save`."""
    _, safe_open = _require_safetensors()
    with safe_open(str(path), framework="numpy") as f:
        meta = f.metadata()
        keys = set(f.keys())
        tensors = {k: f.get_tensor(k) for k in keys}

    def col(name, dtype, ncols=None):  # a dropped zero-size tensor rebuilds as an empty array
        if name in tensors:
            return tensors[name]
        shape = (0, ncols) if ncols is not None else (0,)
        return np.empty(shape, dtype=dtype)

    n_nodes = int(meta["n_nodes"])
    labels = json.loads(meta["labels"])
    labels = tuple(labels) if labels is not None else None
    node_types = tensors.get("truth.node_types") if json.loads(meta["node_types_present"]) else None
    graph = GraphStructure._from_csr(
        n_nodes,
        col("truth.indptr", np.int64),
        col("truth.indices", np.int32),
        col("truth.marks", np.int8),
        kind=meta["kind"],
        labels=labels,
        node_types=node_types,
    )
    rep = json.loads(meta["report"])
    params_meta = json.loads(meta["params"])
    params = SCMParams(
        edges=col("params.edges", np.int64, ncols=2),
        weights=col("params.weights", np.float64),
        noise_scales=col("params.noise_scales", np.float64),
        functional=params_meta["functional"],
        noise=params_meta["noise"],
    )
    x = tensors.get("X")
    if x is None:  # dropped as a zero-size tensor (for example, n=0)
        x = np.empty(tuple(json.loads(meta["x_shape"])), dtype=np.float64)
    return CausalDataset(
        data=x,
        graph=graph,
        report=QAReport(**rep),
        provenance=json.loads(meta["provenance"]),
        params=params,
    )


def save_collection(datasets: list[CausalDataset], directory: str | Path) -> None:
    """Write each dataset to ``directory`` plus a ``collection.json`` manifest.

    The manifest denormalizes each file's header (d, n, nnz, kind, varsortability, scale) so a
    grid is filterable without opening any data buffer; it is a cache, rebuildable from the
    per-file headers.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, ds in enumerate(datasets):
        filename = f"dataset-{i:05d}.safetensors"
        save(ds, directory / filename)
        manifest.append(
            {
                "file": filename,
                "d": ds.graph.n_nodes,
                "n": int(ds.data.shape[0]),
                "n_obs": int(ds.data.shape[1]),
                "nnz": int(ds.params.edges.shape[0]),
                "kind": ds.graph.kind,
                "varsortability": ds.report.varsortability,
                "scale": ds.report.scale,
            }
        )
    (directory / "collection.json").write_text(
        json.dumps({"format": _FORMAT, "format_version": _VERSION, "datasets": manifest}, indent=2)
    )


def load_collection(directory: str | Path) -> list[CausalDataset]:
    """Load every dataset in a collection written by :func:`save_collection`."""
    directory = Path(directory)
    manifest = json.loads((directory / "collection.json").read_text())
    return [load(directory / entry["file"]) for entry in manifest["datasets"]]
