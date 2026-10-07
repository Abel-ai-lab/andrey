"""Shared adapters for adjacency arrays, column names, JSON-safe metadata, and run envelopes."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

import numpy as np

from andrey.core import ARROW, TAIL, GraphStructure, Structure, StructureOutput
from andrey.core.structure import relabel

if TYPE_CHECKING:
    import numpy.typing as npt

    from andrey.core.output import JSONValue


def dag_from_adjacency(adjacency: npt.ArrayLike) -> GraphStructure:
    """Build a ``dag`` structure from an adjacency where any ``A[i, j]`` != 0 is edge ``i -> j``.

    For algorithms that hand back a bare adjacency rather than an endpoint matrix: a 0/1 DAG
    (ExactSearch) or a weighted one already in Andrey's canonical row-is-source orientation (the
    LiNGAM adapters, via :func:`andrey.api.lingam._canonical_weights`). Only the support matters --
    marks are set from the nonzero pattern.
    """
    A = np.asarray(adjacency)
    d = A.shape[0]
    M = np.zeros((d, d), dtype=np.int8)
    rows, cols = np.nonzero(A)  # A[i, j] != 0  =>  edge i -> j
    M[rows, cols] = TAIL  # tail at the source i
    M[cols, rows] = ARROW  # arrow at the target j
    return GraphStructure.from_numpy(M, kind="dag")


def json_safe(value: object) -> JSONValue:
    """Coerce a result value (score, p-values, metadata) into a JSON-safe metadata value.

    numpy scalars/arrays collapse to Python ``int`` / ``float`` / ``bool`` / ``list``; mappings and
    sequences recurse. Raises ``TypeError`` on a value with no JSON-safe form, mirroring the
    fail-fast ``StructureOutput`` metadata check.
    """
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    if isinstance(value, np.ndarray):
        # ndarray.tolist() overloads need a static dtype; correct at runtime for any array.
        return [json_safe(v) for v in value.tolist()]  # ty: ignore[no-matching-overload]
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    raise TypeError(f"cannot coerce {type(value).__name__} to a JSON-safe metadata value")


def structure_output(
    structure: Structure,
    *,
    labels: tuple[str, ...] | None = None,
    ordering: npt.ArrayLike | None = None,
    weighted_adjacency: npt.ArrayLike | None = None,
    metadata: Mapping[str, object] | None = None,
) -> StructureOutput:
    """Assemble the run envelope, coercing ordering / weights / metadata into the store types.

    Metadata values pass through :func:`json_safe` (so a raw ``np.float64`` score is accepted);
    ``ordering`` becomes a plain ``int`` tuple and ``weighted_adjacency`` is cast to ``float64``.
    ``labels`` names the structure's nodes. Delegates to ``StructureOutput.new``, which validates
    the whole envelope.
    """
    if labels is not None:
        structure = relabel(structure, labels)
    meta = None if metadata is None else {str(k): json_safe(v) for k, v in metadata.items()}
    order = None if ordering is None else tuple(int(x) for x in np.asarray(ordering).tolist())
    weights = (
        None if weighted_adjacency is None else np.asarray(weighted_adjacency, dtype=np.float64)
    )
    return StructureOutput.new(structure, ordering=order, weighted_adjacency=weights, metadata=meta)


def node_labels(names: Iterable[object]) -> tuple[str, ...]:
    """Turn column names into node labels: strings, unique so that each names one node."""
    labels = tuple(str(name) for name in names)
    if len(set(labels)) != len(labels):
        raise ValueError(f"column names must be unique to name the nodes, got {list(labels)}")
    return labels


def column_labels(data: object) -> tuple[str, ...] | None:
    """Return a DataFrame's column names as node labels; ``None`` for an array.

    Reads any table with a ``columns`` attribute (pandas, polars), so pandas stays optional. Default
    integer columns ``0..n-1`` name nothing, so such a table numbers its nodes like an array.
    """
    columns = getattr(data, "columns", None)
    if columns is None:
        return None
    names = list(columns)
    return None if names == list(range(len(names))) else node_labels(names)


def shared_labels(label_sets: Iterable[tuple[str, ...] | None]) -> tuple[str, ...] | None:
    """Return the column names several datasets share; ``None`` when none of them is named.

    Raises ``ValueError`` when two datasets name their columns differently.
    """
    named = list(dict.fromkeys(labels for labels in label_sets if labels is not None))
    if len(named) > 1:
        raise ValueError(f"datasets name their columns differently: {named[0]} and {named[1]}")
    return named[0] if named else None


def with_labels(out: StructureOutput, labels: tuple[str, ...] | None) -> StructureOutput:
    """Return ``out`` with its structure's nodes named ``labels``; ``out`` itself for ``None``."""
    if labels is None:
        return out
    return dataclasses.replace(out, _structure=relabel(out.structure, labels))
