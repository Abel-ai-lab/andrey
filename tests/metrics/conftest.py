"""Fixtures for the metrics tests: builders that turn a 0/1 adjacency into Andrey structures."""

from __future__ import annotations

import numpy as np
import pytest

from andrey.core import GraphStructure


def _directed_marks(adj: np.ndarray) -> np.ndarray:
    """Endpoint marks for a 0/1 directed adjacency (``adj[i, j]`` = edge ``i -> j``)."""
    adj = np.asarray(adj)
    n = adj.shape[0]
    marks = np.zeros((n, n), dtype=np.int8)
    for i, j in zip(*np.nonzero(adj)):
        marks[int(i), int(j)] = 1  # TAIL at the source
        marks[int(j), int(i)] = 2  # ARROW at the head
    return marks


@pytest.fixture
def dag_of():
    """Build a ``kind='dag'`` GraphStructure from a 0/1 directed adjacency."""

    def _make(adj, **kwargs) -> GraphStructure:
        return GraphStructure.from_numpy(_directed_marks(adj), kind="dag", **kwargs)

    return _make


@pytest.fixture
def pag_of():
    """Build a ``kind='pag'`` GraphStructure from an explicit endpoint-mark matrix."""

    def _make(marks) -> GraphStructure:
        return GraphStructure.from_numpy(np.asarray(marks, dtype=np.int8), kind="pag")

    return _make
