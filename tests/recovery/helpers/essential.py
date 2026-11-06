"""The interventional essential graph: the class of a DAG that its interventions identify."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from andrey import GraphStructure
from andrey.search.gies import cut_pairs, replace_unprotected


def essential(dag: np.ndarray, family: Iterable[Iterable[int]]) -> GraphStructure:
    """The interventional essential graph of the DAG's marks under the observational setting and
    each setting's targets in ``family``, by column position."""
    family = [sorted(variables) for variables in family]
    rows = np.zeros((len(family) + 1, dag.shape[0]), dtype=bool)
    for row, variables in enumerate(family, start=1):
        rows[row, variables] = True
    marks = replace_unprotected(dag.astype(np.int8), cut_pairs(rows))
    return GraphStructure.from_numpy(marks, kind="cpdag")
