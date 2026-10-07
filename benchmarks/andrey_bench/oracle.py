"""Oracle solution used to validate the scoring pipeline.

``OracleGES`` returns the dataset's true graph, optionally projected to a
CPDAG. A correct scoring and conversion pipeline should therefore give the
oracle a perfect score.

The runner supplies the true graph through ``params["truth_marks"]`` using
``OracleGES.truth_params``.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

import andrey
import andrey.metrics as M
from andrey import GraphStructure

TRUTH_KEY = "truth_marks"


@dataclass
class OracleGES:
    """Oracle adapter that returns the true graph, optionally as a CPDAG."""

    name: str = "oracle.ges"
    package: str = "oracle"
    package_version: str = andrey.__version__
    backend: str = "native"
    mode: str = "serial"
    algorithm: str = "ges"
    output_type: str = "cpdag"
    # A CPDAG is what a perfect MEC-identified (GES) learner reports; keep it configurable so a
    # DAG-lane variant can echo the truth verbatim without projection.
    project_to_cpdag: bool = True

    @staticmethod
    def truth_params(truth: GraphStructure) -> dict[str, np.ndarray]:
        """Return the runner parameters containing the ground-truth graph."""
        return {TRUTH_KEY: np.asarray(truth.to_numpy())}

    def params(self) -> dict[str, Any]:
        """Return solution configuration parameters."""
        return {}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Return the supplied ground-truth graph as endpoint marks.

        Validates that the truth is square and matches the dataset dimensions.

        If enabled, projects the DAG to its CPDAG before returning it.

        """
        if TRUTH_KEY not in params:
            raise KeyError(
                f"OracleGES.fit requires params[{TRUTH_KEY!r}] (the true endpoint-mark array); "
                "the runner must inject it via OracleGES.truth_params(truth)."
            )
        marks = np.asarray(params[TRUTH_KEY])
        if marks.ndim != 2 or marks.shape[0] != marks.shape[1]:
            raise ValueError(f"truth_marks must be a square 2-D mark matrix, got {marks.shape}")
        d = np.asarray(data).shape[1] if np.asarray(data).ndim == 2 else marks.shape[0]
        if marks.shape[0] != d:
            raise ValueError(
                f"truth_marks is {marks.shape[0]}x{marks.shape[0]} but data has {d} variables — "
                "the runner supplied a truth for the wrong dataset."
            )
        if not self.project_to_cpdag:
            return marks.copy()
        truth = GraphStructure.from_numpy(marks, kind="dag")
        return np.asarray(M.to_cpdag(truth).to_numpy())

    def to_structure(self, native: np.ndarray) -> GraphStructure:
        """Convert endpoint marks returned by ``fit`` to a ``GraphStructure``."""
        return GraphStructure.from_numpy(np.asarray(native), kind=self.output_type)
