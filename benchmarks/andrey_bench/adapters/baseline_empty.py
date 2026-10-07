"""Empty-graph baseline solution — the measured correctness floor.

This adapter runs the empty graph (no edges) on every dataset and is scored like any other
solution, so "better than nothing" is a measured record rather than an assumed number. A method
that recovers less than nothing has a worse record, in the same units, on the same dataset.

The floor is a non-trivial bar. The empty graph's SHD to a sparse ground truth is exactly the
truth's edge count ``m``, which on a sparse ER DGP grows only linearly in ``d`` while the number of
node pairs grows quadratically. Beating it therefore forces a solution to recover structure, not
merely to avoid catastrophic errors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class BaselineEmpty:
    """Empty-graph correctness-floor solution.

    Satisfies :class:`andrey_bench.contracts.SolutionAdapter`. Its ``algorithm`` is ``baseline``,
    not the family it is read against: the empty graph implements no method, and filing it as one
    would pool the correctness floor into that method's field on any ``algorithm`` groupby.
    """

    name: str = "baseline.empty"
    package: str = "baseline"
    package_version: str = "1.0"  # a constant graph — versionless, pinned for provenance only
    backend: str = "native"
    mode: str = "serial"
    algorithm: str = "baseline"
    output_type: str = "cpdag"

    def params(self) -> dict[str, Any]:
        """The empty graph has nothing to configure."""
        return {}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Return the ``d×d`` all-zero (no-edge) 0/1 adjacency. The timed region — near-zero cost.

        ``d`` is the data's column count; ``params`` is ignored (the empty graph has no knobs).
        """
        d = int(np.asarray(data).shape[1])
        return np.zeros((d, d), dtype=np.int8)

    def to_structure(self, native: np.ndarray) -> Any:
        """All-zero adjacency -> the empty Andrey ``GraphStructure``; untimed."""
        from andrey_bench.contracts import structure_from_adjacency

        return structure_from_adjacency(native, kind=self.output_type)
