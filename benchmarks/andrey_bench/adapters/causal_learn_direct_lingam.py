"""causal-learn DirectLiNGAM benchmark adapter.

Calls ``DirectLiNGAM(measure="pwling").fit(X)``, the package default. The pairwise-likelihood order
search is deterministic, so no seed applies.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from andrey_bench.adapters.causal_learn_convert import lingam_to_adjacency

_DEFAULT_MEASURE = "pwling"


def _causal_learn_version() -> str:
    """Return the installed causal-learn distribution version."""
    from importlib import metadata

    try:
        return metadata.version("causal-learn")
    except metadata.PackageNotFoundError:  # pragma: no cover - installed in bench-env
        return "unknown"


class CausalLearnDirectLiNGAM:
    """Benchmark adapter for causal-learn DirectLiNGAM."""

    name: str = "causal-learn.direct_lingam"
    package: str = "causal-learn"
    package_version: str = _causal_learn_version()
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "lingam"
    output_type: str = "dag"

    def setup(self) -> None:
        """Import causal-learn before timing; see `contracts.SetupAdapter`."""
        from causallearn.search.FCMBased.lingam import DirectLiNGAM  # noqa: F401

    def params(self) -> dict[str, Any]:
        return {"measure": _DEFAULT_MEASURE}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Return a 0/1 DAG adjacency. The only timed region."""
        from causallearn.search.FCMBased.lingam import DirectLiNGAM

        model = DirectLiNGAM(measure=params.get("measure", _DEFAULT_MEASURE))
        model.fit(np.asarray(data, dtype=np.float64))
        return lingam_to_adjacency(model.adjacency_matrix_)

    def to_structure(self, native: np.ndarray) -> Any:
        """Convert a 0/1 DAG adjacency to an Andrey ``GraphStructure``."""
        from andrey_bench.contracts import structure_from_adjacency

        return structure_from_adjacency(native, kind=self.output_type)
