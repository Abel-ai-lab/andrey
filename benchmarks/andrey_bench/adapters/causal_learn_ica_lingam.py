"""causal-learn ICA-LiNGAM benchmark adapter.

Calls ``ICALiNGAM(random_state=0, max_iter=1000).fit(X)``. FastICA starts from a random unmixing
matrix; causal-learn leaves it unseeded by default. The seed is pinned to the value the Andrey
adapter uses.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from andrey_bench.adapters.causal_learn_convert import lingam_to_adjacency

_DEFAULT_RANDOM_STATE = 0
_DEFAULT_MAX_ITER = 1000


def _causal_learn_version() -> str:
    """Return the installed causal-learn distribution version."""
    from importlib import metadata

    try:
        return metadata.version("causal-learn")
    except metadata.PackageNotFoundError:  # pragma: no cover - installed in bench-env
        return "unknown"


class CausalLearnICALiNGAM:
    """Benchmark adapter for causal-learn ICA-LiNGAM."""

    name: str = "causal-learn.ica_lingam"
    package: str = "causal-learn"
    package_version: str = _causal_learn_version()
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "ica_lingam"
    output_type: str = "dag"

    def setup(self) -> None:
        """Import causal-learn before timing; see `contracts.SetupAdapter`."""
        from causallearn.search.FCMBased.lingam import ICALiNGAM  # noqa: F401

    def params(self) -> dict[str, Any]:
        return {"random_state": _DEFAULT_RANDOM_STATE, "max_iter": _DEFAULT_MAX_ITER}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Return a 0/1 DAG adjacency. The only timed region."""
        from causallearn.search.FCMBased.lingam import ICALiNGAM

        model = ICALiNGAM(
            random_state=int(params.get("random_state", _DEFAULT_RANDOM_STATE)),
            max_iter=int(params.get("max_iter", _DEFAULT_MAX_ITER)),
        )
        model.fit(np.asarray(data, dtype=np.float64))
        return lingam_to_adjacency(model.adjacency_matrix_)

    def to_structure(self, native: np.ndarray) -> Any:
        """Convert a 0/1 DAG adjacency to an Andrey ``GraphStructure``."""
        from andrey_bench.contracts import structure_from_adjacency

        return structure_from_adjacency(native, kind=self.output_type)
