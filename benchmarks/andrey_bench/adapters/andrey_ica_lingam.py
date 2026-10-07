"""Andrey ICA-LiNGAM benchmark adapter.

Calls ``andrey.ica_lingam(X, random_state=0, max_iter=1000)``, which returns a weighted DAG.
``weighted_adjacency[i, j]`` weights edge ``i -> j``, so the 0/1 reduction needs no transpose.
FastICA starts from a random unmixing matrix, so the seed is pinned. The causal-learn adapter uses
the same seed.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

_DEFAULT_RANDOM_STATE = 0
_DEFAULT_MAX_ITER = 1000


def _andrey_version() -> str:
    """Return the installed Andrey version from metadata, else the module attribute."""
    from importlib import metadata

    try:
        return metadata.version("andrey")
    except metadata.PackageNotFoundError:  # editable/dev checkout with no dist metadata
        import andrey

        return getattr(andrey, "__version__", "unknown")


class AndreyICALiNGAM:
    """Benchmark adapter for Andrey ICA-LiNGAM, with one variant and no backend axis."""

    name: str = "andrey.ica_lingam.numpy"
    package: str = "andrey"
    package_version: str = _andrey_version()
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "ica_lingam"
    output_type: str = "dag"

    def setup(self) -> None:
        """Import Andrey and its ICA-LiNGAM implementation before timing.

        The facade loads `andrey.lingam.ica` on its first call. See `contracts.SetupAdapter`.
        """
        import andrey  # noqa: F401
        from andrey.lingam.ica import ica_lingam  # noqa: F401

    def params(self) -> dict[str, Any]:
        return {"random_state": _DEFAULT_RANDOM_STATE, "max_iter": _DEFAULT_MAX_ITER}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Return a 0/1 DAG adjacency. The only timed region."""
        import andrey

        out = andrey.ica_lingam(
            np.asarray(data, dtype=np.float64),
            random_state=int(params.get("random_state", _DEFAULT_RANDOM_STATE)),
            max_iter=int(params.get("max_iter", _DEFAULT_MAX_ITER)),
        )
        return np.ascontiguousarray((np.asarray(out.weighted_adjacency) != 0).astype(np.int8))

    def to_structure(self, native: np.ndarray) -> Any:
        """Convert a 0/1 DAG adjacency to an Andrey ``GraphStructure``."""
        from andrey_bench.contracts import structure_from_adjacency

        return structure_from_adjacency(native, kind=self.output_type)
