"""Andrey LOCI solution adapters for cause-effect pairs.

Wrap ``andrey.loci``, which decides the direction of a two-column table under the location-scale
noise model and returns a two-node DAG: ``0 -> 1``, ``1 -> 0``, or no edge. :class:`AndreyLOCI`
runs the independence criterion (LOCI-H in Immer et al. 2023) with ``alpha=None``, the direction
with the larger p-value; :class:`AndreyLOCILikelihood` runs the likelihood criterion (LOCI-M), the
direction with the larger likelihood. Each fit standardizes both columns first, as the paper does.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def _andrey_version() -> str:
    """The installed Andrey version (distribution metadata, else the module attribute)."""
    from importlib import metadata

    try:
        return metadata.version("andrey")
    except metadata.PackageNotFoundError:  # editable/dev checkout with no dist metadata
        import andrey

        return getattr(andrey, "__version__", "unknown")


class AndreyLOCI:
    """Solution adapter for Andrey's LOCI under the independence criterion.

    Satisfies :class:`andrey_bench.contracts.SolutionAdapter`.
    """

    name: str = "andrey.loci.independence.numpy"
    package: str = "andrey"
    package_version: str = _andrey_version()
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "loci"
    output_type: str = "dag"
    criterion: str = "independence"

    def setup(self) -> None:
        """Fit a tiny pair before timing, so lazy imports and the warning stay out of the fit."""
        from andrey_bench.adapters.fcm_warm import tiny_table

        self.fit(tiny_table()[:, :2], self.params())

    def params(self) -> dict[str, Any]:
        """The criterion, and ``alpha=None``: the larger p-value, or no level at all."""
        return {"alpha": None, "criterion": self.criterion}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run Andrey LOCI; return the two-node DAG's endpoint marks. The only timed region."""
        import warnings

        import andrey

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", andrey.ExperimentalWarning)
            out = andrey.loci(np.asarray(data, dtype=np.float64), **{**self.params(), **params})
        return np.asarray(out.structure.to_numpy())

    def to_structure(self, native: np.ndarray) -> Any:
        """Rebuild the two-node DAG from its endpoint marks outside the timed region."""
        from andrey import GraphStructure

        return GraphStructure.from_numpy(native, kind=self.output_type)


class AndreyLOCILikelihood(AndreyLOCI):
    """Solution adapter for Andrey's LOCI under the likelihood criterion."""

    name: str = "andrey.loci.likelihood.numpy"
    criterion: str = "likelihood"
