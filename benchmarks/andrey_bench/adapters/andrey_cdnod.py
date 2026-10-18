"""Andrey CD-NOD solution adapter for data pooled from several domains.

Calls ``andrey.cdnod(X, domain_index, alpha=0.05, indep_test="fisherz")``, Andrey's defaults. A
``domain_shift`` dataset holds its domains in contiguous equal blocks of rows
(:mod:`andrey_bench.regime_models`), so ``fit`` rebuilds the domain index from ``n_domains``.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from andrey_bench.regime_models import N_DOMAINS, domain_index

_DEFAULTS = {"alpha": 0.05, "indep_test": "fisherz", "n_domains": N_DOMAINS}


def _andrey_version() -> str:
    """The installed Andrey version (distribution metadata, else the module attribute)."""
    from importlib import metadata

    try:
        return metadata.version("andrey")
    except metadata.PackageNotFoundError:  # editable/dev checkout with no dist metadata
        import andrey

        return getattr(andrey, "__version__", "unknown")


class AndreyCDNOD:
    """Solution adapter for Andrey's CD-NOD.

    Satisfies :class:`andrey_bench.contracts.SolutionAdapter`.
    """

    name: str = "andrey.cdnod.numpy"
    package: str = "andrey"
    package_version: str = _andrey_version()
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "cdnod"
    output_type: str = "cpdag"

    def setup(self) -> None:
        """Import Andrey and its CD-NOD implementation before timing; see `contracts.SetupAdapter`."""
        import andrey  # noqa: F401
        from andrey.constraint.cdnod import cdnod  # noqa: F401

    def params(self) -> dict[str, Any]:
        """Return the test, its significance level, and the number of domains in the rows."""
        return dict(_DEFAULTS)

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run Andrey CD-NOD; return the CPDAG's endpoint-mark array. The only timed region."""
        import andrey

        settings = {**_DEFAULTS, **params}
        X = np.asarray(data, dtype=np.float64)
        index = domain_index(len(X), int(settings["n_domains"]))
        out = andrey.cdnod(
            X, index, alpha=float(settings["alpha"]), indep_test=settings["indep_test"]
        )
        return np.asarray(out.structure.to_numpy())

    def to_structure(self, native: np.ndarray) -> Any:
        """Rebuild the CPDAG from its endpoint marks outside the timed region."""
        from andrey import GraphStructure

        return GraphStructure.from_numpy(native, kind=self.output_type)
