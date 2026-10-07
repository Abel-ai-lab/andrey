"""Benchmark Andrey FCI with recorded-sepset or majority collider decisions.

Both configurations use Fisher-Z at ``alpha=0.05`` on numpy with serial execution. The majority
configuration has its own solution name and records ``collider_rule`` in the measured parameters.
Fits return PAG endpoint marks, preserving circles and arrowheads for endpoint-aware scoring.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

# Equalized constraint+latent-lane pins (Fisher-Z CI test at alpha=0.05); runner params override them.
_DEFAULT_ALPHA = 0.05
_DEFAULT_INDEP_TEST = "fisherz"


def _andrey_version() -> str:
    """The installed Andrey version (distribution metadata, else the module attribute)."""
    from importlib import metadata

    try:
        return metadata.version("andrey")
    except metadata.PackageNotFoundError:  # editable/dev checkout with no dist metadata
        import andrey

        return getattr(andrey, "__version__", "unknown")


class AndreyFCI:
    """FCI solution adapter with a separately named configuration for each collider rule."""

    name: str = "andrey.fci.numpy"
    package: str = "andrey"
    package_version: str = _andrey_version()
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "fci"
    output_type: str = "pag"

    def __init__(self, collider_rule: str = "sepsets") -> None:
        if collider_rule not in ("sepsets", "majority"):
            raise ValueError(f"Unsupported collider_rule: {collider_rule!r}")
        self.collider_rule = collider_rule
        if collider_rule == "majority":
            self.name = "andrey.fci.majority.numpy"

    def setup(self) -> None:
        """Import Andrey and its FCI implementation before timing.

        The facade loads `andrey.constraint.fci` on its first call. Setup excludes
        that import from `fit`, including the first timed fit at `--warmup 0`.
        See `contracts.SetupAdapter`.
        """
        import andrey  # noqa: F401
        from andrey.constraint.fci import fci  # noqa: F401

    def params(self) -> dict[str, Any]:
        """Return the CI settings and the selected collider rule."""
        return {
            "alpha": _DEFAULT_ALPHA,
            "indep_test": _DEFAULT_INDEP_TEST,
            "collider_rule": self.collider_rule,
        }

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run Andrey FCI; return the PAG's endpoint-mark array. The only timed region.

        Runner parameters override the adapter's CI settings and collider rule. The returned
        matrix uses ``NULL=0``, ``TAIL=1``, ``ARROW=2``, and ``CIRCLE=3``.
        """
        import andrey

        alpha = float(params.get("alpha", _DEFAULT_ALPHA))
        indep_test = params.get("indep_test", _DEFAULT_INDEP_TEST)
        collider_rule = params.get("collider_rule", self.collider_rule)

        X = np.asarray(data, dtype=np.float64)
        out = andrey.fci(X, alpha=alpha, indep_test=indep_test, collider_rule=collider_rule)
        return np.asarray(out.structure.to_numpy())

    def to_structure(self, native: np.ndarray) -> Any:
        """Rebuild a PAG from its endpoint marks outside the timed region.

        A binary adjacency cannot preserve circle endpoints, so conversion uses the mark matrix.
        """
        from andrey import GraphStructure

        return GraphStructure.from_numpy(native, kind=self.output_type)
