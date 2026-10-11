"""Andrey CAM-UV solution adapter for data with hidden common causes.

Wraps ``andrey.camuv``, which returns a PAG over the observed variables: ``i -> j`` for a direct
cause and ``i <-> j`` for a pair connected through an unobserved variable. The pairs are the
bidirected edges of the returned marks. The defaults match ``lingam.CAMUV``'s significance level;
the parent-set size is set to Andrey's default of 3 on both sides.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

_DEFAULTS = {"alpha": 0.01, "max_explanatory_vars": 3}


def _andrey_version() -> str:
    """The installed Andrey version (distribution metadata, else the module attribute)."""
    from importlib import metadata

    try:
        return metadata.version("andrey")
    except metadata.PackageNotFoundError:  # editable/dev checkout with no dist metadata
        import andrey

        return getattr(andrey, "__version__", "unknown")


class AndreyCAMUV:
    """Solution adapter for Andrey's CAM-UV.

    Satisfies :class:`andrey_bench.contracts.SolutionAdapter`.
    """

    name: str = "andrey.camuv.numpy"
    package: str = "andrey"
    package_version: str = _andrey_version()
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "camuv"
    output_type: str = "pag"
    #: The endpoint marks that carry the pairs this method attributes to a hidden common cause.
    confounded_mark: str = "bidirected"

    def setup(self) -> None:
        """Import Andrey and its CAM-UV implementation before timing.

        The facade loads `andrey.irregular.camuv` on its first call. Setup excludes that import
        from `fit`, including the first timed fit at `--warmup 0`. See `contracts.SetupAdapter`.
        """
        import andrey  # noqa: F401
        from andrey.irregular.camuv import camuv  # noqa: F401

    def params(self) -> dict[str, Any]:
        """Return the HSIC significance level and the largest subset tested, child included."""
        return dict(_DEFAULTS)

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run Andrey CAM-UV; return the PAG's endpoint-mark array. The only timed region.

        Runner parameters override the adapter's settings. The returned matrix uses ``NULL=0``,
        ``TAIL=1``, ``ARROW=2``, and ``CIRCLE=3``.
        """
        import andrey

        X = np.asarray(data, dtype=np.float64)
        out = andrey.camuv(X, **{**_DEFAULTS, **params})
        return np.asarray(out.structure.to_numpy())

    def to_structure(self, native: np.ndarray) -> Any:
        """Rebuild the PAG from its endpoint marks outside the timed region."""
        from andrey import GraphStructure

        return GraphStructure.from_numpy(native, kind=self.output_type)
