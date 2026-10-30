"""Andrey RCD solution adapter for data with hidden common causes.

Wraps ``andrey.rcd``, which returns a PAG over the observed variables: ``i -> j`` for a direct cause
and ``i <-> j`` for a pair the method attributes to a hidden common cause. The pairs are the
bidirected edges of the returned marks. The defaults match ``lingam.RCD``'s, so both solutions run
the same settings.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

_DEFAULTS = {
    "max_depth": 2,
    "cor_alpha": 0.01,
    "alpha": 0.01,
    "shapiro_alpha": 0.01,
}


def _andrey_version() -> str:
    """The installed Andrey version (distribution metadata, else the module attribute)."""
    from importlib import metadata

    try:
        return metadata.version("andrey")
    except metadata.PackageNotFoundError:  # editable/dev checkout with no dist metadata
        import andrey

        return getattr(andrey, "__version__", "unknown")


class AndreyRCD:
    """Solution adapter for Andrey's RCD.

    Satisfies :class:`andrey_bench.contracts.SolutionAdapter`.
    """

    name: str = "andrey.rcd.numpy"
    package: str = "andrey"
    package_version: str = _andrey_version()
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "rcd"
    output_type: str = "pag"
    #: The endpoint marks that carry the pairs this method attributes to a hidden common cause.
    confounded_mark: str = "bidirected"

    def setup(self) -> None:
        """Import Andrey and its RCD implementation before timing.

        The facade loads `andrey.irregular.rcd` on its first call. Setup excludes that import from
        `fit`, including the first timed fit at `--warmup 0`. See `contracts.SetupAdapter`.
        """
        import andrey  # noqa: F401
        from andrey.irregular.rcd import fit_rcd  # noqa: F401

    def params(self) -> dict[str, Any]:
        """Return the search settings: ancestor-set size and the three significance levels."""
        return dict(_DEFAULTS)

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run Andrey RCD; return the PAG's endpoint-mark array. The only timed region.

        Runner parameters override the adapter's settings. The returned matrix uses ``NULL=0``,
        ``TAIL=1``, ``ARROW=2``, and ``CIRCLE=3``.
        """
        import andrey

        X = np.asarray(data, dtype=np.float64)
        out = andrey.rcd(X, **{**_DEFAULTS, **params})
        return np.asarray(out.structure.to_numpy())

    def to_structure(self, native: np.ndarray) -> Any:
        """Rebuild the PAG from its endpoint marks outside the timed region."""
        from andrey import GraphStructure

        return GraphStructure.from_numpy(native, kind=self.output_type)
