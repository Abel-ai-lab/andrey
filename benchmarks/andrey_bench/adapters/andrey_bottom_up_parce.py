"""Andrey BottomUpParceLiNGAM solution adapter for data with hidden common causes.

Wraps ``andrey.bottom_up_parce_lingam``, which returns a PAG over the observed variables: ``i -> j``
for a direct cause and ``i o-o j`` for a pair whose order a hidden common cause leaves unresolved.
The pairs the method attributes to a hidden common cause are the circle-circle edges of the returned
marks. The significance level matches ``lingam.BottomUpParceLiNGAM``'s default.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

_DEFAULTS = {"alpha": 0.1}


def _andrey_version() -> str:
    """The installed Andrey version (distribution metadata, else the module attribute)."""
    from importlib import metadata

    try:
        return metadata.version("andrey")
    except metadata.PackageNotFoundError:  # editable/dev checkout with no dist metadata
        import andrey

        return getattr(andrey, "__version__", "unknown")


class AndreyBottomUpParceLiNGAM:
    """Solution adapter for Andrey's BottomUpParceLiNGAM.

    Satisfies :class:`andrey_bench.contracts.SolutionAdapter`.
    """

    name: str = "andrey.bottom_up_parce.numpy"
    package: str = "andrey"
    package_version: str = _andrey_version()
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "bottom_up_parce"
    output_type: str = "pag"
    #: The endpoint marks that carry the pairs this method attributes to a hidden common cause.
    confounded_mark: str = "circle"

    def setup(self) -> None:
        """Import Andrey and its BottomUpParceLiNGAM implementation before timing.

        The facade loads `andrey.irregular.bottom_up_parce` on its first call. Setup excludes that
        import from `fit`, including the first timed fit at `--warmup 0`. See
        `contracts.SetupAdapter`.
        """
        import andrey  # noqa: F401
        from andrey.irregular.bottom_up_parce import bottom_up_parce_lingam  # noqa: F401

    def params(self) -> dict[str, Any]:
        """Return the independence-test significance level; the search is deterministic."""
        return dict(_DEFAULTS)

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run Andrey BottomUpParceLiNGAM; return the PAG's endpoint-mark array. The timed region.

        Runner parameters override the adapter's settings. The returned matrix uses ``NULL=0``,
        ``TAIL=1``, ``ARROW=2``, and ``CIRCLE=3``.
        """
        import andrey

        X = np.asarray(data, dtype=np.float64)
        out = andrey.bottom_up_parce_lingam(X, **{**_DEFAULTS, **params})
        return np.asarray(out.structure.to_numpy())

    def to_structure(self, native: np.ndarray) -> Any:
        """Rebuild the PAG from its endpoint marks outside the timed region."""
        from andrey import GraphStructure

        return GraphStructure.from_numpy(native, kind=self.output_type)
