"""lingam CAM-UV adapter for data with hidden common causes.

Calls ``lingam.CAMUV(...).fit(X)``. ``adjacency_matrix_`` has ``B[i, j] = 1`` for each parent ``j``
of ``i`` and ``NaN`` in both cells of a pair connected through an unobserved variable. That is the
encoding of lingam's RCD, so the matrix goes through the function that converts an RCD result. It
gives the marks that Andrey's CAM-UV conversion builds from the same parents and pairs.

lingam's CAM-UV fits its smooth functions with pygam, whose releases before 0.12 fail with
current scipy.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

# `num_explanatory_vals` is the largest subset tested, the variable itself included; Andrey calls
# it `max_explanatory_vars`.
_DEFAULTS = {"alpha": 0.01, "num_explanatory_vals": 3}


def _lingam_version() -> str:
    """The installed lingam version (distribution metadata; the module exposes one too)."""
    from importlib import metadata

    try:
        return metadata.version("lingam")
    except metadata.PackageNotFoundError:  # pragma: no cover - installed in bench-env
        return "unknown"


class LingamCAMUV:
    """Solution adapter for lingam's CAM-UV.

    Satisfies :class:`andrey_bench.contracts.SolutionAdapter`.
    """

    name: str = "lingam.camuv"
    package: str = "lingam"
    package_version: str = _lingam_version()
    backend: str = "native"
    mode: str = "serial"
    algorithm: str = "camuv"
    output_type: str = "pag"
    #: The endpoint marks that carry the pairs this method attributes to a hidden common cause.
    confounded_mark: str = "bidirected"

    def setup(self) -> None:
        """Import lingam and the shared conversion before timing; see `contracts.SetupAdapter`."""
        import lingam  # noqa: F401

        from andrey.api._irregular import _adapt_rcd  # noqa: F401

    def params(self) -> dict[str, Any]:
        """Return the HSIC significance level and the largest parent set tried."""
        return dict(_DEFAULTS)

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run lingam CAM-UV; return the PAG's endpoint-mark array. The only timed region.

        Runner parameters override the adapter's settings. The returned matrix uses ``NULL=0``,
        ``TAIL=1``, ``ARROW=2``, and ``CIRCLE=3``.
        """
        import lingam

        from andrey.api._irregular import _adapt_rcd

        X = np.asarray(data, dtype=np.float64)
        model = lingam.CAMUV(**{**_DEFAULTS, **params}).fit(X)
        return np.asarray(_adapt_rcd(model).structure.to_numpy())

    def to_structure(self, native: np.ndarray) -> Any:
        """Rebuild the PAG from its endpoint marks outside the timed region."""
        from andrey import GraphStructure

        return GraphStructure.from_numpy(native, kind=self.output_type)
