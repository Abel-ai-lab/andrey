"""lingam RCD adapter for data with hidden common causes.

Calls ``lingam.RCD(...).fit(X)``. ``adjacency_matrix_`` is the weighted ``B`` of ``x = B x + e``: a
finite nonzero ``B[i, j]`` is the edge ``j -> i`` and a ``NaN`` cell marks a pair with a hidden
common cause. The matrix is converted to PAG endpoint marks by the function that converts Andrey's
RCD result, so both solutions are scored from the same encoding.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

# lingam's own defaults, which Andrey's RCD adapter also uses.
_DEFAULTS = {
    "max_explanatory_num": 2,
    "cor_alpha": 0.01,
    "ind_alpha": 0.01,
    "shapiro_alpha": 0.01,
}


def _lingam_version() -> str:
    """The installed lingam version (distribution metadata; the module exposes one too)."""
    from importlib import metadata

    try:
        return metadata.version("lingam")
    except metadata.PackageNotFoundError:  # pragma: no cover - installed in bench-env
        return "unknown"


class LingamRCD:
    """Solution adapter for lingam's RCD.

    Satisfies :class:`andrey_bench.contracts.SolutionAdapter`.
    """

    name: str = "lingam.rcd"
    package: str = "lingam"
    package_version: str = _lingam_version()
    backend: str = "native"
    mode: str = "serial"
    algorithm: str = "rcd"
    output_type: str = "pag"
    #: The endpoint marks that carry the pairs this method attributes to a hidden common cause.
    confounded_mark: str = "bidirected"

    def setup(self) -> None:
        """Import lingam and the shared conversion before timing; see `contracts.SetupAdapter`."""
        import lingam  # noqa: F401

        from andrey.api._irregular import _adapt_rcd  # noqa: F401

    def params(self) -> dict[str, Any]:
        """Return the search settings: ancestor-set size and the three significance levels."""
        return dict(_DEFAULTS)

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run lingam RCD; return the PAG's endpoint-mark array. The only timed region.

        Runner parameters override the adapter's settings. The conversion is the same call that
        converts Andrey's RCD result. The returned matrix uses ``NULL=0``, ``TAIL=1``,
        ``ARROW=2``, and ``CIRCLE=3``.
        """
        import lingam

        from andrey.api._irregular import _adapt_rcd

        X = np.asarray(data, dtype=np.float64)
        model = lingam.RCD(**{**_DEFAULTS, **params}).fit(X)
        return np.asarray(_adapt_rcd(model).structure.to_numpy())

    def to_structure(self, native: np.ndarray) -> Any:
        """Rebuild the PAG from its endpoint marks outside the timed region."""
        from andrey import GraphStructure

        return GraphStructure.from_numpy(native, kind=self.output_type)
