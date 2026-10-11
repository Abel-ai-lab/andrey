"""lingam BottomUpParceLiNGAM adapter for data with hidden common causes.

Calls ``lingam.BottomUpParceLiNGAM(...).fit(X)``. ``adjacency_matrix_`` is the weighted ``B`` of
``x = B x + e``, with ``NaN`` in both cells of a pair whose order is unknown. The matrix is
converted to PAG endpoint marks, with ``o-o`` at each ``NaN`` pair, by the function that converts
Andrey's result, so both solutions are scored from the same encoding.

The search is deterministic, so ``random_state`` stays unset and no seed applies.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

_DEFAULTS = {"alpha": 0.1}


def _lingam_version() -> str:
    """The installed lingam version (distribution metadata; the module exposes one too)."""
    from importlib import metadata

    try:
        return metadata.version("lingam")
    except metadata.PackageNotFoundError:  # pragma: no cover - installed in bench-env
        return "unknown"


class LingamBottomUpParceLiNGAM:
    """Solution adapter for lingam's BottomUpParceLiNGAM.

    Satisfies :class:`andrey_bench.contracts.SolutionAdapter`.
    """

    name: str = "lingam.bottom_up_parce"
    package: str = "lingam"
    package_version: str = _lingam_version()
    backend: str = "native"
    mode: str = "serial"
    algorithm: str = "bottom_up_parce"
    output_type: str = "pag"
    #: The endpoint marks that carry the pairs this method attributes to a hidden common cause.
    confounded_mark: str = "circle"

    def setup(self) -> None:
        """Import lingam and the shared conversion before timing; see `contracts.SetupAdapter`."""
        import lingam  # noqa: F401

        from andrey.api._irregular import _adapt_bottom_up_parce  # noqa: F401

    def params(self) -> dict[str, Any]:
        """Return the independence-test significance level; the search is deterministic."""
        return dict(_DEFAULTS)

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run lingam BottomUpParceLiNGAM; return the PAG's endpoint-mark array. The timed region.

        Runner parameters override the adapter's settings. The returned matrix uses ``NULL=0``,
        ``TAIL=1``, ``ARROW=2``, and ``CIRCLE=3``.
        """
        import lingam

        from andrey.api._irregular import _adapt_bottom_up_parce

        X = np.asarray(data, dtype=np.float64)
        model = lingam.BottomUpParceLiNGAM(**{**_DEFAULTS, **params}).fit(X)
        return np.asarray(_adapt_bottom_up_parce(model).structure.to_numpy())

    def to_structure(self, native: np.ndarray) -> Any:
        """Rebuild the PAG from its endpoint marks outside the timed region."""
        from andrey import GraphStructure

        return GraphStructure.from_numpy(native, kind=self.output_type)
