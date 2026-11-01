"""Andrey CAM-UV solution adapter for nonlinear additive data with hidden variables.

Calls ``andrey.camuv(X, alpha=0.01, max_depth=2)``, Andrey's defaults and the paper's settings. The
result is a PAG over the observed variables: ``i -> j`` for a direct cause and ``i o-o j`` for a
pair joined through a hidden variable, the pairs of ``metadata["latent_pairs"]``.
:mod:`~andrey_bench.adapters.lingam_camuv` and :mod:`~andrey_bench.adapters.causal_learn_camuv`
pass the same settings to the reference implementations, which mark their pairs ``<->``.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from andrey_bench.adapters.andrey_rcd import _andrey_version

#: The settings, Andrey's defaults: the HSIC significance level and the largest conditioning set of
#: a sink test.
DEFAULTS = {"alpha": 0.01, "max_depth": 2}


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
    #: The endpoint marks that carry the pairs this method attributes to a hidden variable.
    confounded_mark: str = "circle"

    def setup(self) -> None:
        """Import Andrey and its CAM-UV engine before timing; see `contracts.SetupAdapter`."""
        import andrey  # noqa: F401
        from andrey.fcm.camuv import camuv  # noqa: F401

    def params(self) -> dict[str, Any]:
        """Return the HSIC significance level and the largest conditioning set of a sink test."""
        return dict(DEFAULTS)

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run Andrey CAM-UV; return the PAG's endpoint-mark array. The only timed region.

        The returned matrix uses ``NULL=0``, ``TAIL=1``, ``ARROW=2``, and ``CIRCLE=3``.
        """
        import andrey

        out = andrey.camuv(
            np.asarray(data, dtype=np.float64),
            alpha=float(params["alpha"]),
            max_depth=int(params["max_depth"]),
        )
        return np.asarray(out.structure.to_numpy())

    def to_structure(self, native: np.ndarray) -> Any:
        """Rebuild the PAG from its endpoint marks outside the timed region."""
        from andrey import GraphStructure

        return GraphStructure.from_numpy(native, kind=self.output_type)
