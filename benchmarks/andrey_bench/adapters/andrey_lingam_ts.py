"""Andrey VARMA-LiNGAM and longitudinal LiNGAM solution adapters.

VARMA-LiNGAM calls ``andrey.varma_lingam(X, order=(1, 1), criterion=None, prune=True)``: one
autoregressive and one moving-average lag, no order search, and the lag blocks re-estimated by
adaptive Lasso, which sets small weights to zero, so a link is a nonzero weight.
:mod:`~andrey_bench.adapters.lingam_ts` passes the same settings to lingam. Longitudinal LiNGAM
calls ``andrey.longitudinal_lingam(data, tau_max=1, measure="pwling")``, Andrey's defaults and
lingam's.

``fit`` returns the coefficient stack of :mod:`andrey_bench.coefficients` in the ``[effect, cause]``
layout both packages report: ``[B0, B1, M1]`` for VARMA-LiNGAM, and ``[B(t,t), B(t,t-1)]`` for
every time point for longitudinal LiNGAM, ``NaN`` at the first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from andrey_bench.adapters.andrey_pcmci import _andrey_version


@dataclass(frozen=True)
class AndreyVarmaLiNGAM:
    """Solution adapter for Andrey's VARMA-LiNGAM; satisfies ``contracts.SolutionAdapter``."""

    name: str = "andrey.varma_lingam.numpy"
    package: str = "andrey"
    package_version: str = _andrey_version()
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "varma_lingam"
    output_type: str = "temporal"

    def setup(self) -> None:
        """Import Andrey and its VARMA-LiNGAM engine before timing."""
        import andrey  # noqa: F401
        from andrey.temporal.varma import varma_lingam  # noqa: F401

    def params(self) -> dict[str, Any]:
        """The order, the order search, and the pruning."""
        return {"order": [1, 1], "criterion": None, "prune": True}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run Andrey VARMA-LiNGAM and return its ``(3, n, n)`` stack. The timed region."""
        import andrey

        out = andrey.varma_lingam(
            np.asarray(data, dtype=np.float64),
            order=tuple(params["order"]),
            criterion=params["criterion"],
            prune=bool(params["prune"]),
        )
        structure = out.structure
        # The structure stores [cause, effect]; the stack is [effect, cause].
        ar = np.asarray(structure.lag_weights).transpose(0, 2, 1)
        ma = np.asarray(structure.lag_weights_ma).transpose(0, 2, 1)
        return np.ascontiguousarray(np.concatenate([ar, ma]))

    def to_structure(self, native: np.ndarray) -> Any:
        """Stack -> ``CoefficientStack``; untimed."""
        from andrey_bench.coefficients import CoefficientStack

        return CoefficientStack(native, kind="varma")


@dataclass(frozen=True)
class AndreyLongitudinalLiNGAM:
    """Solution adapter for Andrey's longitudinal LiNGAM; satisfies ``contracts.SolutionAdapter``."""

    name: str = "andrey.longitudinal_lingam.numpy"
    package: str = "andrey"
    package_version: str = _andrey_version()
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "longitudinal_lingam"
    output_type: str = "temporal"

    def setup(self) -> None:
        """Import Andrey and its longitudinal LiNGAM engine before timing."""
        import andrey  # noqa: F401
        from andrey.temporal.longitudinal import LongitudinalLiNGAM  # noqa: F401

    def params(self) -> dict[str, Any]:
        """The lags and the independence measure."""
        return {"n_lags": 1, "measure": "pwling"}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run Andrey longitudinal LiNGAM on a ``(T, M, n)`` panel; return its stack. Timed."""
        import andrey

        panel = np.asarray(data, dtype=np.float64)
        out = andrey.longitudinal_lingam(
            list(panel), tau_max=int(params["n_lags"]), measure=str(params["measure"])
        )
        # time_weights covers time points 1..T-1 as [cause, effect]; the first is not estimated.
        weights = np.asarray(out.structure.time_weights).transpose(0, 1, 3, 2)
        first = np.full((1, *weights.shape[1:]), np.nan)
        return np.ascontiguousarray(np.concatenate([first, weights]))

    def to_structure(self, native: np.ndarray) -> Any:
        """Stack -> ``CoefficientStack``; untimed."""
        from andrey_bench.coefficients import CoefficientStack

        return CoefficientStack(native, kind="panel")
