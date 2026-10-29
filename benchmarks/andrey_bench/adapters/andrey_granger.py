"""Andrey Granger solution adapter for multivariate time series.

Wraps ``andrey.granger``, the ordinary least-squares fit of each variable on the past ``maxlag``
steps of every variable, with a test of each lagged coefficient at ``alpha``. The fit returns the
coefficients in causal-learn's ``granger_lasso`` layout, a ``(d, d * maxlag)`` matrix whose
``[i, j + d * (k - 1)]`` entry weights ``x_j(t - k)`` in ``x_i(t)``, so every solution is scored
from the same encoding. A coefficient that is not significant is zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

_DEFAULTS = {"maxlag": 2}


def _andrey_version() -> str:
    """The installed Andrey version (distribution metadata, else the module attribute)."""
    from importlib import metadata

    try:
        return metadata.version("andrey")
    except metadata.PackageNotFoundError:  # editable/dev checkout with no dist metadata
        import andrey

        return getattr(andrey, "__version__", "unknown")


@dataclass(frozen=True)
class AndreyGranger:
    """Solution adapter for Andrey's Granger causality.

    Satisfies :class:`andrey_bench.contracts.SolutionAdapter`.
    """

    name: str = "andrey.granger.numpy"
    mode: str = "serial"
    package: str = "andrey"
    package_version: str = _andrey_version()
    backend: str = "numpy"
    algorithm: str = "granger"
    output_type: str = "temporal"

    def setup(self) -> None:
        """Import Andrey and its Granger implementation before timing.

        The facade loads `andrey.temporal.granger` on its first call. Setup excludes that import
        from `fit`, including the first timed fit at `--warmup 0`. See `contracts.SetupAdapter`.
        """
        import andrey  # noqa: F401
        from andrey.temporal.granger import granger_f_test  # noqa: F401

    def params(self) -> dict[str, Any]:
        """Return the fit settings: the lags."""
        return dict(_DEFAULTS)

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run Andrey Granger; return the ``(d, d * maxlag)`` coefficients. The only timed region."""
        import andrey

        settings = {**_DEFAULTS, **params}
        out = andrey.granger(np.asarray(data, dtype=np.float64), tau_max=int(settings["maxlag"]))
        # lag_weights[k][j, i] weights x_j(t - k - 1) -> x_i(t); the layout wants it at [i, j + d*k]
        return np.concatenate([w.T for w in out.structure.lag_weights], axis=1)

    def to_structure(self, native: np.ndarray) -> Any:
        """Rebuild the temporal structure from the coefficients outside the timed region."""
        from andrey.api._temporal import _adapt_granger

        d = native.shape[0]
        return _adapt_granger(native, d, native.shape[1] // d).structure


def andrey_granger_adapters() -> list[AndreyGranger]:
    """The adapter, as a list for the lane."""
    return [AndreyGranger()]
