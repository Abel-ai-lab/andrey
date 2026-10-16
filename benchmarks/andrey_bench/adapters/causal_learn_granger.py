"""causal-learn Granger adapter for multivariate time series.

Calls ``Granger(maxlag=..., cv=...).granger_lasso(X)``, which returns a ``(d, d * maxlag)``
coefficient matrix whose ``[i, j + d * (k - 1)]`` entry weights ``x_j(t - k)`` in ``x_i(t)``. A
nonzero coefficient is a lagged edge.

causal-learn fits the lasso on the data as given. Andrey standardizes each variable first and
returns the coefficients in the data's units, so its edges do not depend on the units; on
standardized data the two fit the same model.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

_DEFAULTS = {"maxlag": 2, "cv": 5}


def _causal_learn_version() -> str:
    """Return the installed causal-learn distribution version."""
    from importlib import metadata

    try:
        return metadata.version("causal-learn")
    except metadata.PackageNotFoundError:  # pragma: no cover - installed in bench-env
        return "unknown"


class CausalLearnGranger:
    """Solution adapter for causal-learn's lasso Granger causality.

    Satisfies :class:`andrey_bench.contracts.SolutionAdapter`.
    """

    name: str = "causal-learn.granger"
    package: str = "causal-learn"
    package_version: str = _causal_learn_version()
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "granger"
    output_type: str = "temporal"

    def setup(self) -> None:
        """Import causal-learn before timing; see `contracts.SetupAdapter`."""
        from causallearn.search.Granger.Granger import Granger  # noqa: F401

    def params(self) -> dict[str, Any]:
        """Return the fit settings: the lags and the cross-validation folds."""
        return dict(_DEFAULTS)

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run causal-learn's lasso Granger; return its coefficients. The only timed region."""
        from causallearn.search.Granger.Granger import Granger

        settings = {**_DEFAULTS, **params}
        model = Granger(maxlag=int(settings["maxlag"]), cv=int(settings["cv"]))
        return np.asarray(model.granger_lasso(np.asarray(data, dtype=np.float64)))

    def to_structure(self, native: np.ndarray) -> Any:
        """Rebuild the temporal structure from the coefficients outside the timed region."""
        from andrey.api._temporal import _adapt_granger

        d = native.shape[0]
        return _adapt_granger(native, d, native.shape[1] // d).structure
