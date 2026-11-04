"""Andrey DYNOTEARS solution adapter.

Calls ``andrey.dynotears(X, tau_max=..., lambda1=..., w_threshold=..., lag_threshold=...)`` with
the rest at Andrey's defaults (``max_iter=100``, ``h_tol=1e-8``, ``rho_max=1e16``).
:mod:`~andrey_bench.adapters.causalnex_dynotears` passes the same settings to CausalNex.

``fit`` returns the ``(tau_max + 1, n, n)`` window marks of :mod:`andrey_bench.window`: lag 0 the
same-step DAG and lag ``k`` the links ``X_i(t-k) -> X_j(t)``, self-links included, read from the
result's ``TemporalStructure``.

The settings are fields, and a driver names each solution for its settings: ``params`` are not part
of a ``run_id``, so two settings under one name would share a part file and a resume would keep one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from andrey_bench.adapters.andrey_pcmci import ARROW, TAIL, _andrey_version


def window_from_temporal(structure: Any) -> np.ndarray:
    """The window marks of a ``TemporalStructure`` whose lags are ``0..L``, its lag 0 a DAG."""
    from andrey_bench.window import marks_from_edges

    lags = tuple(structure.lags)
    if lags != tuple(range(len(lags))) or len(lags) < 2:
        raise ValueError(f"a window holds lags 0..L with L >= 1, got lags {lags}")
    same_step = np.asarray(structure.lag(0).to_numpy())
    # The arrowhead of a lagged `i -> j` is the mark at j, `marks[j, i]`, so the links are the
    # transpose; a self-link is an arrowhead on the diagonal.
    links = [(np.asarray(structure.lag(k).to_numpy()) == ARROW).T for k in lags[1:]]
    return marks_from_edges((same_step == TAIL) & (same_step.T == ARROW), np.stack(links))


@dataclass(frozen=True)
class AndreyDYNOTEARS:
    """Solution adapter for Andrey's DYNOTEARS; satisfies ``contracts.SolutionAdapter``."""

    name: str
    lambda1: float
    w_threshold: float
    lag_threshold: float
    tau_max: int = 1
    package: str = "andrey"
    package_version: str = _andrey_version()
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "dynotears"
    output_type: str = "temporal"

    def setup(self) -> None:
        """Import Andrey and its DYNOTEARS engine before timing."""
        import andrey  # noqa: F401
        from andrey.temporal.dynotears import dynotears  # noqa: F401

    def params(self) -> dict[str, Any]:
        """The lag window, the penalty, and both thresholds."""
        return {
            "tau_max": self.tau_max,
            "lambda1": self.lambda1,
            "w_threshold": self.w_threshold,
            "lag_threshold": self.lag_threshold,
        }

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run Andrey DYNOTEARS and return its ``(tau_max + 1, n, n)`` window marks. Timed."""
        import andrey

        out = andrey.dynotears(
            np.asarray(data, dtype=np.float64),
            tau_max=int(params["tau_max"]),
            lambda1=float(params["lambda1"]),
            w_threshold=float(params["w_threshold"]),
            lag_threshold=float(params["lag_threshold"]),
        )
        return window_from_temporal(out.structure)

    def to_structure(self, native: np.ndarray) -> Any:
        """Window marks -> ``WindowGraph`` with a DAG at lag 0; untimed."""
        from andrey_bench.window import WindowGraph

        return WindowGraph(native, lag0="dag")
