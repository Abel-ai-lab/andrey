"""Adapter for CausalNex's DYNOTEARS, ``from_pandas_dynamic``, in an environment of its own.

The adapter takes Andrey's settings and passes CausalNex the same model:

| Andrey (:func:`andrey.dynotears`) | CausalNex 0.12.2 |
|---|---|
| ``tau_max`` | ``p`` |
| ``lambda1`` | ``lambda_w`` and ``lambda_a``, both set to it |
| ``w_threshold`` | ``w_threshold=0.0``, then same-step weights kept where ``abs(w) >= w_threshold`` |
| ``lag_threshold`` | lagged weights kept where ``abs(a) >= lag_threshold`` |
| ``max_iter=100``, ``h_tol=1e-8`` | ``max_iter=100``, ``h_tol=1e-8``, CausalNex's defaults |

CausalNex's defaults differ from the paper's: ``lambda_w = lambda_a = 0.1`` and one threshold,
``w_threshold=0.0``, for both blocks. So the fit runs without a threshold and the adapter applies
the two, CausalNex's own rule (a weight below the threshold in magnitude is set to 0) at each block's
value. Two differences stay: CausalNex fits ``X`` and the lagged ``Y`` as given, with no intercept,
where Andrey centers every column first; and CausalNex raises the penalty weight up to ``1e20`` and
stops on ``max_iter`` or ``h_tol`` alone, where Andrey stops at ``rho_max=1e16`` and then removes
the weakest same-step edges until no cycle is left. ``centered=True`` centers ``X`` and ``Y`` on
their column means, built by CausalNex's ``DynamicDataTransformer``, and fits them with
``from_numpy_dynamic``, so the two fits solve the same problem.

``fit`` returns the ``(p + 1, n, n)`` window marks of :mod:`andrey_bench.window`, and raises
``RuntimeError`` when the kept same-step edges form a cycle, which a DAG's marks cannot hold.

**It runs in an environment of its own.** CausalNex 0.12.2 pins Python below 3.11, NumPy below 1.24,
and pandas below 2, so ``benchmarks/start_causalnex.sh`` builds ``.venv-causalnex``, this adapter
declares ``python_env``, and the runner starts its fits there. Only ``fit`` runs in that
environment; :meth:`to_structure` and scoring run in the parent. This module imports nothing but
numpy at import time, because the worker in that environment unpickles the adapter from it.
"""

from __future__ import annotations

import functools
import os
import subprocess
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

#: Name of the environment variable holding the interpreter CausalNex's fits run in.
PYTHON_ENV_VAR = "ANDREY_BENCH_CAUSALNEX_PYTHON"


@functools.lru_cache(maxsize=None)
def _causalnex_version(python: str) -> str:
    """CausalNex's version, as the interpreter that will run the fits reports it."""
    try:
        out = subprocess.run(
            [python, "-c", "import importlib.metadata as m; print(m.version('causalnex'))"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() if out.returncode == 0 else "unknown"


def _version() -> str:
    python = os.environ.get(PYTHON_ENV_VAR, "").strip()
    return _causalnex_version(python) if python else "unknown"


@dataclass(frozen=True)
class CausalnexDYNOTEARS:
    """Benchmark adapter for CausalNex's DYNOTEARS; build it with :func:`causalnex_adapter`."""

    name: str
    lambda1: float
    w_threshold: float
    lag_threshold: float
    tau_max: int = 1
    centered: bool = False
    package: str = "causalnex"
    package_version: str = ""
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "dynotears"
    output_type: str = "temporal"
    python_env: str = PYTHON_ENV_VAR

    def setup(self) -> None:
        """Import CausalNex's DYNOTEARS before timing."""
        import causalnex.structure.dynotears  # noqa: F401

    def params(self) -> dict[str, Any]:
        """The settings, in Andrey's names; :meth:`fit` maps them to CausalNex's."""
        return {
            "tau_max": self.tau_max,
            "lambda1": self.lambda1,
            "w_threshold": self.w_threshold,
            "lag_threshold": self.lag_threshold,
            "centered": self.centered,
        }

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run CausalNex DYNOTEARS and return its ``(p + 1, n, n)`` window marks. Timed."""
        import pandas as pd
        from causalnex.structure.dynotears import from_numpy_dynamic, from_pandas_dynamic
        from causalnex.structure.transformers import DynamicDataTransformer

        from andrey_bench.window import marks_from_edges

        x = np.asarray(data, dtype=np.float64)
        d, p, penalty = x.shape[1], int(params["tau_max"]), float(params["lambda1"])
        frame = pd.DataFrame(x, columns=[str(j) for j in range(d)])
        common = {"lambda_w": penalty, "lambda_a": penalty, "max_iter": 100, "h_tol": 1e-8}
        if params["centered"]:
            now, past = DynamicDataTransformer(p=p).fit_transform(frame, return_df=False)
            model = from_numpy_dynamic(
                now - now.mean(axis=0), past - past.mean(axis=0), w_threshold=0.0, **common
            )
        else:
            model = from_pandas_dynamic(frame, p=p, w_threshold=0.0, **common)
        # Nodes are named "<column>_lag<k>"; every edge ends at lag 0.
        weights = np.zeros((p + 1, d, d))
        for cause, effect, weight in model.edges.data("weight"):
            column, lag = cause.rsplit("_lag", 1)
            target, at = effect.rsplit("_lag", 1)
            if at != "0":
                raise RuntimeError(f"CausalNex returned an edge into {effect}")
            weights[int(lag), int(column), int(target)] = weight
        thresholds = np.r_[params["w_threshold"], np.full(p, params["lag_threshold"])]
        kept = (weights != 0) & (np.abs(weights) >= thresholds[:, None, None])
        return marks_from_edges(kept[0], kept[1:])

    def to_structure(self, native: np.ndarray) -> Any:
        """Window marks -> ``WindowGraph`` with a DAG at lag 0. Runs in the parent; untimed."""
        from andrey_bench.window import WindowGraph

        return WindowGraph(native, lag0="dag")


def causalnex_adapter(**fields: Any) -> CausalnexDYNOTEARS:
    """A :class:`CausalnexDYNOTEARS` carrying its environment's CausalNex version."""
    return CausalnexDYNOTEARS(**fields, package_version=_version())
