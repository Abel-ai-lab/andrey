"""Adapters for lingam's VARMALiNGAM and LongitudinalLiNGAM, in an environment of their own.

VARMALiNGAM runs ``lingam.VARMALiNGAM(order=(1, 1), criterion=None, prune=True,
random_state=0).fit(X)`` and returns ``adjacency_matrices_``, the ``(psis, omegas)`` pair, as one
``[B0, B1, M1]`` stack. ``criterion`` and ``prune`` are passed because lingam's defaults
(``"bic"`` and ``True``) differ from Andrey's; the settings match
:mod:`~andrey_bench.adapters.andrey_lingam_ts`. LongitudinalLiNGAM runs
``lingam.LongitudinalLiNGAM(n_lags=1, measure="pwling", random_state=0).fit(list(panel))`` on a
``(T, M, n)`` panel and returns ``adjacency_matrices_``, ``(T, 2, n, n)``, ``NaN`` at the first time
point. Both stacks use lingam's ``[effect, cause]`` layout (:mod:`andrey_bench.coefficients`).

**It runs in an environment of its own.** lingam 1.13.0, the release that ships both estimators,
does not install on the Python the bench environment uses, so ``benchmarks/start_lingam_ts.sh``
builds ``.venv-lingam-ts`` on Python 3.12, this adapter declares ``python_env``, and the runner
starts its fits there. Only ``fit`` runs in that environment; :meth:`to_structure` and scoring run
in the parent. This module imports nothing but numpy at import time, because the worker in that
environment unpickles the adapter from it.
"""

from __future__ import annotations

import functools
import os
import subprocess
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

#: Name of the environment variable holding the interpreter lingam's fits run in.
PYTHON_ENV_VAR = "ANDREY_BENCH_LINGAM_TS_PYTHON"


@functools.lru_cache(maxsize=None)
def _lingam_version(python: str) -> str:
    """lingam's version, as the interpreter that will run the fits reports it."""
    try:
        out = subprocess.run(
            [python, "-c", "import importlib.metadata as m; print(m.version('lingam'))"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() if out.returncode == 0 else "unknown"


def _version() -> str:
    python = os.environ.get(PYTHON_ENV_VAR, "").strip()
    return _lingam_version(python) if python else "unknown"


@dataclass(frozen=True)
class LingamVarmaLiNGAM:
    """Benchmark adapter for ``lingam.VARMALiNGAM``; build it with :func:`lingam_varma_adapter`."""

    name: str = "lingam.varma_lingam"
    package: str = "lingam"
    package_version: str = ""
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "varma_lingam"
    output_type: str = "temporal"
    python_env: str = PYTHON_ENV_VAR

    def setup(self) -> None:
        """Import lingam before timing."""
        import lingam  # noqa: F401

    def params(self) -> dict[str, Any]:
        """The ``VARMALiNGAM`` arguments passed."""
        return {"order": [1, 1], "criterion": None, "prune": True, "random_state": 0}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run lingam VARMALiNGAM and return its ``(3, n, n)`` stack. Timed."""
        import lingam

        model = lingam.VARMALiNGAM(
            order=tuple(params["order"]),
            criterion=params["criterion"],
            prune=bool(params["prune"]),
            random_state=int(params["random_state"]),
        )
        model.fit(np.asarray(data, dtype=np.float64))
        psis, omegas = model.adjacency_matrices_
        return np.ascontiguousarray(np.concatenate([np.asarray(psis), np.asarray(omegas)]))

    def to_structure(self, native: np.ndarray) -> Any:
        """Stack -> ``CoefficientStack``. Runs in the parent; untimed."""
        from andrey_bench.coefficients import CoefficientStack

        return CoefficientStack(native, kind="varma")


@dataclass(frozen=True)
class LingamLongitudinalLiNGAM:
    """Benchmark adapter for ``lingam.LongitudinalLiNGAM``; build it with
    :func:`lingam_longitudinal_adapter`."""

    name: str = "lingam.longitudinal_lingam"
    package: str = "lingam"
    package_version: str = ""
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "longitudinal_lingam"
    output_type: str = "temporal"
    python_env: str = PYTHON_ENV_VAR

    def setup(self) -> None:
        """Import lingam before timing."""
        import lingam  # noqa: F401

    def params(self) -> dict[str, Any]:
        """The ``LongitudinalLiNGAM`` arguments passed."""
        return {"n_lags": 1, "measure": "pwling", "random_state": 0}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run lingam LongitudinalLiNGAM on a ``(T, M, n)`` panel; return its stack. Timed."""
        import lingam

        model = lingam.LongitudinalLiNGAM(
            n_lags=int(params["n_lags"]),
            measure=str(params["measure"]),
            random_state=int(params["random_state"]),
        )
        model.fit(list(np.asarray(data, dtype=np.float64)))
        return np.ascontiguousarray(np.asarray(model.adjacency_matrices_, dtype=np.float64))

    def to_structure(self, native: np.ndarray) -> Any:
        """Stack -> ``CoefficientStack``. Runs in the parent; untimed."""
        from andrey_bench.coefficients import CoefficientStack

        return CoefficientStack(native, kind="panel")


def lingam_varma_adapter() -> LingamVarmaLiNGAM:
    """A :class:`LingamVarmaLiNGAM` carrying its environment's lingam version."""
    return LingamVarmaLiNGAM(package_version=_version())


def lingam_longitudinal_adapter() -> LingamLongitudinalLiNGAM:
    """A :class:`LingamLongitudinalLiNGAM` carrying its environment's lingam version."""
    return LingamLongitudinalLiNGAM(package_version=_version())
