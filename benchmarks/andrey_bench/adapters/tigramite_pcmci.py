"""Adapter for tigramite's PCMCI with the analytic partial-correlation test.

Calls tigramite's public API only:
``PCMCI(dataframe=DataFrame(X), cond_ind_test=ParCorr(significance="analytic"), verbosity=0)``
then ``run_pcmci(tau_min=1, tau_max=..., pc_alpha=0.2, alpha_level=0.05, fdr_method="none")``.
``tau_min=1`` tests lags ``1..tau_max`` as Andrey's PCMCI does; tigramite's own ``tau_min=0`` also
tests lag 0, which is PCMCI+'s scope. ``pc_alpha`` and ``alpha_level`` are passed rather than left
to the package, so both packages run the same settings; every other ``run_pcmci`` argument keeps
its default. ``ParCorr``'s analytic p-value is Student's t with ``T - |Z| - 2`` degrees of freedom,
the test of the ``andrey.pcmci.student_t`` solutions.

**It runs in an environment of its own.** tigramite is GPL-licensed: it is installed only in
``.venv-tigramite`` (``benchmarks/start_tigramite.sh``), this adapter declares ``python_env``, and
the runner starts its fits there. Only ``fit`` runs in that environment. It returns a plain link
stack, and :meth:`to_structure` and scoring run in the parent, so both packages share one graph
model and one scorer. This module imports nothing but numpy at import time, because the worker in
that environment unpickles the adapter from it.

Orientation: ``results["graph"]`` has shape ``(N, N, tau_max + 1)``, and ``graph[i, j, tau] ==
"-->"`` is the lagged link ``X_i(t-tau) -> X_j(t)``. The link stack is
``links[tau - 1, i, j]``, the layout of Andrey's adapter.
"""

from __future__ import annotations

import functools
import os
import subprocess
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

#: Name of the environment variable holding the interpreter tigramite fits run in.
PYTHON_ENV_VAR = "ANDREY_BENCH_TIGRAMITE_PYTHON"

#: The settings both packages run; the values of ``andrey_pcmci``'s defaults.
DEFAULT_TAU_MAX = 2
DEFAULT_PC_ALPHA = 0.2
DEFAULT_ALPHA_LEVEL = 0.05

#: The only marks ``run_pcmci`` may put on a lagged link at ``tau_min=1``: none, or a directed link.
_LAGGED_MARKS = {"", "-->"}


def links_from_graph(graph: np.ndarray) -> np.ndarray:
    """Convert ``run_pcmci``'s ``(N, N, tau_max + 1)`` graph to the ``(tau_max, N, N)`` link stack.

    The lag-0 slice is dropped: at ``tau_min=1`` it is not tested. Raises ``RuntimeError`` when a
    lagged entry is neither empty nor ``"-->"``, so a change in tigramite's encoding fails the fit
    instead of being read as no link.
    """
    lagged = np.asarray(graph)[:, :, 1:]
    unexpected = set(np.unique(lagged).tolist()) - _LAGGED_MARKS
    if unexpected:
        raise RuntimeError(f"tigramite marked lagged links {sorted(unexpected)}")
    return (lagged == "-->").transpose(2, 0, 1).astype(np.int8)


@functools.lru_cache(maxsize=None)
def _tigramite_version(python: str) -> str:
    """tigramite's distribution version, as the interpreter that will run the fits reports it.

    Read out of that environment: tigramite is not installed in the parent.
    """
    try:
        out = subprocess.run(
            [python, "-c", "import importlib.metadata as m; print(m.version('tigramite'))"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() if out.returncode == 0 else "unknown"


@dataclass(frozen=True)
class TigramitePCMCI:
    """Benchmark adapter for tigramite's PCMCI with ``ParCorr(significance="analytic")``.

    Satisfies :class:`andrey_bench.contracts.SolutionAdapter`, plus the optional ``python_env``.
    Build it with :func:`tigramite_adapter`, which names the lag window and reads the version.
    """

    name: str = f"tigramite.pcmci.parcorr.tau{DEFAULT_TAU_MAX}"
    tau_max: int = DEFAULT_TAU_MAX
    package: str = "tigramite"
    package_version: str = ""
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "pcmci"
    output_type: str = "temporal"
    python_env: str = PYTHON_ENV_VAR

    def setup(self) -> None:
        """Import tigramite before timing; see `contracts.SetupAdapter`."""
        from tigramite import data_processing  # noqa: F401
        from tigramite.independence_tests.parcorr import ParCorr  # noqa: F401
        from tigramite.pcmci import PCMCI  # noqa: F401

    def params(self) -> dict[str, Any]:
        """The ``run_pcmci`` arguments passed, and the test."""
        return {
            "tau_min": 1,
            "tau_max": self.tau_max,
            "pc_alpha": DEFAULT_PC_ALPHA,
            "alpha_level": DEFAULT_ALPHA_LEVEL,
            "fdr_method": "none",
            "cond_ind_test": "ParCorr",
            "significance": "analytic",
        }

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run tigramite PCMCI and return the ``(tau_max, n, n)`` int8 link stack. Timed.

        The graph is read by :func:`links_from_graph`.
        """
        from tigramite import data_processing
        from tigramite.independence_tests.parcorr import ParCorr
        from tigramite.pcmci import PCMCI

        pcmci = PCMCI(
            dataframe=data_processing.DataFrame(np.asarray(data, dtype=np.float64)),
            cond_ind_test=ParCorr(significance=str(params["significance"])),
            verbosity=0,
        )
        results = pcmci.run_pcmci(
            tau_min=int(params["tau_min"]),
            tau_max=int(params["tau_max"]),
            pc_alpha=float(params["pc_alpha"]),
            alpha_level=float(params["alpha_level"]),
            fdr_method=str(params["fdr_method"]),
        )
        return links_from_graph(results["graph"])

    def to_structure(self, native: np.ndarray) -> Any:
        """Link stack -> ``TemporalStructure``. Runs in the parent; untimed."""
        from andrey_bench.contracts import temporal_from_links

        return temporal_from_links(native)


def tigramite_adapter(tau_max: int = DEFAULT_TAU_MAX) -> TigramitePCMCI:
    """A :class:`TigramitePCMCI` named for its lag window, carrying its environment's version.

    The version is read with a subprocess, so it is resolved when a driver builds its solution
    list rather than at import.
    """
    python = os.environ.get(PYTHON_ENV_VAR, "").strip()
    return TigramitePCMCI(
        name=f"tigramite.pcmci.parcorr.tau{tau_max}",
        tau_max=tau_max,
        package_version=_tigramite_version(python) if python else "unknown",
    )
