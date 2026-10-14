"""Adapter for tigramite's PC on non-time-series data with its CMIknn or GPDC test.

tigramite (GPL-3.0) is the reference for Andrey's CMIknn and GPDC. It is called as a black box
through its public API, ``PCMCI(...).run_pcalg_non_timeseries_data(...)``; nothing of its source is
read or copied. Its setting names appear only here and in the benchmarks README, mapped to
Andrey's:

| Andrey | tigramite |
|---|---|
| ``alpha=0.05`` | ``pc_alpha=0.05`` |
| PC's collider rule (separating sets) | ``contemp_collider_rule="none"`` |
| CMIknn ``k_cmi=0.1`` | ``CMIknn(knn=0.1)`` |
| CMIknn ``k_perm=5`` | ``shuffle_neighbors=5`` |
| CMIknn ``n_permutations=500`` | ``sig_samples=500``, ``significance="shuffle_test"`` |
| CMIknn's rank transform | ``transform="ranks"`` |
| CMIknn shuffles single rows (no time order) | ``sig_blocklength=1`` |
| GPDC | ``GPDC(significance="analytic")`` |

Every pin matches Andrey's default, not tigramite's where the two differ (``knn=0.2``,
``pc_alpha=0.01``, the ``"majority"`` collider rule). tigramite sizes the blocks its shuffles move
from the data's autocorrelation, as for a time series; the rows here are independent, so the
block length is pinned to one row, which also avoids a NumPy 2 incompatibility in that estimate.

**It runs in an environment of its own** (``benchmarks/start_tigramite.sh``), so tigramite and
its dependencies stay out of bench-env. Only ``fit`` runs there: it returns a 0/1 CPDAG adjacency,
and :meth:`TigramitePC.to_structure` runs back in the parent.
"""

from __future__ import annotations

import functools
import subprocess
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

#: Name of the environment variable holding the interpreter tigramite fits run in.
PYTHON_ENV_VAR = "ANDREY_BENCH_TIGRAMITE_PYTHON"

_DEFAULT_ALPHA = 0.05

#: Andrey's settings for each test, at Andrey's defaults (tigramite's names are in ``fit``).
CI_TESTS: dict[str, dict[str, Any]] = {
    "cmiknn": {"k_cmi": 0.1, "k_perm": 5, "n_permutations": 500},
    "gpdc": {},
}


@functools.lru_cache(maxsize=None)
def _tigramite_version(python: str) -> str:
    """tigramite's distribution version in the interpreter that will run the fits (the module
    carries no ``__version__``)."""
    try:
        out = subprocess.run(
            [python, "-c", "from importlib import metadata; print(metadata.version('tigramite'))"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() if out.returncode == 0 else "unknown"


def graph_to_adjacency(graph: np.ndarray) -> np.ndarray:
    """tigramite's ``(N, N, 1)`` string graph -> a 0/1 CPDAG adjacency, ``adj[i, j] = 1`` for
    ``i -> j``.

    ``"-->"`` at ``[i, j]`` is ``i -> j`` and ``"<--"`` is ``j -> i``. An unoriented link
    (``"o-o"``) and a conflicting one (``"x-x"``) both become an undirected edge, a symmetric pair.
    """
    links = np.asarray(graph)[:, :, 0]
    d = links.shape[0]
    adj = np.zeros((d, d), dtype=np.int8)
    for i in range(d):
        for j in range(d):
            if i == j or links[i, j] == "":
                continue
            if links[i, j] == "-->":
                adj[i, j] = 1
            elif links[i, j] == "<--":
                adj[j, i] = 1
            elif links[i, j] in ("o-o", "x-x"):
                adj[i, j] = adj[j, i] = 1
            else:
                raise ValueError(f"unexpected link {links[i, j]!r} at ({i}, {j})")
    return adj


@dataclass(frozen=True)
class TigramitePC:
    """Benchmark adapter for tigramite's PC with CMIknn or GPDC.

    Satisfies :class:`andrey_bench.contracts.SolutionAdapter`, plus the optional ``python_env``.
    """

    indep_test: str = "cmiknn"
    name: str = ""
    package: str = "tigramite"
    package_version: str = ""
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "pc"
    output_type: str = "cpdag"
    python_env: str = PYTHON_ENV_VAR

    def __post_init__(self) -> None:
        if self.indep_test not in CI_TESTS:
            raise ValueError(f"Unsupported indep_test: {self.indep_test!r}")
        if not self.name:
            object.__setattr__(self, "name", f"tigramite.pc.{self.indep_test}")

    def params(self) -> dict[str, Any]:
        """``alpha`` and the test with its settings, in Andrey's names."""
        return {"alpha": _DEFAULT_ALPHA, "indep_test": self.indep_test, **CI_TESTS[self.indep_test]}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run tigramite's PC; return a 0/1 CPDAG adjacency. Timed, and run in its own env.

        Single-threaded: CMIknn's neighbor searches get ``workers=1``. Its shuffles move single
        rows (``sig_blocklength=1``).
        """
        from tigramite import data_processing
        from tigramite.pcmci import PCMCI

        settings = {**CI_TESTS[self.indep_test], **params}
        if self.indep_test == "cmiknn":
            from tigramite.independence_tests.cmiknn import CMIknn

            test = CMIknn(
                knn=float(settings["k_cmi"]),
                shuffle_neighbors=int(settings["k_perm"]),
                sig_samples=int(settings["n_permutations"]),
                significance="shuffle_test",
                transform="ranks",
                sig_blocklength=1,
                workers=1,
            )
        else:
            from tigramite.independence_tests.gpdc import GPDC

            test = GPDC(significance="analytic")

        X = np.asarray(data, dtype=np.float64)
        pcmci = PCMCI(dataframe=data_processing.DataFrame(X), cond_ind_test=test, verbosity=0)
        results = pcmci.run_pcalg_non_timeseries_data(
            pc_alpha=float(settings.get("alpha", _DEFAULT_ALPHA)), contemp_collider_rule="none"
        )
        return graph_to_adjacency(results["graph"])

    def to_structure(self, native: np.ndarray) -> Any:
        """0/1 CPDAG adjacency -> Andrey ``GraphStructure``. Runs in bench-env (parent); untimed."""
        from andrey_bench.contracts import structure_from_adjacency

        return structure_from_adjacency(native, kind=self.output_type)


def tigramite_pc_adapter(indep_test: str) -> TigramitePC:
    """A :class:`TigramitePC` carrying the version its own interpreter reports.

    The interpreter is read from :data:`PYTHON_ENV_VAR`; without it, the version is ``"unknown"``
    and the runner stops the run at launch, as for any unset declared environment.
    """
    import os

    python = os.environ.get(PYTHON_ENV_VAR, "")
    version = _tigramite_version(python) if python else "unknown"
    return TigramitePC(indep_test=indep_test, package_version=version)
