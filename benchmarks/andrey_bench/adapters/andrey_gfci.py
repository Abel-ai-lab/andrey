"""Benchmark Andrey GFCI: a BIC-scored GES search, then FCI's tests and orientation rules.

The settings are the latent lane's: Fisher-Z at ``alpha=0.05`` and the standard BIC penalty,
``lambda_value=1.0`` and ``ebic_gamma=0.0``, on numpy. ``andrey.gfci.numpy`` runs in one process;
``andrey.gfci.numpy.parallel`` runs the GES step in the fit's cores as worker processes and
returns the same PAG. Fits return PAG endpoint marks, preserving circles and arrowheads for
endpoint-aware scoring.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from andrey_bench.adapters.andrey_fci import _andrey_version
from andrey_bench.adapters.andrey_ges import _parallel_workers
from andrey_bench.adapters.bic_penalty import ANDREY_EBIC_GAMMA

_DEFAULT_ALPHA = 0.05
_DEFAULT_INDEP_TEST = "fisherz"
_DEFAULT_SCORE_FUNC = "local_score_BIC"
_DEFAULT_LAMBDA = 1.0


class AndreyGFCI:
    """GFCI solution adapter; ``mode="parallel"`` gives the GES step the fit's cores."""

    package: str = "andrey"
    package_version: str = _andrey_version()
    backend: str = "numpy"
    algorithm: str = "gfci"
    output_type: str = "pag"

    def __init__(self, mode: str = "serial") -> None:
        if mode not in ("serial", "parallel"):
            raise ValueError(f"mode must be 'serial' or 'parallel', not {mode!r}")
        self.mode = mode
        self.name = "andrey.gfci.numpy" + ("" if mode == "serial" else ".parallel")

    def setup(self) -> None:
        """Import Andrey and its GFCI implementation before timing; see `contracts.SetupAdapter`."""
        import andrey  # noqa: F401
        from andrey.search.gfci import gfci  # noqa: F401

    def params(self) -> dict[str, Any]:
        """Return the score and test settings."""
        return {
            "alpha": _DEFAULT_ALPHA,
            "indep_test": _DEFAULT_INDEP_TEST,
            "score_func": _DEFAULT_SCORE_FUNC,
            "lambda_value": _DEFAULT_LAMBDA,
            "ebic_gamma": ANDREY_EBIC_GAMMA,
        }

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run Andrey GFCI; return the PAG's endpoint-mark array. The only timed region."""
        import andrey

        workers = 1 if self.mode == "serial" else _parallel_workers()
        with andrey.config(num_workers=workers):
            out = andrey.gfci(
                np.asarray(data, dtype=np.float64),
                alpha=float(params.get("alpha", _DEFAULT_ALPHA)),
                indep_test=params.get("indep_test", _DEFAULT_INDEP_TEST),
                score_func=params.get("score_func", _DEFAULT_SCORE_FUNC),
                lambda_value=float(params.get("lambda_value", _DEFAULT_LAMBDA)),
                ebic_gamma=float(params.get("ebic_gamma", ANDREY_EBIC_GAMMA)),
            )
        return np.asarray(out.structure.to_numpy())

    def to_structure(self, native: np.ndarray) -> Any:
        """Rebuild the PAG from its endpoint marks outside the timed region."""
        from andrey import GraphStructure

        return GraphStructure.from_numpy(native, kind=self.output_type)
