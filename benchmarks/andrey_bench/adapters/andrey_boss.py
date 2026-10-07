"""BOSS benchmark adapter for Andrey."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from andrey_bench.adapters.bic_penalty import ANDREY_LAMBDA, SEARCH_SEED

_DEFAULT_SCORE_FUNC = "local_score_BIC_from_cov"


def _andrey_version() -> str:
    """Return the installed Andrey version."""
    from importlib import metadata

    try:
        return metadata.version("andrey")
    except metadata.PackageNotFoundError:  # editable/dev checkout with no dist metadata
        import andrey

        return getattr(andrey, "__version__", "unknown")


class AndreyBOSS:
    """Benchmark adapter for Andrey BOSS."""

    name: str = "andrey.boss.numpy"
    package: str = "andrey"
    package_version: str = _andrey_version()
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "boss"
    output_type: str = "cpdag"

    def setup(self) -> None:
        """Import Andrey and its BOSS implementation before timing.

        `andrey.boss` loads its search module on first call. Setup excludes that import
        from `fit`, including the first timed fit at `--warmup 0`.
        See `contracts.SetupAdapter`.
        """
        import andrey  # noqa: F401
        from andrey.search.boss import boss  # noqa: F401

    def params(self) -> dict[str, Any]:
        """Covariance BIC at the campaign's deviance-unit penalty, on a fixed search seed."""
        return {
            "score_func": _DEFAULT_SCORE_FUNC,
            "lambda_value": ANDREY_LAMBDA,
            "seed": SEARCH_SEED,
        }

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run BOSS and return CPDAG endpoint marks. The only timed region.

        Pin ``numpy`` because ``BICScore`` dispatches covariance work. Otherwise an available
        accelerator could run part of a CPU-labeled fit.
        """
        import andrey

        score_func = params.get("score_func", _DEFAULT_SCORE_FUNC)
        lambda_value = float(params.get("lambda_value", ANDREY_LAMBDA))
        seed = int(params.get("seed", SEARCH_SEED))

        X = np.asarray(data, dtype=np.float64)
        with andrey.config(backend=self.backend, num_workers=1):
            out = andrey.boss(X, score_func=score_func, lambda_value=lambda_value, seed=seed)
        return np.asarray(out.structure.to_numpy())

    def to_structure(self, native: np.ndarray) -> Any:
        """Rebuild a ``GraphStructure`` directly from endpoint marks. Untimed."""
        from andrey import GraphStructure

        return GraphStructure.from_numpy(native, kind=self.output_type)
