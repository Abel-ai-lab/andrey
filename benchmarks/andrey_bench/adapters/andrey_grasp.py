"""GRaSP benchmark adapter for Andrey."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from andrey_bench.adapters.bic_penalty import ANDREY_LAMBDA, SEARCH_SEED

_DEFAULT_SCORE_FUNC = "local_score_BIC_from_cov"
#: causal-learn's default tuck-relaxation depth and the only value Andrey implements.
_DEFAULT_DEPTH = 3


def _andrey_version() -> str:
    """Return the installed Andrey version."""
    from importlib import metadata

    try:
        return metadata.version("andrey")
    except metadata.PackageNotFoundError:  # editable/dev checkout with no dist metadata
        import andrey

        return getattr(andrey, "__version__", "unknown")


class AndreyGRaSP:
    """Benchmark adapter for Andrey GRaSP."""

    name: str = "andrey.grasp.numpy"
    package: str = "andrey"
    package_version: str = _andrey_version()
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "grasp"
    output_type: str = "cpdag"

    def setup(self) -> None:
        """Import Andrey and its GRaSP implementation before timing.

        `andrey.grasp` loads its search module on first call. Setup excludes that import
        from `fit`, including the first timed fit at `--warmup 0`.
        See `contracts.SetupAdapter`.
        """
        import andrey  # noqa: F401
        from andrey.search.grasp import grasp  # noqa: F401

    def params(self) -> dict[str, Any]:
        """Covariance BIC at the campaign's deviance-unit penalty, on a fixed search seed."""
        return {
            "score_func": _DEFAULT_SCORE_FUNC,
            "lambda_value": ANDREY_LAMBDA,
            "depth": _DEFAULT_DEPTH,
            "seed": SEARCH_SEED,
        }

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run GRaSP and return CPDAG endpoint marks. The only timed region.

        Pin ``numpy`` because ``BICScore`` dispatches covariance work. Otherwise an available
        accelerator could run part of a CPU-labeled fit.
        """
        import andrey

        score_func = params.get("score_func", _DEFAULT_SCORE_FUNC)
        lambda_value = float(params.get("lambda_value", ANDREY_LAMBDA))
        depth = int(params.get("depth", _DEFAULT_DEPTH))
        seed = int(params.get("seed", SEARCH_SEED))

        X = np.asarray(data, dtype=np.float64)
        with andrey.config(backend=self.backend, num_workers=1):
            out = andrey.grasp(
                X, score_func=score_func, lambda_value=lambda_value, depth=depth, seed=seed
            )
        return np.asarray(out.structure.to_numpy())

    def to_structure(self, native: np.ndarray) -> Any:
        """Rebuild a ``GraphStructure`` directly from endpoint marks. Untimed."""
        from andrey import GraphStructure

        return GraphStructure.from_numpy(native, kind=self.output_type)
