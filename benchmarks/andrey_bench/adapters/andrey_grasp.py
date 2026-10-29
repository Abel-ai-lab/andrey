"""GRaSP benchmark adapter for Andrey."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from andrey_bench.adapters.bic_penalty import (
    ANDREY_EBIC_GAMMA,
    ANDREY_LAMBDA,
    SEARCH_SEED,
    andrey_score,
)

_DEFAULT_SCORE_FUNC = "local_score_BIC_from_cov"
#: causal-learn's default tuck-relaxation depth, and Andrey's.
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
            "ebic_gamma": ANDREY_EBIC_GAMMA,
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
        ebic_gamma = float(params.get("ebic_gamma", ANDREY_EBIC_GAMMA))
        depth = int(params.get("depth", _DEFAULT_DEPTH))
        seed = int(params.get("seed", SEARCH_SEED))

        X = np.asarray(data, dtype=np.float64)
        with andrey.config(backend=self.backend, n_jobs=1):
            out = andrey.grasp(
                X,
                score_func=andrey_score(score_func),
                penalty_discount=lambda_value,
                ebic_gamma=ebic_gamma,
                grasp_depth=depth,
                seed=seed,
            )
        return np.asarray(out.structure.to_numpy())

    def to_structure(self, native: np.ndarray) -> Any:
        """Rebuild a ``GraphStructure`` directly from endpoint marks. Untimed."""
        from andrey import GraphStructure

        return GraphStructure.from_numpy(native, kind=self.output_type)


class AndreyGRaSPBDeu(AndreyGRaSP):
    """Andrey GRaSP with the BDeu score, on category codes.

    BDeu takes its default priors (equivalent sample size 1, one expected parent per variable) and
    no penalty weight, so ``params`` names only the score, the depth, and the seed.
    """

    name: str = "andrey.grasp.bdeu"

    def params(self) -> dict[str, Any]:
        """The BDeu score at the default depth, on a fixed search seed."""
        return {"score_func": "local_score_BDeu", "depth": _DEFAULT_DEPTH, "seed": SEARCH_SEED}


class AndreyGRaSPCategoricalBIC(AndreyGRaSP):
    """Andrey GRaSP with the multinomial BIC, on category codes, at the standard penalty."""

    name: str = "andrey.grasp.bic_categorical"

    def params(self) -> dict[str, Any]:
        """The multinomial BIC at ``lambda_value=1`` and the default depth, on a fixed seed."""
        return {
            "score_func": "local_score_BIC_categorical",
            "lambda_value": 1.0,
            "depth": _DEFAULT_DEPTH,
            "seed": SEARCH_SEED,
        }


class AndreyGRaSPDegenGauss(AndreyGRaSP):
    """Andrey GRaSP with the degenerate Gaussian score at the standard penalty, every column named
    categorical: the mixed-data score on category codes."""

    name: str = "andrey.grasp.degen_gauss"

    def params(self) -> dict[str, Any]:
        """The degenerate Gaussian score at ``lambda_value=1`` and the default depth."""
        return {
            "score_func": "local_score_degen_gauss",
            "lambda_value": 1.0,
            "depth": _DEFAULT_DEPTH,
            "seed": SEARCH_SEED,
        }

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run GRaSP with every column named categorical and return CPDAG endpoint marks."""
        import andrey

        X = np.asarray(data, dtype=np.float64)
        with andrey.config(backend=self.backend, n_jobs=1):
            out = andrey.grasp(
                X,
                score_func=andrey_score(str(params["score_func"])),
                penalty_discount=float(params["lambda_value"]),
                grasp_depth=int(params["depth"]),
                seed=int(params["seed"]),
                categorical=list(range(X.shape[1])),
            )
        return np.asarray(out.structure.to_numpy())
