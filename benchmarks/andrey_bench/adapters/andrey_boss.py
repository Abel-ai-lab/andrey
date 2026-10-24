"""BOSS benchmark adapter for Andrey."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from andrey_bench.adapters.bic_penalty import ANDREY_EBIC_GAMMA, ANDREY_LAMBDA, SEARCH_SEED

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
            "ebic_gamma": ANDREY_EBIC_GAMMA,
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
        ebic_gamma = float(params.get("ebic_gamma", ANDREY_EBIC_GAMMA))
        seed = int(params.get("seed", SEARCH_SEED))

        X = np.asarray(data, dtype=np.float64)
        with andrey.config(backend=self.backend, num_workers=1):
            out = andrey.boss(
                X,
                score_func=score_func,
                lambda_value=lambda_value,
                ebic_gamma=ebic_gamma,
                seed=seed,
            )
        return np.asarray(out.structure.to_numpy())

    def to_structure(self, native: np.ndarray) -> Any:
        """Rebuild a ``GraphStructure`` directly from endpoint marks. Untimed."""
        from andrey import GraphStructure

        return GraphStructure.from_numpy(native, kind=self.output_type)


class AndreyBOSSBDeu(AndreyBOSS):
    """Andrey BOSS with the BDeu score, on category codes.

    BDeu takes its default priors (equivalent sample size 1, one expected parent per variable) and
    no penalty weight, so ``params`` names only the score and the seed.
    """

    name: str = "andrey.boss.bdeu"

    def params(self) -> dict[str, Any]:
        """The BDeu score on a fixed search seed."""
        return {"score_func": "local_score_BDeu", "seed": SEARCH_SEED}


class AndreyBOSSCategoricalBIC(AndreyBOSS):
    """Andrey BOSS with the multinomial BIC, on category codes, at the standard penalty."""

    name: str = "andrey.boss.bic_categorical"

    def params(self) -> dict[str, Any]:
        """The multinomial BIC at ``lambda_value=1``, on a fixed search seed."""
        return {
            "score_func": "local_score_BIC_categorical",
            "lambda_value": 1.0,
            "seed": SEARCH_SEED,
        }


class AndreyBOSSDegenGauss(AndreyBOSS):
    """Andrey BOSS with the degenerate Gaussian score at the standard penalty, every column named
    categorical: the mixed-data score on category codes."""

    name: str = "andrey.boss.degen_gauss"

    def params(self) -> dict[str, Any]:
        """The degenerate Gaussian score at ``lambda_value=1``, on a fixed search seed."""
        return {"score_func": "local_score_degen_gauss", "lambda_value": 1.0, "seed": SEARCH_SEED}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run BOSS with every column named categorical and return CPDAG endpoint marks."""
        import andrey

        X = np.asarray(data, dtype=np.float64)
        with andrey.config(backend=self.backend, num_workers=1):
            out = andrey.boss(
                X,
                score_func=str(params["score_func"]),
                lambda_value=float(params["lambda_value"]),
                seed=int(params["seed"]),
                categorical=list(range(X.shape[1])),
            )
        return np.asarray(out.structure.to_numpy())
