"""Andrey's PC, GES, and hill climbing on mixed data: the degenerate and conditional Gaussian tests
and scores.

Each adapter calls the public facade as a user would, without ``categorical``: Andrey infers the
categorical columns, every column of at most ``max_categories`` distinct whole numbers. On the
``mixed_er`` regime that reads exactly the generated categorical columns, since a categorical column
holds at most five codes and a continuous one a distinct value per row. The other packages in the
mixed lane read the columns by the same rule. GES runs uncapped, as the facade does, on the numpy
backend in one process; its mixed-data moves are scored in that process. Hill climbing starts from
the empty graph and has no move limit, as bnlearn's ``hc`` by default, and like bnlearn's conditional
linear Gaussian networks it gives no categorical node a continuous parent.
"""

from __future__ import annotations

import warnings
from typing import Any, Mapping

import numpy as np

from andrey_bench.adapters.bic_penalty import andrey_score

_DEFAULT_ALPHA = 0.05
_DEFAULT_LAMBDA = 1.0
#: The most distinct whole numbers a column may hold to be read as categorical: Andrey's default.
MAX_CATEGORIES = 10


def _andrey_version() -> str:
    from importlib import metadata

    try:
        return metadata.version("andrey")
    except metadata.PackageNotFoundError:  # editable/dev checkout with no dist metadata
        import andrey

        return getattr(andrey, "__version__", "unknown")


class _AndreyMixed:
    """Shared metadata and conversion for the mixed-data adapters."""

    package: str = "andrey"
    package_version: str = _andrey_version()
    backend: str = "numpy"
    mode: str = "serial"
    output_type: str = "cpdag"

    def setup(self) -> None:
        """Import Andrey and its searches before timing; see `contracts.SetupAdapter`."""
        import andrey  # noqa: F401
        from andrey.constraint.pc import pc  # noqa: F401
        from andrey.search.ges import ges  # noqa: F401
        from andrey.search.hc import hc  # noqa: F401

    def to_structure(self, native: np.ndarray) -> Any:
        """Endpoint-mark array -> ``GraphStructure``; untimed."""
        from andrey import GraphStructure

        return GraphStructure.from_numpy(native, kind=self.output_type)


class AndreyPCMixed(_AndreyMixed):
    """Andrey's PC with a mixed-data test, ``alpha=0.05``, categorical columns inferred."""

    algorithm: str = "pc"

    def __init__(self, indep_test: str) -> None:
        self.indep_test = indep_test
        self.name = f"andrey.pc.{indep_test}"

    def params(self) -> dict[str, Any]:
        """The test, its level, and the inference bound."""
        return {
            "alpha": _DEFAULT_ALPHA,
            "indep_test": self.indep_test,
            "max_categories": MAX_CATEGORIES,
        }

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run ``andrey.pc``; return the CPDAG's endpoint marks. The only timed region."""
        import andrey

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", andrey.DataTypeWarning)  # it names the inferred columns
            out = andrey.pc(
                np.asarray(data, dtype=np.float64),
                alpha=float(params.get("alpha", _DEFAULT_ALPHA)),
                indep_test=str(params.get("indep_test", self.indep_test)),
                max_categories=int(params.get("max_categories", MAX_CATEGORIES)),
            )
        return np.asarray(out.structure.to_numpy())


class AndreyGESMixed(_AndreyMixed):
    """Andrey's GES with a mixed-data score, the standard penalty, categorical columns inferred."""

    algorithm: str = "ges"

    def __init__(self, score_func: str) -> None:
        self.score_func = score_func
        self.name = f"andrey.ges.{score_func.removeprefix('local_score_')}"

    def params(self) -> dict[str, Any]:
        """The score, its BIC weight, and the inference bound."""
        return {
            "score_func": self.score_func,
            "lambda_value": _DEFAULT_LAMBDA,
            "max_categories": MAX_CATEGORIES,
        }

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run ``andrey.ges`` serially; return the CPDAG's endpoint marks. The only timed region."""
        import andrey

        with warnings.catch_warnings(), andrey.config(backend="numpy", num_workers=1):
            warnings.simplefilter("ignore", andrey.DataTypeWarning)
            out = andrey.ges(
                np.asarray(data, dtype=np.float64),
                score_func=andrey_score(str(params.get("score_func", self.score_func))),
                penalty_discount=float(params.get("lambda_value", _DEFAULT_LAMBDA)),
                max_categories=int(params.get("max_categories", MAX_CATEGORIES)),
            )
        return np.asarray(out.structure.to_numpy())


class AndreyHCMixed(_AndreyMixed):
    """Andrey's hill climbing with a mixed-data score: the empty start graph, no move limit."""

    algorithm: str = "hc"

    def __init__(self, score_func: str) -> None:
        self.score_func = score_func
        self.name = f"andrey.hc.{score_func.removeprefix('local_score_')}"

    def params(self) -> dict[str, Any]:
        """The score, its BIC weight, and the inference bound."""
        return {
            "score_func": self.score_func,
            "lambda_value": _DEFAULT_LAMBDA,
            "max_categories": MAX_CATEGORIES,
        }

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run ``andrey.hc``; return the CPDAG's endpoint marks. The only timed region."""
        import andrey

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", andrey.DataTypeWarning)
            out = andrey.hc(
                np.asarray(data, dtype=np.float64),
                score_func=andrey_score(str(params.get("score_func", self.score_func))),
                penalty_discount=float(params.get("lambda_value", _DEFAULT_LAMBDA)),
                max_categories=int(params.get("max_categories", MAX_CATEGORIES)),
            )
        return np.asarray(out.structure.to_numpy())


def andrey_mixed_adapters() -> list[_AndreyMixed]:
    """PC with each mixed-data test, then GES and hill climbing with each mixed-data score."""
    return [
        AndreyPCMixed("degen_gauss"),
        AndreyPCMixed("cond_gauss"),
        AndreyGESMixed("local_score_degen_gauss"),
        AndreyGESMixed("local_score_cond_gauss"),
        AndreyHCMixed("local_score_degen_gauss"),
        AndreyHCMixed("local_score_cond_gauss"),
    ]
