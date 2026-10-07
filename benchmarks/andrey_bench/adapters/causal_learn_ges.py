"""causal-learn GES benchmark adapter."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from andrey_bench.adapters.causal_learn_convert import generalgraph_to_adjacency

_DEFAULT_SCORE_FUNC = "local_score_BIC"
#: Matches Andrey's 1.0 penalty after conversion to log-likelihood units.
_DEFAULT_LAMBDA = 0.5
_DEFAULT_MAX_PARENTS = 4


def _causal_learn_version() -> str:
    """Return the installed causal-learn distribution version."""
    from importlib import metadata

    try:
        return metadata.version("causal-learn")
    except metadata.PackageNotFoundError:  # pragma: no cover - installed in bench-env
        return "unknown"


class CausalLearnGES:
    """Benchmark adapter for causal-learn GES."""

    name: str = "causal-learn.ges"
    package: str = "causal-learn"
    package_version: str = _causal_learn_version()
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "ges"
    output_type: str = "cpdag"

    def setup(self) -> None:
        """Import causal-learn before timing; see `contracts.SetupAdapter`.

        The adapter is unpickled in the child, and `fit` imports lazily. Setup excludes that
        import from the first timed fit at `--warmup 0`.
        """
        if not hasattr(np, "mat"):
            np.mat = np.asmatrix  # type: ignore[attr-defined]
        from causallearn.search.ScoreBased.GES import ges  # noqa: F401

    def params(self) -> dict[str, Any]:
        return {
            "score_func": _DEFAULT_SCORE_FUNC,
            "lambda_value": _DEFAULT_LAMBDA,
            "max_parents": _DEFAULT_MAX_PARENTS,
        }

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Return a 0/1 CPDAG adjacency matrix. The only timed region."""
        # causal-learn imports np.mat, which NumPy 2 removed.
        if not hasattr(np, "mat"):
            np.mat = np.asmatrix  # type: ignore[attr-defined]
        from causallearn.search.ScoreBased.GES import ges as _ges

        score_func = params.get("score_func", _DEFAULT_SCORE_FUNC)
        lambda_value = float(params.get("lambda_value", _DEFAULT_LAMBDA))
        max_parents = params.get("max_parents", params.get("maxP", _DEFAULT_MAX_PARENTS))

        X = np.asarray(data, dtype=np.float64)
        record = _ges(
            X,
            score_func=score_func,
            maxP=max_parents,
            parameters={"lambda_value": lambda_value},
        )

        return generalgraph_to_adjacency(record["G"].graph)

    def to_structure(self, native: np.ndarray) -> Any:
        """Convert a 0/1 CPDAG adjacency to an Andrey ``GraphStructure``."""
        from andrey_bench.contracts import structure_from_adjacency

        return structure_from_adjacency(native, kind=self.output_type)
