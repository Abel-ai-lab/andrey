"""causal-learn GRaSP benchmark adapter."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from andrey_bench.adapters.bic_penalty import CAUSAL_LEARN_LAMBDA, SEARCH_SEED
from andrey_bench.adapters.causal_learn_convert import generalgraph_to_adjacency

_DEFAULT_SCORE_FUNC = "local_score_BIC_from_cov"
#: causal-learn's default tuck-relaxation depth and the only value Andrey implements.
_DEFAULT_DEPTH = 3


def _causal_learn_version() -> str:
    """Return the installed causal-learn distribution version."""
    from importlib import metadata

    try:
        return metadata.version("causal-learn")
    except metadata.PackageNotFoundError:  # pragma: no cover - installed in bench-env
        return "unknown"


class CausalLearnGRaSP:
    """Benchmark adapter for causal-learn GRaSP."""

    name: str = "causal-learn.grasp"
    package: str = "causal-learn"
    package_version: str = _causal_learn_version()
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "grasp"
    output_type: str = "cpdag"

    def setup(self) -> None:
        """Import causal-learn GRaSP before timing; see `contracts.SetupAdapter`.

        The adapter is unpickled in the child, and `fit` imports lazily. Setup excludes that
        import from the first timed fit at `--warmup 0`.
        """
        if not hasattr(np, "mat"):
            np.mat = np.asmatrix  # type: ignore[attr-defined]
        from causallearn.search.PermutationBased.GRaSP import grasp  # noqa: F401

    def params(self) -> dict[str, Any]:
        """Covariance BIC at the campaign's log-likelihood-unit penalty, on a fixed search seed."""
        return {
            "score_func": _DEFAULT_SCORE_FUNC,
            "lambda_value": CAUSAL_LEARN_LAMBDA,
            "depth": _DEFAULT_DEPTH,
            "seed": SEARCH_SEED,
        }

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Return a 0/1 CPDAG adjacency matrix. The only timed region.

        ``verbose=False`` suppresses a cached edge count after each DFS round. Unlike BOSS, it does
        not avoid extra search work.

        GRaSP shuffles through the global ``random`` module and has no seed argument. Seed it for
        each child-process fit.
        """
        import random

        # causal-learn imports `np.mat`, which NumPy 2 removed.
        if not hasattr(np, "mat"):
            np.mat = np.asmatrix  # type: ignore[attr-defined]
        from causallearn.search.PermutationBased.GRaSP import grasp as _grasp

        score_func = params.get("score_func", _DEFAULT_SCORE_FUNC)
        lambda_value = float(params.get("lambda_value", CAUSAL_LEARN_LAMBDA))
        depth = int(params.get("depth", _DEFAULT_DEPTH))
        seed = int(params.get("seed", SEARCH_SEED))

        X = np.asarray(data, dtype=np.float64)
        random.seed(seed)
        graph = _grasp(
            X,
            score_func=score_func,
            depth=depth,
            parameters={"lambda_value": lambda_value},
            verbose=False,
        )

        return generalgraph_to_adjacency(graph.graph)

    def to_structure(self, native: np.ndarray) -> Any:
        """Convert a 0/1 CPDAG adjacency to an Andrey ``GraphStructure``."""
        from andrey_bench.contracts import structure_from_adjacency

        return structure_from_adjacency(native, kind=self.output_type)
