"""causal-learn exact-search adapter: `bic_exact_search`, the BIC-optimal DAG.

Runs causal-learn's A* (`search_method="astar"`) with its default optimal path extension, which
does not change the optimum, and no parent cap (`max_parents=None`): the settings of Andrey's
`exact_search`. The package scores the columns as given, without centering them.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from andrey_bench.adapters.causal_learn_ges import _causal_learn_version

_DEFAULT_SEARCH_METHOD = "astar"


class CausalLearnExactSearch:
    """Benchmark adapter for causal-learn's `bic_exact_search`."""

    name: str = "causal-learn.exact_search"
    package: str = "causal-learn"
    package_version: str = _causal_learn_version()
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "exact_search"
    output_type: str = "dag"

    def setup(self) -> None:
        """Import causal-learn's exact search before timing."""
        from causallearn.search.ScoreBased.ExactSearch import bic_exact_search  # noqa: F401

    def params(self) -> dict[str, Any]:
        """A* with the package's path extension and no parent cap."""
        return {
            "search_method": _DEFAULT_SEARCH_METHOD,
            "use_path_extension": True,
            "max_parents": None,
        }

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Return the 0/1 adjacency of the optimal DAG, row -> column. The only timed region."""
        from causallearn.search.ScoreBased.ExactSearch import bic_exact_search

        dag, _stats = bic_exact_search(
            np.asarray(data, dtype=np.float64),
            search_method=params.get("search_method", _DEFAULT_SEARCH_METHOD),
            use_path_extension=bool(params.get("use_path_extension", True)),
            max_parents=params.get("max_parents"),
        )
        return (np.asarray(dag) != 0).astype(np.int8)

    def to_structure(self, native: np.ndarray) -> Any:
        """Convert the 0/1 DAG adjacency to an Andrey ``GraphStructure``; untimed."""
        from andrey_bench.contracts import structure_from_adjacency

        return structure_from_adjacency(native, kind=self.output_type)
