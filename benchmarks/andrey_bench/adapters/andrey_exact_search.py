"""Andrey exact-search adapter: the BIC-optimal DAG by A* over the order graph.

causal-learn's `bic_exact_search` is the counterpart. Both score a node by
`n * log(RSS / n) + n_parents * log(n)`, Andrey on centered columns and causal-learn on the columns
as given; on the harness's standardized data the two scores agree. Both search with A* and no
parent cap. The result is one DAG of the optimal class, so compare `mec_shd` across packages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from andrey_bench.adapters.andrey_hc import _andrey_version

_DEFAULT_SEARCH_METHOD = "astar"


@dataclass(frozen=True)
class AndreyExactSearch:
    """Benchmark adapter for Andrey exact search."""

    name: str = "andrey.exact_search.numpy"
    backend: str = "numpy"
    mode: str = "serial"
    package: str = "andrey"
    package_version: str = _andrey_version()
    algorithm: str = "exact_search"
    output_type: str = "dag"

    def setup(self) -> None:
        """Import Andrey's exact search before timing, outside `fit`."""
        from andrey.search.exact import exact_search  # noqa: F401

    def params(self) -> dict[str, Any]:
        """A* over the order graph."""
        return {"search_method": _DEFAULT_SEARCH_METHOD}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run the search and return DAG endpoint marks. The only timed region.

        Calls the search directly, so the facade's experimental warning and input checks stay
        outside the timing.
        """
        from andrey.search.exact import exact_search

        X = np.asarray(data, dtype=np.float64)
        dag = exact_search(X, search_method=params.get("search_method", _DEFAULT_SEARCH_METHOD))
        return np.asarray(dag.to_numpy())

    def to_structure(self, native: np.ndarray) -> Any:
        """Rebuild a ``GraphStructure`` from endpoint marks; untimed."""
        from andrey import GraphStructure

        return GraphStructure.from_numpy(native, kind=self.output_type)
