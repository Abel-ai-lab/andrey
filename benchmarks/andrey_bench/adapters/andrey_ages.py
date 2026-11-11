"""Andrey AGES adapter: the aggregated graph of the GES solution path.

pcalg's `ages` is the counterpart. Both start the path at BIC, a penalty of `log(n) / 2` per
parameter in log-likelihood units, Andrey's `penalty_discount=1.0` in deviance units, and neither
limits a node's parents.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from andrey_bench.adapters.andrey_hc import _andrey_version

_DEFAULT_LAMBDA = 1.0


@dataclass(frozen=True)
class AndreyAGES:
    """Benchmark adapter for Andrey AGES."""

    name: str = "andrey.ages.numpy"
    backend: str = "numpy"
    mode: str = "serial"
    package: str = "andrey"
    package_version: str = _andrey_version()
    algorithm: str = "ages"
    output_type: str = "cpdag"

    def setup(self) -> None:
        """Import Andrey's AGES before timing, outside `fit`."""
        from andrey.search.ages import ages  # noqa: F401

    def params(self) -> dict[str, Any]:
        """BIC in deviance units as the lowest penalty of the path."""
        return {"lambda_value": _DEFAULT_LAMBDA}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run AGES and return the endpoint marks of its graph. The only timed region.

        Calls the search directly, so the facade's input checks stay outside the timing.
        """
        import andrey
        from andrey.search.ages import ages

        X = np.asarray(data, dtype=np.float64)
        with andrey.config(backend=self.backend):
            graph, _penalties, _taken = ages(
                X, penalty_discount=float(params.get("lambda_value", _DEFAULT_LAMBDA))
            )
        return np.asarray(graph.to_numpy())

    def to_structure(self, native: np.ndarray) -> Any:
        """Rebuild a ``GraphStructure`` from endpoint marks; untimed."""
        from andrey import GraphStructure

        return GraphStructure.from_numpy(native, kind=self.output_type)
