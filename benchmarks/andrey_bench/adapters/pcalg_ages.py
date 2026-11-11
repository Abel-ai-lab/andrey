"""pcalg AGES adapter, the implementation of the method's authors.

`pcalg::ages` aggregates the solution path of GES for every penalty from `lambda_min` up. Its
default `lambda_min = 0.5 * log(n)` is BIC in log-likelihood units, Andrey's
`penalty_discount=1.0` in deviance units. The adapter runs the defaults otherwise: the plain search
(`adaptive = "none"`) and no degree cap.

The returned graph is a partially directed graph that can direct more edges than a CPDAG; it is
read as the other pcalg graphs are, a 0/1 matrix with an undirected edge written both ways.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from andrey_bench import rsession
from andrey_bench.adapters.r_adapter import RAdapter

_DEFAULT_LAMBDA_RULE = "0.5*log(n)"

_R_SOURCE = """
andrey_pcalg_ages <- function(X) {
    fit <- pcalg::ages(X, labels = colnames(X))
    # Row -> column 0/1 matrix with an undirected edge written both ways, named from the graph's own
    # node list, as in the GES adapter.
    m <- as(fit$essgraph, "matrix") * 1
    dimnames(m) <- list(fit$essgraph$.nodes, fit$essgraph$.nodes)
    m
}
"""


class PcalgAGES(RAdapter):
    """pcalg AGES adapter.

    Satisfies :class:`~andrey_bench.contracts.SolutionAdapter` and
    :class:`~andrey_bench.contracts.SetupAdapter`.
    """

    name: str = "pcalg.ages"
    package: str = "pcalg"
    package_version: str = rsession.package_version("pcalg")
    backend: str = "native"
    mode: str = "serial"
    algorithm: str = "ages"
    output_type: str = "cpdag"

    r_package: str = "pcalg"
    r_source: str = _R_SOURCE
    r_function: str = "andrey_pcalg_ages"

    def params(self) -> dict[str, Any]:
        """Declare pcalg's defaults: the path from BIC up, the plain search, no degree cap."""
        return {"lambda_min_rule": _DEFAULT_LAMBDA_RULE, "adaptive": "none", "max_degree": None}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run pcalg AGES and return the 0/1 adjacency of its graph. The only timed region."""
        return self.call(data).astype(np.int8)

    def to_structure(self, native: np.ndarray) -> Any:
        """Convert the 0/1 adjacency to an Andrey `GraphStructure`; untimed."""
        from andrey_bench.contracts import structure_from_adjacency

        return structure_from_adjacency(native, kind=self.output_type)
