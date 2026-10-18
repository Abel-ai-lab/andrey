"""pcalg GIES adapter for interventional data.

`pcalg::gies` maximizes an L0-penalized Gaussian likelihood through `GaussL0penIntScore`. Its default
penalty `lambda = 0.5 * log(n)`, with `n` every row of the data, is BIC in log-likelihood units: the
objective Andrey's `gies` minimizes at `lambda_value=1.0` in deviance units. The score's default
`intercept = FALSE` fits each structural equation through the origin; the adapter sets
`intercept = TRUE`, as Andrey centers each variable on the rows it is scored on.

The fit takes the samples and the 0/1 intervention matrix side by side
(:func:`andrey_bench.adapters.andrey_gies.stack`). Inside the timed region, R turns each row's set of
intervened variables into pcalg's `targets`, one entry per distinct set, and `target.index`, the
entry of each row. `gies` runs its defaults: the forward, backward, and turning phases, repeated
until none improves, and no degree cap (`maxDegree`).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from andrey_bench import rsession
from andrey_bench.adapters.andrey_gies import unstack
from andrey_bench.adapters.r_adapter import RAdapter

_DEFAULT_LAMBDA_RULE = "0.5*log(n)"
_DEFAULT_INTERCEPT = True

_R_SOURCE = """
andrey_pcalg_gies <- function(X, M, intercept) {
    rows <- apply(M, 1, function(r) paste(which(r != 0), collapse = ","))
    settings <- unique(rows)
    targets <- lapply(strsplit(settings, ","), as.integer)
    score <- new("GaussL0penIntScore", X, targets = targets,
                 target.index = match(rows, settings), intercept = intercept)
    fit <- pcalg::gies(score)
    # Row -> column 0/1 matrix with an undirected edge written both ways, named from the graph's own
    # node list, as in the GES adapter.
    m <- as(fit$essgraph, "matrix") * 1
    dimnames(m) <- list(fit$essgraph$.nodes, fit$essgraph$.nodes)
    m
}
"""


class PcalgGIES(RAdapter):
    """pcalg GIES adapter for interventional data.

    Satisfies :class:`~andrey_bench.contracts.SolutionAdapter` and
    :class:`~andrey_bench.contracts.SetupAdapter`.
    """

    name: str = "pcalg.gies"
    package: str = "pcalg"
    package_version: str = rsession.package_version("pcalg")
    backend: str = "native"
    mode: str = "serial"
    algorithm: str = "gies"
    output_type: str = "cpdag"

    r_package: str = "pcalg"
    r_source: str = _R_SOURCE
    r_function: str = "andrey_pcalg_gies"

    def params(self) -> dict[str, Any]:
        """Declare BIC with an intercept, pcalg's three iterated phases, and no degree cap."""
        return {
            "score": "GaussL0penIntScore",
            "lambda_rule": _DEFAULT_LAMBDA_RULE,
            "intercept": _DEFAULT_INTERCEPT,
            "max_degree": None,
            "phase": ["forward", "backward", "turning"],
            "iterate": True,
        }

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run pcalg GIES and return the 0/1 essential-graph adjacency. The only timed region."""
        X, mask = unstack(data)
        session = rsession.session()
        result = session.function(self.r_function)(
            session.matrix(X),
            session.matrix(mask.astype(np.float64)),
            bool(params.get("intercept", _DEFAULT_INTERCEPT)),
        )
        return session.square_matrix(result, names=rsession.columns(X.shape[1])).astype(np.int8)

    def to_structure(self, native: np.ndarray) -> Any:
        """Convert the 0/1 adjacency to an Andrey `GraphStructure`; untimed."""
        from andrey_bench.contracts import structure_from_adjacency

        return structure_from_adjacency(native, kind=self.output_type)
