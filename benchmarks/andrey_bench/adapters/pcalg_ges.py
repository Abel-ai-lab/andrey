"""pcalg GES adapter for R score-based search.

`pcalg::ges` maximizes an L0-penalized Gaussian likelihood through `GaussL0penObsScore`, whose
default penalty `lambda = 0.5 * log(n)` is plain BIC in log-likelihood units - the same objective
Andrey's GES optimizes at `lambda_value=1.0` in deviance units.

pcalg caps vertex degree, which has no setting equivalent to the `max_parents=4` parent cap in
the Andrey and causal-learn adapters. The adapter declares pcalg's default of no degree cap.

`ges` runs pcalg's default search: forward, backward, and turning phases, repeated until none
improves. The other GES adapters run forward then backward once, so pcalg's does more search;
`params` declares both settings.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from andrey_bench import rsession
from andrey_bench.adapters.r_adapter import RAdapter

#: pcalg's default BIC in log-likelihood units, matching the other GES adapters.
_DEFAULT_LAMBDA_RULE = "0.5*log(n)"
#: `pcalg::ges` defaults: three phases, repeated until none improves.
_DEFAULT_PHASE = ("forward", "backward", "turning")
_DEFAULT_ITERATE = True

_R_SOURCE = """
andrey_pcalg_ges <- function(X, phase, iterate) {
    fit <- pcalg::ges(
        new("GaussL0penObsScore", X), phase = strsplit(phase, ",")[[1]], iterate = iterate
    )
    # An EssGraph converts to a 0/1 matrix that is already row -> column, with an undirected edge
    # written both ways, but the coercion drops the dimnames. They are taken from the graph's own
    # node list rather than from `colnames(X)`: reattaching the input's order would *assert* the
    # result is in it, which is exactly the check `square_matrix` exists to perform.
    m <- as(fit$essgraph, "matrix") * 1
    dimnames(m) <- list(fit$essgraph$.nodes, fit$essgraph$.nodes)
    m
}
"""


class PcalgGES(RAdapter):
    """pcalg GES adapter for R score-based search.

    Satisfies :class:`~andrey_bench.contracts.SolutionAdapter` and
    :class:`~andrey_bench.contracts.SetupAdapter`.
    """

    name: str = "pcalg.ges"
    package: str = "pcalg"
    package_version: str = rsession.package_version("pcalg")
    backend: str = "native"
    mode: str = "serial"
    algorithm: str = "ges"
    output_type: str = "cpdag"

    r_package: str = "pcalg"
    r_source: str = _R_SOURCE
    r_function: str = "andrey_pcalg_ges"

    def params(self) -> dict[str, Any]:
        """Declare pcalg's defaults: BIC, no degree cap, and its three-phase iterated search."""
        return {
            "score": "GaussL0penObsScore",
            "lambda_rule": _DEFAULT_LAMBDA_RULE,
            "max_degree": None,
            "phase": list(_DEFAULT_PHASE),
            "iterate": _DEFAULT_ITERATE,
        }

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run pcalg GES and return 0/1 CPDAG adjacency. The only timed region."""
        phase = ",".join(params.get("phase", _DEFAULT_PHASE))
        iterate = bool(params.get("iterate", _DEFAULT_ITERATE))
        return self.call(data, phase, iterate).astype(np.int8)

    def to_structure(self, native: np.ndarray) -> Any:
        """Convert 0/1 CPDAG adjacency to an Andrey `GraphStructure`; untimed."""
        from andrey_bench.contracts import structure_from_adjacency

        return structure_from_adjacency(native, kind=self.output_type)
