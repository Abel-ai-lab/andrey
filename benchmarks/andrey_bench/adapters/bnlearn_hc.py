"""bnlearn hill-climbing adapter for R greedy score-based search.

`bnlearn::hc` returns a **DAG**, not a CPDAG: one member of the equivalence class rather than the
class. Its `output_type` declares this, and `andrey.metrics.score` reports both the directed scores
and the MEC pair (`mec_shd`, `mec_arrowhead_f1`) computed through the DAG's essential graph. Those
two are the columns comparable with a GES CPDAG; the raw `shd` is not.

`score="bic-g"` is Gaussian BIC with bnlearn's default `k = log(n)/2`, matching the objective of
pcalg's `GaussL0penObsScore` and Andrey's `BICScore`. `maxp` caps parents, in the same units
as the `max_parents=4` setting in the Andrey and causal-learn GES adapters.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from andrey_bench import rsession
from andrey_bench.adapters.r_adapter import RAdapter

_DEFAULT_SCORE = "bic-g"
#: No parent cap, matching bnlearn's default and Andrey's uncapped hill climbing.
_DEFAULT_MAX_PARENTS = None

_R_SOURCE = """
andrey_bnlearn_hc <- function(X, score, maxp) {
    fit <- bnlearn::hc(as.data.frame(X), score = score, maxp = maxp)
    # bnlearn's amat is 0/1, row -> column, and carries the data frame's names.
    bnlearn::amat(fit)
}
"""


class BnlearnHC(RAdapter):
    """bnlearn hill-climbing adapter for R greedy score-based search.

    Satisfies :class:`~andrey_bench.contracts.SolutionAdapter` and
    :class:`~andrey_bench.contracts.SetupAdapter`.
    """

    name: str = "bnlearn.hc"
    package: str = "bnlearn"
    package_version: str = rsession.package_version("bnlearn")
    backend: str = "native"
    mode: str = "serial"
    algorithm: str = "hc"
    output_type: str = "dag"

    r_package: str = "bnlearn"
    r_source: str = _R_SOURCE
    r_function: str = "andrey_bnlearn_hc"

    def params(self) -> dict[str, Any]:
        """Gaussian BIC with no parent cap; `None` denotes bnlearn's unbounded `maxp`."""
        return {"score": _DEFAULT_SCORE, "max_parents": _DEFAULT_MAX_PARENTS}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run bnlearn hc and return 0/1 DAG adjacency. The only timed region."""
        cap = params.get("max_parents", _DEFAULT_MAX_PARENTS)
        adjacency = self.call(
            data,
            str(params.get("score", _DEFAULT_SCORE)),
            float("inf") if cap is None else float(cap),
        )
        return adjacency.astype(np.int8)

    def to_structure(self, native: np.ndarray) -> Any:
        """Convert 0/1 DAG adjacency to an Andrey `GraphStructure`; untimed."""
        from andrey_bench.contracts import structure_from_adjacency

        return structure_from_adjacency(native, kind=self.output_type)
