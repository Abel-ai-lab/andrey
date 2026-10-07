"""bnlearn tabu-search adapter - hill climbing that can leave a local optimum.

`bnlearn::tabu` uses the same score-based search and objective as `hc`. It keeps a list of recently
visited structures and accepts a worse move rather than stopping at a local optimum. Its output
is a DAG; compare it through the MEC metrics used for `hc`, rather than raw `shd`.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from andrey_bench import rsession
from andrey_bench.adapters.r_adapter import RAdapter

_DEFAULT_SCORE = "bic-g"
#: No parent cap, matching bnlearn's default and Andrey's uncapped hill climbing.
_DEFAULT_MAX_PARENTS = None
#: bnlearn's defaults: a tabu list of 10 structures, and 10 moves without improvement end the
#: search.
_DEFAULT_TABU = 10
_DEFAULT_MAX_TABU = 10

_R_SOURCE = """
andrey_bnlearn_tabu <- function(X, score, maxp, tabu, max_tabu) {
    fit <- bnlearn::tabu(
        as.data.frame(X), score = score, maxp = maxp, tabu = tabu, max.tabu = max_tabu
    )
    bnlearn::amat(fit)
}
"""


class BnlearnTabu(RAdapter):
    """bnlearn tabu-search adapter for R greedy score-based search.

    Satisfies :class:`~andrey_bench.contracts.SolutionAdapter` and
    :class:`~andrey_bench.contracts.SetupAdapter`.
    """

    name: str = "bnlearn.tabu"
    package: str = "bnlearn"
    package_version: str = rsession.package_version("bnlearn")
    backend: str = "native"
    mode: str = "serial"
    algorithm: str = "tabu"
    output_type: str = "dag"

    r_package: str = "bnlearn"
    r_source: str = _R_SOURCE
    r_function: str = "andrey_bnlearn_tabu"

    def params(self) -> dict[str, Any]:
        """Gaussian BIC, no parent cap, and bnlearn's tabu settings; `None` is an unbounded
        `maxp`."""
        return {
            "score": _DEFAULT_SCORE,
            "max_parents": _DEFAULT_MAX_PARENTS,
            "tabu": _DEFAULT_TABU,
            "max_tabu": _DEFAULT_MAX_TABU,
        }

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run bnlearn tabu and return 0/1 DAG adjacency. The only timed region."""
        cap = params.get("max_parents", _DEFAULT_MAX_PARENTS)
        adjacency = self.call(
            data,
            str(params.get("score", _DEFAULT_SCORE)),
            float("inf") if cap is None else float(cap),
            int(params.get("tabu", _DEFAULT_TABU)),
            int(params.get("max_tabu", _DEFAULT_MAX_TABU)),
        )
        return adjacency.astype(np.int8)

    def to_structure(self, native: np.ndarray) -> Any:
        """Convert 0/1 DAG adjacency to an Andrey `GraphStructure`; untimed."""
        from andrey_bench.contracts import structure_from_adjacency

        return structure_from_adjacency(native, kind=self.output_type)
