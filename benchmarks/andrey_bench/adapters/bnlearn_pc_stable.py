"""bnlearn `pc.stable` adapter - the order-independent PC variant.

`pc.stable` fixes PC's dependence on the order the variables arrive in: each depth's tests are
decided against the adjacency the depth started with, so the skeleton does not change when the
columns are permuted. Its output is a CPDAG.

`test="zf"` selects Fisher-Z, matching the Andrey, causal-learn, and pcalg PC adapters.
bnlearn's default for continuous data is the exact t-test; using it would compare different tests.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from andrey_bench import rsession
from andrey_bench.adapters.r_adapter import RAdapter

_DEFAULT_ALPHA = 0.05
_DEFAULT_INDEP_TEST = "zf"

_R_SOURCE = """
andrey_bnlearn_pc_stable <- function(X, test, alpha) {
    fit <- bnlearn::pc.stable(as.data.frame(X), test = test, alpha = alpha)
    # 0/1, row -> column, with an undirected edge written both ways.
    bnlearn::amat(fit)
}
"""


class BnlearnPCStable(RAdapter):
    """bnlearn `pc.stable` adapter for R constraint-based search.

    Satisfies :class:`~andrey_bench.contracts.SolutionAdapter` and
    :class:`~andrey_bench.contracts.SetupAdapter`.
    """

    name: str = "bnlearn.pc-stable"
    package: str = "bnlearn"
    package_version: str = rsession.package_version("bnlearn")
    backend: str = "native"
    mode: str = "serial"
    algorithm: str = "pc"
    output_type: str = "cpdag"

    r_package: str = "bnlearn"
    r_source: str = _R_SOURCE
    r_function: str = "andrey_bnlearn_pc_stable"

    def params(self) -> dict[str, Any]:
        """Fisher-Z at `alpha=0.05`, matching the other constraint-based adapters."""
        return {"alpha": _DEFAULT_ALPHA, "indep_test": _DEFAULT_INDEP_TEST}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run bnlearn pc.stable and return 0/1 CPDAG adjacency. The only timed region."""
        adjacency = self.call(
            data,
            str(params.get("indep_test", _DEFAULT_INDEP_TEST)),
            float(params.get("alpha", _DEFAULT_ALPHA)),
        )
        return adjacency.astype(np.int8)

    def to_structure(self, native: np.ndarray) -> Any:
        """Convert 0/1 CPDAG adjacency to an Andrey `GraphStructure`; untimed."""
        from andrey_bench.contracts import structure_from_adjacency

        return structure_from_adjacency(native, kind=self.output_type)
