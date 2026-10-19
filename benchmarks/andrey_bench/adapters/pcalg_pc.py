"""pcalg PC adapter for R constraint-based search.

`pcalg::pc` reads sufficient statistics rather than a sample, so the timed fit transfers `X` and
computes `cor(X)` in R. The other PC adapters compute their correlations inside `fit` too; leaving
it out would time pcalg on less work.

`gaussCItest` at `alpha=0.05` is Fisher-Z, matching the Andrey and causal-learn PC adapters.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from andrey_bench import rsession
from andrey_bench.adapters.r_adapter import RAdapter

_DEFAULT_ALPHA = 0.05
_DEFAULT_INDEP_TEST = "gaussCItest"

_R_SOURCE = """
andrey_pcalg_pc <- function(X, alpha) {
    fit <- pcalg::pc(list(C = cor(X), n = nrow(X)), indepTest = pcalg::gaussCItest,
                     alpha = alpha, labels = colnames(X))
    # amat.cpdag writes an edge at the *child's* row, so `i -> j` is amat[j, i] = 1 and an
    # undirected edge is symmetric. Transposed here, the matrix reads row -> column like every
    # other adjacency the benchmark scores.
    t(as(fit, "amat")) * 1
}
"""


class PcalgPC(RAdapter):
    """pcalg PC adapter for R constraint-based search.

    Satisfies :class:`~andrey_bench.contracts.SolutionAdapter` and
    :class:`~andrey_bench.contracts.SetupAdapter`.
    """

    name: str = "pcalg.pc"
    package: str = "pcalg"
    package_version: str = rsession.package_version("pcalg")
    backend: str = "native"
    mode: str = "serial"
    algorithm: str = "pc"
    output_type: str = "cpdag"

    r_package: str = "pcalg"
    r_source: str = _R_SOURCE
    r_function: str = "andrey_pcalg_pc"

    def params(self) -> dict[str, Any]:
        """Fisher-Z at `alpha=0.05`, matching the other constraint-based adapters."""
        return {"alpha": _DEFAULT_ALPHA, "indep_test": _DEFAULT_INDEP_TEST}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run pcalg PC and return 0/1 CPDAG adjacency. The only timed region."""
        return self.call(data, float(params.get("alpha", _DEFAULT_ALPHA))).astype(np.int8)

    def to_structure(self, native: np.ndarray) -> Any:
        """Convert 0/1 CPDAG adjacency to an Andrey `GraphStructure`; untimed."""
        from andrey_bench.contracts import structure_from_adjacency

        return structure_from_adjacency(native, kind=self.output_type)


_R_SOURCE_RCOT = """
andrey_pcalg_pc_rcot <- function(X, alpha) {
    # RCIT's RCoT at Andrey's RCoT settings: 5 features for x and y, 100 for the conditioning set,
    # the Hall-Buckley-Eagleson p-value, and a fixed seed for each query's features.
    rcot <- function(x, y, S, suffStat) {
        z <- if (length(S)) suffStat$X[, S, drop = FALSE] else NULL
        RCIT::RCoT(suffStat$X[, x], suffStat$X[, y], z, approx = "hbe", num_f = 100, num_f2 = 5,
                   seed = 0)$p
    }
    fit <- pcalg::pc(list(X = X), indepTest = rcot, alpha = alpha, labels = colnames(X))
    t(as(fit, "amat")) * 1
}
"""


class PcalgPCRCoT(PcalgPC):
    """pcalg's PC with RCIT's RCoT, the R reference for Andrey's PC with ``indep_test="rcot"``.

    RCIT (Strobl, GitHub ``ericstrobl/RCIT``) is the implementation RCoT was published with; its
    ``RCoT`` runs at Andrey's settings (see the R source); with an empty conditioning set it calls
    ``RIT``, the same test without the regression. ``package_version`` names both packages.
    """

    name: str = "pcalg.pc.rcot"
    package: str = "pcalg+RCIT"
    package_version: str = (
        f"pcalg {rsession.package_version('pcalg')}; RCIT {rsession.package_version('RCIT')}"
    )
    r_source: str = _R_SOURCE_RCOT
    r_function: str = "andrey_pcalg_pc_rcot"

    def setup(self) -> None:
        """Start R, load pcalg and RCIT, and define the entry point. Untimed; see `rsession`."""
        rsession.start("pcalg", "RCIT").define(self.r_source)

    def params(self) -> dict[str, Any]:
        """RCoT at `alpha=0.05`, in Andrey's names."""
        return {"alpha": _DEFAULT_ALPHA, "indep_test": "rcot"}
