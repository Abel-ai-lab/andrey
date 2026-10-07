"""pcalg FCI adapter for R constraint-based search with latent variables.

FCI returns a PAG, so the native array is an endpoint-mark matrix rather than a 0/1 adjacency:
pcalg's `amat.pag` encodes none / circle / arrowhead / tail at the opposite end from Andrey.
:func:`~andrey_bench.adapters.r_convert.marks_from_amat_pag` converts these marks.

Like `pcalg::pc`, FCI takes sufficient statistics, so `cor(X)` is part of the timed fit.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from andrey_bench import rsession
from andrey_bench.adapters.r_adapter import RAdapter
from andrey_bench.adapters.r_convert import marks_from_amat_pag

_DEFAULT_ALPHA = 0.05
_DEFAULT_INDEP_TEST = "gaussCItest"

_R_SOURCE = """
andrey_pcalg_fci <- function(X, alpha) {
    fit <- pcalg::fci(list(C = cor(X), n = nrow(X)), indepTest = pcalg::gaussCItest,
                      alpha = alpha, labels = colnames(X))
    fit@amat
}
"""


class PcalgFCI(RAdapter):
    """pcalg FCI adapter for R constraint-based search with latent variables.

    Satisfies :class:`~andrey_bench.contracts.SolutionAdapter` and
    :class:`~andrey_bench.contracts.SetupAdapter`.
    """

    name: str = "pcalg.fci"
    package: str = "pcalg"
    package_version: str = rsession.package_version("pcalg")
    backend: str = "native"
    mode: str = "serial"
    algorithm: str = "fci"
    output_type: str = "pag"

    r_package: str = "pcalg"
    r_source: str = _R_SOURCE
    r_function: str = "andrey_pcalg_fci"

    def params(self) -> dict[str, Any]:
        """Fisher-Z at `alpha=0.05`, matching the other constraint-based adapters."""
        return {"alpha": _DEFAULT_ALPHA, "indep_test": _DEFAULT_INDEP_TEST}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run pcalg FCI and return Andrey PAG endpoint marks. The only timed region."""
        amat = self.call(data, float(params.get("alpha", _DEFAULT_ALPHA)))
        return marks_from_amat_pag(amat)

    def to_structure(self, native: np.ndarray) -> Any:
        """Convert PAG endpoint marks to an Andrey `GraphStructure`; untimed.

        `structure_from_adjacency` accepts 0/1 adjacency, which cannot represent PAG circles.
        """
        from andrey import GraphStructure

        return GraphStructure.from_numpy(native, kind=self.output_type)
