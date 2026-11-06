"""pcalg FCI-JCI adapter: FCI with context variables under the Joint Causal Inference assumptions.

Calls ``pcalg::fci`` with ``jci`` and ``contextVars``, the context variables' columns, which
``params`` name 0-based as ``context_vars``, and with ``selectionBias = FALSE``, which pcalg needs
for any ``jci`` other than ``"0"``. ``jci = "123"``, the default here, makes JCI assumptions 1-3:
no system variable causes a context variable, no system variable is confounded with a context
variable, and the context variables are all confounded but none causes another. FCI runs to the
end of the possible-d-separation step (``type = "normal"``). The PAG covers every column, context
variables included, and :func:`~andrey_bench.adapters.r_convert.marks_from_amat_pag` decodes it,
as for :class:`~andrey_bench.adapters.pcalg_fci.PcalgFCI`.

Like `pcalg::fci`, FCI-JCI takes sufficient statistics, so `cor(X)` is part of the timed fit.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from andrey_bench import rsession
from andrey_bench.adapters.r_adapter import RAdapter
from andrey_bench.adapters.r_convert import marks_from_amat_pag

_DEFAULT_ALPHA = 0.05
_DEFAULT_INDEP_TEST = "gaussCItest"
_DEFAULT_JCI = "123"

# `M` is a 1 x d 0/1 matrix marking the context columns.
_R_SOURCE = """
andrey_pcalg_fci_jci <- function(X, alpha, M, jci) {
    fit <- pcalg::fci(list(C = cor(X), n = nrow(X)), indepTest = pcalg::gaussCItest,
                      alpha = alpha, labels = colnames(X), type = "normal", jci = jci,
                      contextVars = which(M[1, ] != 0), selectionBias = FALSE)
    fit@amat
}
"""


class PcalgFCIJCI(RAdapter):
    """pcalg FCI-JCI adapter for pooled data with context variables and hidden confounders.

    Satisfies :class:`~andrey_bench.contracts.SolutionAdapter` and
    :class:`~andrey_bench.contracts.SetupAdapter`.
    """

    name: str = "pcalg.fci_jci"
    package: str = "pcalg"
    package_version: str = rsession.package_version("pcalg")
    backend: str = "native"
    mode: str = "serial"
    algorithm: str = "fci_jci"
    output_type: str = "pag"

    r_package: str = "pcalg"
    r_source: str = _R_SOURCE
    r_function: str = "andrey_pcalg_fci_jci"

    def params(self) -> dict[str, Any]:
        """Fisher-Z at `alpha=0.05`, JCI assumptions 1-3, and no context columns until set."""
        return {
            "alpha": _DEFAULT_ALPHA,
            "indep_test": _DEFAULT_INDEP_TEST,
            "jci": _DEFAULT_JCI,
            "context_vars": [],
        }

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run pcalg FCI-JCI and return Andrey PAG endpoint marks. The only timed region.

        Raises ``ValueError`` without ``context_vars``, or with a ``jci`` of ``"0"``, which is
        plain FCI.
        """
        x = np.asarray(data, dtype=np.float64)
        context = [int(j) for j in params.get("context_vars") or ()]
        jci = str(params.get("jci", _DEFAULT_JCI))
        if not context or jci not in ("1", "12", "123"):
            raise ValueError(f"{self.name} needs context_vars and jci 1, 12, or 123")
        mask = np.zeros((1, x.shape[1]))
        mask[0, context] = 1.0
        session = rsession.session()
        amat = self.call(x, float(params.get("alpha", _DEFAULT_ALPHA)), session.matrix(mask), jci)
        return marks_from_amat_pag(amat)

    def to_structure(self, native: np.ndarray) -> Any:
        """Convert PAG endpoint marks to an Andrey `GraphStructure`; untimed."""
        from andrey import GraphStructure

        return GraphStructure.from_numpy(native, kind=self.output_type)
