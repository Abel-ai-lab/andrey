"""bnlearn's conditional Gaussian score and test, on mixed data, through the embedded R session.

bnlearn reads a ``factor`` column as categorical and a ``numeric`` one as continuous. Each entry
point turns every column of at most ``max_categories`` distinct whole numbers into a factor, the
rule Andrey and Tetrad's adapters read the columns by, then runs:

- ``hc`` with ``score="bic-cg"``, the conditional linear Gaussian BIC: Andrey's
  ``score_func="local_score_cond_gauss"`` family. bnlearn's conditional linear Gaussian networks
  give no categorical node a continuous parent, and its search keeps to them. It returns a DAG, so
  the comparable error is the ``mec_shd`` column.
- ``pc.stable`` with ``test="mi-cg"``, the conditional Gaussian mutual-information test: Andrey's
  ``indep_test="cond_gauss"`` family, at ``alpha=0.05``. It returns a CPDAG.

bnlearn's ``k`` for the BIC defaults to ``log(n) / 2`` per parameter, Andrey's ``penalty_discount=1``
in deviance units.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from andrey_bench import rsession
from andrey_bench.adapters.r_adapter import RAdapter

_DEFAULT_ALPHA = 0.05
_MAX_CATEGORIES = 10

_R_SOURCE = """
andrey_bnlearn_typed <- function(X, max_categories) {
    df <- as.data.frame(X)
    df[] <- lapply(df, function(v) {
        if (all(v == round(v)) && length(unique(v)) <= max_categories) factor(v) else v
    })
    df
}
andrey_bnlearn_hc_cg <- function(X, max_categories) {
    fit <- bnlearn::hc(andrey_bnlearn_typed(X, max_categories), score = "bic-cg")
    bnlearn::amat(fit)
}
andrey_bnlearn_pc_cg <- function(X, max_categories, alpha) {
    fit <- bnlearn::pc.stable(andrey_bnlearn_typed(X, max_categories), test = "mi-cg",
                              alpha = alpha)
    bnlearn::amat(fit)
}
"""


class _BnlearnMixed(RAdapter):
    package: str = "bnlearn"
    package_version: str = rsession.package_version("bnlearn")
    backend: str = "native"
    mode: str = "serial"
    r_package: str = "bnlearn"
    r_source: str = _R_SOURCE

    def to_structure(self, native: np.ndarray) -> Any:
        """0/1 adjacency -> ``GraphStructure``; untimed."""
        from andrey_bench.contracts import structure_from_adjacency

        return structure_from_adjacency(native, kind=self.output_type)


class BnlearnHCCondGauss(_BnlearnMixed):
    """bnlearn ``hc`` with the conditional linear Gaussian BIC; returns a DAG."""

    name: str = "bnlearn.hc.cond_gauss"
    algorithm: str = "hc"
    output_type: str = "dag"
    r_function: str = "andrey_bnlearn_hc_cg"

    def params(self) -> dict[str, Any]:
        """The score family, in Andrey's name, and the categorical-column bound."""
        return {"score_func": "local_score_cond_gauss", "max_categories": _MAX_CATEGORIES}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run bnlearn hc; return its 0/1 DAG adjacency. The only timed region."""
        cap = float(params.get("max_categories", _MAX_CATEGORIES))
        return self.call(data, cap).astype(np.int8)


class BnlearnPCStableCondGauss(_BnlearnMixed):
    """bnlearn ``pc.stable`` with the conditional Gaussian test; returns a CPDAG."""

    name: str = "bnlearn.pc-stable.cond_gauss"
    algorithm: str = "pc"
    output_type: str = "cpdag"
    r_function: str = "andrey_bnlearn_pc_cg"

    def params(self) -> dict[str, Any]:
        """The test family, in Andrey's name, its level, and the categorical-column bound."""
        return {
            "indep_test": "cond_gauss",
            "alpha": _DEFAULT_ALPHA,
            "max_categories": _MAX_CATEGORIES,
        }

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run bnlearn pc.stable; return its 0/1 CPDAG adjacency. The only timed region."""
        cap = float(params.get("max_categories", _MAX_CATEGORIES))
        alpha = float(params.get("alpha", _DEFAULT_ALPHA))
        return self.call(data, cap, alpha).astype(np.int8)
