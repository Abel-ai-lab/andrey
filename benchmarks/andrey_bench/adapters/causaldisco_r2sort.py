"""R2-SortnRegress adapter for CausalDisco (Reisach et al., 2023), the sortability control.

R2-SortnRegress orders the variables by the share of each one's variance that a linear regression
on all the others explains, its R2, lowest first, then regresses each variable on the earlier ones
with an adaptive Lasso (BIC) and keeps the selected predictors as its parents. It reads no model of
the links, so a method that orders the variables no better than it does has found no more than
what R2-sortability gives away. CausalDisco (BSD-3-Clause) is the authors' package;
``benchmarks/start_stein.sh`` installs it beside the SCORE and NoGAM references.

:class:`CausalDiscoR2SortnRegress` returns the graph of ``r2_sort_regress``;
:class:`CausalDiscoR2Sort` returns the full DAG of its order
(:mod:`~andrey_bench.adapters.order_dag`), the floor for the share of true edges an order gets
right. :class:`CausalDiscoVarSortnRegress` returns the graph of ``var_sort_regress`` (Reisach et
al., 2021), which orders the variables by their variance instead: the control for data at the
generator's own scale, where the variances grow along the causal order.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

#: The pinned CausalDisco release; ``start_stein.sh`` installs it.
VERSION = "0.3.0"


class CausalDiscoR2SortnRegress:
    """CausalDisco's ``r2_sort_regress`` at its defaults.

    Satisfies :class:`~andrey_bench.contracts.SolutionAdapter`.
    """

    name: str = "causaldisco.r2_sort_regress"
    package: str = "CausalDisco"
    package_version: str = VERSION
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "r2_sort_regress"
    output_type: str = "dag"
    python_env: str = "ANDREY_BENCH_STEIN_PYTHON"

    def params(self) -> dict[str, Any]:
        """No search parameters: the package's defaults."""
        return {}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Return the 0/1 adjacency of the regression weights. The only timed region."""
        from CausalDisco.baselines import r2_sort_regress  # ty: ignore[unresolved-import]

        return (r2_sort_regress(np.asarray(data, dtype=np.float64)) != 0).astype(np.int8)

    def to_structure(self, native: np.ndarray) -> Any:
        """0/1 DAG adjacency -> Andrey ``GraphStructure``; untimed."""
        from andrey_bench.contracts import structure_from_adjacency

        return structure_from_adjacency(native, kind=self.output_type)


class CausalDiscoVarSortnRegress(CausalDiscoR2SortnRegress):
    """CausalDisco's ``var_sort_regress`` at its defaults."""

    name: str = "causaldisco.var_sort_regress"
    algorithm: str = "var_sort_regress"

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Return the 0/1 adjacency of the regression weights. The only timed region."""
        from CausalDisco.baselines import var_sort_regress  # ty: ignore[unresolved-import]

        return (var_sort_regress(np.asarray(data, dtype=np.float64)) != 0).astype(np.int8)


class CausalDiscoR2Sort(CausalDiscoR2SortnRegress):
    """The order ``r2_sort_regress`` regresses along, as its full DAG."""

    name: str = "causaldisco.r2_sort.order"

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Return the full DAG of the variables sorted by R2, lowest first."""
        from CausalDisco.analytics import r2coeff  # ty: ignore[unresolved-import]

        from andrey_bench.adapters.order_dag import order_dag

        return order_dag(np.argsort(r2coeff(np.asarray(data, dtype=np.float64).T)))
