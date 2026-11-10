"""bnlearn structural EM adapter: score-based search on data with missing values.

`bnlearn::structural.em` is Scutari's implementation of Friedman's structural EM. Each iteration
fits the parameters of the current network (`fit`), fills every missing value from it (`impute`),
and runs a score-based search on the completed data (`maximize`, `hc` or `tabu`) starting from the
current network; it stops when the fitted parameters stop changing or after `max.iter` iterations.
The result is a DAG, one member of an equivalence class, so `output_type` is `"dag"` and
`andrey.metrics.score` reports the directed `shd` and the equivalence-class `mec_shd`, the column
comparable with Andrey's CPDAG output (see :mod:`andrey_bench.adapters.bnlearn_hc`).

The call, checked against the installed manual of bnlearn 5.2.1 (license `GPL (>= 2)`):

    structural.em(df, maximize = <hc|tabu>, maximize.args = list(score = "bic-g"),
                  fit = "mle-g", impute = "exact", max.iter = <5|10>)

- `maximize.args = list(score = "bic-g")` is the Gaussian BIC `bnlearn.hc` and `bnlearn.tabu` run
  with, so the complete-data arms are the same searches without the EM layer. No parent cap, and
  `tabu` keeps its defaults: a list of 10 networks and 10 moves without improvement.
- `fit = "mle-g"` is the maximum-likelihood parameter fit of a Gaussian network.
- `impute = "exact"` imputes each missing value with its posterior mean given the observed values
  of its Markov blanket, in closed form for the multivariate normal. It is deterministic, so no R
  seed is set. The default, `"bayes-lw"`, averages a random simulation and would make a fit depend
  on the generator.
- `max.iter = 5` is bnlearn's default. `max_iter=10` matches the ten EM iterations of Andrey's
  `missing="em"`, and the arm's name carries `.iter10`.

With `start = NULL` the first E-step imputes from the empty network fitted to the observed values.
The E-step differs from Andrey's: bnlearn replaces each missing value by a conditional mean and
searches on the completed data, so the imputed values carry no conditional variance, while
Andrey's EM keeps the expected second moments of the model and scores from them. The two packages
share the idea, not the arithmetic, and their searches can end on different networks.

References
----------
.. [1] Friedman, N. (1997). Learning belief networks in the presence of missing values and hidden
   variables. Proceedings of the 14th International Conference on Machine Learning, 125-133.
.. [2] Scutari, M. (2010). Learning Bayesian networks with the bnlearn R package. Journal of
   Statistical Software 35(3), 1-22. The `structural.em` manual page cites [1].
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from andrey_bench import rsession
from andrey_bench.adapters.r_adapter import RAdapter

#: The searches `maximize` runs on each completed data set.
MAXIMIZE = ("hc", "tabu")
#: bnlearn's default for `max.iter`; another value is part of the arm's name.
DEFAULT_MAX_ITER = 5
_SCORE = "bic-g"
_FIT = "mle-g"
_IMPUTE = "exact"

_R_SOURCE = """
andrey_bnlearn_structural_em <- function(X, maximize, score, fit, impute, max_iter) {
    # A missing value reaches R as NaN; bnlearn reads NA.
    X[is.nan(X)] <- NA
    dag <- bnlearn::structural.em(
        as.data.frame(X), maximize = maximize, maximize.args = list(score = score),
        fit = fit, impute = impute, max.iter = max_iter
    )
    # bnlearn's amat is 0/1, row -> column, and carries the data frame's names.
    bnlearn::amat(dag)
}
"""


class BnlearnStructuralEM(RAdapter):
    """bnlearn structural EM with `hc` or `tabu` as the search, on data with missing values.

    Satisfies :class:`~andrey_bench.contracts.SolutionAdapter` and
    :class:`~andrey_bench.contracts.SetupAdapter`. The arm is named
    ``bnlearn.structural_em.<maximize>``, with ``.iter<max_iter>`` appended for a `max_iter`
    other than bnlearn's default of 5.
    """

    package: str = "bnlearn"
    package_version: str = rsession.package_version("bnlearn")
    backend: str = "native"
    mode: str = "serial"
    output_type: str = "dag"

    r_package: str = "bnlearn"
    r_source: str = _R_SOURCE
    r_function: str = "andrey_bnlearn_structural_em"

    def __init__(self, maximize: str = "hc", max_iter: int = DEFAULT_MAX_ITER) -> None:
        if maximize not in MAXIMIZE:
            raise ValueError(f"maximize must be one of {MAXIMIZE}, got {maximize!r}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be at least 1, got {max_iter}")
        self.maximize = maximize
        self.max_iter = int(max_iter)
        self.algorithm = maximize
        suffix = "" if self.max_iter == DEFAULT_MAX_ITER else f".iter{self.max_iter}"
        self.name = f"bnlearn.structural_em.{maximize}{suffix}"

    def params(self) -> dict[str, Any]:
        """The search, its score, the parameter fit, the imputation, and `max.iter`."""
        return {
            "maximize": self.maximize,
            "score": _SCORE,
            "fit": _FIT,
            "impute": _IMPUTE,
            "max_iter": self.max_iter,
        }

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run structural EM on `data`, ``NaN`` where a value is missing; return 0/1 adjacency.

        The only timed region.
        """
        adjacency = self.call(
            data,
            str(params.get("maximize", self.maximize)),
            str(params.get("score", _SCORE)),
            str(params.get("fit", _FIT)),
            str(params.get("impute", _IMPUTE)),
            int(params.get("max_iter", self.max_iter)),
        )
        return adjacency.astype(np.int8)

    def to_structure(self, native: np.ndarray) -> Any:
        """Convert 0/1 DAG adjacency to an Andrey `GraphStructure`; untimed."""
        from andrey_bench.contracts import structure_from_adjacency

        return structure_from_adjacency(native, kind=self.output_type)
