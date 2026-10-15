"""bnlearn's categorical tests and scores, through the embedded R session.

bnlearn reads a ``factor`` column as categorical, so each entry point turns every column of
category codes into a factor, inside the timed call, then runs:

- ``pc.stable`` with ``test="x2-adf"`` or ``"mi-adf"``, at ``alpha=0.05``: the chi-square and
  G-square tests with degrees of freedom adjusted for the empty rows and columns of each table,
  Andrey's ``indep_test="chisq"`` and ``"gsq"``. It returns a CPDAG.
- ``hc`` and ``tabu`` with ``score="bde"``, ``iss=1``, ``prior="vsp"``, and ``beta = 1 / (d - 1)``:
  BDeu with equivalent sample size 1 and the structure prior that makes each other variable a
  parent with probability ``1 / (d - 1)``, Andrey's ``score_func="local_score_BDeu"``. bnlearn's
  prior leaves out the term ``(d - 1) * log(1 - beta)`` per node, the same for every graph, so the
  two rank graphs alike. With ``score="bic"``, the multinomial BIC, Andrey's
  ``score_func="local_score_BIC_categorical"`` at ``lambda_value=1``: bnlearn's ``k`` defaults to
  ``log(n) / 2`` per parameter, and Andrey's scores are ``-2`` times bnlearn's. Both return a DAG,
  so the comparable error is the ``mec_shd`` column.

``tabu`` keeps a list of 10 DAGs and stops after 10 non-improving moves (``tabu`` and
``max.tabu``), Andrey's ``tabu_length`` and ``max_tabu``.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from andrey_bench import rsession
from andrey_bench.adapters.r_adapter import RAdapter

_DEFAULT_ALPHA = 0.05
_DEFAULT_TABU = 10
#: Andrey's test and score names, and bnlearn's for the same objective.
TESTS = {"chisq": "x2-adf", "gsq": "mi-adf"}
SCORES = {"local_score_BDeu": "bde", "local_score_BIC_categorical": "bic"}

_R_SOURCE = """
andrey_bnlearn_factors <- function(X) {
    df <- as.data.frame(X)
    df[] <- lapply(df, factor)
    df
}
andrey_bnlearn_pc_categorical <- function(X, test, alpha) {
    fit <- bnlearn::pc.stable(andrey_bnlearn_factors(X), test = test, alpha = alpha)
    bnlearn::amat(fit)
}
andrey_bnlearn_search_categorical <- function(X, search, score, tabu, max_tabu) {
    df <- andrey_bnlearn_factors(X)
    args <- list(df, score = score)
    if (score == "bde") args <- c(args, list(iss = 1, prior = "vsp", beta = 1 / (ncol(df) - 1)))
    if (search == "tabu") args <- c(args, list(tabu = tabu, max.tabu = max_tabu))
    fit <- do.call(if (search == "tabu") bnlearn::tabu else bnlearn::hc, args)
    bnlearn::amat(fit)
}
"""


class _BnlearnCategorical(RAdapter):
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


class BnlearnPCStableCategorical(_BnlearnCategorical):
    """bnlearn ``pc.stable`` with a categorical test; returns a CPDAG."""

    algorithm: str = "pc"
    output_type: str = "cpdag"
    r_function: str = "andrey_bnlearn_pc_categorical"

    def __init__(self, indep_test: str) -> None:
        if indep_test not in TESTS:
            raise ValueError(f"indep_test must be one of {sorted(TESTS)}, got {indep_test!r}")
        self.indep_test = indep_test
        self.name = f"bnlearn.pc-stable.{indep_test}"

    def params(self) -> dict[str, Any]:
        """The test, in Andrey's name, and its level."""
        return {"indep_test": self.indep_test, "alpha": _DEFAULT_ALPHA}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run bnlearn pc.stable; return its 0/1 CPDAG adjacency. The only timed region."""
        test = TESTS[str(params.get("indep_test", self.indep_test))]
        alpha = float(params.get("alpha", _DEFAULT_ALPHA))
        return self.call(data, test, alpha).astype(np.int8)


class BnlearnHCCategorical(_BnlearnCategorical):
    """bnlearn ``hc`` with a categorical score; returns a DAG."""

    algorithm: str = "hc"
    output_type: str = "dag"
    r_function: str = "andrey_bnlearn_search_categorical"

    def __init__(self, score_func: str) -> None:
        if score_func not in SCORES:
            raise ValueError(f"score_func must be one of {sorted(SCORES)}, got {score_func!r}")
        self.score_func = score_func
        self.name = f"bnlearn.{self.algorithm}.{score_func.removeprefix('local_score_')}"

    def params(self) -> dict[str, Any]:
        """The score, in Andrey's name."""
        return {"score_func": self.score_func}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run the search; return its 0/1 DAG adjacency. The only timed region."""
        score = SCORES[str(params.get("score_func", self.score_func))]
        tabu = int(params.get("tabu_length", _DEFAULT_TABU))
        max_tabu = int(params.get("max_tabu", _DEFAULT_TABU))
        return self.call(data, self.algorithm, score, tabu, max_tabu).astype(np.int8)


class BnlearnTabuCategorical(BnlearnHCCategorical):
    """bnlearn ``tabu`` with a categorical score and a tabu list of 10; returns a DAG."""

    algorithm: str = "tabu"

    def params(self) -> dict[str, Any]:
        """The score, in Andrey's name, and the tabu settings, in Andrey's names."""
        return {
            "score_func": self.score_func,
            "tabu_length": _DEFAULT_TABU,
            "max_tabu": _DEFAULT_TABU,
        }
