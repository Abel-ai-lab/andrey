"""bnlearn adapters for categorical data: `pc.stable`, `hc`, and `tabu` on factor columns.

bnlearn chooses its discrete tests and scores only for a data frame of factors, so each entry point
turns the matrix of category codes into factors first, inside the timed call.

The settings match Andrey's. `test="x2-adf"` and `"mi-adf"` are the chi-square and G-square tests
with degrees of freedom adjusted for empty rows and columns, as Andrey's `chisq` and `gsq` count
them. `score="bde"` with `iss=1` and `prior="vsp"`, `beta = 1/(d - 1)` is BDeu with equivalent
sample size 1 and the structure prior that makes each other variable a parent with probability
`1/(d - 1)`, Andrey's `local_score_BDeu`: bnlearn's prior leaves out the term
`(d - 1) log(1 - beta)` per node, the same for every graph, so the two rank graphs alike. `score="bic"` is the multinomial BIC, Andrey's
`local_score_BIC_discrete` at `lambda_value=1` up to its factor of `-2`.

`hc` and `tabu` return a DAG, scored through its equivalence class like the Gaussian adapters'.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from andrey_bench import rsession
from andrey_bench.adapters.r_adapter import RAdapter

_R_SOURCE = """
andrey_bnlearn_factors <- function(X) {
    df <- as.data.frame(X)
    df[] <- lapply(df, factor)
    df
}
andrey_bnlearn_discrete_pc <- function(X, test, alpha) {
    fit <- bnlearn::pc.stable(andrey_bnlearn_factors(X), test = test, alpha = alpha)
    bnlearn::amat(fit)
}
andrey_bnlearn_discrete_score <- function(X, search, score, tabu) {
    df <- andrey_bnlearn_factors(X)
    args <- list(df, score = score)
    if (score == "bde") args <- c(args, list(iss = 1, prior = "vsp", beta = 1 / (ncol(df) - 1)))
    if (search == "tabu") args <- c(args, list(tabu = tabu, max.tabu = tabu))
    fit <- do.call(if (search == "tabu") bnlearn::tabu else bnlearn::hc, args)
    bnlearn::amat(fit)
}
"""

_DEFAULT_ALPHA = 0.05
_DEFAULT_TEST = "x2-adf"
_DEFAULT_SCORE = "bde"
_DEFAULT_TABU = 10


class _BnlearnDiscrete(RAdapter):
    package: str = "bnlearn"
    package_version: str = rsession.package_version("bnlearn")
    backend: str = "native"
    mode: str = "serial"
    r_package: str = "bnlearn"
    r_source: str = _R_SOURCE

    def to_structure(self, native: np.ndarray) -> Any:
        """Convert 0/1 adjacency to an Andrey `GraphStructure`; untimed."""
        from andrey_bench.contracts import structure_from_adjacency

        return structure_from_adjacency(native, kind=self.output_type)


class BnlearnDiscretePC(_BnlearnDiscrete):
    """bnlearn `pc.stable` with a discrete test. Returns a 0/1 CPDAG adjacency."""

    name: str = "bnlearn.pc-stable.discrete"
    algorithm: str = "pc"
    output_type: str = "cpdag"
    r_function: str = "andrey_bnlearn_discrete_pc"

    def params(self) -> dict[str, Any]:
        """The adjusted chi-square test at `alpha=0.05`."""
        return {"alpha": _DEFAULT_ALPHA, "test": _DEFAULT_TEST}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run bnlearn pc.stable on factor columns. The only timed region."""
        adjacency = self.call(
            data,
            str(params.get("test", _DEFAULT_TEST)),
            float(params.get("alpha", _DEFAULT_ALPHA)),
        )
        return adjacency.astype(np.int8)


class BnlearnDiscreteHC(_BnlearnDiscrete):
    """bnlearn `hc` with a discrete score. Returns a 0/1 DAG adjacency."""

    name: str = "bnlearn.hc.discrete"
    algorithm: str = "hc"
    output_type: str = "dag"
    r_function: str = "andrey_bnlearn_discrete_score"
    search: str = "hc"

    def params(self) -> dict[str, Any]:
        """BDeu with equivalent sample size 1 and the `vsp` structure prior."""
        return {"score": _DEFAULT_SCORE}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run the search on factor columns. The only timed region."""
        adjacency = self.call(
            data,
            self.search,
            str(params.get("score", _DEFAULT_SCORE)),
            int(params.get("tabu", _DEFAULT_TABU)),
        )
        return adjacency.astype(np.int8)


class BnlearnDiscreteTabu(BnlearnDiscreteHC):
    """bnlearn `tabu` with a discrete score; `max.tabu` equals `tabu`, as Andrey's adapter sets."""

    name: str = "bnlearn.tabu.discrete"
    algorithm: str = "tabu"
    search: str = "tabu"

    def params(self) -> dict[str, Any]:
        """BDeu with the `vsp` prior and a tabu list of 10."""
        return {"score": _DEFAULT_SCORE, "tabu": _DEFAULT_TABU}
