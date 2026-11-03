"""CAM 1.0 adapter (R), the reference implementation of Buhlmann, Peters, and Ernest (2014).

Calls ``CAM::CAM`` with the paper's steps on: ``variableSel = TRUE`` (componentwise boosting
preselects each variable's possible parents) and ``pruning = TRUE`` (smooth-term tests at
``cutOffPVal = 0.001``). The function's other arguments keep their defaults, which `params`
records. CAM 1.0 is archived on CRAN; ``benchmarks/start_cam_r.sh`` installs it into the R lane's
library.

Settings that differ from Andrey's defaults are named, not tuned away:

- ``maxNumParents = min(d - 1, round(n / 20))``, a parent cap of 50 at ``n = 1000`` that Andrey does
  not apply by default;
- the preselection keeps variables picked in more than ``atLeastThatMuchSelected = 0.02`` of the
  boosting iterations, at most ``atMostThatManyNeighbors = 10`` of them; Andrey applies the same
  rule;
- the smoothing parameters come from mgcv's GCV optimizer, Andrey's from its own, so near-tied
  fits can land on different local optima.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from andrey_bench import rsession
from andrey_bench.adapters.r_adapter import RAdapter

_R_SOURCE = """
andrey_cam_r <- function(X) {
    fit <- CAM::CAM(
        X, scoreName = "SEMGAM", parsScore = list(numBasisFcts = 10), numCores = 1,
        output = FALSE, variableSel = TRUE, pruning = TRUE
    )
    # A sparse 0/1 matrix with `adj[i, j] = 1` for i -> j and no dimnames.
    m <- as.matrix(fit$Adj) * 1
    dimnames(m) <- list(colnames(X), colnames(X))
    m
}
"""


class CamR(RAdapter):
    """CAM 1.0 through the embedded R session.

    Satisfies :class:`~andrey_bench.contracts.SolutionAdapter` and
    :class:`~andrey_bench.contracts.SetupAdapter`.
    """

    name: str = "cam.cam"
    package: str = "CAM"
    package_version: str = rsession.package_version("CAM")
    backend: str = "native"
    mode: str = "serial"
    algorithm: str = "cam"
    output_type: str = "dag"

    r_package: str = "CAM"
    r_source: str = _R_SOURCE
    r_function: str = "andrey_cam_r"

    def setup(self) -> None:
        """Start R, then fit a tiny table before timing, so first-call costs stay out of the fit."""
        from andrey_bench.adapters.fcm_warm import tiny_table

        super().setup()
        self.call(tiny_table())

    def params(self) -> dict[str, Any]:
        """The paper's steps on, every other argument at CAM 1.0's default."""
        return {
            "scoreName": "SEMGAM",
            "numBasisFcts": 10,
            "variableSel": True,
            "pruning": True,
            "maxNumParents": "min(d - 1, round(n / 20))",
            "atLeastThatMuchSelected": 0.02,
            "atMostThatManyNeighbors": 10,
            "cutOffPVal": 0.001,
        }

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run CAM 1.0 and return a 0/1 DAG adjacency. The only timed region."""
        return self.call(data).astype(np.int8)

    def to_structure(self, native: np.ndarray) -> Any:
        """0/1 DAG adjacency -> Andrey ``GraphStructure``; untimed."""
        from andrey_bench.contracts import structure_from_adjacency

        return structure_from_adjacency(native, kind=self.output_type)
