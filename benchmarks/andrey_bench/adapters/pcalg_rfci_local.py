"""pcalg RFCI adapter that counts the tests, the RFCI arm of the target and latent lane.

The same call as :class:`~andrey_bench.adapters.pcalg_rfci.PcalgRFCI`, Fisher z at ``alpha = 0.05``
unless the lane sets it, with the test wrapped in a counter: ``pcalg::rfci`` runs with an
``indepTest`` that adds one to a count and calls ``pcalg::gaussCItest``, so the count is every test
the fit evaluated, the skeleton search and RFCI's extra tests of the orientation phase. The answer
is the whole PAG, which every target's row and column are read from
(:class:`~andrey_bench.local_lane.LocalAdapter` with ``local = False``).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from andrey_bench import rsession
from andrey_bench.adapters.pcalg_rfci import PcalgRFCI
from andrey_bench.adapters.r_convert import marks_from_amat_pag
from andrey_bench.local_lane import pack_answer, unpack_answer

_R_SOURCE = """
andrey_pcalg_rfci_counted <- function(X, alpha) {
    tests <- 0
    counted <- function(x, y, S, suffStat) {
        tests <<- tests + 1
        pcalg::gaussCItest(x, y, S, suffStat)
    }
    fit <- pcalg::rfci(list(C = cor(X), n = nrow(X)), indepTest = counted, alpha = alpha,
                       labels = colnames(X))
    assign(".andrey_rfci_tests", tests, envir = .GlobalEnv)
    fit@amat
}
"""


class PcalgRFCICounted(PcalgRFCI):
    """pcalg RFCI with its conditional-independence tests counted."""

    local: bool = False
    r_source: str = _R_SOURCE
    r_function: str = "andrey_pcalg_rfci_counted"

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run pcalg RFCI; return the packed PAG marks and the number of tests."""
        amat = self.call(data, float(params.get("alpha", self.params()["alpha"])))
        tests = int(rsession.session().function(".andrey_rfci_tests")[0])
        return pack_answer(marks_from_amat_pag(amat), tests)

    def to_structure(self, native: np.ndarray) -> Any:
        """The PAG's marks as a ``GraphStructure``; untimed."""
        from andrey import GraphStructure

        return GraphStructure.from_numpy(unpack_answer(native)[0], kind=self.output_type)
