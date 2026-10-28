"""The nonlinear conditional-independence tests as the recovery tests and recorded outputs run them.

CMIknn and GPDC draw fewer permutations than their defaults, which keeps the suite fast: a
permutation p-value is calibrated at any number of draws, only coarser. :class:`PValueOnly` is the
smallest registered test: the one call, nothing else.
"""

from __future__ import annotations

from functools import partial

import numpy as np

from andrey.core.ci import FisherZ
from andrey.core.ci_cmiknn import CMIknn
from andrey.core.ci_gpdc import GPDC
from andrey.core.ci_kernel import KCI, RCoT
from tests.recovery.helpers import datagen

#: Recorded-output name -> ``indep_test`` name and the factory the tests build it with.
CI_TESTS = {
    "KCI": ("kci", KCI),
    "RCoT": ("rcot", partial(RCoT, seed=0)),
    "CMIknn": ("cmiknn", partial(CMIknn, n_permutations=100, seed=0)),
    "GPDC": ("gpdc", partial(GPDC, n_null=300, seed=0)),
}


class PValueOnly:
    """A registered test with the single-query call alone: the p-value of Fisher's z-test."""

    def __init__(self, data: np.ndarray) -> None:
        self._test = FisherZ(data)

    def __call__(self, x: int, y: int, condition_set=()) -> float:
        return float(self._test(x, y, condition_set))


def recorded_pvalues(algorithm: str, data: np.ndarray, case: str) -> dict:
    """The p-values of ``case``'s queries under the test ``algorithm``, as recorded outputs.

    ``data`` is the case's input. A query names variables as the case defines them and is asked of
    the columns that hold them.
    """
    test = CI_TESTS[algorithm][1](data)
    column = datagen.column_of(case)
    return {
        "pvalues": np.asarray(
            [
                test(int(column[x]), int(column[y]), tuple(int(column[v]) for v in cond))
                for x, y, cond in datagen.CASES[case]["queries"]
            ]
        )
    }
