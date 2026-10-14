"""The nonlinear conditional-independence tests as the recovery tests and recorded outputs run them.

CMIknn and GPDC draw fewer permutations than their defaults, which keeps the suite fast: a
permutation p-value is calibrated at any number of draws, only coarser.
"""

from __future__ import annotations

from functools import partial

import numpy as np

from andrey.core.ci_cmiknn import CMIknn
from andrey.core.ci_gpdc import GPDC
from andrey.core.ci_kernel import KCI, RCoT

#: Recorded-output name -> ``indep_test`` name and the factory the tests build it with.
CI_TESTS = {
    "KCI": ("kci", KCI),
    "RCoT": ("rcot", partial(RCoT, seed=0)),
    "CMIknn": ("cmiknn", partial(CMIknn, n_permutations=100, seed=0)),
    "GPDC": ("gpdc", partial(GPDC, n_null=300, seed=0)),
}


def recorded_pvalues(algorithm: str, data: np.ndarray, queries) -> dict:
    """The p-values of ``queries`` under the test ``algorithm``, in the recorded-output layout."""
    test = CI_TESTS[algorithm][1](data)
    return {"pvalues": np.asarray([test(x, y, cond) for x, y, cond in queries])}
