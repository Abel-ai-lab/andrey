"""The Fisher-z test both of the authors' adjustment references are called with.

SNAP and LOAD (Schubert et al., 2025, 2026) take the conditional-independence test as a callable
``ci_test(x, y, given) -> p-value`` and read nothing else of the data. The test here is the one
their demo and the experiments' tests use, Fisher's z test on the sample correlation matrix:
the partial correlation of ``x`` and ``y`` given ``given`` from the inverse of the correlation
submatrix, transformed by ``arctanh`` and scaled by ``sqrt(n - |given| - 3)``, with the two-sided
normal tail as the p-value. The tail is ``erfc``, so the module needs NumPy only and a p-value
below 1e-16 does not round to zero.

The references pass the conditioning set as a tuple, a set, or a frozenset, and test a pair in
either order; :class:`FisherZ` evaluates a query once per unordered pair and set and answers a
repeat from its cache, as the demo's test object does. :attr:`FisherZ.n_tests` counts the queries
evaluated: the lane's definition of a test count, distinct conditional-independence queries.
:attr:`FisherZ.calls` counts every call, which is what profiling the demo's test object reports.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from math import erfc, log, sqrt
from typing import Any

import numpy as np

#: The only test the two references are run with.
INDEP_TEST = "fisherz"


def check_settings(settings: Mapping[str, Any], what: str) -> None:
    """Raise ``ValueError`` for a setting the reference ``what`` cannot honor.

    The references call a Fisher-z test only, and neither has the lane's ``max_depth`` (SNAP's own
    cap is ``max_order``), so a request for another test or for that cap is refused instead of
    being run as something else.
    """
    if settings.get("indep_test") != INDEP_TEST:
        got = settings.get("indep_test")
        raise ValueError(f"{what} runs with the {INDEP_TEST!r} test, got indep_test={got!r}")
    if settings.get("max_depth") is not None:
        raise ValueError(f"{what} has no max_depth cap")


class FisherZ:
    """Fisher's z test on the correlation matrix of ``data``, with a cache and a test count.

    Calling the object with two columns and a conditioning set returns the p-value of the test
    that they are independent given the set; ``nan`` is never returned: a singular correlation
    submatrix or a sample too small for the set raises ``ValueError``.
    """

    def __init__(self, data: np.ndarray) -> None:
        data = np.asarray(data, dtype=np.float64)
        self.n_rows = int(data.shape[0])
        self._corr = np.corrcoef(data.T)
        self._cache: dict[tuple[int, int, tuple[int, ...]], float] = {}
        self.calls = 0

    @property
    def n_tests(self) -> int:
        """The distinct queries evaluated: a repeat of a pair and a set, in either order, is not."""
        return len(self._cache)

    def __call__(
        self, x: int | np.integer, y: int | np.integer, given: Iterable[int] = ()
    ) -> float:
        self.calls += 1
        x, y = int(x), int(y)
        key = (min(x, y), max(x, y), tuple(sorted(int(v) for v in given)))
        p = self._cache.get(key)
        if p is None:
            p = self._cache[key] = self._p_value(*key)
        return p

    def _p_value(self, x: int, y: int, given: tuple[int, ...]) -> float:
        dof = self.n_rows - len(given) - 3
        if dof <= 0:
            raise ValueError(
                f"{self.n_rows} rows are too few for a conditioning set of {len(given)}"
            )
        columns = [x, y, *given]
        try:
            inverse = np.linalg.inv(self._corr[np.ix_(columns, columns)])
        except np.linalg.LinAlgError as error:
            raise ValueError(
                "the correlation matrix is singular; the Fisher-z test needs none"
            ) from error
        r = -inverse[0, 1] / sqrt(abs(inverse[0, 0] * inverse[1, 1]))
        if abs(r) >= 1.0:  # a deterministic relation: keep the transform finite
            r = (1.0 - np.finfo(float).eps) * np.sign(r)
        statistic = sqrt(dof) * abs(0.5 * log((1.0 + r) / (1.0 - r)))
        return erfc(statistic / sqrt(2.0))
