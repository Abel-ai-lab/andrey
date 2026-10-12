"""PCMCI: the lagged causal links of a multivariate time series.

PCMCI runs two steps over the lagged copies ``X^i_{t-tau}`` of the variables, ``tau = 1 ..
tau_max``:

1. **PC1 condition selection.** For each variable ``X^j_t``, every lagged variable starts as a
   candidate parent. Round ``p = 0, 1, ...`` tests each candidate against ``X^j_t`` given the ``p``
   strongest other candidates, then drops the candidates whose p-value exceeds ``pc_alpha``. A
   candidate's strength is the smallest absolute test statistic it has had so far; the survivors
   are re-sorted by it, strongest first, after each round. The rounds stop when no candidate has
   ``p`` others left to condition on.
2. **MCI.** Every link ``X^i_{t-tau} -> X^j_t`` is tested given the selected parents of ``X^j_t``
   (without ``X^i_{t-tau}``) and the selected parents of ``X^i_t`` shifted ``tau`` steps back. The
   link is significant when its p-value is at most ``alpha_level``.

Every test reads one lagged design matrix: column ``lag * n + i`` holds ``X^i_{t-lag}`` for ``lag =
0 .. 2 * tau_max``, over the ``T - 2 * tau_max`` time steps where all those lags exist, so every
test uses the same samples. Every test is a partial-correlation test: ``indep_test="student_t"``
(:class:`~andrey.core.ci.StudentT`, the default) or ``"fisherz"``
(:class:`~andrey.core.ci.FisherZ`). Any object with
``statistic_pvalue(x, y, condition_set) -> (statistic, p_value)`` over the design's column indices
can drive :func:`pc1` and :func:`mci`.

References
----------
Runge, J., Nowack, P., Kretschmer, M., Flaxman, S., and Sejdinovic, D. (2019). Detecting and
quantifying causal associations in large nonlinear time series datasets. Science Advances 5(11),
eaau4996. https://doi.org/10.1126/sciadv.aau4996
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from andrey.core.ci import FisherZ, StudentT

# A lagged variable ``(i, lag)``: variable ``i`` at time ``t - lag``.
Link = tuple[int, int]


class LinkTest(Protocol):
    """A conditional-independence test that reports its statistic with the p-value.

    Built once over the lagged design matrix and called with its column indices. A larger p-value
    means more independence; a larger absolute statistic means a stronger dependence.
    """

    def statistic_pvalue(
        self, x: int, y: int, condition_set: Iterable[int] = ()
    ) -> tuple[float, float]: ...


# The ``indep_test`` names: each builds a test over the lagged design.
INDEP_TESTS: dict[str, Callable[[np.ndarray], LinkTest]] = {
    "student_t": StudentT,
    "fisherz": FisherZ,
}


@dataclass(frozen=True)
class PCMCIResult:
    """The PC1 parents and the MCI results of one PCMCI run.

    ``parents[j]`` holds the ``(i, lag)`` parents PC1 selected for variable ``j``, strongest first.
    ``statistics``, ``p_values``, and ``graph`` have shape ``(tau_max, n, n)``: entry
    ``[lag - 1, i, j]`` belongs to the link ``X^i_{t-lag} -> X^j_t``, and ``graph`` is
    ``p_values <= alpha_level``.
    """

    parents: tuple[tuple[Link, ...], ...]
    statistics: np.ndarray
    p_values: np.ndarray
    graph: np.ndarray


def lagged_design(data: np.ndarray, tau_max: int) -> np.ndarray:
    """Return the ``(T - 2 * tau_max, n * (2 * tau_max + 1))`` lagged design of a ``(T, n)`` series.

    Column ``lag * n + i`` holds variable ``i`` at ``t - lag``, row by row for ``t = 2 * tau_max ..
    T - 1``.
    """
    n_steps = data.shape[0]
    window = 2 * tau_max
    return np.hstack([data[window - lag : n_steps - lag] for lag in range(window + 1)])


def _column(link: Link, n_vars: int) -> int:
    """The design column of the lagged variable ``link``."""
    i, lag = link
    return lag * n_vars + i


def pc1(
    test: LinkTest, n_vars: int, target: int, tau_max: int, pc_alpha: float
) -> tuple[Link, ...]:
    """Select the lagged parents of variable ``target`` with the PC1 rounds; strongest first.

    Candidates start in variable-major order, ``(0, 1), (0, 2), ..., (1, 1), ...``, and each round
    re-sorts the survivors stably, so equal strengths keep their earlier order.
    """
    candidates = [(i, lag) for i in range(n_vars) for lag in range(1, tau_max + 1)]
    strength = dict.fromkeys(candidates, math.inf)
    p = 0
    while len(candidates) - 1 >= p:
        removed = set()
        for link in candidates:
            conditions = [c for c in candidates if c != link][:p]
            statistic, p_value = test.statistic_pvalue(
                _column(link, n_vars), target, [_column(c, n_vars) for c in conditions]
            )
            strength[link] = min(strength[link], abs(statistic))
            if p_value > pc_alpha:
                removed.add(link)
        candidates = sorted((c for c in candidates if c not in removed), key=lambda c: -strength[c])
        p += 1
    return tuple(candidates)


def mci(
    test: LinkTest, n_vars: int, parents: tuple[tuple[Link, ...], ...], tau_max: int
) -> tuple[np.ndarray, np.ndarray]:
    """Run the MCI test on every link; return ``(statistics, p_values)``, each ``(tau_max, n, n)``.

    The link ``X^i_{t-lag} -> X^j_t`` is tested given ``parents[j]`` without ``(i, lag)`` and
    ``parents[i]`` shifted ``lag`` steps back.
    """
    statistics = np.zeros((tau_max, n_vars, n_vars))
    p_values = np.ones((tau_max, n_vars, n_vars))
    for j in range(n_vars):
        for i in range(n_vars):
            for lag in range(1, tau_max + 1):
                conditions = {c for c in parents[j] if c != (i, lag)}
                conditions.update((k, s + lag) for k, s in parents[i])
                statistic, p_value = test.statistic_pvalue(
                    _column((i, lag), n_vars), j, sorted(_column(c, n_vars) for c in conditions)
                )
                statistics[lag - 1, i, j] = statistic
                p_values[lag - 1, i, j] = p_value
    return statistics, p_values


def pcmci(
    data: np.ndarray,
    *,
    tau_max: int = 1,
    pc_alpha: float = 0.2,
    alpha_level: float = 0.05,
    indep_test: str = "student_t",
) -> PCMCIResult:
    """Run PC1 and MCI on a ``(T, n)`` series with the ``indep_test`` partial-correlation test.

    Raises :class:`ValueError` when ``indep_test`` is not a key of :data:`INDEP_TESTS`.
    """
    if indep_test not in INDEP_TESTS:
        raise ValueError(f"indep_test must be one of {sorted(INDEP_TESTS)}, got {indep_test!r}")
    data = np.asarray(data, dtype=np.float64)
    n_vars = data.shape[1]
    test = INDEP_TESTS[indep_test](lagged_design(data, tau_max))
    parents = tuple(pc1(test, n_vars, j, tau_max, pc_alpha) for j in range(n_vars))
    statistics, p_values = mci(test, n_vars, parents, tau_max)
    return PCMCIResult(
        parents=parents, statistics=statistics, p_values=p_values, graph=p_values <= alpha_level
    )
