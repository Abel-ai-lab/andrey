"""PCMCI: the lagged causal links of a multivariate time series.

PCMCI runs two steps over the lagged copies ``X^i_{t-tau}`` of the variables, ``tau = 1 ..
tau_max``:

1. **PC1 condition selection.** For each variable ``X^j_t``, every lagged variable starts as a
   candidate parent. Round ``p = 0, 1, ...`` tests each candidate against ``X^j_t`` given the ``p``
   strongest other candidates, then drops the candidates whose p-value exceeds ``pc_alpha``. A
   candidate's strength is the smallest absolute test statistic it has had so far; the survivors
   are re-sorted by it, strongest first, after each round. Strengths equal to ``_KEY_DP`` decimals
   are ordered by :func:`~andrey.search.hc.tie_ranks` of their variables, then by lag, so the
   order follows the variables when the columns are reordered. The rounds stop when no candidate
   has ``p`` others left to condition on.
2. **MCI.** Every link ``X^i_{t-tau} -> X^j_t`` is tested given the selected parents of ``X^j_t``
   (without ``X^i_{t-tau}``) and the selected parents of ``X^i_t`` shifted ``tau`` steps back. The
   link is significant when its p-value is at most ``alpha_level``.

Every test reads one lagged design matrix: column ``lag * n + i`` holds ``X^i_{t-lag}`` for ``lag =
0 .. 2 * tau_max``, over the ``T - 2 * tau_max`` time steps where all those lags exist, so every
test uses the same samples. Every test is a partial-correlation test: ``indep_test="student_t"``
(:class:`~andrey.core.ci.StudentT`, the default) or ``"fisherz"``
(:class:`~andrey.core.ci.FisherZ`). A PC1 round tests all of a target's candidates in one call, and
MCI tests all links into a target together, grouped by conditioning-set size. Any object with
``statistic_pvalues(x, y, conditions) -> (statistics, p_values)`` over the design's column indices
can drive :func:`pc1` and :func:`mci`.

A partial correlation is the correlation of the residuals of ``x`` and ``y`` after a least-squares
fit on the conditioning columns. The test takes it from the inverse of the ``[x, y, *S]``
correlation submatrix; when that submatrix is singular to working precision, as lagged copies of a
strongly autocorrelated series can make it, :class:`DesignTest` takes it from the residuals of the
design instead (:func:`design_test`).

References
----------
Runge, J., Nowack, P., Kretschmer, M., Flaxman, S., and Sejdinovic, D. (2019). Detecting and
quantifying causal associations in large nonlinear time series datasets. Science Advances 5(11),
eaau4996. https://doi.org/10.1126/sciadv.aau4996
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from andrey.core import backend
from andrey.core.ci import FisherZ, PartialCorrelationTest, SingularCorrelationError, StudentT
from andrey.core.score import copies_error
from andrey.search._parallel import _KEY_DP
from andrey.search.hc import tie_ranks

if TYPE_CHECKING:
    from andrey.core.knowledge import LaggedConstraints

# A lagged variable ``(i, lag)``: variable ``i`` at time ``t - lag``.
Link = tuple[int, int]


class LinkTest(Protocol):
    """A conditional-independence test that reports its statistics with the p-values.

    Built once over the lagged design matrix and called with its column indices. A larger p-value
    means more independence; a larger absolute statistic means a stronger dependence.
    """

    def statistic_pvalues(
        self, x: np.ndarray, y: np.ndarray, conditions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Test ``x[i]`` against ``y[i]`` given the columns ``conditions[i]``, for every row.

        ``x`` and ``y`` have shape ``(N,)`` and ``conditions`` shape ``(N, m)``: every query has
        ``m`` conditioners. Returns the ``(N,)`` statistics and p-values.
        """
        ...


# The ``indep_test`` names: each builds a test over the lagged design.
INDEP_TESTS: dict[str, type[PartialCorrelationTest]] = {
    "student_t": StudentT,
    "fisherz": FisherZ,
}


@dataclass(frozen=True)
class PCMCIResult:
    """The PC1 parents and the MCI results of one PCMCI run.

    ``parents[j]`` holds the ``(i, lag)`` parents PC1 selected for variable ``j``, strongest first.
    ``statistics``, ``p_values``, and ``graph`` have shape ``(tau_max, n, n)``: entry
    ``[lag - 1, i, j]`` belongs to the link ``X^i_{t-lag} -> X^j_t``, and ``graph`` is
    ``p_values <= alpha_level``. A link the knowledge forbids reads statistic 0 and p-value 1, and
    a required one 1 and 0.
    """

    parents: tuple[tuple[Link, ...], ...]
    statistics: np.ndarray
    p_values: np.ndarray
    graph: np.ndarray


class DesignTest:
    """A partial-correlation test over a lagged design that answers every query it can.

    A call goes to ``test.statistic_pvalues`` first. When it raises
    :class:`~andrey.core.ci.SingularCorrelationError`, each query of that call is asked alone, and a
    query whose ``[x, y, *S]`` correlation submatrix is singular gets its partial correlation from
    the least-squares residuals of ``design``
    (:meth:`~andrey.core.ci.PartialCorrelationTest.residual_statistic_pvalues`). A call that does
    not raise returns ``test``'s results unchanged.
    """

    def __init__(self, test: PartialCorrelationTest, design: np.ndarray) -> None:
        self.test = test
        self.design = design

    def statistic_pvalues(
        self, x: np.ndarray, y: np.ndarray, conditions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Test ``x[i]`` against ``y[i]`` given the columns ``conditions[i]``, for every row."""
        try:
            return self.test.statistic_pvalues(x, y, conditions)
        except SingularCorrelationError:
            pass
        statistics = np.empty(len(x), dtype=np.float64)
        p_values = np.empty(len(x), dtype=np.float64)
        for k in range(len(x)):
            query = (x[k : k + 1], y[k : k + 1], conditions[k : k + 1])
            try:
                statistic, p_value = self.test.statistic_pvalues(*query)
            except SingularCorrelationError:
                statistic, p_value = self.test.residual_statistic_pvalues(self.design, *query)
            statistics[k], p_values[k] = statistic[0], p_value[0]
        return statistics, p_values


def design_test(indep_test: str, data: np.ndarray, tau_max: int) -> DesignTest:
    """The ``indep_test`` partial-correlation test over the lagged design of ``data``.

    Raises :class:`ValueError` when ``indep_test`` is not a key of :data:`INDEP_TESTS`, and
    :class:`~andrey.core.score.CopiedColumnsError` when design columns repeat others up to scale
    and shift: a variable that copies another, at the same step or some lags apart. A copy at one
    step repeats at every lag of the design, so the error names each group that holds a lag-0
    column, by :func:`design_column_name`.
    """
    if indep_test not in INDEP_TESTS:
        raise ValueError(f"indep_test must be one of {sorted(INDEP_TESTS)}, got {indep_test!r}")
    design = lagged_design(data, tau_max)
    test = INDEP_TESTS[indep_test](design)
    n_vars = data.shape[1]
    groups = [group for group in test.copies() if group[0] < n_vars]
    if groups:
        raise copies_error(groups, design_column_name(n_vars))
    return DesignTest(test, design)


def design_column_name(n_vars: int, name: Callable[[int], str] = str) -> Callable[[int], str]:
    """Name design column ``lag * n_vars + i`` as ``name(i)``, with ``at lag {lag}`` from lag 1."""

    def column(j: int) -> str:
        lag, i = divmod(j, n_vars)
        return name(i) if lag == 0 else f"{name(i)} at lag {lag}"

    return column


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


def _strongest_others(columns: np.ndarray, p: int) -> np.ndarray:
    """Row ``k`` holds the first ``p`` entries of ``columns`` without ``columns[k]``."""
    others = np.broadcast_to(columns[:p], (len(columns), p)).copy()
    for k in range(min(p, len(columns))):  # these rows lose one of the first p entries
        others[k] = np.delete(columns[: p + 1], k)
    return others


def pc1(
    test: LinkTest,
    n_vars: int,
    target: int,
    tau_max: int,
    pc_alpha: float,
    ranks: Sequence[int],
    constraints: LaggedConstraints | None = None,
) -> tuple[Link, ...]:
    """Select the lagged parents of variable ``target`` with the PC1 rounds; strongest first.

    Candidates start in variable-major order, ``(0, 1), (0, 2), ..., (1, 1), ...``. Each round
    re-sorts the survivors by strength rounded to ``_KEY_DP`` decimals, strongest first, and equal
    ones by ``ranks[i]``, then by lag, for the candidate ``(i, lag)``; ``ranks`` holds each
    variable's :func:`~andrey.search.hc.tie_ranks`. Each round is one ``test`` call over all
    remaining candidates. With ``constraints``, a forbidden link is no candidate, and a required one
    is never tested: it stays, with the strength of a partial correlation of ``1``.
    """
    columns = np.array(
        [_column((i, lag), n_vars) for i in range(n_vars) for lag in range(1, tau_max + 1)],
        dtype=np.intp,
    )
    if constraints is not None:
        columns = columns[~constraints.forbidden[columns // n_vars, columns % n_vars, target]]
    tie = np.asarray(ranks, dtype=np.intp)[columns % n_vars] * (tau_max + 1) + columns // n_vars
    strength = np.full(len(columns), math.inf)
    p = 0
    while len(columns) - 1 >= p:
        y = np.full_like(columns, target)
        conditions = _strongest_others(columns, p)
        if constraints is None:
            statistics, p_values = test.statistic_pvalues(columns, y, conditions)
        else:
            statistics, p_values = _known_links_pvalues(
                test,
                constraints.required[columns // n_vars, columns % n_vars, target],
                columns,
                y,
                conditions,
            )
        strength = np.fmin(strength, np.abs(statistics))
        kept = np.flatnonzero(~(p_values > pc_alpha))
        kept = kept[np.lexsort((tie[kept], -np.round(strength[kept], _KEY_DP)))]
        columns, strength, tie = columns[kept], strength[kept], tie[kept]
        p += 1
    return tuple((int(c) % n_vars, int(c) // n_vars) for c in columns)


def _known_links_pvalues(
    test: LinkTest, required: np.ndarray, x: np.ndarray, y: np.ndarray, conditions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``test`` the rows that are not ``required``; a required row reads statistic 1, p-value 0."""
    statistics = np.ones(len(x), dtype=np.float64)
    p_values = np.zeros(len(x), dtype=np.float64)
    free = ~required
    if free.any():
        statistics[free], p_values[free] = test.statistic_pvalues(
            x[free], y[free], conditions[free]
        )
    return statistics, p_values


def mci(
    test: LinkTest,
    n_vars: int,
    parents: tuple[tuple[Link, ...], ...],
    tau_max: int,
    constraints: LaggedConstraints | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the MCI test on every link; return ``(statistics, p_values)``, each ``(tau_max, n, n)``.

    The link ``X^i_{t-lag} -> X^j_t`` is tested given ``parents[j]`` without ``(i, lag)`` and
    ``parents[i]`` shifted ``lag`` steps back. The links into each ``X^j_t`` are tested together,
    one ``test`` call per conditioning-set size. With ``constraints``, no test runs on a forbidden
    link, which reads statistic 0 and p-value 1, or on a required one, which reads 1 and 0.
    """
    statistics = np.zeros((tau_max, n_vars, n_vars))
    p_values = np.ones((tau_max, n_vars, n_vars))
    if constraints is not None:
        statistics[constraints.required[1:]] = 1.0
        p_values[constraints.required[1:]] = 0.0
    for j in range(n_vars):
        groups: dict[int, list[tuple[int, int, list[int]]]] = {}
        for i in range(n_vars):
            for lag in range(1, tau_max + 1):
                if constraints is not None and (
                    constraints.forbidden[lag, i, j] or constraints.required[lag, i, j]
                ):
                    continue
                conditions = {c for c in parents[j] if c != (i, lag)}
                conditions.update((k, s + lag) for k, s in parents[i])
                columns = sorted(_column(c, n_vars) for c in conditions)
                groups.setdefault(len(columns), []).append((i, lag, columns))
        for size, group in groups.items():
            causes = np.array([i for i, _, _ in group], dtype=np.intp)
            lags = np.array([lag for _, lag, _ in group], dtype=np.intp)
            x = lags * n_vars + causes
            conditions = np.array([columns for _, _, columns in group], dtype=np.intp)
            group_statistics, group_p_values = test.statistic_pvalues(
                x, np.full_like(x, j), conditions.reshape(len(group), size)
            )
            statistics[lags - 1, causes, j] = group_statistics
            p_values[lags - 1, causes, j] = group_p_values
    return statistics, p_values


def pcmci(
    data: np.ndarray,
    *,
    tau_max: int = 1,
    pc_alpha: float = 0.2,
    alpha_level: float = 0.05,
    indep_test: str = "student_t",
    constraints: LaggedConstraints | None = None,
) -> PCMCIResult:
    """Run PC1 and MCI on a ``(T, n)`` series with the ``indep_test`` partial-correlation test.

    With ``constraints``, background knowledge over lags ``0`` to ``tau_max``, PC1 and MCI test no
    forbidden or required link (:func:`pc1`, :func:`mci`): a forbidden link is not in the graph and
    a required one is. Raises :class:`ValueError` when ``indep_test`` is not a key of
    :data:`INDEP_TESTS`.
    """
    data = np.asarray(data, dtype=np.float64)
    n_vars = data.shape[1]
    test = design_test(indep_test, data, tau_max)
    ranks = tie_ranks(data).tolist()
    with backend.one_blas_thread():  # the stacked inversions then give the same bits on any host
        parents = tuple(
            pc1(test, n_vars, j, tau_max, pc_alpha, ranks, constraints) for j in range(n_vars)
        )
        statistics, p_values = mci(test, n_vars, parents, tau_max, constraints)
    return PCMCIResult(
        parents=parents, statistics=statistics, p_values=p_values, graph=p_values <= alpha_level
    )
