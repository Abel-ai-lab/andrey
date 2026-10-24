"""Test-wise deletion: conditional-independence tests on data with missing values.

:func:`~andrey.core.ci.make_indep_test` returns a :class:`TestwiseDeletion` when the data hold
``NaN``; data without ``NaN`` get the registered test itself. A query of ``x`` and ``y`` given
``S`` reads only the rows where ``x``, ``y``, and every member of ``S`` are observed: the
registered test is built over those rows of those columns, kept in data order, and asked the same
query, so the p-value is the one that test gives on that block. Each test thus keeps every row its
own variables allow, where deleting every row with a missing value would drop rows for all of them.

Deleting rows per test is sound when the values are missing completely at random (MCAR): the rows a
test keeps are a random sample of all rows, so each test keeps its level and a constraint-based
search stays consistent [1]_ [2]_. When whether a value is missing depends on the variables, the
rows a test keeps are a selected sample. PC's tests can then find dependence where the full
population has none and keep extra edges [2]_; FCI, which allows for selection, stays sound when
no missingness mechanism causally affects another [1]_.

References
----------
.. [1] Strobl, E. V., Visweswaran, S., and Spirtes, P. L. (2018). Fast causal inference with
   non-random missingness by test-wise deletion. International Journal of Data Science and
   Analytics 6(1), 47-62. https://doi.org/10.1007/s41060-017-0094-6
.. [2] Tu, R., Zhang, C., Ackermann, P., Mohan, K., Kjellstrom, H., and Zhang, K. (2019). Causal
   discovery in the presence of missing data. Proceedings of Machine Learning Research 89
   (AISTATS), 1762-1770. https://proceedings.mlr.press/v89/tu19a.html
"""

from __future__ import annotations

import contextlib
from collections import Counter, OrderedDict
from collections.abc import Callable, Generator, Iterable, Mapping
from contextvars import ContextVar
from functools import partial
from typing import Any

import numpy as np
import numpy.typing as npt

from .ci import (
    CGTest,
    ChiSquare,
    CITest,
    DGTest,
    PartialCorrelationTest,
    SingularCorrelationError,
    _canonical,
)
from .score import CopiedColumnsError, copies_error, copies_message, copy_groups

__all__ = ["TestwiseDeletion", "rows_metadata", "rows_tested"]

# The most bytes of row blocks the tests one TestwiseDeletion keeps for reuse hold together.
_CACHE_BYTES = 1 << 27

#: The row counts of the tests run inside :func:`rows_tested`, or ``None`` outside it.
_ROWS: ContextVar[Counter[int] | None] = ContextVar("andrey_rows_tested", default=None)


@contextlib.contextmanager
def rows_tested() -> Generator[Counter[int]]:
    """Count the rows each test-wise deletion test reads inside the block.

    The counter maps a row count to the number of tests that read that many rows; it stays empty
    when no test met a missing value. :func:`rows_metadata` summarizes it.
    """
    counter: Counter[int] = Counter()
    token = _ROWS.set(counter)
    try:
        yield counter
    finally:
        _ROWS.reset(token)


def rows_metadata(counter: Mapping[int, int]) -> dict[str, dict[str, float]]:
    """Return ``{"rows_tested": {"min": ..., "median": ...}}`` for ``counter``; ``{}`` when empty.

    ``min`` is the fewest rows a test read and ``median`` the median over the tests, the mean of
    the two middle counts when the number of tests is even.
    """
    if not counter:
        return {}
    counts = sorted(counter.items())
    total = sum(n for _, n in counts)
    middle = [(total - 1) // 2, total // 2]
    values, seen = [], 0
    for rows, n in counts:
        while middle and middle[0] < seen + n:
            values.append(rows)
            middle.pop(0)
        seen += n
    return {"rows_tested": {"min": counts[0][0], "median": (values[0] + values[1]) / 2}}


class TestwiseDeletion:
    """A conditional-independence test that deletes rows per test on data with missing values.

    Built by :func:`~andrey.core.ci.make_indep_test` over data holding ``NaN``. A query of ``x`` and
    ``y`` given ``S`` keeps the rows observed in every one of the columns ``{x, y} | S``, builds the
    registered test over that block, its columns in data order, and returns that test's p-value
    for the query. Queries are canonicalized as :class:`~andrey.core.ci.FisherZ`'s are. The test of
    a block answers every later query over the same columns; the blocks the reused tests hold stay
    within ``_CACHE_BYTES``, the least recently used dropped first. A nonlinear or a registered
    test, whose state grows with its queries, is built anew for each query.

    A query raises ``ValueError`` when its block has fewer rows than its test needs: 2, or
    ``|S| + 3`` for ``"fisherz"`` and ``"student_t"``, the rows their degrees of freedom need; and
    when a continuous column is constant on the block, as a constant column of the data is rejected
    before a fit. A categorical column with one category on the block is left to the test.

    Parameters
    ----------
    data : ndarray, shape (n_samples, n_features)
        Values with ``NaN`` marking the missing ones; no infinities.
    build : callable
        ``build(block, columns)`` builds the registered test over ``block``, the kept rows of
        ``columns``.
    name : str
        The ``indep_test`` name, for the messages.
    test_class : type or None
        The class ``build`` returns, ``None`` for a registered test. It decides which of the
        partial-correlation methods the wrapper offers, what it reuses, and its BLAS pinning.
    continuous : ndarray of bool, shape (n_features,), or None
        Marks the continuous columns, which a block may not hold constant; ``None`` marks none.
    check_copies : bool
        Raise :class:`~andrey.core.score.CopiedColumnsError` when a partial-correlation block holds
        columns that repeat others up to scale and shift, as
        :func:`~andrey.core.ci.make_indep_test` does on complete data.
    """

    def __init__(
        self,
        data: np.ndarray,
        build: Callable[[np.ndarray, list[int]], CITest],
        *,
        name: str,
        test_class: type | None,
        continuous: np.ndarray | None,
        check_copies: bool,
    ) -> None:
        self._data = data
        self._observed = ~np.isnan(data)
        self._build = build
        self._name = name
        self._partial = test_class is not None and issubclass(test_class, PartialCorrelationTest)
        self._continuous = np.zeros(data.shape[1], dtype=bool) if continuous is None else continuous
        self._most = np.full(data.shape[1], -1, dtype=np.int64)
        self._check_copies = check_copies and self._partial
        self._reuse = test_class in (ChiSquare, DGTest, CGTest) or self._partial
        self._cache: OrderedDict[tuple[int, ...], tuple[CITest, int, int]] = OrderedDict()
        self._held = 0
        self.pin_blas = _pins_blas(test_class, data.shape[0])

    def __call__(self, x: int, y: int, condition_set: Iterable[int] = ()) -> float:
        """Return the p-value for ``x`` independent of ``y`` given ``condition_set``.

        The p-value is the registered test's on the rows observed in all of the query's columns.
        A larger p-value means more independence.
        """
        x_int, y_int, cond = _canonical(x, y, condition_set)
        return float(self._query(lambda test, a, b, s: test(a, b, s), x_int, y_int, cond))

    def _query(self, ask: Callable[..., Any], x: int, y: int, cond: list[int]) -> Any:
        """Return ``ask(test, a, b, s)`` for the canonical query ``x``, ``y``, ``cond``.

        ``test`` is the test over the query's block, and ``a``, ``b``, and ``s`` the query's
        columns as indices of that block. The query's row count enters the running
        :func:`rows_tested` counter once it is answered; a copies error names the data's columns.
        """
        test, columns, rows = self._block_test(x, y, cond)
        at = {c: i for i, c in enumerate(columns)}
        try:
            out = ask(test, at[x], at[y], [at[c] for c in cond])
        except SingularCorrelationError as error:
            raise _in_data_columns(error, columns) from error
        counter = _ROWS.get()
        if counter is not None:
            counter[rows] += 1
        return out

    def _block_test(self, x: int, y: int, cond: list[int]) -> tuple[CITest, list[int], int]:
        """The test over the block of the canonical query, the block's columns, and its rows."""
        columns = tuple(sorted((x, y, *cond)))
        entry = self._cache.get(columns)
        if entry is not None:
            self._cache.move_to_end(columns)
            return entry[0], list(columns), entry[1]
        block = self._block(columns)
        self._check_block(block, len(cond), columns)
        test = self._new_test(block, list(columns))
        if self._reuse and block.nbytes <= _CACHE_BYTES:
            self._cache[columns] = (test, block.shape[0], block.nbytes)
            self._held += block.nbytes
            while self._held > _CACHE_BYTES:
                self._held -= self._cache.popitem(last=False)[1][2]
        return test, list(columns), block.shape[0]

    def _block(self, columns: Iterable[int]) -> np.ndarray:
        """The rows of ``data`` observed in every one of ``columns``, those columns only."""
        index = np.asarray(list(columns), dtype=np.intp)
        rows = np.flatnonzero(self._observed[:, index].all(axis=1))
        return self._data[rows[:, None], index]

    def _check_block(self, block: np.ndarray, size: int, columns: tuple[int, ...]) -> None:
        """Raise ``ValueError`` when ``block`` has too few rows or a constant continuous column."""
        least = size + 3 if self._partial else 2
        if block.shape[0] < least:
            raise ValueError(
                f"indep_test={self._name!r} tests a pair given {size} conditioning variables on "
                f"the {block.shape[0]} rows where all of them are observed, and needs at least "
                f"{least}; drop the columns with the most missing values, or cap the conditioning "
                "sets with max_depth where the method has it"
            )
        # A column can be constant on the block only when one of its values fills every row.
        index = list(columns)
        if not (block.shape[0] <= self._repeats(index)).any():
            return
        flat = np.ptp(block, axis=0) == 0
        if (flat & self._continuous[index]).any():
            raise ValueError(
                f"indep_test={self._name!r} meets a column that is constant on the "
                f"{block.shape[0]} rows where all the columns of a test are observed, so the test "
                "measures no dependence; drop the columns with the most missing values"
            )

    def _repeats(self, columns: list[int]) -> np.ndarray:
        """The most observed rows that share one value, in each of ``columns``.

        Counted when a column is first read; a column that is not continuous counts ``0``, as
        nothing checks it.
        """
        for j in columns:
            if self._most[j] < 0 and not self._continuous[j]:
                self._most[j] = 0
            elif self._most[j] < 0:
                values = self._data[self._observed[:, j], j]
                self._most[j] = int(np.unique(values, return_counts=True)[1].max())
        return self._most[columns]

    def _new_test(self, block: np.ndarray, columns: list[int]) -> CITest:
        """Build the registered test over ``block``; a copies error names the data's columns."""
        try:
            test = self._build(block, columns)
            if self._check_copies:
                groups = test.copies()  # ty: ignore[unresolved-attribute]  # gated by _partial
                if groups:
                    raise copies_error(groups)
        except (CopiedColumnsError, SingularCorrelationError) as error:
            raise _in_data_columns(error, columns) from error
        return test


class PartialCorrelationDeletion(TestwiseDeletion):
    """:class:`TestwiseDeletion` of a partial-correlation test, with its statistics.

    The searches that rank candidates by the partial correlation, and the time-series searches,
    read :meth:`statistic_pvalue`, :meth:`statistic_pvalues`, :meth:`residual_statistic_pvalues`,
    and :meth:`copies`; each reads, for every query, the rows observed in its columns.
    """

    def statistic_pvalue(
        self, x: int, y: int, condition_set: Iterable[int] = ()
    ) -> tuple[float, float]:
        """Return the partial correlation and p-value of ``x`` and ``y`` given ``condition_set``."""
        x_int, y_int, cond = _canonical(x, y, condition_set)
        r, p = self._query(lambda test, a, b, s: test.statistic_pvalue(a, b, s), x_int, y_int, cond)
        return float(r), float(p)

    def statistic_pvalues(
        self, x: np.ndarray, y: np.ndarray, conditions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Partial correlations and p-values of many queries, each on its own rows.

        ``x`` and ``y`` are ``(N,)`` int arrays and ``conditions`` the ``(N, m)`` int array of
        conditioning sets; entry ``i`` equals ``statistic_pvalue(x[i], y[i], conditions[i])``.
        """
        x = np.asarray(x, dtype=np.intp)
        y = np.asarray(y, dtype=np.intp)
        conds = np.asarray(conditions, dtype=np.intp).reshape(len(x), -1)
        r = np.empty(len(x), dtype=np.float64)
        p = np.empty(len(x), dtype=np.float64)
        for k in range(len(x)):
            r[k], p[k] = self.statistic_pvalue(int(x[k]), int(y[k]), conds[k].tolist())
        return r, p

    def residual_statistic_pvalues(
        self, data: np.ndarray, x: np.ndarray, y: np.ndarray, conditions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """:meth:`~andrey.core.ci.PartialCorrelationTest.residual_statistic_pvalues` per query.

        ``data``, the matrix the test was built over, is not read again: each query's residuals
        come from the test's own rows observed in the query's columns.
        """
        del data
        x = np.asarray(x, dtype=np.intp)
        y = np.asarray(y, dtype=np.intp)
        conds = np.asarray(conditions, dtype=np.intp).reshape(len(x), -1)
        r = np.empty(len(x), dtype=np.float64)
        p = np.empty(len(x), dtype=np.float64)
        for k in range(len(x)):
            x_int, y_int, cond = _canonical(int(x[k]), int(y[k]), conds[k].tolist())
            block = self._block(sorted((x_int, y_int, *cond)))

            def residual(test: Any, a: int, b: int, s: list[int]) -> tuple[np.ndarray, ...]:
                return test.residual_statistic_pvalues(
                    block, np.array([a]), np.array([b]), np.array([s], dtype=np.intp)
                )

            rk, pk = self._query(residual, x_int, y_int, cond)
            r[k], p[k] = rk[0], pk[0]
        return r, p

    def copies(self) -> tuple[tuple[int, ...], ...]:
        """The groups of columns that repeat one another up to scale and shift.

        Each pair of columns is compared on the rows where both are observed, by
        :func:`~andrey.core.score.copy_groups`; a pair observed together on at most two rows, or
        constant on them, is no copy.
        """
        if not hasattr(self, "_copies"):
            d = self._data.shape[1]
            corr = np.eye(d)
            for i in range(d):
                for j in range(i + 1, d):
                    rows = self._observed[:, i] & self._observed[:, j]
                    if np.count_nonzero(rows) > 2:
                        pair = self._data[rows][:, [i, j]]
                        if (np.ptp(pair, axis=0) > 0).all():
                            corr[i, j] = corr[j, i] = np.corrcoef(pair, rowvar=False)[0, 1]
            self._copies = copy_groups(corr, self._data.shape[0])
        return self._copies


class BoundedDeletion(TestwiseDeletion):
    """:class:`TestwiseDeletion` of a test that can stop once the side of ``alpha`` is certain."""

    def _bounded_pvalue(
        self,
        x: int,
        y: int,
        condition_set: Iterable[int],
        alpha: float | None,
        exact_above: bool = False,
    ) -> float:
        """The block test's ``_bounded_pvalue`` of the query, on the rows its columns share."""
        x_int, y_int, cond = _canonical(x, y, condition_set)
        return float(
            self._query(
                lambda test, a, b, s: test._bounded_pvalue(a, b, s, alpha, exact_above),
                x_int,
                y_int,
                cond,
            )
        )


def testwise_deletion(
    name: str,
    factory: Callable[..., CITest],
    data: np.ndarray,
    categorical: npt.ArrayLike | None,
    options: Mapping[str, Any],
    *,
    mixed: bool,
    check_copies: bool = True,
) -> TestwiseDeletion:
    """Return the :class:`TestwiseDeletion` of the registered test ``name`` over ``data``.

    ``factory`` builds the test as :func:`~andrey.core.ci.make_indep_test` would, with ``options``,
    and with each block's slice of the ``categorical`` mask when the test is ``mixed``.
    """
    arr = np.asarray(data, dtype=np.float64)
    mask = None if categorical is None else np.asarray(categorical, dtype=bool)
    if mixed:

        def build(block: np.ndarray, columns: list[int]) -> CITest:
            return factory(block, None if mask is None else mask[columns], **options)

    else:

        def build(block: np.ndarray, columns: list[int]) -> CITest:
            return factory(block, **options)

    test_class = _shipped_class(factory)
    if test_class is None:
        continuous = None
    elif test_class is ChiSquare:
        continuous = np.zeros(arr.shape[1], dtype=bool)
    elif mixed:
        continuous = np.ones(arr.shape[1], dtype=bool) if mask is None else ~mask
    else:
        continuous = np.ones(arr.shape[1], dtype=bool)
    if test_class is not None and issubclass(test_class, PartialCorrelationTest):
        kind: type[TestwiseDeletion] = PartialCorrelationDeletion
    elif test_class is not None and hasattr(test_class, "_bounded_pvalue"):
        kind = BoundedDeletion
    else:
        kind = TestwiseDeletion
    return kind(
        arr,
        build,
        name=name,
        test_class=test_class,
        continuous=continuous,
        check_copies=check_copies,
    )


def _shipped_class(factory: Callable[..., CITest]) -> type | None:
    """The class a shipped test's factory builds; ``None`` for a registered one."""
    from . import ci

    if isinstance(factory, type):
        return factory
    if isinstance(factory, partial) and factory.func is ChiSquare:
        return ChiSquare
    if isinstance(factory, partial) and factory.func is ci._nonlinear:
        module, cls = ci._NONLINEAR_CLASSES[factory.args[0]]
        import importlib

        return getattr(importlib.import_module(f"{__package__}.{module}"), cls)
    return None


def _pins_blas(test_class: type | None, n: int) -> bool:
    """Whether the skeleton scans pin BLAS to one thread, as :func:`~andrey.core.ci.pins_blas`
    decides for the test over all ``n`` rows."""
    if test_class is None:
        return False
    pin = getattr(test_class, "pin_blas", None)
    if isinstance(pin, property):  # KCI: unpinned from a row count on
        from .ci_kernel import _KCI_BLAS_THREADS_FROM

        return n < _KCI_BLAS_THREADS_FROM
    if pin is not None:
        return bool(pin)
    return issubclass(test_class, (PartialCorrelationTest, DGTest, CGTest))


def _in_data_columns(
    error: CopiedColumnsError | SingularCorrelationError, columns: list[int]
) -> CopiedColumnsError | SingularCorrelationError:
    """``error`` with its groups of copies, positions in a block, as the data's columns."""
    if not error.groups:
        return error
    groups = tuple(tuple(columns[i] for i in group) for group in error.groups)
    renamed = type(error)(str(error).replace(copies_message(error.groups), copies_message(groups)))
    renamed.groups = groups
    return renamed
