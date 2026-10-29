"""Test-wise deletion: conditional-independence tests on data with missing values.

:func:`~andrey.core.ci.make_indep_test` returns a :class:`TestwiseDeletion` when the data hold
``NaN``; data without ``NaN`` get the registered test itself. A query of ``x`` and ``y`` given
``S`` reads only the rows where ``x``, ``y``, and every member of ``S`` are observed. For
:class:`~andrey.core.ci.FisherZ` and :class:`~andrey.core.ci.StudentT`, a
:class:`GaussianDeletion` computes the query's correlations directly from those rows, many queries
at once, so the p-value is the one the test gives on that block up to rounding. Every other test is
built over those rows of those columns, kept in data order, and asked the same query, so the
p-value is the one that test gives on that block exactly. Each test thus keeps every row its own
variables allow, where deleting every row with a missing value would drop rows for all of them.

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
import functools
from collections import Counter, OrderedDict
from collections.abc import Callable, Generator, Iterable, Mapping
from contextvars import ContextVar
from functools import partial
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from . import backend
from .ci import (
    _BOUND_MARGIN,
    _ONE_MINUS_EPS,
    _SINGULAR_RTOL,
    CGTest,
    ChiSquare,
    CITest,
    DGTest,
    FisherZ,
    PartialCorrelationTest,
    SingularCorrelationError,
    StudentT,
    _canonical,
)
from .score import (
    ZERO_RESIDUAL,
    CopiedColumnsError,
    copies_error,
    copies_message,
    copy_groups,
)

__all__ = ["TestwiseDeletion"]

# The most bytes of row blocks the tests one TestwiseDeletion keeps for reuse hold together.
_CACHE_BYTES = 1 << 27

# The most bytes of row values GaussianDeletion reads at once to compute new blocks' matrices.
_STEP_BYTES = 1 << 23
# The bytes of the Python objects that keep one block's matrix in GaussianDeletion's cache.
_ENTRY_BYTES = 512
# How far inside the exact path's thresholds a query stays to keep GaussianDeletion's own answer:
# a query whose block nears a copy, a singular correlation submatrix, or a constant column is
# answered by the exact path, which then decides it.
_MARGIN = 100.0
# The degrees of freedom of a partial-correlation test given |S| conditioners on n rows are
# n - |S| - offset.
_DF_OFFSET: dict[type, int] = {FisherZ: 3, StudentT: 2}

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
    test, whose state grows with its queries, is built anew for each query. A partial-correlation
    test gets the :class:`GaussianDeletion` subclass, which builds no test per block.

    A query raises ``ValueError`` when its block has fewer rows than its test needs: 2, or
    ``|S| + 3`` for ``"fisherz"`` and ``"student_t"``, the rows their degrees of freedom need; when
    a continuous column is constant on the block, as a constant column of the data is rejected
    before a fit; and when the test fails on the block or gives a p-value that is not a number, as
    a mixed-data test can on a block with fewer rows than columns. A categorical column with one
    category on the block is left to the test.

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
        columns as indices of that block; the answer's last entry is the p-value. The query's row
        count enters the running :func:`rows_tested` counter once it is answered. A copies error
        names the data's columns, and another error of the test on its block, or a p-value that is
        not a number, raises ``ValueError`` saying how many rows the block holds.
        """
        test, columns, rows = self._block_test(x, y, cond)
        at = {c: i for i, c in enumerate(columns)}
        try:
            out = ask(test, at[x], at[y], [at[c] for c in cond])
        except SingularCorrelationError as error:
            raise _in_data_columns(error, columns) from error
        except (ValueError, ArithmeticError) as error:
            raise self._failed(rows, f": {error}") from error
        p = np.asarray(out[-1] if isinstance(out, tuple) else out, dtype=np.float64)
        if not np.isfinite(p).all():
            raise self._failed(rows, " gives no p-value")
        counter = _ROWS.get()
        if counter is not None:
            counter[rows] += 1
        return out

    def _failed(self, rows: int, what: str) -> ValueError:
        """The error of a test that fails on a block of ``rows`` rows, ``what`` saying how."""
        return ValueError(
            f"indep_test={self._name!r} on the {rows} rows where all the columns of one of its "
            f"tests are observed{what}; drop the columns with the most missing values, or cap the "
            "conditioning sets with max_depth where the method has it"
        )

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
        """Build the registered test over ``block``; a copies error names the data's columns, and
        another error of the build raises as :meth:`_query` says."""
        try:
            test = self._build(block, columns)
            if self._check_copies:
                groups = test.copies()  # ty: ignore[unresolved-attribute]  # gated by _partial
                if groups:
                    raise copies_error(groups)
        except (CopiedColumnsError, SingularCorrelationError) as error:
            raise _in_data_columns(error, columns) from error
        except (ValueError, ArithmeticError) as error:
            raise self._failed(block.shape[0], f": {error}") from error
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


class GaussianDeletion(PartialCorrelationDeletion):
    """:class:`PartialCorrelationDeletion` of :class:`~andrey.core.ci.FisherZ` or
    :class:`~andrey.core.ci.StudentT` that answers many queries at once.

    A query's correlation submatrix is the correlation of the rows observed in all of its columns,
    computed directly, with no test built for the block: from sums over those rows of each column's
    values less its observed mean, for every pair of columns once (:meth:`_pairwise`) and for every
    larger block once while it is kept (:meth:`_submatrices`). Its partial correlation, clamp,
    degrees of freedom, and p-value are those of the block's test, so the p-value equals that
    test's up to rounding, as the sums run in another order. A query's answer does not depend on
    the queries evaluated with it, so every query path (a single call, :meth:`batched_call`,
    :meth:`statistic_pvalues`, and the canonical query arrays the skeleton searches pass) gives it
    the same bits.

    A query whose block has fewer rows than the test needs, a continuous column that may be
    constant on the block, columns within ``_MARGIN`` of a copy, a correlation submatrix within
    ``_MARGIN`` of the singular bound, or a partial correlation or p-value that is not a number is
    answered by :class:`PartialCorrelationDeletion` instead, which raises where it raises. So every
    error is the exact path's, at the same query.
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
        super().__init__(
            data,
            build,
            name=name,
            test_class=test_class,
            continuous=continuous,
            check_copies=check_copies,
        )
        self._kind = cast("type[PartialCorrelationTest]", test_class)  # FisherZ or StudentT
        self._offset = _DF_OFFSET[self._kind]
        # Each column less its observed mean, 0 where unobserved, with the observed rows: one row
        # of n values per column, so a block's columns are contiguous.
        values = np.where(self._observed, data, 0.0)
        values -= values.sum(axis=0) / np.maximum(self._observed.sum(axis=0), 1)
        self._values = np.ascontiguousarray(np.where(self._observed, values, 0.0).T)
        self._seen = np.ascontiguousarray(self._observed.T)
        self._pairs: tuple[np.ndarray, np.ndarray] | None = None
        self._blocks: OrderedDict[tuple[int, ...], tuple[np.ndarray, int]] = OrderedDict()
        self._kept = 0
        self._repeats(list(range(data.shape[1])))

    def __call__(self, x: int, y: int, condition_set: Iterable[int] = ()) -> float:
        """Return the p-value for ``x`` independent of ``y`` given ``condition_set``.

        The p-value is the registered test's on the rows observed in all of the query's columns, up
        to rounding. A larger p-value means more independence.
        """
        return self.statistic_pvalue(x, y, condition_set)[1]

    def statistic_pvalue(
        self, x: int, y: int, condition_set: Iterable[int] = ()
    ) -> tuple[float, float]:
        """Return the partial correlation and p-value of ``x`` and ``y`` given ``condition_set``."""
        x_int, y_int, cond = _canonical(x, y, condition_set)
        r, p = self._answer(
            np.array([x_int]), np.array([y_int]), np.array([cond], dtype=np.intp).reshape(1, -1)
        )
        return float(r[0]), float(p[0])

    def statistic_pvalues(
        self, x: np.ndarray, y: np.ndarray, conditions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Partial correlations and p-values of many queries sharing one conditioning-set size.

        ``x`` and ``y`` are ``(N,)`` int arrays and ``conditions`` the ``(N, m)`` int array of
        conditioning sets, each distinct and excluding ``x[i]`` and ``y[i]``; entry ``i`` equals
        ``statistic_pvalue(x[i], y[i], conditions[i])``.
        """
        x = np.asarray(x, dtype=np.intp)
        y = np.asarray(y, dtype=np.intp)
        conds = np.sort(np.asarray(conditions, dtype=np.intp).reshape(len(x), -1), axis=1)
        return self._answer(np.minimum(x, y), np.maximum(x, y), conds)

    def _canonical_pvalues(
        self, x: np.ndarray, y: np.ndarray, conditions: np.ndarray
    ) -> np.ndarray:
        """P-values of many canonical queries sharing one conditioning-set size.

        ``x[i] < y[i]``, and row ``i`` of ``conditions`` is sorted and distinct and excludes both,
        as the skeleton searches build their queries; entry ``i`` equals
        ``self(x[i], y[i], conditions[i])``.
        """
        return self._answer(
            np.asarray(x, dtype=np.intp),
            np.asarray(y, dtype=np.intp),
            np.asarray(conditions, dtype=np.intp).reshape(len(x), -1),
        )[1]

    def batched_call(self, x: int, y: int, condition_sets: Iterable[Iterable[int]]) -> list[float]:
        """Return the p-values of one pair over many conditioning sets, in their order.

        Entry ``i`` equals ``self(x, y, condition_sets[i])``; the sets may differ in size.
        """
        queries = [_canonical(x, y, cond) for cond in condition_sets]
        sizes: dict[int, list[int]] = {}
        for i, (_, _, cond) in enumerate(queries):
            sizes.setdefault(len(cond), []).append(i)
        r = np.empty(len(queries), dtype=np.float64)
        p = np.empty(len(queries), dtype=np.float64)
        rows = np.empty(len(queries), dtype=np.int64)
        flagged = np.empty(len(queries), dtype=bool)
        for size, at in sizes.items():
            arrays = _arrays([queries[i] for i in at], size)
            r[at], p[at], rows[at], flagged[at] = self._fast(*arrays)
        self._settle(queries.__getitem__, r, p, rows, flagged)
        return p.tolist()

    def _answer(
        self, lo: np.ndarray, hi: np.ndarray, conds: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """The partial correlations and p-values of canonical queries of one size, in order."""
        r, p, rows, flagged = self._fast(lo, hi, conds)
        self._settle(lambda k: (int(lo[k]), int(hi[k]), conds[k].tolist()), r, p, rows, flagged)
        return r, p

    def _settle(
        self,
        query: Callable[[int], tuple[int, int, list[int]]],
        r: np.ndarray,
        p: np.ndarray,
        rows: np.ndarray,
        flagged: np.ndarray,
    ) -> None:
        """Answer the ``flagged`` queries by the exact path, in order, and count every query.

        ``query(k)`` is query ``k`` as ``(x, y, conditioning set)``. ``r`` and ``p`` hold the fast
        answers and take the exact ones in place; ``rows`` holds each query's row count. The
        queries are counted in order, so a query that raises does so after the queries before it,
        as the exact path's loop does.
        """
        counter = _ROWS.get()
        start = 0
        for k in np.flatnonzero(flagged).tolist():
            if counter is not None:
                counter.update(rows[start:k].tolist())
            x, y, cond = query(k)
            r[k], p[k] = self._query(
                lambda test, a, b, s: test.statistic_pvalue(a, b, s), x, y, cond
            )
            start = k + 1
        if counter is not None:
            counter.update(rows[start:].tolist())

    def _fast(
        self, lo: np.ndarray, hi: np.ndarray, conds: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(r, p, rows, flagged)`` of canonical queries of one size, without the exact path.

        ``rows`` is each query's row count and ``flagged`` marks the queries the exact path
        answers instead (see the class); their ``r`` and ``p`` are meaningless.
        """
        size = conds.shape[1]
        columns = np.empty((len(lo), size + 2), dtype=np.intp)
        columns[:, 0], columns[:, 1], columns[:, 2:] = lo, hi, conds
        with np.errstate(all="ignore"):
            if size == 0:
                count, corr = self._pairwise()
                rows = count[lo, hi]
                stack = np.empty((len(lo), 2, 2), dtype=np.float64)
                stack[:, 0, 0] = stack[:, 1, 1] = 1.0
                stack[:, 0, 1] = stack[:, 1, 0] = corr[lo, hi]
            else:
                stack, rows = self._submatrices(columns)
            flagged = rows < size + 3
            flagged |= (rows[:, None] <= self._most[columns]).any(axis=1)
            flagged |= ~np.isfinite(stack).all(axis=(1, 2))
            if self._check_copies:
                i, j = _apart(size + 2)
                flagged |= (1.0 - stack[:, i, j] ** 2 <= _MARGIN * ZERO_RESIDUAL).any(axis=1)
            if size == 0:
                r = stack[:, 0, 1]
            elif size == 1:
                rxz, ryz = stack[:, 0, 2], stack[:, 1, 2]
                r = (stack[:, 0, 1] - rxz * ryz) / np.sqrt((1.0 - rxz * rxz) * (1.0 - ryz * ryz))
            else:
                inverse, singular = _inverse(stack, flagged)
                flagged |= singular
                r = -inverse[:, 0, 1] / np.sqrt(np.abs(inverse[:, 0, 0] * inverse[:, 1, 1]))
            r = np.clip(r, -_ONE_MINUS_EPS, _ONE_MINUS_EPS)
            p = np.asarray(self._kind._pvalue(r, rows - size - self._offset), dtype=np.float64)
        flagged |= ~(np.isfinite(r) & np.isfinite(p))
        return r, p, rows, flagged

    def _pairwise(self) -> tuple[np.ndarray, np.ndarray]:
        """The rows each pair of columns shares and the pair's correlation on them, ``(d, d)``.

        Computed once, from sums over the rows both columns observe of the values less each
        column's observed mean: ``[i, j]`` is canonical for ``i < j``.
        """
        if self._pairs is None:
            seen = self._observed.astype(np.float64)
            values = self._values.T
            # One BLAS thread: the products' bits then do not depend on the thread count.
            with np.errstate(all="ignore"), backend.one_blas_thread():
                count = seen.T @ seen
                sums = values.T @ seen  # [i, j]: column i's sum over the rows both observe
                spread = (values * values).T @ seen - sums * sums / count
                cov = values.T @ values - sums * sums.T / count
                corr = np.clip(cov / np.sqrt(spread * spread.T), -1.0, 1.0)
            self._pairs = (np.rint(count).astype(np.int64), corr)
        return self._pairs

    def _submatrices(self, columns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The correlation submatrices of the queries whose columns ``[x, y, *S]`` are the rows of
        ``columns``, on the rows each observes, and those row counts.

        A block's matrix is computed once, over its columns in data order, from the sums over its
        rows of the values less each column's observed mean, and kept for later queries over the
        same columns within ``_CACHE_BYTES``, each taking ``_ENTRY_BYTES`` besides its matrix; a
        query reads it in its own order. The blocks not
        kept are computed in steps of at most ``_STEP_BYTES`` of row values.
        """
        k = columns.shape[1]
        keys = [tuple(row) for row in np.sort(columns, axis=1).tolist()]
        new = list(dict.fromkeys(key for key in keys if key not in self._blocks))
        step = max(1, _STEP_BYTES // (8 * k * self._data.shape[0]))
        for start in range(0, len(new), step):
            chunk = new[start : start + step]
            blocks, counts = _block_correlations(
                self._values, self._seen, np.array(chunk, dtype=np.intp)
            )
            for key, block, count in zip(chunk, blocks, counts.tolist()):
                self._blocks[key] = (block, count)
                self._kept += block.nbytes + _ENTRY_BYTES
        found = [self._blocks[key] for key in keys]
        for key in keys:
            self._blocks.move_to_end(key)
        while self._kept > _CACHE_BYTES:
            self._kept -= self._blocks.popitem(last=False)[1][0].nbytes + _ENTRY_BYTES
        ordered = np.stack([block for block, _ in found])
        rank = np.argsort(np.argsort(columns, axis=1), axis=1)
        at = np.arange(len(columns))[:, None, None]
        stack = ordered[at, rank[:, :, None], rank[:, None, :]]
        return stack, np.array([count for _, count in found], dtype=np.int64)

    def copies(self) -> tuple[tuple[int, ...], ...]:
        """The groups of columns that repeat one another up to scale and shift.

        Each pair of columns is compared on the rows where both are observed, as
        :meth:`PartialCorrelationDeletion.copies` compares them, with the same groups: a pair whose
        correlation from :meth:`_pairwise` is within ``_MARGIN`` of a copy, or is not a number, is
        compared on its rows as the exact path compares it, and every other pair is no copy.
        """
        if not hasattr(self, "_copies"):
            count, corr = self._pairwise()
            with np.errstate(invalid="ignore"):
                near = (count > 2) & ~(1.0 - corr * corr > _MARGIN * ZERO_RESIDUAL)
            exact = np.eye(self._data.shape[1])
            for a, b in zip(*np.nonzero(np.triu(near, 1))):
                rows = self._observed[:, a] & self._observed[:, b]
                pair = self._data[rows][:, [a, b]]
                if (np.ptp(pair, axis=0) > 0).all():
                    exact[a, b] = exact[b, a] = np.corrcoef(pair, rowvar=False)[0, 1]
            self._copies = copy_groups(exact, self._data.shape[0])
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
    if test_class in _DF_OFFSET:
        kind: type[TestwiseDeletion] = GaussianDeletion
    elif test_class is not None and issubclass(test_class, PartialCorrelationTest):
        kind = PartialCorrelationDeletion
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


def _block_correlations(
    values: np.ndarray, seen: np.ndarray, columns: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The correlation matrices of the blocks whose columns are the rows of ``columns``, and their
    row counts.

    ``values`` holds each column's values less its observed mean, ``0`` where unobserved, and
    ``seen`` its observed rows, one row per column. Block ``b`` keeps the rows observed in all of
    ``columns[b]``; its covariances come from the sums over every row with the others zeroed, the
    sums of its own values only, so a block's matrix does not depend on the others computed with
    it. Clipped to ``[-1, 1]``, as :func:`numpy.corrcoef` clips.
    """
    kept = seen[columns[:, 0]].copy()
    for c in range(1, columns.shape[1]):
        kept &= seen[columns[:, c]]
    count = np.count_nonzero(kept, axis=1)
    block = values[columns] * kept[:, None, :]
    sums = block.sum(axis=2)
    # einsum sums each product along its row in NumPy's own loop, the same whatever the BLAS.
    products = np.einsum("bin,bjn->bij", block, block)
    cov = products - sums[:, :, None] * sums[:, None, :] / count[:, None, None]
    spread = np.sqrt(np.diagonal(cov, axis1=1, axis2=2))
    return np.clip(cov / spread[:, :, None] / spread[:, None, :], -1.0, 1.0), count


@functools.cache
def _apart(k: int) -> tuple[np.ndarray, np.ndarray]:
    """The row and column indices of the entries above the diagonal of a ``(k, k)`` matrix."""
    return np.triu_indices(k, 1)


def _arrays(
    queries: list[tuple[int, int, list[int]]], size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The canonical query arrays ``(x, y, conditions)`` of ``queries``, each of ``size``."""
    lo = np.array([x for x, _, _ in queries], dtype=np.intp)
    hi = np.array([y for _, y, _ in queries], dtype=np.intp)
    conds = np.array([cond for _, _, cond in queries], dtype=np.intp).reshape(len(queries), size)
    return lo, hi, conds


def _inverse(stack: np.ndarray, skip: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The inverses of a ``(N, k, k)`` stack of correlation submatrices, and which are near
    singular.

    A matrix is near singular when an entry of its inverse reaches ``1 / _MARGIN`` of the bound
    :func:`~andrey.core.ci._invert_nonsingular` settles a matrix by, or, when the stack has an
    exactly singular matrix, when its smallest eigenvalue is at most ``_MARGIN`` times
    ``_SINGULAR_RTOL`` of its largest. The rows ``skip`` marks are inverted as the identity. Each
    inverse is that of its matrix alone.
    """
    k = stack.shape[-1]
    stack = np.where(skip[:, None, None], np.eye(k), stack)
    singular = np.zeros(len(stack), dtype=bool)
    try:
        inverse = np.linalg.inv(stack)
    except np.linalg.LinAlgError:
        w = np.linalg.eigvalsh(stack)
        singular = ~(w[:, 0] > w[:, -1] * _MARGIN * _SINGULAR_RTOL)
        stack[singular] = np.eye(k)
        inverse = np.linalg.inv(stack)
    limit = 1.0 / (_MARGIN * _BOUND_MARGIN * _SINGULAR_RTOL * k * k)
    return inverse, singular | ~(np.abs(inverse).max(axis=(1, 2)) < limit)


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
    decides for the test over all ``n`` rows; a registered test, which may wrap a pinned one, is
    pinned, which changes no result."""
    if test_class is None:
        return True
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
