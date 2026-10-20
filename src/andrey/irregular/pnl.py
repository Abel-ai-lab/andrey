"""Post-nonlinear (PNL) direction test, the engine behind :func:`andrey.pnl`.

The algorithm, its settings, and its references are described on :func:`andrey.pnl`.

In each direction the fit finds an inner transform ``G1`` of the assumed cause and an invertible
outer transform ``G2`` of the assumed effect that make ``e = G2(effect) - G1(cause)`` independent
of the cause, by maximizing ``-0.5 * sum(e**2) + sum(log|G2'(effect)|)``. Both transforms are sums
of fixed smooth basis functions, so ``G1`` is profiled out by least squares, and a damped Newton
iteration with a log barrier, which keeps ``G2`` increasing, reaches the global optimum of a
convex program. Each variable is first mapped to normal scores by rank.

:func:`pnl_pairs` runs the test on many pairs of one table's columns. The ordered pairs are grouped
by cause, so a column's normal scores, basis designs, and centered HSIC kernel are computed once and
shared by every pair that uses it, and each kernel is built in row blocks into reused buffers. Each
pair's four numbers equal :func:`pnl`'s on that pair, bit for bit.
"""

from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.special import gammainc
from scipy.stats import norm, rankdata

from andrey.core import backend
from andrey.core.independence import _empirical_bandwidth, _standardize

_N_FEATURES = 10
_RIDGE = 1e-8
_BARRIER_FLOOR = 1e-6
_MAX_NEWTON = 60
_TOL = 1e-8

# Kernel elements computed per block (rows times columns), so a block's elementwise steps run in
# cache.
_BLOCK_ELEMENTS = 32768


def _gaussianize(values: np.ndarray) -> np.ndarray:
    """Return the rank-based normal-score transform of ``values`` (marginal mapped to Gaussian).

    Each value is replaced by the standard-normal quantile of its empirical rank. This monotone
    per-variable warp partly undoes the outer nonlinearity, so the fitted outer transform only has
    to correct the residual distortion rather than represent a steep inverse from scratch; it
    matches the "make each variable close to Gaussian" preprocessing the post-nonlinear model
    prescribes. Applied identically to both variables, it does not bias the inferred direction.
    """
    n = values.shape[0]
    if n == 0:
        return values.astype(np.float64)
    ranks = rankdata(values, method="average")
    return norm.ppf((ranks - 0.5) / n)


def _rbf_centers(values: np.ndarray) -> tuple[np.ndarray, float]:
    """Return evenly spread Gaussian-feature centers and a shared bandwidth for ``values``.

    Centers span the 2nd-to-98th percentile range so extreme points do not stretch the grid; the
    bandwidth is the center spacing, giving smoothly overlapping features.
    """
    lo, hi = np.percentile(values, [2.0, 98.0])
    if hi - lo < 1e-9:
        lo, hi = values.min() - 1.0, values.max() + 1.0
    centers = np.linspace(lo, hi, _N_FEATURES)
    bandwidth = (hi - lo) / max(_N_FEATURES - 1, 1)
    return centers, max(bandwidth, 1e-6)


def _design(values: np.ndarray, *, intercept: bool) -> np.ndarray:
    """Return the basis design matrix for ``values``: a linear column plus Gaussian radial features.

    With ``intercept`` an all-ones column leads, absorbing an additive offset (used for the inner
    transform so the outer transform's constant is unidentified and can be dropped).
    """
    centers, bandwidth = _rbf_centers(values)
    gaussians = np.exp(-0.5 * ((values[:, None] - centers[None, :]) / bandwidth) ** 2)
    columns = [values[:, None], gaussians]
    if intercept:
        columns.insert(0, np.ones((values.shape[0], 1)))
    return np.hstack(columns)


def _design_derivative(values: np.ndarray) -> np.ndarray:
    """Return the derivative of the no-intercept design of ``values`` column by column.

    The linear column differentiates to ones; each Gaussian feature to
    ``-(v - c) / bandwidth**2 * feature``. Used for the outer transform's Jacobian term.
    """
    centers, bandwidth = _rbf_centers(values)
    offset = (values[:, None] - centers[None, :]) / bandwidth
    gaussians = np.exp(-0.5 * offset**2)
    gauss_deriv = -(offset / bandwidth) * gaussians
    return np.hstack([np.ones((values.shape[0], 1)), gauss_deriv])


def _fit_disturbance(phi: np.ndarray, outer: np.ndarray, outer_deriv: np.ndarray) -> np.ndarray:
    """Recover the disturbance ``e = G2(effect) - G1(cause)`` under the ``cause -> effect`` model.

    ``phi`` is the cause's design with an intercept; ``outer`` and ``outer_deriv`` are the effect's
    design without one and its derivative. ``G1`` is profiled out by regressing the outer transform
    on the cause's basis, so the objective ``0.5 * ||(I - P) G2(effect)||**2 - sum(log
    G2'(effect))`` is convex in ``G2``'s coefficients; a damped Newton iteration started at the
    identity transform drives it to the global optimum while a backtracking line search keeps
    ``G2'`` positive on every sample.
    """
    # Residualize the outer basis against the inner basis (profiles out G1 by least squares).
    projection, *_ = np.linalg.lstsq(phi, outer, rcond=None)
    residual_basis = outer - phi @ projection
    gram = residual_basis.T @ residual_basis
    gram += _RIDGE * np.eye(gram.shape[0])

    n_params = outer.shape[1]
    weights = np.zeros(n_params)
    weights[0] = 1.0  # identity outer transform: derivative is one everywhere, strictly feasible.

    def objective(w: np.ndarray) -> float:
        slopes = outer_deriv @ w
        if np.any(slopes <= _BARRIER_FLOOR):
            return np.inf
        return 0.5 * float(w @ gram @ w) - float(np.sum(np.log(slopes)))

    value = objective(weights)
    for _ in range(_MAX_NEWTON):
        slopes = outer_deriv @ weights
        inv_slopes = 1.0 / slopes
        gradient = gram @ weights - outer_deriv.T @ inv_slopes
        hessian = gram + outer_deriv.T @ (outer_deriv * (inv_slopes**2)[:, None])
        step = np.linalg.solve(hessian + _RIDGE * np.eye(n_params), -gradient)
        if np.max(np.abs(step)) < _TOL:
            break
        alpha = 1.0
        while alpha > 1e-10:
            candidate = weights + alpha * step
            candidate_value = objective(candidate)
            if candidate_value < value - 1e-4 * alpha * float(gradient @ step):
                break
            alpha *= 0.5
        if alpha <= 1e-10:
            break
        weights, value = candidate, candidate_value

    return residual_basis @ weights


@dataclass(frozen=True)
class _Column:
    """One variable's quantities that every pair it belongs to shares."""

    standardized: np.ndarray  # its normal scores z-scored as the HSIC test does
    width: float  # the HSIC test's kernel width for its row count
    phi: np.ndarray  # the inner design, used when it is the cause
    outer: np.ndarray  # the outer design and its derivative, used when it is the effect
    outer_deriv: np.ndarray


def _column(values: np.ndarray) -> _Column:
    """Return the shared quantities of one variable's raw ``values``."""
    scores = _gaussianize(np.asarray(values, dtype=np.float64).ravel())
    raw = scores.reshape(-1, 1)
    return _Column(
        standardized=_standardize(raw)[:, 0],
        width=_empirical_bandwidth(raw),
        phi=_design(scores, intercept=True),
        outer=_design(scores, intercept=False),
        outer_deriv=_design_derivative(scores),
    )


@dataclass(frozen=True)
class _Buffers:
    """One worker's reused ``n x n`` matrices and the row block the kernels are built in."""

    cause: np.ndarray
    disturbance: np.ndarray
    product: np.ndarray
    block: np.ndarray

    @classmethod
    def for_rows(cls, n: int) -> _Buffers:
        rows = min(n, max(1, _BLOCK_ELEMENTS // max(n, 1)))
        return cls(np.empty((n, n)), np.empty((n, n)), np.empty((n, n)), np.empty((rows, n)))


def _centered_kernel(z: np.ndarray, width: float, out: np.ndarray, block: np.ndarray) -> np.ndarray:
    """Write the centered Gaussian kernel of the standardized 1-D ``z`` into ``out`` and return it.

    The matrix equals, bit for bit, the one :func:`~andrey.core.independence.hsic_statistic_test`
    builds: every element takes the same operations in the same order, ``exp(-0.5 * d**2 * width)``
    and then ``K - (col_sum_j + col_sum_i) / n + total / n**2``, but in row blocks of ``block``'s
    size written into ``out``, so no ``n x n`` temporary is allocated.
    """
    n = z.shape[0]
    rows = block.shape[0]
    for start in range(0, n, rows):
        part = out[start : start + rows]
        np.subtract(z[start : start + rows, None], z[None, :], out=part)
        np.square(part, out=part)
        np.multiply(part, -0.5, out=part)
        np.multiply(part, width, out=part)
        np.exp(part, out=part)
    col_sums = out.sum(axis=0)
    shift = col_sums.sum() / n**2
    for start in range(0, n, rows):
        part = out[start : start + rows]
        offsets = block[: part.shape[0]]
        np.add(col_sums[None, :], col_sums[start : start + rows, None], out=offsets)
        np.divide(offsets, n, out=offsets)
        np.subtract(part, offsets, out=part)
        np.add(part, shift, out=part)
    return out


def _hsic(
    kxc: np.ndarray, trace_x: float, sumsq_x: float, disturbance: np.ndarray, buffers: _Buffers
) -> tuple[float, float]:
    """Return ``hsic_statistic_test(cause, disturbance)`` given the cause's centered kernel.

    ``kxc`` is the cause's centered kernel, ``trace_x`` its trace, and ``sumsq_x`` the sum of its
    squared entries; the statistic and the gamma null's moments take the same operations, in the
    same order, as in :func:`~andrey.core.independence.hsic_statistic_test`.
    """
    n = disturbance.shape[0]
    raw = disturbance.reshape(-1, 1)
    kyc = _centered_kernel(
        _standardize(raw)[:, 0], _empirical_bandwidth(raw), buffers.disturbance, buffers.block
    )
    statistic = np.sum(np.multiply(kxc, kyc, out=buffers.product))
    mean_appr = trace_x * np.trace(kyc) / n
    var_appr = 2 * sumsq_x * np.sum(np.square(kyc, out=buffers.product)) / n / n
    shape = mean_appr**2 / var_appr
    scale = var_appr / mean_appr
    return float(statistic / mean_appr), float(1.0 - gammainc(shape, statistic / scale))


def _run_tasks(
    columns: dict[int, _Column], tasks: list[tuple[int, int]]
) -> list[tuple[float, float]]:
    """Return ``(statistic, p_value)`` for each ``(cause, effect)`` task, in order.

    The cause's kernel is rebuilt only when the cause changes, so tasks sorted by cause share it.
    """
    n = next(iter(columns.values())).standardized.shape[0]
    buffers = _Buffers.for_rows(n)
    results = []
    current = None
    for cause, effect in tasks:
        source, target = columns[cause], columns[effect]
        if cause != current:
            current = cause
            kxc = _centered_kernel(source.standardized, source.width, buffers.cause, buffers.block)
            trace_x = np.trace(kxc)
            sumsq_x = np.sum(np.square(kxc, out=buffers.product))
        disturbance = _fit_disturbance(source.phi, target.outer, target.outer_deriv)
        results.append(_hsic(kxc, trace_x, sumsq_x, disturbance, buffers))
    return results


def _ordered_tests(
    columns: dict[int, _Column], tasks: list[tuple[int, int]], workers: int
) -> dict[tuple[int, int], tuple[float, float]]:
    """Return ``{(cause, effect): (statistic, p_value)}`` for the distinct ``tasks``.

    The tasks, sorted by cause, are cut into ``workers`` contiguous runs, one per worker thread,
    each on one BLAS thread; every task's result is the same whatever the number of workers.
    """
    ordered = sorted(dict.fromkeys(tasks))
    runs = [
        [ordered[k] for k in part]
        for part in np.array_split(np.arange(len(ordered)), max(1, workers))
        if part.size
    ]
    if len(runs) <= 1:
        results = [_run_tasks(columns, ordered)]
    else:
        with backend.one_blas_thread(), ThreadPoolExecutor(max_workers=len(runs)) as pool:
            results = list(pool.map(lambda run: _run_tasks(columns, run), runs))
    return {
        task: result
        for run, run_results in zip(runs, results, strict=True)
        for task, result in zip(run, run_results, strict=True)
    }


def pnl(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """Return ``(pval_forward, pval_backward, stat_forward, stat_backward)`` for the pair ``x, y``.

    ``pval_forward`` is the HSIC independence p-value between ``x`` and the disturbance recovered
    under the ``x -> y`` post-nonlinear model; ``pval_backward`` is the same quantity under the
    ``y -> x`` model. A larger p-value means the assumed cause and the recovered disturbance are
    more independent, so it favors that direction. ``stat_forward`` and ``stat_backward`` are the
    HSIC statistics of the same two tests, each divided by its mean under independence; a smaller
    value means weaker dependence. ``x`` and ``y`` are 1-D arrays of equal length.
    """
    columns = {0: _column(x), 1: _column(y)}
    tests = _ordered_tests(columns, [(0, 1), (1, 0)], backend.worker_count())
    (stat_forward, pval_forward), (stat_backward, pval_backward) = tests[0, 1], tests[1, 0]
    return pval_forward, pval_backward, stat_forward, stat_backward


def pnl_pairs(
    data: np.ndarray,
    pairs: Iterable[tuple[int, int]] | None = None,
    *,
    num_workers: int | None = None,
) -> np.ndarray:
    """Return :func:`pnl`'s four numbers for pairs of ``data``'s columns, one row per pair.

    Row ``k`` is ``(pval_forward, pval_backward, stat_forward, stat_backward)`` for the ``k``-th
    pair ``(i, j)``, where forward means column ``i`` -> column ``j``; it equals
    ``pnl(data[:, i], data[:, j])`` bit for bit. ``pairs`` defaults to every ``i < j`` in
    lexicographic order. ``num_workers`` worker threads share the ordered pairs (``None`` reads
    ``andrey.config`` and ``ANDREY_NUM_WORKERS``, ``-1`` uses every usable core); each holds three
    ``n x n`` float64 matrices.
    """
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"pnl_pairs expects an (n_samples, n_variables) table, got {arr.shape}")
    d = arr.shape[1]
    chosen = [(int(i), int(j)) for i, j in (combinations(range(d), 2) if pairs is None else pairs)]
    for i, j in chosen:
        if not (0 <= i < d and 0 <= j < d):
            raise ValueError(f"pair ({i}, {j}) is out of range for {d} columns")
    if not chosen:
        return np.empty((0, 4))
    columns = {k: _column(arr[:, k]) for k in sorted({k for pair in chosen for k in pair})}
    tasks = [(i, j) for i, j in chosen] + [(j, i) for i, j in chosen]
    tests = _ordered_tests(columns, tasks, backend.worker_count(num_workers))
    return np.array(
        [(tests[i, j][1], tests[j, i][1], tests[i, j][0], tests[j, i][0]) for i, j in chosen],
        dtype=np.float64,
    )
