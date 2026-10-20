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

On the ``cuda`` backend, or under ``auto`` from ``ANDREY_PNL_GPU_THRESHOLD`` rows squared, the
kernels and their sums run in float64 torch on the device, a batch of disturbances at a time; the
fits, the normal scores, and the gamma p-values stay on the CPU, so the results match the CPU's to
rounding.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import combinations, groupby

import numpy as np
from scipy.special import gammainc
from scipy.stats import norm, rankdata

from andrey.core import backend, env
from andrey.core.independence import _empirical_bandwidth, _standardize

_N_FEATURES = 10
_RIDGE = 1e-8
_BARRIER_FLOOR = 1e-6
_MAX_NEWTON = 60
_TOL = 1e-8

# Kernel elements computed per block (rows times columns), so a block's elementwise steps run in
# cache.
_BLOCK_ELEMENTS = 32768

# Smallest kernel matrix (rows squared) at which `auto` runs a table's PNL tests on a CUDA GPU.
# `qa/gpu_calibration_validate.py` on an A100-SXM4-80GB found 26,278 (163 rows) on each of five
# runs, against the CPU path at its default of one worker; on its table of four columns, the GPU
# ran at 1.5x the CPU's speed at 200 rows, 4.2x at 500, and 10x at 1,000 (against four workers:
# 1.1x, 1.8x, 3.6x). PNL computes in float64, so `mps` (float32) is never used.
_PNL_GPU_THRESHOLDS: dict[str, int | None] = {"cuda": 26_278, "mps": None}

# Device memory one batch of disturbance kernels uses.
_DEVICE_BATCH_BYTES = 2**30


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


def _kernel_into(z: np.ndarray, width: float, out: np.ndarray, block: np.ndarray) -> np.ndarray:
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
    raw = disturbance.reshape(-1, 1)
    kyc = _kernel_into(
        _standardize(raw)[:, 0], _empirical_bandwidth(raw), buffers.disturbance, buffers.block
    )
    statistic = np.sum(np.multiply(kxc, kyc, out=buffers.product))
    sumsq_y = np.sum(np.square(kyc, out=buffers.product))
    return _gamma_test(statistic, trace_x, sumsq_x, np.trace(kyc), sumsq_y, kyc.shape[0])


def _gamma_test(
    statistic: float, trace_x: float, sumsq_x: float, trace_y: float, sumsq_y: float, n: int
) -> tuple[float, float]:
    """Return ``(statistic / mean, p_value)`` of the HSIC test from its two kernels' sums.

    ``trace_*`` and ``sumsq_*`` are each centered kernel's trace and sum of squared entries; the
    gamma null's moments take the same operations, in the same order, as in
    :func:`~andrey.core.independence.hsic_statistic_test`.
    """
    mean_appr = trace_x * trace_y / n
    var_appr = 2 * sumsq_x * sumsq_y / n / n
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
            kxc = _kernel_into(source.standardized, source.width, buffers.cause, buffers.block)
            trace_x = np.trace(kxc)
            sumsq_x = np.sum(np.square(kxc, out=buffers.product))
        disturbance = _fit_disturbance(source.phi, target.outer, target.outer_deriv)
        results.append(_hsic(kxc, trace_x, sumsq_x, disturbance, buffers))
    return results


def _device_kernels(z, width: float):
    """Return the centered Gaussian kernels of the rows of ``z`` as an ``(m, n, n)`` tensor.

    ``z`` is an ``(m, n)`` float64 tensor of standardized values. Each kernel takes the steps of
    :func:`_kernel_into`, on ``z``'s device; the sums run in another order, so the entries match
    it to rounding.
    """
    n = z.shape[1]
    k = z[:, :, None] - z[:, None, :]
    k.square_().mul_(-0.5 * width).exp_()
    col_sums = k.sum(dim=1)
    shift = col_sums.sum(dim=1) / n**2
    k.sub_(col_sums[:, None, :] / n).sub_(col_sums[:, :, None] / n).add_(shift[:, None, None])
    return k


def _device_tests(
    columns: dict[int, _Column], ordered: list[tuple[int, int]], workers: int, dev: str
) -> list[tuple[float, float]]:
    """Return ``(statistic, p_value)`` for each ``(cause, effect)`` task in ``ordered``, sorted by
    cause, with the kernels on the torch device ``dev``.

    ``workers`` threads fit the disturbances in order while this thread runs each cause's kernel
    and its disturbances' kernels, a batch of ``_DEVICE_BATCH_BYTES`` at a time. Every column and
    disturbance has ``n`` rows, so they share one kernel width.
    """
    torch = backend.torch()
    n = next(iter(columns.values())).standardized.shape[0]
    batch = max(1, _DEVICE_BATCH_BYTES // (8 * n * n))

    def standardized_disturbance(task: tuple[int, int]) -> np.ndarray:
        source, target = columns[task[0]], columns[task[1]]
        disturbance = _fit_disturbance(source.phi, target.outer, target.outer_deriv)
        return _standardize(disturbance.reshape(-1, 1))[:, 0]

    sums = []  # per task: statistic, the cause's trace and sum of squares, the disturbance's
    with contextlib.ExitStack() as stack:
        if workers > 1:
            stack.enter_context(backend.one_blas_thread())
            pool = stack.enter_context(ThreadPoolExecutor(max_workers=workers))
            fitted = pool.map(standardized_disturbance, ordered)
        else:
            fitted = map(standardized_disturbance, ordered)
        for cause, group in groupby(zip(ordered, fitted, strict=True), key=lambda t: t[0][0]):
            source = columns[cause]
            kxc = _device_kernels(backend.as_tensor(source.standardized[None], dev), source.width)
            kxc = kxc.reshape(-1)
            cause_sums = torch.stack([kxc[:: n + 1].sum(), kxc @ kxc])
            disturbances = [z for _, z in group]
            for start in range(0, len(disturbances), batch):
                z = backend.as_tensor(np.stack(disturbances[start : start + batch]), dev)
                kyc = _device_kernels(z, source.width).reshape(z.shape[0], -1)
                sums.append(
                    torch.column_stack(
                        [
                            kyc @ kxc,
                            cause_sums.expand(z.shape[0], 2),
                            kyc[:, :: n + 1].sum(dim=1),
                            torch.einsum("mk,mk->m", kyc, kyc),
                        ]
                    )
                )
    values = backend.to_numpy(torch.cat(sums))
    return [_gamma_test(st, tx, sx, ty, sy, n) for st, tx, sx, ty, sy in values]


def _ordered_tests(
    columns: dict[int, _Column], tasks: list[tuple[int, int]], workers: int, dev: str = "numpy"
) -> dict[tuple[int, int], tuple[float, float]]:
    """Return ``{(cause, effect): (statistic, p_value)}`` for the distinct ``tasks``.

    On ``numpy`` the tasks, sorted by cause, are cut into ``workers`` contiguous runs, one per
    worker thread, each on one BLAS thread. A task's result does not depend on the number of
    workers, except that the fits' last digits can: a lone worker's fits run on every BLAS thread.
    On a torch device, :func:`_device_tests` runs them.
    """
    ordered = sorted(dict.fromkeys(tasks))
    if dev != "numpy":
        return dict(zip(ordered, _device_tests(columns, ordered, workers, dev), strict=True))
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


def _pnl_backend(n: int, *, amortized: bool) -> str:
    """Where PNL's tests on ``n`` rows run: ``numpy``, or torch on ``cpu`` or ``cuda``."""
    manual = env.PNL_GPU_THRESHOLD.read()
    return backend.select(
        ("numpy", "cpu", "cuda"),
        ("cuda", "numpy"),
        metric=n * n,
        thresholds=_PNL_GPU_THRESHOLDS if manual is None else {"cuda": manual},
        amortized=amortized,
        op="pnl",
    )


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
    dev = _pnl_backend(columns[0].standardized.shape[0], amortized=False)
    tests = _ordered_tests(columns, [(0, 1), (1, 0)], backend.worker_count(), dev)
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
    dev = _pnl_backend(arr.shape[0], amortized=True)
    tests = _ordered_tests(columns, tasks, backend.worker_count(num_workers), dev)
    return np.array(
        [(tests[i, j][1], tests[j, i][1], tests[i, j][0], tests[j, i][0]) for i, j in chosen],
        dtype=np.float64,
    )
