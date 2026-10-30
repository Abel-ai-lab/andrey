"""HSIC independence tests with a gamma-approximation p-value.

Both tests map the two variables through Gaussian kernels, center the kernel matrices, and match
the HSIC statistic to a gamma distribution for a closed-form p-value; a larger p-value means more
independent.

- :func:`hsic_test` z-scores each variable (``ddof=1``; a constant column becomes zero) and sets
  the bandwidth by the empirical rule. :func:`hsic_statistic_test`, used by ANM, runs the same test
  and returns the statistic, divided by its mean under independence, with the p-value. PNL computes
  the same numbers from kernels it shares across pairs.
- :func:`hsic_gamma_from_grams`, used by RCD and ParceLiNGAM with the grams of :func:`gamma_gram`,
  sets the bandwidth by the median distance over the first rows without z-scoring, corrects the
  variance for bias, and returns the statistic with the p-value.
- :func:`hsic_moments_from_grams`, used by RESIT with the grams of :func:`median_gram`, takes each
  side's kernel on its joint squared distances with the median over every pair of rows as the
  bandwidth, and matches the gamma law to the null mean and variance of Gretton et al. (2007).

References
----------
Gretton et al. (2007). A Kernel Statistical Test of Independence. NIPS 20.
https://papers.nips.cc/paper/2007/hash/d5cfead94f5350c12c322b5b664544c1-Abstract.html

Zhang, Peters, Janzing and Scholkopf (2011). Kernel-based Conditional Independence Test
and Application in Causal Discovery. https://webdav.tuebingen.mpg.de/causality/UAI11_KCItest.pdf
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.special import gammainc, gammaincc
from scipy.stats import zscore


def _as_columns(data: np.ndarray) -> np.ndarray:
    """Return ``data`` as a float64 ``(n, d)`` matrix, promoting a 1-D vector to a single column."""
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _standardize(data: np.ndarray) -> np.ndarray:
    """Z-score each column with ``ddof=1`` and map the NaNs from constant columns to zero."""
    standardized = zscore(data, ddof=1, axis=0)
    standardized[np.isnan(standardized)] = 0.0
    return standardized


def _empirical_bandwidth(data: np.ndarray) -> float:
    """Return the Gaussian ``width`` (the ``1/sigma**2`` precision) from the empirical HSIC rule.

    The bandwidth depends only on the sample size and dimensionality: ``sigma`` steps down from 0.8
    to 0.5 to 0.3 as ``n`` crosses 200 and 1200, and the resulting precision is scaled by the number
    of columns.
    """
    n = data.shape[0]
    if n < 200:
        sigma = 0.8
    elif n < 1200:
        sigma = 0.5
    else:
        sigma = 0.3
    precision = 1.0 / sigma**2
    return precision * data.shape[1]


def _gaussian_kernel(data: np.ndarray, width: float) -> np.ndarray:
    """Return the Gaussian (RBF) kernel matrix ``exp(-0.5 * ||x_i - x_j||**2 * width)``."""
    sq_dists = squareform(pdist(data, "sqeuclidean"))
    return np.exp(-0.5 * sq_dists * width)


def _center_kernel(matrix: np.ndarray) -> np.ndarray:
    """Return the doubly-centered kernel matrix ``H K H`` with ``H = I - 1/n``.

    Expanded from the matrix product to the equivalent row/column-sum form (kernel matrices are
    symmetric).
    """
    n = matrix.shape[0]
    col_sums = matrix.sum(axis=0)
    total = col_sums.sum()
    return matrix - (col_sums[None, :] + col_sums[:, None]) / n + (total / n**2)


# Kernel elements computed per block (rows times columns), so a block's elementwise steps run in
# cache.
_BLOCK_ELEMENTS = 32768


def _block_rows(n: int) -> int:
    """Rows of the block :func:`_kernel_into` builds at a time for ``n`` samples."""
    return max(1, min(n, _BLOCK_ELEMENTS // max(n, 1)))


def _kernel_into(z: np.ndarray, width: float, out: np.ndarray, block: np.ndarray) -> np.ndarray:
    """Write the centered Gaussian kernel of the standardized 1-D ``z`` into ``out`` and return it.

    Every element takes ``exp(-0.5 * d**2 * width)``, then ``K - (col_sum_j + col_sum_i) / n +
    total / n**2``, as :func:`_gaussian_kernel` and :func:`_center_kernel` do, but in row blocks of
    ``block``'s size written into ``out``, so no ``n x n`` temporary is allocated.
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


def _gamma_test(
    statistic: float, trace_x: float, sumsq_x: float, trace_y: float, sumsq_y: float, n: int
) -> tuple[float, float]:
    """Return ``(statistic / mean, p_value)`` of the HSIC test from its two kernels' sums.

    ``trace_*`` and ``sumsq_*`` are each centered kernel's trace and sum of squared entries; the
    gamma null's mean and variance are those of :func:`hsic_statistic_test`.
    """
    mean_appr = trace_x * trace_y / n
    var_appr = 2 * sumsq_x * sumsq_y / n / n
    shape = mean_appr**2 / var_appr
    scale = var_appr / mean_appr
    return float(statistic / mean_appr), float(1.0 - gammainc(shape, statistic / scale))


def _centered_kernel(raw: np.ndarray) -> np.ndarray:
    """Build a centered kernel with raw-data bandwidth and z-scored inputs."""
    width = _empirical_bandwidth(raw)
    z = _standardize(raw)
    if z.shape[1] > 1:
        return _center_kernel(_gaussian_kernel(z, width))
    n = z.shape[0]
    return _kernel_into(z[:, 0], width, np.empty((n, n)), np.empty((_block_rows(n), n)))


def hsic_test(x: np.ndarray, y: np.ndarray) -> float:
    """Return the gamma-approximation p-value of the HSIC independence test between ``x`` and ``y``.

    The p-value of :func:`hsic_statistic_test`, which describes the test.
    """
    return hsic_statistic_test(x, y)[1]


def hsic_statistic_test(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return ``(T / mean, p_value)`` of the HSIC independence test between ``x`` and ``y``.

    Each variable is z-scored (``ddof=1``), mapped through a Gaussian kernel whose bandwidth follows
    the empirical rule, and centered. The HSIC V-statistic ``T = sum(Kxc * Kyc)`` is compared
    against a gamma distribution matched to its null mean and variance,

        mean = tr(Kxc) tr(Kyc) / n,   var = 2 sum(Kxc^2) sum(Kyc^2) / n^2,
        shape = mean^2 / var,         scale = var / mean,

    and the p-value is the gamma upper tail ``1 - P(shape, T / scale)`` with ``P`` the regularized
    lower incomplete gamma function (equivalently ``scipy.stats.gamma.sf(T, shape, scale=scale)``).
    A larger p-value means more independent. ``T / mean`` is the statistic relative to its mean
    under independence, the gamma null's ``shape * scale``: near 1 for independent variables and
    larger the more dependent they are. It still orders two tests whose p-values are both zero.
    Accepts 1-D vectors or ``(n, d)`` matrices.
    """
    n = _as_columns(x).shape[0]
    kxc = _centered_kernel(_as_columns(x))
    kyc = _centered_kernel(_as_columns(y))

    statistic = np.sum(kxc * kyc)
    return _gamma_test(statistic, np.trace(kxc), np.sum(kxc**2), np.trace(kyc), np.sum(kyc**2), n)


# ---- LiNGAM-family HSIC (median-distance bandwidth, biased V-statistic) --------------------------


_TRIU_K1_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _triu_k1(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Strict upper-triangle index pair for an ``n x n`` matrix, memoized by ``n``.

    The bandwidth heuristic runs on repeated same-size blocks, so the deterministic index arrays are
    built once per ``n`` and reused.
    """
    idx = _TRIU_K1_CACHE.get(n)
    if idx is None:
        idx = np.triu_indices(n, k=1)
        _TRIU_K1_CACHE[n] = idx
    return idx


def _median_width_gamma(column: np.ndarray) -> float:
    """Median-distance Gaussian bandwidth ``sigma`` from up to the first 100 rows of an ``(n, 1)``.

    ``sigma = sqrt(0.5 * median(positive squared pairwise distances))``; the 100-row cap keeps the
    median heuristic cheap.
    """
    data = column[:100] if column.shape[0] > 100 else column
    n = data.shape[0]
    sq_norms = np.sum(data * data, axis=1)
    sq_dists = sq_norms[None, :] + sq_norms[:, None] - 2.0 * (data @ data.T)
    upper = sq_dists[_triu_k1(n)]
    return float(np.sqrt(0.5 * np.median(upper[upper > 0])))


def _gram_gamma(column: np.ndarray, width: float) -> tuple[np.ndarray, np.ndarray]:
    """Raw and doubly-centered RBF gram matrices of an ``(n, 1)`` column at bandwidth ``sigma``.

    The centered matrix is ``H K H`` with ``H = I - 1/n`` in row/column-sum form (the biased
    estimate).
    """
    n = column.shape[0]
    sq_norms = np.sum(column * column, axis=1)
    sq_dists = sq_norms[None, :] + sq_norms[:, None] - 2.0 * (column @ column.T)
    np.negative(sq_dists, out=sq_dists)  # scale the exponent in place: -sq_dists / 2 / width**2
    sq_dists /= 2.0
    sq_dists /= width**2
    raw = np.exp(sq_dists)
    col_sums = raw.sum(axis=0)
    row_sums = raw.sum(axis=1)
    centered = raw - (col_sums[None, :] + row_sums[:, None]) / n + (row_sums.sum() / n**2)
    return raw, centered


def gamma_gram(column: np.ndarray) -> tuple[np.ndarray, float]:
    """Return one variable's ``(centered_gram, off_diagonal_kernel_mean)`` for the gamma HSIC.

    These are the only per-variable quantities the test needs, so a caller that scores one column
    against many others (RCD, ParceLiNGAM) computes this once per column and reuses it.
    """
    col = _as_columns(column)
    n = col.shape[0]
    raw, centered = _gram_gamma(col, _median_width_gamma(col))
    raw[np.diag_indices(n)] = 0.0
    mu = 1.0 / n / (n - 1) * raw.sum()
    return centered, mu


def hsic_gamma_from_grams(
    gram_x: tuple[np.ndarray, float], gram_y: tuple[np.ndarray, float], n: int
) -> tuple[float, float]:
    """Return ``(statistic, p_value)`` of the gamma HSIC from two cached :func:`gamma_gram` outputs.

    ``n`` is the sample size. The biased HSIC V-statistic ``T = sum(Kxc.T * Kyc) / n`` is matched to
    a gamma law through the finite-sample null mean and bias-corrected variance; the p-value is the
    gamma upper tail. A larger p-value means more independent.
    """
    cen_x, mu_x = gram_x
    cen_y, mu_y = gram_y

    statistic = 1.0 / n * np.sum(cen_x.T * cen_y)

    prod = cen_x * (1.0 / 6.0)  # 1/6 * cen_x * cen_y then squared, fused into one reused buffer
    prod *= cen_y
    prod **= 2
    var = 1.0 / n / (n - 1) * (np.sum(prod) - np.trace(prod))
    var = 72.0 * (n - 4) * (n - 5) / n / (n - 1) / (n - 2) / (n - 3) * var

    mean = 1.0 / n * (1 + mu_x * mu_y - mu_x - mu_y)

    shape = mean**2 / var
    scale = var * n / mean
    return float(statistic), float(1.0 - gammainc(shape, statistic / scale))


# ---- RESIT HSIC (joint median bandwidth, the null moments of Gretton et al. 2007) ---------------


def median_gram(sq_dists: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return one side's ``(centered_gram, off_diagonal_mean, sum_of_squares)`` for the moment HSIC.

    ``sq_dists`` holds the squared Euclidean distances between the rows of the side, one variable or
    several jointly. The Gaussian kernel is ``exp(-D / m)``, ``m`` the median of the positive
    squared distances over every pair of rows: the median heuristic ``sigma = sqrt(m / 2)``, under
    which a common rescaling of the side's coordinates leaves the kernel unchanged. A side whose
    rows are all equal has the all-ones kernel, whose centered gram is zero. The off-diagonal mean
    is of the raw kernel; the sum of squares is of the centered gram.
    """
    n = sq_dists.shape[0]
    upper = sq_dists[_triu_k1(n)]
    positive = upper[upper > 0]
    kernel = np.exp(-sq_dists / np.median(positive)) if positive.size else np.ones((n, n))
    mean = float((kernel.sum() - np.trace(kernel)) / (n * (n - 1)))
    centered = _center_kernel(kernel)
    return centered, mean, float(np.sum(centered * centered))


def hsic_moments_from_grams(
    gram_x: tuple[np.ndarray, float, float], gram_y: tuple[np.ndarray, float, float]
) -> tuple[float, float]:
    """Return ``(statistic, p_value)`` of the HSIC test from two :func:`median_gram` outputs.

    The statistic is ``n HSIC_b = sum(Kc * Lc) / n`` for the centered grams ``Kc`` and ``Lc`` of
    ``n`` rows. Under independence it is matched to a gamma law with the mean and variance of
    ``HSIC_b`` from Gretton et al. (2007), each kernel's squared Hilbert-Schmidt norm estimated by
    ``sum(Kc^2) / (n - 1)^2``:

        mean = (1 + mu_x mu_y - mu_x - mu_y) / n,
        var = 2 (n - 4)(n - 5) / (n (n - 1)(n - 2)(n - 3)) sum(Kc^2) sum(Lc^2) / (n - 1)^4,
        shape = mean^2 / var,   scale = n var / mean,

    with ``mu`` each raw kernel's off-diagonal mean. The p-value is the gamma upper tail, the
    regularized upper incomplete gamma function, which keeps its relative precision far below
    ``1e-16``. A larger p-value means more independent. A side with a zero centered gram, a constant
    variable, is independent of any other: statistic ``0`` and p-value ``1``. Needs ``n >= 6``.
    """
    cen_x, mu_x, sq_x = gram_x
    cen_y, mu_y, sq_y = gram_y
    n = cen_x.shape[0]
    if sq_x == 0 or sq_y == 0:
        return 0.0, 1.0
    statistic = float(np.sum(cen_x * cen_y)) / n
    mean = (1.0 + mu_x * mu_y - mu_x - mu_y) / n
    var = 2.0 * (n - 4) * (n - 5) / (n * (n - 1) * (n - 2) * (n - 3))
    var *= sq_x / (n - 1) ** 2 * sq_y / (n - 1) ** 2
    shape = mean**2 / var
    scale = n * var / mean
    return statistic, float(gammaincc(shape, statistic / scale))
