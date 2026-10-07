"""HSIC independence tests with a gamma-approximation p-value.

Both tests map the two variables through Gaussian kernels, center the kernel matrices, and match
the HSIC statistic to a gamma distribution for a closed-form p-value; a larger p-value means more
independent.

- :func:`hsic_test`, used by ANM, z-scores each variable (``ddof=1``; a constant column becomes
  zero) and sets the bandwidth by the empirical rule, or by the median heuristic on request.
- :func:`hsic_gamma_test`, used by RCD and ParceLiNGAM, sets the bandwidth by the median distance
  over the first rows without z-scoring, corrects the variance for bias, and returns the statistic
  with the p-value.

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
from scipy.special import gammainc
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


def _median_bandwidth(data: np.ndarray) -> float:
    """Return the Gaussian ``width`` (the ``1/sigma**2`` precision) from the median heuristic.

    ``sigma`` is ``sqrt(2)`` times the median of the positive pairwise Euclidean distances; for more
    than 1000 rows a uniform 1000-row subsample is drawn first.
    """
    n = data.shape[0]
    if n > 1000:
        data = data[np.random.permutation(n)[:1000], :]
    dists = squareform(pdist(data, "euclidean"))
    median_dist = np.median(dists[dists > 0])
    sigma = np.sqrt(2.0) * median_dist
    return 1.0 / sigma**2


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


def _resolve_bandwidth(raw: np.ndarray, est_width: str, manual_width: float | None) -> float:
    """Return the Gaussian ``width`` precision for one variable under the chosen estimator."""
    if manual_width is not None:
        return 1.0 / manual_width**2
    if est_width == "empirical":
        return _empirical_bandwidth(raw)
    if est_width == "median":
        return _median_bandwidth(raw)
    raise ValueError(f"unknown est_width {est_width!r}; expected 'empirical' or 'median'")


def _centered_kernel(raw: np.ndarray, est_width: str, manual_width: float | None) -> np.ndarray:
    """Build a centered kernel with raw-data bandwidth and z-scored inputs."""
    width = _resolve_bandwidth(raw, est_width, manual_width)
    return _center_kernel(_gaussian_kernel(_standardize(raw), width))


def measure_hsic_statistic(
    x: np.ndarray,
    y: np.ndarray,
    *,
    est_width: str = "empirical",
    kernel_width_x: float | None = None,
    kernel_width_y: float | None = None,
) -> float:
    """Return the HSIC V-statistic between ``x`` and ``y``.

    The statistic is the Frobenius inner product ``sum(Kxc * Kyc)`` of the two centered Gaussian
    kernel matrices. ``est_width`` selects the bandwidth rule (``'empirical'`` or ``'median'``);
    a per-variable ``kernel_width_*`` overrides it with an explicit ``sigma``.
    """
    kxc = _centered_kernel(_as_columns(x), est_width, kernel_width_x)
    kyc = _centered_kernel(_as_columns(y), est_width, kernel_width_y)
    return float(np.sum(kxc * kyc))


def hsic_test(
    x: np.ndarray,
    y: np.ndarray,
    *,
    est_width: str = "empirical",
    kernel_width_x: float | None = None,
    kernel_width_y: float | None = None,
) -> float:
    """Return the gamma-approximation p-value of the HSIC independence test between ``x`` and ``y``.

    Each variable is z-scored (``ddof=1``), mapped through a Gaussian kernel whose bandwidth follows
    the empirical rule (``est_width='empirical'``, the ANM default) or the median heuristic, and
    centered. The HSIC V-statistic ``T = sum(Kxc * Kyc)`` is compared against a gamma distribution
    matched to its null mean and variance,

        mean = tr(Kxc) tr(Kyc) / n,   var = 2 sum(Kxc^2) sum(Kyc^2) / n^2,
        shape = mean^2 / var,         scale = var / mean,

    and the p-value is the gamma upper tail ``1 - P(shape, T / scale)`` with ``P`` the regularized
    lower incomplete gamma function (equivalently ``scipy.stats.gamma.sf(T, shape, scale=scale)``).
    A larger p-value means more independent. Accepts 1-D vectors or ``(n, d)`` matrices.
    """
    n = _as_columns(x).shape[0]
    kxc = _centered_kernel(_as_columns(x), est_width, kernel_width_x)
    kyc = _centered_kernel(_as_columns(y), est_width, kernel_width_y)

    statistic = np.sum(kxc * kyc)
    mean_appr = np.trace(kxc) * np.trace(kyc) / n
    var_appr = 2 * np.sum(kxc**2) * np.sum(kyc**2) / n / n
    shape = mean_appr**2 / var_appr
    scale = var_appr / mean_appr
    return float(1.0 - gammainc(shape, statistic / scale))


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
    against many others (RCD, ParceLiNGAM) computes this once per column and reuses it. Bit-for-bit
    identical to the grams :func:`hsic_gamma_test` builds inline.
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


def hsic_gamma_test(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return ``(statistic, p_value)`` of the LiNGAM-family HSIC gamma-approximation test.

    The bandwidth is the median-distance heuristic over the first rows; the biased HSIC V-statistic
    ``T = sum(Kxc.T * Kyc) / n`` is compared against a gamma law matched to the finite-sample null
    mean and bias-corrected variance, and the p-value is the gamma upper tail ``1 - P(shape,
    T / scale)``. A larger p-value means more independent. Accepts 1-D vectors or ``(n, 1)`` inputs.
    Callers scoring one variable against many should cache :func:`gamma_gram` and combine
    with :func:`hsic_gamma_from_grams` instead.
    """
    xc = _as_columns(x)
    return hsic_gamma_from_grams(gamma_gram(xc), gamma_gram(_as_columns(y)), xc.shape[0])
