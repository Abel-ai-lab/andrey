"""HSIC independence tests with a gamma-approximation p-value.

Both tests map the two variables through Gaussian kernels, center the kernel matrices, and match
the HSIC statistic to a gamma distribution for a closed-form p-value; a larger p-value means more
independent. The p-value is the gamma law's upper tail, computed through its logarithm
(:func:`_log_upper_gamma`): the logarithm keeps the tail's relative precision far below ``1e-16``,
and stays finite and keeps the order of two tails below the ``1e-308`` where the p-value itself
underflows to ``0``.

- :func:`hsic_test` z-scores each variable (``ddof=1``; a constant column becomes zero) and sets
  the bandwidth by the empirical rule. :func:`hsic_statistic_log_p`, used by ANM and by KCI without
  conditions, runs the same test and returns the statistic, divided by its mean under
  independence, with the p-value's logarithm. PNL computes the same numbers from kernels it shares
  across pairs.
- :func:`hsic_gamma_from_grams` and :func:`hsic_gamma_log_p`, used by RCD, ParceLiNGAM, and GIN
  with the grams of :func:`gamma_gram`, set the bandwidth by the median distance over the first
  rows without z-scoring and correct the variance for bias; the first returns the statistic with
  the p-value, the second the p-value's logarithm. RESIT and CAM-UV run the same test through
  :func:`hsic_log_p` on each side's joint distances with two settings of their own: the
  bandwidth's median over every row (:func:`hsic_gram`) and the variance's product form.
  :func:`hsic_median_statistic_log_p`, used by LOCI, reads the median over every row with the
  bias-corrected variance, and returns the statistic, divided by its null mean, with the p-value's
  logarithm.

References
----------
Gretton et al. (2007). A Kernel Statistical Test of Independence. NIPS 20.
https://papers.nips.cc/paper/2007/hash/d5cfead94f5350c12c322b5b664544c1-Abstract.html

Zhang, Peters, Janzing and Scholkopf (2011). Kernel-based Conditional Independence Test
and Application in Causal Discovery. https://webdav.tuebingen.mpg.de/causality/UAI11_KCItest.pdf
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.special import gammaincc, gammaln
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
    """Return ``(statistic / mean, log_p)`` of the HSIC test from its two kernels' sums.

    ``trace_*`` and ``sumsq_*`` are each centered kernel's trace and sum of squared entries; the
    gamma null's mean and variance, and the p-value's logarithm ``log_p``, are those of
    :func:`hsic_statistic_log_p`.
    """
    mean_appr = trace_x * trace_y / n
    var_appr = 2 * sumsq_x * sumsq_y / n / n
    shape = mean_appr**2 / var_appr
    scale = var_appr / mean_appr
    return float(statistic / mean_appr), _log_upper_gamma(float(shape), float(statistic / scale))


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

    The exponential of the logarithm :func:`hsic_statistic_log_p` returns, which describes the
    test; it underflows to ``0`` below about ``1e-308``.
    """
    return math.exp(hsic_statistic_log_p(x, y)[1])


def hsic_statistic_log_p(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return ``(T / mean, log_p)`` of the HSIC independence test between ``x`` and ``y``.

    Each variable is z-scored (``ddof=1``), mapped through a Gaussian kernel whose bandwidth follows
    the empirical rule, and centered. The HSIC V-statistic ``T = sum(Kxc * Kyc)`` is compared
    against a gamma distribution matched to its null mean and variance,

        mean = tr(Kxc) tr(Kyc) / n,   var = 2 sum(Kxc^2) sum(Kyc^2) / n^2,
        shape = mean^2 / var,         scale = var / mean,

    and ``log_p`` is the natural logarithm of the p-value, the gamma upper tail
    ``Q(shape, T / scale)`` with ``Q`` the regularized upper incomplete gamma function
    (:func:`_log_upper_gamma`). It keeps its precision far below ``1e-16``, and stays finite and
    keeps the order of two tests below the ``1e-308`` where the p-value underflows to ``0``. A
    larger p-value means more independent. ``T / mean`` is the statistic relative to its mean under
    independence, the gamma null's ``shape * scale``: near 1 for independent variables and larger
    the more dependent they are. Accepts 1-D vectors or ``(n, d)`` matrices.
    """
    n = _as_columns(x).shape[0]
    kxc = _centered_kernel(_as_columns(x))
    kyc = _centered_kernel(_as_columns(y))

    statistic = np.sum(kxc * kyc)
    return _gamma_test(statistic, np.trace(kxc), np.sum(kxc**2), np.trace(kyc), np.sum(kyc**2), n)


# ---- LiNGAM-family HSIC (median-distance bandwidth, biased V-statistic) --------------------------


_TRIU_K1_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}
#: Most rows whose upper-triangle indices :func:`_triu_k1` keeps: the default median's block.
_TRIU_CACHED_ROWS = 100


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


def _upper(sq_dists: np.ndarray) -> np.ndarray:
    """The strict upper triangle of a square matrix, row by row.

    Up to :data:`_TRIU_CACHED_ROWS` rows through the memoized indices of :func:`_triu_k1`; a larger
    matrix, which only a median over every row meets, through a boolean mask of one byte per entry
    that is not kept, in the same order.
    """
    n = sq_dists.shape[0]
    if n <= _TRIU_CACHED_ROWS:
        return sq_dists[_triu_k1(n)]
    return sq_dists[np.triu(np.ones((n, n), dtype=bool), k=1)]


def _sq_dists(data: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances between the rows of ``data``, ``x^2 + y^2 - 2xy``."""
    sq_norms = np.sum(data * data, axis=1)
    return sq_norms[None, :] + sq_norms[:, None] - 2.0 * (data @ data.T)


def _median_width(sq_dists: np.ndarray) -> float:
    """``sqrt(0.5 * median(positive squared pairwise distances))`` from squared distances."""
    upper = _upper(sq_dists)
    return float(np.sqrt(0.5 * np.median(upper[upper > 0])))


def _median_width_gamma(column: np.ndarray, rows: int | None = 100) -> float:
    """Median-distance Gaussian bandwidth ``sigma`` from up to the first ``rows`` rows of a matrix.

    ``sigma = sqrt(0.5 * median(positive squared pairwise distances))`` over the rows of an
    ``(n, q)`` matrix, its ``q`` columns read jointly; the 100-row cap keeps the median heuristic
    cheap, and ``rows=None`` reads every row.
    """
    data = column[:rows] if rows is not None and column.shape[0] > rows else column
    return _median_width(_sq_dists(data))


def _gram_from_sq_dists(sq_dists: np.ndarray, width: float) -> tuple[np.ndarray, np.ndarray]:
    """Raw and doubly-centered RBF gram matrices from squared distances, which are overwritten.

    The centered matrix is ``H K H`` with ``H = I - 1/n`` in row/column-sum form (the biased
    estimate).
    """
    n = sq_dists.shape[0]
    np.negative(sq_dists, out=sq_dists)  # scale the exponent in place: -sq_dists / 2 / width**2
    sq_dists /= 2.0
    sq_dists /= width**2
    raw = np.exp(sq_dists)
    col_sums = raw.sum(axis=0)
    row_sums = raw.sum(axis=1)
    centered = raw - (col_sums[None, :] + row_sums[:, None]) / n + (row_sums.sum() / n**2)
    return raw, centered


def _gram_gamma(column: np.ndarray, width: float) -> tuple[np.ndarray, np.ndarray]:
    """Raw and doubly-centered RBF gram matrices of an ``(n, q)`` matrix at bandwidth ``sigma``."""
    return _gram_from_sq_dists(_sq_dists(column), width)


#: One side of the gamma HSIC: its centered gram and its raw kernel's off-diagonal mean, and
#: optionally the centered gram's sum of squares, which the product-form variance then reuses.
Gram = tuple[np.ndarray, float] | tuple[np.ndarray, float, float]


def gamma_gram(column: np.ndarray, *, median_rows: int | None = 100) -> tuple[np.ndarray, float]:
    """Return one side's ``(centered_gram, off_diagonal_kernel_mean)`` for the gamma HSIC.

    These are the only per-variable quantities the test needs, so a caller that scores one column
    against many others (RCD, ParceLiNGAM) computes this once per column and reuses it. ``column``
    is one variable or an ``(n, q)`` matrix of several, read jointly through their Euclidean
    distances. The bandwidth's median reads the first ``median_rows`` rows, or every row with
    ``None`` (RESIT); when it reads every row, the squared distances are computed once for both
    the bandwidth and the kernel.

    The distances are those of the mean-centered column, which are the column's own; computed from
    the raw values as ``x^2 + y^2 - 2xy`` they lose their digits when the column lies far from 0.
    With ``median_rows=None`` the columns are read as given: RESIT passes columns it has centered
    before scaling them and residuals of fits with an intercept, whose means are 0 up to rounding.
    """
    col = _as_columns(column)
    if median_rows is not None:
        col = col - col.mean(axis=0)
    n = col.shape[0]
    if median_rows is None or n <= median_rows:
        sq_dists = _sq_dists(col)
        raw, centered = _gram_from_sq_dists(sq_dists, _median_width(sq_dists))
    else:
        raw, centered = _gram_gamma(col, _median_width_gamma(col, median_rows))
    raw[np.diag_indices(n)] = 0.0
    mu = 1.0 / n / (n - 1) * raw.sum()
    return centered, mu


def hsic_gamma_from_grams(
    gram_x: Gram,
    gram_y: Gram,
    n: int,
    *,
    variance: str = "bias_corrected",
) -> tuple[float, float]:
    """Return ``(statistic, p_value)`` of the gamma HSIC from two cached :func:`gamma_gram` outputs.

    ``n`` is the sample size. The biased HSIC V-statistic ``T = sum(Kxc.T * Kyc) / n`` is matched to
    a gamma law through the finite-sample null mean ``(1 + mu_x mu_y - mu_x - mu_y) / n`` and a null
    variance; the p-value is the exponential of :func:`hsic_gamma_log_p`'s logarithm of the gamma
    upper tail, precise far below ``1e-16`` and ``0`` below about ``1e-308``. A larger p-value
    means more independent.

    ``variance="bias_corrected"`` estimates the variance from the joint products
    ``(Kxc * Kyc / 6)^2``, less their diagonal; ``"product"`` uses the variance's product form in
    Gretton et al. (2007), ``2 (n - 4)(n - 5) / (n (n - 1)(n - 2)(n - 3))`` times each gram's
    ``sum(Kc^2) / (n - 1)^2``, the RESIT paper's; a gram that carries its sum of squares as a third
    element has it read rather than summed again.
    """
    statistic, log_p = _statistic_log_p(gram_x, gram_y, n, variance)
    return statistic, math.exp(log_p)


def hsic_gamma_log_p(
    gram_x: Gram,
    gram_y: Gram,
    n: int,
    *,
    variance: str = "bias_corrected",
) -> float:
    """Return the natural logarithm of the p-value of :func:`hsic_gamma_from_grams`'s test.

    The tail is the regularized upper incomplete gamma function, which keeps its relative precision
    far below ``1e-16``. Below about ``1e-308`` the tail underflows to ``0``, as a strongly
    dependent pair's does once there are a few thousand rows; its logarithm stays finite there and
    keeps the order of two such tails (:func:`_log_upper_gamma`).

    The statistic, ``tr(Kxc Kyc) / n`` of two positive semidefinite centered grams, is never
    negative in exact arithmetic. A negative value within :func:`_rounding_bound` of ``0`` is
    rounding of an exact ``0``, as on a balanced grid of independent values, and gives ``0``, the
    logarithm of ``p = 1``; one further below ``0``, or a non-finite argument, gives ``nan``.
    """
    return _statistic_log_p(gram_x, gram_y, n, variance)[1]


def _statistic_log_p(gram_x: Gram, gram_y: Gram, n: int, variance: str) -> tuple[float, float]:
    """The gamma HSIC's statistic and its p-value's logarithm (:func:`hsic_gamma_log_p`)."""
    statistic, shape, x = _gamma_arguments(gram_x, gram_y, n, variance)
    if 0.0 > statistic >= -_rounding_bound(gram_x, gram_y, n):
        x = 0.0
    return float(statistic), _log_upper_gamma(float(shape), float(x))


def hsic_gram(side: np.ndarray) -> tuple[np.ndarray, float, float]:
    """One side's gram for :func:`hsic_log_p`: :func:`gamma_gram` with the bandwidth's median over
    every row, and the centered gram's sum of squares, which the product-form variance reads, so a
    side tested against many others sums it once.

    ``side`` is one column or several read jointly, as given: the caller centers it when its mean
    is far from 0.
    """
    centered, mu = gamma_gram(side, median_rows=None)
    return centered, mu, float(np.sum(centered * centered))


def hsic_log_p(gram_x: Gram, gram_y: Gram, method: str) -> float:
    """The log p-value of the gamma HSIC of two :func:`hsic_gram` sides, the null variance in its
    product form (:func:`hsic_gamma_log_p`); ``method`` names the caller in the error.

    Raises
    ------
    FloatingPointError
        If the test's value is not finite, so that no choice rests on it.
    """
    log_p = hsic_gamma_log_p(gram_x, gram_y, gram_x[0].shape[0], variance="product")
    if not math.isfinite(log_p):
        raise FloatingPointError(
            f"{method}: an HSIC test gave a non-finite p-value; the kernels could not be computed "
            "from these data"
        )
    return log_p


def hsic_median_statistic_log_p(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return ``(T / mean, log_p)`` of the gamma HSIC between ``x`` and ``y``, its bandwidths the
    median distance over every row.

    Each variable is centered and passed to :func:`gamma_gram` with ``median_rows=None``; the test
    is :func:`hsic_gamma_log_p`'s with the bias-corrected variance. ``T / mean`` is the statistic
    ``T = sum(Kxc * Kyc) / n`` divided by its mean under independence, the gamma law's
    ``shape * scale = 1 + mu_x mu_y - mu_x - mu_y``: near 1 for independent variables and larger
    the more dependent they are.

    Raises
    ------
    FloatingPointError
        If the test's value is not finite, so that no choice rests on it.
    """
    sides = (_as_columns(x), _as_columns(y))
    grams = [gamma_gram(side - side.mean(axis=0), median_rows=None) for side in sides]
    n = grams[0][0].shape[0]
    statistic, log_p = _statistic_log_p(grams[0], grams[1], n, "bias_corrected")
    if not math.isfinite(log_p):
        raise FloatingPointError(
            "an HSIC test gave a non-finite p-value; the kernels could not be computed from these "
            "data"
        )
    (_, mu_x), (_, mu_y) = grams
    return statistic / (1 + mu_x * mu_y - mu_x - mu_y), log_p


def _sum_of_squares(gram: Gram) -> float:
    """A gram's ``sum(Kc^2)``: its third element when it carries one, else summed here."""
    return gram[2] if len(gram) == 3 else np.sum(gram[0] * gram[0])


def _rounding_bound(gram_x: Gram, gram_y: Gram, n: int) -> float:
    """How far rounding can move the statistic ``sum(Kxc.T * Kyc) / n``: ``n^2 eps`` times the
    largest value it can take.

    The statistic sums ``n^2`` products. A floating-point sum of ``N`` terms errs by at most about
    ``N eps`` times the sum of the terms' magnitudes (Higham 2002, Section 3.1), and by
    Cauchy-Schwarz that sum is at most ``sqrt(sum(Kxc^2) sum(Kyc^2))``, which over ``n`` bounds the
    statistic itself. The rounding of the centered entries, a few ``eps`` each against entries of
    order one, adds less. The bound is therefore ``n eps sqrt(sum(Kxc^2) sum(Kyc^2))``.

    References
    ----------
    Higham (2002). Accuracy and Stability of Numerical Algorithms, 2nd ed. SIAM.
    """
    return n * _EPS * math.sqrt(_sum_of_squares(gram_x) * _sum_of_squares(gram_y))


def _gamma_arguments(
    gram_x: Gram, gram_y: Gram, n: int, variance: str
) -> tuple[float, float, float]:
    """The statistic of the gamma HSIC and the gamma law's ``(shape, statistic / scale)``."""
    cen_x, mu_x = gram_x[0], gram_x[1]
    cen_y, mu_y = gram_y[0], gram_y[1]

    statistic = 1.0 / n * np.sum(cen_x.T * cen_y)

    if variance == "bias_corrected":
        prod = cen_x * (1.0 / 6.0)  # 1/6 * cen_x * cen_y then squared, fused into one reused buffer
        prod *= cen_y
        prod **= 2
        var = 1.0 / n / (n - 1) * (np.sum(prod) - np.trace(prod))
        var = 72.0 * (n - 4) * (n - 5) / n / (n - 1) / (n - 2) / (n - 3) * var
    elif variance == "product":
        sq_x, sq_y = _sum_of_squares(gram_x), _sum_of_squares(gram_y)
        var = 2.0 * (n - 4) * (n - 5) / n / (n - 1) / (n - 2) / (n - 3)
        var *= sq_x / (n - 1) ** 2 * sq_y / (n - 1) ** 2
    else:
        raise ValueError(f"variance must be 'bias_corrected' or 'product', got {variance!r}")

    mean = 1.0 / n * (1 + mu_x * mu_y - mu_x - mu_y)

    shape = mean**2 / var
    scale = var * n / mean
    return statistic, shape, statistic / scale


#: Upper-tail value below which :func:`_log_upper_gamma` takes the continued fraction: the smallest
#: normal double, under which ``gammaincc`` loses precision and then underflows to ``0``.
_LOG_TAIL_FLOOR = float(np.finfo(np.float64).tiny)
#: The continued fraction's guard against a zero denominator, and its convergence tolerance.
_CF_TINY = 1e-300
_CF_EPS = 1e-16
_CF_MAX_ITER = 10_000
#: Machine epsilon of a double, the unit of :func:`_rounding_bound`.
_EPS = float(np.finfo(np.float64).eps)


def _log_upper_gamma(a: float, x: float) -> float:
    """``log Q(a, x)``, ``Q`` the regularized upper incomplete gamma function.

    While ``Q`` is a normal double, at least about ``2.2e-308``, the logarithm of ``gammaincc``, so
    every such p-value keeps its bits. Below that, which needs ``x > a + 1``,
    ``log Q = a log x - x - log Gamma(a) + log h`` with ``h`` the continued fraction of
    ``Gamma(a, x)`` by the modified Lentz method (Press et al., Numerical Recipes, Section 6.2),
    which agrees with 60-digit arithmetic to a relative ``5e-13`` over shapes from ``0.3`` to
    ``1000``. A non-finite argument gives ``nan``.
    """
    q = gammaincc(a, x)
    if q >= _LOG_TAIL_FLOOR or not x > a + 1.0:
        return math.log(q) if q > 0 else math.nan
    b = x + 1.0 - a
    c = 1.0 / _CF_TINY
    d = 1.0 / b
    h = d
    for i in range(1, _CF_MAX_ITER + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = d if abs(d) > _CF_TINY else _CF_TINY
        c = b + an / c
        c = c if abs(c) > _CF_TINY else _CF_TINY
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _CF_EPS:
            return a * math.log(x) - x - float(gammaln(a)) + math.log(h)
    return math.nan
