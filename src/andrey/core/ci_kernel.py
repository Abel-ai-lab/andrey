"""Kernel conditional-independence tests for nonlinear data: KCI and RCoT.

Selected through ``indep_test="kci"`` and ``indep_test="rcot"``. Each test follows the
:class:`~andrey.core.ci.CITest` protocol: built once over the data and called as
``test(x, y, condition_set)``, returning a p-value where a larger value means more independent.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import pdist, squareform
from scipy.special import gammaincc

from andrey.search.hc import tie_ranks

from . import backend, env
from .ci import _canonical_ranked, _checked_data
from .independence import hsic_test
from .seeding import resolve_seed
from .warning_policy import SampleSizeWarning, warn_once

__all__ = ["KCI", "RCoT"]

# The smallest sample size Strobl et al. (2019) evaluate; with two conditioning variables, fewer
# rows were measured to reject true independences more often than alpha.
_RCOT_MIN_ROWS = 500

# Each cache of RCoT's features or KCI's matrices holds about this many bytes.
_CACHE_BYTES = 2**28

# Rows from which the skeleton scans run KCI on every BLAS thread. The gain depends on the cores a
# shared node assigns: PC fits with 8 threads ran 0.42x to 1.56x the one-thread speed on 500 rows,
# where the eigendecomposition slows with threads, 0.72x to 2.45x on 1,000, and 1.48x to 2.88x on
# 2,000.
_KCI_BLAS_THREADS_FROM = 2000

# Smallest kernel matrix (rows squared) at which `auto` runs a KCI query with a conditioning set on
# a CUDA GPU. `qa/gpu_calibration_validate.py` on an A100-SXM4-80GB against 8 CPU threads found
# 31,250 (177 rows); per query with two conditioning variables, the GPU took 0.52x the CPU's speed
# at 100 rows, 1.4x at 200, and 9.5x at 1,000. KCI computes in float64, so `mps` (float32) is never
# used.
_KCI_GPU_THRESHOLDS: dict[str, int | None] = {"cuda": 31_250, "mps": None}


def _cache_entries(floats_per_entry: int) -> int:
    """How many entries of ``floats_per_entry`` float64 values fit in ``_CACHE_BYTES``."""
    return max(16, _CACHE_BYTES // (8 * floats_per_entry))


def _standardize(data: np.ndarray) -> np.ndarray:
    """Z-score each column (``ddof=1``); a column of identical values becomes zero.

    The rounded mean of identical values can differ from them, which leaves a standard deviation
    near zero rather than zero, so constancy is read from the range. A varying column whose
    variance overflows or underflows to zero raises.
    """
    with np.errstate(over="ignore", under="ignore"):  # either one raises below
        std = data.std(axis=0, ddof=1)
    if not np.isfinite(std).all():
        raise ValueError("data values too large to standardize; rescale the columns")
    std[np.ptp(data, axis=0) == 0] = np.inf
    if (std == 0).any():
        raise ValueError("data values too small to standardize; rescale the columns")
    return (data - data.mean(axis=0)) / std


def _centered_gaussian_kernel(columns: np.ndarray, width: float) -> np.ndarray:
    """Doubly centered Gaussian kernel ``H K H``, ``K_ij = exp(-|a_i - a_j|^2 / (2 width^2))``."""
    kernel = np.exp(squareform(pdist(columns, "sqeuclidean")) / (-2.0 * width * width))
    col_mean = kernel.mean(axis=0)
    return kernel - col_mean[None, :] - col_mean[:, None] + col_mean.mean()


def _kci_moments(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """KCI's statistic ``tr(A B) / n`` and the mean and variance of its null law.

    Under the null the statistic is ``(1 / n) sum_k lam_k z_k^2``, ``lam`` the eigenvalues of
    ``w w'`` with ``w_t = vec(psi_t phi_t')``, ``A = Psi Psi'`` and ``B = Phi Phi'``. ``w' w`` has
    entries ``A_st B_st``, so the mean ``tr(w w') / n`` and the variance ``2 tr((w w')^2) / n^2``
    come from the elementwise product of ``A`` and ``B``.
    """
    n = a.shape[0]
    ab = a * b
    statistic = float(ab.sum()) / n
    mean = float(np.trace(ab)) / n
    var = 2.0 * float(np.sum(ab * ab)) / n**2
    return statistic, mean, var


def _gamma_pvalue(statistic: float, mean: float, var: float) -> float:
    """Upper tail at ``statistic`` of the gamma law with the given mean and variance."""
    if not np.isfinite([statistic, mean, var]).all():
        raise FloatingPointError("KCI statistic or null moments are not finite")
    if not (mean > 0.0 and var > 0.0):
        return 1.0  # a constant variable: no variation to depend on
    shape = mean * mean / var
    scale = var / mean
    return float(gammaincc(shape, statistic / scale))


class KCI:
    """Kernel conditional-independence test (KCI) with the gamma approximation of its null.

    Each query compares Gaussian-kernel features of ``x`` and ``y`` after kernel ridge regression
    on the conditioning set removes what it explains. The statistic is matched to a gamma law
    through its null mean and variance, which gives the p-value in closed form.

    Parameters
    ----------
    data : ndarray of shape (n_samples, n_variables)
        Observed data, one row per sample. Each column is standardized to unit variance.
    ridge : float, default=1e-3
        Regularization of the kernel ridge regression on the conditioning set. Smaller values fit
        the conditioning set more closely.
    rank : int or None, default=None
        With a conditioning set, replace each kernel matrix by a pivoted incomplete Cholesky
        factorization of at most this many columns, so a query's time grows with the square of the
        number of rows instead of its cube. The p-value then approximates the exact one. ``None``
        computes the exact kernels.

    Raises
    ------
    ValueError
        If ``data`` is not 2-D, has fewer than 2 rows, contains NaN or infinity, or has values
        too large or too small to standardize, ``ridge`` is not positive and finite, or ``rank`` is
        below 1.

    Notes
    -----
    **How it works.** With ``Kx`` the centered Gaussian kernel matrix of a variable and
    ``n`` rows, an empty conditioning set gives ``T = tr(Kx Ky) / n``, with null mean
    ``tr(Kx) tr(Ky) / n^2`` and variance ``2 tr(Kx^2) tr(Ky^2) / n^4``: the gamma HSIC test of
    :func:`~andrey.core.independence.hsic_test`. A conditioning set ``S`` joins ``x`` and ``S``
    into one variable ``xs`` and residualizes with ``R = ridge (Ks + ridge I)^-1``:
    ``A = R Kxs R``, ``B = R Ky R``, ``T = tr(A B) / n``. Under the null, ``T`` is a weighted sum
    of chi-squared variables whose mean ``sum_t A_tt B_tt / n`` and variance
    ``2 sum_st (A_st B_st)^2 / n^2`` come from the elementwise product of ``A`` and ``B``. The
    gamma law with that mean and variance gives the p-value [1]_.

    Every kernel is Gaussian, ``exp(-|a - b|^2 / (2 w^2))``, on standardized columns. Without a
    conditioning set, ``w`` is 0.8 below 200 rows, 0.5 below 1,200, and 0.3 beyond. With ``k``
    conditioning variables, the kernels of ``xs``, ``y``, and ``S`` share
    ``w = c sqrt(k)``, ``c`` 1.2 below 200 rows, 0.7 below 1,200, and 0.4 beyond, and ``S``
    enters ``xs`` at half scale. ``x`` joins the conditioning set and ``y`` does not, so of the
    two variables the one with the lower :func:`~andrey.search.hc.tie_ranks` rank, a rank
    computed from each column's own data, takes the role of ``x``, and the conditioning set's
    columns enter its kernels in rank order: the p-value depends on neither the argument order
    nor the column order.

    With ``rank``, each kernel of a conditional query is a pivoted incomplete Cholesky factor
    ``L`` with ``K ~ L L'`` [2]_, centered, and ``R = I - Ls (Ls' Ls + ridge I)^-1 Ls'``, so
    ``A`` and ``B`` are products of ``n x rank`` factors and no ``n x n`` system is solved. The
    factorization stops before ``rank`` columns when ``L L'`` reproduces the kernel to rounding.

    An exact query with a conditioning set runs on a CUDA GPU, in float64, under the ``cuda``
    backend, or under ``auto`` from ``ANDREY_KCI_GPU_THRESHOLD`` rows squared. Its p-value then
    matches the CPU's to rounding.

    **Limitations.** Not benchmarked yet. Its tests run it on up to 500 rows and two conditioning
    variables. Each query with a conditioning set costs two products of ``n x n`` matrices, plus
    an eigendecomposition for a conditioning set not seen before and two more products for a new
    ``y`` with it; the test keeps up to 256 MB of each kind of matrix, at least one. So time grows
    as the cube of the number of rows and memory as its square: thousands of rows are practical,
    tens of thousands are not. The kernel widths follow the row count and the size of the
    conditioning set, not the data. On simulated draws with up to three conditioning variables and
    500 rows, it rejected a true independence at rates between ``alpha`` and twice ``alpha``; a
    test that rejects too often makes PC and FCI keep edges.

    References
    ----------
    .. [1] Zhang, K., Peters, J., Janzing, D., and Scholkopf, B. (2011). Kernel-based conditional
       independence test and application in causal discovery. UAI, 804-813.
    .. [2] Fine, S., and Scheinberg, K. (2001). Efficient SVM training using low-rank kernel
       representations. Journal of Machine Learning Research 2, 243-264.
    """

    @property
    def pin_blas(self) -> bool:
        """Whether the skeleton scans run the test on one BLAS thread: on fewer than 2,000 rows."""
        return self._n < _KCI_BLAS_THREADS_FROM

    def __init__(self, data: np.ndarray, *, ridge: float = 1e-3, rank: int | None = None) -> None:
        if not 0.0 < ridge < np.inf:
            raise ValueError(f"ridge must be positive and finite, got {ridge}")
        if rank is not None and rank < 1:
            raise ValueError(f"rank must be at least 1, got {rank}")
        arr = _checked_data(data, "KCI", min_rows=2)
        # The test holds data column j at index _tie_rank[j], in C order, so a query reads the
        # same values in the same order whatever the column order or memory layout.
        self._tie_rank = tie_ranks(arr).tolist()
        self._data = _standardize(np.ascontiguousarray(arr[:, np.argsort(self._tie_rank)]))
        self._n = arr.shape[0]
        self._ridge = float(ridge)
        self._rank = None if rank is None else int(rank)
        # R and R Ky R do not depend on x, so queries with the same conditioning set share them.
        entries = max(1, _CACHE_BYTES // (8 * self._n * self._n))
        self._resid = lru_cache(entries)(self._residual_maker)
        self._resid_y = lru_cache(entries)(self._residualized_kernel)

    def __call__(self, x: int, y: int, condition_set: Iterable[int] = ()) -> float:
        """Return the KCI p-value for ``x`` independent of ``y`` given ``condition_set``."""
        x_int, y_int, cond = _canonical_ranked(x, y, condition_set, self._tie_rank)
        data, n = self._data, self._n
        if not (np.ptp(data[:, x_int]) > 0.0 and np.ptp(data[:, y_int]) > 0.0):
            return 1.0  # a constant variable: no variation to depend on
        if not cond:
            return hsic_test(data[:, x_int], data[:, y_int])

        width = self._width(len(cond))
        z = data[:, cond]
        if self._rank is not None:
            return self._low_rank_pvalue(data[:, x_int], data[:, y_int], z, width, self._rank)
        xs = np.column_stack([data[:, x_int], 0.5 * z])
        dev = _kci_backend(n)
        if dev != "numpy":
            return _gamma_pvalue(
                *_kci_moments_torch(xs, data[:, [y_int]], z, width, self._ridge, dev)
            )
        resid = self._resid(tuple(cond))
        kxs = _centered_gaussian_kernel(xs, width)
        return _gamma_pvalue(*_kci_moments(resid @ kxs @ resid, self._resid_y(y_int, tuple(cond))))

    def _width(self, size: int) -> float:
        """The Gaussian kernels' width with ``size`` conditioning variables."""
        n = self._n
        return (1.2 if n < 200 else 0.7 if n < 1200 else 0.4) * np.sqrt(size)

    def _residual_maker(self, cond: tuple[int, ...]) -> np.ndarray:
        """``R = ridge (Ks + ridge I)^-1`` of the conditioning set ``cond``."""
        ks = _centered_gaussian_kernel(self._data[:, list(cond)], self._width(len(cond)))
        eigval, eigvec = np.linalg.eigh(ks)
        shrink = self._ridge / (np.clip(eigval, 0.0, None) + self._ridge)
        return (eigvec * shrink) @ eigvec.T

    def _residualized_kernel(self, y: int, cond: tuple[int, ...]) -> np.ndarray:
        """``B = R Ky R``: the kernel of column ``y`` with what ``cond`` explains removed."""
        resid = self._resid(cond)
        ky = _centered_gaussian_kernel(self._data[:, [y]], self._width(len(cond)))
        return resid @ ky @ resid

    def _low_rank_pvalue(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray, width: float, rank: int
    ) -> float:
        """The p-value with each kernel replaced by its centered incomplete Cholesky factors.

        With ``Ks ~ Ls Ls'``, ``R = ridge (Ks + ridge I)^-1 = I - Ls (Ls' Ls + ridge I)^-1 Ls'``,
        so ``A = R Kxs R = P P'`` with ``P = R Lxs``, and ``B = Q Q'`` with ``Q = R Ly``.
        """
        lxs = _centered_cholesky_factor(np.column_stack([x, 0.5 * z]), width, rank)
        ly = _centered_cholesky_factor(y[:, None], width, rank)
        ls = _centered_cholesky_factor(z, width, rank)
        inner = ls.T @ ls + self._ridge * np.eye(ls.shape[1])
        p = lxs - ls @ np.linalg.solve(inner, ls.T @ lxs)
        q = ly - ls @ np.linalg.solve(inner, ls.T @ ly)
        return _gamma_pvalue(*_kci_moments(p @ p.T, q @ q.T))


def _kci_backend(n: int) -> str:
    """Where a KCI query with a conditioning set on ``n`` rows runs: ``numpy``, or torch on ``cpu``
    or ``cuda``."""
    manual = env.KCI_GPU_THRESHOLD.read()
    return backend.select(
        ("numpy", "cpu", "cuda"),
        ("cuda", "numpy"),
        metric=n * n,
        thresholds=_KCI_GPU_THRESHOLDS if manual is None else {"cuda": manual},
        amortized=True,
        op="kci",
    )


def _kci_moments_torch(
    xs: np.ndarray, y: np.ndarray, z: np.ndarray, width: float, ridge: float, dev: str
) -> tuple[float, float, float]:
    """KCI's statistic and null moments as :meth:`KCI.__call__` computes them, in float64 torch on
    ``dev``."""
    torch = backend.torch()
    n = xs.shape[0]

    def kernel(points: np.ndarray):
        p = torch.as_tensor(points, dtype=torch.float64, device=dev)
        sq = torch.zeros((n, n), dtype=torch.float64, device=dev)
        for c in range(p.shape[1]):
            sq += (p[:, c, None] - p[None, :, c]) ** 2
        k = torch.exp(sq / (-2.0 * width * width))
        col_mean = k.mean(dim=0)
        return k - col_mean[None, :] - col_mean[:, None] + col_mean.mean()

    eigval, eigvec = torch.linalg.eigh(kernel(z))
    resid = (eigvec * (ridge / (eigval.clamp(min=0.0) + ridge))) @ eigvec.T
    del eigvec
    ab = (resid @ kernel(xs) @ resid) * (resid @ kernel(y) @ resid)
    statistic = float(ab.sum()) / n
    mean = float(torch.trace(ab)) / n
    var = 2.0 * float((ab * ab).sum()) / n**2
    return statistic, mean, var


def _centered_cholesky_factor(points: np.ndarray, width: float, rank: int) -> np.ndarray:
    """Centered factor ``H L`` of a pivoted incomplete Cholesky ``K ~ L L'`` of the Gaussian kernel
    of ``points``, with at most ``rank`` columns.

    Each step takes the row with the largest remaining diagonal as the pivot; the factorization
    stops early when every remaining diagonal entry is below ``1e-12`` (the kernel's diagonal is
    1), where ``L L'`` equals ``K`` to rounding.
    """
    n = points.shape[0]
    rank = min(rank, n)
    factor = np.zeros((n, rank))
    diag = np.ones(n)
    scale = -0.5 / (width * width)
    for j in range(rank):
        pivot = int(np.argmax(diag))
        if diag[pivot] < 1e-12:
            factor = factor[:, :j]
            break
        column = np.exp(scale * np.sum((points - points[pivot]) ** 2, axis=1))
        factor[:, j] = (column - factor[:, :j] @ factor[pivot, :j]) / np.sqrt(diag[pivot])
        diag = np.maximum(diag - factor[:, j] ** 2, 0.0)
    return factor - factor.mean(axis=0)


def _hall_buckley_eagleson_pvalue(statistic: float, weights: np.ndarray) -> float:
    """Upper tail at ``statistic`` of ``sum_i weights_i z_i^2``, ``z_i`` standard normal.

    Matches the first three cumulants with a shifted and scaled chi-squared law.
    """
    k1 = float(np.sum(weights))
    k2 = 2.0 * float(np.sum(weights**2))
    k3 = 8.0 * float(np.sum(weights**3))
    if not np.isfinite([statistic, k1, k2, k3]).all():
        raise FloatingPointError("RCoT statistic or null weights are not finite")
    if not (k2 > 0.0 and k3 > 0.0):
        return 1.0  # a constant variable: no variation to depend on
    dof = 8.0 * k2**3 / k3**2
    chi2 = dof + np.sqrt(2.0 * dof) * (statistic - k1) / np.sqrt(k2)
    return float(gammaincc(dof / 2.0, max(chi2, 0.0) / 2.0))


def _median_distance(values: np.ndarray) -> float:
    """The median pairwise distance between the first 500 rows of ``values``; 0 below 2 rows."""
    dists = pdist(values[:500], "euclidean")
    return float(np.median(dists)) if dists.size else 0.0


def _fourier_features(
    values: np.ndarray, count: int, rng: np.random.Generator, scale: float | None = None
) -> np.ndarray:
    """``count`` random Fourier features of the rows of ``values`` for the Gaussian kernel
    ``exp(-|a - b|^2 / (2 s^2))``, ``s`` the median pairwise distance over the first 500 rows, or
    ``scale`` when given.

    The average product of two rows' features converges to their kernel value.
    """
    if scale is None:
        scale = _median_distance(values)
    weights = rng.standard_normal((values.shape[1], count))
    offsets = rng.uniform(0.0, 2.0 * np.pi, count)
    if scale > 0.0:
        weights /= scale
    return np.sqrt(2.0) * np.cos(values @ weights + offsets)


class RCoT:
    """Randomized conditional correlation test (RCoT), a fast approximation of KCI.

    Each query maps ``x``, ``y``, and the conditioning set through random Fourier features of a
    Gaussian kernel, removes from the ``x`` and ``y`` features what a linear ridge regression on
    the conditioning-set features explains, and tests the cross-covariance of what is left. The
    cost of a query grows linearly with the number of rows.

    Parameters
    ----------
    data : ndarray of shape (n_samples, n_variables)
        Observed data, one row per sample. Each column is standardized to unit variance.
    n_features : int, default=5
        Random Fourier features drawn for ``x`` and for ``y``.
    n_features_cond : int, default=100
        Random Fourier features drawn for the conditioning set. More features remove more of the
        conditioning set's effect, at a cost that grows with their square. The default is four
        times the 25 of [1]_: with 25, two conditioning variables that drive both ``x`` and ``y``
        leave enough of their effect to reject most true independences at 500 rows.
    seed : int or None, default=None
        Seed for the random features. ``None`` uses ``ANDREY_SEED``, or ``0`` when it is unset.

    Raises
    ------
    ValueError
        If ``data`` is not 2-D, has fewer than 2 rows, contains NaN or infinity, or has values
        too large or too small to standardize, or a feature count is below 1.

    Warns
    -----
    SampleSizeWarning
        On the first query with a conditioning set, when ``data`` has fewer than 500 rows, the
        smallest sample size [1]_ evaluates.

    Notes
    -----
    **How it works.** A variable ``v`` with median pairwise distance ``s`` (over its first 500
    rows) gets features ``sqrt(2) cos(v W + b)`` with ``W`` normal of standard deviation
    ``1 / s`` and ``b`` uniform on ``[0, 2 pi]``, the random features of the Gaussian kernel
    ``exp(-|v - v'|^2 / (2 s^2))``. The median distance is that kernel's bandwidth, the median
    heuristic [4]_ that [1]_ uses; [1]_ prints the kernel as ``exp(-|v - v'|^2 / s)``. Every
    feature is standardized [1]_. With ``A`` and ``B`` the features of ``x`` and ``y``, and ``C``
    those of the conditioning set, ``A`` and ``B`` are replaced by their residuals on ``C``
    (ridge ``1e-10``). The statistic is ``n`` times the squared Frobenius norm of the residuals'
    cross-covariance matrix, as [1]_ defines it. Under the null it is a weighted sum of
    chi-squared variables whose weights are the eigenvalues of the covariance of the products of
    residuals; the p-value matches the first three cumulants of that sum with a shifted and scaled
    chi-squared law (Hall-Buckley-Eagleson) [2]_ [3]_. An empty conditioning set uses ``A`` and
    ``B``, centered.

    The features of ``x`` and of ``y`` are drawn from generators seeded by ``seed``, the variable,
    and the conditioning set, and those of the conditioning set from one seeded by ``seed`` and the
    set; each column enters by its :func:`~andrey.search.hc.tie_ranks` rank, computed from the
    column's own data. The queries of one pair with different conditioning sets therefore use
    independent features. PC removes an edge when any conditioning set gives a p-value above
    ``alpha``, and features shared by those queries would make them reject together, which keeps
    edges between conditionally independent variables. Queries on the same variable and
    conditioning set share its features, and the test object keeps them, and the residuals, for
    later queries. A p-value does not depend on the order of the queries, of the arguments, or of
    the columns, and the same seed reproduces it.

    **Limitations.** Not benchmarked yet. Its tests run it on up to 500 rows and two conditioning
    variables. Measured on columns independent of each other, its rate of rejecting a true
    conditional independence with two conditioning variables was about twice ``alpha`` at 100 rows
    and near ``alpha`` from 500 rows. With three, four, or six conditioning variables the rate
    averaged about 1.3 times ``alpha`` from 500 to 2,000 rows, with no row count from which it held
    at ``alpha``. The p-value is random through the features: a different seed moves it, most near
    the threshold. The regression on the conditioning-set features removes a strong nonlinear effect
    of that set only approximately, and what remains reads as dependence: with two or more
    conditioning variables that drive both ``x`` and ``y``, the test rejects a true independence
    more often than ``alpha``, increasingly so as the rows grow (on simulated draws with three such
    variables and 2,000 rows, nearly always). RCoT tests conditional uncorrelatedness of
    ``n_features`` features per variable, which can miss a dependence the full kernel test finds,
    such as one between an odd and an even function of the same cause on a few hundred rows.

    References
    ----------
    .. [1] Strobl, E. V., Zhang, K., and Visweswaran, S. (2019). Approximate kernel-based
       conditional independence tests for fast non-parametric causal discovery. Journal of Causal
       Inference 7(1), 20180017.
    .. [2] Hall, P. (1983). Chi squared approximations to the distribution of a sum of independent
       random variables. Annals of Probability 11(4), 1028-1036.
    .. [3] Buckley, M. J., and Eagleson, G. K. (1988). An approximation to the distribution of
       quadratic forms in normal random variables. Australian Journal of Statistics 30A, 150-159.
    .. [4] Gretton, A., Fukumizu, K., Teo, C. H., Song, L., Scholkopf, B., and Smola, A. J.
       (2007). A kernel statistical test of independence. Advances in Neural Information
       Processing Systems 20, 585-592.
    """

    pin_blas = True  # read by pins_blas: the skeleton scans run it on one BLAS thread

    def __init__(
        self,
        data: np.ndarray,
        *,
        n_features: int = 5,
        n_features_cond: int = 100,
        seed: int | None = None,
    ) -> None:
        if n_features < 1 or n_features_cond < 1:
            raise ValueError(
                f"feature counts must be at least 1, got {n_features} and {n_features_cond}"
            )
        arr = _checked_data(data, "RCoT", min_rows=2)
        # The test holds data column j at index _tie_rank[j], in C order, so a query reads the
        # same values in the same order whatever the column order or memory layout.
        self._tie_rank = tie_ranks(arr).tolist()
        self._data = _standardize(np.ascontiguousarray(arr[:, np.argsort(self._tie_rank)]))
        self._n = arr.shape[0]
        self._n_features = int(n_features)
        self._n_features_cond = int(n_features_cond)
        self._seed = resolve_seed(seed)
        # Bandwidths, features, and residuals depend only on their columns, so queries share them.
        n = self._n
        self._scale = lru_cache(maxsize=None)(
            lambda column: _median_distance(self._data[:, [column]])
        )
        self._residual = lru_cache(_cache_entries(n * self._n_features))(self._fit_residual)
        self._cond_factor = lru_cache(_cache_entries(n * self._n_features_cond))(self._factor)

    def __call__(self, x: int, y: int, condition_set: Iterable[int] = ()) -> float:
        """Return the RCoT p-value for ``x`` independent of ``y`` given ``condition_set``."""
        x_int, y_int, cond = _canonical_ranked(x, y, condition_set, self._tie_rank)
        if cond and self._n < _RCOT_MIN_ROWS:
            warn_once(
                f"RCoT on fewer than {_RCOT_MIN_ROWS} rows may reject a true conditional "
                "independence more often than alpha.",
                SampleSizeWarning,
            )
        n = self._n
        a, b = self._residual(x_int, tuple(cond)), self._residual(y_int, tuple(cond))
        cross_cov = a.T @ b / (n - 1)
        statistic = n * float(np.sum(cross_cov**2))
        products = (a[:, :, None] * b[:, None, :]).reshape(n, -1)
        weights = np.clip(np.linalg.eigvalsh(products.T @ products / n), 0.0, None)
        return _hall_buckley_eagleson_pvalue(statistic, weights)

    def _fit_residual(self, column: int, cond: tuple[int, ...]) -> np.ndarray:
        """Standardized features of ``column``, drawn from a generator seeded by the seed,
        ``column``, and ``cond``, minus what a ridge regression on the features of ``cond``
        explains; the features themselves when ``cond`` is empty."""
        # The leading 1 (0 for a conditioning set) keeps the two kinds of draws apart.
        rng = np.random.default_rng([self._seed, 1, self._n_features, column, *cond])
        values = self._data[:, [column]]
        features = _standardize(
            _fourier_features(values, self._n_features, rng, self._scale(column))
        )
        if not cond:
            return features
        c, factor = self._cond_factor(cond)
        return features - c @ cho_solve(factor, c.T @ features)

    def _factor(self, cond: tuple[int, ...]) -> tuple[np.ndarray, tuple]:
        """The conditioning set's standardized features ``C``, drawn from a generator seeded by the
        seed and ``cond``, and the Cholesky factor of ``C'C + ridge (n - 1) I``, that is
        ``(n - 1)(Sigma_CC + ridge I)``."""
        rng = np.random.default_rng([self._seed, 0, self._n_features_cond, *cond])
        values = self._data[:, list(cond)]
        c = _standardize(_fourier_features(values, self._n_features_cond, rng))
        gram = c.T @ c + 1e-10 * (self._n - 1) * np.eye(c.shape[1])
        return c, cho_factor(gram)
