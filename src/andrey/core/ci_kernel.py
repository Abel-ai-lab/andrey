"""Kernel conditional-independence tests for nonlinear data: KCI and RCoT.

Selected through ``indep_test="kci"`` and ``indep_test="rcot"``. Each test follows the
:class:`~andrey.core.ci.CITest` protocol: built once over the data and called as
``test(x, y, condition_set)``, returning a p-value where a larger value means more independent.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.special import gammaincc

from .ci import _canonical, _checked_data
from .independence import hsic_test
from .seeding import resolve_seed


def _standardize(data: np.ndarray) -> np.ndarray:
    """Z-score each column (``ddof=1``); a constant column becomes zero."""
    std = data.std(axis=0, ddof=1)
    std[std == 0] = 1.0
    return (data - data.mean(axis=0)) / std


def _centered_gaussian_kernel(columns: np.ndarray, width: float) -> np.ndarray:
    """Doubly centered Gaussian kernel ``H K H``, ``K_ij = exp(-|a_i - a_j|^2 / (2 width^2))``."""
    kernel = np.exp(squareform(pdist(columns, "sqeuclidean")) / (-2.0 * width * width))
    col_mean = kernel.mean(axis=0)
    return kernel - col_mean[None, :] - col_mean[:, None] + col_mean.mean()


def _gamma_pvalue(statistic: float, mean: float, var: float) -> float:
    """Upper tail at ``statistic`` of the gamma law with the given mean and variance."""
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

    Raises
    ------
    ValueError
        If ``data`` is not 2-D, contains NaN or infinity, or ``ridge`` is not positive.

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
    gamma law with that mean and variance gives the p-value.

    Every kernel is Gaussian, ``exp(-|a - b|^2 / (2 w^2))``, on standardized columns. Without a
    conditioning set, ``w`` is 0.8 below 200 rows, 0.5 below 1,200, and 0.3 beyond. With ``k``
    conditioning variables, the kernels of ``xs``, ``y``, and ``S`` share
    ``w = c sqrt(k)``, ``c`` 1.2 below 200 rows, 0.7 below 1,200, and 0.4 beyond, and ``S``
    enters ``xs`` at half scale. ``x`` joins the conditioning set and ``y`` does not, so a query
    is ordered (lower index first) before it runs: the p-value does not depend on argument
    order.

    **Limitations.** Not benchmarked yet. Each query with a conditioning set costs an
    eigendecomposition and four products of ``n x n`` matrices, so time grows as the cube of the
    number of rows and memory as its square: thousands of rows are practical, tens of thousands
    are not. The kernel widths are fixed, not fitted to the data. With one conditioning variable
    the test rejects a true independence at close to the rate ``alpha``; with two or more it
    rejects more often, increasingly so as the rows grow, so PC and FCI can keep edges a larger
    conditioning set would remove. The unit tests check its level with one conditioning variable
    on 200 rows and run it on up to 300 rows and two conditioning variables.

    References
    ----------
    .. [1] Zhang, K., Peters, J., Janzing, D., and Scholkopf, B. (2011). Kernel-based conditional
       independence test and application in causal discovery. UAI, 804-813.
    """

    def __init__(self, data: np.ndarray, *, ridge: float = 1e-3) -> None:
        if not ridge > 0.0:
            raise ValueError(f"ridge must be positive, got {ridge}")
        arr = _checked_data(data, "KCI")
        self._data = _standardize(arr)
        self._n = arr.shape[0]
        self._ridge = float(ridge)

    def __call__(self, x: int, y: int, condition_set: Iterable[int] = ()) -> float:
        """Return the KCI p-value for ``x`` independent of ``y`` given ``condition_set``."""
        x_int, y_int, cond = _canonical(x, y, condition_set)
        data, n = self._data, self._n
        if not (np.ptp(data[:, x_int]) > 0.0 and np.ptp(data[:, y_int]) > 0.0):
            return 1.0  # a constant variable: no variation to depend on
        if not cond:
            return hsic_test(data[:, x_int], data[:, y_int])

        width = (1.2 if n < 200 else 0.7 if n < 1200 else 0.4) * np.sqrt(len(cond))
        z = data[:, cond]
        kxs = _centered_gaussian_kernel(np.column_stack([data[:, x_int], 0.5 * z]), width)
        ky = _centered_gaussian_kernel(data[:, [y_int]], width)
        ks = _centered_gaussian_kernel(z, width)
        eigval, eigvec = np.linalg.eigh(ks)
        shrink = self._ridge / (np.clip(eigval, 0.0, None) + self._ridge)
        resid = (eigvec * shrink) @ eigvec.T  # R = ridge (Ks + ridge I)^-1
        a = resid @ kxs @ resid
        b = resid @ ky @ resid
        ab = a * b
        statistic = float(ab.sum()) / n
        mean = float(np.trace(ab)) / n
        var = 2.0 * float(np.sum(ab * ab)) / n**2
        return _gamma_pvalue(statistic, mean, var)


def _hall_buckley_eagleson_pvalue(statistic: float, weights: np.ndarray) -> float:
    """Upper tail at ``statistic`` of ``sum_i weights_i z_i^2``, ``z_i`` standard normal.

    Matches the first three cumulants with a shifted and scaled chi-squared law.
    """
    k1 = float(np.sum(weights))
    k2 = 2.0 * float(np.sum(weights**2))
    k3 = 8.0 * float(np.sum(weights**3))
    if not (k2 > 0.0 and k3 > 0.0):
        return 1.0  # a constant variable: no variation to depend on
    dof = 8.0 * k2**3 / k3**2
    chi2 = dof + np.sqrt(2.0 * dof) * (statistic - k1) / np.sqrt(k2)
    return float(gammaincc(dof / 2.0, max(chi2, 0.0) / 2.0))


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
        conditioning set's effect, at a cost that grows with their square.
    seed : int or None, default=None
        Seed for the random features. ``None`` uses ``ANDREY_SEED``, or ``0`` when it is unset.

    Raises
    ------
    ValueError
        If ``data`` is not 2-D or contains NaN or infinity, or a feature count is below 1.

    Notes
    -----
    **How it works.** A variable ``v`` with median pairwise distance ``s`` (over its first 500
    rows) gets features ``sqrt(2) cos(v W + b)`` with ``W`` normal of variance ``2 / s`` and
    ``b`` uniform on ``[0, 2 pi]``, the random features of the kernel ``exp(-|v - v'|^2 / s)``;
    every feature is standardized. With ``A`` and ``B`` the features of ``x`` and ``y``, and
    ``C`` those of the conditioning set, ``A`` and ``B`` are replaced by their residuals on ``C``
    (ridge ``1e-10``). The statistic is ``n`` times the squared Frobenius norm of the residuals'
    cross-covariance matrix. Under the null it is a weighted sum of chi-squared variables whose
    weights are the eigenvalues of the covariance of the products of residuals; the p-value
    matches the first three cumulants of that sum with a shifted and scaled chi-squared law
    (Hall-Buckley-Eagleson) [2]_ [3]_. An empty conditioning set uses ``A`` and ``B``, centered.

    The features of a query are drawn from a generator seeded by ``seed``, the two variables, and
    the conditioning set, so a p-value does not depend on the order of queries or of the
    arguments, and the same seed reproduces it.

    **Limitations.** Not benchmarked yet. The p-value is random through the features: a different
    seed moves it, most near the threshold. The regression on the conditioning-set features
    removes a strong nonlinear effect of that set only approximately; what remains reads as
    dependence, so with a conditioning set of three or more variables that drives both ``x`` and
    ``y``, the test rejects a true independence far more often than ``alpha``, and more so as the
    rows grow. RCoT tests conditional uncorrelatedness of the features, which misses a dependence
    that no feature captures. The unit tests check its level with up to two conditioning
    variables on 500 rows.

    References
    ----------
    .. [1] Strobl, E. V., Zhang, K., and Visweswaran, S. (2019). Approximate kernel-based
       conditional independence tests for fast non-parametric causal discovery. Journal of Causal
       Inference 7(1), 20180017.
    .. [2] Hall, P. (1983). Chi squared approximations to the distribution of a sum of independent
       random variables. Annals of Probability 11(4), 1028-1036.
    .. [3] Buckley, M. J., and Eagleson, G. K. (1988). An approximation to the distribution of
       quadratic forms in normal random variables. Australian Journal of Statistics 30A, 150-159.
    """

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
        arr = _checked_data(data, "RCoT")
        self._data = _standardize(arr)
        self._n = arr.shape[0]
        self._n_features = int(n_features)
        self._n_features_cond = int(n_features_cond)
        self._seed = resolve_seed(seed)

    def __call__(self, x: int, y: int, condition_set: Iterable[int] = ()) -> float:
        """Return the RCoT p-value for ``x`` independent of ``y`` given ``condition_set``."""
        x_int, y_int, cond = _canonical(x, y, condition_set)
        rng = np.random.default_rng([self._seed, x_int, y_int, len(cond), *cond])
        n = self._n
        a = self._features([x_int], self._n_features, rng)
        b = self._features([y_int], self._n_features, rng)
        if cond:
            c = self._features(cond, self._n_features_cond, rng)
            gram = c.T @ c + 1e-10 * (n - 1) * np.eye(c.shape[1])  # (n - 1)(Sigma_CC + ridge I)
            coef = np.linalg.solve(gram, c.T @ np.hstack([a, b]))
            resid = np.hstack([a, b]) - c @ coef
            a, b = resid[:, : a.shape[1]], resid[:, a.shape[1] :]
        cross_cov = a.T @ b / (n - 1)
        statistic = n * float(np.sum(cross_cov**2))
        products = (a[:, :, None] * b[:, None, :]).reshape(n, -1)
        weights = np.clip(np.linalg.eigvalsh(products.T @ products / n), 0.0, None)
        return _hall_buckley_eagleson_pvalue(statistic, weights)

    def _features(self, columns: list[int], count: int, rng: np.random.Generator) -> np.ndarray:
        """Standardized random Fourier features of ``columns`` for the median-distance kernel."""
        values = self._data[:, columns]
        dists = pdist(values[:500], "euclidean")
        scale = float(np.median(dists)) if dists.size else 0.0
        weights = rng.standard_normal((len(columns), count))
        offsets = rng.uniform(0.0, 2.0 * np.pi, count)
        if scale > 0.0:
            weights *= np.sqrt(2.0 / scale)
        return _standardize(np.sqrt(2.0) * np.cos(values @ weights + offsets))
