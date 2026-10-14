"""Gaussian-process regression followed by a distance-correlation test (GPDC), for nonlinear data.

Selected through ``indep_test="gpdc"``. The test follows the :class:`~andrey.core.ci.CITest`
protocol: built once over the data and called as ``test(x, y, condition_set)``, returning a p-value
where a larger value means more independent.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from .ci import _canonical, _checked_data
from .seeding import resolve_seed

__all__ = ["GPDC"]


def _centered_distances(ranks: np.ndarray) -> np.ndarray:
    """The doubly centered matrix of pairwise distances between the entries of ``ranks``."""
    dist = np.abs(ranks[:, None] - ranks[None, :]).astype(np.float64)
    col_mean = dist.mean(axis=0)
    return dist - col_mean[None, :] - col_mean[:, None] + col_mean.mean()


def _squared_distance_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Squared distance correlation from the doubly centered distance matrices ``a`` and ``b``.

    Zero when either variable is constant.
    """
    var_a, var_b = float(np.mean(a * a)), float(np.mean(b * b))
    if not (var_a > 0.0 and var_b > 0.0):
        return 0.0
    return float(np.mean(a * b)) / np.sqrt(var_a * var_b)


class GPDC:
    """Gaussian-process regression, then a distance-correlation test of the residuals (GPDC).

    Each query regresses ``x`` and ``y`` on the conditioning set with a Gaussian process, ranks the
    two residuals, and tests their distance correlation, which is zero only when they are
    independent. The null distribution of the ranked statistic depends only on the number of rows,
    so it is drawn once and reused by every query.

    Parameters
    ----------
    data : ndarray of shape (n_samples, n_variables)
        Observed data, one row per sample. Each column is standardized to unit variance.
    n_null : int, default=1000
        Draws of the null distribution of the distance correlation.
    seed : int or None, default=None
        Seed for the null draws and the tie-breaking. ``None`` uses ``ANDREY_SEED``, or ``0`` when
        it is unset.

    Raises
    ------
    ValueError
        If ``data`` is not 2-D, contains NaN or infinity, or has values too large to
        standardize, or ``n_null`` is below 1.

    Notes
    -----
    **How it works.** For a conditioning set ``z``, ``x`` and ``y`` are each fitted by a
    Gaussian-process regression on ``z`` with a radial-basis-function kernel plus white noise,
    whose length scale and noise level maximize the marginal likelihood [2]_; the residuals are the
    observations minus the fitted means. An empty conditioning set uses ``x`` and ``y`` themselves.
    The residuals are replaced by their ranks, which gives them uniform marginals; tied values
    are ordered by a random key per row and variable, drawn once from ``seed``, so that every
    ranking stays equally likely under the null. The statistic is their squared distance
    correlation, from the doubly centered matrices of their pairwise distances [3]_. After
    ranking, the null is the same statistic between the ranks and a random permutation of them,
    drawn ``n_null`` times [1]_. With ``b`` of the ``n_null`` draws at least as large as the
    statistic, the p-value is ``(b + 1) / (n_null + 1)``: the observed statistic counts as one
    more draw, so the p-value is never zero [4]_.

    **Limitations.** Not benchmarked yet. Its tests run it on up to 500 rows and two conditioning
    variables. The test assumes ``x`` and ``y`` depend on the conditioning set additively, a
    function of it plus noise; a dependence through the noise's scale, such as ``x = z e``, stays
    in the residuals and reads as conditional dependence. The null ignores the degrees of freedom
    the regressions use, as [1]_ does. Each regression costs the cube of the number of rows per
    optimizer step, so thousands of rows take seconds per query. The p-value is a multiple of
    ``1 / (n_null + 1)``, and is at least that.

    References
    ----------
    .. [1] Runge, J., Nowack, P., Kretschmer, M., Flaxman, S., and Sejdinovic, D. (2019).
       Detecting and quantifying causal associations in large nonlinear time series datasets.
       Science Advances 5(11), eaau4996.
    .. [2] Rasmussen, C. E., and Williams, C. K. I. (2006). Gaussian Processes for Machine
       Learning. MIT Press.
    .. [3] Szekely, G. J., Rizzo, M. L., and Bakirov, N. K. (2007). Measuring and testing
       dependence by correlation of distances. Annals of Statistics 35(6), 2769-2794.
    .. [4] Phipson, B., and Smyth, G. K. (2010). Permutation p-values should never be zero:
       calculating exact p-values when permutations are randomly drawn. Statistical Applications
       in Genetics and Molecular Biology 9(1), Article 39.
    """

    def __init__(self, data: np.ndarray, *, n_null: int = 1000, seed: int | None = None) -> None:
        if n_null < 1:
            raise ValueError(f"n_null must be at least 1, got {n_null}")
        arr = _checked_data(data, "GPDC")
        with np.errstate(over="ignore"):  # an overflow raises below
            std = arr.std(axis=0)
        if not np.isfinite(std).all():
            raise ValueError("data values too large to standardize; rescale the columns")
        std[np.ptp(arr, axis=0) == 0] = np.inf  # identical values; their std is rounding, not 0
        self._data = (arr - arr.mean(axis=0)) / std
        self._n = arr.shape[0]
        self._n_null = int(n_null)
        self._seed = resolve_seed(seed)
        self._tie_keys = np.random.default_rng(self._seed).random(arr.shape)
        self._null: np.ndarray | None = None

    def __call__(self, x: int, y: int, condition_set: Iterable[int] = ()) -> float:
        """Return the GPDC p-value for ``x`` independent of ``y`` given ``condition_set``."""
        x_int, y_int, cond = _canonical(x, y, condition_set)
        rx, ry = self._residual(x_int, cond), self._residual(y_int, cond)
        if not (np.ptp(rx) > 0.0 and np.ptp(ry) > 0.0):
            return 1.0  # a constant variable: no variation to depend on
        statistic = _squared_distance_correlation(
            _centered_distances(self._ranks(rx, x_int)), _centered_distances(self._ranks(ry, y_int))
        )
        exceed = np.count_nonzero(self._null_distribution() >= statistic)
        return (exceed + 1) / (self._n_null + 1)

    def _residual(self, column: int, cond: list[int]) -> np.ndarray:
        target = self._data[:, column]
        if not cond:
            return target
        regressor = GaussianProcessRegressor(kernel=RBF() + WhiteKernel(), alpha=0.0)
        features = self._data[:, cond]
        with warnings.catch_warnings():
            # A length scale or noise level at its bound is a valid fit (a target unrelated to the
            # conditioning set, or explained by it exactly), not a failure.
            warnings.simplefilter("ignore", ConvergenceWarning)
            regressor.fit(features, target)
        return target - regressor.predict(features)

    def _ranks(self, values: np.ndarray, column: int) -> np.ndarray:
        """Ranks 0 to n - 1, ties broken by the random key of ``column``."""
        return np.argsort(np.lexsort((self._tie_keys[:, column], values)))

    def _null_distribution(self) -> np.ndarray:
        """``n_null`` distance correlations of the ranks with a random permutation of them."""
        if self._null is None:
            rng = np.random.default_rng([self._seed, self._n])
            a = _centered_distances(np.arange(self._n))
            draws = np.empty(self._n_null)
            for i in range(self._n_null):
                perm = rng.permutation(self._n)
                draws[i] = _squared_distance_correlation(a, a[np.ix_(perm, perm)])
            self._null = draws
        return self._null
