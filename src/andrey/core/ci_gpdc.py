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
        Seed for the null draws. ``None`` uses ``ANDREY_SEED``, or ``0`` when it is unset.

    Raises
    ------
    ValueError
        If ``data`` is not 2-D or contains NaN or infinity, or ``n_null`` is below 1.

    Notes
    -----
    **How it works.** For a conditioning set ``z``, ``x`` and ``y`` are each fitted by a
    Gaussian-process regression on ``z`` with a radial-basis-function kernel plus white noise,
    whose length scale and noise level maximize the marginal likelihood [2]_; the residuals are the
    observations minus the fitted means. An empty conditioning set uses ``x`` and ``y`` themselves.
    The residuals are replaced by their ranks, which gives them uniform marginals, and the
    statistic is their squared distance correlation, from the doubly centered matrices of their
    pairwise distances [3]_. After ranking, the null is the same statistic between the ranks and
    a random permutation of them, drawn ``n_null`` times; the p-value is the share of draws at
    least as large as the statistic [1]_.

    **Limitations.** Not benchmarked yet. Its tests run it on up to 400 rows and two conditioning
    variables. The test assumes ``x`` and ``y`` depend on the conditioning set additively, a
    function of it plus noise; a dependence through the noise's scale, such as ``x = z e``, stays
    in the residuals and reads as conditional dependence. The null ignores the degrees of freedom
    the regressions use, as [1]_ does. Each regression costs the cube of the number of rows per
    optimizer step, so thousands of rows take seconds per query. The p-value cannot fall between
    multiples of ``1 / n_null``.

    References
    ----------
    .. [1] Runge, J., Nowack, P., Kretschmer, M., Flaxman, S., and Sejdinovic, D. (2019).
       Detecting and quantifying causal associations in large nonlinear time series datasets.
       Science Advances 5(11), eaau4996.
    .. [2] Rasmussen, C. E., and Williams, C. K. I. (2006). Gaussian Processes for Machine
       Learning. MIT Press.
    .. [3] Szekely, G. J., Rizzo, M. L., and Bakirov, N. K. (2007). Measuring and testing
       dependence by correlation of distances. Annals of Statistics 35(6), 2769-2794.
    """

    def __init__(self, data: np.ndarray, *, n_null: int = 1000, seed: int | None = None) -> None:
        if n_null < 1:
            raise ValueError(f"n_null must be at least 1, got {n_null}")
        arr = _checked_data(data, "GPDC")
        std = arr.std(axis=0)
        std[std == 0] = 1.0
        self._data = (arr - arr.mean(axis=0)) / std
        self._n = arr.shape[0]
        self._n_null = int(n_null)
        self._seed = resolve_seed(seed)
        self._null: np.ndarray | None = None

    def __call__(self, x: int, y: int, condition_set: Iterable[int] = ()) -> float:
        """Return the GPDC p-value for ``x`` independent of ``y`` given ``condition_set``."""
        x_int, y_int, cond = _canonical(x, y, condition_set)
        rx, ry = self._residual(x_int, cond), self._residual(y_int, cond)
        if not (np.ptp(rx) > 0.0 and np.ptp(ry) > 0.0):
            return 1.0  # a constant variable: no variation to depend on
        statistic = _squared_distance_correlation(
            _centered_distances(self._ranks(rx)), _centered_distances(self._ranks(ry))
        )
        return float(np.mean(self._null_distribution() >= statistic))

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

    @staticmethod
    def _ranks(values: np.ndarray) -> np.ndarray:
        """Ranks 0 to n - 1, ties broken by row order."""
        return np.argsort(np.argsort(values, kind="stable"), kind="stable")

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
