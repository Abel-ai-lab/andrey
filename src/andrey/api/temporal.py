"""Temporal-family facades: VARMA-LiNGAM and Longitudinal LiNGAM.

Each callable runs the native engine and returns one uniform ``StructureOutput`` wrapping a
:class:`~andrey.core.TemporalStructure` (a per-lag stack of graphs plus the lossless coefficient
weights). The mapping onto the public core contract lives in the private ``_temporal`` adapters.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np

from ._adapt import column_labels, shared_labels, with_labels
from ._experimental import warn_experimental
from ._temporal import _adapt_longitudinal_lingam, _adapt_varma_lingam

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from andrey.core import StructureOutput


def varma_lingam(
    data: npt.ArrayLike,
    *,
    order: tuple[int, int] = (1, 1),
    criterion: str | None = None,
    prune: bool = False,
    structural_ma: bool = False,
) -> StructureOutput:
    """Learn a temporal causal structure from one multivariate time series with VARMA-LiNGAM.

    VARMA-LiNGAM [1]_ fits a vector autoregressive moving-average model, then applies LiNGAM to
    the residuals to fix a causal order over the instantaneous (contemporaneous) effects [2]_. Use
    it when a single multivariate series carries both same-step and lagged causal influence.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        A multivariate time series, one row per time step. Coerced to ``float64``. A DataFrame's
        column names become the structure's ``labels``.
    order : tuple of int, default=(1, 1)
        The ``(p, q)`` autoregressive and moving-average lag counts.
    criterion : {"aic", "bic", "hqic"} or None, default=None
        Order-selection criterion. When set, ``(p, q)`` is chosen by that information criterion
        over ``0 <= p <= order[0]`` and ``0 <= q <= order[1]``, excluding ``(0, 0)``; ``None``
        uses ``order`` directly.
    prune : bool, default=False
        Re-estimate the lagged weights by adaptive Lasso on the causal order, which sets small
        weights to zero. With ``False`` every lagged weight is nonzero, so every lag graph links
        every pair of variables, self-loops included.
    structural_ma : bool, default=False
        How the moving-average weights are expressed: ``True`` maps them onto the independent
        noise terms, ``(I - B0) Theta_j (I - B0)^-1`` [1]_; ``False`` uses ``(I - B0) Theta_j``.
        ``B0`` is the lag-0 weight matrix and ``Theta_j`` the fitted moving-average coefficients.
        Ignored when ``prune=True``.

    Returns
    -------
    StructureOutput
        Wraps a ``temporal`` :class:`~andrey.core.TemporalStructure`.
        ``structure.lag_weights[k][i, j]`` is the autoregressive weight of edge ``i -> j`` at lag
        ``k``, lag 0 being instantaneous; ``structure.lag_weights_ma[k]`` holds the moving-average
        weights at lag ``k + 1``, in the same orientation. The lag graphs follow the
        autoregressive weights. ``ordering`` is the instantaneous causal order, and
        ``metadata["algorithm"]`` is ``"varmalingam"``.

    Raises
    ------
    ValueError
        If ``criterion`` is not one of ``"aic"``, ``"bic"``, ``"hqic"``, or ``None``, if ``order``
        is ``(0, 0)``, if ``data`` is not a 2-D numeric matrix or holds ``NaN`` / ``inf``, or if
        a DataFrame's column names repeat.

    Warns
    -----
    ExperimentalWarning
        On the first call in a process, because the method is experimental: it has no published
        benchmark, and its API may change without deprecation.

    References
    ----------
    .. [1] Kawahara, Y., Shimizu, S., and Washio, T. (2011). Analyzing relationships among ARMA
       processes based on non-Gaussianity of external influences. Neurocomputing 74(12-13),
       2212-2221.
    .. [2] Hyvarinen, A., Zhang, K., Shimizu, S., and Hoyer, P. O. (2010). Estimation of a
       structural vector autoregression model using non-Gaussianity. JMLR 11, 1709-1731.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import varma_lingam
    >>> rng = np.random.default_rng(0)
    >>> x = rng.uniform(-1, 1, 300)
    >>> y = np.r_[0.0, 0.8 * x[:-1]] + rng.uniform(-1, 1, 300)
    >>> out = varma_lingam(np.column_stack([x, y]), order=(1, 1), prune=True)
    >>> out.structure.lag(1).oriented_edges()
    [(0, 1, 'directed')]
    """
    warn_experimental("varma_lingam")
    from andrey.temporal.varma import varma_lingam as _native

    X = np.asarray(data, dtype=np.float64)
    labels = column_labels(data)
    causal_order, psis, omegas = _native(
        X, order=order, criterion=criterion, prune=prune, structural_ma=structural_ma
    )
    model = SimpleNamespace(adjacency_matrices_=(psis, omegas), causal_order_=causal_order)
    return with_labels(_adapt_varma_lingam(model), labels)


def longitudinal_lingam(
    data_list: Sequence[npt.ArrayLike], *, n_lags: int = 1, measure: str = "pwling"
) -> StructureOutput:
    """Learn a temporal causal structure from panel data with Longitudinal LiNGAM.

    Longitudinal LiNGAM [1]_ handles the same units measured at several time points: at each
    occasion it regresses out the past ``n_lags`` time points, then applies DirectLiNGAM to orient
    the instantaneous effects. Use it for repeated-measures / panel designs rather than one long
    series.

    Parameters
    ----------
    data_list : sequence of array-like, each of shape (n_samples, n_variables)
        One dataset per time point, over the same units and identical shape. Each is coerced to
        ``float64``. DataFrames must name their columns alike; the names become the structure's
        ``labels``.
    n_lags : int, default=1
        Number of past time points regressed out before the instantaneous fit.
    measure : {"pwling"}, default="pwling"
        DirectLiNGAM pairwise independence measure for the instantaneous fit.

    Returns
    -------
    StructureOutput
        Wraps a ``temporal`` :class:`~andrey.core.TemporalStructure`. The first time point has no
        past, so it is not estimated. ``structure.times`` lists the estimated time points, ``1``
        to ``T - 1``; ``structure.time_weights[t][k][i, j]`` is the weight of edge ``i -> j`` at
        lag ``k`` for the t-th of them (lag 0 is instantaneous), and ``structure.lag_weights``
        repeats the last time point's. A lag that reaches before the first time point is ``NaN``.
        Lagged weights are not pruned, so every lagged graph links every pair of variables,
        self-loops included; judge lagged effects by weight size. ``metadata["causal_orders"]``
        holds one causal order per time point, the first one empty; they need not agree, so
        ``ordering`` is ``None``. ``metadata["algorithm"]`` is ``"longitudinallingam"``.

    Raises
    ------
    NotImplementedError
        If ``measure`` is not ``"pwling"``.
    ValueError
        If ``data_list`` has fewer than two time points, if the per-time arrays are not all 2-D
        with the same shape, or if the DataFrames' column names differ or repeat.

    Warns
    -----
    ExperimentalWarning
        On the first call in a process, because the method is experimental: it has no published
        benchmark, and its API may change without deprecation.

    References
    ----------
    .. [1] Kadowaki, K., Shimizu, S., and Washio, T. (2013). Estimation of causal structures in
       longitudinal data using non-Gaussianity. IEEE International Workshop on Machine Learning
       for Signal Processing (MLSP), 1-6.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import longitudinal_lingam
    >>> rng = np.random.default_rng(0)
    >>> data_list = [rng.uniform(-1, 1, (200, 3)) for _ in range(4)]
    >>> out = longitudinal_lingam(data_list, n_lags=1)
    >>> out.structure.type
    'temporal'
    """
    warn_experimental("longitudinal_lingam")
    from andrey.temporal.longitudinal import LongitudinalLiNGAM

    panel = [np.asarray(X, dtype=np.float64) for X in data_list]
    labels = shared_labels(column_labels(X) for X in data_list)
    model = LongitudinalLiNGAM(n_lags=n_lags, measure=measure).fit(panel)
    return with_labels(_adapt_longitudinal_lingam(model), labels)
