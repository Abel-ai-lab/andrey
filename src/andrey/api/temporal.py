"""Temporal-family facades: VARMA-LiNGAM, Longitudinal LiNGAM, PCMCI, and PCMCI+.

Each callable runs the native engine and returns one uniform ``StructureOutput`` wrapping a
:class:`~andrey.core.TemporalStructure` (a per-lag stack of graphs, plus the lossless coefficient
weights for the LiNGAM methods). The mapping onto the public core contract lives in the private
``_temporal`` adapters.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

from ._adapt import (
    column_labels,
    data_matrix,
    int_in_range,
    open_unit_interval,
    shared_labels,
    with_labels,
)
from ._experimental import warn_experimental
from ._temporal import (
    _adapt_longitudinal_lingam,
    _adapt_pcmci,
    _adapt_pcmciplus,
    _adapt_varma_lingam,
)

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
        The ``(p, q)`` autoregressive and moving-average lag counts, each at least ``0`` and not
        both ``0``.
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
    TypeError
        If ``order`` is not a pair of ints.
    ValueError
        If ``criterion`` is not one of ``"aic"``, ``"bic"``, ``"hqic"``, or ``None``, if ``order``
        holds a negative count or is ``(0, 0)``, or if ``data`` has fewer than 2 columns.

    Warns
    -----
    ExperimentalWarning
        On the first call in a process, because the method is experimental: it has no published
        benchmark, and its API may change without deprecation.

    Notes
    -----
    **Limitations.** Not benchmarked yet. Its tests run it on one series of 500 time steps over 3
    variables. It assumes linear effects and non-Gaussian noise. The likelihood fit stops after 100
    iterations and gives no warning when it has not converged; with ``criterion`` set, it fits one
    model for every ``(p, q)`` in the grid.

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

    X = data_matrix(data)
    if X.shape[1] < 2:
        raise ValueError(f"data needs at least 2 columns (variables), got {X.shape[1]}")
    if not isinstance(order, (tuple, list)) or len(order) != 2:
        raise TypeError(f"order must be a (p, q) pair of ints, got {order!r}")
    order = (int_in_range(order[0], "order[0]", 0), int_in_range(order[1], "order[1]", 0))
    if order == (0, 0):
        raise ValueError("order must have p or q above 0, got (0, 0)")
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
        Number of past time points regressed out before the instantaneous fit, from ``1`` to the
        number of time points minus one.
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
    TypeError
        If ``n_lags`` is not an int.
    ValueError
        If ``data_list`` has fewer than two time points, if ``n_lags`` is outside its range, if
        the per-time arrays differ in shape, or if the DataFrames' column names differ.

    Warns
    -----
    ExperimentalWarning
        On the first call in a process, because the method is experimental: it has no published
        benchmark, and its API may change without deprecation.

    Notes
    -----
    **Limitations.** Not benchmarked yet. Its tests run it on 4 time points of 600 units and 4
    variables. It assumes linear effects and non-Gaussian noise. Each time point's fit uses only the
    units' rows at that time, so its accuracy depends on the number of units, not on the number of
    time points.

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

    panel = [data_matrix(X, name=f"data_list[{t}]") for t, X in enumerate(data_list)]
    if len(panel) < 2:
        raise ValueError(f"data_list must hold at least two time points, got {len(panel)}")
    n_lags = int_in_range(n_lags, "n_lags", 1, len(panel) - 1)
    labels = shared_labels(column_labels(X) for X in data_list)
    model = LongitudinalLiNGAM(n_lags=n_lags, measure=measure).fit(panel)
    return with_labels(_adapt_longitudinal_lingam(model), labels)


def pcmci(
    data: npt.ArrayLike,
    *,
    tau_max: int = 1,
    pc_alpha: float = 0.2,
    alpha_level: float = 0.05,
    indep_test: str = "student_t",
) -> StructureOutput:
    """Learn the lagged causal links of one multivariate time series with PCMCI.

    PCMCI [1]_ works in two steps. The PC1 step selects, for each variable, a few candidate parents
    among the lagged values of all variables. The momentary conditional independence (MCI) step
    then tests each lagged link given the selected parents of both its ends. Conditioning on the
    cause's own past removes the autocorrelation that makes a plain lagged-correlation test report
    false links. Use it on one series whose causal effects take at least one time step.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        A multivariate time series, one row per time step. Coerced to ``float64``. A DataFrame's
        column names become the structure's ``labels``.
    tau_max : int, default=1
        The largest lag tested, at least ``1``: links at lags ``1`` to ``tau_max`` are tested.
    pc_alpha : float, default=0.2
        Significance level of the PC1 step, in ``(0, 1)``: a candidate parent is dropped when a test
        gives it a p-value above ``pc_alpha``. A larger value keeps more parents as conditions for
        the MCI step.
    alpha_level : float, default=0.05
        Significance level of the MCI step, in ``(0, 1)``: a link is in the graph when its p-value
        is at most ``alpha_level``.
    indep_test : {"student_t", "fisherz"}, default="student_t"
        The partial-correlation test behind every p-value. Both report the same partial
        correlation; ``"student_t"`` takes its p-value from the Student-t distribution and
        ``"fisherz"`` from Fisher's z-transform.

    Returns
    -------
    StructureOutput
        Wraps a ``temporal`` :class:`~andrey.core.TemporalStructure` with lags ``1`` to
        ``tau_max`` and no lag 0. ``structure.lag(k)`` is a ``digraph`` whose edge ``i -> j`` is
        the link from variable ``i`` at time ``t - k`` to variable ``j`` at time ``t``; a self-loop
        is a variable's dependence on its own past. The structure has no ``lag_weights``.
        ``metadata["p_values"][k - 1][i][j]`` and ``metadata["statistics"][k - 1][i][j]`` hold the
        MCI p-value and partial correlation of that link, for every pair, significant or not.
        ``metadata["parents"][j]`` lists the ``[i, k]`` pairs (variable ``i`` at lag ``k``) that
        PC1 selected for variable ``j``, strongest first. ``ordering`` is ``None``, and
        ``metadata["algorithm"]`` is ``"pcmci"``.

    Raises
    ------
    TypeError
        If ``tau_max`` is not an int.
    ValueError
        If ``tau_max`` is below 1, if ``pc_alpha`` or ``alpha_level`` is not in ``(0, 1)``, if
        ``indep_test`` is neither ``"student_t"`` nor ``"fisherz"``, or if ``data`` is not a usable
        2-D matrix with at least ``2 * tau_max * (n_variables + 1) + 3`` rows, the length at which
        every test keeps a degree of freedom.

    Warns
    -----
    ExperimentalWarning
        On the first call in a process, because the method is experimental: it has no published
        benchmark, and its API may change without deprecation.

    Notes
    -----
    Every test is a partial-correlation test. The default, ``"student_t"``, takes the p-value from
    the Student-t distribution with ``n - |S| - 2`` degrees of freedom, for ``|S|`` conditions, as
    the authors' reference implementation does. ``"fisherz"`` uses Fisher's z-transform instead; its
    p-values differ slightly, so a link close to ``alpha_level`` can flip.

    All tests use the same time steps, the last ``n_samples - 2 * tau_max``, where every lag a test
    can condition on exists: the MCI test of a link at lag ``k`` conditions on the cause's parents,
    which reach back ``k + tau_max`` steps. PC1 runs in rounds ``p = 0, 1, ...``: each round tests
    every remaining candidate given the ``p`` strongest other candidates, one conditioning set per
    candidate, then drops those that test independent. A candidate's strength is the smallest
    absolute partial correlation it has had so far.

    **Limitations.** Not benchmarked yet. Its tests run it on one series of 1,000 time steps over 5
    variables with ``tau_max=2``. It assumes linear effects, a stationary series, and no hidden
    common causes. It does not test same-step (lag 0) links: an effect within one time step is
    missed and can show up as a false lagged link. An MCI test conditions on up to
    ``2 * n_variables * tau_max - 1`` lagged values, so the series needs many more time steps than
    that. The MCI step alone runs ``n_variables ** 2 * tau_max`` tests.

    References
    ----------
    .. [1] Runge, J., Nowack, P., Kretschmer, M., Flaxman, S., and Sejdinovic, D. (2019). Detecting
       and quantifying causal associations in large nonlinear time series datasets. Science
       Advances 5(11), eaau4996.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import pcmci
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal(500)
    >>> y = np.r_[0.0, 0.8 * x[:-1]] + rng.standard_normal(500)
    >>> out = pcmci(np.column_stack([x, y]), tau_max=2, alpha_level=0.01)
    >>> out.structure.lag(1).oriented_edges()
    [(0, 1, 'directed')]
    >>> out.structure.lag(2).oriented_edges()
    []
    """
    warn_experimental("pcmci")
    from andrey.temporal.pcmci import pcmci as _native

    tau_max = int_in_range(tau_max, "tau_max", 1)
    pc_alpha = open_unit_interval(pc_alpha, "pc_alpha")
    alpha_level = open_unit_interval(alpha_level, "alpha_level")
    X = data_matrix(data)
    # An MCI test conditions on up to 2 * n * tau_max - 1 lagged columns of the last
    # n_samples - 2 * tau_max steps; this keeps a degree of freedom in every test.
    min_rows = 2 * tau_max * (X.shape[1] + 1) + 3
    if X.shape[0] < min_rows:
        raise ValueError(
            f"data needs at least {min_rows} rows (time steps) for tau_max={tau_max} over "
            f"{X.shape[1]} variables, got {X.shape[0]}"
        )
    result = _native(
        X, tau_max=tau_max, pc_alpha=pc_alpha, alpha_level=alpha_level, indep_test=indep_test
    )
    return with_labels(_adapt_pcmci(result), column_labels(data))


def pcmciplus(
    data: npt.ArrayLike,
    *,
    tau_max: int = 1,
    pc_alpha: float = 0.01,
    collider_rule: str = "majority",
    indep_test: str = "student_t",
) -> StructureOutput:
    """Learn the lagged and same-step causal links of one multivariate time series with PCMCI+.

    PCMCI+ [1]_ extends PCMCI to links within one time step. It first selects, for each variable, a
    few candidate parents among the lagged values of all variables. It then tests every lagged
    candidate and every pair of variables at the same step, given subsets of the same-step
    neighbors plus the selected lagged parents of both ends, and orients the same-step links with
    the collider and orientation rules of the PC algorithm. Lagged links always point forward in
    time. Use it on one series whose causal effects can act within a time step.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        A multivariate time series, one row per time step. Coerced to ``float64``. A DataFrame's
        column names become the structure's ``labels``.
    tau_max : int, default=1
        The largest lag tested, at least ``1``: links at lags ``0`` to ``tau_max`` are tested.
    pc_alpha : float, default=0.01
        Significance level of every test, in ``(0, 1)``: a link is removed when a test gives it a
        p-value above ``pc_alpha``.
    collider_rule : {"majority", "conservative", "none"}, default="majority"
        How an unshielded triple ``a - b - c`` with a same-step link ``b - c`` is judged a
        collider ``a -> b <- c``. ``"majority"`` and ``"conservative"`` test ``a`` and ``c`` again
        given every subset of their same-step neighbors and look at the subsets that separate
        them: ``"majority"`` calls the triple a collider when fewer than half hold ``b``, and
        ``"conservative"`` when none does; a triple neither rule can decide is ambiguous and
        orients nothing. ``"none"`` calls it a collider when ``b`` is not in the one separating
        set the skeleton search found.
    indep_test : {"student_t", "fisherz"}, default="student_t"
        The partial-correlation test behind every p-value. Both report the same partial
        correlation; ``"student_t"`` takes its p-value from the Student-t distribution and
        ``"fisherz"`` from Fisher's z-transform.

    Returns
    -------
    StructureOutput
        Wraps a ``temporal`` :class:`~andrey.core.TemporalStructure` with lags ``0`` to
        ``tau_max``. ``structure.lag(0)`` is a ``cpdag`` of the same-step links: ``i -> j`` is
        oriented, and ``i -- j`` is a link whose direction the data cannot tell or whose
        orientations conflict; ``metadata["conflicts"]`` lists the latter as ``[i, j]`` pairs,
        ``i < j``. ``structure.lag(k)``, ``k >= 1``, is a ``digraph`` whose edge ``i -> j`` is
        the link from variable ``i`` at time ``t - k`` to variable ``j`` at time ``t``; a
        self-loop is a variable's dependence on its own past. The structure has no
        ``lag_weights``. ``metadata["p_values"][k][i][j]`` is the largest p-value among the
        skeleton-phase tests of the link between variable ``i`` at ``t - k`` and variable ``j``
        at ``t`` (the first step's tests for a lagged link that step removed), and
        ``metadata["statistics"][k][i][j]`` the partial correlation of that test; lag ``0`` is
        symmetric with ``1`` and ``0`` on the diagonal. ``metadata["parents"][j]`` lists the
        ``[i, k]`` pairs (variable ``i`` at lag ``k``) the first step selected for variable
        ``j``, strongest first, and ``metadata["ambiguous_triples"]`` the ``[i, k, m, j]``
        triples (variable ``i`` at lag ``k``, then ``m`` and ``j`` at lag ``0``) the collider rule
        left undecided. ``ordering`` is ``None``, and ``metadata["algorithm"]`` is
        ``"pcmciplus"``.

    Raises
    ------
    TypeError
        If ``tau_max`` is not an int.
    ValueError
        If ``tau_max`` is below 1, if ``pc_alpha`` is not in ``(0, 1)``, if ``collider_rule`` or
        ``indep_test`` is not one of its choices, or if ``data`` is not a usable 2-D matrix with
        at least ``2 * tau_max * (n_variables + 1) + n_variables + 1`` rows, the length at which
        every test keeps a degree of freedom.

    Warns
    -----
    ExperimentalWarning
        On the first call in a process, because the method is experimental: it has no published
        benchmark, and its API may change without deprecation.

    Notes
    -----
    The method runs in four phases. The lagged phase selects the lagged parents of each variable
    as PCMCI's PC1 step does. The skeleton phase starts from those lagged links and from a link
    between every two variables at the same step. Round ``p = 0, 1, ...`` tests each linked pair
    given each subset of ``p`` same-step neighbors of the later end, with the lagged parents of
    both ends added, and removes the link at the first p-value above ``pc_alpha``. A same-step
    pair is tested from both ends, and the neighbor sets stay fixed during a round, so the links
    found do not depend on the order of the variables. The collider phase orients the
    unshielded triples by ``collider_rule``, and the rule phase applies Meek's rules R1 to R3 to
    the same-step links until none applies. Each collider or rule pass collects its orientations
    before applying them, and marks a link that gets both orientations as a conflict. With the
    ``"majority"`` or ``"conservative"`` rule the orientations do not depend on the order of the
    variables either; ``"none"`` reads the one separating set the skeleton phase stored, which
    can change with the order.

    All tests use the same time steps, the last ``n_samples - 2 * tau_max``. The default test,
    ``"student_t"``, takes the p-value from the Student-t distribution with ``n - |S| - 2``
    degrees of freedom, for ``|S|`` conditions.

    **Limitations.** Not benchmarked yet. Its tests run it on one series of 1,000 time steps over
    5 variables with ``tau_max=2``. It assumes linear effects, a stationary series, and no hidden
    common causes. The skeleton and collider phases test every subset of a variable's same-step
    neighbors, so the number of tests grows exponentially with the number of same-step links at
    one variable. A test conditions on up to ``2 * n_variables * tau_max + n_variables - 2``
    values, so the series needs many more time steps than that.

    References
    ----------
    .. [1] Runge, J. (2020). Discovering contemporaneous and lagged causal relations in
       autocorrelated nonlinear time series datasets. Proceedings of the 36th Conference on
       Uncertainty in Artificial Intelligence (UAI), PMLR 124, 1388-1397.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import pcmciplus
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal(500)
    >>> y = 0.8 * x + rng.standard_normal(500)
    >>> z = np.r_[0.0, 0.8 * y[:-1]] + rng.standard_normal(500)
    >>> out = pcmciplus(np.column_stack([x, y, z]))
    >>> out.structure.lag(0).oriented_edges()
    [(0, 1, 'undirected')]
    >>> out.structure.lag(1).oriented_edges()
    [(1, 2, 'directed')]
    """
    warn_experimental("pcmciplus")
    from andrey.temporal.pcmciplus import pcmciplus as _native

    tau_max = int_in_range(tau_max, "tau_max", 1)
    pc_alpha = open_unit_interval(pc_alpha, "pc_alpha")
    X = data_matrix(data)
    # A test conditions on up to 2 * n * tau_max + n - 2 columns of the last
    # n_samples - 2 * tau_max steps; this keeps a degree of freedom in every test.
    min_rows = 2 * tau_max * (X.shape[1] + 1) + X.shape[1] + 1
    if X.shape[0] < min_rows:
        raise ValueError(
            f"data needs at least {min_rows} rows (time steps) for tau_max={tau_max} over "
            f"{X.shape[1]} variables, got {X.shape[0]}"
        )
    result = _native(
        X,
        tau_max=tau_max,
        pc_alpha=pc_alpha,
        collider_rule=collider_rule,
        indep_test=indep_test,
    )
    return with_labels(_adapt_pcmciplus(result), column_labels(data))
