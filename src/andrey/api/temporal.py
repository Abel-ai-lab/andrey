"""Temporal-family facades: VARMA-LiNGAM, Longitudinal LiNGAM, PCMCI, PCMCI+, LPCMCI, and Granger.

Each callable runs the native engine and returns one uniform ``StructureOutput`` wrapping a
:class:`~andrey.core.TemporalStructure` (a per-lag stack of graphs, plus the lossless coefficient
weights for the LiNGAM and Granger methods). The mapping onto the public core contract lives in
the private ``_temporal`` adapters.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np

from ._adapt import (
    column_labels,
    column_namer,
    copies_named,
    data_matrix,
    int_in_range,
    open_unit_interval,
    shared_labels,
    with_labels,
)
from ._temporal import (
    _adapt_granger,
    _adapt_longitudinal_lingam,
    _adapt_lpcmci,
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
    max_iter: int = 100,
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
        Re-estimate the weights on the causal order: adaptive Lasso sets small weights to zero and
        least squares refits the rest. With ``False`` every lagged weight is nonzero, so every lag
        graph links every pair of variables, self-loops included.
    structural_ma : bool, default=False
        How the moving-average weights are expressed: ``True`` maps them onto the independent
        noise terms, ``(I - B0) Theta_j (I - B0)^-1`` [1]_; ``False`` uses ``(I - B0) Theta_j``.
        ``B0`` is the lag-0 weight matrix and ``Theta_j`` the fitted moving-average coefficients.
        Ignored when ``prune=True``.
    max_iter : int, default=100
        Most iterations of the likelihood fit, at least ``1``. With ``criterion`` set, every fit
        in the grid gets this limit.

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
        If ``order`` is not a pair of ints, or ``max_iter`` is not an int.
    ValueError
        If ``criterion`` is not one of ``"aic"``, ``"bic"``, ``"hqic"``, or ``None``, if ``order``
        holds a negative count or is ``(0, 0)``, if ``max_iter`` is below 1, or if ``data`` has
        fewer than 2 columns.

    Notes
    -----
    DirectLiNGAM orders the innovations of the fitted model. Each pairwise regression in its
    ordering takes the least-squares slope, so every residual it scores is uncorrelated with its
    regressor in the sample.

    With ``prune=True`` each variable is regressed on its causal predecessors at the same step,
    the lagged values of every variable, and the lagged independent noise terms. Adaptive Lasso
    selects the regressors, and an ordinary least-squares fit with an intercept on the selected
    ones gives their weights, so the kept weights are not shrunk toward zero. The noise terms come
    from a lag-0 matrix refit the same way. The regressions start at time step ``max(p, q)``: the
    first steps serve only as lagged values. No lagged value wraps around from the end of the
    series, so no step is regressed on a later one.

    **Column order.** The fit runs on the columns sorted by a rank computed from each column's own
    data, as for :func:`~andrey.hc`, so reordering the columns relabels the result and changes
    nothing else, unless two columns are identical.

    **Limitations.** Measured on simulated linear series with non-Gaussian noise, one autoregressive
    and one moving-average lag, from 3 to 20 variables with 100 to 1,000 time steps, with
    ``order=(1, 1)`` and ``prune=True``. Its median errors (missing and extra cross-links) stayed
    below the empty graph's from 300 time steps at every size; at 100 time steps they exceeded it
    from 10 variables on. The fit takes minutes at 20 variables and 1,000 time steps. It assumes
    linear effects and non-Gaussian noise. The likelihood fit can stop before it converges, and
    gives no warning; a larger ``max_iter`` does not prevent this. The result then depends on the
    versions of the numerical libraries, so the same series can give a different graph in another
    environment. With ``criterion`` set, it fits one model for every ``(p, q)`` in the grid.

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
    from andrey.temporal.varma import varma_lingam as _native

    X = data_matrix(data)
    if X.shape[1] < 2:
        raise ValueError(f"data needs at least 2 columns (variables), got {X.shape[1]}")
    if not isinstance(order, (tuple, list)) or len(order) != 2:
        raise TypeError(f"order must be a (p, q) pair of ints, got {order!r}")
    order = (int_in_range(order[0], "order[0]", 0), int_in_range(order[1], "order[1]", 0))
    if order == (0, 0):
        raise ValueError("order must have p or q above 0, got (0, 0)")
    max_iter = int_in_range(max_iter, "max_iter", 1)
    labels = column_labels(data)
    causal_order, psis, omegas = _native(
        X,
        order=order,
        criterion=criterion,
        prune=prune,
        structural_ma=structural_ma,
        max_iter=max_iter,
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

    Notes
    -----
    At each time point DirectLiNGAM orders the residuals of the lag regressions. Each pairwise
    regression in its ordering takes the least-squares slope, so every residual it scores is
    uncorrelated with its regressor in the sample, and listing every unit twice leaves the causal
    orders unchanged.

    **Column order.** The fit runs on the variables sorted by a rank computed from each column's own
    data over all time points, as for :func:`~andrey.hc`, so reordering the columns relabels the
    result and changes nothing else, unless two columns are identical.

    **Limitations.** Measured on simulated panels with non-Gaussian noise, a dense same-step DAG and
    a dense lag-1 matrix drawn anew at every time point, and a hidden common cause at the first,
    from 2 to 5 variables at 20 or 50 time points with 100 to 1,000 units, with ``n_lags=1``. Its
    median errors (missing and extra cross-links) stayed far below the empty graph's at every size.
    It assumes linear effects and non-Gaussian noise. Each time point's fit uses only the units'
    rows at that time, so its accuracy depends on the number of units, not on the number of time
    points.

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
        every test keeps a degree of freedom. Also if a variable repeats another up to scale and
        shift, at the same step or up to ``2 * tau_max`` steps apart, before any test runs; one
        message names every group of copies, a lagged value as ``'x' at lag k``, and says to keep
        one column per group.

    Notes
    -----
    Every test is a partial-correlation test. The default, ``"student_t"``, takes the p-value from
    the Student-t distribution with ``n - |S| - 2`` degrees of freedom, for ``|S|`` conditions, as
    the authors' reference implementation does. ``"fisherz"`` uses Fisher's z-transform instead; its
    p-values differ slightly, so a link close to ``alpha_level`` can flip. The partial correlation
    is the correlation of the least-squares residuals of the two tested values given the
    conditions. It is computed from the correlation matrix, and from the residuals themselves when
    the conditions are collinear to working precision, as lagged copies of a strongly
    autocorrelated series can be.

    All tests use the same time steps, the last ``n_samples - 2 * tau_max``, where every lag a test
    can condition on exists: the MCI test of a link at lag ``k`` conditions on the cause's parents,
    which reach back ``k + tau_max`` steps. PC1 runs in rounds ``p = 0, 1, ...``: each round tests
    every remaining candidate given the ``p`` strongest other candidates, one conditioning set per
    candidate, then drops those that test independent. A candidate's strength is the smallest
    absolute partial correlation it has had so far. Equal strengths are ordered by each variable's
    rank, as in :func:`~andrey.hc`; the :doc:`column-order guide </docs/guides/column-order>` says
    what the rank keeps.

    **Limitations.** Measured on simulated linear-Gaussian series with an autodependency in every
    variable and one cross-link per variable at lag 1 or 2, from 10 to 160 variables with 500 to
    2,000 time steps, with ``tau_max=2``. It found nearly every true link at every size, but its
    false links grew with the number of variables: its median errors (lag-resolved SHD) stayed below
    the empty graph's at 10 variables, came close to it at 20, and exceeded it from 40 on. It
    assumes linear effects, a stationary series, and no hidden common causes. It does not test
    same-step (lag 0) links: an effect within one time step is missed and can show up as a false
    lagged link. An MCI test conditions on up to ``2 * n_variables * tau_max - 1`` lagged values, so
    the series needs many more time steps than that. The MCI step alone runs
    ``n_variables ** 2 * tau_max`` tests.

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
    from andrey.temporal.pcmci import design_column_name
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
    labels = column_labels(data)
    with copies_named(labels, design_column_name(X.shape[1], column_namer(labels))):
        result = _native(
            X, tau_max=tau_max, pc_alpha=pc_alpha, alpha_level=alpha_level, indep_test=indep_test
        )
    return with_labels(_adapt_pcmci(result), labels)


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
        every test keeps a degree of freedom. Also if a variable repeats another up to scale and
        shift, at the same step or up to ``2 * tau_max`` steps apart, before any test runs; one
        message names every group of copies, a lagged value as ``'x' at lag k``, and says to keep
        one column per group.

    Notes
    -----
    The method runs in four phases. The lagged phase selects the lagged parents of each variable
    as PCMCI's PC1 step does, with equal strengths ordered by each variable's rank. The skeleton
    phase starts from those lagged links and from a link between every two variables at the same
    step. Round ``p = 0, 1, ...`` tests each linked pair given each subset of ``p`` same-step
    neighbors of the later end, with the lagged parents of both ends added, and removes the link
    at the first p-value above ``pc_alpha``. A same-step pair is tested from both ends, first
    given the neighbors of its higher-numbered variable, and the neighbor sets stay fixed during a
    round, so the links found do not depend on the order of the variables. The collider phase
    orients the unshielded triples by ``collider_rule``, and the rule phase applies Meek's rules
    R1 to R3 to the same-step links until none applies. Each collider or rule pass collects its
    orientations before applying them, and marks a link that gets both orientations as a
    conflict. With the ``"majority"`` or ``"conservative"`` rule the orientations do not depend on
    the order of the variables either; ``"none"`` reads the one separating set the skeleton phase
    stored, which can change with the order.

    All tests use the same time steps, the last ``n_samples - 2 * tau_max``. The default test,
    ``"student_t"``, takes the p-value from the Student-t distribution with ``n - |S| - 2``
    degrees of freedom, for ``|S|`` conditions. The partial correlation is the correlation of the
    least-squares residuals of the two tested values given the conditions. It is computed from the
    correlation matrix, and from the residuals themselves when the conditions are collinear to
    working precision, as lagged copies of a strongly autocorrelated series can be.

    **Limitations.** Measured on simulated linear-Gaussian series with strong autocorrelation and
    1.5 cross-links per variable, 30% of them within a time step and the rest at lags 1 to 5, from
    10 to 80 variables with 500 to 2,000 time steps, with ``tau_max=5``. Its median errors (endpoint
    SHD) stayed well below the empty graph's at every size. It assumes linear effects, a stationary
    series, and no hidden common causes. The skeleton and collider phases test every subset of a
    variable's same-step neighbors, so the number of tests grows exponentially with the number of
    same-step links at one variable. A test conditions on up to
    ``2 * n_variables * tau_max + n_variables - 2`` values, so the series needs many more time steps
    than that.

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
    from andrey.temporal.pcmci import design_column_name
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
    labels = column_labels(data)
    with copies_named(labels, design_column_name(X.shape[1], column_namer(labels))):
        result = _native(
            X,
            tau_max=tau_max,
            pc_alpha=pc_alpha,
            collider_rule=collider_rule,
            indep_test=indep_test,
        )
    return with_labels(_adapt_pcmciplus(result), labels)


def lpcmci(
    data: npt.ArrayLike,
    *,
    tau_max: int = 1,
    pc_alpha: float = 0.05,
    n_preliminary_iterations: int = 1,
    indep_test: str = "student_t",
) -> StructureOutput:
    """Learn a time-series partial ancestral graph that allows hidden common causes with LPCMCI.

    LPCMCI [1]_ finds the lagged and same-step links of one multivariate time series when some
    variables that drive several observed ones are not measured. Like FCI it returns a partial
    ancestral graph: an arrowhead says an end is not a cause of the other, a tail that it is, and a
    circle leaves it open, so a link with arrowheads at both ends is a hidden common cause. It
    orients links while it removes them and conditions every test on the parents found so far,
    which keeps true links in strongly autocorrelated series. Use it on one series when an
    unmeasured variable may confound the measured ones.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        A multivariate time series, one row per time step. Coerced to ``float64``. A DataFrame's
        column names become the structure's ``labels``.
    tau_max : int, default=1
        The largest lag, at least ``1``: the graph covers the time steps ``t - tau_max`` to ``t``,
        and links at lags ``0`` to ``tau_max`` are tested.
    pc_alpha : float, default=0.05
        Significance level of every test, in ``(0, 1)``: a link is removed when a test gives it a
        p-value above ``pc_alpha``.
    n_preliminary_iterations : int, default=1
        How many preliminary rounds run before the final one, at least ``0``. Each round learns
        parents that the next one conditions on from its first test.
    indep_test : {"student_t", "fisherz"}, default="student_t"
        The partial-correlation test behind every p-value. Both report the same partial
        correlation; ``"student_t"`` takes its p-value from the Student-t distribution and
        ``"fisherz"`` from Fisher's z-transform.

    Returns
    -------
    StructureOutput
        Wraps a ``temporal`` :class:`~andrey.core.TemporalStructure` with lags ``0`` to
        ``tau_max``. ``structure.lag(0)`` is a ``pag`` of the same-step links: ``i -> j``,
        ``i o-> j``, ``i <-> j`` (a hidden common cause), and ``i o-o j``. ``structure.lag(k)``,
        ``k >= 1``, is a ``digraph`` of the lagged links whose earlier end is a tail, the definite
        causes: its edge ``i -> j`` says variable ``i`` at time ``t - k`` causes variable ``j`` at
        time ``t``, and a self-loop is a variable's effect on its own future. The structure has no
        ``lag_weights``. ``metadata["links"][k][i][j]`` holds every link between variable ``i``
        at ``t - k`` and variable ``j`` at ``t`` as three characters, ``""`` when there is none:
        the mark at ``i`` (``-`` tail, ``<`` arrowhead, ``o`` circle, ``x`` conflict), ``-``,
        and the mark at ``j`` (``>``, ``-``, ``o``, or ``x``), such as ``"o->"``, ``"<->"``, or
        ``"-->"``; lag ``0`` lists both orders. ``metadata["conflicts"]`` lists the ``[i, k, j]``
        links with a conflict mark, which ``structure.lag(0)`` shows as a circle and
        ``structure.lag(k)`` leaves out. ``metadata["p_values"][k][i][j]`` is the largest p-value
        of the final round's tests of that pair, and ``metadata["statistics"][k][i][j]`` the
        partial correlation of that test; lag ``0`` is symmetric with ``1`` and ``0`` on the
        diagonal. ``metadata["parents"][j]`` lists the ``[i, k]`` pairs (variable ``i`` at lag
        ``k``) the last preliminary round found to be parents of variable ``j``. ``ordering`` is
        ``None``, and ``metadata["algorithm"]`` is ``"lpcmci"``.

    Raises
    ------
    TypeError
        If ``tau_max`` or ``n_preliminary_iterations`` is not an int.
    ValueError
        If ``tau_max`` is below 1, ``n_preliminary_iterations`` below 0, ``pc_alpha`` not in
        ``(0, 1)``, ``indep_test`` neither ``"student_t"`` nor ``"fisherz"``, or if ``data`` is
        not a usable 2-D matrix with at least ``n_variables * (tau_max + 1) + 2 * tau_max + 1``
        rows, the length at which every test keeps a degree of freedom. Also if a variable repeats
        another up to scale and shift, at the same step or up to ``2 * tau_max`` steps apart,
        before any test runs; one message names every group of copies, a lagged value as
        ``'x' at lag k``, and says to keep one column per group.

    Notes
    -----
    The graph starts complete, with every lagged link ``o->`` (time order rules out the later end
    as a cause of the earlier one) and every same-step link ``o-o``. A round removes links in
    steps ``p = 0, 1, ...``: it tests a link given ``p`` of the neighbors of its later end that are
    not known non-causes of it, plus the parents of both ends, and tests the autodependency links
    first, then the cross links lag by lag. A same-step link is also tested from its other end.
    After each step that removes a link, orientation rules mark new tails and arrowheads on the
    lagged links and the round restarts at ``p = 0``, as in the paper, so that the separating sets
    found later stay weakly minimal. In these passes the rules R9' and R10' orient lagged links
    only, but their paths may run through same-step links, directed or not, as the paper's rules
    allow. Once nothing is left to test, the full set of rules runs, including the collider
    rule, which first tests the two links of a triple again given each stored separating set of
    the outer pair plus the parents of their ends. A preliminary round keeps only the parents it
    found and starts over from the complete graph. The final round also tests the links that may
    join two variables neither of which causes the other, given subsets of a wider neighborhood.
    Whether a node lies outside the separating set of two others is a majority vote over the
    distinct separating sets stored and found among the neighbors of the two. A mark that rules
    orient both ways is a conflict. The candidates of a step are ordered by statistics frozen when
    its group of links starts, so the order in which a group's links are tested changes nothing.
    Equal statistics are ordered by each variable's rank, as in :func:`~andrey.hc`; the
    :doc:`column-order guide </docs/guides/column-order>` says what the rank keeps. The collider
    rule stores every separating set its tests find, each made weakly minimal by dropping first the
    node whose removal leaves the largest p-value, so the stored sets do not depend on the order of
    the variables.

    All tests use the same time steps, the last ``n_samples - 2 * tau_max``, and every node a test
    conditions on lies within ``tau_max`` steps of the later end. A test's partial correlation is
    computed from the correlation matrix, and from the least-squares residuals when the conditions
    are collinear to working precision. The defaults are the authors' published defaults. The
    earlier end of a lagged link is searched with the empty set only, since time order already
    rules out the later end as its cause.

    **Limitations.** Measured on simulated linear-Gaussian series with strong autocorrelation, one
    cross-link per variable, 30% of them within a time step and the rest at lags 1 to 3, and 30% of
    the series unobserved, from 5 to 20 variables with 500 to 2,000 time steps, with ``tau_max=3``.
    Its median errors (endpoint SHD against the true PAG of the observed series) stayed below the
    empty graph's at every size. It assumes linear effects, a stationary series, and no selection
    bias; hidden common causes are allowed. The graph covers ``tau_max + 1`` time steps, and a
    hidden cause that acts through earlier steps can add links within that window. The removal and
    rule steps test subsets of a node's neighbors, so the number of tests grows exponentially with
    the number of links at one node. A test conditions on up to ``n_variables * (tau_max + 1) - 2``
    values, so the series needs many more time steps than that.

    References
    ----------
    .. [1] Gerhardus, A., and Runge, J. (2020). High-recall causal discovery for autocorrelated
       time series with latent confounders. Advances in Neural Information Processing Systems 33
       (NeurIPS 2020), 12615-12625.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import lpcmci
    >>> rng = np.random.default_rng(0)
    >>> hidden = rng.standard_normal(1000)
    >>> x, y = np.zeros(1000), np.zeros(1000)
    >>> for t in range(1, 1000):
    ...     x[t] = 0.5 * x[t - 1] + 0.7 * hidden[t] + rng.standard_normal()
    ...     y[t] = 0.6 * y[t - 1] + 0.6 * hidden[t] + rng.standard_normal()
    >>> out = lpcmci(np.column_stack([x, y]), tau_max=2)
    >>> out.metadata["links"][0][0][1]
    '<->'
    >>> out.structure.lag(1).oriented_edges()
    [(0, 0, 'directed'), (1, 1, 'directed')]
    """
    from andrey.temporal.lpcmci import lpcmci as _native
    from andrey.temporal.pcmci import design_column_name

    tau_max = int_in_range(tau_max, "tau_max", 1)
    n_preliminary_iterations = int_in_range(n_preliminary_iterations, "n_preliminary_iterations", 0)
    pc_alpha = open_unit_interval(pc_alpha, "pc_alpha")
    X = data_matrix(data)
    # A test conditions on up to n * (tau_max + 1) - 2 nodes of the window, over the last
    # n_samples - 2 * tau_max steps; this keeps a degree of freedom in every test.
    min_rows = X.shape[1] * (tau_max + 1) + 2 * tau_max + 1
    if X.shape[0] < min_rows:
        raise ValueError(
            f"data needs at least {min_rows} rows (time steps) for tau_max={tau_max} over "
            f"{X.shape[1]} variables, got {X.shape[0]}"
        )
    labels = column_labels(data)
    with copies_named(labels, design_column_name(X.shape[1], column_namer(labels))):
        result = _native(
            X,
            tau_max=tau_max,
            pc_alpha=pc_alpha,
            n_preliminary_iterations=n_preliminary_iterations,
            indep_test=indep_test,
        )
    return with_labels(_adapt_lpcmci(result), labels)


def granger(data: npt.ArrayLike, *, maxlag: int = 2, cv: int = 5) -> StructureOutput:
    """Learn the lagged causal graph of a multivariate time series by lasso Granger causality.

    A variable Granger-causes another when its past improves the prediction of the other's
    present [1]_. Each variable is regressed on the past ``maxlag`` steps of every variable by the
    lasso [2]_, with the penalty chosen by cross-validation; a nonzero coefficient is a lagged
    edge. Use it for one multivariate series whose effects take at least one time step.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        A multivariate time series, one row per time step. Coerced to ``float64``. A DataFrame's
        column names become the structure's ``labels``.
    maxlag : int, default=2
        Number of past time steps each regression uses, at least 1.
    cv : int, default=5
        Number of cross-validation folds that choose each regression's penalty, at least 2.

    Returns
    -------
    StructureOutput
        Wraps a ``temporal`` :class:`~andrey.core.TemporalStructure` with lags ``1`` to ``maxlag``
        and no lag 0. ``structure.lag(k)`` is a ``digraph`` with the edge ``i -> j`` when
        ``x_i(t - k)`` has a nonzero coefficient in the regression of ``x_j(t)``, self-loops
        included; ``structure.lag_weights[k - 1][i, j]`` is that coefficient, in the data's units.
        ``ordering`` is ``None``, and ``metadata["algorithm"]`` is ``"granger"``.

    Raises
    ------
    TypeError
        If ``maxlag`` or ``cv`` is not an int.
    ValueError
        If ``maxlag`` is less than 1, if ``cv`` is less than 2, if ``data`` has fewer than
        ``maxlag + cv`` rows, so that some fold would hold no time step, or if ``data`` has fewer
        than 2 columns.

    Notes
    -----
    An edge says that the past of one variable helps predict another beyond the past of every
    variable in ``data``. Granger causality misses effects within one time step, and a common
    cause left out of ``data`` can produce edges.

    Each regression is fitted on standardized variables, so the edges do not depend on the units of
    any variable; the coefficients are converted back to the data's units. Cross-validation chooses
    the penalty that predicts best, which often keeps small coefficients on lags with no effect:
    judge the edges by the size of their weights. The folds are consecutive blocks of time steps,
    and the fit is deterministic. With ``andrey.config(num_workers=k)``, the regressions run on
    ``k`` threads and return the same coefficients.

    The penalty is the plain lasso of Arnold et al. [2]_. Shojaie and Michailidis [3]_ propose a
    truncating lasso penalty that also selects the lag order; it is not implemented here.

    **Limitations.** Measured on simulated linear-Gaussian VAR(2) series with no same-step links,
    standardized, from 10 to 160 variables with 1,000 time steps, with ``maxlag=2``. It found most
    true lagged edges but returned many false ones, so its median errors (missing and extra lagged
    edges) exceeded the empty graph's at every size: on half the series at 10 variables and on all
    of them from 20 on. It assumes linear effects and tests no same-step effects. Each variable's
    regression takes ``maxlag`` lags of every variable as inputs, so the work grows with ``maxlag``
    and with the square of the number of variables.

    References
    ----------
    .. [1] Granger, C. W. J. (1969). Investigating causal relations by econometric models and
       cross-spectral methods. Econometrica 37(3), 424-438.
    .. [2] Arnold, A., Liu, Y., and Abe, N. (2007). Temporal causal modeling with graphical Granger
       methods. KDD, 66-75.
    .. [3] Shojaie, A., and Michailidis, G. (2010). Discovering graphical Granger causality using
       the truncating lasso penalty. Bioinformatics 26(18), i517-i523.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import granger
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal(500)
    >>> y = np.r_[0.0, 0.8 * x[:-1]] + rng.standard_normal(500)
    >>> out = granger(np.column_stack([x, y]), maxlag=1)
    >>> round(float(out.structure.lag_weights[0][0, 1]), 1)
    0.8
    """
    from andrey.temporal.granger import granger_lasso

    maxlag = int_in_range(maxlag, "maxlag", 1)
    cv = int_in_range(cv, "cv", 2)
    X = data_matrix(data, min_rows=maxlag + cv)  # LassoCV needs a time step in every fold
    if X.shape[1] < 2:
        raise ValueError(f"data needs at least 2 columns (variables), got {X.shape[1]}")
    peak = np.abs(X).max(axis=0)  # divide first, so tiny or huge units cannot underflow the std
    spread = (X / peak).std(axis=0)  # data_matrix rejects a constant column
    coeff = granger_lasso((X / peak - (X / peak).mean(axis=0)) / spread, maxlag=maxlag, cv=cv)
    scale = spread * peak
    # coeff[i, j + d*k] weights standardized x_j(t-k-1) in standardized x_i(t): rescale to units.
    # Units far apart can overflow the ratio; a zero weight stays zero rather than becoming NaN.
    with np.errstate(over="ignore", invalid="ignore"):
        ratio = scale[:, None] / np.tile(scale, maxlag)[None, :]
        coeff = np.where(coeff == 0, 0.0, coeff * ratio)
    return with_labels(_adapt_granger(coeff, X.shape[1], maxlag), column_labels(data))
