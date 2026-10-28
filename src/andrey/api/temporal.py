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

from andrey.api._method import method_spec
from andrey.core.ci_missing import rows_metadata, rows_tested
from andrey.core.knowledge import Knowledge, resolve_lagged

from ._adapt import (
    DROP_UNITS,
    FILL_STEPS,
    column_labels,
    column_namer,
    copies_named,
    data_matrix,
    int_in_range,
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


@method_spec("varma_lingam")
def varma_lingam(
    data: npt.ArrayLike,
    *,
    order: tuple[int, int] = (1, 1),
    criterion: str | None = None,
    prune: bool = False,
    structural_ma: bool = False,
    max_iter: int = 100,
    measure: str = "pwling",
    knowledge: Knowledge | None = None,
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
        Most iterations of the likelihood fit, at least ``1``. The optimizer's function evaluations
        are limited to 150 per iteration; each iteration spends about one per parameter, so a fit
        of more than about 150 parameters, such as ``order=(1, 1)`` on 8 variables, can use them up
        first. With ``criterion`` set, every fit in the grid gets these limits. A fit that reaches
        either without converging warns with :class:`~andrey.SearchLimitWarning`; raise
        ``max_iter`` to let it run on.
    measure : {"pwling"}, default="pwling"
        How the instantaneous order search, DirectLiNGAM on the innovations, picks the most
        exogenous variable; ``"pwling"`` is DirectLiNGAM's pairwise likelihood-ratio measure.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`, read as links between lagged
        variables over the autoregressive lags ``0`` to ``order[0]``: required and forbidden
        links, the pairs allowed to be linked, and temporal tiers, all honored. A link
        ``(cause, effect, lag)`` is the link from ``cause`` at time ``t - lag`` to ``effect`` at
        time ``t``, lag ``0`` being the same step; a forbidden ``(cause, effect)`` pair forbids it
        at every lag, and a required link names its lag. At lag ``0``, DirectLiNGAM on the
        innovations honors the knowledge as :func:`~andrey.direct_lingam` does: the causal order
        puts the cause of each required link, and each earlier tier, first, and the regression
        drops forbidden parents and keeps required ones unpenalized. With ``prune=True``, each
        variable's regression also drops its forbidden lagged regressors and keeps its required
        ones unpenalized. With ``prune=False`` every lagged weight stays, so a required lagged
        link is in the graph, and a forbidden lagged link raises. Tiers forbid every link from a
        later tier into an earlier one, and a pair left out of ``allowed`` has no link at any
        lag; a variable's links to its own past stay allowed. The moving-average weights are not
        constrained.

    Returns
    -------
    StructureOutput
        Wraps a ``temporal`` :class:`~andrey.core.TemporalStructure`.
        ``structure.lag_weights[k][i, j]`` is the autoregressive weight of edge ``i -> j`` at lag
        ``k``, lag 0 being instantaneous; ``structure.lag_weights_ma[k]`` holds the moving-average
        weights at lag ``k + 1``, in the same orientation. The lag graphs follow the
        autoregressive weights. ``ordering`` is the instantaneous causal order, and
        ``metadata["algorithm"]`` is ``"varma_lingam"``.

    Raises
    ------
    TypeError
        If ``knowledge`` is neither ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If an argument has the wrong type, lies outside its range, or is not one of its choices; the
        message names the method, the argument, what it accepts, and the value. Also if ``order`` is
        ``(0, 0)``, or if ``data`` has fewer than 2 columns. Also if ``knowledge`` names a column
        ``data`` lacks, contradicts itself, requires a link without a lag or beyond ``order[0]``,
        forbids a lagged link while ``prune=False``, or requires a link at a lag beyond the ``p``
        that ``criterion`` selected.

    Notes
    -----
    The likelihood fit runs on the series with each column divided by its standard deviation, and
    its matrices and residuals are mapped back to the data's units, so rescaling a column changes
    the fit only through rounding.

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

    **Missing values.** A ``NaN`` in ``data`` raises ``ValueError``: the VARMA fit and the LiNGAM
    step on its residuals read every time step, so a step with a missing value has no place in them.
    Fill in the missing steps first, as dropping one would join the steps around it, or use
    :func:`~andrey.pcmci`, :func:`~andrey.pcmciplus`, or :func:`~andrey.lpcmci`, whose tests delete
    rows per test.

    **Limitations.** Measured on simulated linear non-Gaussian VARMA time series, of ``max(1000, 4 *
    (d + 1))`` time steps for ``d`` variables and of five times as many, with ``prune=True``, on 4
    cores with a cap of 1 h and 28 GB per fit. The median F1 score of its directed cross-links
    (same-step and lag 1), which is 0 for the empty graph, was 0.66 to 0.88 at every size through 40
    variables with the shorter series and 0.90 to 0.96 through 20 with the series five times as
    long, the largest sizes its fits finished within the caps. Every ladder ended when three fits of
    the next size ran past 1 h. The median fit took 1,845 s at 20 variables with the series five
    times as long. On series with one autoregressive and one moving-average lag, from 3 to 20
    variables with 100 to 1,000 time steps, with ``order=(1, 1)`` and ``prune=True``, its median
    errors (missing and extra cross-links) stayed below the empty graph's from 300 time steps at
    every size; at 100 time steps they exceeded it from 10 variables on. It assumes linear effects
    and non-Gaussian noise. Besides stopping at ``max_iter``, which warns, the likelihood fit can
    stop before it converges for another reason, such as a failed line search, and gives no
    warning; a larger ``max_iter`` does not prevent this. The result then depends on the versions of
    the numerical libraries. Where the fit stops, even when it converges, also depends on the BLAS
    kernels the CPU runs, so the same series can give a different graph on another type of CPU or in
    another environment. With ``criterion`` set, it fits one model for every ``(p, q)`` in the grid.
    The numbers are a reference, not a hard limit: they change with the data generator, the density,
    the method's settings, and the machine.

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

    X = data_matrix(data, method="varma_lingam", fix=FILL_STEPS)
    if X.shape[1] < 2:
        raise ValueError(f"data needs at least 2 columns (variables), got {X.shape[1]}")
    if tuple(order) == (0, 0):
        raise ValueError("varma_lingam: order must have p or q above 0, got (0, 0)")
    labels = column_labels(data)
    constraints = resolve_lagged(knowledge, X.shape[1], labels, order[0], method="varma_lingam")
    causal_order, psis, omegas = _native(
        X,
        order=order,
        criterion=criterion,
        prune=prune,
        structural_ma=structural_ma,
        max_iter=max_iter,
        measure=measure,
        constraints=constraints,
        name=column_namer(labels),
    )
    model = SimpleNamespace(adjacency_matrices_=(psis, omegas), causal_order_=causal_order)
    return with_labels(_adapt_varma_lingam(model), labels)


@method_spec("longitudinal_lingam")
def longitudinal_lingam(
    data: Sequence[npt.ArrayLike],
    *,
    tau_max: int = 1,
    measure: str = "pwling",
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn a temporal causal structure from panel data with Longitudinal LiNGAM.

    Longitudinal LiNGAM [1]_ handles the same units measured at several time points: at each
    occasion it regresses out the past ``tau_max`` time points, then applies DirectLiNGAM to orient
    the instantaneous effects. Use it for repeated-measures / panel designs rather than one long
    series.

    Parameters
    ----------
    data : sequence of array-like, each of shape (n_samples, n_variables)
        One dataset per time point, over the same units and identical shape. Each is coerced to
        ``float64``. DataFrames must name their columns alike; the names become the structure's
        ``labels``.
    tau_max : int, default=1
        The largest lag: the number of past time points regressed out before the instantaneous
        fit, from ``1`` to the number of time points minus one.
    measure : {"pwling"}, default="pwling"
        How the instantaneous order search picks the most exogenous variable; ``"pwling"`` is
        DirectLiNGAM's pairwise likelihood-ratio measure.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`, read as links between lagged
        variables over lags ``0`` to ``tau_max``, the same at every time point: required and
        forbidden links, the pairs allowed to be linked, and temporal tiers, all honored. A link
        ``(cause, effect, lag)`` is the link from ``cause`` at time point ``t - lag`` to
        ``effect`` at time point ``t``, lag ``0`` being the same time point; a forbidden
        ``(cause, effect)`` pair forbids it at every lag, and a required link names its lag. At
        lag ``0``, each time point's DirectLiNGAM honors the knowledge as
        :func:`~andrey.direct_lingam` does: the causal order puts the cause of each required
        link, and each earlier tier, first, and the regression drops forbidden parents and keeps
        required ones unpenalized. A forbidden lagged link is left out of the least-squares fit
        of its effect's lagged weights, so its weight is ``0``; every other lagged weight stays,
        so a required lagged link is in the graph. Tiers forbid every link from a later tier into
        an earlier one, and a pair left out of ``allowed`` has no link at any lag; a variable's
        links to its own past stay allowed.

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
        ``ordering`` is ``None``. ``metadata["algorithm"]`` is ``"longitudinal_lingam"``.

    Raises
    ------
    TypeError
        If ``knowledge`` is neither ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If an argument has the wrong type, lies outside its range, or is not one of its choices; the
        message names the method, the argument, what it accepts, and the value. Also if
        ``data`` has fewer than two time points, if ``tau_max`` is at least the number of time
        points, if the per-time arrays differ in shape, or if the DataFrames' column names differ.
        Also if ``knowledge`` names a column the data lacks, contradicts itself, or requires a link
        without a lag or beyond ``tau_max``.

    Notes
    -----
    At each time point DirectLiNGAM orders the residuals of the lag regressions. Each pairwise
    regression in its ordering takes the least-squares slope, so every residual it scores is
    uncorrelated with its regressor in the sample, and listing every unit twice leaves the causal
    orders unchanged.

    **Column order.** The fit runs on the variables sorted by a rank computed from each column's own
    data over all time points, as for :func:`~andrey.hc`, so reordering the columns relabels the
    result and changes nothing else, unless two columns are identical.

    **Missing values.** A ``NaN`` in a dataset raises ``ValueError``: each time point's fit compares
    the residuals of every variable over the same units, so a unit with a missing value has no place
    in it. Drop the units that hold one from every dataset first, which suits values missing
    completely at random.

    **Limitations.** Measured on simulated linear non-Gaussian panels, each unit observed at 20 time
    points, on 4 cores with a cap of 1 h and 28 GB per fit. The median F1 score of its directed
    cross-links (same-step and lag 1), which is 0 for the empty graph, was 0.98 to 1 at every size
    through 40 variables with 2,000 units and 0.94 to 1 through 40 with 500 units, the largest sizes
    its fits finished within the caps. Every ladder ended at the largest planned size. The median
    fit took 41 s at 40 variables with 2,000 units. It assumes linear effects and non-Gaussian
    noise. Each time point's fit uses only the units' rows at that time, so its accuracy depends on
    the number of units, not on the number of time points. The numbers are a reference, not a hard
    limit: they change with the data generator, the density, the method's settings, and the machine.

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
    >>> panel = [rng.uniform(-1, 1, (200, 3)) for _ in range(4)]
    >>> out = longitudinal_lingam(panel, tau_max=1)
    >>> out.structure.type
    'temporal'
    """
    from andrey.temporal.longitudinal import LongitudinalLiNGAM

    panel = [
        data_matrix(X, name=f"data[{t}]", method="longitudinal_lingam", fix=DROP_UNITS)
        for t, X in enumerate(data)
    ]
    if len(panel) < 2:
        raise ValueError(f"data must hold at least two time points, got {len(panel)}")
    tau_max = int_in_range(tau_max, "tau_max", 1, len(panel) - 1)
    labels = shared_labels(column_labels(X) for X in data)
    constraints = resolve_lagged(
        knowledge, panel[0].shape[1], labels, tau_max, method="longitudinal_lingam"
    )
    model = LongitudinalLiNGAM(n_lags=tau_max, measure=measure, constraints=constraints).fit(panel)
    return with_labels(_adapt_longitudinal_lingam(model), labels)


@method_spec("pcmci")
def pcmci(
    data: npt.ArrayLike,
    *,
    tau_max: int = 1,
    pc_alpha: float = 0.2,
    alpha: float = 0.05,
    indep_test: str | None = None,
    max_depth: int | None = None,
    knowledge: Knowledge | None = None,
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
    alpha : float, default=0.05
        Significance level of the MCI step, in ``(0, 1)``, the level that decides the returned
        links: a link is in the graph when its p-value is at most ``alpha``.
    indep_test : {"student_t", "fisherz"} or None, default=None
        The conditional-independence test behind every p-value, a partial-correlation test. Both
        report the same partial correlation; ``"student_t"`` takes its p-value from the Student-t
        distribution and ``"fisherz"`` from Fisher's z-transform. ``None`` picks ``"fisherz"``.
    max_depth : int or None, default=None
        The maximum conditioning-set size of every conditional-independence test of the PC1
        step, at least ``0``: round ``p`` of PC1 conditions on the ``p`` strongest other
        candidates, and the rounds stop after ``p = max_depth``; ``None`` sets no limit. A smaller
        cap runs fewer tests and can keep candidate parents that a larger conditioning set would
        remove. The MCI step conditions on the selected parents of both ends, however many.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`, read as links between lagged
        variables: required and forbidden links, the pairs allowed to be linked, and temporal
        tiers, all honored. A link ``(cause, effect, lag)`` is the link from ``cause`` at time
        ``t - lag`` to ``effect`` at time ``t``; a forbidden ``(cause, effect)`` pair forbids it at
        every lag, and a required link names its lag, from ``1`` to ``tau_max``. No test runs on
        a forbidden or a required link: a forbidden link is in no conditioning set and not in the
        graph, and a required one is a parent in every PC1 round and in the graph. Tiers forbid
        every link from a later tier into an earlier one, and a pair left out of ``allowed`` has
        no link at any lag; a variable's links to its own past stay allowed.

    Returns
    -------
    StructureOutput
        Wraps a ``temporal`` :class:`~andrey.core.TemporalStructure` with lags ``1`` to
        ``tau_max`` and no lag 0. ``structure.lag(k)`` is a ``digraph`` whose edge ``i -> j`` is
        the link from variable ``i`` at time ``t - k`` to variable ``j`` at time ``t``; a self-loop
        is a variable's dependence on its own past. The structure has no ``lag_weights``.
        ``metadata["p_values"][k - 1][i][j]`` and ``metadata["statistics"][k - 1][i][j]`` hold the
        MCI p-value and partial correlation of that link, for every pair, significant or not; a
        link the knowledge forbids reads p-value 1 and statistic 0, and a required one reads 0 and
        1. ``metadata["parents"][j]`` lists the ``[i, k]`` pairs (variable ``i`` at lag ``k``)
        that PC1 selected for variable ``j``, strongest first. ``ordering`` is ``None``, and
        ``metadata["algorithm"]`` is ``"pcmci"``.

    Raises
    ------
    TypeError
        If ``knowledge`` is neither ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If an argument has the wrong type, lies outside its range, or is not one of its choices; the
        message names the method, the argument, what it accepts, and the value. Also if ``data`` is
        not a usable 2-D matrix with at least ``2 * tau_max * (n_variables + 1) + 3`` rows, the
        length at which every test keeps a degree of freedom. Also if a variable repeats another up
        to scale and shift, at the same step or up to ``2 * tau_max`` steps apart, before any test
        runs; one message names every group of copies, a lagged value as ``'x' at lag k``, and says
        to keep one column per group. Also if ``knowledge`` names a column ``data`` lacks,
        contradicts itself, or requires a link without a lag or outside lags ``1`` to ``tau_max``.

    Notes
    -----
    Every test is a partial-correlation test. The default, ``"fisherz"``, takes the p-value from
    Fisher's z-transform of the partial correlation. ``"student_t"`` takes it from the Student-t
    distribution with ``n - |S| - 2`` degrees of freedom, for ``|S|`` conditions, as the authors'
    reference implementation does; its p-values differ slightly, so a link close to ``alpha`` can
    flip. The partial correlation
    is the correlation of the least-squares residuals of the two tested values given the
    conditions. It is computed from the correlation matrix, and from the residuals themselves when
    the conditions are collinear to working precision, as lagged copies of a strongly
    autocorrelated series can be.

    Without missing values, all tests use the same time steps, the last ``n_samples - 2 * tau_max``,
    where every lag a test can condition on exists: the MCI test of a link at lag ``k`` conditions
    on the cause's parents, which reach back ``k + tau_max`` steps. PC1 runs in rounds ``p = 0, 1,
    ...``: each round tests every remaining candidate given the ``p`` strongest other candidates,
    one conditioning set per candidate, then drops those that test independent. A candidate's
    strength is the smallest absolute partial correlation it has had so far. Equal strengths are
    ordered by each variable's rank, as in :func:`~andrey.hc`; the :doc:`column-order guide
    </docs/guides/column-order>` says what the rank keeps.

    **Missing values.** A ``NaN`` in ``data`` marks a missing value. Every row of the lagged design
    that reads it holds it, and each test reads only the design rows where all of its lagged
    variables are observed (test-wise deletion) [2]_ [3]_, so the tests no longer share one set of
    time steps; ``metadata["rows_tested"]`` gives the fewest and the median design rows a test read.
    When values are missing completely at random, the rows a test keeps are a random sample, so each
    test keeps its level [3]_. A test with fewer than ``|S| + 3`` rows, or with a column constant on
    its rows, raises ``ValueError``.

    **Limitations.** Measured on simulated linear-Gaussian time series with lagged links, of
    ``max(1000, 8 * (d + 1))`` time steps for ``d`` variables and of five times as many, with
    ``tau_max=2``, on 4 cores with a cap of 1 h and 28 GB per fit. The largest sizes through which
    its median errors (lag-resolved SHD) stayed below the empty graph's, and through which its fits
    finished within the caps, were 20 and 80 variables with the shorter series or the series five
    times as long. Every ladder ended after two sizes in a row did not beat the empty graph. The
    median fit took 0.87 s at 80 variables with the series five times as long. On series with an
    autodependency in every variable and one cross-link per variable at lag 1 or 2, from 10 to 160
    variables with 500 to 2,000 time steps, it found nearly every true link at every size, but its
    false links grew with the number of variables. It assumes linear effects, a stationary series,
    and no hidden common causes. It does not test same-step (lag 0) links: an effect within one time
    step is missed and can show up as a false lagged link. An MCI test conditions on up to ``2 *
    n_variables * tau_max - 1`` lagged values, so the series needs many more time steps than that.
    The MCI step alone runs ``n_variables ** 2 * tau_max`` tests. Missing values are not benchmarked
    yet; the tests run them on up to 5 variables with up to a fifth of the values missing completely
    at random. The numbers are a reference, not a hard limit: they change with the data generator,
    the density, the method's settings, and the machine.

    References
    ----------
    .. [1] Runge, J., Nowack, P., Kretschmer, M., Flaxman, S., and Sejdinovic, D. (2019). Detecting
       and quantifying causal associations in large nonlinear time series datasets. Science
       Advances 5(11), eaau4996.
    .. [2] Strobl, E. V., Visweswaran, S., and Spirtes, P. L. (2018). Fast causal inference with
       non-random missingness by test-wise deletion. International Journal of Data Science and
       Analytics 6(1), 47-62.
    .. [3] Tu, R., Zhang, C., Ackermann, P., Mohan, K., Kjellstrom, H., and Zhang, K. (2019). Causal
       discovery in the presence of missing data. Proceedings of Machine Learning Research 89,
       1762-1770.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import pcmci
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal(500)
    >>> y = np.r_[0.0, 0.8 * x[:-1]] + rng.standard_normal(500)
    >>> out = pcmci(np.column_stack([x, y]), tau_max=2, alpha=0.01)
    >>> out.structure.lag(1).oriented_edges()
    [(0, 1, 'directed')]
    >>> out.structure.lag(2).oriented_edges()
    []
    """
    from andrey.temporal.pcmci import design_column_name
    from andrey.temporal.pcmci import pcmci as _native

    X = data_matrix(data, method="pcmci", missing=True)
    # An MCI test conditions on up to 2 * n * tau_max - 1 lagged columns of the last
    # n_samples - 2 * tau_max steps; this keeps a degree of freedom in every test.
    min_rows = 2 * tau_max * (X.shape[1] + 1) + 3
    if X.shape[0] < min_rows:
        raise ValueError(
            f"data needs at least {min_rows} rows (time steps) for tau_max={tau_max} over "
            f"{X.shape[1]} variables, got {X.shape[0]}"
        )
    labels = column_labels(data)
    constraints = resolve_lagged(
        knowledge, X.shape[1], labels, tau_max, method="pcmci", contemporaneous=False
    )
    with (
        copies_named(labels, design_column_name(X.shape[1], column_namer(labels))),
        rows_tested() as rows,
    ):
        result = _native(
            X,
            tau_max=tau_max,
            pc_alpha=pc_alpha,
            alpha_level=alpha,
            indep_test="fisherz" if indep_test is None else indep_test,
            constraints=constraints,
            max_depth=max_depth,
        )
    return with_labels(_adapt_pcmci(result, rows_metadata(rows)), labels)


@method_spec("pcmciplus")
def pcmciplus(
    data: npt.ArrayLike,
    *,
    tau_max: int = 1,
    alpha: float = 0.01,
    collider_rule: str = "majority",
    indep_test: str | None = None,
    max_depth: int | None = None,
    max_p_global: int | None = None,
    knowledge: Knowledge | None = None,
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
    alpha : float, default=0.01
        Significance level of every test, in ``(0, 1)``: a link is removed when a test gives it a
        p-value above ``alpha``. The default is the level of the method's published experiments
        [1]_.
    collider_rule : {"majority", "conservative", "none"}, default="majority"
        How an unshielded triple ``a - b - c`` with a same-step link ``b - c`` is judged a
        collider ``a -> b <- c``. ``"majority"`` and ``"conservative"`` test ``a`` and ``c`` again
        given every subset of their same-step neighbors and look at the subsets that separate
        them: ``"majority"`` calls the triple a collider when fewer than half hold ``b``, and
        ``"conservative"`` when none does; a triple neither rule can decide is ambiguous and
        orients nothing. ``"none"`` calls it a collider when ``b`` is not in the one separating
        set the skeleton search found.
    indep_test : {"student_t", "fisherz"} or None, default=None
        The conditional-independence test behind every p-value, a partial-correlation test. Both
        report the same partial correlation; ``"student_t"`` takes its p-value from the Student-t
        distribution and ``"fisherz"`` from Fisher's z-transform. ``None`` picks ``"fisherz"``.
    max_depth : int or None, default=None
        The maximum conditioning-set size of every conditional-independence test of the lagged
        skeleton (phase 1, PC1 as in :func:`~andrey.pcmci`), at least ``0``; ``None`` sets no
        limit. A smaller cap runs fewer tests and can keep lagged parents that a larger
        conditioning set would remove. It does not cap the skeleton and collider phases, which
        add the lagged parents of both ends to every conditioning set; ``max_p_global`` caps those.
    max_p_global : int or None, default=None
        The most conditions a test of the skeleton and collider phases takes beyond the default
        conditions, at least ``0``: every such test conditions on the lagged parents of both ends,
        the default conditions, and on a subset of at most ``max_p_global`` same-step neighbors;
        the cap also bounds the subsets the collider vote reads. It is tigramite's name for this
        cap, as in :func:`~andrey.lpcmci`. Unlike ``max_depth``, it does not count the default
        conditions, so a test can condition on more than ``max_p_global`` values. ``None`` sets no
        limit. The number of tests grows exponentially with the same-step links at one variable; a
        cap bounds it and can keep links that a larger conditioning set would remove.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`, read as links between lagged
        variables: required and forbidden links, the pairs allowed to be linked, and temporal
        tiers, all honored. A link ``(cause, effect, lag)`` is the link from ``cause`` at time
        ``t - lag`` to ``effect`` at time ``t``, lag ``0`` being the same step; a forbidden
        ``(cause, effect)`` pair forbids it at every lag, and a required link names its lag, from
        ``0`` to ``tau_max``. The lagged and skeleton phases test neither: a required link stays
        and a forbidden one is never linked. The collider phase tests a pair kept apart for the
        subsets that separate it, as it does any unlinked pair; with ``collider_rule="none"``,
        which reads the skeleton phase's separating set, a triple over such a pair is ambiguous.
        At the same step, a pair forbidden in both directions is not linked, and a link the
        knowledge allows in one direction only is directed that way before the colliders, which
        never reverse it. Tiers forbid every link from a later tier into an earlier one, and a
        pair left out of ``allowed`` has no link at any lag; a variable's links to its own past
        stay allowed. The rule phase applies R1 to R3 only, so a same-step link that Meek's rule R4
        would direct from the knowledge can stay undirected.

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
        symmetric with ``1`` and ``0`` on the diagonal. A link the knowledge forbids, never
        tested, reads p-value 1 and statistic 0, and a required one reads 0 and 1.
        ``metadata["parents"][j]`` lists the
        ``[i, k]`` pairs (variable ``i`` at lag ``k``) the first step selected for variable
        ``j``, strongest first, and ``metadata["ambiguous_triples"]`` the ``[i, k, m, j]``
        triples (variable ``i`` at lag ``k``, then ``m`` and ``j`` at lag ``0``) the collider rule
        left undecided. ``ordering`` is ``None``, and ``metadata["algorithm"]`` is
        ``"pcmciplus"``.

    Raises
    ------
    TypeError
        If ``knowledge`` is neither ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If an argument has the wrong type, lies outside its range, or is not one of its choices; the
        message names the method, the argument, what it accepts, and the value. Also if ``data`` is
        not a usable 2-D matrix with at least ``2 * tau_max * (n_variables + 1) + n_variables + 1``
        rows, the length at which every test keeps a degree of freedom. Also if a variable repeats
        another up to scale and shift, at the same step or up to ``2 * tau_max`` steps apart, before
        any test runs; one message names every group of copies, a lagged value as ``'x' at lag k``,
        and says to keep one column per group. Also if ``knowledge`` names a column ``data`` lacks,
        contradicts itself, requires a link without a lag or beyond ``tau_max``, or directs
        same-step links into a directed cycle over the links the skeleton phase kept.

    Notes
    -----
    The method runs in four phases. The lagged phase selects the lagged parents of each variable
    as PCMCI's PC1 step does, with equal strengths ordered by each variable's rank. The skeleton
    phase starts from those lagged links and from a link between every two variables at the same
    step. Round ``p = 0, 1, ...`` tests each linked pair given each subset of ``p`` same-step
    neighbors of the later end, with the lagged parents of both ends added, and removes the link
    at the first p-value above ``alpha``. A same-step pair is tested from both ends, first
    given the neighbors of its higher-numbered variable, and the neighbor sets stay fixed during a
    round, so the links found do not depend on the order of the variables. The collider phase
    orients the unshielded triples by ``collider_rule``, and the rule phase applies Meek's rules
    R1 to R3 to the same-step links until none applies. Each collider or rule pass collects its
    orientations before applying them, and marks a link that gets both orientations as a
    conflict. With the ``"majority"`` or ``"conservative"`` rule the orientations do not depend on
    the order of the variables either; ``"none"`` reads the one separating set the skeleton phase
    stored, which can change with the order.

    Without missing values, all tests use the same time steps, the last ``n_samples - 2 * tau_max``.
    The default test, ``"fisherz"``, takes the p-value from Fisher's z-transform of the partial
    correlation, and ``"student_t"`` from the Student-t distribution with ``n - |S| - 2`` degrees of
    freedom, for ``|S|`` conditions. The partial correlation is the correlation
    of the least-squares residuals of the two tested values given the conditions. It is computed
    from the correlation matrix, and from the residuals themselves when the conditions are collinear
    to working precision, as lagged copies of a strongly autocorrelated series can be.

    **Missing values.** A ``NaN`` in ``data`` marks a missing value. Every row of the lagged design
    that reads it holds it, and each test reads only the design rows where all of its lagged
    variables are observed (test-wise deletion) [2]_ [3]_, so the tests no longer share one set of
    time steps; ``metadata["rows_tested"]`` gives the fewest and the median design rows a test read.
    When values are missing completely at random, the rows a test keeps are a random sample, so each
    test keeps its level [3]_. A test with fewer than ``|S| + 3`` rows, or with a column constant on
    its rows, raises ``ValueError``.

    **Limitations.** Measured on simulated linear-Gaussian time series with lagged and same-step
    links, of ``max(1000, 20 * (d + 1))`` time steps for ``d`` variables and of five times as many,
    with ``tau_max=5``, on 4 cores with a cap of 1 h and 28 GB per fit. The largest sizes through
    which its median errors (endpoint SHD) stayed below the empty graph's, and through which its
    fits finished within the caps, were 160 and 160 variables with the shorter series or the series
    five times as long. Every ladder ended when the data generator drew no stationary series at 320
    variables, so no fit ran. The median fit took 336 s at 160 variables with the series five times
    as long. It assumes linear effects, a stationary series, and no hidden common causes. The
    skeleton and collider phases test every subset of a variable's same-step neighbors, so the
    number of tests grows exponentially with the number of same-step links at one variable. A test
    conditions on up to ``2 * n_variables * tau_max + n_variables - 2`` values, so the series needs
    many more time steps than that. Missing values are not benchmarked yet; the tests run them on up
    to 5 variables with up to a fifth of the values missing completely at random. The numbers are a
    reference, not a hard limit: they change with the data generator, the density, the method's
    settings, and the machine.

    References
    ----------
    .. [1] Runge, J. (2020). Discovering contemporaneous and lagged causal relations in
       autocorrelated nonlinear time series datasets. Proceedings of the 36th Conference on
       Uncertainty in Artificial Intelligence (UAI), PMLR 124, 1388-1397.
    .. [2] Strobl, E. V., Visweswaran, S., and Spirtes, P. L. (2018). Fast causal inference with
       non-random missingness by test-wise deletion. International Journal of Data Science and
       Analytics 6(1), 47-62.
    .. [3] Tu, R., Zhang, C., Ackermann, P., Mohan, K., Kjellstrom, H., and Zhang, K. (2019). Causal
       discovery in the presence of missing data. Proceedings of Machine Learning Research 89,
       1762-1770.

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

    X = data_matrix(data, method="pcmciplus", missing=True)
    # A test conditions on up to 2 * n * tau_max + n - 2 columns of the last
    # n_samples - 2 * tau_max steps; this keeps a degree of freedom in every test.
    min_rows = 2 * tau_max * (X.shape[1] + 1) + X.shape[1] + 1
    if X.shape[0] < min_rows:
        raise ValueError(
            f"data needs at least {min_rows} rows (time steps) for tau_max={tau_max} over "
            f"{X.shape[1]} variables, got {X.shape[0]}"
        )
    labels = column_labels(data)
    constraints = resolve_lagged(knowledge, X.shape[1], labels, tau_max, method="pcmciplus")
    with (
        copies_named(labels, design_column_name(X.shape[1], column_namer(labels))),
        rows_tested() as rows,
    ):
        result = _native(
            X,
            tau_max=tau_max,
            pc_alpha=alpha,
            collider_rule=collider_rule,
            indep_test="fisherz" if indep_test is None else indep_test,
            constraints=constraints,
            max_depth=max_depth,
            max_p_global=max_p_global,
        )
    return with_labels(_adapt_pcmciplus(result, rows_metadata(rows)), labels)


@method_spec("lpcmci")
def lpcmci(
    data: npt.ArrayLike,
    *,
    tau_max: int = 1,
    alpha: float = 0.05,
    n_preliminary_iterations: int = 1,
    indep_test: str | None = None,
    max_p_global: int | None = None,
    knowledge: Knowledge | None = None,
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
    alpha : float, default=0.05
        Significance level of every test, in ``(0, 1)``: a link is removed when a test gives it a
        p-value above ``alpha``.
    n_preliminary_iterations : int, default=1
        How many preliminary rounds run before the final one, at least ``0``. Each round learns
        parents that the next one conditions on from its first test.
    indep_test : {"student_t", "fisherz"} or None, default=None
        The conditional-independence test behind every p-value, a partial-correlation test. Both
        report the same partial correlation; ``"student_t"`` takes its p-value from the Student-t
        distribution and ``"fisherz"`` from Fisher's z-transform. ``None`` picks ``"fisherz"``.
    max_p_global : int or None, default=None
        The most conditions a test takes beyond the default conditions, at least ``0``: every test
        of the removal phases conditions on the parents of both ends, the default conditions, and
        on a searched subset of at most ``max_p_global`` neighbors; the cap also bounds the subsets
        the majority vote reads. It is tigramite's name for this cap. Unlike ``max_depth`` on
        :func:`~andrey.pc` or :func:`~andrey.pcmci`, it does not count the default conditions, so a
        test can condition on more than ``max_p_global`` nodes. ``None`` sets no limit, as in the
        paper. The number of tests grows exponentially with the links at one node, so on many
        strongly related series an uncapped run may not finish. A cap bounds the number of tests,
        though a capped run can still take long, and can keep links that a larger conditioning set
        would remove.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`, read as links between lagged
        variables: required and forbidden links, the pairs allowed to be linked, and temporal
        tiers, all honored. A link ``(cause, effect, lag)`` is the link from ``cause`` at time
        ``t - lag`` to ``effect`` at time ``t``, lag ``0`` being the same step; a forbidden
        ``(cause, effect)`` pair forbids it at every lag, and a required link names its lag, from
        ``0`` to ``tau_max``. A forbidden lagged link, and a same-step pair forbidden in both
        directions, is never linked. A same-step link forbidden in one direction carries an
        arrowhead at the cause, as in :func:`~andrey.fci`: the cause is not an ancestor of the
        effect. A required link is ``-->`` from the start and no test or rule removes it. No rule
        changes a mark the knowledge sets. Tiers forbid every link from a later tier into an
        earlier one, and a pair left out of ``allowed`` has no link at any lag; a variable's links
        to its own past stay allowed.

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
        diagonal and on pairs no test ran on, and a required link, never tested, reads ``0`` and
        ``1``. ``metadata["parents"][j]`` lists the ``[i, k]`` pairs (variable ``i`` at lag
        ``k``) the last preliminary round found to be parents of variable ``j``. ``ordering`` is
        ``None``, and ``metadata["algorithm"]`` is ``"lpcmci"``.

    Raises
    ------
    TypeError
        If ``knowledge`` is neither ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If an argument has the wrong type, lies outside its range, or is not one of its choices; the
        message names the method, the argument, what it accepts, and the value. Also if ``data`` is
        not a usable 2-D matrix with at least ``n_variables * (tau_max + 1) + 2 * tau_max + 1``
        rows, the length at which every test keeps a degree of freedom. Also if a variable repeats
        another up to scale and shift, at the same step or up to ``2 * tau_max`` steps apart, before
        any test runs; one message names every group of copies, a lagged value as ``'x' at lag k``,
        and says to keep one column per group. Also if ``knowledge`` names a column ``data`` lacks,
        contradicts itself, or requires a link without a lag or beyond ``tau_max``.

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

    Without missing values, all tests use the same time steps, the last ``n_samples - 2 * tau_max``,
    and every node a test conditions on lies within ``tau_max`` steps of the later end. A test's
    partial correlation is computed from the correlation matrix, and from the least-squares
    residuals when the conditions are collinear to working precision. The defaults are the authors'
    published defaults. The earlier end of a lagged link is searched with the empty set only, since
    time order already rules out the later end as its cause.

    **Missing values.** A ``NaN`` in ``data`` marks a missing value. Every row of the lagged design
    that reads it holds it, and each test reads only the design rows where all of its lagged
    variables are observed (test-wise deletion) [2]_ [3]_, so the tests no longer share one set of
    time steps; ``metadata["rows_tested"]`` gives the fewest and the median design rows a test read.
    When values are missing completely at random, the rows a test keeps are a random sample, so each
    test keeps its level [3]_. A test with fewer than ``|S| + 3`` rows, or with a column constant on
    its rows, raises ``ValueError``.

    **Limitations.** Measured on simulated linear-Gaussian time series with 30% of the series
    unobserved, of ``max(1000, 12 * (d + 1))`` time steps for ``d`` variables and of five times as
    many, the unobserved series counted among the variables, with ``tau_max=3``, on 4 cores with a
    cap of 1 h and 28 GB per fit. The largest sizes through which its median errors (endpoint SHD
    against the PAG of the observed series) stayed below the empty graph's, and through which its
    fits finished within the caps, were 40 and 40 variables with the shorter series or the series
    five times as long. At 40 variables with the shorter series, 1 of 10 fits ran past 1 h; at 40
    variables with the series five times as long, 2 of 5 fits ran past 1 h. Every ladder ended when
    three fits of the next size ran past 1 h. The median fit took 263 s at 40 variables with the
    series five times as long. It assumes linear effects, a stationary series, and no selection
    bias; hidden common causes are allowed. The graph covers ``tau_max + 1`` time steps, and a
    hidden cause that acts through earlier steps can add links within that window. The removal and
    rule steps test subsets of a node's neighbors, so the number of tests grows exponentially with
    the number of links at one node. A test conditions on up to ``n_variables * (tau_max + 1) - 2``
    values, so the series needs many more time steps than that. The numbers are a reference, not a
    hard limit: they change with the data generator, the density, the method's settings, and the
    machine. Missing values are not benchmarked yet; the tests run them on up to 5 variables with up
    to a fifth of the values missing completely at random.

    References
    ----------
    .. [1] Gerhardus, A., and Runge, J. (2020). High-recall causal discovery for autocorrelated
       time series with latent confounders. Advances in Neural Information Processing Systems 33
       (NeurIPS 2020), 12615-12625.
    .. [2] Strobl, E. V., Visweswaran, S., and Spirtes, P. L. (2018). Fast causal inference with
       non-random missingness by test-wise deletion. International Journal of Data Science and
       Analytics 6(1), 47-62.
    .. [3] Tu, R., Zhang, C., Ackermann, P., Mohan, K., Kjellstrom, H., and Zhang, K. (2019). Causal
       discovery in the presence of missing data. Proceedings of Machine Learning Research 89,
       1762-1770.

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

    X = data_matrix(data, method="lpcmci", missing=True)
    # A test conditions on up to n * (tau_max + 1) - 2 nodes of the window, over the last
    # n_samples - 2 * tau_max steps; this keeps a degree of freedom in every test.
    min_rows = X.shape[1] * (tau_max + 1) + 2 * tau_max + 1
    if X.shape[0] < min_rows:
        raise ValueError(
            f"data needs at least {min_rows} rows (time steps) for tau_max={tau_max} over "
            f"{X.shape[1]} variables, got {X.shape[0]}"
        )
    labels = column_labels(data)
    constraints = resolve_lagged(knowledge, X.shape[1], labels, tau_max, method="lpcmci")
    with (
        copies_named(labels, design_column_name(X.shape[1], column_namer(labels))),
        rows_tested() as rows,
    ):
        result = _native(
            X,
            tau_max=tau_max,
            pc_alpha=alpha,
            n_preliminary_iterations=n_preliminary_iterations,
            indep_test="fisherz" if indep_test is None else indep_test,
            max_p_global=max_p_global,
            constraints=constraints,
        )
    return with_labels(_adapt_lpcmci(result, rows_metadata(rows)), labels)


@method_spec("granger")
def granger(
    data: npt.ArrayLike, *, tau_max: int = 1, cv: int = 5, knowledge: Knowledge | None = None
) -> StructureOutput:
    """Learn the lagged causal graph of a multivariate time series by lasso Granger causality.

    A variable Granger-causes another when its past improves the prediction of the other's
    present [1]_. Each variable is regressed on the past ``tau_max`` steps of every variable by the
    lasso [2]_, with the penalty chosen by cross-validation; a nonzero coefficient is a lagged
    edge. Use it for one multivariate series whose effects take at least one time step.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        A multivariate time series, one row per time step. Coerced to ``float64``. A DataFrame's
        column names become the structure's ``labels``.
    tau_max : int, default=1
        The largest lag: each regression uses the past ``tau_max`` time steps, at least ``1``.
    cv : int, default=5
        Number of cross-validation folds that choose each regression's penalty, at least 2.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`, read as links between lagged
        variables over lags ``1`` to ``tau_max``: required and forbidden links, the pairs allowed
        to be linked, and temporal tiers, all honored. A link ``(cause, effect, lag)`` is the link
        from ``cause`` at time ``t - lag`` to ``effect`` at time ``t``; a forbidden
        ``(cause, effect)`` pair forbids it at every lag, and a required link names its lag, from
        ``1`` to ``tau_max`` (the method learns no same-step links). Each variable's regression
        drops its forbidden lagged regressors and keeps its required ones unpenalized: the lasso
        selects among the other regressors on what the required ones leave of the variable, and a
        required link's weight is the least-squares weight of what the selected regressors leave of
        the variable. Tiers forbid every link from a later tier into an earlier one, and a pair
        left out of ``allowed`` has no link at any lag; a variable's links to its own past stay
        allowed.

    Returns
    -------
    StructureOutput
        Wraps a ``temporal`` :class:`~andrey.core.TemporalStructure` with lags ``1`` to ``tau_max``
        and no lag 0. ``structure.lag(k)`` is a ``digraph`` with the edge ``i -> j`` when
        ``x_i(t - k)`` has a nonzero coefficient in the regression of ``x_j(t)``, self-loops
        included; ``structure.lag_weights[k - 1][i, j]`` is that coefficient, in the data's units.
        ``ordering`` is ``None``, and ``metadata["algorithm"]`` is ``"granger"``.

    Raises
    ------
    TypeError
        If ``knowledge`` is neither ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If an argument has the wrong type, lies outside its range, or is not one of its choices; the
        message names the method, the argument, what it accepts, and the value. Also if ``data`` has
        fewer than ``tau_max + cv`` rows, so that some fold would hold no time step, or if ``data``
        has fewer than 2 columns. Also if ``knowledge`` names a column ``data`` lacks, contradicts
        itself, or requires a link without a lag or beyond ``tau_max``.

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

    **Missing values.** A ``NaN`` in ``data`` raises ``ValueError``: each variable's lasso
    regression reads every row of the lagged design, so a step with a missing value has no place in
    it. Fill in the missing steps first, as dropping one would join the steps around it, or use
    :func:`~andrey.pcmci`, :func:`~andrey.pcmciplus`, or :func:`~andrey.lpcmci`, whose tests delete
    rows per test.

    **Limitations.** Measured on simulated linear-Gaussian VAR(2) series with no same-step links,
    standardized, from 10 to 160 variables with 1,000 time steps, with ``tau_max=2``. It found most
    true lagged edges but returned many false ones, so its median errors (missing and extra lagged
    edges) exceeded the empty graph's at every size: on half the series at 10 variables and on all
    of them from 20 on. It assumes linear effects and tests no same-step effects. Each variable's
    regression takes ``tau_max`` lags of every variable as inputs, so the work grows with
    ``tau_max`` and with the square of the number of variables. The numbers are a reference, not a
    hard limit: they change with the data generator, the density, the method's settings, and the
    machine.

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
    >>> out = granger(np.column_stack([x, y]))
    >>> round(float(out.structure.lag_weights[0][0, 1]), 1)
    0.8
    """
    from andrey.temporal.granger import granger_lasso

    # LassoCV needs a time step in every fold.
    X = data_matrix(data, min_rows=tau_max + cv, method="granger", fix=FILL_STEPS)
    if X.shape[1] < 2:
        raise ValueError(f"data needs at least 2 columns (variables), got {X.shape[1]}")
    labels = column_labels(data)
    constraints = resolve_lagged(
        knowledge, X.shape[1], labels, tau_max, method="granger", contemporaneous=False
    )
    peak = np.abs(X).max(axis=0)  # divide first, so tiny or huge units cannot underflow the std
    spread = (X / peak).std(axis=0)  # data_matrix rejects a constant column
    coeff = granger_lasso(
        (X / peak - (X / peak).mean(axis=0)) / spread,
        maxlag=tau_max,
        cv=cv,
        forbidden=None if constraints is None else constraints.forbidden[1:],
        required=None if constraints is None else constraints.required[1:],
    )
    scale = spread * peak
    # coeff[i, j + d*k] weights standardized x_j(t-k-1) in standardized x_i(t): rescale to units.
    # Units far apart can overflow the ratio; a zero weight stays zero rather than becoming NaN.
    with np.errstate(over="ignore", invalid="ignore"):
        ratio = scale[:, None] / np.tile(scale, tau_max)[None, :]
        coeff = np.where(coeff == 0, 0.0, coeff * ratio)
    return with_labels(_adapt_granger(coeff, X.shape[1], tau_max), labels)
