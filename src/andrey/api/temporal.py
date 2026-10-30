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
    categorical_metadata,
    checked_data,
    column_labels,
    column_namer,
    copies_named,
    data_matrix,
    default_test,
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
    measure : {"pwling", "skew", "tanh"}, default="pwling"
        How the instantaneous order search, DirectLiNGAM on the innovations, picks the most
        exogenous variable, as in :func:`~andrey.direct_lingam` [3]_: ``"pwling"`` approximates the
        pairwise likelihood ratios by entropies, and ``"tanh"`` and ``"skew"`` by a correlation of
        nonlinear functions of the standardized innovations.
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
        ones unpenalized; a required lagged link is in the graph even where its estimated weight
        is ``0``. With ``prune=False`` every lagged weight stays, so a required lagged link is in
        the graph, and a forbidden lagged link raises. Tiers forbid every link from a
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

    **Choosing the measure.** ``"skew"`` and ``"tanh"`` suit the noise they suit in
    :func:`~andrey.direct_lingam`, whose Notes and Limitations give the numbers; here they read the
    standardized innovations.

    **Column order.** The fit runs on the columns sorted by a rank computed from the data, as for
    :func:`~andrey.hc`, so reordering the columns relabels the result and changes nothing else,
    unless two columns are identical.

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
    .. [3] Hyvarinen, A., and Smith, S. M. (2013). Pairwise likelihood ratios for estimation of
       non-Gaussian structural equation models. JMLR 14, 111-152.

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
    required = None if constraints is None else constraints.required
    return with_labels(_adapt_varma_lingam(model, required), labels)


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
    measure : {"pwling", "skew", "tanh"}, default="pwling"
        How the instantaneous order search picks the most exogenous variable, as in
        :func:`~andrey.direct_lingam` [2]_: ``"pwling"`` approximates the pairwise likelihood ratios
        by entropies, and ``"tanh"`` and ``"skew"`` by a correlation of nonlinear functions of the
        standardized columns.
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
        and a required lagged link is in the graph even where its least-squares weight is ``0``.
        Tiers forbid every link from a later tier into an earlier one, and a pair left out of
        ``allowed`` has no link at any lag; a variable's links to its own past stay allowed.

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

    **Choosing the measure.** ``"skew"`` and ``"tanh"`` suit the noise they suit in
    :func:`~andrey.direct_lingam`, whose Notes and Limitations give the numbers; here they read the
    standardized columns of each time point.

    **Column order.** The fit runs on the variables sorted by a rank computed from the data over all
    time points, as for :func:`~andrey.hc`, so reordering the columns relabels the result and
    changes nothing else, unless two columns are identical.

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
    .. [2] Hyvarinen, A., and Smith, S. M. (2013). Pairwise likelihood ratios for estimation of
       non-Gaussian structural equation models. JMLR 14, 111-152.

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
    required = None if constraints is None else constraints.required
    return with_labels(_adapt_longitudinal_lingam(model, required), labels)


@method_spec("pcmci")
def pcmci(
    data: npt.ArrayLike,
    *,
    tau_max: int = 1,
    pc1_alpha: float = 0.2,
    alpha: float = 0.05,
    fdr_method: str = "fdr_bh",
    indep_test: str | None = None,
    categorical: Sequence[str | int] | None = None,
    seed: int | None = None,
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
    pc1_alpha : float, default=0.2
        Significance level of the PC1 step, in ``(0, 1)``: a candidate parent is dropped when a test
        gives it a p-value above ``pc1_alpha``. A larger value keeps more parents as conditions for
        the MCI step.
    alpha : float, default=0.05
        Significance level of the MCI step, in ``(0, 1)``, the level that decides the returned
        links: a link is in the graph when its p-value is at most ``alpha``.
    fdr_method : {"none", "fdr_bh"}, default="fdr_bh"
        How the MCI p-values are corrected for the number of links tested. ``"fdr_bh"`` replaces
        them with their Benjamini-Hochberg adjusted values [4]_ over all the links together, every
        lag and every ordered pair of variables, a variable's link to its own past included. A link
        is then in the graph when its adjusted p-value is at most ``alpha``, which holds the false
        discovery rate, the expected share of false links among the returned links, at or below
        ``alpha`` when the tests are independent or positively dependent. The adjusted p-value is
        never below the tested one, so ``"fdr_bh"`` can remove links and never adds one, and
        ``metadata["p_values"]`` holds the adjusted values. ``"none"`` keeps each p-value as
        tested, so about ``alpha`` of the absent links are in the graph. A link the knowledge
        forbids is no test: it is not counted and keeps p-value ``1``; a required link is counted
        with p-value ``0``. See Notes.
    indep_test : str or None, default=None
        The conditional-independence test behind every p-value. ``"fisherz"`` and ``"student_t"``
        test partial correlation, for linear effects in continuous data: ``"fisherz"`` applies
        Fisher's z-transform and the normal distribution, ``"student_t"`` the Student-t
        distribution, as the authors' reference implementation does. ``"normal_scores"`` applies
        ``"fisherz"`` to the normal scores of each column of the lagged values, for monotone
        effects. ``"chisq"`` (Pearson's
        chi-square) and ``"gsq"`` (G-square, the likelihood ratio) test categorical series, coded
        by whole numbers. ``"degen_gauss"`` (degenerate Gaussian) and ``"cond_gauss"`` (conditional
        Gaussian) test mixed series, continuous and categorical columns, by a likelihood ratio.
        Four more tests detect nonlinear dependence in continuous data, at a far higher cost per
        test: ``"cmiknn"``, a nearest-neighbor conditional mutual information test, and
        ``"gpdc"``, Gaussian-process regression with a distance correlation test, the nonlinear
        tests the method was published with; ``"kci"``, the kernel test; and ``"rcot"``, its
        random-feature approximation. ``None`` picks from the declared column types: ``"gsq"``
        when every column is categorical, ``"cond_gauss"`` when some are, else ``"fisherz"``. A
        column is declared categorical by ``categorical`` or by a DataFrame's ``category``,
        ``bool``, or text type, never by its values.
        ``metadata["indep_test"]`` names the test the fit used: the one picked from the
        declared column types when this is ``None``.
    categorical : sequence of str or int, optional
        The categorical columns of ``data``, for a mixed-data test: column labels (strings) or
        0-based positions. ``None`` takes a DataFrame's ``category``, ``bool``, and text columns,
        never a column by its values. ``[]`` marks none. A categorical test reads every column as
        categorical, and a continuous test raises ``ValueError`` on a declared column, named here or
        by its type. Every lagged value of a column has its type.
    seed : int or None, default=None
        ``None``, the default, is the configured seed, ``ANDREY_SEED`` or ``0`` when it is unset,
        not a fresh random one, so a fit repeats. Seed for the random draws of ``indep_test``, which
        only ``"rcot"``, ``"cmiknn"``, and ``"gpdc"`` make; the other tests ignore it. With those
        tests, ``metadata["seed"]`` records the seed used.
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
        MCI p-value, adjusted under ``fdr_method="fdr_bh"``, and test statistic of that link (the
        partial correlation under ``"fisherz"`` and ``"student_t"``; see Notes), for every pair,
        significant or not; a link the knowledge forbids reads p-value 1 and statistic 0, and a
        required one reads 0 and 1.
        ``metadata["parents"][j]`` lists the ``[i, k]`` pairs (variable ``i`` at lag ``k``) that
        PC1 selected for variable ``j``, strongest first. Under a mixed-data test,
        ``metadata["categorical"]`` lists the categorical columns it read, by name or 0-based
        position; with a randomized ``indep_test``, ``metadata["seed"]`` is the seed used.
        ``ordering`` is ``None``, and ``metadata["algorithm"]`` is ``"pcmci"``.

    Raises
    ------
    TypeError
        If ``knowledge`` is neither ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If an argument has the wrong type, lies outside its range, or is not one of its choices; the
        message names the method, the argument, what it accepts, and the value. Also if ``data`` is
        not a usable 2-D matrix with at least ``2 * tau_max * (n_variables + 1) + 3`` rows, the
        length at which every test keeps a degree of freedom. With ``"fisherz"`` or ``"student_t"``,
        also if a variable repeats another up to scale and shift, at the same step or up to ``2 *
        tau_max`` steps apart, before any test runs; one message names every group of copies, a
        lagged value as ``'x' at lag k``, and says to keep one column per group. With ``"chisq"`` or
        ``"gsq"``, also if ``data`` holds a value that is not a whole number; with ``"degen_gauss"``
        or ``"cond_gauss"``, if ``categorical`` names a column ``data`` lacks; and, with a
        continuous test, if a column is declared categorical, by ``categorical`` or by its type.
        Also if ``knowledge`` names a column ``data`` lacks, contradicts itself, or requires a link
        without a lag or outside lags ``1`` to ``tau_max``.

    Notes
    -----
    Every test reads the lagged values as the columns of one design. ``"fisherz"`` and
    ``"student_t"`` report the partial correlation, ``"chisq"`` and ``"gsq"`` their chi-square
    statistic, ``"degen_gauss"`` and ``"cond_gauss"`` the likelihood ratio, ``"cmiknn"`` its
    estimate of the conditional mutual information, ``"gpdc"`` the squared distance correlation,
    ``"rcot"`` its statistic, and ``"kci"`` its statistic over the statistic's mean under
    independence. A candidate's strength is the smallest absolute statistic it has had, as in the
    authors' reference implementation, tigramite; strengths compare within one test only. A test
    registered with :func:`~andrey.core.ci.register_indep_test` that reports only p-values is
    asked one query at a time, its candidates are ranked by p-value, smallest first, and its
    statistic reads ``-log10(p)``.
    The default, ``"fisherz"``, takes the p-value from Fisher's z-transform of the partial
    correlation. ``"student_t"`` takes it from the Student-t
    distribution with ``n - |S| - 2`` degrees of freedom, for ``|S|`` conditions, as the authors'
    reference implementation does; its p-values differ slightly, so a link close to ``alpha`` can
    flip. The partial correlation
    is the correlation of the least-squares residuals of the two tested values given the
    conditions. It is computed from the correlation matrix, and from the residuals themselves when
    the conditions are collinear to working precision, as lagged copies of a strongly
    autocorrelated series can be.

    **Choosing fdr_method.** The MCI step tests ``n_variables ** 2 * tau_max`` links, each at level
    ``alpha``. With ``"none"`` about ``alpha`` of the absent links are returned by chance, so the
    extra links grow with the number of variables (see Limitations). ``"fdr_bh"`` [4]_ sorts the
    ``m`` p-values of the links and gives the one of rank ``k`` the smallest value, over its rank
    and every higher one, of ``p * m / rank``, at most ``1``; equal p-values get equal results. It
    returns fewer links than ``"none"``: the false ones, and also true ones whose p-value is not
    small enough for the number of tests. The authors' reference implementation, tigramite,
    corrects the MCI p-values of the same links in the same way with ``fdr_method="fdr_bh"`` and
    thresholds the adjusted values at ``alpha_level``, as ``alpha`` does here. The PC1 step is not
    corrected. The guarantee of [4]_ needs independent or positively dependent tests, which the
    MCI tests of one series are not shown to be.

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
    test keeps its level [3]_. A partial-correlation test with fewer than ``|S| + 3`` rows, another
    test with fewer than ``2``, or a test with a continuous column constant on its rows raises
    ``ValueError``. The categorical, mixed-data, and nonlinear tests delete rows the same way.

    **Limitations.** Measured with ``fdr_method="none"`` on simulated linear-Gaussian time series
    with lagged links, of ``max(1000, 8 * (d + 1))`` time steps for ``d`` variables and of five
    times as many, with ``tau_max=2``, on 4 cores with a cap of 1 h and 28 GB per fit. The largest
    sizes through which its median errors (lag-resolved SHD) stayed below the empty graph's, and
    through which its fits finished within the caps, were 20 and 80 variables with the shorter
    series or the series five times as long. Every ladder ended after two sizes in a row did not
    beat the empty graph. The median fit took 0.87 s at 80 variables with the series five times as
    long. On series with an autodependency in every variable and one cross-link per variable at lag
    1 or 2, from 10 to 160 variables with 500 to 2,000 time steps, it found nearly every true link
    at every size, but its false links grew with the number of variables. On 20 simulated
    linear-Gaussian VAR series per cell, with 2 expected parents of each variable at each lag, no
    same-step links, from 5 to 80 variables, with 500 and 1,000 time steps, and ``tau_max`` of 1 and
    2 (20 cells), the default ``fdr_method="fdr_bh"`` had fewer mean errors (missing plus extra
    lagged links) than ``"none"`` in 16 cells, by the paired 95% bootstrap interval of the
    difference, the same in 3, and more in 1: 12.5 against 11.7 with 5 variables, 500 steps, and
    ``tau_max=2``. From 20 variables on it was better in every cell. At 80 variables, 1,000 steps,
    and ``tau_max=2``, ``"none"`` had 779 errors, above the empty graph's 481, with 616 extra links
    and 163 missed true links on average; ``"fdr_bh"`` had 252, with 13 extra and 239 missed. The
    mean errors were above the empty graph's in 5 of the 20 cells with ``"none"``, in none with
    ``"fdr_bh"``. ``"fdr_bh"`` trades missed true links for far fewer extra ones: with 5 to 10
    variables the two differ by at most 2.8 errors, and with 5 variables and ``tau_max=2``
    ``"fdr_bh"`` removed the extra links (1.2 to 0.4 at 500 steps) and missed more true ones (10.4
    to 12.1). With a partial-correlation test it assumes linear effects; ``"cmiknn"`` and ``"gpdc"``
    lift that assumption at a far higher cost per test and are not benchmarked yet. It assumes a
    stationary series and no hidden common causes. It does not test same-step (lag 0) links: an
    effect within one time step is missed and can show up as a false lagged link. An MCI test
    conditions on up to ``2 * n_variables * tau_max - 1`` lagged values, so the series needs many
    more time steps than that. The MCI step alone runs ``n_variables ** 2 * tau_max`` tests. Missing
    values are not benchmarked yet; the tests run them on up to 5 variables with up to a fifth of
    the values missing completely at random. The numbers are a reference, not a hard limit: they
    change with the data generator, the density, the method's settings, and the machine.

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
    .. [4] Benjamini, Y., and Hochberg, Y. (1995). Controlling the false discovery rate: a
       practical and powerful approach to multiple testing. Journal of the Royal Statistical
       Society B 57(1), 289-300.

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

    indep_test, categorical = default_test(data, indep_test, categorical=categorical)
    X, mask = checked_data(
        data,
        categorical,
        "indep_test",
        indep_test,
        method="pcmci",
        missing=True,
    )
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
            pc_alpha=pc1_alpha,
            alpha_level=alpha,
            fdr_method=fdr_method,
            indep_test=indep_test,
            constraints=constraints,
            max_depth=max_depth,
            categorical=mask,
            seed=seed,
        )
    metadata = {**categorical_metadata(mask, data), **rows_metadata(rows)}
    return with_labels(_adapt_pcmci(result, metadata), labels)


@method_spec("pcmciplus")
def pcmciplus(
    data: npt.ArrayLike,
    *,
    tau_max: int = 1,
    alpha: float = 0.01,
    fdr_method: str = "fdr_bh",
    collider_rule: str = "majority",
    indep_test: str | None = None,
    categorical: Sequence[str | int] | None = None,
    seed: int | None = None,
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
    fdr_method : {"none", "fdr_bh"}, default="fdr_bh"
        How the p-values of the lagged links are corrected for the number of links tested.
        ``"fdr_bh"`` replaces the p-values of lags ``1`` to ``tau_max`` with their
        Benjamini-Hochberg adjusted values [4]_ over all those links together, every lag and every
        ordered pair of variables, a variable's link to its own past included; the same-step
        p-values stay as tested. The adjusted p-value is never below the tested one. ``"none"``
        keeps each p-value as tested. The links themselves do not change with either: the skeleton
        phase decides them at ``alpha``, so ``fdr_method`` only changes ``metadata["p_values"]``,
        and under ``"fdr_bh"`` a lagged link of the graph whose adjusted p-value is above ``alpha``
        is one the correction would not keep. A link the knowledge forbids is no test: it is not
        counted and keeps p-value ``1``; a required link is counted with p-value ``0``. See Notes.
    collider_rule : {"sepsets", "conservative", "majority"}, default="majority"
        How an unshielded triple ``a - b - c`` with a same-step link ``b - c`` is judged a
        collider ``a -> b <- c``. ``"sepsets"`` calls it a collider when ``b`` is not in the one
        separating set the skeleton search found. ``"majority"`` and ``"conservative"`` test ``a``
        and ``c`` again given every subset of their same-step neighbors and look at the subsets
        that separate them: ``"majority"`` calls the triple a collider when fewer than half hold
        ``b``, and ``"conservative"`` when none does; a triple neither rule can decide is ambiguous
        and orients nothing. The default, ``"majority"``, is tigramite's, and its orientations do
        not depend on the order of the variables.
    indep_test : str or None, default=None
        The conditional-independence test behind every p-value. ``"fisherz"`` and ``"student_t"``
        test partial correlation, for linear effects in continuous data: ``"fisherz"`` applies
        Fisher's z-transform and the normal distribution, ``"student_t"`` the Student-t
        distribution, as the authors' reference implementation does. ``"normal_scores"`` applies
        ``"fisherz"`` to the normal scores of each column of the lagged values, for monotone
        effects. ``"chisq"`` (Pearson's
        chi-square) and ``"gsq"`` (G-square, the likelihood ratio) test categorical series, coded
        by whole numbers. ``"degen_gauss"`` (degenerate Gaussian) and ``"cond_gauss"`` (conditional
        Gaussian) test mixed series, continuous and categorical columns, by a likelihood ratio.
        Four more tests detect nonlinear dependence in continuous data, at a far higher cost per
        test: ``"cmiknn"``, a nearest-neighbor conditional mutual information test, and
        ``"gpdc"``, Gaussian-process regression with a distance correlation test, the nonlinear
        tests the method was published with; ``"kci"``, the kernel test; and ``"rcot"``, its
        random-feature approximation. ``None`` picks from the declared column types: ``"gsq"``
        when every column is categorical, ``"cond_gauss"`` when some are, else ``"fisherz"``. A
        column is declared categorical by ``categorical`` or by a DataFrame's ``category``,
        ``bool``, or text type, never by its values.
        ``metadata["indep_test"]`` names the test the fit used: the one picked from the
        declared column types when this is ``None``.
    categorical : sequence of str or int, optional
        The categorical columns of ``data``, for a mixed-data test: column labels (strings) or
        0-based positions. ``None`` takes a DataFrame's ``category``, ``bool``, and text columns,
        never a column by its values. ``[]`` marks none. A categorical test reads every column as
        categorical, and a continuous test raises ``ValueError`` on a declared column, named here or
        by its type. Every lagged value of a column has its type.
    seed : int or None, default=None
        ``None``, the default, is the configured seed, ``ANDREY_SEED`` or ``0`` when it is unset,
        not a fresh random one, so a fit repeats. Seed for the random draws of ``indep_test``, which
        only ``"rcot"``, ``"cmiknn"``, and ``"gpdc"`` make; the other tests ignore it. With those
        tests, ``metadata["seed"]`` records the seed used.
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
        subsets that separate it, as it does any unlinked pair; with ``collider_rule="sepsets"``,
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
        orientations conflict; ``metadata["conflicts"]`` lists the latter as ``[i, 0, j]`` links
        (variable ``i`` at lag ``0`` and ``j``), ``i < j``. ``structure.lag(k)``, ``k >= 1``, is a
        ``digraph`` whose edge ``i -> j`` is the link from variable ``i`` at time ``t - k`` to
        variable ``j`` at time ``t``; a self-loop is a variable's dependence on its own past. The
        structure has no ``lag_weights``. ``metadata["p_values"][k][i][j]`` is the largest p-value
        among the skeleton-phase tests of the link between variable ``i`` at ``t - k`` and
        variable ``j`` at ``t`` (the first step's tests for a lagged link that step removed; for
        ``k >= 1`` its Benjamini-Hochberg adjusted value under ``fdr_method="fdr_bh"``), and
        ``metadata["statistics"][k][i][j]`` the statistic of that test (see Notes); lag ``0`` is
        symmetric with ``1`` and ``0`` on the diagonal. A link the knowledge forbids, never
        tested, reads p-value 1 and statistic 0, and a required one reads 0 and 1. Under a
        mixed-data test, ``metadata["categorical"]`` lists the categorical columns it read; with a
        randomized ``indep_test``, ``metadata["seed"]`` is the seed used.
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
        rows, the length at which every test keeps a degree of freedom. With ``"fisherz"`` or
        ``"student_t"``, also if a variable repeats another up to scale and shift, at the same step
        or up to ``2 * tau_max`` steps apart, before any test runs; one message names every group of
        copies, a lagged value as ``'x' at lag k``, and says to keep one column per group. With
        ``"chisq"`` or ``"gsq"``, also if ``data`` holds a value that is not a whole number; with
        ``"degen_gauss"`` or ``"cond_gauss"``, if ``categorical`` names a column ``data`` lacks;
        and, with a continuous test, if a column is declared categorical, by ``categorical`` or by
        its type. Also if ``knowledge`` names a column ``data`` lacks, contradicts itself, requires
        a link without a lag or beyond ``tau_max``, or directs same-step links into a directed cycle
        over the links the skeleton phase kept.

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
    the order of the variables either; ``"sepsets"`` reads the one separating set the skeleton phase
    stored, which can change with the order.

    **Choosing fdr_method.** The skeleton phase tests each lagged link at level ``alpha`` and keeps
    the links whose tests all stay at or below it, so absent links are kept by chance, more of them
    as the number of variables grows, whatever ``fdr_method`` is. ``"fdr_bh"`` [4]_ sorts the ``m``
    lagged-link p-values and gives the one of rank ``k`` the smallest value, over its rank and
    every higher one, of ``p * m / rank``, at most ``1``; equal p-values get equal results. The
    reported p-values then show which lagged links of the graph the correction would keep, where
    ``"none"`` reports them as tested. The authors' reference implementation, tigramite, corrects
    the lagged p-values it returns in the same way with ``fdr_method="fdr_bh"`` and leaves the
    graph as the skeleton phase decided it, as this method does. The guarantee of [4]_ needs
    independent or positively dependent tests, which these tests are not shown to be.

    Without missing values, all tests use the same time steps, the last ``n_samples - 2 * tau_max``.
    Every test reads the lagged values as the columns of one design. ``"fisherz"`` and
    ``"student_t"`` report the partial correlation, ``"chisq"`` and ``"gsq"`` their chi-square
    statistic, ``"degen_gauss"`` and ``"cond_gauss"`` the likelihood ratio, ``"cmiknn"`` its
    estimate of the conditional mutual information, ``"gpdc"`` the squared distance correlation,
    ``"rcot"`` its statistic, and ``"kci"`` its statistic over the statistic's mean under
    independence. A candidate's strength is the smallest absolute statistic it has had, as in the
    authors' reference implementation, tigramite; strengths compare within one test only. A test
    registered with :func:`~andrey.core.ci.register_indep_test` that reports only p-values is
    asked one query at a time, its candidates are ranked by p-value, smallest first, and its
    statistic reads ``-log10(p)``.
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
    test keeps its level [3]_. A partial-correlation test with fewer than ``|S| + 3`` rows, another
    test with fewer than ``2``, or a test with a continuous column constant on its rows raises
    ``ValueError``. The categorical, mixed-data, and nonlinear tests delete rows the same way.

    **Limitations.** Measured on simulated linear-Gaussian time series with lagged and same-step
    links, of ``max(1000, 20 * (d + 1))`` time steps for ``d`` variables and of five times as many,
    with ``tau_max=5``, on 4 cores with a cap of 1 h and 28 GB per fit. The largest sizes through
    which its median errors (endpoint SHD) stayed below the empty graph's, and through which its
    fits finished within the caps, were 160 and 160 variables with the shorter series or the series
    five times as long. Every ladder ended when the data generator drew no stationary series at 320
    variables, so no fit ran. The median fit took 336 s at 160 variables with the series five times
    as long. With a partial-correlation test it assumes linear effects; ``"cmiknn"`` and ``"gpdc"``
    lift that assumption at a far higher cost per test and are not benchmarked yet. It assumes a
    stationary series and no hidden common causes. The skeleton and collider phases test every
    subset of a variable's same-step neighbors, so the number of tests grows exponentially with the
    number of same-step links at one variable. A test conditions on up to ``2 * n_variables *
    tau_max + n_variables - 2`` values, so the series needs many more time steps than that. Missing
    values are not benchmarked yet; the tests run them on up to 5 variables with up to a fifth of
    the values missing completely at random. The numbers are a reference, not a hard limit: they
    change with the data generator, the density, the method's settings, and the machine.

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
    .. [4] Benjamini, Y., and Hochberg, Y. (1995). Controlling the false discovery rate: a
       practical and powerful approach to multiple testing. Journal of the Royal Statistical
       Society B 57(1), 289-300.

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

    indep_test, categorical = default_test(data, indep_test, categorical=categorical)
    X, mask = checked_data(
        data,
        categorical,
        "indep_test",
        indep_test,
        method="pcmciplus",
        missing=True,
    )
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
            fdr_method=fdr_method,
            collider_rule=collider_rule,
            indep_test=indep_test,
            constraints=constraints,
            max_depth=max_depth,
            max_p_global=max_p_global,
            categorical=mask,
            seed=seed,
        )
    metadata = {**categorical_metadata(mask, data), **rows_metadata(rows)}
    return with_labels(_adapt_pcmciplus(result, metadata), labels)


@method_spec("lpcmci")
def lpcmci(
    data: npt.ArrayLike,
    *,
    tau_max: int = 1,
    alpha: float = 0.05,
    n_preliminary_iterations: int = 1,
    indep_test: str | None = None,
    categorical: Sequence[str | int] | None = None,
    seed: int | None = None,
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
    indep_test : str or None, default=None
        The conditional-independence test behind every p-value. ``"fisherz"`` and ``"student_t"``
        test partial correlation, for linear effects in continuous data: ``"fisherz"`` applies
        Fisher's z-transform and the normal distribution, ``"student_t"`` the Student-t
        distribution, as the authors' reference implementation does. ``"normal_scores"`` applies
        ``"fisherz"`` to the normal scores of each column of the lagged values, for monotone
        effects. ``"chisq"`` (Pearson's
        chi-square) and ``"gsq"`` (G-square, the likelihood ratio) test categorical series, coded
        by whole numbers. ``"degen_gauss"`` (degenerate Gaussian) and ``"cond_gauss"`` (conditional
        Gaussian) test mixed series, continuous and categorical columns, by a likelihood ratio.
        Four more tests detect nonlinear dependence in continuous data, at a far higher cost per
        test: ``"cmiknn"``, a nearest-neighbor conditional mutual information test, and
        ``"gpdc"``, Gaussian-process regression with a distance correlation test, the nonlinear
        tests the method was published with; ``"kci"``, the kernel test; and ``"rcot"``, its
        random-feature approximation. ``None`` picks from the declared column types: ``"gsq"``
        when every column is categorical, ``"cond_gauss"`` when some are, else ``"fisherz"``. A
        column is declared categorical by ``categorical`` or by a DataFrame's ``category``,
        ``bool``, or text type, never by its values.
        ``metadata["indep_test"]`` names the test the fit used: the one picked from the
        declared column types when this is ``None``.
    categorical : sequence of str or int, optional
        The categorical columns of ``data``, for a mixed-data test: column labels (strings) or
        0-based positions. ``None`` takes a DataFrame's ``category``, ``bool``, and text columns,
        never a column by its values. ``[]`` marks none. A categorical test reads every column as
        categorical, and a continuous test raises ``ValueError`` on a declared column, named here or
        by its type. Every lagged value of a column has its type.
    seed : int or None, default=None
        ``None``, the default, is the configured seed, ``ANDREY_SEED`` or ``0`` when it is unset,
        not a fresh random one, so a fit repeats. Seed for the random draws of ``indep_test``, which
        only ``"rcot"``, ``"cmiknn"``, and ``"gpdc"`` make; the other tests ignore it. With those
        tests, ``metadata["seed"]`` records the seed used.
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
        the mark at ``i`` (``-`` tail, ``<`` arrowhead, ``o`` circle, ``x`` conflict), ``-``, and
        the mark at ``j`` (``>``, ``-``, ``o``, or ``x``), such as ``"o->"``, ``"<->"``, or
        ``"-->"``; lag ``0`` lists both orders. ``metadata["conflicts"]`` lists the ``[i, k, j]``
        links with a conflict mark, which ``structure.lag(0)`` shows as a circle and
        ``structure.lag(k)`` leaves out. ``metadata["p_values"][k][i][j]`` is the largest p-value of
        the final round's tests of that pair, and ``metadata["statistics"][k][i][j]`` the statistic
        of that test (see Notes); lag ``0`` is symmetric with ``1`` and ``0`` on the diagonal and on
        pairs no test ran on, and a required link, never tested, reads ``0`` and ``1``. Under a
        mixed-data test, ``metadata["categorical"]`` lists the categorical columns it read; with a
        randomized ``indep_test``, ``metadata["seed"]`` is the seed used. ``metadata["parents"][j]``
        lists the ``[i, k]`` pairs (variable ``i`` at lag ``k``) the last preliminary round found to
        be parents of variable ``j``. ``ordering`` is ``None``, and ``metadata["algorithm"]`` is
        ``"lpcmci"``.

    Raises
    ------
    TypeError
        If ``knowledge`` is neither ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If an argument has the wrong type, lies outside its range, or is not one of its choices; the
        message names the method, the argument, what it accepts, and the value. Also if ``data`` is
        not a usable 2-D matrix with at least ``n_variables * (tau_max + 1) + 2 * tau_max + 1``
        rows, the length at which every test keeps a degree of freedom. With ``"fisherz"`` or
        ``"student_t"``, also if a variable repeats another up to scale and shift, at the same step
        or up to ``2 * tau_max`` steps apart, before any test runs; one message names every group of
        copies, a lagged value as ``'x' at lag k``, and says to keep one column per group. With
        ``"chisq"`` or ``"gsq"``, also if ``data`` holds a value that is not a whole number; with
        ``"degen_gauss"`` or ``"cond_gauss"``, if ``categorical`` names a column ``data`` lacks;
        and, with a continuous test, if a column is declared categorical, by ``categorical`` or by
        its type. Also if ``knowledge`` names a column ``data`` lacks, contradicts itself, or
        requires a link without a lag or beyond ``tau_max``.

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
    and every node a test conditions on lies within ``tau_max`` steps of the later end.
    Every test reads the lagged values as the columns of one design. ``"fisherz"`` and
    ``"student_t"`` report the partial correlation, ``"chisq"`` and ``"gsq"`` their chi-square
    statistic, ``"degen_gauss"`` and ``"cond_gauss"`` the likelihood ratio, ``"cmiknn"`` its
    estimate of the conditional mutual information, ``"gpdc"`` the squared distance correlation,
    ``"rcot"`` its statistic, and ``"kci"`` its statistic over the statistic's mean under
    independence. A candidate's strength is the smallest absolute statistic it has had, as in the
    authors' reference implementation, tigramite; strengths compare within one test only. A test
    registered with :func:`~andrey.core.ci.register_indep_test` that reports only p-values is
    asked one query at a time, its candidates are ranked by p-value, smallest first, and its
    statistic reads ``-log10(p)``.
    A test's partial correlation is computed from the correlation matrix, and from the least-squares
    residuals when the conditions are collinear to working precision. The defaults are the authors'
    published defaults. The earlier end of a lagged link is searched with the empty set only, since
    time order already rules out the later end as its cause.

    **Missing values.** A ``NaN`` in ``data`` marks a missing value. Every row of the lagged design
    that reads it holds it, and each test reads only the design rows where all of its lagged
    variables are observed (test-wise deletion) [2]_ [3]_, so the tests no longer share one set of
    time steps; ``metadata["rows_tested"]`` gives the fewest and the median design rows a test read.
    When values are missing completely at random, the rows a test keeps are a random sample, so each
    test keeps its level [3]_. A partial-correlation test with fewer than ``|S| + 3`` rows, another
    test with fewer than ``2``, or a test with a continuous column constant on its rows raises
    ``ValueError``. The categorical, mixed-data, and nonlinear tests delete rows the same way.

    **Limitations.** Measured on simulated linear-Gaussian time series with 30% of the series
    unobserved, of ``max(1000, 12 * (d + 1))`` time steps for ``d`` variables and of five times as
    many, the unobserved series counted among the variables, with ``tau_max=3``, on 4 cores with a
    cap of 1 h and 28 GB per fit. The largest sizes through which its median errors (endpoint SHD
    against the PAG of the observed series) stayed below the empty graph's, and through which its
    fits finished within the caps, were 40 and 40 variables with the shorter series or the series
    five times as long. At 40 variables with the shorter series, 1 of 10 fits ran past 1 h; at 40
    variables with the series five times as long, 2 of 5 fits ran past 1 h. Every ladder ended when
    three fits of the next size ran past 1 h. The median fit took 263 s at 40 variables with the
    series five times as long. With a partial-correlation test it assumes linear effects;
    ``"cmiknn"`` and ``"gpdc"`` lift that assumption at a far higher cost per test and are not
    benchmarked yet. It assumes a stationary series and no selection bias; hidden common causes are
    allowed. The graph covers ``tau_max + 1`` time steps, and a
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

    indep_test, categorical = default_test(data, indep_test, categorical=categorical)
    X, mask = checked_data(
        data,
        categorical,
        "indep_test",
        indep_test,
        method="lpcmci",
        missing=True,
    )
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
            indep_test=indep_test,
            max_p_global=max_p_global,
            constraints=constraints,
            categorical=mask,
            seed=seed,
        )
    metadata = {**categorical_metadata(mask, data), **rows_metadata(rows)}
    return with_labels(_adapt_lpcmci(result, metadata), labels)


@method_spec("granger")
def granger(
    data: npt.ArrayLike,
    *,
    tau_max: int = 1,
    alpha: float = 0.05,
    fdr_method: str = "fdr_bh",
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn the lagged causal graph of a multivariate time series by Granger causality.

    A variable Granger-causes another when its past improves the prediction of the other's
    present [1]_. Each variable is regressed by least squares on the past ``tau_max`` steps of every
    variable, a vector autoregression [2]_, and each lagged coefficient is tested against zero
    given every other lagged variable: the F-test of the regression with and without that
    regressor, the square of the coefficient's t-ratio. A lagged edge is a coefficient whose
    p-value is at most ``alpha``. Use it for one multivariate series whose effects take at least
    one time step.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        A multivariate time series, one row per time step. Coerced to ``float64``. A DataFrame's
        column names become the structure's ``labels``.
    tau_max : int, default=1
        The largest lag: each regression uses the past ``tau_max`` time steps, at least ``1``.
    alpha : float, default=0.05
        Significance level of the test of each lagged coefficient, in the open interval (0, 1): a
        larger value returns more edges.
    fdr_method : {"none", "fdr_bh"}, default="fdr_bh"
        How the p-values are corrected for the number of links tested. ``"fdr_bh"`` replaces them
        with their Benjamini-Hochberg adjusted values [3]_ over all the links together, every lag
        and every ordered pair of variables, a variable's link to its own past included. A link is
        then an edge when its adjusted p-value is at most ``alpha``, which holds the false
        discovery rate, the expected share of false edges among the returned edges, at or below
        ``alpha`` when the tests are independent or positively dependent. The adjusted p-value is
        never below the tested one, so ``"fdr_bh"`` can remove edges and never adds one, and
        ``metadata["p_values"]`` holds the adjusted values. ``"none"`` keeps each p-value as
        tested, so about ``alpha`` of the absent links are edges. A link the knowledge forbids is
        no test: it is not counted and keeps p-value ``1``; a required link is counted with
        p-value ``0``. See Notes.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`, read as links between lagged
        variables over lags ``1`` to ``tau_max``: required and forbidden links, the pairs allowed
        to be linked, and temporal tiers, all honored. A link ``(cause, effect, lag)`` is the link
        from ``cause`` at time ``t - lag`` to ``effect`` at time ``t``; a forbidden
        ``(cause, effect)`` pair forbids it at every lag, and a required link names its lag, from
        ``1`` to ``tau_max`` (the method learns no same-step links). Each variable's regression
        leaves out its forbidden lagged regressors, so the others are tested given the ones that
        stay, and a required link is in the graph whatever its test says, with the regression's
        coefficient as its weight. Tiers forbid every link from a later tier into an earlier one,
        and a pair left out of ``allowed`` has no link at any lag; a variable's links to its own
        past stay allowed.

    Returns
    -------
    StructureOutput
        Wraps a ``temporal`` :class:`~andrey.core.TemporalStructure` with lags ``1`` to ``tau_max``
        and no lag 0. ``structure.lag(k)`` is a ``digraph`` with the edge ``i -> j`` when
        ``x_i(t - k)`` is significant in the regression of ``x_j(t)``, self-loops included;
        ``structure.lag_weights[k - 1][i, j]`` is that regression coefficient, in the data's units,
        and ``0`` where there is no edge. ``ordering`` is ``None``, and ``metadata["algorithm"]``
        is ``"granger"``. ``metadata["p_values"][k - 1][i][j]`` is the p-value of the link
        ``i -> j`` at lag ``k``, for every pair, significant or not, adjusted under
        ``fdr_method="fdr_bh"``; a link the knowledge forbids reads p-value 1, and a required one 0.

    Raises
    ------
    TypeError
        If ``knowledge`` is neither ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If an argument has the wrong type, lies outside its range, or is not one of its choices; the
        message names the method, the argument, what it accepts, and the value. Also if ``data`` has
        fewer than ``(n_variables + 1) * tau_max + 2`` rows, so that the regression has no degree of
        freedom left, or fewer than 2 columns, or if a regression is singular, as when one variable
        is a linear combination of others. Also if ``knowledge`` names a column ``data`` lacks,
        contradicts itself, or requires a link without a lag or beyond ``tau_max``.

    Notes
    -----
    An edge says that the past of one variable helps predict another beyond the past of every
    variable in ``data``. Granger causality misses effects within one time step, and a common
    cause left out of ``data`` can produce edges.

    The test of each coefficient is the F-test of one regressor in Granger's regression [1]_, the
    test a vector autoregression reports for a coefficient [2]_. Its p-value takes the coefficient's
    t-ratio against the Student ``t`` distribution with the regression's residual degrees of
    freedom, which is exact for Gaussian noise; statsmodels' ``VARResults.pvalues`` takes the same
    t-ratio against the normal distribution, which gives the same p-values as the series grows. Each
    link is tested at ``alpha``, and ``fdr_method="none"`` makes no correction for the
    ``n_variables**2 * tau_max`` tests, so on a series with many variables about ``alpha`` of the
    absent links appear as edges, which the default ``"fdr_bh"`` avoids; with ``"none"``, lower
    ``alpha`` or judge the edges by their p-values and weights. The regressions are fitted on
    standardized variables, so the edges do not depend on the units of any variable; the
    coefficients are converted back to the data's units. The fit is deterministic and has no random
    component.

    **Choosing fdr_method.** The method tests ``n_variables ** 2 * tau_max`` links, each at level
    ``alpha``. With ``"none"`` about ``alpha`` of the absent links are returned by chance, so the
    extra edges grow with the number of variables (see Limitations). ``"fdr_bh"`` [3]_ sorts the
    ``m`` p-values of the links and gives the one of rank ``k`` the smallest value, over its rank
    and every higher one, of ``p * m / rank``, at most ``1``; equal p-values get equal results. It
    returns fewer edges than ``"none"``: the false ones, and also true ones whose p-value is not
    small enough for the number of tests (see Limitations). The guarantee of [3]_ needs independent
    or positively dependent tests, which the coefficient tests of one vector autoregression are not
    shown to be.

    **Missing values.** A ``NaN`` in ``data`` raises ``ValueError``: each variable's regression
    reads every row of the lagged design, so a step with a missing value has no place in it. Fill
    in the missing steps first, as dropping one would join the steps around it, or use
    :func:`~andrey.pcmci`, :func:`~andrey.pcmciplus`, or :func:`~andrey.lpcmci`, whose tests delete
    rows per test.

    **Limitations.** On 20 simulated linear-Gaussian VAR series per cell, with 2 expected parents of
    each variable at each lag, no same-step links, and standardized, from 5 to 80 variables, with
    500 and 1,000 time steps, and ``tau_max`` of 1 and 2 (20 cells), the default
    ``fdr_method="fdr_bh"`` had fewer mean errors (missing plus extra lagged edges) than ``"none"``
    in 15 cells, by the paired 95% bootstrap interval of the difference, the same in 4, and more in
    1: 11.9 against 10.9 with 5 variables, 500 steps, and ``tau_max=2``. From 20 variables on it was
    better in every cell. At 80 variables, 1,000 steps, and ``tau_max=2``, ``"none"`` had 768
    errors, above the empty graph's 481, with 603 extra edges and 165 missed true edges on average;
    ``"fdr_bh"`` had 256, with 13 extra and 243 missed. The mean errors were above the empty graph's
    in 5 of the 20 cells with ``"none"``, in none with ``"fdr_bh"``. ``"fdr_bh"`` trades missed true
    edges for far fewer extra ones: with 5 to 10 variables the two differ by at most 2.7 errors, and
    with 5 variables and ``tau_max=2`` ``"fdr_bh"`` removed the extra edges (1.1 to 0.4 at 500
    steps) and missed more true ones (9.8 to 11.5). On 10 series of 160 variables with 1,000 steps
    and ``tau_max=2``, the median errors were 2,855 with ``"none"``, above the empty graph's 973,
    and 546 with ``"fdr_bh"``. It assumes linear effects and tests no same-step effects. The work
    grows with ``tau_max`` and with the square of the number of variables. The numbers are a
    reference, not a hard limit: they change with the data generator, the density, the method's
    settings, and the machine.

    References
    ----------
    .. [1] Granger, C. W. J. (1969). Investigating causal relations by econometric models and
       cross-spectral methods. Econometrica 37(3), 424-438.
    .. [2] Lutkepohl, H. (2005). New Introduction to Multiple Time Series Analysis. Springer.
    .. [3] Benjamini, Y., and Hochberg, Y. (1995). Controlling the false discovery rate: a
       practical and powerful approach to multiple testing. Journal of the Royal Statistical
       Society B 57(1), 289-300.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import granger
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal(2000)
    >>> y = np.r_[0.0, 0.8 * x[:-1]] + rng.standard_normal(2000)
    >>> out = granger(np.column_stack([x, y]))
    >>> round(float(out.structure.lag_weights[0][0, 1]), 1)
    0.8
    >>> bool(out.metadata["p_values"][0][0][1] < 0.05)
    True
    """
    from andrey.temporal.granger import granger_f_test

    X = data_matrix(data, min_rows=2, method="granger", fix=FILL_STEPS)
    if X.shape[1] < 2:
        raise ValueError(f"data needs at least 2 columns (variables), got {X.shape[1]}")
    min_rows = (X.shape[1] + 1) * tau_max + 2  # a degree of freedom left in every regression
    if X.shape[0] < min_rows:
        raise ValueError(
            f"data needs at least {min_rows} rows (time steps) for tau_max={tau_max} over "
            f"{X.shape[1]} variables, got {X.shape[0]}"
        )
    labels = column_labels(data)
    constraints = resolve_lagged(
        knowledge, X.shape[1], labels, tau_max, method="granger", contemporaneous=False
    )
    required = None if constraints is None else constraints.required[1:]
    peak = np.abs(X).max(axis=0)  # divide first, so tiny or huge units cannot underflow the std
    spread = (X / peak).std(axis=0)  # data_matrix rejects a constant column
    coeff, p_values = granger_f_test(
        (X / peak - (X / peak).mean(axis=0)) / spread,
        maxlag=tau_max,
        alpha=alpha,
        fdr_method=fdr_method,
        forbidden=None if constraints is None else constraints.forbidden[1:],
        required=required,
    )
    scale = spread * peak
    # coeff[i, j + d*k] weights standardized x_j(t-k-1) in standardized x_i(t): rescale to units.
    # Units far apart can overflow the ratio; a zero weight stays zero rather than becoming NaN.
    with np.errstate(over="ignore", invalid="ignore"):
        ratio = scale[:, None] / np.tile(scale, tau_max)[None, :]
        coeff = np.where(coeff == 0, 0.0, coeff * ratio)
    return with_labels(_adapt_granger(coeff, X.shape[1], tau_max, required, p_values), labels)
