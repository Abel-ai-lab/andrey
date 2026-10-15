"""Facades for the pairwise direction tests (ANM and PNL) and the hidden-confounder methods.

ANM and PNL test both directions in a two-column table. RCD, CAM-UV, and BottomUpParceLiNGAM
return a PAG over the observed variables. RCD and CAM-UV list the pairs a hidden variable links in
``metadata["confounded_pairs"]``; BottomUpParceLiNGAM leaves the variables whose order such a
cause hides unordered.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._adapt import column_labels, data_matrix, int_in_range, open_unit_interval, with_labels
from ._experimental import warn_experimental
from ._irregular import (
    _adapt_anm,
    _adapt_bottom_up_parce,
    _adapt_camuv,
    _adapt_pnl,
    _adapt_rcd,
)

if TYPE_CHECKING:
    import numpy.typing as npt

    from andrey.core import StructureOutput


def anm(data: npt.ArrayLike, *, alpha: float | None = None) -> StructureOutput:
    """Test the direction of a variable pair with the additive noise model (ANM).

    ANM [1]_ fits ``y = f(x) + e`` in each direction by Gaussian-process regression. An HSIC test
    [2]_ checks whether the residual ``e`` is independent of the proposed cause, which holds only in
    the causal direction. The two p-values determine the edge, and the two test statistics decide a
    tie.

    Parameters
    ----------
    data : array-like of shape (n_samples, 2)
        The two variables, one per column: column 0 is ``x`` and column 1 is ``y``. Coerced to
        ``float64``. A DataFrame's column names become the structure's ``labels``.
    alpha : float or None, default=None
        ``None`` chooses the direction with the larger p-value, or with the smaller test statistic
        when the p-values are equal. A significance level such as ``0.05`` draws an edge only when
        exactly one p-value exceeds it. See Notes.

    Returns
    -------
    StructureOutput
        Contains a two-node :class:`~andrey.core.GraphStructure` of kind ``dag`` with the chosen
        edge, or no edge. ``metadata["algorithm"]`` is ``"ANM"``; ``pval_forward`` is the
        independence p-value for ``x -> y`` and ``pval_backward`` is the p-value for ``y -> x``;
        ``stat_forward`` and ``stat_backward`` are the HSIC statistics of the same two tests, each
        divided by its mean under independence.

    Raises
    ------
    ValueError
        If ``data`` does not have 2 columns, or if ``alpha`` is not in ``(0, 1)``.

    Notes
    -----
    With ``alpha=None``, the edge points in the direction with the larger p-value. On a large
    sample, or a pair that fits neither direction, both p-values can be zero; equal p-values leave
    the choice to the HSIC statistics, each divided by its mean under independence, and the edge
    points in the direction with the smaller one, the weaker dependence between the proposed cause
    and its residual. Equal statistics give no edge. The rule therefore produces an edge even for
    unrelated variables, a deterministic pair whose p-values are both near zero, or a pair whose
    direction cannot be identified. Mooij et al. (2016) [3]_ compare p-values to avoid choosing a
    significance level and found the comparison among the best methods in their benchmark.

    With a significance level, only the p-values decide; a direction passes when its p-value
    exceeds ``alpha``:

    - only ``x -> y`` passes: the edge ``x -> y``;
    - only ``y -> x`` passes: the edge ``y -> x``;
    - both pass: no edge, since neither direction is rejected;
    - neither passes: no edge, since both directions are rejected.

    Hoyer et al. (2008) [1]_ use a level of 0.02. Both p-values and both statistics remain in
    ``metadata``.

    The direction is identifiable when ``f`` is nonlinear or the noise is non-Gaussian; a linear
    pair with Gaussian noise fits an additive noise model in both directions [1]_.

    Each column is standardized to zero mean and unit variance first, so the result does not
    depend on the units of either variable. The regression is a Gaussian process with a scaled RBF
    kernel plus a white-noise term, its hyperparameters fitted by maximum marginal likelihood. The
    HSIC test uses Gaussian kernels and a gamma approximation of its null distribution. The fit is
    deterministic.

    **Limitations.** Measured on simulated cause-effect pairs with a nonlinear additive-noise
    mechanism and Gaussian noise, from 200 to 8,000 rows. It named the true direction on three pairs
    in four at 200 rows and on every pair from 4,000 rows. The median fit took over three minutes at
    8,000 rows, on 4 cores. Each regression and HSIC test builds matrices with one row and one
    column per sample, so memory grows with the square of ``n_samples`` and the regression's time
    with its cube.

    References
    ----------
    .. [1] Hoyer, P. O., Janzing, D., Mooij, J. M., Peters, J., and Scholkopf, B. (2008).
       Nonlinear causal discovery with additive noise models. NeurIPS 21, 689-696.
    .. [2] Gretton, A., Fukumizu, K., Teo, C. H., Song, L., Scholkopf, B., and Smola, A. J. (2008).
       A kernel statistical test of independence. NeurIPS 20, 585-592.
    .. [3] Mooij, J. M., Peters, J., Janzing, D., Zscheischler, J., and Scholkopf, B. (2016).
       Distinguishing cause from effect using observational data: methods and benchmarks. JMLR
       17(32), 1-102.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import anm
    >>> rng = np.random.default_rng(0)
    >>> x = rng.uniform(-1, 1, 300)
    >>> y = x * x * x + 0.3 * rng.uniform(-1, 1, 300)
    >>> out = anm(np.column_stack([x, y]))
    >>> out.structure.oriented_edges()
    [(0, 1, 'directed')]
    >>> out.metadata["pval_forward"] > 0.05 >= out.metadata["pval_backward"]
    True
    """
    from andrey.irregular.anm import anm as _native

    if np.ndim(data) != 2 or np.shape(data)[1] != 2:
        raise ValueError(
            f"anm expects an (n_samples, 2) table of two variables, got {np.shape(data)}"
        )
    X = data_matrix(data)
    if alpha is not None:
        alpha = open_unit_interval(alpha, "alpha")
    labels = column_labels(data)
    X = X / np.abs(X).max(axis=0)  # first, so a column in tiny or huge units cannot underflow
    Z = (X - X.mean(axis=0)) / X.std(axis=0)  # data_matrix rejects a constant column
    pval_forward, pval_backward, stat_forward, stat_backward = _native(Z[:, 0], Z[:, 1])
    return _adapt_anm(
        pval_forward,
        pval_backward,
        stat_forward=stat_forward,
        stat_backward=stat_backward,
        alpha=alpha,
        labels=labels,
    )


def pnl(data: npt.ArrayLike, *, alpha: float | None = None) -> StructureOutput:
    """Test the direction of a variable pair with the post-nonlinear (PNL) model.

    PNL [1]_ [2]_ fits ``y = f2(f1(x) + e)`` in each direction. An HSIC test checks whether the
    estimated noise ``e`` is independent of the proposed cause. The two p-values determine the edge,
    and the two test statistics decide a tie.

    Parameters
    ----------
    data : array-like of shape (n_samples, 2)
        The two variables, one per column: column 0 is ``x`` and column 1 is ``y``. Coerced to
        ``float64``. A DataFrame's column names become the structure's ``labels``.
    alpha : float or None, default=None
        ``None`` chooses the direction with the larger p-value, or with the smaller test statistic
        when the p-values are equal. A significance level such as ``0.05`` draws an edge only when
        exactly one p-value exceeds it. See Notes.

    Returns
    -------
    StructureOutput
        Contains a two-node :class:`~andrey.core.GraphStructure` of kind ``dag`` with the chosen
        edge, or no edge. ``metadata["algorithm"]`` is ``"PNL"``; ``pval_forward`` is the
        independence p-value for ``x -> y`` and ``pval_backward`` is the p-value for ``y -> x``;
        ``stat_forward`` and ``stat_backward`` are the HSIC statistics of the same two tests, each
        divided by its mean under independence.

    Raises
    ------
    ValueError
        If ``data`` does not have 2 columns, or if ``alpha`` is not in ``(0, 1)``.

    Notes
    -----
    With ``alpha=None``, the edge points in the direction with the larger p-value. On a large
    sample, or a pair that fits neither direction, both p-values can be zero; equal p-values leave
    the choice to the HSIC statistics, each divided by its mean under independence, and the edge
    points in the direction with the smaller one, the weaker dependence between the proposed cause
    and its estimated noise. Equal statistics give no edge. The rule therefore produces an edge
    even for unrelated variables or a pair whose direction cannot be identified, such as a linear
    Gaussian pair. Mooij et al. (2016) [4]_ compare p-values to avoid choosing a significance level
    and found the comparison among the best methods in their benchmark of additive-noise models.

    With a significance level, only the p-values decide; a direction passes when its p-value
    exceeds ``alpha``:

    - only ``x -> y`` passes: the edge ``x -> y``;
    - only ``y -> x`` passes: the edge ``y -> x``;
    - both pass: no edge, since neither direction is rejected;
    - neither passes: no edge, since both directions are rejected.

    Zhang and Hyvarinen (2010) [2]_ use a level of 0.01 for PNL; Hoyer et al. (2008) [3]_ use 0.02
    for additive-noise models. The result depends on ``alpha``. Estimated noise can appear dependent
    on the true cause, so a standard level can reject the true direction; no theory establishes
    the right level (Mooij et al., 2016). Both p-values and both statistics remain in
    ``metadata``.

    PNL fits each transform as a sum of smooth basis functions, so the fit is deterministic but can
    miss a sharply varying nonlinearity.

    **Limitations.** Measured on simulated cause-effect pairs with a post-nonlinear mechanism, a
    linear inner transform and an invertible nonlinear outer one, and Laplace noise, from 200 to
    5,000 rows. It named the true direction on three pairs in four at 200 rows and on every pair
    from 1,000 rows. The median fit took under 2 s at 5,000 rows, on 4 cores. Each HSIC test builds
    kernel matrices with one row and one column per sample, so time and memory grow with the square
    of the row count.

    References
    ----------
    .. [1] Zhang, K., and Hyvarinen, A. (2009). On the identifiability of the post-nonlinear
       causal model. UAI, 647-655.
    .. [2] Zhang, K., and Hyvarinen, A. (2010). Distinguishing causes from effects using nonlinear
       acyclic causal models. JMLR Workshop and Conference Proceedings 6, 157-164.
    .. [3] Hoyer, P. O., Janzing, D., Mooij, J. M., Peters, J., and Scholkopf, B. (2008).
       Nonlinear causal discovery with additive noise models. NeurIPS 21, 689-696.
    .. [4] Mooij, J. M., Peters, J., Janzing, D., Zscheischler, J., and Scholkopf, B. (2016).
       Distinguishing cause from effect using observational data: methods and benchmarks. JMLR
       17(32), 1-102.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import pnl
    >>> rng = np.random.default_rng(2)
    >>> x = rng.uniform(-1, 1, 300)
    >>> y = np.tanh(x**3 + 0.3 * rng.uniform(-1, 1, 300))
    >>> out = pnl(np.column_stack([x, y]))
    >>> out.structure.oriented_edges()
    [(0, 1, 'directed')]
    >>> out.metadata["pval_forward"] > 0.05 >= out.metadata["pval_backward"]
    True
    """
    from andrey.irregular.pnl import pnl as _native

    if np.ndim(data) != 2 or np.shape(data)[1] != 2:
        raise ValueError(
            f"pnl expects an (n_samples, 2) table of two variables, got {np.shape(data)}"
        )
    X = data_matrix(data)
    if alpha is not None:
        alpha = open_unit_interval(alpha, "alpha")
    labels = column_labels(data)
    pval_forward, pval_backward, stat_forward, stat_backward = _native(X[:, 0], X[:, 1])
    return _adapt_pnl(
        pval_forward,
        pval_backward,
        stat_forward=stat_forward,
        stat_backward=stat_backward,
        alpha=alpha,
        labels=labels,
    )


def rcd(
    data: npt.ArrayLike,
    *,
    max_explanatory_vars: int = 2,
    cor_alpha: float = 0.01,
    ind_alpha: float = 0.01,
    shapiro_alpha: float = 0.01,
) -> StructureOutput:
    """Learn a linear non-Gaussian graph with hidden common causes by repetitive causal discovery.

    RCD [1]_ repeatedly regresses variables on candidate ancestors and tests the residuals for
    independence, to find each variable's ancestors, then its parents, and finally the pairs that
    share a hidden common cause.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observations, one variable per column. A DataFrame's column names become the structure's
        ``labels``.
    max_explanatory_vars : int, default=2
        Most variables regressed out together when testing whether one variable is an ancestor of
        another, at least 1. Larger values find more ancestors and take longer.
    cor_alpha : float, default=0.01
        Significance level of the Pearson correlation tests, in ``(0, 1)``.
    ind_alpha : float, default=0.01
        Significance level of the HSIC independence tests, in ``(0, 1)``.
    shapiro_alpha : float, default=0.01
        Significance level of the Shapiro-Wilk tests that decide whether a residual is
        non-Gaussian, in ``(0, 1)``.

    Returns
    -------
    StructureOutput
        A :class:`~andrey.core.GraphStructure` of kind ``pag``: ``i -> j`` for a direct cause and
        ``i <-> j`` for a pair with a hidden common cause. ``metadata["algorithm"]`` is ``"RCD"``;
        ``metadata["confounded_pairs"]`` lists the ``<->`` pairs as sorted ``[i, j]`` lists;
        ``weighted_adjacency`` holds the edge coefficients.

    Raises
    ------
    TypeError
        If ``max_explanatory_vars`` is not an int.
    ValueError
        If ``data`` has fewer than 6 rows, if ``max_explanatory_vars`` is less than 1, or if a
        significance level is not in ``(0, 1)``.

    Notes
    -----
    RCD assumes linear relations and non-Gaussian noise. It needs more rows than DirectLiNGAM: its
    tests compare residuals, and a weak hidden cause leaves only a small trace in them.

    **Limitations.** Measured on linear data with uniform noise, one node in five hidden, and 2,000
    rows, from 4 to 24 observed variables: its median directed-edge errors were below the empty
    graph's at every size but 8 variables, where they equaled it. A fit took about 2 s at 4
    variables and 18 s at 24, on two cores. Larger problems are untested; the work grows with the
    number of variables and with ``max_explanatory_vars``.

    References
    ----------
    .. [1] Maeda, T. N., and Shimizu, S. (2020). RCD: Repetitive causal discovery of linear
       non-Gaussian acyclic models with latent confounders. AISTATS, PMLR 108, 735-745.

    Examples
    --------
    >>> import numpy as np
    >>> import andrey
    >>> rng = np.random.default_rng(0)
    >>> u = lambda: rng.uniform(-1, 1, 2000)
    >>> h = u()  # hidden: a common cause of x0 and x1
    >>> x0, x1 = 1.5 * h + u(), 1.5 * h + u()
    >>> data = np.column_stack([x0, x1, x1 + u(), x0 + u()])
    >>> out = andrey.rcd(data)
    >>> out.metadata["confounded_pairs"]
    [[0, 1]]
    """
    from andrey.irregular.rcd import fit_rcd

    X = data_matrix(data, min_rows=6)  # HSIC's null variance has (n - 4)(n - 5)
    max_explanatory_vars = int_in_range(max_explanatory_vars, "max_explanatory_vars", 1)
    model = fit_rcd(
        X,
        max_explanatory_num=max_explanatory_vars,
        cor_alpha=open_unit_interval(cor_alpha, "cor_alpha"),
        ind_alpha=open_unit_interval(ind_alpha, "ind_alpha"),
        shapiro_alpha=open_unit_interval(shapiro_alpha, "shapiro_alpha"),
    )
    return with_labels(_adapt_rcd(model), column_labels(data))


def camuv(
    data: npt.ArrayLike, *, alpha: float = 0.01, max_explanatory_vars: int = 3
) -> StructureOutput:
    """Learn a nonlinear additive graph with hidden variables by CAM-UV.

    CAM-UV [1]_ extends causal additive models: each variable is a sum of nonlinear functions of
    its parents plus noise. It finds each variable's direct parents, and the pairs connected
    through an unobserved variable.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observations, one variable per column. A DataFrame's column names become the structure's
        ``labels``.
    alpha : float, default=0.01
        Significance level of the HSIC independence tests, in ``(0, 1)``.
    max_explanatory_vars : int, default=3
        Largest set of variables tested together, the variable itself included, at least 2: parent
        sets have up to ``max_explanatory_vars - 1`` variables. Larger values find more parents and
        take longer.

    Returns
    -------
    StructureOutput
        A :class:`~andrey.core.GraphStructure` of kind ``pag``: ``i -> j`` for a direct cause and
        ``i <-> j`` for a pair connected through an unobserved variable.
        ``metadata["algorithm"]`` is ``"CAMUV"``; ``metadata["confounded_pairs"]`` lists the
        ``<->`` pairs as sorted ``[i, j]`` lists.

    Raises
    ------
    TypeError
        If ``max_explanatory_vars`` is not an int.
    ValueError
        If ``data`` has fewer than 6 rows, if ``alpha`` is not in ``(0, 1)``, or if
        ``max_explanatory_vars`` is less than 2.

    Warns
    -----
    ExperimentalWarning
        On the first call in a process, because the method is experimental: it has no published
        benchmark, and its API may change without deprecation.

    Notes
    -----
    A ``<->`` pair may share a hidden common cause or be linked by a causal path through a hidden
    variable; CAM-UV does not tell the two apart. It estimates no edge weights.

    **Limitations.** Not benchmarked yet. Its tests run it on at most 6 variables with 300 rows, and
    on 1,000 rows with 5 variables. It searches every subset of up to ``max_explanatory_vars``
    variables, and starts the search over after each parent it adds. Each HSIC test builds kernel
    matrices with one row and one column per sample, so time and memory grow with the square of the
    row count.

    References
    ----------
    .. [1] Maeda, T. N., and Shimizu, S. (2021). Causal additive models with unobserved variables.
       UAI, PMLR 161, 97-106.

    Examples
    --------
    >>> import numpy as np
    >>> import andrey
    >>> rng = np.random.default_rng(0)
    >>> u = lambda: rng.uniform(-1, 1, 1000)
    >>> h = u()  # hidden: a common cause of x0 and x1
    >>> x0, x1 = np.sin(3 * h) + 0.3 * u(), np.cos(3 * h) + 0.3 * u()
    >>> data = np.column_stack([x0, x1, np.tanh(2 * x1) + 0.3 * u(), np.tanh(2 * x0) + 0.3 * u()])
    >>> out = andrey.camuv(data)
    >>> out.metadata["confounded_pairs"]
    [[0, 1]]
    """
    warn_experimental("camuv")
    from andrey.irregular.camuv import camuv as _native

    X = data_matrix(data, min_rows=6)  # as rcd: HSIC on fewer rows decides nothing
    alpha = open_unit_interval(alpha, "alpha")
    max_explanatory_vars = int_in_range(max_explanatory_vars, "max_explanatory_vars", 2)
    parents, confounded = _native(X, alpha=alpha, max_explanatory_vars=max_explanatory_vars)
    return with_labels(_adapt_camuv(parents, confounded, X.shape[1]), column_labels(data))


def bottom_up_parce_lingam(data: npt.ArrayLike, *, alpha: float = 0.1) -> StructureOutput:
    """Learn a linear non-Gaussian causal order that tolerates hidden common causes.

    BottomUpParceLiNGAM [1]_ finds sinks one at a time, from the last variable in the causal order
    up. Where a hidden common cause hides the order, it leaves the variables unordered rather than
    guessing.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observations, one variable per column. A DataFrame's column names become the structure's
        ``labels``.
    alpha : float, default=0.1
        Significance level of the independence test that accepts a sink, Bonferroni-corrected
        across the remaining variables, in ``(0, 1)``.

    Returns
    -------
    StructureOutput
        A :class:`~andrey.core.GraphStructure` of kind ``pag``: ``i -> j`` for a direct cause and
        ``i o-o j`` between variables whose order a hidden common cause leaves unresolved.
        ``metadata["algorithm"]`` is ``"BottomUpParceLiNGAM"``; ``metadata["causal_order"]`` gives
        the order, with the unordered variables nested in one list; ``weighted_adjacency`` holds
        the edge coefficients.

    Raises
    ------
    ValueError
        If ``data`` has fewer than 6 rows or no more rows than columns, or if ``alpha`` is not in
        ``(0, 1)``.

    Notes
    -----
    The search finds sinks from the bottom up and stops at the first variable it cannot separate,
    leaving it and every variable above it in one unordered block: the variables a hidden cause
    confounds, their observed ancestors, and any other confounded pairs. An ``o-o`` pair need not
    share a cause, and edges among the block's variables stay unoriented. The edge weights come
    from an adaptive-Lasso regression of each variable on those before it, which needs more rows
    than columns.

    **Limitations.** Suited to a few variables. Measured on linear data with uniform noise, one node
    in five hidden, and 2,000 rows: its median directed-edge errors were below the empty graph's at
    4 observed variables, and above it at every size from 8 to 24. From 16 variables it left none of
    the confounded pairs unordered. A fit took about 2 s at 4 variables and 96 s at 24, on two
    cores.

    References
    ----------
    .. [1] Tashiro, T., Shimizu, S., Hyvarinen, A., and Washio, T. (2014). ParceLiNGAM: A causal
       ordering method robust against latent confounders. Neural Computation 26(1), 57-83.

    Examples
    --------
    >>> import numpy as np
    >>> import andrey
    >>> rng = np.random.default_rng(0)
    >>> u = lambda: rng.uniform(-1, 1, 2000)
    >>> h = u()  # hidden: a common cause of x0 and x1
    >>> x0, x1 = 1.5 * h + u(), 1.5 * h + u()
    >>> data = np.column_stack([x0, x1, x1 + u(), x0 + u()])
    >>> out = andrey.bottom_up_parce_lingam(data, alpha=0.01)
    >>> out.metadata["causal_order"]
    [[0, 1], 2, 3]
    """
    from andrey.irregular.bottom_up_parce import bottom_up_parce_lingam as _native

    X = data_matrix(data, min_rows=6)  # HSIC's null variance has (n - 4)(n - 5)
    if X.shape[0] <= X.shape[1]:
        raise ValueError(
            f"data needs more rows than columns, got {X.shape[0]} rows and {X.shape[1]} columns"
        )
    model = _native(X, alpha=open_unit_interval(alpha, "alpha"))
    return with_labels(_adapt_bottom_up_parce(model), column_labels(data))
