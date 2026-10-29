"""Facades for the pairwise direction tests (ANM and PNL) and the hidden-confounder methods.

ANM and PNL test both directions in a two-column table. RCD and BottomUpParceLiNGAM return a PAG
over the observed variables. RCD lists the pairs a hidden variable links in
``metadata["confounded_pairs"]``; BottomUpParceLiNGAM leaves the variables whose order such a
cause hides unordered.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from andrey.api._method import method_spec
from andrey.core.ci_missing import rows_metadata
from andrey.core.knowledge import Knowledge, resolve

from ._adapt import (
    column_labels,
    column_namer,
    data_matrix,
    with_labels,
)
from ._irregular import (
    _adapt_anm,
    _adapt_bottom_up_parce,
    _adapt_pnl,
    _adapt_rcd,
)

if TYPE_CHECKING:
    import numpy.typing as npt

    from andrey.core import StructureOutput


def _complete_pairs(
    X: np.ndarray, labels: tuple[str, ...] | None, method: str
) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    """Return the rows of the pair ``X`` where both columns are observed, and ``rows_tested``.

    Without a missing value, ``X`` returns as it is with no metadata. Both tests of the pair read
    the kept rows, which for two columns are the rows test-wise deletion keeps. Raises
    ``ValueError`` when fewer than 2 rows are kept, or a column is constant on them.
    """
    observed = ~np.isnan(X).any(axis=1)
    if observed.all():
        return X, {}
    kept = X[observed]
    rows = kept.shape[0]
    if rows < 2:
        raise ValueError(
            f"{method} needs at least 2 rows where both columns are observed, got {rows}"
        )
    flat = np.flatnonzero(np.ptp(kept, axis=0) == 0).tolist()
    if flat:
        raise ValueError(
            f"{method}: data column {column_namer(labels)(flat[0])} is constant on the {rows} rows "
            "where both columns are observed, so no direction can be told; drop the rows that "
            "hold the missing values"
        )
    return kept, rows_metadata({rows: 2})


@method_spec("anm")
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
        edge, or no edge. ``metadata["algorithm"]`` is ``"anm"``. ``metadata["p_values"]`` holds
        the independence p-value for ``x -> y`` under ``"forward"`` and for ``y -> x`` under
        ``"backward"``; ``metadata["statistics"]`` holds the HSIC statistics of the same two tests,
        each divided by its mean under independence, under the same keys.

    Raises
    ------
    ValueError
        If an argument has the wrong type, lies outside its range, or is not one of its choices; the
        message names the method, the argument, what it accepts, and the value. Also if ``data``
        does not have 2 columns.

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

    **Missing values.** A ``NaN`` in ``data`` marks a missing value. Both tests read only the rows
    where both columns are observed, which for a pair are the rows test-wise deletion keeps;
    ``metadata["rows_tested"]`` gives their count as both ``min`` and ``median``. When values are
    missing completely at random, those rows are a random sample of all rows. Fewer than 2 such
    rows, or a column constant on them, raises ``ValueError``.

    **Limitations.** Measured on simulated cause-effect pairs with a nonlinear additive-noise
    mechanism and Gaussian noise, with rows doubling from 1,000, on 4 cores with a cap of 1 h and 28
    GB per fit. It named the true direction on 95% to 100% of the 20 pairs at every size through
    16,000 rows, the largest its fits finished within the caps. The ladder ended when three fits of
    the next size ran out of the 28 GB. The median fit took 380 s at 8,000 rows. At 200 rows it
    named the true direction on three pairs in four. Each regression and HSIC test builds matrices
    with one row and one column per sample, so memory grows with the square of ``n_samples`` and the
    regression's time with its cube. Missing values are not benchmarked yet; the tests run them on
    pairs with up to a fifth of the values missing completely at random. The numbers are a
    reference, not a hard limit: they change with the data generator, the density, the method's
    settings, and the machine.

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
    >>> out.metadata["p_values"]["forward"] > 0.05 >= out.metadata["p_values"]["backward"]
    True
    """
    from andrey.irregular.anm import anm as _native

    if np.ndim(data) != 2 or np.shape(data)[1] != 2:
        raise ValueError(
            f"anm expects an (n_samples, 2) table of two variables, got {np.shape(data)}"
        )
    X = data_matrix(data, method="anm", missing=True)
    labels = column_labels(data)
    X, rows = _complete_pairs(X, labels, "anm")
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
        extra=rows,
    )


@method_spec("pnl")
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
        edge, or no edge. ``metadata["algorithm"]`` is ``"pnl"``. ``metadata["p_values"]`` holds
        the independence p-value for ``x -> y`` under ``"forward"`` and for ``y -> x`` under
        ``"backward"``; ``metadata["statistics"]`` holds the HSIC statistics of the same two tests,
        each divided by its mean under independence, under the same keys.

    Raises
    ------
    ValueError
        If an argument has the wrong type, lies outside its range, or is not one of its choices; the
        message names the method, the argument, what it accepts, and the value. Also if ``data``
        does not have 2 columns.

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

    The kernel matrices of the two HSIC tests run on a CUDA GPU, in float64, under the ``cuda``
    backend, or under ``auto`` from ``ANDREY_PNL_GPU_THRESHOLD`` rows squared once the GPU is
    running in the process; the fits stay on the CPU, and the p-values then match the CPU's to
    rounding.

    On the CPU, ``n_jobs`` above 1 (``andrey.config`` or ``ANDREY_N_JOBS``) fits the two
    directions on two threads, each limited to one BLAS thread; with one worker the BLAS threads
    are not limited. On a machine with several cores the fits then round differently, so the
    p-values and statistics with one worker and with several can differ in their last digits, and
    the direction can differ when the two p-values nearly tie. Each thread holds three ``n x n``
    float64 matrices; when the threads asked for would hold more than 4 GiB, fewer run, with a
    ``PerformanceWarning``.

    **Missing values.** A ``NaN`` in ``data`` marks a missing value. Both tests read only the rows
    where both columns are observed, which for a pair are the rows test-wise deletion keeps;
    ``metadata["rows_tested"]`` gives their count as both ``min`` and ``median``. When values are
    missing completely at random, those rows are a random sample of all rows. Fewer than 2 such
    rows, or a column constant on them, raises ``ValueError``.

    **Limitations.** Measured on simulated cause-effect pairs with a post-nonlinear mechanism, a
    linear inner transform and an invertible nonlinear outer one, and Laplace noise, with rows
    doubling from 1,000, on 4 cores with a cap of 1 h and 28 GB per fit. It named the true direction
    on 100% of the 20 pairs at every size through 32,000 rows, the largest its fits finished within
    the caps. The ladder ended when three fits of the next size ran out of the 28 GB. The median fit
    took 80 s at 32,000 rows. At 200 rows it named the true direction on three pairs in four. Each
    HSIC test builds kernel matrices with one row and one column per sample, so time and memory grow
    with the square of the row count: each worker thread holds three such float64 matrices, 600 MB
    at 5,000 rows. Missing values are not benchmarked yet; the tests run them on pairs with up to a
    fifth of the values missing completely at random. The numbers are a reference, not a hard limit:
    they change with the data generator, the density, the method's settings, and the machine.

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
    >>> out.metadata["p_values"]["forward"] > 0.05 >= out.metadata["p_values"]["backward"]
    True
    """
    from andrey.irregular.pnl import pnl as _native

    if np.ndim(data) != 2 or np.shape(data)[1] != 2:
        raise ValueError(
            f"pnl expects an (n_samples, 2) table of two variables, got {np.shape(data)}"
        )
    X = data_matrix(data, method="pnl", missing=True)
    labels = column_labels(data)
    X, rows = _complete_pairs(X, labels, "pnl")
    pval_forward, pval_backward, stat_forward, stat_backward = _native(X[:, 0], X[:, 1])
    return _adapt_pnl(
        pval_forward,
        pval_backward,
        stat_forward=stat_forward,
        stat_backward=stat_backward,
        alpha=alpha,
        labels=labels,
        extra=rows,
    )


@method_spec("rcd")
def rcd(
    data: npt.ArrayLike,
    *,
    max_depth: int = 2,
    cor_alpha: float = 0.01,
    alpha: float = 0.01,
    shapiro_alpha: float = 0.01,
    knowledge: Knowledge | None = None,
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
    max_depth : int, default=2
        The maximum conditioning-set size of an ancestor test, at least ``1``: the most variables
        regressed out together when testing whether one variable is an ancestor of another. Larger
        values find more ancestors and take longer. The default ``2`` is the setting of Maeda and
        Shimizu's experiments [1]_, who advise a small value when there are fewer rows than
        variables.
    cor_alpha : float, default=0.01
        Significance level of the Pearson correlation tests, in ``(0, 1)``.
    alpha : float, default=0.01
        Significance level of the HSIC independence tests, in ``(0, 1)``, which decide the
        ancestors and so the returned edges. The default is the level of the method's published
        experiments [1]_, as are those of ``cor_alpha`` and ``shapiro_alpha``.
    shapiro_alpha : float, default=0.01
        Significance level of the Shapiro-Wilk tests that decide whether a residual is
        non-Gaussian, in ``(0, 1)``.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the pairs
        allowed to be adjacent, and temporal tiers, all honored. The ancestor search runs as
        without knowledge. The parent sets it leads to then drop each forbidden parent, an edge
        into an earlier tier and a pair outside ``allowed`` included, and gain each required one,
        and the weights are fitted on those parents. A pair forbidden in both directions is never
        marked ``<->``; a pair forbidden in one direction can be, since ``<->`` claims a hidden
        common cause, not an edge.

    Returns
    -------
    StructureOutput
        A :class:`~andrey.core.GraphStructure` of kind ``pag``: ``i -> j`` for a direct cause and
        ``i <-> j`` for a pair with a hidden common cause. ``metadata["algorithm"]`` is ``"rcd"``;
        ``metadata["confounded_pairs"]`` lists the ``<->`` pairs as sorted ``[i, j]`` lists;
        ``weighted_adjacency`` holds the edge coefficients.

    Raises
    ------
    TypeError
        If ``knowledge`` is neither ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If an argument has the wrong type, lies outside its range, or is not one of its choices; the
        message names the method, the argument, what it accepts, and the value. Also if ``data`` has
        fewer than 6 rows. Also if ``knowledge`` names a column ``data`` lacks or contradicts
        itself, as :class:`~andrey.Knowledge` lists, or if its required edges close a directed cycle
        with the parents the search found.

    Notes
    -----
    RCD assumes linear relations and non-Gaussian noise. It needs more rows than DirectLiNGAM: its
    tests compare residuals, and a weak hidden cause leaves only a small trace in them.

    **Missing values.** A ``NaN`` in ``data`` raises ``ValueError``: the causal order compares the
    residuals of every variable over the same rows, so a row with a missing value has no place in
    it. Drop the incomplete rows first, which suits values missing completely at random, or use
    :func:`~andrey.pc` or :func:`~andrey.fci`, whose tests delete rows per test.

    **Limitations.** Measured on simulated linear data with uniform noise from Erdos-Renyi graphs in
    which one node in five is hidden (at least one), the hidden ones counted among the variables, on
    4 cores with a cap of 1 h and 28 GB per fit. The largest sizes through which its median endpoint
    errors against the PAG of the observed variables stayed below the empty graph's, and through
    which its fits finished within the caps, were 30 and 40 variables at mean degree 2 with 10,000
    rows; none and 60 at mean degree 2 with 2,000 rows (below it only from 10 through 30); none and
    20 at mean degree 4 with 10,000 rows (below it only from 10 through 20); and none and 10 at mean
    degree 4 with 2,000 rows. At 40 variables (mean degree 2, 10,000 rows), 2 of 10 fits ran out of
    memory. The ladders ended after two sizes in a row did not beat the empty graph, except at mean
    degree 2 with 10,000 rows, where the ladder ended when three fits of the next size ran out of
    the 28 GB, and at mean degree 4 with 10,000 rows, where the ladder ended when three fits of the
    next size ran past 1 h. The median fit took 45 s at 60 variables (mean degree 2, 2,000 rows).
    The work grows with the number of variables and with ``max_depth``. The numbers are a
    reference, not a hard limit: they change with the data generator, the density, the method's
    settings, and the machine.

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

    X = data_matrix(data, min_rows=6, method="rcd")  # HSIC's null variance has (n - 4)(n - 5)
    labels = column_labels(data)
    constraints = resolve(knowledge, X.shape[1], labels, method="rcd")
    model = fit_rcd(
        X,
        max_explanatory_num=max_depth,
        cor_alpha=cor_alpha,
        ind_alpha=alpha,
        shapiro_alpha=shapiro_alpha,
        constraints=constraints,
    )
    return with_labels(_adapt_rcd(model), labels)


@method_spec("bottom_up_parce_lingam")
def bottom_up_parce_lingam(
    data: npt.ArrayLike, *, alpha: float = 0.05, knowledge: Knowledge | None = None
) -> StructureOutput:
    """Learn a linear non-Gaussian causal order that tolerates hidden common causes.

    BottomUpParceLiNGAM [1]_ finds sinks one at a time, from the last variable in the causal order
    up. Where a hidden common cause hides the order, it leaves the variables unordered rather than
    guessing.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observations, one variable per column. A DataFrame's column names become the structure's
        ``labels``.
    alpha : float, default=0.05
        Significance level of the independence test that accepts a sink, Bonferroni-corrected
        across the remaining variables, in ``(0, 1)``. The default is the level of the method's
        published experiments [1]_.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the pairs
        allowed to be adjacent, and temporal tiers, all honored. A variable is taken as a sink only
        when no remaining variable belongs to a later tier or is the effect of one of its required
        edges, so the order keeps the tiers and the required edges, and each regression drops the
        parents the knowledge forbids, an edge into an earlier tier and a pair outside ``allowed``
        included, and keeps the required ones unpenalized. In the unordered block, a pair
        forbidden in both directions is not adjacent, an ``o-o`` pair forbidden in one direction
        gets an arrowhead at the cause, as in :func:`~andrey.fci`, and a required edge is directed
        (see Notes).

    Returns
    -------
    StructureOutput
        A :class:`~andrey.core.GraphStructure` of kind ``pag``: ``i -> j`` for a direct cause and
        ``i o-o j`` between variables whose order a hidden common cause leaves unresolved.
        ``metadata["algorithm"]`` is ``"bottom_up_parce_lingam"``; ``metadata["partial_order"]``
        gives the order, with the unordered variables nested in one list; ``weighted_adjacency``
        holds the edge coefficients.

    Raises
    ------
    TypeError
        If ``knowledge`` is neither ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If an argument has the wrong type, lies outside its range, or is not one of its choices; the
        message names the method, the argument, what it accepts, and the value. Also if ``data`` has
        fewer than 6 rows or no more rows than columns. Also if ``knowledge`` names a column
        ``data`` lacks or contradicts itself, as :class:`~andrey.Knowledge` lists.

    Notes
    -----
    The search finds sinks from the bottom up and stops at the first variable it cannot separate,
    leaving it and every variable above it in one unordered block: the variables a hidden cause
    confounds, their observed ancestors, and any other confounded pairs. Candidates whose test
    statistics are equal after rounding are chosen by each column's rank, as in :func:`~andrey.hc`;
    the :doc:`column-order guide </docs/guides/column-order>` says what the rank keeps. An ``o-o``
    pair need not share a cause, and edges among the block's variables stay unoriented. The edge
    weights come from an adaptive-Lasso regression of each variable on those before it, which needs
    more rows than columns.

    **Knowledge.** ``knowledge`` honors required and forbidden edges, ``allowed``, and tiers. The
    order puts each required edge's cause first and keeps the tiers, and each regression drops the
    forbidden parents and keeps the required ones unpenalized, so a required parent's weight is
    estimated and can be ``0``. A required edge whose two ends both stay in the unordered block is
    directed there, with the least-squares weight of the child on its required parents in the
    block, while the block's other pairs stay ``o-o``. Every required edge is in the graph.

    **Missing values.** A ``NaN`` in ``data`` raises ``ValueError``: the causal order compares the
    residuals of every variable over the same rows, so a row with a missing value has no place in
    it. Drop the incomplete rows first, which suits values missing completely at random, or use
    :func:`~andrey.pc` or :func:`~andrey.fci`, whose tests delete rows per test.

    **Limitations.** Measured on simulated linear data with uniform noise from Erdos-Renyi graphs in
    which one node in five is hidden (at least one), the hidden ones counted among the variables, on
    4 cores with a cap of 1 h and 28 GB per fit. The largest sizes through which its median endpoint
    errors against the PAG of the observed variables stayed below the empty graph's, and through
    which its fits finished within the caps, were 10 and 20 variables at mean degree 2 with 2,000 or
    10,000 rows; none and 20 at mean degree 4 with 10,000 rows (below it only from 10 through 20);
    and none and 20 at mean degree 4 with 2,000 rows (below it only at 10). The ladders ended after
    two sizes in a row did not beat the empty graph, except at mean degree 4 with 10,000 rows, where
    the ladder ended when three fits of the next size ran past 1 h. The median fit took 1,902 s at
    20 variables (mean degree 4, 10,000 rows). On the same data with 2,000 rows, from 16 observed
    variables it left none of the confounded pairs unordered. The numbers are a reference, not a
    hard limit: they change with the data generator, the density, the method's settings, and the
    machine.

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
    >>> out.metadata["partial_order"]
    [[0, 1], 2, 3]
    """
    from andrey.irregular.bottom_up_parce import bottom_up_parce_lingam as _native

    # HSIC's null variance has (n - 4)(n - 5).
    X = data_matrix(data, min_rows=6, method="bottom_up_parce_lingam")
    if X.shape[0] <= X.shape[1]:
        raise ValueError(
            f"data needs more rows than columns, got {X.shape[0]} rows and {X.shape[1]} columns"
        )
    labels = column_labels(data)
    constraints = resolve(knowledge, X.shape[1], labels, method="bottom_up_parce_lingam")
    model = _native(X, alpha=alpha, constraints=constraints)
    if constraints is None:
        return with_labels(_adapt_bottom_up_parce(model), labels)
    one_way = constraints.forbidden & ~constraints.forbidden.T
    return with_labels(
        _adapt_bottom_up_parce(model, one_way, required=constraints.required), labels
    )
