"""Facade adapters of the continuous-optimization learners: NOTEARS and DAGMA.

``notears`` and ``dagma`` share :mod:`andrey.search.continuous`: a loss over a weight matrix, a
sparsity penalty, and an acyclicity function, minimized from the zero matrix. ``robust=True``
follows the first fit with the likelihood path of Deng et al. (2024).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from andrey.api._adapt import (
    column_labels,
    copies_named,
    data_matrix,
    fit_by_em,
    fit_missing,
    json_safe,
    structure_output,
)
from andrey.api._method import method_spec
from andrey.core.backend import one_blas_thread
from andrey.core.ci_missing import EM
from andrey.core.knowledge import Knowledge, resolve

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from andrey.core import GraphStructure, StructureOutput
    from andrey.search.continuous import ContinuousResult


def _output(
    result: ContinuousResult, labels: tuple[str, ...] | None, em: dict[str, object]
) -> StructureOutput:
    """Wrap a fit as a weighted DAG with its bookkeeping and the record of its EM, if any."""
    return structure_output(
        result.dag,
        labels=labels,
        weighted_adjacency=result.weighted_adjacency,
        metadata={
            "h_final": json_safe(result.h_final),
            "edges_removed": result.edges_removed,
            "path_steps": result.path_steps,
            **em,
        },
    )


@method_spec("notears")
def notears(
    data: npt.ArrayLike,
    *,
    lambda1: float = 0.1,
    w_threshold: float = 0.3,
    max_iter: int = 100,
    h_tol: float = 1e-8,
    rho_max: float = 1e16,
    robust: bool = False,
    mcp_lambda: float = 0.4,
    mcp_delta: float = 0.2,
    mcp_decay: float = 0.8,
    max_path: int = 50,
    standardize: bool = False,
    missing: str | None = None,
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn a weighted DAG with NOTEARS, a continuous-optimization structure learner.

    NOTEARS [1]_ minimizes the least-squares loss of a linear model plus an ``l1`` penalty on the
    weights, under the constraint that the weight matrix is acyclic, written as the smooth equation
    ``tr(exp(W * W)) - d = 0``. With ``robust=True`` the fit is followed by the likelihood path of
    [2]_, which replaces the loss by the Gaussian likelihood and the penalty by the quasi-MCP.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``.
    lambda1 : float, default=0.1
        Weight of the ``l1`` penalty of the first fit, at least ``0``. Larger values give sparser
        graphs.
    w_threshold : float, default=0.3
        An edge is kept when its weight magnitude reaches this value, at least ``0``.
    max_iter : int, default=100
        Most dual steps of the augmented Lagrangian, at least ``1``. Stopping on it rather than on
        ``h_tol`` or ``rho_max`` warns with :class:`~andrey.SearchLimitWarning`.
    h_tol : float, default=1e-8
        The fit stops when the acyclicity function falls to this value, above ``0``.
    rho_max : float, default=1e16
        The fit stops when the penalty weight on the acyclicity function reaches this value, above
        ``1``, the weight the fit starts from.
    robust : bool, default=False
        ``True`` follows the first fit with the likelihood path of [2]_ (see Notes).
    mcp_lambda : float, default=0.4
        The penalty level of the first step of the path, above ``0``. Used only with ``robust``.
    mcp_delta : float, default=0.2
        The weight magnitude above which the quasi-MCP penalty stops growing, above ``0``, at the
        first step of the path. Used only with ``robust``.
    mcp_decay : float, default=0.8
        The factor, in ``(0, 1)``, by which ``mcp_lambda`` and ``mcp_delta`` shrink after each
        accepted step. Used only with ``robust``.
    max_path : int, default=50
        Most accepted steps of the path, at least ``1``. Used only with ``robust``. Reaching it
        warns with :class:`~andrey.SearchLimitWarning`.
    standardize : bool, default=False
        Scale each column to unit variance first, so the fit does not read the columns' units;
        the weights are then those of the scaled columns. Limitations gives the fit's errors on
        standardized columns.
    missing : {"em"} or None, default=None
        How missing values (``NaN``) are handled. ``"em"`` fits the method by structural
        expectation-maximization (EM) over the expected covariance of the data, which assumes values
        missing completely at random or at random given observed values. ``None`` refuses missing
        values. On data with missing values, ``metadata["missing"]`` records the handling used and
        ``metadata["em"]`` the EM record. See **Missing values.** in Notes.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the pairs
        allowed to be adjacent, and temporal tiers, all honored. Forbidden edges, edges into an
        earlier tier, and pairs outside the allowed set are held at zero; a required edge is left
        out of the penalty and the threshold and is never removed by the cycle repair.

    Returns
    -------
    StructureOutput
        Wraps a ``dag`` :class:`~andrey.core.GraphStructure`; ``weighted_adjacency[i, j]`` is the
        coefficient of edge ``i -> j``. ``metadata["algorithm"]`` is ``"notears"``,
        ``metadata["h_final"]`` the acyclicity function of the fit before the threshold,
        ``metadata["edges_removed"]`` the edges the threshold kept that the cycle repair dropped,
        and ``metadata["path_steps"]`` the accepted steps of the path, ``0`` without ``robust``.
        ``metadata["tie_break"]`` is the tie-break rule the fit ran under,
        ``andrey.config.tie_break``.

    Raises
    ------
    TypeError
        If an argument has the wrong type, such as a string for a number; the message names the
        method, the argument, what it accepts, and the value. Also if ``knowledge`` is neither
        ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If an argument lies outside its range or is not one of its choices; the message names the
        method, the argument, what it accepts, and the value. Also if ``data`` is not a finite
        numeric matrix of two or more rows without a constant column, or if ``knowledge`` names a
        column ``data`` lacks or contradicts itself, as :class:`~andrey.Knowledge` lists.

    Warns
    -----
    ExperimentalWarning
        On the first call in a process: the method has no published benchmark, and its API may
        change without deprecation.
    SearchLimitWarning
        When the fit stops on ``max_iter`` rather than on ``h_tol`` or ``rho_max``, or, with
        ``robust``, when the path takes ``max_path`` steps.
    DataTypeWarning
        When a column holds at most 10 distinct whole numbers and so may hold category codes, which
        the fit reads as numbers. Each message warns once per process.

    Notes
    -----
    **When to use.** NOTEARS suits i.i.d. continuous observational data from a linear structural
    equation model, in which each variable is a weighted sum of its causes plus noise, with no
    unmeasured common causes [1]_. Its default least-squares fit reads the columns in their given
    units, so its graph can depend on how each column is scaled [4]_ [5]_; ``robust=True`` follows
    it with the Gaussian-likelihood path of [2]_. Its ``l1`` penalty favors sparse graphs, and its
    acyclicity function takes a matrix exponential of the full weight matrix, so it suits a moderate
    number of variables; it returns a DAG with the coefficient of each edge. With nonlinear effects,
    use :func:`~andrey.cam`; when the noise is non-Gaussian and every edge should be oriented, use
    :func:`~andrey.direct_lingam`; when a CPDAG is enough on linear-Gaussian data, use
    :func:`~andrey.ges`. With values missing completely at random or at random given observed
    values, ``missing="em"`` fits it on every row by structural EM [7]_ [8]_.

    The first fit writes the weights as ``W = W+ - W-`` with both parts non-negative and solves the
    augmented Lagrangian ``loss + 0.5 * rho * h**2 + alpha * h + lambda1 * sum(W+ + W-)`` with
    L-BFGS-B, where ``loss = 0.5 * tr((I - W).T @ S @ (I - W))`` for the sample covariance ``S``.
    After each solve ``rho`` grows tenfold while ``h`` has not fallen to a quarter of its previous
    value, and then ``alpha += rho * h``, until ``h <= h_tol`` or ``rho >= rho_max``. Edges below
    ``w_threshold`` are removed, and then the weakest remaining edges until no cycle is left.

    With ``robust=True`` the thresholded weights start a path of refits [2]_. Each step minimizes
    the same augmented Lagrangian with the Gaussian negative log-likelihood, with one noise variance
    per variable and up to a constant,

    ``0.5 * sum_j log(((I - W).T @ S @ (I - W))[j, j]) - log|det(I - W)|``,

    in place of the loss, and with the quasi-MCP in place of the ``l1`` penalty. The quasi-MCP of an
    entry ``t`` is ``lam * (|t| - t**2 / (2 * delta))`` for ``|t| <= delta`` and ``lam * delta / 2``
    above it. Adam [3]_ with step size ``3e-4`` solves each inner problem in place of L-BFGS-B. A
    step starts from the last accepted weights. Its result is thresholded at ``w_threshold`` and
    its cycles repaired, and the step is accepted when the likelihood of that graph is lower than
    the last accepted one's by more than a ``1e-6`` fraction. ``lam`` and ``delta`` then shrink by
    ``mcp_decay``, and the path ends at the first step that does not lower the likelihood. [2]_
    shows that, with a small enough penalty, the global minimizers of the penalized likelihood have
    the same DAG structures for the raw and the standardized data; the path finds a local solution,
    and the cost of a step is the number of its Adam iterations times a matrix exponential.

    Ties between equal weights go to the column whose rank, computed from the data, comes
    first, as for :func:`~andrey.hc`, so the result does not depend on the column order unless two
    columns are identical. The :doc:`column-order guide </docs/guides/column-order>` lists what
    reordering the columns does to each method. The fit draws no random numbers and runs on one BLAS
    thread, so the host's thread count does not change it.

    **Missing values.** With ``missing=None``, a ``NaN`` in ``data`` raises ``ValueError``: the loss
    fits every variable over the same rows, so a row with a missing value has no place in it. The
    loss reads the data only through their covariance, so ``missing="em"`` runs structural EM [7]_
    [8]_, which Gao et al. [8]_ run with NOTEARS: under a linear-Gaussian model of the current
    graph, each row's missing values are Gaussian given its observed ones, and their conditional
    means and covariances give the expected covariance of the data (the E-step); the fit runs on it,
    with all of its settings, and the model's coefficients and noise variances are fit again by
    least squares on the parents the fit returns (the M-step). Gao et al. [8]_ keep, in the M-step,
    the weights the fit returns and one noise variance shared by every variable; here the weights
    and each variable's noise variance are fit again. The weights it returns are the last round's
    fit on its expected covariance. ``standardize`` scales the expected covariance to unit
    variances. A row that observes no value is dropped first: it holds no information on the model.
    It runs 10 rounds, as Gao et al. [8]_ do, from no edges and each column's mean and variance over
    its observed values, and it alternates as Friedman [7]_ describes: before each fit after the
    first, 5 parameter steps refit the current graph's coefficients and noise variances, each on the
    expected covariance under the step before it. Friedman [7]_ leaves the number of parameter steps
    open. The first fit reads the expected covariance under the model with no edge, where every
    covariance is shrunk toward zero. Measured with ``ges``, ``boss``, and ``grasp`` at 2,000 rows,
    the first search returned many extra edges; with one refit per round the later rounds removed
    them slowly and the graph was still changing after 10 rounds, and with 5 parameter steps it had
    settled within them in most fits. ``metadata["em"]`` records ``iterations``, the rounds;
    ``parameter_steps``, the parameter steps before each fit after the first; ``log_likelihood``,
    the observed-data log-likelihood of the model after each round, which can fall in a round that
    changes the graph; and ``graph_changed``, whether each round's fit returned another graph than
    the round before. When the last round changed the graph, EM has not settled on one, which is
    more common with few rows, and the fit logs a warning. EM holds when values are missing
    completely at random or at random given observed values; when whether a value is missing depends
    on the missing values themselves, the expected covariance is biased, and the data cannot show
    it. On complete data, ``"em"`` gives the complete-data result. Without ``"em"``, drop the
    incomplete rows first, which suits values missing completely at random, or use
    :func:`~andrey.pc` or :func:`~andrey.fci`, whose tests delete rows per test.

    **Limitations.** Measured on simulated linear data from sparse random graphs (Erdos-Renyi and
    scale-free, mean degree 2) with Gaussian and uniform noise, 1,000 rows, and 10, 20, 50 and 100
    variables, five graphs for each combination, at the default settings. Errors are SHD against the
    true CPDAG, medians over those 20 fits; the empty graph's are 9.5, 19.5, 49.5 and 99.5. The
    least-squares fit changes when the columns are rescaled or standardized [4]_ [5]_, and [6]_
    traces this to the score. With a small enough penalty, the global minimizers of the penalized
    likelihood that ``robust=True`` uses have the same graph on raw and standardized columns [2]_.
    The robust fit, though, starts from the weights of the least-squares fit, and its penalty and
    threshold act on the weights in data units; rescaling the columns by powers of two, all by one
    factor or each by its own, changed its graph in every fit tried at 10 and 20 variables. Both
    that starting point and the penalty and threshold on the weights in data units are the published
    method's [2]_. The optimization is also harder on standardized data, and more so as the number
    of variables grows [2]_. On standardized columns the default fit's median errors were 7, 18.5,
    44.5 and 92 at those sizes, below the empty graph's in 61 of the 80 fits; on
    ``scale="unitless"`` data of ``andrey.data.sample_scm`` they were 8, 17, 43.5 and 89, below it
    in 78 of 80. With ``robust=True`` they were 10, 14.5 and 30.5 on standardized columns and 6.5,
    14.5 and 32.5 on unitless data at 10, 20 and 50 variables; at 50 variables one graph per
    combination was fit (four fits), and the one fit tried at 100 variables did not finish within
    900 s, so there is no figure for it. On the same data, ``ges`` had median errors of 2, 4, 19.5
    and 79 on standardized columns and 3, 7.5, 29 and 90 on unitless data, and ``calm`` 3.5, 6, 18
    and 36, and 4, 12, 31.5 and 55. The default fit's median time was 0.6 s at 10 variables and 54 s
    at 100 on standardized columns (0.4 s and 15.5 s on unitless data), on one core; with
    ``robust=True`` it was 23 s at 10 variables and 198 s at 50 on standardized columns (5.3 s and
    69 s on unitless data), because each Adam iteration of a path step computes a matrix
    exponential. A fit can stop at a local optimum, and only weights that reach ``w_threshold``
    become edges. Nothing was measured beyond 100 variables, at other densities, row counts or noise
    types, or on non-linear data. With ``missing="em"``, measured in the setting of Gao et al. [8]_:
    linear-Gaussian data with unit noise variances on raw columns, from Erdos-Renyi graphs of mean
    degree 1 and 2 and scale-free graphs of mean degree 2, with 20 variables and 100 rows, 20
    datasets per setting, at the default settings on one core. With 10 to 50% of the values missing
    completely at random or at random given observed values, its mean errors (SHD) stayed below the
    empty graph's in 22 of the 24 settings. With 10% missing completely at random they were 0.6,
    1.8, and 3.25 on the three graphs, against 10, 20, and 19 for the empty graph and 0.2, 1.25, and
    1.55 on the complete data; with 50% they were 8.65, 19.5, and 18.7, not below the empty graph's
    on the two graphs of mean degree 2. It had fewer errors than :func:`~andrey.pc`, whose tests
    delete rows per test, in 16 of the 24 settings and tied in 8, and fewer than the default fit on
    data with column means filled in in 22, with 2 ties; dropping the incomplete rows left enough
    rows to compare in 6 settings, and it had fewer errors in each. It runs the fit 10 times: the
    median fit took 8 to 42 s, against 1.9 to 3.8 s on the complete data. On standardized columns
    (Erdos-Renyi, mean degree 2, 20 variables, 500 rows, 20% missing), where the least-squares fit
    is weaker, its mean errors were 15.35 completely at random and 16.0 at random, against 15.25 on
    the complete data and 20 for the empty graph; ``pc`` had 7.5 there, and the default fit on
    mean-filled data 13.55 with values missing completely at random, a difference not rerun on fresh
    seeds. Under values missing not at random it has no guarantee. The numbers are a reference, not
    a hard limit: they change with the data generator, the density, the method's settings, and the
    machine.

    References
    ----------
    .. [1] Zheng, X., Aragam, B., Ravikumar, P., and Xing, E. P. (2018). DAGs with NO TEARS:
       Continuous optimization for structure learning. NeurIPS 31, 9472-9483.
    .. [2] Deng, C., Bello, K., Ravikumar, P., and Aragam, B. (2024). Markov equivalence and
       consistency in differentiable structure learning. NeurIPS 37.
    .. [3] Kingma, D. P. and Ba, J. (2015). Adam: A method for stochastic optimization. ICLR 2015.
    .. [4] Kaiser, M. and Sipos, M. (2022). Unsuitability of NOTEARS for causal graph discovery
       when dealing with dimensional quantities. Neural Processing Letters 54(3), 1587-1595.
    .. [5] Reisach, A. G., Seiler, C., and Weichwald, S. (2021). Beware of the simulated DAG!
       Causal discovery benchmarks may be easy to game. NeurIPS 34.
    .. [6] Ng, I., Huang, B., and Zhang, K. (2024). Structure learning with continuous
       optimization: A sober look and beyond. CLeaR 2024, PMLR 236, 71-105.
    .. [7] Friedman, N. (1997). Learning belief networks in the presence of missing values and
       hidden variables. Proceedings of the 14th International Conference on Machine Learning,
       125-133.
    .. [8] Gao, E., Ng, I., Gong, M., Shen, L., Huang, W., Liu, T., Zhang, K., and Bondell, H.
       (2022). MissDAG: Causal discovery in the presence of missing data with continuous additive
       noise models. Advances in Neural Information Processing Systems 35.

    Examples
    --------
    >>> import warnings
    >>> import numpy as np
    >>> from andrey import notears
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal(500)
    >>> y = x + 0.5 * rng.standard_normal(500)
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     out = notears(np.column_stack([x, y]))
    >>> out.structure.kind
    'dag'
    """
    from andrey.search.continuous import notears as _notears

    X = data_matrix(data, method="notears", missing=missing == EM, fix=fit_by_em("notears"))
    labels = column_labels(data)
    constraints = resolve(knowledge, X.shape[1], labels, method="notears")

    def fit(cov: np.ndarray | None) -> tuple[ContinuousResult, GraphStructure]:
        found = _notears(
            X,
            lambda1=lambda1,
            w_threshold=w_threshold,
            max_iter=max_iter,
            h_tol=h_tol,
            rho_max=rho_max,
            robust=robust,
            mcp_lambda=mcp_lambda,
            mcp_delta=mcp_delta,
            mcp_decay=mcp_decay,
            max_path=max_path,
            standardize=standardize,
            labels=labels,
            constraints=constraints,
            cov=cov,
        )
        return found, found.dag

    with copies_named(labels), one_blas_thread():
        result, em = fit_missing(X, missing, fit)
    return _output(result, labels, em)


@method_spec("dagma")
def dagma(
    data: npt.ArrayLike,
    *,
    lambda1: float = 0.03,
    w_threshold: float = 0.3,
    stages: int = 5,
    warm_iter: int = 30000,
    max_iter: int = 60000,
    lr: float = 3e-4,
    robust: bool = False,
    mcp_lambda: float = 0.4,
    mcp_delta: float = 0.2,
    mcp_decay: float = 0.8,
    max_path: int = 50,
    standardize: bool = False,
    missing: str | None = None,
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn a weighted DAG with DAGMA, a continuous-optimization structure learner.

    DAGMA [1]_ minimizes the least-squares loss of a linear model plus an ``l1`` penalty on the
    weights, with the acyclicity function ``-log det(s * I - W * W) + d * log(s)`` as a barrier
    that is lowered along a central path. With ``robust=True`` the fit is followed by the
    likelihood path of [2]_, which replaces the loss by the Gaussian likelihood and the penalty by
    the quasi-MCP.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``.
    lambda1 : float, default=0.03
        Weight of the ``l1`` penalty of the first fit, at least ``0``. Larger values give sparser
        graphs.
    w_threshold : float, default=0.3
        An edge is kept when its weight magnitude reaches this value, at least ``0``.
    stages : int, default=5
        Stages of the central path, at least ``1`` (``T`` in the paper). Stage ``t`` weights the
        loss by ``0.1**t`` and uses ``s`` from ``(1, 0.9, 0.8, 0.7, 0.6)``, the last value
        repeating.
    warm_iter : int, default=30000
        Most Adam steps of each stage but the last, at least ``1``.
    max_iter : int, default=60000
        Most Adam steps of the last stage, at least ``1``. A last stage that takes them all
        without converging warns with :class:`~andrey.SearchLimitWarning`.
    lr : float, default=3e-4
        Step size of Adam, above ``0``.
    robust : bool, default=False
        ``True`` follows the first fit with the likelihood path of [2]_ (see Notes).
    mcp_lambda : float, default=0.4
        The penalty level of the first step of the path, above ``0``. Used only with ``robust``.
    mcp_delta : float, default=0.2
        The weight magnitude above which the quasi-MCP penalty stops growing, above ``0``, at the
        first step of the path. Used only with ``robust``.
    mcp_decay : float, default=0.8
        The factor, in ``(0, 1)``, by which ``mcp_lambda`` and ``mcp_delta`` shrink after each
        accepted step. Used only with ``robust``.
    max_path : int, default=50
        Most accepted steps of the path, at least ``1``. Used only with ``robust``. Reaching it
        warns with :class:`~andrey.SearchLimitWarning`.
    standardize : bool, default=False
        Scale each column to unit variance first, so the fit does not read the columns' units;
        the weights are then those of the scaled columns. Limitations gives the fit's errors on
        standardized columns.
    missing : {"em"} or None, default=None
        How missing values (``NaN``) are handled. ``"em"``, experimental, fits the method by
        structural expectation-maximization (EM) over the expected covariance of the data, which
        assumes values missing completely at random or at random given observed values. ``None``
        refuses missing values. On data with missing values, ``metadata["missing"]`` records the
        handling used and ``metadata["em"]`` the EM record. See **Missing values.** in Notes.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the pairs
        allowed to be adjacent, and temporal tiers, all honored. Forbidden edges, edges into an
        earlier tier, and pairs outside the allowed set are held at zero; a required edge is left
        out of the penalty and the threshold and is never removed by the cycle repair.

    Returns
    -------
    StructureOutput
        Wraps a ``dag`` :class:`~andrey.core.GraphStructure`; ``weighted_adjacency[i, j]`` is the
        coefficient of edge ``i -> j``. ``metadata["algorithm"]`` is ``"dagma"``,
        ``metadata["h_final"]`` the log-determinant acyclicity function at ``s = 1`` before the
        threshold, ``metadata["edges_removed"]`` the edges the threshold kept that the cycle repair
        dropped, and ``metadata["path_steps"]`` the accepted steps of the path, ``0`` without
        ``robust``. ``metadata["tie_break"]`` is the tie-break rule the fit ran under,
        ``andrey.config.tie_break``.

    Raises
    ------
    TypeError
        If an argument has the wrong type, such as a string for a number; the message names the
        method, the argument, what it accepts, and the value. Also if ``knowledge`` is neither
        ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If an argument lies outside its range or is not one of its choices; the message names the
        method, the argument, what it accepts, and the value. Also if ``data`` is not a finite
        numeric matrix of two or more rows without a constant column, or if ``knowledge`` names a
        column ``data`` lacks or contradicts itself, as :class:`~andrey.Knowledge` lists.

    Warns
    -----
    ExperimentalWarning
        On the first call in a process: the method has no published benchmark, and its API may
        change without deprecation.
    SearchLimitWarning
        When the last stage takes ``max_iter`` steps without converging, or, with ``robust``, when
        the path takes ``max_path`` steps.
    DataTypeWarning
        When a column holds at most 10 distinct whole numbers and so may hold category codes, which
        the fit reads as numbers. Each message warns once per process.

    Notes
    -----
    **When to use.** DAGMA suits i.i.d. continuous observational data from a linear structural
    equation model, in which each variable is a weighted sum of its causes plus noise, with no
    unmeasured common causes [1]_. Its default least-squares fit reads the columns in their given
    units, so its graph can depend on how each column is scaled [4]_ [5]_; ``robust=True`` follows
    it with the Gaussian-likelihood path of [2]_. Its ``l1`` penalty favors sparse graphs, and its
    acyclicity function takes a log-determinant of the full weight matrix, so it suits a moderate
    number of variables; it returns a DAG with the coefficient of each edge. With nonlinear effects,
    use :func:`~andrey.cam`; when the noise is non-Gaussian and every edge should be oriented, use
    :func:`~andrey.direct_lingam`; when a CPDAG is enough on linear-Gaussian data, use
    :func:`~andrey.ges`. With values missing completely at random or at random given observed
    values, ``missing="em"`` fits it on every row by structural EM [7]_ [8]_.

    The first fit starts from the zero matrix. Stage ``t`` of the central path runs Adam [3]_
    (betas ``0.99`` and ``0.999``) on ``mu * (loss + lambda1 * |W|_1) + h(W)`` with
    ``loss = 0.5 * tr((I - W).T @ S @ (I - W))`` for the sample covariance ``S`` and
    ``mu = 0.1**t``, and stops after ``warm_iter`` steps (``max_iter`` in the last stage), or
    earlier when, at a multiple of 1000 steps, the objective has changed by at most a ``1e-6``
    fraction since the last such step. A step that leaves the domain where ``s * I - W * W`` is an
    M-matrix is undone and the step size halved; a stage that leaves it at its first step starts
    again with half the step size and ``s`` raised by ``0.1``. The diagonal of ``W`` is held at
    zero. Edges below ``w_threshold`` are then removed, and then the weakest remaining edges until
    no cycle is left.

    With ``robust=True`` the thresholded weights start a path of refits [2]_. Each step runs the
    central path again from the last accepted weights, with the Gaussian negative log-likelihood,
    with one noise variance per variable and up to a constant,

    ``0.5 * sum_j log(((I - W).T @ S @ (I - W))[j, j]) - log|det(I - W)|``,

    in place of the loss, and the quasi-MCP in place of the ``l1`` penalty. The quasi-MCP of an
    entry ``t`` is ``lam * (|t| - t**2 / (2 * delta))`` for ``|t| <= delta`` and ``lam * delta / 2``
    above it. The result of a step is thresholded at ``w_threshold`` and its cycles repaired, and
    the step is accepted when the likelihood of that graph is lower than the last accepted one's by
    more than a ``1e-6`` fraction. ``lam`` and ``delta`` then shrink by ``mcp_decay``, and the path
    ends at the first step that does not lower the likelihood. [2]_ shows that, with a small enough
    penalty, the global minimizers of the penalized likelihood have the same DAG structures for the
    raw and the standardized data; the path finds a local solution, and each step repeats the whole
    central path.

    Ties between equal weights go to the column whose rank, computed from the data, comes
    first, as for :func:`~andrey.hc`, so the result does not depend on the column order unless two
    columns are identical. The :doc:`column-order guide </docs/guides/column-order>` lists what
    reordering the columns does to each method. The fit draws no random numbers and runs on one BLAS
    thread, so the host's thread count does not change it.

    **Missing values.** With ``missing=None``, a ``NaN`` in ``data`` raises ``ValueError``: the loss
    fits every variable over the same rows, so a row with a missing value has no place in it. The
    loss reads the data only through their covariance, so ``missing="em"``, experimental, runs
    structural EM [7]_ [8]_, which Gao et al. [8]_ run with NOTEARS: under a linear-Gaussian model
    of the current graph, each row's missing values are Gaussian given its observed ones, and their
    conditional means and covariances give the expected covariance of the data (the E-step); the fit
    runs on it, with all of its settings, and the model's coefficients and noise variances are fit
    again by least squares on the parents the fit returns (the M-step). Gao et al. [8]_ keep, in the
    M-step, the weights the fit returns and one noise variance shared by every variable; here the
    weights and each variable's noise variance are fit again. The weights it returns are the last
    round's fit on its expected covariance. ``standardize`` scales the expected covariance to unit
    variances. A row that observes no value is dropped first: it holds no information on the model.
    It runs 10 rounds, as Gao et al. [8]_ do, from no edges and each column's mean and variance over
    its observed values, and it alternates as Friedman [7]_ describes: before each fit after the
    first, 5 parameter steps refit the current graph's coefficients and noise variances, each on the
    expected covariance under the step before it. Friedman [7]_ leaves the number of parameter steps
    open. The first fit reads the expected covariance under the model with no edge, where every
    covariance is shrunk toward zero. Measured with ``ges``, ``boss``, and ``grasp`` at 2,000 rows,
    the first search returned many extra edges; with one refit per round the later rounds removed
    them slowly and the graph was still changing after 10 rounds, and with 5 parameter steps it had
    settled within them in most fits. ``metadata["em"]`` records ``iterations``, the rounds;
    ``parameter_steps``, the parameter steps before each fit after the first; ``log_likelihood``,
    the observed-data log-likelihood of the model after each round, which can fall in a round that
    changes the graph; and ``graph_changed``, whether each round's fit returned another graph than
    the round before. When the last round changed the graph, EM has not settled on one, which is
    more common with few rows, and the fit logs a warning. EM holds when values are missing
    completely at random or at random given observed values; when whether a value is missing depends
    on the missing values themselves, the expected covariance is biased, and the data cannot show
    it. On complete data, ``"em"`` gives the complete-data result. Without ``"em"``, drop the
    incomplete rows first, which suits values missing completely at random, or use
    :func:`~andrey.pc` or :func:`~andrey.fci`, whose tests delete rows per test.

    **Limitations.** Measured on simulated linear data from sparse random graphs (Erdos-Renyi and
    scale-free, mean degree 2) with Gaussian and uniform noise, 1,000 rows, and 10, 20, 50 and 100
    variables, five graphs for each combination, at the default settings. Errors are SHD against the
    true CPDAG, medians over those 20 fits; the empty graph's are 9.5, 19.5, 49.5 and 99.5. The
    least-squares fit changes when the columns are rescaled or standardized [4]_ [5]_, and [6]_
    traces this to the score. With a small enough penalty, the global minimizers of the penalized
    likelihood that ``robust=True`` uses have the same graph on raw and standardized columns [2]_.
    The robust fit, though, starts from the weights of the least-squares fit, and its penalty and
    threshold act on the weights in data units; rescaling the columns by powers of two, all by one
    factor or each by its own, changed its graph in every fit tried at 10 and 20 variables. Both
    that starting point and the penalty and threshold on the weights in data units are the published
    method's [2]_. The optimization is also harder on standardized data, and more so as the number
    of variables grows [2]_. On standardized columns the default fit's median errors were 12, 25.5,
    64 and 129.5 at those sizes, below the empty graph's in 7 of the 80 fits; on
    ``scale="unitless"`` data of ``andrey.data.sample_scm`` they were 9.5, 19, 49.5 and 99.5, below
    it in 38 of 80. With ``robust=True`` they were 12, 25, 65 and 135.5 on standardized columns
    (below the empty graph's in 10 of 80 fits) and 9.5, 18, 48 and 98.5 on unitless data (44 of 80).
    On the same data, ``ges`` had median errors of 2, 4, 19.5 and 79 on standardized columns and 3,
    7.5, 29 and 90 on unitless data, and ``calm`` 3.5, 6, 18 and 36, and 4, 12, 31.5 and 55. The
    default fit's median time was 2.2 s at 10 variables and 28 s at 100 on standardized columns (1.8
    s and 21 s on unitless data), on one core; with ``robust=True`` it was 21 s at 10 variables and
    212 s at 100 on standardized columns (27 s and 310 s on unitless data), because each path step
    repeats the whole central path. A fit can stop at a local optimum, and only weights that reach
    ``w_threshold`` become edges. Nothing was measured beyond 100 variables, at other densities, row
    counts or noise types, or on non-linear data. ``missing="em"`` is experimental and not
    benchmarked yet; its tests run it on a collider of 3 variables with 1,000 rows and 30% of one
    variable's values missing at random, which it learns, and on 8 variables with 500 rows and
    values missing at random, 30% of each column that can go missing, where over 4 graphs it made no
    more errors in total than the fit on the complete rows, fewer than the fit with each missing
    value replaced by its column's mean, and fewer than the empty graph. It runs the fit 10 times,
    so it takes about 10 times as long, and under values missing not at random it has no guarantee.
    The numbers are a reference, not a hard limit: they change with the data generator, the density,
    the method's settings, and the machine.

    References
    ----------
    .. [1] Bello, K., Aragam, B., and Ravikumar, P. (2022). DAGMA: Learning DAGs via M-matrices and
       a log-determinant acyclicity characterization. NeurIPS 35, 8226-8239.
    .. [2] Deng, C., Bello, K., Ravikumar, P., and Aragam, B. (2024). Markov equivalence and
       consistency in differentiable structure learning. NeurIPS 37.
    .. [3] Kingma, D. P. and Ba, J. (2015). Adam: A method for stochastic optimization. ICLR 2015.
    .. [4] Kaiser, M. and Sipos, M. (2022). Unsuitability of NOTEARS for causal graph discovery
       when dealing with dimensional quantities. Neural Processing Letters 54(3), 1587-1595.
    .. [5] Reisach, A. G., Seiler, C., and Weichwald, S. (2021). Beware of the simulated DAG!
       Causal discovery benchmarks may be easy to game. NeurIPS 34.
    .. [6] Ng, I., Huang, B., and Zhang, K. (2024). Structure learning with continuous
       optimization: A sober look and beyond. CLeaR 2024, PMLR 236, 71-105.
    .. [7] Friedman, N. (1997). Learning belief networks in the presence of missing values and
       hidden variables. Proceedings of the 14th International Conference on Machine Learning,
       125-133.
    .. [8] Gao, E., Ng, I., Gong, M., Shen, L., Huang, W., Liu, T., Zhang, K., and Bondell, H.
       (2022). MissDAG: Causal discovery in the presence of missing data with continuous additive
       noise models. Advances in Neural Information Processing Systems 35.

    Examples
    --------
    >>> import warnings
    >>> import numpy as np
    >>> from andrey import dagma
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal(500)
    >>> y = x + 0.5 * rng.standard_normal(500)
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     out = dagma(np.column_stack([x, y]))
    >>> out.structure.kind
    'dag'
    """
    from andrey.search.continuous import dagma as _dagma

    X = data_matrix(data, method="dagma", missing=missing == EM, fix=fit_by_em("dagma"))
    labels = column_labels(data)
    constraints = resolve(knowledge, X.shape[1], labels, method="dagma")

    def fit(cov: np.ndarray | None) -> tuple[ContinuousResult, GraphStructure]:
        found = _dagma(
            X,
            lambda1=lambda1,
            w_threshold=w_threshold,
            stages=stages,
            warm_iter=warm_iter,
            max_iter=max_iter,
            lr=lr,
            robust=robust,
            mcp_lambda=mcp_lambda,
            mcp_delta=mcp_delta,
            mcp_decay=mcp_decay,
            max_path=max_path,
            standardize=standardize,
            labels=labels,
            constraints=constraints,
            cov=cov,
        )
        return found, found.dag

    with copies_named(labels), one_blas_thread():
        result, em = fit_missing(X, missing, fit)
    return _output(result, labels, em)
