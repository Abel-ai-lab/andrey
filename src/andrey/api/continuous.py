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
    json_safe,
    structure_output,
)
from andrey.api._method import method_spec
from andrey.core.backend import one_blas_thread
from andrey.core.knowledge import Knowledge, resolve

if TYPE_CHECKING:
    import numpy.typing as npt

    from andrey.core import StructureOutput
    from andrey.search.continuous import ContinuousResult


def _output(
    result: ContinuousResult,
    labels: tuple[str, ...] | None,
    robust: bool,
) -> StructureOutput:
    """Wrap a fit as a weighted DAG with its bookkeeping."""
    return structure_output(
        result.dag,
        labels=labels,
        weighted_adjacency=result.weighted_adjacency,
        metadata={
            "n_edges": int((result.weighted_adjacency != 0).sum()),
            "h_final": json_safe(result.h_final),
            "edges_removed": result.edges_removed,
            "robust": robust,
            "path_steps": result.path_steps,
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
        Most dual steps of the augmented Lagrangian, at least ``1``.
    h_tol : float, default=1e-8
        The fit stops when the acyclicity function falls to this value, above ``0``.
    rho_max : float, default=1e16
        The fit stops when the penalty weight on the acyclicity function reaches this value, above
        ``0``.
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
        Most accepted steps of the path, at least ``1``. Used only with ``robust``.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the pairs
        allowed to be adjacent, and temporal tiers, all honored. Forbidden edges, edges into an
        earlier tier, and pairs outside the allowed set are held at zero; a required edge is left
        out of the penalty and the threshold and is never removed by the cycle repair.

    Returns
    -------
    StructureOutput
        Wraps a ``dag`` :class:`~andrey.core.GraphStructure`; ``weighted_adjacency[i, j]`` is the
        coefficient of edge ``i -> j``. ``metadata`` holds ``"algorithm"`` (``"notears"``),
        ``"n_edges"``, ``"h_final"`` (the acyclicity function of the fit before the threshold),
        ``"edges_removed"`` (edges the threshold kept that the cycle repair dropped), ``"robust"``,
        and ``"path_steps"`` (accepted steps of the path, ``0`` without ``robust``).

    Raises
    ------
    TypeError
        If ``knowledge`` is neither ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If an argument has the wrong type, lies outside its range, or is not one of its choices; the
        message names the method, the argument, what it accepts, and the value. Also if ``data`` is
        not a finite numeric matrix of two or more rows without a constant column, or if
        ``knowledge`` names a column ``data`` lacks or contradicts itself, as
        :class:`~andrey.Knowledge` lists.

    Warns
    -----
    ExperimentalWarning
        On the first call in a process: the method has no published benchmark, and its API may
        change without deprecation.

    Notes
    -----
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

    Ties between equal weights go to the column whose rank, computed from its own data, comes
    first, as for :func:`~andrey.hc`, so the result does not depend on the column order unless two
    columns are identical. The :doc:`column-order guide </docs/guides/column-order>` lists what
    reordering the columns does to each method. The fit draws no random numbers and runs on one BLAS
    thread, so the host's thread count does not change it.

    **Missing values.** A ``NaN`` in ``data`` raises ``ValueError``: the loss fits every variable
    over the same rows, so a row with a missing value has no place in it. Drop the incomplete rows
    first, which suits values missing completely at random, or use :func:`~andrey.pc` or
    :func:`~andrey.fci`, whose tests delete rows per test.

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
    types, or on non-linear data. The numbers are a reference, not a hard limit: they change with
    the data generator, the density, the method's settings, and the machine.

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

    X = data_matrix(data, method="notears")
    labels = column_labels(data)
    constraints = resolve(knowledge, X.shape[1], labels, method="notears")
    with copies_named(labels), one_blas_thread():
        result = _notears(
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
            labels=labels,
            constraints=constraints,
        )
    return _output(result, labels, robust)


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
        Most Adam steps of the last stage, at least ``1``.
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
        Most accepted steps of the path, at least ``1``. Used only with ``robust``.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the pairs
        allowed to be adjacent, and temporal tiers, all honored. Forbidden edges, edges into an
        earlier tier, and pairs outside the allowed set are held at zero; a required edge is left
        out of the penalty and the threshold and is never removed by the cycle repair.

    Returns
    -------
    StructureOutput
        Wraps a ``dag`` :class:`~andrey.core.GraphStructure`; ``weighted_adjacency[i, j]`` is the
        coefficient of edge ``i -> j``. ``metadata`` holds ``"algorithm"`` (``"dagma"``),
        ``"n_edges"``, ``"h_final"`` (the log-determinant acyclicity function at ``s = 1`` before
        the threshold), ``"edges_removed"`` (edges the threshold kept that the cycle repair
        dropped), ``"robust"``, and ``"path_steps"`` (accepted steps of the path, ``0`` without
        ``robust``).

    Raises
    ------
    TypeError
        If ``knowledge`` is neither ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If an argument has the wrong type, lies outside its range, or is not one of its choices; the
        message names the method, the argument, what it accepts, and the value. Also if ``data`` is
        not a finite numeric matrix of two or more rows without a constant column, or if
        ``knowledge`` names a column ``data`` lacks or contradicts itself, as
        :class:`~andrey.Knowledge` lists.

    Warns
    -----
    ExperimentalWarning
        On the first call in a process: the method has no published benchmark, and its API may
        change without deprecation.

    Notes
    -----
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

    Ties between equal weights go to the column whose rank, computed from its own data, comes
    first, as for :func:`~andrey.hc`, so the result does not depend on the column order unless two
    columns are identical. The :doc:`column-order guide </docs/guides/column-order>` lists what
    reordering the columns does to each method. The fit draws no random numbers and runs on one BLAS
    thread, so the host's thread count does not change it.

    **Missing values.** A ``NaN`` in ``data`` raises ``ValueError``: the loss fits every variable
    over the same rows, so a row with a missing value has no place in it. Drop the incomplete rows
    first, which suits values missing completely at random, or use :func:`~andrey.pc` or
    :func:`~andrey.fci`, whose tests delete rows per test.

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
    counts or noise types, or on non-linear data. The numbers are a reference, not a hard limit:
    they change with the data generator, the density, the method's settings, and the machine.

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

    X = data_matrix(data, method="dagma")
    labels = column_labels(data)
    constraints = resolve(knowledge, X.shape[1], labels, method="dagma")
    with copies_named(labels), one_blas_thread():
        result = _dagma(
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
            labels=labels,
            constraints=constraints,
        )
    return _output(result, labels, robust)
