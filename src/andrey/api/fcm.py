"""Functional-causal-model facades: CAM, RESIT, and CAM-UV fit each variable from its causes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from andrey.api._adapt import (
    column_labels,
    column_namer,
    dag_from_adjacency,
    data_matrix,
    structure_output,
)
from andrey.api._marks import marks_from_signed
from andrey.api._method import method_spec
from andrey.core import GraphStructure
from andrey.core.backend import one_blas_thread
from andrey.core.knowledge import Knowledge, resolve
from andrey.core.score import copies_error, copy_groups

if TYPE_CHECKING:
    import numpy.typing as npt

    from andrey.core import StructureOutput

#: Fewest distinct values of a column: a smooth term's basis has at least three functions.
MIN_DISTINCT = 3


def _smooth_columns(X: np.ndarray, labels: tuple[str, ...] | None, method: str) -> None:
    """Raise ``ValueError`` when a column cannot carry a smooth term or repeats another.

    A column with fewer than :data:`MIN_DISTINCT` distinct values, such as a binary one, has no
    smooth basis, and a column that repeats another up to scale and shift explains it with no
    residual. The messages name ``method`` and every such column.
    """
    name = column_namer(labels)
    few = [j for j in range(X.shape[1]) if np.unique(X[:, j]).size < MIN_DISTINCT]
    if few:
        names = [name(j) for j in few]
        if len(names) == 1:
            columns, verb, pronoun = f"column {names[0]}", "has", "it"
        else:
            columns = f"columns {', '.join(names[:-1])} and {names[-1]}"
            verb, pronoun = "have", "them"
        raise ValueError(
            f"data {columns} {verb} fewer than {MIN_DISTINCT} distinct values; {method} fits a "
            f"smooth function of each variable, which needs at least {MIN_DISTINCT}, so drop "
            f"{pronoun} before the fit"
        )
    if X.shape[1] > 1:
        groups = copy_groups(np.corrcoef(X, rowvar=False), X.shape[0])
        if groups:
            raise copies_error(groups, name)


@method_spec("cam")
def cam(
    data: npt.ArrayLike,
    *,
    alpha: float = 0.001,
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn a DAG with CAM, the causal additive model (nonlinear additive effects).

    CAM [1]_ models each variable as a sum of smooth functions of its parents plus Gaussian noise,
    ``x_j = sum_k f_jk(x_k) + e_j``, which identifies the DAG when the functions are nonlinear
    [6]_ [7]_. It preselects each variable's possible parents by boosting, adds edges greedily by
    the drop in the log residual variance until the DAG is complete within them, and prunes the
    edges whose smooth terms are not significant. It returns a DAG and a causal order.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``.
    alpha : float, default=0.001
        The significance level of the pruning test: a parent stays when its smooth term's p-value
        is at most ``alpha``, in ``(0, 1)``. ``0.001`` is the paper's value, which it keeps at every
        sample size and advises raising for small samples.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the pairs
        allowed to be adjacent, and temporal tiers, all honored. Forbidden edges, edges into an
        earlier tier, and pairs outside the allowed set are never possible parents; the boosting
        that preselects the possible parents reads the data alone. The search starts from the
        required edges and the pruning never removes one. The causal order puts the cause of each
        required edge, and every earlier tier, first.

    Returns
    -------
    StructureOutput
        Wraps a ``dag`` :class:`~andrey.core.GraphStructure`; ``metadata["algorithm"]`` is
        ``"cam"`` and ``metadata["tie_break"]`` the rule ties followed. ``ordering`` is a causal
        order of the search's DAG before pruning, as a tuple of column indices, causes first.

    Raises
    ------
    TypeError
        If ``knowledge`` is neither ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If an argument has the wrong type, lies outside its range, or is not one of its choices; the
        message names the method, the argument, what it accepts, and the value. Also if ``data`` is
        not a usable 2-D matrix: not numeric, holding ``NaN`` or ``inf``, or with a constant column,
        a column of fewer than 3 distinct values, or a column that repeats another up to scale and
        shift, each named in the message. Also if ``knowledge`` names a column ``data`` lacks or
        contradicts itself, as :class:`~andrey.Knowledge` lists.

    Warns
    -----
    ExperimentalWarning
        On the first call in a process: the method has no published benchmark, and its API may
        change without deprecation.

    See Also
    --------
    andrey.anm : Decides the direction between two variables under an additive noise model.
    andrey.direct_lingam : Linear effects with non-Gaussian noise.
    andrey.pc, andrey.ges : Return the equivalence class, which is what linear Gaussian data
        identify.

    Notes
    -----
    **When to use it.** CAM suits continuous data whose effects are nonlinear and additive in the
    parents, with additive noise and no unmeasured common causes [1]_. Its score is the Gaussian
    likelihood of that model.

    **When not to.** With linear effects and Gaussian noise only the Markov equivalence class is
    identified, and the orientation of the edges it leaves undirected is arbitrary: use
    :func:`~andrey.pc` or :func:`~andrey.ges`. Noise whose scale depends on the parents
    (location-scale data) can reverse edges, because the model's noise variance is fixed.

    **The steps.** The paper's settings, in its three steps [1]_:

    1. *Neighbor preselection.* Each variable is regressed on all the others by componentwise L2
       boosting [2]_, 100 iterations of step size 0.1, with one P-spline base learner per other
       variable [3]_: a cubic B-spline basis on 20 equidistant interior knots over the variable's
       range, a second-order difference penalty, and the penalty weight that gives the learner 4
       degrees of freedom. The possible parents are the ten variables picked most often, among
       those picked at least three times. When more than ten qualify, the ones picked more often
       than the eleventh stay, so a tie at the tenth place keeps fewer than ten rather than
       choosing by column order.
    2. *Order search.* From the empty graph, the search adds the edge ``i -> j`` whose fit most
       lowers the log residual variance of ``j``, among the possible parents that close no cycle,
       whatever the sign of the change, and refits only ``j``'s remaining candidates, until none
       is left. Each fit is an additive model of penalized thin plate regression splines with ten
       basis functions per parent and smoothing chosen by GCV [4]_; with fewer than 30 rows per
       parent each term gets ``ceil(n_samples / (3 n_parents))`` functions, at least 3, and a
       variable with fewer distinct values than that gets one per distinct value. The paper states
       no cap on a variable's parents, and the search sets none; a candidate whose model would have
       as many coefficients as rows is never added.
    3. *Pruning.* Each variable is fit on its parents in that DAG, and a parent stays when its
       smooth term's p-value [5]_ is at most ``alpha``, the paper's rule.

    The rescaling of a column changes no step: the base learners and the smooth terms read a
    variable through an affine map to a fixed interval, and each gain is a ratio of residual
    variances. The fit draws no random numbers and runs on one BLAS thread.

    **Ties.** Gains are compared rounded to 9 decimal places, as :func:`~andrey.hc` compares its
    score changes per row, so two gains that differ only in their rounding tie. A tie goes to the
    edge whose cause, then effect, has the lower rank computed from each column's own data, as for
    :func:`~andrey.hc`, and so do the variables the causal order could place next; the fit runs on
    the columns in that rank order, so the result does not depend on the column order unless two
    columns are identical. The :doc:`column-order guide </docs/guides/column-order>` says what the
    rank keeps under a new column order.

    **Missing values.** A ``NaN`` in ``data`` raises ``ValueError``: every fit reads every row of
    its variables, so a row with a missing value has no place in it. Drop the incomplete rows
    first, which suits values missing completely at random, or use :func:`~andrey.pc` or
    :func:`~andrey.fci`, whose tests delete rows per test.

    **Limitations.** Not benchmarked against other packages yet. Measured on simulated data with
    1,000 rows from Erdos-Renyi graphs of mean degree 2 with standardized columns, ten graphs at 20
    variables and two or three at 100, on one core. On additive-noise data with Gaussian noise it
    oriented 88.5% to 95% of the true edges correctly at 20 variables, depending on the generator,
    and 95% at 100, with median errors (SHD) of 2 to 2.5 and 9 against the empty graph's 20 and 100;
    with Gumbel noise at 20 variables it reversed 22.5% of them. Location-scale data, whose noise
    scale changes with the parents, break the model's fixed noise variance: it reversed 42% of the
    true edges at 20 variables and 32% at 100, with median errors of 11.5 and 54. Linear Gaussian
    data identify only the Markov equivalence class, and there the two directions of an edge fit
    equally, their gains differing by less than ``1e-6``, so the direction of an edge the class
    leaves undirected is arbitrary: it reversed 36% of the true edges at 20 variables, with median
    errors of 11 at 20 variables and 84 at 100. A fit took a median of 3 s at 20 variables and 22 s
    at 100, about 40% of it in the neighbor preselection and 55% in the order search. The smoothing
    of each fit is a local minimum of its GCV score, which can have several, so two edges of nearly
    equal gain can be added in either order. Binary columns, and any column with fewer than 3
    distinct values, are not supported: a smooth term needs at least 3, so such a column raises
    ``ValueError``. The numbers are a reference, not a hard limit: they change with the data
    generator, the density, the method's settings, and the machine.

    References
    ----------
    .. [1] Buhlmann, P., Peters, J., and Ernest, J. (2014). CAM: Causal additive models,
       high-dimensional order search and penalized regression. Annals of Statistics 42(6),
       2526-2556.
    .. [2] Buhlmann, P., and Hothorn, T. (2007). Boosting algorithms: Regularization, prediction
       and model fitting. Statistical Science 22(4), 477-505.
    .. [3] Eilers, P. H. C., and Marx, B. D. (1996). Flexible smoothing with B-splines and
       penalties. Statistical Science 11(2), 89-121.
    .. [4] Wood, S. N. (2003). Thin plate regression splines. Journal of the Royal Statistical
       Society B 65(1), 95-114.
    .. [5] Wood, S. N. (2013). On p-values for smooth components of an extended generalized
       additive model. Biometrika 100(1), 221-228.
    .. [6] Hoyer, P. O., Janzing, D., Mooij, J. M., Peters, J., and Scholkopf, B. (2009). Nonlinear
       causal discovery with additive noise models. NeurIPS 21, 689-696.
    .. [7] Peters, J., Mooij, J. M., Janzing, D., and Scholkopf, B. (2014). Causal discovery with
       continuous additive noise models. Journal of Machine Learning Research 15, 2009-2053.

    Examples
    --------
    >>> import warnings
    >>> import numpy as np
    >>> from andrey import cam
    >>> rng = np.random.default_rng(0)
    >>> x = rng.uniform(-2, 2, 300)
    >>> y = np.sin(2 * x) + 0.3 * rng.standard_normal(300)
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     out = cam(np.column_stack([x, y]))
    >>> out.structure.kind, out.ordering
    ('dag', (0, 1))
    """
    from andrey.fcm.cam import cam as _cam
    from andrey.search.hc import tie_ranks

    X = data_matrix(data, method="cam")
    labels = column_labels(data)
    _smooth_columns(X, labels, "cam")
    constraints = resolve(knowledge, X.shape[1], labels, method="cam")
    # The fit runs on the columns in rank order, so no step reads their positions.
    order = np.argsort(tie_ranks(X), kind="stable")
    forbidden = required = precedence = None
    if constraints is not None:
        ranked = constraints.permuted(order)
        forbidden, required, precedence = ranked.forbidden, ranked.required, ranked.precedence()
    with one_blas_thread():
        result = _cam(
            X[:, order],
            alpha=alpha,
            forbidden=forbidden,
            required=required,
            precedence=precedence,
        )
    back = np.argsort(order)
    dag = result.dag[np.ix_(back, back)]
    return structure_output(dag_from_adjacency(dag), labels=labels, ordering=order[result.order])


#: Fewest rows of the HSIC test, whose null variance has the factor ``(n - 4)(n - 5)``.
MIN_ROWS = 6


def _regression_rows(X: np.ndarray, method: str) -> None:
    """Raise ``ValueError`` when a fit on all the other variables has as many coefficients as rows.

    A fit of one variable on the ``d - 1`` others, each smooth term of at least 3 basis functions,
    2 coefficients after its centering constraint, plus the intercept, has ``2d - 1`` coefficients,
    which need ``2d`` rows. No fit has more terms.
    """
    n, d = X.shape
    if n < 2 * d:
        raise ValueError(
            f"data needs at least {2 * d} rows (samples) over {d} variables, got {n}: {method} "
            f"can regress a variable on the other {d - 1} with a smooth term of at least 2 "
            "coefficients each, plus an intercept, which needs more rows than coefficients"
        )


@method_spec("resit")
def resit(
    data: npt.ArrayLike,
    *,
    alpha: float = 0.05,
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn a DAG with RESIT, regression with subsequent independence tests (additive noise).

    RESIT [1]_ models each variable as a function of its parents plus noise independent of its
    non-descendants, ``x_j = f_j(parents) + e_j``, which identifies the DAG when the functions are
    nonlinear or the noise is not Gaussian [1]_ [3]_. It finds a causal order from the sink up,
    taking as the next sink the variable whose additive-model residual is most independent of the
    others by an HSIC test [2]_, then removes each parent whose removal leaves the residual
    independent of the variable's predecessors. It returns a DAG and a causal order.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``.
    alpha : float, default=0.05
        The significance level of the edge-removal tests: a parent goes when the residual without
        it is independent of the variable's predecessors at a p-value above ``alpha``, in
        ``(0, 1)``. The causal order does not read it. ``0.05`` is the paper's value.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the pairs
        allowed to be adjacent, and temporal tiers, all honored. A variable is the next sink only
        when no remaining variable is in a later tier or is the effect of a required edge out of
        it. Forbidden edges, edges into an earlier tier, and pairs outside the allowed set are in
        no regression and never in the result; each test still reads every remaining variable or
        predecessor. The edge removal never removes a required edge.

    Returns
    -------
    StructureOutput
        Wraps a ``dag`` :class:`~andrey.core.GraphStructure`; ``metadata["algorithm"]`` is
        ``"resit"`` and ``metadata["tie_break"]`` the rule ties followed. ``ordering`` is the
        causal order of the sink search, as a tuple of column indices, causes first.

    Raises
    ------
    TypeError
        If ``knowledge`` is neither ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If an argument has the wrong type, lies outside its range, or is not one of its choices; the
        message names the method, the argument, what it accepts, and the value. Also if ``data`` is
        not a usable 2-D matrix: not numeric, holding ``NaN`` or ``inf``, with fewer than 6 rows or
        fewer than twice as many rows as columns, or with a constant column, a column of fewer than
        3 distinct values, or a column that repeats another up to scale and shift, each named in
        the message. Also if ``knowledge`` names a column ``data`` lacks or contradicts itself, as
        :class:`~andrey.Knowledge` lists.

    Warns
    -----
    ExperimentalWarning
        On the first call in a process: the method has no published benchmark, and its API may
        change without deprecation.

    See Also
    --------
    andrey.cam : Additive effects with Gaussian noise, scored by the likelihood.
    andrey.anm : Decides the direction between two variables under an additive noise model.
    andrey.pc, andrey.ges : Return the equivalence class, which is what linear Gaussian data
        identify.

    Notes
    -----
    **When to use it.** RESIT suits continuous data whose effects are nonlinear and additive in the
    parents, with additive noise of any distribution and no unmeasured common causes [1]_. It reads
    the noise through independence tests, not a likelihood, so the noise need not be Gaussian.

    **When not to.** With linear effects and Gaussian noise only the Markov equivalence class is
    identified: use :func:`~andrey.pc` or :func:`~andrey.ges`. Noise whose scale depends on the
    parents breaks the independence of the residuals. Each test of the causal order reads all the
    remaining variables jointly, so its power falls as the variables grow, and an early wrong sink
    carries through the rest of the order [1]_.

    **The steps.** The paper's settings, in its two phases [1]_:

    1. *Causal order.* Each remaining variable is regressed on the other remaining ones and its
       residual tested against them jointly; the variable whose residual has the largest p-value
       is the next sink, and the variables still remaining are its possible parents.
    2. *Edge removal.* Each variable's possible parents are tried one at a time, in the rank
       order of **Ties**: a parent goes when the residual of the fit on the other current parents
       has a p-value above ``alpha`` against all the variable's predecessors in the order. With one
       parent left, the residual without it is the variable itself.

    Each regression is an additive model of penalized thin plate regression splines with ten
    basis functions per term and smoothing chosen by GCV [4]_, with
    ``ceil(n_samples / (3 n_parents))`` functions per term, at least 3, when there are fewer than
    30 rows per term, and at most one per distinct value. Each test is the HSIC test [2]_ with
    Gaussian kernels on the residual and on the other side's joint squared distances, each
    bandwidth the median of the positive squared distances over every pair of rows, and the null
    approximated by a gamma law matched to its mean and variance.

    The kernels read the columns divided by their standard deviations, and each smooth term reads
    a variable through an affine map to a fixed interval, so rescaling a column changes no step.
    The paper's kernel reads several raw columns at once and so weighs each by its units; on
    columns of one scale, such as standardized data, the two kernels are the same. The fit draws
    no random numbers and runs on one BLAS thread.

    **Ties.** p-values are compared as their logarithms rounded to 9 decimal places, as
    :func:`~andrey.hc` compares its score changes per row, so two p-values that differ only in
    their rounding tie; so do p-values too small to represent. A tie goes to the variable with the
    lower rank computed from each column's own data, as for :func:`~andrey.hc`, and the edge
    removal tries the parents in that rank order; the fit runs on the columns in that rank order,
    so the result does not depend on the column order unless two columns are identical. The
    :doc:`column-order guide </docs/guides/column-order>` says what the rank keeps under a new
    column order.

    **Missing values.** A ``NaN`` in ``data`` raises ``ValueError``: every fit reads every row of
    its variables, so a row with a missing value has no place in it. Drop the incomplete rows
    first, which suits values missing completely at random, or use :func:`~andrey.pc` or
    :func:`~andrey.fci`, whose tests delete rows per test.

    **Limitations.** Not benchmarked against other packages yet. Binary columns, and any column
    with fewer than 3 distinct values, are not supported: every regression fits a smooth term of
    each parent, and such a column raises ``ValueError`` before the fit. Measured on simulated
    additive-noise data with 1,000 rows from Erdos-Renyi graphs of mean degree 2 with standardized
    columns, on one core. It oriented 92% of the true edges correctly at 5 variables and 80% at
    10, with median errors (SHD) of 1 and 5 against the empty graph's 5 and 10. From 20 variables
    its median errors on these data exceed the empty graph's, 23 against 20 at 20 variables and 89
    against 50 at 50: each test of the causal order reads all the remaining variables at once and
    loses power, and an early wrong sink carries through the rest of the order. At 20 variables,
    over ten graphs per generator, it oriented 65% to 76.5% of the true edges correctly on four
    additive-noise generators, the most with Gumbel noise, with median errors of 17 to 23 against
    the empty graph's 20; it oriented 43.5% on linear Gaussian data and 25.5% on location-scale
    data, whose noise scale changes with the parents. A fit took a median of 2 s at 5 variables,
    48 s at 20, 3.8 minutes at 30, and 24 minutes at 50. The causal order fits about ``d^2 / 2``
    additive models of up to ``d - 1`` terms; their cost grows with the terms up to about 35, past
    which each term's basis shrinks to keep three rows per basis function, so a fit's cost grows
    about as ``d^3.3`` up to 50 variables and as ``d^2`` beyond. A fit passes about a minute near
    20 variables, 10 minutes near 40, and an hour near 75; at 100 variables, a fit stopped after
    an hour had run 3,565 of the 5,049 tests of its causal order, and the measured cost per test
    puts the whole fit near two hours. The edge removal tests each parent against all the
    variable's predecessors at once, and many of those p-values lie near ``alpha``, so the order
    the parents are tried in changes the result: trying them in the data's column order instead of
    by rank gave the same causal orders at 20 variables but graphs a median of 10 to 17 edges
    apart, at about the same accuracy. The smoothing of each fit is a local minimum of its GCV
    score, which can have several, so two variables of nearly equal p-value can be taken as the
    sink in either order. The numbers are a reference, not a hard limit: they change with the data
    generator, the density, the method's settings, and the machine.

    References
    ----------
    .. [1] Peters, J., Mooij, J. M., Janzing, D., and Scholkopf, B. (2014). Causal discovery with
       continuous additive noise models. Journal of Machine Learning Research 15, 2009-2053.
    .. [2] Gretton, A., Fukumizu, K., Teo, C. H., Song, L., Scholkopf, B., and Smola, A. J. (2007).
       A kernel statistical test of independence. NeurIPS 20, 585-592.
    .. [3] Hoyer, P. O., Janzing, D., Mooij, J. M., Peters, J., and Scholkopf, B. (2009). Nonlinear
       causal discovery with additive noise models. NeurIPS 21, 689-696.
    .. [4] Wood, S. N. (2003). Thin plate regression splines. Journal of the Royal Statistical
       Society B 65(1), 95-114.

    Examples
    --------
    >>> import warnings
    >>> import numpy as np
    >>> from andrey import resit
    >>> rng = np.random.default_rng(0)
    >>> x = rng.uniform(-2, 2, 300)
    >>> y = np.sin(2 * x) + 0.3 * rng.standard_normal(300)
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     out = resit(np.column_stack([x, y]))
    >>> out.structure.kind, out.ordering
    ('dag', (0, 1))
    """
    from andrey.fcm.resit import resit as _resit
    from andrey.search.hc import tie_ranks

    X = data_matrix(data, method="resit", min_rows=MIN_ROWS)
    labels = column_labels(data)
    _smooth_columns(X, labels, "resit")
    _regression_rows(X, "resit")
    constraints = resolve(knowledge, X.shape[1], labels, method="resit")
    # The fit runs on the columns in rank order, so no step reads their positions.
    order = np.argsort(tie_ranks(X), kind="stable")
    forbidden = required = precedence = None
    if constraints is not None:
        ranked = constraints.permuted(order)
        forbidden, required, precedence = ranked.forbidden, ranked.required, ranked.precedence()
    with one_blas_thread():
        result = _resit(
            X[:, order],
            alpha=alpha,
            forbidden=forbidden,
            required=required,
            precedence=precedence,
        )
    back = np.argsort(order)
    dag = result.dag[np.ix_(back, back)]
    return structure_output(dag_from_adjacency(dag), labels=labels, ordering=order[result.order])


@method_spec("camuv")
def camuv(
    data: npt.ArrayLike,
    *,
    alpha: float = 0.01,
    max_depth: int = 2,
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn a PAG of nonlinear additive effects with hidden variables by CAM-UV.

    CAM-UV [1]_ models each variable, observed or hidden, as a sum of nonlinear functions of its
    direct causes plus independent noise. Where a hidden variable lies between two observed ones,
    as a common cause or on a causal path, their relation cannot be identified by regression and
    independence tests, and CAM-UV marks the pair instead of guessing. It finds each variable's
    direct causes and those pairs.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``.
    alpha : float, default=0.01
        The significance level of every HSIC independence test, in ``(0, 1)``: of the sink tests,
        the tests that remove a candidate parent, and the tests that mark a pair. ``0.01`` is the
        paper's value.
    max_depth : int, default=2
        The maximum conditioning-set size of a sink test, at least ``1``: the most variables
        tested together as a variable's new candidate parents, so the search runs over sets of up
        to ``max_depth + 1`` variables. Larger values find parents that only show together and
        take longer. The default ``2`` is the paper's setting, sets of at most 3 variables.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the pairs
        allowed to be adjacent, and temporal tiers, all honored. A forbidden edge, an edge into an
        earlier tier, and a pair outside ``allowed`` never make a variable a candidate parent, so a
        variable is not tried as the sink of a set that holds a forbidden parent of it, and none
        is in the result. A required edge's cause is a candidate parent from the start, and no
        step removes it; its pair is never marked. A pair forbidden in both directions is never
        marked ``o-o``.

    Returns
    -------
    StructureOutput
        A :class:`~andrey.core.GraphStructure` of kind ``pag``: ``i -> j`` for a direct cause and
        ``i o-o j`` for a pair joined through a hidden variable, whose relation the method cannot
        determine. ``metadata["algorithm"]`` is ``"camuv"``, ``metadata["latent_pairs"]`` lists
        the ``o-o`` pairs as sorted ``[i, j]`` lists, and ``metadata["tie_break"]`` is the rule
        ties followed.

    Raises
    ------
    TypeError
        If ``knowledge`` is neither ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If an argument has the wrong type, lies outside its range, or is not one of its choices; the
        message names the method, the argument, what it accepts, and the value. Also if ``data`` is
        not a usable 2-D matrix: not numeric, holding ``NaN`` or ``inf``, with fewer than 6 rows or
        fewer than twice as many rows as columns, or with a constant column, a column of fewer than
        3 distinct values, or a column that repeats another up to scale and shift, each named in
        the message. Also if ``knowledge`` names a column ``data`` lacks or contradicts itself, as
        :class:`~andrey.Knowledge` lists.

    Warns
    -----
    ExperimentalWarning
        On the first call in a process: the method has no published benchmark, and its API may
        change without deprecation.

    See Also
    --------
    andrey.cam, andrey.resit : Nonlinear additive effects with no hidden variables.
    andrey.rcd : Linear non-Gaussian effects with hidden common causes.
    andrey.fci : Hidden variables with any effects, from conditional independence alone.

    Notes
    -----
    **When to use it.** CAM-UV suits continuous data whose effects are nonlinear and additive in
    the causes, with independent noise of any distribution, where some causes may be unmeasured
    [1]_. An ``o-o`` pair shares a hidden common cause, or one of its variables causes the other
    through a hidden variable; the method does not tell the two apart.

    **When not to.** With linear effects, a path through a hidden variable can be regressed away
    and the pairs are identified differently: use :func:`~andrey.rcd` for non-Gaussian noise.
    Without hidden variables, :func:`~andrey.cam` and :func:`~andrey.resit` also give a causal
    order.

    **The steps.** Every pair of variables is first tested for dependence; then come the paper's
    three steps, with its settings [1]_:

    1. *Candidate parents.* Over every set of 2, then up to ``max_depth + 1`` variables, each
       member dependent on every other member is regressed on the others and its candidate
       parents, and its residual is tested against the others' residuals on their own candidates,
       jointly. The member of largest p-value is a sink of the set when that p-value is above
       ``alpha`` and its residual on its candidates alone is dependent on each other member's,
       every p-value below ``alpha``; the other members then become its candidates. After a sweep
       that adds a candidate, the search starts over from sets of 2. A set that holds a pair
       already related, one a candidate of the other, is skipped: the sink test identifies a sink
       only when each member's candidates lie outside the set.
    2. *Parents.* Each variable's candidates are tried one at a time, in the rank order of
       **Ties**: a candidate goes when the variable's residual on its other candidates is
       independent of the candidate's residual on its own, at a p-value above ``alpha``.
    3. *Hidden paths.* A dependent pair with no edge whose residuals, each on its parents, are
       dependent at a p-value below ``alpha`` is marked ``o-o``.

    The two dependence screens come from the authors' code; the paper's algorithms test every
    member of a set and every pair. Under the model they add no conclusion: two independent
    variables share no noise, so their residuals are independent too, and neither a sink test nor
    step 3 would accept them. With tests at ``alpha`` they skip most sink tests, and a parent whose
    dependence the test misses, such as a weak parent beside a strong one, is not found.

    The paper does not say what to do when the parents form a directed cycle, which its model
    rules out and failed tests can leave: every edge on a directed cycle is dropped, except a
    required one, and step 3 tests the pair. Each regression is an additive model of penalized thin
    plate regression splines with ten basis functions per term and smoothing chosen by GCV [3]_,
    with ``ceil(n_samples / (3 n_terms))`` functions per term, at least 3, when there are fewer
    than 30 rows per term, and at most one per distinct value. Each test is the HSIC test [2]_ with
    Gaussian kernels, each bandwidth the median of the positive squared distances over every pair
    of rows, a side of several residuals read through their joint distances, and the null
    approximated by a gamma law matched to its mean and variance.

    Every step reads the columns divided by their standard deviations, so rescaling a column
    changes no step; the paper's joint kernel reads raw residuals and so weighs each by its units,
    which is the same on standardized data. The fit draws no random numbers and runs on one BLAS
    thread.

    **Ties.** p-values are compared as their logarithms rounded to 9 decimal places, as
    :func:`~andrey.hc` compares its score changes per row, so two p-values that differ only in
    their rounding tie; so do p-values too small to represent. A tie goes to the variable with the
    lower rank computed from each column's own data, as for :func:`~andrey.hc`; the sets run in
    that rank order and the parents are tried in it, since the fit runs on the columns in that rank
    order, so the result does not depend on the column order unless two columns are identical. The
    :doc:`column-order guide </docs/guides/column-order>` says what the rank keeps under a new
    column order.

    **Missing values.** A ``NaN`` in ``data`` raises ``ValueError``: every fit reads every row of
    its variables, so a row with a missing value has no place in it. Drop the incomplete rows
    first, which suits values missing completely at random, or use :func:`~andrey.pc` or
    :func:`~andrey.fci`, whose tests delete rows per test.

    **Limitations.** Not benchmarked against other packages yet. Binary columns, and any column with
    fewer than 3 distinct values, are not supported: every regression fits a smooth term of each
    regressor, and such a column raises ``ValueError`` before the fit. Measured on simulated
    additive-noise data with 1,000 rows from Erdos-Renyi graphs of mean degree 2 with standardized
    columns, ten graphs per setting, on one core. With 20 observed variables and 2 or 4 hidden ones,
    33% and 43% of the pairs it marked ``o-o`` were joined through a hidden variable, and it marked
    17% and 20% of the pairs that were; on data with no hidden variable it marked a mean of 3 pairs
    a fit on nonlinear data and 6 on linear Gaussian data, each a false mark. Its median errors in
    the directed edges (SHD) were 10.5 and 11 against the empty graph's 16 and 13 with hidden
    variables, and 6.5 against 20 without, orienting 78% to 79% of the true edges. Each sink test
    reads the candidates found before it, so the order the sets are visited in changes the result:
    visiting them in the data's column order instead of by rank gave graphs a median of 7 to 17.5
    pairs apart, at about the same accuracy. With one hidden variable per ten observed ones and one
    to three graphs per size, a fit took a median of 16 s at 20 variables, 5.5 minutes at 100, and
    51 minutes at 300, where its directed errors nearly reached the empty graph's, 232 against 239;
    its cost grows about as ``d^2``, so a fit passes about a minute near 40 variables, 10 minutes
    near 130, and an hour near 330. The dependence screens skip most of the sink tests: without them
    a fit took 49 minutes at 50 variables, against 1.8 with them. The numbers are a reference, not a
    hard limit: they change with the data generator, the density, the method's settings, and the
    machine.

    References
    ----------
    .. [1] Maeda, T. N., and Shimizu, S. (2021). Causal additive models with unobserved variables.
       UAI, PMLR 161, 97-106.
    .. [2] Gretton, A., Fukumizu, K., Teo, C. H., Song, L., Scholkopf, B., and Smola, A. J. (2007).
       A kernel statistical test of independence. NeurIPS 20, 585-592.
    .. [3] Wood, S. N. (2004). Stable and efficient multiple smoothing parameter estimation for
       generalized additive models. Journal of the American Statistical Association 99, 673-686.

    Examples
    --------
    >>> import warnings
    >>> import numpy as np
    >>> from andrey import camuv
    >>> rng = np.random.default_rng(0)
    >>> x = rng.uniform(-2, 2, 300)
    >>> y = np.sin(2 * x) + 0.3 * rng.uniform(-1, 1, 300)
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     out = camuv(np.column_stack([x, y]))
    >>> out.structure.kind, out.structure.oriented_edges()
    ('pag', [(0, 1, 'directed')])
    """
    from andrey.fcm.camuv import camuv as _camuv
    from andrey.search.hc import tie_ranks

    X = data_matrix(data, method="camuv", min_rows=MIN_ROWS)
    labels = column_labels(data)
    _smooth_columns(X, labels, "camuv")
    _regression_rows(X, "camuv")
    constraints = resolve(knowledge, X.shape[1], labels, method="camuv")
    # The fit runs on the columns in rank order, so no step reads their positions.
    order = np.argsort(tie_ranks(X), kind="stable")
    forbidden = required = None
    if constraints is not None:
        ranked = constraints.permuted(order)
        forbidden, required = ranked.forbidden, ranked.required
    with one_blas_thread():
        result = _camuv(
            X[:, order], alpha=alpha, max_depth=max_depth, forbidden=forbidden, required=required
        )
    back = np.argsort(order)
    edges, pairs = result.edges[np.ix_(back, back)], result.pairs[np.ix_(back, back)]
    # B[i, j] != 0 is the edge j -> i, and a NaN pair is o-o.
    B = np.where(pairs, np.nan, edges.T.astype(np.float64))
    structure = GraphStructure.from_numpy(marks_from_signed(B, nan_mark="circle"), kind="pag")
    latent = [[int(i), int(j)] for i, j in zip(*np.nonzero(np.triu(pairs)), strict=True)]
    return structure_output(structure, labels=labels, metadata={"latent_pairs": latent})
