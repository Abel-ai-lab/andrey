"""Functional-causal-model facades: CAM and RESIT fit each variable as a function of its parents."""

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
from andrey.api._method import method_spec
from andrey.core.backend import one_blas_thread
from andrey.core.knowledge import Knowledge, resolve
from andrey.core.score import copies_error, copy_groups
from andrey.fcm.gam import BasisCache

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


def _required_rows(X: np.ndarray, required: np.ndarray, labels: tuple[str, ...] | None) -> None:
    """Raise ``ValueError`` when a column's required parents have more coefficients than rows."""
    name = column_namer(labels)
    bases = BasisCache(X)
    n = X.shape[0]
    for j in np.flatnonzero(required.any(axis=0)).tolist():
        parents = np.flatnonzero(required[:, j]).tolist()
        need = bases.n_coef(parents)
        if need >= n:
            raise ValueError(
                f"cam: knowledge requires {len(parents)} parents of column {name(j)}, whose "
                f"additive model has {need} coefficients, but the fit needs more rows than "
                f"coefficients and data has {n} rows"
            )


@method_spec("cam")
def cam(
    data: npt.ArrayLike,
    *,
    alpha: float = 0.001,
    max_parents: int | None = None,
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
        is at most ``alpha``, in ``(0, 1)``. ``0.001`` is the paper's value at every sample size. A
        larger value keeps more of the true edges a small sample tests weakly, and more false edges
        with them (see Notes).
    max_parents : int or None, default=None
        A cap on every node's parents, at least ``0``: the order search adds no edge that gives a
        node more, so no node of the returned DAG has more than ``max_parents`` parents. A cap can
        stop the search short of the best-scoring DAG. ``None`` sets no cap, as in the paper.
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
        contradicts itself, as :class:`~andrey.Knowledge` lists; if its required edges give a node
        more than ``max_parents`` parents; or if a column's required parents need more
        coefficients than ``data`` has rows, naming the column.

    Warns
    -----
    ExperimentalWarning
        On the first call in a process: the method has no published benchmark, and its API may
        change without deprecation.
    DataTypeWarning
        When a column holds at most 10 distinct whole numbers and so may hold category codes,
        which the fit reads as numbers. Each message warns once per process.

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
       choosing by column order. When more than ten tie at the top count, none is picked more
       often than the eleventh, and the variable has no possible parents.
    2. *Order search.* From the empty graph, the search adds the edge ``i -> j`` whose fit most
       lowers the log residual variance of ``j``, among the possible parents that close no cycle,
       whatever the sign of the change, and refits only ``j``'s remaining candidates, until none
       is left. Each fit is an additive model of penalized thin plate regression splines with ten
       basis functions per parent and smoothing chosen by GCV [4]_; with fewer than 30 rows per
       parent each term gets ``ceil(n_samples / (3 n_parents))`` functions, at least 3, and a
       variable with fewer distinct values than that gets one per distinct value. The paper states
       no cap on a variable's parents, so ``max_parents`` sets none by default; a candidate whose
       model would have as many coefficients as rows is never added.
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
    and at 100 variables, on one core. On additive-noise data with Gaussian noise it oriented 88.5%
    to 95% of the true edges correctly at 20 variables, depending on the generator, and 97.5% at
    100, with median errors (SHD) of 2 to 2.5 and 8 against the empty graph's 20 and 100; with
    Gumbel noise at 20 variables it reversed 22.5% of them. Location-scale data, whose noise scale
    changes with the parents, break the model's fixed noise variance: it reversed 42% of the true
    edges at 20 variables and 32% at 100, with median errors of 11.5 and 55.5. Linear Gaussian data
    identify only the Markov equivalence class, and there the two directions of an edge fit equally,
    their gains differing by less than ``1e-6``, so the direction of an edge the class leaves
    undirected is arbitrary: it reversed 36% of the true edges at 20 variables and 32% at 100, with
    median errors of 11 and 68. The model has no hidden common causes: with 2 hidden variables among
    22, over ten graphs of the Gaussian additive-noise generator, its median errors were 12 against
    the empty graph's 16, mostly extra edges, and it joined 42 of the 49 pairs of observed variables
    that share only a hidden cause. Effects that interact are outside the model too: with ``x3`` the
    product of ``x1`` and ``x2`` plus noise, on 1,000 rows and standard normal causes, whose product
    has no additive part, it found at most one of the two edges into ``x3`` and returned an edge out
    of ``x3`` in 6 of 10 fits; with causes uniform on ``(0, 2)`` it made ``x3`` the cause of both,
    and added an edge between them, in 8 of 10. On the additive-noise data a fit took a median of 3
    s at 20 variables and 21 s at 100, about 40% of it in the neighbor preselection and 55% in the
    order search. The smoothing of each fit is a local minimum of its GCV score, which can have
    several, so two edges of nearly equal gain can be added in either order. Binary columns, and any
    column with fewer than 3 distinct values, are not supported: a smooth term needs at least 3, so
    such a column raises ``ValueError``. Categorical columns are not modeled: text labels raise
    ``ValueError``, and integer codes are read as numbers, a smooth function of the code, with a
    :class:`~andrey.DataTypeWarning`. The numbers are a reference, not a hard limit: they change
    with the data generator, the density, the method's settings, and the machine.

    **Large graphs and many rows.** On the rescaled additive-noise data with 1,000 rows, over five
    graphs at 200 variables, three at 500, and two at 1,000 and at 2,000, its median errors stayed
    below the empty graph's at every size but grew faster than the variables, from 8% of the empty
    graph's at 100 variables to 37% at 2,000. Nearly all of them were extra edges, and on one graph
    each at 100 and 500 variables all but one lay between variables unrelated or more than three
    edges apart in the true graph: the preselection picks, among more variables, those that fit a
    variable's residuals best by chance, and the pruning tests them on the same rows, so of the
    searched edges between such pairs it kept 1.1% and 5.4%. The preselection fits ``d - 1`` base
    learners for each of the ``d`` variables, so its cost grows as ``d^2`` and takes over the fit,
    85% of it at 1,000 variables and 88% at 2,000; a fit's cost grows about as ``d^1.6`` up to 1,000
    variables and ``d^1.9`` from 1,000 to 2,000. A fit passes about a minute near 200 variables, 10
    minutes near 800, and an hour near 2,000, where it took a median of 57 minutes and peaked at
    0.64 GB of memory. With 20 variables, over three graphs each, a fit took a median of 5 s at
    5,000 rows and 17.5 s at 20,000 and peaked at 140 MB and 214 MB, about 100 MB of it the imports;
    the median errors stayed at 2. Above 2,000 distinct values a column's thin plate basis is built
    on a fixed subsample of 2,000 of them: at 5,000 and 20,000 rows that returned the same graph as
    knots on every value in 6 of 6 fits, in up to a fifth less time.

    **Small samples and skewed causes.** With few rows per variable it keeps false edges: on
    independent columns it kept a median of 26 at 40 rows and 60 variables and 7.5 at 200 rows and
    60 variables, against 0 at 500 rows and 20. The search keeps each variable's best-fitting
    candidates and the pruning tests them on the same rows, and the smooth-term tests run slightly
    liberal at small samples: an unrelated parent among five reached a p-value of at most 0.01 in
    3.3% of fits at 100 rows. A larger ``alpha`` trades false edges for true ones: at 100 rows,
    ``alpha=0.01`` found 2 more of 100 true adjacencies on additive-noise graphs of 10 variables and
    5 more false ones. A strongly skewed cause can be reversed: with a lognormal cause of log-scale
    2 and an effect of ``log(x)`` plus noise, it reversed the edge in 6 of 10 fits. A
    variance-stabilizing transform such as the log of a positive column suits data whose effect
    stays nonlinear on the new scale: with an effect of ``sin(2 log(x) / 3)`` at log-scale 3, it
    found the edge in 6 of 10 fits on ``x`` and 10 of 10 on ``log(x)``. An effect that becomes
    linear, as ``log(x)`` itself does, leaves a linear Gaussian pair, whose direction no fit
    identifies.

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
        _required_rows(X, constraints.required, labels)
        ranked = constraints.permuted(order)
        forbidden, required, precedence = ranked.forbidden, ranked.required, ranked.precedence()
    with one_blas_thread():
        result = _cam(
            X[:, order],
            alpha=alpha,
            forbidden=forbidden,
            required=required,
            precedence=precedence,
            max_parents=max_parents,
        )
    back = np.argsort(order)
    dag = result.dag[np.ix_(back, back)]
    return structure_output(dag_from_adjacency(dag), labels=labels, ordering=order[result.order])


#: Fewest rows of the HSIC test, whose null variance has the factor ``(n - 4)(n - 5)``.
MIN_ROWS = 6


def _regression_rows(X: np.ndarray) -> None:
    """Raise ``ValueError`` when the first regression has as many coefficients as rows.

    The first step fits each variable on the ``d - 1`` others, each smooth term of at least 3
    basis functions, 2 coefficients after its centering constraint, plus the intercept: ``2d - 1``
    coefficients, which need ``2d`` rows. A later fit has fewer terms or fewer coefficients.
    """
    n, d = X.shape
    if n < 2 * d:
        raise ValueError(
            f"data needs at least {2 * d} rows (samples) over {d} variables, got {n}: resit "
            f"regresses each variable on the other {d - 1} with a smooth term of at least 2 "
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
    FloatingPointError
        If an HSIC test gives a non-finite value, rather than let it steer a choice.

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

    The kernels read the columns centered and divided by their standard deviations, and each smooth
    term reads a variable through an affine map to a fixed interval, so rescaling or shifting a
    column changes no step; the centering also keeps a large offset from cancelling the distances'
    digits. The fit draws no random numbers and runs on one BLAS thread.

    **Ties.** p-values are compared as their logarithms rounded to 9 decimal places, as
    :func:`~andrey.hc` compares its score changes per row, so two p-values that differ only in
    their rounding tie. The logarithms are computed directly, so p-values below the smallest
    double, about ``1e-308``, which strongly dependent residuals reach with a few thousand rows,
    keep their order. A tie goes to the variable with the lower rank computed from each column's
    own data, as for :func:`~andrey.hc`, and the edge removal tries the parents in that rank order;
    the fit runs on the columns in that rank order, so the result does not depend on the column
    order unless two columns are identical.
    The :doc:`column-order guide </docs/guides/column-order>` says what the rank keeps under a new
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
    puts the whole fit near two hours. Its accuracy ends sooner: over five graphs per size, its
    median errors are at most half the empty graph's up to 10 variables and 80% or more of it from
    12 variables. Each HSIC test builds matrices with one row and one column per sample, so time and
    memory grow with the square of ``n_samples``: at 5 variables a fit took 2 s and peaked at 0.17
    GB with 1,000 rows, 25 s and 0.75 GB with 4,000, and 98 s and 2.6 GB with 8,000; on Sachs's
    7,466 rows and 11 variables it took 6 minutes and 2.3 GB, and kept all 55 possible edges, as
    every removal test rejected. The edge removal tests each parent against all the variable's
    predecessors at once, and many of those p-values lie near ``alpha``, so the order the parents
    are tried in changes the result: trying them in the data's column order instead of by rank gave
    the same causal orders at 20 variables but graphs a median of 10 to 17 edges apart, at about the
    same accuracy. A cause with a long tail can reverse its edge: with a lognormal cause of log-sd 3
    and an effect of its logarithm plus noise, on 500 rows, the true order came out in 0 of 5 fits,
    against 5 of 5 at log-sd 1 and 2; replacing the cause by its ranks before the fit gave 5 of 5
    at every log-sd, while its logarithm, which makes that pair linear and Gaussian, did not. The
    smoothing of each fit is a local minimum of its GCV score, which can have several, so two
    variables of nearly equal p-value can be taken as the sink in either order. The numbers are a
    reference, not a hard limit: they change with the data generator, the density, the method's
    settings, and the machine.

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
    _regression_rows(X)
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
