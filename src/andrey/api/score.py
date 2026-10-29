"""Score-based facade adapters.

``ges``, ``gies``, ``xges``, ``exact_search``, ``hc``, and ``tabu`` use :mod:`andrey.core` for BIC
scoring and orientation; ``xges``, ``hc``, and ``tabu`` use the Schur delta-BIC. Each also scores
categorical data with BDeu and the multinomial BIC, and mixed data with the degenerate and
conditional Gaussian BICs. Reported objectives are stored in ``metadata``.

CALM lazily imports its stochastic continuous optimizer from the ``[torch]`` extra and returns a
weighted DAG. Its output varies across torch versions, so determinism and quality tests replace a
committed baseline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from andrey.api._adapt import (
    MAX_CATEGORIES,
    categorical_metadata,
    checked_data,
    column_labels,
    copies_named,
    data_matrix,
    default_score,
    int_in_range,
    json_safe,
    structure_output,
)
from andrey.api._method import method_spec
from andrey.core import backend
from andrey.core.knowledge import Knowledge, resolve

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from andrey.core import StructureOutput


@method_spec("ges")
def ges(
    data: npt.ArrayLike,
    *,
    score_func: str | None = None,
    penalty_discount: float = 1.0,
    ebic_gamma: float = 1.0,
    max_parents: int | None = None,
    categorical: Sequence[str | int] | None = None,
    max_categories: int = MAX_CATEGORIES,
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn a CPDAG from observational data with greedy equivalence search (GES).

    GES searches CPDAGs, which represent equivalence classes of graphs. It adds edges while they
    improve a score, the BIC (Bayesian information criterion) by default, then removes edges in a
    backward sweep [1]_.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``.
    score_func : str or None, default=None
        Local score objective: ``"bic"``, the linear-Gaussian BIC, for continuous data. ``"bdeu"``
        is the BDeu score [3]_ and ``"discrete_bic"`` the multinomial BIC [7]_, for categorical data
        coded by whole numbers. ``"degen_gauss"`` (degenerate Gaussian) [4]_ and ``"cond_gauss"``
        (conditional Gaussian) [5]_ are BIC scores for mixed data, continuous and categorical
        columns. ``None`` picks from the declared column types: ``"bdeu"`` when every column is
        categorical, ``"cond_gauss"`` when some are, else ``"bic"``. A column is declared
        categorical by ``categorical`` or by a DataFrame's ``category``, ``bool``, or text type,
        never by its values. See Notes.
    penalty_discount : float, default=1.0
        Multiplier on the BIC penalty of ``log(n_samples)`` per parameter: per parent for the
        linear-Gaussian BIC. Larger values give sparser graphs; ``1.0`` with ``ebic_gamma=0.0`` is
        the standard BIC. Must be finite and non-negative. BDeu ignores it; for the multinomial BIC
        it weighs ``log(n)`` per free parameter of a variable's conditional probability table.
    ebic_gamma : float, default=1.0
        Weight on the extended-BIC term of the linear-Gaussian BIC: each variable's score adds
        ``2 * ebic_gamma * log C(n_variables - 1, n_parents)``, which grows with the number of
        parent sets a variable can choose from. Larger values give sparser graphs; ``0.0`` is the
        standard BIC. Must be finite and non-negative. The other scores ignore it. See Notes.
    max_parents : int or None, default=None
        A cap on every node's parents, at least ``0``: no DAG the returned CPDAG represents gives a
        node more than ``max_parents`` parents, and some DAG reaches each node's count. A move whose
        CPDAG would break the cap, in either phase, is skipped for the next best move. A cap can
        stop the search short of the lowest-scoring graph, which GES otherwise finds. ``None`` sets
        no cap; the forward phase then keeps its gate of ``n_variables / 2``. See Notes.
    categorical : sequence of str or int, optional
        The categorical columns of ``data``, for a mixed-data score: column labels (strings) or
        0-based positions. ``None`` takes a DataFrame's ``category``, ``bool``, and text columns; a
        mixed-data setting passed explicitly also infers every column of at most ``max_categories``
        distinct whole numbers, named in a ``DataTypeWarning``. ``[]`` marks none. The other scores
        take no column types and accept only ``None`` or ``[]``. See *Data types* in the
        :doc:`getting-started guide </docs/guides/getting-started>`.
    max_categories : int, default=10
        The most distinct whole numbers a column may hold to be inferred as categorical, at least
        ``2``. With a continuous score, columns within it get a one-time ``DataTypeWarning`` that
        they may be category codes.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the pairs
        allowed to be adjacent, and temporal tiers, all honored. No move adds a forbidden edge or
        removes a required one, and the search starts from the required edges. The result directs
        every edge the knowledge decides, then Meek's rules direct the edges that follow, as
        :class:`~andrey.Knowledge` describes.

    Returns
    -------
    StructureOutput
        Wraps a ``cpdag`` :class:`~andrey.core.GraphStructure`; ``metadata["algorithm"]`` is
        ``"ges"`` and ``metadata["score"]`` the score of the result; lower is better. For BIC, it is
        the sum over variables of ``n * log(residual variance) + penalty_discount * log(n) *
        n_parents + 2 * ebic_gamma * log C(n_variables - 1, n_parents)``; for BDeu, ``-2`` times the
        log of the structure prior times the marginal likelihood; for the multinomial BIC and the
        mixed-data scores, see Notes. Directed marks
        (``->``) are oriented edges, undirected (``--``) the ones the equivalence class leaves open.
        Under a mixed-data setting, ``metadata["categorical"]`` lists the categorical columns it
        read, by name or 0-based position.

    Raises
    ------
    TypeError
        If ``categorical`` is not a list of column names and positions, or ``knowledge`` is neither
        ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If an argument has the wrong type, lies outside its range, or is not one of its choices; the
        message names the method, the argument, what it accepts, and the value. Also if a given
        ``max_parents`` is below the parents some graph of the required edges of ``knowledge`` gives
        a node; with a categorical score, if ``data`` holds a number that is not whole; or, with a
        mixed-data score, if ``categorical`` names a column ``data`` lacks, or if a categorical
        column holds a number that is not whole or a missing value. Also if ``categorical`` is given
        with another score. With the BIC score, also if a column repeats another up to scale and
        shift given more than two rows or, with more rows than columns, is a linear function of
        other columns. One message names every group of copies and says to keep one column per
        group. Also if ``knowledge`` names a column ``data`` lacks or contradicts itself, as
        :class:`~andrey.Knowledge` lists.

    Warns
    -----
    DataTypeWarning
        When a mixed-data score infers the categorical columns, or finds none and so runs as the
        linear-Gaussian form, or when a continuous setting meets columns that look like category
        codes. Each message warns once per process.

    See Also
    --------
    andrey.fci : Allows unmeasured common causes and selection bias.
    andrey.direct_lingam : Orients every edge when the noise is non-Gaussian.

    Notes
    -----
    **When to use it.** GES suits data with no unmeasured common causes. Its default BIC score
    assumes continuous data, where each variable depends linearly on its causes with Gaussian noise;
    in that setting it is a fast, accurate default. When every column is categorical, the default is
    ``"bdeu"``, and on mixed data ``"cond_gauss"``;
    ``"discrete_bic"`` and ``"degen_gauss"`` also score those data.

    **When not to.** When unmeasured common causes are plausible, use :func:`~andrey.fci`. When the
    noise is non-Gaussian and every edge should be oriented, use :func:`~andrey.direct_lingam`.

    **Choosing the penalty.** The linear-Gaussian BIC charges a variable ``penalty_discount *
    log(n_samples)`` per parent and adds ``2 * ebic_gamma * log C(n_variables - 1, n_parents)``,
    where ``C(n_variables - 1, n_parents)`` counts the parent sets of that size: this extended BIC
    [8]_ charges more as the number of variables grows. A larger value of either gives a sparser
    graph, with fewer false edges and more missed ones. The standard BIC, ``ebic_gamma=0.0``,
    charges a fixed ``log(n_samples)`` per edge while the candidate edges grow with the square of
    the number of variables, so with many variables it keeps many false edges. On simulated
    linear-Gaussian data with 20 to 400 variables and 200 to 4,000 rows, the default
    ``ebic_gamma=1.0`` made fewer errors (SHD) than the standard BIC on average at every number of
    variables and rows measured, and the most where the rows are few for the variables: 24 against
    211 with 200 variables and 2,000 rows (200 true edges), and 46 against 2,162 with 400 variables
    and 200 rows (400 true edges). It misses more weak edges: on four Gaussian networks with
    published parameters, the share of true adjacencies it found fell from 0.84 to 0.77, while the
    share of its adjacencies that were true rose from 0.72 to 0.94. With about
    ``10 * n_variables`` rows or more, ``penalty_discount=2.0`` at ``ebic_gamma=0.0`` made about as
    few errors as the default, and more with fewer rows. Pass ``ebic_gamma=0.0`` for the standard
    BIC. FGES, a fast GES, multiplied the BIC penalty by 4 when searching continuous data on graphs
    of up to a million variables [2]_.

    **How it works.** Without ``max_parents``, a variable that already has more than
    ``n_variables / 2`` parents gains no new parents in the forward phase; the backward phase has
    no such gate. With ``max_parents``, that gate gives way to the cap. One move can add several
    parents to a variable at once, and rebuilding the CPDAG after a move can direct or undirect
    other edges, so the search tests the CPDAG after each move and takes the best move whose CPDAG
    keeps every variable within the cap. A variable's count is its directed parents plus the
    largest set of its undirected neighbors that are adjacent to one another: a graph of the class
    directs such a set into it and no larger one, since two neighbors that are not adjacent would
    add a collider. A cap of ``1`` therefore allows a chain or a tree of undirected edges but not
    a collider. Moves whose score changes are equal after rounding are chosen by each column's
    rank, as in :func:`~andrey.hc`; the :doc:`column-order guide </docs/guides/column-order>` says
    what the rank keeps under a new column order and on another machine.

    **Categorical data.** ``"bdeu"`` takes each distinct value of a column as one
    category, so the codes need not start at 0 or be consecutive; a DataFrame's text, ``bool``, and
    ``category`` columns are read the same way. It scores a variable given its parents by the
    marginal likelihood of its counts in each parent configuration, under a Dirichlet prior with an
    equivalent sample size of 1 spread evenly over the cells [3]_. A structure prior with one
    expected parent per variable multiplies it, from three variables up. Equivalent graphs score the
    same, as GES requires. Moves are scored in this process, without the worker processes the BIC
    search can use. ``"discrete_bic"`` reads the columns the same way and scores a
    variable by ``-2`` times the log-likelihood of its counts in each parent configuration, plus
    ``penalty_discount * log(n)`` for each of the ``q * (r - 1)`` free parameters, with ``r`` its
    categories and ``q`` its parents' joint categories [7]_. Both scores are on the scale of the
    BIC: ``-2`` times a log score.

    **Mixed data.** ``"degen_gauss"`` and ``"cond_gauss"`` score data with
    continuous and categorical columns; the categorical columns are read as :func:`~andrey.pc`
    describes. Each is a BIC: ``-2`` times the fitted log-likelihood of a variable given its
    parents, plus ``penalty_discount * log(n)`` per parameter; the degenerate Gaussian's leaves out
    a constant per variable, which does not change which graph scores best. Equivalent graphs score
    the same. The degenerate Gaussian score [4]_ replaces a categorical column of ``k`` categories
    by ``k - 1`` indicator columns and scores every column as Gaussian, by the residual covariance
    of a linear regression on the parents' columns; on continuous columns only, it equals
    ``"bic"`` at ``ebic_gamma=0.0``. The conditional Gaussian score [5]_ fits the
    continuous variables of a variable and its parents in each joint category of their categorical
    ones, with a Gaussian of
    that category's own mean and a covariance shrunk toward that of all rows by one pseudo-row [6]_,
    and divides by the same fit of the parents alone, so a categorical variable can have continuous
    parents. It counts parameters for every joint category, so a parent with many categories costs
    many parameters; on continuous columns only, it differs from ``"bic"`` at
    ``ebic_gamma=0.0`` by a constant per variable, so it finds the same graph. Moves are scored in
    this process.

    **Missing values.** A ``NaN`` in ``data`` raises ``ValueError``. The score adds one term per
    variable, each over every row, and terms over different rows do not add up to the score of one
    dataset, so the search cannot delete rows per test as the conditional-independence searches do.
    Drop the incomplete rows first, which suits values missing completely at random, or use
    :func:`~andrey.pc` or :func:`~andrey.fci`, whose tests delete rows per test.

    **Limitations.** The benchmark numbers in this paragraph are for the standard BIC,
    ``ebic_gamma=0.0``. Measured on simulated linear-Gaussian data from Erdos-Renyi graphs, with
    ``max_parents=4``, on 4 cores with a cap of 1 h and 28 GB per fit. The largest sizes through
    which its median errors (SHD) stayed below the empty graph's, and through which its fits
    finished within the caps, were 50 and 200 variables at mean degree 1 with 2,000 rows and mean
    degree 2 with 500 rows; 400 and 1,600 at mean degree 2 with 10,000 rows; 200 and 800 at mean
    degree 2 with 2,000 rows; 800 and 3,200 at mean degree 4 with 2,000 rows; and none and 50 at
    mean degree 8 with 2,000 rows. Every ladder ended after two sizes in a row did not beat the
    empty graph. The median fit took 2,626 s at 3,200 variables (mean degree 4, 2,000 rows). The
    measured build applied ``max_parents=4`` as a gate on the forward phase: a node holding 4
    parents gained no new parent while edges were added, and one move could still give a node more.
    On this release, ``max_parents=4`` caps every DAG of the returned class at 4 parents, a stricter
    rule that is not benchmarked yet. With the same gate, mean degree 2, and ten rows per variable,
    at 200 variables its median errors were 206 against the empty graph's 200 at
    ``penalty_discount=1.0`` and 43 at ``penalty_discount=2.0``. On the 223-variable ANDES network
    with 1,000 simulated rows and a forward-phase gate of ``n_variables / 2`` parents, its median
    errors passed the empty graph's (612 against 338). With fewer rows than columns the standard BIC
    is unreliable; the extended BIC made far fewer errors there (see Choosing the penalty). A parent
    set of ``n_samples - 1`` or more variables fits its child exactly, so the search never selects
    one, nor any parent set that leaves its child a residual variance of at most ``1e-12`` times its
    variance. On simulated mixed data (conditional linear Gaussian networks on sparse random graphs,
    the first half of the variables categorical with 2 to 5 categories), from 10 to 40 variables
    with 2,000 rows, its median errors at 40 variables were 8 with ``"degen_gauss"`` and
    6.5 with ``"cond_gauss"``, against 40, and the median fit took 0.75 s and 1.2 s on 4
    cores. On simulated categorical networks with 2 or 3 categories per variable, of the same sizes
    and rows, ``"bdeu"`` gave 6 against 40 at 40 variables, in 0.60 s. The multinomial
    BIC is not benchmarked with GES yet: its tests check it against the formula and run GES on small
    fixed categorical networks. The numbers are a reference, not a hard limit: they change with the
    data generator, the density, the method's settings, and the machine.

    References
    ----------
    .. [1] Chickering, D. M. (2002). Optimal structure identification with greedy search. JMLR 3,
       507-554.
    .. [2] Ramsey, J., Glymour, M., Sanchez-Romero, R., and Glymour, C. (2017). A million variables
       and more: the Fast Greedy Equivalence Search algorithm for learning high-dimensional
       graphical causal models, with an application to functional magnetic resonance images.
       International Journal of Data Science and Analytics 3(2), 121-129.
    .. [3] Heckerman, D., Geiger, D., and Chickering, D. M. (1995). Learning Bayesian networks:
       the combination of knowledge and statistical data. Machine Learning 20(3), 197-243.
    .. [4] Andrews, B., Ramsey, J., and Cooper, G. F. (2019). Learning high-dimensional directed
       acyclic graphs with mixed data-types. Proceedings of Machine Learning Research 104, 4-21.
    .. [5] Andrews, B., Ramsey, J., and Cooper, G. F. (2018). Scoring Bayesian networks of mixed
       variables. International Journal of Data Science and Analytics 6(1), 3-18.
    .. [6] Friedman, J. H. (1989). Regularized discriminant analysis. Journal of the American
       Statistical Association 84(405), 165-175.
    .. [7] Schwarz, G. (1978). Estimating the dimension of a model. Annals of Statistics 6(2),
       461-464.
    .. [8] Chen, J., and Chen, Z. (2008). Extended Bayesian information criteria for model
       selection with large model spaces. Biometrika 95(3), 759-771.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from andrey import ges
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal(500)
    >>> y = rng.standard_normal(500)
    >>> z = x + y + 0.5 * rng.standard_normal(500)
    >>> ges(pd.DataFrame({"x": x, "y": y, "z": z})).structure.oriented_edges()
    [('x', 'z', 'directed'), ('y', 'z', 'directed')]

    On categorical data, with BDeu:

    >>> a = rng.integers(0, 2, 2000)
    >>> b = rng.integers(0, 2, 2000)
    >>> c = np.where(rng.random(2000) < 0.8, a + b, rng.integers(0, 3, 2000))
    >>> data = pd.DataFrame({"a": a, "b": b, "c": c})
    >>> ges(data, score_func="bdeu").structure.oriented_edges()
    [('a', 'c', 'directed'), ('b', 'c', 'directed')]

    On mixed data, with a category column and the conditional Gaussian score:

    >>> g = rng.integers(0, 3, 2000)
    >>> u = rng.standard_normal(2000)
    >>> v = np.array([-1.0, 0.0, 1.0])[g] + u + 0.5 * rng.standard_normal(2000)
    >>> data = pd.DataFrame({"g": pd.Categorical(g), "u": u, "v": v})
    >>> ges(data, score_func="cond_gauss").structure.oriented_edges()
    [('g', 'v', 'directed'), ('u', 'v', 'directed')]
    """
    from andrey.search.ges import ges as _ges

    score_func, categorical = default_score(data, score_func, categorical=categorical)
    X, mask = checked_data(
        data, categorical, "score_func", score_func, max_categories=max_categories, method="ges"
    )
    labels = column_labels(data)
    constraints = resolve(knowledge, X.shape[1], labels, method="ges")
    with copies_named(labels):
        cpdag, score = _ges(
            X,
            score_func=score_func,
            penalty_discount=penalty_discount,
            ebic_gamma=ebic_gamma,
            max_parents=max_parents,
            categorical=mask,
            constraints=constraints,
        )
    return structure_output(
        cpdag,
        labels=labels,
        metadata={
            "score": json_safe(score),
            **categorical_metadata(mask, data),
        },
    )


@method_spec("gies")
def gies(
    data: npt.ArrayLike,
    interventions: npt.ArrayLike | None = None,
    *,
    score_func: str | None = None,
    penalty_discount: float = 1.0,
    ebic_gamma: float = 1.0,
    max_parents: int | None = None,
    categorical: Sequence[str | int] | None = None,
    max_categories: int = MAX_CATEGORIES,
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn an interventional essential graph with greedy interventional equivalence search (GIES).

    GIES learns from data that mixes observation with interventions [1]_: in a row where a variable
    was intervened on, its value was set from outside rather than by its causes. Each variable is
    scored only on the rows where it was not intervened on, and an edge with exactly one end
    intervened on in some rows gets its direction from the data. So the result can orient edges
    that a CPDAG learned from observational data leaves undirected.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``.
    interventions : array-like of shape (n_samples, n_variables), default=None
        ``True`` (or 1) where the row's value of the variable was set by an intervention, ``False``
        (or 0) where its causes set it. ``None`` means no interventions: observational data. Every
        variable needs at least two rows without an intervention, with different values.
    score_func : str or None, default=None
        Local score objective, as for :func:`~andrey.ges`: ``"bic"``, the linear-Gaussian BIC, for
        continuous data. ``"bdeu"`` is the BDeu score and ``"discrete_bic"`` the multinomial BIC,
        for categorical data coded by whole numbers. ``"degen_gauss"`` (degenerate Gaussian) and
        ``"cond_gauss"`` (conditional Gaussian) are BIC scores for mixed data, continuous and
        categorical columns. ``None`` picks from the declared column types: ``"bdeu"`` when every
        column is categorical, ``"cond_gauss"`` when some are, else ``"bic"``. A column is
        declared categorical by ``categorical`` or by a DataFrame's ``category``, ``bool``, or text
        type, never by its values. Every score reads a variable on the rows without an
        intervention on it; see Notes.
    penalty_discount : float, default=1.0
        Multiplier on the BIC penalty of ``log(n_samples)`` per parameter: per parent for the
        linear-Gaussian BIC. ``n_samples`` counts every row. Larger values give sparser graphs;
        ``1.0`` with ``ebic_gamma=0.0`` is the standard BIC. Must be finite and non-negative. BDeu
        ignores it; for the multinomial BIC it weighs ``log(n_samples)`` per free parameter of a
        variable's conditional probability table.
    ebic_gamma : float, default=1.0
        Weight on the extended-BIC term of the linear-Gaussian BIC: each variable's score adds
        ``2 * ebic_gamma * log C(n_variables - 1, n_parents)``, which grows with the number of
        parent sets a variable can choose from. Larger values give sparser graphs; ``0.0`` is the
        standard BIC. Must be finite and non-negative. The other scores ignore it. See Notes.
    max_parents : int or None, default=None
        A cap on every node's parents, at least ``0``: no DAG the returned graph represents gives a
        node more than ``max_parents`` parents (every DAG of a class has the same largest number of
        parents). A step that would give a node more is skipped for the next best step. A cap can
        stop the search short of the lowest-scoring graph. ``None`` sets no cap. See Notes.
    categorical : sequence of str or int, optional
        The categorical columns of ``data``, for a mixed-data score: column labels (strings) or
        0-based positions. ``None`` takes a DataFrame's ``category``, ``bool``, and text columns; a
        mixed-data setting passed explicitly also infers every column of at most ``max_categories``
        distinct whole numbers, named in a ``DataTypeWarning``. ``[]`` marks none. The other scores
        take no column types and accept only ``None`` or ``[]``. See *Data types* in the
        :doc:`getting-started guide </docs/guides/getting-started>`.
    max_categories : int, default=10
        The most distinct whole numbers a column may hold to be inferred as categorical, at least
        ``2``. With a continuous score, columns within it get a one-time ``DataTypeWarning`` that
        they may be category codes.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the pairs
        allowed to be adjacent, and temporal tiers, all honored. No move adds a forbidden edge or
        removes a required one, and the search starts from the required edges. The result directs
        every edge the knowledge decides, then Meek's rules direct the edges that follow, as
        :class:`~andrey.Knowledge` describes.

    Returns
    -------
    StructureOutput
        Wraps a ``cpdag`` :class:`~andrey.core.GraphStructure`, the interventional essential
        graph: directed marks (``->``) are edges every graph in the class orients alike,
        undirected (``--``) the ones it leaves open. ``metadata["algorithm"]`` is ``"gies"`` and
        ``metadata["score"]`` the score of the result, the sum over variables of each one's local
        score on the rows without an intervention on it; lower is better. For BIC, a variable's
        term is ``n_v * log(residual variance) + penalty_discount * log(n_samples) * n_parents + 2 *
        ebic_gamma * log C(n_variables - 1, n_parents)``, where ``n_v`` counts those rows; for the
        other scores, see :func:`~andrey.ges`. Under a mixed-data setting,
        ``metadata["categorical"]`` lists the categorical columns it read, by name or 0-based
        position.

    Raises
    ------
    TypeError
        If ``categorical`` is not a list of column names and positions, or ``knowledge`` is neither
        ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If an argument has the wrong type, lies outside its range, or is not one of its choices; the
        message names the method, the argument, what it accepts, and the value. Also if ``data`` is
        not a 2-D numeric matrix with at least two rows, no ``NaN`` or ``inf``, and no constant
        column; if ``interventions`` does not have ``data``'s shape, holds a value other than 0 or
        1, or leaves a variable fewer than two rows, or one value, without an intervention; with a
        categorical score, if ``data`` holds a number that is not whole; or, with a mixed-data
        score, if ``categorical`` names a column ``data`` lacks, or if a categorical column holds a
        number that is not whole. Also if ``categorical`` is given with another score. With the BIC
        score, also if a column repeats another up to scale and shift given more than two rows or,
        with more rows than columns, is a linear function of other columns. One message names every
        group of copies and says to keep one column per group. Also if a given ``max_parents`` is
        below the parents some graph of the required edges of ``knowledge`` gives a node, or if
        ``knowledge`` names a column ``data`` lacks or contradicts itself, as
        :class:`~andrey.Knowledge` lists.

    Warns
    -----
    DataTypeWarning
        When a mixed-data score infers the categorical columns, or finds none and so runs as the
        linear-Gaussian form, or when a continuous setting meets columns that look like category
        codes. Each message warns once per process.

    See Also
    --------
    andrey.ges : Searches the same way on observational data, without the turning phase.

    Notes
    -----
    **When to use it.** GIES makes the assumptions of :func:`~andrey.ges`: no unmeasured common
    causes and, with the default BIC score on continuous data, linear relations with Gaussian
    noise. When every column is categorical, the default is ``"bdeu"``, and on mixed data
    ``"cond_gauss"``; ``"discrete_bic"`` and ``"degen_gauss"`` also score those data, as
    :func:`~andrey.ges` describes. An intervention replaces the intervened variable's mechanism and
    leaves every other variable's unchanged.

    **Scores under interventions.** Each score reads a variable only on the rows where it was not
    intervened on, as the likelihood of interventional data does [1]_, and every BIC charges
    ``log(n_samples)`` per parameter, with ``n_samples`` counting every row. BDeu and the
    multinomial BIC count each column's categories over every row, so a category seen only under
    an intervention stays one of the variable's categories; BDeu is then the score of Cooper and Yoo
    [3]_ for a mix of experimental and observational data, with its uniform prior. The mixed-data
    scores read the categories, and the conditional Gaussian score its cells, from those rows alone;
    their interventional form follows the same per-variable rule [1]_, and no paper defines these
    scores under interventions. Under every score, two graphs of one interventional class get the
    same score: they differ only in how the edges the class leaves undirected are directed [1]_, the
    two ends of such an edge are intervened on in the same rows, and on any set of rows each score
    gives Markov equivalent graphs the same score.

    **How it works.** The search starts from the empty graph and repeats three phases: adding
    edges, removing edges, and turning one edge around, each while it improves the score, until a
    round of removing and turning changes nothing. Each step takes its phase's move that improves
    the score most, from one equivalence class to a neighboring one; the classes are those of the
    interventional setting: two graphs are equivalent when they agree on the data from every
    observed combination of intervened variables. Steps whose score changes are equal after rounding
    are chosen by each column's rank, as in :func:`~andrey.hc`; the
    :doc:`column-order guide </docs/guides/column-order>` says what the rank keeps.

    **Parent cap.** GIES as published has no parent cap; its authors note that the cost of a step
    grows with the largest set of mutually adjacent variables joined by undirected edges, and that
    bounding the number of edges at each variable keeps it polynomial in the number of variables
    [1]_. An addition or a turn builds a graph of the new class that gives its target variable new
    parents; with ``max_parents``, a step whose graph gives that variable more than ``max_parents``
    parents is skipped for the next best step. Every graph of a class has the same largest number of
    parents, so the cap then holds for every graph of the new class, and a removal never breaks it.
    Under a cap of ``k``, no set of mutually adjacent variables joined by undirected edges holds
    more than ``k + 1`` variables, since some graph of the class gives one of them all the others
    as parents.

    **Choosing the penalty.** The score charges a variable ``penalty_discount * log(n_samples)`` per
    parent plus the extended-BIC term ``2 * ebic_gamma * log C(n_variables - 1, n_parents)``
    [2]_, which grows with the number of variables; :func:`~andrey.ges` describes the choice.
    ``ebic_gamma=0.0`` with ``penalty_discount=1.0`` is the standard BIC. On simulated observational
    linear-Gaussian data with 20 to 400 variables and 200 to 4,000 rows, the default
    ``ebic_gamma=1.0`` made fewer errors (SHD) than the standard BIC on average at every number of
    variables and rows measured: 23 against 210 with 200 variables and 2,000 rows (200 true
    edges).

    **Missing values.** A ``NaN`` in ``data`` raises ``ValueError``. The score adds one term per
    variable, each over every row, and terms over different rows do not add up to the score of one
    dataset, so the search cannot delete rows per test as the conditional-independence searches do.
    Drop the incomplete rows first, which suits values missing completely at random, or use
    :func:`~andrey.pc` or :func:`~andrey.fci`, whose tests delete rows per test.

    **Limitations.** Measured at the standard BIC (``ebic_gamma=0.0``) on simulated linear-Gaussian
    data from Erdos-Renyi graphs with single-variable interventions on 40% of the variables, the
    rows split evenly between observation and each intervention, on 4 cores with a cap of 1 h and 28
    GB per fit. The largest sizes through which its median errors (SHD against the interventional
    essential graph) stayed below the empty graph's, and through which its fits finished within the
    caps, were 50 and 200 variables at mean degree 1 with 2,000 rows; 400 and 1,600 at mean degree 2
    with 10,000 rows; 200 and 800 at mean degree 2 with 2,000 rows; 100 and 400 at mean degree 2
    with 500 rows; 400 and 800 at mean degree 4 with 2,000 rows; and 800 and 800 at mean degree 8
    with 2,000 rows. The ladders ended after two sizes in a row did not beat the empty graph, except
    at mean degree 4 or 8 with 2,000 rows, where they ended when three fits of the next size ran
    past 1 h. The median fit took 1,231 s at 1,600 variables (mean degree 2, 10,000 rows in total).
    Each distinct set of rows without an intervention on a variable keeps its own score over those
    rows, under the BIC a covariance matrix of all the variables, so memory grows with the number
    of intervened variables times the square of the variable count. A parent set of ``m - 1`` or
    more variables, where ``m`` counts the rows without an intervention on the child, fits its
    child exactly under the BIC, so the search never selects one, nor any parent set that leaves its
    child a residual variance of at most ``1e-12`` times its variance. The categorical and
    mixed-data scores are not benchmarked with GIES yet: its tests run them on small fixed
    networks, with and without interventions. The numbers are a reference, not a hard limit: they
    change with the data generator, the density, the method's settings, and the machine.

    References
    ----------
    .. [1] Hauser, A., and Buhlmann, P. (2012). Characterization and greedy learning of
       interventional Markov equivalence classes of directed acyclic graphs. JMLR 13, 2409-2464.
    .. [2] Chen, J., and Chen, Z. (2008). Extended Bayesian information criteria for model
       selection with large model spaces. Biometrika 95(3), 759-771.
    .. [3] Cooper, G. F., and Yoo, C. (1999). Causal discovery from a mixture of experimental and
       observational data. Proceedings of the Fifteenth Conference on Uncertainty in Artificial
       Intelligence, 116-125.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from andrey import gies
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal(600)
    >>> y = x + 0.5 * rng.standard_normal(600)
    >>> y[400:] = rng.standard_normal(200)  # the last 200 rows set y from outside
    >>> intervened = np.zeros((600, 2), dtype=bool)
    >>> intervened[400:, 1] = True
    >>> gies(pd.DataFrame({"x": x, "y": y}), intervened).structure.oriented_edges()
    [('x', 'y', 'directed')]

    On categorical data, declared by a pandas ``category`` type, the default score is BDeu:

    >>> a = rng.integers(0, 3, 2000)
    >>> b = np.where(rng.random(2000) < 0.8, a, rng.integers(0, 3, 2000))
    >>> b[1000:] = rng.integers(0, 3, 1000)  # the last 1000 rows set b from outside
    >>> intervened = np.zeros((2000, 2), dtype=bool)
    >>> intervened[1000:, 1] = True
    >>> data = pd.DataFrame({"a": pd.Categorical(a), "b": pd.Categorical(b)})
    >>> gies(data, intervened).structure.oriented_edges()
    [('a', 'b', 'directed')]
    """
    from andrey.search.gies import gies as _gies
    from andrey.search.gies import intervention_mask

    score_func, categorical = default_score(data, score_func, categorical=categorical)
    X, mask = checked_data(
        data, categorical, "score_func", score_func, max_categories=max_categories, method="gies"
    )
    labels = column_labels(data)
    constraints = resolve(knowledge, X.shape[1], labels, method="gies")
    intervened = None if interventions is None else intervention_mask(interventions, X)
    with copies_named(labels):
        graph, score = _gies(
            X,
            intervened,
            score_func=score_func,
            penalty_discount=penalty_discount,
            ebic_gamma=ebic_gamma,
            max_parents=max_parents,
            categorical=mask,
            constraints=constraints,
        )
    metadata = {"score": json_safe(score), **categorical_metadata(mask, data)}
    return structure_output(graph, labels=labels, metadata=metadata)


@method_spec("xges")
def xges(
    data: npt.ArrayLike,
    *,
    score_func: str | None = None,
    penalty_discount: float = 1.0,
    ebic_gamma: float = 1.0,
    max_parents: int | None = None,
    extended_search: bool = True,
    categorical: Sequence[str | int] | None = None,
    max_categories: int = MAX_CATEGORIES,
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn a CPDAG from observational data with extremely greedy equivalence search (XGES).

    XGES searches the equivalence classes that GES searches [2]_, with GES's moves, the turning move
    of GIES [3]_, and a decomposable score, the BIC by default, but applies them in a different
    order [1]_ (see Notes). On linear-Gaussian data it often finds a graph with a better score than
    GES.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``.
    score_func : str or None, default=None
        Local score objective, as for :func:`~andrey.ges`: ``"bic"``, the linear-Gaussian BIC, for
        continuous data. ``"bdeu"`` is the BDeu score and ``"discrete_bic"`` the multinomial BIC,
        for categorical data coded by whole numbers. ``"degen_gauss"`` (degenerate Gaussian) and
        ``"cond_gauss"`` (conditional Gaussian) are BIC scores for mixed data, continuous and
        categorical columns. ``None`` picks from the declared column types: ``"bdeu"`` when every
        column is categorical, ``"cond_gauss"`` when some are, else ``"bic"``. A column is
        declared categorical by ``categorical`` or by a DataFrame's ``category``, ``bool``, or text
        type, never by its values.
    penalty_discount : float, default=1.0
        Multiplier on the BIC penalty of ``log(n_samples)`` per parameter: per parent for the
        linear-Gaussian BIC. Larger values give sparser graphs; ``1.0`` with ``ebic_gamma=0.0`` is
        the standard BIC. Must be finite and non-negative. BDeu ignores it; for the multinomial BIC
        it weighs ``log(n)`` per free parameter of a variable's conditional probability table.
    ebic_gamma : float, default=1.0
        Weight on the extended-BIC term of the linear-Gaussian BIC: each variable's score adds
        ``2 * ebic_gamma * log C(n_variables - 1, n_parents)``, which grows with the number of
        parent sets a variable can choose from. Larger values give sparser graphs; ``0.0`` is the
        standard BIC. Must be finite and non-negative. The other scores ignore it. See Notes.
    max_parents : int or None, default=None
        A cap on every node's parents, at least ``0``: no DAG the returned CPDAG represents gives a
        node more than ``max_parents`` parents (every DAG of a class has the same largest number of
        parents). A move that would give a node more is never taken, in the base or the extended
        search. A cap can stop the search short of the lowest-scoring graph. ``None`` sets no cap.
        See Notes.
    extended_search : bool, default=True
        Run the extended search after the base search (see Notes). ``False`` returns the base
        search's graph, which is faster.
    categorical : sequence of str or int, optional
        The categorical columns of ``data``, for a mixed-data score: column labels (strings) or
        0-based positions. ``None`` takes a DataFrame's ``category``, ``bool``, and text columns; a
        mixed-data setting passed explicitly also infers every column of at most ``max_categories``
        distinct whole numbers, named in a ``DataTypeWarning``. ``[]`` marks none. The other scores
        take no column types and accept only ``None`` or ``[]``. See *Data types* in the
        :doc:`getting-started guide </docs/guides/getting-started>`.
    max_categories : int, default=10
        The most distinct whole numbers a column may hold to be inferred as categorical, at least
        ``2``. With a continuous score, columns within it get a one-time ``DataTypeWarning`` that
        they may be category codes.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the pairs
        allowed to be adjacent, and temporal tiers, all honored. No move adds a forbidden edge or
        removes a required one, and the search starts from the required edges. The result directs
        every edge the knowledge decides, then Meek's rules direct the edges that follow, as
        :class:`~andrey.Knowledge` describes.

    Returns
    -------
    StructureOutput
        Wraps a ``cpdag`` :class:`~andrey.core.GraphStructure`; ``metadata["algorithm"]`` is
        ``"xges"`` and ``metadata["score"]`` the score of the result, computed as for
        :func:`andrey.ges`; lower is better. Directed marks (``->``) are oriented edges,
        undirected (``--``) the ones the equivalence class leaves open. Under a mixed-data setting,
        ``metadata["categorical"]`` lists the categorical columns it read, by name or 0-based
        position.

    Raises
    ------
    TypeError
        If ``categorical`` is not a list of column names and positions, or ``knowledge`` is neither
        ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If an argument has the wrong type, lies outside its range, or is not one of its choices; the
        message names the method, the argument, what it accepts, and the value. Also if ``data`` is
        not a 2-D numeric matrix with at least two rows and no ``NaN`` or ``inf``, or if a column is
        constant; with a categorical score, if ``data`` holds a number that is not whole; or, with
        a mixed-data score, if ``categorical`` names a column ``data`` lacks, or if a categorical
        column holds a number that is not whole. Also if ``categorical`` is given with another
        score. With the BIC score, also if a column repeats another column up to scale and shift
        given more than two rows or, with more rows than columns, is a linear function of other
        columns. One message names every group of copies and says to keep one column per group.
        Also if a given ``max_parents`` is below the parents some graph of the required edges of
        ``knowledge`` gives a node, or if ``knowledge`` names a column ``data`` lacks or contradicts
        itself, as :class:`~andrey.Knowledge` lists.

    Warns
    -----
    DataTypeWarning
        When a mixed-data score infers the categorical columns, or finds none and so runs as the
        linear-Gaussian form, or when a continuous setting meets columns that look like category
        codes. Each message warns once per process.

    See Also
    --------
    andrey.ges : Adds edges, then removes them, in two separate phases.

    Notes
    -----
    **When to use it.** XGES makes the assumptions of :func:`~andrey.ges`: no unmeasured common
    causes and, with the default BIC score on continuous data, linear relations with Gaussian
    noise. When every column is categorical, the default is ``"bdeu"``, and on mixed data
    ``"cond_gauss"``; ``"discrete_bic"`` and ``"degen_gauss"`` also score those data, as
    :func:`~andrey.ges` describes. Its moves need only a score that adds one term per variable and
    gives equivalent graphs the same score [1]_, which each of these does. It returns the
    equivalence class.

    **How it works.** Each step takes the best improving deletion, else reversal of a directed edge,
    else addition; the search stops when no move improves the score [1]_. The extended search then
    deletes each edge of that graph, best score first, forbids adding it back in that direction, and
    reruns the search. It keeps a result whose score is lower by more than ``2e-7 * n_samples``, and
    stops once a fresh pass over the edges keeps nothing. Moves whose score changes are equal after
    rounding are taken in the order of each column's rank, as in :func:`~andrey.hc`; the
    :doc:`column-order guide </docs/guides/column-order>` says what the rank keeps.

    **Parent cap.** XGES as published [1]_ has no parent cap. An addition or a reversal builds a
    graph of the new class that gives its target variable new parents; with ``max_parents``, a move
    whose graph gives that variable more than ``max_parents`` parents is never taken. Every graph
    of a class has the same largest number of parents, so the cap then holds for every graph of the
    new class, and a deletion never breaks it.

    **Choosing the penalty.** The score charges a variable ``penalty_discount * log(n_samples)`` per
    parent plus the extended-BIC term ``2 * ebic_gamma * log C(n_variables - 1, n_parents)``
    [4]_, which grows with the number of variables; :func:`~andrey.ges` describes the choice.
    ``ebic_gamma=0.0`` with ``penalty_discount=1.0`` is the standard BIC. On simulated
    linear-Gaussian data with 20 to 400 variables and 200 to 4,000 rows, the default
    ``ebic_gamma=1.0`` made fewer errors (SHD) than the standard BIC on average at every number of
    variables and rows measured: 8 against 199 with 200 variables and 2,000 rows (200 true
    edges).

    **Missing values.** A ``NaN`` in ``data`` raises ``ValueError``. The score adds one term per
    variable, each over every row, and terms over different rows do not add up to the score of one
    dataset, so the search cannot delete rows per test as the conditional-independence searches do.
    Drop the incomplete rows first, which suits values missing completely at random, or use
    :func:`~andrey.pc` or :func:`~andrey.fci`, whose tests delete rows per test.

    **Limitations.** The benchmark numbers in this paragraph are for the standard BIC,
    ``ebic_gamma=0.0``. Measured on simulated linear-Gaussian data from Erdos-Renyi graphs, on 4
    cores with a cap of 1 h and 28 GB per fit. The largest sizes through which its median errors
    (SHD) stayed below the empty graph's, and through which its fits finished within the caps, were
    50 and 200 variables at mean degree 1 with 2,000 rows and mean degree 2 with 500 rows; 400 and
    1,600 at mean degree 2 with 10,000 rows and mean degree 4 with 2,000 rows; 200 and 800 at mean
    degree 2 with 2,000 rows; and none and 50 at mean degree 8 with 2,000 rows. Every ladder ended
    after two sizes in a row did not beat the empty graph. The median fit took 828 s at 1,600
    variables (mean degree 4, 2,000 rows). With ten rows per variable at mean degree 2, where its
    errors exceeded the empty graph's from 500 variables, the BIC score still preferred the returned
    graphs to the true one. At 1,000 variables with ten rows per variable, the median fit took under
    a minute at mean degree 2 and under five minutes at mean degree 4, on one core, and a few
    seconds with ``extended_search=False``. The extended search reruns the base search for each edge
    it tries to delete and retries the edges after keeping a result, so its run time grows with the
    number of edges and of results kept. Its BIC score assumes linear relations with Gaussian noise.
    With fewer rows than columns the standard BIC is unreliable. A parent set of ``n_samples - 1``
    or more variables fits its child exactly, so the search never selects one, nor any parent set
    that leaves its child a residual variance of at most ``1e-12`` times its variance. The
    categorical and mixed-data scores are not benchmarked with XGES yet: its tests run them on small
    fixed networks. Under them, each candidate parent is scored by two full local scores rather
    than a rank-one update. The numbers are a reference, not a hard limit: they change with the
    data generator, the density, the method's settings, and the machine.

    References
    ----------
    .. [1] Nazaret, A., and Blei, D. (2024). Extremely greedy equivalence search. Proceedings of
       the Fortieth Conference on Uncertainty in Artificial Intelligence, PMLR 244, 2716-2745.
    .. [2] Chickering, D. M. (2002). Optimal structure identification with greedy search. JMLR 3,
       507-554.
    .. [3] Hauser, A., and Buhlmann, P. (2012). Characterization and greedy learning of
       interventional Markov equivalence classes of directed acyclic graphs. JMLR 13, 2409-2464.
    .. [4] Chen, J., and Chen, Z. (2008). Extended Bayesian information criteria for model
       selection with large model spaces. Biometrika 95(3), 759-771.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from andrey import xges
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal(500)
    >>> y = rng.standard_normal(500)
    >>> z = x + y + 0.5 * rng.standard_normal(500)
    >>> xges(pd.DataFrame({"x": x, "y": y, "z": z})).structure.oriented_edges()
    [('x', 'z', 'directed'), ('y', 'z', 'directed')]

    On categorical data, with BDeu:

    >>> a = rng.integers(0, 2, 2000)
    >>> b = rng.integers(0, 2, 2000)
    >>> c = np.where(rng.random(2000) < 0.8, a + b, rng.integers(0, 3, 2000))
    >>> data = pd.DataFrame({"a": a, "b": b, "c": c})
    >>> xges(data, score_func="bdeu").structure.oriented_edges()
    [('a', 'c', 'directed'), ('b', 'c', 'directed')]
    """
    from andrey.search.xges import xges as _xges

    score_func, categorical = default_score(data, score_func, categorical=categorical)
    X, mask = checked_data(
        data, categorical, "score_func", score_func, max_categories=max_categories, method="xges"
    )
    labels = column_labels(data)
    constraints = resolve(knowledge, X.shape[1], labels, method="xges")
    with copies_named(labels):
        cpdag, score = _xges(
            X,
            score_func=score_func,
            penalty_discount=penalty_discount,
            ebic_gamma=ebic_gamma,
            extended_search=extended_search,
            max_parents=max_parents,
            categorical=mask,
            constraints=constraints,
        )
    metadata = {"score": json_safe(score), **categorical_metadata(mask, data)}
    return structure_output(cpdag, labels=labels, metadata=metadata)


@method_spec("hc")
def hc(
    data: npt.ArrayLike,
    *,
    score_func: str | None = None,
    penalty_discount: float = 1.0,
    ebic_gamma: float = 1.0,
    max_iter: int | None = None,
    max_parents: int | None = None,
    categorical: Sequence[str | int] | None = None,
    max_categories: int = MAX_CATEGORIES,
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn a CPDAG from observational data with hill-climbing search.

    The search repeatedly adds, removes, or reverses the edge that most improves the score, a
    covariance-based delta-BIC by default, then converts the DAG to its Markov equivalence class
    [1]_ [2]_.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``.
    score_func : str or None, default=None
        Local score objective: ``"bic"``, a covariance-based Gaussian BIC for continuous data;
        ``"bdeu"``, the BDeu score, or ``"discrete_bic"``, the multinomial BIC, for categorical data
        coded by whole numbers; or ``"degen_gauss"`` and ``"cond_gauss"``, the degenerate and
        conditional Gaussian BICs for mixed data, as for :func:`~andrey.ges`. ``None`` picks from
        the declared column types: ``"bdeu"`` when every column is categorical, ``"cond_gauss"``
        when some are, else ``"bic"``. A column is declared categorical by ``categorical`` or by a
        DataFrame's ``category``, ``bool``, or text type, never by its values.
    penalty_discount : float, default=1.0
        Multiplier on the BIC penalty of ``log(n_samples)`` per parameter: per parent for the
        Gaussian BIC. Larger values give sparser graphs; ``1.0`` with ``ebic_gamma=0.0`` is the
        standard BIC. Must be finite and non-negative. BDeu ignores it.
    ebic_gamma : float, default=1.0
        Weight on the extended-BIC term of the linear-Gaussian BIC: each variable's score adds
        ``2 * ebic_gamma * log C(n_variables - 1, n_parents)``, which grows with the number of
        parent sets a variable can choose from. Larger values give sparser graphs; ``0.0`` is the
        standard BIC. Must be finite and non-negative. The other scores ignore it. See
        Notes.
    max_iter : int or None, default=None
        Maximum moves, at least ``1``; ``None`` sets no limit. The search still terminates, since
        each move lowers the score. A given ``max_iter`` also bounds the edge count, as each move
        adds at most one edge.
    max_parents : int or None, default=None
        A cap on every node's parents, at least ``0``: the search takes no move that gives a node
        more, so no DAG the returned CPDAG represents gives a node more than ``max_parents`` parents
        (every DAG of a class has the same largest number of parents). A cap can stop the search
        short of the lowest-scoring graph. ``None`` sets no cap.
    categorical : sequence of str or int, optional
        The categorical columns of ``data``, for a mixed-data score: column labels (strings) or
        0-based positions. ``None`` takes a DataFrame's ``category``, ``bool``, and text columns; a
        mixed-data setting passed explicitly also infers every column of at most ``max_categories``
        distinct whole numbers, named in a ``DataTypeWarning``. ``[]`` marks none. The other scores
        take no column types and accept only ``None`` or ``[]``. See *Data types* in the
        :doc:`getting-started guide </docs/guides/getting-started>`.
    max_categories : int, default=10
        The most distinct whole numbers a column may hold to be inferred as categorical, at least
        ``2``. With a continuous score, columns within it get a one-time ``DataTypeWarning`` that
        they may be category codes.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the pairs
        allowed to be adjacent, and temporal tiers, all honored. No move adds a forbidden edge or
        removes a required one, and the search starts from the required edges. The result directs
        every edge the knowledge decides, then Meek's rules direct the edges that follow, as
        :class:`~andrey.Knowledge` describes.

    Returns
    -------
    StructureOutput
        Wraps a ``cpdag`` :class:`~andrey.core.GraphStructure`; ``metadata["algorithm"]`` is
        ``"hc"`` and ``metadata["score"]`` the score of the DAG found, computed as for
        :func:`andrey.ges`; lower is better. Directed marks (``->``) are oriented edges, undirected
        (``--``) the ones the equivalence class leaves open. Under a mixed-data setting,
        ``metadata["categorical"]`` lists the categorical columns it read, by name or 0-based
        position.

    Raises
    ------
    TypeError
        If ``categorical`` is not a list of column names and positions, or ``knowledge`` is neither
        ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If an argument has the wrong type, lies outside its range, or is not one of its choices; the
        message names the method, the argument, what it accepts, and the value. Also if the required
        edges of ``knowledge`` give a node more than ``max_parents`` parents; if ``data`` or
        ``categorical`` does not suit the score, as for :func:`~andrey.ges`. With the BIC score,
        also if a column repeats another up to scale and shift given more than two rows or, with
        more rows than columns, is a linear function of other columns. One message names every group
        of copies and says to keep one column per group. Also if ``knowledge`` names a column
        ``data`` lacks or contradicts itself, as :class:`~andrey.Knowledge` lists, or, under a
        mixed-data score, requires an edge from a continuous into a categorical column.

    Warns
    -----
    SearchLimitWarning
        When the search takes a given ``max_iter`` moves; a larger ``max_iter`` may find a better
        graph.
    DataTypeWarning
        When a mixed-data score infers the categorical columns, or finds none and so runs as the
        linear-Gaussian form, or when a continuous setting meets columns that look like category
        codes. Each message warns once per process.

    Notes
    -----
    The search starts from the empty graph and stops when no single move improves the score, or
    after ``max_iter`` moves when set. Under a mixed-data score, no move gives a categorical column
    a continuous parent, so the DAG found is a conditional linear Gaussian network.

    Under a score-equivalent score, adding ``x -> y`` or ``y -> x`` from the empty graph gives the
    same score. Such ties recur during search, and choosing one move can change later moves and
    the final CPDAG. Each column gets a rank computed from its own data to choose between tied
    moves, and the rank moves with the column. The
    :doc:`column-order guide </docs/guides/column-order>` says what the rank keeps under a new
    column order and on another machine, and how ``andrey.config(tie_break="position")`` orders the
    ties by column position instead.

    **Choosing the penalty.** The score charges a variable ``penalty_discount * log(n_samples)`` per
    parent plus the extended-BIC term ``2 * ebic_gamma * log C(n_variables - 1, n_parents)`` [3]_,
    which grows with the number of variables; :func:`~andrey.ges` describes the choice.
    ``ebic_gamma=0.0`` with ``penalty_discount=1.0`` is the standard BIC. On simulated
    linear-Gaussian data with 20 to 400 variables and 200 to 4,000 rows, the default
    ``ebic_gamma=1.0`` made fewer errors (SHD) than the standard BIC on average at every number of
    variables and rows measured: 109 against 272 with 200 variables and 2,000 rows (200 true edges).
    Its mean errors stayed below the empty graph's in each of the 65 settings, where the standard
    BIC's exceeded it in 37.

    **Missing values.** A ``NaN`` in ``data`` raises ``ValueError``. The score adds one term per
    variable, each over every row, and terms over different rows do not add up to the score of one
    dataset, so the search cannot delete rows per test as the conditional-independence searches do.
    Drop the incomplete rows first, which suits values missing completely at random, or use
    :func:`~andrey.pc` or :func:`~andrey.fci`, whose tests delete rows per test.

    **Limitations.** The benchmark numbers in this paragraph are for the standard BIC,
    ``ebic_gamma=0.0``. Measured on simulated linear-Gaussian data from Erdos-Renyi graphs, with
    ``max_iter=100000``, on 4 cores with a cap of 1 h and 28 GB per fit. The largest sizes through
    which its median errors (SHD) stayed below the empty graph's, and through which its fits
    finished within the caps, were 50 and 1,600 variables at mean degree 1 with 2,000 rows; 200 and
    1,600 at mean degree 2 with 10,000 rows; 100 and 3,200 at mean degree 2 with 2,000 rows; 20 and
    1,600 at mean degree 2 with 500 rows; none and 800 at mean degree 4 with 2,000 rows (below it
    only at 100); and none and 800 at mean degree 8 with 2,000 rows. The ladders ended after two
    sizes in a row did not beat the empty graph, except at mean degree 2 with 2,000 rows, where the
    ladder ended at the largest planned size. Each accuracy stop was set after larger sizes had run.
    The median fit took 148 s at 3,200 variables (mean degree 2, 2,000 rows). It has no move limit
    by default; run time grows with the moves the search takes. A graph that no single move improves
    need not be the best-scoring one. Its BIC score assumes linear relations with Gaussian noise.
    With fewer rows than columns the standard BIC is unreliable. A parent set of ``n_samples - 1``
    or more variables fits its child exactly, so the search never selects one, nor any parent set
    that leaves its child a residual variance of at most ``1e-12`` times its variance. On simulated
    mixed data (conditional linear Gaussian networks on sparse random graphs, the first half of the
    variables categorical with 2 to 5 categories), from 10 to 40 variables with 2,000 rows, its
    median errors at 40 variables were 12.5 with ``"degen_gauss"`` and 11 with
    ``"cond_gauss"``, against 40. On simulated categorical networks with 2 or 3
    categories per variable, of the same sizes and rows, they were 18 with ``"bdeu"``
    and 20 with ``"discrete_bic"``. Every median fit took under 0.2 s at 40
    variables, on 4 cores. The numbers are a reference, not a hard limit: they change with the data
    generator, the density, the method's settings, and the machine.

    References
    ----------
    .. [1] Chickering, D. M., Geiger, D., and Heckerman, D. (1995). Learning Bayesian networks:
       search methods and experimental results. AISTATS, PMLR R0, 112-128.
    .. [2] Scutari, M. (2010). Learning Bayesian networks with the bnlearn R package. Journal of
       Statistical Software 35(3). https://doi.org/10.18637/jss.v035.i03
    .. [3] Chen, J., and Chen, Z. (2008). Extended Bayesian information criteria for model
       selection with large model spaces. Biometrika 95(3), 759-771.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import hc
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal(500)
    >>> y = x + 0.1 * rng.standard_normal(500)
    >>> hc(np.column_stack([x, y])).structure.kind
    'cpdag'
    """
    from andrey.search.hc import hc as _hc

    score_func, categorical = default_score(
        data,
        score_func,
        categorical=categorical,
    )
    X, mask = checked_data(
        data, categorical, "score_func", score_func, max_categories=max_categories, method="hc"
    )
    labels = column_labels(data)
    constraints = resolve(knowledge, X.shape[1], labels, method="hc")
    with copies_named(labels):
        cpdag, score = _hc(
            X,
            score_func=score_func,
            penalty_discount=penalty_discount,
            ebic_gamma=ebic_gamma,
            max_iter=max_iter,
            max_parents=max_parents,
            categorical=mask,
            constraints=constraints,
        )
    metadata = {"score": json_safe(score), **categorical_metadata(mask, data)}
    return structure_output(cpdag, labels=labels, metadata=metadata)


@method_spec("tabu")
def tabu(
    data: npt.ArrayLike,
    *,
    score_func: str | None = None,
    penalty_discount: float = 1.0,
    ebic_gamma: float = 1.0,
    tabu_length: int = 10,
    max_tabu: int | None = None,
    max_iter: int | None = None,
    max_parents: int | None = None,
    categorical: Sequence[str | int] | None = None,
    max_categories: int = MAX_CATEGORIES,
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn a CPDAG from observational data with tabu search.

    Tabu search is hill climbing that also takes non-improving moves and never returns to a recently
    visited DAG [1]_ [2]_ [3]_. Moves are scored as in :func:`andrey.hc`. The result is the CPDAG of
    the best-scoring DAG visited.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``.
    score_func : str or None, default=None
        Local score objective: ``"bic"``, a covariance-based Gaussian BIC for continuous data;
        ``"bdeu"`` or ``"discrete_bic"``, the BDeu score or the multinomial BIC for categorical data
        coded by whole numbers; or ``"degen_gauss"`` or ``"cond_gauss"``, the degenerate or
        conditional Gaussian BIC for mixed data; each as for :func:`~andrey.ges`. ``None`` picks
        from the declared column types: ``"bdeu"`` when every column is categorical,
        ``"cond_gauss"`` when some are, else ``"bic"``. A column is declared categorical by
        ``categorical`` or by a DataFrame's ``category``, ``bool``, or text type, never by its
        values.
    penalty_discount : float, default=1.0
        Multiplier on the BIC penalty of ``log(n_samples)`` per parameter: per parent for the
        Gaussian BIC. Larger values give sparser graphs; ``1.0`` with ``ebic_gamma=0.0`` is the
        standard BIC. Must be finite and non-negative. BDeu ignores it.
    ebic_gamma : float, default=1.0
        Weight on the extended-BIC term of the linear-Gaussian BIC: each variable's score adds
        ``2 * ebic_gamma * log C(n_variables - 1, n_parents)``, which grows with the number of
        parent sets a variable can choose from. Larger values give sparser graphs; ``0.0`` is the
        standard BIC. Must be finite and non-negative. The other scores ignore it. See
        Notes.
    tabu_length : int, default=10
        Number of most recently visited DAGs a move may not return to. At least ``1``.
    max_tabu : int or None, default=None
        Non-improving moves allowed without a better DAG, from ``1`` to ``tabu_length``. ``None``
        uses ``tabu_length``.
    max_iter : int or None, default=None
        Maximum moves, improving or not, at least ``1``; ``None`` sets no limit. The search still
        terminates: non-improving runs are bounded by ``max_tabu``, and each new best DAG scores
        strictly lower.
    max_parents : int or None, default=None
        A cap on every node's parents, at least ``0``: the search takes no move that gives a node
        more, so no DAG the returned CPDAG represents gives a node more than ``max_parents`` parents
        (every DAG of a class has the same largest number of parents). A cap can stop the search
        short of the lowest-scoring graph. ``None`` sets no cap.
    categorical : sequence of str or int, optional
        The categorical columns of ``data``, for a mixed-data score: column labels (strings) or
        0-based positions. ``None`` takes a DataFrame's ``category``, ``bool``, and text columns; a
        mixed-data setting passed explicitly also infers every column of at most ``max_categories``
        distinct whole numbers, named in a ``DataTypeWarning``. ``[]`` marks none. The other scores
        take no column types and accept only ``None`` or ``[]``. See *Data types* in the
        :doc:`getting-started guide </docs/guides/getting-started>`.
    max_categories : int, default=10
        The most distinct whole numbers a column may hold to be inferred as categorical, at least
        ``2``. With a continuous score, columns within it get a one-time ``DataTypeWarning`` that
        they may be category codes.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the pairs
        allowed to be adjacent, and temporal tiers, all honored. No move adds a forbidden edge or
        removes a required one, and the search starts from the required edges. The result directs
        every edge the knowledge decides, then Meek's rules direct the edges that follow, as
        :class:`~andrey.Knowledge` describes.

    Returns
    -------
    StructureOutput
        Wraps a ``cpdag`` :class:`~andrey.core.GraphStructure`; ``metadata["algorithm"]`` is
        ``"tabu"`` and ``metadata["score"]`` the best DAG's score, computed as for
        :func:`andrey.ges`; lower is better. Directed marks (``->``) are oriented edges, undirected
        (``--``) the ones the equivalence class leaves open.

    Raises
    ------
    TypeError
        If ``knowledge`` is neither ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If an argument has the wrong type, lies outside its range, or is not one of its choices; the
        message names the method, the argument, what it accepts, and the value. Also if ``max_tabu``
        is greater than ``tabu_length``; if the required edges of ``knowledge`` give a node more
        than ``max_parents`` parents; or, with a categorical score, if ``data`` holds a number
        that is not whole. With the BIC score, also if a column repeats another up to scale and
        shift given more than two rows or, with more rows than columns, is a linear function of
        other columns. One message names every group of copies and says to keep one column per
        group. Also if ``knowledge`` names a column ``data`` lacks or contradicts itself, as
        :class:`~andrey.Knowledge` lists.

    Warns
    -----
    SearchLimitWarning
        When the search takes a given ``max_iter`` moves; a larger ``max_iter`` may find a better
        graph.
    DataTypeWarning
        When a mixed-data score infers the categorical columns, or finds none and so runs as the
        linear-Gaussian form, or when a continuous setting meets columns that look like category
        codes. Each message warns once per process.

    Notes
    -----
    The search starts from the empty graph. Each step adds the current DAG to the tabu list, which
    keeps the last ``tabu_length`` DAGs visited, then takes the best improving move outside the
    list. Without one, it stops if ``max_tabu`` non-improving moves have occurred since the best
    score last improved; otherwise it takes the best non-improving move, leaving the score unchanged
    or worse. It also stops when every legal move leads into the list, or after ``max_iter`` moves
    when set.

    Until it leaves its first local optimum, the search takes the same moves as :func:`andrey.hc`,
    so with the same ``max_iter`` its score is never worse than hill climbing's. Ties between
    moves break as they do for :func:`andrey.hc`.

    **Choosing the penalty.** The score charges a variable ``penalty_discount * log(n_samples)`` per
    parent plus the extended-BIC term ``2 * ebic_gamma * log C(n_variables - 1, n_parents)`` [4]_,
    which grows with the number of variables; :func:`~andrey.ges` describes the choice.
    ``ebic_gamma=0.0`` with ``penalty_discount=1.0`` is the standard BIC. On simulated
    linear-Gaussian data with 20 to 400 variables and 200 to 4,000 rows, the default
    ``ebic_gamma=1.0`` made fewer errors (SHD) than the standard BIC on average at every number of
    variables and rows measured: 85 against 276 with 200 variables and 2,000 rows (200 true edges).
    Its mean errors stayed below the empty graph's in each of the 65 settings, where the standard
    BIC's exceeded it in 35.

    **Missing values.** A ``NaN`` in ``data`` raises ``ValueError``. The score adds one term per
    variable, each over every row, and terms over different rows do not add up to the score of one
    dataset, so the search cannot delete rows per test as the conditional-independence searches do.
    Drop the incomplete rows first, which suits values missing completely at random, or use
    :func:`~andrey.pc` or :func:`~andrey.fci`, whose tests delete rows per test.

    **Limitations.** The benchmark numbers in this paragraph are for the standard BIC,
    ``ebic_gamma=0.0``. Measured on simulated linear-Gaussian data from Erdos-Renyi graphs, with
    ``max_iter=100000`` and ``max_tabu=10``, on 4 cores with a cap of 1 h and 28 GB per fit. The
    largest sizes through which its median errors (SHD) stayed below the empty graph's, and through
    which its fits finished within the caps, were 50 and 200 variables at mean degree 1 with 2,000
    rows and mean degree 2 with 500 rows; 200 and 800 at mean degree 2 with 10,000 rows; 100 and 800
    at mean degree 2 with 2,000 rows; 20 and 800 at mean degree 4 with 2,000 rows (also below it
    from 100 through 200); and none and 50 at mean degree 8 with 2,000 rows. Every ladder ended
    after two sizes in a row did not beat the empty graph. The accuracy stop at mean degree 2 with
    2,000 rows was set after larger sizes had run. The median fit took 6.6 s at 800 variables (mean
    degree 4, 2,000 rows). It takes hill climbing's moves plus up to ``max_tabu`` non-improving
    moves between new best scores, with no move limit by default; run time grows with the moves
    taken. It avoids only the last ``tabu_length`` DAGs, so a better DAG beyond that reach may be
    missed. Its BIC score assumes linear relations with Gaussian noise. With fewer rows than columns
    the standard BIC is unreliable. A parent set of ``n_samples - 1`` or more variables fits its
    child exactly, so the search never selects one, nor any parent set that leaves its child a
    residual variance of at most ``1e-12`` times its variance. With a categorical score, its tests
    run it on small fixed categorical networks. The numbers are a reference, not a hard limit: they
    change with the data generator, the density, the method's settings, and the machine.

    References
    ----------
    .. [1] Glover, F. (1986). Future paths for integer programming and links to artificial
       intelligence. Computers & Operations Research 13(5), 533-549.
       https://doi.org/10.1016/0305-0548(86)90048-1
    .. [2] Russell, S. J., and Norvig, P. (2009). Artificial Intelligence: A Modern Approach,
       3rd edition. Prentice Hall.
    .. [3] Scutari, M. (2010). Learning Bayesian networks with the bnlearn R package. Journal of
       Statistical Software 35(3). https://doi.org/10.18637/jss.v035.i03
    .. [4] Chen, J., and Chen, Z. (2008). Extended Bayesian information criteria for model
       selection with large model spaces. Biometrika 95(3), 759-771.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import tabu
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal(500)
    >>> y = x + 0.1 * rng.standard_normal(500)
    >>> tabu(np.column_stack([x, y])).structure.kind
    'cpdag'
    """
    from andrey.search.tabu import tabu as _tabu

    score_func, categorical = default_score(data, score_func, categorical=categorical)
    X, mask = checked_data(
        data, categorical, "score_func", score_func, max_categories=max_categories, method="tabu"
    )
    max_tabu = int_in_range(
        tabu_length if max_tabu is None else max_tabu, "max_tabu", 1, tabu_length
    )
    labels = column_labels(data)
    constraints = resolve(knowledge, X.shape[1], labels, method="tabu")
    with copies_named(labels):
        cpdag, score = _tabu(
            X,
            score_func=score_func,
            penalty_discount=penalty_discount,
            ebic_gamma=ebic_gamma,
            tabu_length=tabu_length,
            max_tabu=max_tabu,
            max_iter=max_iter,
            max_parents=max_parents,
            categorical=mask,
            constraints=constraints,
        )
    metadata = {"score": json_safe(score), **categorical_metadata(mask, data)}
    return structure_output(cpdag, labels=labels, metadata=metadata)


@method_spec("exact_search")
def exact_search(
    data: npt.ArrayLike,
    *,
    search_method: str = "astar",
    score_func: str | None = None,
    penalty_discount: float = 1.0,
    ebic_gamma: float = 1.0,
    max_parents: int | None = None,
    categorical: Sequence[str | int] | None = None,
    max_categories: int = MAX_CATEGORIES,
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn the globally optimal DAG for a decomposable score by exact search.

    The optimum is identified up to its Markov equivalence class, and A*
    (``search_method="astar"``) [1]_ and dynamic programming (``"dp"``) [2]_ return the same DAG of
    that class. Runtime is exponential in the variable count, so keep the problem small (roughly
    <= 20 variables).

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``.
    search_method : {"astar", "dp"}, default="astar"
        Exact search strategy: A* search or dynamic programming.
    score_func : str or None, default=None
        Local score: ``"bic"``, the linear-Gaussian BIC; ``"bdeu"`` or ``"discrete_bic"``, the BDeu
        score or the multinomial BIC for categorical data coded by whole numbers; or
        ``"degen_gauss"`` or ``"cond_gauss"``, the degenerate or conditional Gaussian BIC for mixed
        data, as for :func:`~andrey.ges`. Each BIC uses the penalty of ``penalty_discount=1``, and
        the linear-Gaussian BIC adds the extended-BIC term that ``ebic_gamma`` weighs. ``None``
        picks from the declared column types: ``"bdeu"`` when every column is categorical,
        ``"cond_gauss"`` when some are, else ``"bic"``. A column is declared categorical by
        ``categorical`` or by a DataFrame's ``category``, ``bool``, or text type, never by its
        values.
    penalty_discount : float, default=1.0
        Multiplier on the BIC penalty of ``log(n_samples)`` per parameter: per parent for the
        linear-Gaussian BIC. Larger values give sparser graphs; ``1.0`` with ``ebic_gamma=0.0`` is
        the standard BIC. Must be finite and non-negative. BDeu ignores it; for the multinomial BIC
        it weighs ``log(n)`` per free parameter of a variable's conditional probability table.
    ebic_gamma : float, default=1.0
        Weight on the extended-BIC term of the linear-Gaussian BIC: each variable's score adds
        ``2 * ebic_gamma * log C(n_variables - 1, n_parents)``, which grows with the number of
        parent sets a variable can choose from. Larger values give sparser graphs; ``0.0`` is the
        standard BIC. Must be finite and non-negative. The other scores ignore it. See Notes.
    max_parents : int or None, default=None
        A cap on every node's parents, at least ``0``: the search scores only parent sets within it
        and returns the best-scoring DAG whose every node keeps within it, so no DAG the returned
        DAG represents gives a node more than ``max_parents`` parents (every DAG of a class has the
        same largest number of parents). ``None`` sets no cap.
    categorical : sequence of str or int, optional
        The categorical columns of ``data``, for a mixed-data score: column labels (strings) or
        0-based positions. ``None`` takes a DataFrame's ``category``, ``bool``, and text columns; a
        mixed-data setting passed explicitly also infers every column of at most ``max_categories``
        distinct whole numbers, named in a ``DataTypeWarning``. ``[]`` marks none. The other scores
        take no column types and accept only ``None`` or ``[]``. See *Data types* in the
        :doc:`getting-started guide </docs/guides/getting-started>`.
    max_categories : int, default=10
        The most distinct whole numbers a column may hold to be inferred as categorical, at least
        ``2``. With a continuous score, columns within it get a one-time ``DataTypeWarning`` that
        they may be category codes.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the pairs
        allowed to be adjacent, and temporal tiers, all honored. Each variable's parent sets exclude
        the forbidden parents and include the required ones, so the DAG found is the best one that
        satisfies the knowledge.

    Returns
    -------
    StructureOutput
        Wraps a ``dag`` :class:`~andrey.core.GraphStructure` (one DAG of the optimal equivalence
        class); ``metadata["algorithm"]`` is ``"exact_search"``. Under a mixed-data
        setting, ``metadata["categorical"]`` lists the categorical columns it read, by name or
        0-based position.

    Raises
    ------
    TypeError
        If ``categorical`` is not a list of column names and positions, or ``knowledge`` is neither
        ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If an argument has the wrong type, lies outside its range, or is not one of its choices; the
        message names the method, the argument, what it accepts, and the value. Also if the required
        edges of ``knowledge`` give a node more than ``max_parents`` parents; if ``data`` or
        ``categorical`` does not suit the score, as for :func:`~andrey.ges`. With the BIC score,
        also if a column repeats another up to scale and shift given more than two rows or, with
        more rows than columns, is a linear function of other columns. One message names every group
        of copies and says to keep one column per group. Also if ``knowledge`` names a column
        ``data`` lacks or contradicts itself, as :class:`~andrey.Knowledge` lists.

    Warns
    -----
    DataTypeWarning
        When a mixed-data score infers the categorical columns, or finds none and so runs as the
        linear-Gaussian form, or when a continuous setting meets columns that look like category
        codes. Each message warns once per process.

    Notes
    -----
    **When to use it.** With ``"bic"``, exact search makes the assumptions of
    :func:`~andrey.ges`: continuous data, no unmeasured common causes, and linear relations with
    Gaussian noise; the categorical and mixed-data scores below make theirs. It suits a few
    variables, where it returns a graph with the best score rather than a greedy search's local
    optimum.

    **Tied DAGs.** Every DAG of the optimal equivalence class has the same score, so the search's
    own pick among them would follow last-bit rounding and the column order. Exact search instead
    reports one fixed member of the class it found: it keeps the edges every DAG of the class
    shares, and those the knowledge directs, and orients the others by a rank computed from each
    column's own data, as :func:`~andrey.hc` ranks tied moves. The rank moves with the column, so
    reordering the columns reorders the graph's nodes and changes nothing else, unless two columns
    are identical. The rank is arbitrary with respect to causal structure. Rounding differences,
    such as another CPU's BLAS kernel makes, leave the reported DAG unchanged whenever they leave
    the class unchanged; the :doc:`column-order guide </docs/guides/column-order>` says what the
    rank keeps.

    **Categorical data.** ``"bdeu"`` [5]_ and ``"discrete_bic"`` [6]_
    read each distinct value of a column as one category and score a variable by its counts in
    each configuration of its parents, as :func:`~andrey.ges` describes. Both give equivalent graphs
    the same score.

    **Mixed data.** ``"degen_gauss"`` [3]_ and ``"cond_gauss"`` [4]_ score
    data with continuous and categorical columns, as :func:`~andrey.ges` describes, and read the
    categorical columns as :func:`~andrey.pc` does. Both give equivalent graphs the same score, so
    the optimum is again identified up to its equivalence class.

    **Choosing the penalty.** With ``"bic"``, the score charges a variable
    ``log(n_samples)`` per parent plus the extended-BIC term ``2 * ebic_gamma * log C(n_variables -
    1, n_parents)`` [7]_, which grows with the number of variables; :func:`~andrey.ges` describes
    the choice. ``ebic_gamma=0.0`` is the standard BIC. On simulated linear-Gaussian data with 20
    variables and 200 or 1,000 rows, the best-scoring graph under the default ``ebic_gamma=1.0`` had
    fewer errors (SHD) than the standard BIC's on average in each of the 10 settings of graph and
    rows measured: 2.0 against 9.6 with 200 rows and 20 true edges. Larger graphs were not
    measured.

    **Missing values.** A ``NaN`` in ``data`` raises ``ValueError``. The score adds one term per
    variable, each over every row, and terms over different rows do not add up to the score of one
    dataset, so the search cannot delete rows per test as the conditional-independence searches do.
    Drop the incomplete rows first, which suits values missing completely at random, or use
    :func:`~andrey.pc` or :func:`~andrey.fci`, whose tests delete rows per test.

    **Limitations.** The benchmark numbers in this paragraph are for ``ebic_gamma=0.0``. Measured on
    simulated linear-Gaussian data from Erdos-Renyi graphs, on 4 cores with a cap of 1 h and 28 GB
    per fit. The largest sizes through which its median errors (SHD) stayed below the empty graph's,
    and through which its fits finished within the caps, were 24 and 24 variables at mean degree 1
    with 2,000 rows; and 22 and 22 at mean degree 2 with 500, 2,000, or 10,000 rows and mean degree
    4 or 8 with 2,000 rows. Every ladder ended when three fits of the next size ran past 1 h. The
    median fit took 1,457 s at 24 variables (mean degree 1, 2,000 rows). With a mixed-data or
    categorical score it is not benchmarked yet: its tests run it with a mixed-data score on 5
    variables with 3,000 rows, and with a categorical score on small fixed categorical networks. It
    scores every possible parent set of every variable, so its run time at least doubles with each
    added variable. The ``"dp"`` search also keeps one entry per subset of the variables, so its
    memory doubles as well. With fewer rows than columns the standard BIC is unreliable. A parent
    set of ``n_samples - 1`` or more variables fits its child exactly, so the search never selects
    one, nor any parent set that leaves its child a residual variance of at most ``1e-12`` times its
    variance. The numbers are a reference, not a hard limit: they change with the data generator,
    the density, the method's settings, and the machine.

    References
    ----------
    .. [1] Yuan, C., and Malone, B. (2013). Learning optimal Bayesian networks: a shortest path
       perspective. https://doi.org/10.1613/jair.4039
    .. [2] Silander, T., and Myllymaki, P. (2006). A simple approach for finding the globally
       optimal Bayesian network structure. https://arxiv.org/abs/1206.6875
    .. [3] Andrews, B., Ramsey, J., and Cooper, G. F. (2019). Learning high-dimensional directed
       acyclic graphs with mixed data-types. Proceedings of Machine Learning Research 104, 4-21.
    .. [4] Andrews, B., Ramsey, J., and Cooper, G. F. (2018). Scoring Bayesian networks of mixed
       variables. International Journal of Data Science and Analytics 6(1), 3-18.
    .. [5] Heckerman, D., Geiger, D., and Chickering, D. M. (1995). Learning Bayesian networks:
       the combination of knowledge and statistical data. Machine Learning 20(3), 197-243.
    .. [6] Schwarz, G. (1978). Estimating the dimension of a model. Annals of Statistics 6(2),
       461-464.
    .. [7] Chen, J., and Chen, Z. (2008). Extended Bayesian information criteria for model
       selection with large model spaces. Biometrika 95(3), 759-771.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import exact_search
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal(500)
    >>> y = x + 0.1 * rng.standard_normal(500)
    >>> z = y + 0.1 * rng.standard_normal(500)
    >>> exact_search(np.column_stack([x, y, z])).structure.kind
    'dag'
    """
    from andrey.search.exact import exact_search as _exact

    score_func, categorical = default_score(data, score_func, categorical=categorical)
    X, mask = checked_data(
        data,
        categorical,
        "score_func",
        score_func,
        max_categories=max_categories,
        method="exact_search",
    )
    labels = column_labels(data)
    constraints = resolve(knowledge, X.shape[1], labels, method="exact_search")
    with copies_named(labels):
        dag = _exact(
            X,
            search_method=search_method,
            score_func=score_func,
            penalty_discount=penalty_discount,
            ebic_gamma=ebic_gamma,
            categorical=mask,
            constraints=constraints,
            max_parents=max_parents,
        )
    return structure_output(
        dag,
        labels=labels,
        metadata={**categorical_metadata(mask, data)},
    )


@method_spec("calm")
def calm(
    data: npt.ArrayLike,
    *,
    lambda1: float = 0.005,
    screen_alpha: float = 0.01,
    use_moral_graph: bool = True,
    standardize: bool = False,
    w_threshold: float = 0.3,
    gate_threshold: float = 0.5,
    max_iter: int = 25,
    subproblem_iter: int = 3000,
    lr: float = 0.01,
    tau: float = 0.5,
    rho_init: float = 0.001,
    rho_mult: float = 3.0,
    rho_max: float = 1e16,
    h_tol: float = 1e-8,
    seed: int | None = None,
) -> StructureOutput:
    """Learn a weighted DAG with CALM, a continuous-optimization structure learner.

    CALM (Continuous and Acyclicity-constrained L0-penalized likelihood with an estimated Moral
    graph) [1]_ fits a linear-Gaussian DAG with torch. By default it keeps only the candidate edges
    of an estimated moral graph, then minimizes the Gaussian likelihood plus a penalty per edge
    under an acyclicity constraint. It is a continuous-optimization method in the line of NOTEARS
    (:func:`~andrey.notears`), with a likelihood in place of least squares, gates that count edges
    in place of an ``l1`` penalty, and a candidate set restricted to the moral graph. The optimizer
    draws random numbers and is seeded on every call; see ``seed``.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``.
    lambda1 : float, default=0.005
        Weight of the l1 penalty on the sum of the edge gates, at least ``0``. The gates
        approximate a count of edges, so it prices each edge: larger values give sparser graphs.
    screen_alpha : float, default=0.01
        Significance level of the Fisher-Z tests that estimate the moral graph, in ``(0, 1)``.
        Only pairs the screen keeps can become edges; smaller values keep fewer.
    use_moral_graph : bool, default=True
        With ``False`` every pair is a candidate edge, and ``screen_alpha`` has no effect.
    standardize : bool, default=False
        Scale each column to unit variance first.
    w_threshold : float, default=0.3
        An edge is kept when its weight magnitude reaches this value, at least ``0``.
    gate_threshold : float, default=0.5
        An edge is kept when its gate probability reaches this value, in ``[0, 1]``.
    max_iter : int, default=25
        Most quadratic-penalty rounds, each an Adam subproblem, at least ``1``. Stopping on this
        limit before the acyclicity residual clears ``h_tol`` or ``rho`` reaches ``rho_max`` warns
        with :class:`~andrey.SearchLimitWarning`; raise it to let the fit run on.
    subproblem_iter : int, default=3000
        Most Adam steps per round, at least ``1``.
    lr : float, default=0.01
        Adam's learning rate, above ``0``.
    tau : float, default=0.5
        The gate's Gumbel-sigmoid temperature, above ``0``.
    rho_init : float, default=0.001
        The acyclicity penalty ``rho`` of the first round, above ``0``.
    rho_mult : float, default=3.0
        The factor by which ``rho`` grows between rounds, at least ``1``.
    rho_max : float, default=1e16
        The largest ``rho``: the rounds end when it is reached, above ``0``.
    h_tol : float, default=1e-8
        The acyclicity residual of the noise-free gates at or below which the rounds end, at
        least ``0``.
    seed : int or None, default=None
        Seed for the optimizer's torch random number generator. ``None`` uses the
        ``ANDREY_SEED`` environment variable, or ``0`` when it is unset, so every call is seeded
        and repeated calls return the same graph on one machine with the same torch version.
        The draws go to the columns in order of each column's rank, as in :func:`~andrey.hc` (the
        :doc:`column-order guide </docs/guides/column-order>`). ``metadata["seed"]`` records the
        seed used.

    Returns
    -------
    StructureOutput
        Wraps a ``dag`` :class:`~andrey.core.GraphStructure`; ``weighted_adjacency[i, j]`` is the
        coefficient of edge ``i -> j``. ``metadata`` holds ``"algorithm"`` (``"calm"``), the
        ``"seed"`` used, ``"n_edges"``, ``"objective"`` (half the sum of the log residual
        variances plus ``lambda1`` times the edge count; lower is better), and ``"h_final"`` (the
        acyclicity residual of the returned graph, ``0`` when it is acyclic).

    Raises
    ------
    ImportError
        If the ``[torch]`` extra is not installed.
    ValueError
        If an argument has the wrong type, lies outside its range, or is not one of its choices; the
        message names the method, the argument, what it accepts, and the value. Also if the fit's
        loss or weights stop being finite, for example when the data's scale overflows its
        covariance; ``standardize=True`` rescales the columns.

    Notes
    -----
    CALM first estimates the moral graph with IAMB [2]_: for each variable it adds, one at a time,
    the variable most dependent on it given those already added (Fisher-Z at level
    ``screen_alpha``), then drops any member the others make independent of it. The most dependent
    variable has the largest absolute partial correlation; equal ones are chosen by each column's
    rank, as in :func:`~andrey.hc`. A pair is a candidate edge, in either direction, when each is in
    the other's set.

    Each candidate edge ``i -> j`` has a weight and a gate. During the fit a gate is a
    Gumbel-Softmax sample [3]_, ``sigmoid((logit + noise) / tau)``, so ``lambda1`` times the sum
    of the gates approximates a penalty per edge. With ``B`` the gated weights and ``S`` the
    sample covariance, the fit minimizes the Gaussian negative log-likelihood with one noise
    variance per variable [4]_, ``0.5 * sum_j log(((I - B).T @ S @ (I - B))[j, j]) -
    log|det(I - B)|``, plus that penalty and ``0.5 * rho * h**2``. Here ``h = tr((I + G / d)**d) -
    d`` [5]_ for the gates ``G`` and ``d`` variables, and ``h`` is zero exactly when the gated
    graph is acyclic. Adam runs up to ``subproblem_iter`` steps per round; a round ends early once,
    after 50 steps, the loss changes by less than a ``1e-7`` fraction from one step to the next.
    Between rounds ``rho`` grows by the factor ``rho_mult``, the quadratic penalty method [6]_,
    until ``h`` of the noise-free gates ``sigmoid(logit / tau)`` is at most ``h_tol``, ``rho``
    reaches ``rho_max``, or ``max_iter`` rounds have run; stopping on ``max_iter`` warns with
    :class:`~andrey.SearchLimitWarning`.

    If the edges that ``gate_threshold`` and ``w_threshold`` keep still form a cycle, edges are
    removed in order of increasing weight magnitude until none does.

    The defaults are a shorter schedule than the paper's [1]_, which starts ``rho`` at ``1e-5``,
    runs 40,000 steps per round at learning rate ``0.001``, and keeps every edge whose gate
    reaches ``0.5``. ``rho_init=1e-5, subproblem_iter=40000, lr=0.001, w_threshold=0.0`` with a
    larger ``max_iter`` runs that schedule.

    **Missing values.** A ``NaN`` in ``data`` raises ``ValueError``: the loss fits every variable
    over the same rows, so a row with a missing value has no place in it. Drop the incomplete rows
    first, which suits values missing completely at random, or use :func:`~andrey.pc` or
    :func:`~andrey.fci`, whose tests delete rows per test.

    **Limitations.** Measured on simulated linear-Gaussian data from Erdos-Renyi graphs, on 4 cores
    with a cap of 1 h and 28 GB per fit. The largest sizes through which its median errors (SHD)
    stayed below the empty graph's, and through which its fits finished within the caps, were 200
    and 200 variables at mean degree 1 with 2,000 rows; 400 and 400 at mean degree 2 with 500,
    2,000, or 10,000 rows and mean degree 4 with 2,000 rows; and none and 400 at mean degree 8 with
    2,000 rows (below it only from 50 through 400). At 400 variables at mean degree 2 with 500 rows
    and mean degree 4 or 8 with 2,000 rows, 1 or 2 of 5 fits ran past 1 h. Every ladder ended when
    three fits of the next size ran past 1 h. The median fit took 3,332 s at 400 variables (mean
    degree 4, 2,000 rows). With mean degree 2 and ten rows per variable, the median fit took 18 s at
    10 variables and 59 s at 50, where its median errors were 15; the paper's schedule
    (``lr=0.001``, ``max_iter=100``, ``subproblem_iter=40000``, ``rho_init=1e-5``,
    ``w_threshold=0.0``) took 174 s and 472 s for median errors of 6 and 12. Every optimizer step
    computes a determinant and a matrix power of the full weight matrix, one row and one column per
    variable, so a step's cost grows with the cube of the variable count; the defaults allow up to
    75,000 steps (``max_iter`` times ``subproblem_iter``). The fit is nonconvex and can stop at a
    local optimum: on a collider whose two parents are independent, it can return both edges
    reversed and an edge between the parents, with the defaults and with the paper's schedule alike.
    The numbers are a reference, not a hard limit: they change with the data generator, the density,
    the method's settings, and the machine.

    References
    ----------
    .. [1] Jin, K., Ng, I., Zhang, K., and Huang, B. (2026). Revisiting differentiable structure
       learning: inconsistency of l1 penalty and beyond. AAAI 40(27), 22399-22407.
       https://doi.org/10.1609/aaai.v40i27.39398
    .. [2] Tsamardinos, I., Aliferis, C. F., and Statnikov, A. (2003). Algorithms for large scale
       Markov blanket discovery. FLAIRS 2003, 376-380.
    .. [3] Jang, E., Gu, S., and Poole, B. (2017). Categorical reparameterization with
       Gumbel-Softmax. ICLR 2017.
    .. [4] Ng, I., Ghassami, A., and Zhang, K. (2020). On the role of sparsity and DAG constraints
       for learning linear DAGs. NeurIPS 33, 17943-17954.
    .. [5] Yu, Y., Chen, J., Gao, T., and Yu, M. (2019). DAG-GNN: DAG structure learning with graph
       neural networks. ICML 2019, PMLR 97, 7154-7163.
    .. [6] Ng, I., Lachapelle, S., Ke, N. R., Lacoste-Julien, S., and Zhang, K. (2022). On the
       convergence of continuous constrained optimization for structure learning. AISTATS 2022,
       PMLR 151, 8176-8198.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import calm
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal(500)
    >>> y = x + 0.1 * rng.standard_normal(500)
    >>> out = calm(np.column_stack([x, y]), seed=0)  # doctest: +SKIP
    >>> out.structure.kind  # doctest: +SKIP
    'dag'
    """
    if backend.torch() is None:
        raise ImportError("andrey.calm needs torch for its continuous optimizer; install '[torch]'")
    from andrey.search.calm import calm as _calm

    X = data_matrix(data, method="calm")
    labels = column_labels(data)
    with copies_named(labels):
        result = _calm(
            X,
            lambda1=lambda1,
            alpha=screen_alpha,
            tau=tau,
            rho_init=rho_init,
            rho_mult=rho_mult,
            rho_max=rho_max,
            h_tol=h_tol,
            max_outer=max_iter,
            subproblem_iter=subproblem_iter,
            lr=lr,
            w_threshold=w_threshold,
            gate_threshold=gate_threshold,
            standardize=standardize,
            use_moral_graph=use_moral_graph,
            seed=seed,  # ty: ignore[invalid-argument-type]  # an int here
        )
    return structure_output(
        result.dag,
        labels=labels,
        weighted_adjacency=result.weighted_adjacency,
        metadata={
            "n_edges": int((result.weighted_adjacency != 0).sum()),
            "h_final": json_safe(result.h_final),
            "objective": json_safe(result.objective),
        },
    )
