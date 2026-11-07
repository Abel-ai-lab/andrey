"""Score-based facade adapters.

``ges``, ``gies``, ``xges``, ``exact_search``, ``hc``, and ``tabu`` use :mod:`andrey.core` for BIC
scoring and orientation; ``xges``, ``hc``, and ``tabu`` use the Schur delta-BIC. Each also scores
categorical data with BDeu and the multinomial BIC, and mixed data with the degenerate and
conditional Gaussian BICs. With ``env``, ``ges``, ``hc``, and ``tabu`` append the environment index
to the data as a last column no other column causes, and drop it from the result again, its edges
listed in ``metadata`` (:func:`~andrey.api._adapt.cpdag_search`). ``gnies`` scores environments that
differ in noise variances (:mod:`andrey.search.gnies`) and searches with the GIES engine. Reported
objectives are stored in ``metadata``.

CALM lazily imports its stochastic continuous optimizer from the ``[torch]`` extra and returns a
weighted DAG. Its output varies across torch versions, so determinism and quality tests replace a
committed baseline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from andrey.api._adapt import (
    categorical_metadata,
    column_labels,
    column_namer,
    copies_named,
    cpdag_search,
    data_matrix,
    declared_columns,
    fit_inputs,
    json_safe,
    label_of,
    structure_output,
    tabu_cap,
)
from andrey.api._method import method_spec
from andrey.core import backend
from andrey.core.knowledge import Knowledge, resolve

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy.typing as npt

    from andrey.core import StructureOutput


@method_spec("ges")
def ges(
    data: npt.ArrayLike,
    env: npt.ArrayLike | str | None = None,
    *,
    score_func: str | None = None,
    penalty_discount: float = 1.0,
    ebic_gamma: float = 1.0,
    equivalent_sample_size: float = 1.0,
    max_parents: int | None = None,
    categorical: Sequence[str | int] | None = None,
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn a CPDAG from observational data with greedy equivalence search (GES).

    GES searches CPDAGs, which represent equivalence classes of graphs. It adds edges while they
    improve a score, the BIC (Bayesian information criterion) by default, then removes edges in a
    backward sweep [1]_.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables), or a list or mapping of them
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names become
        the structure's ``labels``. A column that ``env`` names is not a variable: it leaves the
        data before the fit, and ``categorical``, ``knowledge``, and the result's positions count
        the other columns. A list of DataFrames or 2-D arrays gives each environment's rows apart,
        labeled by position ``0, 1, ...``, and a mapping labels each by its key: the rows are
        stacked in that order, and ``env`` stays ``None``. A position is the environment's number,
        so with ``"bic"`` and three or more environments, the order of the list can change the
        result (see Notes).
    env : array-like of shape (n_samples,) or (n_samples, 1), str, or None, default=None
        The environment of each row: one label per row, strings or finite numbers, not both, or the
        name of a column of the DataFrame ``data`` that holds them. The environment index then joins
        the search as a variable that no data variable causes; see Notes. Numbers are coerced to
        ``float64`` and read as numbers, whole numbers under a categorical or mixed-data setting;
        string labels are coded ``0, 1, ...`` in sorted order, whatever the row order. When every
        row has the same value, the data come from one environment, and the graph equals the one
        without ``env``. ``None``, the default, fits ``data`` as one environment.
    score_func : str or None, default=None
        Local score objective: ``"bic"``, the linear-Gaussian BIC, for continuous data. ``"bdeu"``
        is the BDeu score [3]_ and ``"discrete_bic"`` the multinomial BIC [7]_, for categorical data
        coded by whole numbers. ``"degen_gauss"`` (degenerate Gaussian) [4]_ and ``"cond_gauss"``
        (conditional Gaussian) [5]_ are BIC scores for mixed data, continuous and categorical
        columns. ``None`` picks from the declared column types: ``"bdeu"`` when every column is
        categorical, ``"cond_gauss"`` when some are, else ``"bic"``. A column is declared
        categorical by ``categorical`` or by a DataFrame's ``category``, ``bool``, or text type,
        never by its values. See Notes.
        ``metadata["score_func"]`` names the score the fit used: the one picked from the
        declared column types when this is ``None``.
    penalty_discount : float, default=1.0
        Multiplier on the BIC penalty of ``log(n_samples)`` per parameter: per parent for the
        linear-Gaussian BIC. Larger values give sparser graphs; ``1.0`` with ``ebic_gamma=0.0`` is
        the standard BIC. Must be finite and non-negative. BDeu ignores it; for the multinomial BIC
        it weighs ``log(n)`` per free parameter of a variable's conditional probability table.
    ebic_gamma : float, default=1.0
        Weight on the extended-BIC term of the linear-Gaussian BIC: each variable's score adds
        ``2 * ebic_gamma * log C(n_variables - 1, n_parents)``, which grows with the number of
        parent sets a variable can choose from. Larger values give sparser graphs; ``0.0`` is the
        standard BIC. Must be from 0 to 1. Only ``"bic"`` uses it: passing it with another score
        raises ``ValueError``. See Notes.
    equivalent_sample_size : float, default=1.0
        The equivalent sample size of the BDeu score: how many rows the uniform prior over each
        variable's conditional distributions is worth. A larger value weighs the prior more
        against the data, and the graph can change with it. Must be finite and positive. Only
        ``"bdeu"`` uses it: passing it with another score raises ``ValueError``.
    max_parents : int or None, default=None
        A cap on every node's parents, at least ``0``: no DAG the returned CPDAG represents gives a
        node more than ``max_parents`` parents, and some DAG reaches each node's count. A move whose
        CPDAG would break the cap, in either phase, is skipped for the next best move. A cap can
        stop the search short of the lowest-scoring graph, which GES otherwise finds. ``None`` sets
        no cap; the forward phase then keeps its gate of ``n_variables / 2``. With ``env``, the
        index counts as a parent of each variable it points into. See Notes.
    categorical : sequence of str or int, or None, default=None
        The categorical columns of ``data``, for a mixed-data score: column labels (strings) or
        0-based positions. ``None`` takes a DataFrame's ``category``, ``bool``, and text columns,
        never a column by its values. ``[]`` marks none. A categorical score reads every column as
        categorical, and a continuous score raises ``ValueError`` on a declared column, named here
        or by its type. See *Data types* in the
        :doc:`getting-started guide </docs/guides/getting-started>`.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the pairs
        allowed to be adjacent, and temporal tiers, all honored. No move adds a forbidden edge or
        removes a required one, and the search starts from the required edges. The result directs
        every edge the knowledge decides, then Meek's rules direct the edges that follow, as
        :class:`~andrey.Knowledge` describes. With ``env``, the environment index is unconstrained.

    Returns
    -------
    StructureOutput
        Wraps a ``cpdag`` :class:`~andrey.core.GraphStructure`; ``metadata["algorithm"]`` is
        ``"ges"`` and ``metadata["score"]`` the score of the result; lower is better. For BIC, it is
        the sum over variables of ``n * log(residual variance) + penalty_discount * log(n) *
        n_parents + 2 * ebic_gamma * log C(n_variables - 1, n_parents)``; for BDeu, ``-2`` times the
        log of the structure prior times the marginal likelihood; for the multinomial BIC and the
        mixed-data scores, see Notes. Directed marks (``->``) are oriented edges, undirected
        (``--``) the ones the equivalence class leaves open. Under a mixed-data setting,
        ``metadata["categorical"]`` lists the categorical columns it read, by name or 0-based
        position. With ``env``, the graph over the data variables is partially directed: the index
        directs edges that the CPDAG of the pooled rows leaves undirected, so the graph stands for
        the DAGs that agree on every direction the index identifies, and is tagged ``cpdag``, as
        :func:`~andrey.cdnod`'s is. ``metadata["index_edges"]`` then lists the index's edges, each
        ``env -> variable``, as ``[variable, "directed"]`` pairs, and
        ``metadata["changing_variables"]`` the same variables, by name or 0-based position, in
        column order: the variables whose causal module, the distribution given their causes,
        changes across the environments, when the index is only a cause and no hidden common cause
        acts. Their descendants change in distribution too, through them, and are not listed. Both
        are empty for a single environment, and ``metadata["score"]`` is the score of the graph over
        the data and the index. ``metadata["tie_break"]`` is the tie-break rule the fit ran under,
        ``andrey.config.tie_break``, and ``metadata`` also records the penalties its score used,
        under their parameter names.

    Raises
    ------
    TypeError
        If an argument has the wrong type, such as a string for a number; the message names the
        method, the argument, what it accepts, and the value. Also if ``categorical`` is not a list
        of column names and positions, or ``knowledge`` is neither ``None`` nor a
        :class:`~andrey.Knowledge`.
    ValueError
        If an argument lies outside its range or is not one of its choices; the message names the
        method, the argument, what it accepts, and the value. Also if a given ``max_parents`` is
        below the parents some graph of the required edges of ``knowledge`` gives a node; with a
        categorical score, if ``data`` holds a number that is not whole; or, with a mixed-data
        score, if ``categorical`` names a column ``data`` lacks, or if a categorical column holds a
        number that is not whole or a missing value. Also if a continuous score meets a column
        declared categorical, by ``categorical`` or by its type. With the BIC score, also if a
        column repeats another up to scale and shift given more than two rows or, with more rows
        than columns, is a linear function of other columns. One message names every group of copies
        and says to keep one column per group. Also if ``knowledge`` names a column ``data`` lacks
        or contradicts itself, as :class:`~andrey.Knowledge` lists. Also if ``env`` does not hold
        one label per row, misses a label or is infinite, or mixes strings and numbers, or, with a
        categorical or mixed-data score, holds a number that is not whole.

    Warns
    -----
    DataTypeWarning
        When a mixed-data score finds no declared categorical column and so runs as the
        linear-Gaussian form, or when a setting reads columns that look like category codes as
        numbers. Each message warns once per process.

    Notes
    -----
    **When to use.** GES suits i.i.d. observational data with no unmeasured common causes, in which
    every independence follows from the graph (faithfulness) [1]_. Its default BIC score assumes
    continuous data, where each variable depends linearly on its causes with Gaussian noise; when
    every column is categorical the default score is ``"bdeu"``, on mixed data ``"cond_gauss"``, and
    ``"discrete_bic"`` and ``"degen_gauss"`` also score those data. It suits sparse graphs, where
    every variable has few parents [2]_; it returns a CPDAG. When unmeasured common causes are
    plausible, use :func:`~andrey.fci`; when the noise is non-Gaussian and every edge should be
    oriented, use :func:`~andrey.direct_lingam`; when some rows come from interventions, use
    :func:`~andrey.gies`. With ``env``, it takes rows pooled from several environments, such as
    sites, batches, or experiments, whose shifts change variables it does not know;
    :func:`~andrey.gies` takes the changed variables when they are known, :func:`~andrey.cdnod` runs
    PC with the same index, and :func:`~andrey.gfci` takes ``env`` when unmeasured common causes are
    plausible.

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

    **Several environments.** With ``env``, the environment index joins the data as a last variable
    that no Insert gives a parent: the background knowledge Joint Causal Inference gives a context
    variable [9]_, and the index CD-NOD adds to PC [10]_. Pooling makes the environment a common
    cause of the variables it shifts; with the index among the variables, the score explains two
    variables the environments shift together without an edge between them. After every move, the
    edges at the index are directed out of it and Meek's rules direct the edges that follow, so the
    search moves over partially directed graphs, and the result can direct edges that the CPDAG of
    the pooled rows leaves undirected. Every move is then scored in this process. The index is
    dropped from the graph. ``"bic"`` reads the index as a number; ``"degen_gauss"`` and
    ``"cond_gauss"``, with ``categorical=[]`` for continuous data, and the categorical scores read
    each label as a category. It assumes no hidden common cause; :func:`~andrey.gfci` allows them.

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
    with 2,000 rows, its median errors at 40 variables were 8 with ``"degen_gauss"`` and 6.5 with
    ``"cond_gauss"``, against 40, and the median fit took 0.75 s and 1.2 s on 4 cores. On simulated
    categorical networks with 2 or 3 categories per variable, of the same sizes and rows, ``"bdeu"``
    gave 6 against 40 at 40 variables, in 0.60 s. The multinomial BIC is not benchmarked with GES
    yet: its tests check it against the formula and run GES on small fixed categorical networks.
    With ``env``, GES is experimental and not benchmarked yet; its tests run it on up to 12
    variables with 2,000 rows from two environments, and on Sachs's 11 proteins, 7,466 cells from
    nine conditions, with ``"cond_gauss"``. ``"bic"`` reads a numeric ``env`` as a number, so with
    three or more environments a shift that does not follow the labels' values, such as one in the
    middle environment of three coded ``0, 1, 2``, can go unnoticed, and recoding the labels can
    change the result. ``max_parents`` counts the index as a parent of each variable it points into.
    The conditions under which an Insert or a Delete is valid are proved for CPDAGs [1]_; with
    ``env`` the search applies them to the partially directed graphs the index gives. The numbers
    are a reference, not a hard limit: they change with the data generator, the density, the
    method's settings, and the machine.

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
    .. [9] Mooij, J. M., Magliacane, S., and Claassen, T. (2020). Joint causal inference from
       multiple contexts. JMLR 21(99), 1-108.
    .. [10] Huang, B., Zhang, K., Zhang, J., Ramsey, J., Sanchez-Romero, R., Glymour, C., and
       Scholkopf, B. (2020). Causal discovery from heterogeneous/nonstationary data. JMLR 21(89),
       1-53.

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

    return cpdag_search(
        "ges",
        _ges,
        data,
        env,
        score_func=score_func,
        categorical=categorical,
        knowledge=knowledge,
        penalty_discount=penalty_discount,
        ebic_gamma=ebic_gamma,
        equivalent_sample_size=equivalent_sample_size,
        max_parents=max_parents,
    )


@method_spec("gies")
def gies(
    data: npt.ArrayLike,
    env: npt.ArrayLike | str | None = None,
    *,
    targets: Mapping | npt.ArrayLike | str | None = None,
    score_func: str | None = None,
    penalty_discount: float = 1.0,
    ebic_gamma: float = 1.0,
    equivalent_sample_size: float = 1.0,
    max_parents: int | None = None,
    categorical: Sequence[str | int] | None = None,
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn an interventional essential graph with greedy interventional equivalence search (GIES).

    GIES learns from data that mixes observation with interventions [1]_: in a row where a variable
    was intervened on, its value was set from outside rather than by its causes. The rows come from
    environments, each with its own set of intervened variables, its targets; an observational
    environment has none. Each variable is scored only on the rows where it was not intervened on,
    and an edge with exactly one end intervened on in some environment gets its direction from the
    data. So the result can orient edges that a CPDAG learned from observational data leaves
    undirected.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables), or a list or mapping of them
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``. A column that ``env`` or ``targets`` names is not a
        variable: it leaves the data before the fit, and ``categorical``, ``knowledge``,
        ``targets``, and the result's positions count the other columns.
        A list of DataFrames or 2-D arrays gives each environment's rows apart, labeled by
        position ``0, 1, ...``, and a mapping labels each by its key: the rows are stacked in that
        order, ``env`` stays ``None``, and ``targets`` maps those labels to their variables.
    env : array-like of shape (n_samples,) or (n_samples, 1), str, or None, default=None
        The environment of each row: one label per row, strings or finite numbers, not both, or the
        name of a column of the DataFrame ``data`` that holds them. ``None`` takes the environments
        from ``targets``: its distinct values, or none, for observational data.
    targets : mapping, str, array-like of shape (n_samples, n_variables), or None, default=None
        The variables intervened on in each environment, named by column label or 0-based position.
        A mapping ``{label: [variables]}`` takes each label of ``env`` to a list of its targets;
        ``[]`` marks an observational environment, and every label of ``env`` needs a list:
        ``None``, or a label the mapping leaves out, raises ``ValueError``, as a key that is not a
        label of ``env`` does. When the labels of ``env`` are numbers, a text key names the label
        it spells, ``"1"`` the label ``1``, so a mapping read back from JSON passes as it is. A
        string names a column of the DataFrame ``data`` whose cells give
        each row's targets: a label or a position, a list of them, or an empty cell (``None``,
        ``NaN``, ``""``, ``[]``) for an observational row. An array is ``True`` (or 1) where the
        row's value of the variable was set by an intervention and ``False`` (or 0) where its
        causes set it. With a column or an array, every row of one environment of ``env`` must
        give the same targets. ``None``, without ``env``, means observational data. Every variable
        needs at least two rows without an intervention on it, with different values.
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
        ``metadata["score_func"]`` names the score the fit used: the one picked from the
        declared column types when this is ``None``.
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
        standard BIC. Must be from 0 to 1. Only ``"bic"`` uses it: passing it with another score
        raises ``ValueError``. See Notes.
    equivalent_sample_size : float, default=1.0
        The equivalent sample size of the BDeu score: how many rows the uniform prior over each
        variable's conditional distributions is worth. A larger value weighs the prior more
        against the data, and the graph can change with it. Must be finite and positive. Only
        ``"bdeu"`` uses it: passing it with another score raises ``ValueError``.
    max_parents : int or None, default=None
        A cap on every node's parents, at least ``0``: no DAG the returned graph represents gives a
        node more than ``max_parents`` parents (every DAG of a class has the same largest number of
        parents). A step that would give a node more is skipped for the next best step. A cap can
        stop the search short of the lowest-scoring graph. ``None`` sets no cap. See Notes.
    categorical : sequence of str or int, or None, default=None
        The categorical columns of ``data``, for a mixed-data score: column labels (strings) or
        0-based positions. ``None`` takes a DataFrame's ``category``, ``bool``, and text columns,
        never a column by its values. ``[]`` marks none. A categorical score reads every column as
        categorical, and a continuous score raises ``ValueError`` on a declared column, named here
        or by its type. See *Data types* in the
        :doc:`getting-started guide </docs/guides/getting-started>`.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the pairs
        allowed to be adjacent, and temporal tiers, all honored. No move adds a forbidden edge or
        removes a required one, and the search starts from the required edges. The result directs
        every edge the knowledge decides, then Meek's rules direct the edges that follow, as
        :class:`~andrey.Knowledge` describes.

    Returns
    -------
    StructureOutput
        Wraps a ``cpdag`` :class:`~andrey.core.GraphStructure`, the interventional essential graph:
        directed marks (``->``) are edges every graph in the class orients alike, undirected
        (``--``) the ones it leaves open. ``metadata["algorithm"]`` is ``"gies"`` and
        ``metadata["score"]`` the score of the result, the sum over variables of each one's local
        score on the rows without an intervention on it; lower is better. For BIC, a variable's term
        is ``n_v * log(residual variance) + penalty_discount * log(n_samples) * n_parents + 2 *
        ebic_gamma * log C(n_variables - 1, n_parents)``, where ``n_v`` counts those rows; for the
        other scores, see :func:`~andrey.ges`. Under a mixed-data setting,
        ``metadata["categorical"]`` lists the categorical columns it read, by name or 0-based
        position. ``metadata["tie_break"]`` is the tie-break rule the fit ran under,
        ``andrey.config.tie_break``, and ``metadata`` also records the penalties its score used,
        under their parameter names.

    Raises
    ------
    TypeError
        If an argument has the wrong type, such as a string for a number; the message names the
        method, the argument, what it accepts, and the value. Also if ``categorical`` is not a list
        of column names and positions, or ``knowledge`` is neither ``None`` nor a
        :class:`~andrey.Knowledge`.
    ValueError
        If an argument lies outside its range or is not one of its choices; the message names the
        method, the argument, what it accepts, and the value. Also if ``data`` is not a 2-D numeric
        matrix with at least two rows, no ``NaN`` or ``inf``, and no constant column; if ``data``
        holds each environment's rows apart with ``env`` or with ``targets`` other than a mapping,
        or its tables differ in kind or columns, or one has no rows; if a ``targets`` matrix does
        not have ``data``'s shape, holds a value other than 0 or 1, or leaves a variable fewer than
        two rows, or one value, without an intervention; if an environment's targets are unknown, or
        ``targets`` names a label ``env`` lacks or names one label by two keys, such as ``1`` and
        ``"1"``; with a categorical score, if ``data`` holds a number that is not whole; or, with a
        mixed-data score, if ``categorical`` names a column ``data`` lacks, or if a categorical
        column holds a number that is not whole. Also if a continuous score meets a column declared
        categorical, by ``categorical`` or by its type. With the BIC score, also if a column repeats
        another up to scale and shift given more than two rows or, with more rows than columns, is a
        linear function of other columns. One message names every group of copies and says to keep
        one column per group. Also if a given ``max_parents`` is below the parents some graph of the
        required edges of ``knowledge`` gives a node, or if ``knowledge`` names a column ``data``
        lacks or contradicts itself, as :class:`~andrey.Knowledge` lists. Also if ``env`` comes
        without ``targets``; if ``env`` does not hold one label per row, misses a label or holds
        ``inf``, or mixes strings and numbers; if a ``targets`` mapping comes without ``env``, names
        a label ``env`` lacks, or leaves an environment's targets unknown; if a column or an array
        of ``targets`` gives two rows of one environment different targets; if ``targets`` names a
        variable ``data`` lacks; or if ``env`` or ``targets`` names a column ``data`` lacks.

    Warns
    -----
    DataTypeWarning
        When a mixed-data score finds no declared categorical column and so runs as the
        linear-Gaussian form, or when a setting reads columns that look like category codes as
        numbers. Each message warns once per process.
    BoundaryWarning
        When the data are larger than the largest size whose fits finished within the time and
        memory caps of the published benchmark, at the measured setting nearest the data's rows, and
        fits of the next size ran past a cap. It warns once per process and names that size and the
        release card; the size is a reference, not a hard limit.

    Notes
    -----
    **When to use.** GIES suits independent rows drawn from the observational distribution and from
    the distributions of known interventions, with each row's environment in ``env`` and the
    variables set in each environment in ``targets``; the rows are not identically distributed
    [1]_. It needs the targets of every environment. It makes the assumptions of
    :func:`~andrey.ges`, no unmeasured common causes and, with the default BIC score on continuous
    data, linear relations with Gaussian noise, and assumes that an intervention replaces the
    intervened variable's mechanism and leaves every other variable's unchanged; categorical and
    mixed data take the scores and defaults of :func:`~andrey.ges`. It suits sparse graphs: its
    authors note that bounding the number of edges at each variable keeps the cost of a step
    polynomial in the number of variables [1]_. It returns the interventional essential graph, whose
    directed edges are those every graph of the interventional equivalence class orients alike, so
    interventions can orient edges that a CPDAG leaves undirected. On observational data alone, use
    :func:`~andrey.ges`; when the rows come from several environments and which variables changed
    is unknown, use :func:`~andrey.cdnod`, which learns from ``env`` alone; and with unmeasured
    common causes, :func:`~andrey.fci` takes observational rows only.

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
    networks, with and without interventions. Every environment's targets must be known: GIES
    reads no environment whose targets are unknown. The numbers are a reference, not a hard limit:
    they change with the data generator, the density, the method's settings, and the machine.

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
    >>> env = np.repeat(["observed", "set y"], [400, 200])
    >>> data = pd.DataFrame({"x": x, "y": y})
    >>> gies(data, env, targets={"observed": [], "set y": ["y"]}).structure.oriented_edges()
    [('x', 'y', 'directed')]

    The same data with a column that names each row's targets, from which ``env`` follows:

    >>> data["target"] = np.where(env == "set y", "y", "")
    >>> gies(data, targets="target").structure.oriented_edges()
    [('x', 'y', 'directed')]

    On categorical data, declared by a pandas ``category`` type, the default score is BDeu:

    >>> a = rng.integers(0, 3, 2000)
    >>> b = np.where(rng.random(2000) < 0.8, a, rng.integers(0, 3, 2000))
    >>> b[1000:] = rng.integers(0, 3, 1000)  # the last 1000 rows set b from outside
    >>> env = np.repeat([0, 1], 1000)
    >>> data = pd.DataFrame({"a": pd.Categorical(a), "b": pd.Categorical(b)})
    >>> gies(data, env, targets={0: [], 1: ["b"]}).structure.oriented_edges()
    [('a', 'b', 'directed')]
    """
    from andrey.core.environments import normalize_env_targets
    from andrey.search.gies import gies as _gies
    from andrey.search.gies import intervention_mask

    if env is not None and np.ndim(env) == 2 and np.shape(env)[1] > 1:
        raise ValueError(
            "gies: the second argument is env, one label per row; a 0/1 matrix of the intervened "
            "variables goes in targets since 0.14.0"
        )
    data, codes, _, by_code = normalize_env_targets(
        data, env, targets, require_known_targets=True, method="gies"
    )
    assert score_func is not None  # method_spec picks it from the column types
    X, mask, labels, constraints = fit_inputs(
        data, "score_func", score_func, categorical, knowledge, method="gies"
    )
    intervened = None
    if targets is not None:  # mask[i] = targets_by_code[env_codes[i]]
        rows = np.zeros((len(by_code), X.shape[1]), dtype=bool)
        for code, variables in by_code.items():
            rows[code, sorted(variables or ())] = True
        intervened = intervention_mask(rows[codes], X)
    with copies_named(labels):
        graph, score = _gies(
            X,
            intervened,
            score_func=score_func,
            penalty_discount=penalty_discount,
            ebic_gamma=ebic_gamma,
            equivalent_sample_size=equivalent_sample_size,
            max_parents=max_parents,
            categorical=mask,
            constraints=constraints,
        )
    metadata = {"score": json_safe(score), **categorical_metadata(mask, data)}
    return structure_output(graph, labels=labels, metadata=metadata)


@method_spec("gnies")
def gnies(
    data: npt.ArrayLike,
    env: npt.ArrayLike | str | None = None,
    *,
    targets: Mapping | npt.ArrayLike | str | None = None,
    penalty_discount: float = 1.0,
    ebic_gamma: float = 1.0,
    max_parents: int | None = None,
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn an interventional essential graph and its unknown targets with greedy search (GnIES).

    GnIES learns from data pooled from several environments that share one linear model with
    Gaussian noise and differ in the noise variances of some variables, the targets, which it need
    not be told [1]_. It learns the targets together with the graph's interventional equivalence
    class, so every edge at a target gets its direction from the data.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables), or a list or mapping of them
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``. A column that ``env`` or ``targets`` names is not a
        variable: it leaves the data before the fit, and ``knowledge``, ``targets``, and the
        result's positions count the other columns.
        A list of DataFrames or 2-D arrays gives each environment's rows apart, labeled by
        position ``0, 1, ...``, and a mapping labels each by its key: the rows are stacked in that
        order, ``env`` stays ``None``, and ``targets``, when given, maps
        those labels to their known variables.
    env : array-like of shape (n_samples,) or (n_samples, 1), str, or None, default=None
        The environment of each row: one label per row, strings or finite numbers, not both, or the
        name of a column of the DataFrame ``data`` that holds them. GnIES needs two or more
        environments, each with at least two rows, and at least as many rows as variables once each
        environment's mean is taken out: ``n_samples - n_environments >= n_variables``. One of them
        may be observational, but none has to be. ``None`` takes the environments from ``targets``,
        as a column or a 0/1 matrix.
    targets : mapping, str, array-like of shape (n_samples, n_variables), or None, default=None
        The variables known to be intervened on, by column label or 0-based position. ``None``,
        the setting GnIES is built for, leaves every environment's targets unknown, and the search
        learns them. A mapping ``{label: [variables]}`` lists the known targets of each label of
        ``env``; ``None``, or a label the mapping leaves out, marks an environment whose targets
        are unknown, and ``[]`` adds none. When the labels of ``env`` are numbers, a text key
        names the label it spells, ``"1"`` the label ``1``. GnIES fits one set of targets for all
        environments: the
        variables the mapping lists stay targets, and while some environment's targets are
        unknown the search may add more. When every environment's targets are known (a mapping
        that gives each a list, a string naming a column of the DataFrame ``data`` whose cells
        give each row's targets, or a 0/1 array that is ``True`` where the row's value of the
        variable was intervened on), the targets are fixed and only the graph is searched.
    penalty_discount : float, default=1.0
        Multiplier on the BIC penalty of ``log(n_samples)`` per parameter: per parent, and per extra
        noise variance of a target, ``n_environments - 1`` each. ``n_samples`` counts every row.
        Larger values give sparser graphs and fewer targets; ``1.0`` with ``ebic_gamma=0.0`` is
        the standard BIC. Must be finite and non-negative.
    ebic_gamma : float, default=1.0
        Weight on the extended-BIC term: each variable's score adds ``2 * ebic_gamma * log
        C(n_variables - 1, n_parents)``, which grows with the number of parent sets a variable can
        choose from. Larger values give sparser graphs; ``0.0`` is the standard BIC. The term
        charges parents only, not targets. Must be from 0 to 1. See Notes.
    max_parents : int or None, default=None
        A cap on every node's parents, at least ``0``: no DAG the returned graph represents gives a
        node more than ``max_parents`` parents. Each search over graphs skips a step that would
        give a node more for the next best step, so a cap can stop it short of the lowest-scoring
        graph. ``None`` sets no cap.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the pairs
        allowed to be adjacent, and temporal tiers, all honored in each search over graphs. No
        move adds a forbidden edge or removes a required one, and each search starts from the
        required edges. The result directs every edge the knowledge decides, then Meek's rules
        direct the edges that follow, as :class:`~andrey.Knowledge` describes.

    Returns
    -------
    StructureOutput
        Wraps a ``cpdag`` :class:`~andrey.core.GraphStructure`, the interventional essential graph
        of the learned targets: directed marks (``->``) are edges every graph in the class orients
        alike, among them every edge at a target, and undirected (``--``) the ones it leaves open.
        ``metadata["algorithm"]`` is ``"gnies"``; ``metadata["score"]`` the score of the result,
        lower is better (see Notes); ``metadata["changing_variables"]`` lists the targets, known and
        learned, by label or 0-based position, in column order: the variables whose noise variance
        differs between some two environments. A shift in a variable's mean is not among them, since
        each environment's mean is taken out, and neither is a change made alike in every
        environment. With ``env``, ``metadata["environments"]`` holds its labels in sorted order, so
        ``targets=dict.fromkeys(out.metadata["environments"], out.metadata["changing_variables"])``
        gives every environment the learned targets as known: a refit on it searches the graph alone
        and returns the same graph and score. Without ``env``, ``targets`` gives every environment's
        targets, which are then known. ``metadata["tie_break"]`` is the tie-break rule the fit ran
        under, ``andrey.config.tie_break``, and ``metadata`` also records the penalties its score
        used, under their parameter names.

    Raises
    ------
    TypeError
        If an argument has the wrong type, such as a string for a number; the message names the
        method, the argument, what it accepts, and the value. Also if ``knowledge`` is neither
        ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If an argument lies outside its range or is not one of its choices; the message names the
        method, the argument, what it accepts, and the value. Also if ``data`` is not a 2-D numeric
        matrix with at least two rows, no ``NaN`` or ``inf``, and no constant column; if a column is
        declared categorical by a DataFrame's ``category``, ``bool``, or text type; if the rows come
        from fewer than two environments, an environment has fewer than two rows,
        ``n_samples - n_environments`` is below the variable count, or a column is constant within
        one environment; if a column repeats another up to scale and shift or is a linear function
        of other columns, within the environments (one message names every group of copies and says
        to keep one column per group). Also if ``data`` holds each environment's rows apart with
        ``env`` or with ``targets`` other than a mapping, or its tables differ in kind or columns,
        or one has no rows; if ``env`` does not hold one label per row, misses a label or holds
        ``inf``, or mixes strings and numbers; if a ``targets`` mapping comes without ``env``, names
        a label ``env`` lacks, or names one label by two keys; if a column or an array of
        ``targets`` gives two rows of one environment different targets, or an array does not have
        ``data``'s shape or holds a value other than 0 or 1; if ``targets`` names a variable
        ``data`` lacks; or if ``env`` or ``targets`` names a column ``data`` lacks. Also if a given
        ``max_parents`` is below the parents some graph of the required edges of ``knowledge`` gives
        a node; if the required edges give a known target more parents than the smallest
        environment's rows minus 2; or if ``knowledge`` names a column ``data`` lacks or contradicts
        itself, as :class:`~andrey.Knowledge` lists.

    Warns
    -----
    DataTypeWarning
        When a column holds at most 10 distinct whole numbers and so may hold category codes,
        which the fit reads as numbers. Each message warns once per process.

    Notes
    -----
    **When to use.** GnIES suits independent rows from two or more environments, with or without an
    observational one, of one linear structural equation model with Gaussian noise, in which every
    environment keeps the same weights and differs only in the noise variances of some variables,
    its targets, which it learns without being told them [1]_. It assumes no unmeasured common
    causes and a faithful distribution. It returns the learned targets and the interventional
    essential graph, which directs every edge at a target, so the environments can orient edges that
    a CPDAG leaves undirected. Its authors simulate graphs of 10 variables and present it as most
    useful with many environments of few rows each; measured with 100 or more rows per environment,
    it learned graphs of 10 to 100 variables with errors far below the empty graph's, a fit taking
    about a second at 10 variables and three minutes at 100 (see Limitations). When an intervention
    replaces a variable's mechanism and the targets are known, use :func:`~andrey.gies`; when how
    the environments differ is not linear and Gaussian, :func:`~andrey.cdnod`; and on observational
    data alone, :func:`~andrey.ges`.

    **How it works.** Each environment's columns are centered by their own means. For a set of
    targets, a variable outside it is scored by the Gaussian likelihood of its regression on its
    parents with one noise variance over every environment, and a target with one set of
    coefficients and a noise variance per environment, fitted by alternating weighted least
    squares from the regression over every environment. The search keeps one set of targets for
    all environments [1]_. The equivalence class of a set of targets holds the graphs with the
    same adjacencies and v-structures that orient every edge at a target alike [4]_, and all of
    them get the same score [1]_. For each set it visits, a greedy search over these classes [2]_,
    which extends greedy equivalence search [3]_, adds, removes, and turns edges from the empty
    graph while the score improves. The outer search starts from the known targets, adds the
    variable whose set scores best while that improves the score, then removes added variables
    the same way. Steps whose score changes are equal after rounding are chosen by each column's
    rank, as in :func:`~andrey.hc`; the :doc:`column-order guide </docs/guides/column-order>`
    says what the rank keeps. With every environment's targets known, one search over graphs
    runs, with the union of the targets.

    **The score.** ``metadata["score"]`` is the sum of each variable's term. Outside the targets
    it is ``n_samples * log(residual variance) + penalty_discount * log(n_samples) * n_parents``;
    at a target, ``sum_e n_e * log(residual variance in e) + penalty_discount * log(n_samples) *
    (n_parents + n_environments - 1)``, with ``n_e`` the rows of environment ``e``. Each term also
    adds ``2 * ebic_gamma * log C(n_variables - 1, n_parents)``. Variances are divided by the row
    count.

    **Choosing the penalty.** The paper's score is the standard BIC, ``penalty_discount=1.0`` with
    ``ebic_gamma=0.0`` [1]_. The default adds the extended-BIC term [5]_, as :func:`~andrey.ges`
    and :func:`~andrey.gies` do, which describe the choice; with 10 variables it adds ``2 * log 9
    = 4.4`` to the first parent's penalty, against ``log(n_samples)``, 8.3 with 4,000 rows. On the
    simulated data of the Limitations, paired per setting, the default made fewer errors than the
    standard BIC on 20 datasets in 7 of the 21 settings of 10 variables on the SHD and in 1 on the
    target error, and more in none; on 20 more datasets of those 7 settings, the SHD difference
    held in 3 and the target-error difference in none. With 50 and 100 variables its median SHD was
    1 and 2.5, against 16.5 and 54.5 at the standard BIC.

    **Data scaling.** Rescaling or shifting a column alike in every row does not change the
    result, nor does shifting one environment's rows. Do not standardize each environment on its
    own: that rescales the noise variances the method reads, and changes the result.

    **Missing values.** A ``NaN`` in ``data`` raises ``ValueError``. The score adds one term per
    variable, each over every row, and terms over different rows do not add up to the score of one
    dataset, so the search cannot delete rows per test as the conditional-independence searches do.
    Drop the incomplete rows first, which suits values missing completely at random, or use
    :func:`~andrey.pc` or :func:`~andrey.fci`, whose tests delete rows per test.

    **Limitations.** Measured with the targets unknown on simulated data from the model above:
    Erdos-Renyi graphs of 10 variables at mean degree 2.7 or 4, two to ten environments with 10 to
    1,000 rows each, with or without an observational one, one or three targets per environment, 20
    datasets per setting, one core per fit; errors count the SHD against the interventional
    essential graph and the targets learned wrongly or missed. With 100 or more rows per environment
    its errors were below the empty graph's on both counts in every setting. With 10 rows per
    environment its target errors were 1.85 to 3 with 3 true targets and 4.35 and 7.7 with 9, and
    its SHD was not reliably below the empty graph's at mean degree 4 (20 against 20) or with the
    targets' noise variances set to 3 to 4 instead of 5 to 10 (11.9 against 14; the 95% interval of
    the difference reached above 0 on both sets of 20 datasets). Outside its model, with 1,000 rows
    per environment: after hard interventions, which replace a target's mechanism, its mean SHD was
    3.75 against the empty graph's 14; a shifted mean alone is invisible to it, so it learned almost
    no targets (3.1 target errors with 3 true targets) and returned close to the observational class
    (SHD 2.55 against it); after a target's incoming weights were halved it beat the empty graph on
    neither count (SHD 11.6 against 14, 3.05 target errors with 3 true targets); with Laplace noise
    its SHD was 1.65, and with additive nonlinear relations 7.55. With 1,000 rows in each of four
    environments, three of them with one target, its median SHD at 20, 30, 50, and 100 variables was
    0, 0, 1, and 2.5, against the empty graph's 27, 40, 68, and 135, and its median fit took 3.5,
    9.0, 30, and 176 s. At 10 variables its median fit took 0.3 to 1.3 s per setting. Each set of
    targets it visits costs one search over graphs from the empty graph: each forward step scores
    every variable not yet a target, and each backward step every target, about
    ``(n_targets + 1) * n_variables - n_targets * (n_targets - 1) / 2`` searches per fit, with
    ``n_targets`` the targets it returns. Given every environment's targets, one search over graphs
    runs, and it can stop at a graph that scores worse than one the target search reaches: with the
    true targets given, its mean SHD was higher than when it learned them in 17 of the 21 settings.
    A target's coefficients are shared across environments and its variances are not, so a target
    with ``k`` parents needs at least ``k + 2`` rows in every environment: no parent set larger than
    the smallest environment's row count minus 2 is selected for a target, and a variable that the
    required edges of ``knowledge`` give more parents than that is not learned as one. The numbers
    are a reference, not a hard limit: they change with the data generator, the density, the
    method's settings, and the machine.

    References
    ----------
    .. [1] Gamella, J. L., Taeb, A., Heinze-Deml, C., and Buhlmann, P. (2022). Characterization
       and greedy learning of Gaussian structural causal models under unknown interventions.
       arXiv:2211.14897.
    .. [2] Hauser, A., and Buhlmann, P. (2012). Characterization and greedy learning of
       interventional Markov equivalence classes of directed acyclic graphs. JMLR 13, 2409-2464.
    .. [3] Chickering, D. M. (2002). Optimal structure identification with greedy search. JMLR 3,
       507-554.
    .. [4] Tian, J., and Pearl, J. (2001). Causal discovery from changes. Proceedings of the
       Seventeenth Conference on Uncertainty in Artificial Intelligence, 512-521.
    .. [5] Chen, J., and Chen, Z. (2008). Extended Bayesian information criteria for model
       selection with large model spaces. Biometrika 95(3), 759-771.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from andrey import gnies
    >>> rng = np.random.default_rng(0)
    >>> env = np.repeat(["observed", "noisy y"], 500)
    >>> x = rng.standard_normal(1000)
    >>> y = 0.8 * x + rng.standard_normal(1000) * np.where(env == "noisy y", 3.0, 1.0)
    >>> out = gnies(pd.DataFrame({"x": x, "y": y}), env)
    >>> out.structure.oriented_edges()
    [('x', 'y', 'directed')]
    >>> out.metadata["changing_variables"]
    ['y']
    """
    from andrey.core.environments import (
        check_constant_columns,
        check_environment_rows,
        normalize_env_targets,
    )
    from andrey.search.gnies import gnies as _gnies

    data, codes, envs, by_code = normalize_env_targets(
        data, env, targets, require_known_targets=False, method="gnies"
    )
    declared = declared_columns(data, None)
    if declared is not None:
        columns = ", ".join(str(c) for c, hit in zip(data.columns, declared, strict=True) if hit)
        raise ValueError(
            f"gnies reads continuous data, but data columns {columns} are declared categorical by "
            "a category, bool, or text type; pass them as numbers or leave them out"
        )
    X = data_matrix(data, method="gnies")
    labels = column_labels(data)
    name = column_namer(labels)
    variable = label_of(labels)
    # An environment taken from targets is named by its targets.
    shown = [[variable(j) for j in sorted(e)] if isinstance(e, frozenset) else e for e in envs]
    sizes = np.bincount(codes, minlength=len(envs))
    if len(envs) < 2:
        raise ValueError(
            "gnies needs rows from two or more environments; pass env, each row's environment"
        )
    check_environment_rows(codes, shown, "gnies", 2)
    n, d = X.shape
    if n - len(envs) < d:
        raise ValueError(
            f"gnies: {n} rows in {len(envs)} environments leave {n - len(envs)} once each "
            f"environment's mean is taken out, fewer than the {d} variables; the score needs at "
            "least as many"
        )
    for k in range(len(envs)):
        check_constant_columns(
            X[codes == k],
            name,
            "gnies",
            f"the rows of environment {shown[k]!r}, where the model needs a positive noise "
            "variance",
        )
    constraints = resolve(knowledge, d, labels, method="gnies")
    known = frozenset().union(*(s for s in by_code.values() if s is not None))
    fixed = all(s is not None for s in by_code.values())
    if constraints is not None:
        required = constraints.required.sum(axis=0)
        most = int(sizes.min()) - 2
        over = sorted(j for j in known if required[j] > most)
        if over:
            raise ValueError(
                f"gnies: the required edges give {int(required[over[0]])} parents to the known "
                f"target {name(over[0])}; a target can have at most {most}, the smallest "
                f"environment's {most + 2} rows minus 2"
            )
    with copies_named(labels):
        graph, score, learned = _gnies(
            X,
            codes,
            known=known,
            fixed=fixed,
            penalty_discount=penalty_discount,
            ebic_gamma=ebic_gamma,
            max_parents=max_parents,
            constraints=constraints,
            column_name=name,
        )
    metadata = {
        "score": json_safe(score),
        "changing_variables": [variable(j) for j in learned],
        **({} if env is None else {"environments": envs}),
    }
    return structure_output(graph, labels=labels, metadata=metadata)


@method_spec("xges")
def xges(
    data: npt.ArrayLike,
    *,
    score_func: str | None = None,
    penalty_discount: float = 1.0,
    ebic_gamma: float = 1.0,
    equivalent_sample_size: float = 1.0,
    max_parents: int | None = None,
    extended_search: bool = True,
    categorical: Sequence[str | int] | None = None,
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
        ``metadata["score_func"]`` names the score the fit used: the one picked from the
        declared column types when this is ``None``.
    penalty_discount : float, default=1.0
        Multiplier on the BIC penalty of ``log(n_samples)`` per parameter: per parent for the
        linear-Gaussian BIC. Larger values give sparser graphs; ``1.0`` with ``ebic_gamma=0.0`` is
        the standard BIC. Must be finite and non-negative. BDeu ignores it; for the multinomial BIC
        it weighs ``log(n)`` per free parameter of a variable's conditional probability table.
    ebic_gamma : float, default=1.0
        Weight on the extended-BIC term of the linear-Gaussian BIC: each variable's score adds
        ``2 * ebic_gamma * log C(n_variables - 1, n_parents)``, which grows with the number of
        parent sets a variable can choose from. Larger values give sparser graphs; ``0.0`` is the
        standard BIC. Must be from 0 to 1. Only ``"bic"`` uses it: passing it with another score
        raises ``ValueError``. See Notes.
    equivalent_sample_size : float, default=1.0
        The equivalent sample size of the BDeu score: how many rows the uniform prior over each
        variable's conditional distributions is worth. A larger value weighs the prior more
        against the data, and the graph can change with it. Must be finite and positive. Only
        ``"bdeu"`` uses it: passing it with another score raises ``ValueError``.
    max_parents : int or None, default=None
        A cap on every node's parents, at least ``0``: no DAG the returned CPDAG represents gives a
        node more than ``max_parents`` parents (every DAG of a class has the same largest number of
        parents). A move that would give a node more is never taken, in the base or the extended
        search. A cap can stop the search short of the lowest-scoring graph. ``None`` sets no cap.
        See Notes.
    extended_search : bool, default=True
        Run the extended search after the base search (see Notes). ``False`` returns the base
        search's graph, which is faster.
    categorical : sequence of str or int, or None, default=None
        The categorical columns of ``data``, for a mixed-data score: column labels (strings) or
        0-based positions. ``None`` takes a DataFrame's ``category``, ``bool``, and text columns,
        never a column by its values. ``[]`` marks none. A categorical score reads every column as
        categorical, and a continuous score raises ``ValueError`` on a declared column, named here
        or by its type. See *Data types* in the
        :doc:`getting-started guide </docs/guides/getting-started>`.
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
        :func:`andrey.ges`; lower is better. Directed marks (``->``) are oriented edges, undirected
        (``--``) the ones the equivalence class leaves open. Under a mixed-data setting,
        ``metadata["categorical"]`` lists the categorical columns it read, by name or 0-based
        position. ``metadata["tie_break"]`` is the tie-break rule the fit ran under,
        ``andrey.config.tie_break``, and ``metadata`` also records the penalties its score used,
        under their parameter names.

    Raises
    ------
    TypeError
        If an argument has the wrong type, such as a string for a number; the message names the
        method, the argument, what it accepts, and the value. Also if ``categorical`` is not a list
        of column names and positions, or ``knowledge`` is neither ``None`` nor a
        :class:`~andrey.Knowledge`.
    ValueError
        If an argument lies outside its range or is not one of its choices; the message names the
        method, the argument, what it accepts, and the value. Also if ``data`` is not a 2-D numeric
        matrix with at least two rows and no ``NaN`` or ``inf``, or if a column is constant; with a
        categorical score, if ``data`` holds a number that is not whole; or, with a mixed-data
        score, if ``categorical`` names a column ``data`` lacks, or if a categorical column holds a
        number that is not whole. Also if a continuous score meets a column declared categorical, by
        ``categorical`` or by its type. With the BIC score, also if a column repeats another column
        up to scale and shift given more than two rows or, with more rows than columns, is a linear
        function of other columns. One message names every group of copies and says to keep one
        column per group. Also if a given ``max_parents`` is below the parents some graph of the
        required edges of ``knowledge`` gives a node, or if ``knowledge`` names a column ``data``
        lacks or contradicts itself, as :class:`~andrey.Knowledge` lists.

    Warns
    -----
    DataTypeWarning
        When a mixed-data score finds no declared categorical column and so runs as the
        linear-Gaussian form, or when a setting reads columns that look like category codes as
        numbers. Each message warns once per process.

    Notes
    -----
    **When to use.** XGES makes the assumptions of :func:`~andrey.ges`: i.i.d. observational data,
    no unmeasured common causes and, with the default BIC score on continuous data, linear
    relations with Gaussian noise. Categorical and mixed data take the scores and defaults of
    :func:`~andrey.ges`; its moves need only a score that adds one term per variable and gives
    equivalent graphs the same score [1]_, which each of these does. It suits the same sparse
    graphs as GES and on linear-Gaussian data often finds a graph with a better score; it returns a
    CPDAG. When unmeasured common causes are plausible, use :func:`~andrey.fci`; when the noise is
    non-Gaussian and every edge should be oriented, use :func:`~andrey.direct_lingam`; when some
    rows come from interventions, use :func:`~andrey.gies`.

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

    assert score_func is not None  # method_spec picks it from the column types
    X, mask, labels, constraints = fit_inputs(
        data, "score_func", score_func, categorical, knowledge, method="xges"
    )
    with copies_named(labels):
        cpdag, score = _xges(
            X,
            score_func=score_func,
            penalty_discount=penalty_discount,
            ebic_gamma=ebic_gamma,
            equivalent_sample_size=equivalent_sample_size,
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
    env: npt.ArrayLike | str | None = None,
    *,
    score_func: str | None = None,
    penalty_discount: float = 1.0,
    ebic_gamma: float = 1.0,
    equivalent_sample_size: float = 1.0,
    max_iter: int | None = None,
    seed: int | None = None,
    n_starts: int = 1,
    n_perturbations: int | None = None,
    max_parents: int | None = None,
    categorical: Sequence[str | int] | None = None,
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn a CPDAG from observational data with hill-climbing search.

    The search repeatedly adds, removes, or reverses the edge that most improves the score, a
    covariance-based delta-BIC by default, then converts the DAG to its Markov equivalence class
    [1]_ [2]_.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables), or a list or mapping of them
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names become
        the structure's ``labels``. A column that ``env`` names is not a variable: it leaves the
        data before the fit, and ``categorical``, ``knowledge``, and the result's positions count
        the other columns. A list of DataFrames or 2-D arrays gives each environment's rows apart,
        labeled by position ``0, 1, ...``, and a mapping labels each by its key: the rows are
        stacked in that order, and ``env`` stays ``None``. A position is the environment's number,
        so with ``"bic"`` and three or more environments, the order of the list can change the
        result (see Notes).
    env : array-like of shape (n_samples,) or (n_samples, 1), str, or None, default=None
        The environment of each row: one label per row, strings or finite numbers, not both, or the
        name of a column of the DataFrame ``data`` that holds them. The environment index then joins
        the search as a variable that no data variable causes; see Notes. Numbers are coerced to
        ``float64`` and read as numbers, whole numbers under a categorical or mixed-data setting;
        string labels are coded ``0, 1, ...`` in sorted order, whatever the row order. When every
        row has the same value, the data come from one environment, and the graph equals the one
        without ``env``. ``None``, the default, fits ``data`` as one environment.
    score_func : str or None, default=None
        Local score objective: ``"bic"``, a covariance-based Gaussian BIC for continuous data;
        ``"bdeu"``, the BDeu score, or ``"discrete_bic"``, the multinomial BIC, for categorical data
        coded by whole numbers; or ``"degen_gauss"`` and ``"cond_gauss"``, the degenerate and
        conditional Gaussian BICs for mixed data, as for :func:`~andrey.ges`. ``None`` picks from
        the declared column types: ``"bdeu"`` when every column is categorical, ``"cond_gauss"``
        when some are, else ``"bic"``. A column is declared categorical by ``categorical`` or by a
        DataFrame's ``category``, ``bool``, or text type, never by its values.
        ``metadata["score_func"]`` names the score the fit used: the one picked from the
        declared column types when this is ``None``.
    penalty_discount : float, default=1.0
        Multiplier on the BIC penalty of ``log(n_samples)`` per parameter: per parent for the
        Gaussian BIC. Larger values give sparser graphs; ``1.0`` with ``ebic_gamma=0.0`` is the
        standard BIC. Must be finite and non-negative. BDeu ignores it.
    ebic_gamma : float, default=1.0
        Weight on the extended-BIC term of the linear-Gaussian BIC: each variable's score adds
        ``2 * ebic_gamma * log C(n_variables - 1, n_parents)``, which grows with the number of
        parent sets a variable can choose from. Larger values give sparser graphs; ``0.0`` is the
        standard BIC. Must be from 0 to 1. Only ``"bic"`` uses it: passing it with another score
        raises ``ValueError``. See Notes.
    equivalent_sample_size : float, default=1.0
        The equivalent sample size of the BDeu score: how many rows the uniform prior over each
        variable's conditional distributions is worth. A larger value weighs the prior more
        against the data, and the graph can change with it. Must be finite and positive. Only
        ``"bdeu"`` uses it: passing it with another score raises ``ValueError``.
    max_iter : int or None, default=None
        Most moves of each climb, at least ``1``; ``None`` sets no limit. The search still
        terminates, since each move lowers the score. A given ``max_iter`` also bounds the edge
        count of the first climb, as each move adds at most one edge.
    seed : int or None, default=None
        ``None``, the default, is the configured seed, ``ANDREY_SEED`` or ``0`` when it is unset,
        not a fresh random one, so a fit repeats. Seed for the random edge changes that begin each
        climb after the first, which only ``n_starts`` above ``1`` makes.
    n_starts : int, default=1
        Number of climbs, at least ``1``. The first starts from the empty graph, or the required
        edges of ``knowledge``; each later one from the best DAG so far after ``n_perturbations``
        random edge changes drawn from ``seed`` and the climb's index. The result is the
        best-scoring search; a later one replaces an earlier one only when it scores better by more
        than ``1e-6``. Each climb builds its own caches and drops the previous one's, so more starts
        take more time but little more memory. See Notes.
    n_perturbations : int or None, default=None
        Number of random edge additions, removals, or reversals made to the best DAG so far before
        each climb after the first, at least ``1``; ``None`` uses the number of variables. It
        changes nothing when ``n_starts`` is ``1``. See Notes.
    max_parents : int or None, default=None
        A cap on every node's parents, at least ``0``: the search takes no move that gives a node
        more, so no DAG the returned CPDAG represents gives a node more than ``max_parents`` parents
        (every DAG of a class has the same largest number of parents). A cap can stop the search
        short of the lowest-scoring graph. ``None`` sets no cap. With ``env``, the index counts as a
        parent of each variable it points into.
    categorical : sequence of str or int, or None, default=None
        The categorical columns of ``data``, for a mixed-data score: column labels (strings) or
        0-based positions. ``None`` takes a DataFrame's ``category``, ``bool``, and text columns,
        never a column by its values. ``[]`` marks none. A categorical score reads every column as
        categorical, and a continuous score raises ``ValueError`` on a declared column, named here
        or by its type. See *Data types* in the
        :doc:`getting-started guide </docs/guides/getting-started>`.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the pairs
        allowed to be adjacent, and temporal tiers, all honored. No move adds a forbidden edge or
        removes a required one, and the search starts from the required edges. The result directs
        every edge the knowledge decides, then Meek's rules direct the edges that follow, as
        :class:`~andrey.Knowledge` describes. With ``env``, the environment index is unconstrained.

    Returns
    -------
    StructureOutput
        Wraps a ``cpdag`` :class:`~andrey.core.GraphStructure`; ``metadata["algorithm"]`` is
        ``"hc"`` and ``metadata["score"]`` the score of the DAG found, computed as for
        :func:`andrey.ges`; lower is better. Directed marks (``->``) are oriented edges, undirected
        (``--``) the ones the equivalence class leaves open. Under a mixed-data setting,
        ``metadata["categorical"]`` lists the categorical columns it read, by name or 0-based
        position. With ``n_starts`` above ``1``, ``metadata["seed"]`` is the seed used and
        ``metadata["n_perturbations"]`` the number of changes before each later climb. With ``env``,
        the graph over the data variables is partially directed: the index directs edges that the
        CPDAG of the pooled rows leaves undirected, so the graph stands for the DAGs that agree on
        every direction the index identifies, and is tagged ``cpdag``, as :func:`~andrey.cdnod`'s
        is. ``metadata["index_edges"]`` then lists the index's edges, each ``env -> variable``, as
        ``[variable, "directed"]`` pairs, and ``metadata["changing_variables"]`` the same variables,
        by name or 0-based position, in column order: the variables whose causal module, the
        distribution given their causes, changes across the environments, when the index is only a
        cause and no hidden common cause acts. Their descendants change in distribution too, through
        them, and are not listed, unless the climb stops with a chain directed against the index;
        see Notes. Both are empty for a single environment, and ``metadata["score"]`` is the score
        of the DAG found over the data and the index. ``metadata["tie_break"]`` is the
        tie-break rule the fit ran under, ``andrey.config.tie_break``, and ``metadata`` also records
        the penalties its score used, under their parameter names.

    Raises
    ------
    TypeError
        If an argument has the wrong type, such as a string for a number; the message names the
        method, the argument, what it accepts, and the value. Also if ``categorical`` is not a list
        of column names and positions, or ``knowledge`` is neither ``None`` nor a
        :class:`~andrey.Knowledge`.
    ValueError
        If an argument lies outside its range or is not one of its choices; the message names the
        method, the argument, what it accepts, and the value. Also if the required edges of
        ``knowledge`` give a node more than ``max_parents`` parents; if ``data`` or ``categorical``
        does not suit the score, as for :func:`~andrey.ges`. With the BIC score, also if a column
        repeats another up to scale and shift given more than two rows or, with more rows than
        columns, is a linear function of other columns. One message names every group of copies and
        says to keep one column per group. Also if ``knowledge`` names a column ``data`` lacks or
        contradicts itself, as :class:`~andrey.Knowledge` lists, or, under a mixed-data score,
        requires an edge from a continuous into a categorical column. Also if ``env`` does not hold
        one label per row, misses a label or is infinite, or mixes strings and numbers, or, with a
        categorical or mixed-data score, holds a number that is not whole.

    Warns
    -----
    SearchLimitWarning
        When a climb takes a given ``max_iter`` moves, once per call; a larger ``max_iter`` may
        find a better graph.
    DataTypeWarning
        When a mixed-data score finds no declared categorical column and so runs as the
        linear-Gaussian form, or when a setting reads columns that look like category codes as
        numbers. Each message warns once per process.

    Notes
    -----
    **When to use.** Hill climbing suits i.i.d. observational data with no unmeasured common causes.
    Its default BIC score assumes continuous data, where each variable depends linearly on its
    causes with Gaussian noise; categorical and mixed data take the scores and defaults of
    :func:`~andrey.ges`. Each move adds, removes, or reverses one edge of a DAG [1]_, and the search
    suits sparse graphs: on dense ones it more often stops at a DAG that no single move improves but
    that scores worse than the true one. It returns the CPDAG of the DAG found. When unmeasured
    common causes are plausible, use :func:`~andrey.fci`; when the noise is non-Gaussian and every
    edge should be oriented, use :func:`~andrey.direct_lingam`; with nonlinear effects, use
    :func:`~andrey.cam`. With ``env``, it takes rows pooled from several environments, such as
    sites, batches, or experiments, whose shifts change variables it does not know;
    :func:`~andrey.gies` takes the changed variables when they are known, and :func:`~andrey.cdnod`
    runs PC with the same index. With ``env``, set ``n_starts`` above ``1`` (see **Several
    environments.**).

    The search starts from the empty graph and stops when no single move improves the score, or
    after ``max_iter`` moves when set; with ``n_starts``, each climb stops so. Under a mixed-data
    score, no move gives a categorical column a continuous parent, so the DAG found is a
    conditional linear Gaussian network.

    Under a score-equivalent score, adding ``x -> y`` or ``y -> x`` from the empty graph gives the
    same score. Such ties recur during search, and choosing one move can change later moves and
    the final CPDAG. Each column gets a rank computed from the data to choose between tied
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

    **Choosing n_starts.** The search can stop at a DAG that no single move improves but that scores
    worse than the true one, most often on dense graphs. Each later start climbs again from the best
    DAG so far after ``n_perturbations`` random edge additions, removals, or reversals, and the fit
    keeps the best score: hill climbing with random restarts, which changes many edges at each
    restart [1]_ [4]_. No change breaks the knowledge, ``max_parents``, the rule of the mixed-data
    scores, or acyclicity: a draw that would close a cycle changes nothing, and at most
    ``3 * n_perturbations`` draws are made. The next climb most often undoes a single change, so
    ``n_perturbations`` defaults to the number of variables.

    On simulated linear-Gaussian data with 100 variables at mean degree 4 and 1,000 rows, at the
    defaults, 8 starts lowered the mean errors (SHD) over 20 datasets from 161.6 to 93.5 and took a
    median of 5.3 times as long as one start. On categorical data drawn from the ALARM network with
    1,000 rows, under BDeu, they lowered them from 29.6 to 16.9.

    **Several environments.** With ``env``, the environment index joins the data as a last variable
    that no move gives a parent: the background knowledge Joint Causal Inference gives a context
    variable [5]_, and the index CD-NOD adds to PC [6]_. Pooling makes the environment a common
    cause of the variables it shifts; with the index among the variables, the score explains two
    variables the environments shift together without an edge between them. The edges at the index
    of the DAG found are then directed out of it, and Meek's rules direct the edges that follow, so
    the graph can direct edges that the CPDAG of the pooled rows leaves undirected. The index is
    dropped from the graph. ``"bic"`` reads the index as a number; ``"degen_gauss"`` and
    ``"cond_gauss"``, with ``categorical=[]`` for continuous data, and the categorical scores read
    each label as a category. It assumes no hidden common cause; :func:`~andrey.gfci` allows them.

    Through the ties between the two directions of an edge, one climb can direct a chain against
    the index and then keep an edge from the index to each variable on it: no single move improves
    the score there, though the true graph scores better. ``index_edges`` and
    ``changing_variables`` then list every variable on the chain, not only the ones the environments
    change. Each later start climbs again from a changed copy of the best DAG so far and can leave
    such a DAG, so with ``env`` set ``n_starts`` above ``1``. :func:`~andrey.tabu`, which also takes
    moves that do not improve the score, can leave it in one run.

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
    median errors at 40 variables were 12.5 with ``"degen_gauss"`` and 11 with ``"cond_gauss"``,
    against 40. On simulated categorical networks with 2 or 3 categories per variable, of the same
    sizes and rows, they were 18 with ``"bdeu"`` and 20 with ``"discrete_bic"``. Every median fit
    took under 0.2 s at 40 variables, on 4 cores. With ``env``, hill climbing is experimental and
    not benchmarked yet; its tests run it on up to 12 variables with 2,000 rows from two
    environments, and on Sachs's 11 proteins, 7,466 cells from nine conditions, with
    ``"cond_gauss"``. ``"bic"`` reads a numeric ``env`` as a number, so with three or more
    environments a shift that does not follow the labels' values, such as one in the middle
    environment of three coded ``0, 1, 2``, can go unnoticed, and recoding the labels can change the
    result. ``max_parents`` counts the index as a parent of each variable it points into. At the
    default ``n_starts=1``, ``changing_variables`` often lists variables the environments change
    only through their causes (see **Several environments.**). The numbers are a reference, not a
    hard limit: they change with the data generator, the density, the method's settings, and the
    machine.

    References
    ----------
    .. [1] Chickering, D. M., Geiger, D., and Heckerman, D. (1995). Learning Bayesian networks:
       search methods and experimental results. AISTATS, PMLR R0, 112-128.
    .. [2] Scutari, M. (2010). Learning Bayesian networks with the bnlearn R package. Journal of
       Statistical Software 35(3). https://doi.org/10.18637/jss.v035.i03
    .. [3] Chen, J., and Chen, Z. (2008). Extended Bayesian information criteria for model
       selection with large model spaces. Biometrika 95(3), 759-771.
    .. [4] Scutari, M., Graafland, C. E., and Gutierrez, J. M. (2019). Who learns better Bayesian
       network structures: accuracy and speed of structure learning algorithms. International
       Journal of Approximate Reasoning 115, 235-253. https://doi.org/10.1016/j.ijar.2019.10.003
    .. [5] Mooij, J. M., Magliacane, S., and Claassen, T. (2020). Joint causal inference from
       multiple contexts. JMLR 21(99), 1-108.
    .. [6] Huang, B., Zhang, K., Zhang, J., Ramsey, J., Sanchez-Romero, R., Glymour, C., and
       Scholkopf, B. (2020). Causal discovery from heterogeneous/nonstationary data. JMLR 21(89),
       1-53.

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

    assert seed is not None  # method_spec picks the seed
    return cpdag_search(
        "hc",
        _hc,
        data,
        env,
        score_func=score_func,
        categorical=categorical,
        knowledge=knowledge,
        penalty_discount=penalty_discount,
        ebic_gamma=ebic_gamma,
        equivalent_sample_size=equivalent_sample_size,
        max_iter=max_iter,
        max_parents=max_parents,
        n_starts=n_starts,
        n_perturbations=n_perturbations,
        seed=seed,
    )


@method_spec("tabu")
def tabu(
    data: npt.ArrayLike,
    env: npt.ArrayLike | str | None = None,
    *,
    score_func: str | None = None,
    penalty_discount: float = 1.0,
    ebic_gamma: float = 1.0,
    equivalent_sample_size: float = 1.0,
    tabu_length: int = 10,
    max_tabu: int | None = None,
    max_iter: int | None = None,
    seed: int | None = None,
    n_starts: int = 1,
    n_perturbations: int | None = None,
    max_parents: int | None = None,
    categorical: Sequence[str | int] | None = None,
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn a CPDAG from observational data with tabu search.

    Tabu search is hill climbing that also takes non-improving moves and never returns to a recently
    visited DAG [1]_ [2]_ [3]_. Moves are scored as in :func:`andrey.hc`. The result is the CPDAG of
    the best-scoring DAG visited.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables), or a list or mapping of them
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names become
        the structure's ``labels``. A column that ``env`` names is not a variable: it leaves the
        data before the fit, and ``categorical``, ``knowledge``, and the result's positions count
        the other columns. A list of DataFrames or 2-D arrays gives each environment's rows apart,
        labeled by position ``0, 1, ...``, and a mapping labels each by its key: the rows are
        stacked in that order, and ``env`` stays ``None``. A position is the environment's number,
        so with ``"bic"`` and three or more environments, the order of the list can change the
        result (see Notes).
    env : array-like of shape (n_samples,) or (n_samples, 1), str, or None, default=None
        The environment of each row: one label per row, strings or finite numbers, not both, or the
        name of a column of the DataFrame ``data`` that holds them. The environment index then joins
        the search as a variable that no data variable causes; see Notes. Numbers are coerced to
        ``float64`` and read as numbers, whole numbers under a categorical or mixed-data setting;
        string labels are coded ``0, 1, ...`` in sorted order, whatever the row order. When every
        row has the same value, the data come from one environment, and the graph equals the one
        without ``env``. ``None``, the default, fits ``data`` as one environment.
    score_func : str or None, default=None
        Local score objective: ``"bic"``, a covariance-based Gaussian BIC for continuous data;
        ``"bdeu"`` or ``"discrete_bic"``, the BDeu score or the multinomial BIC for categorical data
        coded by whole numbers; or ``"degen_gauss"`` or ``"cond_gauss"``, the degenerate or
        conditional Gaussian BIC for mixed data; each as for :func:`~andrey.ges`. ``None`` picks
        from the declared column types: ``"bdeu"`` when every column is categorical,
        ``"cond_gauss"`` when some are, else ``"bic"``. A column is declared categorical by
        ``categorical`` or by a DataFrame's ``category``, ``bool``, or text type, never by its
        values.
        ``metadata["score_func"]`` names the score the fit used: the one picked from the
        declared column types when this is ``None``.
    penalty_discount : float, default=1.0
        Multiplier on the BIC penalty of ``log(n_samples)`` per parameter: per parent for the
        Gaussian BIC. Larger values give sparser graphs; ``1.0`` with ``ebic_gamma=0.0`` is the
        standard BIC. Must be finite and non-negative. BDeu ignores it.
    ebic_gamma : float, default=1.0
        Weight on the extended-BIC term of the linear-Gaussian BIC: each variable's score adds
        ``2 * ebic_gamma * log C(n_variables - 1, n_parents)``, which grows with the number of
        parent sets a variable can choose from. Larger values give sparser graphs; ``0.0`` is the
        standard BIC. Must be from 0 to 1. Only ``"bic"`` uses it: passing it with another score
        raises ``ValueError``. See Notes.
    equivalent_sample_size : float, default=1.0
        The equivalent sample size of the BDeu score: how many rows the uniform prior over each
        variable's conditional distributions is worth. A larger value weighs the prior more
        against the data, and the graph can change with it. Must be finite and positive. Only
        ``"bdeu"`` uses it: passing it with another score raises ``ValueError``.
    tabu_length : int, default=10
        Number of most recently visited DAGs a move may not return to. At least ``1``.
    max_tabu : int or None, default=None
        Non-improving moves allowed without a better DAG, from ``1`` to ``tabu_length``. ``None``
        uses ``tabu_length``.
    max_iter : int or None, default=None
        Most moves of each search, improving or not, at least ``1``; ``None`` sets no limit. The
        search still terminates: non-improving runs are bounded by ``max_tabu``, and each new best
        DAG scores strictly lower.
    seed : int or None, default=None
        ``None``, the default, is the configured seed, ``ANDREY_SEED`` or ``0`` when it is unset,
        not a fresh random one, so a fit repeats. Seed for the random edge changes that begin each
        search after the first, which only ``n_starts`` above ``1`` makes.
    n_starts : int, default=1
        Number of searches, at least ``1``. The first starts from the empty graph, or the required
        edges of ``knowledge``; each later one from the best DAG so far after ``n_perturbations``
        random edge changes drawn from ``seed`` and the search's index, with an empty tabu list.
        The result is the best-scoring search; a later one replaces an earlier one only when it
        scores better by more than ``1e-6``. Each search builds its own caches and drops the
        previous one's, so more starts take more time but little more memory. See Notes.
    n_perturbations : int or None, default=None
        Number of random edge additions, removals, or reversals made to the best DAG so far before
        each search after the first, at least ``1``; ``None`` uses the number of variables. It
        changes nothing when ``n_starts`` is ``1``. See Notes.
    max_parents : int or None, default=None
        A cap on every node's parents, at least ``0``: the search takes no move that gives a node
        more, so no DAG the returned CPDAG represents gives a node more than ``max_parents`` parents
        (every DAG of a class has the same largest number of parents). A cap can stop the search
        short of the lowest-scoring graph. ``None`` sets no cap. With ``env``, the index counts as a
        parent of each variable it points into.
    categorical : sequence of str or int, or None, default=None
        The categorical columns of ``data``, for a mixed-data score: column labels (strings) or
        0-based positions. ``None`` takes a DataFrame's ``category``, ``bool``, and text columns,
        never a column by its values. ``[]`` marks none. A categorical score reads every column as
        categorical, and a continuous score raises ``ValueError`` on a declared column, named here
        or by its type. See *Data types* in the
        :doc:`getting-started guide </docs/guides/getting-started>`.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the pairs
        allowed to be adjacent, and temporal tiers, all honored. No move adds a forbidden edge or
        removes a required one, and the search starts from the required edges. The result directs
        every edge the knowledge decides, then Meek's rules direct the edges that follow, as
        :class:`~andrey.Knowledge` describes. With ``env``, the environment index is unconstrained.

    Returns
    -------
    StructureOutput
        Wraps a ``cpdag`` :class:`~andrey.core.GraphStructure`; ``metadata["algorithm"]`` is
        ``"tabu"`` and ``metadata["score"]`` the best DAG's score, computed as for
        :func:`andrey.ges`; lower is better. Directed marks (``->``) are oriented edges, undirected
        (``--``) the ones the equivalence class leaves open. Under a mixed-data setting,
        ``metadata["categorical"]`` lists the categorical columns it read, by name or 0-based
        position. With ``n_starts`` above ``1``, ``metadata["seed"]`` is the seed used and
        ``metadata["n_perturbations"]`` the number of changes before each later search. With
        ``env``, the graph over the data variables is partially directed: the index directs edges
        that the CPDAG of the pooled rows leaves undirected, so the graph stands for the DAGs that
        agree on every direction the index identifies, and is tagged ``cpdag``, as
        :func:`~andrey.cdnod`'s is. ``metadata["index_edges"]`` then lists the index's edges, each
        ``env -> variable``, as ``[variable, "directed"]`` pairs, and
        ``metadata["changing_variables"]`` the same variables, by name or 0-based position, in
        column order: the variables whose causal module, the distribution given their causes,
        changes across the environments, when the index is only a cause and no hidden common cause
        acts. Their descendants change in distribution too, through them, and are not listed. Both
        are empty for a single environment, and ``metadata["score"]`` is the score of the DAG found
        over the data and the index. ``metadata["tie_break"]`` is the tie-break rule the fit ran
        under, ``andrey.config.tie_break``, and ``metadata`` also records the penalties its score
        used, under their parameter names.

    Raises
    ------
    TypeError
        If an argument has the wrong type, such as a string for a number; the message names the
        method, the argument, what it accepts, and the value. Also if ``knowledge`` is neither
        ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If an argument lies outside its range or is not one of its choices; the message names the
        method, the argument, what it accepts, and the value. Also if ``max_tabu`` is greater than
        ``tabu_length``; if the required edges of ``knowledge`` give a node more than
        ``max_parents`` parents; or, with a categorical score, if ``data`` holds a number that is
        not whole. With the BIC score, also if a column repeats another up to scale and shift given
        more than two rows or, with more rows than columns, is a linear function of other columns.
        One message names every group of copies and says to keep one column per group. Also if
        ``knowledge`` names a column ``data`` lacks or contradicts itself, as
        :class:`~andrey.Knowledge` lists. Also if ``env`` does not hold one label per row, misses a
        label or is infinite, or mixes strings and numbers, or, with a categorical or mixed-data
        score, holds a number that is not whole.

    Warns
    -----
    SearchLimitWarning
        When a search takes a given ``max_iter`` moves, once per call; a larger ``max_iter`` may
        find a better graph.
    DataTypeWarning
        When a mixed-data score finds no declared categorical column and so runs as the
        linear-Gaussian form, or when a setting reads columns that look like category codes as
        numbers. Each message warns once per process.

    Notes
    -----
    **When to use.** Tabu search suits the data of :func:`~andrey.hc`: i.i.d. observational data
    with no unmeasured common causes. Its default BIC score assumes continuous data, where each
    variable depends linearly on its causes with Gaussian noise; categorical and mixed data take the
    scores and defaults of :func:`~andrey.ges`. Its non-improving moves [1]_ let it leave a DAG that
    no single move improves, where hill climbing stops, and it suits sparse graphs: on dense ones it
    can still stop at a DAG that scores worse than the true one. It returns the CPDAG of the
    best-scoring DAG visited. When unmeasured common causes are plausible, use :func:`~andrey.fci`;
    when the noise is non-Gaussian and every edge should be oriented, use
    :func:`~andrey.direct_lingam`; with nonlinear effects, use :func:`~andrey.cam`. With ``env``, it
    takes rows pooled from several environments, such as sites, batches, or experiments, whose
    shifts change variables it does not know; :func:`~andrey.gies` takes the changed variables when
    they are known, and :func:`~andrey.cdnod` runs PC with the same index.

    The search starts from the empty graph. Each step adds the current DAG to the tabu list, which
    keeps the last ``tabu_length`` DAGs visited, then takes the best improving move outside the
    list. Without one, it stops if ``max_tabu`` non-improving moves have occurred since the best
    score last improved; otherwise it takes the best non-improving move, leaving the score unchanged
    or worse. It also stops when every legal move leads into the list, or after ``max_iter`` moves
    when set; with ``n_starts``, each search stops so. Under a mixed-data score, no move gives a
    categorical column a continuous parent, so the DAG found is a conditional linear Gaussian
    network.

    Until it leaves its first local optimum, the search takes the same moves as :func:`andrey.hc`,
    so with the same ``max_iter`` and one start its score is never worse than hill climbing's. Ties
    between moves break as they do for :func:`andrey.hc`.

    **Choosing the penalty.** The score charges a variable ``penalty_discount * log(n_samples)`` per
    parent plus the extended-BIC term ``2 * ebic_gamma * log C(n_variables - 1, n_parents)`` [4]_,
    which grows with the number of variables; :func:`~andrey.ges` describes the choice.
    ``ebic_gamma=0.0`` with ``penalty_discount=1.0`` is the standard BIC. On simulated
    linear-Gaussian data with 20 to 400 variables and 200 to 4,000 rows, the default
    ``ebic_gamma=1.0`` made fewer errors (SHD) than the standard BIC on average at every number of
    variables and rows measured: 85 against 276 with 200 variables and 2,000 rows (200 true edges).
    Its mean errors stayed below the empty graph's in each of the 65 settings, where the standard
    BIC's exceeded it in 35.

    **Choosing n_starts.** The tabu list lets the search leave a local optimum only for the moves
    ``max_tabu`` allows, so it can still stop at a DAG that scores worse than the true one, most
    often on dense graphs. Each later start searches again, with an empty tabu list, from the best
    DAG so far after ``n_perturbations`` random edge additions, removals, or reversals, and the fit
    keeps the best score: tabu search with the random restarts of hill climbing, which change many
    edges at each restart [5]_ [6]_. No change breaks the knowledge, ``max_parents``, the rule of
    the mixed-data scores, or acyclicity: a draw that would close a cycle changes nothing, and at
    most ``3 * n_perturbations`` draws are made. The next search most often undoes a single change,
    so ``n_perturbations`` defaults to the number of variables.

    On simulated linear-Gaussian data with 100 variables at mean degree 4 and 1,000 rows, at the
    defaults, 8 starts lowered the mean errors (SHD) over 20 datasets from 136.1 to 75.6 and took a
    median of 4.8 times as long as one start. On categorical data drawn from the ALARM network with
    1,000 rows, under BDeu, they lowered them from 23.8 to 13.0.

    **Several environments.** With ``env``, the environment index joins the data as a last variable
    that no move gives a parent: the background knowledge Joint Causal Inference gives a context
    variable [7]_, and the index CD-NOD adds to PC [8]_. Pooling makes the environment a common
    cause of the variables it shifts; with the index among the variables, the score explains two
    variables the environments shift together without an edge between them. The edges at the index
    of the DAG found are then directed out of it, and Meek's rules direct the edges that follow, so
    the graph can direct edges that the CPDAG of the pooled rows leaves undirected. The index is
    dropped from the graph. ``"bic"`` reads the index as a number; ``"degen_gauss"`` and
    ``"cond_gauss"``, with ``categorical=[]`` for continuous data, and the categorical scores read
    each label as a category. It assumes no hidden common cause; :func:`~andrey.gfci` allows them.

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
    run it on small fixed categorical networks. With ``env``, Tabu search is experimental and not
    benchmarked yet; its tests run it on up to 12 variables with 2,000 rows from two environments,
    and on Sachs's 11 proteins, 7,466 cells from nine conditions, with ``"cond_gauss"``. ``"bic"``
    reads a numeric ``env`` as a number, so with three or more environments a shift that does not
    follow the labels' values, such as one in the middle environment of three coded ``0, 1, 2``, can
    go unnoticed, and recoding the labels can change the result. ``max_parents`` counts the index as
    a parent of each variable it points into. The numbers are a reference, not a hard limit: they
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
    .. [5] Chickering, D. M., Geiger, D., and Heckerman, D. (1995). Learning Bayesian networks:
       search methods and experimental results. AISTATS, PMLR R0, 112-128.
    .. [6] Scutari, M., Graafland, C. E., and Gutierrez, J. M. (2019). Who learns better Bayesian
       network structures: accuracy and speed of structure learning algorithms. International
       Journal of Approximate Reasoning 115, 235-253. https://doi.org/10.1016/j.ijar.2019.10.003
    .. [7] Mooij, J. M., Magliacane, S., and Claassen, T. (2020). Joint causal inference from
       multiple contexts. JMLR 21(99), 1-108.
    .. [8] Huang, B., Zhang, K., Zhang, J., Ramsey, J., Sanchez-Romero, R., Glymour, C., and
       Scholkopf, B. (2020). Causal discovery from heterogeneous/nonstationary data. JMLR 21(89),
       1-53.

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

    assert seed is not None  # method_spec picks the seed
    return cpdag_search(
        "tabu",
        _tabu,
        data,
        env,
        score_func=score_func,
        categorical=categorical,
        knowledge=knowledge,
        penalty_discount=penalty_discount,
        ebic_gamma=ebic_gamma,
        equivalent_sample_size=equivalent_sample_size,
        tabu_length=tabu_length,
        max_tabu=tabu_cap(max_tabu, tabu_length),
        max_iter=max_iter,
        max_parents=max_parents,
        n_starts=n_starts,
        n_perturbations=n_perturbations,
        seed=seed,
    )


@method_spec("exact_search")
def exact_search(
    data: npt.ArrayLike,
    *,
    search_method: str = "astar",
    score_func: str | None = None,
    penalty_discount: float = 1.0,
    ebic_gamma: float = 1.0,
    equivalent_sample_size: float = 1.0,
    max_parents: int | None = None,
    categorical: Sequence[str | int] | None = None,
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn the CPDAG of the globally optimal DAGs for a decomposable score by exact search.

    The optimum is identified up to its Markov equivalence class, so the search returns the class's
    CPDAG, as the other score searches do; A* (``search_method="astar"``) [1]_ and dynamic
    programming (``"dp"``) [2]_ find the same DAG of that class, which ``metadata["dag"]`` keeps.
    Runtime is exponential in the variable count, so keep the problem small (roughly <= 20
    variables).

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
        data, as for :func:`~andrey.ges`. ``None`` picks from the declared column types:
        ``"bdeu"`` when every column is categorical, ``"cond_gauss"`` when some are, else
        ``"bic"``. A column is declared categorical by ``categorical`` or by a DataFrame's
        ``category``, ``bool``, or text type, never by its values.
        ``metadata["score_func"]`` names the score the fit used: the one picked from the
        declared column types when this is ``None``.
    penalty_discount : float, default=1.0
        Multiplier on the BIC penalty of ``log(n_samples)`` per parameter: per parent for the
        linear-Gaussian BIC. Larger values give sparser graphs; ``1.0`` with ``ebic_gamma=0.0`` is
        the standard BIC. Must be finite and non-negative. BDeu ignores it; for the multinomial BIC
        it weighs ``log(n)`` per free parameter of a variable's conditional probability table.
    ebic_gamma : float, default=1.0
        Weight on the extended-BIC term of the linear-Gaussian BIC: each variable's score adds
        ``2 * ebic_gamma * log C(n_variables - 1, n_parents)``, which grows with the number of
        parent sets a variable can choose from. Larger values give sparser graphs; ``0.0`` is the
        standard BIC. Must be from 0 to 1. Only ``"bic"`` uses it: passing it with another score
        raises ``ValueError``. See Notes.
    equivalent_sample_size : float, default=1.0
        The equivalent sample size of the BDeu score: how many rows the uniform prior over each
        variable's conditional distributions is worth. A larger value weighs the prior more
        against the data, and the graph can change with it. Must be finite and positive. Only
        ``"bdeu"`` uses it: passing it with another score raises ``ValueError``.
    max_parents : int or None, default=None
        A cap on every node's parents, at least ``0``: the search scores only parent sets within it
        and finds the best-scoring DAG whose every node keeps within it, so no DAG the returned
        CPDAG represents gives a node more than ``max_parents`` parents (every DAG of a class has
        the same largest number of parents). ``None`` sets no cap.
    categorical : sequence of str or int, or None, default=None
        The categorical columns of ``data``, for a mixed-data score: column labels (strings) or
        0-based positions. ``None`` takes a DataFrame's ``category``, ``bool``, and text columns,
        never a column by its values. ``[]`` marks none. A categorical score reads every column as
        categorical, and a continuous score raises ``ValueError`` on a declared column, named here
        or by its type. See *Data types* in the
        :doc:`getting-started guide </docs/guides/getting-started>`.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the pairs
        allowed to be adjacent, and temporal tiers, all honored. Each variable's parent sets exclude
        the forbidden parents and include the required ones, so the DAG found is the best one that
        satisfies the knowledge, and the returned graph directs the edges the knowledge decides.

    Returns
    -------
    StructureOutput
        Wraps a ``cpdag`` :class:`~andrey.core.GraphStructure`, the optimal equivalence class:
        directed marks (``->``) are the edges every DAG of the class shares, or the knowledge
        directs, and undirected (``--``) the ones the class leaves open. ``metadata["dag"]`` lists
        the edges of the DAG the search found as ``[i, j]`` pairs for ``i -> j``, 0-based column
        positions in ascending order; ``metadata["algorithm"]`` is ``"exact_search"``. Under a
        mixed-data setting, ``metadata["categorical"]`` lists the categorical columns it read, by
        name or 0-based position. ``metadata["tie_break"]`` is the tie-break rule the fit ran under,
        ``andrey.config.tie_break``, and ``metadata`` also records the penalties its score used,
        under their parameter names.

    Raises
    ------
    TypeError
        If an argument has the wrong type, such as a string for a number; the message names the
        method, the argument, what it accepts, and the value. Also if ``categorical`` is not a list
        of column names and positions, or ``knowledge`` is neither ``None`` nor a
        :class:`~andrey.Knowledge`.
    ValueError
        If an argument lies outside its range or is not one of its choices; the message names the
        method, the argument, what it accepts, and the value. Also if the required edges of
        ``knowledge`` give a node more than ``max_parents`` parents; if ``data`` or ``categorical``
        does not suit the score, as for :func:`~andrey.ges`. With the BIC score, also if a column
        repeats another up to scale and shift given more than two rows or, with more rows than
        columns, is a linear function of other columns. One message names every group of copies and
        says to keep one column per group. Also if ``knowledge`` names a column ``data`` lacks or
        contradicts itself, as :class:`~andrey.Knowledge` lists.

    Warns
    -----
    DataTypeWarning
        When a mixed-data score finds no declared categorical column and so runs as the
        linear-Gaussian form, or when a setting reads columns that look like category codes as
        numbers. Each message warns once per process.
    BoundaryWarning
        When the data are larger than the largest size whose fits finished within the time and
        memory caps of the published benchmark, at the measured setting nearest the data's rows, and
        fits of the next size ran past a cap. It warns once per process and names that size and the
        release card; the size is a reference, not a hard limit.

    Notes
    -----
    **When to use.** With ``"bic"``, exact search makes the assumptions of :func:`~andrey.ges`:
    i.i.d. continuous observational data, no unmeasured common causes, and linear relations with
    Gaussian noise; the categorical and mixed-data scores below make theirs. Its run time grows
    exponentially with the number of variables, so it suits a few variables, roughly 20 or fewer,
    where it finds a DAG with the best score [1]_ [2]_ rather than a greedy search's local
    optimum. It returns the CPDAG of that DAG's equivalence class and keeps the DAG in
    ``metadata["dag"]``. With more variables, use :func:`~andrey.ges` or :func:`~andrey.boss`;
    when unmeasured common causes are plausible, use :func:`~andrey.fci`.

    **Tied DAGs.** Every DAG of the optimal equivalence class has the same score, so the search's
    own pick among them would follow last-bit rounding and the column order. The returned CPDAG
    does not depend on the pick; for ``metadata["dag"]``, exact search reports one fixed member of
    the class it found: it keeps the edges every DAG of the class
    shares, and those the knowledge directs, and orients the others by a rank computed from the
    data, as :func:`~andrey.hc` ranks tied moves. The rank moves with the column, so reordering the
    columns reorders the graph's nodes and changes nothing else, unless two columns are identical.
    The rank is arbitrary with respect to causal structure. Rounding differences, such as another
    CPU's BLAS kernel makes, leave the reported DAG unchanged whenever they leave the class
    unchanged; the :doc:`column-order guide </docs/guides/column-order>` says what the rank keeps.

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
    'cpdag'
    """
    import numpy as np

    from andrey.core.knowledge import mpdag
    from andrey.core.orient import dag2cpdag, to_structure
    from andrey.core.structure import ARROW, TAIL
    from andrey.search.exact import exact_search as _exact

    assert score_func is not None  # method_spec picks it from the column types
    X, mask, labels, constraints = fit_inputs(
        data, "score_func", score_func, categorical, knowledge, method="exact_search"
    )
    with copies_named(labels):
        dag = _exact(
            X,
            search_method=search_method,
            score_func=score_func,
            penalty_discount=penalty_discount,
            ebic_gamma=ebic_gamma,
            equivalent_sample_size=equivalent_sample_size,
            categorical=mask,
            constraints=constraints,
            max_parents=max_parents,
        )
    marks = dag.to_numpy()
    cpdag = dag2cpdag(marks)
    if constraints is not None:
        cpdag = mpdag(cpdag, constraints)
    edges = np.argwhere((marks == TAIL) & (marks.T == ARROW))
    return structure_output(
        to_structure(cpdag, kind="cpdag"),
        labels=labels,
        metadata={"dag": edges, **categorical_metadata(mask, data)},
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
    knowledge: Knowledge | None = None,
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
        Scale each column to unit variance first, so the fit does not read the columns' units;
        the weights are then those of the scaled columns.
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
        ``None``, the default, is the configured seed, ``ANDREY_SEED`` or ``0`` when it is unset,
        not a fresh random one, so a fit repeats. Seed for the optimizer's torch random number
        generator. Repeated calls return the same graph on one machine with the same torch version.
        The draws go to the columns in order of each column's rank, as in :func:`~andrey.hc` (the
        :doc:`column-order guide </docs/guides/column-order>`). ``metadata["seed"]`` records the
        seed used.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the pairs
        allowed to be adjacent, and temporal tiers, all honored. The candidate edges are the pairs
        the moral-graph screen keeps and the pairs of the required edges; the screen reads the data
        alone. Forbidden edges, edges into an earlier tier, and pairs outside the allowed set have
        their gates held at zero. A required edge's gate is held at one, so the acyclicity term
        counts it, the penalty and both thresholds leave it, and the cycle repair never removes it.

    Returns
    -------
    StructureOutput
        Wraps a ``dag`` :class:`~andrey.core.GraphStructure`; ``weighted_adjacency[i, j]`` is the
        coefficient of edge ``i -> j``. ``metadata["algorithm"]`` is ``"calm"``,
        ``metadata["seed"]`` the seed used, ``metadata["objective"]`` half the sum of the log
        residual variances plus ``lambda1`` times the edge count (lower is better), and
        ``metadata["h_final"]`` the acyclicity residual ``h`` of the noise-free gates before the
        thresholds, the value the rounds compare with ``h_tol``. ``metadata["tie_break"]`` is the
        tie-break rule the fit ran under, ``andrey.config.tie_break``.

    Raises
    ------
    ImportError
        If the ``[torch]`` extra is not installed.
    TypeError
        If an argument has the wrong type, such as a string for a number; the message names the
        method, the argument, what it accepts, and the value. Also if ``knowledge`` is neither
        ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If an argument lies outside its range or is not one of its choices; the message names the
        method, the argument, what it accepts, and the value. Also if the fit's loss or weights stop
        being finite, for example when the data's scale overflows its covariance;
        ``standardize=True`` rescales the columns. Also if ``knowledge`` names a column ``data``
        lacks or contradicts itself, as :class:`~andrey.Knowledge` lists.

    Warns
    -----
    SearchLimitWarning
        When the fit stops on ``max_iter`` rounds before the acyclicity residual clears ``h_tol`` or
        ``rho`` reaches ``rho_max``; raise ``max_iter`` to let it run on.
    DataTypeWarning
        When a column holds at most 10 distinct whole numbers and so may hold category codes, which
        the fit reads as numbers. Each message warns once per process.
    BoundaryWarning
        When the data are larger than the largest size whose fits finished within the time and
        memory caps of the published benchmark, at the measured setting nearest the data's rows, and
        fits of the next size ran past a cap. It warns once per process and names that size and the
        release card; the size is a reference, not a hard limit.

    Notes
    -----
    **When to use.** CALM suits i.i.d. continuous observational data from a linear model with
    Gaussian noise, each variable with its own noise variance, and no unmeasured common causes
    [1]_; the Fisher-Z tests of its screen assume the same model. Under that model the data
    identify the DAG only up to its Markov equivalence class, so an edge the class leaves
    undirected gets a direction the data do not determine. The screen keeps only the edges of the
    estimated moral graph, which joins each variable to its parents, its children, and its
    children's other parents, so it suits sparse graphs; it returns a DAG with the coefficient of
    each edge. When the equivalence class is enough, :func:`~andrey.ges` returns it as a CPDAG; when
    the noise is non-Gaussian and every edge should be oriented, use :func:`~andrey.direct_lingam`;
    with nonlinear effects, use :func:`~andrey.cam`.

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
    constraints = resolve(knowledge, X.shape[1], labels, method="calm")
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
            constraints=constraints,
        )
    return structure_output(
        result.dag,
        labels=labels,
        weighted_adjacency=result.weighted_adjacency,
        metadata={
            "h_final": json_safe(result.h_final),
            "objective": json_safe(result.objective),
        },
    )
