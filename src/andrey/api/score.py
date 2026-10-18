"""Score-based facade adapters.

``ges``, ``gies``, ``xges``, ``exact_search``, ``hc``, and ``tabu`` use :mod:`andrey.core` for BIC
scoring and orientation; ``xges``, ``hc``, and ``tabu`` use the Schur delta-BIC. ``ges`` and ``hc``
also score categorical data with BDeu, and mixed data with the degenerate and conditional Gaussian
BICs. Reported objectives are stored in ``metadata``.

CALM lazily imports its stochastic continuous optimizer from the ``[torch]`` extra and returns a
weighted DAG. Its output varies across torch versions, so determinism and quality tests replace a
committed baseline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from andrey.api._adapt import (
    MAX_CATEGORIES,
    categorical_metadata,
    checked_data,
    column_labels,
    data_kind,
    data_matrix,
    flag,
    int_in_range,
    json_safe,
    structure_output,
)
from andrey.core import backend
from andrey.core.seeding import resolve_seed

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from andrey.core import StructureOutput


def ges(
    data: npt.ArrayLike,
    *,
    score_func: str = "local_score_BIC",
    lambda_value: float = 1.0,
    categorical: Sequence[str | int] | None = None,
    max_categories: int = MAX_CATEGORIES,
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
    score_func : str, default="local_score_BIC"
        Local score objective: ``"local_score_BIC"``, the linear-Gaussian BIC, for
        continuous data. ``"local_score_BDeu"`` is the BDeu score [3]_ and
        ``"local_score_BIC_categorical"`` the multinomial BIC [7]_, for categorical data coded by
        whole numbers. ``"local_score_degen_gauss"`` (degenerate Gaussian) [4]_ and
        ``"local_score_cond_gauss"`` (conditional Gaussian) [5]_ are BIC scores for mixed data,
        continuous and categorical columns. See Notes.
    lambda_value : float, default=1.0
        Multiplier on the BIC penalty of ``log(n_samples)`` per parameter: per parent for the
        linear-Gaussian BIC. Larger values give sparser graphs; ``1.0`` is the standard BIC. Must
        be finite and non-negative. BDeu ignores it; for the multinomial BIC it weighs ``log(n)``
        per free parameter of a variable's conditional probability table.
    categorical : sequence of str or int, optional
        The categorical columns of ``data``, for a mixed-data score: DataFrame column names or
        0-based positions. ``None`` infers them: a DataFrame's ``category``, ``bool``, and text
        columns, and every column of at most ``max_categories`` distinct whole numbers, named in a
        ``DataTypeWarning``. ``[]`` marks none. The other scores take no column types and reject it.
        See *Data types* in the :doc:`getting-started guide </docs/guides/getting-started>`.
    max_categories : int, default=10
        The most distinct whole numbers a column may hold to be inferred as categorical, at least
        ``2``. With a continuous score, columns within it get a one-time ``DataTypeWarning`` that
        they may be category codes.

    Returns
    -------
    StructureOutput
        Wraps a ``cpdag`` :class:`~andrey.core.GraphStructure`; ``metadata["algorithm"]`` is
        ``"GES"`` and ``metadata["score"]`` the score of the result; lower is better. For BIC, it is
        the sum over variables of ``n * log(residual variance) + lambda_value * log(n) *
        n_parents``; for BDeu, ``-2`` times the log of the structure prior times the marginal
        likelihood; for the multinomial BIC and the mixed-data scores, see Notes. Directed marks
        (``->``) are oriented edges, undirected (``--``) the ones the equivalence class leaves open.
        Under a mixed-data setting, ``metadata["categorical"]`` lists the categorical columns it
        read, by name or 0-based position.

    Raises
    ------
    NotImplementedError
        If ``score_func`` names no available score.
    TypeError
        If ``categorical`` is not a list of column names and positions, or ``max_categories`` is not
        an int.
    ValueError
        If ``lambda_value`` is negative or non-finite, if ``max_categories`` is less than ``2``,
        with a categorical score, if ``data`` holds a number that is not whole, or, with
        a mixed-data score, if ``categorical`` names a column ``data`` lacks, or if a categorical
        column holds a number that is not whole or a missing value. Also if ``categorical`` is given
        with another score. With the BIC score, also if a column repeats
        another up to scale and shift given more than two rows or, with more rows than columns, is a
        linear function of other columns.

    Warns
    -----
    DataTypeWarning
        When a mixed-data score infers the categorical columns, or finds none and so runs as the
        linear-Gaussian form, or when a continuous setting meets columns that look like category
        codes; each message warns once per process.

    See Also
    --------
    andrey.fci : Allows unmeasured common causes and selection bias.
    andrey.direct_lingam : Orients every edge when the noise is non-Gaussian.

    Notes
    -----
    **When to use it.** GES suits data with no unmeasured common causes. Its default BIC score
    assumes continuous data, where each variable depends linearly on its causes with Gaussian noise;
    in that setting it is a fast, accurate default. For categorical data, use
    ``"local_score_BDeu"`` or ``"local_score_BIC_categorical"``, and for mixed data
    ``"local_score_degen_gauss"`` or ``"local_score_cond_gauss"``.

    **When not to.** When unmeasured common causes are plausible, use :func:`~andrey.fci`. When the
    noise is non-Gaussian and every edge should be oriented, use :func:`~andrey.direct_lingam`.

    **Choosing lambda_value.** A larger value raises the penalty per parent, giving a sparser graph:
    fewer false edges and more missed ones. ``1.0`` is the standard BIC. With many variables and
    about ``10 * n_variables`` samples, ``lambda_value=2.0`` gave more accurate graphs than ``1.0``
    on the published benchmarks. FGES, a fast GES, multiplied the BIC penalty by 4 when searching
    continuous data on graphs of up to a million variables [2]_.

    **How it works.** In the forward phase, a variable with more than ``n_variables / 2`` parents
    gets no new edges. The backward phase has no cap.

    **Categorical data.** ``"local_score_BDeu"`` takes each distinct value of a column as one
    category, so the codes need not start at 0 or be consecutive; a DataFrame's text, ``bool``, and
    ``category`` columns are read the same way. It scores a variable given its parents by the
    marginal likelihood of its counts in each parent configuration, under a Dirichlet prior with an
    equivalent sample size of 1 spread evenly over the cells [3]_. A structure prior with one
    expected parent per variable multiplies it, from three variables up. Equivalent graphs score the
    same, as GES requires. Moves are scored in this process, without the worker processes the BIC
    search can use. ``"local_score_BIC_categorical"`` reads the columns the same way and scores a
    variable by ``-2`` times the log-likelihood of its counts in each parent configuration, plus
    ``lambda_value * log(n)`` for each of the ``q * (r - 1)`` free parameters, with ``r`` its
    categories and ``q`` its parents' joint categories [7]_. Both scores are on the scale of the
    BIC: ``-2`` times a log score.

    **Mixed data.** ``"local_score_degen_gauss"`` and ``"local_score_cond_gauss"`` score data with
    continuous and categorical columns; the categorical columns are read as :func:`~andrey.pc`
    describes. Each is a BIC: ``-2`` times the fitted log-likelihood of a variable given its
    parents, plus ``lambda_value * log(n)`` per parameter; the degenerate Gaussian's leaves out a
    constant per variable, which does not change which graph scores best. Equivalent graphs score
    the same. The degenerate Gaussian score [4]_ replaces a categorical column of ``k`` categories
    by ``k - 1`` indicator columns and scores every column as Gaussian, by the residual covariance
    of a linear regression on the parents' columns; on continuous columns only, it equals
    ``"local_score_BIC"``. The conditional Gaussian score [5]_ fits the continuous variables of a
    variable and its parents in each joint category of their categorical ones, with a Gaussian of
    that category's own mean and a covariance shrunk toward that of all rows by one pseudo-row [6]_,
    and divides by the same fit of the parents alone, so a categorical variable can have continuous
    parents. It counts parameters for every joint category, so a parent with many categories costs
    many parameters; on continuous columns only, it differs from ``"local_score_BIC"`` by a
    constant per variable, so it finds the same graph. Moves are scored in this process.

    **Limitations.** Measured on simulated linear-Gaussian data from sparse random graphs (mean
    degree 2), from 20 to 200 variables with ten rows per variable and parents capped at 4 per
    variable rather than ``n_variables / 2``. At ``lambda_value=1.0`` its median errors (SHD) stayed
    below the empty graph's up to 150 variables (141 against 150) and passed it at 200 (206 against
    200), where ``lambda_value=2.0`` brought them to 43. The median fit took 0.15 s at 20 variables
    and 40 s at 150, on 128 cores. On the 223-variable ANDES network with 1,000 simulated rows and
    the default parent cap, its median errors passed the empty graph's (612 against 338). With
    fewer rows than columns the BIC score is unreliable. A parent set of ``n_samples - 1`` or more
    variables fits its child exactly, so the search never selects one, nor any parent set that
    leaves its child a residual variance of at most ``1e-12`` times its variance. On simulated
    mixed data (conditional linear Gaussian networks on sparse random graphs, the first half of the
    variables categorical with 2 to 5 categories), from 10 to 40 variables with 2,000 rows, its
    median errors at 40 variables were 8 with ``"local_score_degen_gauss"`` and 6.5 with
    ``"local_score_cond_gauss"``, against 40, and the median fit took 0.75 s and 1.2 s on 4 cores.
    On simulated categorical networks with 2 or 3 categories per variable, of the same sizes and
    rows, ``"local_score_BDeu"`` gave 6 against 40 at 40 variables, in 0.60 s. The multinomial BIC
    is not benchmarked with GES yet: its tests check it against the formula and run GES on small
    fixed categorical networks.

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
    >>> ges(data, score_func="local_score_BDeu").structure.oriented_edges()
    [('a', 'c', 'directed'), ('b', 'c', 'directed')]

    On mixed data, with a category column and the conditional Gaussian score:

    >>> g = rng.integers(0, 3, 2000)
    >>> u = rng.standard_normal(2000)
    >>> v = np.array([-1.0, 0.0, 1.0])[g] + u + 0.5 * rng.standard_normal(2000)
    >>> data = pd.DataFrame({"g": pd.Categorical(g), "u": u, "v": v})
    >>> ges(data, score_func="local_score_cond_gauss").structure.oriented_edges()
    [('g', 'v', 'directed'), ('u', 'v', 'directed')]
    """
    from andrey.search.ges import ges as _ges

    X, mask = checked_data(
        data, categorical, "score_func", score_func, max_categories=max_categories
    )
    labels = column_labels(data)
    cpdag, score = _ges(X, score_func=score_func, lambda_value=lambda_value, categorical=mask)
    return structure_output(
        cpdag,
        labels=labels,
        metadata={
            "algorithm": "GES",
            "score": json_safe(score),
            **categorical_metadata(mask, data),
        },
    )


def gies(
    data: npt.ArrayLike,
    interventions: npt.ArrayLike | None = None,
    *,
    lambda_value: float = 1.0,
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
        variable needs at least two rows without an intervention.
    lambda_value : float, default=1.0
        Multiplier on the BIC penalty of ``log(n_samples)`` per parent. Larger values give
        sparser graphs; ``1.0`` is the standard BIC. Must be finite and non-negative.

    Returns
    -------
    StructureOutput
        Wraps a ``cpdag`` :class:`~andrey.core.GraphStructure`, the interventional essential
        graph: directed marks (``->``) are edges every graph in the class orients alike,
        undirected (``--``) the ones it leaves open. ``metadata["algorithm"]`` is ``"GIES"`` and
        ``metadata["score"]`` the BIC score of the result, the sum over variables of
        ``n_v * log(residual variance) + lambda_value * log(n_samples) * n_parents``, where
        ``n_v`` counts the rows without an intervention on the variable; lower is better.

    Raises
    ------
    ValueError
        If ``data`` is not a 2-D numeric matrix with at least two rows, no ``NaN`` or ``inf``, and
        no constant column; if ``interventions`` does not have ``data``'s shape, holds a value
        other than 0 or 1, or leaves a variable fewer than two rows, or one value, without an
        intervention; if ``lambda_value`` is negative or non-finite; or if a column repeats another
        up to scale and shift given more than two rows or, with more rows than columns, is a linear
        function of other columns.

    See Also
    --------
    andrey.ges : Searches the same way on observational data, without the turning phase.

    Notes
    -----
    **When to use it.** GIES makes the assumptions of :func:`~andrey.ges`: continuous data, no
    unmeasured common causes, and linear relations with Gaussian noise. An intervention replaces
    the intervened variable's mechanism and leaves every other variable's unchanged.

    **How it works.** The search starts from the empty graph and repeats three phases: adding
    edges, removing edges, and turning one edge around, each while it improves the score, until a
    round of removing and turning changes nothing. No variable's parent count is capped. Each step
    moves to the best neighboring equivalence class, and the classes are those of the
    interventional setting: two graphs are equivalent when they agree on the data from every
    observed combination of intervened variables.

    **Limitations.** Measured on simulated linear-Gaussian data that mixes observation with
    interventions, from sparse random graphs (mean degree 2), from 10 to 200 variables with ten
    rows per variable and four intervention targets per ten variables. Its median errors (SHD
    against the true interventional essential graph) stayed below the empty graph's at every size
    (181 against 198 at 200 variables, where one of the five fits had more). The median fit took
    0.25 s at 100 variables and 1.0 s at 200, on 4 cores. Each distinct set of rows without an
    intervention on a variable keeps its own covariance matrix of all the variables, so memory
    grows with the number of intervened variables times the square of the variable count.

    References
    ----------
    .. [1] Hauser, A., and Buhlmann, P. (2012). Characterization and greedy learning of
       interventional Markov equivalence classes of directed acyclic graphs. JMLR 13, 2409-2464.

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
    """
    from andrey.search.gies import gies as _gies
    from andrey.search.gies import intervention_mask

    X = data_matrix(data)
    labels = column_labels(data)
    mask = None if interventions is None else intervention_mask(interventions, X)
    graph, score = _gies(X, mask, lambda_value=lambda_value)
    metadata = {"algorithm": "GIES", "score": json_safe(score)}
    return structure_output(graph, labels=labels, metadata=metadata)


def xges(
    data: npt.ArrayLike,
    *,
    score_func: str = "local_score_BIC",
    lambda_value: float = 1.0,
    extended_search: bool = True,
) -> StructureOutput:
    """Learn a CPDAG from observational data with extremely greedy equivalence search (XGES).

    XGES searches the equivalence classes that GES searches [2]_, with GES's moves, the turning move
    of GIES [3]_, and the BIC score, but applies them in a different order [1]_ (see Notes). On
    linear-Gaussian data it often finds a graph with a better score than GES.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``.
    score_func : {"local_score_BIC"}, default="local_score_BIC"
        Only the Gaussian BIC score is supported.
    lambda_value : float, default=1.0
        Multiplier on the BIC penalty of ``log(n_samples)`` per parent. Larger values give
        sparser graphs; ``1.0`` is the standard BIC. Must be finite and non-negative.
    extended_search : bool, default=True
        Run the extended search after the base search (see Notes). ``False`` returns the base
        search's graph, which is faster.

    Returns
    -------
    StructureOutput
        Wraps a ``cpdag`` :class:`~andrey.core.GraphStructure`; ``metadata["algorithm"]`` is
        ``"XGES"`` and ``metadata["score"]`` the BIC score, computed as for :func:`andrey.ges`;
        lower is better. Directed marks (``->``) are oriented edges, undirected (``--``) the ones
        the equivalence class leaves open.

    Raises
    ------
    NotImplementedError
        If ``score_func`` is not ``"local_score_BIC"``.
    TypeError
        If ``extended_search`` is not a bool.
    ValueError
        If ``data`` is not a 2-D numeric matrix with at least two rows and no ``NaN`` or ``inf``, if
        ``lambda_value`` is negative or non-finite, or if a column is constant, repeats another
        column up to scale and shift given more than two rows, or, with more rows than columns, is a
        linear function of other columns.

    See Also
    --------
    andrey.ges : Adds edges, then removes them, in two separate phases.

    Notes
    -----
    **When to use it.** XGES makes the assumptions of :func:`~andrey.ges`: continuous data, no
    unmeasured common causes, and linear relations with Gaussian noise. It returns the equivalence
    class.

    **How it works.** Each step takes the best improving deletion, else reversal of a directed edge,
    else addition; the search stops when no move improves the score [1]_. The extended search then
    deletes each edge of that graph, best score first, forbids adding it back in that direction, and
    reruns the search. It keeps a result whose score is lower by more than ``2e-7 * n_samples``, and
    stops once a fresh pass over the edges keeps nothing. No variable has a parent limit.

    **Limitations.** Measured on simulated linear-Gaussian data from sparse random graphs (mean
    degree 2) and denser ones (mean degree 4), from 25 to 1,000 variables with ten rows per
    variable. On the denser graphs its median errors (SHD) stayed below the empty graph's at every
    size. On the sparse graphs they exceeded it from 500 variables on, where the BIC score still
    preferred the returned graphs to the true one. At 1,000 variables, the median fit took under a
    minute on the sparse graphs and under five minutes on the denser ones, on one core; with
    ``extended_search=False``, a few seconds. The extended search reruns the base search for each
    edge it tries to delete and retries the edges after keeping a result, so its run time grows
    with the number of edges and of results kept. Its BIC score assumes linear relations with
    Gaussian noise.
    With fewer rows than columns the BIC score is unreliable. A parent set of ``n_samples - 1`` or
    more variables fits its child exactly, so the search never selects one, nor any parent set that
    leaves its child a residual variance of at most ``1e-12`` times its variance.

    References
    ----------
    .. [1] Nazaret, A., and Blei, D. (2024). Extremely greedy equivalence search. Proceedings of
       the Fortieth Conference on Uncertainty in Artificial Intelligence, PMLR 244, 2716-2745.
    .. [2] Chickering, D. M. (2002). Optimal structure identification with greedy search. JMLR 3,
       507-554.
    .. [3] Hauser, A., and Buhlmann, P. (2012). Characterization and greedy learning of
       interventional Markov equivalence classes of directed acyclic graphs. JMLR 13, 2409-2464.

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
    """
    from andrey.search.xges import xges as _xges

    X = data_matrix(data)
    extended_search = flag(extended_search, "extended_search")
    labels = column_labels(data)
    cpdag, score = _xges(
        X, score_func=score_func, lambda_value=lambda_value, extended_search=extended_search
    )
    metadata = {"algorithm": "XGES", "score": json_safe(score)}
    return structure_output(cpdag, labels=labels, metadata=metadata)


def hc(
    data: npt.ArrayLike,
    *,
    score_func: str = "local_score_BIC_from_cov",
    lambda_value: float = 1.0,
    max_iter: int | None = None,
    categorical: Sequence[str | int] | None = None,
    max_categories: int = MAX_CATEGORIES,
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
    score_func : str, default="local_score_BIC_from_cov"
        Local score objective: ``"local_score_BIC_from_cov"``, a covariance-based Gaussian BIC for
        continuous data; ``"local_score_BDeu"``, the BDeu score, or
        ``"local_score_BIC_categorical"``, the multinomial BIC, for categorical data coded by whole
        numbers; or ``"local_score_degen_gauss"`` and ``"local_score_cond_gauss"``, the degenerate
        and conditional Gaussian BICs for mixed data, as for :func:`~andrey.ges`.
    lambda_value : float, default=1.0
        Multiplier on the BIC penalty of ``log(n_samples)`` per parameter: per parent for the
        Gaussian BIC. Larger values give sparser graphs; ``1.0`` is the standard BIC. Must be
        finite and non-negative. BDeu ignores it.
    max_iter : int or None, default=None
        Maximum moves, at least ``1``; ``None`` sets no limit. The search still terminates, since
        each move lowers the score. A given ``max_iter`` also bounds the edge count, as each move
        adds at most one edge.
    categorical : sequence of str or int, optional
        The categorical columns of ``data``, for a mixed-data score: DataFrame column names or
        0-based positions. ``None`` infers them: a DataFrame's ``category``, ``bool``, and text
        columns, and every column of at most ``max_categories`` distinct whole numbers, named in a
        ``DataTypeWarning``. ``[]`` marks none. The other scores take no column types and reject it.
        See *Data types* in the :doc:`getting-started guide </docs/guides/getting-started>`.
    max_categories : int, default=10
        The most distinct whole numbers a column may hold to be inferred as categorical, at least
        ``2``. With a continuous score, columns within it get a one-time ``DataTypeWarning`` that
        they may be category codes.

    Returns
    -------
    StructureOutput
        Wraps a ``cpdag`` :class:`~andrey.core.GraphStructure`; ``metadata["algorithm"]`` is
        ``"HC"`` and ``metadata["score"]`` the score of the DAG found, computed as for
        :func:`andrey.ges`; lower is better. Directed marks (``->``) are oriented edges, undirected
        (``--``) the ones the equivalence class leaves open. Under a mixed-data setting,
        ``metadata["categorical"]`` lists the categorical columns it read, by name or 0-based
        position.

    Raises
    ------
    NotImplementedError
        If ``score_func`` names no available score.
    TypeError
        If a given ``max_iter`` is not an int, ``categorical`` is not a list of column names and
        positions, or ``max_categories`` is not an int.
    ValueError
        If ``lambda_value`` is negative or non-finite, if a given ``max_iter`` is less than ``1``,
        or if ``data`` or ``categorical`` does not suit the score, as for :func:`~andrey.ges`. With
        the BIC score, also if a column repeats another up to scale and shift given more than two
        rows or, with more rows than columns, is a linear function of other columns.

    Warns
    -----
    SearchLimitWarning
        When the search takes a given ``max_iter`` moves; a larger ``max_iter`` may find a better
        graph.
    DataTypeWarning
        When a mixed-data score infers the categorical columns, or finds none and so runs as the
        linear-Gaussian form, or when a continuous setting meets columns that look like category
        codes; each message warns once per process.

    Notes
    -----
    The search starts from the empty graph and stops when no single move improves the score, or
    after ``max_iter`` moves when set. Under a mixed-data score, no move gives a categorical column
    a continuous parent, so the DAG found is a conditional linear Gaussian network.

    Under a score-equivalent score, adding ``x -> y`` or ``y -> x`` from the empty graph gives the
    same score. Such ties recur during search, and choosing one move can change later moves and
    the final CPDAG. Each column gets a rank computed from its own data to choose between tied
    moves. The rank moves with the column, so reordering the columns reorders the graph's nodes
    and changes nothing else, unless two columns are identical. The rank is arbitrary with respect
    to causal structure.

    A continuous column ranks by whether its values rise, fall, or stay equal from row to row,
    which last-bit differences in the data, such as a BLAS kernel's on another CPU, leave
    unchanged; a categorical column ranks by which rows share a category, which recoding the
    categories leaves unchanged. So the same data give the same graph on other machines, unless two
    tied score changes round to different values.

    **Limitations.** Measured on simulated linear-Gaussian data from sparse random graphs, from 10
    to 3,200 variables with ten rows per variable. From 50 variables on, its median errors (SHD)
    exceeded the empty graph's. The median fit took under a second at 800 variables and about 20 s
    at 3,200, on 4 cores. It has no move limit by default; run time grows with the moves the search
    takes. A graph that no single move improves need not be the best-scoring one. Its BIC score
    assumes linear relations with Gaussian noise. With fewer rows than columns the BIC score is
    unreliable. A parent set of ``n_samples - 1`` or more variables fits its child exactly, so the
    search never selects one, nor any parent set that leaves its child a residual variance of at
    most ``1e-12`` times its variance. On simulated mixed data (conditional linear Gaussian
    networks on sparse random graphs, the first half of the variables categorical with 2 to 5
    categories), from 10 to 40 variables with 2,000 rows, its median errors at 40 variables were
    12.5 with ``"local_score_degen_gauss"`` and 11 with ``"local_score_cond_gauss"``, against 40. On
    simulated categorical networks with 2 or 3 categories per variable, of the same sizes and rows,
    they were 18 with ``"local_score_BDeu"`` and 20 with ``"local_score_BIC_categorical"``. Every
    median fit took under 0.2 s at 40 variables, on 4 cores.

    References
    ----------
    .. [1] Chickering, D. M., Geiger, D., and Heckerman, D. (1995). Learning Bayesian networks:
       search methods and experimental results. AISTATS, PMLR R0, 112-128.
    .. [2] Scutari, M. (2010). Learning Bayesian networks with the bnlearn R package. Journal of
       Statistical Software 35(3). https://doi.org/10.18637/jss.v035.i03

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

    X, mask = checked_data(
        data, categorical, "score_func", score_func, max_categories=max_categories
    )
    max_iter = None if max_iter is None else int_in_range(max_iter, "max_iter", 1)
    labels = column_labels(data)
    cpdag, score = _hc(
        X, score_func=score_func, lambda_value=lambda_value, max_iter=max_iter, categorical=mask
    )
    metadata = {"algorithm": "HC", "score": json_safe(score), **categorical_metadata(mask, data)}
    return structure_output(cpdag, labels=labels, metadata=metadata)


def tabu(
    data: npt.ArrayLike,
    *,
    score_func: str = "local_score_BIC_from_cov",
    lambda_value: float = 1.0,
    tabu_length: int = 10,
    max_tabu: int | None = None,
    max_iter: int | None = None,
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
    score_func : str, default="local_score_BIC_from_cov"
        Local score objective: ``"local_score_BIC_from_cov"``, a covariance-based Gaussian BIC for
        continuous data, or ``"local_score_BDeu"`` or ``"local_score_BIC_categorical"``, the BDeu
        score or the multinomial BIC for categorical data coded by whole numbers, as for
        :func:`~andrey.ges`.
    lambda_value : float, default=1.0
        Multiplier on the BIC penalty of ``log(n_samples)`` per parameter: per parent for the
        Gaussian BIC. Larger values give sparser graphs; ``1.0`` is the standard BIC. Must be finite
        and non-negative. BDeu ignores it.
    tabu_length : int, default=10
        Number of most recently visited DAGs a move may not return to. At least ``1``.
    max_tabu : int or None, default=None
        Non-improving moves allowed without a better DAG, from ``1`` to ``tabu_length``. ``None``
        uses ``tabu_length``.
    max_iter : int or None, default=None
        Maximum moves, improving or not, at least ``1``; ``None`` sets no limit. The search still
        terminates: non-improving runs are bounded by ``max_tabu``, and each new best DAG scores
        strictly lower.

    Returns
    -------
    StructureOutput
        Wraps a ``cpdag`` :class:`~andrey.core.GraphStructure`; ``metadata["algorithm"]`` is
        ``"Tabu"`` and ``metadata["score"]`` the best DAG's score, computed as for
        :func:`andrey.ges`; lower is better. Directed marks (``->``) are oriented edges, undirected
        (``--``) the ones the equivalence class leaves open.

    Raises
    ------
    NotImplementedError
        If ``score_func`` names no score tabu search takes.
    TypeError
        If ``tabu_length``, ``max_tabu``, or a given ``max_iter`` is not an int.
    ValueError
        If ``lambda_value`` is negative or non-finite, if ``tabu_length`` or a given ``max_iter`` is
        less than ``1``, if ``max_tabu`` is outside ``1..tabu_length``, or, with a categorical
        score, if ``data`` holds a number that is not whole. With the BIC score, also if a column
        repeats another up to scale and shift given more than two rows or, with more rows than
        columns, is a linear function of other columns.

    Warns
    -----
    SearchLimitWarning
        When the search takes a given ``max_iter`` moves; a larger ``max_iter`` may find a better
        graph.

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

    **Limitations.** Measured on simulated linear-Gaussian data from sparse random graphs, from 10
    to 3,200 variables with ten rows per variable. From 50 variables on, its median errors (SHD)
    exceeded the empty graph's. The median fit took under 2 s at 800 variables and under a minute
    at 3,200, on 4 cores. It takes hill climbing's moves plus up to ``max_tabu`` non-improving
    moves between new best scores, with no move limit by default; run time grows with the moves
    taken. It avoids only the last ``tabu_length`` DAGs, so a better DAG beyond that reach may be
    missed. Its BIC score assumes linear relations with Gaussian noise. With fewer rows than
    columns the BIC score is unreliable. A parent set of ``n_samples - 1`` or more variables fits
    its child exactly, so the search never selects one, nor any parent set that leaves its child a
    residual variance of at most ``1e-12`` times its variance. With a categorical score, its tests
    run it on small fixed categorical networks.

    References
    ----------
    .. [1] Glover, F. (1986). Future paths for integer programming and links to artificial
       intelligence. Computers & Operations Research 13(5), 533-549.
       https://doi.org/10.1016/0305-0548(86)90048-1
    .. [2] Russell, S. J., and Norvig, P. (2009). Artificial Intelligence: A Modern Approach,
       3rd edition. Prentice Hall.
    .. [3] Scutari, M. (2010). Learning Bayesian networks with the bnlearn R package. Journal of
       Statistical Software 35(3). https://doi.org/10.18637/jss.v035.i03

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
    from andrey.search.tabu import TABU_SCORES
    from andrey.search.tabu import tabu as _tabu

    if score_func not in TABU_SCORES:
        raise NotImplementedError(f"tabu supports {TABU_SCORES}, not {score_func!r}")
    if data_kind(score_func) == "categorical":
        X, _ = checked_data(data, None, "score_func", score_func)
    else:
        X = data_matrix(data)
    tabu_length = int_in_range(tabu_length, "tabu_length", 1)
    max_tabu = int_in_range(
        tabu_length if max_tabu is None else max_tabu, "max_tabu", 1, tabu_length
    )
    max_iter = None if max_iter is None else int_in_range(max_iter, "max_iter", 1)
    labels = column_labels(data)
    cpdag, score = _tabu(
        X,
        score_func=score_func,
        lambda_value=lambda_value,
        tabu_length=tabu_length,
        max_tabu=max_tabu,
        max_iter=max_iter,
    )
    metadata = {"algorithm": "Tabu", "score": json_safe(score)}
    return structure_output(cpdag, labels=labels, metadata=metadata)


def exact_search(
    data: npt.ArrayLike,
    *,
    search_method: str = "astar",
    score_func: str = "local_score_BIC",
    categorical: Sequence[str | int] | None = None,
    max_categories: int = MAX_CATEGORIES,
) -> StructureOutput:
    """Learn the globally optimal DAG for a decomposable score by exact search.

    The optimum is identified up to its Markov equivalence class. A* (``search_method="astar"``)
    [1]_ and dynamic programming (``"dp"``) [2]_ may return different representatives. Runtime is
    exponential in the variable count, so keep the problem small (roughly <= 20 variables).

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``.
    search_method : {"astar", "dp"}, default="astar"
        Exact search strategy: A* search or dynamic programming.
    score_func : str, default="local_score_BIC"
        Local score: ``"local_score_BIC"``, the linear-Gaussian BIC; ``"local_score_BDeu"`` or
        ``"local_score_BIC_categorical"``, the BDeu score or the multinomial BIC for categorical
        data coded by whole numbers; or ``"local_score_degen_gauss"`` or
        ``"local_score_cond_gauss"``, the degenerate or conditional Gaussian BIC for mixed data, as
        for :func:`~andrey.ges`. Each BIC uses the standard penalty, ``lambda_value=1``.
    categorical : sequence of str or int, optional
        The categorical columns of ``data``, for a mixed-data score: DataFrame column names or
        0-based positions. ``None`` infers them: a DataFrame's ``category``, ``bool``, and text
        columns, and every column of at most ``max_categories`` distinct whole numbers, named in a
        ``DataTypeWarning``. ``[]`` marks none. The other scores take no column types and reject it.
        See *Data types* in the :doc:`getting-started guide </docs/guides/getting-started>`.
    max_categories : int, default=10
        The most distinct whole numbers a column may hold to be inferred as categorical, at least
        ``2``. With a continuous score, columns within it get a one-time ``DataTypeWarning`` that
        they may be category codes.

    Returns
    -------
    StructureOutput
        Wraps a ``dag`` :class:`~andrey.core.GraphStructure` (one representative of the optimal
        equivalence class); ``metadata["algorithm"]`` is ``"ExactSearch"``. Under a mixed-data
        setting, ``metadata["categorical"]`` lists the categorical columns it read, by name or
        0-based position.

    Raises
    ------
    NotImplementedError
        If ``score_func`` names no available score.
    TypeError
        If ``categorical`` is not a list of column names and positions, or ``max_categories`` is not
        an int.
    ValueError
        If ``search_method`` is not ``"astar"`` or ``"dp"``, or if ``data`` or ``categorical``
        does not suit the score, as for :func:`~andrey.ges`. With the BIC score, also if a column
        repeats another up to scale and shift given more than two rows or, with more rows than
        columns, is a linear function of other columns.

    Warns
    -----
    DataTypeWarning
        When a mixed-data score infers the categorical columns, or finds none and so runs as the
        linear-Gaussian form, or when a continuous setting meets columns that look like category
        codes; each message warns once per process.

    Notes
    -----
    **When to use it.** Exact search makes the assumptions of :func:`~andrey.ges`: continuous
    data, no unmeasured common causes, and linear relations with Gaussian noise. It suits a few
    variables, where it returns a graph with the lowest BIC rather than a greedy search's local
    optimum.

    **Categorical data.** ``"local_score_BDeu"`` [5]_ and ``"local_score_BIC_categorical"`` [6]_
    read each distinct value of a column as one category and score a variable by its counts in
    each configuration of its parents, as :func:`~andrey.ges` describes. Both give equivalent graphs
    the same score.

    **Mixed data.** ``"local_score_degen_gauss"`` [3]_ and ``"local_score_cond_gauss"`` [4]_ score
    data with continuous and categorical columns, as :func:`~andrey.ges` describes, and read the
    categorical columns as :func:`~andrey.pc` does. Both give equivalent graphs the same score, so
    the optimum is again identified up to its equivalence class.

    **Limitations.** Measured on simulated linear-Gaussian data from sparse random graphs (mean
    degree 2), from 5 to 15 variables with ten rows per variable. Its median errors (SHD) stayed
    below the empty graph's at every size (7 against 15 at 15 variables). The median fit took
    0.11 s at 10 variables and 6.4 s at 15, on 4 cores. With a mixed-data or categorical score it
    is not benchmarked yet: its tests run it with a mixed-data score on 5 variables with 3,000 rows,
    and with a categorical score on small fixed categorical networks.
    It scores every possible parent set of every variable, so its run time at least doubles with
    each added variable. The ``"dp"`` search also keeps one entry per subset of the variables, so
    its memory doubles as well.
    With fewer rows than columns the BIC score is unreliable. A parent set of ``n_samples - 1`` or
    more variables fits its child exactly, so the search never selects one, nor any parent set that
    leaves its child a residual variance of at most ``1e-12`` times its variance.

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
    from andrey.search.exact import EXACT_SCORES
    from andrey.search.exact import exact_search as _exact

    if score_func not in EXACT_SCORES:
        raise NotImplementedError(f"exact_search supports {EXACT_SCORES}, not {score_func!r}")
    X, mask = checked_data(
        data, categorical, "score_func", score_func, max_categories=max_categories
    )
    labels = column_labels(data)
    dag = _exact(X, search_method=search_method, score_func=score_func, categorical=mask)
    return structure_output(
        dag,
        labels=labels,
        metadata={"algorithm": "ExactSearch", **categorical_metadata(mask, data)},
    )


def calm(data: npt.ArrayLike, *, seed: int | None = None, **params: Any) -> StructureOutput:
    """Learn a weighted DAG with CALM, a continuous-optimization structure learner.

    CALM (Continuous and Acyclicity-constrained L0-penalized likelihood with an estimated Moral
    graph) [1]_ fits a linear-Gaussian DAG with torch. By default it keeps only the candidate edges
    of an estimated moral graph, then minimizes the Gaussian likelihood plus a penalty per edge
    under an acyclicity constraint. The optimizer draws random numbers and is seeded on every call;
    see ``seed``.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``.
    seed : int or None, default=None
        Seed for the optimizer's torch random number generator. ``None`` uses the
        ``ANDREY_SEED`` environment variable, or ``0`` when it is unset, so every call is seeded
        and repeated calls return the same graph on one machine with the same torch version.
        ``metadata["seed"]`` records the seed used.
    **params : Any
        Optimizer settings passed to the solver; a name not listed here raises ``TypeError``.

        - ``lambda1`` (default ``0.005``): penalty per edge. Larger values give sparser graphs.
        - ``alpha`` (default ``0.01``): significance level of the Fisher-Z tests that estimate the
          moral graph. Only pairs it keeps can become edges; smaller values keep fewer.
        - ``use_moral_graph`` (default ``True``): ``False`` makes every pair a candidate edge.
        - ``standardize`` (default ``False``): scale each column to unit variance first.
        - ``w_threshold`` (default ``0.3``) and ``gate_threshold`` (default ``0.5``): an edge is
          kept when its weight magnitude and its gate probability reach these values.
        - ``max_outer`` (default ``25``), ``subproblem_iter`` (default ``3000``), and ``lr``
          (default ``0.01``): quadratic-penalty rounds, Adam steps per round, and learning rate.
        - ``tau`` (default ``0.5``), ``rho_init`` (default ``0.001``), ``rho_mult`` (default
          ``3.0``), ``rho_max`` (default ``1e16``), and ``h_tol`` (default ``1e-8``): gate
          temperature and the acyclicity-penalty schedule.
        - ``labels`` (default ``None``): node names; a DataFrame's column names take precedence.

    Returns
    -------
    StructureOutput
        Wraps a ``dag`` :class:`~andrey.core.GraphStructure`; ``weighted_adjacency[i, j]`` is the
        coefficient of edge ``i -> j``. ``metadata`` holds ``"algorithm"`` (``"CALM"``), the
        ``"seed"`` used, ``"n_edges"``, ``"objective"`` (half the sum of the log residual
        variances plus ``lambda1`` times the edge count; lower is better), and ``"h_final"`` (the
        acyclicity residual of the returned graph, ``0`` when it is acyclic).

    Raises
    ------
    ImportError
        If the ``[torch]`` extra is not installed.
    TypeError
        If ``params`` holds a name the solver does not accept.
    ValueError
        If the fit's loss or weights stop being finite, for example when the data's scale
        overflows its covariance; ``standardize=True`` rescales the columns.

    Notes
    -----
    CALM first estimates the moral graph with IAMB [2]_: for each variable it adds, one at a time,
    the variable most dependent on it given those already added (Fisher-Z at level ``alpha``),
    then drops any member the others make independent of it. A pair is a candidate edge, in
    either direction, when each is in the other's set.

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
    reaches ``rho_max``, or ``max_outer`` rounds have run.

    An edge is kept when its noise-free gate reaches ``gate_threshold`` and its weight magnitude
    reaches ``w_threshold``. If the kept edges still form a cycle, edges are removed in order of
    increasing weight magnitude until none does.

    The defaults are a shorter schedule than the paper's [1]_, which starts ``rho`` at ``1e-5``,
    runs 40,000 steps per round at learning rate ``0.001``, and keeps every edge whose gate
    reaches ``0.5``. ``rho_init=1e-5, subproblem_iter=40000, lr=0.001, w_threshold=0.0`` with a
    larger ``max_outer`` runs that schedule.

    **Limitations.** Measured on simulated linear-Gaussian data from sparse random graphs (mean
    degree 2), from 10 to 50 variables with ten rows per variable. Its median errors (SHD) stayed
    below the empty graph's at every size (15 against 50 at 50 variables). The median fit took 18 s
    at 10 variables and 59 s at 50, on 4 cores. The paper's schedule (``lr=0.001``,
    ``max_outer=100``, ``subproblem_iter=40000``, ``rho_init=1e-5``, ``w_threshold=0.0``) took 174 s
    and 472 s for median errors of 6 and 12. Every optimizer step computes a determinant and a
    matrix power of the full weight matrix, one row and one column per variable, so a step's cost
    grows with the cube of the variable count; the defaults allow up to 75,000 steps (``max_outer``
    times ``subproblem_iter``). The fit is nonconvex and can stop at a local optimum: on a collider
    whose two parents are independent, it can return both edges reversed and an edge between the
    parents, with the defaults and with the paper's schedule alike.

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

    X = data_matrix(data)
    labels = column_labels(data)
    used_seed = resolve_seed(seed)  # CALM uses this seed for torch.
    result = _calm(X, seed=used_seed, **params)
    return structure_output(
        result.dag,
        labels=labels,
        weighted_adjacency=result.weighted_adjacency,
        metadata={
            "algorithm": "CALM",
            "seed": used_seed,
            "n_edges": int((result.weighted_adjacency != 0).sum()),
            "h_final": json_safe(result.h_final),
            "objective": json_safe(result.objective),
        },
    )
