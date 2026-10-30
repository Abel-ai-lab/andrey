"""Constraint-based facade adapters.

The underlying searches use :mod:`andrey.core` for the conditional-independence tests (Fisher-Z and
Student-t on continuous data, chi-square and G-square on categorical data, degenerate and
conditional Gaussian likelihood ratios on mixed data, and the nonlinear tests) and orientation.
``pc`` and ``cdnod`` produce CPDAGs; ``fci`` and ``gfci`` produce PAGs. GFCI also uses GES with a
BIC, BDeu, or mixed-data score.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from andrey.api._adapt import (
    categorical_metadata,
    check_categories,
    checked_data,
    column_labels,
    column_namer,
    copies_named,
    data_kind,
    default_score,
    default_test,
    structure_output,
)
from andrey.api._method import method_spec
from andrey.core.ci_missing import rows_metadata, rows_tested
from andrey.core.knowledge import Knowledge, resolve

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from andrey.core import StructureOutput


@method_spec("pc")
def pc(
    data: npt.ArrayLike,
    *,
    alpha: float = 0.05,
    indep_test: str | None = None,
    collider_order: str | None = None,
    collider_rule: str = "sepsets",
    categorical: Sequence[str | int] | None = None,
    seed: int | None = None,
    max_depth: int | None = None,
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn a CPDAG from observational data with the PC algorithm.

    PC [1]_ [2]_ uses conditional-independence tests to remove edges from a complete graph. These
    tests check whether variables are independent given other variables. It then orients colliders,
    where arrows meet at a variable, and applies Meek's rules [4]_. The result is a CPDAG,
    representing an equivalence class of graphs whose unresolved edges stay undirected.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``.
    alpha : float, default=0.05
        Significance level of the conditional-independence test. Smaller values give sparser graphs.
    indep_test : str or None, default=None
        Conditional-independence test. ``"fisherz"`` and ``"student_t"`` test partial correlation,
        for continuous data: ``"fisherz"`` applies Fisher's z-transform and the normal distribution;
        ``"student_t"`` applies the Student-t distribution, which is exact for Gaussian data at any
        sample size. ``"normal_scores"`` applies ``"fisherz"`` to the normal scores of each column,
        for data whose columns are monotone transformations of Gaussian variables: its p-values do
        not change under a strictly increasing transformation of a column. ``"chisq"`` (Pearson's
        chi-square) and ``"gsq"`` (G-square, the likelihood ratio) test categorical data, coded by
        whole numbers. ``"degen_gauss"`` (degenerate
        Gaussian) and ``"cond_gauss"`` (conditional Gaussian) test mixed data, continuous and
        categorical columns, by a likelihood ratio. Four more tests detect nonlinear dependence in
        continuous data, at a far higher cost per test: ``"kci"`` is the kernel test; ``"rcot"`` its
        random-feature approximation; ``"cmiknn"`` a nearest-neighbor conditional mutual information
        test; and ``"gpdc"`` Gaussian-process regression with a distance correlation test. ``None``
        picks from the declared column types: ``"chisq"`` when every column is categorical,
        ``"cond_gauss"`` when some are, else ``"fisherz"``. A column is declared categorical by
        ``categorical`` or by a DataFrame's ``category``, ``bool``, or text type, never by its
        values. See Notes. ``None`` picks from the declared column types: ``"chisq"`` when every
        column is categorical, ``"cond_gauss"`` when some are, else ``"fisherz"``. A column is
        declared categorical by ``categorical`` or by a DataFrame's ``category``, ``bool``, or text
        type, never by its values.
        ``metadata["indep_test"]`` names the test the fit used: the one picked from the
        declared column types when this is ``None``.
    collider_order : {"p_value", "column"} or None, default=None
        The order in which colliders are oriented, which decides between two colliders that would
        orient one edge in opposite directions. ``"p_value"`` orients first the collider whose
        endpoints were separated with the largest p-value, so the colliders do not depend on the
        column order. ``"column"`` orients them in column order and keeps the first of two that
        conflict. ``None`` follows ``andrey.config.tie_break``: ``"p_value"`` under ``"rank"``,
        ``"column"`` under ``"position"``. See Notes.
    collider_rule : {"sepsets", "conservative", "majority"}, default="sepsets"
        How PC decides whether ``z`` is a collider ``x -> z <- y`` on a triple ``x - z - y`` whose
        ends are not adjacent. ``"sepsets"`` makes it a collider when ``z`` is not in the
        separating set that removed the edge ``x - y``. ``"conservative"`` [11]_ and
        ``"majority"`` [3]_ test ``x`` and ``y`` again given every subset of ``x``'s neighbors and
        every subset of ``y``'s neighbors, and count the subsets that separate them
        (``p >= alpha``): ``"conservative"`` makes ``z`` a collider when none of them contains it
        and a non-collider when all do; ``"majority"`` when fewer than half contain it and when
        more than half do. Any other triple is ambiguous, and no rule orients through it. Both run
        up to ``2**degree(x) + 2**degree(y)`` extra tests per pair. See Notes.
    categorical : sequence of str or int, optional
        The categorical columns of ``data``, for a mixed-data test: column labels (strings) or
        0-based positions. ``None`` takes a DataFrame's ``category``, ``bool``, and text columns,
        never a column by its values. ``[]`` marks none. A categorical test reads every column as
        categorical, and a continuous test raises ``ValueError`` on a declared column, named here or
        by its type. See *Data types* in the
        :doc:`getting-started guide </docs/guides/getting-started>`.
    seed : int or None, default=None
        ``None``, the default, is the configured seed, ``ANDREY_SEED`` or ``0`` when it is unset,
        not a fresh random one, so a fit repeats. Seed for the random draws of ``indep_test``, which
        only ``"rcot"``, ``"cmiknn"``, and ``"gpdc"`` make; the other tests ignore it. With those
        tests, ``metadata["seed"]`` records the seed used.
    max_depth : int or None, default=None
        The maximum conditioning-set size of every conditional-independence test, at least ``0``:
        ``0`` runs only unconditional tests, and ``None`` sets no limit. A smaller cap runs fewer
        tests and can keep edges that a larger conditioning set would remove. It also caps the
        subsets the conservative and majority rules test.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the pairs
        allowed to be adjacent, and temporal tiers, all honored. The adjacency search starts from
        the allowed pairs and never removes a required pair; the tests can remove any other allowed
        pair. A pair the knowledge makes non-adjacent is tested for a separating set, so colliders
        over it are oriented as usual; when the search finds none within ``max_depth``, no rule
        reads its unshielded triples as colliders or non-colliders. The result directs every edge
        the knowledge decides, then Meek's rules direct the edges that follow, giving the maximally
        oriented PDAG (MPDAG) :class:`~andrey.Knowledge` describes; knowledge that orients the edges
        the search kept into a directed cycle raises ``ValueError``.

    Returns
    -------
    StructureOutput
        Wraps a ``cpdag`` :class:`~andrey.core.GraphStructure`; ``metadata["algorithm"]`` is
        ``"pc"``. Directed marks (``->``) are oriented edges, undirected (``--``) the ones the
        equivalence class leaves open. Under a mixed-data setting, ``metadata["categorical"]`` lists
        the categorical columns it read, by name or 0-based position. With ``collider_rule``
        ``"conservative"`` or ``"majority"``, ``metadata["ambiguous_triples"]`` lists every
        unshielded triple no rule reads as a collider or a non-collider, as ``[x, z, y]``: 0-based
        column positions, the middle node second and ``x < y``, in ascending order.
        ``metadata["conflicts"]`` lists the edges two colliders ask to orient both ways, as
        ``[x, y]`` pairs of 0-based column positions with ``x < y``, in ascending order; Notes say
        which orientation each keeps.
        With a randomized ``indep_test``, ``metadata["seed"]`` is the seed used.

    Raises
    ------
    TypeError
        If ``categorical`` is not a list of column names and positions, or ``knowledge`` is neither
        ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If an argument has the wrong type, lies outside its range, or is not one of its choices; the
        message names the method, the argument, what it accepts, and the value. Also if ``data`` has
        too few samples for a test or a singular correlation matrix; with ``"chisq"`` or ``"gsq"``,
        if ``data`` holds a value that is not a whole number; or, with ``"degen_gauss"`` or
        ``"cond_gauss"``, if ``categorical`` names a column ``data`` lacks, or if a categorical
        column holds a number that is not whole or a missing value. Also if a continuous test meets
        a column declared categorical, by ``categorical`` or by its type. With ``"fisherz"`` or
        ``"student_t"``, also if a column repeats another up to scale and shift, before any test
        runs; one message names every group of copies and says to keep one column per group. Also if
        ``knowledge`` names a column ``data`` lacks or contradicts itself, as
        :class:`~andrey.Knowledge` lists.

    Warns
    -----
    PerformanceWarning
        When a conditioning pass of size two or greater requires at least ten million CI tests;
        each message warns once per process. The warning does not limit the search.
    DataTypeWarning
        When a mixed-data test finds no declared categorical column and so runs as the
        linear-Gaussian form, or when a setting reads columns that look like category codes as
        numbers; each message warns once per process.

    See Also
    --------
    andrey.fci : Allows unmeasured common causes and selection bias.
    andrey.direct_lingam : Orients every edge when the noise is non-Gaussian.

    Notes
    -----
    **When to use it.** PC suits data with no unmeasured common causes. Its default Fisher-Z test
    assumes continuous data, where each variable depends linearly on its causes with roughly
    Gaussian noise; for categorical data, use ``"chisq"`` or ``"gsq"``, for mixed data
    ``"degen_gauss"`` or ``"cond_gauss"``, and for nonlinear effects in continuous data ``"kci"``,
    ``"rcot"``, ``"cmiknn"``, or ``"gpdc"``, at a far higher cost per test.
    It is fastest on sparse graphs, where every variable has few neighbors [5]_.

    **When not to.** When unmeasured common causes are plausible, use :func:`~andrey.fci`. When the
    noise is non-Gaussian and every edge should be oriented, use :func:`~andrey.direct_lingam`.

    **Choosing alpha.** A test removes an edge when it cannot reject independence at ``alpha``. A
    larger ``alpha`` therefore keeps more edges, true and false. On the Sachs data in the
    getting-started guide, raising ``alpha`` from 0.05 to 0.2 found more known links, but a smaller
    share of the links found were known. No value is right in general: it trades missed edges
    against false ones, so a result is best reported with how the graph changes with ``alpha``.

    **How it works.** Edge removal is order-independent (PC-stable) [3]_. At each conditioning-set
    size, every test uses the adjacencies from the start of that size. Edges found independent are
    removed together. A removed pair's separating set combines all conditioning sets that separated
    it. Two colliders conflict when they would orient one edge in opposite directions, so the order
    in which they are oriented can change the result [3]_. A collider is skipped when it would give
    an edge an arrowhead at both ends, so of two conflicting colliders the one oriented first is
    kept. With ``collider_order="p_value"`` (the default), each collider ``x -> y <- z`` ranks by
    the largest p-value among the conditioning sets that separated ``x`` and ``z``, and colliders
    are oriented from the highest rank to the lowest. When two conflicting colliders rank equal to
    nine decimals, the edge they dispute stays undirected, and Meek's rules do not orient it. The
    colliders therefore do not depend on the column order. This adapts the conflict rule of Ramsey
    [10]_, which ranks a collider by separating sets that additional tests choose; Andrey ranks it
    by the tests that removed the edge, so orienting runs no new test. With
    ``collider_order="column"``, colliders are oriented by middle node, then by endpoint pair, in
    column order, so the result can depend on the column order. After the colliders, Meek's rules
    run in rounds: each round collects every orientation they imply on the graph as it stands and
    applies them together, except that an edge two of them would direct both ways stays
    undirected, and so do orientations that would together close a directed cycle. The rounds
    repeat until one orients nothing, so Meek's rules do not depend on the column order. Such
    conflicts arise only when no DAG on the skeleton has exactly the colliders found, as test errors
    can cause; otherwise the rounds give the CPDAG that firing the rules one at a time gives. Under
    ``andrey.config(tie_break="position")``, the rules fire one at a time in column order instead,
    and of two firings that would orient one edge in opposite directions the first decides. The
    :doc:`column-order guide </docs/guides/column-order>` lists what reordering the columns does to
    each method.

    **Collider rules.** A test error can make the separating set that removed an edge disagree with
    other sets that also separate the pair, and ``collider_rule="sepsets"`` then decides a collider
    from that one set. ``"conservative"``, conservative PC [11]_, and ``"majority"``, majority-rule
    PC [3]_, decide each triple ``x - z - y`` on the final skeleton instead. They test ``x`` and
    ``y`` given each subset of ``x``'s neighbors, then each subset of ``y``'s, the empty set
    included, and keep the subsets that separate them; a subset both offer counts twice. Under
    ``"conservative"``, ``z`` is a collider when no kept subset contains it, a non-collider when
    every one does, and ambiguous otherwise. Under ``"majority"``, ``z`` is a collider when fewer
    than half contain it, a non-collider when more than half do, and ambiguous at exactly half. A
    triple whose ends no subset separates is ambiguous under both: the search separated them only
    given sets holding a node whose edge to that end it removed later. An ambiguous triple is never
    oriented as a collider, and Meek's rules do not read it as a non-collider, so edges around it
    can stay undirected. The colliders are then oriented in ``collider_order`` as under
    ``"sepsets"``, so with ``"p_value"`` the result still does not depend on the column order.
    Conservative PC's orientations are correct under adjacency-faithfulness, a weaker assumption
    than PC's [11]_, at the cost of more undirected edges; the majority rule orients more of them
    [3]_.

    **Missing values.** A ``NaN`` in ``data`` marks a missing value. Each test reads only the rows
    where all of its variables are observed (test-wise deletion) [12]_ [13]_, so it keeps rows that
    deleting every incomplete row first would drop; ``metadata["rows_tested"]`` gives the fewest and
    the median rows a test read. When values are missing completely at random, the rows a test keeps
    are a random sample, so each test keeps its level [13]_. When whether a value is missing depends
    on the variables, the rows a test keeps are a selected sample, and the tests can find dependence
    where the population has none and keep extra edges [13]_. A test raises ``ValueError`` when it
    has fewer rows than it needs, ``|S| + 3`` for ``"fisherz"`` and ``"student_t"`` and 2 for the
    other tests, or when a continuous column is constant on its rows. With missing values, the
    skeleton search runs in one process.

    **Limitations.** Measured on simulated linear-Gaussian data from Erdos-Renyi graphs, on 4 cores
    with a cap of 1 h and 28 GB per fit. The largest sizes through which its median errors (SHD)
    stayed below the empty graph's, and through which its fits finished within the caps, were 50 and
    200 variables at mean degree 1 with 2,000 rows; 400 and 1,600 at mean degree 2 with 500, 2,000,
    or 10,000 rows; and 3,200 and 3,200 at mean degree 4 or 8 with 2,000 rows. At 1,600 variables
    (mean degree 2, 10,000 rows), 2 of 5 fits ran past 1 h. The ladders ended after two sizes in a
    row did not beat the empty graph, except at mean degree 4 or 8 with 2,000 rows, where they ended
    at the largest planned size. The median fit took 1,279 s at 3,200 variables (mean degree 8,
    2,000 rows). The numbers below were measured with ``collider_order="column"``. On the
    223-variable ANDES network with 1,000 simulated rows, its median errors passed the empty graph's
    (372 against 338). Its run time grows combinatorially with the largest neighborhood. On
    simulated mixed data (conditional linear Gaussian networks on sparse random graphs, the first
    half of the variables categorical with 2 to 5 categories), from 10 to 40 variables with 2,000
    rows, its median errors stayed below the empty graph's at every size: at 40 variables, 18.5 with
    ``"degen_gauss"`` and 9.5 with ``"cond_gauss"``, against 40. On simulated categorical networks
    with 2 or 3 categories per variable, of the same sizes and rows, they were 14 with ``"chisq"``
    and 16 with ``"gsq"`` at 40 variables. The median fit at 40 variables took 0.38 s with
    ``"degen_gauss"``, 0.42 s with ``"cond_gauss"``, and 0.18 s with either categorical test, on 4
    cores. ``collider_rule="conservative"`` and ``"majority"`` are not benchmarked yet; their tests
    run them on up to 8 variables. Missing values are not benchmarked yet; the tests run them on up
    to 20 variables with up to a fifth of the values missing completely at random. The numbers are a
    reference, not a hard limit: they change with the data generator, the density, the method's
    settings, and the machine.

    **Categorical data.** ``"chisq"`` and ``"gsq"`` take each distinct value of a column as one
    category, so the codes need not start at 0 or be consecutive; a DataFrame's text, ``bool``, and
    ``category`` columns are read the same way. A test compares the counts of ``x`` and ``y`` with
    independence within each configuration of the conditioning set that occurs in the data [2]_.
    The p-value refers the statistic to a chi-square law whose degrees of freedom are its
    mean when ``y`` is shuffled within each configuration: ``(r_x - 1) * (r_y - 1)`` per
    configuration for large counts, with ``r_x`` and ``r_y`` the values present in it [6]_, and
    more for few rows, where the usual degrees of freedom over-reject a true independence; with
    none, the p-value is 1. A table with fewer rows than cells has most cells empty: the p-values
    are then a little too large, ``"gsq"``'s more than ``"chisq"``'s, so prefer ``"chisq"``. Over
    35,808 true independences of random discrete networks, with 3 to 5 categories, conditioning
    sets of 0 to 3 variables, and 500 to 5,000 rows, a test at level 0.05 rejected 4.9%
    (``"chisq"``) and 3.8% (``"gsq"``); the usual degrees of freedom rejected 11.1% and 21.7%.

    **Mixed data.** ``"degen_gauss"`` and ``"cond_gauss"`` test data with continuous and categorical
    columns. Each distinct value of a categorical column is one category, and a DataFrame column of
    text or categories needs no coding. ``"degen_gauss"``, the degenerate Gaussian test [7]_,
    replaces a categorical column of ``k`` categories by ``k - 1`` indicator columns and treats
    every column as Gaussian, so it suits effects that are linear in the continuous variables and
    add up across categories. ``"cond_gauss"``, the conditional Gaussian test [8]_, fits the
    continuous variables in each joint category of the categorical ones with a Gaussian of that
    category's own mean and covariance, so a category can change a variance as well as a mean. Each
    category's covariance is shrunk toward that of all rows by one pseudo-row [9]_, so a joint
    category with few rows takes mostly the covariance of all rows; to tell its own apart, a test
    needs several rows in each joint category of its categorical variables. Each compares the
    likelihoods of the models with and without the dependence, a chi-square statistic on the
    difference in their numbers of parameters. With no categorical column, both are the
    likelihood-ratio form of the partial-correlation test, which agrees with ``"fisherz"`` as the
    rows grow.

    References
    ----------
    .. [1] Spirtes, P., and Glymour, C. (1991). An algorithm for fast recovery of sparse causal
       graphs. Social Science Computer Review 9(1), 62-72.
    .. [2] Spirtes, P., Glymour, C., and Scheines, R. (2000). Causation, Prediction, and Search,
       2nd ed. MIT Press.
    .. [3] Colombo, D., and Maathuis, M. H. (2014). Order-independent constraint-based causal
       structure learning. JMLR 15, 3921-3962.
    .. [4] Meek, C. (1995). Causal inference and causal explanation with background knowledge.
       UAI, 403-410.
    .. [5] Kalisch, M., and Buhlmann, P. (2007). Estimating high-dimensional directed acyclic graphs
       with the PC-algorithm. JMLR 8, 613-636.
    .. [6] Tsamardinos, I., Brown, L. E., and Aliferis, C. F. (2006). The max-min hill-climbing
       Bayesian network structure learning algorithm. Machine Learning 65(1), 31-78.
    .. [7] Andrews, B., Ramsey, J., and Cooper, G. F. (2019). Learning high-dimensional directed
       acyclic graphs with mixed data-types. Proceedings of Machine Learning Research 104, 4-21.
    .. [8] Andrews, B., Ramsey, J., and Cooper, G. F. (2018). Scoring Bayesian networks of mixed
       variables. International Journal of Data Science and Analytics 6(1), 3-18.
    .. [9] Friedman, J. H. (1989). Regularized discriminant analysis. Journal of the American
       Statistical Association 84(405), 165-175.
    .. [10] Ramsey, J. (2016). Improving accuracy and scalability of the PC algorithm by maximizing
       P-value. arXiv:1610.00378, Section 5.
    .. [11] Ramsey, J., Zhang, J., and Spirtes, P. (2006). Adjacency-faithfulness and conservative
       causal inference. UAI, 401-408.
    .. [12] Strobl, E. V., Visweswaran, S., and Spirtes, P. L. (2018). Fast causal inference with
       non-random missingness by test-wise deletion. International Journal of Data Science and
       Analytics 6(1), 47-62.
    .. [13] Tu, R., Zhang, C., Ackermann, P., Mohan, K., Kjellstrom, H., and Zhang, K. (2019).
       Causal discovery in the presence of missing data. Proceedings of Machine Learning Research
       89, 1762-1770.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from andrey import pc
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal(500)
    >>> y = rng.standard_normal(500)
    >>> z = x + y + 0.5 * rng.standard_normal(500)
    >>> pc(pd.DataFrame({"x": x, "y": y, "z": z})).structure.oriented_edges()
    [('x', 'z', 'directed'), ('y', 'z', 'directed')]

    On categorical data, with a chi-square test:

    >>> a = rng.integers(0, 2, 2000)
    >>> b = rng.integers(0, 2, 2000)
    >>> c = np.where(rng.random(2000) < 0.8, a + b, rng.integers(0, 3, 2000))
    >>> data = pd.DataFrame({"a": a, "b": b, "c": c})
    >>> pc(data, indep_test="chisq").structure.oriented_edges()
    [('a', 'c', 'directed'), ('b', 'c', 'directed')]

    On mixed data, with a category column and the degenerate Gaussian test:

    >>> g = rng.integers(0, 3, 2000)
    >>> u = rng.standard_normal(2000)
    >>> v = np.array([-1.0, 0.0, 1.0])[g] + u + 0.5 * rng.standard_normal(2000)
    >>> data = pd.DataFrame({"g": pd.Categorical(g), "u": u, "v": v})
    >>> pc(data, indep_test="degen_gauss").structure.oriented_edges()
    [('g', 'v', 'directed'), ('u', 'v', 'directed')]
    """
    from andrey.constraint.pc import pc as _pc
    from andrey.constraint.pc import resolve_collider_order, resolve_in_column_order

    indep_test, categorical = default_test(data, indep_test, categorical=categorical)
    X, mask = checked_data(
        data,
        categorical,
        "indep_test",
        indep_test,
        method="pc",
        missing=True,
    )
    labels = column_labels(data)
    constraints = resolve(knowledge, X.shape[1], labels, method="pc")
    ambiguous: set[tuple[int, int, int]] = set()
    conflicts: set[tuple[int, int]] = set()
    with copies_named(labels), rows_tested() as rows:
        cpdag = _pc(
            X,
            alpha=alpha,
            indep_test=indep_test,
            collider_order=resolve_collider_order(collider_order),
            collider_rule=collider_rule,
            categorical=mask,
            max_depth=max_depth,
            constraints=constraints,
            in_column_order=resolve_in_column_order(),
            seed=seed,
            _ambiguous=ambiguous,
            _conflicts=conflicts,
        )
    metadata = {
        "conflicts": [list(edge) for edge in sorted(conflicts)],
        **categorical_metadata(mask, data),
        **_ambiguous_metadata(collider_rule, ambiguous),
        **rows_metadata(rows),
    }
    return structure_output(cpdag, labels=labels, metadata=metadata)


def _ambiguous_metadata(
    collider_rule: str, ambiguous: set[tuple[int, int, int]]
) -> dict[str, list[list[int]]]:
    """Return the ambiguous triples as metadata, or ``{}`` under ``collider_rule="sepsets"``.

    The triples become ``{"ambiguous_triples": [[a, b, c], ...]}``, in ascending order.
    """
    if collider_rule == "sepsets":
        return {}
    return {"ambiguous_triples": [list(triple) for triple in sorted(ambiguous)]}


@method_spec("fci")
def fci(
    data: npt.ArrayLike,
    *,
    alpha: float = 0.05,
    indep_test: str | None = None,
    collider_rule: str = "sepsets",
    categorical: Sequence[str | int] | None = None,
    possible_dsep: str | None = None,
    seed: int | None = None,
    max_depth: int | None = None,
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn a PAG from observational data with FCI, sound under latent confounders.

    FCI extends PC to handle unmeasured common causes (latent confounders) and selection bias
    [1]_ [2]_. It uses conditional-independence tests to remove edges, then orients their endpoints.
    The result is a PAG, whose circles (``o``) mark endpoints the data cannot determine. Use it when
    unmeasured confounding is plausible.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``.
    alpha : float, default=0.05
        Significance level of the conditional-independence test. Smaller values give sparser graphs.
    indep_test : str or None, default=None
        Conditional-independence test. ``"fisherz"`` and ``"student_t"`` test partial correlation,
        for continuous data: ``"fisherz"`` applies Fisher's z-transform and the normal distribution;
        ``"student_t"`` applies the Student-t distribution, which is exact for Gaussian data at any
        sample size. ``"normal_scores"`` applies ``"fisherz"`` to the normal scores of each column,
        for data whose columns are monotone transformations of Gaussian variables: its p-values do
        not change under a strictly increasing transformation of a column. ``"chisq"`` (Pearson's
        chi-square) and ``"gsq"`` (G-square, the likelihood ratio) test categorical data, coded by
        whole numbers. ``"degen_gauss"`` (degenerate
        Gaussian) and ``"cond_gauss"`` (conditional Gaussian) test mixed data, continuous and
        categorical columns. Four more tests detect nonlinear dependence in continuous data, at a
        far higher cost per test: ``"kci"`` is the kernel test; ``"rcot"`` its random-feature
        approximation; ``"cmiknn"`` a nearest-neighbor conditional mutual information test; and
        ``"gpdc"`` Gaussian-process regression with a distance correlation test. ``None`` picks from
        the declared column types: ``"chisq"`` when every column is categorical, ``"cond_gauss"``
        when some are, else ``"fisherz"``. A column is declared categorical by ``categorical`` or by
        a DataFrame's ``category``, ``bool``, or text type, never by its values. See Notes. ``None``
        picks from the declared column types: ``"chisq"`` when every column is categorical,
        ``"cond_gauss"`` when some are, else ``"fisherz"``. A column is declared categorical by
        ``categorical`` or by a DataFrame's ``category``, ``bool``, or text type, never by its
        values.
        ``metadata["indep_test"]`` names the test the fit used: the one picked from the
        declared column types when this is ``None``.
    collider_rule : {"sepsets", "conservative", "majority"}, default="sepsets"
        How FCI decides whether ``z`` is a collider on a triple ``x - z - y`` where ``x`` and ``y``
        are not adjacent. ``"sepsets"`` makes it a collider when ``z`` is not in the separating set
        that removed the edge ``x - y``. ``"conservative"`` [4]_ [5]_ and ``"majority"`` [4]_ test
        ``x`` and ``y`` again given every subset of ``x``'s neighbors and every subset of ``y``'s
        neighbors, and count the subsets that separate them (``p >= alpha``): ``"conservative"``
        makes ``z`` a collider when none of them contains it and a non-collider when all do;
        ``"majority"`` when fewer than half contain it and when more than half do. Any other
        triple is ambiguous, and no rule orients through it. Both run up to
        ``2**degree(x) + 2**degree(y)`` extra tests per pair. See Notes.
    categorical : sequence of str or int, optional
        The categorical columns of ``data``, for a mixed-data test: column labels (strings) or
        0-based positions. ``None`` takes a DataFrame's ``category``, ``bool``, and text columns,
        never a column by its values. ``[]`` marks none. A categorical test reads every column as
        categorical, and a continuous test raises ``ValueError`` on a declared column, named here or
        by its type. See *Data types* in the
        :doc:`getting-started guide </docs/guides/getting-started>`.
    possible_dsep : {"order_free", "fast"} or None, default=None
        The order of FCI's possible-d-separation step [2]_ [3]_, which removes edges the first step
        kept, one at a time, so each removal changes what the step tests after it.
        ``"order_free"`` visits the pairs and nodes by a rank computed from each column's own data,
        so the result does not depend on the column order unless two columns are identical.
        ``"fast"`` visits them in column order. ``None`` follows ``andrey.config.tie_break``:
        ``"order_free"`` under ``"rank"``, ``"fast"`` under ``"position"``. See Notes and
        Limitations.
    seed : int or None, default=None
        ``None``, the default, is the configured seed, ``ANDREY_SEED`` or ``0`` when it is unset,
        not a fresh random one, so a fit repeats. Seed for the random draws of ``indep_test``, which
        only ``"rcot"``, ``"cmiknn"``, and ``"gpdc"`` make; the other tests ignore it. With those
        tests, ``metadata["seed"]`` records the seed used.
    max_depth : int or None, default=None
        The maximum conditioning-set size of every conditional-independence test, at least ``0``:
        ``0`` runs only unconditional tests, and ``None`` sets no limit. A smaller cap runs fewer
        tests and can keep edges that a larger conditioning set would remove. It caps the tests of
        the adjacency search, the possible-d-separation step, the conservative and majority
        rules, and the discriminating-path rule.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the pairs
        allowed to be adjacent, and temporal tiers, all honored. The adjacency search starts from
        the allowed pairs and never removes a required pair; the tests can remove any other allowed
        pair. A pair the knowledge makes non-adjacent is tested for a separating set, so colliders
        over it are oriented as usual; when the search finds none within ``max_depth``, no rule
        reads its unshielded triples as colliders or non-colliders. In the PAG, a forbidden
        ``a -> b`` on an adjacent pair puts an arrowhead at ``a`` and can leave a circle at ``b``,
        and a required one is ``a -> b``; knowledge that puts an arrowhead at ``a`` on an edge
        ``a - b`` while its required edges lead from ``a`` to ``b`` raises ``ValueError``.

    Returns
    -------
    StructureOutput
        Wraps a ``pag`` :class:`~andrey.core.GraphStructure`; ``metadata["algorithm"]`` is
        ``"fci"``. Endpoints carry arrowheads, tails, or circle marks (``o``); a bidirected edge
        (``<->``) marks a probable latent common cause. Under a mixed-data setting,
        ``metadata["categorical"]`` lists the categorical columns it read, by name or 0-based
        position. With ``collider_rule`` ``"conservative"`` or ``"majority"``,
        ``metadata["ambiguous_triples"]`` lists every unshielded triple no rule reads as a collider
        or a non-collider, as ``[x, z, y]``: 0-based column positions, the middle node second and
        ``x < y``, in ascending order.
        With a randomized ``indep_test``, ``metadata["seed"]`` is the seed used.

    Raises
    ------
    TypeError
        If ``categorical`` is not a list of column names and positions, or ``knowledge`` is neither
        ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If an argument has the wrong type, lies outside its range, or is not one of its choices; the
        message names the method, the argument, what it accepts, and the value. Also if ``data`` has
        too few samples for a test or a singular correlation matrix, or if ``data`` or
        ``categorical`` does not suit the test, as for :func:`~andrey.pc`. With ``"fisherz"`` or
        ``"student_t"``, also if a column repeats another up to scale and shift, before any test
        runs; one message names every group of copies and says to keep one column per group. Also if
        ``knowledge`` names a column ``data`` lacks or contradicts itself, as
        :class:`~andrey.Knowledge` lists.

    Warns
    -----
    DataTypeWarning
        When a mixed-data test finds no declared categorical column and so runs as the
        linear-Gaussian form, or when a setting reads columns that look like category codes as
        numbers; each message warns once per process.

    See Also
    --------
    andrey.pc : Orients more of the graph, with fewer tests, when every common cause is measured.

    Notes
    -----
    **When to use it.** FCI suits data that may be affected by unmeasured common causes or sample
    selection. Its default Fisher-Z test assumes continuous data with linear effects and roughly
    Gaussian noise; for categorical data, use ``"chisq"`` or ``"gsq"``, and for mixed data
    ``"degen_gauss"`` or ``"cond_gauss"``, which read the data as :func:`~andrey.pc` describes. Its
    PAG claims less than PC's CPDAG: an arrowhead at a node rules out that node causing the node at
    the edge's other end, and a circle leaves the mark open.

    **When not to.** When every common cause is measured, :func:`~andrey.pc` orients more of the
    graph with fewer tests. FCI's possible-d-separation step runs tests PC does not.

    **Choosing alpha.** A larger ``alpha`` keeps more edges, true and false.

    **How it works.** FCI runs in three steps. First, it removes edges by conditional-independence
    tests, as PC does, and orients colliders from the separating sets. Next, it tests each remaining
    pair given subsets of its possible-d-separation set: nodes that can separate the pair when some
    causes are unmeasured. It removes pairs found independent and orients colliders again. Last, it
    applies Zhang's orientation rules R1-R10 [3]_ until no mark changes. An endpoint no rule
    resolves stays a circle.

    The rules run in rounds: each round collects the marks every rule implies on the graph as it
    stands and sets them together, except that an endpoint two of them would mark one as an
    arrowhead and one as a tail stays a circle. The rounds repeat until no mark changes. Rule R4
    decides an endpoint by every shortest discriminating path, so paths whose tests disagree leave
    it a circle. The orientation rules therefore do not depend on the column order, and with
    ``possible_dsep="order_free"`` neither does the result, unless two columns are identical. With
    ``"fast"`` the result can depend on the column order through the possible-d-separation step. A
    fixed column order gives reproducible results. Under ``andrey.config(tie_break="position")``,
    the rules instead fire one at a time in column order, and of two firings that conflict the
    first decides. ``possible_dsep`` then defaults to ``"fast"``. The
    :doc:`column-order guide </docs/guides/column-order>` lists what reordering the columns does to
    each method.

    Rule R5 turns a cycle of circle-circle edges into undirected edges only if the whole cycle,
    including the closing edge, is uncovered, that is, no two nodes two steps apart are adjacent.
    It considers every circle path, up to a fixed search bound, so a covered route through a node
    never hides an uncovered route through the same node. Rule R8 checks each pair of endpoints in
    both orders, so whether it fires does not depend on the node labels.

    **Possible-d-separation set.** FCI builds each possible-d-separation set [2]_ [3]_ with a search
    that walks out from one endpoint of the pair and keeps a node when a node it was reached from
    has a possibly directed path, one with no arrowhead pointing back along it, to the endpoint or
    to the node. FCI checks a node when the search reaches it, against the nodes it was reached
    from so far, so the set depends on the order the search visits nodes. The set can be smaller
    than the published definition of the set [2]_, so FCI can keep an edge that the published set
    would remove, even with exact tests. FCI keeps this search because the published set grows
    quickly with the graph, the search's set is the one FCI has been measured with, and on
    simulated data the published set did not give more accurate PAGs. FCI tests every subset of the
    set, except the empty set and the subsets of either endpoint's neighbors, which the first step
    already tested. It visits the pairs one at a time, removes a pair as soon as a subset separates
    it, and records that subset, so each removal changes the sets of the pairs after it. With
    ``possible_dsep="order_free"``, the pairs, the nodes of each search, and the subsets follow a
    tie rank computed from each column's own data, as for :func:`~andrey.hc`, which moves with its
    column when the columns are reordered; with ``"fast"``, they follow the column order.

    **Collider rules.** ``collider_rule="conservative"``, conservative FCI [4]_ after the idea of
    conservative PC [5]_, and ``"majority"``, majority-rule FCI [4]_, decide each triple
    ``x - z - y`` after the possible-d-separation step, on the final skeleton. They test ``x`` and
    ``y`` given each subset of ``x``'s neighbors, then each subset of ``y``'s, the empty set
    included, and keep the subsets that separate them; a subset both offer counts twice. Under
    ``"conservative"``, ``z`` is a collider when no kept subset contains it, a non-collider when
    every one does, and ambiguous otherwise. Under ``"majority"``, ``z`` is a collider when fewer
    than half contain it, a non-collider when more than half do, and ambiguous at exactly half. A
    triple whose ends no subset separates is a collider under both: no kept subset contains ``z``,
    and the separating set that removed the edge, from a possible-d-separation set or an earlier
    neighborhood, does not vote. With exact tests this can be wrong where only nodes beyond both
    ends' neighbors separate them and ``z`` is a non-collider, so on such graphs the voting rules
    can orient a collider that ``"sepsets"`` does not. An ambiguous triple is never oriented as a
    collider, and rules R1-R10 do not read it as a non-collider, so endpoints around it can stay
    circles. A vote reads only the skeleton and the tests, so the rules keep the result independent
    of the column order wherever ``"sepsets"`` does.

    **Separating sets.** When several conditioning sets of the same size make a pair independent
    in the first step, FCI records their union as the pair's separating set. FCI keeps the union
    because it does not depend on the order of the tests, and on simulated data it gave more
    accurate PAGs than the first set found.

    **Missing values.** A ``NaN`` in ``data`` marks a missing value. Each test reads only the rows
    where all of its variables are observed (test-wise deletion) [6]_ [7]_, so it keeps rows that
    deleting every incomplete row first would drop; ``metadata["rows_tested"]`` gives the fewest and
    the median rows a test read. When values are missing completely at random, the rows a test keeps
    are a random sample, so each test keeps its level [7]_. When whether a value is missing depends
    on the variables, the rows a test keeps are a selected sample; FCI allows for selection, and
    stays sound when no missingness mechanism causally affects another [6]_. A test raises
    ``ValueError`` when it has fewer rows than it needs, ``|S| + 3`` for ``"fisherz"`` and
    ``"student_t"`` and 2 for the other tests, or when a continuous column is constant on its rows.
    With missing values, the skeleton search runs in one process.

    **Limitations.** Measured on simulated linear-Gaussian data from Erdos-Renyi graphs with no
    hidden variable, on 4 cores with a cap of 1 h and 28 GB per fit. The largest sizes through which
    its median errors (SHD against the PAG of the observed variables) stayed below the empty
    graph's, and through which its fits finished within the caps, were 200 and 800 variables at mean
    degree 2 with 10 or 50 rows per variable; and 1,600 and 1,600 at mean degree 4 with 10 or 50
    rows per variable. The ladders ended after two sizes in a row did not beat the empty graph,
    except at mean degree 4 with 10 or 50 rows per variable, where they ended when three fits of the
    next size ran past 1 h. The median fit took 116 s at 1,600 variables (mean degree 4, 50 rows per
    variable). Measured with ``possible_dsep="fast"`` on simulated linear-Gaussian data from random
    graphs with mean degree 2 or 4 and one variable in ten or one in five hidden, from 20 to 800
    variables counting the hidden ones, with 10 or 50 rows per variable, on 4 cores. Its median
    endpoint errors, each mismatched tail, arrowhead, or circle, stayed below the empty graph's at
    every size with mean degree 4, and up to 400 variables with mean degree 2, where they passed it
    at 800 (2,582 against 1,738 with one variable in ten hidden and ten rows per variable). With
    those settings the median fit took 0.0064 s at 20 variables and 114 s at 800; with one variable
    in five hidden and 50 rows per variable it took 78 s at 400 and 4,829 s at 800, where two of
    five fits ran past 7,200 s. The default, ``"order_free"``, took 0.99 to 1.05 times as long as
    ``"fast"`` up to 200 variables and 0.78 to 1.42 times at 400 and 800, and its median endpoint
    errors were within 2% of ``"fast"``'s. On some graphs from 400 variables its order needs more
    tests than the column order: on one with 400 variables, mean degree 4, one variable in five
    hidden, and 50 rows per variable, it took 34 s against 23 to 29 s for ``"fast"`` over five
    column orders. ``possible_dsep="fast"`` gives the column-order mode. On the 223-variable ANDES
    network with 1,000 simulated rows, its median endpoint errors passed the empty graph's (696
    against 676). Rule R5 stops searching a pair after a fixed number of partial paths and leaves
    its circles, so on graphs with many circle-circle edges it can miss an undirected cycle. With
    the mixed-data tests, it is experimental and not benchmarked yet; its tests run it on 5
    variables with 3,000 rows. ``collider_rule="conservative"`` and ``"majority"`` are not
    benchmarked yet; their tests run them on up to 8 variables. The numbers are a reference, not a
    hard limit: they change with the data generator, the density, the method's settings, and the
    machine. Missing values are not benchmarked yet; the tests run them on up to 20 variables with
    up to a fifth of the values missing completely at random.

    References
    ----------
    .. [1] Spirtes, P., Meek, C., and Richardson, T. (1995). Causal inference in the presence of
       latent variables and selection bias. UAI, 499-506.
    .. [2] Spirtes, P., Glymour, C., and Scheines, R. (2000). Causation, Prediction, and Search,
       2nd ed. MIT Press.
    .. [3] Zhang, J. (2008). On the completeness of orientation rules for causal discovery in the
       presence of latent confounders and selection bias. Artificial Intelligence 172(16-17),
       1873-1896.
    .. [4] Colombo, D., and Maathuis, M. H. (2014). Order-independent constraint-based causal
       structure learning. JMLR 15, 3921-3962.
    .. [5] Ramsey, J., Zhang, J., and Spirtes, P. (2006). Adjacency-faithfulness and conservative
       causal inference. UAI, 401-408.
    .. [6] Strobl, E. V., Visweswaran, S., and Spirtes, P. L. (2018). Fast causal inference with
       non-random missingness by test-wise deletion. International Journal of Data Science and
       Analytics 6(1), 47-62.
    .. [7] Tu, R., Zhang, C., Ackermann, P., Mohan, K., Kjellstrom, H., and Zhang, K. (2019). Causal
       discovery in the presence of missing data. Proceedings of Machine Learning Research 89,
       1762-1770.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import fci
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal(500)
    >>> y = x + 0.1 * rng.standard_normal(500)
    >>> fci(np.column_stack([x, y])).structure.kind
    'pag'
    """
    from andrey.constraint.fci import fci as _fci
    from andrey.constraint.pc import resolve_in_column_order, resolve_possible_dsep

    indep_test, categorical = default_test(data, indep_test, categorical=categorical)
    X, mask = checked_data(
        data,
        categorical,
        "indep_test",
        indep_test,
        method="fci",
        missing=True,
    )
    labels = column_labels(data)
    constraints = resolve(knowledge, X.shape[1], labels, method="fci")
    ambiguous: set[tuple[int, int, int]] = set()
    with copies_named(labels), rows_tested() as rows:
        pag = _fci(
            X,
            alpha=alpha,
            indep_test=indep_test,
            collider_rule=collider_rule,
            categorical=mask,
            possible_dsep=resolve_possible_dsep(possible_dsep),
            max_depth=max_depth,
            constraints=constraints,
            in_column_order=resolve_in_column_order(),
            seed=seed,
            _ambiguous=ambiguous,
        )
    metadata = {
        **categorical_metadata(mask, data),
        **_ambiguous_metadata(collider_rule, ambiguous),
        **rows_metadata(rows),
    }
    return structure_output(pag, labels=labels, metadata=metadata)


@method_spec("gfci")
def gfci(
    data: npt.ArrayLike,
    *,
    score_func: str | None = None,
    penalty_discount: float = 1.0,
    ebic_gamma: float = 0.0,
    equivalent_sample_size: float = 1.0,
    indep_test: str | None = None,
    alpha: float = 0.05,
    collider_rule: str = "sepsets",
    categorical: Sequence[str | int] | None = None,
    possible_dsep: str | None = None,
    seed: int | None = None,
    max_depth: int | None = None,
    max_parents: int | None = None,
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn a PAG from observational data with GFCI, a score-then-FCI hybrid.

    GFCI [1]_ starts from the adjacencies of a score-based GES search, removes those that
    conditional-independence tests reject, and orients the endpoints with FCI's rules to account
    for possible latent confounding. Its scored skeleton is usually more accurate than FCI's
    test-only one.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``.
    score_func : str or None, default=None
        Local score driving the GES skeleton phase, as for :func:`~andrey.ges`: ``"bic"``, the
        linear-Gaussian BIC; ``"bdeu"`` or ``"discrete_bic"`` for categorical data, coded by whole
        numbers; or ``"degen_gauss"`` or ``"cond_gauss"``, the degenerate or conditional Gaussian
        BIC for mixed data. ``None`` picks from the declared column types, as for
        :func:`~andrey.ges`.
        ``metadata["score_func"]`` names the score the fit used: the one picked from the
        declared column types when this is ``None``.
    penalty_discount : float, default=1.0
        Multiplier on the BIC penalty of ``log(n_samples)`` per parameter in the GES phase. Larger
        values give sparser graphs; ``1.0`` with ``ebic_gamma=0.0`` is the standard BIC. Must be
        finite and non-negative. BDeu ignores it.
    ebic_gamma : float, default=0.0
        Weight on the extended-BIC term of the linear-Gaussian BIC: each variable's score adds
        ``2 * ebic_gamma * log C(n_variables - 1, n_parents)``, which grows with the number of
        parent sets a variable can choose from. Larger values give sparser graphs; ``0.0`` is the
        standard BIC. Must be finite and non-negative. The other scores ignore it. See Notes.
    equivalent_sample_size : float, default=1.0
        The equivalent sample size of the BDeu score: how many rows the uniform prior over each
        variable's conditional distributions is worth. A larger value weighs the prior more
        against the data, and the graph can change with it. Must be finite and positive. Only
        ``"bdeu"`` uses it: passing it with another score raises ``ValueError``.
    indep_test : str or None, default=None
        Conditional-independence test for the FCI phase. ``"fisherz"`` (Fisher's z) and
        ``"student_t"`` (Student-t) test partial correlation, for continuous data, and
        ``"normal_scores"`` does so on each column's normal scores. ``"chisq"``
        (Pearson's chi-square) and ``"gsq"`` (G-square, the likelihood ratio) test categorical data,
        coded by whole numbers, and ``"degen_gauss"`` and ``"cond_gauss"`` mixed data. ``"kci"``,
        ``"rcot"``, ``"cmiknn"``, and ``"gpdc"`` also detect nonlinear dependence in continuous
        data, at a far higher cost per test. Each as for :func:`~andrey.pc`. ``None`` picks from the
        declared column types: ``"chisq"`` when every column is categorical, ``"cond_gauss"`` when
        some are, else ``"fisherz"``. A column is declared categorical by ``categorical`` or by a
        DataFrame's ``category``, ``bool``, or text type, never by its values.
        ``metadata["indep_test"]`` names the test the fit used: the one picked from the
        declared column types when this is ``None``.
    alpha : float, default=0.05
        Significance level of the conditional-independence test. Smaller values give sparser graphs.
    collider_rule : {"sepsets", "conservative", "majority"}, default="sepsets"
        How GFCI decides whether ``z`` is a collider on a triple ``x - z - y`` where the tests
        removed the GES edge between ``x`` and ``y``; on the other triples whose ends are not
        adjacent, the GES CPDAG decides (see Notes). ``"sepsets"`` makes it a collider when ``z``
        is not in the recorded separating set of ``x`` and ``y``. ``"conservative"`` [3]_ [5]_ and
        ``"majority"`` [3]_ test ``x`` and ``y`` given every subset of ``x``'s neighbors and every
        subset of ``y``'s neighbors and count the subsets that separate them (``p >= alpha``), as
        for :func:`~andrey.fci`: ``"conservative"`` makes ``z`` a collider when none of them
        contains it and a non-collider when all do; ``"majority"`` when fewer than half contain it
        and when more than half do. Any other triple is ambiguous, and no rule orients through it.
        Both run ``2**degree(x) + 2**degree(y)`` tests per pair.
    categorical : sequence of str or int, optional
        The categorical columns of ``data``, for the mixed-data score and test: DataFrame column
        names or 0-based positions. ``None`` takes a DataFrame's ``category``, ``bool``, and text
        columns, never a column by its values. ``[]`` marks none. A categorical setting reads every
        column as categorical, and a continuous setting raises ``ValueError`` on a declared column,
        named here or by its type. See *Data types* in the
        :doc:`getting-started guide </docs/guides/getting-started>`.
    possible_dsep : {"order_free", "fast"} or None, default=None
        The order of GFCI's possible-d-separation step [2]_ [4]_, which removes edges the adjacency
        search kept, one at a time, so each removal changes what the step tests after it.
        ``"order_free"`` visits the pairs and nodes by a rank computed from each column's own data,
        so the result does not depend on the column order unless two columns are identical.
        ``"fast"`` visits them in column order. ``None`` follows ``andrey.config.tie_break``:
        ``"order_free"`` under ``"rank"``, ``"fast"`` under ``"position"``. See Notes and
        Limitations.
    seed : int or None, default=None
        ``None``, the default, is the configured seed, ``ANDREY_SEED`` or ``0`` when it is unset,
        not a fresh random one, so a fit repeats. Seed for the random draws of ``indep_test``, which
        only ``"rcot"``, ``"cmiknn"``, and ``"gpdc"`` make; the other tests ignore it. With those
        tests, ``metadata["seed"]`` records the seed used.
    max_depth : int or None, default=None
        The maximum conditioning-set size of every conditional-independence test, at least ``0``:
        ``0`` runs only unconditional tests, and ``None`` sets no limit. A smaller cap runs fewer
        tests and can keep edges that a larger conditioning set would remove. It caps the tests of
        the adjacency search, the separating sets for the pairs GES leaves non-adjacent, the
        possible-d-separation step, the conservative and majority rules, and the discriminating-path
        rule, but not the GES phase.
    max_parents : int or None, default=None
        A cap on every node's parents in the GES phase, at least ``0``: no DAG the GES CPDAG
        represents gives a node more than ``max_parents`` parents, as :func:`~andrey.ges` applies
        it. The test phase starts from that CPDAG's adjacencies and removes edges, so a node of the
        PAG keeps at most its CPDAG neighbors; the cap does not bound the arrowheads the
        orientation rules put at a node. ``None`` sets no cap; the GES phase then keeps its gate of
        ``n_variables / 2``.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the pairs
        allowed to be adjacent, and temporal tiers, all honored. The GES phase honors it as
        :func:`~andrey.ges` does, and the test phase starts from the adjacencies GES found and never
        removes a required pair. In the PAG, a forbidden ``a -> b`` on an adjacent pair puts an
        arrowhead at ``a`` and can leave a circle at ``b``, and a required one is ``a -> b``.

    Returns
    -------
    StructureOutput
        Wraps a ``pag`` :class:`~andrey.core.GraphStructure`; ``metadata["algorithm"]`` is
        ``"gfci"``. Endpoints carry arrowheads, tails, or circle marks (``o``); a bidirected edge
        (``<->``) marks a probable latent common cause. Under a mixed-data setting,
        ``metadata["categorical"]`` lists the categorical columns it read, by name or 0-based
        position. With ``collider_rule`` ``"conservative"`` or ``"majority"``,
        ``metadata["ambiguous_triples"]`` lists the triples the rule left ambiguous, as
        ``[x, z, y]``: 0-based column positions, the middle node second and ``x < y``, in
        ascending order.
        With a randomized ``indep_test``, ``metadata["seed"]`` is the seed used.

    Raises
    ------
    TypeError
        If ``categorical`` is not a list of column names and positions, or ``knowledge`` is neither
        ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If an argument has the wrong type, lies outside its range, or is not one of its choices; the
        message names the method, the argument, what it accepts, and the value. Also if
        ``score_func`` and ``indep_test`` read different data (continuous, categorical, or mixed),
        or if ``data`` or ``categorical`` does not suit them, as for :func:`~andrey.pc`. With the
        BIC score, also if a column repeats another up to scale and shift given more than two rows
        or, with more rows than columns, is a linear function of other columns. One message names
        every group of copies and says to keep one column per group. Also if a given
        ``max_parents`` is below the parents some graph of the required edges of ``knowledge``
        gives a node, or if ``knowledge`` names a column ``data`` lacks or contradicts itself, as
        :class:`~andrey.Knowledge` lists.

    Warns
    -----
    DataTypeWarning
        When the mixed-data score or test finds no declared categorical column and so runs as the
        linear-Gaussian form, or when a setting reads columns that look like category codes as
        numbers; each message warns once per process.

    Notes
    -----
    GFCI first runs :func:`~andrey.ges` and keeps its CPDAG. FCI's adjacency search then starts
    from the CPDAG's adjacencies instead of a complete graph: it removes an edge ``x - y`` at the
    smallest size at which a subset of ``x``'s or ``y``'s neighbors makes the two independent, and
    records the union of those subsets as their separating set. On a triple ``x - z - y`` whose
    ends are not adjacent, ``z`` is a collider when the CPDAG has ``x -> z <- y``; when the CPDAG
    had the edge ``x - y``, the separating set decides, as ``collider_rule`` describes. FCI's
    possible-d-separation step can remove more edges, after which the colliders are decided again.
    No step adds an edge, so the PAG's adjacencies are a subset of the GES CPDAG's. Zhang's rules
    R1-R10 [2]_ then orient the endpoints, in rounds as in :func:`~andrey.fci`; an endpoint no rule
    resolves stays a circle. A rule that needs the separating set of a pair GES left non-adjacent
    uses the union of the smallest subsets of the two endpoints' CPDAG neighbors that make the pair
    independent, which does not depend on the column order. Under
    ``andrey.config(tie_break="position")``, the rules fire one at a time in column order, such a
    pair's separating set is the first subset found in column order, and ``possible_dsep`` defaults
    to ``"fast"``.

    With ``max_parents``, the GES phase takes no move that gives a node more parents than the cap,
    as :func:`~andrey.ges` does; the published GFCI [1]_ has no such cap. The test phase only
    removes edges, but its orientation rules can put more arrowheads at a node than the cap.

    Rule R5 turns a cycle of circle-circle edges into undirected edges only when the whole cycle,
    closing edge included, is uncovered, that is, no two nodes two steps apart on it are adjacent.
    It considers every circle path, up to a fixed search bound, and rule R8 checks each pair of
    endpoints in both orders. GFCI shares FCI's possible-d-separation step [2]_ [4]_, in the order
    ``possible_dsep`` selects, and its separating sets; the Notes of :func:`~andrey.fci` describe
    both and the reasons for keeping them.

    **Choosing the penalty.** The GES phase scores a variable with the linear-Gaussian BIC, which
    charges ``penalty_discount * log(n_samples)`` per parent, plus the extended-BIC term ``2 *
    ebic_gamma * log C(n_variables - 1, n_parents)`` [6]_, where ``C(n_variables - 1, n_parents)``
    counts the parent sets of that size. The term grows with the number of variables, so a larger
    ``ebic_gamma`` gives a sparser GES graph, with fewer false adjacencies and more missed ones; the
    tests then start from fewer adjacencies and never add one. ``ebic_gamma=0.0`` is the standard
    BIC, GFCI's default; :func:`~andrey.ges` and most other score-based searches default to ``1.0``.
    On simulated linear-Gaussian data with 20 to 400 variables and 200 to 4,000 rows,
    ``ebic_gamma=1.0`` lowered GFCI's mean errors (SHD against the true graph) per true edge by
    0.064 (95% interval 0.055 to 0.074), against 0.78 for GES. In 11 of 65 settings of graph,
    variables, and rows it made more errors on average than ``0.0``, and in three of them the
    difference's 95% interval lies above zero: 300.6 against 291.0 with 200 variables, mean degree
    4, and 2,000 rows (400 true edges); 149.2 against 140.6 with 100 variables, mean degree 4,
    ``scale="unitless"`` data, and 1,000 rows (200 true edges); and 38.4 against 36.2 with 20
    variables, mean degree 4, ``scale="unitless"`` data, and 200 rows (40 true edges). Across the
    295 datasets with both fits, it made more errors on 56, fewer on 198, and the same on 41. Its
    mean errors stayed below the empty graph's in each of the 65 settings, where the standard BIC's
    exceeded it in 16. Pass ``ebic_gamma=1.0`` to use the extended BIC.

    **Missing values.** A ``NaN`` in ``data`` raises ``ValueError``: its GES phase scores every
    variable over the same rows, and score terms over different rows do not add up to the score of
    one dataset. Drop the incomplete rows first, which suits values missing completely at random, or
    use :func:`~andrey.fci`, whose tests delete rows per test.

    **Limitations.** Measured on simulated linear-Gaussian data from Erdos-Renyi graphs with no
    hidden variable, on 4 cores with a cap of 1 h and 28 GB per fit. The largest sizes through which
    its median errors (SHD against the PAG of the observed variables) stayed below the empty
    graph's, and through which its fits finished within the caps, were 1,600 and 1,600 variables at
    mean degree 2 or 4 with 10 rows per variable; and 3,200 and 3,200 at mean degree 2 or 4 with 50
    rows per variable. The ladders ended when three fits of the next size ran past 1 h, except at
    mean degree 2 or 4 with 50 rows per variable, where they ended at the largest planned size. The
    median fit took 1,480 s at 3,200 variables (mean degree 4, 50 rows per variable). Measured with
    ``possible_dsep="fast"`` on simulated linear-Gaussian data from random graphs with mean degree 2
    or 4 and one variable in ten or one in five hidden, from 20 to 800 variables counting the hidden
    ones, with 10 or 50 rows per variable, on 4 cores. Its median endpoint errors, each mismatched
    tail, arrowhead, or circle, stayed below the empty graph's wherever its fits finished (197
    against 444 at 200 variables with mean degree 2, one variable in ten hidden, and ten rows per
    variable). With those settings the median fit took 0.53 s at 100 variables, 2.6 s at 200, and
    132 s at 800. Every fit at 800 variables finished within 7,200 s in four of the eight settings;
    with mean degree 4, one variable in five hidden, and 50 rows per variable, none did, and three
    of five fits at 400 ran past 3,600 s. The default, ``"order_free"``, took 1.00 to 1.02 times as
    long as ``"fast"`` up to 200 variables and 0.87 to 1.03 times at 400 and 800, and its median
    endpoint errors were within 2% of ``"fast"``'s. On some graphs from 400 variables its order
    needs more tests than the column order: on one with 400 variables, mean degree 2, one variable
    in five hidden, and 50 rows per variable, it took 407 s against 120 to 336 s for ``"fast"`` over
    five column orders. ``possible_dsep="fast"`` gives the column-order mode. It runs a full
    :func:`~andrey.ges` search before any test, so it takes at least as long as GES on the same
    data. Its BIC score and Fisher-Z tests assume linear relations with Gaussian noise. With fewer
    rows than columns the BIC score is unreliable. A parent set of ``n_samples - 1`` or more
    variables fits its child exactly, so the GES search never selects one, nor any parent set that
    leaves its child a residual variance of at most ``1e-12`` times its variance. Rule R5 has the
    search bound described under :func:`~andrey.fci`. With the mixed-data score and test, it is
    experimental and not benchmarked yet; its tests run it on 5 variables with 3,000 rows.
    ``collider_rule="conservative"`` and ``"majority"`` are not benchmarked yet; their tests run
    them on up to 8 variables. The numbers are a reference, not a hard limit: they change with the
    data generator, the density, the method's settings, and the machine.

    References
    ----------
    .. [1] Ogarrio, J. M., Spirtes, P., and Ramsey, J. (2016). A hybrid causal search algorithm
       for latent variable models. PGM, PMLR 52, 368-379.
    .. [2] Zhang, J. (2008). On the completeness of orientation rules for causal discovery in the
       presence of latent confounders and selection bias. Artificial Intelligence 172(16-17),
       1873-1896.
    .. [3] Colombo, D., and Maathuis, M. H. (2014). Order-independent constraint-based causal
       structure learning. JMLR 15, 3921-3962.
    .. [4] Spirtes, P., Glymour, C., and Scheines, R. (2000). Causation, Prediction, and Search,
       2nd ed. MIT Press.
    .. [5] Ramsey, J., Zhang, J., and Spirtes, P. (2006). Adjacency-faithfulness and conservative
       causal inference. UAI, 401-408.
    .. [6] Chen, J., and Chen, Z. (2008). Extended Bayesian information criteria for model
       selection with large model spaces. Biometrika 95(3), 759-771.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import gfci
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal(500)
    >>> y = x + 0.1 * rng.standard_normal(500)
    >>> gfci(np.column_stack([x, y])).structure.kind
    'pag'
    """
    from andrey.constraint.pc import resolve_in_column_order, resolve_possible_dsep
    from andrey.search.gfci import gfci as _gfci

    score_func, _ = default_score(data, score_func, categorical=categorical)
    indep_test, categorical = default_test(data, indep_test, categorical=categorical)
    # Both phases read the same data, so the score and the test must agree on its type.
    if data_kind(score_func) != data_kind(indep_test):
        raise ValueError(
            "gfci's score and test must read the data alike: score_func='bdeu' or "
            "'discrete_bic' with indep_test 'chisq' or 'gsq' for categories, "
            "'degen_gauss' or 'cond_gauss' with 'degen_gauss' or "
            "'cond_gauss' for mixed data, or a BIC score with 'fisherz'; got "
            f"score_func={score_func!r}, indep_test={indep_test!r}"
        )
    X, mask = checked_data(data, categorical, "indep_test", indep_test, method="gfci")
    labels = column_labels(data)
    constraints = resolve(knowledge, X.shape[1], labels, method="gfci")
    ambiguous: set[tuple[int, int, int]] = set()
    with copies_named(labels):
        pag = _gfci(
            X,
            score_func=score_func,
            penalty_discount=penalty_discount,
            ebic_gamma=ebic_gamma,
            equivalent_sample_size=equivalent_sample_size,
            alpha=alpha,
            indep_test=indep_test,
            collider_rule=collider_rule,
            categorical=mask,
            possible_dsep=resolve_possible_dsep(possible_dsep),
            max_depth=max_depth,
            max_parents=max_parents,
            constraints=constraints,
            in_column_order=resolve_in_column_order(),
            seed=seed,
            _ambiguous=ambiguous,
        )
    metadata = {
        **categorical_metadata(mask, data),
        **_ambiguous_metadata(collider_rule, ambiguous),
    }
    return structure_output(pag, labels=labels, metadata=metadata)


@method_spec("cdnod")
def cdnod(
    data: npt.ArrayLike,
    domain_index: npt.ArrayLike,
    *,
    alpha: float = 0.05,
    indep_test: str | None = None,
    collider_order: str | None = None,
    collider_rule: str = "sepsets",
    categorical: Sequence[str | int] | None = None,
    dependence_threshold: float | None = None,
    seed: int | None = None,
    max_depth: int | None = None,
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn a partially directed graph over the data variables from multi-domain data with CDNOD.

    CDNOD [1]_ adds a domain/context index as an extra causal parent, so distribution shifts
    across regimes are modeled, then runs PC-style discovery. It applies to samples pooled across
    domains, regimes, or time blocks. The index is appended as a trailing pseudo-node and removed
    from the result.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``.
    domain_index : array-like of shape (n_samples,) or (n_samples, 1)
        Domain or context index of each row of ``data``, such as ``0`` and ``1`` for two domains, or
        the time step for data that drift over time. Coerced to ``float64``; whole numbers with
        ``"chisq"``, ``"gsq"``, ``"degen_gauss"``, or ``"cond_gauss"``. See Notes for more than two
        domains. When every row has the same value, the data come from one domain, and the result
        equals :func:`~andrey.pc`'s.
    alpha : float, default=0.05
        Significance level of the conditional-independence test. Smaller values give sparser graphs.
    indep_test : str or None, default=None
        Conditional-independence test. ``"fisherz"`` (Fisher's z) and ``"student_t"`` (Student-t)
        test partial correlation, for continuous data, and ``"normal_scores"`` does so on each
        column's normal scores. ``"chisq"`` (Pearson's chi-square) and
        ``"gsq"`` (G-square, the likelihood ratio) test categorical data, coded by whole numbers,
        and ``"degen_gauss"`` and ``"cond_gauss"`` mixed data. ``"kci"``, ``"rcot"``, ``"cmiknn"``,
        and ``"gpdc"`` also detect nonlinear dependence in continuous data, at a far higher cost per
        test. Each as for :func:`~andrey.pc`; see Notes for the index. ``None`` picks from the
        declared column types: ``"chisq"`` when every column is categorical, ``"cond_gauss"`` when
        some are, else ``"fisherz"``. A column is declared categorical by ``categorical`` or by a
        DataFrame's ``category``, ``bool``, or text type, never by its values.
        ``metadata["indep_test"]`` names the test the fit used: the one picked from the
        declared column types when this is ``None``.
    collider_order : {"p_value", "column"} or None, default=None
        The order in which colliders are oriented, as for :func:`~andrey.pc`: ``"p_value"`` by the
        largest p-value that separated each collider's endpoints, so the colliders do not depend on
        the column order, and ``"column"`` in column order. ``None`` follows
        ``andrey.config.tie_break``, which also sets whether Meek's rules run in order-free rounds
        or fire one at a time in column order, as for :func:`~andrey.pc`. Under either order, the
        edges out of the index are directed first, and no collider puts an arrowhead at the index.
    collider_rule : {"sepsets", "conservative", "majority"}, default="sepsets"
        How an unshielded triple ``x - z - y`` is decided, as for :func:`~andrey.pc`: by the
        separating set that removed the edge ``x - y`` (``"sepsets"``), or by a vote over the
        subsets of ``x``'s and ``y``'s neighbors that separate them, with the index among the
        variables, which makes ``z`` a collider when none of them contains it and a non-collider
        when all do (``"conservative"`` [3]_), or when fewer than half and when more than half do
        (``"majority"`` [4]_). Any other triple is ambiguous, and no rule orients through it. A
        triple whose middle node is the index is not voted: the index is never a collider.
    categorical : sequence of str or int, optional
        The categorical columns of ``data``, for a mixed-data test: column labels (strings) or
        0-based positions. ``None`` takes a DataFrame's ``category``, ``bool``, and text columns,
        never a column by its values. ``[]`` marks none. The domain index is categorical too, so
        ``data`` may have no categorical column. A categorical test reads every column as
        categorical, and a continuous test raises ``ValueError`` on a declared column, named here or
        by its type. See *Data types* in the
        :doc:`getting-started guide </docs/guides/getting-started>`.
    dependence_threshold : float or None, default=None
        Runs the independent-change step at this threshold, in ``(0, 1)``, on its module
        dependence measure: two modules, the distributions of a candidate cause and of its effect
        given it, whose measure is at most the threshold count as changing independently. ``None``
        skips the step. See Notes.
    seed : int or None, default=None
        ``None``, the default, is the configured seed, ``ANDREY_SEED`` or ``0`` when it is unset,
        not a fresh random one, so a fit repeats. Seed for the random draws of ``indep_test``, which
        only ``"rcot"``, ``"cmiknn"``, and ``"gpdc"`` make; the other tests ignore it. With those
        tests, ``metadata["seed"]`` records the seed used.
    max_depth : int or None, default=None
        The maximum conditioning-set size of every conditional-independence test, at least ``0``:
        ``0`` runs only unconditional tests, and ``None`` sets no limit. A smaller cap runs fewer
        tests and can keep edges that a larger conditioning set would remove. It also caps the
        subsets the conservative and majority rules test.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the pairs
        allowed to be adjacent, and temporal tiers, all honored. The adjacency search starts from
        the allowed pairs and never removes a required pair; the tests can remove any other allowed
        pair. A pair the knowledge makes non-adjacent is tested for a separating set, so colliders
        over it are oriented as usual; when the search finds none within ``max_depth``, no rule
        reads its unshielded triples as colliders or non-colliders. The result directs every edge
        the knowledge decides, then Meek's rules direct the edges that follow, as
        :class:`~andrey.Knowledge` describes; knowledge that orients the edges the search kept into
        a directed cycle raises ``ValueError``. The domain index is unconstrained.

    Returns
    -------
    StructureOutput
        Wraps a :class:`~andrey.core.GraphStructure` over the ``d`` data variables;
        ``metadata["algorithm"]`` is ``"cdnod"``. Directed marks (``->``) are oriented edges,
        undirected (``--``) the ones left open. The graph is partially directed, not the CPDAG of
        an observational equivalence class: the index orients edges that the true DAG's CPDAG
        leaves undirected, and the graph stands for the DAGs that agree on every direction CD-NOD
        can identify [1]_. Andrey has no kind for such a graph, so it is tagged ``cpdag``, and
        :mod:`andrey.metrics` scores a ``cpdag`` against the truth's CPDAG; score the result
        against the graph CD-NOD can identify instead.
        Under a mixed-data setting, ``metadata["categorical"]`` lists the categorical columns it
        read, by name or 0-based position. With ``collider_rule`` ``"conservative"`` or
        ``"majority"``, ``metadata["ambiguous_triples"]`` lists every unshielded triple no rule
        reads as a collider or a non-collider, as ``[x, z, y]``: 0-based column positions, the
        middle node second and ``x < y``, in ascending order; the index, in a triple through it, is
        numbered ``n_variables``. ``metadata["conflicts"]`` lists the edges between variables that
        two colliders ask to orient both ways, as ``[x, y]`` pairs of 0-based column positions with
        ``x < y``, in ascending order, decided as in :func:`~andrey.pc`.
        With a randomized ``indep_test``, ``metadata["seed"]`` is the seed used.

    Raises
    ------
    TypeError
        If ``categorical`` is not a list of column names and positions, or ``knowledge`` is neither
        ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If an argument has the wrong type, lies outside its range, or is not one of its choices; the
        message names the method, the argument, what it accepts, and the value. Also if
        ``domain_index`` does not hold one value per row of ``data`` or holds ``NaN`` / ``inf``, if
        ``data`` has too few rows for a test, if ``data`` or ``categorical`` does not suit the test,
        as for :func:`~andrey.pc`, or, with a categorical or mixed-data test, if ``domain_index``
        holds a value that is not a whole number. With ``"fisherz"`` or ``"student_t"``, also if a
        column of ``data``, or ``domain_index`` with more than one domain, repeats another up to
        scale and shift, before any test runs; one message names every group of copies and says to
        keep one column per group. Also if ``knowledge`` names a column ``data`` lacks or
        contradicts itself, as :class:`~andrey.Knowledge` lists.

    Warns
    -----
    DataTypeWarning
        When a mixed-data test finds no declared categorical column and so runs as the
        linear-Gaussian form, or when a setting reads columns that look like category codes as
        numbers; each message warns once per process.

    Notes
    -----
    The index is added as a variable that can only be a cause. When a variable's distribution
    changes with the index, the edge from the index into it is directed, and Meek's rules can then
    orient edges that PC leaves undirected on the same data.

    ``"fisherz"`` and ``"student_t"`` test the index as a number, by linear correlation, so they
    find a change in a variable's mean that follows the index's values. With two domains the coding
    does not matter. With three domains coded ``0, 1, 2``, a shift in domain ``1`` alone can go
    undetected, and recoding the domains can change the result. A change in a variable's noise
    variance alone, or in the size of an effect whose cause has mean zero, leaves the mean
    unchanged, so the index keeps no edge to that variable. The paper uses a kernel
    conditional-independence test [1]_ [2]_ that detects such changes; Andrey has none. ``"chisq"``,
    ``"gsq"``, ``"degen_gauss"``, and ``"cond_gauss"`` take each index value as a category, so they
    find a shift in any domain, whatever the coding. With ``"degen_gauss"`` or ``"cond_gauss"``, the
    index is categorical, so every data column may be continuous.

    With ``dependence_threshold`` set, the independent-change step [1]_ orients undirected edges
    between two variables that both keep an edge to the index. For a candidate direction
    ``x -> y``, it measures how strongly the change of ``P(x, Z)`` across the index values and the
    change of ``P(y | x, Z)`` depend on each other, where ``Z`` is a set of neighbors that blocks
    their common causes. The measure is the normalized HSIC of the two modules' kernel embeddings
    at each row's index value, with the Gaussian kernel widths of [2]_. ``x -> y`` is oriented when
    that measure is at most the threshold and the measure for ``y -> x`` is above it; when both are
    above it, the edge stays undirected. An orientation that would close a directed cycle is
    skipped, so that edge stays undirected unless Meek's rules orient it. Meek's rules run before
    and after the step. The measure compares the index values as separate domains, so the step needs
    many domains, each with many rows: with two domains, the two changes always depend on each
    other, and nothing is oriented; a per-row index, such as a time step, makes every row its own
    domain, and the step cannot measure a change. The paper leaves the threshold open. For two
    modules that change independently, the measure sits near ``1 / (K - 1)`` over ``K`` domains, so
    the threshold has to rise as the number of domains falls; the 0.06 below is the value the
    measurements used, not a default. The step is off by default, because with ``"fisherz"`` or
    ``"student_t"`` only a variable whose mean follows the index reaches it, and such a shared trend
    makes the changes depend on each other.

    **Missing values.** A ``NaN`` in ``data`` marks a missing value. Each test reads only the rows
    where all of its variables are observed (test-wise deletion) [5]_ [6]_, so it keeps rows that
    deleting every incomplete row first would drop; ``metadata["rows_tested"]`` gives the fewest and
    the median rows a test read. When values are missing completely at random, the rows a test keeps
    are a random sample, so each test keeps its level [6]_. When whether a value is missing depends
    on the variables, the rows a test keeps are a selected sample, and the tests can find dependence
    where the population has none and keep extra edges [6]_. A test raises ``ValueError`` when it
    has fewer rows than it needs, ``|S| + 3`` for ``"fisherz"`` and ``"student_t"`` and 2 for the
    other tests, or when a continuous column is constant on its rows. ``domain_index`` must be
    complete, and ``dependence_threshold``, whose kernel measures read every row, raises
    ``ValueError`` on data with missing values. With missing values, the skeleton search runs in one
    process.

    **Limitations.** Measured on simulated linear-Gaussian data from Erdos-Renyi graphs whose means
    shift across 4 domains, on 4 cores with a cap of 1 h and 28 GB per fit. The largest sizes
    through which its median errors (SHD) stayed below the empty graph's, and through which its fits
    finished within the caps, were 50 and 50 variables at mean degree 1 with 2,000 rows; 100 and 100
    at mean degree 2 with 2,000 or 10,000 rows; 200 and 200 at mean degree 2 with 500 rows and mean
    degree 4 with 2,000 rows; and 400 and 400 at mean degree 8 with 2,000 rows. At 100 variables
    (mean degree 2, 10,000 rows pooled from 4 domains), 1 of 5 fits ran past 1 h. Every ladder ended
    when three fits of the next size ran past 1 h. The median fit took 39 s at 400 variables (mean
    degree 8, 2,000 rows pooled from 4 domains). The independent-change step is not benchmarked yet;
    its tests run it on at most 3 variables with 600 rows from 20 domains. It runs the PC search
    with the index as one more variable, so it costs about as much as :func:`~andrey.pc` on the data
    plus one column. Its partial-correlation tests assume linear relations with Gaussian noise and
    find only a change in a variable's mean that is linear in the index: on two variables from two
    domains with 600 rows, a mean shift of the cause oriented the edge on 20 of 20 seeds, and a
    change of its variance alone, or of the weight of the effect on a mean-zero cause alone, on
    none. Without the independent-change step, an edge between two variables that both change with
    the index stays undirected unless the collider or Meek's rules orient it. With ``"fisherz"`` or
    ``"student_t"``, the step reaches only variables whose means follow the index. On two variables
    from 20 domains with 600 rows, both means rising linearly by about 1 across the domains, the
    cause's noise scale drawn per domain, and the effect's weight on it drawn per domain from (0.5,
    2.5), at the threshold 0.06 it oriented the edge on 1 of 20 seeds. Given the true skeleton of
    ten complete three-variable DAGs from 20 domains with 600 rows, no mean trend, and weights in
    (-2, 2) drawn per domain, it oriented 18 of their 30 edges correctly and 4 against the truth.
    The paper fits the kernel widths, the index kernel, and the ridge of the step by
    Gaussian-process likelihood; Andrey fixes them. With a mixed-data test, it is experimental and
    not benchmarked yet; its tests run it on 5 variables with 3,000 rows.
    ``collider_rule="conservative"`` and ``"majority"`` are not benchmarked yet; their tests run
    them on up to 8 variables. Missing values are not benchmarked yet; the tests run them on up to
    20 variables with up to a fifth of the values missing completely at random. The numbers are a
    reference, not a hard limit: they change with the data generator, the density, the method's
    settings, and the machine.

    References
    ----------
    .. [1] Huang, B., Zhang, K., Zhang, J., Ramsey, J., Sanchez-Romero, R., Glymour, C., and
       Scholkopf, B. (2020). Causal discovery from heterogeneous/nonstationary data. JMLR 21(89),
       1-53.
    .. [2] Zhang, K., Peters, J., Janzing, D., and Scholkopf, B. (2011). Kernel-based conditional
       independence test and application in causal discovery. UAI, 804-813.
    .. [3] Ramsey, J., Zhang, J., and Spirtes, P. (2006). Adjacency-faithfulness and conservative
       causal inference. UAI, 401-408.
    .. [4] Colombo, D., and Maathuis, M. H. (2014). Order-independent constraint-based causal
       structure learning. JMLR 15, 3921-3962.
    .. [5] Strobl, E. V., Visweswaran, S., and Spirtes, P. L. (2018). Fast causal inference with
       non-random missingness by test-wise deletion. International Journal of Data Science and
       Analytics 6(1), 47-62.
    .. [6] Tu, R., Zhang, C., Ackermann, P., Mohan, K., Kjellstrom, H., and Zhang, K. (2019). Causal
       discovery in the presence of missing data. Proceedings of Machine Learning Research 89,
       1762-1770.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import cdnod
    >>> rng = np.random.default_rng(0)
    >>> domain_index = np.repeat([0.0, 1.0], 300)
    >>> x = 2.0 * domain_index + rng.standard_normal(600)
    >>> y = x + 0.5 * rng.standard_normal(600)
    >>> cdnod(np.column_stack([x, y]), domain_index).structure.oriented_edges()
    [(0, 1, 'directed')]
    """
    from andrey.constraint.cdnod import cdnod as _cdnod
    from andrey.constraint.pc import resolve_collider_order, resolve_in_column_order

    indep_test, categorical = default_test(data, indep_test, categorical=categorical)
    # A mixed-data test takes the index as categorical, so every data column may be continuous.
    X, mask = checked_data(
        data,
        categorical,
        "indep_test",
        indep_test,
        index_categorical=True,
        method="cdnod",
        missing=True,
    )
    labels = column_labels(data)
    constraints = resolve(knowledge, X.shape[1], labels, method="cdnod")
    c = np.asarray(domain_index, dtype=np.float64)
    if data_kind(indep_test) != "continuous":
        check_categories(c, None, f"indep_test={indep_test!r}", name="domain_index")
    if dependence_threshold is not None and np.isnan(X).any():
        raise ValueError(
            "cdnod's dependence_threshold step compares kernel measures over every row, so it "
            "does not accept missing values; leave dependence_threshold=None, whose tests delete "
            "rows per test, or drop the rows that hold NaN"
        )
    # The search appends the index as column d, so a copy of it is named as the index.
    d, column = X.shape[1], column_namer(labels)
    ambiguous: set[tuple[int, int, int]] = set()
    conflicts: set[tuple[int, int]] = set()
    with (
        copies_named(labels, lambda j: "domain_index" if j == d else column(j)),
        rows_tested() as rows,
    ):
        cpdag = _cdnod(
            X,
            c,
            alpha=alpha,
            indep_test=indep_test,
            collider_order=resolve_collider_order(collider_order),
            collider_rule=collider_rule,
            categorical=mask,
            dependence_threshold=dependence_threshold,
            max_depth=max_depth,
            constraints=constraints,
            in_column_order=resolve_in_column_order(),
            seed=seed,
            _ambiguous=ambiguous,
            _conflicts=conflicts,
        )
    metadata = {
        "conflicts": [list(edge) for edge in sorted(conflicts)],
        **categorical_metadata(mask, data),
        **_ambiguous_metadata(collider_rule, ambiguous),
        **rows_metadata(rows),
    }
    return structure_output(cpdag, labels=labels, metadata=metadata)
