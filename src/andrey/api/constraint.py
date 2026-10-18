"""Constraint-based facade adapters.

The underlying searches use :mod:`andrey.core` for the conditional-independence tests (Fisher-Z and
Student-t on continuous data, chi-square and G-square on categorical data, degenerate and
conditional Gaussian likelihood ratios on mixed data) and orientation. ``pc`` and ``cdnod`` produce
CPDAGs; ``fci`` and ``gfci`` produce PAGs. GFCI also uses GES with a BIC, BDeu, or mixed-data score.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from andrey.api._adapt import (
    MAX_CATEGORIES,
    categorical_metadata,
    check_categories,
    checked_data,
    column_labels,
    data_kind,
    open_unit_interval,
    structure_output,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from andrey.core import StructureOutput


def pc(
    data: npt.ArrayLike,
    *,
    alpha: float = 0.05,
    indep_test: str = "fisherz",
    collider_order: str = "p_value",
    categorical: Sequence[str | int] | None = None,
    max_categories: int = MAX_CATEGORIES,
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
    indep_test : {"fisherz", "student_t", "chisq", "gsq", "degen_gauss", "cond_gauss"}
        Conditional-independence test, by default ``"fisherz"``. ``"fisherz"`` and ``"student_t"``
        test partial correlation, for continuous data: ``"fisherz"`` applies Fisher's z-transform
        and the normal distribution; ``"student_t"`` applies the Student-t distribution, which is
        exact for Gaussian data at any sample size. ``"chisq"`` (Pearson's chi-square) and
        ``"gsq"`` (G-square, the likelihood ratio) test categorical data, coded by whole numbers.
        ``"degen_gauss"`` (degenerate Gaussian) and ``"cond_gauss"`` (conditional Gaussian) test
        mixed data, continuous and categorical columns, by a likelihood ratio. See Notes.
    collider_order : {"p_value", "column"}, default="p_value"
        The order in which colliders are oriented, which decides between two colliders that would
        orient one edge in opposite directions. ``"p_value"`` orients first the collider whose
        endpoints were separated with the largest p-value, so the colliders do not depend on the
        column order. ``"column"`` orients them in column order and keeps the first of two that
        conflict. See Notes.
    categorical : sequence of str or int, optional
        The categorical columns of ``data``, for a mixed-data test: DataFrame column names or
        0-based positions. ``None`` infers them: a DataFrame's ``category``, ``bool``, and text
        columns, and every column of at most ``max_categories`` distinct whole numbers, named in a
        ``DataTypeWarning``. ``[]`` marks none. The other tests take no column types and reject it.
        See *Data types* in the :doc:`getting-started guide </docs/guides/getting-started>`.
    max_categories : int, default=10
        The most distinct whole numbers a column may hold to be inferred as categorical, at least
        ``2``. With a continuous test, columns within it get a one-time ``DataTypeWarning`` that
        they may be category codes.

    Returns
    -------
    StructureOutput
        Wraps a ``cpdag`` :class:`~andrey.core.GraphStructure`; ``metadata["algorithm"]`` is
        ``"PC"``. Directed marks (``->``) are oriented edges, undirected (``--``) the ones the
        equivalence class leaves open. Under a mixed-data setting, ``metadata["categorical"]`` lists
        the categorical columns it read, by name or 0-based position.

    Raises
    ------
    NotImplementedError
        If ``indep_test`` names no available test.
    TypeError
        If ``categorical`` is not a list of column names and positions, or ``max_categories`` is not
        an int.
    ValueError
        If ``data`` has too few samples for a test or a singular correlation matrix, if ``alpha`` is
        not in ``(0, 1)``, if ``max_categories`` is less than ``2``, with ``"chisq"`` or ``"gsq"``,
        if ``data`` holds a value that is not a whole number, or, with ``"degen_gauss"`` or
        ``"cond_gauss"``, if ``categorical`` names a column ``data`` lacks, or if a categorical
        column holds a number that is not whole or a missing value. Also if ``categorical`` is given
        with another test, or if ``collider_order`` is not ``"p_value"`` or ``"column"``.

    Warns
    -----
    PerformanceWarning
        When a conditioning pass of size two or greater requires at least ten million CI tests;
        each message warns once per process. The warning does not limit the search.
    DataTypeWarning
        When a mixed-data test infers the categorical columns, or finds none and so runs as the
        linear-Gaussian form, or when a continuous setting meets columns that look like category
        codes; each message warns once per process.

    See Also
    --------
    andrey.fci : Allows unmeasured common causes and selection bias.
    andrey.direct_lingam : Orients every edge when the noise is non-Gaussian.

    Notes
    -----
    **When to use it.** PC suits data with no unmeasured common causes. Its default Fisher-Z test
    assumes continuous data, where each variable depends linearly on its causes with roughly
    Gaussian noise; for categorical data, use ``"chisq"`` or ``"gsq"``, and for mixed data
    ``"degen_gauss"`` or ``"cond_gauss"``. It is fastest on sparse graphs, where every variable has
    few neighbors [5]_.

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
    kept. With the default ``collider_order="p_value"``, each collider ``x -> y <- z`` ranks by the
    largest p-value among the conditioning sets that separated ``x`` and ``z``, and colliders are
    oriented from the highest rank to the lowest. When two conflicting colliders rank exactly
    equal, the edge they dispute stays undirected, and Meek's rules do not orient it. The colliders
    therefore do not depend on the column order. This adapts the conflict rule of Ramsey [10]_,
    which ranks a collider by separating sets that additional tests choose; Andrey ranks it by the
    tests that removed the edge, so orienting runs no new test. With ``collider_order="column"``,
    colliders are oriented by middle node, then by endpoint pair, in column order, so the result
    can depend on the column order. Meek's rules can still depend on the column order: when two
    rule firings would orient one edge in opposite directions, the one reached first in column
    order decides.

    **Limitations.** The benchmark numbers below were measured with ``collider_order="column"``; the
    default is not benchmarked yet. On simulated linear-Gaussian data from sparse random graphs
    (mean degree 2), from 20 to 400 variables with ten rows per variable, its median errors (SHD)
    stayed below the empty graph's at every size, 357 against 400 at 400 variables. The median fit
    took 0.0076 s at 20 variables and 0.82 s at 400, on 16 cores. On the 223-variable ANDES network
    with 1,000 simulated rows, its median errors passed the empty graph's (372 against 338). It is
    practical to roughly 1,000 variables on sparse data with ten rows per variable; runtime grows
    combinatorially with the largest neighborhood, so sparser graphs can stay feasible beyond that.
    On simulated mixed data (conditional linear Gaussian networks on sparse random graphs, the
    first half of the variables categorical with 2 to 5 categories), from 10 to 40 variables with
    2,000 rows, its median errors stayed below the empty graph's at every size: at 40 variables,
    18.5 with ``"degen_gauss"`` and 9.5 with ``"cond_gauss"``, against 40. On simulated categorical
    networks with 2 or 3 categories per variable, of the same sizes and rows, they were 14 with
    ``"chisq"`` and 16 with ``"gsq"`` at 40 variables. The median fit at 40 variables took 0.38 s
    with ``"degen_gauss"``, 0.42 s with ``"cond_gauss"``, and 0.18 s with either categorical test,
    on 4 cores.

    **Categorical data.** ``"chisq"`` and ``"gsq"`` take each distinct value of a column as one
    category, so the codes need not start at 0 or be consecutive; a DataFrame's text, ``bool``, and
    ``category`` columns are read the same way. A test compares the counts of ``x`` and ``y`` with
    independence within each configuration of the conditioning set that occurs in the data [2]_.
    Each configuration adds ``(r_x - 1) * (r_y - 1)`` degrees of freedom, counting only the values
    of ``x`` and ``y`` present in it [6]_; with none, the p-value is 1. A test needs several rows
    per cell of its table: with many categories or a large conditioning set, most cells are empty,
    and the p-value is unreliable.

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

    X, mask = checked_data(
        data, categorical, "indep_test", indep_test, max_categories=max_categories
    )
    alpha = open_unit_interval(alpha, "alpha")
    labels = column_labels(data)
    cpdag = _pc(
        X, alpha=alpha, indep_test=indep_test, collider_order=collider_order, categorical=mask
    )
    return structure_output(
        cpdag, labels=labels, metadata={"algorithm": "PC", **categorical_metadata(mask, data)}
    )


def fci(
    data: npt.ArrayLike,
    *,
    alpha: float = 0.05,
    indep_test: str = "fisherz",
    collider_rule: str = "sepsets",
    categorical: Sequence[str | int] | None = None,
    max_categories: int = MAX_CATEGORIES,
    possible_dsep: str = "fast",
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
    indep_test : {"fisherz", "student_t", "chisq", "gsq", "degen_gauss", "cond_gauss"}
        Conditional-independence test, by default ``"fisherz"``. ``"fisherz"`` and ``"student_t"``
        test partial correlation, for continuous data: ``"fisherz"`` applies Fisher's z-transform
        and the normal distribution; ``"student_t"`` applies the Student-t distribution, which is
        exact for Gaussian data at any sample size. ``"chisq"`` (Pearson's chi-square) and
        ``"gsq"`` (G-square, the likelihood ratio) test categorical data, coded by whole numbers.
        ``"degen_gauss"`` (degenerate Gaussian) and ``"cond_gauss"`` (conditional Gaussian) test
        mixed data, continuous and categorical columns. See Notes.
    collider_rule : {"sepsets", "majority"}, default="sepsets"
        How FCI decides whether ``z`` is a collider on a triple ``x - z - y`` where ``x`` and ``y``
        are not adjacent. ``"sepsets"`` makes it a collider when ``z`` is not in the separating set
        that removed the edge ``x - y``. ``"majority"`` tests ``x`` and ``y`` given every subset of
        ``x``'s neighbors and every subset of ``y``'s neighbors, and makes it a collider when
        fewer than half of the separating subsets (``p >= alpha``) contain ``z`` [4]_; at exactly
        half the triple stays unoriented. Majority runs up to ``2**degree`` extra tests per triple.
    categorical : sequence of str or int, optional
        The categorical columns of ``data``, for a mixed-data test: DataFrame column names or
        0-based positions. ``None`` infers them: a DataFrame's ``category``, ``bool``, and text
        columns, and every column of at most ``max_categories`` distinct whole numbers, named in a
        ``DataTypeWarning``. ``[]`` marks none. The other tests take no column types and reject it.
        See *Data types* in the :doc:`getting-started guide </docs/guides/getting-started>`.
    max_categories : int, default=10
        The most distinct whole numbers a column may hold to be inferred as categorical, at least
        ``2``. With a continuous test, columns within it get a one-time ``DataTypeWarning`` that
        they may be category codes.
    possible_dsep : {"fast", "order_free"}, default="fast"
        How FCI builds each possible-d-separation set [2]_ [3]_, the nodes it conditions on to
        remove edges the first step kept. ``"fast"`` checks each node when the search reaches it,
        so the set depends on the order the search visits nodes. ``"order_free"`` checks every
        node once the search ends, so the set does not depend on that order; its sets can be
        larger, so it can run more tests. See Notes.

    Returns
    -------
    StructureOutput
        Wraps a ``pag`` :class:`~andrey.core.GraphStructure`; ``metadata["algorithm"]`` is
        ``"FCI"``. Endpoints carry arrowheads, tails, or circle marks (``o``); a bidirected edge
        (``<->``) marks a probable latent common cause. Under a mixed-data setting,
        ``metadata["categorical"]`` lists the categorical columns it read, by name or 0-based
        position.

    Raises
    ------
    NotImplementedError
        If ``indep_test`` names no available test.
    TypeError
        If ``categorical`` is not a list of column names and positions, or ``max_categories`` is not
        an int.
    ValueError
        If ``data`` has too few samples for a test or a singular correlation matrix, if ``alpha``
        is not in ``(0, 1)``, if ``collider_rule`` or ``possible_dsep`` is not a supported choice,
        or if ``data`` or ``categorical`` does not suit the test, as for :func:`~andrey.pc`.

    Warns
    -----
    DataTypeWarning
        When a mixed-data test infers the categorical columns, or finds none and so runs as the
        linear-Gaussian form, or when a continuous setting meets columns that look like category
        codes; each message warns once per process.

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

    The result can depend on the column order, for example, through the order in which the
    possible-d-separation step visits pairs and conditioning sets, and with
    ``possible_dsep="fast"`` through the sets themselves. A fixed column order gives reproducible
    results.

    Rule R5 turns a cycle of circle-circle edges into undirected edges only if the whole cycle,
    including the closing edge, is uncovered, that is, no two nodes two steps apart are adjacent.
    It considers every circle path, up to a fixed search bound, so a covered route through a node
    never hides an uncovered route through the same node. Rule R8 checks each pair of endpoints in
    both orders, so whether it fires does not depend on the node labels.

    **Possible-d-separation set.** FCI builds each possible-d-separation set [2]_ [3]_ with a search
    that walks out from one endpoint of the pair and keeps a node when a node it was reached from
    has a possibly directed path, one with no arrowhead pointing back along it, to the endpoint or
    to the node. With ``possible_dsep="fast"``, FCI checks a node when the search reaches it,
    against the nodes it was reached from so far, so the set depends on the order the search visits
    nodes. With ``"order_free"``, it checks every node once the search ends, against all the nodes
    it was reached from, so the set does not depend on that order; it holds every node of the
    ``"fast"`` set and sometimes more. Either set can be smaller than the published definition of
    the set [2]_, so FCI can keep an edge that the published set would remove, even with exact
    tests. FCI keeps this search because the published set grows quickly with the graph, the
    search's set is the one FCI has been measured with, and on simulated data the published set did
    not give more accurate PAGs. FCI tests every subset of the set, except the empty set and the
    subsets of either endpoint's neighbors, which the first step already tested.

    **Separating sets.** When several conditioning sets of the same size make a pair independent
    in the first step, FCI records their union as the pair's separating set. FCI keeps the union
    because it does not depend on the order of the tests, and on simulated data it gave more
    accurate PAGs than the first set found.

    **Limitations.** Measured on simulated linear-Gaussian data from sparse random graphs (mean
    degree 2) with one variable in ten hidden, from 20 to 800 variables counting the hidden ones,
    with ten rows per variable. Its median endpoint errors, each mismatched tail, arrowhead, or
    circle, stayed below the empty graph's up to 400 variables (836 against 884) and passed it at
    800 (2,579 against 1,772). The median fit took 0.061 s at 20 variables and 72 s at 800, on 16
    cores. On the 223-variable ANDES network with 1,000 simulated rows, its median endpoint errors
    passed the empty graph's (696 against 676). Rule R5 stops searching a pair after a fixed number
    of partial paths and leaves its circles, so on graphs with many circle-circle edges it can miss
    an undirected cycle. ``"order_free"`` is order-free only under exact (oracle) tests; on sampled
    data the result can still depend on the column order. With the mixed-data tests, it is
    experimental and not benchmarked yet; its tests run it on 5 variables with 3,000 rows.

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

    X, mask = checked_data(
        data, categorical, "indep_test", indep_test, max_categories=max_categories
    )
    alpha = open_unit_interval(alpha, "alpha")
    labels = column_labels(data)
    pag = _fci(
        X,
        alpha=alpha,
        indep_test=indep_test,
        collider_rule=collider_rule,
        categorical=mask,
        possible_dsep=possible_dsep,
    )
    return structure_output(
        pag, labels=labels, metadata={"algorithm": "FCI", **categorical_metadata(mask, data)}
    )


def gfci(
    data: npt.ArrayLike,
    *,
    score_func: str = "local_score_BIC",
    lambda_value: float = 1.0,
    indep_test: str = "fisherz",
    alpha: float = 0.05,
    collider_rule: str = "sepsets",
    categorical: Sequence[str | int] | None = None,
    max_categories: int = MAX_CATEGORIES,
    possible_dsep: str = "fast",
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
    score_func : str, default="local_score_BIC"
        Local score driving the GES skeleton phase, as for :func:`~andrey.ges`:
        ``"local_score_BIC"``, the linear-Gaussian BIC; ``"local_score_BDeu"`` or
        ``"local_score_BIC_categorical"`` for categorical data, coded by whole numbers; or
        ``"local_score_degen_gauss"`` or ``"local_score_cond_gauss"``, the degenerate or conditional
        Gaussian BIC for mixed data.
    lambda_value : float, default=1.0
        Multiplier on the BIC penalty of ``log(n_samples)`` per parameter in the GES phase. Larger
        values give sparser graphs; ``1.0`` is the standard BIC. Must be finite and non-negative.
        BDeu ignores it.
    indep_test : {"fisherz", "student_t", "chisq", "gsq", "degen_gauss", "cond_gauss"}
        Conditional-independence test for the FCI phase, by default ``"fisherz"``. ``"fisherz"``
        (Fisher's z) and ``"student_t"`` (Student-t) test partial correlation, for continuous data.
        ``"chisq"`` (Pearson's chi-square) and ``"gsq"`` (G-square, the likelihood ratio) test
        categorical data, coded by whole numbers, and ``"degen_gauss"`` and ``"cond_gauss"`` mixed
        data, as for :func:`~andrey.pc`.
    alpha : float, default=0.05
        Significance level of the conditional-independence test. Smaller values give sparser graphs.
    collider_rule : {"sepsets", "majority"}, default="sepsets"
        How GFCI decides whether ``z`` is a collider on a triple ``x - z - y`` where the tests
        removed the GES edge between ``x`` and ``y``; on the other triples whose ends are not
        adjacent, the GES CPDAG decides (see Notes). ``"sepsets"`` makes it a collider when ``z``
        is not in the recorded separating set of ``x`` and ``y``. ``"majority"`` tests ``x`` and
        ``y`` given every subset of ``x``'s neighbors and every subset of ``y``'s neighbors, and
        makes it a collider when fewer than half of the separating subsets (``p >= alpha``)
        contain ``z`` [3]_; at exactly half the triple stays unoriented. Majority runs up to
        ``2**degree`` extra tests per triple.
    categorical : sequence of str or int, optional
        The categorical columns of ``data``, for the mixed-data score and test: DataFrame column
        names or 0-based positions. ``None`` infers them: a DataFrame's ``category``, ``bool``, and
        text columns, and every column of at most ``max_categories`` distinct whole numbers, named
        in a ``DataTypeWarning``. ``[]`` marks none. The other settings reject it. See *Data types*
        in the :doc:`getting-started guide </docs/guides/getting-started>`.
    max_categories : int, default=10
        The most distinct whole numbers a column may hold to be inferred as categorical, at least
        ``2``. With a continuous setting, columns within it get a one-time ``DataTypeWarning`` that
        they may be category codes.
    possible_dsep : {"fast", "order_free"}, default="fast"
        How GFCI builds each possible-d-separation set [2]_ [4]_, the nodes it conditions on to
        remove edges the adjacency search kept. ``"fast"`` checks each node when the search reaches
        it, so the set depends on the order the search visits nodes. ``"order_free"`` checks every
        node once the search ends, so the set does not depend on that order; its sets can be
        larger, so it can run more tests. See Notes.

    Returns
    -------
    StructureOutput
        Wraps a ``pag`` :class:`~andrey.core.GraphStructure`; ``metadata["algorithm"]`` is
        ``"GFCI"``. Endpoints carry arrowheads, tails, or circle marks (``o``); a bidirected edge
        (``<->``) marks a probable latent common cause. Under a mixed-data setting,
        ``metadata["categorical"]`` lists the categorical columns it read, by name or 0-based
        position.

    Raises
    ------
    NotImplementedError
        If ``score_func`` names no available score, or ``indep_test`` no available test.
    TypeError
        If ``categorical`` is not a list of column names and positions, or ``max_categories`` is not
        an int.
    ValueError
        If ``alpha`` is not in ``(0, 1)``, if ``lambda_value`` is negative or non-finite, if
        ``collider_rule`` or ``possible_dsep`` is not a supported choice, if ``score_func`` and
        ``indep_test`` read different data (continuous, categorical, or mixed), or if ``data`` or
        ``categorical`` does not suit them, as for :func:`~andrey.pc`. With the BIC score, also if a
        column repeats another up to scale and shift given more than two rows or, with more rows
        than columns, is a linear function of other columns.

    Warns
    -----
    DataTypeWarning
        When the mixed-data score and test infers the categorical columns, or finds none and so runs
        as the linear-Gaussian form, or when a continuous setting meets columns that look like
        category codes; each message warns once per process.

    Notes
    -----
    GFCI first runs :func:`~andrey.ges` and keeps its CPDAG. FCI's adjacency search then starts
    from the CPDAG's adjacencies instead of a complete graph: it removes an edge ``x - y`` when a
    subset of ``x``'s or ``y``'s neighbors makes the two independent, and records that subset as
    their separating set. On a triple ``x - z - y`` whose ends are not adjacent, ``z`` is a
    collider when the CPDAG has ``x -> z <- y``; when the CPDAG had the edge ``x - y``, the
    separating set decides, as ``collider_rule`` describes. FCI's possible-d-separation step can
    remove more edges, after which the colliders are decided again. No step adds an edge, so the
    PAG's adjacencies are a subset of the GES CPDAG's. Zhang's rules R1-R10 [2]_ then orient the
    endpoints; an endpoint no rule resolves stays a circle. A rule that needs the separating set
    of a pair GES left non-adjacent uses the first subset, by increasing size, of the two
    endpoints' CPDAG neighbors that makes the pair independent.

    Rule R5 turns a cycle of circle-circle edges into undirected edges only when the whole cycle,
    closing edge included, is uncovered, that is, no two nodes two steps apart on it are adjacent.
    It considers every circle path, up to a fixed search bound, and rule R8 checks each pair of
    endpoints in both orders. GFCI shares FCI's possible-d-separation sets [2]_ [4]_, built as
    ``possible_dsep`` selects, and its separating sets; the Notes of :func:`~andrey.fci` describe
    both and the reasons for keeping them. With ``possible_dsep="fast"``, GFCI checks each node
    when the search reaches it, against the nodes it was reached from so far, so the set depends on
    the order the search visits nodes. With ``"order_free"``, it checks every node once the search
    ends, against all the nodes it was reached from, so the set does not depend on that order, and
    it can be larger.

    **Limitations.** Measured on simulated linear-Gaussian data from sparse random graphs (mean
    degree 2) with one variable in ten hidden, from 20 to 200 variables counting the hidden ones (18
    to 180 observed), with ten rows per variable. Its median endpoint errors, each mismatched tail,
    arrowhead, or circle, stayed below the empty graph's at every size (197 against 444 at 180
    observed variables). The median fit took 0.54 s at 90 observed variables and 2.6 s at 180, on 4
    cores. It runs a full :func:`~andrey.ges` search before any test,
    so it takes at least as long as GES on the same data. Its BIC score and Fisher-Z tests assume
    linear relations with Gaussian noise. With fewer rows than columns the BIC score is
    unreliable. A parent set of ``n_samples - 1`` or more variables fits its child exactly, so the
    GES search never selects one, nor any parent set that leaves its child a residual variance of at
    most ``1e-12`` times its variance. Rule R5 has the search bound described under
    :func:`~andrey.fci`. ``"order_free"`` is order-free only under exact (oracle) tests; on sampled
    data the result can still depend on the column order. With the mixed-data score and test, it is
    experimental and not benchmarked yet; its tests run it on 5 variables with 3,000 rows.

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
    from andrey.search.gfci import gfci as _gfci

    # Both phases read the same data, so the score and the test must agree on its type.
    if data_kind(score_func) != data_kind(indep_test):
        raise ValueError(
            "gfci's score and test must read the data alike: score_func='local_score_BDeu' or "
            "'local_score_BIC_categorical' with indep_test 'chisq' or 'gsq' for categories, "
            "'local_score_degen_gauss' or 'local_score_cond_gauss' with 'degen_gauss' or "
            "'cond_gauss' for mixed data, or a BIC score with 'fisherz'; got "
            f"score_func={score_func!r}, indep_test={indep_test!r}"
        )
    X, mask = checked_data(
        data, categorical, "indep_test", indep_test, max_categories=max_categories
    )
    alpha = open_unit_interval(alpha, "alpha")
    labels = column_labels(data)
    pag = _gfci(
        X,
        score_func=score_func,
        lambda_value=lambda_value,
        alpha=alpha,
        indep_test=indep_test,
        collider_rule=collider_rule,
        categorical=mask,
        possible_dsep=possible_dsep,
    )
    return structure_output(
        pag, labels=labels, metadata={"algorithm": "GFCI", **categorical_metadata(mask, data)}
    )


def cdnod(
    data: npt.ArrayLike,
    domain_index: npt.ArrayLike,
    *,
    alpha: float = 0.05,
    indep_test: str = "fisherz",
    collider_order: str = "p_value",
    categorical: Sequence[str | int] | None = None,
    max_categories: int = MAX_CATEGORIES,
    dependence_threshold: float | None = None,
) -> StructureOutput:
    """Learn a partially directed graph over the data variables from multi-domain data with CDNOD.

    CDNOD [1]_ adds a domain/context index as an extra causal parent, so distribution shifts
    across regimes are modeled, then runs PC-style discovery. Use it when samples are pooled across
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
    indep_test : {"fisherz", "student_t", "chisq", "gsq", "degen_gauss", "cond_gauss"}
        Conditional-independence test, by default ``"fisherz"``. ``"fisherz"`` (Fisher's z) and
        ``"student_t"`` (Student-t) test partial correlation, for continuous data. ``"chisq"``
        (Pearson's chi-square) and ``"gsq"`` (G-square, the likelihood ratio) test categorical data,
        coded by whole numbers, and ``"degen_gauss"`` and ``"cond_gauss"`` mixed data, as for
        :func:`~andrey.pc`; see Notes for the index.
    collider_order : {"p_value", "column"}, default="p_value"
        The order in which colliders are oriented, as for :func:`~andrey.pc`: ``"p_value"`` by the
        largest p-value that separated each collider's endpoints, so the colliders do not depend on
        the column order, and ``"column"`` in column order. Under either order, the edges out of
        the index are directed first, and no collider puts an arrowhead at the index.
    categorical : sequence of str or int, optional
        The categorical columns of ``data``, for a mixed-data test: DataFrame column names or
        0-based positions. ``None`` infers them: a DataFrame's ``category``, ``bool``, and text
        columns, and every column of at most ``max_categories`` distinct whole numbers, named in a
        ``DataTypeWarning``. ``[]`` marks none. The domain index is categorical too, so ``data`` may
        have no categorical column. The other tests take no column types and reject it. See *Data
        types* in the :doc:`getting-started guide </docs/guides/getting-started>`.
    max_categories : int, default=10
        The most distinct whole numbers a column may hold to be inferred as categorical, at least
        ``2``. With a continuous test, columns within it get a one-time ``DataTypeWarning`` that
        they may be category codes.
    dependence_threshold : float or None, default=None
        Runs the independent-change step at this threshold, in ``(0, 1)``, on its module
        dependence measure: two modules whose measure is at most the threshold count as changing
        independently. ``None`` skips the step. See Notes.

    Returns
    -------
    StructureOutput
        Wraps a :class:`~andrey.core.GraphStructure` over the ``d`` data variables;
        ``metadata["algorithm"]`` is ``"CDNOD"``. Directed marks (``->``) are oriented edges,
        undirected (``--``) the ones left open. The graph is partially directed, not the CPDAG of
        an observational equivalence class: the index orients edges that the true DAG's CPDAG
        leaves undirected, and the graph stands for the DAGs that agree on every direction CD-NOD
        can identify [1]_. Andrey has no kind for such a graph, so it is tagged ``cpdag``, and
        :mod:`andrey.metrics` scores a ``cpdag`` against the truth's CPDAG; score the result
        against the graph CD-NOD can identify instead.
        Under a mixed-data setting, ``metadata["categorical"]`` lists the categorical columns it
        read, by name or 0-based position.

    Raises
    ------
    NotImplementedError
        If ``indep_test`` names no available test.
    TypeError
        If ``categorical`` is not a list of column names and positions, or ``max_categories`` is not
        an int.
    ValueError
        If ``domain_index`` does not hold one value per row of ``data`` or holds ``NaN`` /
        ``inf``, if ``data`` has too few rows for a test, if ``alpha`` or ``dependence_threshold``
        is not in ``(0, 1)``, if ``data`` or ``categorical`` does not suit the test, as for
        :func:`~andrey.pc`, or, with a categorical or mixed-data test, if ``domain_index`` holds a
        value that is not a whole number. Also if ``collider_order`` is not ``"p_value"`` or
        ``"column"``.

    Warns
    -----
    DataTypeWarning
        When a mixed-data test infers the categorical columns, or finds none and so runs as the
        linear-Gaussian form, or when a continuous setting meets columns that look like category
        codes; each message warns once per process.

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
    unchanged, so the index keeps no edge to that variable. The paper uses a kernel test [1]_ that
    detects such changes; Andrey has no kernel test. ``"chisq"``, ``"gsq"``, ``"degen_gauss"``, and
    ``"cond_gauss"`` take each index value as a category, so they find a shift in any domain,
    whatever the coding. With ``"degen_gauss"`` or ``"cond_gauss"``, the index is categorical, so
    every data column may be continuous.

    With ``dependence_threshold`` set, the independent-change step [1]_ orients undirected edges
    between two variables that both keep an edge to the index. For a candidate direction
    ``x -> y``, it measures how strongly the change of ``P(x)`` across the index values and the
    change of ``P(y | x)`` depend on each other, given a set of neighbors that blocks their common
    causes. ``x -> y`` is oriented when that measure is at most the threshold and the measure for
    ``y -> x`` is above it; when both are above it, the edge stays undirected. An orientation
    that would close a directed cycle is skipped, so that edge stays undirected unless Meek's
    rules orient it. Meek's rules run before and after the step. The measure compares the index
    values as separate domains, so the step needs many domains, each with many rows: with two
    domains, the two changes always depend on each other, and nothing is oriented; a per-row
    index, such as a time step, makes every row its own domain, and the step cannot measure a
    change. The paper leaves the threshold open. For two modules that change independently, the
    measure sits near ``1 / (K - 1)`` over ``K`` domains, so the threshold has to rise as the
    number of domains falls; the 0.06 below is the value the measurements used, not a default.
    The step is off by default, because with these tests only a variable whose mean follows the
    index reaches it, and such a shared trend makes the changes depend on each other.

    **Limitations.** Measured on simulated linear-Gaussian data from sparse random graphs (mean
    degree 2), with 2,000 rows pooled from four domains of equal size, in which one variable in
    three shifts its mean with the domain, from 10 to 50 variables with 20 datasets per size. Its
    median errors (SHD against the CPDAG the domain index makes identifiable) stayed below the empty
    graph's at every size (12 against 50 at 50 variables). The median fit took 0.18 s at 50
    variables, on 4 cores. The independent-change step is not benchmarked yet; its tests run it on
    at most 4 variables with 600 rows from 20 domains. It runs the PC search with the index as one
    more variable, so it costs about as much as :func:`~andrey.pc` on the data plus one column. Its
    partial-correlation tests assume linear relations with Gaussian noise and find only a change in
    a variable's mean that is linear in the index: on two variables from two domains with 600 rows,
    a mean shift of the cause oriented the edge on 20 of 20 seeds, and a change of its variance
    alone, or of the weight of the effect on a mean-zero cause alone, on none. Without the
    independent-change step, an edge between two variables that both change with the index stays
    undirected unless the collider or Meek's rules orient it. The step reaches only variables whose
    means follow the index. On two variables from 20 domains with 600 rows, both means rising
    linearly by about 1 across the domains, the cause's noise scale drawn per domain, and the
    effect's weight on it drawn per domain from (0.5, 2.5), at the threshold 0.06 it oriented the
    edge on 1 of 20 seeds. Given the true skeleton of ten complete three-variable DAGs from 20
    domains with 600 rows, no mean trend, and weights in (-2, 2) drawn per domain, it oriented 18 of
    their 30 edges correctly and 4 against the truth. The paper fits the kernel widths, the index
    kernel, and the ridge of the step by Gaussian-process likelihood; Andrey fixes them. With a
    mixed-data test, it is experimental and not benchmarked yet; its tests run it on 5 variables
    with 3,000 rows.

    References
    ----------
    .. [1] Huang, B., Zhang, K., Zhang, J., Ramsey, J., Sanchez-Romero, R., Glymour, C., and
       Scholkopf, B. (2020). Causal discovery from heterogeneous/nonstationary data. JMLR 21(89),
       1-53.

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

    # A mixed-data test takes the index as categorical, so every data column may be continuous.
    X, mask = checked_data(
        data,
        categorical,
        "indep_test",
        indep_test,
        max_categories=max_categories,
        index_categorical=True,
    )
    alpha = open_unit_interval(alpha, "alpha")
    labels = column_labels(data)
    c = np.asarray(domain_index, dtype=np.float64)
    if data_kind(indep_test) != "continuous":
        check_categories(c, None, f"indep_test={indep_test!r}", name="domain_index")
    if dependence_threshold is not None:
        dependence_threshold = open_unit_interval(dependence_threshold, "dependence_threshold")
    cpdag = _cdnod(
        X,
        c,
        alpha=alpha,
        indep_test=indep_test,
        collider_order=collider_order,
        categorical=mask,
        dependence_threshold=dependence_threshold,
    )
    return structure_output(
        cpdag, labels=labels, metadata={"algorithm": "CDNOD", **categorical_metadata(mask, data)}
    )
