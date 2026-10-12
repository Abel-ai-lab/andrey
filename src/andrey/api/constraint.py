"""Constraint-based facade adapters.

The underlying searches use :mod:`andrey.core` for the conditional-independence tests (Fisher-Z on
continuous data, chi-square and G-square on categorical data) and orientation. ``pc`` and
``cdnod`` produce CPDAGs; ``fci`` and ``gfci`` produce PAGs. GFCI also uses GES with BIC or BDeu
scoring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from andrey.api._adapt import (
    check_categories,
    column_labels,
    data_matrix,
    open_unit_interval,
    structure_output,
)
from andrey.api._experimental import warn_experimental
from andrey.core.ci import DISCRETE_INDEP_TESTS

if TYPE_CHECKING:
    import numpy.typing as npt

    from andrey.core import StructureOutput


def pc(data: npt.ArrayLike, *, alpha: float = 0.05, indep_test: str = "fisherz") -> StructureOutput:
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
    indep_test : {"fisherz", "chisq", "gsq"}, default="fisherz"
        Conditional-independence test. ``"fisherz"`` tests partial correlation (Fisher's z), for
        continuous data. ``"chisq"`` (Pearson's chi-square) and ``"gsq"`` (G-square, the
        likelihood ratio) test categorical data, coded by whole numbers; see Notes.

    Returns
    -------
    StructureOutput
        Wraps a ``cpdag`` :class:`~andrey.core.GraphStructure`; ``metadata["algorithm"]`` is
        ``"PC"``. Directed marks (``->``) are oriented edges, undirected (``--``) the ones the
        equivalence class leaves open.

    Raises
    ------
    NotImplementedError
        If ``indep_test`` names no available test.
    ValueError
        If ``data`` has too few samples for a test or a singular correlation matrix, if ``alpha``
        is not in ``(0, 1)``, or, with ``"chisq"`` or ``"gsq"``, if ``data`` holds a value that is
        not a whole number.

    Warns
    -----
    PerformanceWarning
        When a conditioning pass of size two or greater requires at least ten million CI tests;
        each message warns once per process. The warning does not limit the search.

    See Also
    --------
    andrey.fci : Allows unmeasured common causes and selection bias.
    andrey.direct_lingam : Orients every edge when the noise is non-Gaussian.

    Notes
    -----
    **When to use it.** PC suits data with no unmeasured common causes. Its default Fisher-Z test
    assumes continuous data, where each variable depends linearly on its causes with roughly
    Gaussian noise; for categorical data, use ``"chisq"`` or ``"gsq"``. It is fastest on sparse
    graphs, where every variable has few neighbors [5]_. An undirected edge (``--``) is one the
    data cannot orient.

    **When not to.** When unmeasured common causes are plausible, use :func:`~andrey.fci`. When the
    noise is non-Gaussian and every edge should be oriented, use :func:`~andrey.direct_lingam`.

    **Choosing alpha.** A test removes an edge when it cannot reject independence at ``alpha``. A
    larger ``alpha`` therefore keeps more edges, true and false. On the Sachs data in the
    getting-started guide, raising ``alpha`` from 0.05 to 0.2 found more known links, but a smaller
    share of the links found were known. No value is right in general. Choose it by weighing the
    cost of a missed edge against a false one, and report how the graph changes with it.

    **How it works.** Edge removal is order-independent (PC-stable) [3]_. At each conditioning-set
    size, every test uses the adjacencies from the start of that size. Edges found independent are
    removed together. A removed pair's separating set combines all conditioning sets that separated
    it. If two colliders conflict, the one found first in column order is kept.

    **Limitations.** Measured on simulated linear-Gaussian data from sparse random graphs (mean
    degree 2), from 20 to 400 variables with ten rows per variable: its median errors (SHD) stayed
    below the empty graph's at every size, 357 against 400 at 400 variables. The median fit took
    0.0076 s at 20 variables and 0.82 s at 400, on 16 cores. On the 223-variable ANDES network with
    1,000 simulated rows, its median errors passed the empty graph's (372 against 338). It is
    practical to roughly 1,000 variables on sparse data with ten rows per variable; runtime grows
    combinatorially with the largest neighborhood, so sparser graphs can stay feasible beyond that.

    **Categorical data.** ``"chisq"`` and ``"gsq"`` take each distinct value of a column as one
    category, so the codes need not start at 0 or be consecutive. A test compares the counts of
    ``x`` and ``y`` with independence within each configuration of the conditioning set that
    occurs in the data [2]_. Each configuration adds ``(r_x - 1) * (r_y - 1)`` degrees of freedom,
    counting only the values of ``x`` and ``y`` present in it [6]_; with none, the p-value is 1.
    A test needs several rows per cell of its table: with many categories or a large conditioning
    set, most cells are empty, and the p-value is unreliable.

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
    """
    from andrey.constraint.pc import pc as _pc

    X = data_matrix(data)
    alpha = open_unit_interval(alpha, "alpha")
    if indep_test in DISCRETE_INDEP_TESTS:
        check_categories(X, data, f"indep_test={indep_test!r}")
    labels = column_labels(data)
    return structure_output(
        _pc(X, alpha=alpha, indep_test=indep_test), labels=labels, metadata={"algorithm": "PC"}
    )


def fci(
    data: npt.ArrayLike,
    *,
    alpha: float = 0.05,
    indep_test: str = "fisherz",
    collider_rule: str = "sepsets",
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
    indep_test : {"fisherz", "chisq", "gsq"}, default="fisherz"
        Conditional-independence test. ``"fisherz"`` tests partial correlation (Fisher's z), for
        continuous data. ``"chisq"`` (Pearson's chi-square) and ``"gsq"`` (G-square, the
        likelihood ratio) test categorical data, coded by whole numbers; see Notes.
    collider_rule : {"sepsets", "majority"}, default="sepsets"
        How FCI decides whether ``z`` is a collider on a triple ``x - z - y`` where ``x`` and ``y``
        are not adjacent. ``"sepsets"`` makes it a collider when ``z`` is not in the separating set
        that removed the edge ``x - y``. ``"majority"`` tests ``x`` and ``y`` given every subset of
        ``x``'s neighbors and every subset of ``y``'s neighbors, and makes it a collider when
        fewer than half of the separating subsets (``p >= alpha``) contain ``z`` [4]_; at exactly
        half the triple stays unoriented. Majority runs up to ``2**degree`` extra tests per triple.

    Returns
    -------
    StructureOutput
        Wraps a ``pag`` :class:`~andrey.core.GraphStructure`; ``metadata["algorithm"]`` is
        ``"FCI"``. Endpoints carry arrowheads, tails, or circle marks (``o``); a bidirected edge
        (``<->``) marks a probable latent common cause.

    Raises
    ------
    NotImplementedError
        If ``indep_test`` names no available test.
    ValueError
        If ``data`` has too few samples for a test or a singular correlation matrix, if ``alpha``
        is not in ``(0, 1)``, if ``collider_rule`` is not a supported choice, or, with
        ``"chisq"`` or ``"gsq"``, if ``data`` holds a value that is not a whole number.

    See Also
    --------
    andrey.pc : Orients more of the graph, with fewer tests, when every common cause is measured.

    Notes
    -----
    **When to use it.** FCI suits data that may be affected by unmeasured common causes or sample
    selection. Its default Fisher-Z test assumes continuous data with linear effects and roughly
    Gaussian noise; for categorical data, use ``"chisq"`` or ``"gsq"``, which read the data as
    :func:`~andrey.pc` describes. Its PAG claims less than PC's CPDAG: an arrowhead at a node rules
    out that node causing the node at the edge's other end, and a circle leaves the mark open.

    **When not to.** When every common cause is measured, :func:`~andrey.pc` orients more of the
    graph with fewer tests. FCI's possible-d-separation step runs tests PC does not.

    **Choosing alpha.** As for :func:`~andrey.pc`, a larger ``alpha`` keeps more edges, true and
    false.

    **Choosing collider_rule.** ``"sepsets"`` uses the separating set recorded when the edge was
    removed. ``"majority"`` tests the pair again given every subset of either endpoint's neighbors
    and follows the majority [4]_, at the cost of the extra tests described under the parameter.

    **How it works.** FCI runs in three steps. First, it removes edges by conditional-independence
    tests, as PC does, and orients colliders from the separating sets. Next, it tests each remaining
    pair given subsets of its possible-d-separation set: nodes that can separate the pair when some
    causes are unmeasured. It removes pairs found independent and orients colliders again. Last, it
    applies Zhang's orientation rules R1-R10 [3]_ until no mark changes. An endpoint no rule
    resolves stays a circle.

    The result can depend on the column order, for example, through the possible-d-separation step.
    Keep a fixed column order for reproducible results.

    Rule R5 turns a cycle of circle-circle edges into undirected edges only if the whole cycle,
    including the closing edge, is uncovered, that is, no two nodes two steps apart are adjacent.

    **Limitations.** Measured on simulated linear-Gaussian data from sparse random graphs (mean
    degree 2) with one variable in ten hidden, from 20 to 800 variables counting the hidden ones,
    with ten rows per variable. Its median endpoint errors, each mismatched tail, arrowhead, or
    circle, stayed below the empty graph's up to 400 variables (836 against 884) and passed it at
    800 (2,579 against 1,772). The median fit took 0.061 s at 20 variables and 72 s at 800, on 16
    cores. On the 223-variable ANDES network with 1,000 simulated rows, its median endpoint errors
    passed the empty graph's (696 against 676).

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

    X = data_matrix(data)
    alpha = open_unit_interval(alpha, "alpha")
    if indep_test in DISCRETE_INDEP_TESTS:
        check_categories(X, data, f"indep_test={indep_test!r}")
    labels = column_labels(data)
    return structure_output(
        _fci(X, alpha=alpha, indep_test=indep_test, collider_rule=collider_rule),
        labels=labels,
        metadata={"algorithm": "FCI"},
    )


def gfci(
    data: npt.ArrayLike,
    *,
    score_func: str = "local_score_BIC",
    lambda_value: float = 1.0,
    indep_test: str = "fisherz",
    alpha: float = 0.05,
    collider_rule: str = "sepsets",
) -> StructureOutput:
    """Learn a PAG from observational data with GFCI, a score-then-FCI hybrid.

    GFCI [1]_ builds a skeleton with score-based GES, then applies FCI orientation to account for
    possible latent confounding. Its scored skeleton is usually more accurate than FCI's test-only
    one.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``.
    score_func : {"local_score_BIC", "local_score_BDeu"}, default="local_score_BIC"
        Local score driving the GES skeleton phase: the linear-Gaussian BIC, or BDeu for
        categorical data, coded by whole numbers, as for :func:`~andrey.ges`.
    lambda_value : float, default=1.0
        Multiplier on the BIC penalty of ``log(n_samples)`` per parent in the GES phase. Larger
        values give sparser graphs; ``1.0`` is the standard BIC. Must be finite and non-negative.
        BDeu ignores it.
    indep_test : {"fisherz", "chisq", "gsq"}, default="fisherz"
        Conditional-independence test for the FCI phase. ``"fisherz"`` tests partial correlation
        (Fisher's z), for continuous data. ``"chisq"`` (Pearson's chi-square) and ``"gsq"``
        (G-square, the likelihood ratio) test categorical data, coded by whole numbers, as for
        :func:`~andrey.pc`.
    alpha : float, default=0.05
        Significance level of the conditional-independence test. Smaller values give sparser graphs.
    collider_rule : {"sepsets", "majority"}, default="sepsets"
        How GFCI decides whether ``z`` is a collider on a triple ``x - z - y`` where ``x`` and ``y``
        are not adjacent. ``"sepsets"`` makes it a collider when ``z`` is not in the recorded
        separating set of ``x`` and ``y``. ``"majority"`` tests ``x`` and ``y`` given every subset
        of ``x``'s neighbors and every subset of ``y``'s neighbors, and makes it a collider when
        fewer than half of the separating subsets (``p >= alpha``) contain ``z`` [3]_; at exactly
        half the triple stays unoriented. Majority runs up to ``2**degree`` extra tests per triple.

    Returns
    -------
    StructureOutput
        Wraps a ``pag`` :class:`~andrey.core.GraphStructure`; ``metadata["algorithm"]`` is
        ``"GFCI"``. Endpoints carry arrowheads, tails, or circle marks (``o``); a bidirected edge
        (``<->``) marks a probable latent common cause.

    Raises
    ------
    NotImplementedError
        If ``score_func`` names no available score, or ``indep_test`` no available test.
    ValueError
        If ``alpha`` is not in ``(0, 1)``, if ``lambda_value`` is negative or non-finite, if
        ``collider_rule`` is not a supported choice, if one of ``score_func`` and ``indep_test`` is
        categorical and the other is not, or, with categorical settings, if ``data`` holds a value
        that is not a whole number.

    Warns
    -----
    ExperimentalWarning
        On the first call in a process, because the method is experimental: it has no published
        benchmark, and its API may change without deprecation.

    Notes
    -----
    GFCI keeps the adjacencies of the GES CPDAG and tests each non-adjacent pair given subsets of
    the two endpoints' neighbors to find its separating set. FCI's possible-d-separation step can
    remove more edges but never adds one, so the PAG's adjacencies are a subset of the GES CPDAG's.
    Zhang's rules R1-R10 [2]_ then orient the endpoints; an endpoint no rule resolves stays a
    circle.

    Rule R5 turns a cycle of circle-circle edges into undirected edges only when the whole cycle,
    closing edge included, is uncovered, that is, no two nodes two steps apart on it are adjacent.

    **Limitations.** Not benchmarked yet. Its tests run it on at most 12 variables with 600 rows,
    and on 1,200 rows with 7 variables. It runs a full :func:`~andrey.ges` search before any test,
    so it takes at least as long as GES on the same data. Its BIC score and Fisher-Z tests assume
    linear relations with Gaussian noise.

    References
    ----------
    .. [1] Ogarrio, J. M., Spirtes, P., and Ramsey, J. (2016). A hybrid causal search algorithm
       for latent variable models. PGM, PMLR 52, 368-379.
    .. [2] Zhang, J. (2008). On the completeness of orientation rules for causal discovery in the
       presence of latent confounders and selection bias. Artificial Intelligence 172(16-17),
       1873-1896.
    .. [3] Colombo, D., and Maathuis, M. H. (2014). Order-independent constraint-based causal
       structure learning. JMLR 15, 3921-3962.

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
    warn_experimental("gfci")
    from andrey.search.gfci import gfci as _gfci

    X = data_matrix(data)
    alpha = open_unit_interval(alpha, "alpha")
    # Both phases read the same data, so the score and the test must agree on its type.
    if (score_func == "local_score_BDeu") != (indep_test in DISCRETE_INDEP_TESTS):
        raise ValueError(
            "gfci's score and test must read the data alike: score_func='local_score_BDeu' with "
            "indep_test 'chisq' or 'gsq' for categories, or a BIC score with 'fisherz'; got "
            f"score_func={score_func!r}, indep_test={indep_test!r}"
        )
    if indep_test in DISCRETE_INDEP_TESTS:
        check_categories(X, data, f"indep_test={indep_test!r}")
    labels = column_labels(data)
    pag = _gfci(
        X,
        score_func=score_func,
        lambda_value=lambda_value,
        alpha=alpha,
        indep_test=indep_test,
        collider_rule=collider_rule,
    )
    return structure_output(pag, labels=labels, metadata={"algorithm": "GFCI"})


def cdnod(
    data: npt.ArrayLike,
    c_indx: npt.ArrayLike,
    *,
    alpha: float = 0.05,
    indep_test: str = "fisherz",
) -> StructureOutput:
    """Learn a CPDAG over the data variables from nonstationary / multi-domain data with CDNOD.

    CDNOD [1]_ adds a domain/context index as an extra causal parent, so distribution shifts
    across regimes are modeled, then runs PC-style discovery. Use it when samples are pooled across
    domains, regimes, or time blocks. The index is appended as a trailing pseudo-node and removed
    from the result.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``.
    c_indx : array-like of shape (n_samples,) or (n_samples, 1)
        Domain or context index of each row of ``data``, such as ``0`` and ``1`` for two domains, or
        the time step for data that drift over time. Coerced to ``float64``; whole numbers with
        ``"chisq"`` or ``"gsq"``. See Notes for more than two domains. When every row has the same
        value, the data come from one domain, and the result equals :func:`~andrey.pc`'s.
    alpha : float, default=0.05
        Significance level of the conditional-independence test. Smaller values give sparser graphs.
    indep_test : {"fisherz", "chisq", "gsq"}, default="fisherz"
        Conditional-independence test. ``"fisherz"`` tests partial correlation (Fisher's z), for
        continuous data. ``"chisq"`` (Pearson's chi-square) and ``"gsq"`` (G-square, the
        likelihood ratio) test categorical data, coded by whole numbers, as for
        :func:`~andrey.pc`; see Notes for the index.

    Returns
    -------
    StructureOutput
        Wraps a ``cpdag`` :class:`~andrey.core.GraphStructure` over the ``d`` data variables;
        ``metadata["algorithm"]`` is ``"CDNOD"``. Directed marks (``->``) are oriented edges,
        undirected (``--``) the ones the equivalence class leaves open.

    Raises
    ------
    NotImplementedError
        If ``indep_test`` names no available test.
    ValueError
        If ``c_indx`` does not hold one value per row of ``data`` or holds ``NaN`` / ``inf``, if
        ``data`` has too few rows for a test, if ``alpha`` is not in ``(0, 1)``, or, with
        ``"chisq"`` or ``"gsq"``, if ``data`` or ``c_indx`` holds a value that is not a whole
        number.

    Warns
    -----
    ExperimentalWarning
        On the first call in a process, because the method is experimental: it has no published
        benchmark, and its API may change without deprecation.

    Notes
    -----
    The index is added as a variable that can only be a cause. When a variable's distribution
    changes with the index, the edge from the index into it is directed, and Meek's rules can then
    orient edges that PC leaves undirected on the same data.

    Fisher-Z tests the index as a number, by linear correlation, so it finds a shift only when the
    shift follows the index's values. With two domains the coding does not matter. With three
    domains coded ``0, 1, 2``, a shift in domain ``1`` alone can go undetected, and recoding the
    domains can change the result. ``"chisq"`` and ``"gsq"`` take each index value as a category,
    so they find a shift in any domain, whatever the coding.

    **Limitations.** Not benchmarked yet. Its tests run it on at most 10 variables with 500 rows,
    and on 1,200 rows with 7 variables. It runs the PC search with the index as one more variable,
    so it costs about as much as :func:`~andrey.pc` on the data plus one column. Its Fisher-Z tests
    assume linear relations with Gaussian noise.

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
    >>> c_indx = np.repeat([0.0, 1.0], 300)
    >>> x = 2.0 * c_indx + rng.standard_normal(600)
    >>> y = x + 0.5 * rng.standard_normal(600)
    >>> cdnod(np.column_stack([x, y]), c_indx).structure.oriented_edges()
    [(0, 1, 'directed')]
    """
    warn_experimental("cdnod")
    from andrey.constraint.cdnod import cdnod as _cdnod

    X = data_matrix(data)
    alpha = open_unit_interval(alpha, "alpha")
    labels = column_labels(data)
    c = np.asarray(c_indx, dtype=np.float64)
    if indep_test in DISCRETE_INDEP_TESTS:
        check_categories(X, data, f"indep_test={indep_test!r}")
        check_categories(c, None, f"indep_test={indep_test!r}", name="c_indx")
    cpdag = _cdnod(X, c, alpha=alpha, indep_test=indep_test)
    return structure_output(cpdag, labels=labels, metadata={"algorithm": "CDNOD"})
