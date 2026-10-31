"""Local-structure facade adapters: one variable's Markov blanket or parents and children.

``iamb``, ``inter_iamb``, and ``mmpc`` share :mod:`andrey.constraint.blanket`. With ``target`` they
return that variable's set; without, every variable's, joined by the AND rule into the graph
``structure`` names: the CPDAG, the skeleton, or, for the blankets, the moral graph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from andrey.api._adapt import (
    categorical_metadata,
    checked_data,
    column_labels,
    copies_named,
    default_test,
    structure_output,
)
from andrey.api._method import method_spec
from andrey.core import NULL, TAIL
from andrey.core.ci_missing import rows_metadata, rows_tested
from andrey.core.knowledge import Knowledge, _node, column_position, resolve
from andrey.core.orient import to_structure

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from andrey.core import StructureOutput

#: The ``metadata`` key of each method's set, and the ``algorithm`` it reports.
_SETS = {"iamb": "markov_blanket", "inter_iamb": "markov_blanket", "mmpc": "parents_and_children"}


def _fit(
    method: str,
    data: npt.ArrayLike,
    *,
    target: str | int | None,
    structure: str,
    alpha: float,
    indep_test: str | None,
    categorical: Sequence[str | int] | None,
    seed: int | None,
    max_depth: int | None,
    knowledge: Knowledge | None,
) -> StructureOutput:
    """Check the settings, run the search, and wrap the graph ``structure`` names or the set."""
    from andrey.search.hybrid import local_set, local_structure
    from andrey.spec.concepts import BLANKET_STRUCTURES

    indep_test, categorical = default_test(data, indep_test, categorical=categorical)
    X, mask = checked_data(
        data,
        categorical,
        "indep_test",
        indep_test,
        method=method,
        missing=True,
    )
    labels = column_labels(data)
    d = X.shape[1]
    t = None if target is None else column_position(_node(target, "target"), labels, d, "target")
    constraints = resolve(knowledge, d, labels, method=method)
    settings = {
        "alpha": alpha,
        "indep_test": indep_test,
        "categorical": mask,
        "seed": seed,
        "max_depth": max_depth,
        "constraints": constraints,
    }
    metadata: dict[str, object] = {}
    kind = BLANKET_STRUCTURES[structure] if t is None else "undirected"
    with copies_named(labels), rows_tested() as rows:
        if t is None:
            marks = local_structure(X, method, structure, **settings)
        else:
            members = local_set(X, method, t, **settings)
            adjacent = np.zeros((d, d), dtype=bool)
            adjacent[t, members] = adjacent[members, t] = True
            marks = np.where(adjacent, TAIL, NULL).astype(np.int8)
            name = (lambda j: labels[j]) if labels is not None else int
            metadata |= {"target": name(t), _SETS[method]: [name(j) for j in members]}
    metadata |= categorical_metadata(mask, data) | rows_metadata(rows)
    return structure_output(to_structure(marks, kind=kind), labels=labels, metadata=metadata)


@method_spec("iamb")
def iamb(
    data: npt.ArrayLike,
    *,
    target: str | int | None = None,
    structure: str = "cpdag",
    alpha: float = 0.05,
    indep_test: str | None = None,
    categorical: Sequence[str | int] | None = None,
    seed: int | None = None,
    max_depth: int | None = None,
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn a variable's Markov blanket with incremental association (IAMB).

    A variable's Markov blanket is the smallest set of other variables given which it is
    independent of all the rest: in a causal graph, its parents, its children, and its children's
    other parents. IAMB [1]_ grows the blanket by conditional-independence tests, then shrinks it.
    Without a target, every variable's blanket is learned and joined into the graph ``structure``
    names, by default the CPDAG of the Grow-Shrink structure learner [2]_ [3]_.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``.
    target : str, int, or None, default=None
        The variable whose set to learn: a column label (a string) or a 0-based position (an
        integer). ``None`` learns every variable's set and keeps a pair when each variable is in
        the other's set.
    structure : {"cpdag", "skeleton", "moral"}, default="cpdag"
        The graph returned when ``target`` is ``None``. ``"cpdag"``: the CPDAG, the causal graphs
        the tests cannot tell apart, whose directed edges they all share. ``"skeleton"``: the same
        adjacencies with no directions, each variable joined to its parents and children.
        ``"moral"``: the moral graph, each variable joined to every member of its Markov blanket,
        a child's other parents included, for feature selection or as a superset of the skeleton;
        an edge there does not say that the two variables are adjacent. With ``target``, the result
        is the target's Markov blanket, whatever ``structure`` is.
    alpha : float, default=0.05
        Significance level of the conditional-independence test. Smaller values give smaller sets.
    indep_test : str or None, default=None
        Conditional-independence test, any of :func:`~andrey.pc`'s: ``"fisherz"``,
        ``"student_t"``, or ``"normal_scores"`` for continuous data, ``"chisq"`` or ``"gsq"`` for
        categorical data, ``"degen_gauss"`` or ``"cond_gauss"`` for mixed data, or a nonlinear
        test. ``None`` picks from the declared column types, as for :func:`~andrey.pc`.
        ``metadata["indep_test"]`` names the test the fit used: the one picked from the
        declared column types when this is ``None``.
    categorical : sequence of str or int, optional
        The categorical columns of ``data``, for a mixed-data test, as for :func:`~andrey.pc`. A
        continuous test raises ``ValueError`` on a declared column.
    seed : int or None, default=None
        ``None``, the default, is the configured seed, ``ANDREY_SEED`` or ``0`` when it is unset,
        not a fresh random one, so a fit repeats. Seed for the random draws of ``indep_test``, which
        only ``"rcot"``, ``"cmiknn"``, and ``"gpdc"`` make; the other tests ignore it. With those
        tests, ``metadata["seed"]`` records the seed used.
    max_depth : int or None, default=None
        The maximum conditioning-set size of every conditional-independence test, at least ``0``:
        ``0`` runs only unconditional tests, and ``None`` sets no limit. A smaller cap runs fewer
        tests and can keep variables that a larger conditioning set would remove. Every test
        conditions on the whole current blanket, so the blanket stops growing once it holds more
        than ``max_depth`` members, counting those ``knowledge`` requires. A collider test
        conditions on the middle node as well, so with ``0`` no triple is decided and the CPDAG
        directs only the edges ``knowledge`` decides.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the pairs
        allowed to be adjacent, and temporal tiers, all honored. Each end of a required edge is in
        the other's blanket from the start, and no test removes it. A pair the knowledge makes
        non-adjacent, forbidden both ways or outside ``allowed``, stays a candidate, since a child's
        other parent belongs to the blanket without being adjacent, and is left out of the result.
        Under ``structure="cpdag"``, the result directs every edge the knowledge
        decides, by a tier or an edge forbidden one way, before the colliders, which never reverse
        it, and Meek's rules direct the edges that follow, giving the maximally oriented PDAG
        (MPDAG) :class:`~andrey.Knowledge` describes; knowledge that orients the edges the search
        kept into a directed cycle raises ``ValueError``. The undirected outputs do not read tiers
        or edges forbidden one way.

    Returns
    -------
    StructureOutput
        Without ``target``, wraps the graph ``structure`` names: a ``cpdag``
        :class:`~andrey.core.GraphStructure` for ``"cpdag"``, whose undirected edges the tests
        cannot orient, or an ``undirected`` one, whose edges have no arrowhead, for ``"skeleton"``
        and ``"moral"``. With ``target``, an ``undirected`` one whose edges join ``target`` to each
        member of its Markov blanket, which holds its parents, its children, and its children's
        other parents; ``metadata["target"]`` names it and ``metadata["markov_blanket"]`` lists the
        members in the order they joined, by column label or 0-based position.
        ``metadata["algorithm"]`` is ``"iamb"``. Under a mixed-data setting,
        ``metadata["categorical"]`` lists the categorical columns it read.
        With a randomized ``indep_test``, ``metadata["seed"]`` is the seed used.

    Raises
    ------
    TypeError
        If ``target`` is neither a column label nor a position, ``categorical`` is not a list of
        column names and positions, or ``knowledge`` is neither ``None`` nor a
        :class:`~andrey.Knowledge`.
    ValueError
        If an argument has the wrong type, lies outside its range, or is not one of its choices; the
        message names the method, the argument, what it accepts, and the value. Also if ``target``
        names a column ``data`` lacks, or if ``data`` or ``categorical`` does not suit the test, as
        for :func:`~andrey.pc`. With ``"fisherz"`` or ``"student_t"``, also if a column repeats
        another up to scale and shift. Also if ``knowledge`` names a column ``data`` lacks or
        contradicts itself, as :class:`~andrey.Knowledge` lists, or under ``structure="cpdag"``
        orients the edges the search kept into a directed cycle.

    Warns
    -----
    DataTypeWarning
        As for :func:`~andrey.pc`, when a mixed-data test finds no declared categorical column or a
        setting reads columns that look like category codes as numbers; once per process.

    See Also
    --------
    andrey.inter_iamb : Shrinks after each addition.
    andrey.mmpc : The parents and children alone.

    Notes
    -----
    **How it works.** Grow: add the candidate most dependent on the target given the current
    blanket, among those whose p-value is at or below ``alpha``, until none is. Shrink: visit the
    members in the order they joined and drop each one the test finds independent of the target
    given the other current members. With exact tests, the result is the true Markov blanket [1]_.

    Candidates whose strengths agree to nine decimals go to the column whose rank, computed from the
    data, comes first, as for :func:`~andrey.hc`, so the sets do not depend on the column order
    unless two columns are identical. The strength is the absolute partial correlation for
    ``"fisherz"`` and ``"student_t"``, which keeps its order where p-values underflow to zero, and
    the p-value for the other tests. Where the p-values of several candidates underflow to zero, the
    candidate with the strongest statistic joins first, as the association measure of [1]_ orders
    them.

    **From blankets to a graph.** Without a target, a pair is kept when each variable is in the
    other's blanket, which gives the moral graph (``structure="moral"``). The skeleton drops each
    pair ``x - y`` that tests independent given some subset, the empty set included, of the smaller
    of ``x``'s blanket without ``y`` and ``y``'s blanket without ``x``, or of either when both hold
    as many members [2]_ [3]_: a child's other parent is in the blanket without being adjacent, and
    such a subset separates the two. Every pair is tested against the moral graph, so no verdict
    depends on another. The CPDAG decides each triple ``x - z - y`` of the skeleton with ``x`` and
    ``y`` not adjacent: ``z`` is a collider, ``x -> z <- y``, when ``x`` and ``y`` test dependent
    given ``z`` together with every subset of the smaller of their blankets without ``x``, ``y``,
    and ``z`` [2]_, and a non-collider when one such set separates them. The colliders are oriented
    as :func:`~andrey.pc` orients them, the most clearly dependent first: by the smallest absolute
    partial correlation over the collider's tests for ``"fisherz"`` and ``"student_t"``, and by the
    largest p-value, smallest first, for the other tests. When two colliders that rank equal to nine
    decimals dispute an edge, it stays undirected. Meek's rules [4]_ then orient the edges that
    follow, in rounds, as for :func:`~andrey.pc`, so the result does not depend on the column order;
    under ``andrey.config(tie_break="position")`` the colliders and the rules follow the column
    order instead. With exact tests, the result is the true CPDAG [2]_.

    **Missing values.** A ``NaN`` in ``data`` marks a missing value. Each test reads only the rows
    where all of its variables are observed (test-wise deletion) [5]_ [6]_, so it keeps rows that
    deleting every incomplete row first would drop; ``metadata["rows_tested"]`` gives the fewest and
    the median rows a test read. When values are missing completely at random, the rows a test keeps
    are a random sample, so each test keeps its level [6]_. When whether a value is missing depends
    on the variables, the rows a test keeps are a selected sample, and the tests can find dependence
    where the population has none and keep extra edges [6]_. A test raises ``ValueError`` when it
    has fewer rows than it needs, ``|S| + 3`` for ``"fisherz"`` and ``"student_t"`` and 2 for the
    other tests, or when a continuous column is constant on its rows.

    **Limitations.** Measured on simulated linear-Gaussian data with Fisher-z tests at alpha 0.05,
    on one core: 20, 50, and 100 variables with 1,000 and 5,000 rows and mean degree 2 and 4, with
    the columns standardized and not, and 200 variables with the same rows and degrees. In every
    setting the mean errors of the CPDAG, the skeleton, and the moral graph stay below the empty
    graph's. The CPDAG is closest to it on unstandardized data at 100 variables, 1,000 rows, and
    mean degree 2, with 79 pairs wrong against 100 for the empty graph, and on standardized data at
    20 variables, 1,000 rows, and mean degree 4, with 24 against 40. The median CPDAG fit took 0.03
    to 6.1 s up to 100 variables, the longest at 5,000 rows and mean degree 4, and 13 to 17 s at 200
    variables. The benchmark covers continuous data; the categorical and mixed-data tests are not in
    it yet. Tested with exact d-separation on 15 variables. It conditions on the whole current
    blanket, so each test needs more rows as the blanket grows, and a candidate the grow step adds
    in error enlarges every later conditioning set. The skeleton and the CPDAG test every subset of
    the smaller blanket for each pair of the moral graph and each unshielded triple, and a collider
    takes all of them, so a triple whose smaller blanket holds ``m`` variables takes up to ``2**m``
    tests. On simulated linear-Gaussian data with 50 variables, 5,000 rows, and mean degree 4, the
    smaller blankets held at most 16 variables, the collider tests took 0.07 to 0.7 s on five
    datasets, and learning the blankets 0.8 to 1.0 s. Missing values are not benchmarked yet; the
    tests run them on up to 20 variables with up to a fifth of the values missing completely at
    random.

    References
    ----------
    .. [1] Tsamardinos, I., Aliferis, C. F., and Statnikov, A. (2003). Algorithms for large scale
       Markov blanket discovery. FLAIRS, 376-380.
    .. [2] Margaritis, D. (2003). Learning Bayesian network model structure from data. PhD thesis,
       Carnegie Mellon University, CMU-CS-03-153.
    .. [3] Margaritis, D., and Thrun, S. (2000). Bayesian network induction via local
       neighborhoods. Advances in Neural Information Processing Systems 12, 505-511.
    .. [4] Meek, C. (1995). Causal inference and causal explanation with background knowledge.
       UAI, 403-410.
    .. [5] Strobl, E. V., Visweswaran, S., and Spirtes, P. L. (2018). Fast causal inference with
       non-random missingness by test-wise deletion. International Journal of Data Science and
       Analytics 6(1), 47-62.
    .. [6] Tu, R., Zhang, C., Ackermann, P., Mohan, K., Kjellstrom, H., and Zhang, K. (2019). Causal
       discovery in the presence of missing data. Proceedings of Machine Learning Research 89,
       1762-1770.

    Examples
    --------
    >>> import warnings
    >>> import numpy as np
    >>> from andrey import iamb
    >>> rng = np.random.default_rng(0)
    >>> a, b = rng.standard_normal((2, 1000))
    >>> c = a + b + 0.5 * rng.standard_normal(1000)
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     out = iamb(np.column_stack([a, b, c]), target=0)
    ...     graph = iamb(np.column_stack([a, b, c]))
    >>> sorted(out.metadata["markov_blanket"])
    [1, 2]
    >>> graph.structure.oriented_edges()
    [(0, 2, 'directed'), (1, 2, 'directed')]
    """
    return _fit(
        "iamb",
        data,
        target=target,
        structure=structure,
        alpha=alpha,
        indep_test=indep_test,
        categorical=categorical,
        seed=seed,
        max_depth=max_depth,
        knowledge=knowledge,
    )


@method_spec("inter_iamb")
def inter_iamb(
    data: npt.ArrayLike,
    *,
    target: str | int | None = None,
    structure: str = "cpdag",
    alpha: float = 0.05,
    indep_test: str | None = None,
    categorical: Sequence[str | int] | None = None,
    seed: int | None = None,
    max_depth: int | None = None,
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn a variable's Markov blanket with interleaved IAMB (Inter-IAMB).

    Inter-IAMB [1]_ is IAMB with its shrink step after every addition, so a member that becomes
    independent leaves before it enlarges the later conditioning sets. Without a target, every
    variable's blanket is learned and joined into the graph ``structure`` names, by default the
    CPDAG of the Grow-Shrink structure learner [2]_ [3]_.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``.
    target : str, int, or None, default=None
        The variable whose set to learn: a column label (a string) or a 0-based position (an
        integer). ``None`` learns every variable's set and keeps a pair when each variable is in
        the other's set.
    structure : {"cpdag", "skeleton", "moral"}, default="cpdag"
        The graph returned when ``target`` is ``None``. ``"cpdag"``: the CPDAG, the causal graphs
        the tests cannot tell apart, whose directed edges they all share. ``"skeleton"``: the same
        adjacencies with no directions, each variable joined to its parents and children.
        ``"moral"``: the moral graph, each variable joined to every member of its Markov blanket,
        a child's other parents included, for feature selection or as a superset of the skeleton;
        an edge there does not say that the two variables are adjacent. With ``target``, the result
        is the target's Markov blanket, whatever ``structure`` is.
    alpha : float, default=0.05
        Significance level of the conditional-independence test. Smaller values give smaller sets.
    indep_test : str or None, default=None
        Conditional-independence test, any of :func:`~andrey.pc`'s: ``"fisherz"``,
        ``"student_t"``, or ``"normal_scores"`` for continuous data, ``"chisq"`` or ``"gsq"`` for
        categorical data, ``"degen_gauss"`` or ``"cond_gauss"`` for mixed data, or a nonlinear
        test. ``None`` picks from the declared column types, as for :func:`~andrey.pc`.
        ``metadata["indep_test"]`` names the test the fit used: the one picked from the
        declared column types when this is ``None``.
    categorical : sequence of str or int, optional
        The categorical columns of ``data``, for a mixed-data test, as for :func:`~andrey.pc`. A
        continuous test raises ``ValueError`` on a declared column.
    seed : int or None, default=None
        ``None``, the default, is the configured seed, ``ANDREY_SEED`` or ``0`` when it is unset,
        not a fresh random one, so a fit repeats. Seed for the random draws of ``indep_test``, which
        only ``"rcot"``, ``"cmiknn"``, and ``"gpdc"`` make; the other tests ignore it. With those
        tests, ``metadata["seed"]`` records the seed used.
    max_depth : int or None, default=None
        The maximum conditioning-set size of every conditional-independence test, at least ``0``:
        ``0`` runs only unconditional tests, and ``None`` sets no limit. A smaller cap runs fewer
        tests and can keep variables that a larger conditioning set would remove. Every test
        conditions on the whole current blanket, so the blanket stops growing once it holds more
        than ``max_depth`` members, counting those ``knowledge`` requires. A collider test
        conditions on the middle node as well, so with ``0`` no triple is decided and the CPDAG
        directs only the edges ``knowledge`` decides.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the pairs
        allowed to be adjacent, and temporal tiers, all honored. Each end of a required edge is in
        the other's blanket from the start, and no test removes it. A pair the knowledge makes
        non-adjacent, forbidden both ways or outside ``allowed``, stays a candidate, since a child's
        other parent belongs to the blanket without being adjacent, and is left out of the result.
        Under ``structure="cpdag"``, the result directs every edge the knowledge
        decides, by a tier or an edge forbidden one way, before the colliders, which never reverse
        it, and Meek's rules direct the edges that follow, giving the maximally oriented PDAG
        (MPDAG) :class:`~andrey.Knowledge` describes; knowledge that orients the edges the search
        kept into a directed cycle raises ``ValueError``. The undirected outputs do not read tiers
        or edges forbidden one way.

    Returns
    -------
    StructureOutput
        Without ``target``, wraps the graph ``structure`` names: a ``cpdag``
        :class:`~andrey.core.GraphStructure` for ``"cpdag"``, whose undirected edges the tests
        cannot orient, or an ``undirected`` one, whose edges have no arrowhead, for ``"skeleton"``
        and ``"moral"``. With ``target``, an ``undirected`` one whose edges join ``target`` to each
        member of its Markov blanket, which holds its parents, its children, and its children's
        other parents; ``metadata["target"]`` names it and ``metadata["markov_blanket"]`` lists the
        members in the order they joined, by column label or 0-based position.
        ``metadata["algorithm"]`` is ``"inter_iamb"``. Under a mixed-data setting,
        ``metadata["categorical"]`` lists the categorical columns it read.
        With a randomized ``indep_test``, ``metadata["seed"]`` is the seed used.

    Raises
    ------
    TypeError
        If ``target`` is neither a column label nor a position, ``categorical`` is not a list of
        column names and positions, or ``knowledge`` is neither ``None`` nor a
        :class:`~andrey.Knowledge`.
    ValueError
        If an argument has the wrong type, lies outside its range, or is not one of its choices; the
        message names the method, the argument, what it accepts, and the value. Also if ``target``
        names a column ``data`` lacks, or if ``data`` or ``categorical`` does not suit the test, as
        for :func:`~andrey.pc`. With ``"fisherz"`` or ``"student_t"``, also if a column repeats
        another up to scale and shift. Also if ``knowledge`` names a column ``data`` lacks or
        contradicts itself, as :class:`~andrey.Knowledge` lists, or under ``structure="cpdag"``
        orients the edges the search kept into a directed cycle.

    Warns
    -----
    DataTypeWarning
        As for :func:`~andrey.pc`, when a mixed-data test finds no declared categorical column or a
        setting reads columns that look like category codes as numbers; once per process.

    See Also
    --------
    andrey.iamb : Shrinks once, after the blanket stops growing.

    Notes
    -----
    **How it works.** Each step adds the candidate most dependent on the target given the current
    blanket, among those whose p-value is at or below ``alpha``, then visits the members in the
    order they joined and drops each one the test finds independent of the target given the other
    current members. The search stops when no candidate qualifies, or when the blanket after a
    step repeats one it held after an earlier step, which on sample data could otherwise make it
    cycle. With exact tests, the result is the true Markov blanket [1]_.

    Candidates whose strengths agree to nine decimals go to the column whose rank, computed from the
    data, comes first, as for :func:`~andrey.hc`, so the sets do not depend on the column order
    unless two columns are identical. The strength is the absolute partial correlation for
    ``"fisherz"`` and ``"student_t"``, which keeps its order where p-values underflow to zero, and
    the p-value for the other tests.

    **From blankets to a graph.** Without a target, a pair is kept when each variable is in the
    other's blanket, which gives the moral graph (``structure="moral"``). The skeleton drops each
    pair ``x - y`` that tests independent given some subset, the empty set included, of the smaller
    of ``x``'s blanket without ``y`` and ``y``'s blanket without ``x``, or of either when both hold
    as many members [2]_ [3]_: a child's other parent is in the blanket without being adjacent, and
    such a subset separates the two. Every pair is tested against the moral graph, so no verdict
    depends on another. The CPDAG decides each triple ``x - z - y`` of the skeleton with ``x`` and
    ``y`` not adjacent: ``z`` is a collider, ``x -> z <- y``, when ``x`` and ``y`` test dependent
    given ``z`` together with every subset of the smaller of their blankets without ``x``, ``y``,
    and ``z`` [2]_, and a non-collider when one such set separates them. The colliders are oriented
    as :func:`~andrey.pc` orients them, the most clearly dependent first: by the smallest absolute
    partial correlation over the collider's tests for ``"fisherz"`` and ``"student_t"``, and by the
    largest p-value, smallest first, for the other tests. When two colliders that rank equal to nine
    decimals dispute an edge, it stays undirected. Meek's rules [4]_ then orient the edges that
    follow, in rounds, as for :func:`~andrey.pc`, so the result does not depend on the column order;
    under ``andrey.config(tie_break="position")`` the colliders and the rules follow the column
    order instead. With exact tests, the result is the true CPDAG [2]_.

    **Missing values.** A ``NaN`` in ``data`` marks a missing value. Each test reads only the rows
    where all of its variables are observed (test-wise deletion) [5]_ [6]_, so it keeps rows that
    deleting every incomplete row first would drop; ``metadata["rows_tested"]`` gives the fewest and
    the median rows a test read. When values are missing completely at random, the rows a test keeps
    are a random sample, so each test keeps its level [6]_. When whether a value is missing depends
    on the variables, the rows a test keeps are a selected sample, and the tests can find dependence
    where the population has none and keep extra edges [6]_. A test raises ``ValueError`` when it
    has fewer rows than it needs, ``|S| + 3`` for ``"fisherz"`` and ``"student_t"`` and 2 for the
    other tests, or when a continuous column is constant on its rows.

    **Limitations.** Measured on simulated linear-Gaussian data with 20, 50, and 100 variables,
    1,000 and 5,000 rows, and mean degree 2 and 4, with Fisher-z tests at alpha 0.05, on one core.
    In every setting the mean errors of the CPDAG, the skeleton, and the moral graph stay below
    the empty graph's; the CPDAG is closest at 20 variables, 1,000 rows, and mean degree 4, with
    24 pairs wrong against 40 for the empty graph. A fit took 0.06 to 8.9 s, the longest at 100
    variables, 5,000 rows, and mean degree 4. Tested with exact d-separation on 15 variables. It
    conditions on the whole current blanket, so each test needs more rows as the blanket grows.
    The skeleton and the CPDAG test every subset of the smaller blanket for each pair of the moral
    graph and each unshielded triple, and a collider takes all of them, so a triple whose smaller
    blanket holds ``m`` variables takes up to ``2**m`` tests. On simulated linear-Gaussian data
    with 50 variables, 5,000 rows, and mean degree 4, the smaller blankets held at most 16
    variables, the collider tests took 0.06 to 0.7 s on five datasets, and learning the blankets
    0.9 to 1.3 s. Missing values are not benchmarked yet; the tests run them on up to 20 variables
    with up to a fifth of the values missing completely at random.

    References
    ----------
    .. [1] Tsamardinos, I., Aliferis, C. F., and Statnikov, A. (2003). Algorithms for large scale
       Markov blanket discovery. FLAIRS, 376-380.
    .. [2] Margaritis, D. (2003). Learning Bayesian network model structure from data. PhD thesis,
       Carnegie Mellon University, CMU-CS-03-153.
    .. [3] Margaritis, D., and Thrun, S. (2000). Bayesian network induction via local
       neighborhoods. Advances in Neural Information Processing Systems 12, 505-511.
    .. [4] Meek, C. (1995). Causal inference and causal explanation with background knowledge.
       UAI, 403-410.
    .. [5] Strobl, E. V., Visweswaran, S., and Spirtes, P. L. (2018). Fast causal inference with
       non-random missingness by test-wise deletion. International Journal of Data Science and
       Analytics 6(1), 47-62.
    .. [6] Tu, R., Zhang, C., Ackermann, P., Mohan, K., Kjellstrom, H., and Zhang, K. (2019). Causal
       discovery in the presence of missing data. Proceedings of Machine Learning Research 89,
       1762-1770.

    Examples
    --------
    >>> import warnings
    >>> import numpy as np
    >>> from andrey import inter_iamb
    >>> rng = np.random.default_rng(0)
    >>> a, b = rng.standard_normal((2, 1000))
    >>> c = a + b + 0.5 * rng.standard_normal(1000)
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     out = inter_iamb(np.column_stack([a, b, c]), target=0)
    >>> sorted(out.metadata["markov_blanket"])
    [1, 2]
    """
    return _fit(
        "inter_iamb",
        data,
        target=target,
        structure=structure,
        alpha=alpha,
        indep_test=indep_test,
        categorical=categorical,
        seed=seed,
        max_depth=max_depth,
        knowledge=knowledge,
    )


@method_spec("mmpc")
def mmpc(
    data: npt.ArrayLike,
    *,
    target: str | int | None = None,
    structure: str = "skeleton",
    alpha: float = 0.05,
    indep_test: str | None = None,
    categorical: Sequence[str | int] | None = None,
    seed: int | None = None,
    max_depth: int | None = None,
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn a variable's parents and children with max-min parents and children (MMPC).

    MMPC [1]_ finds the variables adjacent to a target in the causal graph, its parents and
    children, without telling them apart, by conditional-independence tests. Every variable's
    set together gives the skeleton :func:`~andrey.mmhc` searches, which ``structure="cpdag"``
    orients into a CPDAG by the collider tests of the Grow-Shrink structure learner [2]_ [3]_.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``.
    target : str, int, or None, default=None
        The variable whose set to learn: a column label (a string) or a 0-based position (an
        integer). ``None`` learns every variable's set and keeps a pair when each variable is in
        the other's set.
    structure : {"skeleton", "cpdag"}, default="skeleton"
        The graph returned when ``target`` is ``None``. ``"skeleton"``: the adjacencies with no
        directions, each variable joined to its parents and children, the graph the sets give.
        ``"cpdag"``: the CPDAG, the causal graphs the tests cannot tell apart, whose directed edges
        they all share, which tests every unshielded triple for a collider and costs more. A
        variable's parents and children hold no child's other parents, so there is no moral graph.
        With ``target``, the result is the target's parents and children, whatever ``structure`` is.
    alpha : float, default=0.05
        Significance level of the conditional-independence test. Smaller values give smaller sets.
    indep_test : str or None, default=None
        Conditional-independence test, any of :func:`~andrey.pc`'s: ``"fisherz"``,
        ``"student_t"``, or ``"normal_scores"`` for continuous data, ``"chisq"`` or ``"gsq"`` for
        categorical data, ``"degen_gauss"`` or ``"cond_gauss"`` for mixed data, or a nonlinear
        test. ``None`` picks from the declared column types, as for :func:`~andrey.pc`.
        ``metadata["indep_test"]`` names the test the fit used: the one picked from the
        declared column types when this is ``None``.
    categorical : sequence of str or int, optional
        The categorical columns of ``data``, for a mixed-data test, as for :func:`~andrey.pc`. A
        continuous test raises ``ValueError`` on a declared column.
    seed : int or None, default=None
        ``None``, the default, is the configured seed, ``ANDREY_SEED`` or ``0`` when it is unset,
        not a fresh random one, so a fit repeats. Seed for the random draws of ``indep_test``, which
        only ``"rcot"``, ``"cmiknn"``, and ``"gpdc"`` make; the other tests ignore it. With those
        tests, ``metadata["seed"]`` records the seed used.
    max_depth : int or None, default=None
        The maximum conditioning-set size of every conditional-independence test, at least ``0``:
        ``0`` runs only unconditional tests, and ``None`` sets no limit. A smaller cap runs fewer
        tests and can keep variables that a larger conditioning set would remove. A test
        conditions on a subset of the current set; the set itself may grow larger. A collider test
        conditions on the middle node as well, so with ``0`` no triple is decided and the CPDAG
        directs only the edges ``knowledge`` decides.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the pairs
        allowed to be adjacent, and temporal tiers, all honored. Each end of a required edge is in
        the other's set from the start, and no test removes it; a pair the knowledge makes
        non-adjacent, forbidden both ways or outside ``allowed``, is never a candidate.
        Under ``structure="cpdag"``, the result directs every edge the knowledge
        decides, by a tier or an edge forbidden one way, before the colliders, which never reverse
        it, and Meek's rules direct the edges that follow, giving the maximally oriented PDAG
        (MPDAG) :class:`~andrey.Knowledge` describes; knowledge that orients the edges the search
        kept into a directed cycle raises ``ValueError``. The undirected outputs do not read tiers
        or edges forbidden one way.

    Returns
    -------
    StructureOutput
        Without ``target``, wraps the graph ``structure`` names: an ``undirected``
        :class:`~andrey.core.GraphStructure` for ``"skeleton"``, whose edges have no arrowhead and
        join each pair in each other's set, or a ``cpdag`` one for ``"cpdag"``, whose undirected
        edges the tests cannot orient. With ``target``, an ``undirected`` one whose edges
        join ``target`` to each of its parents and children; ``metadata["target"]`` names it and
        ``metadata["parents_and_children"]`` lists the members in the order they joined, by column
        label or 0-based position. ``metadata["algorithm"]`` is ``"mmpc"``. Under a mixed-data
        setting, ``metadata["categorical"]`` lists the categorical columns it read.
        With a randomized ``indep_test``, ``metadata["seed"]`` is the seed used.

    Raises
    ------
    TypeError
        If ``target`` is neither a column label nor a position, ``categorical`` is not a list of
        column names and positions, or ``knowledge`` is neither ``None`` nor a
        :class:`~andrey.Knowledge`.
    ValueError
        If an argument has the wrong type, lies outside its range, or is not one of its choices; the
        message names the method, the argument, what it accepts, and the value. Also if ``target``
        names a column ``data`` lacks, or if ``data`` or ``categorical`` does not suit the test, as
        for :func:`~andrey.pc`. With ``"fisherz"`` or ``"student_t"``, also if a column repeats
        another up to scale and shift. Also if ``knowledge`` names a column ``data`` lacks or
        contradicts itself, as :class:`~andrey.Knowledge` lists, or under ``structure="cpdag"``
        orients the edges the search kept into a directed cycle.

    Warns
    -----
    DataTypeWarning
        As for :func:`~andrey.pc`, when a mixed-data test finds no declared categorical column or a
        setting reads columns that look like category codes as numbers; once per process.

    See Also
    --------
    andrey.mmhc : Orients the skeleton by hill climbing.
    andrey.iamb : The Markov blanket, which adds the children's other parents.

    Notes
    -----
    **How it works.** A candidate's association with the target is its weakest over every
    conditioning set drawn from the current members: the largest p-value among those tests. The
    forward phase adds, one at a time, the candidate whose weakest association is the strongest,
    and drops for good every candidate that some conditioning set separates (a p-value above
    ``alpha``). The backward phase visits the members in the order they joined and drops each one
    some subset of the other members separates. A member stays only when the target is in that
    member's own set as well, the symmetry check of [1]_, which removes a descendant of the
    target that no subset of the target's set separates from it. With exact tests, the result is
    the true set of parents and children [1]_.

    Two candidates whose largest p-values agree to nine decimals are compared by their smallest
    strength, and candidates whose strengths also agree go to the column whose rank, computed from
    the data, comes first, as for :func:`~andrey.hc`, so the sets do not depend on the column order
    unless two columns are identical. The strength is the absolute partial correlation for
    ``"fisherz"`` and ``"student_t"``, which keeps its order where p-values underflow to zero, and
    the p-value for the other tests. Where the p-values of several candidates are zero to that
    precision, the candidate with the strongest statistic joins first, as the tie rule of [1]_
    orders them.

    **From sets to a graph.** Without a target, a pair is kept when each variable is in the other's
    set, which gives the skeleton (``structure="skeleton"``). The CPDAG decides each triple
    ``x - z - y`` of the skeleton with ``x`` and ``y`` not adjacent: ``z`` is a collider,
    ``x -> z <- y``, when ``x`` and ``y`` test dependent given ``z`` together with every subset of
    the smaller of their blankets without ``x``, ``y``, and ``z`` [2]_ [3]_, and a non-collider when
    one such set separates them. A variable's blanket here is its set together with its members'
    sets, before the symmetry check, which holds its children's other parents. The colliders are
    oriented as :func:`~andrey.pc` orients them, the most clearly dependent first: by the smallest
    absolute partial correlation over the collider's tests for ``"fisherz"`` and ``"student_t"``,
    and by the largest p-value, smallest first, for the other tests. When two colliders that rank
    equal to nine decimals dispute an edge, it stays undirected. Meek's rules [4]_ then orient the
    edges that follow, in rounds, as for :func:`~andrey.pc`, so the result does not depend on the
    column order; under ``andrey.config(tie_break="position")`` the colliders and the rules follow
    the column order instead. With exact tests, the result is the true CPDAG [2]_.

    **Missing values.** A ``NaN`` in ``data`` marks a missing value. Each test reads only the rows
    where all of its variables are observed (test-wise deletion) [5]_ [6]_, so it keeps rows that
    deleting every incomplete row first would drop; ``metadata["rows_tested"]`` gives the fewest and
    the median rows a test read. When values are missing completely at random, the rows a test keeps
    are a random sample, so each test keeps its level [6]_. When whether a value is missing depends
    on the variables, the rows a test keeps are a selected sample, and the tests can find dependence
    where the population has none and keep extra edges [6]_. A test raises ``ValueError`` when it
    has fewer rows than it needs, ``|S| + 3`` for ``"fisherz"`` and ``"student_t"`` and 2 for the
    other tests, or when a continuous column is constant on its rows.

    **Limitations.** Measured on simulated linear-Gaussian data with Fisher-z tests at alpha 0.05,
    on one core: 20, 50, and 100 variables with 1,000 and 5,000 rows and mean degree 2 and 4, with
    the columns standardized and not, and 200 variables with 1,000 and 5,000 rows. Wherever fits
    finished, the mean skeleton errors stay below the empty graph's; the skeleton is closest to it
    on unstandardized data at 100 variables, 1,000 rows, and mean degree 2, with 56.9 pairs wrong
    against 100, and the CPDAG with 78.1 against 100. The median skeleton fit took 0.01 to 1.6 s up
    to 100 variables and 0.3 to 3.3 s at 200 variables. The CPDAG fit took up to 30 s where it
    finished and reached the 600 s cap at 100 variables, mean degree 4: on 6 of 20 datasets at 1,000
    rows and 19 of 20 at 5,000 on standardized data. At 200 variables and 5,000 rows, MMPC's
    forward step often faces candidates whose dependence is too strong to rank by p-value, and it
    admits the strongest, which is a true neighbor 99.7% of the time. Borderline false edges
    between unrelated pairs, with a statistic just past the cutoff, then survive: about 2 more per
    graph than with an arbitrary order, which happens to admit a strongly correlated non-neighbor
    that removes them. At mean degree 2 the skeleton there makes 75.5 errors on average against 73.9
    for bnlearn's, and 86.9 against 84.2 on 10 fresh datasets. The benchmark covers continuous data; the categorical and mixed-data tests are
    in the discrete-network lanes only. Tested with exact d-separation on 15 variables. It tests
    each candidate against every subset of the current members, so its cost grows exponentially
    with the size of the set; the symmetry check runs the search once more for each member. The
    CPDAG tests every subset of the smaller blanket for each unshielded triple, and a collider
    takes all of them. These blankets reach two steps from a variable, so they are larger than a
    Markov blanket, and a triple whose smaller blanket holds ``m`` variables takes up to ``2**m``
    tests. On simulated linear-Gaussian data with 50 variables, 5,000 rows, and mean degree 4, the
    collider tests took 1.6 to 2.2 s on four datasets and 277 s on a fifth, whose smaller blanket
    held 24 variables (38 million tests), and learning the sets 0.13 to 0.18 s. Missing values are
    not benchmarked yet; the tests run them on up to 20 variables with up to a fifth of the values
    missing completely at random.

    References
    ----------
    .. [1] Tsamardinos, I., Brown, L. E., and Aliferis, C. F. (2006). The max-min hill-climbing
       Bayesian network structure learning algorithm. Machine Learning 65(1), 31-78.
    .. [2] Margaritis, D. (2003). Learning Bayesian network model structure from data. PhD thesis,
       Carnegie Mellon University, CMU-CS-03-153.
    .. [3] Margaritis, D., and Thrun, S. (2000). Bayesian network induction via local
       neighborhoods. Advances in Neural Information Processing Systems 12, 505-511.
    .. [4] Meek, C. (1995). Causal inference and causal explanation with background knowledge.
       UAI, 403-410.
    .. [5] Strobl, E. V., Visweswaran, S., and Spirtes, P. L. (2018). Fast causal inference with
       non-random missingness by test-wise deletion. International Journal of Data Science and
       Analytics 6(1), 47-62.
    .. [6] Tu, R., Zhang, C., Ackermann, P., Mohan, K., Kjellstrom, H., and Zhang, K. (2019). Causal
       discovery in the presence of missing data. Proceedings of Machine Learning Research 89,
       1762-1770.

    Examples
    --------
    >>> import warnings
    >>> import numpy as np
    >>> from andrey import mmpc
    >>> rng = np.random.default_rng(0)
    >>> a, b = rng.standard_normal((2, 1000))
    >>> c = a + b + 0.5 * rng.standard_normal(1000)
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     out = mmpc(np.column_stack([a, b, c]), target=0)
    >>> out.metadata["parents_and_children"]
    [2]
    """
    return _fit(
        "mmpc",
        data,
        target=target,
        structure=structure,
        alpha=alpha,
        indep_test=indep_test,
        categorical=categorical,
        seed=seed,
        max_depth=max_depth,
        knowledge=knowledge,
    )
