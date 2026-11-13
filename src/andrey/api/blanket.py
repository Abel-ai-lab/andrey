"""Local-structure facade adapters: one variable's Markov blanket or parents and children.

``iamb``, ``inter_iamb``, ``gs``, ``fast_iamb``, ``mmpc``, ``hiton_pc``, ``hiton_mb``, ``mmmb``,
and ``pcmb`` share :mod:`andrey.constraint.blanket`. With ``target`` they return that variable's
set, and ``hiton_mb``, ``mmmb``, and ``pcmb`` its parents and children as well; without, every
variable's, joined by the AND rule into the graph ``structure`` names: the CPDAG, the skeleton, or,
for the blankets, the moral graph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from andrey.api._adapt import (
    categorical_metadata,
    copies_named,
    fit_inputs,
    label_of,
    structure_output,
)
from andrey.api._local import target_metadata
from andrey.api._method import method_spec
from andrey.core import NULL, TAIL
from andrey.core.ci import counting_tests
from andrey.core.ci_missing import rows_metadata, rows_tested
from andrey.core.knowledge import Knowledge, as_node, column_position
from andrey.core.orient import to_structure

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from andrey.core import StructureOutput


def fit_blanket(
    method: str,
    data: npt.ArrayLike,
    *,
    target: str | int | None,
    structure: str,
    alpha: float,
    indep_test: str | None,
    degrees_of_freedom: str,
    categorical: Sequence[str | int] | None,
    seed: int | None,
    max_depth: int | None,
    knowledge: Knowledge | None,
) -> StructureOutput:
    """Check the settings, run the search, and wrap the graph ``structure`` names or the sets."""
    from andrey.constraint.blanket import local_search, local_structure
    from andrey.spec.concepts import BLANKET_STRUCTURES

    assert indep_test is not None  # method_spec picks it from the column types
    X, mask, labels, constraints = fit_inputs(
        data, "indep_test", indep_test, categorical, knowledge, method=method, missing=True
    )
    d = X.shape[1]
    t = None if target is None else column_position(as_node(target, "target"), labels, d, "target")
    settings = {
        "alpha": alpha,
        "indep_test": indep_test,
        "degrees_of_freedom": degrees_of_freedom,
        "categorical": mask,
        "seed": seed,
        "max_depth": max_depth,
        "constraints": constraints,
    }
    metadata: dict[str, object] = {}
    kind = BLANKET_STRUCTURES[structure] if t is None else "undirected"
    with copies_named(labels), rows_tested() as rows, counting_tests() as tests:
        if t is None:
            marks = local_structure(X, method, structure, **settings)
        else:
            search = local_search(
                X,
                alpha=alpha,
                indep_test=indep_test,
                degrees_of_freedom=degrees_of_freedom,
                categorical=mask,
                seed=seed,
                max_depth=max_depth,
                constraints=constraints,
            )
            with search.pinned():
                pc, mb, _ = search.local(method, t)
            sets = {"markov_blanket": mb, "parents_and_children": pc}
            members = mb if mb is not None else pc
            assert members is not None  # every learner learns one of the two
            adjacent = np.zeros((d, d), dtype=bool)
            adjacent[t, members] = adjacent[members, t] = True
            marks = np.where(adjacent, TAIL, NULL).astype(np.int8)
            metadata |= target_metadata(
                t, {k: v for k, v in sets.items() if v is not None}, label_of(labels)
            )
    metadata |= categorical_metadata(mask, data) | rows_metadata(rows) | {"n_tests": tests.n}
    return structure_output(to_structure(marks, kind=kind), labels=labels, metadata=metadata)


@method_spec("iamb")
def iamb(
    data: npt.ArrayLike,
    *,
    target: str | int | None = None,
    structure: str = "cpdag",
    alpha: float = 0.05,
    indep_test: str | None = None,
    degrees_of_freedom: str = "mean",
    categorical: Sequence[str | int] | None = None,
    missing: str | None = None,
    seed: int | None = None,
    max_depth: int | None = None,
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn a CPDAG, or one variable's Markov blanket, with incremental association (IAMB).

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
        Significance level of the conditional-independence test, in ``(0, 1)``. Smaller values give
        smaller sets.
    indep_test : str or None, default=None
        Conditional-independence test, any of :func:`~andrey.pc`'s: ``"fisherz"``,
        ``"student_t"``, or ``"normal_scores"`` for continuous data, ``"chisq"`` or ``"gsq"`` for
        categorical data, ``"degen_gauss"`` or ``"cond_gauss"`` for mixed data, or a nonlinear
        test. ``None`` picks from the declared column types, as for :func:`~andrey.pc`.
        ``metadata["indep_test"]`` names the test the fit used: the one picked from the
        declared column types when this is ``None``.
    degrees_of_freedom : {"mean", "nonempty"}, default="mean"
        Degrees of freedom of the categorical tests, ``"chisq"`` and ``"gsq"``. ``"mean"`` is the
        mean of the statistic when ``y`` is shuffled within each configuration of the conditioning
        set; ``"nonempty"`` is ``(r_x - 1) * (r_y - 1)`` per configuration over the values of ``x``
        and ``y`` present in it, the table's empty rows and columns struck. The two differ only on
        sparse tables. Only ``"chisq"`` and ``"gsq"`` read it: passing it with another test raises
        ``ValueError``.
    categorical : sequence of str or int, or None, default=None
        The categorical columns of ``data``, for a mixed-data test, as for :func:`~andrey.pc`. A
        continuous test raises ``ValueError`` on a declared column.
    missing : {"testwise"} or None, default=None
        How missing values (``NaN``) are handled. ``"testwise"`` deletes rows per test: each test
        reads only the rows where all of its variables are observed. ``None`` picks the method's
        default, ``"testwise"``. On data with missing values, ``metadata["missing"]`` records the
        handling used. See **Missing values.** in Notes.
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
        Under ``structure="cpdag"``, the result directs every edge the knowledge decides, by a tier
        or an edge forbidden one way, before the colliders, which never reverse it, and Meek's rules
        direct the edges that follow, giving the maximally oriented PDAG (MPDAG)
        :class:`~andrey.Knowledge` describes; knowledge that orients the edges the search kept into
        a directed cycle raises ``ValueError``. The undirected outputs do not read tiers or edges
        forbidden one way, and with them or a ``target``, ``required_paths`` and ``forbidden_paths``
        raise ``NotImplementedError``. In a CPDAG, ``required_paths`` and ``forbidden_paths`` keep
        the DAGs the graph stands for in which each listed path holds or does not, as
        :class:`~andrey.Knowledge` describes; a claim no such DAG meets is dropped with a
        :class:`~andrey.KnowledgeWarning` and listed in ``metadata["unmet_knowledge"]``; when the
        graph found has no consistent extension, and so stands for no DAG, a claim on a pair joined
        by an undirected edge directs that edge, Meek's rules follow, ``metadata["edge_claims"]``
        lists the claims applied this way, and the result is not guaranteed to stand for a DAG.

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
        ``metadata["categorical"]`` lists the categorical columns it read. With a randomized
        ``indep_test``, ``metadata["seed"]`` is the seed used. ``metadata["n_tests"]`` is the
        number of conditional-independence tests the fit evaluated; a query answered again from a
        kept answer is not a new test. ``metadata["tie_break"]`` is the tie-break rule the fit ran
        under, ``andrey.config.tie_break``.

    Raises
    ------
    TypeError
        If an argument has the wrong type, such as a string for a number; the message names the
        method, the argument, what it accepts, and the value. Also if ``target`` is neither a column
        label nor a position, ``categorical`` is not a list of column names and positions, or
        ``knowledge`` is neither ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If an argument lies outside its range or is not one of its choices; the message names the
        method, the argument, what it accepts, and the value. Also if ``target`` names a column
        ``data`` lacks, or if ``data`` or ``categorical`` does not suit the test, as for
        :func:`~andrey.pc`. With ``"fisherz"`` or ``"student_t"``, also if a column repeats another
        up to scale and shift. Also if ``knowledge`` names a column ``data`` lacks or contradicts
        itself, as :class:`~andrey.Knowledge` lists, or under ``structure="cpdag"`` orients the
        edges the search kept into a directed cycle.
    NotImplementedError
        If ``knowledge`` holds a part the method does not honor, such as ``required_paths`` or
        ``forbidden_paths`` with ``target`` or an undirected ``structure``; the message names the
        methods that honor it.

    Warns
    -----
    DataTypeWarning
        As for :func:`~andrey.pc`, when a mixed-data test finds no declared categorical column or a
        setting reads columns that look like category codes as numbers; once per process.

    Notes
    -----
    **When to use.** IAMB suits i.i.d. observational data with no unmeasured common causes, in
    which every independence follows from the graph (faithfulness) [1]_. Its default Fisher-Z test
    assumes continuous data with linear effects and Gaussian noise; the other tests read
    categorical or mixed data or detect nonlinear effects, as :func:`~andrey.pc` describes. It is
    designed for many variables [1]_, each with a small blanket, since every test conditions on the
    whole current blanket. With ``target``, it returns an undirected graph joining ``target`` to
    each member of its Markov blanket; without, the graph ``structure`` names: the CPDAG by default,
    or the undirected skeleton or moral graph. For the parents and children alone, use
    :func:`~andrey.mmpc`; when unmeasured common causes are plausible, use :func:`~andrey.fci`.

    **How it works.** The grow phase adds the candidate most dependent on the target given the
    current blanket, among those whose p-value is at or below ``alpha``, until none is; the shrink
    phase visits the members in the order they joined and drops each one the test finds independent
    of the target given the other current members. With exact tests, the result is the true Markov
    blanket [1]_.

    The strength is the absolute partial correlation for ``"fisherz"`` and ``"student_t"``, compared
    to nine decimals, which keeps its order where p-values underflow to zero, and the p-value for
    the other tests, compared to nine significant digits, so p-values far below ``1e-9`` keep their
    order. Candidates whose strengths agree, as p-values that are exactly zero do, go to the column
    whose rank, computed from the data, comes first, as for :func:`~andrey.hc`, so the sets do not
    depend on the column order unless two columns are identical.

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
    datasets, and learning the blankets 0.8 to 1.0 s. The skeleton step also separates true edges,
    more often as the blankets grow: on simulated linear-Gaussian data of mean degree 4 (10 to 50
    variables, 500 or 2,000 rows, 120 datasets) it separated 6% of the truly adjacent pairs whose
    smaller blanket held 2 variables, 21% at 4, 43% at 6, and about half at 7 to 9. Required edges
    enlarge the blankets: with true required edges on 5 or 10% of the variable pairs, the step
    separated about one true pair per graph at 50 variables that the fit without them kept, and 0.4
    averaged over 10, 20, and 50 variables, on 240 datasets. Missing values are not benchmarked yet;
    the tests run them on up to 20 variables with up to a fifth of the values missing completely at
    random. ``required_paths`` is supported and ``forbidden_paths`` is experimental. On simulated
    linear-Gaussian data (10 to 50 variables, mean degree 2 or 4, 500 or 2,000 rows, true claims on
    5 to 30% of the variable pairs) the graph found had no consistent extension, and so stood for no
    DAG, in 106 of 240 datasets, and in 20 of 20 with 50 variables, 2,000 rows, and mean degree 4.
    On such a graph a claim acts only on an edge, as :class:`~andrey.Knowledge` describes, and most
    claims are dropped: of 70 required paths per fit, 4 directed an undirected edge, 13 were met by
    an edge the data directed, and 53 were dropped, and of 86 forbidden paths, 1, 2, and 83. Over
    every dataset the fits dropped 57% of the required and 51% of the forbidden paths. With true
    required paths the CPDAG was no farther from the class of graphs that the truth and the claims
    define than without them in any of the 36 settings, and closer in 25. With true forbidden paths
    alone it was farther from that class in 3 of the 6 settings with 50 variables and mean degree 2,
    by 0.65 to 1.35 edges per graph over 40 datasets, since the class is more directed than the
    graph the dropped claims leave; no single setting was farther on two separate sets of 20
    datasets. The CPDAG was closer to the true DAG in 35 of the 36 settings with required paths and
    32 with forbidden paths, and farther in none. The numbers are a reference, not a hard limit:
    they change with the data generator, the density, the method's settings, and the machine.

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
    >>> import numpy as np
    >>> from andrey import iamb
    >>> rng = np.random.default_rng(0)
    >>> a, b = rng.standard_normal((2, 1000))
    >>> c = a + b + 0.5 * rng.standard_normal(1000)
    >>> out = iamb(np.column_stack([a, b, c]), target=0)
    >>> graph = iamb(np.column_stack([a, b, c]))
    >>> sorted(out.metadata["markov_blanket"])
    [1, 2]
    >>> graph.structure.oriented_edges()
    [(0, 2, 'directed'), (1, 2, 'directed')]
    """
    return fit_blanket(
        "iamb",
        data,
        target=target,
        structure=structure,
        alpha=alpha,
        indep_test=indep_test,
        degrees_of_freedom=degrees_of_freedom,
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
    degrees_of_freedom: str = "mean",
    categorical: Sequence[str | int] | None = None,
    missing: str | None = None,
    seed: int | None = None,
    max_depth: int | None = None,
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn a CPDAG, or one variable's Markov blanket, with interleaved IAMB (Inter-IAMB).

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
        Significance level of the conditional-independence test, in ``(0, 1)``. Smaller values give
        smaller sets.
    indep_test : str or None, default=None
        Conditional-independence test, any of :func:`~andrey.pc`'s: ``"fisherz"``,
        ``"student_t"``, or ``"normal_scores"`` for continuous data, ``"chisq"`` or ``"gsq"`` for
        categorical data, ``"degen_gauss"`` or ``"cond_gauss"`` for mixed data, or a nonlinear
        test. ``None`` picks from the declared column types, as for :func:`~andrey.pc`.
        ``metadata["indep_test"]`` names the test the fit used: the one picked from the
        declared column types when this is ``None``.
    degrees_of_freedom : {"mean", "nonempty"}, default="mean"
        Degrees of freedom of the categorical tests, ``"chisq"`` and ``"gsq"``. ``"mean"`` is the
        mean of the statistic when ``y`` is shuffled within each configuration of the conditioning
        set; ``"nonempty"`` is ``(r_x - 1) * (r_y - 1)`` per configuration over the values of ``x``
        and ``y`` present in it, the table's empty rows and columns struck. The two differ only on
        sparse tables. Only ``"chisq"`` and ``"gsq"`` read it: passing it with another test raises
        ``ValueError``.
    categorical : sequence of str or int, or None, default=None
        The categorical columns of ``data``, for a mixed-data test, as for :func:`~andrey.pc`. A
        continuous test raises ``ValueError`` on a declared column.
    missing : {"testwise"} or None, default=None
        How missing values (``NaN``) are handled. ``"testwise"`` deletes rows per test: each test
        reads only the rows where all of its variables are observed. ``None`` picks the method's
        default, ``"testwise"``. On data with missing values, ``metadata["missing"]`` records the
        handling used. See **Missing values.** in Notes.
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
        Under ``structure="cpdag"``, the result directs every edge the knowledge decides, by a tier
        or an edge forbidden one way, before the colliders, which never reverse it, and Meek's rules
        direct the edges that follow, giving the maximally oriented PDAG (MPDAG)
        :class:`~andrey.Knowledge` describes; knowledge that orients the edges the search kept into
        a directed cycle raises ``ValueError``. The undirected outputs do not read tiers or edges
        forbidden one way, and with them or a ``target``, ``required_paths`` and ``forbidden_paths``
        raise ``NotImplementedError``. In a CPDAG, ``required_paths`` and ``forbidden_paths`` keep
        the DAGs the graph stands for in which each listed path holds or does not, as
        :class:`~andrey.Knowledge` describes; a claim no such DAG meets is dropped with a
        :class:`~andrey.KnowledgeWarning` and listed in ``metadata["unmet_knowledge"]``; when the
        graph found has no consistent extension, and so stands for no DAG, a claim on a pair joined
        by an undirected edge directs that edge, Meek's rules follow, ``metadata["edge_claims"]``
        lists the claims applied this way, and the result is not guaranteed to stand for a DAG.

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
        ``metadata["categorical"]`` lists the categorical columns it read. With a randomized
        ``indep_test``, ``metadata["seed"]`` is the seed used. ``metadata["n_tests"]`` is the
        number of conditional-independence tests the fit evaluated; a query answered again from a
        kept answer is not a new test. ``metadata["tie_break"]`` is the tie-break rule the fit ran
        under, ``andrey.config.tie_break``.

    Raises
    ------
    TypeError
        If an argument has the wrong type, such as a string for a number; the message names the
        method, the argument, what it accepts, and the value. Also if ``target`` is neither a column
        label nor a position, ``categorical`` is not a list of column names and positions, or
        ``knowledge`` is neither ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If an argument lies outside its range or is not one of its choices; the message names the
        method, the argument, what it accepts, and the value. Also if ``target`` names a column
        ``data`` lacks, or if ``data`` or ``categorical`` does not suit the test, as for
        :func:`~andrey.pc`. With ``"fisherz"`` or ``"student_t"``, also if a column repeats another
        up to scale and shift. Also if ``knowledge`` names a column ``data`` lacks or contradicts
        itself, as :class:`~andrey.Knowledge` lists, or under ``structure="cpdag"`` orients the
        edges the search kept into a directed cycle.
    NotImplementedError
        If ``knowledge`` holds a part the method does not honor, such as ``required_paths`` or
        ``forbidden_paths`` with ``target`` or an undirected ``structure``; the message names the
        methods that honor it.

    Warns
    -----
    DataTypeWarning
        As for :func:`~andrey.pc`, when a mixed-data test finds no declared categorical column or a
        setting reads columns that look like category codes as numbers; once per process.

    Notes
    -----
    **When to use.** Inter-IAMB suits i.i.d. observational data with no unmeasured common causes, in
    which every independence follows from the graph (faithfulness) [1]_. Its default Fisher-Z test
    assumes continuous data with linear effects and Gaussian noise; the other tests read categorical
    or mixed data or detect nonlinear effects, as :func:`~andrey.pc` describes. It is designed for
    many variables whose Markov blankets are small relative to the rows [1]_: shrinking after each
    addition aims to keep the current blanket, and so the conditioning sets, small at every step.
    With ``target``, it returns an undirected graph joining ``target`` to each member of its Markov
    blanket, which holds its parents, its children, and its children's other parents; without, the
    graph ``structure`` names: the CPDAG by default, or the undirected skeleton or moral graph. For
    the parents and children alone, use :func:`~andrey.mmpc`; when unmeasured common causes are
    plausible, use :func:`~andrey.fci`.

    **How it works.** Each step adds the candidate most dependent on the target given the current
    blanket, among those whose p-value is at or below ``alpha``, then visits the members in the
    order they joined and drops each one the test finds independent of the target given the other
    current members. The search stops when no candidate qualifies, or when the blanket after a
    step repeats one it held after an earlier step, which on sample data could otherwise make it
    cycle. With exact tests, the result is the true Markov blanket [1]_.

    The strength is the absolute partial correlation for ``"fisherz"`` and ``"student_t"``, compared
    to nine decimals, which keeps its order where p-values underflow to zero, and the p-value for
    the other tests, compared to nine significant digits, so p-values far below ``1e-9`` keep their
    order. Candidates whose strengths agree, as p-values that are exactly zero do, go to the column
    whose rank, computed from the data, comes first, as for :func:`~andrey.hc`, so the sets do not
    depend on the column order unless two columns are identical.

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
    In every setting the mean errors of the CPDAG, the skeleton, and the moral graph stay below the
    empty graph's; the CPDAG is closest at 20 variables, 1,000 rows, and mean degree 4, with 24
    pairs wrong against 40 for the empty graph. A fit took 0.06 to 8.9 s, the longest at 100
    variables, 5,000 rows, and mean degree 4. Tested with exact d-separation on 15 variables. It
    conditions on the whole current blanket, so each test needs more rows as the blanket grows. The
    skeleton and the CPDAG test every subset of the smaller blanket for each pair of the moral graph
    and each unshielded triple, and a collider takes all of them, so a triple whose smaller blanket
    holds ``m`` variables takes up to ``2**m`` tests. On simulated linear-Gaussian data with 50
    variables, 5,000 rows, and mean degree 4, the smaller blankets held at most 16 variables, the
    collider tests took 0.06 to 0.7 s on five datasets, and learning the blankets 0.9 to 1.3 s. The
    skeleton step also separates true edges, more often as the blankets grow: on simulated
    linear-Gaussian data of mean degree 4 (10 to 50 variables, 500 or 2,000 rows, 120 datasets) it
    separated 6% of the truly adjacent pairs whose smaller blanket held 2 variables, 21% at 4, 41%
    at 6, and about half at 7 to 9. Required edges enlarge the blankets: with true required edges on
    5 or 10% of the variable pairs, the step separated about one true pair per graph at 50 variables
    that the fit without them kept, and 0.3 to 0.4 averaged over 10, 20, and 50 variables, on 240
    datasets. Missing values are not benchmarked yet; the tests run them on up to 20 variables with
    up to a fifth of the values missing completely at random. ``required_paths`` is supported and
    ``forbidden_paths`` is experimental. On simulated linear-Gaussian data (10 to 50 variables, mean
    degree 2 or 4, 500 or 2,000 rows, true claims on 5 to 30% of the variable pairs) the graph found
    had no consistent extension, and so stood for no DAG, in 105 of 240 datasets, and in 20 of 20
    with 50 variables, 2,000 rows, and mean degree 4. On such a graph a claim acts only on an edge,
    as :class:`~andrey.Knowledge` describes, and most claims are dropped: of 71 required paths per
    fit, 4 directed an undirected edge, 13 were met by an edge the data directed, and 54 were
    dropped, and of 86 forbidden paths, 1, 2, and 83. Over every dataset the fits dropped 56% of the
    required and 51% of the forbidden paths. With true required paths the CPDAG was no farther from
    the class of graphs that the truth and the claims define than without them in any of the 36
    settings, and closer in 24. With true forbidden paths alone it was farther from that class in 3
    of the 6 settings with 50 variables and mean degree 2, by 0.65 to 1.35 edges per graph over 40
    datasets, since the class is more directed than the graph the dropped claims leave; no single
    setting was farther on two separate sets of 20 datasets. The CPDAG was closer to the true DAG in
    35 of the 36 settings with required paths and 31 with forbidden paths, and farther in none. The
    numbers are a reference, not a hard limit: they change with the data generator, the density, the
    method's settings, and the machine.

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
    >>> import numpy as np
    >>> from andrey import inter_iamb
    >>> rng = np.random.default_rng(0)
    >>> a, b = rng.standard_normal((2, 1000))
    >>> c = a + b + 0.5 * rng.standard_normal(1000)
    >>> out = inter_iamb(np.column_stack([a, b, c]), target=0)
    >>> sorted(out.metadata["markov_blanket"])
    [1, 2]
    """
    return fit_blanket(
        "inter_iamb",
        data,
        target=target,
        structure=structure,
        alpha=alpha,
        indep_test=indep_test,
        degrees_of_freedom=degrees_of_freedom,
        categorical=categorical,
        seed=seed,
        max_depth=max_depth,
        knowledge=knowledge,
    )


@method_spec("gs")
def gs(
    data: npt.ArrayLike,
    *,
    target: str | int | None = None,
    structure: str = "cpdag",
    alpha: float = 0.05,
    indep_test: str | None = None,
    degrees_of_freedom: str = "mean",
    categorical: Sequence[str | int] | None = None,
    missing: str | None = None,
    seed: int | None = None,
    max_depth: int | None = None,
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn a CPDAG, or one variable's Markov blanket, with Grow-Shrink (GS).

    A variable's Markov blanket is the smallest set of other variables given which it is
    independent of all the rest: in a causal graph, its parents, its children, and its children's
    other parents. GS [1]_ [2]_ visits the other variables in one fixed order, grows the blanket by
    conditional-independence tests, then shrinks it. Without a target, every variable's blanket is
    learned and joined into the graph ``structure`` names, by default the CPDAG of the Grow-Shrink
    structure learner [1]_ [2]_.

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
        Significance level of the conditional-independence test, in ``(0, 1)``. Smaller values give
        smaller sets.
    indep_test : str or None, default=None
        Conditional-independence test, any of :func:`~andrey.pc`'s: ``"fisherz"``,
        ``"student_t"``, or ``"normal_scores"`` for continuous data, ``"chisq"`` or ``"gsq"`` for
        categorical data, ``"degen_gauss"`` or ``"cond_gauss"`` for mixed data, or a nonlinear
        test. ``None`` picks from the declared column types, as for :func:`~andrey.pc`.
        ``metadata["indep_test"]`` names the test the fit used: the one picked from the
        declared column types when this is ``None``.
    degrees_of_freedom : {"mean", "nonempty"}, default="mean"
        Degrees of freedom of the categorical tests, ``"chisq"`` and ``"gsq"``. ``"mean"`` is the
        mean of the statistic when ``y`` is shuffled within each configuration of the conditioning
        set; ``"nonempty"`` is ``(r_x - 1) * (r_y - 1)`` per configuration over the values of ``x``
        and ``y`` present in it, the table's empty rows and columns struck. The two differ only on
        sparse tables. Only ``"chisq"`` and ``"gsq"`` read it: passing it with another test raises
        ``ValueError``.
    categorical : sequence of str or int, or None, default=None
        The categorical columns of ``data``, for a mixed-data test, as for :func:`~andrey.pc`. A
        continuous test raises ``ValueError`` on a declared column.
    missing : {"testwise"} or None, default=None
        How missing values (``NaN``) are handled. ``"testwise"`` deletes rows per test: each test
        reads only the rows where all of its variables are observed. ``None`` picks the method's
        default, ``"testwise"``. On data with missing values, ``metadata["missing"]`` records the
        handling used. See **Missing values.** in Notes.
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
        Under ``structure="cpdag"``, the result directs every edge the knowledge decides, by a tier
        or an edge forbidden one way, before the colliders, which never reverse it, and Meek's rules
        direct the edges that follow, giving the maximally oriented PDAG (MPDAG)
        :class:`~andrey.Knowledge` describes; knowledge that orients the edges the search kept into
        a directed cycle raises ``ValueError``. The undirected outputs do not read tiers or edges
        forbidden one way, and with them or a ``target``, ``required_paths`` and ``forbidden_paths``
        raise ``NotImplementedError``. In a CPDAG, ``required_paths`` and ``forbidden_paths`` keep
        the DAGs the graph stands for in which each listed path holds or does not, as
        :class:`~andrey.Knowledge` describes; a claim no such DAG meets is dropped with a
        :class:`~andrey.KnowledgeWarning` and listed in ``metadata["unmet_knowledge"]``; when the
        graph found has no consistent extension, and so stands for no DAG, a claim on a pair joined
        by an undirected edge directs that edge, Meek's rules follow, ``metadata["edge_claims"]``
        lists the claims applied this way, and the result is not guaranteed to stand for a DAG.

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
        ``metadata["algorithm"]`` is ``"gs"``. Under a mixed-data setting,
        ``metadata["categorical"]`` lists the categorical columns it read. With a randomized
        ``indep_test``, ``metadata["seed"]`` is the seed used. ``metadata["n_tests"]`` is the
        number of conditional-independence tests the fit evaluated; a query answered again from a
        kept answer is not a new test. ``metadata["tie_break"]`` is the tie-break rule the fit ran
        under, ``andrey.config.tie_break``.

    Raises
    ------
    TypeError
        If an argument has the wrong type, such as a string for a number; the message names the
        method, the argument, what it accepts, and the value. Also if ``target`` is neither a column
        label nor a position, ``categorical`` is not a list of column names and positions, or
        ``knowledge`` is neither ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If an argument lies outside its range or is not one of its choices; the message names the
        method, the argument, what it accepts, and the value. Also if ``target`` names a column
        ``data`` lacks, or if ``data`` or ``categorical`` does not suit the test, as for
        :func:`~andrey.pc`. With ``"fisherz"`` or ``"student_t"``, also if a column repeats another
        up to scale and shift. Also if ``knowledge`` names a column ``data`` lacks or contradicts
        itself, as :class:`~andrey.Knowledge` lists, or under ``structure="cpdag"`` orients the
        edges the search kept into a directed cycle.
    NotImplementedError
        If ``knowledge`` holds a part the method does not honor, such as ``required_paths`` or
        ``forbidden_paths`` with ``target`` or an undirected ``structure``; the message names the
        methods that honor it.

    Warns
    -----
    ExperimentalWarning
        On the first call in a process: the method is not supported (see Limitations), and its
        API may change without deprecation.
    DataTypeWarning
        As for :func:`~andrey.pc`, when a mixed-data test finds no declared categorical column or a
        setting reads columns that look like category codes as numbers; once per process.

    Notes
    -----
    **When to use.** GS suits i.i.d. observational data with no unmeasured common causes, in
    which every independence follows from the graph (faithfulness) [1]_. Its papers state it for any
    independence test and run it on categorical data with a chi-square test [1]_ [2]_; the default
    Fisher-Z test assumes continuous data with linear effects and Gaussian noise, and the other
    tests read categorical or mixed data or detect nonlinear effects, as :func:`~andrey.pc`
    describes. It takes a number of tests proportional to the number of variables for each blanket
    [1]_, and every test conditions on the whole current blanket, so it suits variables with small
    blankets. With ``target``, it returns an undirected graph joining ``target`` to each member of
    its Markov blanket; without, the graph ``structure`` names: the CPDAG by default, or the
    undirected skeleton or moral graph. For the parents and children alone, use
    :func:`~andrey.mmpc`; when unmeasured common causes are plausible, use :func:`~andrey.fci`.

    **How it works.** The other variables are ordered once, the most dependent on the target
    first, by the strength of their unconditional association, which takes one test per variable;
    the papers leave the order open and suggest visiting the most associated variables first [2]_.
    The grow phase visits the variables outside the blanket in that order, tests each given the
    current blanket, and adds it at once when its p-value is at or below ``alpha``, so the next one
    is tested given the larger blanket; it passes over them again until a pass adds none. With
    exact tests, the first pass adds at least the parents and children and the second the
    children's other parents [2]_. The shrink phase visits the members in the same order and drops
    each one the test finds independent of the target given the other current members, again until
    a pass drops none. A test whose conditioning set is unchanged since the variable's last test is
    not run again. With exact tests, the result is the true Markov blanket [1]_ [2]_.

    The strength is the absolute partial correlation for ``"fisherz"`` and ``"student_t"``, compared
    to nine decimals, which keeps its order where p-values underflow to zero, and the p-value for
    the other tests, compared to nine significant digits, so p-values far below ``1e-9`` keep their
    order, as the papers order the variables by their probability of dependence [1]_. Variables
    whose strengths agree go to the column whose rank, computed from the data, comes first, as for
    :func:`~andrey.hc`, so the sets do not depend on the column order unless two columns are
    identical.

    **From blankets to a graph.** Without a target, a pair is kept when each variable is in the
    other's blanket, which gives the moral graph (``structure="moral"``). The skeleton drops each
    pair ``x - y`` that tests independent given some subset, the empty set included, of the smaller
    of ``x``'s blanket without ``y`` and ``y``'s blanket without ``x``, or of either when both hold
    as many members [1]_ [2]_: a child's other parent is in the blanket without being adjacent, and
    such a subset separates the two. Every pair is tested against the moral graph, so no verdict
    depends on another. The CPDAG decides each triple ``x - z - y`` of the skeleton with ``x`` and
    ``y`` not adjacent: ``z`` is a collider, ``x -> z <- y``, when ``x`` and ``y`` test dependent
    given ``z`` together with every subset of the smaller of their blankets without ``x``, ``y``,
    and ``z`` [1]_ [2]_, and a non-collider when one such set separates them. The colliders are
    oriented as :func:`~andrey.pc` orients them, the most clearly dependent first: by the smallest
    absolute partial correlation over the collider's tests for ``"fisherz"`` and ``"student_t"``,
    and by the largest p-value, smallest first, for the other tests. When two colliders that rank
    equal to nine decimals dispute an edge, it stays undirected. Meek's rules [3]_ then orient the
    edges that follow, in rounds, as for :func:`~andrey.pc`, so the result does not depend on the
    column order; under ``andrey.config(tie_break="position")`` the colliders and the rules follow
    the column order instead. The papers instead remove the edges that close directed cycles,
    reverse them, and orient an edge along any directed path [1]_ [2]_; with exact tests both give
    the true CPDAG.

    **Missing values.** A ``NaN`` in ``data`` marks a missing value. Each test reads only the rows
    where all of its variables are observed (test-wise deletion) [4]_ [5]_, so it keeps rows that
    deleting every incomplete row first would drop; ``metadata["rows_tested"]`` gives the fewest and
    the median rows a test read. When values are missing completely at random, the rows a test keeps
    are a random sample, so each test keeps its level [5]_. When whether a value is missing depends
    on the variables, the rows a test keeps are a selected sample, and the tests can find dependence
    where the population has none and keep extra edges [5]_. A test raises ``ValueError`` when it
    has fewer rows than it needs, ``|S| + 3`` for ``"fisherz"`` and ``"student_t"`` and 2 for the
    other tests, or when a continuous column is constant on its rows.

    **Limitations.** Not benchmarked yet. Tested with exact d-separation on 15 variables and on
    simulated linear-Gaussian data with 20 variables and 2,000 rows; missing values on up to 20
    variables with up to a fifth of the values missing completely at random. At a fixed ``alpha``,
    the grow phase tests every variable outside the blanket, so the false members grow with the
    number of variables: of the many variables independent of the target, those whose p-values fall
    below ``alpha`` by chance join, each pass tests the ones left out again, and the shrink phase,
    which retests the members given nearly the same set with the same test, keeps most of them. It
    conditions on the whole current blanket, so each test needs more rows as the blanket grows, and
    a variable the grow phase adds in error enlarges every later conditioning set. The skeleton and
    the CPDAG test every subset of the smaller blanket for each pair of the moral graph and each
    unshielded triple, so a triple whose smaller blanket holds ``m`` variables takes up to ``2**m``
    tests. The numbers are a reference, not a hard limit: they change with the data generator, the
    density, the method's settings, and the machine.

    References
    ----------
    .. [1] Margaritis, D., and Thrun, S. (2000). Bayesian network induction via local
       neighborhoods. Advances in Neural Information Processing Systems 12, 505-511.
    .. [2] Margaritis, D. (2003). Learning Bayesian network model structure from data. PhD thesis,
       Carnegie Mellon University, CMU-CS-03-153.
    .. [3] Meek, C. (1995). Causal inference and causal explanation with background knowledge.
       UAI, 403-410.
    .. [4] Strobl, E. V., Visweswaran, S., and Spirtes, P. L. (2018). Fast causal inference with
       non-random missingness by test-wise deletion. International Journal of Data Science and
       Analytics 6(1), 47-62.
    .. [5] Tu, R., Zhang, C., Ackermann, P., Mohan, K., Kjellstrom, H., and Zhang, K. (2019). Causal
       discovery in the presence of missing data. Proceedings of Machine Learning Research 89,
       1762-1770.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import gs
    >>> rng = np.random.default_rng(0)
    >>> a, b = rng.standard_normal((2, 1000))
    >>> c = a + b + 0.5 * rng.standard_normal(1000)
    >>> out = gs(np.column_stack([a, b, c]), target=0)
    >>> sorted(out.metadata["markov_blanket"])
    [1, 2]
    """
    return fit_blanket(
        "gs",
        data,
        target=target,
        structure=structure,
        alpha=alpha,
        indep_test=indep_test,
        degrees_of_freedom=degrees_of_freedom,
        categorical=categorical,
        seed=seed,
        max_depth=max_depth,
        knowledge=knowledge,
    )


@method_spec("fast_iamb")
def fast_iamb(
    data: npt.ArrayLike,
    *,
    target: str | int | None = None,
    structure: str = "cpdag",
    alpha: float = 0.05,
    indep_test: str | None = None,
    degrees_of_freedom: str = "mean",
    categorical: Sequence[str | int] | None = None,
    missing: str | None = None,
    seed: int | None = None,
    max_depth: int | None = None,
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn a CPDAG, or one variable's Markov blanket, with Fast-IAMB.

    A variable's Markov blanket is the smallest set of other variables given which it is
    independent of all the rest: in a causal graph, its parents, its children, and its children's
    other parents. Fast-IAMB [1]_ grows the blanket by conditional-independence tests, several
    variables at a time, the most dependent first, and shrinks it after each addition. Without a
    target, every variable's blanket is learned and joined into the graph ``structure`` names, by
    default the CPDAG of the Grow-Shrink structure learner [2]_ [3]_.

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
        Significance level of the conditional-independence test, in ``(0, 1)``. Smaller values give
        smaller sets.
    indep_test : str or None, default=None
        Conditional-independence test, any of :func:`~andrey.pc`'s: ``"fisherz"``,
        ``"student_t"``, or ``"normal_scores"`` for continuous data, ``"chisq"`` or ``"gsq"`` for
        categorical data, ``"degen_gauss"`` or ``"cond_gauss"`` for mixed data, or a nonlinear
        test. ``None`` picks from the declared column types, as for :func:`~andrey.pc`.
        ``metadata["indep_test"]`` names the test the fit used: the one picked from the
        declared column types when this is ``None``.
    degrees_of_freedom : {"mean", "nonempty"}, default="mean"
        Degrees of freedom of the categorical tests, ``"chisq"`` and ``"gsq"``. ``"mean"`` is the
        mean of the statistic when ``y`` is shuffled within each configuration of the conditioning
        set; ``"nonempty"`` is ``(r_x - 1) * (r_y - 1)`` per configuration over the values of ``x``
        and ``y`` present in it, the table's empty rows and columns struck. The two differ only on
        sparse tables. Only ``"chisq"`` and ``"gsq"`` read it: passing it with another test raises
        ``ValueError``.
    categorical : sequence of str or int, or None, default=None
        The categorical columns of ``data``, for a mixed-data test, as for :func:`~andrey.pc`. A
        continuous test raises ``ValueError`` on a declared column.
    missing : {"testwise"} or None, default=None
        How missing values (``NaN``) are handled. ``"testwise"`` deletes rows per test: each test
        reads only the rows where all of its variables are observed. ``None`` picks the method's
        default, ``"testwise"``. On data with missing values, ``metadata["missing"]`` records the
        handling used. See **Missing values.** in Notes.
    seed : int or None, default=None
        ``None``, the default, is the configured seed, ``ANDREY_SEED`` or ``0`` when it is unset,
        not a fresh random one, so a fit repeats. Seed for the random draws of ``indep_test``, which
        only ``"rcot"``, ``"cmiknn"``, and ``"gpdc"`` make; the other tests ignore it. With those
        tests, ``metadata["seed"]`` records the seed used.
    max_depth : int or None, default=None
        The maximum conditioning-set size of every conditional-independence test, at least ``0``:
        ``0`` runs only unconditional tests, and ``None`` sets no limit. A smaller cap runs fewer
        tests and can keep variables that a larger conditioning set would remove. Every test
        conditions on the whole current blanket, so an addition stops once the blanket holds more
        than ``max_depth`` members, counting those ``knowledge`` requires, and the search stops
        while it does. A collider test
        conditions on the middle node as well, so with ``0`` no triple is decided and the CPDAG
        directs only the edges ``knowledge`` decides.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the pairs
        allowed to be adjacent, and temporal tiers, all honored. Each end of a required edge is in
        the other's blanket from the start, and no test removes it. A pair the knowledge makes
        non-adjacent, forbidden both ways or outside ``allowed``, stays a candidate, since a child's
        other parent belongs to the blanket without being adjacent, and is left out of the result.
        Under ``structure="cpdag"``, the result directs every edge the knowledge decides, by a tier
        or an edge forbidden one way, before the colliders, which never reverse it, and Meek's rules
        direct the edges that follow, giving the maximally oriented PDAG (MPDAG)
        :class:`~andrey.Knowledge` describes; knowledge that orients the edges the search kept into
        a directed cycle raises ``ValueError``. The undirected outputs do not read tiers or edges
        forbidden one way, and with them or a ``target``, ``required_paths`` and ``forbidden_paths``
        raise ``NotImplementedError``. In a CPDAG, ``required_paths`` and ``forbidden_paths`` keep
        the DAGs the graph stands for in which each listed path holds or does not, as
        :class:`~andrey.Knowledge` describes; a claim no such DAG meets is dropped with a
        :class:`~andrey.KnowledgeWarning` and listed in ``metadata["unmet_knowledge"]``; when the
        graph found has no consistent extension, and so stands for no DAG, a claim on a pair joined
        by an undirected edge directs that edge, Meek's rules follow, ``metadata["edge_claims"]``
        lists the claims applied this way, and the result is not guaranteed to stand for a DAG.

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
        ``metadata["algorithm"]`` is ``"fast_iamb"``. Under a mixed-data setting,
        ``metadata["categorical"]`` lists the categorical columns it read. With a randomized
        ``indep_test``, ``metadata["seed"]`` is the seed used. ``metadata["n_tests"]`` is the
        number of conditional-independence tests the fit evaluated; a query answered again from a
        kept answer is not a new test. ``metadata["tie_break"]`` is the tie-break rule the fit ran
        under, ``andrey.config.tie_break``.

    Raises
    ------
    TypeError
        If an argument has the wrong type, such as a string for a number; the message names the
        method, the argument, what it accepts, and the value. Also if ``target`` is neither a column
        label nor a position, ``categorical`` is not a list of column names and positions, or
        ``knowledge`` is neither ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If an argument lies outside its range or is not one of its choices; the message names the
        method, the argument, what it accepts, and the value. Also if ``target`` names a column
        ``data`` lacks, or if ``data`` or ``categorical`` does not suit the test, as for
        :func:`~andrey.pc`. With ``"fisherz"`` or ``"student_t"``, also if a column repeats another
        up to scale and shift. Also if ``knowledge`` names a column ``data`` lacks or contradicts
        itself, as :class:`~andrey.Knowledge` lists, or under ``structure="cpdag"`` orients the
        edges the search kept into a directed cycle.
    NotImplementedError
        If ``knowledge`` holds a part the method does not honor, such as ``required_paths`` or
        ``forbidden_paths`` with ``target`` or an undirected ``structure``; the message names the
        methods that honor it.

    Warns
    -----
    ExperimentalWarning
        On the first call in a process: the method is not supported (see Limitations), and its
        API may change without deprecation.
    DataTypeWarning
        As for :func:`~andrey.pc`, when a mixed-data test finds no declared categorical column or a
        setting reads columns that look like category codes as numbers; once per process.

    Notes
    -----
    **When to use.** Fast-IAMB suits i.i.d. observational data with no unmeasured common causes, in
    which every independence follows from the graph (faithfulness) [1]_. Its paper runs it on
    categorical data with a G-square test [1]_; the default Fisher-Z test assumes continuous data
    with linear effects and Gaussian noise, and the other tests read categorical or mixed data or
    detect nonlinear effects, as :func:`~andrey.pc` describes. It is designed to run fewer tests
    than :func:`~andrey.iamb` by adding several variables after each round of tests [1]_, and every
    test conditions on the whole current blanket, so it suits variables with small blankets. With
    ``target``, it returns an undirected graph joining ``target`` to each member of its Markov
    blanket; without, the graph ``structure`` names: the CPDAG by default, or the undirected
    skeleton or moral graph. For the parents and children alone, use :func:`~andrey.mmpc`; when
    unmeasured common causes are plausible, use :func:`~andrey.fci`.

    **How it works.** Each round tests every variable outside the blanket given the current
    blanket, the first round given no variable, and orders those whose p-value is at or below
    ``alpha`` from the most dependent to the least. It adds them in that order without another
    test while the data hold enough rows for the next test: on categorical data, while the rows are
    more than five times the cells of the table of the target, the variable, and the blanket so far,
    the product of their numbers of categories [1]_; every row of the data counts, also where a test
    reads fewer. A variable that fails this rule ends the round and marks the data as insufficient.
    With a continuous column the table has no cells to count, so every variable that tests
    dependent in the round joins. A shrink pass then visits the members in the order they joined
    and drops each one the test finds independent of the target given the other current members.
    The search stops when no variable tests dependent; when the data were insufficient and the
    shrink dropped nothing, taking the variables the rule kept out as independent [1]_; or when the
    blanket after a shrink repeats one it held after an earlier shrink, which ends a cycle of
    additions and removals the paper leaves open. With exact tests, the result is the true Markov
    blanket [1]_.

    The strength is the absolute partial correlation for ``"fisherz"`` and ``"student_t"``, compared
    to nine decimals, which keeps its order where p-values underflow to zero, and the p-value for
    the other tests, compared to nine significant digits, so p-values far below ``1e-9`` keep their
    order, as the paper orders the variables by the significance of their tests [1]_. Variables
    whose strengths agree go to the column whose rank, computed from the data, comes first, as for
    :func:`~andrey.hc`, so the sets do not depend on the column order unless two columns are
    identical.

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

    **Limitations.** Not benchmarked yet. Tested with exact d-separation on 15 variables, on
    simulated linear-Gaussian data with 20 variables and 2,000 rows, and on simulated categorical
    data with 10 variables and 3,000 rows; missing values on up to 20 variables with up to a fifth
    of the values missing completely at random. At a fixed ``alpha``, every round tests every
    variable outside the blanket, so the false members grow with the number of variables: of the
    many variables independent of the target, those whose p-values fall below ``alpha`` by chance
    join, and the shrink, which retests them given nearly the same set with the same test, keeps
    most of them. Without the rule, on continuous data, the first round adds every variable that
    depends on the target at all, ancestors and descendants included, and the shrink must remove
    them given all the others. It conditions on the whole current blanket, so each test needs more
    rows as the blanket grows. The skeleton and the CPDAG test every subset of the smaller blanket
    for each pair of the moral graph and each unshielded triple, so a triple whose smaller blanket
    holds ``m`` variables takes up to ``2**m`` tests. The numbers are a reference, not a hard limit:
    they change with the data generator, the density, the method's settings, and the machine.

    References
    ----------
    .. [1] Yaramakala, S., and Margaritis, D. (2005). Speculative Markov blanket discovery for
       optimal feature selection. Fifth IEEE International Conference on Data Mining (ICDM).
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
    >>> import numpy as np
    >>> from andrey import fast_iamb
    >>> rng = np.random.default_rng(0)
    >>> a, b = rng.standard_normal((2, 1000))
    >>> c = a + b + 0.5 * rng.standard_normal(1000)
    >>> out = fast_iamb(np.column_stack([a, b, c]), target=0)
    >>> sorted(out.metadata["markov_blanket"])
    [1, 2]
    """
    return fit_blanket(
        "fast_iamb",
        data,
        target=target,
        structure=structure,
        alpha=alpha,
        indep_test=indep_test,
        degrees_of_freedom=degrees_of_freedom,
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
    degrees_of_freedom: str = "mean",
    categorical: Sequence[str | int] | None = None,
    missing: str | None = None,
    seed: int | None = None,
    max_depth: int | None = None,
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn a skeleton, or one variable's parents and children, with MMPC.

    Max-min parents and children (MMPC) [1]_ finds the variables adjacent to a target in the causal
    graph, its parents and children, without telling them apart, by conditional-independence tests.
    Every variable's set together gives the skeleton :func:`~andrey.mmhc` searches, which
    ``structure="cpdag"`` orients into a CPDAG by the collider tests of the Grow-Shrink structure
    learner [2]_ [3]_.

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
        Significance level of the conditional-independence test, in ``(0, 1)``. Smaller values give
        smaller sets.
    indep_test : str or None, default=None
        Conditional-independence test, any of :func:`~andrey.pc`'s: ``"fisherz"``,
        ``"student_t"``, or ``"normal_scores"`` for continuous data, ``"chisq"`` or ``"gsq"`` for
        categorical data, ``"degen_gauss"`` or ``"cond_gauss"`` for mixed data, or a nonlinear
        test. ``None`` picks from the declared column types, as for :func:`~andrey.pc`.
        ``metadata["indep_test"]`` names the test the fit used: the one picked from the
        declared column types when this is ``None``.
    degrees_of_freedom : {"mean", "nonempty"}, default="mean"
        Degrees of freedom of the categorical tests, ``"chisq"`` and ``"gsq"``. ``"mean"`` is the
        mean of the statistic when ``y`` is shuffled within each configuration of the conditioning
        set; ``"nonempty"`` is ``(r_x - 1) * (r_y - 1)`` per configuration over the values of ``x``
        and ``y`` present in it, the table's empty rows and columns struck. The two differ only on
        sparse tables. Only ``"chisq"`` and ``"gsq"`` read it: passing it with another test raises
        ``ValueError``.
    categorical : sequence of str or int, or None, default=None
        The categorical columns of ``data``, for a mixed-data test, as for :func:`~andrey.pc`. A
        continuous test raises ``ValueError`` on a declared column.
    missing : {"testwise"} or None, default=None
        How missing values (``NaN``) are handled. ``"testwise"`` deletes rows per test: each test
        reads only the rows where all of its variables are observed. ``None`` picks the method's
        default, ``"testwise"``. On data with missing values, ``metadata["missing"]`` records the
        handling used. See **Missing values.** in Notes.
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
        non-adjacent, forbidden both ways or outside ``allowed``, is never a candidate. Under
        ``structure="cpdag"``, the result directs every edge the knowledge decides, by a tier or an
        edge forbidden one way, before the colliders, which never reverse it, and Meek's rules
        direct the edges that follow, giving the maximally oriented PDAG (MPDAG)
        :class:`~andrey.Knowledge` describes; knowledge that orients the edges the search kept into
        a directed cycle raises ``ValueError``. The undirected outputs do not read tiers or edges
        forbidden one way, and with them or a ``target``, ``required_paths`` and ``forbidden_paths``
        raise ``NotImplementedError``. In a CPDAG, ``required_paths`` and ``forbidden_paths`` keep
        the DAGs the graph stands for in which each listed path holds or does not, as
        :class:`~andrey.Knowledge` describes; a claim no such DAG meets is dropped with a
        :class:`~andrey.KnowledgeWarning` and listed in ``metadata["unmet_knowledge"]``; when the
        graph found has no consistent extension, and so stands for no DAG, a claim on a pair joined
        by an undirected edge directs that edge, Meek's rules follow, ``metadata["edge_claims"]``
        lists the claims applied this way, and the result is not guaranteed to stand for a DAG.

    Returns
    -------
    StructureOutput
        Without ``target``, wraps the graph ``structure`` names: an ``undirected``
        :class:`~andrey.core.GraphStructure` for ``"skeleton"``, whose edges have no arrowhead and
        join each pair in each other's set, or a ``cpdag`` one for ``"cpdag"``, whose undirected
        edges the tests cannot orient. With ``target``, an ``undirected`` one whose edges join
        ``target`` to each of its parents and children; ``metadata["target"]`` names it and
        ``metadata["parents_and_children"]`` lists the members in the order they joined, by column
        label or 0-based position. ``metadata["algorithm"]`` is ``"mmpc"``. Under a mixed-data
        setting, ``metadata["categorical"]`` lists the categorical columns it read. With a
        randomized ``indep_test``, ``metadata["seed"]`` is the seed used. ``metadata["n_tests"]``
        is the number of conditional-independence tests the fit evaluated; a query answered again
        from a kept answer is not a new test. ``metadata["tie_break"]`` is the tie-break rule the
        fit ran under, ``andrey.config.tie_break``.

    Raises
    ------
    TypeError
        If an argument has the wrong type, such as a string for a number; the message names the
        method, the argument, what it accepts, and the value. Also if ``target`` is neither a column
        label nor a position, ``categorical`` is not a list of column names and positions, or
        ``knowledge`` is neither ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If an argument lies outside its range or is not one of its choices; the message names the
        method, the argument, what it accepts, and the value. Also if ``target`` names a column
        ``data`` lacks, or if ``data`` or ``categorical`` does not suit the test, as for
        :func:`~andrey.pc`. With ``"fisherz"`` or ``"student_t"``, also if a column repeats another
        up to scale and shift. Also if ``knowledge`` names a column ``data`` lacks or contradicts
        itself, as :class:`~andrey.Knowledge` lists, or under ``structure="cpdag"`` orients the
        edges the search kept into a directed cycle.
    NotImplementedError
        If ``knowledge`` holds a part the method does not honor, such as ``required_paths`` or
        ``forbidden_paths`` with ``target`` or an undirected ``structure``; the message names the
        methods that honor it.

    Warns
    -----
    DataTypeWarning
        As for :func:`~andrey.pc`, when a mixed-data test finds no declared categorical column or a
        setting reads columns that look like category codes as numbers; once per process.

    Notes
    -----
    **When to use.** MMPC suits i.i.d. observational data with no unmeasured common causes, in
    which every independence follows from the graph (faithfulness) [1]_. Its default Fisher-Z test
    assumes continuous data with linear effects and Gaussian noise; the other tests read
    categorical or mixed data or detect nonlinear effects, as :func:`~andrey.pc` describes. It is
    designed for many variables, each with few parents and children [1]_, since each candidate is
    tested given subsets of the members found so far. With ``target``, it returns an undirected
    graph joining ``target`` to each of its parents and children, without telling the two apart;
    without, the graph ``structure`` names: the undirected skeleton by default, or the CPDAG. For
    the Markov blanket, which adds the children's other parents, use :func:`~andrey.iamb`; to
    orient the skeleton by a score search, use :func:`~andrey.mmhc`.

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
    rows and 19 of 20 at 5,000 on standardized data. At 200 variables and 5,000 rows, MMPC's forward
    step often faces candidates whose dependence is too strong to rank by p-value, and it admits the
    strongest, which is a true neighbor 99.7% of the time. Borderline false edges between unrelated
    pairs, with a statistic just past the cutoff, then survive: about 2 more per graph than with an
    arbitrary order, which happens to admit a strongly correlated non-neighbor that removes them. At
    mean degree 2 the skeleton there makes 75.5 errors on average, and 86.9 on 10 fresh datasets.
    The continuous benchmark does not cover the categorical and mixed-data tests, which only the
    discrete-network lanes cover. Tested with exact d-separation on 15 variables. It tests each
    candidate against the subsets of the current members, up to the first size that separates it, so
    its cost grows exponentially with the size of the set; the symmetry check runs the search once
    more for each member. The CPDAG tests every subset of the smaller blanket for each unshielded
    triple, and a collider takes all of them. These blankets reach two steps from a variable, so
    they are larger than a Markov blanket, and a triple whose smaller blanket holds ``m`` variables
    takes up to ``2**m`` tests. On simulated linear-Gaussian data with 50 variables, 5,000 rows, and
    mean degree 4, the collider tests took 1.6 to 2.2 s on four datasets and 277 s on a fifth, whose
    smaller blanket held 24 variables (38 million tests), and learning the sets 0.13 to 0.18 s.
    Missing values are not benchmarked yet; the tests run them on up to 20 variables with up to a
    fifth of the values missing completely at random. ``required_paths`` is supported and
    ``forbidden_paths`` is experimental. On simulated linear-Gaussian data (10 to 50 variables, mean
    degree 2 or 4, 500 or 2,000 rows, true claims on 5 to 30% of the variable pairs) the graph found
    had no consistent extension, and so stood for no DAG, in 125 of 240 datasets, and in 19 of 20
    with 50 variables, 2,000 rows, and mean degree 4. On such a graph a claim acts only on an edge,
    as :class:`~andrey.Knowledge` describes, and most claims are dropped: of 69 required paths per
    fit, 4 directed an undirected edge, 11 were met by an edge the data directed, and 54 were
    dropped, and of 85 forbidden paths, 1, 2, and 82. Over every dataset the fits dropped 62% of the
    required and 60% of the forbidden paths. With true required paths the CPDAG was no farther from
    the class of graphs that the truth and the claims define than without them in any of the 36
    settings, and closer in 26. With true forbidden paths alone it was farther from that class in 4
    of the 6 settings with 50 variables and mean degree 2, by 0.72 to 1.95 edges per graph over 40
    datasets, since the class is more directed than the graph the dropped claims leave; no single
    setting was farther on two separate sets of 20 datasets. The CPDAG was closer to the true DAG in
    35 of the 36 settings with required paths and 32 with forbidden paths, and farther in none. The
    numbers are a reference, not a hard limit: they change with the data generator, the density, the
    method's settings, and the machine.

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
    >>> import numpy as np
    >>> from andrey import mmpc
    >>> rng = np.random.default_rng(0)
    >>> a, b = rng.standard_normal((2, 1000))
    >>> c = a + b + 0.5 * rng.standard_normal(1000)
    >>> out = mmpc(np.column_stack([a, b, c]), target=0)
    >>> out.metadata["parents_and_children"]
    [2]
    """
    return fit_blanket(
        "mmpc",
        data,
        target=target,
        structure=structure,
        alpha=alpha,
        indep_test=indep_test,
        degrees_of_freedom=degrees_of_freedom,
        categorical=categorical,
        seed=seed,
        max_depth=max_depth,
        knowledge=knowledge,
    )


@method_spec("hiton_pc")
def hiton_pc(
    data: npt.ArrayLike,
    *,
    target: str | int | None = None,
    structure: str = "skeleton",
    alpha: float = 0.05,
    indep_test: str | None = None,
    degrees_of_freedom: str = "mean",
    categorical: Sequence[str | int] | None = None,
    missing: str | None = None,
    seed: int | None = None,
    max_depth: int | None = 3,
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn a skeleton, or one variable's parents and children, with HITON-PC.

    Semi-interleaved HITON-PC [1]_ finds the variables adjacent to a target in the causal graph, its
    parents and children, without telling them apart, by conditional-independence tests. Every
    variable's set together gives the skeleton, which ``structure="cpdag"`` orients into a CPDAG by
    the collider tests of the Grow-Shrink structure learner [2]_ [3]_.

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
        Significance level of the conditional-independence test, in ``(0, 1)``. Smaller values give
        smaller sets.
    indep_test : str or None, default=None
        Conditional-independence test, any of :func:`~andrey.pc`'s: ``"fisherz"``,
        ``"student_t"``, or ``"normal_scores"`` for continuous data, ``"chisq"`` or ``"gsq"`` for
        categorical data, ``"degen_gauss"`` or ``"cond_gauss"`` for mixed data, or a nonlinear
        test. ``None`` picks from the declared column types, as for :func:`~andrey.pc`.
        ``metadata["indep_test"]`` names the test the fit used: the one picked from the
        declared column types when this is ``None``.
    degrees_of_freedom : {"mean", "nonempty"}, default="mean"
        Degrees of freedom of the categorical tests, ``"chisq"`` and ``"gsq"``. ``"mean"`` is the
        mean of the statistic when ``y`` is shuffled within each configuration of the conditioning
        set; ``"nonempty"`` is ``(r_x - 1) * (r_y - 1)`` per configuration over the values of ``x``
        and ``y`` present in it, the table's empty rows and columns struck. The two differ only on
        sparse tables. Only ``"chisq"`` and ``"gsq"`` read it: passing it with another test raises
        ``ValueError``.
    categorical : sequence of str or int, or None, default=None
        The categorical columns of ``data``, for a mixed-data test, as for :func:`~andrey.pc`. A
        continuous test raises ``ValueError`` on a declared column.
    missing : {"testwise"} or None, default=None
        How missing values (``NaN``) are handled. ``"testwise"`` deletes rows per test: each test
        reads only the rows where all of its variables are observed. ``None`` picks the method's
        default, ``"testwise"``. On data with missing values, ``metadata["missing"]`` records the
        handling used. See **Missing values.** in Notes.
    seed : int or None, default=None
        ``None``, the default, is the configured seed, ``ANDREY_SEED`` or ``0`` when it is unset,
        not a fresh random one, so a fit repeats. Seed for the random draws of ``indep_test``, which
        only ``"rcot"``, ``"cmiknn"``, and ``"gpdc"`` make; the other tests ignore it. With those
        tests, ``metadata["seed"]`` records the seed used.
    max_depth : int or None, default=3
        The maximum conditioning-set size of every conditional-independence test, at least ``0``:
        ``0`` runs only unconditional tests, and ``None`` sets no limit. The paper leaves the cap
        open and runs this search with caps from 1 to 4 [1]_; the default ``3`` is the cap of its
        Markov-blanket experiments. A smaller cap runs fewer tests and can keep variables that a
        larger conditioning set would remove. A test
        conditions on a subset of the current set; the set itself may grow larger. A collider test
        conditions on the middle node as well, so with ``0`` no triple is decided and the CPDAG
        directs only the edges ``knowledge`` decides.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the pairs
        allowed to be adjacent, and temporal tiers, all honored. Each end of a required edge is in
        the other's set from the start, and no test removes it; a pair the knowledge makes
        non-adjacent, forbidden both ways or outside ``allowed``, is never a candidate. Under
        ``structure="cpdag"``, the result directs every edge the knowledge decides, by a tier or an
        edge forbidden one way, before the colliders, which never reverse it, and Meek's rules
        direct the edges that follow, giving the maximally oriented PDAG (MPDAG)
        :class:`~andrey.Knowledge` describes; knowledge that orients the edges the search kept into
        a directed cycle raises ``ValueError``. The undirected outputs do not read tiers or edges
        forbidden one way, and with them or a ``target``, ``required_paths`` and ``forbidden_paths``
        raise ``NotImplementedError``. In a CPDAG, ``required_paths`` and ``forbidden_paths`` keep
        the DAGs the graph stands for in which each listed path holds or does not, as
        :class:`~andrey.Knowledge` describes; a claim no such DAG meets is dropped with a
        :class:`~andrey.KnowledgeWarning` and listed in ``metadata["unmet_knowledge"]``; when the
        graph found has no consistent extension, and so stands for no DAG, a claim on a pair joined
        by an undirected edge directs that edge, Meek's rules follow, ``metadata["edge_claims"]``
        lists the claims applied this way, and the result is not guaranteed to stand for a DAG.

    Returns
    -------
    StructureOutput
        Without ``target``, wraps the graph ``structure`` names: an ``undirected``
        :class:`~andrey.core.GraphStructure` for ``"skeleton"``, whose edges have no arrowhead and
        join each pair in each other's set, or a ``cpdag`` one for ``"cpdag"``, whose undirected
        edges the tests cannot orient. With ``target``, an ``undirected`` one whose edges join
        ``target`` to each of its parents and children; ``metadata["target"]`` names it and
        ``metadata["parents_and_children"]`` lists the members in the order they joined, by column
        label or 0-based position. ``metadata["algorithm"]`` is ``"hiton_pc"``. Under a mixed-data
        setting, ``metadata["categorical"]`` lists the categorical columns it read. With a
        randomized ``indep_test``, ``metadata["seed"]`` is the seed used. ``metadata["n_tests"]``
        is the number of conditional-independence tests the fit evaluated; a query answered again
        from a kept answer is not a new test. ``metadata["tie_break"]`` is the tie-break rule the
        fit ran under, ``andrey.config.tie_break``.

    Raises
    ------
    TypeError
        If an argument has the wrong type, such as a string for a number; the message names the
        method, the argument, what it accepts, and the value. Also if ``target`` is neither a column
        label nor a position, ``categorical`` is not a list of column names and positions, or
        ``knowledge`` is neither ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If an argument lies outside its range or is not one of its choices; the message names the
        method, the argument, what it accepts, and the value. Also if ``target`` names a column
        ``data`` lacks, or if ``data`` or ``categorical`` does not suit the test, as for
        :func:`~andrey.pc`. With ``"fisherz"`` or ``"student_t"``, also if a column repeats another
        up to scale and shift. Also if ``knowledge`` names a column ``data`` lacks or contradicts
        itself, as :class:`~andrey.Knowledge` lists, or under ``structure="cpdag"`` orients the
        edges the search kept into a directed cycle.
    NotImplementedError
        If ``knowledge`` holds a part the method does not honor, such as ``required_paths`` or
        ``forbidden_paths`` with ``target`` or an undirected ``structure``; the message names the
        methods that honor it.

    Warns
    -----
    ExperimentalWarning
        On the first call in a process: the method is not supported (see Limitations), and its
        API may change without deprecation.
    DataTypeWarning
        As for :func:`~andrey.pc`, when a mixed-data test finds no declared categorical column or a
        setting reads columns that look like category codes as numbers; once per process.

    Notes
    -----
    **When to use.** HITON-PC suits i.i.d. observational data with no unmeasured common causes, in
    which every independence follows from the graph (faithfulness) [1]_. Its paper runs it on
    categorical data with a G-square test and on continuous data with Fisher's z [1]_; the default
    Fisher-Z test assumes continuous data with linear effects and Gaussian noise, and the other
    tests read categorical or mixed data or detect nonlinear effects, as :func:`~andrey.pc`
    describes. It is designed for many variables, each with few parents and children [1]_: it
    orders the candidates once and tests each one against the subsets of the members found so far
    only when it joins. With ``target``, it returns an undirected graph joining ``target`` to each
    of its parents and children, without telling the two apart; without, the graph ``structure``
    names: the undirected skeleton by default, or the CPDAG. For the Markov blanket, which adds the
    children's other parents, use :func:`~andrey.hiton_mb`, which builds on this search, or
    :func:`~andrey.iamb`; :func:`~andrey.mmpc` learns the same set by another search.

    **How it works.** A variable is a candidate when its unconditional test with the target finds
    it dependent (a p-value at or below ``alpha``); the others are dropped for good. The candidates
    join the set one at a time, the most dependent first [1]_, and each newcomer is tested against
    the subsets of the members before it and dropped for good when one separates it (a p-value
    above ``alpha``). The members already in the set are not tested again until the last candidate
    has joined; then each member, in joining order, is tested against the subsets of the other
    current members and dropped when one separates it, so a later member is tested without the
    members dropped before it. This is the semi-interleaved variant of [1]_; the interleaved variant
    tests every member after each newcomer. A member stays only when the target is in that member's
    own set as well, the symmetry check of [1]_, which removes a descendant of the target that no
    subset of the target's set separates from it. The paper's experiments ran the search without the
    check [1]_, which can keep such a descendant. With exact tests, the result is the true set of
    parents and children [1]_.

    The candidates are ordered by the p-value of their unconditional test, the measure of
    association of [1]_, compared to nine significant digits, so p-values far below ``1e-9`` keep
    their order. Where p-values agree, as when they underflow to zero, the larger test statistic
    comes first: the absolute partial correlation for ``"fisherz"`` and ``"student_t"`` and the
    statistic for ``"chisq"`` and ``"gsq"``. Candidates whose statistics also agree go to the
    column whose rank, computed from the data, comes first, as for :func:`~andrey.hc`, so the sets
    do not depend on the column order unless two columns are identical.

    On categorical data, a test runs only when ``data`` holds at least five rows per cell of its
    contingency table, the reliability rule of [1]_, whose threshold the paper leaves open and its
    authors set to five. A candidate whose unconditional test fails the rule is taken as
    independent of the target, and a conditioning set that fails it is not tested, so it drops
    nothing. The rule counts every row of ``data``, also where a test deletes rows with missing
    values.

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

    **Limitations.** Not benchmarked yet. Tested with exact d-separation on 15 variables, on
    simulated linear-Gaussian data with 20 variables and 2,000 rows, and on categorical data with
    10 variables and 3,000 rows; missing values on up to 20 variables with up to a fifth of the
    values missing completely at random. A newcomer is tested only against the members before it,
    so a member that a later one separates stays until the last candidate has joined and enlarges
    the conditioning sets until then. Under the default ``max_depth=3``, a variable that only a
    larger conditioning set separates from the target stays [1]_. Each candidate takes up to one
    test per subset of the current set, so the cost grows with the size of the set, up to
    ``max_depth`` members at a time, and the symmetry check runs the search once more for each
    member. The CPDAG tests every subset of the smaller blanket for each unshielded triple, so a
    triple whose smaller blanket holds ``m`` variables takes up to ``2**m`` tests. The numbers are a
    reference, not a hard limit: they change with the data generator, the density, the method's
    settings, and the machine.

    References
    ----------
    .. [1] Aliferis, C. F., Statnikov, A., Tsamardinos, I., Mani, S., and Koutsoukos, X. D.
       (2010). Local causal and Markov blanket induction for causal discovery and feature
       selection for classification part I: Algorithms and empirical evaluation. Journal of
       Machine Learning Research 11, 171-234.
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
    >>> import numpy as np
    >>> from andrey import hiton_pc
    >>> rng = np.random.default_rng(0)
    >>> a, b = rng.standard_normal((2, 1000))
    >>> c = a + b + 0.5 * rng.standard_normal(1000)
    >>> out = hiton_pc(np.column_stack([a, b, c]), target=0)
    >>> out.metadata["parents_and_children"]
    [2]
    """
    return fit_blanket(
        "hiton_pc",
        data,
        target=target,
        structure=structure,
        alpha=alpha,
        indep_test=indep_test,
        degrees_of_freedom=degrees_of_freedom,
        categorical=categorical,
        seed=seed,
        max_depth=max_depth,
        knowledge=knowledge,
    )


@method_spec("hiton_mb")
def hiton_mb(
    data: npt.ArrayLike,
    *,
    target: str | int | None = None,
    structure: str = "cpdag",
    alpha: float = 0.05,
    indep_test: str | None = None,
    degrees_of_freedom: str = "mean",
    categorical: Sequence[str | int] | None = None,
    missing: str | None = None,
    seed: int | None = None,
    max_depth: int | None = 3,
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn a CPDAG, or one variable's Markov blanket, with HITON-MB.

    A variable's Markov blanket is the smallest set of other variables given which it is
    independent of all the rest: in a causal graph, its parents, its children, and its children's
    other parents (its spouses). Semi-interleaved HITON-MB [1]_ finds the parents and children of
    the target and of each of them with HITON-PC (:func:`~andrey.hiton_pc`), then admits each
    child's other parent by one conditional-independence test. Without a target, every variable's
    sets are learned and joined into the graph ``structure`` names, by default the CPDAG of the
    Grow-Shrink structure learner [2]_ [3]_.

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
        Significance level of the conditional-independence test, in ``(0, 1)``. Smaller values give
        smaller sets.
    indep_test : str or None, default=None
        Conditional-independence test, any of :func:`~andrey.pc`'s: ``"fisherz"``,
        ``"student_t"``, or ``"normal_scores"`` for continuous data, ``"chisq"`` or ``"gsq"`` for
        categorical data, ``"degen_gauss"`` or ``"cond_gauss"`` for mixed data, or a nonlinear
        test. ``None`` picks from the declared column types, as for :func:`~andrey.pc`.
        ``metadata["indep_test"]`` names the test the fit used: the one picked from the
        declared column types when this is ``None``.
    degrees_of_freedom : {"mean", "nonempty"}, default="mean"
        Degrees of freedom of the categorical tests, ``"chisq"`` and ``"gsq"``. ``"mean"`` is the
        mean of the statistic when ``y`` is shuffled within each configuration of the conditioning
        set; ``"nonempty"`` is ``(r_x - 1) * (r_y - 1)`` per configuration over the values of ``x``
        and ``y`` present in it, the table's empty rows and columns struck. The two differ only on
        sparse tables. Only ``"chisq"`` and ``"gsq"`` read it: passing it with another test raises
        ``ValueError``.
    categorical : sequence of str or int, or None, default=None
        The categorical columns of ``data``, for a mixed-data test, as for :func:`~andrey.pc`. A
        continuous test raises ``ValueError`` on a declared column.
    missing : {"testwise"} or None, default=None
        How missing values (``NaN``) are handled. ``"testwise"`` deletes rows per test: each test
        reads only the rows where all of its variables are observed. ``None`` picks the method's
        default, ``"testwise"``. On data with missing values, ``metadata["missing"]`` records the
        handling used. See **Missing values.** in Notes.
    seed : int or None, default=None
        ``None``, the default, is the configured seed, ``ANDREY_SEED`` or ``0`` when it is unset,
        not a fresh random one, so a fit repeats. Seed for the random draws of ``indep_test``, which
        only ``"rcot"``, ``"cmiknn"``, and ``"gpdc"`` make; the other tests ignore it. With those
        tests, ``metadata["seed"]`` records the seed used.
    max_depth : int or None, default=3
        The maximum conditioning-set size of every conditional-independence test, at least ``0``:
        ``0`` runs only unconditional tests, and ``None`` sets no limit. The paper leaves the cap
        open and runs this search with a cap of 3 [1]_, the default. A smaller cap runs fewer tests
        and can keep variables that a larger conditioning set would remove. A test of HITON-PC
        conditions on a subset of the current set, which may grow larger; a spouse's test
        conditions on its separating set together with the common child, so a candidate whose
        separating set already holds ``max_depth`` variables is not tested and does not join. A
        collider test conditions on the middle node as well, so with ``0`` no spouse joins, no
        triple is decided, and the CPDAG directs only the edges ``knowledge`` decides.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the pairs
        allowed to be adjacent, and temporal tiers, all honored. Each end of a required edge is in
        the other's parents and children from the start, and no test removes it; a pair the
        knowledge makes non-adjacent, forbidden both ways or outside ``allowed``, is never a
        candidate, as a parent or child or as a spouse, and is left out of the result. Under
        ``structure="cpdag"``, the result directs every edge the knowledge decides, by a tier or an
        edge forbidden one way, before the colliders, which never reverse it, and Meek's rules
        direct the edges that follow, giving the maximally oriented PDAG (MPDAG)
        :class:`~andrey.Knowledge` describes; knowledge that orients the edges the search kept into
        a directed cycle raises ``ValueError``. The undirected outputs do not read tiers or edges
        forbidden one way, and with them or a ``target``, ``required_paths`` and ``forbidden_paths``
        raise ``NotImplementedError``. In a CPDAG, ``required_paths`` and ``forbidden_paths`` keep
        the DAGs the graph stands for in which each listed path holds or does not, as
        :class:`~andrey.Knowledge` describes; a claim no such DAG meets is dropped with a
        :class:`~andrey.KnowledgeWarning` and listed in ``metadata["unmet_knowledge"]``; when the
        graph found has no consistent extension, and so stands for no DAG, a claim on a pair joined
        by an undirected edge directs that edge, Meek's rules follow, ``metadata["edge_claims"]``
        lists the claims applied this way, and the result is not guaranteed to stand for a DAG.

    Returns
    -------
    StructureOutput
        Without ``target``, wraps the graph ``structure`` names: a ``cpdag``
        :class:`~andrey.core.GraphStructure` for ``"cpdag"``, whose undirected edges the tests
        cannot orient, or an ``undirected`` one, whose edges have no arrowhead, for ``"skeleton"``
        and ``"moral"``. With ``target``, an ``undirected`` one whose edges join ``target`` to each
        member of its Markov blanket; ``metadata["target"]`` names it,
        ``metadata["markov_blanket"]`` lists the members, the parents and children first, then the
        spouses, each in the order they joined, and ``metadata["parents_and_children"]`` lists the
        parents and children alone, by column label or 0-based position. ``metadata["algorithm"]``
        is ``"hiton_mb"``. Under a mixed-data setting, ``metadata["categorical"]`` lists the
        categorical columns it read. With a randomized ``indep_test``, ``metadata["seed"]`` is the
        seed used. ``metadata["n_tests"]`` is the number of conditional-independence tests the fit
        evaluated; a query answered again from a kept answer is not a new test.
        ``metadata["tie_break"]`` is the tie-break rule the fit ran under,
        ``andrey.config.tie_break``.

    Raises
    ------
    TypeError
        If an argument has the wrong type, such as a string for a number; the message names the
        method, the argument, what it accepts, and the value. Also if ``target`` is neither a column
        label nor a position, ``categorical`` is not a list of column names and positions, or
        ``knowledge`` is neither ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If an argument lies outside its range or is not one of its choices; the message names the
        method, the argument, what it accepts, and the value. Also if ``target`` names a column
        ``data`` lacks, or if ``data`` or ``categorical`` does not suit the test, as for
        :func:`~andrey.pc`. With ``"fisherz"`` or ``"student_t"``, also if a column repeats another
        up to scale and shift. Also if ``knowledge`` names a column ``data`` lacks or contradicts
        itself, as :class:`~andrey.Knowledge` lists, or under ``structure="cpdag"`` orients the
        edges the search kept into a directed cycle.
    NotImplementedError
        If ``knowledge`` holds a part the method does not honor, such as ``required_paths`` or
        ``forbidden_paths`` with ``target`` or an undirected ``structure``; the message names the
        methods that honor it.

    Warns
    -----
    ExperimentalWarning
        On the first call in a process: the method is not supported (see Limitations), and its
        API may change without deprecation.
    DataTypeWarning
        As for :func:`~andrey.pc`, when a mixed-data test finds no declared categorical column or a
        setting reads columns that look like category codes as numbers; once per process.

    Notes
    -----
    **When to use.** HITON-MB suits i.i.d. observational data with no unmeasured common causes, in
    which every independence follows from the graph (faithfulness) [1]_. Its paper runs it on
    categorical data with a G-square test and on continuous data with Fisher's z [1]_; the default
    Fisher-Z test assumes continuous data with linear effects and Gaussian noise, and the other
    tests read categorical or mixed data or detect nonlinear effects, as :func:`~andrey.pc`
    describes. It is designed for many variables, each with few parents and children [1]_: its
    tests condition on subsets of a parents-and-children set, never on the whole blanket. With
    ``target``, it returns an undirected graph joining ``target`` to each member of its Markov
    blanket, and ``metadata["parents_and_children"]`` tells the parents and children from the
    spouses; without, the graph ``structure`` names: the CPDAG by default, or the undirected
    skeleton or moral graph. For the parents and children alone, use :func:`~andrey.hiton_pc`;
    :func:`~andrey.iamb` learns the same blanket by growing and shrinking it.

    **How it works.** HITON-PC gives the target's parents and children, with its symmetry check, as
    :func:`~andrey.hiton_pc` describes, and the parents and children of each of them. A variable
    outside the target's set that lies in the set of one of its members ``y`` is a candidate
    spouse, and HITON-PC recorded a conditioning set that separated it from the target: the
    target's own search did, or, for a variable the symmetry check removed, that variable's search.
    The candidate joins the blanket when the test finds it dependent on the target (a p-value at
    or below ``alpha``) given that set together with ``y`` [1]_, which with exact tests happens
    exactly when ``y`` is a common child of the two. A candidate is tested only with a ``y`` whose
    parents and children hold it [1]_ [4]_, the ``y`` in the order they joined, and it joins at the
    first one that admits it; a ``y`` already in the separating set is not tested, as that test
    repeats the one that separated the two. The paper defines HITON-MB with the symmetry check of
    HITON-PC [1]_, which this search runs; its experiments ran their searches without the check
    unless they say otherwise [1]_. With exact tests, the result is the true Markov blanket [1]_.

    On categorical data, a test runs only when ``data`` holds at least five rows per cell of its
    contingency table, the reliability rule of [1]_, whose threshold the paper leaves open and its
    authors set to five, in HITON-PC as :func:`~andrey.hiton_pc` describes and in the spouse step.
    A spouse test that fails the rule, or that would condition on more than ``max_depth``
    variables, is not run and admits nothing, so the candidate joins only through another common
    child. The rule counts every row of ``data``, also where a test deletes rows with missing
    values.

    **From sets to a graph.** Without a target, a pair is kept in the skeleton
    (``structure="skeleton"``) when each variable is in the other's parents and children, and in
    the moral graph (``structure="moral"``) when each is in the other's Markov blanket. The CPDAG
    decides each triple ``x - z - y`` of the skeleton with ``x`` and ``y`` not adjacent: ``z`` is a
    collider, ``x -> z <- y``, when ``x`` and ``y`` test dependent given ``z`` together with every
    subset of the smaller of their blankets without ``x``, ``y``, and ``z`` [2]_ [3]_, and a
    non-collider when one such set separates them. The colliders are oriented as :func:`~andrey.pc`
    orients them, the most clearly dependent first: by the smallest absolute partial correlation
    over the collider's tests for ``"fisherz"`` and ``"student_t"``, and by the largest p-value,
    smallest first, for the other tests. When two colliders that rank equal to nine decimals
    dispute an edge, it stays undirected. Meek's rules [5]_ then orient the edges that follow, in
    rounds, as for :func:`~andrey.pc`, so the result does not depend on the column order; under
    ``andrey.config(tie_break="position")`` the colliders and the rules follow the column order
    instead. With exact tests, the result is the true CPDAG [2]_.

    **Missing values.** A ``NaN`` in ``data`` marks a missing value. Each test reads only the rows
    where all of its variables are observed (test-wise deletion) [6]_ [7]_, so it keeps rows that
    deleting every incomplete row first would drop; ``metadata["rows_tested"]`` gives the fewest and
    the median rows a test read. When values are missing completely at random, the rows a test keeps
    are a random sample, so each test keeps its level [7]_. When whether a value is missing depends
    on the variables, the rows a test keeps are a selected sample, and the tests can find dependence
    where the population has none and keep extra edges [7]_. A test raises ``ValueError`` when it
    has fewer rows than it needs, ``|S| + 3`` for ``"fisherz"`` and ``"student_t"`` and 2 for the
    other tests, or when a continuous column is constant on its rows.

    **Limitations.** Not benchmarked yet. Tested with exact d-separation on 15 variables, on
    simulated linear-Gaussian data with 20 variables and 2,000 rows, and on categorical data with
    10 variables and 3,000 rows; missing values on up to 20 variables with up to a fifth of the
    values missing completely at random. The parents and children come from HITON-PC, with its
    limits: a newcomer is tested only against the members before it, so a member that a later one
    separates stays until the last candidate has joined and enlarges the conditioning sets until
    then. A spouse is found only through a child that the search found, so an edge its tests miss
    also hides that child's other parents, and each spouse's verdict rests on the one separating
    set HITON-PC recorded for it. On the linear-Gaussian data above, the moral graph had up to about
    a third as many wrong pairs as true ones, most of them spouses behind missed edges, and the
    paper found that HITON-MB needs more rows than HITON-PC [1]_. Under the default
    ``max_depth=3``, a variable that only a larger conditioning set separates stays, and a spouse
    whose separating set holds three variables does not join. The search runs HITON-PC for the
    target, for each of its parents and children, and, for the symmetry check, for each of their
    members, so its cost grows with the sets two steps from the target. The CPDAG tests every subset
    of the smaller blanket for each unshielded triple, so a triple whose smaller blanket holds ``m``
    variables takes up to ``2**m`` tests. The numbers are a reference, not a hard limit: they change
    with the data generator, the density, the method's settings, and the machine.

    References
    ----------
    .. [1] Aliferis, C. F., Statnikov, A., Tsamardinos, I., Mani, S., and Koutsoukos, X. D.
       (2010). Local causal and Markov blanket induction for causal discovery and feature
       selection for classification part I: Algorithms and empirical evaluation. Journal of
       Machine Learning Research 11, 171-234.
    .. [2] Margaritis, D. (2003). Learning Bayesian network model structure from data. PhD thesis,
       Carnegie Mellon University, CMU-CS-03-153.
    .. [3] Margaritis, D., and Thrun, S. (2000). Bayesian network induction via local
       neighborhoods. Advances in Neural Information Processing Systems 12, 505-511.
    .. [4] Pena, J. M., Nilsson, R., Bjorkegren, J., and Tegner, J. (2007). Towards scalable and
       data efficient learning of Markov boundaries. International Journal of Approximate
       Reasoning 45(2), 211-232.
    .. [5] Meek, C. (1995). Causal inference and causal explanation with background knowledge.
       UAI, 403-410.
    .. [6] Strobl, E. V., Visweswaran, S., and Spirtes, P. L. (2018). Fast causal inference with
       non-random missingness by test-wise deletion. International Journal of Data Science and
       Analytics 6(1), 47-62.
    .. [7] Tu, R., Zhang, C., Ackermann, P., Mohan, K., Kjellstrom, H., and Zhang, K. (2019). Causal
       discovery in the presence of missing data. Proceedings of Machine Learning Research 89,
       1762-1770.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import hiton_mb
    >>> rng = np.random.default_rng(0)
    >>> a, b = rng.standard_normal((2, 1000))
    >>> c = a + b + 0.5 * rng.standard_normal(1000)
    >>> out = hiton_mb(np.column_stack([a, b, c]), target=0)
    >>> out.metadata["markov_blanket"], out.metadata["parents_and_children"]
    ([2, 1], [2])
    >>> hiton_mb(np.column_stack([a, b, c])).structure.oriented_edges()
    [(0, 2, 'directed'), (1, 2, 'directed')]
    """
    return fit_blanket(
        "hiton_mb",
        data,
        target=target,
        structure=structure,
        alpha=alpha,
        indep_test=indep_test,
        degrees_of_freedom=degrees_of_freedom,
        categorical=categorical,
        seed=seed,
        max_depth=max_depth,
        knowledge=knowledge,
    )


@method_spec("mmmb")
def mmmb(
    data: npt.ArrayLike,
    *,
    target: str | int | None = None,
    structure: str = "cpdag",
    alpha: float = 0.05,
    indep_test: str | None = None,
    degrees_of_freedom: str = "mean",
    categorical: Sequence[str | int] | None = None,
    missing: str | None = None,
    seed: int | None = None,
    max_depth: int | None = None,
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn a CPDAG, or one variable's Markov blanket, with MMMB.

    A variable's Markov blanket is the smallest set of other variables given which it is
    independent of all the rest: in a causal graph, its parents, its children, and its children's
    other parents (its spouses). Max-min Markov blanket (MMMB) [1]_ finds the parents and children
    of the target and of each of them with MMPC (:func:`~andrey.mmpc`), then admits each child's
    other parent by one conditional-independence test. It runs the authors' corrected form: MMPC
    with its symmetry check [2]_, and a spouse test only through a variable adjacent to both [3]_.
    Without a target, every variable's sets are learned and joined into the graph ``structure``
    names, by default the CPDAG of the Grow-Shrink structure learner [4]_ [5]_.

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
        Significance level of the conditional-independence test, in ``(0, 1)``. Smaller values give
        smaller sets.
    indep_test : str or None, default=None
        Conditional-independence test, any of :func:`~andrey.pc`'s: ``"fisherz"``,
        ``"student_t"``, or ``"normal_scores"`` for continuous data, ``"chisq"`` or ``"gsq"`` for
        categorical data, ``"degen_gauss"`` or ``"cond_gauss"`` for mixed data, or a nonlinear
        test. ``None`` picks from the declared column types, as for :func:`~andrey.pc`.
        ``metadata["indep_test"]`` names the test the fit used: the one picked from the
        declared column types when this is ``None``.
    degrees_of_freedom : {"mean", "nonempty"}, default="mean"
        Degrees of freedom of the categorical tests, ``"chisq"`` and ``"gsq"``. ``"mean"`` is the
        mean of the statistic when ``y`` is shuffled within each configuration of the conditioning
        set; ``"nonempty"`` is ``(r_x - 1) * (r_y - 1)`` per configuration over the values of ``x``
        and ``y`` present in it, the table's empty rows and columns struck. The two differ only on
        sparse tables. Only ``"chisq"`` and ``"gsq"`` read it: passing it with another test raises
        ``ValueError``.
    categorical : sequence of str or int, or None, default=None
        The categorical columns of ``data``, for a mixed-data test, as for :func:`~andrey.pc`. A
        continuous test raises ``ValueError`` on a declared column.
    missing : {"testwise"} or None, default=None
        How missing values (``NaN``) are handled. ``"testwise"`` deletes rows per test: each test
        reads only the rows where all of its variables are observed. ``None`` picks the method's
        default, ``"testwise"``. On data with missing values, ``metadata["missing"]`` records the
        handling used. See **Missing values.** in Notes.
    seed : int or None, default=None
        ``None``, the default, is the configured seed, ``ANDREY_SEED`` or ``0`` when it is unset,
        not a fresh random one, so a fit repeats. Seed for the random draws of ``indep_test``, which
        only ``"rcot"``, ``"cmiknn"``, and ``"gpdc"`` make; the other tests ignore it. With those
        tests, ``metadata["seed"]`` records the seed used.
    max_depth : int or None, default=None
        The maximum conditioning-set size of every conditional-independence test, at least ``0``:
        ``0`` runs only unconditional tests, and ``None`` sets no limit. A smaller cap runs fewer
        tests and can keep variables that a larger conditioning set would remove. A test of MMPC
        conditions on a subset of the current set, which may grow larger; a spouse's test
        conditions on its separating set together with the common child, so a candidate whose
        separating set already holds ``max_depth`` variables is not tested and does not join. A
        collider test conditions on the middle node as well, so with ``0`` no spouse joins, no
        triple is decided, and the CPDAG directs only the edges ``knowledge`` decides.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the pairs
        allowed to be adjacent, and temporal tiers, all honored. Each end of a required edge is in
        the other's parents and children from the start, and no test removes it; a pair the
        knowledge makes non-adjacent, forbidden both ways or outside ``allowed``, is never a
        candidate, as a parent or child or as a spouse, and is left out of the result. Under
        ``structure="cpdag"``, the result directs every edge the knowledge decides, by a tier or an
        edge forbidden one way, before the colliders, which never reverse it, and Meek's rules
        direct the edges that follow, giving the maximally oriented PDAG (MPDAG)
        :class:`~andrey.Knowledge` describes; knowledge that orients the edges the search kept into
        a directed cycle raises ``ValueError``. The undirected outputs do not read tiers or edges
        forbidden one way, and with them or a ``target``, ``required_paths`` and ``forbidden_paths``
        raise ``NotImplementedError``. In a CPDAG, ``required_paths`` and ``forbidden_paths`` keep
        the DAGs the graph stands for in which each listed path holds or does not, as
        :class:`~andrey.Knowledge` describes; a claim no such DAG meets is dropped with a
        :class:`~andrey.KnowledgeWarning` and listed in ``metadata["unmet_knowledge"]``; when the
        graph found has no consistent extension, and so stands for no DAG, a claim on a pair joined
        by an undirected edge directs that edge, Meek's rules follow, ``metadata["edge_claims"]``
        lists the claims applied this way, and the result is not guaranteed to stand for a DAG.

    Returns
    -------
    StructureOutput
        Without ``target``, wraps the graph ``structure`` names: a ``cpdag``
        :class:`~andrey.core.GraphStructure` for ``"cpdag"``, whose undirected edges the tests
        cannot orient, or an ``undirected`` one, whose edges have no arrowhead, for ``"skeleton"``
        and ``"moral"``. With ``target``, an ``undirected`` one whose edges join ``target`` to each
        member of its Markov blanket; ``metadata["target"]`` names it,
        ``metadata["markov_blanket"]`` lists the members, the parents and children first, then the
        spouses, each in the order they joined, and ``metadata["parents_and_children"]`` lists the
        parents and children alone, by column label or 0-based position. ``metadata["algorithm"]``
        is ``"mmmb"``. Under a mixed-data setting, ``metadata["categorical"]`` lists the
        categorical columns it read. With a randomized ``indep_test``, ``metadata["seed"]`` is the
        seed used. ``metadata["n_tests"]`` is the number of conditional-independence tests the fit
        evaluated; a query answered again from a kept answer is not a new test.
        ``metadata["tie_break"]`` is the tie-break rule the fit ran under,
        ``andrey.config.tie_break``.

    Raises
    ------
    TypeError
        If an argument has the wrong type, such as a string for a number; the message names the
        method, the argument, what it accepts, and the value. Also if ``target`` is neither a column
        label nor a position, ``categorical`` is not a list of column names and positions, or
        ``knowledge`` is neither ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If an argument lies outside its range or is not one of its choices; the message names the
        method, the argument, what it accepts, and the value. Also if ``target`` names a column
        ``data`` lacks, or if ``data`` or ``categorical`` does not suit the test, as for
        :func:`~andrey.pc`. With ``"fisherz"`` or ``"student_t"``, also if a column repeats another
        up to scale and shift. Also if ``knowledge`` names a column ``data`` lacks or contradicts
        itself, as :class:`~andrey.Knowledge` lists, or under ``structure="cpdag"`` orients the
        edges the search kept into a directed cycle.
    NotImplementedError
        If ``knowledge`` holds a part the method does not honor, such as ``required_paths`` or
        ``forbidden_paths`` with ``target`` or an undirected ``structure``; the message names the
        methods that honor it.

    Warns
    -----
    ExperimentalWarning
        On the first call in a process: the method is not supported (see Limitations), and its
        API may change without deprecation.
    DataTypeWarning
        As for :func:`~andrey.pc`, when a mixed-data test finds no declared categorical column or a
        setting reads columns that look like category codes as numbers; once per process.

    Notes
    -----
    **When to use.** MMMB suits i.i.d. observational data with no unmeasured common causes, in
    which every independence follows from the graph (faithfulness) [1]_ [3]_. Its paper runs it on
    categorical data with a G-square test [1]_; the default Fisher-Z test assumes continuous data
    with linear effects and Gaussian noise, and the other tests read categorical or mixed data or
    detect nonlinear effects, as :func:`~andrey.pc` describes. It is designed for many variables,
    each with few parents and children: its tests condition on subsets of a parents-and-children
    set, never on the whole blanket, so the rows it needs depend on how densely the graph is
    connected rather than on the size of the blanket [1]_. With ``target``, it returns an
    undirected graph joining ``target`` to each member of its Markov blanket, and
    ``metadata["parents_and_children"]`` tells the parents and children from the spouses; without,
    the graph ``structure`` names: the CPDAG by default, or the undirected skeleton or moral graph.
    For the parents and children alone, use :func:`~andrey.mmpc`; :func:`~andrey.hiton_mb` builds
    the same blanket on HITON-PC, and :func:`~andrey.iamb` learns it by growing and shrinking it.

    **How it works.** MMPC gives the target's parents and children, with its symmetry check, as
    :func:`~andrey.mmpc` describes, and the parents and children of each of them. A variable
    outside the target's set that lies in the set of one of its members ``y`` is a candidate
    spouse, and MMPC recorded a conditioning set that separated it from the target: the target's
    own search did, or, for a variable the symmetry check removed, that variable's search. The
    candidate joins the blanket when the test finds it dependent on the target (a p-value at or
    below ``alpha``) given that set together with ``y`` [1]_, which with exact tests happens
    exactly when ``y`` is a common child of the two. A candidate is tested only with a ``y`` whose
    parents and children hold it [3]_, the ``y`` in the order they joined, and it joins at the
    first one that admits it; a ``y`` already in the separating set is not tested, as that test
    repeats the one that separated the two. With exact tests, the result is the true Markov
    blanket [3]_.

    The MMMB of [1]_ differs in two steps, and Pena et al. [6]_ give a graph on which each returns
    a wrong blanket under exact tests. Its MMPC has no symmetry check, so the target's set can keep
    a descendant that no subset of the set separates from it; the authors added the check [2]_.
    It tests a candidate with every member of the target's parents and children, so a candidate
    can join through a child of the target that is not its own child; the authors' later template
    tests it only with a ``y`` whose set holds it [3]_, the rule this search follows. The
    experiments of [1]_ run a test of MMPC only with at least five rows per parameter on average;
    this search runs every test, as :func:`~andrey.mmpc` does.

    **From sets to a graph.** Without a target, a pair is kept in the skeleton
    (``structure="skeleton"``) when each variable is in the other's parents and children, and in
    the moral graph (``structure="moral"``) when each is in the other's Markov blanket. The CPDAG
    decides each triple ``x - z - y`` of the skeleton with ``x`` and ``y`` not adjacent: ``z`` is a
    collider, ``x -> z <- y``, when ``x`` and ``y`` test dependent given ``z`` together with every
    subset of the smaller of their blankets without ``x``, ``y``, and ``z`` [4]_ [5]_, and a
    non-collider when one such set separates them. The colliders are oriented as :func:`~andrey.pc`
    orients them, the most clearly dependent first: by the smallest absolute partial correlation
    over the collider's tests for ``"fisherz"`` and ``"student_t"``, and by the largest p-value,
    smallest first, for the other tests. When two colliders that rank equal to nine decimals
    dispute an edge, it stays undirected. Meek's rules [7]_ then orient the edges that follow, in
    rounds, as for :func:`~andrey.pc`, so the result does not depend on the column order; under
    ``andrey.config(tie_break="position")`` the colliders and the rules follow the column order
    instead. With exact tests, the result is the true CPDAG [4]_.

    **Missing values.** A ``NaN`` in ``data`` marks a missing value. Each test reads only the rows
    where all of its variables are observed (test-wise deletion) [8]_ [9]_, so it keeps rows that
    deleting every incomplete row first would drop; ``metadata["rows_tested"]`` gives the fewest and
    the median rows a test read. When values are missing completely at random, the rows a test keeps
    are a random sample, so each test keeps its level [9]_. When whether a value is missing depends
    on the variables, the rows a test keeps are a selected sample, and the tests can find dependence
    where the population has none and keep extra edges [9]_. A test raises ``ValueError`` when it
    has fewer rows than it needs, ``|S| + 3`` for ``"fisherz"`` and ``"student_t"`` and 2 for the
    other tests, or when a continuous column is constant on its rows.

    **Limitations.** Not benchmarked yet. Tested with exact d-separation on 15 variables, on
    simulated linear-Gaussian data with 20 variables and 2,000 rows, and on categorical data with
    10 variables and 3,000 rows; missing values on up to 20 variables with up to a fifth of the
    values missing completely at random. A spouse is found only through a child that the search
    found, so an edge its tests miss also hides that child's other parents, and each spouse's
    verdict rests on the one separating set MMPC recorded for it. The search runs MMPC for the
    target, for each of its parents and children, and, for the symmetry check, for each of their
    members, so its cost grows with the sets two steps from the target. The CPDAG tests every
    subset of the smaller blanket for each unshielded triple, so a triple whose smaller blanket
    holds ``m`` variables takes up to ``2**m`` tests. The numbers are a reference, not a hard limit:
    they change with the data generator, the density, the method's settings, and the machine.

    References
    ----------
    .. [1] Tsamardinos, I., Aliferis, C. F., and Statnikov, A. (2003). Time and sample efficient
       discovery of Markov blankets and direct causal relations. Proceedings of the Ninth ACM
       SIGKDD International Conference on Knowledge Discovery and Data Mining, 673-678.
    .. [2] Tsamardinos, I., Brown, L. E., and Aliferis, C. F. (2006). The max-min hill-climbing
       Bayesian network structure learning algorithm. Machine Learning 65(1), 31-78.
    .. [3] Aliferis, C. F., Statnikov, A., Tsamardinos, I., Mani, S., and Koutsoukos, X. D.
       (2010). Local causal and Markov blanket induction for causal discovery and feature
       selection for classification part I: Algorithms and empirical evaluation. Journal of
       Machine Learning Research 11, 171-234.
    .. [4] Margaritis, D. (2003). Learning Bayesian network model structure from data. PhD thesis,
       Carnegie Mellon University, CMU-CS-03-153.
    .. [5] Margaritis, D., and Thrun, S. (2000). Bayesian network induction via local
       neighborhoods. Advances in Neural Information Processing Systems 12, 505-511.
    .. [6] Pena, J. M., Nilsson, R., Bjorkegren, J., and Tegner, J. (2007). Towards scalable and
       data efficient learning of Markov boundaries. International Journal of Approximate
       Reasoning 45(2), 211-232.
    .. [7] Meek, C. (1995). Causal inference and causal explanation with background knowledge.
       UAI, 403-410.
    .. [8] Strobl, E. V., Visweswaran, S., and Spirtes, P. L. (2018). Fast causal inference with
       non-random missingness by test-wise deletion. International Journal of Data Science and
       Analytics 6(1), 47-62.
    .. [9] Tu, R., Zhang, C., Ackermann, P., Mohan, K., Kjellstrom, H., and Zhang, K. (2019). Causal
       discovery in the presence of missing data. Proceedings of Machine Learning Research 89,
       1762-1770.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import mmmb
    >>> rng = np.random.default_rng(0)
    >>> a, b = rng.standard_normal((2, 1000))
    >>> c = a + b + 0.5 * rng.standard_normal(1000)
    >>> out = mmmb(np.column_stack([a, b, c]), target=0)
    >>> out.metadata["markov_blanket"], out.metadata["parents_and_children"]
    ([2, 1], [2])
    >>> mmmb(np.column_stack([a, b, c])).structure.oriented_edges()
    [(0, 2, 'directed'), (1, 2, 'directed')]
    """
    return fit_blanket(
        "mmmb",
        data,
        target=target,
        structure=structure,
        alpha=alpha,
        indep_test=indep_test,
        degrees_of_freedom=degrees_of_freedom,
        categorical=categorical,
        seed=seed,
        max_depth=max_depth,
        knowledge=knowledge,
    )


@method_spec("pcmb")
def pcmb(
    data: npt.ArrayLike,
    *,
    target: str | int | None = None,
    structure: str = "cpdag",
    alpha: float = 0.01,
    indep_test: str | None = None,
    degrees_of_freedom: str = "nonempty",
    categorical: Sequence[str | int] | None = None,
    missing: str | None = None,
    seed: int | None = None,
    max_depth: int | None = None,
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn a CPDAG, or one variable's Markov blanket, with PCMB.

    A variable's Markov blanket is the smallest set of other variables given which it is
    independent of all the rest: in a causal graph, its parents, its children, and its children's
    other parents (its spouses). Parents-and-children based Markov boundary (PCMB) [1]_ finds the
    parents and children of the target and of each of them with its own search, GetPCD followed by
    a symmetry check, then admits each child's other parent by one conditional-independence test
    given the set that separated it from the target and the common child. Without a target, every
    variable's sets are learned and joined into the graph ``structure`` names, by default the
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
    alpha : float, default=0.01
        Significance level of the conditional-independence test, in ``(0, 1)``. Smaller values give
        smaller sets. Pena et al. [1]_ run PCMB at 0.01, the default.
    indep_test : str or None, default=None
        Conditional-independence test, any of :func:`~andrey.pc`'s: ``"fisherz"``,
        ``"student_t"``, or ``"normal_scores"`` for continuous data, ``"chisq"`` or ``"gsq"`` for
        categorical data, ``"degen_gauss"`` or ``"cond_gauss"`` for mixed data, or a nonlinear
        test. ``None`` picks from the declared column types, as for :func:`~andrey.pc`.
        ``metadata["indep_test"]`` names the test the fit used: the one picked from the
        declared column types when this is ``None``.
    degrees_of_freedom : {"mean", "nonempty"}, default="nonempty"
        Degrees of freedom of the categorical tests, ``"chisq"`` and ``"gsq"``. ``"mean"`` is the
        mean of the statistic when ``y`` is shuffled within each configuration of the conditioning
        set; ``"nonempty"`` is ``(r_x - 1) * (r_y - 1)`` per configuration over the values of ``x``
        and ``y`` present in it, the table's empty rows and columns struck. The two differ only on
        sparse tables. Only ``"chisq"`` and ``"gsq"`` read it: passing it with another test raises
        ``ValueError``. PCMB defaults to ``"nonempty"``, the degrees of freedom of the G-square
        test its paper runs, after Spirtes, Glymour and Scheines [1]_ [7]_.
    categorical : sequence of str or int, or None, default=None
        The categorical columns of ``data``, for a mixed-data test, as for :func:`~andrey.pc`. A
        continuous test raises ``ValueError`` on a declared column.
    missing : {"testwise"} or None, default=None
        How missing values (``NaN``) are handled. ``"testwise"`` deletes rows per test: each test
        reads only the rows where all of its variables are observed. ``None`` picks the method's
        default, ``"testwise"``. On data with missing values, ``metadata["missing"]`` records the
        handling used. See **Missing values.** in Notes.
    seed : int or None, default=None
        ``None``, the default, is the configured seed, ``ANDREY_SEED`` or ``0`` when it is unset,
        not a fresh random one, so a fit repeats. Seed for the random draws of ``indep_test``, which
        only ``"rcot"``, ``"cmiknn"``, and ``"gpdc"`` make; the other tests ignore it. With those
        tests, ``metadata["seed"]`` records the seed used.
    max_depth : int or None, default=None
        The maximum conditioning-set size of every conditional-independence test, at least ``0``:
        ``0`` runs only unconditional tests, and ``None`` sets no limit. A smaller cap runs fewer
        tests and can keep variables that a larger conditioning set would remove. A test of GetPCD
        conditions on a subset of the current set, which may grow larger; a spouse's test
        conditions on its separating set together with the common child, so a candidate whose
        separating set already holds ``max_depth`` variables is not tested and does not join. A
        collider test conditions on the middle node as well, so with ``0`` no spouse joins, no
        triple is decided, and the CPDAG directs only the edges ``knowledge`` decides.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the pairs
        allowed to be adjacent, and temporal tiers, all honored. Each end of a required edge is in
        the other's parents and children from the start, and no test removes it; a pair the
        knowledge makes non-adjacent, forbidden both ways or outside ``allowed``, is never a
        candidate, as a parent or child or as a spouse, and is left out of the result. Under
        ``structure="cpdag"``, the result directs every edge the knowledge decides, by a tier or an
        edge forbidden one way, before the colliders, which never reverse it, and Meek's rules
        direct the edges that follow, giving the maximally oriented PDAG (MPDAG)
        :class:`~andrey.Knowledge` describes; knowledge that orients the edges the search kept into
        a directed cycle raises ``ValueError``. The undirected outputs do not read tiers or edges
        forbidden one way, and with them or a ``target``, ``required_paths`` and ``forbidden_paths``
        raise ``NotImplementedError``. In a CPDAG, ``required_paths`` and ``forbidden_paths`` keep
        the DAGs the graph stands for in which each listed path holds or does not, as
        :class:`~andrey.Knowledge` describes; a claim no such DAG meets is dropped with a
        :class:`~andrey.KnowledgeWarning` and listed in ``metadata["unmet_knowledge"]``; when the
        graph found has no consistent extension, and so stands for no DAG, a claim on a pair joined
        by an undirected edge directs that edge, Meek's rules follow, ``metadata["edge_claims"]``
        lists the claims applied this way, and the result is not guaranteed to stand for a DAG.

    Returns
    -------
    StructureOutput
        Without ``target``, wraps the graph ``structure`` names: a ``cpdag``
        :class:`~andrey.core.GraphStructure` for ``"cpdag"``, whose undirected edges the tests
        cannot orient, or an ``undirected`` one, whose edges have no arrowhead, for ``"skeleton"``
        and ``"moral"``. With ``target``, an ``undirected`` one whose edges join ``target`` to each
        member of its Markov blanket; ``metadata["target"]`` names it,
        ``metadata["markov_blanket"]`` lists the members, the parents and children first, then the
        spouses, each in the order they joined, and ``metadata["parents_and_children"]`` lists the
        parents and children alone, by column label or 0-based position. ``metadata["algorithm"]``
        is ``"pcmb"``. Under a mixed-data setting, ``metadata["categorical"]`` lists the
        categorical columns it read. With a randomized ``indep_test``, ``metadata["seed"]`` is the
        seed used. ``metadata["n_tests"]`` is the number of conditional-independence tests the fit
        evaluated; a query answered again from a kept answer is not a new test.
        ``metadata["tie_break"]`` is the tie-break rule the fit ran under,
        ``andrey.config.tie_break``.

    Raises
    ------
    TypeError
        If an argument has the wrong type, such as a string for a number; the message names the
        method, the argument, what it accepts, and the value. Also if ``target`` is neither a column
        label nor a position, ``categorical`` is not a list of column names and positions, or
        ``knowledge`` is neither ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If an argument lies outside its range or is not one of its choices; the message names the
        method, the argument, what it accepts, and the value. Also if ``target`` names a column
        ``data`` lacks, or if ``data`` or ``categorical`` does not suit the test, as for
        :func:`~andrey.pc`. With ``"fisherz"`` or ``"student_t"``, also if a column repeats another
        up to scale and shift. Also if ``knowledge`` names a column ``data`` lacks or contradicts
        itself, as :class:`~andrey.Knowledge` lists, or under ``structure="cpdag"`` orients the
        edges the search kept into a directed cycle.
    NotImplementedError
        If ``knowledge`` holds a part the method does not honor, such as ``required_paths`` or
        ``forbidden_paths`` with ``target`` or an undirected ``structure``; the message names the
        methods that honor it.

    Warns
    -----
    ExperimentalWarning
        On the first call in a process: the method is not supported (see Limitations), and its
        API may change without deprecation.
    DataTypeWarning
        As for :func:`~andrey.pc`, when a mixed-data test finds no declared categorical column or a
        setting reads columns that look like category codes as numbers; once per process.

    Notes
    -----
    **When to use.** PCMB suits i.i.d. observational data with no unmeasured common causes, in
    which every independence follows from the graph (faithfulness) [1]_. Its paper runs it on
    categorical data with a G-square test at the level 0.01 [1]_; the default Fisher-Z test assumes
    continuous data with linear effects and Gaussian noise, and the other tests read categorical or
    mixed data or detect nonlinear effects, as :func:`~andrey.pc` describes; the paper names
    Fisher's z for continuous data [1]_. It is designed for many variables and few rows, each
    variable with few parents and children: its tests condition on subsets of a
    parents-and-children set, never on the whole blanket, so the rows it needs depend on how
    densely the graph is connected rather than on the size of the blanket [1]_. With ``target``, it
    returns an undirected graph joining ``target`` to each member of its Markov blanket, and
    ``metadata["parents_and_children"]`` tells the parents and children from the spouses; without,
    the graph ``structure`` names: the CPDAG by default, or the undirected skeleton or moral graph.
    :func:`~andrey.mmmb` and :func:`~andrey.hiton_mb` build the same blanket on another
    parents-and-children search, and :func:`~andrey.iamb` learns it by growing and shrinking it.

    **How it works.** GetPCD [1]_ grows a set of candidate parents and children of the target. A
    candidate's separating set is, among the subsets of the current set, the one with the largest
    p-value, and each pass runs three steps: every candidate its separating set separates from the
    target (a p-value above ``alpha``) leaves for good; the candidate whose largest p-value is the
    smallest joins; and every earlier member that a subset of the other members separates leaves
    for good, all at once. The search stops when no candidate is left. The set it returns can hold
    a descendant of the target other than a child, which no subset of the set separates from it,
    so a member stays only when the target is in the member's own set (the symmetry check), which
    gives the parents and children [1]_. A variable outside the target's parents and children that
    lies in those of one of its members ``y`` is a candidate spouse, with the set that separated it
    from the target: the target's own search found one, or, for a variable the symmetry check
    removed, that variable's search. The candidate joins the blanket when the test finds it
    dependent on the target (a p-value at or below ``alpha``) given that set together with ``y``,
    which with exact tests happens exactly when ``y`` is a common child of the two [1]_. A
    candidate is tested only with a ``y`` whose parents and children hold it [1]_, the ``y`` in the
    order they joined, and it joins at the first one that admits it; a ``y`` already in the
    separating set is not tested, as that test repeats the one that separated the two. Candidates
    whose p-values agree to nine significant digits, as when they underflow to zero, are ordered by
    the larger test statistic, then by the column ranks, where the paper breaks ties at random
    [1]_. With exact tests, the result is the true Markov blanket [1]_.

    On categorical data, a test runs only when ``data`` holds at least five rows per degree of
    freedom of its contingency table, ``(r_x - 1) * (r_y - 1)`` times the product of the
    conditioning variables' categories, the reliability rule of [1]_. In GetPCD, a conditioning set
    that fails the rule is not tested and takes no part, and a candidate whose unconditional test
    fails it never joins; a spouse test that fails it, or that would condition on more than
    ``max_depth`` variables, is not run and admits nothing, so the candidate joins only through
    another common child. The rule counts every row of ``data``, also where a test deletes rows
    with missing values.

    **From sets to a graph.** Without a target, a pair is kept in the skeleton
    (``structure="skeleton"``) when each variable is in the other's parents and children, and in
    the moral graph (``structure="moral"``) when each is in the other's Markov blanket. The CPDAG
    decides each triple ``x - z - y`` of the skeleton with ``x`` and ``y`` not adjacent: ``z`` is a
    collider, ``x -> z <- y``, when ``x`` and ``y`` test dependent given ``z`` together with every
    subset of the smaller of their blankets without ``x``, ``y``, and ``z`` [2]_ [3]_, and a
    non-collider when one such set separates them. The colliders are oriented as :func:`~andrey.pc`
    orients them, the most clearly dependent first: by the smallest absolute partial correlation
    over the collider's tests for ``"fisherz"`` and ``"student_t"``, and by the largest p-value,
    smallest first, for the other tests. When two colliders that rank equal to nine decimals
    dispute an edge, it stays undirected. Meek's rules [4]_ then orient the edges that follow, in
    rounds, as for :func:`~andrey.pc`, so the result does not depend on the column order; under
    ``andrey.config(tie_break="position")`` the colliders and the rules follow the column order
    instead. With exact tests, the result is the true CPDAG [2]_.

    **Missing values.** A ``NaN`` in ``data`` marks a missing value. Each test reads only the rows
    where all of its variables are observed (test-wise deletion) [5]_ [6]_, so it keeps rows that
    deleting every incomplete row first would drop; ``metadata["rows_tested"]`` gives the fewest and
    the median rows a test read. When values are missing completely at random, the rows a test keeps
    are a random sample, so each test keeps its level [6]_. When whether a value is missing depends
    on the variables, the rows a test keeps are a selected sample, and the tests can find dependence
    where the population has none and keep extra edges [6]_. A test raises ``ValueError`` when it
    has fewer rows than it needs, ``|S| + 3`` for ``"fisherz"`` and ``"student_t"`` and 2 for the
    other tests, or when a continuous column is constant on its rows.

    **Limitations.** Not benchmarked yet. Tested with exact d-separation on 15 variables, on
    simulated linear-Gaussian data with 20 variables and 2,000 rows, and on categorical data with
    10 variables and 3,000 rows; missing values on up to 20 variables with up to a fifth of the
    values missing completely at random. A spouse is found only through a child that the search
    found, so an edge its tests miss also hides that child's other parents, and each spouse's
    verdict rests on the one separating set GetPCD recorded for it. GetPCD tests every candidate
    against every subset of the current set, so a set of ``m`` members takes up to ``2**m`` tests
    per candidate; the search runs it for the target, for each of its parents and children, and,
    for the symmetry check, for each of their members, so its cost grows with the sets two steps
    from the target. The CPDAG tests every subset of the smaller blanket for each unshielded
    triple, so a triple whose smaller blanket holds ``m`` variables takes up to ``2**m`` tests. The
    numbers are a reference, not a hard limit: they change with the data generator, the density,
    the method's settings, and the machine.

    References
    ----------
    .. [1] Pena, J. M., Nilsson, R., Bjorkegren, J., and Tegner, J. (2007). Towards scalable and
       data efficient learning of Markov boundaries. International Journal of Approximate
       Reasoning 45(2), 211-232.
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
    .. [7] Spirtes, P., Glymour, C., and Scheines, R. (1993). Causation, Prediction, and Search.
       Springer.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import pcmb
    >>> rng = np.random.default_rng(0)
    >>> a, b = rng.standard_normal((2, 1000))
    >>> c = a + b + 0.5 * rng.standard_normal(1000)
    >>> out = pcmb(np.column_stack([a, b, c]), target=0)
    >>> out.metadata["markov_blanket"], out.metadata["parents_and_children"]
    ([2, 1], [2])
    >>> pcmb(np.column_stack([a, b, c])).structure.oriented_edges()
    [(0, 2, 'directed'), (1, 2, 'directed')]
    """
    return fit_blanket(
        "pcmb",
        data,
        target=target,
        structure=structure,
        alpha=alpha,
        indep_test=indep_test,
        degrees_of_freedom=degrees_of_freedom,
        categorical=categorical,
        seed=seed,
        max_depth=max_depth,
        knowledge=knowledge,
    )
