"""Target-centered facade adapters: the edges at one variable and their directions.

``mb_by_mb`` runs :func:`andrey.constraint.sequential.sequential` with the choices of its paper and
returns the target's edges as a CPDAG holds them.
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
from andrey.api._local import target_column, target_metadata
from andrey.api._method import method_spec
from andrey.core import ARROW, TAIL
from andrey.core.ci import counting_tests
from andrey.core.ci_missing import rows_metadata, rows_tested
from andrey.core.orient import to_structure

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from andrey.core import StructureOutput
    from andrey.core.knowledge import Knowledge


def _edge_sets(star: np.ndarray, t: int) -> dict[str, list[int]]:
    """Return the target's parents, children, and undirected neighbors in ``star``, by position."""
    sets: dict[str, list[int]] = {"parents": [], "children": [], "undirected": []}
    for j in np.flatnonzero(star[t]).tolist():
        near, far = star[t, j], star[j, t]
        if near == ARROW and far == TAIL:
            sets["parents"].append(j)
        elif near == TAIL and far == ARROW:
            sets["children"].append(j)
        else:
            sets["undirected"].append(j)
    return sets


@method_spec("mb_by_mb")
def mb_by_mb(
    data: npt.ArrayLike,
    *,
    target: str | int | None = None,
    blanket: str = "iamb_forward",
    alpha: float = 0.05,
    indep_test: str | None = None,
    categorical: Sequence[str | int] | None = None,
    missing: str | None = None,
    seed: int | None = None,
    max_depth: int | None = None,
    local_max_depth: int | None = 5,
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn the edges at one variable and their directions, its CPDAG's, with MB-by-MB.

    MB-by-MB [1]_ finds the direct causes and the direct effects of a target without learning the
    whole graph. It learns the target's Markov blanket and the local structure over it, then the
    blanket and local structure of a neighbor, and so on outward, one blanket at a time, until no
    further blanket can direct an edge at the target. The answer holds the edges the target has in
    the CPDAG, the causal graphs the tests cannot tell apart, with their directions.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``.
    target : str, int, or None, default=None
        Required: the variable to answer for, a column label (a string) or a 0-based position (an
        integer). ``None`` raises ``ValueError``.
    blanket : {"iamb_forward", "iamb", "inter_iamb", "gs", "fast_iamb"}, default="iamb_forward"
        The Markov blanket learner run for each variable the search visits. ``"iamb_forward"`` is
        IAMB's grow phase without its shrink, the learner of the paper's runs [1]_, which keeps a
        variable added in error rather than lose a member; the others are :func:`~andrey.iamb`,
        :func:`~andrey.inter_iamb`, :func:`~andrey.gs`, and :func:`~andrey.fast_iamb`.
    alpha : float, default=0.05
        Significance level of the conditional-independence test, in ``(0, 1)``. The paper runs
        0.01 and 0.05 [1]_.
    indep_test : str or None, default=None
        Conditional-independence test, any of :func:`~andrey.pc`'s: ``"fisherz"``,
        ``"student_t"``, or ``"normal_scores"`` for continuous data, ``"chisq"`` or ``"gsq"`` for
        categorical data, ``"degen_gauss"`` or ``"cond_gauss"`` for mixed data, or a nonlinear
        test. ``None`` picks from the declared column types, as for :func:`~andrey.pc`. The
        paper's runs use the likelihood-ratio test on categorical data, ``"gsq"`` [1]_.
        ``metadata["indep_test"]`` names the test the fit used: the one picked from the
        declared column types when this is ``None``.
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
        The maximum conditioning-set size of every conditional-independence test, the blanket
        learner's and the local structures', at least ``0``; ``None``, the default, sets no limit.
        Under a cap, the blanket learner stops growing at ``max_depth + 1`` members.
    local_max_depth : int or None, default=5
        The maximum size of the subsets the local structures test, at least ``0`` and never above
        ``max_depth``; ``None`` sets no limit. The paper leaves the cap open and its runs use 5
        [1]_, the default, on the local structures alone, as its complexity bounds only them by the
        cap: the same cap on the blanket learner stops its grow phase before the blanket is whole,
        so the errors rise with the sample.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the pairs
        allowed to be adjacent, and temporal tiers, all honored. Each end of a required edge is in
        the other's blanket from the start, and no test removes it. A pair the knowledge makes
        non-adjacent, forbidden both ways or outside ``allowed``, never shares an edge. Every edge
        a tier or an edge forbidden one way directs is directed so as each local structure joins
        the growing graph, over any v-structure's arrowhead, and Meek's rules, R4 included, direct
        the edges that follow. ``required_paths`` and ``forbidden_paths`` raise
        ``NotImplementedError``.

    Returns
    -------
    StructureOutput
        Wraps a ``cpdag`` :class:`~andrey.core.GraphStructure` over every column holding only the
        edges at ``target``: ``x -> target`` for a direct cause, ``target -> x`` for a direct
        effect, and ``x - target`` where the CPDAG leaves the direction undecided. An absent edge
        away from the target means nothing. ``metadata["target"]`` names the target, and
        ``metadata["parents"]``, ``metadata["children"]``, and ``metadata["undirected"]`` list
        those columns, by column label or 0-based position. ``metadata["blankets_learned"]`` lists
        the variables whose Markov blanket the search learned, in order, the target first, so its
        length is the number of blankets the answer took. ``metadata["algorithm"]`` is
        ``"mb_by_mb"``. Under a mixed-data setting, ``metadata["categorical"]`` lists the
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
        method, the argument, what it accepts, and the value. Also if ``target`` is ``None`` or
        names a column ``data`` lacks, or if ``data`` or ``categorical`` does not suit the test, as
        for :func:`~andrey.pc`. With ``"fisherz"`` or ``"student_t"``, also if a column repeats
        another up to scale and shift. Also if ``knowledge`` names a column ``data`` lacks or
        contradicts itself, as :class:`~andrey.Knowledge` lists.
    NotImplementedError
        If ``knowledge`` holds ``required_paths`` or ``forbidden_paths``; the message names the
        methods that honor them.

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
    **When to use.** MB-by-MB suits i.i.d. observational data with no unmeasured common causes, in
    which every independence follows from the graph (faithfulness), when you want the direct causes
    and effects of one variable rather than the whole graph [1]_. Its paper runs it on categorical
    data with the likelihood-ratio test; the default Fisher-Z test assumes continuous data with
    linear effects and Gaussian noise, and the other tests read categorical or mixed data or detect
    nonlinear effects, as :func:`~andrey.pc` describes. It is designed for large graphs in which
    each variable has a small Markov blanket: it learns only the blankets the answer needs, often
    two or three [1]_. It returns the edges at the target with the directions the CPDAG gives them;
    for the whole CPDAG, use :func:`~andrey.pc`, and for the target's neighbors without directions,
    :func:`~andrey.mmpc`.

    **How it works.** A waitlist starts with the target [1]_. Each step takes the waitlist's head
    ``x``, learns its Markov blanket ``MB(x)`` with the ``blanket`` learner, and appends the members
    neither waiting nor done to the waitlist. It then learns the local structure over ``x`` and its
    blanket: ``x`` and a member are adjacent unless some subset of the rest of the blanket
    separates them, every other pair is tested against every subset of the other variables there,
    and a triple ``a - b - c`` with ``a`` and ``c`` separated by a set without ``b`` is a
    v-structure ``a -> b <- c`` [1]_. When the blanket and ``x`` lie inside those of a variable
    done earlier, or every member of the blanket is done, the local structure is read from what is
    already learned instead [1]_. Only the edges at ``x`` and the v-structures that contain ``x``
    join the growing graph: the rest of the local structure may come from variables outside the
    blanket [1]_.
    Meek's rules [2]_ then direct the edges that follow, reading a missing edge as an absence only
    where a test separated the pair, and a triple as a non-collider only where the separating set
    holds its middle [1]_. Last, every waitlist variable that each path to the target in the
    growing graph reaches only through a directed edge is dropped, since its structure cannot
    direct an edge at the target; the search stops when the waitlist is empty [1]_. With exact
    tests, the edges at the target and their directions are those of the true CPDAG [1]_. An edge
    that two v-structures direct both ways is left undirected.

    Every choice between candidates of equal strength goes to the column whose rank, computed from
    the data, comes first, as for :func:`~andrey.iamb`, and a pair separated at some size keeps the
    set with the largest p-value, so the answer does not depend on the column order unless two
    columns are identical.

    **Missing values.** A ``NaN`` in ``data`` marks a missing value. Each test reads only the rows
    where all of its variables are observed (test-wise deletion) [3]_ [4]_, so it keeps rows that
    deleting every incomplete row first would drop; ``metadata["rows_tested"]`` gives the fewest and
    the median rows a test read. When values are missing completely at random, the rows a test keeps
    are a random sample, so each test keeps its level [4]_. When whether a value is missing depends
    on the variables, the rows a test keeps are a selected sample, and the tests can find dependence
    where the population has none and keep extra edges [4]_. A test raises ``ValueError`` when it
    has fewer rows than it needs, ``|S| + 3`` for ``"fisherz"`` and ``"student_t"`` and 2 for the
    other tests, or when a continuous column is constant on its rows.

    **Limitations.** Not benchmarked yet, and no reference implementation exists to compare with.
    Tested with exact d-separation on every target of 11 random graphs of up to 30 variables and of
    a 5 x 5 chessboard, and on simulated linear-Gaussian data with 20 variables and categorical data
    with 10. Each local structure tests every pair of a blanket against every subset of the rest, up
    to ``local_max_depth`` members, so a blanket of ``m`` variables takes up to about
    ``m**2 * 2**m`` tests without the cap. Under the default ``local_max_depth=5``, a pair that
    only a larger set separates keeps its edge. The default blanket learner never drops a variable
    it added, so a large sample, in which more variables test dependent, gives larger blankets, and
    each test of the blanket learner conditions on the whole set so far. The numbers are a
    reference, not a hard limit: they change with the data generator, the density, the method's
    settings, and the machine.

    References
    ----------
    .. [1] Wang, C., Zhou, Y., Zhao, Q., and Geng, Z. (2014). Discovering and orienting the edges
       connected to a target variable in a DAG via a sequential local learning approach.
       Computational Statistics and Data Analysis 77, 252-266.
    .. [2] Meek, C. (1995). Causal inference and causal explanation with background knowledge.
       UAI, 403-410.
    .. [3] Strobl, E. V., Visweswaran, S., and Spirtes, P. L. (2018). Fast causal inference with
       non-random missingness by test-wise deletion. International Journal of Data Science and
       Analytics 6(1), 47-62.
    .. [4] Tu, R., Zhang, C., Ackermann, P., Mohan, K., Kjellstrom, H., and Zhang, K. (2019). Causal
       discovery in the presence of missing data. Proceedings of Machine Learning Research 89,
       1762-1770.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import mb_by_mb
    >>> rng = np.random.default_rng(0)
    >>> a, b = rng.standard_normal((2, 1000))
    >>> c = a + b + 0.5 * rng.standard_normal(1000)
    >>> out = mb_by_mb(np.column_stack([a, b, c]), target=2)
    >>> out.metadata["parents"], out.metadata["children"]
    ([0, 1], [])
    """
    from andrey.constraint.blanket import local_search
    from andrey.constraint.sequential import MB_BY_MB, sequential

    assert indep_test is not None  # method_spec picks it from the column types
    X, mask, labels, constraints = fit_inputs(
        data, "indep_test", indep_test, categorical, knowledge, method="mb_by_mb", missing=True
    )
    d = X.shape[1]
    t = target_column(target, labels, d, "mb_by_mb")
    name = label_of(labels)
    with copies_named(labels), rows_tested() as rows, counting_tests() as tests:
        search = local_search(
            X,
            alpha=alpha,
            indep_test=indep_test,
            categorical=mask,
            seed=seed,
            max_depth=max_depth,
            constraints=constraints,
        )
        result = sequential(search, t, blanket, MB_BY_MB, local_depth=local_max_depth)
    star = result.network.star(t, d)
    metadata = target_metadata(t, _edge_sets(star, t), name)
    metadata["blankets_learned"] = [name(x) for x in result.learned]
    metadata |= categorical_metadata(mask, data) | rows_metadata(rows) | {"n_tests": tests.n}
    return structure_output(to_structure(star, kind="cpdag"), labels=labels, metadata=metadata)
