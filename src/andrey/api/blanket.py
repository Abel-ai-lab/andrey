"""Local-structure facade adapters: one variable's Markov blanket or parents and children.

``iamb``, ``inter_iamb``, and ``mmpc`` share :mod:`andrey.constraint.blanket`. With ``target`` they
return that variable's set; without, every variable's, joined by the AND rule.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from andrey.api._adapt import (
    MAX_CATEGORIES,
    categorical_metadata,
    checked_data,
    column_labels,
    copies_named,
    structure_output,
)
from andrey.api._method import method_spec
from andrey.core import NULL, TAIL
from andrey.core.ci_missing import rows_metadata, rows_tested
from andrey.core.knowledge import _node, column_position
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
    alpha: float,
    indep_test: str,
    categorical: Sequence[str | int] | None,
    max_categories: int,
) -> StructureOutput:
    """Check the settings, run the search, and wrap its sets as undirected edges."""
    from andrey.search.hybrid import local_graph, local_set

    X, mask = checked_data(
        data,
        categorical,
        "indep_test",
        indep_test,
        max_categories=max_categories,
        method=method,
        missing=True,
    )
    labels = column_labels(data)
    d = X.shape[1]
    t = None if target is None else column_position(_node(target, "target"), labels, d, "target")
    settings = {"alpha": alpha, "indep_test": indep_test, "categorical": mask}
    metadata: dict[str, object] = {}
    with copies_named(labels), rows_tested() as rows:
        if t is None:
            adjacent = local_graph(X, method, **settings)
        else:
            members = local_set(X, method, t, **settings)
            adjacent = np.zeros((d, d), dtype=bool)
            adjacent[t, members] = adjacent[members, t] = True
            name = (lambda j: labels[j]) if labels is not None else int
            metadata |= {"target": name(t), _SETS[method]: [name(j) for j in members]}
    marks = np.where(adjacent, TAIL, NULL).astype(np.int8)
    metadata |= categorical_metadata(mask, data) | rows_metadata(rows)
    return structure_output(to_structure(marks, kind="cpdag"), labels=labels, metadata=metadata)


@method_spec("iamb")
def iamb(
    data: npt.ArrayLike,
    *,
    target: str | int | None = None,
    alpha: float = 0.05,
    indep_test: str = "fisherz",
    categorical: Sequence[str | int] | None = None,
    max_categories: int = MAX_CATEGORIES,
) -> StructureOutput:
    """Learn a variable's Markov blanket with incremental association (IAMB).

    A variable's Markov blanket is the smallest set of other variables given which it is
    independent of all the rest: in a causal graph, its parents, its children, and its children's
    other parents. IAMB [1]_ grows the blanket by conditional-independence tests, then shrinks it.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``.
    target : str, int, or None, default=None
        The variable whose set to learn: a column label (a string) or a 0-based position (an
        integer). ``None`` learns every variable's set and keeps a pair when each variable is in
        the other's set.
    alpha : float, default=0.05
        Significance level of the conditional-independence test. Smaller values give smaller sets.
    indep_test : str, default="fisherz"
        Conditional-independence test, any of :func:`~andrey.pc`'s: ``"fisherz"`` or
        ``"student_t"`` for continuous data, ``"chisq"`` or ``"gsq"`` for categorical data,
        ``"degen_gauss"`` or ``"cond_gauss"`` for mixed data, or a nonlinear test.
    categorical : sequence of str or int, optional
        The categorical columns of ``data``, for a mixed-data test, as for :func:`~andrey.pc`. The
        other tests take no column types and reject it.
    max_categories : int, default=10
        The most distinct whole numbers a column may hold to be inferred as categorical, at least
        ``2``, as for :func:`~andrey.pc`.

    Returns
    -------
    StructureOutput
        Wraps a ``cpdag`` :class:`~andrey.core.GraphStructure` whose edges are all undirected.
        With ``target``, they join ``target`` to each member of its Markov blanket, which holds
        its parents, its children, and its children's other parents; without, they join each pair
        in each other's blanket, the moral graph. An edge here does not say that the two variables
        are adjacent in the causal graph. ``metadata["algorithm"]`` is ``"iamb"``; with
        ``target``, ``metadata["target"]`` names it and ``metadata["markov_blanket"]`` lists the
        members in the order they joined, by column label or 0-based position. Under a mixed-data
        setting, ``metadata["categorical"]`` lists the categorical columns it read.

    Raises
    ------
    TypeError
        If ``target`` is neither a column label nor a position, or ``categorical`` is not a list of
        column names and positions.
    ValueError
        If an argument has the wrong type, lies outside its range, or is not one of its choices; the
        message names the method, the argument, what it accepts, and the value. Also if ``target``
        names a column ``data`` lacks, or if ``data`` or ``categorical`` does not suit the test, as
        for :func:`~andrey.pc`. With ``"fisherz"`` or ``"student_t"``, also if a column repeats
        another up to scale and shift.

    Warns
    -----
    ExperimentalWarning
        On the first call in a process: the method has no published benchmark, and its API may
        change without deprecation.
    DataTypeWarning
        As for :func:`~andrey.pc`, when a mixed-data test infers the categorical columns or a
        continuous setting meets columns that look like category codes; once per process.

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

    Candidates whose strengths agree to nine decimals go to the column whose rank, computed from
    its own data, comes first, as for :func:`~andrey.hc`, so the sets do not depend on the column
    order unless two columns are identical. The strength is the absolute partial correlation for
    ``"fisherz"`` and ``"student_t"``, which keeps its order where p-values underflow to zero, and
    the p-value for the other tests.

    **Missing values.** A ``NaN`` in ``data`` marks a missing value. Each test reads only the rows
    where all of its variables are observed (test-wise deletion) [2]_ [3]_, so it keeps rows that
    deleting every incomplete row first would drop; ``metadata["rows_tested"]`` gives the fewest and
    the median rows a test read. When values are missing completely at random, the rows a test keeps
    are a random sample, so each test keeps its level [3]_. When whether a value is missing depends
    on the variables, the rows a test keeps are a selected sample, and the tests can find dependence
    where the population has none and keep extra edges [3]_. A test raises ``ValueError`` when it
    has fewer rows than it needs, ``|S| + 3`` for ``"fisherz"`` and ``"student_t"`` and 2 for the
    other tests, or when a continuous column is constant on its rows.

    **Limitations.** Not benchmarked yet. Its tests run it on 20 variables with 2,000 rows of
    simulated linear-Gaussian data, and on 15 variables with exact d-separation tests. It
    conditions on the whole current blanket, so each test needs more rows as the blanket grows,
    and a candidate the grow step adds in error enlarges every later conditioning set. Missing
    values are not benchmarked yet; the tests run them on up to 20 variables with up to a fifth of
    the values missing completely at random.

    References
    ----------
    .. [1] Tsamardinos, I., Aliferis, C. F., and Statnikov, A. (2003). Algorithms for large scale
       Markov blanket discovery. FLAIRS, 376-380.
    .. [2] Strobl, E. V., Visweswaran, S., and Spirtes, P. L. (2018). Fast causal inference with
       non-random missingness by test-wise deletion. International Journal of Data Science and
       Analytics 6(1), 47-62.
    .. [3] Tu, R., Zhang, C., Ackermann, P., Mohan, K., Kjellstrom, H., and Zhang, K. (2019). Causal
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
    >>> sorted(out.metadata["markov_blanket"])
    [1, 2]
    """
    return _fit(
        "iamb",
        data,
        target=target,
        alpha=alpha,
        indep_test=indep_test,
        categorical=categorical,
        max_categories=max_categories,
    )


@method_spec("inter_iamb")
def inter_iamb(
    data: npt.ArrayLike,
    *,
    target: str | int | None = None,
    alpha: float = 0.05,
    indep_test: str = "fisherz",
    categorical: Sequence[str | int] | None = None,
    max_categories: int = MAX_CATEGORIES,
) -> StructureOutput:
    """Learn a variable's Markov blanket with interleaved IAMB (Inter-IAMB).

    Inter-IAMB [1]_ is IAMB with its shrink step after every addition, so a member that becomes
    independent leaves before it enlarges the later conditioning sets.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``.
    target : str, int, or None, default=None
        The variable whose set to learn: a column label (a string) or a 0-based position (an
        integer). ``None`` learns every variable's set and keeps a pair when each variable is in
        the other's set.
    alpha : float, default=0.05
        Significance level of the conditional-independence test. Smaller values give smaller sets.
    indep_test : str, default="fisherz"
        Conditional-independence test, any of :func:`~andrey.pc`'s: ``"fisherz"`` or
        ``"student_t"`` for continuous data, ``"chisq"`` or ``"gsq"`` for categorical data,
        ``"degen_gauss"`` or ``"cond_gauss"`` for mixed data, or a nonlinear test.
    categorical : sequence of str or int, optional
        The categorical columns of ``data``, for a mixed-data test, as for :func:`~andrey.pc`. The
        other tests take no column types and reject it.
    max_categories : int, default=10
        The most distinct whole numbers a column may hold to be inferred as categorical, at least
        ``2``, as for :func:`~andrey.pc`.

    Returns
    -------
    StructureOutput
        Wraps a ``cpdag`` :class:`~andrey.core.GraphStructure` whose edges are all undirected.
        With ``target``, they join ``target`` to each member of its Markov blanket, which holds
        its parents, its children, and its children's other parents; without, they join each pair
        in each other's blanket, the moral graph. An edge here does not say that the two variables
        are adjacent in the causal graph. ``metadata["algorithm"]`` is ``"inter_iamb"``; with
        ``target``, ``metadata["target"]`` names it and ``metadata["markov_blanket"]`` lists the
        members in the order they joined, by column label or 0-based position. Under a mixed-data
        setting, ``metadata["categorical"]`` lists the categorical columns it read.

    Raises
    ------
    TypeError
        If ``target`` is neither a column label nor a position, or ``categorical`` is not a list of
        column names and positions.
    ValueError
        If an argument has the wrong type, lies outside its range, or is not one of its choices; the
        message names the method, the argument, what it accepts, and the value. Also if ``target``
        names a column ``data`` lacks, or if ``data`` or ``categorical`` does not suit the test, as
        for :func:`~andrey.pc`. With ``"fisherz"`` or ``"student_t"``, also if a column repeats
        another up to scale and shift.

    Warns
    -----
    ExperimentalWarning
        On the first call in a process: the method has no published benchmark, and its API may
        change without deprecation.
    DataTypeWarning
        As for :func:`~andrey.pc`, when a mixed-data test infers the categorical columns or a
        continuous setting meets columns that look like category codes; once per process.

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

    Candidates whose strengths agree to nine decimals go to the column whose rank, computed from
    its own data, comes first, as for :func:`~andrey.hc`, so the sets do not depend on the column
    order unless two columns are identical. The strength is the absolute partial correlation for
    ``"fisherz"`` and ``"student_t"``, which keeps its order where p-values underflow to zero, and
    the p-value for the other tests.

    **Missing values.** A ``NaN`` in ``data`` marks a missing value. Each test reads only the rows
    where all of its variables are observed (test-wise deletion) [2]_ [3]_, so it keeps rows that
    deleting every incomplete row first would drop; ``metadata["rows_tested"]`` gives the fewest and
    the median rows a test read. When values are missing completely at random, the rows a test keeps
    are a random sample, so each test keeps its level [3]_. When whether a value is missing depends
    on the variables, the rows a test keeps are a selected sample, and the tests can find dependence
    where the population has none and keep extra edges [3]_. A test raises ``ValueError`` when it
    has fewer rows than it needs, ``|S| + 3`` for ``"fisherz"`` and ``"student_t"`` and 2 for the
    other tests, or when a continuous column is constant on its rows.

    **Limitations.** Not benchmarked yet. Its tests run it on 20 variables with 2,000 rows of
    simulated linear-Gaussian data, and on 15 variables with exact d-separation tests. It
    conditions on the whole current blanket, so each test needs more rows as the blanket grows.
    Missing values are not benchmarked yet; the tests run them on up to 20 variables with up to a
    fifth of the values missing completely at random.

    References
    ----------
    .. [1] Tsamardinos, I., Aliferis, C. F., and Statnikov, A. (2003). Algorithms for large scale
       Markov blanket discovery. FLAIRS, 376-380.
    .. [2] Strobl, E. V., Visweswaran, S., and Spirtes, P. L. (2018). Fast causal inference with
       non-random missingness by test-wise deletion. International Journal of Data Science and
       Analytics 6(1), 47-62.
    .. [3] Tu, R., Zhang, C., Ackermann, P., Mohan, K., Kjellstrom, H., and Zhang, K. (2019). Causal
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
        alpha=alpha,
        indep_test=indep_test,
        categorical=categorical,
        max_categories=max_categories,
    )


@method_spec("mmpc")
def mmpc(
    data: npt.ArrayLike,
    *,
    target: str | int | None = None,
    alpha: float = 0.05,
    indep_test: str = "fisherz",
    categorical: Sequence[str | int] | None = None,
    max_categories: int = MAX_CATEGORIES,
) -> StructureOutput:
    """Learn a variable's parents and children with max-min parents and children (MMPC).

    MMPC [1]_ finds the variables adjacent to a target in the causal graph, its parents and
    children, without telling them apart, by conditional-independence tests. Every variable's
    set together gives the skeleton :func:`~andrey.mmhc` searches.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``.
    target : str, int, or None, default=None
        The variable whose set to learn: a column label (a string) or a 0-based position (an
        integer). ``None`` learns every variable's set and keeps a pair when each variable is in
        the other's set.
    alpha : float, default=0.05
        Significance level of the conditional-independence test. Smaller values give smaller sets.
    indep_test : str, default="fisherz"
        Conditional-independence test, any of :func:`~andrey.pc`'s: ``"fisherz"`` or
        ``"student_t"`` for continuous data, ``"chisq"`` or ``"gsq"`` for categorical data,
        ``"degen_gauss"`` or ``"cond_gauss"`` for mixed data, or a nonlinear test.
    categorical : sequence of str or int, optional
        The categorical columns of ``data``, for a mixed-data test, as for :func:`~andrey.pc`. The
        other tests take no column types and reject it.
    max_categories : int, default=10
        The most distinct whole numbers a column may hold to be inferred as categorical, at least
        ``2``, as for :func:`~andrey.pc`.

    Returns
    -------
    StructureOutput
        Wraps a ``cpdag`` :class:`~andrey.core.GraphStructure` whose edges are all undirected.
        With ``target``, they join ``target`` to each of its parents and children; without, they
        form the skeleton: each pair in each other's set. ``metadata["algorithm"]`` is ``"mmpc"``;
        with ``target``, ``metadata["target"]`` names it and ``metadata["parents_and_children"]``
        lists the members in the order they joined, by column label or 0-based position. Under a
        mixed-data setting, ``metadata["categorical"]`` lists the categorical columns it read.

    Raises
    ------
    TypeError
        If ``target`` is neither a column label nor a position, or ``categorical`` is not a list of
        column names and positions.
    ValueError
        If an argument has the wrong type, lies outside its range, or is not one of its choices; the
        message names the method, the argument, what it accepts, and the value. Also if ``target``
        names a column ``data`` lacks, or if ``data`` or ``categorical`` does not suit the test, as
        for :func:`~andrey.pc`. With ``"fisherz"`` or ``"student_t"``, also if a column repeats
        another up to scale and shift.

    Warns
    -----
    ExperimentalWarning
        On the first call in a process: the method has no published benchmark, and its API may
        change without deprecation.
    DataTypeWarning
        As for :func:`~andrey.pc`, when a mixed-data test infers the categorical columns or a
        continuous setting meets columns that look like category codes; once per process.

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
    its own data, comes first, as for :func:`~andrey.hc`, so the sets do not depend on the column
    order unless two columns are identical. The strength is the absolute partial correlation for
    ``"fisherz"`` and ``"student_t"``, which keeps its order where p-values underflow to zero, and
    the p-value for the other tests.

    **Missing values.** A ``NaN`` in ``data`` marks a missing value. Each test reads only the rows
    where all of its variables are observed (test-wise deletion) [2]_ [3]_, so it keeps rows that
    deleting every incomplete row first would drop; ``metadata["rows_tested"]`` gives the fewest and
    the median rows a test read. When values are missing completely at random, the rows a test keeps
    are a random sample, so each test keeps its level [3]_. When whether a value is missing depends
    on the variables, the rows a test keeps are a selected sample, and the tests can find dependence
    where the population has none and keep extra edges [3]_. A test raises ``ValueError`` when it
    has fewer rows than it needs, ``|S| + 3`` for ``"fisherz"`` and ``"student_t"`` and 2 for the
    other tests, or when a continuous column is constant on its rows.

    **Limitations.** Not benchmarked yet. Its tests run it on 20 variables with 2,000 rows of
    simulated linear-Gaussian data, and on 15 variables with exact d-separation tests. It tests
    each candidate against every subset of the current members, so its cost grows exponentially
    with the size of the set; the symmetry check runs the search once more for each member. Missing
    values are not benchmarked yet; the tests run them on up to 20 variables with up to a fifth of
    the values missing completely at random.

    References
    ----------
    .. [1] Tsamardinos, I., Brown, L. E., and Aliferis, C. F. (2006). The max-min hill-climbing
       Bayesian network structure learning algorithm. Machine Learning 65(1), 31-78.
    .. [2] Strobl, E. V., Visweswaran, S., and Spirtes, P. L. (2018). Fast causal inference with
       non-random missingness by test-wise deletion. International Journal of Data Science and
       Analytics 6(1), 47-62.
    .. [3] Tu, R., Zhang, C., Ackermann, P., Mohan, K., Kjellstrom, H., and Zhang, K. (2019). Causal
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
        alpha=alpha,
        indep_test=indep_test,
        categorical=categorical,
        max_categories=max_categories,
    )
