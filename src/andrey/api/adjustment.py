"""Treatment-outcome facade adapters.

Each method answers for one ordered pair of columns, a treatment and an outcome: the answer is a
record in ``metadata["effects"]`` (:mod:`andrey.core.effects`) with the adjustment sets it found,
next to the graph its paper licenses. ``ldp`` finds a valid adjustment set for the total effect;
``ld3`` finds the parents of the outcome and whether the treatment is one of them; ``ldecc`` finds
the edges at the treatment and its locally valid parent sets; ``local_optimal_adjustment`` finds,
for two variables, which one causes the other and the optimal adjustment set; ``locpc_cde`` decides
whether the controlled direct effect is identifiable and finds the outcome's parents from a graph it
learns around the outcome.
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
from andrey.api._method import method_spec
from andrey.core.ci_missing import rows_metadata, rows_tested

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from andrey.core import StructureOutput
    from andrey.core.knowledge import Knowledge


def _fit(
    method: str, data, treatment, outcome, alpha, indep_test, degrees_of_freedom, categorical, seed
):
    """Run the engine ``method`` of :mod:`andrey.constraint.partition` on the pair.

    Returns its result, the fit's columns, labels, and namer, and the shared metadata.
    """
    from andrey.api._local import pair_columns
    from andrey.constraint.blanket import local_search
    from andrey.constraint.partition import ld3 as ld3_engine
    from andrey.constraint.partition import ldp as ldp_engine
    from andrey.core.ci import counting_tests

    X, mask, labels, _ = fit_inputs(
        data, "indep_test", indep_test, categorical, None, method=method, missing=True
    )
    d = X.shape[1]
    x, y = pair_columns(treatment, outcome, labels, d, method)
    others = [j for j in range(d) if j not in (x, y)]
    with copies_named(labels), rows_tested() as rows, counting_tests() as tests:
        search = local_search(
            X,
            alpha=alpha,
            indep_test=indep_test,
            degrees_of_freedom=degrees_of_freedom,
            categorical=mask,
            seed=seed,
            ranked=False,
        )
        with search.pinned():
            result = (ldp_engine if method == "ldp" else ld3_engine)(search, x, y, others)
    name = label_of(labels)
    metadata = {
        "partition": {k: [name(j) for j in v] for k, v in result.partition.items()},
        **categorical_metadata(mask, data),
        **rows_metadata(rows),
        "n_tests": tests.n,
    }
    return result, (x, y, d, labels, name), metadata


@method_spec("ldp")
def ldp(
    data: npt.ArrayLike,
    *,
    treatment: str | int | None = None,
    outcome: str | int | None = None,
    alpha: float = 0.005,
    indep_test: str | None = None,
    degrees_of_freedom: str = "mean",
    categorical: Sequence[str | int] | None = None,
    missing: str | None = None,
    seed: int | None = None,
) -> StructureOutput:
    """Find a valid adjustment set for a treatment and an outcome by partitioning the others (LDP).

    LDP sorts every other variable by its causal relation to the pair (confounder, instrument,
    cause of the outcome, descendant of the treatment, and so on) from tests that read the pair
    and the variables already sorted, never a graph [1]_. The confounders it finds are a valid
    adjustment set for the total effect of the treatment on the outcome when its Z5 criterion
    passes: an instrument it found is independent of the outcome given the treatment and the
    confounders. Otherwise it gives no set. It runs a number of tests quadratic in the number of
    variables, most of them conditioning on one or two.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``.
    treatment : str or int
        Required: the cause of the pair, a column label (a string) or a 0-based position (an
        integer).
    outcome : str or int
        Required: the effect of the pair, a column label (a string) or a 0-based position (an
        integer). The outcome must not cause the treatment.
    alpha : float, default=0.005
        Significance level of every test, in ``(0, 1)``: a p-value above ``alpha`` is an
        independence, any other a dependence. The default is the level of the authors' code; the
        experiments of [1]_ use 0.01, 0.005, or 0.001 by setting.
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

    Returns
    -------
    StructureOutput
        Wraps a ``dag`` :class:`~andrey.core.GraphStructure` over every column with no edges: the
        labels are classes of paths to the pair, not adjacencies. ``metadata["effects"]`` holds
        one record (:mod:`andrey.core.effects`) for the pair, by column label or 0-based position:
        ``estimand`` is ``"total"`` and ``nonzero`` is ``None``; when the Z5 criterion passes,
        ``identifiable`` is ``True`` and ``adjustment_sets`` is ``[Z1]``, the confounders found,
        with ``set_kind`` ``"valid"``; when it fails, ``identifiable`` is ``None`` and
        ``adjustment_sets`` is ``[]``: the method gives no answer, which does not mean that no
        valid set exists. ``metadata["partition"]`` maps each label to its variables: ``"Z1"``
        (confounders), ``"Z4"`` (causes of the outcome independent of the treatment), ``"Z5"``
        (instruments), ``"Z5,7"`` (instruments or descendants of the treatment, left unsplit when
        no confounder is found), ``"Z7"`` (descendants of the treatment whose paths to the
        outcome all pass through it), ``"Z8"`` (independent of both), ``"Zpost"`` (colliders,
        mediators, or descendants of the outcome), and ``"unlabeled"``; see Notes.
        ``metadata["algorithm"]`` is ``"ldp"``. Under a mixed-data setting,
        ``metadata["categorical"]`` lists the categorical columns it read. With a randomized
        ``indep_test``, ``metadata["seed"]`` is the seed used. ``metadata["n_tests"]`` is the
        number of conditional-independence tests the fit evaluated.

    Raises
    ------
    TypeError
        If an argument has the wrong type, such as a string for a number; the message names the
        method, the argument, what it accepts, and the value. Also if ``treatment`` or ``outcome``
        is neither a column label nor a position, or ``categorical`` is not a list of column
        names and positions.
    ValueError
        If an argument lies outside its range or is not one of its choices; the message names the
        method, the argument, what it accepts, and the value. Also if ``treatment`` or ``outcome``
        is missing or names a column ``data`` lacks, or both name one column; or if ``data`` or
        ``categorical`` does not suit the test, as for :func:`~andrey.pc`. With ``"fisherz"`` or
        ``"student_t"``, also if a column repeats another up to scale and shift.

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
    **When to use.** LDP suits observational data where you want to estimate the effect of one
    variable on another and need a set of covariates to adjust for, without learning a graph,
    and the covariates may include descendants of the treatment as well as confounders. Its claim
    [1]_: when every independence follows from the causal graph (faithfulness), the tests are
    exact, and the outcome does not cause the treatment, the set it returns blocks every back-door
    path and holds no descendant of the treatment, even with hidden variables (Theorem 4.5). It
    finds one when the data hold a cause of the outcome independent of the treatment and an
    instrument independent of the confounders (its conditions C1 and C2). The default Fisher-Z
    test assumes continuous data with linear effects and Gaussian noise; the other tests read
    categorical or mixed data or detect nonlinear effects, as :func:`~andrey.pc` describes. To
    test whether the treatment causes the outcome directly, use :func:`~andrey.ld3`.

    **How it works.** Algorithm 1 of [1]_ in eight steps. A variable independent of both is
    ``Z8``; one independent of the treatment and dependent on it given the outcome is ``Z4``; one
    dependent on the outcome and independent of it given the treatment is ``Z5,7``. A remaining
    variable dependent on a ``Z4`` variable, or made independent of it by the pair, is
    post-treatment. Of the rest, those the treatment and the others make independent of the
    outcome form ``Zmix``. A remaining variable that is independent of a ``Zmix`` or ``Z5,7``
    candidate and dependent on it given the treatment is a confounder: the two meet in a collider
    at the treatment, and the candidate is a confounder or an instrument. The other remaining
    variables are post-treatment. A ``Zmix`` variable independent of such a candidate is a
    confounder, and any other is post-treatment. A candidate dependent on a confounder found in
    that collider test is a confounder, and any other is an instrument; this step reuses those
    tests and runs none. Each query is tested once and its p-value kept. Where the algorithm is
    open, the authors' code decides: every candidate in such a collider is kept, Step 6's second
    pass covers ``Zmix`` alone, and ``Z7`` is ``Z5,7`` less the candidates. Only ``Z4``, ``Z7``,
    and ``Z8`` are labeled right in every graph (Theorem 4.6); the other labels need stronger
    conditions than the adjustment set does (Remark 4.1). The set and the labels do not depend on
    the column order; the number of tests can, since a search stops at the first variable that
    decides it.

    **Missing values.** A ``NaN`` in ``data`` marks a missing value. Each test reads only the rows
    where all of its variables are observed (test-wise deletion) [2]_ [3]_, so it keeps rows that
    deleting every incomplete row first would drop; ``metadata["rows_tested"]`` gives the fewest and
    the median rows a test read. When values are missing completely at random, the rows a test keeps
    are a random sample, so each test keeps its level [3]_. When whether a value is missing depends
    on the variables, the rows a test keeps are a selected sample of their own. A test raises
    ``ValueError`` when it has fewer rows than it needs, ``|S| + 3`` for ``"fisherz"`` and
    ``"student_t"`` and 2 for the other tests, or when a continuous column is constant on its rows.

    **Limitations.** Not benchmarked yet. Tested with exact d-separation on the partition graphs
    of [1]_ and on random graphs, and on simulated linear-Gaussian data. Without a confounder in
    the data there is no instrument to check, so the method gives no set even when the empty set
    is valid; it also gives none for some effects that a set identifies, such as a zero effect
    [4]_. The test of Step 5 conditions on all the variables Step 4 kept, so it needs more rows as
    they grow. With 5,000 rows a fit ran about 40 tests on the 10-node graph of [1]_ and about 120
    on its 18-node graph, each in under a second. The numbers are a reference, not a hard limit:
    they change with the data generator, the density, the method's settings, and the machine.

    References
    ----------
    .. [1] Maasch, J., Pan, W., Gupta, S., Kuleshov, V., Gan, K., and Wang, F. (2024). Local
       discovery by partitioning: polynomial-time causal discovery around exposure-outcome pairs.
       Proceedings of the Fortieth Conference on Uncertainty in Artificial Intelligence,
       Proceedings of Machine Learning Research 244, 2350-2382.
    .. [2] Strobl, E. V., Visweswaran, S., and Spirtes, P. L. (2018). Fast causal inference with
       non-random missingness by test-wise deletion. International Journal of Data Science and
       Analytics 6(1), 47-62.
    .. [3] Tu, R., Zhang, C., Ackermann, P., Mohan, K., Kjellstrom, H., and Zhang, K. (2019). Causal
       discovery in the presence of missing data. Proceedings of Machine Learning Research 89,
       1762-1770.
    .. [4] Schubert, M., Claassen, T., and Magliacane, S. (2026). Local causal discovery for
       statistically efficient causal inference. Proceedings of the 29th International Conference
       on Artificial Intelligence and Statistics, Proceedings of Machine Learning Research 300.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from andrey import ldp
    >>> rng = np.random.default_rng(0)
    >>> c, i, p = rng.standard_normal((3, 5000))
    >>> x = c + i + rng.standard_normal(5000)
    >>> y = x + c + p + rng.standard_normal(5000)
    >>> df = pd.DataFrame({"c": c, "i": i, "p": p, "x": x, "y": y})
    >>> out = ldp(df, treatment="x", outcome="y")
    >>> out.metadata["effects"][0]["adjustment_sets"]
    [['c']]
    """
    from andrey.api._adapt import dag_from_adjacency
    from andrey.core.effects import effect_record

    assert indep_test is not None  # method_spec picks it from the column types
    result, (x, y, d, labels, name), metadata = _fit(
        "ldp", data, treatment, outcome, alpha, indep_test, degrees_of_freedom, categorical, seed
    )
    record = effect_record(
        x,
        y,
        estimand="total",
        identifiable=True if result.valid else None,
        adjustment_sets=[result.partition["Z1"]] if result.valid else [],
        set_kind="valid" if result.valid else None,
        name=name,
    )
    adjacency = np.zeros((d, d), dtype=np.int8)
    return structure_output(
        dag_from_adjacency(adjacency), labels=labels, metadata={"effects": [record], **metadata}
    )


@method_spec("ld3")
def ld3(
    data: npt.ArrayLike,
    *,
    treatment: str | int | None = None,
    outcome: str | int | None = None,
    alpha: float = 0.005,
    indep_test: str | None = None,
    degrees_of_freedom: str = "mean",
    categorical: Sequence[str | int] | None = None,
    missing: str | None = None,
    seed: int | None = None,
) -> StructureOutput:
    """Test whether a treatment directly causes an outcome, and find the outcome's parents (LD3).

    LD3 sorts every other variable by its causal relation to the pair from tests that read the
    pair and the variables already sorted, keeps those that are parents of the outcome, and tests
    the treatment against the outcome given the parents that are confounders or mediators [1]_.
    The treatment is a parent of the outcome (the structural direct criterion of fairness
    analysis) when the two stay dependent. The parents it finds are a valid adjustment set for
    the weighted controlled direct effect. It runs a number of tests linear in the number of
    variables.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``.
    treatment : str or int
        Required: the cause of the pair, a column label (a string) or a 0-based position (an
        integer).
    outcome : str or int
        Required: the effect of the pair, a column label (a string) or a 0-based position (an
        integer). The outcome must have no descendant among the columns.
    alpha : float, default=0.005
        Significance level of every test, in ``(0, 1)``: a p-value above ``alpha`` is an
        independence, any other a dependence. The default is the level of the authors' code; the
        experiments of [1]_ use 0.01 and 0.05 too.
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

    Returns
    -------
    StructureOutput
        Wraps a ``dag`` :class:`~andrey.core.GraphStructure` over every column whose edges point
        into the outcome, one from each parent found, the treatment included when the criterion
        holds. ``metadata["effects"]`` holds one record (:mod:`andrey.core.effects`) for the pair,
        by column label or 0-based position: ``estimand`` is ``"direct"``; ``nonzero`` is the
        structural direct criterion, whether the treatment is a parent of the outcome;
        ``identifiable`` is ``None``, since the method assumes every parent of the outcome is
        observed and does not test it; ``adjustment_sets`` is ``[parents]``, the parents of the
        outcome found other than the treatment, possibly empty, with ``set_kind``
        ``"parents_of_outcome"``. ``metadata["partition"]`` maps each label to its variables:
        ``"Z1,3"`` (the confounders and mediators found to be parents of the outcome), ``"Z4"``
        (causes of the outcome independent of the treatment, parents or not), ``"Z5,7"``
        (instruments or descendants of the treatment), ``"Z8"`` (independent of both), and
        ``"unlabeled"``; a variable that passes both the ``"Z8"`` and the ``"Z4"`` test, which
        only a test error allows, is under both. ``metadata["algorithm"]`` is ``"ld3"``. Under a
        mixed-data setting, ``metadata["categorical"]`` lists the categorical columns it read.
        With a randomized ``indep_test``, ``metadata["seed"]`` is the seed used.
        ``metadata["n_tests"]`` is the number of conditional-independence tests the fit
        evaluated.

    Raises
    ------
    TypeError
        If an argument has the wrong type, such as a string for a number; the message names the
        method, the argument, what it accepts, and the value. Also if ``treatment`` or ``outcome``
        is neither a column label nor a position, or ``categorical`` is not a list of column
        names and positions.
    ValueError
        If an argument lies outside its range or is not one of its choices; the message names the
        method, the argument, what it accepts, and the value. Also if ``treatment`` or ``outcome``
        is missing or names a column ``data`` lacks, or both name one column; or if ``data`` or
        ``categorical`` does not suit the test, as for :func:`~andrey.pc`. With ``"fisherz"`` or
        ``"student_t"``, also if a column repeats another up to scale and shift.

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
    **When to use.** LD3 suits observational data where you want to know whether one variable
    affects an outcome directly, not only through others, such as whether a protected attribute
    affects a decision directly, and to estimate that direct effect, without learning a graph.
    Its claim [1]_: when every independence follows from the causal graph (faithfulness), the
    tests are exact, the outcome has no descendant among the columns (its assumption A1), and
    every parent of the outcome is observed (A2), it finds exactly the parents of the outcome and
    whether the treatment is one (Theorem 1; see Limitations for graphs where the second part
    fails); hidden variables that are not parents of the outcome are allowed (Theorem 2), and
    the parents other than the treatment are a valid adjustment set for the weighted controlled
    direct effect (Theorem 3). The default
    Fisher-Z test assumes continuous data with linear effects and Gaussian noise; the other tests
    read categorical or mixed data or detect nonlinear effects, as :func:`~andrey.pc` describes.
    For an adjustment set for the total effect, use :func:`~andrey.ldp`.

    **How it works.** Algorithm 1 of [1]_. Three tests sort each variable on its own: independent
    of both is ``Z8``; dependent on the outcome and independent of it given the treatment is
    ``Z5,7``; independent of the treatment and dependent on it given the outcome is ``Z4``. A
    remaining variable dependent on the outcome given the treatment, the ``Z4`` variables, and the
    other remaining variables is a parent of the outcome. A ``Z4`` variable dependent on the
    outcome given the treatment, those parents, and the other ``Z4`` variables is a parent too.
    The criterion holds when the treatment and the outcome are dependent given the parents found
    among the remaining variables. Each query is tested once. The answer does not depend on the
    column order.

    **Missing values.** A ``NaN`` in ``data`` marks a missing value. Each test reads only the rows
    where all of its variables are observed (test-wise deletion) [2]_ [3]_, so it keeps rows that
    deleting every incomplete row first would drop; ``metadata["rows_tested"]`` gives the fewest and
    the median rows a test read. When values are missing completely at random, the rows a test keeps
    are a random sample, so each test keeps its level [3]_. When whether a value is missing depends
    on the variables, the rows a test keeps are a selected sample of their own. A test raises
    ``ValueError`` when it has fewer rows than it needs, ``|S| + 3`` for ``"fisherz"`` and
    ``"student_t"`` and 2 for the other tests, or when a continuous column is constant on its rows.

    **Limitations.** Not benchmarked yet. Tested with exact d-separation on the graph of [1]_
    and on random graphs whose outcome has no descendants, and on simulated linear-Gaussian data.
    The criterion's test leaves out the parents independent of the treatment: when one of them
    also causes a mediator that is a parent, as in the graph of Figure D.1 of [1]_, the test
    finds a dependence through that mediator, and the criterion says the treatment is a parent
    when it is not. A descendant of the outcome among the columns, or a hidden parent of it,
    breaks the claim.
    The parent tests condition on most of the other variables, so they need more rows as the
    number of variables grows. With 5,000 rows a fit ran about 65 tests on the 18-node graph of
    [1]_ and about 190 on random 50-node graphs, each in under a second. The numbers are a
    reference, not a hard limit: they change with the data generator, the density, the method's
    settings, and the machine.

    References
    ----------
    .. [1] Maasch, J., Gan, K., Chen, V., Orfanoudaki, A., Akpinar, N.-J., and Wang, F. (2025).
       Local causal discovery for structural evidence of direct discrimination. Proceedings of the
       AAAI Conference on Artificial Intelligence 39(18), 19349-19357.
    .. [2] Strobl, E. V., Visweswaran, S., and Spirtes, P. L. (2018). Fast causal inference with
       non-random missingness by test-wise deletion. International Journal of Data Science and
       Analytics 6(1), 47-62.
    .. [3] Tu, R., Zhang, C., Ackermann, P., Mohan, K., Kjellstrom, H., and Zhang, K. (2019). Causal
       discovery in the presence of missing data. Proceedings of Machine Learning Research 89,
       1762-1770.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from andrey import ld3
    >>> rng = np.random.default_rng(0)
    >>> c, p = rng.standard_normal((2, 5000))
    >>> x = c + rng.standard_normal(5000)
    >>> m = x + rng.standard_normal(5000)
    >>> y = m + c + p + rng.standard_normal(5000)
    >>> df = pd.DataFrame({"c": c, "p": p, "x": x, "m": m, "y": y})
    >>> out = ld3(df, treatment="x", outcome="y")
    >>> record = out.metadata["effects"][0]
    >>> record["nonzero"], record["adjustment_sets"]
    (False, [['c', 'p', 'm']])
    """
    from andrey.api._adapt import dag_from_adjacency
    from andrey.core.effects import effect_record

    assert indep_test is not None  # method_spec picks it from the column types
    result, (x, y, d, labels, name), metadata = _fit(
        "ld3", data, treatment, outcome, alpha, indep_test, degrees_of_freedom, categorical, seed
    )
    record = effect_record(
        x,
        y,
        estimand="direct",
        nonzero=result.direct,
        adjustment_sets=[result.parents],
        set_kind="parents_of_outcome",
        name=name,
    )
    adjacency = np.zeros((d, d), dtype=np.int8)
    adjacency[result.parents, y] = 1
    if result.direct:
        adjacency[x, y] = 1
    return structure_output(
        dag_from_adjacency(adjacency), labels=labels, metadata={"effects": [record], **metadata}
    )


def _ldecc_record(result, x: int, y: int, name) -> dict:
    """Build the ``effects`` record of an LDECC answer: its locally valid parent sets.

    The outcome a parent of the treatment makes the effect zero in every graph the answer allows,
    and no set valid; with one set and no outcome in it, the effect is identified; otherwise the
    sets are listed with the outcome left out of a set that holds it.
    """
    from andrey.core.effects import effect_record

    sets = result.parent_sets
    if y in result.parents:
        return effect_record(x, y, estimand="total", nonzero=False, identifiable=False, name=name)
    if len(sets) == 1:
        return effect_record(
            x,
            y,
            estimand="total",
            identifiable=True,
            adjustment_sets=sets,
            set_kind="local_parents",
            name=name,
        )
    kept = [[v for v in s if v != y] for s in sets]
    return effect_record(
        x,
        y,
        estimand="total",
        adjustment_sets=kept,
        set_kind="local_parents" if kept else None,
        name=name,
    )


@method_spec("ldecc")
def ldecc(
    data: npt.ArrayLike,
    *,
    treatment: str | int | None = None,
    outcome: str | int | None = None,
    blanket: str = "iamb",
    alpha: float = 0.05,
    check: bool = True,
    indep_test: str | None = None,
    degrees_of_freedom: str = "mean",
    categorical: Sequence[str | int] | None = None,
    missing: str | None = None,
    seed: int | None = None,
    max_depth: int | None = None,
) -> StructureOutput:
    """Find the parent sets that adjust for a treatment's effect on an outcome, locally (LDECC).

    LDECC learns the edges at the treatment and their directions without the whole graph [1]_. It
    learns the treatment's Markov blanket and its neighbors among it, then runs the tests of the PC
    algorithm over the other pairs of variables and reads each independence it finds for what it
    says about the treatment's neighbors, stopping once every neighbor is directed. Its answer is
    the set of effects of the treatment on the outcome, one per parent set of the treatment that
    the local structure allows, each estimated by adjusting for that set (the IDA approach).

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``.
    treatment : str or int
        Required: the cause of the pair, a column label (a string) or a 0-based position (an
        integer).
    outcome : str or int
        Required: the effect of the pair, a column label (a string) or a 0-based position (an
        integer).
    blanket : str, default="iamb"
        The Markov blanket learner of the first step: :func:`~andrey.iamb`, the learner of the
        paper's runs [1]_, or :func:`~andrey.inter_iamb`, :func:`~andrey.gs`,
        :func:`~andrey.fast_iamb`, :func:`~andrey.hiton_mb`, :func:`~andrey.mmmb`,
        :func:`~andrey.pcmb`, ``"iamb_forward"`` (IAMB's grow phase without its shrink), or
        ``"total_conditioning"`` (each variable the test finds dependent on the treatment given
        all the other variables) by name. A superset of the blanket keeps the answer right under
        exact tests and costs tests.
    alpha : float, default=0.05
        Significance level of every test, in ``(0, 1)``: a p-value above ``alpha`` is an
        independence, any other a dependence. The default is the level of the authors' code; the
        experiments of [1]_ use ``0.01 * 32000 / n`` for ``n`` rows.
    check : bool, default=True
        How an eager collider check directs parents when neither variable of the pair is a
        neighbor of the treatment. ``True`` (the paper's LDECC-checks, Remark 21 of [1]_) marks
        the minimal neighbor separators of the two as parents only when they are the same set,
        which needs weaker assumptions and was more accurate in the paper's experiments; ``False``
        marks both.
    indep_test : str or None, default=None
        Conditional-independence test, any of :func:`~andrey.pc`'s: ``"fisherz"``,
        ``"student_t"``, or ``"normal_scores"`` for continuous data, ``"chisq"`` or ``"gsq"`` for
        categorical data, ``"degen_gauss"`` or ``"cond_gauss"`` for mixed data, or a nonlinear
        test. ``None`` picks from the declared column types, as for :func:`~andrey.pc`. The
        paper's runs use Fisher's z test [1]_. ``metadata["indep_test"]`` names the test the fit
        used: the one picked from the declared column types when this is ``None``.
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
        The maximum conditioning-set size of every conditional-independence test, at least ``0``;
        ``None``, the default, sets no limit. A test the cap rules out is not run, and the step
        that needs it draws no conclusion from it. Under a cap, the blanket learner stops growing
        at ``max_depth + 1`` members.

    Returns
    -------
    StructureOutput
        Wraps a ``cpdag`` :class:`~andrey.core.GraphStructure` over every column holding only the
        edges at the treatment: ``v -> treatment`` for a parent, ``treatment -> v`` for a child, and
        ``v - treatment`` for a neighbor left undirected. An absent edge away from the treatment
        means nothing. ``metadata["effects"]`` holds one record (:mod:`andrey.core.effects`) for the
        pair, by column label or 0-based position: ``estimand`` is ``"total"`` and ``set_kind``
        ``"local_parents"``; ``adjustment_sets`` lists the locally valid parent sets, the parents
        with each subset of the undirected neighbors that holds no pair found to be a non-collider
        at the treatment, the outcome left out of a set that holds it. With one set,
        ``identifiable`` is ``True``; with several, ``None``. When the outcome is a parent of the
        treatment, the effect is zero in every graph the answer allows and no set is valid:
        ``nonzero`` and ``identifiable`` are ``False`` and ``adjustment_sets`` is ``[]``. When the
        marks conflict, so that no parent set is locally valid, ``adjustment_sets`` is ``[]`` and
        ``identifiable`` ``None``: the method gives no answer. ``nonzero`` is ``None`` otherwise.
        ``metadata["algorithm"]`` is ``"ldecc"``. Under a mixed-data setting,
        ``metadata["categorical"]`` lists the categorical columns it read. With a randomized
        ``indep_test``, ``metadata["seed"]`` is the seed used. ``metadata["n_tests"]`` is the number
        of conditional-independence tests the fit evaluated; a query asked again reads its kept
        answer and is not a new test. ``metadata["tie_break"]`` is the tie-break rule the fit ran
        under, ``andrey.config.tie_break``.

    Raises
    ------
    TypeError
        If an argument has the wrong type, such as a string for a number; the message names the
        method, the argument, what it accepts, and the value. Also if ``treatment`` or ``outcome``
        is neither a column label nor a position, or ``categorical`` is not a list of column
        names and positions.
    ValueError
        If an argument lies outside its range or is not one of its choices; the message names the
        method, the argument, what it accepts, and the value. Also if ``treatment`` or ``outcome``
        is missing or names a column ``data`` lacks, or both name one column; or if ``data`` or
        ``categorical`` does not suit the test, as for :func:`~andrey.pc`. With ``"fisherz"`` or
        ``"student_t"``, also if a column repeats another up to scale and shift.

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
    **When to use.** LDECC suits observational data where you want the effect of one variable on
    another, with no hidden variables, and learning the whole graph would cost too many tests: it
    learns only the edges at the treatment. Its claim [1]_: when every independence follows from
    the causal graph (faithfulness) and the tests are exact, the sets it returns give the same
    effects as the treatment's parent sets over the graphs the CPDAG allows (Theorem 5). Where the
    parents of the treatment form unshielded colliders with variables far from it, its eager
    collider checks direct them with fewer tests than learning the structure outward from the
    treatment, and in other graphs with more (Props. 6-8). The default Fisher-Z test assumes
    continuous data with linear effects and Gaussian noise; the other tests read categorical or
    mixed data or detect nonlinear effects, as :func:`~andrey.pc` describes.

    **How it works.** Fig. 5 of [1]_. The first step learns the treatment's Markov blanket with
    ``blanket``. The neighbors are the blanket members that no subset of the rest of the blanket
    separates from the treatment, the rule of [2]_ (App. A.1.1, Lemma A.2) for this step: each
    member is tested against the subsets of the blanket without it. A neighbor outside a spouse's
    separating set and dependent on the spouse given it is a child, an unshielded collider at the
    neighbor. The PC algorithm's tests then run over the pairs that do not hold the treatment, the
    treatment adjacent to its neighbors only. For each independence of ``a`` and ``b`` given
    ``S``: two neighbors separated without the treatment are parents, as are the neighbors in
    ``S``; two neighbors separated by a set holding the treatment are a non-collider at it, and a
    neighbor that makes them dependent again is a child; otherwise, when ``a`` and ``b`` are
    dependent given ``S`` and the treatment, the treatment is a collider or a descendant of one
    between them, and their minimal neighbor separators, the smallest sets of neighbors that
    separate each from the treatment, are parents (Prop. 4; with ``check``, only when the two are
    equal). A neighbor joined to a parent by a non-collider is a child. A neighbor marked both
    ways is a child, as in the authors' code. Each query is tested once. Variables and subsets
    are visited in an order the data fix, the treatment's neighbors first and the treatment first
    in a conditioning set, as in the authors' code, and the first subset that separates a pair is
    kept, so the answer does not depend on the column order; it does not depend on the columns'
    units when the test does not.

    **Missing values.** A ``NaN`` in ``data`` marks a missing value. Each test reads only the rows
    where all of its variables are observed (test-wise deletion) [3]_ [4]_, so it keeps rows that
    deleting every incomplete row first would drop; ``metadata["rows_tested"]`` gives the fewest and
    the median rows a test read. When values are missing completely at random, the rows a test keeps
    are a random sample, so each test keeps its level [4]_. When whether a value is missing depends
    on the variables, the rows a test keeps are a selected sample of their own. A test raises
    ``ValueError`` when it has fewer rows than it needs, ``|S| + 3`` for ``"fisherz"`` and
    ``"student_t"`` and 2 for the other tests, or when a continuous column is constant on its rows.

    **Limitations.** Not benchmarked yet. Tested with exact d-separation on the graphs of [1]_
    and [2]_ and on random graphs, and on simulated linear-Gaussian data. In some graphs the
    eager collider checks leave a parent of the treatment undirected that the CPDAG directs [2]_
    (App. A.2): the answer then lists more parent sets than the CPDAG allows, the treatment's
    true parents among them. When some neighbor stays undirected, the PC algorithm's tests run
    over every pair, as many as learning the whole skeleton, and the sets number up to two to the
    power of the undirected neighbors. On the 25-node graphs of [1]_ (App. C.1) with 8,000 to
    32,000 rows, a fit ran a median of 310 to 350 tests, in under a second. The numbers are a
    reference, not a hard limit: they change with the data generator, the density, the method's
    settings, and the machine.

    References
    ----------
    .. [1] Gupta, S., Childers, D., and Lipton, Z. C. (2023). Local causal discovery for
       estimating causal effects. Proceedings of the Second Conference on Causal Learning and
       Reasoning, Proceedings of Machine Learning Research 213, 408-447.
    .. [2] Schubert, M., Claassen, T., and Magliacane, S. (2026). Local causal discovery for
       statistically efficient causal inference. Proceedings of the 29th International Conference
       on Artificial Intelligence and Statistics, Proceedings of Machine Learning Research 300.
    .. [3] Strobl, E. V., Visweswaran, S., and Spirtes, P. L. (2018). Fast causal inference with
       non-random missingness by test-wise deletion. International Journal of Data Science and
       Analytics 6(1), 47-62.
    .. [4] Tu, R., Zhang, C., Ackermann, P., Mohan, K., Kjellstrom, H., and Zhang, K. (2019). Causal
       discovery in the presence of missing data. Proceedings of Machine Learning Research 89,
       1762-1770.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from andrey import ldecc
    >>> rng = np.random.default_rng(0)
    >>> a, b = rng.standard_normal((2, 5000))
    >>> x = a + b + rng.standard_normal(5000)
    >>> m = x + rng.standard_normal(5000)
    >>> y = m + a + b + rng.standard_normal(5000)
    >>> df = pd.DataFrame({"a": a, "b": b, "x": x, "m": m, "y": y})
    >>> out = ldecc(df, treatment="x", outcome="y")
    >>> out.metadata["effects"][0]["adjustment_sets"]
    [['a', 'b']]
    """
    from andrey.api._local import pair_columns
    from andrey.constraint.blanket import local_search
    from andrey.constraint.ldecc import ldecc as engine
    from andrey.core import ARROW, TAIL
    from andrey.core.ci import counting_tests
    from andrey.core.orient import to_structure

    assert indep_test is not None  # method_spec picks it from the column types
    X, mask, labels, _ = fit_inputs(
        data, "indep_test", indep_test, categorical, None, method="ldecc", missing=True
    )
    d = X.shape[1]
    x, y = pair_columns(treatment, outcome, labels, d, "ldecc")
    with copies_named(labels), rows_tested() as rows, counting_tests() as tests:
        search = local_search(
            X,
            alpha=alpha,
            indep_test=indep_test,
            degrees_of_freedom=degrees_of_freedom,
            categorical=mask,
            seed=seed,
            max_depth=max_depth,
        )
        with search.pinned():
            result = engine(search, x, blanket=blanket, check=check)
    name = label_of(labels)
    star = np.zeros((d, d), dtype=np.int8)
    for v in result.neighbors:
        near = ARROW if v in result.parents else TAIL
        far = ARROW if v in result.children else TAIL
        star[x, v], star[v, x] = near, far
    metadata = {
        "effects": [_ldecc_record(result, x, y, name)],
        **categorical_metadata(mask, data),
        **rows_metadata(rows),
        "n_tests": tests.n,
    }
    return structure_output(to_structure(star, kind="cpdag"), labels=labels, metadata=metadata)


@method_spec("local_optimal_adjustment")
def local_optimal_adjustment(
    data: npt.ArrayLike,
    *,
    variables: Sequence[str | int] | None = None,
    blanket: str = "gs",
    alpha: float = 0.01,
    indep_test: str | None = None,
    degrees_of_freedom: str = "mean",
    categorical: Sequence[str | int] | None = None,
    missing: str | None = None,
    seed: int | None = None,
) -> StructureOutput:
    """Find the relation of two variables and the optimal adjustment set between them (LOAD).

    LOAD [1]_ takes two variables without knowing which causes which. It learns the edges at each
    one Markov blanket at a time, as MB-by-MB does, and tests, from those edges alone, whether one
    is an ancestor of the other in every causal graph the tests cannot tell apart (explicit
    ancestry), in some of them (possible ancestry), or in none [2]_. When one is an explicit
    ancestor, the treatment, it checks from the edges of the treatment's undirected neighbors
    whether its effect on the other, the outcome, is identifiable, and if so returns the optimal
    adjustment set: of all valid sets, the one whose estimate of a linear effect has the smallest
    asymptotic variance [3]_. Otherwise it returns the parent sets of the treatment that its edges
    allow, one estimate each [4]_. It learns only the blankets the answer needs, not the whole
    graph.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``.
    variables : sequence of str or int, or None, default=None
        Required: the two variables to answer for, in any order: a list of column labels (strings)
        or 0-based positions (integers). The method answers for both ordered pairs of them.
        ``None`` raises ``ValueError``.
    blanket : str, default="gs"
        The Markov blanket learner run for each variable the search visits: :func:`~andrey.gs`,
        Grow-Shrink, the learner of the paper's runs [1]_, or ``"iamb_forward"``, IAMB's grow phase
        without its shrink, ``"total_conditioning"``, each variable the test finds dependent on it
        given all the other variables, or :func:`~andrey.iamb`, :func:`~andrey.inter_iamb`,
        :func:`~andrey.fast_iamb`, :func:`~andrey.hiton_mb`, :func:`~andrey.mmmb`, or
        :func:`~andrey.pcmb` by name.
    alpha : float, default=0.01
        Significance level of every test, in ``(0, 1)``: a p-value above ``alpha`` is an
        independence, any other a dependence. The default is the level of the experiments of [1]_
        and of the authors' code.
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

    Returns
    -------
    StructureOutput
        Wraps a ``cpdag`` :class:`~andrey.core.GraphStructure` over every column holding only the
        edges at the two variables, as their own searches directed them: ``a -> v`` for a direct
        cause of ``v``, ``v -> a`` for a direct effect, and ``a - v`` where the search left the
        direction undecided. An absent edge away from the two means nothing.
        ``metadata["effects"]`` holds two records (:mod:`andrey.core.effects`), the first with the
        first of ``variables`` as the treatment, by column label or 0-based position, each for the
        total effect: ``nonzero`` is ``True`` when the treatment is an explicit ancestor of the
        outcome, ``False`` when it is an ancestor in no graph the tests allow, and ``None`` when it
        is one in some; ``identifiable`` is ``True`` when one set is a valid adjustment set in
        every such graph, with ``adjustment_sets`` the optimal set (``set_kind`` ``"optimal"``) for
        a nonzero effect and ``[]`` for a zero one; otherwise it is ``False`` and
        ``adjustment_sets`` lists the locally valid parent sets of the treatment (``set_kind``
        ``"local_parents"``), the outcome left out, one estimate each. A zero effect between two
        adjacent variables is not identifiable in this sense: the edge into the treatment is a
        back-door path no set blocks. ``metadata["blankets_learned"]`` lists the variables whose
        Markov blanket the search learned, in order; the last step learns blankets over the
        variables that are not possible descendants of the treatment, so a variable can appear
        twice. ``metadata["algorithm"]`` is ``"local_optimal_adjustment"``. Under a mixed-data
        setting, ``metadata["categorical"]`` lists the categorical columns it read. With a
        randomized ``indep_test``, ``metadata["seed"]`` is the seed used. ``metadata["n_tests"]``
        is the number of conditional-independence tests the fit evaluated; a query answered again
        from a kept answer is not a new test. ``metadata["tie_break"]`` is the tie-break rule the
        fit ran under, ``andrey.config.tie_break``.

    Raises
    ------
    TypeError
        If an argument has the wrong type, such as a string for a number; the message names the
        method, the argument, what it accepts, and the value. Also if ``variables`` is a string or
        a member is neither a column label nor a position, or ``categorical`` is not a list of
        column names and positions.
    ValueError
        If an argument lies outside its range or is not one of its choices; the message names the
        method, the argument, what it accepts, and the value. Also if ``variables`` is missing,
        does not name exactly two different columns, or names a column ``data`` lacks; or if
        ``data`` or ``categorical`` does not suit the test, as for :func:`~andrey.pc`. With
        ``"fisherz"`` or ``"student_t"``, also if a column repeats another up to scale and shift.

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
    **When to use.** LOAD suits i.i.d. observational data with no unmeasured common causes, in
    which every independence follows from the graph (faithfulness), when you want to estimate the
    effect between two variables and do not know which one causes the other [1]_. Its claim: with
    exact tests it finds the relation of the two, whether the effect is identifiable, and the
    optimal adjustment set when it is (Corollary 4.1, Theorem 4.3 of [1]_). It is designed for
    large graphs in which each variable has a small Markov blanket. The default Fisher-Z test
    assumes continuous data with linear effects and Gaussian noise; the other tests read
    categorical or mixed data or detect nonlinear effects, as :func:`~andrey.pc` describes. For a
    pair whose order you know and an adjustment set that need not be optimal, see
    :func:`~andrey.ldp`.

    **How it works.** Algorithm 3 of [1]_ in four steps. Step 1 learns the edges at each of the two
    variables with MB-by-MB [5]_ as [1]_ runs it: Grow-Shrink blankets, each local structure learned
    by the skeleton step of PC over a variable and its blanket, Meek's rules, and every blanket,
    local structure, and separating set kept for the later searches. A variable is an explicit
    ancestor of the other when the other is its child, or, when the two are not adjacent, when the
    test finds them dependent given the first one's parents and undirected neighbors; it is a
    possible ancestor when the other is its child or undirected neighbor, or, when not adjacent,
    when they test dependent given its parents alone [2]_. With no explicit ancestor, each direction
    gets the parent sets of its treatment when the treatment is a possible ancestor, and a zero
    effect otherwise. Step 2 learns the edges of each undirected neighbor of the treatment and keeps
    the effect identifiable only when no such neighbor is adjacent to the outcome and each tests
    independent of the outcome given its parents and the treatment (Lemma 4.3 of [1]_). Step 3 marks
    as possible descendants of the treatment its children and undirected neighbors and each other
    variable that tests dependent on it given its parents. Step 4 runs MB-by-MB at the outcome over
    the variables that are not possible descendants, with the treatment and the outcome, and the
    outcome's parents there, the treatment left out, are the optimal set (Lemma 4.4 of [1]_). The
    parent sets of a treatment are its parents with each subset of its other undirected neighbors
    whose members are adjacent to every other parent in the set, so that no new v-structure forms
    [4]_. The variable whose rank, computed from the data, comes first is searched first, ties
    between candidates go by rank as for :func:`~andrey.mb_by_mb`, and the neighbors of Step 2 go in
    rank order, so the answer does not depend on the column order unless two columns are identical.
    An edge that two v-structures direct both ways, which exact tests never give, is left
    undirected, so each end reads the other as an undirected neighbor.

    **Zero effects.** A zero effect between two adjacent variables, the treatment a child of the
    outcome, is not identifiable here: the edge into the treatment is a back-door path that no set
    blocks, so the record lists the treatment's parent sets. This is the definition
    :func:`~andrey.metrics.effect_oracle` scores on the true graph; [1]_ counts every zero effect
    as identified.

    **Missing values.** A ``NaN`` in ``data`` marks a missing value. Each test reads only the rows
    where all of its variables are observed (test-wise deletion) [6]_ [7]_, so it keeps rows that
    deleting every incomplete row first would drop; ``metadata["rows_tested"]`` gives the fewest and
    the median rows a test read. When values are missing completely at random, the rows a test keeps
    are a random sample, so each test keeps its level [7]_. When whether a value is missing depends
    on the variables, the rows a test keeps are a selected sample of their own. A test raises
    ``ValueError`` when it has fewer rows than it needs, ``|S| + 3`` for ``"fisherz"`` and
    ``"student_t"`` and 2 for the other tests, or when a continuous column is constant on its rows.

    **Limitations.** Not benchmarked yet. Tested with exact d-separation on random graphs of up to
    20 variables, against the relation, the identifiability, and the optimal set of the true
    graph, and on simulated linear-Gaussian data. Each local structure tests a pair against the
    subsets of the neighbors of its ends with no cap, so a variable with many neighbors costs many
    tests, and the parent sets of a treatment grow with the subsets of its undirected neighbors.
    The numbers are a reference, not a hard limit: they change with the data generator, the
    density, the method's settings, and the machine.

    References
    ----------
    .. [1] Schubert, M., Claassen, T., and Magliacane, S. (2026). Local causal discovery for
       statistically efficient causal inference. Proceedings of the 29th International Conference
       on Artificial Intelligence and Statistics.
    .. [2] Fang, Z., Liu, Y., Geng, Z., Zhu, S., and He, Y. (2022). A local method for identifying
       causal relations under Markov equivalence. Artificial Intelligence 305, 103669.
    .. [3] Henckel, L., Perkovic, E., and Maathuis, M. H. (2022). Graphical criteria for efficient
       total effect estimation via adjustment in causal linear models. Journal of the Royal
       Statistical Society Series B 84(2), 579-599.
    .. [4] Maathuis, M. H., Kalisch, M., and Buhlmann, P. (2009). Estimating high-dimensional
       intervention effects from observational data. The Annals of Statistics 37(6A), 3133-3164.
    .. [5] Wang, C., Zhou, Y., Zhao, Q., and Geng, Z. (2014). Discovering and orienting the edges
       connected to a target variable in a DAG via a sequential local learning approach.
       Computational Statistics and Data Analysis 77, 252-266.
    .. [6] Strobl, E. V., Visweswaran, S., and Spirtes, P. L. (2018). Fast causal inference with
       non-random missingness by test-wise deletion. International Journal of Data Science and
       Analytics 6(1), 47-62.
    .. [7] Tu, R., Zhang, C., Ackermann, P., Mohan, K., Kjellstrom, H., and Zhang, K. (2019). Causal
       discovery in the presence of missing data. Proceedings of Machine Learning Research 89,
       1762-1770.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from andrey import local_optimal_adjustment
    >>> rng = np.random.default_rng(0)
    >>> c, i, p = rng.standard_normal((3, 5000))
    >>> x = c + i + rng.standard_normal(5000)
    >>> y = x + c + p + rng.standard_normal(5000)
    >>> df = pd.DataFrame({"c": c, "i": i, "p": p, "x": x, "y": y})
    >>> out = local_optimal_adjustment(df, variables=["x", "y"])
    >>> effect = out.metadata["effects"][0]
    >>> effect["nonzero"], effect["identifiable"], effect["adjustment_sets"]
    (True, True, [['c', 'p']])
    """
    from andrey.api._local import variable_columns
    from andrey.constraint.blanket import local_search
    from andrey.constraint.local_optimal_adjustment import local_optimal_adjustment as search_pair
    from andrey.core.ci import counting_tests
    from andrey.core.effects import effect_record
    from andrey.core.knowledge import as_node, column_position
    from andrey.core.orient import to_structure

    assert indep_test is not None  # method_spec picks it from the column types
    method = "local_optimal_adjustment"
    X, mask, labels, _ = fit_inputs(
        data, "indep_test", indep_test, categorical, None, method=method, missing=True
    )
    d = X.shape[1]
    a, b = variable_columns(variables, labels, d, method, minimum=2, maximum=2)
    assert variables is not None  # variable_columns raises on None
    first = column_position(as_node(list(variables)[0], method), labels, d, method)
    with copies_named(labels), rows_tested() as rows, counting_tests() as tests:
        search = local_search(
            X,
            alpha=alpha,
            indep_test=indep_test,
            degrees_of_freedom=degrees_of_freedom,
            categorical=mask,
            seed=seed,
        )
        with search.pinned():
            result = search_pair(search, (a, b), blanket)
    name = label_of(labels)
    records = [
        effect_record(
            answer.treatment,
            answer.outcome,
            estimand="total",
            nonzero=answer.nonzero,
            identifiable=answer.identifiable,
            adjustment_sets=answer.adjustment_sets,
            set_kind=answer.set_kind,
            name=name,
        )
        for answer in sorted(result.answers, key=lambda answer: answer.treatment != first)
    ]
    metadata = {
        "effects": records,
        "blankets_learned": [name(v) for v in result.learned],
        **categorical_metadata(mask, data),
        **rows_metadata(rows),
        "n_tests": tests.n,
    }
    return structure_output(
        to_structure(result.marks, kind="cpdag"), labels=labels, metadata=metadata
    )


@method_spec("locpc_cde")
def locpc_cde(
    data: npt.ArrayLike,
    *,
    treatment: str | int | None = None,
    outcome: str | int | None = None,
    alpha: float = 0.05,
    indep_test: str | None = None,
    degrees_of_freedom: str = "mean",
    categorical: Sequence[str | int] | None = None,
    missing: str | None = None,
    seed: int | None = None,
    max_depth: int | None = None,
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Find whether a direct effect is identifiable, and the outcome's parents (LocPC-CDE).

    LocPC-CDE learns the graph around the outcome one hop at a time with PC's tests, run only at
    the variables near the outcome, until every edge at the outcome is oriented, so that its
    parents are known and are a valid adjustment set for the controlled direct effect, or until a
    larger hop adds no variable, so that an edge at the outcome left unoriented stays so [1]_. It
    stops at once when the treatment is not adjacent to the outcome or is its child: the direct
    effect is then zero.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``.
    treatment : str or int
        Required: the cause of the pair, a column label (a string) or a 0-based position (an
        integer).
    outcome : str or int
        Required: the effect of the pair, a column label (a string) or a 0-based position (an
        integer).
    alpha : float, default=0.05
        Significance level of every test, in ``(0, 1)``: a p-value above ``alpha`` is an
        independence, any other a dependence. The default is the level of the paper's experiments
        and of the authors' code [1]_.
    indep_test : str or None, default=None
        Conditional-independence test, any of :func:`~andrey.pc`'s: ``"fisherz"``,
        ``"student_t"``, or ``"normal_scores"`` for continuous data, ``"chisq"`` or ``"gsq"`` for
        categorical data, ``"degen_gauss"`` or ``"cond_gauss"`` for mixed data, or a nonlinear
        test. ``None`` picks from the declared column types, as for :func:`~andrey.pc`. The
        paper's runs use Fisher-Z on linear-Gaussian data and the likelihood-ratio test,
        ``"gsq"``, on binary data [1]_. ``metadata["indep_test"]`` names the test the fit used:
        the one picked from the declared column types when this is ``None``.
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
        The maximum conditioning-set size of every conditional-independence test, at least ``0``;
        ``None``, the default, sets no limit, as the paper's runs do.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the pairs
        allowed to be adjacent, and temporal tiers, all honored. A required edge is joined and
        never tested; a pair the knowledge makes non-adjacent, forbidden both ways or outside
        ``allowed``, never shares an edge. Every edge a tier or an edge forbidden one way directs
        is directed before the colliders, and Meek's rules, R4 included, direct the edges that
        follow. A variable the knowledge places before another, by a tier or a required edge, is
        not its descendant, so their edge is not tested again when the later one is visited
        [1]_. ``required_paths`` and ``forbidden_paths`` raise ``NotImplementedError``.

    Returns
    -------
    StructureOutput
        Wraps a ``cpdag`` :class:`~andrey.core.GraphStructure` over every column, the local
        essential graph of the outcome at the last hop learned: the edges at every variable the
        tests visited, directed where those tests decide. An edge between two variables within
        the hop of the outcome is an adjacency of the true graph; an edge from one of them to a
        variable farther out may join a descendant that no set of the inner variable's neighbors
        separates; an absent edge between two variables outside the hop means nothing.
        ``metadata["effects"]`` holds one record (:mod:`andrey.core.effects`) for the pair, by
        column label or 0-based position, with ``estimand`` ``"direct"`` and the paper's verdict in
        ``reason``: ``"not adjacent"`` or ``"treatment is a child"``, where ``nonzero`` is
        ``False``; ``"oriented"``, where every edge at the outcome is directed, ``identifiable``
        is ``True``, ``nonzero`` says whether the treatment is a parent, and ``adjustment_sets``
        is ``[parents]``, the outcome's other parents, with ``set_kind``
        ``"parents_of_outcome"``; or ``"not orientable"``, where ``identifiable`` is ``False``
        and ``nonzero`` is ``True`` when the treatment's edge into the outcome is directed, else
        ``None``. After an early stop with a zero effect, ``identifiable`` and the set are as for
        ``"oriented"`` when every edge at the outcome is directed, else ``None`` and ``[]``.
        ``metadata["hop"]`` is the last hop learned, ``metadata["visited"]`` lists the variables
        whose tests ran, in order, the outcome first, and ``metadata["double_bar"]`` lists the
        undirected edges ``[inner, outer]`` the no-non-collider rule marks. Under a mixed-data
        setting, ``metadata["categorical"]`` lists the categorical columns it read. With a
        randomized ``indep_test``, ``metadata["seed"]`` is the seed used.
        ``metadata["n_tests"]`` is the number of conditional-independence tests the fit
        evaluated; a query answered again from a kept answer is not a new test.
        ``metadata["tie_break"]`` is the tie-break rule the fit ran under,
        ``andrey.config.tie_break``.

    Raises
    ------
    TypeError
        If an argument has the wrong type, such as a string for a number; the message names the
        method, the argument, what it accepts, and the value. Also if ``treatment`` or ``outcome``
        is neither a column label nor a position, ``categorical`` is not a list of column names and
        positions, or ``knowledge`` is neither ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If an argument lies outside its range or is not one of its choices; the message names the
        method, the argument, what it accepts, and the value. Also if ``treatment`` or ``outcome``
        is missing or names a column ``data`` lacks, or both name one column; or if ``data`` or
        ``categorical`` does not suit the test, as for :func:`~andrey.pc`. With ``"fisherz"`` or
        ``"student_t"``, also if a column repeats another up to scale and shift. Also if
        ``knowledge`` names a column ``data`` lacks or contradicts itself, as
        :class:`~andrey.Knowledge` lists.
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
    **When to use.** LocPC-CDE suits i.i.d. observational data with no unmeasured common causes,
    when you want to know whether the controlled direct effect of one variable on another can be
    estimated, the effect with every other parent of the outcome held fixed, and which variables to
    adjust for, without learning the whole graph. Its claim [1]_: when every independence the tests
    near the outcome read follows from the graph (local faithfulness) and the tests are exact, the
    effect is identifiable exactly when the CPDAG directs every edge at the outcome, and then the
    outcome's parents are a valid adjustment set; an edge at the outcome that the graph learned
    around it directs is directed so in the CPDAG, so an identifiable effect is often found without
    learning the whole graph. The default Fisher-Z test assumes continuous data with linear effects
    and Gaussian noise; the other tests read categorical or mixed data or detect nonlinear effects,
    as :func:`~andrey.pc` describes. Regressing the outcome on the treatment and the set estimates
    the effect in a linear model. For the whole CPDAG, use :func:`~andrey.pc`.

    **How it works.** LocPC [1]_ (Algorithm 1) learns the graph around the outcome at hop ``h``.
    It joins the outcome to every other variable and runs PC's levels over it: at size ``s``,
    the outcome is tested against each neighbor given each size-``s`` subset of its neighbors at
    the start of the level, and the edge goes at the first set that separates them. The next
    round does the same for the outcome's neighbors, then for theirs, ``h + 1`` rounds in all; a
    variable visited again is tested again with its new neighbors, which removes an edge to a
    descendant that the first round could not separate. The colliders among the variables
    within ``h`` edges of the outcome are oriented, by the most clearly separated pair first,
    Meek's rules [2]_ run among them, and an undirected edge from an inner variable to an outer
    one is double-bar when no outer variable needs the outer end to be separated from the inner
    one. LocPC-CDE (Algorithm 2) runs LocPC at hop 0, 1, 2, ..., keeping every separating set
    and test result for the next hop, while the treatment is adjacent to the outcome and not its
    child and some edge at the outcome is undirected or double-bar, and stops when the hop adds
    no variable. Variables, neighbors, and conditioning sets are visited in the order of
    :func:`~andrey.core.ties.tie_ranks`, so the answer does not depend on the column order.

    **When it stops.** Algorithm 2 of [1]_ also stops after a hop when the variables that undirected
    and double-bar edges join to the outcome within the hop have no undirected edge leaving them and
    at most one double-bar edge each leaving them (the non-orientability criterion), which its
    Theorem 13 states makes the effect not identifiable. Andrey leaves this stop out and grows the
    hop until every edge at the outcome is oriented or the hop adds no variable, the strategy that
    Section 6 of [1]_ describes before the criterion. With exact tests, on the pairs ``x -> y`` of
    300 random graphs of 6 to 15 variables, the stop ended 23 of 3,091 searches for an identifiable
    direct effect while an edge at the outcome was undirected: a double-bar edge from a variable
    near the outcome to one farther out was directed into the nearer one, by a collider with another
    variable near the outcome, once the hop reached the farther one. Without it, every verdict on
    those graphs is the CPDAG's.

    **Missing values.** A ``NaN`` in ``data`` marks a missing value. Each test reads only the rows
    where all of its variables are observed (test-wise deletion) [3]_ [4]_, so it keeps rows that
    deleting every incomplete row first would drop; ``metadata["rows_tested"]`` gives the fewest and
    the median rows a test read. When values are missing completely at random, the rows a test keeps
    are a random sample, so each test keeps its level [4]_. When whether a value is missing depends
    on the variables, the rows a test keeps are a selected sample of their own. A test raises
    ``ValueError`` when it has fewer rows than it needs, ``|S| + 3`` for ``"fisherz"`` and
    ``"student_t"`` and 2 for the other tests, or when a continuous column is constant on its rows.

    **Limitations.** Not benchmarked yet. Tested with exact d-separation on random graphs and the
    paper's figures, and on simulated linear-Gaussian data. Without the paper's early stop (see
    **When it stops.**), a non-identifiable effect is searched until a hop adds no variable: with
    exact tests the mean hop of a search rose from 1.41 to 1.79 on random graphs of 6 to 15
    variables and from 1.46 to 2.16 on graphs of 20 to 30. An unmeasured common cause breaks the
    claim. With 5,000 rows a fit ran about 40 to 250 tests on the paper's random 10-node graphs, and
    on its 50-node graphs about 200 to 2,000 for an identifiable effect and 300 to 5,000 for one
    that is not, each in under a second. The numbers are a reference, not a hard limit: they change
    with the data generator, the density, the method's settings, and the machine.

    References
    ----------
    .. [1] Loranchet, T., and Assaad, C. K. (2026). Local Markov equivalence for PC-style local
       causal discovery and identification of controlled direct effects. Proceedings of the Fifth
       Conference on Causal Learning and Reasoning, PMLR 323, 84-118.
    .. [2] Meek, C. (1995). Causal inference and causal explanation with background knowledge.
       Proceedings of the Eleventh Conference on Uncertainty in Artificial Intelligence, 403-410.
    .. [3] Strobl, E. V., Visweswaran, S., and Spirtes, P. L. (2018). Fast causal inference with
       non-random missingness by test-wise deletion. International Journal of Data Science and
       Analytics 6(1), 47-62.
    .. [4] Tu, R., Zhang, C., Ackermann, P., Mohan, K., Kjellstrom, H., and Zhang, K. (2019). Causal
       discovery in the presence of missing data. Proceedings of Machine Learning Research 89,
       1762-1770.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from andrey import locpc_cde
    >>> rng = np.random.default_rng(0)
    >>> a, b, c = rng.standard_normal((3, 5000))
    >>> x = a + rng.standard_normal(5000)
    >>> y = x + b + rng.standard_normal(5000)
    >>> df = pd.DataFrame({"a": a, "b": b, "c": c, "x": x, "y": y})
    >>> record = locpc_cde(df, treatment="x", outcome="y").metadata["effects"][0]
    >>> record["identifiable"], record["nonzero"], record["adjustment_sets"]
    (True, True, [['b']])
    """
    from andrey.api._local import pair_columns
    from andrey.constraint.blanket import local_search
    from andrey.constraint.locpc import locpc_cde as locpc_cde_engine
    from andrey.core.ci import counting_tests
    from andrey.core.effects import effect_record
    from andrey.core.orient import to_structure

    assert indep_test is not None  # method_spec picks it from the column types
    X, mask, labels, constraints = fit_inputs(
        data, "indep_test", indep_test, categorical, knowledge, method="locpc_cde", missing=True
    )
    d = X.shape[1]
    x, y = pair_columns(treatment, outcome, labels, d, "locpc_cde")
    with copies_named(labels), rows_tested() as rows, counting_tests() as tests:
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
            answer = locpc_cde_engine(search, x, y)
    graph = answer.graph
    parents = graph.parents(y)
    oriented = not graph.non_arrow(y)
    if answer.reason in ("not adjacent", "treatment is a child"):
        nonzero = False
    elif x in parents:
        nonzero = True
    else:
        nonzero = None
    if answer.reason == "not orientable":
        identifiable = False
    else:
        identifiable = True if oriented else None
    name = label_of(labels)
    record = effect_record(
        x,
        y,
        estimand="direct",
        nonzero=nonzero,
        identifiable=identifiable,
        adjustment_sets=[[v for v in parents if v != x]] if identifiable else [],
        set_kind="parents_of_outcome" if identifiable else None,
        name=name,
        reason=answer.reason,
    )
    metadata = {
        "effects": [record],
        "hop": answer.hop,
        "visited": [name(v) for v in graph.visited],
        "double_bar": [[name(a), name(b)] for a, b in sorted(graph.double_bar)],
        **categorical_metadata(mask, data),
        **rows_metadata(rows),
        "n_tests": tests.n,
    }
    return structure_output(
        to_structure(graph.marks, kind="cpdag"), labels=labels, metadata=metadata
    )
