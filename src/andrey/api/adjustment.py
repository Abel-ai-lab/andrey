"""Treatment-outcome facade adapters.

Each method answers for one ordered pair of columns, a treatment and an outcome: the answer is a
record in ``metadata["effects"]`` (:mod:`andrey.core.effects`) with the adjustment sets it found,
next to the graph its paper licenses. ``covariate_rules`` decides whether the treatment affects the
outcome and finds valid adjustment sets among pretreatment covariates; ``ldp`` finds a valid
adjustment set for the total effect; ``ld3`` finds the parents of the outcome and whether the
treatment is one of them.
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


def _fit(
    method: str,
    data,
    treatment,
    outcome,
    alpha,
    indep_test,
    degrees_of_freedom,
    categorical,
    seed,
    max_depth=None,
):
    """Run the engine of ``method`` on the pair.

    The engines are in :mod:`andrey.constraint.partition` for ``ldp`` and ``ld3`` and in
    :mod:`andrey.constraint.covariates` for ``covariate_rules``. Returns its result, the fit's
    columns, labels, and namer, and the metadata: the method's own key (``partition`` or
    ``rule_counts``), then the shared ones.
    """
    from andrey.api._local import pair_columns
    from andrey.constraint.blanket import local_search
    from andrey.constraint.covariates import CovariateRulesResult
    from andrey.constraint.covariates import covariate_rules as rules_engine
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
            if method == "covariate_rules":
                result = rules_engine(search, x, y, others, max_depth=max_depth)
            else:
                result = (ldp_engine if method == "ldp" else ld3_engine)(search, x, y, others)
    name = label_of(labels)
    own = (
        {"rule_counts": result.counts}
        if isinstance(result, CovariateRulesResult)
        else {"partition": {k: [name(j) for j in v] for k, v in result.partition.items()}}
    )
    metadata = {
        **own,
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


@method_spec("covariate_rules")
def covariate_rules(
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
) -> StructureOutput:
    """Decide whether a treatment affects an outcome, and find covariate sets to adjust for.

    Every column other than the treatment and the outcome is a covariate measured before the
    treatment. Rule R1 finds a covariate ``w`` and a set ``Z`` of the others such that ``w`` and
    the outcome are dependent given ``Z`` and independent given ``Z`` and the treatment: then the
    treatment affects the outcome and ``Z`` is a valid adjustment set for the effect. Rule R2 finds
    a set ``Z`` that makes the treatment and the outcome independent, or a covariate ``w`` and a
    set ``Z`` of the others such that ``w`` and the treatment are dependent given ``Z`` while ``w``
    and the outcome are independent given ``Z``: then the treatment does not affect the outcome
    [1]_. The method checks every covariate and every set of the others, so the number of tests
    doubles with each covariate unless ``max_depth`` caps the sets.

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
    alpha : float, default=0.05
        Significance level of every test, in ``(0, 1)``: a p-value above ``alpha`` is an
        independence, any other a dependence. [1]_ leaves the level open.
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
        tests and can leave undecided a pair that a larger conditioning set would decide. R1's
        second test conditions on the treatment as well, so R1 checks sets of at most
        ``max_depth - 1`` covariates and R2 sets of at most ``max_depth``. ``None``, the default,
        checks every set, as [1]_ does for ten covariates; for several tens of covariates [1]_
        caps the sets.

    Returns
    -------
    StructureOutput
        Wraps a ``dag`` :class:`~andrey.core.GraphStructure` over every column with no edges: the
        rules decide whether the treatment is an ancestor of the outcome, not adjacencies.
        ``metadata["effects"]`` holds one record (:mod:`andrey.core.effects`) for the pair, by
        column label or 0-based position, with ``estimand`` ``"total"``. When only R1 applies, the
        treatment affects the outcome: ``nonzero`` and ``identifiable`` are ``True`` and
        ``adjustment_sets`` holds every set ``Z`` for which R1 applies, each a valid adjustment
        set, with ``set_kind`` ``"valid"``. When only R2 applies, the treatment does not affect
        the outcome: ``nonzero`` is ``False``, ``identifiable`` is ``None``, and
        ``adjustment_sets`` is ``[]``. When neither or both apply, the answer is undecided:
        ``nonzero`` and ``identifiable`` are ``None`` and ``adjustment_sets`` is ``[]``.
        ``metadata["rule_counts"]`` counts the rules: ``"R1"`` and ``"R2"`` the candidates each
        applied to, ``"R1_checked"`` and ``"R2_checked"`` the candidates each was checked on; a
        candidate is a covariate and a set of the others, or for R2's first test a set alone.
        ``metadata["algorithm"]`` is ``"covariate_rules"``. Under a mixed-data setting,
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
    **When to use.** The covariate selection rules suit observational data where you want to
    know whether one variable affects another, and which measured covariates to adjust for to
    estimate the effect, when every covariate was measured before the treatment and hidden
    variables may confound the pair. Their claim [1]_: when every independence follows from the
    causal graph (faithfulness), the graph has no cycle and no selection bias, the outcome does
    not cause the treatment, and neither the treatment nor the outcome causes any covariate,
    every answer the rules give is right with exact tests (Theorem 2), and when neither rule
    applies, the independences among the observed variables cannot tell whether the treatment
    affects the outcome (Theorem 3). The default Fisher-Z test assumes continuous data with
    linear effects and Gaussian noise; the other tests read categorical or mixed data or detect
    nonlinear effects, as :func:`~andrey.pc` describes. When some covariates may come after the
    treatment, use :func:`~andrey.ldp`.

    **How it works.** For each set ``Z`` of covariates, by increasing size, the method tests the
    outcome against each other covariate and the treatment given ``Z``, the treatment against
    each other covariate given ``Z``, and the outcome against each other covariate given ``Z``
    and the treatment, and counts every covariate and set to which a rule applies. The answer
    follows from the two counts: an effect when only R1 applies, no effect when only R2 applies,
    and undecided otherwise, which with exact tests is the answer of [1]_, where the two rules
    never both apply. On finite samples [1]_ decides between the counts with two thresholds and a
    classifier trained on simulations it does not publish, and reports an undecided answer when
    the effects estimated with the R1 sets differ; the method uses one level, ``alpha``, and
    returns the sets, so estimating and comparing the effects is left to you. The answer and the
    counts do not depend on the column order.

    **Missing values.** A ``NaN`` in ``data`` marks a missing value. Each test reads only the rows
    where all of its variables are observed (test-wise deletion) [2]_ [3]_, so it keeps rows that
    deleting every incomplete row first would drop; ``metadata["rows_tested"]`` gives the fewest and
    the median rows a test read. When values are missing completely at random, the rows a test keeps
    are a random sample, so each test keeps its level [3]_. When whether a value is missing depends
    on the variables, the rows a test keeps are a selected sample of their own. A test raises
    ``ValueError`` when it has fewer rows than it needs, ``|S| + 3`` for ``"fisherz"`` and
    ``"student_t"`` and 2 for the other tests, or when a continuous column is constant on its rows.

    **Limitations.** Not benchmarked yet. Tested with exact d-separation on random graphs with
    hidden pretreatment covariates, and on simulated linear-Gaussian data. With many candidates
    and one level, each rule applies by chance to some of them, so on finite samples both rules
    often apply and the answer is undecided. With ten covariates a fit runs 16,384 tests, about
    half a second with 10,000 rows, and each added covariate doubles the count. The numbers are a
    reference, not a hard limit: they change with the data generator, the density, the method's
    settings, and the machine.

    References
    ----------
    .. [1] Entner, D., Hoyer, P. O., and Spirtes, P. (2013). Data-driven covariate selection for
       nonparametric estimation of causal effects. Proceedings of the Sixteenth International
       Conference on Artificial Intelligence and Statistics, Proceedings of Machine Learning
       Research 31, 256-264.
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
    >>> from andrey import covariate_rules
    >>> rng = np.random.default_rng(0)
    >>> a, c = rng.standard_normal((2, 5000))
    >>> x = a + c + rng.standard_normal(5000)
    >>> y = x + c + rng.standard_normal(5000)
    >>> df = pd.DataFrame({"a": a, "c": c, "x": x, "y": y})
    >>> out = covariate_rules(df, treatment="x", outcome="y")
    >>> record = out.metadata["effects"][0]
    >>> record["nonzero"], record["adjustment_sets"]
    (True, [['c']])
    """
    from andrey.api._adapt import dag_from_adjacency
    from andrey.core.effects import effect_record

    assert indep_test is not None  # method_spec picks it from the column types
    result, (x, y, d, labels, name), metadata = _fit(
        "covariate_rules",
        data,
        treatment,
        outcome,
        alpha,
        indep_test,
        degrees_of_freedom,
        categorical,
        seed,
        max_depth,
    )
    effect = {"+-": True, "0": False, "?": None}[result.verdict]
    record = effect_record(
        x,
        y,
        estimand="total",
        nonzero=effect,
        identifiable=True if effect else None,
        adjustment_sets=result.sets,
        set_kind="valid" if result.sets else None,
        name=name,
    )
    adjacency = np.zeros((d, d), dtype=np.int8)
    return structure_output(
        dag_from_adjacency(adjacency), labels=labels, metadata={"effects": [record], **metadata}
    )
