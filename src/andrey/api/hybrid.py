"""Hybrid facade adapters: a restrict phase of conditional-independence tests, then a score search.

``mmhc`` and ``rsmax2`` share :mod:`andrey.search.hybrid`: the restrict phase learns the candidate
pairs and the maximize phase runs :func:`andrey.hc` or :func:`andrey.tabu` among them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from andrey.api._adapt import (
    MAX_CATEGORIES,
    categorical_metadata,
    checked_data,
    column_labels,
    copies_named,
    data_kind,
    int_in_range,
    json_safe,
    structure_output,
)
from andrey.api._method import method_spec
from andrey.core.knowledge import Knowledge, resolve

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from andrey.core import StructureOutput


def _fit(
    name: str,
    data: npt.ArrayLike,
    *,
    restrict: str,
    maximize: str,
    alpha: float,
    indep_test: str,
    score_func: str,
    lambda_value: float,
    ebic_gamma: float,
    max_iter: int | None,
    tabu_length: int,
    max_tabu: int | None,
    categorical: Sequence[str | int] | None,
    max_categories: int,
    seed: int | None,
    knowledge: Knowledge | None,
) -> StructureOutput:
    """Check the settings, run the restrict and maximize phases, and wrap the CPDAG."""
    from andrey.search.hc import HC_SCORES
    from andrey.search.hybrid import rsmax2 as _rsmax2
    from andrey.search.tabu import TABU_SCORES

    scores = HC_SCORES if maximize == "hc" else TABU_SCORES
    if score_func not in scores:
        accepted = ", ".join(repr(score) for score in scores)
        raise ValueError(
            f"{name}: score_func must be one of {accepted} with maximize={maximize!r}, got "
            f"{score_func!r}"
        )
    # Both phases read the same data, so the score and the test must agree on its type.
    if data_kind(score_func) != data_kind(indep_test):
        raise ValueError(
            f"{name}'s score and test must read the data alike: score_func='local_score_BDeu' or "
            "'local_score_BIC_categorical' with indep_test 'chisq' or 'gsq' for categories, "
            "'local_score_degen_gauss' or 'local_score_cond_gauss' with 'degen_gauss' or "
            "'cond_gauss' for mixed data, or 'local_score_BIC_from_cov' with a test of continuous "
            f"data; got score_func={score_func!r}, indep_test={indep_test!r}"
        )
    X, mask = checked_data(
        data, categorical, "indep_test", indep_test, max_categories=max_categories, method=name
    )
    max_tabu = int_in_range(
        tabu_length if max_tabu is None else max_tabu, "max_tabu", 1, tabu_length
    )
    labels = column_labels(data)
    constraints = resolve(knowledge, X.shape[1], labels, method=name)
    with copies_named(labels):
        cpdag, score, _ = _rsmax2(
            X,
            restrict=restrict,
            maximize=maximize,
            alpha=alpha,
            indep_test=indep_test,
            score_func=score_func,
            lambda_value=lambda_value,
            ebic_gamma=ebic_gamma,
            max_iter=max_iter,
            tabu_length=tabu_length,
            max_tabu=max_tabu,
            categorical=mask,
            constraints=constraints,
            seed=seed,
        )
    metadata = {
        "score": json_safe(score),
        "restrict": restrict,
        "maximize": maximize,
        **categorical_metadata(mask, data),
    }
    return structure_output(cpdag, labels=labels, metadata=metadata)


@method_spec("mmhc")
def mmhc(
    data: npt.ArrayLike,
    *,
    alpha: float = 0.05,
    indep_test: str = "fisherz",
    score_func: str = "local_score_BIC_from_cov",
    lambda_value: float = 1.0,
    ebic_gamma: float = 1.0,
    max_iter: int | None = None,
    categorical: Sequence[str | int] | None = None,
    max_categories: int = MAX_CATEGORIES,
    seed: int | None = None,
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn a CPDAG from observational data with max-min hill climbing (MMHC).

    MMHC [1]_ first learns a skeleton with the max-min parents and children search (MMPC), which
    uses conditional-independence tests, then runs hill climbing over DAGs whose edges join only
    the pairs of that skeleton. It is :func:`~andrey.rsmax2` with ``restrict="mmpc"`` and
    ``maximize="hc"``.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``.
    alpha : float, default=0.05
        Significance level of the conditional-independence test. Smaller values give sparser
        skeletons, so the score search considers fewer pairs.
    indep_test : str, default="fisherz"
        Conditional-independence test of the MMPC phase, any of :func:`~andrey.pc`'s: ``"fisherz"``
        or ``"student_t"`` for continuous data, ``"chisq"`` or ``"gsq"`` for categorical data,
        ``"degen_gauss"`` or ``"cond_gauss"`` for mixed data, or a nonlinear test. It must read the
        same kind of data as ``score_func``.
    score_func : str, default="local_score_BIC_from_cov"
        Local score of the hill-climbing phase, any of :func:`~andrey.hc`'s:
        ``"local_score_BIC_from_cov"`` or its other name ``"local_score_BIC"`` (Gaussian BIC) for
        continuous data, ``"local_score_BDeu"`` or ``"local_score_BIC_categorical"`` for
        categorical data, ``"local_score_degen_gauss"`` or ``"local_score_cond_gauss"`` for mixed
        data.
    lambda_value : float, default=1.0
        Multiplier on the BIC penalty, as for :func:`~andrey.hc`. BDeu ignores it.
    ebic_gamma : float, default=1.0
        Weight on the extended-BIC term of the linear-Gaussian BIC: each variable's score adds
        ``2 * ebic_gamma * log C(n_variables - 1, n_parents)``, which grows with the number of
        parent sets a variable can choose from. Larger values give sparser graphs; ``0.0`` is the
        standard BIC. Must be finite and non-negative. The other scores ignore it. See Notes.
    max_iter : int or None, default=None
        Maximum hill-climbing moves, at least ``1``; ``None`` sets no limit.
    categorical : sequence of str or int, optional
        The categorical columns of ``data``, for a mixed-data test and score, as for
        :func:`~andrey.pc`. The other settings take no column types and reject it.
    max_categories : int, default=10
        The most distinct whole numbers a column may hold to be inferred as categorical, at least
        ``2``, as for :func:`~andrey.pc`.
    seed : int or None, default=None
        Seed for the random draws of ``indep_test``, which only ``"rcot"``, ``"cmiknn"``, and
        ``"gpdc"`` make; the other tests ignore it. ``None`` uses the ``ANDREY_SEED`` environment
        variable, or ``0`` when it is unset. ``metadata["seed"]`` records the seed used.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the pairs
        allowed to be adjacent, and temporal tiers, all honored. The score search considers the
        pairs of the MMPC skeleton and of the required edges, less the pairs the knowledge makes
        non-adjacent, and then honors the knowledge as :func:`~andrey.hc` does. The MMPC phase
        reads the data alone.

    Returns
    -------
    StructureOutput
        Wraps a ``cpdag`` :class:`~andrey.core.GraphStructure`; ``metadata["algorithm"]`` is
        ``"mmhc"``, ``metadata["score"]`` the score of the DAG found, computed as for
        :func:`andrey.ges` (lower is better), and ``metadata["restrict"]`` and
        ``metadata["maximize"]`` are ``"mmpc"`` and ``"hc"``. Under a mixed-data setting,
        ``metadata["categorical"]`` lists the categorical columns it read.
        ``metadata["seed"]`` is the seed used.

    Raises
    ------
    TypeError
        If ``categorical`` is not a list of column names and positions, or ``knowledge`` is neither
        ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If an argument has the wrong type, lies outside its range, or is not one of its choices; the
        message names the method, the argument, what it accepts, and the value. Also if
        ``score_func`` and ``indep_test`` read different kinds of data, or if ``data`` or
        ``categorical`` does not suit the test or score, as for :func:`~andrey.pc` and
        :func:`~andrey.hc`. Also if ``knowledge`` names a column ``data`` lacks or contradicts
        itself, as :class:`~andrey.Knowledge` lists.

    Warns
    -----
    ExperimentalWarning
        On the first call in a process: the method has no published benchmark, and its API may
        change without deprecation.
    SearchLimitWarning
        When the search takes a given ``max_iter`` moves.
    DataTypeWarning
        As for :func:`~andrey.pc`, when a mixed-data setting infers the categorical columns or a
        continuous setting meets columns that look like category codes; once per process.

    See Also
    --------
    andrey.rsmax2 : Other restrict and maximize phases.
    andrey.mmpc : The MMPC phase alone, for one variable or all.

    Notes
    -----
    **How it works.** The MMPC phase estimates each variable's parents and children: a forward
    phase adds, one at a time, the candidate whose weakest association with the variable, over
    every conditioning set drawn from the members so far, is the strongest, and a backward phase
    drops each member some subset of the others separates (:func:`~andrey.mmpc`). A pair enters
    the skeleton when each variable is in the other's set, the symmetry correction of [1]_. The
    hill-climbing phase starts from the empty graph, or from the required edges, and adds,
    removes, or reverses one edge at a time while the score improves, with every pair outside the
    skeleton forbidden in both directions. The result is the CPDAG of the DAG it stops at.

    Ties between candidates in the MMPC phase and between moves in the hill-climbing phase go to
    the column whose rank, computed from its own data, comes first, as for :func:`~andrey.hc`, so
    the result does not depend on the column order unless two columns are identical. The
    :doc:`column-order guide </docs/guides/column-order>` lists what reordering the columns does to
    each method.

    **Choosing the penalty.** The hill-climbing phase scores as :func:`~andrey.hc` does: the
    linear-Gaussian BIC charges a variable ``lambda_value * log(n_samples)`` per parent plus the
    extended-BIC term ``2 * ebic_gamma * log C(n_variables - 1, n_parents)`` [2]_, and
    :func:`~andrey.ges` describes the choice. ``ebic_gamma=0.0`` with ``lambda_value=1.0`` is the
    standard BIC. On simulated linear-Gaussian data with 20 to 400 variables and 200 to 4,000 rows,
    the default ``ebic_gamma=1.0`` made fewer or as many errors (SHD) as the standard BIC on average
    in 64 of 65 settings of graph, variables, and rows, and 0.2 more in the last (7.0 against 6.8
    with 20 variables, mean degree 2, ``scale="unitless"``, and 1,000 rows); with 200 variables and
    2,000 rows it made 49 errors against 73 (200 true edges). The gain is smaller than hill
    climbing's, since the MMPC phase has already left out most false edges.

    **Missing values.** A ``NaN`` in ``data`` raises ``ValueError``: its hill-climbing phase scores
    every variable over the same rows, and score terms over different rows do not add up to the
    score of one dataset. Drop the incomplete rows first, which suits values missing completely at
    random, or use :func:`~andrey.pc` or :func:`~andrey.fci`, whose tests delete rows per test.

    **Limitations.** Not benchmarked yet. Its tests run it on 20 variables with 2,000 rows of
    simulated linear-Gaussian data. An edge the MMPC phase misses is never added, and the
    hill-climbing phase stops at a local optimum. MMPC tests each candidate against every subset
    of the current members, so its cost grows exponentially with the size of a variable's
    parents-and-children set.

    References
    ----------
    .. [1] Tsamardinos, I., Brown, L. E., and Aliferis, C. F. (2006). The max-min hill-climbing
       Bayesian network structure learning algorithm. Machine Learning 65(1), 31-78.
    .. [2] Chen, J., and Chen, Z. (2008). Extended Bayesian information criteria for model
       selection with large model spaces. Biometrika 95(3), 759-771.

    Examples
    --------
    >>> import warnings
    >>> import numpy as np
    >>> from andrey import mmhc
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal(500)
    >>> y = x + 0.1 * rng.standard_normal(500)
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     out = mmhc(np.column_stack([x, y]))
    >>> out.structure.kind
    'cpdag'
    """
    return _fit(
        "mmhc",
        data,
        restrict="mmpc",
        maximize="hc",
        alpha=alpha,
        indep_test=indep_test,
        score_func=score_func,
        lambda_value=lambda_value,
        ebic_gamma=ebic_gamma,
        max_iter=max_iter,
        tabu_length=10,
        max_tabu=None,
        categorical=categorical,
        max_categories=max_categories,
        seed=seed,
        knowledge=knowledge,
    )


@method_spec("rsmax2")
def rsmax2(
    data: npt.ArrayLike,
    *,
    restrict: str = "mmpc",
    maximize: str = "hc",
    alpha: float = 0.05,
    indep_test: str = "fisherz",
    score_func: str = "local_score_BIC_from_cov",
    lambda_value: float = 1.0,
    ebic_gamma: float = 1.0,
    max_iter: int | None = None,
    tabu_length: int = 10,
    max_tabu: int | None = None,
    categorical: Sequence[str | int] | None = None,
    max_categories: int = MAX_CATEGORIES,
    seed: int | None = None,
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn a CPDAG from observational data with a restrict phase and a maximize phase.

    The restrict phase learns the pairs that may be adjacent with conditional-independence tests;
    the maximize phase runs a score search over DAGs whose edges join only those pairs [1]_ [5]_.
    A small candidate set makes the score search faster on many variables.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``.
    restrict : {"mmpc", "iamb", "inter_iamb", "pc"}, default="mmpc"
        The restrict phase. ``"mmpc"`` keeps a pair when each variable is in the other's MMPC
        parents and children [2]_, as :func:`~andrey.mmhc` does. ``"iamb"`` and ``"inter_iamb"``
        keep a pair when each variable is in the other's Markov blanket from IAMB or Inter-IAMB
        [3]_, the moral graph, which also joins the parents of a common child. ``"pc"`` keeps the
        skeleton of :func:`~andrey.pc`'s adjacency search [4]_. See Notes.
    maximize : {"hc", "tabu"}, default="hc"
        The maximize phase: :func:`~andrey.hc` or :func:`~andrey.tabu`.
    alpha : float, default=0.05
        Significance level of the restrict phase's tests. Smaller values keep fewer pairs.
    indep_test : str, default="fisherz"
        Conditional-independence test of the restrict phase, any of :func:`~andrey.pc`'s. It must
        read the same kind of data as ``score_func``.
    score_func : str, default="local_score_BIC_from_cov"
        Local score of the maximize phase, any of :func:`~andrey.hc`'s; with ``maximize="tabu"``,
        any of :func:`~andrey.tabu`'s, which take no mixed data.
    lambda_value : float, default=1.0
        Multiplier on the BIC penalty, as for :func:`~andrey.hc`. BDeu ignores it.
    ebic_gamma : float, default=1.0
        Weight on the extended-BIC term of the linear-Gaussian BIC: each variable's score adds
        ``2 * ebic_gamma * log C(n_variables - 1, n_parents)``, which grows with the number of
        parent sets a variable can choose from. Larger values give sparser graphs; ``0.0`` is the
        standard BIC. Must be finite and non-negative. The other scores ignore it. See Notes.
    max_iter : int or None, default=None
        Maximum moves of the maximize phase, at least ``1``; ``None`` sets no limit.
    tabu_length : int, default=10
        With ``maximize="tabu"``, the number of most recently visited DAGs a move may not return
        to, at least ``1``, as for :func:`~andrey.tabu`; hill climbing ignores it.
    max_tabu : int or None, default=None
        With ``maximize="tabu"``, the non-improving moves allowed without a better DAG, from ``1``
        to ``tabu_length``; ``None`` uses ``tabu_length``. Hill climbing ignores it.
    categorical : sequence of str or int, optional
        The categorical columns of ``data``, for a mixed-data test and score, as for
        :func:`~andrey.pc`. The other settings take no column types and reject it.
    max_categories : int, default=10
        The most distinct whole numbers a column may hold to be inferred as categorical, at least
        ``2``, as for :func:`~andrey.pc`.
    seed : int or None, default=None
        Seed for the random draws of ``indep_test``, which only ``"rcot"``, ``"cmiknn"``, and
        ``"gpdc"`` make; the other tests ignore it. ``None`` uses the ``ANDREY_SEED`` environment
        variable, or ``0`` when it is unset. ``metadata["seed"]`` records the seed used.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the pairs
        allowed to be adjacent, and temporal tiers, all honored. The maximize phase considers the
        pairs the restrict phase keeps and the pairs of the required edges, less the pairs the
        knowledge makes non-adjacent, and then honors the knowledge as :func:`~andrey.hc` does.
        The restrict phase reads the data alone.

    Returns
    -------
    StructureOutput
        Wraps a ``cpdag`` :class:`~andrey.core.GraphStructure`; ``metadata["algorithm"]`` is
        ``"rsmax2"``, ``metadata["score"]`` the score of the DAG found, computed as for
        :func:`andrey.ges` (lower is better), and ``metadata["restrict"]`` and
        ``metadata["maximize"]`` name the phases. Under a mixed-data setting,
        ``metadata["categorical"]`` lists the categorical columns it read.
        ``metadata["seed"]`` is the seed used.

    Raises
    ------
    TypeError
        If ``categorical`` is not a list of column names and positions, or ``knowledge`` is neither
        ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If an argument has the wrong type, lies outside its range, or is not one of its choices; the
        message names the method, the argument, what it accepts, and the value. Also if
        ``score_func`` is not a score ``maximize`` takes, if ``score_func`` and ``indep_test`` read
        different kinds of data, if ``max_tabu`` is greater than ``tabu_length``, or if ``data`` or
        ``categorical`` does not suit the test or score, as for :func:`~andrey.pc` and
        :func:`~andrey.hc`. Also if ``knowledge`` names a column ``data`` lacks or contradicts
        itself, as :class:`~andrey.Knowledge` lists.

    Warns
    -----
    ExperimentalWarning
        On the first call in a process: the method has no published benchmark, and its API may
        change without deprecation.
    SearchLimitWarning
        When the maximize phase takes a given ``max_iter`` moves.
    PerformanceWarning
        With ``restrict="pc"``, when a conditioning pass needs at least ten million tests, as for
        :func:`~andrey.pc`.
    DataTypeWarning
        As for :func:`~andrey.pc`, when a mixed-data setting infers the categorical columns or a
        continuous setting meets columns that look like category codes; once per process.

    See Also
    --------
    andrey.mmhc : ``restrict="mmpc"`` with ``maximize="hc"``.

    Notes
    -----
    **How it works.** The restrict phase follows the sparse candidate idea of [1]_: tests narrow
    the pairs a score search must consider. Each choice finds every true adjacency when its tests
    are exact. The MMPC skeleton and PC's skeleton then hold only the true adjacencies; the moral
    graph also joins two parents of a common child, which the maximize phase can leave
    non-adjacent. The maximize phase starts from the empty graph, or from the required edges, with
    every pair outside the candidates forbidden in both directions, and returns the CPDAG of the
    DAG it ends with. :func:`~andrey.mmpc`, :func:`~andrey.iamb`, and :func:`~andrey.inter_iamb`
    describe their searches; ``"pc"`` runs :func:`~andrey.pc`'s adjacency search without
    orienting.

    Ties go to the column whose rank, computed from its own data, comes first, as for
    :func:`~andrey.hc`, so the result does not depend on the column order unless two columns are
    identical.

    **Choosing the penalty.** The maximize phase scores as :func:`~andrey.hc` does: the
    linear-Gaussian BIC charges a variable ``lambda_value * log(n_samples)`` per parent plus the
    extended-BIC term ``2 * ebic_gamma * log C(n_variables - 1, n_parents)`` [6]_, and
    :func:`~andrey.ges` describes the choice. ``ebic_gamma=0.0`` with ``lambda_value=1.0`` is the
    standard BIC. With ``restrict="pc"`` and ``maximize="tabu"``, on simulated linear-Gaussian data
    with 20 to 400 variables and 200 to 4,000 rows, the default ``ebic_gamma=1.0`` made fewer or as
    many errors (SHD) as the standard BIC on average in 63 of 65 settings of graph, variables, and
    rows, and more in two: 38.0 against 36.3 on the 46-variable ECOLI70 network with 1,000
    simulated rows, where one of three draws moved by 7, and 39.2 against 38.8 with 50 variables,
    mean degree 4, ``scale="unitless"``, and 1,000 rows. With 200 variables and 2,000 rows it made
    42 errors against 65 (200 true edges). The default ``restrict="mmpc"`` with ``maximize="hc"``
    is :func:`~andrey.mmhc`'s search, whose numbers are given there.

    **Missing values.** A ``NaN`` in ``data`` raises ``ValueError``: its score phase scores every
    variable over the same rows, and score terms over different rows do not add up to the score of
    one dataset. Drop the incomplete rows first, which suits values missing completely at random, or
    use :func:`~andrey.pc` or :func:`~andrey.fci`, whose tests delete rows per test.

    **Limitations.** Not benchmarked yet. Its tests run it on 20 variables with 2,000 rows of
    simulated linear-Gaussian data. An edge the restrict phase misses is never added, and the
    maximize phase stops at a local optimum. MMPC tests each candidate against every subset of
    the current members, so its cost grows exponentially with the size of a variable's
    parents-and-children set. IAMB and Inter-IAMB condition on the whole current blanket, so each
    of their tests needs more rows as the blankets grow.

    References
    ----------
    .. [1] Friedman, N., Nachman, I., and Pe'er, D. (1999). Learning Bayesian network structure
       from massive datasets: the "sparse candidate" algorithm. UAI, 206-215.
    .. [2] Tsamardinos, I., Brown, L. E., and Aliferis, C. F. (2006). The max-min hill-climbing
       Bayesian network structure learning algorithm. Machine Learning 65(1), 31-78.
    .. [3] Tsamardinos, I., Aliferis, C. F., and Statnikov, A. (2003). Algorithms for large scale
       Markov blanket discovery. FLAIRS, 376-380.
    .. [4] Colombo, D., and Maathuis, M. H. (2014). Order-independent constraint-based causal
       structure learning. JMLR 15, 3921-3962.
    .. [5] Scutari, M. (2010). Learning Bayesian networks with the bnlearn R package. Journal of
       Statistical Software 35(3). https://doi.org/10.18637/jss.v035.i03
    .. [6] Chen, J., and Chen, Z. (2008). Extended Bayesian information criteria for model
       selection with large model spaces. Biometrika 95(3), 759-771.

    Examples
    --------
    >>> import warnings
    >>> import numpy as np
    >>> from andrey import rsmax2
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal(500)
    >>> y = x + 0.1 * rng.standard_normal(500)
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     out = rsmax2(np.column_stack([x, y]), restrict="pc", maximize="tabu")
    >>> out.metadata["restrict"], out.metadata["maximize"]
    ('pc', 'tabu')
    """
    return _fit(
        "rsmax2",
        data,
        restrict=restrict,
        maximize=maximize,
        alpha=alpha,
        indep_test=indep_test,
        score_func=score_func,
        lambda_value=lambda_value,
        ebic_gamma=ebic_gamma,
        max_iter=max_iter,
        tabu_length=tabu_length,
        max_tabu=max_tabu,
        categorical=categorical,
        max_categories=max_categories,
        seed=seed,
        knowledge=knowledge,
    )
