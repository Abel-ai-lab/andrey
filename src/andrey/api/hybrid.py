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
    open_unit_interval,
    structure_output,
)
from andrey.api._experimental import warn_experimental
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
    max_iter: int | None,
    tabu_length: int,
    max_tabu: int | None,
    categorical: Sequence[str | int] | None,
    max_categories: int,
    knowledge: Knowledge | None,
) -> StructureOutput:
    """Check the settings, run the restrict and maximize phases, and wrap the CPDAG."""
    from andrey.search.hc import HC_SCORES
    from andrey.search.hybrid import MAXIMIZES, RESTRICTS
    from andrey.search.hybrid import rsmax2 as _rsmax2
    from andrey.search.tabu import TABU_SCORES

    if restrict not in RESTRICTS:
        raise ValueError(f"restrict must be one of {RESTRICTS}, got {restrict!r}")
    if maximize not in MAXIMIZES:
        raise ValueError(f"maximize must be one of {MAXIMIZES}, got {maximize!r}")
    scores = HC_SCORES if maximize == "hc" else TABU_SCORES
    if score_func not in scores:
        raise NotImplementedError(
            f"{name} with maximize={maximize!r} supports score_func {scores}, not {score_func!r}"
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
        data, categorical, "indep_test", indep_test, max_categories=max_categories
    )
    alpha = open_unit_interval(alpha, "alpha")
    max_iter = None if max_iter is None else int_in_range(max_iter, "max_iter", 1)
    tabu_length = int_in_range(tabu_length, "tabu_length", 1)
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
            max_iter=max_iter,
            tabu_length=tabu_length,
            max_tabu=max_tabu,
            categorical=mask,
            constraints=constraints,
        )
    metadata = {
        "algorithm": name.upper(),
        "score": json_safe(score),
        "restrict": restrict,
        "maximize": maximize,
        **categorical_metadata(mask, data),
    }
    return structure_output(cpdag, labels=labels, metadata=metadata)


def mmhc(
    data: npt.ArrayLike,
    *,
    alpha: float = 0.05,
    indep_test: str = "fisherz",
    score_func: str = "local_score_BIC_from_cov",
    lambda_value: float = 1.0,
    max_iter: int | None = None,
    categorical: Sequence[str | int] | None = None,
    max_categories: int = MAX_CATEGORIES,
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
        ``"local_score_BIC_from_cov"`` (Gaussian BIC) for continuous data, ``"local_score_BDeu"``
        or ``"local_score_BIC_categorical"`` for categorical data, ``"local_score_degen_gauss"`` or
        ``"local_score_cond_gauss"`` for mixed data.
    lambda_value : float, default=1.0
        Multiplier on the BIC penalty, as for :func:`~andrey.hc`. BDeu ignores it.
    max_iter : int or None, default=None
        Maximum hill-climbing moves, at least ``1``; ``None`` sets no limit.
    categorical : sequence of str or int, optional
        The categorical columns of ``data``, for a mixed-data test and score, as for
        :func:`~andrey.pc`. The other settings take no column types and reject it.
    max_categories : int, default=10
        The most distinct whole numbers a column may hold to be inferred as categorical, at least
        ``2``, as for :func:`~andrey.pc`.
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
        ``"MMHC"``, ``metadata["score"]`` the score of the DAG found, computed as for
        :func:`andrey.ges` (lower is better), and ``metadata["restrict"]`` and
        ``metadata["maximize"]`` are ``"mmpc"`` and ``"hc"``. Under a mixed-data setting,
        ``metadata["categorical"]`` lists the categorical columns it read.

    Raises
    ------
    NotImplementedError
        If ``indep_test`` or ``score_func`` names no available test or score.
    TypeError
        If a given ``max_iter`` is not an int, ``categorical`` is not a list of column names and
        positions, or ``max_categories`` is not an int. Also if ``knowledge`` is neither ``None``
        nor a :class:`~andrey.Knowledge`.
    ValueError
        If ``alpha`` is not in ``(0, 1)``, if ``score_func`` and ``indep_test`` read different
        kinds of data, if a given ``max_iter`` is less than ``1``, or if ``data`` or
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

    **Limitations.** Not benchmarked yet. Its tests run it on 20 variables with 2,000 rows of
    simulated linear-Gaussian data. An edge the MMPC phase misses is never added, and the
    hill-climbing phase stops at a local optimum. MMPC tests each candidate against every subset
    of the current members, so its cost grows exponentially with the size of a variable's
    parents-and-children set.

    References
    ----------
    .. [1] Tsamardinos, I., Brown, L. E., and Aliferis, C. F. (2006). The max-min hill-climbing
       Bayesian network structure learning algorithm. Machine Learning 65(1), 31-78.

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
    warn_experimental("mmhc")
    return _fit(
        "mmhc",
        data,
        restrict="mmpc",
        maximize="hc",
        alpha=alpha,
        indep_test=indep_test,
        score_func=score_func,
        lambda_value=lambda_value,
        max_iter=max_iter,
        tabu_length=10,
        max_tabu=None,
        categorical=categorical,
        max_categories=max_categories,
        knowledge=knowledge,
    )
