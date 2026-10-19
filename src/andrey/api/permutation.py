"""Permutation-based facade adapters.

``boss`` and ``grasp`` are native -- they run a permutation/score search on the :mod:`andrey.core`
compute substrate and return a CPDAG ``GraphStructure`` with its objective score. Both score
candidate parent additions through the batched Schur delta-BIC surface over the linear-Gaussian BIC
(:class:`andrey.core.score.BICScore`), or by full local scores over BDeu or a mixed-data BIC. Their
search order is drawn from a private RNG seeded by ``seed`` (or the global ``ANDREY_SEED``), so a
run is reproducible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from andrey.api._adapt import (
    MAX_CATEGORIES,
    categorical_metadata,
    checked_data,
    column_labels,
    copies_named,
    int_in_range,
    json_safe,
    structure_output,
)
from andrey.core.seeding import resolve_seed

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from andrey.core import StructureOutput


def boss(
    data: npt.ArrayLike,
    *,
    score_func: str = "local_score_BIC_from_cov",
    lambda_value: float = 1.0,
    seed: int | None = None,
    categorical: Sequence[str | int] | None = None,
    max_categories: int = MAX_CATEGORIES,
) -> StructureOutput:
    """Learn a CPDAG with BOSS (Best Order Score Search).

    BOSS searches variable orders (permutations) using linear-Gaussian BIC (Bayesian information
    criterion). It converts the best-scoring order to a CPDAG, which represents an equivalence class
    of graphs. It is fast and accurate on linear-Gaussian data.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``.
    score_func : str, default="local_score_BIC_from_cov"
        Local score objective, by default ``"local_score_BIC_from_cov"``, the covariance-based
        linear-Gaussian BIC; ``"local_score_BDeu"`` is the BDeu score for categorical data coded by
        whole numbers, and ``"local_score_degen_gauss"`` and ``"local_score_cond_gauss"`` are the
        degenerate and conditional Gaussian BICs for mixed data, as for :func:`~andrey.ges`.
    lambda_value : float, default=1.0
        Multiplier on the BIC penalty of ``log(n_samples)`` per parameter: per parent for the
        linear-Gaussian BIC. Larger values give sparser graphs; ``1.0`` is the standard BIC. Must
        be finite and non-negative. BDeu ignores it.
    seed : int or None, default=None
        Seed for the random order in which the search visits variables. ``None`` uses the
        ``ANDREY_SEED`` environment variable, or ``0`` when it is unset, so repeated calls return
        the same graph. Different seeds can reach different graphs.
    categorical : sequence of str or int, optional
        The categorical columns of ``data``, for a mixed-data score: DataFrame column names or
        0-based positions. ``None`` infers them: a DataFrame's ``category``, ``bool``, and text
        columns, and every column of at most ``max_categories`` distinct whole numbers, named in a
        ``DataTypeWarning``. ``[]`` marks none. The other scores take no column types and reject it.
        See *Data types* in the :doc:`getting-started guide </docs/guides/getting-started>`.
    max_categories : int, default=10
        The most distinct whole numbers a column may hold to be inferred as categorical, at least
        ``2``. With a continuous score, columns within it get a one-time ``DataTypeWarning`` that
        they may be category codes.

    Returns
    -------
    StructureOutput
        Wraps a ``cpdag`` :class:`~andrey.core.GraphStructure`; ``metadata["algorithm"]`` is
        ``"BOSS"`` and ``metadata["score"]`` the score of the DAG read off the final order; for the
        BIC, the sum over variables of ``n * log(residual variance) + lambda_value * log(n) *
        n_parents``, and for BDeu, ``-2`` times the log of the structure prior times the marginal
        likelihood; lower is better. Directed marks (``->``) are oriented edges, undirected
        (``--``) the ones the equivalence class leaves open. Under a mixed-data setting,
        ``metadata["categorical"]`` lists the categorical columns it read, by name or 0-based
        position.

    Raises
    ------
    NotImplementedError
        If ``score_func`` names no available score.
    TypeError
        If ``categorical`` is not a list of column names and positions, or ``max_categories`` is not
        an int.
    ValueError
        If ``lambda_value`` is negative or non-finite, or if ``data`` or ``categorical`` does not
        suit the score, as for :func:`~andrey.ges`. With the BIC score, also if a column repeats
        another up to scale and shift given more than two rows or, with more rows than columns, is a
        linear function of other columns. One message names every group of copies and says to keep
        one column per group.

    Warns
    -----
    DataTypeWarning
        When a mixed-data score infers the categorical columns, or finds none and so runs as the
        linear-Gaussian form, or when a continuous setting meets columns that look like category
        codes; each message warns once per process.

    See Also
    --------
    andrey.grasp : Greedy relaxation of the sparsest permutation algorithm [2]_, searching the
        same space of orders with different moves.
    andrey.ges : Searches over equivalence classes for the same kind of data.
    andrey.fci : Allows unmeasured common causes and selection bias.
    andrey.direct_lingam : Orients every edge when the noise is non-Gaussian.

    Notes
    -----
    **When to use it.** BOSS suits the same data as :func:`~andrey.ges`: continuous data with linear
    effects, Gaussian noise, and no unmeasured common causes. It was designed for hundreds of highly
    connected variables [1]_. For categorical data, use ``"local_score_BDeu"``, and for mixed data
    ``"local_score_degen_gauss"`` or ``"local_score_cond_gauss"``.

    **When not to.** When unmeasured common causes are plausible, use :func:`~andrey.fci`. When the
    noise is non-Gaussian and every edge should be oriented, use :func:`~andrey.direct_lingam`.

    **Choosing seed.** The search visits variables in a random order and stops when no move improves
    the order. Different seeds can therefore reach different graphs; comparing a few shows whether
    a result depends on the seed.

    **How it works.** BOSS searches variable orders for the best DAG with the lowest total score
    [1]_. Each variable uses a grow-shrink tree to cache its best parent set for any set of
    variables allowed to precede it. Each sweep moves every variable to the position that most
    improves the total score. Sweeps repeat until no move improves it. The final parent sets form a
    DAG, which is reduced to its CPDAG.

    A move or parent change must improve the score by more than ``1e-6``, a fixed absolute tolerance
    shared with :func:`~andrey.grasp`. Scores within this tolerance of the best count as ties. The
    earliest order position wins; among candidate parents, the column with the lowest tie rank
    wins.

    **Column order.** Each column gets a tie rank computed from its own data, as for
    :func:`~andrey.hc`, and the search runs on the columns sorted by it: it starts from that order,
    and the seed shuffles the order in which it visits them. The
    :doc:`column-order guide </docs/guides/column-order>` says what the rank keeps under a new
    column order and on another machine.

    **Categorical data.** ``"local_score_BDeu"`` reads every column as categorical, each distinct
    value one category, and scores a variable given its parents with the BDeu score of
    :func:`~andrey.ges`: a Dirichlet prior with an equivalent sample size of 1 and a structure prior
    of one expected parent per variable, on the scale of the BIC. Each candidate parent set is
    scored by full local scores.

    **Mixed data.** ``"local_score_degen_gauss"`` [3]_ and ``"local_score_cond_gauss"`` [4]_ score
    data with continuous and categorical columns, as :func:`~andrey.ges` describes, and read the
    categorical columns as :func:`~andrey.pc` does. Each candidate parent set is scored by full
    local scores, without the Schur update the BIC search uses.

    **Limitations.** Measured on simulated linear-Gaussian data from sparse random graphs (mean
    degree 2), from 20 to 800 variables with ten rows per variable, at ``lambda_value=4.0``: its
    median errors (SHD) stayed below the empty graph's at every size, 24 against 800 at 800
    variables. The median fit took 0.031 s at 20 variables and 292 s at 800, on 16 cores. On the
    223-variable ANDES network with 1,000 simulated rows, at ``lambda_value=1.0``, its median errors
    passed the empty graph's (506 against 338). With fewer rows than columns the BIC score is
    unreliable. A parent set of ``n_samples - 1`` or more variables fits its child exactly, so the
    search never selects one, nor any parent set that leaves its child a residual variance of at
    most ``1e-12`` times its variance. Against :func:`~andrey.exact_search` with BDeu on simulated
    categorical networks of 4 to 8 variables, BOSS stopped at a DAG that scores worse than the
    optimum on some draws; comparing a few seeds shows whether a result depends on the seed. BDeu is
    not benchmarked with BOSS yet. The mixed-data scores are experimental and not benchmarked yet;
    their tests run BOSS on 5 variables with 3,000 rows.

    References
    ----------
    .. [1] Andrews, B., Ramsey, J., Sanchez Romero, R., Camchong, J., and Kummerfeld, E. (2023).
       Fast Scalable and Accurate Discovery of DAGs Using the Best Order Score Search and Grow
       Shrink Trees. NeurIPS 36. https://doi.org/10.52202/075280-2794
    .. [2] Lam, W.-Y., Andrews, B., and Ramsey, J. (2022). Greedy Relaxations of the Sparsest
       Permutation Algorithm. UAI, PMLR 180:1052-1062. https://proceedings.mlr.press/v180/lam22a.html
    .. [3] Andrews, B., Ramsey, J., and Cooper, G. F. (2019). Learning high-dimensional directed
       acyclic graphs with mixed data-types. Proceedings of Machine Learning Research 104, 4-21.
    .. [4] Andrews, B., Ramsey, J., and Cooper, G. F. (2018). Scoring Bayesian networks of mixed
       variables. International Journal of Data Science and Analytics 6(1), 3-18.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from andrey import boss
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal(500)
    >>> y = rng.standard_normal(500)
    >>> z = x + y + 0.5 * rng.standard_normal(500)
    >>> boss(pd.DataFrame({"x": x, "y": y, "z": z})).structure.oriented_edges()
    [('x', 'z', 'directed'), ('y', 'z', 'directed')]
    """
    from andrey.search.boss import PERMUTATION_SCORES
    from andrey.search.boss import boss as _boss

    if score_func not in PERMUTATION_SCORES:
        raise NotImplementedError(f"BOSS supports {PERMUTATION_SCORES}, not {score_func!r}")
    X, mask = checked_data(
        data, categorical, "score_func", score_func, max_categories=max_categories
    )
    labels = column_labels(data)
    with copies_named(labels):
        cpdag, score = _boss(
            X,
            lambda_value=lambda_value,
            random_state=resolve_seed(seed),
            score_func=score_func,
            categorical=mask,
        )
    metadata = {"algorithm": "BOSS", "score": json_safe(score), **categorical_metadata(mask, data)}
    return structure_output(cpdag, labels=labels, metadata=metadata)


def grasp(
    data: npt.ArrayLike,
    *,
    score_func: str = "local_score_BIC_from_cov",
    lambda_value: float = 1.0,
    depth: int = 3,
    seed: int | None = None,
    categorical: Sequence[str | int] | None = None,
    max_categories: int = MAX_CATEGORIES,
) -> StructureOutput:
    """Learn a CPDAG with GRaSP (Greedy Relaxation of the Sparsest Permutation).

    GRaSP searches variable orders (permutations) using linear-Gaussian BIC (Bayesian information
    criterion). It converts the best-scoring order to a CPDAG, which represents an equivalence class
    of graphs. It searches the same space as :func:`~andrey.boss` using chains of up to ``depth``
    tucks [1]_, the moves described in Notes.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``.
    score_func : str, default="local_score_BIC_from_cov"
        Local score objective, by default ``"local_score_BIC_from_cov"``, the covariance-based
        linear-Gaussian BIC; ``"local_score_BDeu"`` is the BDeu score for categorical data coded by
        whole numbers, and ``"local_score_degen_gauss"`` and ``"local_score_cond_gauss"`` are the
        degenerate and conditional Gaussian BICs for mixed data, as for :func:`~andrey.ges`.
    lambda_value : float, default=1.0
        Multiplier on the BIC penalty of ``log(n_samples)`` per parameter: per parent for the
        linear-Gaussian BIC. Larger values give sparser graphs; ``1.0`` is the standard BIC. Must
        be finite and non-negative. BDeu ignores it.
    depth : int, default=3
        Maximum number of tucks in one chain of moves, at least ``1``. The depth-first search
        follows a chain of tucks that leave the score unchanged until one improves it or the chain
        reaches ``depth`` tucks. ``1`` tries single tucks only; larger depths search more chains,
        which can reach a better score, and run longer.
    seed : int or None, default=None
        Seed for the random starting order and the order in which the search visits variables.
        ``None`` uses the ``ANDREY_SEED`` environment variable, or ``0`` when it is unset, so
        repeated calls return the same graph. Different seeds can reach different graphs.
    categorical : sequence of str or int, optional
        The categorical columns of ``data``, for a mixed-data score: DataFrame column names or
        0-based positions. ``None`` infers them: a DataFrame's ``category``, ``bool``, and text
        columns, and every column of at most ``max_categories`` distinct whole numbers, named in a
        ``DataTypeWarning``. ``[]`` marks none. The other scores take no column types and reject it.
        See *Data types* in the :doc:`getting-started guide </docs/guides/getting-started>`.
    max_categories : int, default=10
        The most distinct whole numbers a column may hold to be inferred as categorical, at least
        ``2``. With a continuous score, columns within it get a one-time ``DataTypeWarning`` that
        they may be category codes.

    Returns
    -------
    StructureOutput
        Wraps a ``cpdag`` :class:`~andrey.core.GraphStructure`; ``metadata["algorithm"]`` is
        ``"GRaSP"`` and ``metadata["score"]`` the score of the DAG read off the final order; for the
        BIC, the sum over variables of ``n * log(residual variance) + lambda_value * log(n) *
        n_parents``, and for BDeu, ``-2`` times the log of the structure prior times the marginal
        likelihood; lower is better. Directed marks (``->``) are oriented edges, undirected
        (``--``) the ones the equivalence class leaves open. Under a mixed-data setting,
        ``metadata["categorical"]`` lists the categorical columns it read, by name or 0-based
        position.

    Raises
    ------
    NotImplementedError
        If ``score_func`` names no available score.
    TypeError
        If ``depth`` is not an int, ``categorical`` is not a list of column names and positions, or
        ``max_categories`` is not an int.
    ValueError
        If ``depth`` is less than ``1``, ``lambda_value`` is negative or non-finite, or if ``data``
        or ``categorical`` does not suit the score, as for :func:`~andrey.ges`. With the BIC score,
        also if a column repeats another up to scale and shift given more than two rows or, with
        more rows than columns, is a linear function of other columns. One message names every
        group of copies and says to keep one column per group.

    Warns
    -----
    DataTypeWarning
        When a mixed-data score infers the categorical columns, or finds none and so runs as the
        linear-Gaussian form, or when a continuous setting meets columns that look like category
        codes; each message warns once per process.

    See Also
    --------
    andrey.boss : Searches the same space of orders with different moves; faster on the published
        benchmarks at 400 variables.
    andrey.ges : Searches over equivalence classes for the same kind of data.
    andrey.fci : Allows unmeasured common causes and selection bias.
    andrey.direct_lingam : Orients every edge when the noise is non-Gaussian.

    Notes
    -----
    **When to use it.** GRaSP suits the same data as :func:`~andrey.ges`: continuous data with
    linear effects, Gaussian noise, and no unmeasured common causes. Its authors report accurate
    search even on dense graphs [1]_. For categorical data, use ``"local_score_BDeu"``, and for
    mixed data ``"local_score_degen_gauss"`` or ``"local_score_cond_gauss"``.

    **When not to.** When unmeasured common causes are plausible, use :func:`~andrey.fci`. When the
    noise is non-Gaussian and every edge should be oriented, use :func:`~andrey.direct_lingam`.

    **GRaSP or BOSS.** Both search the same space of orders with different moves. Either can reach
    the better score on a given dataset. On the published benchmarks, GRaSP took far longer than
    :func:`~andrey.boss` at 400 variables.

    **Choosing seed.** Both the starting order and the order of visiting variables are random, so
    different seeds can reach different graphs; comparing a few shows whether a result depends on
    the seed.

    **How it works.** GRaSP searches variable orders for the best DAG with the lowest total score
    [1]_. Each variable uses a grow-shrink tree [2]_ to cache its best parent set for any set of
    variables allowed to precede it. The search starts from a shuffled order and changes it with
    tucks. A tuck on an edge ``x -> y`` moves ``y`` just before ``x``, along with its ancestors that
    sit between ``x`` and ``y``. A depth-first search keeps the first chain of up to ``depth`` tucks
    that improves the total score. It repeats until no chain improves it. The final parent sets form
    a DAG, which is reduced to its CPDAG.

    Tucks must improve the score by more than ``1e-6``, the fixed absolute tolerance used by
    :func:`~andrey.boss`. A tuck within this tolerance counts as leaving the score unchanged.

    **Column order.** Each column gets a tie rank computed from its own data, as for
    :func:`~andrey.hc`, and the search runs on the columns sorted by it: the seed shuffles the
    sorted columns into the starting order and the order of visiting variables, and ties between
    candidate parents go to the lowest rank. The
    :doc:`column-order guide </docs/guides/column-order>` says what the rank keeps under a new
    column order and on another machine.

    **Categorical data.** ``"local_score_BDeu"`` reads every column as categorical, each distinct
    value one category, and scores a variable given its parents with the BDeu score of
    :func:`~andrey.ges`: a Dirichlet prior with an equivalent sample size of 1 and a structure prior
    of one expected parent per variable, on the scale of the BIC. Each candidate parent set is
    scored by full local scores.

    **Mixed data.** ``"local_score_degen_gauss"`` [3]_ and ``"local_score_cond_gauss"`` [4]_ score
    data with continuous and categorical columns, as :func:`~andrey.ges` describes, and read the
    categorical columns as :func:`~andrey.pc` does. Each candidate parent set is scored by full
    local scores, without the Schur update the BIC search uses.

    **Limitations.** Measured on simulated linear-Gaussian data from sparse random graphs (mean
    degree 2), from 20 to 400 variables with ten rows per variable, at ``lambda_value=4.0``: its
    median errors (SHD) stayed below the empty graph's at every size, 0 against 400 at 400
    variables. The median fit took 0.037 s at 20 variables, 40 s at 200, and 1,040 s at 400, on 16
    cores; larger graphs were not measured. On the 223-variable ANDES network with 1,000 simulated
    rows, at ``lambda_value=1.0``, its median errors passed the empty graph's (471 against 338).
    With fewer rows than columns the BIC score is unreliable. A parent set of ``n_samples - 1`` or
    more variables fits its child exactly, so the search never selects one, nor any parent set that
    leaves its child a residual variance of at most ``1e-12`` times its variance. Against
    :func:`~andrey.exact_search` with BDeu on simulated categorical networks of 4 to 8 variables,
    GRaSP stopped at a DAG that scores worse than the optimum on some draws; comparing a few seeds
    shows whether a result depends on the seed. BDeu is not benchmarked with GRaSP yet. The
    mixed-data scores are experimental and not benchmarked yet; their tests run GRaSP on 5 variables
    with 3,000 rows.

    References
    ----------
    .. [1] Lam, W.-Y., Andrews, B., and Ramsey, J. (2022). Greedy Relaxations of the Sparsest
       Permutation Algorithm. UAI, PMLR 180:1052-1062. https://proceedings.mlr.press/v180/lam22a.html
    .. [2] Andrews, B., Ramsey, J., Sanchez Romero, R., Camchong, J., and Kummerfeld, E. (2023).
       Fast Scalable and Accurate Discovery of DAGs Using the Best Order Score Search and Grow
       Shrink Trees. NeurIPS 36. https://doi.org/10.52202/075280-2794
    .. [3] Andrews, B., Ramsey, J., and Cooper, G. F. (2019). Learning high-dimensional directed
       acyclic graphs with mixed data-types. Proceedings of Machine Learning Research 104, 4-21.
    .. [4] Andrews, B., Ramsey, J., and Cooper, G. F. (2018). Scoring Bayesian networks of mixed
       variables. International Journal of Data Science and Analytics 6(1), 3-18.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from andrey import grasp
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal(500)
    >>> y = rng.standard_normal(500)
    >>> z = x + y + 0.5 * rng.standard_normal(500)
    >>> grasp(pd.DataFrame({"x": x, "y": y, "z": z})).structure.oriented_edges()
    [('x', 'z', 'directed'), ('y', 'z', 'directed')]
    """
    from andrey.search.boss import PERMUTATION_SCORES
    from andrey.search.grasp import grasp as _grasp

    if score_func not in PERMUTATION_SCORES:
        raise NotImplementedError(f"GRaSP supports {PERMUTATION_SCORES}, not {score_func!r}")
    depth = int_in_range(depth, "depth", 1)
    X, mask = checked_data(
        data, categorical, "score_func", score_func, max_categories=max_categories
    )
    labels = column_labels(data)
    with copies_named(labels):
        cpdag, score = _grasp(
            X,
            lambda_value=lambda_value,
            depth=depth,
            random_state=resolve_seed(seed),
            score_func=score_func,
            categorical=mask,
        )
    metadata = {"algorithm": "GRaSP", "score": json_safe(score), **categorical_metadata(mask, data)}
    return structure_output(cpdag, labels=labels, metadata=metadata)
