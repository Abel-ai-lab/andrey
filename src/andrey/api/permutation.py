"""Permutation-based facade adapters.

``boss`` and ``grasp`` are native -- they run a permutation/score search on the :mod:`andrey.core`
compute substrate and return a CPDAG ``GraphStructure`` with its objective score. Both score
candidate parent additions through the batched Schur delta-BIC surface over the linear-Gaussian BIC
(:class:`andrey.core.score.BICScore`), or by full local scores over BDeu, the multinomial BIC, or a
mixed-data BIC. Their search order is drawn from a private RNG seeded by ``seed`` (or the global
``ANDREY_SEED``), so a run is reproducible. ``igsp`` and ``ut_igsp`` search variable orders over
minimal I-MAPs with interventional data (:mod:`andrey.search.igsp`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from andrey.api._adapt import (
    categorical_metadata,
    checked_data,
    column_labels,
    copies_named,
    default_score,
    default_test,
    json_safe,
    label_of,
    structure_output,
)
from andrey.api._method import method_spec
from andrey.core.knowledge import Knowledge, resolve

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy.typing as npt

    from andrey.core import StructureOutput


@method_spec("boss")
def boss(
    data: npt.ArrayLike,
    *,
    score_func: str | None = None,
    penalty_discount: float = 1.0,
    ebic_gamma: float = 1.0,
    equivalent_sample_size: float = 1.0,
    seed: int | None = None,
    n_starts: int = 1,
    max_parents: int | None = None,
    categorical: Sequence[str | int] | None = None,
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn a CPDAG from observational data with BOSS (Best Order Score Search).

    BOSS searches variable orders (permutations) using linear-Gaussian BIC (Bayesian information
    criterion). It converts the best-scoring order to a CPDAG, which represents an equivalence class
    of graphs. It is fast and accurate on linear-Gaussian data.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``.
    score_func : str or None, default=None
        Local score objective: ``"bic"``, the covariance-based linear-Gaussian BIC; ``"bdeu"`` and
        ``"discrete_bic"`` are the BDeu score and the multinomial BIC for categorical data coded by
        whole numbers, and ``"degen_gauss"`` and ``"cond_gauss"`` are the degenerate and conditional
        Gaussian BICs for mixed data, as for :func:`~andrey.ges`. ``None`` picks from the declared
        column types: ``"bdeu"`` when every column is categorical, ``"cond_gauss"`` when some are,
        else ``"bic"``. A column is declared categorical by ``categorical`` or by a DataFrame's
        ``category``, ``bool``, or text type, never by its values.
        ``metadata["score_func"]`` names the score the fit used: the one picked from the
        declared column types when this is ``None``.
    penalty_discount : float, default=1.0
        Multiplier on the BIC penalty of ``log(n_samples)`` per parameter: per parent for the
        linear-Gaussian BIC, per free parameter of the conditional probability table for the
        multinomial BIC. Larger values give sparser graphs; ``1.0`` with ``ebic_gamma=0.0`` is the
        standard BIC. Must be finite and non-negative. BDeu ignores it.
    ebic_gamma : float, default=1.0
        Weight on the extended-BIC term of the linear-Gaussian BIC: each variable's score adds
        ``2 * ebic_gamma * log C(n_variables - 1, n_parents)``, which grows with the number of
        parent sets a variable can choose from. Larger values give sparser graphs; ``0.0`` is the
        standard BIC. Must be from 0 to 1. Only ``"bic"`` uses it: passing it with another score
        raises ``ValueError``. See Notes.
    equivalent_sample_size : float, default=1.0
        The equivalent sample size of the BDeu score: how many rows the uniform prior over each
        variable's conditional distributions is worth. A larger value weighs the prior more
        against the data, and the graph can change with it. Must be finite and positive. Only
        ``"bdeu"`` uses it: passing it with another score raises ``ValueError``.
    seed : int or None, default=None
        ``None``, the default, is the configured seed, ``ANDREY_SEED`` or ``0`` when it is unset,
        not a fresh random one, so a fit repeats. Seed for the random order in which the search
        visits variables. Different seeds can reach different graphs.
    n_starts : int, default=1
        Number of searches, at least ``1``. The first starts from the order described in Notes;
        each later one starts from a random order drawn from ``seed`` and its own index. The result
        is the best-scoring search; a later one replaces an earlier one only when it scores better
        by more than ``1e-6``. Each search builds its own caches and drops the previous one's, so
        more starts take more time but little more memory. See Notes.
    max_parents : int or None, default=None
        A cap on every node's parents, at least ``0``: the grow step of each variable's parent
        search adds no parent to a set that holds ``max_parents`` members, so no DAG the returned
        CPDAG represents gives a node more than ``max_parents`` parents (every DAG of a class has
        the same largest number of parents). A cap can stop the search short of the lowest-scoring
        graph. ``None`` sets no cap.
    categorical : sequence of str or int, or None, default=None
        The categorical columns of ``data``, for a mixed-data score: column labels (strings) or
        0-based positions. ``None`` takes a DataFrame's ``category``, ``bool``, and text columns,
        never a column by its values. ``[]`` marks none. A categorical score reads every column as
        categorical, and a continuous score raises ``ValueError`` on a declared column, named here
        or by its type. See *Data types* in the
        :doc:`getting-started guide </docs/guides/getting-started>`.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the pairs
        allowed to be adjacent, and temporal tiers, all honored. Each variable's parents exclude the
        forbidden ones and include the required ones, and every causal order the search visits puts
        the cause of each required edge, and each earlier tier, first. The result directs every edge
        the knowledge decides, then Meek's rules direct the edges that follow, as
        :class:`~andrey.Knowledge` describes.

    Returns
    -------
    StructureOutput
        Wraps a ``cpdag`` :class:`~andrey.core.GraphStructure`; ``metadata["algorithm"]`` is
        ``"boss"``, ``metadata["seed"]`` the seed used, and ``metadata["score"]`` the score of the
        DAG read off the final order; for the BIC, the sum over variables of ``n * log(residual
        variance) + penalty_discount * log(n) * n_parents + 2 * ebic_gamma * log C(n_variables - 1,
        n_parents)``, for BDeu, ``-2`` times the log of the structure prior times the marginal
        likelihood, and for the multinomial BIC, ``-2`` times the log-likelihood plus the penalty;
        lower is better. Directed marks (``->``) are oriented edges, undirected (``--``) the ones
        the equivalence class leaves open. Under a mixed-data setting, ``metadata["categorical"]``
        lists the categorical columns it read, by name or 0-based position.
        ``metadata["tie_break"]`` is the tie-break rule the fit ran under,
        ``andrey.config.tie_break``, and ``metadata`` also records the penalties its score used,
        under their parameter names.

    Raises
    ------
    TypeError
        If an argument has the wrong type, such as a string for a number; the message names the
        method, the argument, what it accepts, and the value. Also if ``categorical`` is not a list
        of column names and positions, or ``knowledge`` is neither ``None`` nor a
        :class:`~andrey.Knowledge`.
    ValueError
        If an argument lies outside its range or is not one of its choices; the message names the
        method, the argument, what it accepts, and the value. Also if the required edges of
        ``knowledge`` give a node more than ``max_parents`` parents; if ``data`` or ``categorical``
        does not suit the score, as for :func:`~andrey.ges`. With the BIC score, also if a column
        repeats another up to scale and shift given more than two rows or, with more rows than
        columns, is a linear function of other columns. One message names every group of copies and
        says to keep one column per group. Also if ``knowledge`` names a column ``data`` lacks or
        contradicts itself, as :class:`~andrey.Knowledge` lists.

    Warns
    -----
    DataTypeWarning
        When a mixed-data score finds no declared categorical column and so runs as the
        linear-Gaussian form, or when a setting reads columns that look like category codes as
        numbers. Each message warns once per process.
    BoundaryWarning
        When the data are larger than the largest size whose fits finished within the time and
        memory caps of the published benchmark, at the measured setting nearest the data's rows, and
        fits of the next size ran past a cap. It warns once per process and names that size and the
        release card; the size is a reference, not a hard limit.

    Notes
    -----
    **When to use.** BOSS suits the same data as :func:`~andrey.ges`: i.i.d. observational
    continuous data with linear effects, Gaussian noise, and no unmeasured common causes. When every
    column is categorical, the default score is ``"bdeu"``, on mixed data ``"cond_gauss"``, and
    ``"discrete_bic"`` and ``"degen_gauss"`` also score those data. It was designed for hundreds of
    highly connected variables [1]_; it returns a CPDAG. When unmeasured common causes are
    plausible, use :func:`~andrey.fci`; when the noise is non-Gaussian and every edge should be
    oriented, use :func:`~andrey.direct_lingam`.

    **Choosing seed.** The search visits variables in a random order and stops when no move improves
    the order. Different seeds can therefore reach different graphs; comparing a few shows whether
    a result depends on the seed.

    **Choosing n_starts.** The search can stop at an order that scores worse than the true graph,
    most often on dense graphs. Its authors run one search [1]_; ``n_starts`` runs later searches
    from random orders and keeps the best score. On simulated linear-Gaussian data with 100
    variables at mean degree 4 and 1,000 rows, at the defaults, 8 starts lowered the mean errors
    (SHD) over 20 datasets from 7.9 to 2.1 and took a median of 7.7 times as long as one start.

    **How it works.** BOSS searches variable orders for the best DAG with the lowest total score
    [1]_. Each variable uses a grow-shrink tree to cache its best parent set for any set of
    variables allowed to precede it. Each sweep moves every variable to the position that most
    improves the total score. Sweeps repeat until no move improves it. The final parent sets form a
    DAG, which is reduced to its CPDAG.

    A move or parent change must improve the score by more than ``1e-6``, a fixed absolute tolerance
    shared with :func:`~andrey.grasp` [2]_. Scores within this tolerance of the best count as ties.
    The earliest order position wins; among candidate parents, the column with the lowest tie rank
    wins. Under the linear-Gaussian BIC each comparison is decided as full local scores decide it: a
    fast update whose rounding-error bound could change the comparison is scored in full.

    **Column order.** Each column gets a tie rank computed from the data, as for
    :func:`~andrey.hc`, and the search runs on the columns sorted by it: it starts from that order,
    and the seed shuffles the order in which it visits them. The
    :doc:`column-order guide </docs/guides/column-order>` says what the rank keeps under a new
    column order and on another machine.

    **Categorical data.** ``"bdeu"`` reads every column as categorical, each distinct
    value one category, and scores a variable given its parents with the BDeu score of
    :func:`~andrey.ges`: a Dirichlet prior with an equivalent sample size of 1 and a structure prior
    of one expected parent per variable, on the scale of the BIC. ``"discrete_bic"``
    reads the columns the same way and scores a variable with the multinomial BIC of
    :func:`~andrey.ges`: ``-2`` times the log-likelihood of its counts in each configuration of its
    parents, plus ``penalty_discount * log(n)`` for each free parameter. Under either score each
    column gets its tie rank as a categorical column, and each candidate parent set is scored by
    full local scores.

    **Mixed data.** ``"degen_gauss"`` [3]_ and ``"cond_gauss"`` [4]_ score
    data with continuous and categorical columns, as :func:`~andrey.ges` describes, and read the
    categorical columns as :func:`~andrey.pc` does. Each candidate parent set is scored by full
    local scores, without the Schur update the BIC search uses.

    **Choosing the penalty.** The linear-Gaussian BIC charges a variable ``penalty_discount *
    log(n_samples)`` per parent plus the extended-BIC term ``2 * ebic_gamma * log C(n_variables - 1,
    n_parents)`` [5]_, which grows with the number of variables; :func:`~andrey.ges` describes the
    choice. ``ebic_gamma=0.0`` with ``penalty_discount=1.0`` is the standard BIC. On simulated
    linear-Gaussian data with 20 to 400 variables and 200 to 4,000 rows, at ``seed=0``, the default
    ``ebic_gamma=1.0`` made fewer errors (SHD) than the standard BIC on average in 64 of 65 settings
    of graph, variables, and rows. In the last, 20 variables with mean degree 4 on
    ``scale="unitless"`` data and 1,000 rows, it made 9.6 errors against 8.0, within the spread of
    the five datasets (-3 to +5 each). With 200 variables and 2,000 rows it made 5 errors against
    174 (200 true edges). Its mean errors stayed below the empty graph's in each of the 65 settings,
    where the standard BIC's exceeded it in 22.

    **Missing values.** A ``NaN`` in ``data`` raises ``ValueError``. The score adds one term per
    variable, each over every row, and terms over different rows do not add up to the score of one
    dataset, so the search cannot delete rows per test as the conditional-independence searches do.
    Drop the incomplete rows first, which suits values missing completely at random, or use
    :func:`~andrey.pc` or :func:`~andrey.fci`, whose tests delete rows per test.

    **Limitations.** The benchmark numbers in this paragraph are for ``ebic_gamma=0.0``. Measured on
    simulated linear-Gaussian data from Erdos-Renyi graphs, with ``penalty_discount=4.0``, on 4
    cores with a cap of 1 h and 28 GB per fit. The largest sizes through which its median errors
    (SHD) stayed below the empty graph's, and through which its fits finished within the caps, were
    1,600 and 1,600 variables at mean degree 1 with 2,000 rows and mean degree 2 with 500 or 2,000
    rows; 800 and 800 at mean degree 2 with 10,000 rows and mean degree 4 with 2,000 rows; and none
    and 400 at mean degree 8 with 2,000 rows (below it only from 50 through 400). At 1,600 variables
    (mean degree 2, 2,000 rows), 1 of 5 fits ran past 1 h. The ladders ended when three fits of the
    next size ran past 1 h, except at mean degree 8 with 2,000 rows, where the ladder ended when
    three fits of the next size ran out of the 28 GB. The median fit took 3,214 s at 1,600 variables
    (mean degree 2, 2,000 rows). On the 223-variable ANDES network with 1,000 simulated rows, at
    ``penalty_discount=1.0``, its median errors passed the empty graph's (506 against 338). With
    fewer rows than columns the standard BIC is unreliable. A parent set of ``n_samples - 1`` or
    more variables fits its child exactly, so the search never selects one, nor any parent set that
    leaves its child a residual variance of at most ``1e-12`` times its variance. Against
    :func:`~andrey.exact_search` on small simulated categorical networks, with BDeu or the
    multinomial BIC, BOSS stopped at a DAG that scores worse than the optimum on some draws;
    comparing a few seeds shows whether a result depends on the seed. Neither categorical score is
    benchmarked with BOSS yet. The mixed-data scores are experimental and not benchmarked yet; their
    tests run BOSS on 5 variables with 3,000 rows. The numbers are a reference, not a hard limit:
    they change with the data generator, the density, the method's settings, and the machine.

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
    .. [5] Chen, J., and Chen, Z. (2008). Extended Bayesian information criteria for model
       selection with large model spaces. Biometrika 95(3), 759-771.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from andrey import boss
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal(500)
    >>> y = rng.standard_normal(500)
    >>> z = x + y + 0.5 * rng.standard_normal(500)
    >>> boss(pd.DataFrame({"x": x, "y": y, "z": z}), seed=0).structure.oriented_edges()
    [('x', 'z', 'directed'), ('y', 'z', 'directed')]
    """
    from andrey.search.boss import boss as _boss

    score_func, categorical = default_score(
        data,
        score_func,
        categorical=categorical,
    )
    X, mask = checked_data(data, categorical, "score_func", score_func, method="boss")
    labels = column_labels(data)
    constraints = resolve(knowledge, X.shape[1], labels, method="boss")
    with copies_named(labels):
        cpdag, score = _boss(
            X,
            penalty_discount=penalty_discount,
            ebic_gamma=ebic_gamma,
            equivalent_sample_size=equivalent_sample_size,
            random_state=seed,  # ty: ignore[invalid-argument-type]  # an int: method_spec resolves it
            n_starts=n_starts,
            score_func=score_func,
            categorical=mask,
            constraints=constraints,
            max_parents=max_parents,
        )
    metadata = {"score": json_safe(score), **categorical_metadata(mask, data)}
    return structure_output(cpdag, labels=labels, metadata=metadata)


@method_spec("grasp")
def grasp(
    data: npt.ArrayLike,
    *,
    score_func: str | None = None,
    penalty_discount: float = 1.0,
    ebic_gamma: float = 1.0,
    equivalent_sample_size: float = 1.0,
    grasp_depth: int = 3,
    seed: int | None = None,
    n_starts: int = 1,
    max_parents: int | None = None,
    categorical: Sequence[str | int] | None = None,
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn a CPDAG from observational data with GRaSP.

    GRaSP (Greedy Relaxation of the Sparsest Permutation) searches variable orders (permutations)
    using linear-Gaussian BIC (Bayesian information criterion). It converts the best-scoring order
    to a CPDAG, which represents an equivalence class of graphs. It searches the same space as
    :func:`~andrey.boss` using chains of up to ``grasp_depth`` tucks [1]_, the moves described in
    Notes.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``.
    score_func : str or None, default=None
        Local score objective: ``"bic"``, the covariance-based linear-Gaussian BIC; ``"bdeu"`` and
        ``"discrete_bic"`` are the BDeu score and the multinomial BIC for categorical data coded by
        whole numbers, and ``"degen_gauss"`` and ``"cond_gauss"`` are the degenerate and conditional
        Gaussian BICs for mixed data, as for :func:`~andrey.ges`. ``None`` picks from the declared
        column types: ``"bdeu"`` when every column is categorical, ``"cond_gauss"`` when some are,
        else ``"bic"``. A column is declared categorical by ``categorical`` or by a DataFrame's
        ``category``, ``bool``, or text type, never by its values.
        ``metadata["score_func"]`` names the score the fit used: the one picked from the
        declared column types when this is ``None``.
    penalty_discount : float, default=1.0
        Multiplier on the BIC penalty of ``log(n_samples)`` per parameter: per parent for the
        linear-Gaussian BIC, per free parameter of the conditional probability table for the
        multinomial BIC. Larger values give sparser graphs; ``1.0`` with ``ebic_gamma=0.0`` is the
        standard BIC. Must be finite and non-negative. BDeu ignores it.
    ebic_gamma : float, default=1.0
        Weight on the extended-BIC term of the linear-Gaussian BIC: each variable's score adds
        ``2 * ebic_gamma * log C(n_variables - 1, n_parents)``, which grows with the number of
        parent sets a variable can choose from. Larger values give sparser graphs; ``0.0`` is the
        standard BIC. Must be from 0 to 1. Only ``"bic"`` uses it: passing it with another score
        raises ``ValueError``. See Notes.
    equivalent_sample_size : float, default=1.0
        The equivalent sample size of the BDeu score: how many rows the uniform prior over each
        variable's conditional distributions is worth. A larger value weighs the prior more
        against the data, and the graph can change with it. Must be finite and positive. Only
        ``"bdeu"`` uses it: passing it with another score raises ``ValueError``.
    grasp_depth : int, default=3
        Maximum number of tucks in one chain of moves, at least ``1``. The depth-first search
        follows a chain of tucks that leave the score unchanged until one improves it or the chain
        reaches ``grasp_depth`` tucks. ``1`` tries single tucks only; larger depths search more
        chains, which can reach a better score, and run longer.
    seed : int or None, default=None
        ``None``, the default, is the configured seed, ``ANDREY_SEED`` or ``0`` when it is unset,
        not a fresh random one, so a fit repeats. Seed for the random starting order and the order
        in which the search visits variables. Different seeds can reach different graphs.
    n_starts : int, default=1
        Number of searches, at least ``1``, each from its own random starting order: the first
        from ``seed``'s, each later one from an order drawn from ``seed`` and its own index. The
        result is the best-scoring search; a later one replaces an earlier one only when it scores
        better by more than ``1e-6``. Each search builds its own caches and drops the previous
        one's, so more starts take more time but little more memory. See Notes.
    max_parents : int or None, default=None
        A cap on every node's parents, at least ``0``: the grow step of each variable's parent
        search adds no parent to a set that holds ``max_parents`` members, so no DAG the returned
        CPDAG represents gives a node more than ``max_parents`` parents (every DAG of a class has
        the same largest number of parents). A cap can stop the search short of the lowest-scoring
        graph. ``None`` sets no cap.
    categorical : sequence of str or int, or None, default=None
        The categorical columns of ``data``, for a mixed-data score: column labels (strings) or
        0-based positions. ``None`` takes a DataFrame's ``category``, ``bool``, and text columns,
        never a column by its values. ``[]`` marks none. A categorical score reads every column as
        categorical, and a continuous score raises ``ValueError`` on a declared column, named here
        or by its type. See *Data types* in the
        :doc:`getting-started guide </docs/guides/getting-started>`.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the pairs
        allowed to be adjacent, and temporal tiers, all honored. Each variable's parents exclude the
        forbidden ones and include the required ones, and every causal order the search visits puts
        the cause of each required edge, and each earlier tier, first. The result directs every edge
        the knowledge decides, then Meek's rules direct the edges that follow, as
        :class:`~andrey.Knowledge` describes.

    Returns
    -------
    StructureOutput
        Wraps a ``cpdag`` :class:`~andrey.core.GraphStructure`; ``metadata["algorithm"]`` is
        ``"grasp"``, ``metadata["seed"]`` the seed used, and ``metadata["score"]`` the score of the
        DAG read off the final order; for the BIC, the sum over variables of ``n * log(residual
        variance) + penalty_discount * log(n) * n_parents + 2 * ebic_gamma * log C(n_variables - 1,
        n_parents)``, for BDeu, ``-2`` times the log of the structure prior times the marginal
        likelihood, and for the multinomial BIC, ``-2`` times the log-likelihood plus the penalty;
        lower is better. Directed marks (``->``) are oriented edges, undirected (``--``) the ones
        the equivalence class leaves open. Under a mixed-data setting, ``metadata["categorical"]``
        lists the categorical columns it read, by name or 0-based position.
        ``metadata["tie_break"]`` is the tie-break rule the fit ran under,
        ``andrey.config.tie_break``, and ``metadata`` also records the penalties its score used,
        under their parameter names.

    Raises
    ------
    TypeError
        If an argument has the wrong type, such as a string for a number; the message names the
        method, the argument, what it accepts, and the value. Also if ``categorical`` is not a list
        of column names and positions, or ``knowledge`` is neither ``None`` nor a
        :class:`~andrey.Knowledge`.
    ValueError
        If an argument lies outside its range or is not one of its choices; the message names the
        method, the argument, what it accepts, and the value. Also if the required edges of
        ``knowledge`` give a node more than ``max_parents`` parents; if ``data`` or ``categorical``
        does not suit the score, as for :func:`~andrey.ges`. With the BIC score, also if a column
        repeats another up to scale and shift given more than two rows or, with more rows than
        columns, is a linear function of other columns. One message names every group of copies and
        says to keep one column per group. Also if ``knowledge`` names a column ``data`` lacks or
        contradicts itself, as :class:`~andrey.Knowledge` lists.

    Warns
    -----
    DataTypeWarning
        When a mixed-data score finds no declared categorical column and so runs as the
        linear-Gaussian form, or when a setting reads columns that look like category codes as
        numbers. Each message warns once per process.
    BoundaryWarning
        When the data are larger than the largest size whose fits finished within the time and
        memory caps of the published benchmark, at the measured setting nearest the data's rows, and
        fits of the next size ran past a cap. It warns once per process and names that size and the
        release card; the size is a reference, not a hard limit.

    Notes
    -----
    **When to use.** GRaSP suits the same data as :func:`~andrey.ges`: i.i.d. observational
    continuous data with linear effects, Gaussian noise, and no unmeasured common causes. When every
    column is categorical, the default score is ``"bdeu"``, on mixed data ``"cond_gauss"``, and
    ``"discrete_bic"`` and ``"degen_gauss"`` also score those data. Its authors report accurate
    search even on dense graphs [1]_; it returns a CPDAG. When unmeasured common causes are
    plausible, use :func:`~andrey.fci`; when the noise is non-Gaussian and every edge should be
    oriented, use :func:`~andrey.direct_lingam`.

    **GRaSP or BOSS.** Both search the same space of orders with different moves. Either can reach
    the better score on a given dataset. On the published benchmarks, GRaSP took far longer than
    :func:`~andrey.boss` at 400 variables.

    **Choosing seed.** Both the starting order and the order of visiting variables are random, so
    different seeds can reach different graphs; comparing a few shows whether a result depends on
    the seed.

    **Choosing n_starts.** The search can stop at a DAG that scores worse than the true one, most
    often on dense graphs. Its authors run one search [1]_; ``n_starts`` runs later searches from
    random orders and keeps the best score. On simulated linear-Gaussian data with 100 variables at
    mean degree 4 and 1,000 rows, at the defaults, 8 starts lowered the mean errors (SHD) over 20
    datasets from 18.2 to 4.6 and took a median of 8.3 times as long as one start.

    **How it works.** GRaSP searches variable orders for the best DAG with the lowest total score
    [1]_. Each variable uses a grow-shrink tree [2]_ to cache its best parent set for any set of
    variables allowed to precede it. The search starts from a shuffled order and changes it with
    tucks. A tuck on an edge ``x -> y`` moves ``y`` just before ``x``, along with its ancestors that
    sit between ``x`` and ``y``. A depth-first search tries a tuck on every edge of the current DAG,
    in random order. A tuck that leaves the score unchanged starts a chain, which goes on with tucks
    on covered edges, those where ``y``'s parents are ``x`` and ``x``'s parents, to DAGs the chain
    has not visited, up to ``grasp_depth`` tucks. The search keeps the first tuck or chain that
    improves the total score and repeats until none does. This is the paper's GRaSP2 at the depths
    of its simulations [1]_; the paper runs its lower tiers, GRaSP0 and GRaSP1, first, and GRaSP
    does not. The final parent sets form a DAG, which is reduced to its CPDAG.

    Tucks must improve the score by more than ``1e-6``, the fixed absolute tolerance used by
    :func:`~andrey.boss`. A tuck within this tolerance counts as leaving the score unchanged. Under
    the linear-Gaussian BIC each comparison is decided as full local scores decide it, as for
    :func:`~andrey.boss`.

    **Column order.** Each column gets a tie rank computed from the data, as for
    :func:`~andrey.hc`, and the search runs on the columns sorted by it: the seed shuffles the
    sorted columns into the starting order and the order of visiting variables, and ties between
    candidate parents go to the lowest rank. The
    :doc:`column-order guide </docs/guides/column-order>` says what the rank keeps under a new
    column order and on another machine.

    **Categorical data.** ``"bdeu"`` reads every column as categorical, each distinct
    value one category, and scores a variable given its parents with the BDeu score of
    :func:`~andrey.ges`: a Dirichlet prior with an equivalent sample size of 1 and a structure prior
    of one expected parent per variable, on the scale of the BIC. ``"discrete_bic"``
    reads the columns the same way and scores a variable with the multinomial BIC of
    :func:`~andrey.ges`: ``-2`` times the log-likelihood of its counts in each configuration of its
    parents, plus ``penalty_discount * log(n)`` for each free parameter. Under either score each
    column gets its tie rank as a categorical column, and each candidate parent set is scored by
    full local scores.

    **Mixed data.** ``"degen_gauss"`` [3]_ and ``"cond_gauss"`` [4]_ score
    data with continuous and categorical columns, as :func:`~andrey.ges` describes, and read the
    categorical columns as :func:`~andrey.pc` does. Each candidate parent set is scored by full
    local scores, without the Schur update the BIC search uses.

    **Choosing the penalty.** The linear-Gaussian BIC charges a variable ``penalty_discount *
    log(n_samples)`` per parent plus the extended-BIC term ``2 * ebic_gamma * log C(n_variables - 1,
    n_parents)`` [5]_, which grows with the number of variables; :func:`~andrey.ges` describes the
    choice. ``ebic_gamma=0.0`` with ``penalty_discount=1.0`` is the standard BIC. On simulated
    linear-Gaussian data with 20 to 400 variables and 200 to 4,000 rows, at ``seed=0``, the default
    ``ebic_gamma=1.0`` made fewer errors (SHD) than the standard BIC on average in 63 of 65 settings
    of graph, variables, and rows, and more in two: 5.4 against 5.0 with 20 variables, mean degree
    4, and 1,000 rows, and 33.4 against 26.0 with 50 variables, mean degree 4, and 1,000 rows, where
    the five datasets changed by -50 to +113 errors. With 200 variables and 2,000 rows it made 8
    errors against 179 (200 true edges). Its mean errors stayed below the empty graph's in each of
    the 65 settings, where the standard BIC's exceeded it in 21.

    **Missing values.** A ``NaN`` in ``data`` raises ``ValueError``. The score adds one term per
    variable, each over every row, and terms over different rows do not add up to the score of one
    dataset, so the search cannot delete rows per test as the conditional-independence searches do.
    Drop the incomplete rows first, which suits values missing completely at random, or use
    :func:`~andrey.pc` or :func:`~andrey.fci`, whose tests delete rows per test.

    **Limitations.** The benchmark numbers in this paragraph are for ``ebic_gamma=0.0``. Measured on
    simulated linear-Gaussian data from Erdos-Renyi graphs, with ``penalty_discount=4.0``, on 4
    cores with a cap of 1 h and 28 GB per fit. The largest sizes through which its median errors
    (SHD) stayed below the empty graph's, and through which its fits finished within the caps, were
    800 and 800 variables at mean degree 1, 4, or 8 with 2,000 rows and mean degree 2 with 500,
    2,000, or 10,000 rows. At 800 variables at mean degree 1, 4, or 8 with 2,000 rows, 1 or 2 of 5
    fits ran past 1 h. Every ladder ended when three fits of the next size ran past 1 h. The median
    fit took 3,108 s at 800 variables (mean degree 8, 2,000 rows). At ``penalty_discount=1.0`` on
    denser graphs (mean degree 4) of 50 to 200 variables, it stopped at a DAG that scores worse than
    the true DAG in 26 of 60 fits, one that no chain of up to ``grasp_depth`` tucks improves. On the
    223-variable ANDES network with 1,000 simulated rows, at ``penalty_discount=1.0``, its median
    errors passed the empty graph's (471 against 338). With fewer rows than columns the standard BIC
    is unreliable. A parent set of ``n_samples - 1`` or more variables fits its child exactly, so
    the search never selects one, nor any parent set that leaves its child a residual variance of at
    most ``1e-12`` times its variance. Against :func:`~andrey.exact_search` on small simulated
    categorical networks, with BDeu or the multinomial BIC, GRaSP stopped at a DAG that scores worse
    than the optimum on some draws; comparing a few seeds shows whether a result depends on the
    seed. Neither categorical score is benchmarked with GRaSP yet. The mixed-data scores are
    experimental and not benchmarked yet; their tests run GRaSP on 5 variables with 3,000 rows. The
    numbers are a reference, not a hard limit: they change with the data generator, the density, the
    method's settings, and the machine.

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
    .. [5] Chen, J., and Chen, Z. (2008). Extended Bayesian information criteria for model
       selection with large model spaces. Biometrika 95(3), 759-771.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from andrey import grasp
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal(500)
    >>> y = rng.standard_normal(500)
    >>> z = x + y + 0.5 * rng.standard_normal(500)
    >>> grasp(pd.DataFrame({"x": x, "y": y, "z": z}), seed=0).structure.oriented_edges()
    [('x', 'z', 'directed'), ('y', 'z', 'directed')]
    """
    from andrey.search.grasp import grasp as _grasp

    score_func, categorical = default_score(
        data,
        score_func,
        categorical=categorical,
    )
    X, mask = checked_data(data, categorical, "score_func", score_func, method="grasp")
    labels = column_labels(data)
    constraints = resolve(knowledge, X.shape[1], labels, method="grasp")
    with copies_named(labels):
        cpdag, score = _grasp(
            X,
            penalty_discount=penalty_discount,
            ebic_gamma=ebic_gamma,
            equivalent_sample_size=equivalent_sample_size,
            depth=grasp_depth,
            random_state=seed,  # ty: ignore[invalid-argument-type]  # an int: method_spec resolves it
            n_starts=n_starts,
            score_func=score_func,
            categorical=mask,
            constraints=constraints,
            max_parents=max_parents,
        )
    metadata = {"score": json_safe(score), **categorical_metadata(mask, data)}
    return structure_output(cpdag, labels=labels, metadata=metadata)


@method_spec("igsp")
def igsp(
    data: npt.ArrayLike,
    env: npt.ArrayLike | str | None = None,
    *,
    targets: Mapping | npt.ArrayLike | str | None = None,
    alpha: float | None = None,
    invariance_alpha: float | None = None,
    indep_test: str | None = None,
    invariance_test: str | None = None,
    n_starts: int = 10,
    seed: int | None = None,
    categorical: Sequence[str | int] | None = None,
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn an interventional essential graph with IGSP from settings with known targets.

    IGSP (interventional greedy sparsest permutation) learns from data pooled from an
    observational environment and interventional ones, the settings, whose targets, the variables
    each one intervened on, are known [1]_ [2]_. It searches variable orders for the DAG with the
    fewest edges and, among those, the fewest edges the settings contradict: each order gives the
    DAG of the edges its conditional-independence tests keep on the observational rows, and
    invariance tests between each setting and the observational rows say which directions the
    settings support. It returns the DAG's interventional essential graph under the targets.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables), or a list or mapping of them
        Observed data, one row per sample, from every environment. Coerced to ``float64``. A
        DataFrame's column names become the structure's ``labels``. A column that ``env`` or
        ``targets`` names is not a variable: it leaves the data before the fit, and
        ``categorical``, ``knowledge``, ``targets``, and the result's positions count the other
        columns.
        A list of DataFrames or 2-D arrays gives each environment's rows apart, labeled by
        position ``0, 1, ...``, and a mapping labels each by its key: the rows are stacked in that
        order, ``env`` stays ``None``, and ``targets`` maps those labels to their variables.
    env : array-like of shape (n_samples,) or (n_samples, 1), str, or None, default=None
        The environment of each row: one label per row, strings or numbers, not both, or the name of
        a column of the DataFrame ``data`` that holds them. ``None`` takes the environments from
        ``targets``: its distinct values.
    targets : mapping, str, array-like of shape (n_samples, n_variables), or None, default=None
        The variables intervened on in each environment, named by column label or 0-based position.
        A mapping ``{label: [variables]}`` takes each label of ``env`` to a list of its targets;
        ``[]`` marks an observational environment, and every label of ``env`` needs a list:
        ``None``, or a label the mapping leaves out, raises ``ValueError``, as a key that is not a
        label of ``env`` does. A string names a column of the DataFrame ``data`` whose cells give
        each row's targets: a label or a position, a list of them, or an empty cell (``None``,
        ``NaN``, ``""``, ``[]``) for an observational row. An array is ``True`` (or 1) where the
        row's value of the variable was set by an intervention. With a column or an array, every
        row of one environment of ``env`` must give the same targets; without ``env``, rows with
        the same targets are one environment. At least one environment must be observational and
        one not, and every environment needs at least 3 rows. See Notes.
    alpha : float or None, default=None
        Significance level of the conditional-independence tests, in ``(0, 1)``: two variables
        are independent when a test's p-value exceeds it. Smaller values keep more edges out.
        ``None`` uses one over the number of observational rows, 0.001 at 1,000 rows, so the level
        falls as the rows grow. Also the level of the invariance tests when ``invariance_alpha``
        is ``None``. ``metadata["alpha"]`` gives the level used. See Notes for the choice of the
        default.
    invariance_alpha : float or None, default=None
        Significance level of the invariance tests, in ``(0, 1)``: a variable's distribution
        given a set of others differs between a setting and the observational rows when the test
        gives a p-value at most this level. ``None`` uses the level of the conditional-independence
        tests, ``metadata["alpha"]``.
    indep_test : str or None, default=None
        The conditional-independence test, on the observational rows. For continuous data, on the
        partial correlation: ``"fisherz"``, ``"student_t"``, and ``"normal_scores"``; for
        categorical data coded by whole numbers: ``"chisq"`` and ``"gsq"``; for mixed data, with
        the categorical columns named by ``categorical``: ``"degen_gauss"`` and ``"cond_gauss"``;
        for nonlinear effects in continuous data, each far slower: ``"kci"``, ``"rcot"``,
        ``"cmiknn"``, and ``"gpdc"``, as for :func:`~andrey.pc`. ``None`` picks from the declared
        column types: ``"gsq"`` when every column is categorical, ``"cond_gauss"`` when some are,
        else ``"fisherz"``. ``metadata["indep_test"]`` names the test the fit used.
    invariance_test : {"gaussian", "indicator"} or None, default=None
        The invariance test, which asks whether a variable's distribution given a set of others is
        the same in a setting as in the observational rows. ``"gaussian"`` compares the linear
        regressions of the two groups of rows: their coefficients and intercept by the Chow test
        and their residual variances by an F-test, each at half the level. ``"indicator"`` runs
        ``indep_test`` between the variable and a 0/1 indicator of the setting's rows given the
        others. ``None`` picks ``"gaussian"`` under ``"fisherz"``, ``"student_t"``, and
        ``"normal_scores"``, else ``"indicator"``. ``"gaussian"`` reads every column as numbers,
        so with a categorical or mixed-data ``indep_test`` it raises ``ValueError``.
        ``metadata["invariance_test"]`` names the test the fit used. See Notes.
    n_starts : int, default=10
        Number of searches, at least ``1``, each from a minimum-degree order of the variables:
        the first breaks ties by each column's rank, and each later one draws them, and the order
        in which it visits moves, from ``seed`` and its own index. Returns the best-scoring search,
        the one that found the fewest edges, then the fewest contradicted ones; a later one
        replaces an earlier one only when it is strictly better. See Notes for the choice of the
        default.
    seed : int or None, default=None
        ``None``, the default, is the configured seed, ``ANDREY_SEED`` or ``0`` when it is unset,
        not a fresh random one, so a fit repeats. Seed for the searches after the first, and for
        the random draws of ``"rcot"``, ``"cmiknn"``, and ``"gpdc"``. With ``n_starts=1`` and
        another test, the result does not depend on it.
    categorical : sequence of str or int, or None, default=None
        The categorical columns of ``data``, for a mixed-data test: column labels (strings) or
        0-based positions. ``None`` takes a DataFrame's ``category``, ``bool``, and text columns,
        never a column by its values. ``[]`` marks none. A categorical test reads every column as
        categorical, and a continuous test raises ``ValueError`` on a declared column, named here
        or by its type.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the pairs
        allowed to be adjacent, and temporal tiers, all honored. Every DAG the search visits has
        the required edges and none of the forbidden ones, and every order it visits puts the
        cause of each required edge, and each earlier tier, first. The result directs every edge
        the knowledge decides, then Meek's rules direct the edges that follow, as
        :class:`~andrey.Knowledge` describes.

    Returns
    -------
    StructureOutput
        Wraps a ``cpdag`` :class:`~andrey.core.GraphStructure`, the interventional essential graph
        of the DAG found under the observational environment and each setting's targets: directed
        marks (``->``) are the edges that every DAG of the class orients alike, undirected (``--``)
        the ones it leaves open. ``metadata["algorithm"]`` is ``"igsp"``; ``metadata["score"]`` is
        the DAG's number of edges, lower is better, and ``metadata["n_contradicting"]`` its number
        of I-contradicting edges, the second part of the objective; ``metadata["indep_test"]`` and
        ``metadata["invariance_test"]`` name the tests used, and ``metadata["alpha"]`` is the level
        of the conditional-independence tests; ``metadata["seed"]`` is the seed used when the fit
        draws random numbers. Under a mixed-data test, ``metadata["categorical"]`` lists the
        categorical columns it read. ``metadata["tie_break"]`` is the tie-break rule the fit ran
        under, ``andrey.config.tie_break``.

    Raises
    ------
    TypeError
        If an argument has the wrong type, such as a string for a number; the message names the
        method, the argument, what it accepts, and the value. Also if ``categorical`` is not a list
        of column names and positions, or ``knowledge`` is neither ``None`` nor a
        :class:`~andrey.Knowledge`.
    ValueError
        If an argument lies outside its range or is not one of its choices; the message names the
        method, the argument, what it accepts, and the value. If ``env`` or ``targets`` is
        malformed, or leaves an environment's targets unknown, as :func:`~andrey.gies` lists; if no
        environment is observational, or every one is, as with ``env`` and ``targets`` both
        ``None``; if an environment has fewer than 3 rows; if ``invariance_test="gaussian"`` comes
        with a categorical or mixed-data ``indep_test``; if ``data`` holds a missing or infinite
        value, a column constant on every row or on the observational rows, or does not suit the
        test, as for :func:`~andrey.pc`. With a partial-correlation test, also if a column repeats
        another up to scale and shift on the observational rows; one message names every group of
        copies. Also if ``knowledge`` names a column ``data`` lacks or contradicts itself, as
        :class:`~andrey.Knowledge` lists.

    Warns
    -----
    DataTypeWarning
        When a column holds at most 10 distinct whole numbers and so may hold category codes,
        which a continuous test reads as numbers. Each message warns once per process.

    Notes
    -----
    **When to use.** IGSP suits continuous data from an observational environment and settings whose
    targets are all known, with linear effects, Gaussian noise, and no unmeasured common causes [1]_
    [2]_. Its authors claim it for perfect interventions, which cut a variable from its causes, and
    for imperfect ones, which change its mechanism, such as a smaller weight on each cause or a
    perfect intervention that succeeds on some rows only. Their simulations use 10 or 20 variables
    with one or two settings of 4 to 10 targets and 1,000 to 100,000 rows per setting [1]_, or up to
    20 single-variable or 45 two-variable settings [2]_. It returns the interventional essential
    graph, the same kind of graph :func:`~andrey.gies` returns from the same inputs; the authors
    show a 7-variable DAG where the score search of GIES is not consistent and IGSP is. In the
    authors' settings, :func:`~andrey.gies` and :func:`~andrey.ut_igsp` told every target often had
    a lower SHD (see Limitations). When some targets are unknown, use :func:`~andrey.ut_igsp`; with
    no observational environment, use :func:`~andrey.gies`.

    **Choosing alpha and n_starts.** The defaults come from simulated data in the authors' settings
    [1]_ [2]_: linear effects and Gaussian noise on 7 to 20 variables, with one or two settings of
    up to 10 targets or up to 45 settings of one or two targets, perfect, inhibiting, or imperfect,
    and 1,000 to 100,000 rows per setting: 87 groups of 20 datasets, one fit each, one set of
    datasets with no rerun. A candidate's SHD counts as lower or higher than another's in a group
    when the 95% bootstrap interval of the mean difference, paired by dataset, lies wholly below or
    above 0. Among the levels the authors swept, ``alpha=1e-3`` was worse in none of the groups and
    lower than ``1e-4`` in 4 groups, ``5e-3`` in 16, ``1e-2`` in 20, and ``0.05`` in 83, by 0.2 to
    10.7 edges; ``5e-4`` tied with it in every group. On 20 variables at mean degree 3 with 8
    perfect single-variable settings and 1,000 or 10,000 rows per setting, outside these groups, no
    level was lower than ``1e-3``. At ``alpha=1e-3``, 10 starts and 5 were each worse in none of the
    groups and tied with each other in every group; against one start, 10 were lower in 21 groups
    and 5 in 19, by 0.4 to 1.7 edges, at a median of 0.10 s per fit against 0.07 s for 5. The
    default level, one over the observational rows, as for :func:`~andrey.ut_igsp`, equals ``1e-3``
    at 1,000 observational rows, and in the 50 groups at 10,000 and 100,000 rows per setting it had
    fewer SHD errors than ``1e-3`` in 1, by 0.28 edges, and more in none.

    **How it works.** A variable order gives a DAG, its minimal I-MAP: ``i -> j`` for ``i`` before
    ``j`` unless the conditional-independence test finds them independent given every other
    variable before ``j``, on the observational rows (all environments whose targets are ``[]``,
    pooled). An edge ``i -> j`` is I-contradicting when the settings say it points the wrong way
    [2]_: with single-variable settings, when ``j``'s distribution is the same under every
    setting that targets ``i`` alone, or ``i``'s changes under one that targets ``j`` alone; with
    larger target sets, when, given some set of ``j``'s other neighbors, ``j``'s distribution is
    the same under every setting that targets ``i`` and not ``j``, or, given every set of ``i``'s
    other neighbors, ``i``'s changes under some setting that targets ``j`` and not ``i``. The
    objective is the number of edges, then the number of I-contradicting edges. The search starts
    from the order of a minimum-degree elimination of the graph of pairs dependent given all other
    variables, and moves by reversing I-covered edges: ``i -> j`` where ``j``'s parents are ``i``
    and ``i``'s parents, and ``j``'s distribution does not change under a setting that targets
    ``i`` alone. A depth-first search visits the reversals of I-contradicting edges first and
    backtracks at a dead end, up to 5 reversals away; it restarts from a DAG with fewer edges
    whenever it finds one, and keeps the DAG with the lowest objective it saw [1]_ [2]_. The
    result is the best of ``n_starts`` such searches, returned as its interventional essential
    graph under the targets [3]_.

    **Invariance tests.** Under ``"gaussian"``, a variable given a set of others is invariant when
    the linear regressions with an intercept agree in the two groups of rows: the Chow test of
    equal coefficients and intercept and the F-test of equal residual variances [4]_, exact under
    Gaussian noise, each at half the level. With no other variable it compares the means and
    variances, so it sees a perfect intervention that keeps the mean. A group with no more rows
    than the regression has parameters leaves nothing to compare, and the variable reads as
    invariant. Under ``"normal_scores"`` the regressions read every column's normal scores over
    all rows. Under ``"indicator"``, a variable is invariant when ``indep_test`` finds it
    independent of a 0/1 indicator of the setting's rows given the others, on the rows of the
    setting and the observational ones [5]_; under ``"fisherz"`` and ``"student_t"`` this sees a
    change of the mean alone, which is why ``"gaussian"`` is their default.

    **Environments and targets.** Every environment whose targets are ``[]`` is observational, and
    their rows are pooled. Every other environment is a setting, compared with the observational
    rows alone, never with another setting. With ``env``, two environments with the same targets
    stay two settings; without ``env``, rows with the same targets are one setting. A setting's
    targets are used as given: a target whose distribution the intervention leaves unchanged
    orients nothing.

    **Column order.** Each column gets a tie rank computed from the data on every row, as for
    :func:`~andrey.hc`, and the search runs on the columns sorted by it: the first search breaks
    ties between orders and between moves by the rank, and the later ones shuffle them. Reordering
    the columns relabels the result, and the labels of the environments and the order of the rows
    change nothing for the tests that draw no random numbers. The
    :doc:`column-order guide </docs/guides/column-order>` says what the rank keeps.

    **Missing values.** A ``NaN`` in ``data`` raises ``ValueError``: every invariance test
    regresses or tests a variable on all of its conditioning variables in each environment. Drop
    the incomplete rows first.

    **Limitations.** Measured on simulated data in the authors' settings, the 87 groups above with 5
    fits per dataset, one core per fit, at ``alpha`` ``1e-4``, ``1e-3``, and ``1e-2``, with 10
    starts and with 5: in every group its SHD was below the empty graph's and every fit finished. On
    10 variables with 2 settings of 4 targets and 1,000 rows per setting, the objective often ties
    the true DAG with a wrong one, and the first start, the minimum-degree order with its ties
    broken by the column ranks, is the same in every fit, so fits with few starts repeat one result
    there. Unless a level is named, the numbers below are at ``alpha=1e-3``, the default's level at
    1,000 observational rows. On the same data, :func:`~andrey.gies` had a lower SHD in 35 of the 87
    groups, by 0.35 to 2.63, and a higher one in 6, and :func:`~andrey.ut_igsp` told every target a
    lower one in 27, by 0.38 to 1.95, and a higher one in none. A perfect intervention that leaves
    its target's distribution unchanged, such as cutting the causes of a variable that has none
    while keeping its noise, orients nothing, and the search then reverses some edges out of that
    target: 823 of 3,960 such edges over the 87 groups, against 16 for :func:`~andrey.gies`. Outside
    the authors' setting, one fit per dataset: with 100 rows per setting on 10 variables its SHD was
    4.65 against the empty graph's 8; at mean degree 4 on 20 variables it was 11.8 against 40; with
    one unknown target per setting, 0.85 and 0.95 against 0.05 and 0.15 for :func:`~andrey.ut_igsp`
    and 15 for the empty graph. With additive nonlinear effects on 10 variables its SHD was 2.05 and
    2.3 with the Gaussian tests and 5.0 and 1.15 with ``indep_test="kci"``, at 200 and 500 rows per
    setting, against 8, a kernel fit taking 1.7 s and 13.8 s; on categorical data with
    ``indep_test="gsq"``, 6.4 at 1,000 rows per setting and 0.9 at 10,000, against 10. With 10
    single-variable settings of 1,000 rows, its SHD was 3.9 at 50 variables, 15.3 at 100, and 42.9
    at 200, against 38, 75, and 150, a median fit taking 1.6 s, 33 s, and 726 s and 0.5 GB of memory
    at 100; at 400 no fit finished within an hour, while :func:`~andrey.gies` reached 0.2, 0.1, and
    0.3 at 50 to 200 variables in under 2 s. Its time grows with the number of rows. With settings
    of more than one target, each edge's test runs over the subsets of a neighborhood, so dense
    graphs are slow. The search's cost grows with the number of I-covered edges to the power of its
    depth, and each start tests every pair given all other variables, so time grows fast with the
    number of variables. Unmeasured common causes are outside its model and not measured. One level
    serves both tests unless ``invariance_alpha`` is given. In :func:`~andrey.bootstrap`, the
    averaged graph is made a CPDAG without the targets, so an edge that only the settings orient
    comes back undirected there; each fit and the edge frequencies keep its direction. The numbers
    are a reference, not a hard limit: they change with the data generator, the density, the
    method's settings, and the machine.

    References
    ----------
    .. [1] Wang, Y., Solus, L., Yang, K. D., and Uhler, C. (2017). Permutation-based causal
       inference algorithms with interventions. NeurIPS 30.
       https://proceedings.neurips.cc/paper/2017/hash/275d7fb2fd45098ad5c3ece2ed4a2824-Abstract.html
    .. [2] Yang, K. D., Katcoff, A., and Uhler, C. (2018). Characterizing and learning equivalence
       classes of causal DAGs under interventions. ICML, PMLR 80, 5537-5546.
       https://proceedings.mlr.press/v80/yang18a.html
    .. [3] Hauser, A., and Buhlmann, P. (2012). Characterization and greedy learning of
       interventional Markov equivalence classes of directed acyclic graphs. JMLR 13, 2409-2464.
    .. [4] Chow, G. C. (1960). Tests of equality between sets of coefficients in two linear
       regressions. Econometrica 28(3), 591-605.
    .. [5] Heinze-Deml, C., Peters, J., and Meinshausen, N. (2018). Invariant causal prediction for
       nonlinear models. Journal of Causal Inference 6(2).

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from andrey import igsp
    >>> rng = np.random.default_rng(0)
    >>> env = np.repeat(["observed", "set y"], 2000)
    >>> x = rng.standard_normal(4000)
    >>> y = np.where(env == "set y", rng.standard_normal(4000), x + rng.standard_normal(4000))
    >>> targets = {"observed": [], "set y": ["y"]}
    >>> out = igsp(pd.DataFrame({"x": x, "y": y}), env, targets=targets, seed=0)
    >>> out.structure.oriented_edges()
    [('x', 'y', 'directed')]
    """
    from andrey.core.environments import normalize_env_targets
    from andrey.search.igsp import igsp as _igsp

    data, codes, env_labels, by_code = normalize_env_targets(
        data, env, targets, require_known_targets=True, method="igsp"
    )
    indep_test, invariance_test, X, mask = _chosen_tests(
        "igsp", data, indep_test, invariance_test, categorical
    )
    labels = column_labels(data)
    named = [_env_name(env_labels[c], env is None, labels) for c in sorted(by_code)]
    _check_environments("igsp", X, labels, codes, by_code, named, env is None)
    constraints = resolve(knowledge, X.shape[1], labels, method="igsp")
    observational = [code for code, t in by_code.items() if t == frozenset()]
    level = 1.0 / int(np.isin(codes, observational).sum()) if alpha is None else alpha
    with copies_named(labels):
        graph, (edges, contradicting) = _igsp(
            X,
            codes,
            by_code,
            alpha=level,
            invariance_alpha=level if invariance_alpha is None else invariance_alpha,
            indep_test=indep_test,
            invariance_test=invariance_test,
            categorical=mask,
            n_starts=n_starts,
            seed=seed,  # ty: ignore[invalid-argument-type]  # an int: method_spec resolves it
            constraints=constraints,
        )
    metadata = {
        "score": edges,
        "n_contradicting": contradicting,
        "invariance_test": invariance_test,
        "alpha": level,
        **categorical_metadata(mask, data),
    }
    return structure_output(graph, labels=labels, metadata=metadata)


@method_spec("ut_igsp")
def ut_igsp(
    data: npt.ArrayLike,
    env: npt.ArrayLike | str | None = None,
    *,
    targets: Mapping | npt.ArrayLike | str | None = None,
    alpha: float | None = None,
    invariance_alpha: float | None = None,
    indep_test: str | None = None,
    invariance_test: str | None = None,
    n_starts: int = 10,
    seed: int | None = None,
    categorical: Sequence[str | int] | None = None,
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn an interventional essential graph and unknown intervention targets with UT-IGSP.

    UT-IGSP (unknown-target interventional greedy sparsest permutation) learns from data pooled
    from an observational environment and interventional ones, the settings, whose targets, the
    variables each one intervened on, may be known in part or not at all [1]_. It searches
    variable orders for the DAG with the fewest edges and targets: each order gives the DAG of the
    edges its conditional-independence tests keep on the observational rows, and a setting's
    targets are its known ones plus every variable whose distribution given its parents differs
    from the observational one. It returns the DAG's interventional essential graph under the
    targets it found, with the targets.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables), or a list or mapping of them
        Observed data, one row per sample, from every environment. Coerced to ``float64``. A
        DataFrame's column names become the structure's ``labels``. A column that ``env`` or
        ``targets`` names is not a variable: it leaves the data before the fit, and
        ``categorical``, ``knowledge``, ``targets``, and the result's positions count the other
        columns.
        A list of DataFrames or 2-D arrays gives each environment's rows apart, labeled by
        position ``0, 1, ...``, and a mapping labels each by its key: the rows are stacked in that
        order, ``env`` stays ``None``, and ``targets`` maps those labels to their variables.
    env : array-like of shape (n_samples,) or (n_samples, 1), str, or None, default=None
        The environment of each row: one label per row, strings or numbers, not both, or the name of
        a column of the DataFrame ``data`` that holds them. ``None`` takes the environments from
        ``targets``: its distinct values.
    targets : mapping, str, array-like of shape (n_samples, n_variables), or None, default=None
        The known targets of each environment, named by column label or 0-based position. A
        mapping ``{label: [variables]}`` takes a label of ``env`` to its known targets: ``[]``
        marks an observational environment, a list a setting with those known targets, and
        ``None``, or a label the mapping leaves out, a setting whose targets are all unknown; a key
        that is not a label of ``env`` raises ``ValueError``. When the labels of ``env`` are
        numbers, a text key names the label it spells, ``"1"`` the label ``1``, so
        ``metadata["targets"]`` read back from JSON passes as it is. A string names a column of the
        DataFrame ``data`` whose cells give each row's targets: a label or a position, a list of
        them, or an empty cell (``None``, ``NaN``, ``""``, ``[]``) for an observational row. An
        array is ``True`` (or 1) where the row's value of the variable was set by an
        intervention. With a column or an array an empty row is observational, so only the
        mapping can name a setting with no known targets; without ``env``, rows with the same
        targets are one environment. ``None`` with ``env`` makes every environment a setting with
        unknown targets, which leaves none observational and raises. Every environment needs at
        least 3 rows. See Notes.
    alpha : float or None, default=None
        Significance level of the conditional-independence tests, in ``(0, 1)``: two variables
        are independent when a test's p-value exceeds it. Smaller values keep more edges out.
        ``None`` uses one over the number of observational rows, 0.001 at 1,000 rows, so the level
        falls as the rows grow. Also the level of the invariance tests when ``invariance_alpha``
        is ``None``. ``metadata["alpha"]`` gives the level used. See Notes for the choice of the
        default.
    invariance_alpha : float or None, default=None
        Significance level of the invariance tests, in ``(0, 1)``: a variable is a target of a
        setting when the test of its distribution given its parents, there against the
        observational rows, gives a p-value at most this level. ``None`` uses the level of the
        conditional-independence tests, ``metadata["alpha"]``.
    indep_test : str or None, default=None
        The conditional-independence test, on the observational rows. For continuous data, on the
        partial correlation: ``"fisherz"``, ``"student_t"``, and ``"normal_scores"``; for
        categorical data coded by whole numbers: ``"chisq"`` and ``"gsq"``; for mixed data, with
        the categorical columns named by ``categorical``: ``"degen_gauss"`` and ``"cond_gauss"``;
        for nonlinear effects in continuous data, each far slower: ``"kci"``, ``"rcot"``,
        ``"cmiknn"``, and ``"gpdc"``, as for :func:`~andrey.pc`. ``None`` picks from the declared
        column types: ``"gsq"`` when every column is categorical, ``"cond_gauss"`` when some are,
        else ``"fisherz"``. ``metadata["indep_test"]`` names the test the fit used.
    invariance_test : {"gaussian", "indicator"} or None, default=None
        The invariance test, which asks whether a variable's distribution given a set of others is
        the same in a setting as in the observational rows. ``"gaussian"`` compares the linear
        regressions of the two groups of rows: their coefficients and intercept by the Chow test
        and their residual variances by an F-test, each at half the level. ``"indicator"`` runs
        ``indep_test`` between the variable and a 0/1 indicator of the setting's rows given the
        others. ``None`` picks ``"gaussian"`` under ``"fisherz"``, ``"student_t"``, and
        ``"normal_scores"``, else ``"indicator"``. ``"gaussian"`` reads every column as numbers,
        so with a categorical or mixed-data ``indep_test`` it raises ``ValueError``.
        ``metadata["invariance_test"]`` names the test the fit used. See Notes.
    n_starts : int, default=10
        Number of searches, at least ``1``, each from a minimum-degree order of the variables:
        the first breaks ties by each column's rank, and each later one draws them, and the order
        in which it visits moves, from ``seed`` and its own index. Returns the best-scoring search,
        the one that found the lowest objective; a later one replaces an earlier one only when it
        is strictly lower. See Notes for the choice of the default.
    seed : int or None, default=None
        ``None``, the default, is the configured seed, ``ANDREY_SEED`` or ``0`` when it is unset,
        not a fresh random one, so a fit repeats. Seed for the searches after the first, and for
        the random draws of ``"rcot"``, ``"cmiknn"``, and ``"gpdc"``. With ``n_starts=1`` and
        another test, the result does not depend on it.
    categorical : sequence of str or int, or None, default=None
        The categorical columns of ``data``, for a mixed-data test: column labels (strings) or
        0-based positions. ``None`` takes a DataFrame's ``category``, ``bool``, and text columns,
        never a column by its values. ``[]`` marks none. A categorical test reads every column as
        categorical, and a continuous test raises ``ValueError`` on a declared column, named here
        or by its type.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the pairs
        allowed to be adjacent, and temporal tiers, all honored. Every DAG the search visits has
        the required edges and none of the forbidden ones, and every order it visits puts the
        cause of each required edge, and each earlier tier, first. The result directs every edge
        the knowledge decides, then Meek's rules direct the edges that follow, as
        :class:`~andrey.Knowledge` describes.

    Returns
    -------
    StructureOutput
        Wraps a ``cpdag`` :class:`~andrey.core.GraphStructure`, the interventional essential graph
        of the DAG found under the observational environment and each setting's targets in
        ``metadata["targets"]``: directed marks (``->``) are the edges that every DAG of the class
        orients alike, undirected (``--``) the ones it leaves open. ``metadata["algorithm"]`` is
        ``"ut_igsp"``; ``metadata["targets"]`` maps each environment to its targets, known and
        learned, in column order, ``[]`` for an observational one: positions for an array or a
        DataFrame whose columns are ``0..d-1``, and column labels as text otherwise. Its keys are
        the labels of ``env`` in sorted order, as Python scalars, so the label ``1`` is the key
        ``1``, and the mapping passes back as ``targets=``: as it is, after ``save`` and ``load``,
        or from ``andrey run --json``, whose keys are the labels' JSON text. Passed back, the
        targets are known, the least each setting has: UT-IGSP may still learn more, so its refit
        can differ, while :func:`~andrey.gies` refits the graph under exactly these targets. A
        setting where no target was found maps to ``[]``, which ``targets=`` reads as observational
        and pools with the observational rows: a different fit, which raises when every setting maps
        to ``[]``. Compared with the true targets, such as those of :mod:`andrey.data`, the mapping
        holds the known targets it was given, and each setting's targets are its differences from
        the environments marked ``[]``, the true ones only when those environments are
        observational. Without ``env``, an environment's key is its known targets joined by commas,
        labels sorted as text or positions sorted, ``""`` for the observational one: a name, not a
        label of ``env``. ``metadata["score"]`` is the objective of the DAG found, its number of
        edges plus its number of targets over the settings, lower is better;
        ``metadata["indep_test"]`` and ``metadata["invariance_test"]`` name the tests used, and
        ``metadata["alpha"]`` is the level of the conditional-independence tests;
        ``metadata["seed"]`` is the seed used when the fit draws random numbers. Under a mixed-data
        test, ``metadata["categorical"]`` lists the categorical columns it read.
        ``metadata["tie_break"]`` is the tie-break rule the fit ran under,
        ``andrey.config.tie_break``.

    Raises
    ------
    TypeError
        If an argument has the wrong type, such as a string for a number; the message names the
        method, the argument, what it accepts, and the value. Also if ``categorical`` is not a list
        of column names and positions, or ``knowledge`` is neither ``None`` nor a
        :class:`~andrey.Knowledge`.
    ValueError
        If an argument lies outside its range or is not one of its choices; the message names the
        method, the argument, what it accepts, and the value. If ``env`` or ``targets`` is
        malformed, or ``data`` holds each environment's rows apart in a form it cannot read, as
        :func:`~andrey.gies` lists; if no environment is observational, or every one is; if an
        environment has fewer than 3 rows; if, without ``env``, two environments' known targets join
        to one key, as when a column label holds a comma; if ``invariance_test="gaussian"`` comes
        with a categorical or mixed-data ``indep_test``; if ``data`` holds a missing or infinite
        value, a column constant on every row or on the observational rows, or does not suit the
        test, as for :func:`~andrey.pc`. With a partial-correlation test, also if a column repeats
        another up to scale and shift on the observational rows; one message names every group of
        copies. Also if ``knowledge`` names a column ``data`` lacks or contradicts itself, as
        :class:`~andrey.Knowledge` lists.

    Warns
    -----
    DataTypeWarning
        When a column holds at most 10 distinct whole numbers and so may hold category codes,
        which a continuous test reads as numbers. Each message warns once per process.

    Notes
    -----
    **When to use.** UT-IGSP suits continuous data from an observational environment and several
    settings, with linear effects, Gaussian noise, and no unmeasured common causes, where the
    settings' targets are known in part or not at all [1]_. Its authors claim it for perfect
    interventions, which cut a variable from its causes, and imperfect ones, which change its
    mechanism, such as a shift of its mean, a change of its noise variance, or a change of the
    weights of its causes; their simulations use about 20 variables, 5 settings, and 1,000 or more
    rows per setting. It returns the interventional essential graph and each setting's targets. When
    every target is known, use :func:`~andrey.gies`; with no observational environment, or with
    environments that differ in ways that are not interventions, use :func:`~andrey.cdnod`.

    **Choosing alpha and n_starts.** The defaults come from simulated data with linear effects,
    Gaussian noise, and 5 settings, 5 seeds per dataset, mostly in the authors' setting of 20
    variables at mean degree 1.5. A comparison's errors count as lower or higher when the 95%
    bootstrap interval of the mean difference in errors, paired by dataset, lies wholly below or
    above 0. On 15 groups of datasets with mean shifts and perfect interventions at 300 to 5,000
    rows per setting, 20 datasets each and 10 in the one at 50 variables, 10 starts made fewer SHD
    errors than 5 in 4 groups, by 0.17 to 0.42 edges, 2 of them outside the authors' setting at 300
    rows per setting or 50 variables, and more in none, at a median of 0.24 s per fit on 20
    variables against 0.21 s. With 5 starts, ``alpha=0.001`` made fewer SHD errors than the
    authors' level, ``alpha=1e-5`` [1]_, in 7 groups and more in 2, those by about 0.3 edges at
    3,000 rows per setting with no unknown targets, and ``alpha=0.01`` made more in 10; at 1,000
    rows or fewer and at 50 variables ``0.001`` lowered the mean SHD by 0.8 to 2.5 edges. Neither
    change raised the target errors in any group. The level was then chosen on 56 further groups at
    1,000 to 5,000 rows per setting, 20 datasets each, with mean shifts, perfect interventions, and
    changes of the noise variance or of the weights, 0 to 3 unknown targets per setting, and in 4
    groups no known ones, among one over the number of observational rows, ``alpha=0.001``, and
    ``alpha=1e-5``. Against ``alpha=0.001`` the default made fewer SHD errors in 4 comparisons, all
    at 5,000 rows, by 0.2 to 0.65 edges, and more in none.

    **How it works.** A variable order gives a DAG, its minimal I-MAP: ``i -> j`` for ``i`` before
    ``j`` unless the conditional-independence test finds them independent given every other
    variable before ``j``, on the observational rows (all environments whose targets are ``[]``,
    pooled). A setting's targets in that DAG are its known ones plus every variable whose
    distribution given its parents in the DAG is not invariant between the setting and the
    observational rows. The objective is the number of edges plus the number of targets over the
    settings [1]_. The search starts from the order of a minimum-degree elimination of the graph
    of pairs dependent given all other variables, and moves by reversing I-covered edges: ``i ->
    j`` where ``j``'s parents are ``i`` and ``i``'s parents, and, in each setting with ``i`` among
    its known targets, ``j``'s distribution given its parents is not invariant. A depth-first
    search over reversals that do not raise the objective moves to a DAG that lowers it, up to 5
    reversals away, and repeats until none does [1]_. The result is the best of ``n_starts`` such
    searches, returned as its interventional essential graph under the targets found [2]_.

    **Invariance tests.** Under ``"gaussian"``, a variable given a set of others is invariant when
    the linear regressions with an intercept agree in the two groups of rows: the Chow test of
    equal coefficients and intercept and the F-test of equal residual variances [3]_, exact under
    Gaussian noise, each at half the level, so a variable is a target when either rejects. A
    regressor constant within a group, as under a hard intervention without noise, costs that
    group one degree of freedom. A group with no more rows than the regression has parameters, as
    a setting of 3 rows given 2 other variables, leaves nothing to compare, and the variable reads
    as invariant: such a setting learns no target from that query and blocks the reversal of an
    edge from one of its known targets into that variable. Under ``"normal_scores"`` the
    regressions read every column's normal scores over all rows. Under ``"indicator"``, a variable
    is invariant when ``indep_test`` finds it independent of a 0/1 indicator of the setting's rows
    given the others, on the rows of the setting and the observational ones [4]_; under
    ``"fisherz"`` and ``"student_t"`` this sees a change of the mean alone, which is why
    ``"gaussian"`` is their default.

    **Environments and targets.** Every environment whose targets are ``[]`` is observational, and
    their rows are pooled. Every other environment is a setting, compared with the observational
    rows alone, never with another setting. A setting's known targets count as targets whatever
    their tests say. Only the mapping form of ``targets`` gives a setting with no known targets:
    in the column and array forms an empty row is observational, so with ``env`` such a setting
    would be pooled with the observational rows, and without ``env`` settings with the same known
    targets merge into one.

    **Column order.** Each column gets a tie rank computed from the data on every row, as for
    :func:`~andrey.hc`, and the search runs on the columns sorted by it: the first search breaks
    ties between orders and between moves by the rank, and the later ones shuffle them. Reordering
    the columns relabels the result, and the labels of the environments and the order of the rows
    change nothing for the tests that draw no random numbers. The
    :doc:`column-order guide </docs/guides/column-order>` says what the rank keeps.

    **Missing values.** A ``NaN`` in ``data`` raises ``ValueError``: every invariance test
    regresses or tests a variable on all of its conditioning variables in each environment. Drop
    the incomplete rows first.

    **Limitations.** Measured on simulated data, 20 datasets per setting (10 at 50 and 100 variables
    and on categorical data), one core per fit. In the authors' setting every fit finished, and in
    all 56 groups its SHD was below the empty graph's 15 and, where the given targets missed some,
    its target errors were below those of the given targets alone. Outside the authors' setting, at
    the default level: with 100 rows per setting its SHD was 8.35 against the empty graph's 15, and
    with no unknown targets it named 1.50 false targets per dataset; on a denser graph, 50 variables
    at mean degree 5 with 3 settings, it did not clearly beat the empty graph, its SHD 130.2 and
    123.9 against 125 at 500 and 2,000 rows per setting; with additive nonlinear effects and the
    Gaussian tests its SHD was 1.55 and 1.4, and its target errors 1.85 with no unknown targets and
    4.85 with 3 per setting. Its authors state a nonparametric version but run none; with
    ``indep_test="kci"`` on 10 variables with additive nonlinear effects its SHD was 5.3 and 5.5 at
    200 rows per setting and 2.4 and 2.35 at 400, against the empty graph's 8, a fit taking 8 to 30
    s. On categorical data with ``indep_test="gsq"``, 10 variables, its SHD was 5.4 at 1,000 rows
    per setting and 1.6 at 5,000, against 10. With 2,000 rows per setting and 3 unknown targets per
    setting, its SHD was 1.5 at 50 variables and 4.8 at 100, against 38 and 75, a median fit taking
    1.7 s and 15.2 s. A median fit on 20 variables took 0.24 s. The search's cost grows with the
    number of I-covered edges to the power of its depth, and each start tests every pair given all
    other variables, so time grows fast with the number of variables. The objective counts targets,
    so a setting whose intervention leaves its target's distribution unchanged orients nothing
    through that target. Unmeasured common causes are outside its model and not measured. One level
    serves both tests unless ``invariance_alpha`` is given. The numbers are a reference, not a hard
    limit: they change with the data generator, the density, the method's settings, and the machine.

    References
    ----------
    .. [1] Squires, C., Wang, Y., and Uhler, C. (2020). Permutation-based causal structure learning
       with unknown intervention targets. UAI, PMLR 124, 1039-1048.
       https://proceedings.mlr.press/v124/squires20a.html
    .. [2] Yang, K. D., Katcoff, A., and Uhler, C. (2018). Characterizing and learning equivalence
       classes of causal DAGs under interventions. ICML, PMLR 80, 5537-5546.
       https://proceedings.mlr.press/v80/yang18a.html
    .. [3] Chow, G. C. (1960). Tests of equality between sets of coefficients in two linear
       regressions. Econometrica 28(3), 591-605.
    .. [4] Heinze-Deml, C., Peters, J., and Meinshausen, N. (2018). Invariant causal prediction for
       nonlinear models. Journal of Causal Inference 6(2).

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from andrey import ut_igsp
    >>> rng = np.random.default_rng(0)
    >>> env = np.repeat(["observed", "shift x", "unknown"], 1000)
    >>> x = rng.standard_normal(3000) + 2.0 * (env == "shift x")
    >>> y = x + rng.standard_normal(3000) + 2.0 * (env == "unknown")
    >>> data = pd.DataFrame({"x": x, "y": y})
    >>> targets = {"observed": [], "shift x": ["x"], "unknown": None}
    >>> out = ut_igsp(data, env, targets=targets, seed=0)
    >>> out.structure.oriented_edges()
    [('x', 'y', 'directed')]
    >>> out.metadata["targets"]
    {'observed': [], 'shift x': ['x'], 'unknown': ['y']}

    Number labels stay numbers, so the result passes back as ``targets``:

    >>> codes = np.repeat([0, 1, 2], 1000)
    >>> ut_igsp(data, codes, targets={0: [], 1: ["x"], 2: None}, seed=0).metadata["targets"]
    {0: [], 1: ['x'], 2: ['y']}
    """
    from andrey.core.environments import normalize_env_targets
    from andrey.search.igsp import ut_igsp as _ut_igsp

    data, codes, env_labels, by_code = normalize_env_targets(
        data, env, targets, require_known_targets=False, method="ut_igsp"
    )
    indep_test, invariance_test, X, mask = _chosen_tests(
        "ut_igsp", data, indep_test, invariance_test, categorical
    )
    labels = column_labels(data)
    named = [_env_name(env_labels[c], env is None, labels) for c in sorted(by_code)]
    shared = [name for name in dict.fromkeys(named) if named.count(name) > 1]
    if shared:
        raise ValueError(
            "ut_igsp names each environment by its known targets joined by commas when env is "
            f"None, and {shared} names more than one, as a column label holds a comma or is "
            "empty; pass env"
        )
    _check_environments("ut_igsp", X, labels, codes, by_code, named, env is None)
    constraints = resolve(knowledge, X.shape[1], labels, method="ut_igsp")
    observational = [code for code, t in by_code.items() if t == frozenset()]
    level = 1.0 / int(np.isin(codes, observational).sum()) if alpha is None else alpha
    with copies_named(labels):
        graph, found, score = _ut_igsp(
            X,
            codes,
            by_code,
            alpha=level,
            invariance_alpha=level if invariance_alpha is None else invariance_alpha,
            indep_test=indep_test,
            invariance_test=invariance_test,
            categorical=mask,
            n_starts=n_starts,
            seed=seed,  # ty: ignore[invalid-argument-type]  # an int: method_spec resolves it
            constraints=constraints,
        )
    variables = label_of(labels)
    metadata = {
        "targets": {
            named[c]: [variables(j) for j in sorted(found.get(c, ()))] for c in sorted(by_code)
        },
        "score": score[0] if len(score) == 1 else list(score),
        "invariance_test": invariance_test,
        "alpha": level,
        **categorical_metadata(mask, data),
    }
    return structure_output(graph, labels=labels, metadata=metadata)


def _env_name(label: object, from_targets: bool, labels: tuple[str, ...] | None) -> object:
    """Return an environment's key in ``metadata["targets"]``.

    The key is its label of ``env``, or, when the environments come from the targets, its known
    targets joined by commas: their labels sorted as text, or their positions sorted.
    """
    if not from_targets:
        return label.item() if isinstance(label, np.generic) else label
    positions = sorted(label)  # ty: ignore[invalid-argument-type]
    return ",".join(sorted(labels[j] for j in positions) if labels else map(str, positions))


def _chosen_tests(
    method: str,
    data: npt.ArrayLike,
    indep_test: str | None,
    invariance_test: str | None,
    categorical: Sequence[str | int] | None,
) -> tuple[str, str, np.ndarray, np.ndarray | None]:
    """Pick the tests of ``igsp`` or ``ut_igsp`` and check the data against them.

    Returns the conditional-independence test, the invariance test (``None`` picks it to match),
    the checked data, and the categorical mask. Raises ``ValueError`` when ``"gaussian"`` comes
    with a categorical or mixed-data test, which reads category codes.
    """
    from andrey.core.ci import DISCRETE_INDEP_TESTS, MIXED_INDEP_TESTS
    from andrey.core.invariance import default_invariance_test

    indep_test, categorical = default_test(data, indep_test, categorical=categorical)
    invariance_test = invariance_test or default_invariance_test(indep_test)
    if invariance_test == "gaussian" and indep_test in DISCRETE_INDEP_TESTS + MIXED_INDEP_TESTS:
        raise ValueError(
            f"{method}: invariance_test='gaussian' regresses on the columns' values and so cannot "
            f"read categorical columns, which indep_test={indep_test!r} reads; use 'indicator' "
            "or None"
        )
    X, mask = checked_data(data, categorical, "indep_test", indep_test, method=method)
    return indep_test, invariance_test, X, mask


def _check_environments(
    method: str,
    X: np.ndarray,
    labels: tuple[str, ...] | None,
    codes: np.ndarray,
    by_code: Mapping[int, frozenset[int] | None],
    named: list[object],
    from_targets: bool,
) -> None:
    """Raise ``ValueError`` unless the environments of ``igsp`` or ``ut_igsp`` can be compared.

    Some environment must be observational (targets ``[]``) and some not, every one needs 3 rows,
    and no column may be constant on the observational rows, where the conditional-independence
    tests run. ``named`` gives each environment's label of ``env`` for the messages, by code; when
    ``from_targets``, the environments come from the targets, and the messages name each one by
    its list of targets instead.
    """
    if from_targets:
        variable = label_of(labels)
        named = [sorted(variable(j) for j in by_code[c] or ()) for c in sorted(by_code)]
    observational = [code for code, t in by_code.items() if t == frozenset()]
    if not observational and from_targets:
        raise ValueError(
            f"{method} needs an observational environment, rows with no targets, but every row "
            f"has some: the rows' targets are {named}; give some rows no targets, an empty cell "
            "of the targets column or an all-zero row of the targets array"
        )
    if not observational:
        raise ValueError(
            f"{method} needs an observational environment, but none of {named} has targets []; "
            "give one no targets: map its label of env to [] in a targets mapping, or give its "
            "rows an empty cell of a targets column or an all-zero row of a targets array"
        )
    if len(observational) == len(by_code):
        rest = "no row has targets" if from_targets else f"every one of {named} has targets []"
        raise ValueError(
            f"{method} needs at least one interventional environment besides the observational "
            f"ones, but {rest}; for observational data use grasp or boss"
        )
    sizes = np.bincount(codes, minlength=len(by_code))
    small = [named[c] for c in np.flatnonzero(sizes < 3)]
    if small:
        which = "the rows with targets" if from_targets else "the environments"
        raise ValueError(
            f"{method} needs at least 3 rows in every environment; {which} {small} have fewer"
        )
    rows = np.isin(codes, observational)
    constant = np.flatnonzero(np.ptp(X[rows], axis=0) == 0)
    if constant.size:
        columns = [labels[j] if labels else int(j) for j in constant]
        raise ValueError(
            f"{method}: columns {columns} are constant on the observational rows, where the "
            "conditional-independence tests run; drop them"
        )
