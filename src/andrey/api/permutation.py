"""Permutation-based facade adapters.

``boss`` and ``grasp`` are native -- they run a permutation/score search on the :mod:`andrey.core`
compute substrate and return a CPDAG ``GraphStructure`` with its objective score. Both score
candidate parent additions through the batched Schur delta-BIC surface over the linear-Gaussian BIC
(:class:`andrey.core.score.BICScore`), or by full local scores over BDeu, the multinomial BIC, or a
mixed-data BIC. Their search order is drawn from a private RNG seeded by ``seed`` (or the global
``ANDREY_SEED``), so a run is reproducible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from andrey.api._adapt import (
    categorical_metadata,
    checked_data,
    column_labels,
    copies_named,
    default_score,
    json_safe,
    structure_output,
)
from andrey.api._method import method_spec
from andrey.core.knowledge import Knowledge, resolve

if TYPE_CHECKING:
    from collections.abc import Sequence

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
    """Learn a CPDAG with BOSS (Best Order Score Search).

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
        standard BIC. Must be finite and non-negative. The other scores ignore it. See Notes.
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
    categorical : sequence of str or int, optional
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

    Raises
    ------
    TypeError
        If ``categorical`` is not a list of column names and positions, or ``knowledge`` is neither
        ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If an argument has the wrong type, lies outside its range, or is not one of its choices; the
        message names the method, the argument, what it accepts, and the value. Also if the required
        edges of ``knowledge`` give a node more than ``max_parents`` parents; if ``data`` or
        ``categorical`` does not suit the score, as for :func:`~andrey.ges`. With the BIC score,
        also if a column repeats another up to scale and shift given more than two rows or, with
        more rows than columns, is a linear function of other columns. One message names every group
        of copies and says to keep one column per group. Also if ``knowledge`` names a column
        ``data`` lacks or contradicts itself, as :class:`~andrey.Knowledge` lists.

    Warns
    -----
    DataTypeWarning
        When a mixed-data score finds no declared categorical column and so runs as the
        linear-Gaussian form, or when a setting reads columns that look like category codes as
        numbers. Each message warns once per process.

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
    connected variables [1]_. When every column is categorical, the default score is
    ``"bdeu"``, and on mixed data ``"cond_gauss"``;
    ``"discrete_bic"`` and ``"degen_gauss"`` also score those data.

    **When not to.** When unmeasured common causes are plausible, use :func:`~andrey.fci`. When the
    noise is non-Gaussian and every edge should be oriented, use :func:`~andrey.direct_lingam`.

    **Choosing seed.** The search visits variables in a random order and stops when no move improves
    the order. Different seeds can therefore reach different graphs; comparing a few shows whether
    a result depends on the seed.

    **Choosing n_starts.** The search can stop at an order that scores worse than the true graph,
    most often on dense graphs. Its authors run one search [1]_; Tetrad's ``numStarts`` [6]_ runs
    later searches from random orders and keeps the best score, as ``n_starts`` does. On simulated
    linear-Gaussian data with 100 variables at mean degree 4 and 1,000 rows, at the defaults, 8
    starts lowered the mean errors (SHD) over 20 datasets from 7.9 to 2.1 and took a median of 7.7
    times as long as one start.

    **How it works.** BOSS searches variable orders for the best DAG with the lowest total score
    [1]_. Each variable uses a grow-shrink tree to cache its best parent set for any set of
    variables allowed to precede it. Each sweep moves every variable to the position that most
    improves the total score. Sweeps repeat until no move improves it. The final parent sets form a
    DAG, which is reduced to its CPDAG.

    A move or parent change must improve the score by more than ``1e-6``, a fixed absolute tolerance
    shared with :func:`~andrey.grasp`. Scores within this tolerance of the best count as ties. The
    earliest order position wins; among candidate parents, the column with the lowest tie rank
    wins. Under the linear-Gaussian BIC each comparison is decided as full local scores decide it:
    a fast update whose rounding-error bound could change the comparison is scored in full.

    **Column order.** Each column gets a tie rank computed from its own data, as for
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
    .. [6] Ramsey, J. D., Zhang, K., Glymour, M., Sanchez Romero, R., Huang, B., Ebert-Uphoff, I.,
       Samarasinghe, S., Barnes, E. A., and Glymour, C. (2018). TETRAD - A toolbox for causal
       discovery. 8th International Workshop on Climate Informatics.

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
    """Learn a CPDAG with GRaSP (Greedy Relaxation of the Sparsest Permutation).

    GRaSP searches variable orders (permutations) using linear-Gaussian BIC (Bayesian information
    criterion). It converts the best-scoring order to a CPDAG, which represents an equivalence class
    of graphs. It searches the same space as :func:`~andrey.boss` using chains of up to
    ``grasp_depth`` tucks [1]_, the moves described in Notes.

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
        standard BIC. Must be finite and non-negative. The other scores ignore it. See Notes.
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
    categorical : sequence of str or int, optional
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

    Raises
    ------
    TypeError
        If ``categorical`` is not a list of column names and positions, or ``knowledge`` is neither
        ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If an argument has the wrong type, lies outside its range, or is not one of its choices; the
        message names the method, the argument, what it accepts, and the value. Also if the required
        edges of ``knowledge`` give a node more than ``max_parents`` parents; if ``data`` or
        ``categorical`` does not suit the score, as for :func:`~andrey.ges`. With the BIC score,
        also if a column repeats another up to scale and shift given more than two rows or, with
        more rows than columns, is a linear function of other columns. One message names every group
        of copies and says to keep one column per group. Also if ``knowledge`` names a column
        ``data`` lacks or contradicts itself, as :class:`~andrey.Knowledge` lists.

    Warns
    -----
    DataTypeWarning
        When a mixed-data score finds no declared categorical column and so runs as the
        linear-Gaussian form, or when a setting reads columns that look like category codes as
        numbers. Each message warns once per process.

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
    search even on dense graphs [1]_. When every column is categorical, the default score is
    ``"bdeu"``, and on mixed data ``"cond_gauss"``;
    ``"discrete_bic"`` and ``"degen_gauss"`` also score those data.

    **When not to.** When unmeasured common causes are plausible, use :func:`~andrey.fci`. When the
    noise is non-Gaussian and every edge should be oriented, use :func:`~andrey.direct_lingam`.

    **GRaSP or BOSS.** Both search the same space of orders with different moves. Either can reach
    the better score on a given dataset. On the published benchmarks, GRaSP took far longer than
    :func:`~andrey.boss` at 400 variables.

    **Choosing seed.** Both the starting order and the order of visiting variables are random, so
    different seeds can reach different graphs; comparing a few shows whether a result depends on
    the seed.

    **Choosing n_starts.** The search can stop at a DAG that scores worse than the true one, most
    often on dense graphs. Its authors run one search [1]_; Tetrad's ``numStarts`` [6]_ runs later
    searches from random orders and keeps the best score, as ``n_starts`` does. On simulated
    linear-Gaussian data with 100 variables at mean degree 4 and 1,000 rows, at the defaults, 8
    starts lowered the mean errors (SHD) over 20 datasets from 18.2 to 4.6 and took a median of 8.3
    times as long as one start.

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

    **Column order.** Each column gets a tie rank computed from its own data, as for
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
    .. [6] Ramsey, J. D., Zhang, K., Glymour, M., Sanchez Romero, R., Huang, B., Ebert-Uphoff, I.,
       Samarasinghe, S., Barnes, E. A., and Glymour, C. (2018). TETRAD - A toolbox for causal
       discovery. 8th International Workshop on Climate Informatics.

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
