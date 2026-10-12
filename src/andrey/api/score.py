"""Score-based facade adapters.

``ges``, ``gies``, ``xges``, ``exact_search``, and ``hc`` use :mod:`andrey.core` for BIC scoring
and orientation; ``xges`` and ``hc`` use the Schur delta-BIC. Reported objectives are stored in
``metadata``.

CALM lazily imports its stochastic continuous optimizer from the ``[torch]`` extra and returns a
weighted DAG. Its output varies across torch versions, so determinism and quality tests replace a
committed baseline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from andrey.api._adapt import (
    column_labels,
    data_matrix,
    flag,
    int_in_range,
    json_safe,
    structure_output,
)
from andrey.api._experimental import warn_experimental
from andrey.core import backend
from andrey.core.seeding import resolve_seed

if TYPE_CHECKING:
    import numpy.typing as npt

    from andrey.core import StructureOutput


def ges(
    data: npt.ArrayLike, *, score_func: str = "local_score_BIC", lambda_value: float = 1.0
) -> StructureOutput:
    """Learn a CPDAG from observational data with greedy equivalence search (GES).

    GES searches CPDAGs, which represent equivalence classes of graphs. It adds edges while they
    improve BIC (Bayesian information criterion), then removes edges in a backward sweep [1]_.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``.
    score_func : {"local_score_BIC"}, default="local_score_BIC"
        Local score objective. Only the Gaussian BIC score is supported.
    lambda_value : float, default=1.0
        Multiplier on the BIC penalty of ``log(n_samples)`` per parent. Larger values give
        sparser graphs; ``1.0`` is the standard BIC. Must be finite and non-negative.

    Returns
    -------
    StructureOutput
        Wraps a ``cpdag`` :class:`~andrey.core.GraphStructure`; ``metadata["algorithm"]`` is
        ``"GES"`` and ``metadata["score"]`` the BIC score of the result, the sum over variables
        of ``n * log(residual variance) + lambda_value * log(n) * n_parents``; lower is better.
        Directed marks (``->``) are oriented edges, undirected (``--``) the ones the equivalence
        class leaves open.

    Raises
    ------
    NotImplementedError
        If ``score_func`` is not ``"local_score_BIC"``.
    ValueError
        If ``lambda_value`` is negative or non-finite.

    See Also
    --------
    andrey.fci : Allows unmeasured common causes and selection bias.
    andrey.direct_lingam : Orients every edge when the noise is non-Gaussian.

    Notes
    -----
    **When to use it.** GES suits continuous data with no unmeasured common causes. Its BIC score
    assumes that each variable depends linearly on its causes with Gaussian noise. In that setting
    it is a fast, accurate default. It returns the equivalence class.

    **When not to.** When unmeasured common causes are plausible, use :func:`~andrey.fci`. When the
    noise is non-Gaussian and every edge should be oriented, use :func:`~andrey.direct_lingam`.

    **Choosing lambda_value.** A larger value raises the penalty per parent, giving a sparser graph:
    fewer false edges and more missed ones. ``1.0`` is the standard BIC. With many variables and
    about ``10 * n_variables`` samples, ``lambda_value=2.0`` gave more accurate graphs than ``1.0``
    on the published benchmarks. FGES, a fast GES, multiplied the BIC penalty by 4 when searching
    continuous data on graphs of up to a million variables [2]_.

    **How it works.** In the forward phase, a variable with more than ``n_variables / 2`` parents
    gets no new edges. The backward phase has no cap.

    **Limitations.** Measured on simulated linear-Gaussian data from sparse random graphs (mean
    degree 2), from 20 to 200 variables with ten rows per variable and parents capped at 4 per
    variable rather than ``n_variables / 2``. At ``lambda_value=1.0`` its median errors (SHD) stayed
    below the empty graph's up to 150 variables (141 against 150) and passed it at 200 (206 against
    200), where ``lambda_value=2.0`` brought them to 43. The median fit took 0.15 s at 20 variables
    and 40 s at 150, on 128 cores. On the 223-variable ANDES network with 1,000 simulated rows and
    the default parent cap, its median errors passed the empty graph's (612 against 338).

    References
    ----------
    .. [1] Chickering, D. M. (2002). Optimal structure identification with greedy search. JMLR 3,
       507-554.
    .. [2] Ramsey, J., Glymour, M., Sanchez-Romero, R., and Glymour, C. (2017). A million variables
       and more: the Fast Greedy Equivalence Search algorithm for learning high-dimensional
       graphical causal models, with an application to functional magnetic resonance images.
       International Journal of Data Science and Analytics 3(2), 121-129.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from andrey import ges
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal(500)
    >>> y = rng.standard_normal(500)
    >>> z = x + y + 0.5 * rng.standard_normal(500)
    >>> ges(pd.DataFrame({"x": x, "y": y, "z": z})).structure.oriented_edges()
    [('x', 'z', 'directed'), ('y', 'z', 'directed')]
    """
    from andrey.search.ges import ges as _ges

    X = data_matrix(data)
    labels = column_labels(data)
    cpdag, score = _ges(X, score_func=score_func, lambda_value=lambda_value)
    return structure_output(
        cpdag,
        labels=labels,
        metadata={"algorithm": "GES", "score": json_safe(score)},
    )


def gies(data: npt.ArrayLike, *, lambda_value: float = 1.0) -> StructureOutput:
    """Learn a CPDAG from observational data with greedy interventional equivalence search (GIES).

    GIES generalizes GES to interventional data [1]_. This function takes observational data
    only, where GIES's score and search are those of GES: it returns the same CPDAG and score as
    :func:`andrey.ges` with the same ``lambda_value``.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``.
    lambda_value : float, default=1.0
        Multiplier on the BIC penalty of ``log(n_samples)`` per parent. Larger values give
        sparser graphs; ``1.0`` is the standard BIC. Must be finite and non-negative.

    Returns
    -------
    StructureOutput
        Wraps a ``cpdag`` :class:`~andrey.core.GraphStructure`; ``metadata["algorithm"]`` is
        ``"GIES"`` and ``metadata["score"]`` the BIC score of the result, computed as for
        :func:`andrey.ges`; lower is better. Directed marks (``->``) are oriented edges,
        undirected (``--``) the ones the equivalence class leaves open.

    Raises
    ------
    ValueError
        If ``lambda_value`` is negative or non-finite.

    Warns
    -----
    ExperimentalWarning
        On the first call in a process, because the method is experimental: it has no published
        benchmark, and its API may change without deprecation.

    Notes
    -----
    **Limitations.** Not benchmarked yet. Its tests run it on at most 10 variables with 600 rows. It
    takes observational data only, so it returns the graph and score that :func:`~andrey.ges`
    returns on the same data, and the limitations of GES apply.

    References
    ----------
    .. [1] Hauser, A., and Buhlmann, P. (2012). Characterization and greedy learning of
       interventional Markov equivalence classes of directed acyclic graphs. JMLR 13, 2409-2464.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import gies
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal(500)
    >>> y = x + 0.1 * rng.standard_normal(500)
    >>> gies(np.column_stack([x, y])).structure.kind
    'cpdag'
    """
    warn_experimental("gies")
    from andrey.search.gies import gies as _gies

    X = data_matrix(data)
    labels = column_labels(data)
    cpdag, score = _gies(X, lambda_value=lambda_value)
    metadata = {"algorithm": "GIES", "score": json_safe(score)}
    return structure_output(cpdag, labels=labels, metadata=metadata)


def xges(
    data: npt.ArrayLike,
    *,
    score_func: str = "local_score_BIC",
    lambda_value: float = 1.0,
    extended_search: bool = True,
) -> StructureOutput:
    """Learn a CPDAG from observational data with extremely greedy equivalence search (XGES).

    XGES searches the equivalence classes that GES searches [2]_, with GES's moves, the turning move
    of GIES [3]_, and the BIC score, but applies them in a different order [1]_ (see Notes). On
    linear-Gaussian data it often finds a graph with a better score than GES.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``.
    score_func : {"local_score_BIC"}, default="local_score_BIC"
        Only the Gaussian BIC score is supported.
    lambda_value : float, default=1.0
        Multiplier on the BIC penalty of ``log(n_samples)`` per parent. Larger values give
        sparser graphs; ``1.0`` is the standard BIC. Must be finite and non-negative.
    extended_search : bool, default=True
        Run the extended search after the base search (see Notes). ``False`` returns the base
        search's graph, which is faster.

    Returns
    -------
    StructureOutput
        Wraps a ``cpdag`` :class:`~andrey.core.GraphStructure`; ``metadata["algorithm"]`` is
        ``"XGES"`` and ``metadata["score"]`` the BIC score, computed as for :func:`andrey.ges`;
        lower is better. Directed marks (``->``) are oriented edges, undirected (``--``) the ones
        the equivalence class leaves open.

    Raises
    ------
    NotImplementedError
        If ``score_func`` is not ``"local_score_BIC"``.
    TypeError
        If ``extended_search`` is not a bool.
    ValueError
        If ``data`` is not a 2-D numeric matrix with at least two rows, no ``NaN`` or ``inf``, and
        no constant column, or if ``lambda_value`` is negative or non-finite.

    Warns
    -----
    ExperimentalWarning
        On the first call in a process, because the method is experimental: it has no published
        benchmark, and its API may change without deprecation.

    See Also
    --------
    andrey.ges : Adds edges, then removes them, in two separate phases.

    Notes
    -----
    **When to use it.** XGES makes the assumptions of :func:`~andrey.ges`: continuous data, no
    unmeasured common causes, and linear relations with Gaussian noise. It returns the equivalence
    class.

    **How it works.** Each step takes the best improving deletion, else reversal of a directed edge,
    else addition; the search stops when no move improves the score [1]_. The extended search then
    deletes each edge of that graph, best score first, forbids adding it back in that direction, and
    reruns the search. It keeps a result whose score is lower by more than ``2e-7 * n_samples``, and
    stops once a fresh pass over the edges keeps nothing. No variable has a parent limit.

    **Limitations.** Not benchmarked yet. Its tests run it on at most 258 variables with 512 rows,
    and on 10,000 rows with 10 variables. The extended search reruns the base search for each edge
    it tries to delete and retries the edges after keeping a result, so its run time grows with the
    number of edges and of results kept; ``extended_search=False`` skips it. Its BIC score assumes
    linear relations with Gaussian noise.

    References
    ----------
    .. [1] Nazaret, A., and Blei, D. (2024). Extremely greedy equivalence search. Proceedings of
       the Fortieth Conference on Uncertainty in Artificial Intelligence, PMLR 244, 2716-2745.
    .. [2] Chickering, D. M. (2002). Optimal structure identification with greedy search. JMLR 3,
       507-554.
    .. [3] Hauser, A., and Buhlmann, P. (2012). Characterization and greedy learning of
       interventional Markov equivalence classes of directed acyclic graphs. JMLR 13, 2409-2464.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from andrey import xges
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal(500)
    >>> y = rng.standard_normal(500)
    >>> z = x + y + 0.5 * rng.standard_normal(500)
    >>> xges(pd.DataFrame({"x": x, "y": y, "z": z})).structure.oriented_edges()
    [('x', 'z', 'directed'), ('y', 'z', 'directed')]
    """
    warn_experimental("xges")
    from andrey.search.xges import xges as _xges

    X = data_matrix(data)
    extended_search = flag(extended_search, "extended_search")
    labels = column_labels(data)
    cpdag, score = _xges(
        X, score_func=score_func, lambda_value=lambda_value, extended_search=extended_search
    )
    metadata = {"algorithm": "XGES", "score": json_safe(score)}
    return structure_output(cpdag, labels=labels, metadata=metadata)


def hc(
    data: npt.ArrayLike,
    *,
    score_func: str = "local_score_BIC_from_cov",
    lambda_value: float = 1.0,
    max_iter: int = 200,
) -> StructureOutput:
    """Learn a CPDAG from observational data with hill-climbing search.

    Hill-climbing is a lightweight greedy baseline [1]_ [2]_. It repeatedly adds, removes, or
    reverses the edge that most improves a covariance-based delta-BIC, then converts the DAG to its
    Markov equivalence class.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``.
    score_func : {"local_score_BIC_from_cov"}, default="local_score_BIC_from_cov"
        Local score objective; a covariance-based Gaussian BIC.
    lambda_value : float, default=1.0
        Multiplier on the BIC penalty of ``log(n_samples)`` per parent. Larger values give
        sparser graphs; ``1.0`` is the standard BIC. Must be finite and non-negative.
    max_iter : int, default=200
        Most moves the search takes, at least ``1``. Each move adds at most one edge, so the result
        has at most ``max_iter`` edges.

    Returns
    -------
    StructureOutput
        Wraps a ``cpdag`` :class:`~andrey.core.GraphStructure`; ``metadata["algorithm"]`` is
        ``"HC"`` and ``metadata["score"]`` the BIC score of the DAG found, computed as for
        :func:`andrey.ges`; lower is better. Directed marks (``->``) are oriented edges,
        undirected (``--``) the ones the equivalence class leaves open.

    Raises
    ------
    NotImplementedError
        If ``score_func`` is not ``"local_score_BIC_from_cov"``.
    TypeError
        If ``max_iter`` is not an int.
    ValueError
        If ``lambda_value`` is negative or non-finite, or if ``max_iter`` is less than ``1``.

    Warns
    -----
    ExperimentalWarning
        On the first call in a process, because the method is experimental: it has no published
        benchmark, and its API may change without deprecation.
    SearchLimitWarning
        When the search takes ``max_iter`` moves: it stopped at the limit, and a larger
        ``max_iter`` may find a better graph.

    Notes
    -----
    The search starts from the empty graph and stops when no single move improves the score, or
    after ``max_iter`` moves.

    **Limitations.** Not benchmarked yet. Its tests run it on at most 45 variables with 800 rows,
    and on 40,000 rows with 24 variables. A graph that no single move improves need not be the
    best-scoring one. Its BIC score assumes linear relations with Gaussian noise.

    References
    ----------
    .. [1] Chickering, D. M., Geiger, D., and Heckerman, D. (1995). Learning Bayesian networks:
       search methods and experimental results. AISTATS, PMLR R0, 112-128.
    .. [2] Scutari, M. (2010). Learning Bayesian networks with the bnlearn R package. Journal of
       Statistical Software 35(3). https://doi.org/10.18637/jss.v035.i03

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import hc
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal(500)
    >>> y = x + 0.1 * rng.standard_normal(500)
    >>> hc(np.column_stack([x, y])).structure.kind
    'cpdag'
    """
    warn_experimental("hc")
    from andrey.search.hc import hc as _hc

    X = data_matrix(data)
    max_iter = int_in_range(max_iter, "max_iter", 1)
    labels = column_labels(data)
    cpdag, score = _hc(X, score_func=score_func, lambda_value=lambda_value, max_iter=max_iter)
    metadata = {"algorithm": "HC", "score": json_safe(score)}
    return structure_output(cpdag, labels=labels, metadata=metadata)


def exact_search(data: npt.ArrayLike, *, search_method: str = "astar") -> StructureOutput:
    """Learn the globally optimal DAG for the BIC score by exact search.

    The optimum is identified up to its Markov equivalence class. A* (``search_method="astar"``)
    [1]_ and dynamic programming (``"dp"``) [2]_ may return different representatives. Runtime is
    exponential in the variable count, so keep the problem small (roughly <= 20 variables).

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``.
    search_method : {"astar", "dp"}, default="astar"
        Exact search strategy: A* search or dynamic programming.

    Returns
    -------
    StructureOutput
        Wraps a ``dag`` :class:`~andrey.core.GraphStructure` (one representative of the optimal
        equivalence class); ``metadata["algorithm"]`` is ``"ExactSearch"``.

    Raises
    ------
    ValueError
        If ``search_method`` is not ``"astar"`` or ``"dp"``.

    Warns
    -----
    ExperimentalWarning
        On the first call in a process, because the method is experimental: it has no published
        benchmark, and its API may change without deprecation.

    Notes
    -----
    **Limitations.** Not benchmarked yet. Its tests run it on at most 5 variables with 1,000 rows.
    It scores every possible parent set of every variable, so its run time at least doubles with
    each added variable. The ``"dp"`` search also keeps one entry per subset of the variables, so
    its memory doubles as well.

    References
    ----------
    .. [1] Yuan, C., and Malone, B. (2013). Learning optimal Bayesian networks: a shortest path
       perspective. https://doi.org/10.1613/jair.4039
    .. [2] Silander, T., and Myllymaki, P. (2006). A simple approach for finding the globally
       optimal Bayesian network structure. https://arxiv.org/abs/1206.6875

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import exact_search
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal(500)
    >>> y = x + 0.1 * rng.standard_normal(500)
    >>> z = y + 0.1 * rng.standard_normal(500)
    >>> exact_search(np.column_stack([x, y, z])).structure.kind
    'dag'
    """
    warn_experimental("exact_search")
    from andrey.search.exact import exact_search as _exact

    X = data_matrix(data)
    labels = column_labels(data)
    dag = _exact(X, search_method=search_method)
    return structure_output(dag, labels=labels, metadata={"algorithm": "ExactSearch"})


def calm(data: npt.ArrayLike, *, seed: int | None = None, **params: Any) -> StructureOutput:
    """Learn a weighted DAG with CALM, a continuous-optimization structure learner.

    CALM (Continuous and Acyclicity-constrained L0-penalized likelihood with an estimated Moral
    graph) [1]_ fits a linear-Gaussian DAG with torch. By default it keeps only the candidate edges
    of an estimated moral graph, then minimizes the Gaussian likelihood plus a penalty per edge
    under an acyclicity constraint. The optimizer draws random numbers and is seeded on every call;
    see ``seed``.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``.
    seed : int or None, default=None
        Seed for the optimizer's torch random number generator. ``None`` uses the
        ``ANDREY_SEED`` environment variable, or ``0`` when it is unset, so every call is seeded
        and repeated calls return the same graph on one machine with the same torch version.
        ``metadata["seed"]`` records the seed used.
    **params : Any
        Optimizer settings passed to the solver; a name not listed here raises ``TypeError``.

        - ``lambda1`` (default ``0.005``): penalty per edge. Larger values give sparser graphs.
        - ``alpha`` (default ``0.01``): significance level of the Fisher-Z tests that estimate the
          moral graph. Only pairs it keeps can become edges; smaller values keep fewer.
        - ``use_moral_graph`` (default ``True``): ``False`` makes every pair a candidate edge.
        - ``standardize`` (default ``False``): scale each column to unit variance first.
        - ``w_threshold`` (default ``0.3``) and ``gate_threshold`` (default ``0.5``): an edge is
          kept when its weight magnitude and its gate probability reach these values.
        - ``max_outer`` (default ``25``), ``subproblem_iter`` (default ``3000``), and ``lr``
          (default ``0.01``): augmented-Lagrangian rounds, Adam steps per round, and learning
          rate.
        - ``tau`` (default ``0.5``), ``rho_init`` (default ``0.001``), ``rho_mult`` (default
          ``3.0``), ``rho_max`` (default ``1e16``), and ``h_tol`` (default ``1e-8``): gate
          temperature and the acyclicity-penalty schedule.
        - ``labels`` (default ``None``): node names; a DataFrame's column names take precedence.

    Returns
    -------
    StructureOutput
        Wraps a ``dag`` :class:`~andrey.core.GraphStructure`; ``weighted_adjacency[i, j]`` is the
        coefficient of edge ``i -> j``. ``metadata`` holds ``"algorithm"`` (``"CALM"``), the
        ``"seed"`` used, ``"n_edges"``, ``"objective"`` (half the sum of the log residual
        variances plus ``lambda1`` times the edge count; lower is better), and ``"h_final"`` (the
        acyclicity residual of the returned graph, ``0`` when it is acyclic).

    Raises
    ------
    ImportError
        If the ``[torch]`` extra is not installed.
    TypeError
        If ``params`` holds a name the solver does not accept.

    Warns
    -----
    ExperimentalWarning
        On the first call in a process, because the method is experimental: it has no published
        benchmark, and its API may change without deprecation.

    Notes
    -----
    **Limitations.** Not benchmarked yet. Its tests run it on at most 6 variables with 800 rows, and
    on 2,000 rows with 5 variables. Every optimizer step computes a determinant and a matrix power
    of the full weight matrix, one row and one column per variable, so a step's cost grows with the
    cube of the variable count; the defaults allow up to 75,000 steps (``max_outer`` times
    ``subproblem_iter``).

    References
    ----------
    .. [1] Jin, K., Ng, I., Zhang, K., and Huang, B. (2026). Revisiting differentiable structure
       learning: inconsistency of l1 penalty and beyond. AAAI 40(27), 22399-22407.
       https://doi.org/10.1609/aaai.v40i27.39398

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import calm
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal(500)
    >>> y = x + 0.1 * rng.standard_normal(500)
    >>> out = calm(np.column_stack([x, y]), seed=0)  # doctest: +SKIP
    >>> out.structure.kind  # doctest: +SKIP
    'dag'
    """
    warn_experimental("calm")
    if backend.torch() is None:
        raise ImportError("andrey.calm needs torch for its continuous optimiser; install '[torch]'")
    from andrey.search.calm import calm as _calm

    X = data_matrix(data)
    labels = column_labels(data)
    used_seed = resolve_seed(seed)  # CALM uses this seed for torch.
    result = _calm(X, seed=used_seed, **params)
    return structure_output(
        result.dag,
        labels=labels,
        weighted_adjacency=result.weighted_adjacency,
        metadata={
            "algorithm": "CALM",
            "seed": used_seed,
            "n_edges": int((result.weighted_adjacency != 0).sum()),
            "h_final": json_safe(result.h_final),
            "objective": json_safe(result.objective),
        },
    )
