"""Permutation-based facade adapters.

``boss`` and ``grasp`` are native -- they run a permutation/score search on the :mod:`andrey.core`
compute substrate and return a CPDAG ``GraphStructure`` with its objective score. Both score
candidate parent additions through the batched Schur delta-BIC surface over the linear-Gaussian
BIC (:class:`andrey.core.score.BICScore`). Their search order is drawn from a private RNG seeded by
``seed`` (or the global ``ANDREY_SEED``), so a run is reproducible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from andrey.api._adapt import column_labels, json_safe, structure_output
from andrey.core.seeding import resolve_seed

if TYPE_CHECKING:
    import numpy.typing as npt

    from andrey.core import StructureOutput


def boss(
    data: npt.ArrayLike,
    *,
    score_func: str = "local_score_BIC_from_cov",
    lambda_value: float = 1.0,
    seed: int | None = None,
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
    score_func : {"local_score_BIC_from_cov"}, default="local_score_BIC_from_cov"
        Local score objective (covariance-based BIC). Only this option is supported.
    lambda_value : float, default=1.0
        Multiplier on the BIC penalty of ``log(n_samples)`` per parent. Larger values give
        sparser graphs; ``1.0`` is the standard BIC. Must be finite and non-negative.
    seed : int or None, default=None
        Seed for the random order in which the search visits variables. ``None`` uses the
        ``ANDREY_SEED`` environment variable, or ``0`` when it is unset, so repeated calls return
        the same graph. Different seeds can reach different graphs.

    Returns
    -------
    StructureOutput
        Wraps a ``cpdag`` :class:`~andrey.core.GraphStructure`; ``metadata["algorithm"]`` is
        ``"BOSS"`` and ``metadata["score"]`` the BIC score of the DAG read off the final order,
        the sum over variables of ``n * log(residual variance) + lambda_value * log(n) *
        n_parents``; lower is better. Directed marks (``->``) are oriented edges, undirected
        (``--``) the ones the equivalence class leaves open.

    Raises
    ------
    NotImplementedError
        If ``score_func`` is not ``"local_score_BIC_from_cov"``.
    ValueError
        If ``data`` is not a 2-D numeric matrix or holds ``NaN`` / ``inf``, if ``lambda_value``
        is negative or non-finite, or if a DataFrame's column names repeat.

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
    connected variables [1]_.

    **When not to.** When unmeasured common causes are plausible, use :func:`~andrey.fci`. When the
    noise is non-Gaussian and every edge should be oriented, use :func:`~andrey.direct_lingam`.

    **Choosing lambda_value.** As for :func:`~andrey.ges`, a larger value gives a sparser graph, and
    ``1.0`` is the standard BIC.

    **Choosing seed.** The search visits variables in a random order and stops when no move improves
    the order. Different seeds can therefore reach different graphs. When the result matters,
    compare a few seeds.

    **How it works.** BOSS searches variable orders for the best DAG with the lowest total score
    [1]_. Each variable uses a grow-shrink tree to cache its best parent set for any set of
    variables allowed to precede it. Each sweep moves every variable to the position that most
    improves the total score. Sweeps repeat until no move improves it. The final parent sets form a
    DAG, which is reduced to its CPDAG.

    A move or parent change must improve the score by more than ``1e-6``, a fixed absolute tolerance
    shared with :func:`~andrey.grasp`. Scores within this tolerance of the best count as ties. The
    earliest order position wins; among candidate parents, the lowest column index wins.

    References
    ----------
    .. [1] Andrews, B., Ramsey, J., Sanchez Romero, R., Camchong, J., and Kummerfeld, E. (2023).
       Fast Scalable and Accurate Discovery of DAGs Using the Best Order Score Search and Grow
       Shrink Trees. NeurIPS 36. https://doi.org/10.52202/075280-2794
    .. [2] Lam, W.-Y., Andrews, B., and Ramsey, J. (2022). Greedy Relaxations of the Sparsest
       Permutation Algorithm. UAI, PMLR 180:1052-1062. https://proceedings.mlr.press/v180/lam22a.html

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

    if score_func != "local_score_BIC_from_cov":
        raise NotImplementedError(f"BOSS supports 'local_score_BIC_from_cov', not {score_func!r}")
    X = np.asarray(data, dtype=np.float64)
    labels = column_labels(data)
    cpdag, score = _boss(X, lambda_value=lambda_value, random_state=resolve_seed(seed))
    metadata = {"algorithm": "BOSS", "score": json_safe(score)}
    return structure_output(cpdag, labels=labels, metadata=metadata)


def grasp(
    data: npt.ArrayLike,
    *,
    score_func: str = "local_score_BIC_from_cov",
    lambda_value: float = 1.0,
    depth: int = 3,
    seed: int | None = None,
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
    score_func : {"local_score_BIC_from_cov"}, default="local_score_BIC_from_cov"
        Local score objective (covariance-based BIC). Only this option is supported.
    lambda_value : float, default=1.0
        Multiplier on the BIC penalty of ``log(n_samples)`` per parent. Larger values give
        sparser graphs; ``1.0`` is the standard BIC. Must be finite and non-negative.
    depth : int, default=3
        Maximum number of tucks in one chain of moves. Only ``3`` is supported.
    seed : int or None, default=None
        Seed for the random starting order and the order in which the search visits variables.
        ``None`` uses the ``ANDREY_SEED`` environment variable, or ``0`` when it is unset, so
        repeated calls return the same graph. Different seeds can reach different graphs.

    Returns
    -------
    StructureOutput
        Wraps a ``cpdag`` :class:`~andrey.core.GraphStructure`; ``metadata["algorithm"]`` is
        ``"GRaSP"`` and ``metadata["score"]`` the BIC score of the DAG read off the final order,
        the sum over variables of ``n * log(residual variance) + lambda_value * log(n) *
        n_parents``; lower is better. Directed marks (``->``) are oriented edges, undirected
        (``--``) the ones the equivalence class leaves open.

    Raises
    ------
    NotImplementedError
        If ``score_func`` is not ``"local_score_BIC_from_cov"``, or ``depth`` is not ``3``.
    ValueError
        If ``data`` is not a 2-D numeric matrix or holds ``NaN`` / ``inf``, if ``lambda_value``
        is negative or non-finite, or if a DataFrame's column names repeat.

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
    search even on dense graphs [1]_.

    **When not to.** When unmeasured common causes are plausible, use :func:`~andrey.fci`. When the
    noise is non-Gaussian and every edge should be oriented, use :func:`~andrey.direct_lingam`.

    **GRaSP or BOSS.** Both search the same space of orders with different moves. Either can reach
    the better score on a given dataset. On the published benchmarks, GRaSP took far longer than
    :func:`~andrey.boss` at 400 variables.

    **Choosing lambda_value.** As for :func:`~andrey.ges`, a larger value gives a sparser graph, and
    ``1.0`` is the standard BIC.

    **Choosing seed.** Both the starting order and the order of visiting variables are random, so
    different seeds can reach different graphs. When the result matters, compare a few seeds.

    **How it works.** GRaSP searches variable orders for the best DAG with the lowest total score
    [1]_. Each variable uses a grow-shrink tree [2]_ to cache its best parent set for any set of
    variables allowed to precede it. The search starts from a shuffled order and changes it with
    tucks. A tuck on an edge ``x -> y`` moves ``y`` just before ``x``, along with its ancestors that
    sit between ``x`` and ``y``. A depth-first search keeps the first chain of up to ``depth`` tucks
    that improves the total score. It repeats until no chain improves it. The final parent sets form
    a DAG, which is reduced to its CPDAG.

    Tucks must improve the score by more than ``1e-6``, the fixed absolute tolerance used by
    :func:`~andrey.boss`. A tuck within this tolerance counts as leaving the score unchanged.

    References
    ----------
    .. [1] Lam, W.-Y., Andrews, B., and Ramsey, J. (2022). Greedy Relaxations of the Sparsest
       Permutation Algorithm. UAI, PMLR 180:1052-1062. https://proceedings.mlr.press/v180/lam22a.html
    .. [2] Andrews, B., Ramsey, J., Sanchez Romero, R., Camchong, J., and Kummerfeld, E. (2023).
       Fast Scalable and Accurate Discovery of DAGs Using the Best Order Score Search and Grow
       Shrink Trees. NeurIPS 36. https://doi.org/10.52202/075280-2794

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

    if score_func != "local_score_BIC_from_cov":
        raise NotImplementedError(f"GRaSP supports 'local_score_BIC_from_cov', not {score_func!r}")
    if depth != 3:
        raise NotImplementedError(f"GRaSP supports depth=3, not {depth!r}")
    X = np.asarray(data, dtype=np.float64)
    labels = column_labels(data)
    cpdag, score = _grasp(X, lambda_value=lambda_value, random_state=resolve_seed(seed))
    metadata = {"algorithm": "GRaSP", "score": json_safe(score)}
    return structure_output(cpdag, labels=labels, metadata=metadata)
