"""LiNGAM-family facade adapters.

Each wraps a fitted LiNGAM estimator's causal order + weighted adjacency into a ``StructureOutput``.
The LiNGAM coefficient matrix ``B`` follows ``x_i = sum_j B[i, j] x_j``, so ``B[i, j]`` weights edge
``j -> i`` -- the transpose of a standard adjacency. Andrey exposes one convention:
``weighted_adjacency[i, j]`` is the weight of edge ``i -> j`` (row = source), matching
``GraphStructure`` and every other adapter -- so the adapter stores ``B.T`` and builds the ``dag``
from it (:func:`_canonical_weights`). Each estimator takes its own ``random_state``, resolved from
the explicit argument else the global ``ANDREY_SEED``.

``multi_group_direct_lingam`` is the one multi-output member: joint estimation over a list of
datasets yields one shared causal order plus a *per-group* weighted adjacency, so it returns a
``list[StructureOutput]`` -- one envelope per group, each with the shared ordering and that group's
own DAG + weights.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from andrey.api._adapt import (
    column_labels,
    dag_from_adjacency,
    data_matrix,
    int_in_range,
    shared_labels,
    structure_output,
)
from andrey.api._boundary import warn_past_boundary
from andrey.core.knowledge import Knowledge, resolve

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from andrey.core import StructureOutput


def _canonical_weights(B: npt.ArrayLike) -> np.ndarray:
    """Transpose a LiNGAM ``B`` to Andrey's canonical adjacency (``W[i, j]`` weights ``i -> j``).

    The LiNGAM coefficient ``B[i, j]`` is the weight of edge ``j -> i`` (from ``x_i = sum_j B[i, j]
    x_j``), so ``B.T`` gives the standard row-is-source adjacency the public API guarantees. The
    ``dag`` built from ``W`` and the stored ``weighted_adjacency`` then share one orientation.
    """
    return np.ascontiguousarray(np.asarray(B, dtype=np.float64).T)


def _pruning_rows(X: np.ndarray, name: str = "data") -> np.ndarray:
    """Return ``X``, checked to have the rows the adaptive-Lasso pruning step needs.

    The step regresses the last variable of the causal order on the other ``d - 1`` with a
    BIC-tuned Lasso, whose noise variance comes from a least-squares fit with an intercept. That fit
    needs more rows than its ``d`` coefficients, so ``X`` needs at least ``d + 1`` rows. Raises
    ``ValueError``, naming ``name``, with fewer.
    """
    n, d = X.shape
    if n <= d:
        raise ValueError(
            f"{name} needs at least {d + 1} rows (samples) over {d} variables, got {n}: the "
            f"pruning step regresses the last variable of the causal order on the other {d - 1} "
            "and estimates the noise variance from that least-squares fit, which needs more rows "
            "than coefficients"
        )
    return X


def direct_lingam(
    data: npt.ArrayLike,
    *,
    random_state: int | None = None,
    measure: str = "pwling",
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn a fully oriented DAG with DirectLiNGAM (linear, non-Gaussian noise).

    DirectLiNGAM uses non-Gaussianity - noise that is not Gaussian - to determine a complete causal
    order, then estimates linear edge weights. It returns a single fully oriented DAG (directed
    acyclic graph), with edge weights and a total ordering. PC and GES instead return an equivalence
    class.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``.
    random_state : int or None, default=None
        Ignored. DirectLiNGAM with ``measure="pwling"`` is deterministic; the argument exists so
        that ``direct_lingam`` accepts ``random_state`` as ``ica_lingam`` does.
    measure : {"pwling"}, default="pwling"
        How the order search picks the most exogenous remaining variable at each step. Only
        ``"pwling"``, the pairwise likelihood-ratio measure [2]_, is supported.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the pairs
        allowed to be adjacent, and temporal tiers, all honored. The causal order puts the cause of
        each required edge, and each earlier tier, first; the regression on that order drops the
        forbidden parents and does not penalize the required ones. A variable with a required parent
        takes least-squares weights on the parents it keeps; the others keep the adaptive Lasso's.
        Every required edge is in the DAG, even where its least-squares weight is ``0``.

    Returns
    -------
    StructureOutput
        Wraps a ``dag`` :class:`~andrey.core.GraphStructure`; ``metadata["algorithm"]`` is
        ``"DirectLiNGAM"``. ``ordering`` is the causal order as a tuple of column indices, causes
        first. ``weighted_adjacency[i, j]`` is the coefficient of edge ``i -> j``, estimated by
        adaptive Lasso, or by least squares for a copy (see Notes) and for a variable with a
        required parent; the DAG has an edge where the coefficient is nonzero and at every required
        edge.

    Raises
    ------
    NotImplementedError
        If ``measure`` is not ``"pwling"``.
    TypeError
        If ``knowledge`` is neither ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If ``data`` is not a usable 2-D matrix: not numeric, holding ``NaN`` or ``inf``, or with a
        constant column, named in the message. Also if it has fewer than ``n_variables + 1`` rows,
        the fewest with which the pruning step can estimate the edge weights. Also if ``knowledge``
        names a column ``data`` lacks or contradicts itself, as :class:`~andrey.Knowledge` lists.

    See Also
    --------
    andrey.pc, andrey.ges : Return the equivalence class, which is what Gaussian noise identifies.
    andrey.fci : Allows unmeasured common causes and selection bias.

    Notes
    -----
    **When to use it.** DirectLiNGAM suits continuous data with linear effects, independent
    non-Gaussian noise, and no unmeasured common causes [1]_. These assumptions allow the causal
    order to be identified. The result includes a DAG with every edge oriented, the edge weights,
    and the order.

    **When not to.** With Gaussian noise, the order cannot be identified, so the returned order is
    unreliable. Use :func:`~andrey.pc` or :func:`~andrey.ges`, which return the equivalence class.
    When unmeasured common causes are plausible, use :func:`~andrey.fci`.

    **Cost.** Each step compares the remaining variables in pairs, so runtime grows quickly with the
    number of variables. With the ``torch`` extra and a CUDA GPU, entropy computations move to the
    GPU once they are large enough.

    **Ties.** Two candidates that are interchangeable in the sample, such as two binary columns
    with the same number of ones, have equal scores up to rounding, which differs between CPUs.
    Each step treats candidates as tied when the square roots of their negated scores lie within
    ``1e-10`` of the best one, and picks among them by a rank computed from each column's own data,
    as :func:`~andrey.hc` does. The :doc:`column-order guide </docs/guides/column-order>` says what
    the rank keeps under a new column order and on another machine.

    **Copies.** Two columns are copies when one leaves at most ``1e-12`` of the other's variance
    unexplained, as for a column and a rescaled or rank-reversed copy of it. Each step scores one
    column per group of copies, the one that ranks first by the same rank, and when it is picked
    the other copies follow it directly in the order. Each of them gets a single edge from it,
    weighted by the scale between the two, and no edges to later columns, which take their edges
    from the column it copies.

    **Missing values.** A ``NaN`` in ``data`` raises ``ValueError``: the causal order compares the
    residuals of every variable over the same rows, so a row with a missing value has no place in
    it. Drop the incomplete rows first, which suits values missing completely at random, or use
    :func:`~andrey.pc` or :func:`~andrey.fci`, whose tests delete rows per test.

    **Limitations.** Measured on simulated linear data with uniform noise from Erdos-Renyi graphs
    with no hidden variable, on 4 cores with a cap of 1 h and 28 GB per fit. The largest sizes
    through which its median errors (SHD) stayed below the empty graph's, and through which its fits
    finished within the caps, were 400 and 400 variables at mean degree 1, 2, 4, or 8 with 2,000
    rows; 200 and 200 at mean degree 2 with 10,000 rows; and 200 and 400 at mean degree 2 with 500
    rows. The ladders ended when three fits of the next size ran past 1 h, except at mean degree 2
    with 500 rows, where the ladder ended when three fits of the next size, with more variables than
    rows, stopped with an error. The median fit took 2,211 s at 400 variables (mean degree 8, 2,000
    rows). Measured on simulated data from scale-free graphs with linear effects and uniform noise,
    with ten rows per variable. From 20 to 200 variables its median errors (SHD) stayed below the
    empty graph's, 13 against 199 at 200, and the median fit took 0.19 s at 20 variables and 545 s
    at 200, on 16 CPU cores. In a separate run on shared nodes, a CPU fit at 300 variables did not
    finish its warm-up and timed fit within 3,600 s; with an NVIDIA A100 GPU, a fit took 528 s at
    300 variables and 1,410 s at 400. The numbers are a reference, not a hard limit: they change
    with the data generator, the density, the method's settings, and the machine.

    References
    ----------
    .. [1] Shimizu, S., Inazumi, T., Sogawa, Y., Hyvarinen, A., Kawahara, Y., Washio, T., Hoyer,
       P. O., and Bollen, K. (2011). DirectLiNGAM: A direct method for learning a linear
       non-Gaussian structural equation model. JMLR 12, 1225-1248.
    .. [2] Hyvarinen, A., and Smith, S. M. (2013). Pairwise likelihood ratios for estimation of
       non-Gaussian structural equation models. JMLR 14, 111-152.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import direct_lingam
    >>> rng = np.random.default_rng(0)
    >>> x = rng.uniform(-1, 1, 500)
    >>> y = 0.8 * x + rng.uniform(-1, 1, 500)
    >>> out = direct_lingam(np.column_stack([x, y]))
    >>> out.structure.kind, out.ordering
    ('dag', (0, 1))
    """
    from andrey.lingam.direct import direct_lingam as _native

    X = _pruning_rows(data_matrix(data, method="direct_lingam"))
    labels = column_labels(data)
    constraints = resolve(knowledge, X.shape[1], labels, method="direct_lingam")
    warn_past_boundary("direct_lingam", X.shape)
    order, B = _native(X, measure=measure, constraints=constraints)
    weighted = _canonical_weights(B)
    edges = weighted != 0 if constraints is None else (weighted != 0) | constraints.required
    return structure_output(
        dag_from_adjacency(edges),
        labels=labels,
        ordering=order,
        weighted_adjacency=weighted,
        metadata={"algorithm": "DirectLiNGAM"},
    )


def ica_lingam(
    data: npt.ArrayLike,
    *,
    random_state: int | None = None,
    max_iter: int = 1000,
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn a fully oriented DAG with ICA-LiNGAM (linear, non-Gaussian noise).

    ICA-LiNGAM uses FastICA to estimate an unmixing matrix, which separates mixed signals. It
    reorders this matrix into a causal order [1]_. It estimates edge weights by regressing each
    variable on those before it, using adaptive Lasso. Like DirectLiNGAM, it returns one weighted
    DAG (directed acyclic graph) with a total ordering, rather than an equivalence class. FastICA
    starts from a random point, so the order can change with ``random_state``.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``.
    random_state : int or None, default=None
        Seed for FastICA's starting point, from ``0`` to ``2**32 - 1``. ``None`` uses the
        ``ANDREY_SEED`` environment variable, or ``0`` when it is unset.
    max_iter : int, default=1000
        Maximum FastICA iterations; at least ``1``.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the pairs
        allowed to be adjacent, and temporal tiers, all honored. The order search reads the unmixing
        weight of every edge the knowledge forbids as ``0``, the reverse of each required edge and
        every edge into an earlier tier included, and puts the cause of each required edge, and each
        earlier tier, first. The regression on that order drops the forbidden parents and does not
        penalize the required ones. A variable with a required parent takes least-squares weights on
        the parents it keeps; the others keep the adaptive Lasso's. Every required edge is in the
        DAG, even where its least-squares weight is ``0``.

    Returns
    -------
    StructureOutput
        Wraps a ``dag`` :class:`~andrey.core.GraphStructure`; ``metadata["algorithm"]`` is
        ``"ICALiNGAM"``. ``ordering`` is the causal order as a tuple of column indices, causes
        first. ``weighted_adjacency[i, j]`` is the coefficient of edge ``i -> j``, estimated by
        adaptive Lasso, or by least squares for a variable with a required parent; the DAG has an
        edge where the coefficient is nonzero and at every required edge.

    Raises
    ------
    RuntimeError
        If no causal order can be read from the FastICA result.
    TypeError
        If ``max_iter`` is not an int, or ``knowledge`` is neither ``None`` nor a
        :class:`~andrey.Knowledge`.
    ValueError
        If ``max_iter`` is less than ``1``, if ``random_state`` is outside ``0`` to
        ``2**32 - 1``, or if ``data`` has fewer than ``n_variables + 1`` rows, the fewest with which
        the pruning step can estimate the edge weights. Also if ``knowledge`` names a column
        ``data`` lacks or contradicts itself, as :class:`~andrey.Knowledge` lists.

    Warns
    -----
    sklearn.exceptions.ConvergenceWarning
        When FastICA does not converge within ``max_iter`` iterations.

    See Also
    --------
    andrey.direct_lingam : The same model with a deterministic order search; it recovered most
        edges on the published benchmarks.

    Notes
    -----
    **When to use it.** ICA-LiNGAM assumes linear effects, independent non-Gaussian noise, and no
    unmeasured common causes, like :func:`~andrey.direct_lingam`. On the published benchmark data,
    it scored worse than the empty graph at the benchmarks' sample size, while DirectLiNGAM
    recovered most edges. Prefer :func:`~andrey.direct_lingam` unless ICA is needed.

    **Choosing random_state and max_iter.** Different values of ``random_state`` can give different
    orders; compare a few when the result matters. If FastICA reaches ``max_iter`` before
    converging, scikit-learn issues a ``ConvergenceWarning``; then raise ``max_iter``.

    **Column order and machines.** When several variables can come next in the causal order, the one
    whose column ranks first by its own values goes first. FastICA itself can reach another solution
    under a new column order or on another machine when it converges slowly, at few rows per
    variable; the :doc:`column-order guide </docs/guides/column-order>` gives the measured limit.

    **Missing values.** A ``NaN`` in ``data`` raises ``ValueError``: the independent components
    unmix every variable over the same rows, so a row with a missing value has no place in them.
    Drop the incomplete rows first, which suits values missing completely at random, or use
    :func:`~andrey.pc` or :func:`~andrey.fci`, whose tests delete rows per test.

    **Limitations.** Measured on simulated linear data with uniform noise from Erdos-Renyi graphs
    with no hidden variable, on 4 cores with a cap of 1 h and 28 GB per fit. The largest sizes
    through which its median errors (SHD) stayed below the empty graph's, and through which its fits
    finished within the caps, were 50 and 400 variables at mean degree 1 or 2 with 2,000 rows; 100
    and 400 at mean degree 2 with 10,000 rows; 20 and 400 at mean degree 2 with 500 rows and mean
    degree 4 with 2,000 rows; and none and 400 at mean degree 8 with 2,000 rows. The ladders ended
    when three fits of the next size ran past 1 h, except at mean degree 2 with 500 rows, where the
    ladder ended when three fits of the next size, with more variables than rows, stopped with an
    error, and at mean degree 8 with 2,000 rows, where the ladder ended after two sizes in a row did
    not beat the empty graph. The accuracy stop at mean degree 8 with 2,000 rows was set after
    larger sizes had run. The median fit took 516 s at 400 variables (mean degree 2, 10,000 rows).
    Measured on simulated data from scale-free graphs with linear effects and uniform noise, from 20
    to 100 variables with ten rows per variable: its median errors (SHD) passed the empty graph's at
    every size, 27 against 19 at 20 variables and 181 against 99 at 100. The median fit took 0.44 s
    at 20 variables and 15.8 s at 100, on 16 cores. On Sachs's protein-signaling data (853 rows, 11
    variables), its errors were below the empty graph's (14 against 17). The numbers are a
    reference, not a hard limit: they change with the data generator, the density, the method's
    settings, and the machine. Its causal order depends on the columns' scale, as its authors note
    [2]: matching the rows of the unmixing matrix to the variables, and pruning its coefficients
    from the smallest, both read the data's units, and the matching assumes every noise term has
    the same scale [1]. On simulated linear data with uniform noise and weights between 0.5 and 2
    whose noise terms all have unit variance, at 50 variables, mean degree 4 and 2,000 rows, its
    errors (SHD) summed over 5 datasets were 9 on the columns as drawn and 1,784 on the same columns
    standardized, against 500 for the empty graph. `direct_lingam` finds the order in a way that
    does not depend on the scale [2].

    References
    ----------
    .. [1] Shimizu, S., Hoyer, P. O., Hyvarinen, A., and Kerminen, A. (2006). A linear
       non-Gaussian acyclic model for causal discovery. JMLR 7, 2003-2030.
    .. [2] Shimizu, S., Inazumi, T., Sogawa, Y., Hyvarinen, A., Kawahara, Y., Washio, T., Hoyer,
       P. O., and Bollen, K. (2011). DirectLiNGAM: A direct method for learning a linear
       non-Gaussian structural equation model. JMLR 12, 1225-1248.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import ica_lingam
    >>> rng = np.random.default_rng(0)
    >>> x = rng.uniform(-1, 1, 500)
    >>> y = 0.8 * x + rng.uniform(-1, 1, 500)
    >>> out = ica_lingam(np.column_stack([x, y]), random_state=0)
    >>> out.structure.kind
    'dag'
    """
    from andrey.lingam.ica import ica_lingam as _native

    X = _pruning_rows(data_matrix(data, method="ica_lingam"))
    max_iter = int_in_range(max_iter, "max_iter", 1)
    labels = column_labels(data)
    constraints = resolve(knowledge, X.shape[1], labels, method="ica_lingam")
    warn_past_boundary("ica_lingam", X.shape)
    order, B = _native(X, random_state=random_state, max_iter=max_iter, constraints=constraints)
    weighted = _canonical_weights(B)
    edges = weighted != 0 if constraints is None else (weighted != 0) | constraints.required
    return structure_output(
        dag_from_adjacency(edges),
        labels=labels,
        ordering=order,
        weighted_adjacency=weighted,
        metadata={"algorithm": "ICALiNGAM"},
    )


def multi_group_direct_lingam(
    data_groups: Sequence[npt.ArrayLike],
    *,
    random_state: int | None = None,
    knowledge: Knowledge | None = None,
) -> list[StructureOutput]:
    """Jointly learn one fully oriented DAG per group with multi-group DirectLiNGAM.

    All groups share the same ``d`` variables. The estimator exploits non-Gaussianity across the
    datasets to fix a single shared causal order, then estimates a separate weighted adjacency for
    each group [1]_. It applies when several datasets share one causal order but may differ in
    edge weights.

    Parameters
    ----------
    data_groups : sequence of array-like, each of shape (n_samples, n_variables)
        Two or more datasets over the same ``d`` variables; sample counts may differ per group.
        Each is coerced to ``float64``. DataFrames must name their columns alike; the names become
        the structures' ``labels``.
    random_state : int or None, default=None
        Ignored. The method is deterministic; the argument exists so that
        ``multi_group_direct_lingam`` accepts ``random_state`` as the other LiNGAM methods do.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the pairs
        allowed to be adjacent, and temporal tiers, all honored. It holds for every group. The
        shared causal order puts the cause of each required edge, and each earlier tier, first. The
        regression on that order drops the forbidden parents and does not penalize the required
        ones. A variable with a required parent takes least-squares weights on the parents it keeps;
        the others keep the adaptive Lasso's. Every required edge is in every group's DAG, even
        where its least-squares weight is ``0``.

    Returns
    -------
    list[StructureOutput]
        One envelope per group, in input order. Each wraps a ``dag``
        :class:`~andrey.core.GraphStructure`; all share the same ``ordering`` (the joint causal
        order) while carrying that group's own ``weighted_adjacency`` (``W[i, j]`` weights
        ``i -> j``). ``metadata["algorithm"]`` is ``"MultiGroupDirectLiNGAM"``, with ``group_index``
        and ``n_groups``.

    Raises
    ------
    TypeError
        If ``knowledge`` is neither ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If fewer than two groups are given, if the groups disagree on the variable count or on
        their column names, or if a group is not a usable 2-D matrix or has fewer than
        ``n_variables + 1`` rows, the fewest with which the pruning step can estimate the edge
        weights. Also if ``knowledge`` names a column the groups lack or contradicts itself, as
        :class:`~andrey.Knowledge` lists.

    Notes
    -----
    **Ties.** Each step sums the groups' scores and breaks ties as :func:`~andrey.direct_lingam`
    does, ranking each column by its data stacked over the groups. Columns that are copies in any
    group are handled as copies are in :func:`~andrey.direct_lingam`; a copy gets its single edge
    only in the groups where it is one.

    **Missing values.** A ``NaN`` in a dataset raises ``ValueError``: the causal order compares the
    residuals of every variable over the same rows, so a row with a missing value has no place in
    it. Drop the incomplete rows first, which suits values missing completely at random, or use
    :func:`~andrey.pc` or :func:`~andrey.fci`, whose tests delete rows per test.

    **Limitations.** Measured on simulated linear data with uniform noise in 3 groups that share one
    Erdos-Renyi graph, on 4 cores with a cap of 1 h and 28 GB per fit. The largest sizes through
    which its median errors (SHD) stayed below the empty graph's, and through which its fits
    finished within the caps, were 200 and 200 variables at mean degree 2 or 4 with 1,000 or 5,000
    rows per group. Every ladder ended at the largest planned size. The median fit took 1,996 s at
    200 variables (mean degree 4, 5,000 rows per group, 3 groups). It assumes linear effects and
    non-Gaussian noise in every group, and it gives every group the one shared causal order, even
    when a group's own order differs. The numbers are a reference, not a hard limit: they change
    with the data generator, the density, the method's settings, and the machine.

    References
    ----------
    .. [1] Shimizu, S. (2012). Joint estimation of linear non-Gaussian acyclic models.
       Neurocomputing 81, 104-107.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import multi_group_direct_lingam
    >>> rng = np.random.default_rng(0)
    >>> def group(weight):
    ...     x = rng.uniform(-1, 1, 1000)
    ...     y = weight * x + rng.uniform(-1, 1, 1000)
    ...     return np.column_stack([y, x])
    >>> outs = multi_group_direct_lingam([group(0.8), group(-1.5)])
    >>> [out.ordering for out in outs]
    [(1, 0), (1, 0)]
    >>> [round(float(out.weighted_adjacency[1, 0]), 1) for out in outs]
    [0.8, -1.5]
    """
    from andrey.lingam.multi_group import multi_group_direct_lingam as _native

    groups = [
        _pruning_rows(
            data_matrix(X, name=f"data_groups[{k}]", method="multi_group_direct_lingam"),
            name=f"data_groups[{k}]",
        )
        for k, X in enumerate(data_groups)
    ]
    if len(groups) < 2:
        raise ValueError(f"data_groups must hold at least two datasets, got {len(groups)}")
    if len({X.shape[1] for X in groups}) > 1:
        widths = [X.shape[1] for X in groups]
        raise ValueError(f"every dataset in data_groups needs the same columns, got {widths}")
    labels = shared_labels(column_labels(X) for X in data_groups)
    constraints = resolve(knowledge, groups[0].shape[1], labels, method="multi_group_direct_lingam")
    order, adjacency_matrices = _native(groups, constraints=constraints)
    n_groups = len(groups)
    outputs = []
    for k, B in enumerate(adjacency_matrices):
        weighted = _canonical_weights(B)
        edges = weighted != 0 if constraints is None else (weighted != 0) | constraints.required
        outputs.append(
            structure_output(
                dag_from_adjacency(edges),
                labels=labels,
                ordering=order,
                weighted_adjacency=weighted,
                metadata={
                    "algorithm": "MultiGroupDirectLiNGAM",
                    "group_index": k,
                    "n_groups": n_groups,
                },
            )
        )
    return outputs
