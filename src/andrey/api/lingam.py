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

from andrey.api._adapt import column_labels, dag_from_adjacency, shared_labels, structure_output
from andrey.api._experimental import warn_experimental

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


def direct_lingam(
    data: npt.ArrayLike, *, random_state: int | None = None, measure: str = "pwling"
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

    Returns
    -------
    StructureOutput
        Wraps a ``dag`` :class:`~andrey.core.GraphStructure`; ``metadata["algorithm"]`` is
        ``"DirectLiNGAM"``. ``ordering`` is the causal order as a tuple of column indices, causes
        first. ``weighted_adjacency[i, j]`` is the coefficient of edge ``i -> j``, estimated by
        adaptive Lasso; the DAG has an edge exactly where the coefficient is nonzero.

    Raises
    ------
    NotImplementedError
        If ``measure`` is not ``"pwling"``.
    ValueError
        If ``data`` is not a 2-D numeric matrix or holds ``NaN`` / ``inf``, or if a DataFrame's
        column names repeat.

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
    number of variables. On the published benchmarks, a 200-variable fit took minutes on a CPU. With
    the ``torch`` extra and a CUDA GPU, entropy computations move to the GPU once they are large
    enough.

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

    X = np.asarray(data, dtype=np.float64)
    labels = column_labels(data)
    if X.ndim != 2:
        raise ValueError(
            f"direct_lingam expects a 2-D (n_samples, n_variables) matrix, got {X.ndim}-D"
        )
    order, B = _native(X, measure=measure)
    weighted = _canonical_weights(B)
    return structure_output(
        dag_from_adjacency(weighted),
        labels=labels,
        ordering=order,
        weighted_adjacency=weighted,
        metadata={"algorithm": "DirectLiNGAM"},
    )


def ica_lingam(
    data: npt.ArrayLike, *, random_state: int | None = None, max_iter: int = 1000
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

    Returns
    -------
    StructureOutput
        Wraps a ``dag`` :class:`~andrey.core.GraphStructure`; ``metadata["algorithm"]`` is
        ``"ICALiNGAM"``. ``ordering`` is the causal order as a tuple of column indices, causes
        first. ``weighted_adjacency[i, j]`` is the coefficient of edge ``i -> j``, estimated by
        adaptive Lasso; the DAG has an edge exactly where the coefficient is nonzero.

    Raises
    ------
    RuntimeError
        If no causal order can be read from the FastICA result.
    ValueError
        If ``data`` is not a 2-D numeric matrix or holds ``NaN`` / ``inf``, if ``max_iter`` is
        less than ``1``, if ``random_state`` is outside ``0`` to ``2**32 - 1``, or if a
        DataFrame's column names repeat.

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

    References
    ----------
    .. [1] Shimizu, S., Hoyer, P. O., Hyvarinen, A., and Kerminen, A. (2006). A linear
       non-Gaussian acyclic model for causal discovery. JMLR 7, 2003-2030.

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

    X = np.asarray(data, dtype=np.float64)
    labels = column_labels(data)
    order, B = _native(X, random_state=random_state, max_iter=max_iter)
    weighted = _canonical_weights(B)
    return structure_output(
        dag_from_adjacency(weighted),
        labels=labels,
        ordering=order,
        weighted_adjacency=weighted,
        metadata={"algorithm": "ICALiNGAM"},
    )


def multi_group_direct_lingam(
    data_groups: Sequence[npt.ArrayLike], *, random_state: int | None = None
) -> list[StructureOutput]:
    """Jointly learn one fully oriented DAG per group with multi-group DirectLiNGAM.

    All groups share the same ``d`` variables. The estimator exploits non-Gaussianity across the
    datasets to fix a single shared causal order, then estimates a separate weighted adjacency for
    each group [1]_. Use it when several datasets share one causal order but may differ in edge
    weights.

    Parameters
    ----------
    data_groups : sequence of array-like, each of shape (n_samples, n_variables)
        Two or more datasets over the same ``d`` variables; sample counts may differ per group.
        Each is coerced to ``float64``. DataFrames must name their columns alike; the names become
        the structures' ``labels``.
    random_state : int or None, default=None
        Ignored. The method is deterministic; the argument exists so that
        ``multi_group_direct_lingam`` accepts ``random_state`` as the other LiNGAM methods do.

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
    ValueError
        If fewer than two groups are given, if the groups disagree on the variable count or on
        their column names, if any matrix is not 2-D numeric or holds ``NaN`` / ``inf``, or if a
        DataFrame's column names repeat.

    Warns
    -----
    ExperimentalWarning
        On the first call in a process, because the method is experimental: it has no published
        benchmark, and its API may change without deprecation.

    References
    ----------
    .. [1] Shimizu, S. (2012). Joint estimation of linear non-Gaussian acyclic models.
       Neurocomputing 81, 104-107.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import multi_group_direct_lingam
    >>> rng = np.random.default_rng(0)
    >>> def group():
    ...     x = rng.uniform(-1, 1, 500)
    ...     y = 0.8 * x + rng.uniform(-1, 1, 500)
    ...     return np.column_stack([x, y])
    >>> outs = multi_group_direct_lingam([group(), group()])
    >>> len(outs), outs[0].structure.kind
    (2, 'dag')
    """
    warn_experimental("multi_group_direct_lingam")
    from andrey.lingam.multi_group import multi_group_direct_lingam as _native

    groups = [np.asarray(X, dtype=np.float64) for X in data_groups]
    labels = shared_labels(column_labels(X) for X in data_groups)
    order, adjacency_matrices = _native(groups)  # one shared order + a per-group weighted adjacency
    n_groups = len(groups)
    outputs = []
    for k, B in enumerate(adjacency_matrices):
        weighted = _canonical_weights(B)
        outputs.append(
            structure_output(
                dag_from_adjacency(weighted),
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
