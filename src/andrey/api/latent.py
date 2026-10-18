"""Latent-variable family facade: GIN.

The callable runs the GIN engine and returns one uniform ``StructureOutput``. GIN recovers a
linear non-Gaussian latent-variable model from the observed data alone, so the wrapped graph spans
both the observed indicators and the discovered latents: each latent points to its observed
children, an ordered latent also points to every later latent, and ``node_types`` flags which
nodes are latent. The recovered clustering and latent causal order are stored in ``metadata``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from andrey.api._adapt import column_labels, data_matrix, open_unit_interval

if TYPE_CHECKING:
    import numpy.typing as npt

    from andrey.core import StructureOutput


def gin(
    data: npt.ArrayLike, *, alpha: float = 0.05, labels: tuple[str, ...] | None = None
) -> StructureOutput:
    """Learn a latent-variable DAG over observed indicators and their hidden latents with GIN.

    GIN exploits the Generalized Independent Noise condition in a linear non-Gaussian model to
    group observed variables under shared latent parents and to order those latents [1]_. It
    applies when the observed variables are noisy indicators of a few unobserved common causes
    rather than direct causes of one another.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed indicator data, one row per sample. Coerced to ``float64``.
    alpha : float, default=0.05
        Significance level of the pooled independence test, in ``(0, 1)``. A group of variables
        forms a cluster, and a latent is the next in the order, when the test's p-value is at
        least ``alpha``, so larger values make each test harder to pass.
    labels : tuple of str or None, default=None
        Optional names for the observed variables, one per column. Defaults to a DataFrame's
        column names, else ``X1..Xn``. Latents are auto-named ``L1..Lm``.

    Returns
    -------
    StructureOutput
        Wraps a ``dag`` :class:`~andrey.core.GraphStructure` over the observed variables (indices
        ``0`` to ``n_variables - 1``) and one latent per cluster, named ``L1`` to ``Lm``: the
        ordered latents first, in causal order, then the unordered ones; ``structure.node_types``
        flags the latents. Each latent points to the observed variables in its cluster. An ordered
        latent also points to every later latent and to every unordered one, and unordered latents
        have no edge between them: edges between latents follow the order only and do not mean a
        direct effect was found. An observed variable in no cluster has no edges.
        ``metadata["algorithm"]`` is ``"GIN"``, ``metadata["clusters"]`` lists the clusters as
        lists of column indices, ``metadata["causal_order"]`` lists the ordered ones earliest
        latent first, ``metadata["unordered"]`` lists the rest by smallest member, and
        ``metadata["n_latents"]`` is the latent count.

    Raises
    ------
    ValueError
        If ``alpha`` is not in ``(0, 1)``, or if ``labels`` does not give one name per column of
        ``data``.

    Notes
    -----
    A group of observed variables forms a cluster when the linear combination of the group with
    the smallest covariance with the other variables not yet in a cluster is independent of each of
    them; groups grow from pairs upward, and clusters that share a variable merge. The latents are
    then ordered one at a time, given the latents already ordered: the next latent is a root, whose
    cluster passes the independence test at ``alpha`` paired with each other remaining cluster [1]_.
    When several candidates pass, the one whose lowest p-value is highest is taken, so latents with
    no causal path between them are still ordered, arbitrarily; equal p-values are decided by each
    column's rank, as in :func:`~andrey.hc`, so reordering whole clusters does not change the
    choice. When none passes, the ordering stops and the remaining latents stay unordered. A single
    remaining latent is ordered last without a test. Each comparison rejects a true root with
    probability about ``alpha``, the test's nominal size, so a correct model can still leave the
    latents from that step on unordered.

    **Limitations.** Measured on simulated linear measurement models with uniform noise and 1,000
    rows: 2 to 4 latents, each a cause of every later one and measured by three indicators of its
    own. With 2 and 3 latents its clusters matched the true ones (median adjusted Rand index 1);
    with 4 latents (12 indicators) the median index was 0.89, and a median of one latent was missed.
    The median fit took 3.5 s with 6 indicators, 15 s with 9, and 38 s with 12, on 4 cores. Given
    the true clusters of a chain of three latents and of a complete three-latent model, with 1,000
    rows, the order stopped early on 1 of 10 seeds for each, and no latent was put after one it
    causes. It tests every group of observed variables of each size, from pairs upward, so the
    number of tests grows combinatorially with the variable count, and each HSIC test's time and
    memory grow with the square of the row count. Clusters have at least two variables, so a latent
    with a single indicator is not found. Each cluster gets one latent, so indicators that share two
    or more latents appear as one. It assumes every observed variable measures a latent: variables
    that depend on no other variable pass the cluster test trivially and are grouped into one
    cluster.

    References
    ----------
    .. [1] Xie, F., Cai, R., Huang, B., Glymour, C., Hao, Z., and Zhang, K. (2020). Generalized
       independent noise condition for estimating latent variable causal graphs. Advances in
       Neural Information Processing Systems 33 (NeurIPS 2020).

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import gin
    >>> rng = np.random.default_rng(0)
    >>> latent = rng.uniform(-1, 1, 500)
    >>> data = np.column_stack([
    ...     0.8 * latent + 0.1 * rng.uniform(-1, 1, 500),
    ...     1.2 * latent + 0.1 * rng.uniform(-1, 1, 500),
    ...     0.9 * latent + 0.1 * rng.uniform(-1, 1, 500),
    ... ])
    >>> out = gin(data)
    >>> out.metadata["clusters"], out.structure.labels
    ([[0, 1, 2]], ('X1', 'X2', 'X3', 'L1'))
    """
    from andrey.latent.gin import gin_structure

    X = data_matrix(data)
    alpha = open_unit_interval(alpha, "alpha")
    if labels is not None and len(labels) != X.shape[1]:
        raise ValueError(
            f"labels must name each of the {X.shape[1]} columns of data, got {len(labels)} names"
        )
    names = labels if labels is not None else column_labels(data)
    return gin_structure(X, alpha=alpha, labels=names)
