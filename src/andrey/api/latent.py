"""Latent-variable family facade: GIN.

The callable runs the GIN engine and returns one uniform ``StructureOutput``. GIN recovers a
linear non-Gaussian latent-variable model from the observed data alone, so the wrapped graph spans
both the observed indicators and the discovered latents: each latent points to its observed children
and to every later latent in causal order, and ``node_types`` flags which nodes are latent. The
recovered clustering and latent causal order are stored in ``metadata``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from andrey.api._adapt import column_labels
from andrey.api._experimental import warn_experimental

if TYPE_CHECKING:
    import numpy.typing as npt

    from andrey.core import StructureOutput


def gin(
    data: npt.ArrayLike, *, alpha: float = 0.05, labels: tuple[str, ...] | None = None
) -> StructureOutput:
    """Learn a latent-variable DAG over observed indicators and their hidden latents with GIN.

    GIN exploits the Generalized Independent Noise condition in a linear non-Gaussian model to
    group observed variables under shared latent parents and to order those latents [1]_. Reach
    for it when the observed variables are noisy indicators of a few unobserved common causes
    rather than direct causes of one another.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed indicator data, one row per sample. Coerced to ``float64``.
    alpha : float, default=0.05
        Significance level of the pooled independence test that finds the clusters: a group of
        variables forms a cluster when the test's p-value is at least ``alpha``, so larger values
        give fewer clusters.
    labels : tuple of str or None, default=None
        Optional names for the observed variables, one per column. Defaults to a DataFrame's
        column names, else ``X1..Xn``. Latents are auto-named ``L1..Lm``.

    Returns
    -------
    StructureOutput
        Wraps a ``dag`` :class:`~andrey.core.GraphStructure` over the observed variables (indices
        ``0`` to ``n_variables - 1``) and one latent per cluster, appended in causal order and
        named ``L1`` to ``Lm``; ``structure.node_types`` flags the latents. Each latent points to
        the observed variables in its cluster and to every later latent: edges between latents
        follow the order only and do not mean a direct effect was found. An observed variable in
        no cluster has no edges. ``metadata["algorithm"]`` is ``"GIN"``, ``metadata["clusters"]``
        lists the clusters as lists of column indices, ``metadata["causal_order"]`` lists them
        earliest latent first (clusters whose order could not be resolved come last), and
        ``metadata["n_latents"]`` is the latent count.

    Raises
    ------
    ValueError
        If ``labels`` does not give one name per column of ``data``, or if a DataFrame's column
        names repeat.
    numpy.linalg.LinAlgError
        If ``data`` holds ``NaN`` / ``inf``.

    Warns
    -----
    ExperimentalWarning
        On the first call in a process, because the method is experimental: it has no published
        benchmark, and its API may change without deprecation.

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
    warn_experimental("gin")
    from andrey.latent.gin import gin_structure

    names = labels if labels is not None else column_labels(data)
    return gin_structure(np.asarray(data, dtype=np.float64), alpha=alpha, labels=names)
