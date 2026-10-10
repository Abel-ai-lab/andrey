"""Facade for the pairwise post-nonlinear (PNL) direction test.

The native engine tests both directions in a two-column table. The result contains a two-node
graph with the chosen edge, if any, and both independence p-values in metadata.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._adapt import column_labels, data_matrix, open_unit_interval
from ._experimental import warn_experimental
from ._irregular import _adapt_pnl

if TYPE_CHECKING:
    import numpy.typing as npt

    from andrey.core import StructureOutput


def pnl(data: npt.ArrayLike, *, alpha: float | None = None) -> StructureOutput:
    """Test the direction of a variable pair with the post-nonlinear (PNL) model.

    PNL [1]_ [2]_ fits ``y = f2(f1(x) + e)`` in each direction. An HSIC test checks whether the
    estimated noise ``e`` is independent of the proposed cause. The two p-values determine the edge.

    Parameters
    ----------
    data : array-like of shape (n_samples, 2)
        The two variables, one per column: column 0 is ``x`` and column 1 is ``y``. Coerced to
        ``float64``. A DataFrame's column names become the structure's ``labels``.
    alpha : float or None, default=None
        ``None`` chooses the direction with the larger p-value. A significance level such as
        ``0.05`` draws an edge only when exactly one p-value exceeds it. See Notes.

    Returns
    -------
    StructureOutput
        Contains a two-node :class:`~andrey.core.GraphStructure` of kind ``dag`` with the chosen
        edge, or no edge. ``metadata["algorithm"]`` is ``"PNL"``; ``pval_forward`` is the
        independence p-value for ``x -> y`` and ``pval_backward`` is the p-value for ``y -> x``.

    Raises
    ------
    ValueError
        If ``data`` does not have 2 columns, or if ``alpha`` is not in ``(0, 1)``.

    Warns
    -----
    ExperimentalWarning
        On the first call in a process, because the method is experimental: it has no published
        benchmark, and its API may change without deprecation.

    Notes
    -----
    With ``alpha=None``, the edge points in the direction with the larger p-value; equal values
    give no edge. Unequal p-values therefore produce an edge even for unrelated variables or a
    pair whose direction cannot be identified, such as a linear Gaussian pair. Mooij et al. (2016)
    [4]_ use this rule to avoid choosing a significance level and found it among the best methods
    in their benchmark of additive-noise models.

    With a significance level, a direction passes when its p-value exceeds ``alpha``:

    - only ``x -> y`` passes: the edge ``x -> y``;
    - only ``y -> x`` passes: the edge ``y -> x``;
    - both pass: no edge, since neither direction is rejected;
    - neither passes: no edge, since both directions are rejected.

    Zhang and Hyvarinen (2010) [2]_ use a level of 0.01 for PNL; Hoyer et al. (2008) [3]_ use 0.02
    for additive-noise models. The result depends on ``alpha``. Estimated noise can appear dependent
    on the true cause, so a standard level can reject the true direction; no theory establishes
    the right level (Mooij et al., 2016). Both p-values remain in ``metadata``.

    PNL fits each transform as a sum of smooth basis functions, so the fit is deterministic but can
    miss a sharply varying nonlinearity.

    References
    ----------
    .. [1] Zhang, K., and Hyvarinen, A. (2009). On the identifiability of the post-nonlinear
       causal model. UAI, 647-655.
    .. [2] Zhang, K., and Hyvarinen, A. (2010). Distinguishing causes from effects using nonlinear
       acyclic causal models. JMLR Workshop and Conference Proceedings 6, 157-164.
    .. [3] Hoyer, P. O., Janzing, D., Mooij, J. M., Peters, J., and Scholkopf, B. (2008).
       Nonlinear causal discovery with additive noise models. NeurIPS 21, 689-696.
    .. [4] Mooij, J. M., Peters, J., Janzing, D., Zscheischler, J., and Scholkopf, B. (2016).
       Distinguishing cause from effect using observational data: methods and benchmarks. JMLR
       17(32), 1-102.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import pnl
    >>> rng = np.random.default_rng(2)
    >>> x = rng.uniform(-1, 1, 300)
    >>> y = np.tanh(x**3 + 0.3 * rng.uniform(-1, 1, 300))
    >>> out = pnl(np.column_stack([x, y]))
    >>> out.structure.oriented_edges()
    [(0, 1, 'directed')]
    >>> out.metadata["pval_forward"] > 0.05 >= out.metadata["pval_backward"]
    True
    """
    warn_experimental("pnl")
    from andrey.irregular.pnl import pnl as _native

    if np.ndim(data) != 2 or np.shape(data)[1] != 2:
        raise ValueError(
            f"pnl expects an (n_samples, 2) table of two variables, got {np.shape(data)}"
        )
    X = data_matrix(data)
    if alpha is not None:
        alpha = open_unit_interval(alpha, "alpha")
    labels = column_labels(data)
    pval_forward, pval_backward = _native(X[:, 0], X[:, 1])
    return _adapt_pnl(pval_forward, pval_backward, alpha=alpha, labels=labels)
