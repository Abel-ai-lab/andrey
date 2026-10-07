"""Edge-coefficient error: mean absolute error of estimated weights on the true edges.

Scored only when a weighted adjacency is present (LiNGAM-family output), so it hangs off the
special weights field rather than the graph topology.
"""

from __future__ import annotations

import numpy as np

from andrey.core.output import StructureOutput


def _weights(w: object) -> np.ndarray:
    """Dense ``float64`` weighted adjacency (``W[i, j]`` = weight of ``i -> j``) from any input."""
    if isinstance(w, StructureOutput):
        dense = w.weighted_adjacency
        if dense is None:
            raise ValueError("StructureOutput carries no edge weights")
        return dense
    return np.asarray(w, dtype=np.float64)


def coefficient_mae(estimated_w: object, true_w: object) -> float:
    """Mean absolute coefficient error over the true edges (edges absent in truth are ignored).

    A missed true edge contributes its full ``|w_true|`` (so this blends coefficient error with
    recall on the true support); returns 0.0 when the truth has no weighted edges. Both inputs use
    the canonical ``W[i, j]`` = weight of edge ``i -> j`` orientation -- note LiNGAM's ``B`` is the
    transpose, and adapters normalize to this before storing.

    Examples
    --------
    >>> coefficient_mae([[0.0, 0.9], [0.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]])
    0.09999999999999998
    """
    est = _weights(estimated_w)
    true_ = _weights(true_w)
    if est.shape != true_.shape:
        raise ValueError(f"weight-shape mismatch: estimated {est.shape}, true {true_.shape}")
    mask = true_ != 0
    if not mask.any():
        return 0.0
    return float(np.abs(est[mask] - true_[mask]).mean())
