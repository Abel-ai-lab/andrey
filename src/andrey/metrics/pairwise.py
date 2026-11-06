"""Pairwise-direction metrics: cause-effect direction accuracy for two-variable methods (ANM, PNL).

Scored over a set of ordered pairs, each with a predicted and a true direction. Accuracy alone is
gameable by abstaining on the hard pairs, so ``decision_rate`` is reported alongside it.
"""

from __future__ import annotations

import numpy as np


def _signs(labels: object) -> np.ndarray:
    """Per-pair direction as ``{-1, 0, +1}`` (``+1`` = ``X -> Y``, ``0`` = abstain)."""
    return np.sign(np.asarray(labels, dtype=np.float64))


def direction_accuracy(pred: object, truth: object, *, weights: object | None = None) -> float:
    """Return the fraction of decided pairs whose predicted direction matches the true one.

    An abstention is dropped before scoring, so report :func:`decision_rate` with it: a high
    accuracy from few decisions then shows.

    Parameters
    ----------
    pred : array-like of shape (n_pairs,)
        Each pair's predicted direction: positive for ``X -> Y``, negative for ``Y -> X``, and 0
        for an abstention.
    truth : array-like of shape (n_pairs,)
        Each pair's true direction, signed the same way.
    weights : array-like of shape (n_pairs,) or None, default=None
        Each pair's weight, such as the weights of a benchmark of cause-effect pairs; ``None``
        weighs every pair 1.

    Returns
    -------
    float
        The weighted share of the decided pairs whose direction matches, or 0.0 when no pair is
        decided or the decided pairs' weights sum to 0.

    Examples
    --------
    >>> direction_accuracy([1, 1, -1], [1, -1, -1])
    0.6666666666666666
    """
    p, t = _signs(pred), _signs(truth)
    if p.shape != t.shape:
        raise ValueError(f"pred/truth length mismatch: {p.shape} vs {t.shape}")
    w = np.ones_like(p) if weights is None else np.asarray(weights, dtype=np.float64)
    decided = p != 0
    if not decided.any() or w[decided].sum() == 0:
        return 0.0
    correct = (p == t) & decided
    return float(w[correct].sum() / w[decided].sum())


def decision_rate(pred: object, *, weights: object | None = None) -> float:
    """Return the fraction of pairs the method decides, those with ``pred != 0``.

    Parameters
    ----------
    pred : array-like of shape (n_pairs,)
        Each pair's predicted direction: positive for ``X -> Y``, negative for ``Y -> X``, and 0
        for an abstention.
    weights : array-like of shape (n_pairs,) or None, default=None
        Each pair's weight, such as the weights of a benchmark of cause-effect pairs; ``None``
        weighs every pair 1.

    Returns
    -------
    float
        The weighted share of all pairs that are decided: 1.0 when every pair is decided, and 0.0
        when the weights sum to 0.

    Examples
    --------
    >>> decision_rate([1, 0, -1, 0])
    0.5
    """
    p = _signs(pred)
    w = np.ones_like(p) if weights is None else np.asarray(weights, dtype=np.float64)
    total = w.sum()
    return float(w[p != 0].sum() / total) if total else 0.0
