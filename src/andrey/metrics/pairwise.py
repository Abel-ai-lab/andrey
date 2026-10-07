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
    """Fraction of decided pairs whose predicted cause -> effect direction matches the truth.

    ``pred`` and ``truth`` are per-pair signed labels (``+1`` = ``X -> Y``, ``-1`` = ``Y -> X``);
    a ``pred`` of 0 is an abstention, dropped before scoring. ``weights`` gives optional per-pair
    weights (the Tuebingen-pairs standard). Returns 0.0 when nothing is decided; pair it with
    :func:`decision_rate` so a high accuracy from few decisions is visible.

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
    """Fraction of pairs the method commits to (``pred != 0``); 1.0 when it decides every pair.

    Examples
    --------
    >>> decision_rate([1, 0, -1, 0])
    0.5
    """
    p = _signs(pred)
    w = np.ones_like(p) if weights is None else np.asarray(weights, dtype=np.float64)
    total = w.sum()
    return float(w[p != 0].sum() / total) if total else 0.0
