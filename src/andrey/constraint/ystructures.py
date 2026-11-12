"""Y-structures, the engine behind :func:`andrey.y_structures`.

The algorithm, its settings, and its references are described on :func:`andrey.y_structures`.

A claim that ``x`` is an ancestor of ``y`` rests on two auxiliary variables ``v`` and ``w``. The
extended Y-structure needs a minimal independence and a minimal dependence, each made by ``x``:

- ``v`` and ``y`` are dependent, and independent given ``x``;
- ``v`` and ``w`` are independent, and dependent given ``x``.

The Y-structure also needs the first line with ``w`` in place of ``v``, so the two play the same
role. Every test is marginal or conditions on ``x`` alone. The marginal tests run once for every
pair; the tests given ``x`` run only for the pairs a pattern at ``x`` could use, each pair once,
asked from its smaller column, so the claims do not depend on which tests run.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from andrey.constraint.blanket import LocalSearch

__all__ = ["Claim", "y_structures"]


@dataclass(frozen=True)
class Claim:
    """The claim that ``cause`` is an ancestor of ``effect``, by 0-based column position.

    ``patterns`` holds the auxiliary pairs ``(v, w)`` of the patterns found, ascending; for the
    Y-structure, whose two roles are the same, each pair once as ``v < w``. ``score`` is the
    largest, over the patterns, of the smaller of ``-log p`` for ``v`` and ``y`` and for ``w``
    and ``y``, the marginal tests' p-values: ``inf`` where a p-value is ``0``.
    """

    cause: int
    effect: int
    score: float
    patterns: tuple[tuple[int, int], ...]


def y_structures(
    search: LocalSearch, columns: Sequence[int], *, extended: bool = False
) -> list[Claim]:
    """Every claim about an ordered pair of ``columns``, by descending score, then by position.

    ``search`` answers the tests at its ``alpha``: a p-value below ``alpha`` is a dependence, any
    other an independence. The auxiliary variables range over every column.
    """
    d, alpha = search.d, search.alpha
    marginal = _marginal(search, d)
    dependent = marginal < alpha
    np.fill_diagonal(dependent, False)
    independent = ~dependent
    np.fill_diagonal(independent, False)
    with np.errstate(divide="ignore"):
        surprise = -np.log(marginal)
    claims = []
    for x in columns:
        for y, pairs in _patterns(search, x, columns, dependent, independent, extended).items():
            score = max(min(surprise[v, y], surprise[w, y]) for v, w in pairs)
            claims.append(Claim(int(x), int(y), float(score), tuple(sorted(pairs))))
    return sorted(claims, key=lambda c: (-c.score, c.cause, c.effect))


def _marginal(search: LocalSearch, d: int) -> np.ndarray:
    """The marginal p-value of every pair, ``1`` on the diagonal."""
    p = np.ones((d, d))
    for a in range(d - 1):
        xs = list(range(a + 1, d))
        p[a, xs] = p[xs, a] = search.scan(a, xs)[1]
    return p


def _given(search: LocalSearch, x: int, pairs: Iterable[tuple[int, int]]) -> dict:
    """The p-value of each pair ``(a, b)``, ``a < b``, given ``x``."""
    grouped: dict[int, list[int]] = defaultdict(list)
    for a, b in sorted(pairs):
        grouped[a].append(b)
    found = {}
    for a, bs in grouped.items():
        for b, p in zip(bs, search.scan(a, bs, (x,))[1], strict=True):
            found[a, b] = float(p)
    return found


def _patterns(
    search: LocalSearch,
    x: int,
    columns: Sequence[int],
    dependent: np.ndarray,
    independent: np.ndarray,
    extended: bool,
) -> dict[int, set[tuple[int, int]]]:
    """The auxiliary pairs ``(v, w)`` of the patterns at ``x``, by effect ``y`` in ``columns``."""
    alpha = search.alpha
    apart = independent.copy()
    apart[x, :] = apart[:, x] = False
    # v needs a partner w independent of it; y is an effect of x in columns.
    capable = np.flatnonzero(apart.any(axis=1))
    effects = [int(y) for y in columns if y != x]
    pairs = {(min(v, y), max(v, y)) for y in effects for v in capable if v != y and dependent[v, y]}
    p_sep = _given(search, x, pairs)
    separated: dict[int, set[int]] = defaultdict(set)
    for y in effects:
        for v in capable:
            key = (min(v, y), max(v, y))
            if key in p_sep and p_sep[key] >= alpha:
                separated[int(v)].add(y)
    pairs = {
        (min(v, w), max(v, w))
        for v in separated
        for w in np.flatnonzero(apart[v])
        if extended or int(w) in separated
    }
    p_con = _given(search, x, pairs)
    found: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for v, ys in separated.items():
        for w in np.flatnonzero(apart[v]):
            w = int(w)
            if p_con.get((min(v, w), max(v, w)), alpha) >= alpha:
                continue
            for y in ys:
                if extended:
                    found[y].add((v, w))
                elif y in separated.get(w, ()):
                    found[y].add((min(v, w), max(v, w)))
    return found
