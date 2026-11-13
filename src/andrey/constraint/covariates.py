"""The covariate selection rules, the engine behind :func:`andrey.covariate_rules`.

The rules, their assumptions, and their reference are described on the facade. The engine reads
the tests through :meth:`~andrey.constraint.blanket.LocalSearch.scan` only: for each set ``Z`` of
covariates, the outcome against every other covariate and the treatment given ``Z``, the treatment
against every other covariate given ``Z``, and the outcome against every other covariate given
``Z`` and the treatment. A p-value above ``alpha`` is an independence, any other a dependence.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Literal

from andrey.constraint.blanket import LocalSearch

__all__ = ["CovariateRulesResult", "Verdict", "covariate_rules"]

#: The rules' answer: ``"+-"`` an effect with an admissible set, ``"0"`` no effect, ``"?"``
#: undecided.
Verdict = Literal["+-", "0", "?"]


@dataclass(frozen=True)
class CovariateRulesResult:
    """The rules' answer, by 0-based column position.

    ``verdict`` is ``"+-"`` when only R1 applies, ``"0"`` when only R2 applies, and ``"?"`` when
    neither or both apply. ``sets`` are the distinct sets ``Z`` for which R1 applies, each
    ascending and in the order they were first found, when the verdict is ``"+-"``, else empty.
    ``counts`` holds the applications of each rule (``"R1"``, ``"R2"``) and the candidates each
    rule was checked on (``"R1_checked"``, ``"R2_checked"``): R1 and R2 (ii)-(iii) count pairs
    ``(w, Z)``, R2 (i) counts sets ``Z``.
    """

    verdict: Verdict
    sets: list[list[int]]
    counts: dict[str, int]


def covariate_rules(
    search: LocalSearch,
    x: int,
    y: int,
    covariates: Sequence[int],
    *,
    max_depth: int | None = None,
) -> CovariateRulesResult:
    """Rules R1 and R2 of Entner, Hoyer, and Spirtes (2013) for the treatment ``x`` and the outcome
    ``y`` over the ``covariates`` ``W``, every set ``Z`` of them checked.

    R1 applies to a covariate ``w`` and a set ``Z`` of the others when ``w`` and ``y`` are dependent
    given ``Z`` and independent given ``Z`` and ``x``. R2 applies to a set ``Z`` when ``x`` and
    ``y`` are independent given ``Z`` (i), and to ``w`` and ``Z`` when ``w`` and ``x`` are
    dependent given ``Z`` and ``w`` and ``y`` independent given ``Z`` (ii, iii). With
    ``max_depth``, no test conditions on more than ``max_depth`` variables: R1 is checked on sets of
    at most ``max_depth - 1`` covariates, since its second test also conditions on ``x``, and R2 on
    sets of at most ``max_depth``.
    """
    alpha = search.alpha
    covariates = sorted(covariates)
    top = len(covariates) if max_depth is None else min(len(covariates), max_depth)
    counts = dict.fromkeys(("R1", "R2", "R1_checked", "R2_checked"), 0)
    found: dict[tuple[int, ...], None] = {}
    for size in range(top + 1):
        r1 = max_depth is None or size + 1 <= max_depth
        for z in combinations(covariates, size):
            rest = [w for w in covariates if w not in z]
            _, p_y = search.scan(y, [*rest, x], z)  # w vs y | Z, then x vs y | Z
            counts["R2_checked"] += 1
            counts["R2"] += int(p_y[-1] > alpha)  # R2 (i)
            if not rest:
                continue
            _, p_x = search.scan(x, rest, z)
            for k in range(len(rest)):
                counts["R2_checked"] += 1
                counts["R2"] += int(p_x[k] <= alpha < p_y[k])  # R2 (ii), (iii)
            if not r1:
                continue
            _, p_yx = search.scan(y, rest, (*z, x))
            for k in range(len(rest)):
                counts["R1_checked"] += 1
                if p_y[k] <= alpha < p_yx[k]:  # R1 (i), (ii)
                    counts["R1"] += 1
                    found.setdefault(z)
    if counts["R1"] and not counts["R2"]:
        return CovariateRulesResult("+-", [list(z) for z in found], counts)
    return CovariateRulesResult("0" if counts["R2"] and not counts["R1"] else "?", [], counts)
