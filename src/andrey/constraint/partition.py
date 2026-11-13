"""LDP and LD3, the engines behind :func:`andrey.ldp` and :func:`andrey.ld3`.

The algorithms, their settings, and their references are described on the two functions. Both label
each variable other than the treatment ``x`` and the outcome ``y`` by its causal relation to the
pair (the partitions ``Z1`` to ``Z8`` of Maasch et al. 2024, Table 1), from tests whose conditioning
sets are the labels found before. Each query ``(a, b | S)`` is asked of the test once and its
p-value kept, so a later step that needs it again, such as LDP's Step 7, runs no test. A p-value
above ``alpha`` is an independence, any other a dependence.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from andrey.constraint.blanket import LocalSearch

__all__ = ["LD3_LABELS", "LDP_LABELS", "LD3Result", "LDPResult", "ld3", "ldp"]

#: LDP's labels, in the order ``metadata["partition"]`` lists them: the partitions it names, the
#: Z5-or-Z7 label it leaves unsplit when it finds no confounder, the post-treatment superset of Z2,
#: Z3, and Z6, and the variables it gives no label.
LDP_LABELS = ("Z1", "Z4", "Z5", "Z5,7", "Z7", "Z8", "Zpost", "unlabeled")

#: LD3's labels: the confounders and mediators that are parents of the outcome, then as for LDP.
LD3_LABELS = ("Z1,3", "Z4", "Z5,7", "Z8", "unlabeled")


@dataclass(frozen=True)
class LDPResult:
    """LDP's answer, by 0-based column position.

    ``partition`` maps each label of :data:`LDP_LABELS` to its members, ascending. ``valid`` is
    the Z5 criterion: when it holds, ``partition["Z1"]`` is a valid adjustment set for the pair.
    """

    partition: dict[str, list[int]]
    valid: bool


@dataclass(frozen=True)
class LD3Result:
    """LD3's answer, by 0-based column position.

    ``partition`` maps each label of :data:`LD3_LABELS` to its members, ascending (a variable that
    passes both the ``Z8`` and the ``Z4`` test is under both). ``parents`` are the parents of the
    outcome found among the other variables, ascending; ``direct`` is the structural direct
    criterion, whether the treatment is a parent of the outcome.
    """

    partition: dict[str, list[int]]
    parents: list[int]
    direct: bool


class _Tests:
    """The answers of ``search`` to the queries asked, each asked once."""

    def __init__(self, search: LocalSearch) -> None:
        self.search = search
        self.alpha = search.alpha
        self.kept: dict[tuple[int, int, tuple[int, ...]], float] = {}

    @staticmethod
    def _key(a: int, b: int, cond: Iterable[int]) -> tuple[int, int, tuple[int, ...]]:
        return (min(a, b), max(a, b), tuple(sorted(cond)))

    def ask(self, a: int, bs: Sequence[int], cond: Sequence[int] = ()) -> None:
        """Keep the p-value of ``a`` against each of ``bs`` given ``cond``, asking the test
        together for the queries not kept yet."""
        new = [b for b in bs if self._key(a, b, cond) not in self.kept]
        if new:
            _, pvalues = self.search.scan(a, new, tuple(cond))
            for b, p in zip(new, pvalues, strict=True):
                self.kept[self._key(a, b, cond)] = float(p)

    def independent(self, a: int, b: int, cond: Sequence[int] = ()) -> bool:
        """Whether ``a`` and ``b`` test independent given ``cond``."""
        key = self._key(a, b, cond)
        if key not in self.kept:
            self.ask(a, [b], cond)
        return self.kept[key] > self.alpha


def _partition(labels: Sequence[str], sets: dict[str, list[int]]) -> dict[str, list[int]]:
    return {label: sorted(sets.get(label, [])) for label in labels}


def ldp(search: LocalSearch, x: int, y: int, others: Sequence[int]) -> LDPResult:
    """LDP (Maasch et al. 2024, Algorithm 1) for the treatment ``x`` and the outcome ``y`` over
    the variables ``others``.

    Steps 1 to 3 test each variable against the pair alone. Step 4 removes post-treatment
    variables with the causes of the outcome (``Z4``) found in Step 2; Step 5 keeps the variables
    the treatment and the rest make independent of the outcome (``Zmix``); Step 6 finds the
    confounders by the collider each forms at the treatment with an instrument candidate; Step 7
    splits those candidates into confounders and instruments from the tests already run; Step 8 is
    the Z5 criterion. Where Algorithm 1 is open the authors' code decides: every candidate that
    forms the collider joins ``Z1,5``, not only the first; Step 6's second pass covers Step 5's
    variables only, and ``Z7`` is ``Z5,7`` less ``Z1,5``; Step 7 compares with the confounders of
    Step 6's first pass, those adjacent to the outcome, whose tests it has run.
    """
    tests = _Tests(search)
    indep = tests.independent
    others = list(others)
    tests.ask(x, others)
    tests.ask(y, others)

    # Steps 1-3: the tests that read the pair alone, each asked only when its branch needs it.
    tests.ask(x, [z for z in others if indep(x, z) and not indep(y, z)], (y,))
    z8, z4, z57, rest = [], [], [], []
    for z in others:
        if indep(x, z) and indep(y, z):
            z8.append(z)
        elif indep(x, z) and not indep(x, z, (y,)):
            z4.append(z)
        else:
            rest.append(z)
    tests.ask(y, [z for z in rest if not indep(y, z)], (x,))
    z57 = [z for z in rest if not indep(y, z) and indep(y, z, (x,))]
    rest = [z for z in rest if z not in z57]

    # Step 4: a variable dependent on a cause of the outcome, or made independent of it by the
    # pair, is post-treatment.
    post = []
    if z4:
        post = [z for z in rest if any(not indep(z, w) or indep(z, w, (x, y)) for w in z4)]
        rest = [z for z in rest if z not in post]

    # Step 5: the variables the treatment and the rest make independent of the outcome.
    zmix = [z for z in rest if not indep(y, z) and indep(y, z, (x, *(w for w in rest if w != z)))]
    rest = [z for z in rest if z not in zmix]

    # Step 6: a remaining variable that forms a collider at the treatment with a candidate of
    # Zmix or Z5,7 is a confounder; the candidate is a confounder or an instrument (Z1,5).
    mix = [*zmix, *z57]
    z1: list[int] = []
    z15: list[int] = []
    adjacent: list[int] = []
    resolved = bool(mix) and bool(rest)
    if resolved:
        for z in rest:
            witnesses = [w for w in mix if indep(w, z) and not indep(w, z, (x,))]
            if witnesses:
                z1.append(z)
                z15.extend(w for w in witnesses if w not in z15)
            else:
                post.append(z)
        adjacent = list(z1)
        for w in zmix:
            if w in z15:
                continue
            if any(indep(u, w) for u in z15):
                z1.append(w)
            else:
                post.append(w)
    z7 = [z for z in z57 if z not in z15] if z15 or z1 else []
    unsplit = [] if z15 or z1 else z57

    # Step 7: a candidate dependent on a confounder adjacent to the outcome is a confounder, the
    # others are instruments.
    z5 = []
    for u in z15:
        if any(not indep(u, a) for a in adjacent):
            z1.append(u)
        else:
            z5.append(u)

    # Step 8: the Z5 criterion.
    valid = any(indep(v, y, (x, *z1)) for v in z5)

    labeled = {*z1, *z4, *z5, *unsplit, *z7, *z8, *post}
    sets = {
        "Z1": z1,
        "Z4": z4,
        "Z5": z5,
        "Z5,7": unsplit,
        "Z7": z7,
        "Z8": z8,
        "Zpost": post,
        "unlabeled": [z for z in others if z not in labeled],
    }
    return LDPResult(partition=_partition(LDP_LABELS, sets), valid=valid)


def ld3(search: LocalSearch, x: int, y: int, others: Sequence[int]) -> LD3Result:
    """LD3 (Maasch et al. 2025, Algorithm 1) for the treatment ``x`` and the outcome ``y`` over
    the variables ``others``.

    Lines 3 to 5 are three separate tests of each variable against the pair; line 8 keeps each
    remaining variable dependent on the outcome given the treatment, the ``Z4`` variables, and the
    other remaining ones; line 10 keeps each ``Z4`` variable dependent on the outcome given the
    treatment, the parents line 8 found, and the other ``Z4`` variables; line 12 tests the
    treatment against the outcome given the parents line 8 found.
    """
    tests = _Tests(search)
    indep = tests.independent
    others = list(others)
    tests.ask(x, others)
    tests.ask(y, others)
    tests.ask(y, [z for z in others if not indep(y, z)], (x,))
    tests.ask(x, [z for z in others if indep(x, z)], (y,))
    z8 = [z for z in others if indep(x, z) and indep(y, z)]
    z57 = [z for z in others if not indep(y, z) and indep(y, z, (x,))]
    z4 = [z for z in others if indep(x, z) and not indep(x, z, (y,))]
    found = {*z8, *z57, *z4}
    rest = [z for z in others if z not in found]

    parents13 = [z for z in rest if not indep(z, y, (x, *z4, *(w for w in rest if w != z)))]
    parents4 = [w for w in z4 if not indep(w, y, (x, *parents13, *(v for v in z4 if v != w)))]
    direct = not indep(x, y, parents13)

    sets = {
        "Z1,3": parents13,
        "Z4": z4,
        "Z5,7": z57,
        "Z8": z8,
        "unlabeled": [z for z in rest if z not in parents13],
    }
    return LD3Result(
        partition=_partition(LD3_LABELS, sets),
        parents=sorted([*parents13, *parents4]),
        direct=direct,
    )
