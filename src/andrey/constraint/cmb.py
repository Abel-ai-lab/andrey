"""CMB's causal identities: each neighbor of a target labeled a parent, a child, or undecided.

Gao and Ji's Causal Markov Blanket discovery (Algorithms 1 and 2) reads the direction of a
target's edges from how the dependence between two of its parents and children changes when the
target joins the conditioning set, retests an inconclusive pair given one member's Markov blanket,
and visits each undecided neighbor in turn to learn the target's identity there. A visited
neighbor's identities also direct, by acyclicity (Meek's rule R2), the target's edges that close a
triangle with it. :func:`cmb` runs it over a :class:`~andrey.constraint.blanket.LocalSearch`.

References
----------
Gao, T., and Ji, Q. (2015). Local causal discovery of direct causes and effects. Advances in
Neural Information Processing Systems 28, 2512-2520.

Meek, C. (1995). Causal inference and causal explanation with background knowledge. UAI, 403-410.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from andrey.constraint.blanket import LocalSearch

#: The identities of a target's parents and children (Gao and Ji's Algorithms 1 and 2): a direct
#: cause, a direct effect, a direction the tests leave open, and a pair whose tests are
#: inconclusive so far. A lower code replaces a higher one, never the reverse.
PARENT, CHILD, UNDECIDED, INCONCLUSIVE = 1, 2, 3, 4

Pair = tuple[int, int]


@dataclass
class Identities:
    """One variable's identities as :func:`cmb` builds them.

    ``codes`` maps each of the variable's parents and children to its code, ``0`` until a test
    labels it; a member of ``fixed`` keeps the code the knowledge gave it. ``undecided`` holds the
    pairs whose dependence vanishes once the variable is conditioned on (at least one of the two is
    a child), ``inconclusive`` the pairs whose tests decided nothing so far, and ``separated`` the
    set given which each pair of parents tested independent.
    """

    codes: dict[int, int]
    fixed: frozenset[int] = frozenset()
    undecided: list[Pair] = field(default_factory=list)
    inconclusive: list[Pair] = field(default_factory=list)
    separated: dict[Pair, frozenset[int]] = field(default_factory=dict)

    def put(self, x: int, code: int) -> None:
        """Give ``x`` the code ``code`` unless it holds a lower one or the knowledge fixed it."""
        if x not in self.fixed and (self.codes[x] == 0 or code < self.codes[x]):
            self.codes[x] = code

    def holds(self, code: int) -> bool:
        """Whether some member has the code ``code``."""
        return code in self.codes.values()

    def partners(self, x: int) -> list[int]:
        """The other end of each pair in ``undecided`` that holds ``x``."""
        return [b if a == x else a for a, b in self.undecided if x in (a, b)]

    def inconclusive_with(self, x: int, y: int) -> bool:
        """Whether ``(x, y)``, in either order, is in ``inconclusive``."""
        return (x, y) in self.inconclusive or (y, x) in self.inconclusive

    def acyclic(self, x: int, there: dict[int, int]) -> None:
        """Meek's rule R2 over each triangle the variable ``t`` forms with ``x`` and a member ``w``.

        ``there`` holds the identities at ``x``: ``there[w]`` is :data:`CHILD` for ``x -> w`` and
        :data:`PARENT` for ``w -> x``. A path ``t -> x -> w`` makes ``w`` a child of ``t`` and
        ``w -> x -> t`` makes it a parent; where ``w``'s own code is known, ``t -> w -> x`` makes
        ``x`` a child and ``x -> w -> t`` a parent. Only these codes act, through :meth:`put`.
        """
        for w in self.codes:
            code = there.get(w, 0)
            if w == x or code not in (PARENT, CHILD):
                continue
            if self.codes[x] == code:
                self.put(w, code)
            if self.codes[w] == CHILD and code == PARENT:
                self.put(x, CHILD)
            if self.codes[w] == PARENT and code == CHILD:
                self.put(x, PARENT)


@dataclass(frozen=True)
class CMBResult:
    """What :func:`cmb` returns: ``codes`` maps each of the target's parents and children to
    :data:`PARENT`, :data:`CHILD`, or :data:`UNDECIDED`; ``blanket`` is the target's Markov
    blanket; ``learned`` lists the variables whose Markov blanket the run learned, in order, the
    target first."""

    codes: dict[int, int]
    blanket: list[int]
    learned: list[int]


class _Run:
    """One run of :func:`cmb`: the identities of every variable it visited and the tests it read."""

    def __init__(self, search: LocalSearch, blanket: str, decided: np.ndarray | None) -> None:
        self.search = search
        self.blanket = blanket
        self.decided = decided
        self.learned: list[int] = []
        self.done: dict[int, Identities] = {}
        self.visiting: set[int] = set()  # variables whose run is open
        self.answers: dict[tuple[int, int, tuple[int, ...]], bool] = {}

    def local(self, x: int) -> tuple[list[int], list[int]]:
        """``x``'s parents and children, in rank order, and its Markov blanket."""
        pc, mb, _ = self.search.local(self.blanket, x)
        assert pc is not None and mb is not None  # a "pc_spouses" learner returns both
        if x not in self.learned:
            self.learned.append(x)
        return sorted(pc, key=lambda v: self.search.ranks[v]), mb

    def independent(self, x: int, y: int, given: frozenset[int]) -> bool:
        """Whether the test finds ``x`` and ``y`` independent given ``given``: a p-value above
        ``alpha``."""
        cond = tuple(sorted(given))
        key = (x, y, cond)
        if key not in self.answers:
            self.answers[key] = bool(self.search.scan(x, [y], cond)[1][0] > self.search.alpha)
        return self.answers[key]

    def causal_search(self, t: int, pc: list[int], z: frozenset[int], node: Identities) -> None:
        """Algorithm 2: label ``t``'s parents and children ``pc`` from each pair's tests given
        ``z`` and given ``z`` with ``t``, in place in ``node``."""
        if len(pc) == 1:
            node.put(pc[0], UNDECIDED)
        depth = self.search.max_depth
        for x, y in itertools.combinations(pc, 2):
            given = z - {x, y, t}
            # A pair whose test with t would condition on more than max_depth variables runs no
            # test and decides nothing.
            capped = depth is not None and len(given) + 1 > depth
            apart = not capped and self.independent(x, y, given)
            apart_given_t = not capped and self.independent(x, y, given | {t})
            if apart and not apart_given_t:  # C2: x -> t <- y
                node.put(x, PARENT)
                node.put(y, PARENT)
                node.separated.setdefault((x, y), given)
                if (x, y) in node.inconclusive:
                    node.inconclusive.remove((x, y))
            elif apart_given_t and not apart:  # C3: at least one of the two is a child
                node.put(y, CHILD if node.codes[x] == PARENT else UNDECIDED)
                node.put(x, CHILD if node.codes[y] == PARENT else UNDECIDED)
                if (x, y) not in node.undecided:
                    node.undecided.append((x, y))
                if (x, y) in node.inconclusive:
                    node.inconclusive.remove((x, y))
            else:
                fresh = all(node.codes[v] in (0, INCONCLUSIVE) for v in (x, y))
                if fresh and (x, y) not in node.inconclusive:
                    node.inconclusive.append((x, y))
                node.put(x, INCONCLUSIVE)
                node.put(y, INCONCLUSIVE)
        for x in pc:
            if node.codes[x] == PARENT:
                for y in node.partners(x):
                    node.put(y, CHILD)

    def identify(self, t: int) -> Identities:
        """Algorithm 1 for ``t``: its parents' and children's identities, each variable visited
        once; a variable whose own run is still open is not visited again."""
        if t in self.done:
            return self.done[t]
        self.visiting.add(t)
        pc, _ = self.local(t)
        node = self._seeded(t, pc)
        # Step 1: the tests given the empty set.
        self.causal_search(t, pc, frozenset(), node)
        # Step 2: each inconclusive pair retested given one member's blanket (Lemma 2).
        tried: set[Pair] = set()
        while node.holds(INCONCLUSIVE):
            pair = next((p for p in node.inconclusive if p not in tried), None)
            if pair is None:
                break
            tried.add(pair)
            x, y = pair
            _, mb = self.local(x)
            self.causal_search(t, pc, frozenset(mb) - {t, y}, node)
        # Meek's rule R3 (Algorithm 1, lines 12-14): a neighbor inconclusive with two parents is a
        # parent when it lies in the set that separated them, where no descendant of t can lie.
        parents = [x for x in pc if node.codes[x] == PARENT]
        for x, y in itertools.combinations(parents, 2):
            sepset = node.separated.get((x, y))
            if sepset is None:
                continue
            for w in pc:
                if w in sepset and node.inconclusive_with(x, w) and node.inconclusive_with(y, w):
                    node.put(w, PARENT)
        for x in pc:
            if node.codes[x] == INCONCLUSIVE:
                node.codes[x] = UNDECIDED
        # Step 3: each undecided neighbor visited for the target's identity there.
        for x in pc:
            if node.codes[x] != UNDECIDED or x in self.visiting:
                continue
            at_x = self.identify(x).codes
            there = at_x.get(t, 0)
            if there == CHILD:
                node.put(x, PARENT)
                for y in node.partners(x):
                    node.put(y, CHILD)
            elif there == PARENT:
                node.put(x, CHILD)
            node.acyclic(x, at_x)
            if not node.holds(UNDECIDED):
                break
        self.done[t] = node
        self.visiting.discard(t)
        return node

    def _seeded(self, t: int, pc: list[int]) -> Identities:
        """``t``'s identities before any test: the codes the knowledge's directions give, fixed."""
        codes = dict.fromkeys(pc, 0)
        if self.decided is None:
            return Identities(codes)
        for x in pc:
            if self.decided[x, t]:
                codes[x] = PARENT
            elif self.decided[t, x]:
                codes[x] = CHILD
        return Identities(codes, fixed=frozenset(x for x in pc if codes[x]))


def cmb(
    search: LocalSearch, target: int, blanket: str, *, decided: np.ndarray | None = None
) -> CMBResult:
    """The identity of each of ``target``'s parents and children by CMB [1]_.

    ``blanket`` names a ``"pc_spouses"`` learner of
    :data:`~andrey.constraint.blanket.LEARNERS`, which returns both a variable's parents and
    children and its Markov blanket. Algorithm 2 tests each pair of the target's parents and
    children given a set ``Z`` and given ``Z`` with the target: independent then dependent makes
    both parents (C2); dependent then independent makes one a child, the other's partner a child
    once the other is a parent (C3); a pair neither decides is inconclusive, and each member no
    test has labeled yet is too. Step 1 takes ``Z`` empty; Step 2 retests every pair given the
    blanket of the inconclusive pair's member of lower rank, less the target and the other member
    (Lemma 2), pair by pair, until no member is inconclusive. A neighbor still inconclusive with
    two parents is a parent when it lies in the set that separated them (Meek's rule R3 [2]_): the
    target is a collider between them, so no descendant of it lies in that set; the rest become
    undecided. Step 3 visits each undecided neighbor ``x`` in rank order and runs the same
    steps there, never returning to a variable whose run is open: the target a child of ``x``
    makes ``x`` a parent and each partner of ``x`` a child, and the target a parent of ``x`` makes
    ``x`` a child. Each visited ``x`` also directs, by acyclicity (Meek's rule R2 [2]_), the
    target's edges that close a triangle with it: a path ``target -> x -> w`` makes ``w`` a child,
    ``w -> x -> target`` makes ``w`` a parent, and a direction already given to ``w`` directs
    ``x`` the same way. A lower code is never replaced by a higher one, and a direction ``decided``
    gives (``decided[u, v]``: ``u -> v``) is fixed before any test. Tests are asked once, the
    lower-ranked member of the pair first; a pair whose test with the target would condition on
    more than the search's ``max_depth`` variables runs neither test and decides nothing.

    References
    ----------
    .. [1] Gao, T., and Ji, Q. (2015). Local causal discovery of direct causes and effects.
       Advances in Neural Information Processing Systems 28, 2512-2520.
    .. [2] Meek, C. (1995). Causal inference and causal explanation with background knowledge.
       UAI, 403-410.
    """
    run = _Run(search, blanket, decided)
    with search.pinned():
        node = run.identify(target)
        _, mb = run.local(target)
    return CMBResult(codes=dict(node.codes), blanket=list(mb), learned=list(run.learned))
