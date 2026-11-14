"""The sequential search behind the target-centered local methods: MB-by-MB [1]_, MMB-by-MMB [2]_,
and the local structure step of LOAD [3]_.

:func:`sequential` learns the edges at one target, and their marks, without the whole graph. It
keeps a waitlist that starts at the target. It takes the waitlist's head ``x``, learns ``x``'s
Markov blanket ``MB(x)`` with a learner of :data:`~andrey.constraint.blanket.LEARNERS`, and learns
the local structure ``L_x`` over ``MB+(x)``, the blanket together with ``x``
(:func:`learn_local`). It puts the edges at ``x`` and the v-structures that contain ``x`` into the
growing graph (:func:`merge`), which holds no other edge of ``L_x``, orients that graph
(:func:`orient`), and extends the waitlist (:func:`extend`), until a stop rule ends the search
(:func:`open_reach`, :func:`stop_rule`). :class:`Rules` names the choice each method's paper makes
at each step, and :data:`MB_BY_MB`, :data:`LOAD`, and :data:`MMB_BY_MMB` hold them.

A node ``y`` of ``MB(x)`` is adjacent to ``x`` exactly when no subset of ``MB(x)`` without ``y``
separates them ([1]_ Thm. 1, [2]_ Thm. 1, [3]_ Lemma A.2), so MB-by-MB tests every pair at ``x``
against the subsets of ``MB(x) \\ {y}``, never against ``x``'s shrinking adjacencies; the tests
of an adjacent pair grow as ``2 ** (|MB(x)| - 1)``. The same holds at ``y``, so MMB-by-MMB tests
the pair against the subsets of the smaller of ``MB(x) \\ {y}`` and ``MB(y) \\ {x}``, as the
Grow-Shrink search [4]_ and MBCS* [5]_ limit it, and the tests of an adjacent pair grow with the
smaller blanket. LOAD runs the skeleton step of PC
over ``MB+(x)`` as its Algorithm 11 states: a pair at ``x`` is tested like every other, against the
subsets of the adjacencies of either end, so its tests grow with the neighborhoods of the skeleton.
The v-structures of ``L_x`` that contain ``x`` are true ones ([1]_ Thm. 2, [2]_ Thm. 2); its other
edges may come from variables outside ``MB+(x)`` and stay out of the growing graph. With PC's
search, a v-structure at ``x`` can rest on an edge away from ``x`` that only a larger subset of the
blanket separates; MMB-by-MMB checks each from the far variables' own blankets
(:func:`confirm_colliders`), as its authors' code does. A pair the growing graph leaves apart is
known to be non-adjacent only when a test separated it; the orientation rules read a non-adjacency
only then.

Every candidate order follows the blanket learner's joining order, and every choice between tied
p-values the ranks of :func:`~andrey.core.ties.tie_ranks`, through
:meth:`~andrey.constraint.blanket.LocalSearch.strongest` and
:meth:`~andrey.constraint.blanket.LocalSearch.largest_sets`, so the answer does not depend on the
column order; it does not depend on the columns' units when the test does not.

References
----------
.. [1] Wang, C., Zhou, Y., Zhao, Q., and Geng, Z. (2014). Discovering and orienting the edges
   connected to a target variable in a DAG via a sequential local learning approach.
   Computational Statistics and Data Analysis 77, 252-266.
.. [2] Xie, F., Li, Z., Wu, P., Zeng, Y., Liu, C., and Geng, Z. (2024). Local causal structure
   learning in the presence of latent variables. Proceedings of the 41st International
   Conference on Machine Learning, PMLR 235, 54511-54530.
.. [3] Schubert, M., Claassen, T., and Magliacane, S. (2026). Local causal discovery for
   statistically efficient causal inference. Proceedings of the 29th International Conference
   on Artificial Intelligence and Statistics.
.. [4] Margaritis, D., and Thrun, S. (1999). Bayesian network induction via local
   neighborhoods. Advances in Neural Information Processing Systems 12, 505-511.
.. [5] Pellet, J.-P., and Elisseeff, A. (2008). Finding latent causes in causal networks: an
   efficient approach based on Markov blankets. Advances in Neural Information Processing
   Systems 21, 1249-1256.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import combinations
from typing import Literal

import numpy as np

from andrey.constraint.blanket import LEARNERS, LocalSearch
from andrey.constraint.fci import fci_orient, knowledge_marks
from andrey.constraint.pc import undirect_conflicts
from andrey.core import ARROW, CIRCLE, NULL, TAIL
from andrey.core.ci import counting_tests
from andrey.core.knowledge import Constraints
from andrey.core.orient import meek

#: A pair of variables, the lower position first.
Pair = tuple[int, int]
#: An unshielded collider ``(a, b, c)``: ``a *-> b <-* c``, the lower endpoint first.
Triple = tuple[int, int, int]


def _pair(u: int, v: int) -> Pair:
    return (u, v) if u < v else (v, u)


@dataclass(frozen=True)
class Rules:
    """The choice a method makes at each step of :func:`sequential`.

    ``local`` decides the pairs of ``L_x`` that do not hold ``x``: ``"ic"`` tests each against
    every subset of the other members of ``MB+(x)``, as the IC search of MB-by-MB [1]_; ``"pc"``
    and ``"pc_smaller"`` test each against the subsets of the current adjacencies of either end,
    level by level, as the PC search of MMB-by-MMB [2]_; ``"skeleton"`` does the same, as the
    skeleton step of PC that LOAD runs over ``MB+(x)`` ([3]_ Alg. 11). With ``"ic"`` and ``"pc"`` a
    pair ``x - y`` is tested against the subsets of ``MB(x)`` without ``y``; with ``"pc_smaller"``
    against those of ``MB(y)`` without ``x`` instead when that blanket holds ``x`` and has fewer
    members, which learns ``MB(y)`` ([2]_ Thm. 1 at ``y``, [4]_, [5]_ Sect. 4.2); with
    ``"skeleton"`` it is tested like every other pair, so the number of tests follows the
    neighborhoods of the skeleton, not ``MB(x)``. ``expand`` decides what joins the waitlist:
    ``"blanket"`` the members of ``MB(x)``, as soon as it is learned [1]_; ``"adjacent"`` the
    variables adjacent to ``x``, after the merge [2]_. ``orient`` decides the orientation:
    ``"meek"`` Meek's rules R1-R3 on a CPDAG [1]_; ``"pag"`` the FCI rules without selection bias
    on a PAG [2]_. ``stop`` decides the end: ``"blocked"`` drops from the waitlist, after every
    step, each variable that no path of edges without an arrowhead joins to the target, and stops
    when the waitlist is empty ([1]_ Steps 6-7); ``"r1_r3"`` stops at the first of the rules
    R1-R3 ([2]_ Thm. 3, :func:`stop_rule`), and ``"r1_r2"`` at the first of R1 and R2.
    ``colliders`` decides which v-structures at ``x`` join the growing graph: ``"local"`` every one
    of ``L_x`` [1]_ [2]_; ``"own_blanket"`` those the variables off ``x`` confirm from their own
    blankets (:func:`confirm_colliders`), as the code of MMB-by-MMB's authors checks them.
    ``stop_after`` decides when the rules ``"r1_r3"`` and ``"r1_r2"`` are first read: ``"first"``
    after every step [2]_; ``"degree"`` only once the search has taken more steps than the target
    has neighbors after its own step, as the authors' code reads them. An empty waitlist ends the
    search either way.

    R3 can stop the search before the target's marks are those of the PAG: it stops at an edge
    ``t o-> v`` whose circle a discriminating path through ``v`` decides, as on the DAG
    ``0 -> 3 <- 1 -> 4 <- 2 -> 3``, ``0 -> 5 <- 1``, ``5 -> 4``, where the PAG holds ``1 -> 4``
    and R3 stops at ``1 o-> 4`` after the target ``1`` alone; with ``stop_after="degree"`` the
    search goes on to three more blankets, which bring in the path. R1 and R2 stop only when the
    marks at the target are decided or no variable is left to learn. With the PC search, ``L_x``
    can keep an edge ``b - c`` away from ``x`` that only a larger subset of the blanket
    separates, and a v-structure ``x *-> b <-* c`` then brings in an edge the PAG lacks;
    ``colliders="own_blanket"`` drops it, since ``b``'s own step tests the pair against every
    subset of ``MB(b)``. With ``"pc_smaller"`` the separating set of a pair at ``x`` can hold
    variables outside ``MB+(x)``, so a v-structure ``x *-> b <-* c`` over such an edge ``b - c`` can
    come in even when ``b`` lies in every separating set inside the blanket; ``"pc_smaller"``
    therefore needs ``colliders="own_blanket"``, which checks ``b - c`` against ``b``'s own step.
    """

    local: Literal["ic", "pc", "pc_smaller", "skeleton"] = "ic"
    expand: Literal["blanket", "adjacent"] = "blanket"
    orient: Literal["meek", "pag"] = "meek"
    stop: Literal["blocked", "r1_r3", "r1_r2"] = "blocked"
    colliders: Literal["local", "own_blanket"] = "local"
    stop_after: Literal["first", "degree"] = "first"

    def __post_init__(self) -> None:
        for name, value, choices in (
            ("local", self.local, ("ic", "pc", "pc_smaller", "skeleton")),
            ("expand", self.expand, ("blanket", "adjacent")),
            ("orient", self.orient, ("meek", "pag")),
            ("stop", self.stop, ("blocked", "r1_r3", "r1_r2")),
            ("colliders", self.colliders, ("local", "own_blanket")),
            ("stop_after", self.stop_after, ("first", "degree")),
        ):
            if value not in choices:
                raise ValueError(f"{name} must be one of {choices}, got {value!r}")
        if self.stop_after == "degree" and self.stop == "blocked":
            raise ValueError("stop_after='degree' needs stop='r1_r3' or 'r1_r2'")
        if self.local == "pc_smaller" and self.colliders != "own_blanket":
            raise ValueError("local='pc_smaller' needs colliders='own_blanket'")


#: MB-by-MB as Wang et al. state it [1]_.
MB_BY_MB = Rules()
#: MB-by-MB as LOAD runs it ([3]_ Alg. 11): the skeleton step of PC in place of IC's search.
LOAD = Rules(local="skeleton")
#: MMB-by-MMB [2]_, with the collider check and the stop guard of its authors' code, each pair at
#: ``x`` tested on the smaller of the two blankets.
MMB_BY_MMB = Rules(
    local="pc_smaller",
    expand="adjacent",
    orient="pag",
    stop="r1_r3",
    colliders="own_blanket",
    stop_after="degree",
)


@dataclass(frozen=True)
class LocalGraph:
    """A local structure ``L_x``: its variables ``nodes`` (``MB+(x)``, ascending), its adjacent
    pairs, and its unshielded colliders."""

    nodes: tuple[int, ...]
    adjacent: frozenset[Pair]
    colliders: frozenset[Triple]

    def restrict(self, nodes: Iterable[int]) -> LocalGraph:
        """The substructure over ``nodes``: the pairs and colliders that lie inside them."""
        keep = frozenset(nodes)
        return LocalGraph(
            tuple(sorted(keep)),
            frozenset(p for p in self.adjacent if keep.issuperset(p)),
            frozenset(t for t in self.colliders if keep.issuperset(t)),
        )


@dataclass
class Cache:
    """What one run of :func:`sequential` leaves for the next over the same
    :class:`~andrey.constraint.blanket.LocalSearch`, as LOAD runs the search for several targets
    ([3]_ Alg. 11): each variable's blanket in joining order, each local structure, a separating
    set for each pair a test separated, keyed by the pair, the first one found, and the own local
    structures the collider check learned, with their separating sets (:func:`own_step`)."""

    blankets: dict[int, list[int]] = field(default_factory=dict)
    local: dict[int, LocalGraph] = field(default_factory=dict)
    sepsets: dict[Pair, tuple[int, ...]] = field(default_factory=dict)
    own: dict[int, tuple[LocalGraph, dict[Pair, tuple[int, ...]]]] = field(default_factory=dict)

    def separate(self, u: int, v: int, sepset: Iterable[int]) -> None:
        """Record ``sepset`` for ``(u, v)`` unless the pair has one."""
        self.sepsets.setdefault(_pair(u, v), tuple(sorted(sepset)))


class Network:
    """The growing graph over the variables its edges touch, as an endpoint-mark matrix.

    ``marks[i, j]`` is the mark at ``nodes[i]`` on its edge with ``nodes[j]``; ``end`` is the mark
    a new edge gets at both ends: a tail for a CPDAG, a circle for a PAG.
    """

    def __init__(self, end: int) -> None:
        self.end = end
        self.nodes: list[int] = []
        self.index: dict[int, int] = {}
        self._marks = np.zeros((0, 0), dtype=np.int8)

    @property
    def marks(self) -> np.ndarray:
        """The marks over :attr:`nodes`, a view that writes through."""
        k = len(self.nodes)
        return self._marks[:k, :k]

    def _slot(self, node: int) -> int:
        k = self.index.get(node)
        if k is None:
            k = len(self.nodes)
            if k == self._marks.shape[0]:
                grown = np.zeros((max(8, 2 * k),) * 2, dtype=np.int8)
                grown[:k, :k] = self._marks
                self._marks = grown
            self.nodes.append(node)
            self.index[node] = k
        return k

    def mark(self, u: int, v: int) -> int:
        """The mark at ``u`` on the edge ``u - v``, ``NULL`` when there is none."""
        i, j = self.index.get(u), self.index.get(v)
        return NULL if i is None or j is None else int(self._marks[i, j])

    def add(self, u: int, v: int) -> None:
        """Add the edge ``u - v`` with :attr:`end` at both ends, unless it is present."""
        i, j = self._slot(u), self._slot(v)
        if self._marks[i, j] == NULL:
            self._marks[i, j] = self._marks[j, i] = self.end

    def arrow(self, at: int, other: int) -> None:
        """Put an arrowhead at ``at`` on its present edge with ``other``."""
        self._marks[self.index[at], self.index[other]] = ARROW

    def neighbors(self, u: int) -> list[int]:
        """The variables adjacent to ``u``, in the order they joined the graph."""
        i = self.index.get(u)
        if i is None:
            return []
        return [self.nodes[j] for j in np.flatnonzero(self.marks[i]).tolist()]

    def dense(self, d: int) -> np.ndarray:
        """The marks over all ``d`` variables."""
        out = np.zeros((d, d), dtype=np.int8)
        out[np.ix_(self.nodes, self.nodes)] = self.marks
        return out

    def star(self, target: int, d: int) -> np.ndarray:
        """The marks over all ``d`` variables of the edges at ``target`` alone."""
        out = np.zeros((d, d), dtype=np.int8)
        for v in self.neighbors(target):
            out[target, v], out[v, target] = self.mark(target, v), self.mark(v, target)
        return out

    def local(self, nodes: Sequence[int]) -> LocalGraph:
        """The substructure over ``nodes``: its pairs, and as colliders the unshielded triples whose
        middle has an arrowhead from both ends ([1]_ Step 3, the case where every member of the
        blanket is done, so that every pair inside it is decided)."""
        inside = [v for v in sorted(nodes) if v in self.index]
        adjacent = {_pair(u, v) for u, v in combinations(inside, 2) if self.mark(u, v) != NULL}
        colliders = set()
        for b in inside:
            into = [a for a in inside if a != b and self.mark(b, a) == ARROW]
            for a, c in combinations(into, 2):
                if _pair(a, c) not in adjacent:
                    colliders.add((a, b, c))
        return LocalGraph(tuple(sorted(nodes)), frozenset(adjacent), frozenset(colliders))


@dataclass
class Result:
    """One run of :func:`sequential`.

    ``network`` is the graph the search grew; its edges at the target are the answer. ``done``
    lists the variables whose blanket and local structure the search used, in order; ``learned``
    those whose blanket this run learned, in order, the collider check's included, the rest coming
    from the ``cache``. ``stopped`` names
    the rule that ended it: ``"empty"`` (an empty waitlist), ``"determined"`` (R1: every mark at
    the target decided), or ``"blocked"`` (R3); the ``"blocked"`` rules of :class:`Rules` always
    end with an empty waitlist. ``n_tests`` counts the tests the run computed
    (:func:`~andrey.core.ci.counting_tests`).
    """

    network: Network
    done: list[int]
    learned: list[int]
    stopped: str
    n_tests: int
    cache: Cache


def sequential(
    search: LocalSearch,
    target: int,
    blanket: str,
    rules: Rules = MB_BY_MB,
    cache: Cache | None = None,
    *,
    local_depth: int | None = None,
) -> Result:
    """Learn the edges at ``target`` and their marks, one Markov blanket at a time.

    ``blanket`` names the learner of :data:`~andrey.constraint.blanket.LEARNERS` that gives each
    blanket; a learner of parents and children alone is refused. ``rules`` holds the choices of
    :class:`Rules`. ``cache`` carries the blankets, local structures, and separating sets of earlier
    runs over the same ``search`` and receives this run's; it is :attr:`Result.cache`.
    ``local_depth`` caps the conditioning sets of the local structures alone, below the search's
    ``max_depth``, which caps every test, the blanket learner's included.

    Each step takes the waitlist's head ``x`` and learns ``MB(x)``. ``L_x`` is a cached local
    structure, else the substructure of a done variable's ``L_y`` when ``MB+(x)`` lies inside
    ``MB+(y)`` (the first such ``y`` done), else the growing graph's substructure when every member
    of ``MB(x)`` is done, else learned from the data (:func:`learn_local`) ([1]_ Step 3, [2]_
    Alg. 1, [3]_ Alg. 11). With ``colliders="own_blanket"``, :func:`confirm_colliders` keeps the
    v-structures at ``x`` that the variables off ``x`` confirm. Then :func:`merge`, :func:`orient`,
    the waitlist (:func:`extend`), and the stop rule, read from the first step or, with
    ``stop_after="degree"``, once more steps are done than the target has neighbors after its own
    step. With the CPDAG rules, an edge whose two ends got arrowheads, from v-structures
    that disagree, is left undirected at the end. With the CPDAG rules and ``search`` constraints
    that direct edges (tiers, edges forbidden one way), each step directs those edges after the
    merge, over any arrowhead a v-structure put on them (:func:`direct`), and Meek's rule R4 joins
    R1-R3. With the PAG rules and ``search`` constraints, each step sets the endpoints the
    knowledge decides on the edges after the merge (:func:`mark_knowledge`), and the FCI rules
    keep them.

    Raises
    ------
    ValueError
        If ``blanket`` names a learner of parents and children alone.
    """
    if LEARNERS[blanket].kind == "pc":
        raise ValueError(f"{blanket!r} learns parents and children, not a Markov blanket")
    cache = Cache() if cache is None else cache
    network = Network(CIRCLE if rules.orient == "pag" else TAIL)
    decided = known = None
    if rules.orient == "meek" and search.constraints is not None:
        decided = search.constraints.decided()
        decided = decided if decided.any() else None
    elif search.constraints is not None:
        known = search.constraints
    waitlist, done, learned = [target], [], []
    stopped = "empty"
    degree = 0

    def blanket_of(v: int) -> list[int]:
        return learn_blanket(search, blanket, v, cache, learned)

    def own(v: int) -> tuple[LocalGraph, dict[Pair, tuple[int, ...]]]:
        return own_step(
            search, v, blanket, rules.local, cache, learned, depth=local_depth, of=blanket_of
        )

    with counting_tests() as tests, search.pinned():
        while waitlist:
            x = waitlist.pop(0)
            mb = learn_blanket(search, blanket, x, cache, learned)
            if rules.expand == "blanket":
                extend(waitlist, mb, done)
            done.append(x)
            local = _local_structure(
                search, x, mb, rules.local, cache, network, done, local_depth, blanket_of
            )
            if rules.colliders == "own_blanket":
                local = confirm_colliders(local, x, search, own)
            merge(network, local, x)
            if decided is not None:
                direct(network, decided)
            if known is not None:
                mark_knowledge(network, known)
            orient(network, cache.sepsets, rules.orient, search, rule4=decided is not None)
            if rules.expand == "adjacent":
                adjacent = set(network.neighbors(x))
                extend(waitlist, [y for y in mb if y in adjacent], done)
            if len(done) == 1:
                degree = len(network.neighbors(target))
            if rules.stop == "blocked":
                reach = open_reach(network, target)
                waitlist[:] = [w for w in waitlist if w in reach]
            elif rules.stop_after == "first" or len(done) > degree:
                rule = stop_rule(network, target, waitlist, r3=rules.stop == "r1_r3")
                if rule is not None:
                    stopped = rule
                    break
    if rules.orient == "meek" and network.nodes:
        network.marks[...] = undirect_conflicts(network.marks)
    return Result(network, done, learned, stopped, tests.n, cache)


def learn_blanket(
    search: LocalSearch, blanket: str, x: int, cache: Cache, learned: list[int]
) -> list[int]:
    """``x``'s blanket from ``cache``, else learned by the learner ``blanket`` and kept in ``cache``
    with the separating set of each variable it leaves out; ``x`` then joins ``learned``."""
    if x not in cache.blankets:
        _, mb, sepsets = search.local(blanket, x)
        cache.blankets[x] = list(mb or [])
        for y, sepset in sepsets.items():
            cache.separate(x, y, sepset)
        learned.append(x)
    return cache.blankets[x]


def extend(waitlist: list[int], members: Iterable[int], done: Iterable[int]) -> None:
    """Append to ``waitlist``, in place and in the order given, each of ``members`` neither in it
    nor ``done``."""
    seen = set(waitlist) | set(done)
    for y in members:
        if y not in seen:
            waitlist.append(y)
            seen.add(y)


def _local_structure(
    search: LocalSearch,
    x: int,
    mb: list[int],
    how: str,
    cache: Cache,
    network: Network,
    done: list[int],
    depth: int | None = None,
    of: Callable[[int], Sequence[int]] | None = None,
) -> LocalGraph:
    """``L_x`` by the cases of :func:`sequential`, kept in ``cache``; ``of`` gives a variable's
    blanket, as :func:`learn_local` reads it."""
    if x in cache.local:
        return cache.local[x]
    nodes = {x, *mb}
    local = None
    for y in done:
        if y != x and y in cache.local and nodes <= {y, *cache.blankets[y]}:
            local = cache.local[y].restrict(nodes)
            break
    if local is None:
        if set(mb) <= set(done):
            local = network.local(sorted(nodes))
        else:
            local = learn_local(search, x, mb, how, cache, depth=depth, of=of)
    cache.local[x] = local
    return local


def learn_local(
    search: LocalSearch,
    x: int,
    mb: Sequence[int],
    how: str,
    cache: Cache | None = None,
    *,
    depth: int | None = None,
    of: Callable[[int], Sequence[int]] | None = None,
) -> LocalGraph:
    """The local structure over ``MB+(x)``, ``x`` with its blanket ``mb``, learned from the data.

    With ``how="ic"`` and ``how="pc"`` a pair at ``x`` is tested against the subsets of ``mb``
    without its other end; with ``how="pc_smaller"`` a pair ``x - y`` is tested against the subsets
    of ``of(y)``, ``y``'s blanket, without ``x`` instead when ``of`` is given and that blanket holds
    ``x`` and has fewer members than ``mb``; those blankets are asked first, in the order of
    ``mb``. With ``how="ic"`` every other pair is tested against
    the subsets of the other members of ``MB+(x)``, and with ``how="pc"`` and ``"pc_smaller"``
    against the subsets of either end's adjacencies as they stand when the level begins. With
    ``how="skeleton"`` every pair, those at ``x`` included, is tested as ``"pc"`` tests the others.
    Sizes grow from zero; at each size every subset is tested (in one call where the test can), and
    a pair that some set of that size separates is removed after the level, with the set of the
    largest p-value, ties to the set whose ranks come first
    (:meth:`~andrey.constraint.blanket.LocalSearch.largest_sets`). No test conditions on more than
    the search's ``max_depth`` variables, nor on more than ``depth``. With knowledge, a pair it
    makes non-adjacent is never adjacent and a required one is never tested. A triple ``a - b - c``
    with ``a``, ``c`` separated is a collider when ``b`` is not in their set. Each set found goes
    into ``cache``.
    """
    nodes = tuple(sorted({x, *mb}))
    constraints = search.constraints
    support = None if constraints is None else constraints.support()
    required = None if constraints is None else constraints.required | constraints.required.T
    adjacent = {p for p in combinations(nodes, 2) if support is None or support[p]}
    fixed = {p for p in adjacent if required is not None and required[p]}
    blanket = sorted(mb)
    # The members' blankets in the blanket's joining order, so the order of the learned blankets,
    # and the separating sets their learner records first, do not depend on the column order.
    far = {y: of(y) for y in mb} if how == "pc_smaller" and of is not None else None
    sepsets: dict[Pair, tuple[int, ...]] = {}
    caps = [c for c in (search.max_depth, depth) if c is not None]
    cap = min(caps) if caps else None
    size = 0
    while cap is None or size <= cap:
        neighbors = {v: {u for p in adjacent if v in p for u in p if u != v} for v in nodes}
        removed: list[tuple[Pair, tuple[int, ...]]] = []
        tested = False
        for pair in sorted(adjacent - fixed):
            hits: list[tuple[tuple[int, ...], float]] = []
            within: set[int] | None = None
            sides = _sides(search, x, pair, blanket, nodes, neighbors, how, far)
            for first, other, among in sides:
                if len(among) < size:
                    continue
                tested = True
                hits += search.separating(first, other, among, size, within)
                within = set(among)
            if hits:
                sets = [s for s, _ in hits]
                best = search.largest_sets(np.array([[p for _, p in hits]]), sets)[0]
                removed.append((pair, tuple(best)))
        if not tested:
            break
        for pair, sepset in removed:
            adjacent.discard(pair)
            sepsets[pair] = sepset
        size += 1
    if cache is not None:
        for (u, v), sepset in sepsets.items():
            cache.separate(u, v, sepset)
    colliders = set()
    for b in nodes:
        around = sorted(u for p in adjacent if b in p for u in p if u != b)
        for a, c in combinations(around, 2):
            sepset = sepsets.get((a, c))
            if sepset is not None and b not in sepset:
                colliders.add((a, b, c))
    return LocalGraph(nodes, frozenset(adjacent), frozenset(colliders))


def _sides(
    search: LocalSearch,
    x: int,
    pair: Pair,
    blanket: list[int],
    nodes: tuple[int, ...],
    neighbors: Mapping[int, set[int]],
    how: str,
    far: Mapping[int, Sequence[int]] | None = None,
) -> list[tuple[int, int, list[int]]]:
    """The tests that decide ``pair`` in :func:`learn_local`: each as ``(first, other, among)``,
    the pair's ends in the order the test takes them and the ascending variables to condition on;
    ``far`` holds the blankets of ``x``'s members for ``how="pc_smaller"``.
    """
    u, v = pair
    if x in pair and how != "skeleton":
        y = v if u == x else u
        if how == "pc_smaller" and far is not None:
            other = far[y]
            if x in other and len(other) < len(blanket):
                return [(y, x, sorted(z for z in other if z != x))]
        return [(x, y, [z for z in blanket if z != y])]
    first = search.strongest({u: 0.0, v: 0.0})
    other = v if first == u else u
    if how == "ic":
        return [(first, other, [z for z in nodes if z not in pair])]
    return [
        (first, other, sorted(neighbors[first] - {other})),
        (other, first, sorted(neighbors[other] - {first})),
    ]


def own_step(
    search: LocalSearch,
    v: int,
    blanket: str,
    how: str,
    cache: Cache,
    learned: list[int],
    *,
    depth: int | None = None,
    of: Callable[[int], Sequence[int]] | None = None,
) -> tuple[LocalGraph, dict[Pair, tuple[int, ...]]]:
    """``v``'s own local structure, learned from the data over ``MB+(v)`` (:func:`learn_local`),
    and the separating sets it found, kept in ``cache.own`` for :func:`confirm_colliders`.

    ``v``'s blanket comes from ``cache`` or is learned on demand (:func:`learn_blanket`), as a
    step's blanket is. The local structure and its sets stay out of ``cache.local`` and
    ``cache.sepsets``, so a later step at ``v`` learns or reads its ``L_v`` as it would without
    the check.
    """
    if v not in cache.own:
        mb = learn_blanket(search, blanket, v, cache, learned)
        found = Cache()
        local = learn_local(search, v, mb, how, found, depth=depth, of=of)
        cache.own[v] = (local, found.sepsets)
    return cache.own[v]


def confirm_colliders(
    local: LocalGraph,
    x: int,
    search: LocalSearch,
    own: Callable[[int], tuple[LocalGraph, Mapping[Pair, tuple[int, ...]]]],
) -> LocalGraph:
    """``local`` with only the colliders at ``x`` that the variables off ``x`` confirm from their
    own blankets, as the code of MMB-by-MMB's authors checks its v-structures; ``own(v)`` gives
    ``v``'s own local structure and its separating sets (:func:`own_step`).

    A collider with ``x`` at an end, ``x *-> b <-* c``, is kept when ``b`` and ``c`` are adjacent
    in ``b``'s own local structure, where the pair is tested against every subset of ``MB(b)``
    without ``c``. A collider ``a *-> x <-* c`` is kept when ``a`` and ``c`` are separated by a
    set without ``x`` in the own local structure of ``a`` or, failing that, of ``c``, the one of
    lower rank asked first. With exact tests and uncapped sets, a true unshielded collider always
    passes, so the check removes only false ones. The colliders are checked in the order of the
    ranks of their members other than ``x``, so the blankets learned on demand join ``cache`` in
    an order free of the column order.
    """
    rank = {v: k for k, v in enumerate(search.ordered(dict.fromkeys(local.nodes, 0)))}
    kept = {t for t in local.colliders if x not in t}
    at_x = [t for t in local.colliders if x in t]
    for a, b, c in sorted(at_x, key=lambda t: sorted(rank[v] for v in t if v != x)):
        if b == x:
            ends = sorted((a, c), key=rank.__getitem__)
            sets = (own(u)[1].get((a, c)) for u in ends)
            confirmed = any(s is not None and x not in s for s in sets)
        else:
            confirmed = _pair(b, c if a == x else a) in own(b)[0].adjacent
        if confirmed:
            kept.add((a, b, c))
    return replace(local, colliders=frozenset(kept))


def merge(network: Network, local: LocalGraph, x: int) -> None:
    """Put into ``network`` the edges of ``local`` at ``x`` and its colliders that contain ``x``:
    the edges a collider names are added and get an arrowhead at its middle ([1]_ Step 4, [2]_
    Alg. 1 line 12). An arrowhead once placed stays, so the merge does not depend on the order of
    the steps."""
    for u, v in local.adjacent:
        if x in (u, v):
            network.add(u, v)
    for a, b, c in local.colliders:
        if x in (a, b, c):
            network.add(a, b)
            network.add(c, b)
            network.arrow(b, a)
            network.arrow(b, c)


def direct(network: Network, decided: np.ndarray) -> None:
    """Direct in place each edge of ``network`` that ``decided`` directs, ``decided[u, v]`` meaning
    ``u -> v`` (:meth:`~andrey.core.knowledge.Constraints.decided`: a tier or an edge forbidden one
    way), over any arrowhead a v-structure put on it."""
    nodes = network.nodes
    marks = network.marks
    rows, cols = np.nonzero(decided[np.ix_(nodes, nodes)] & (marks != NULL))
    marks[rows, cols] = TAIL
    marks[cols, rows] = ARROW


def mark_knowledge(network: Network, constraints: Constraints) -> None:
    """Set in place the endpoints ``constraints`` decide on the edges of a PAG ``network``, as
    :func:`~andrey.constraint.fci.knowledge_marks` sets them on FCI's graph: an arrowhead at the
    end of an edge forbidden out of it alone (a tier, or an edge forbidden one way), and a tail at
    the cause of a required edge. Path knowledge is not read."""
    nodes = network.nodes
    inside = np.ix_(nodes, nodes)
    local = replace(
        constraints,
        forbidden=constraints.forbidden[inside],
        required=constraints.required[inside],
        tier=constraints.tier[nodes],
        required_paths=None,
        forbidden_paths=None,
        confounded=None,
    )
    knowledge_marks(network.marks, local)


def _apart(network: Network, sepsets: Mapping[Pair, tuple[int, ...]]) -> np.ndarray:
    """``[i, j]``: ``network.nodes[i]`` and ``[j]`` share no edge and a test separated them."""
    nodes = network.nodes
    k = len(nodes)
    apart = np.zeros((k, k), dtype=bool)
    marks = network.marks
    for i, j in combinations(range(k), 2):
        if marks[i, j] == NULL and _pair(nodes[i], nodes[j]) in sepsets:
            apart[i, j] = apart[j, i] = True
    return apart


def unknown_triples(network: Network, sepsets: Mapping[Pair, tuple[int, ...]]) -> frozenset[Triple]:
    """The unshielded triples ``a - b - c`` of ``network``, in its positions, that no recorded
    separating set of ``a`` and ``c`` holding ``b`` shows to be non-colliders.

    Meek's rule R1 and the FCI rules read an unshielded triple as a non-collider; in a graph that
    holds only some edges, ``a`` and ``c`` may be adjacent without an edge yet, so a rule may read
    the triple only when a test separated ``a`` and ``c`` given a set that holds ``b`` ([1]_, the
    procedure Orient-Undirected-Edges).
    """
    nodes, marks = network.nodes, network.marks
    unknown = set()
    for b in range(len(nodes)):
        around = np.flatnonzero(marks[b]).tolist()
        for a, c in combinations(around, 2):
            if marks[a, c] != NULL:
                continue
            sepset = sepsets.get(_pair(nodes[a], nodes[c]))
            if sepset is None or nodes[b] not in sepset:
                unknown.add((a, b, c))
    return frozenset(unknown)


def orient(
    network: Network,
    sepsets: Mapping[Pair, tuple[int, ...]],
    how: str,
    search: LocalSearch,
    *,
    rule4: bool = False,
) -> None:
    """Orient ``network`` in place: with ``how="meek"`` by Meek's rules R1-R3 [1]_, and R4 with
    ``rule4`` (for edges background knowledge directed), with ``how="pag"`` by the FCI rules
    without selection bias [2]_.

    The rules read a non-adjacency only for a pair a test separated, and an unshielded triple as a
    non-collider only when that test's set holds its middle (:func:`unknown_triples`). The FCI
    rules' discriminating-path tests run through ``search`` (its ``scan``), so they are counted.
    """
    if not network.nodes:
        return
    unknown = unknown_triples(network, sepsets)
    if how == "meek":
        network.marks[...] = meek(network.marks, rule4=rule4, ambiguous=unknown)
        return
    nodes, index = network.nodes, network.index
    local_sepsets = {}
    for i, j in combinations(range(len(nodes)), 2):
        sepset = sepsets.get(_pair(nodes[i], nodes[j]))
        if sepset is not None:
            local_sepsets[i, j] = tuple(index[z] for z in sepset if z in index)

    def test(x: int, y: int, condition_set: Iterable[int] = ()) -> float:
        cond = [nodes[z] for z in condition_set]
        return float(search.scan(nodes[x], [nodes[y]], cond)[1][0])

    marks = network.marks
    fci_orient(
        marks,
        test,
        search.alpha,
        local_sepsets,
        unknown,
        max_depth=search.max_depth,
        selection_bias=False,
        apart=_apart(network, sepsets),
    )


def open_reach(network: Network, target: int) -> set[int]:
    """The variables a path of edges without an arrowhead joins to ``target`` in ``network``: in a
    CPDAG, the undirected paths. A waitlist variable outside it is one whose every path to the
    target is blocked by a directed edge, so learning it cannot orient the target's edges ([1]_
    Step 6)."""
    reach, stack = {target}, [target]
    while stack:
        u = stack.pop()
        for v in network.neighbors(u):
            if v not in reach and ARROW not in (network.mark(u, v), network.mark(v, u)):
                reach.add(v)
                stack.append(v)
    return reach


def _undecided(network: Network, target: int, v: int) -> bool:
    """Whether the edge ``target - v`` has a mark left to decide: a circle at either end, or a
    tail at both ends of a CPDAG's undirected edge."""
    near, far = network.mark(target, v), network.mark(v, target)
    if network.end == CIRCLE:
        return CIRCLE in (near, far)
    return near == TAIL and far == TAIL


def stop_rule(
    network: Network, target: int, waitlist: Sequence[int], *, r3: bool = True
) -> str | None:
    """The first of MMB-by-MMB's stop rules that holds ([2]_ Thm. 3), or ``None``.

    ``"determined"`` (R1): no edge at ``target`` has a mark left to decide. ``"empty"`` (R2): the
    waitlist is empty. ``"blocked"`` (R3, read only with ``r3``): every path from ``target`` that
    begins with an edge whose marks are not all decided meets an arrowhead pointing away from the
    target before it reaches a waitlist variable; a path walks from ``u`` to ``v`` while ``v``'s
    mark on their edge is not an arrowhead.
    """
    undecided = [v for v in network.neighbors(target) if _undecided(network, target, v)]
    if not undecided:
        return "determined"
    if not waitlist:
        return "empty"
    if not r3:
        return None
    reach: set[int] = set()
    stack = [v for v in undecided if network.mark(v, target) != ARROW]
    reach.update(stack)
    while stack:
        u = stack.pop()
        for v in network.neighbors(u):
            if v != target and v not in reach and network.mark(v, u) != ARROW:
                reach.add(v)
                stack.append(v)
    if reach.isdisjoint(waitlist):
        return "blocked"
    return None
