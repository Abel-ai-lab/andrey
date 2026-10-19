"""CDNOD, the engine behind :func:`andrey.cdnod`.

The algorithm, its settings, and its references are described on :func:`andrey.cdnod`.

The context index joins the data as a last variable. Every edge the context keeps is oriented out
of it, then colliders are oriented as PC orients them, before Meek's rules run. With a
``dependence_threshold``, the independent-change step (Huang et al. 2020, Section 4.2, Algorithm
3) then orients edges between variables that both change with the index, and Meek's rules run
again. The step follows Meek's rules, so it never contradicts an orientation they force. The step
skips any orientation that would close a directed cycle, since the paper's causal graph is a DAG
(Section 2.2) and its Algorithm 3 does not address the case. The context node is dropped from the
returned graph, which is partially directed and tagged ``cpdag``.

The step measures, for each hypothesis ``cause -> effect`` given a conditioning set ``Z``, the
dependence between how ``P(cause, Z)`` and ``P(effect | cause, Z)`` change across the index:
Eqs. 6 and 12 of the paper give the Gram matrices of the kernel embeddings of those two modules at
each row's index value, and Eq. 15, the normalized HSIC of the two, is the measure (Eqs. 16 and
17). A measure at most ``dependence_threshold`` counts as independent. The paper leaves several
choices open, and these are fixed here: every column is standardized; each Gaussian kernel on the
data has the width ``sigma`` 0.8 below 200 rows, 0.5 from 200 to 1,199 rows, and 0.3 from 1,200
rows (Zhang et al. 2011), times the square root of its column count; the index kernel is 1 for
two rows with the same index value and 0 otherwise, so each value is one domain; and the ridge
``lambda`` is 0.1. The paper fits the data kernels, the index kernel, and ``lambda`` by
Gaussian-process likelihood.

The conditioning sets follow the paper's restriction to neighbors of one end, ``vl``, of the
pair ``(vl, vk)``, and both orders of each pair are tried. A neighbor ``z`` of ``vl`` qualifies
when it is not ``vk``, the context node, or a descendant of either end along directed edges, and
the edge is ``z -> vl`` or ``z -- vl``. A walk from ``z`` to ``vk`` that avoids ``vl`` and the
context node and has no definite collider then sorts it: ``z -> vl`` with a walk that ends in an
arrow into ``vk`` puts ``z`` in the deconfounding set ``Z1``; any other qualifying ``z`` with a
walk that ends in an arrow into ``vk`` or an undirected edge at ``vk`` puts it in the potential
set ``Z2``.
"""

from __future__ import annotations

from collections import deque
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np

from andrey.constraint.pc import (
    _discover_skeleton,
    _orient_colliders,
    _undirect_conflicts,
    check_collider_order,
    check_knowledge_cycle,
)
from andrey.core import ARROW, NULL, TAIL, GraphStructure
from andrey.core.independence import _center_kernel, _gaussian_kernel
from andrey.core.knowledge import orient
from andrey.core.orient import _is_directed, _reaches_directed, meek, to_structure

if TYPE_CHECKING:
    from andrey.core.knowledge import Constraints

_RIDGE = 0.1


def cdnod(
    data: np.ndarray,
    domain_index: np.ndarray,
    *,
    alpha: float = 0.05,
    indep_test: str = "fisherz",
    collider_order: str = "p_value",
    categorical: np.ndarray | None = None,
    dependence_threshold: float | None = None,
    max_depth: int | None = None,
    constraints: Constraints | None = None,
) -> GraphStructure:
    """Discover a partially directed graph from multi-domain ``data`` with CDNOD and ``indep_test``.

    ``data`` is an ``(n_samples, n_features)`` array; ``domain_index`` is the ``(n_samples,)`` or
    ``(n_samples, 1)`` domain/context index (one label per row); ``alpha`` is the independence-test
    significance level in ``(0, 1)`` (a pair is separated when its p-value exceeds ``alpha``);
    ``indep_test`` selects the CI test. ``collider_order`` orders the colliders as in
    :func:`~andrey.constraint.pc.pc`. ``dependence_threshold``, when given, runs the
    independent-change step with that threshold on the module dependence measure; ``None`` skips
    it. The context is appended as an extra variable, the tests run over the augmented matrix, and
    the result is projected back to the ``n_features`` data variables; with a single domain the
    context is left out and the search runs on ``data`` alone. ``categorical``, a boolean mask of
    the data's categorical columns, reaches a mixed-data test, which also takes the context as
    categorical. ``max_depth`` caps the conditioning-set size of the skeleton search.
    ``constraints``, over the data's columns, constrain the skeleton search and the orientations as
    in :func:`~andrey.constraint.pc.pc`; the context is unconstrained. Every edge they decide is
    directed before the independent-change step, which orients only undirected edges, so it never
    orients against them. Returns a ``GraphStructure`` tagged ``"cpdag"``: its directions include
    those the index identifies, so it is not an observational CPDAG.
    """
    check_collider_order(collider_order)
    X = np.asarray(data, dtype=np.float64)
    context = np.asarray(domain_index, dtype=np.float64)
    # One value per row: a wider index reshaped to a column would pair rows with the wrong values.
    if context.shape not in ((X.shape[0],), (X.shape[0], 1)):
        raise ValueError(
            f"domain_index must hold one value per row of data, as shape ({X.shape[0]},) or "
            f"({X.shape[0]}, 1); got shape {context.shape}"
        )
    if not np.isfinite(context).all():
        raise ValueError("domain_index contains NaN or inf")
    context = context.reshape(-1, 1)
    d = X.shape[1]
    ranked = collider_order == "p_value"
    rule4 = constraints is not None
    if np.ptp(context) == 0:
        # One domain: no change to detect, and a constant has no correlation to test, so the
        # search runs on ``data`` alone and gives PC's graph.
        skeleton, sepsets, pvalues = _discover_skeleton(
            X,
            alpha,
            indep_test,
            categorical=categorical,
            record_pvalues=ranked,
            max_depth=max_depth,
            constraints=constraints,
        )
        pdag = _orient_with_context(
            skeleton,
            sepsets,
            pvalues,
            context=None,
            order=collider_order,
            constraints=constraints,
        )
    else:
        augmented = np.concatenate([X, context], axis=1)
        mask = None if categorical is None else np.append(np.asarray(categorical, bool), True)
        padded = None if constraints is None else constraints.padded(d + 1)
        skeleton, sepsets, pvalues = _discover_skeleton(
            augmented,
            alpha,
            indep_test,
            categorical=mask,
            record_pvalues=ranked,
            max_depth=max_depth,
            constraints=padded,
        )
        pdag = _orient_with_context(
            skeleton, sepsets, pvalues, context=d, order=collider_order, constraints=padded
        )
        if dependence_threshold is not None:
            changing = np.nonzero(skeleton[d] != NULL)[0].tolist()
            pdag = _orient_by_independent_changes(
                meek(pdag, rule4=rule4), X, context[:, 0], changing, dependence_threshold
            )
    completed = _undirect_conflicts(meek(pdag, rule4=rule4))
    return to_structure(completed[:d, :d], kind="cpdag")


def _orient_with_context(
    skeleton: np.ndarray,
    sepsets: dict[tuple[int, int], tuple[int, ...]],
    pvalues: dict[tuple[int, int], float],
    *,
    context: int | None,
    order: str = "p_value",
    constraints: Constraints | None = None,
) -> np.ndarray:
    """Orient the augmented skeleton: context edges outward, then unshielded colliders.

    Every edge the ``context`` node keeps becomes ``context -> neighbor`` first, so the context is
    a source in every triple it touches, and every edge ``constraints`` decide is directed with it;
    decided edges that close a directed cycle raise ``ValueError``
    (:func:`~andrey.constraint.pc.check_knowledge_cycle`).
    Colliders are then oriented as PC orients them, in ``order``
    (:func:`~andrey.constraint.pc._orient_colliders`), which skips a collider when an edge it would
    orient already has an arrowhead at the other end; so none puts an arrowhead at the context or
    reverses a decided edge. With ``context=None``, for a skeleton without a context node, only the
    decided edges and the colliders are oriented.
    """
    pdag = skeleton.copy()
    if context is not None:
        neighbors = np.flatnonzero(skeleton[context] != NULL)
        pdag[context, neighbors] = TAIL
        pdag[neighbors, context] = ARROW
    if constraints is not None:
        pdag = orient(pdag, constraints)
        check_knowledge_cycle((pdag == TAIL) & (pdag.T == ARROW))
    return _orient_colliders(pdag, sepsets, pvalues, order=order)


def _orient_by_independent_changes(
    pdag: np.ndarray,
    data: np.ndarray,
    context: np.ndarray,
    changing: list[int],
    threshold: float,
) -> np.ndarray:
    """Orient undirected edges among the ``changing`` variables by independent module changes.

    Huang et al. 2020, Algorithm 3. ``pdag`` holds the data variables first and may hold more
    nodes after them, such as the context node, which never enter a conditioning set. ``data`` is
    ``(n_samples, n_features)`` and ``context`` the index, one value per row. Starting from
    ``n = 0``, each undirected pair ``(vl, vk)`` whose sets give ``|Z1| + |Z2| >= n`` is tested
    with ``Z1`` and each ``n - |Z1|`` members of ``Z2``. ``vk -> vl`` is oriented when the measure
    for ``vk -> vl`` is at most ``threshold`` and the one for ``vl -> vk`` is above it, and the
    members of ``Z2`` in that test that share an undirected edge with ``vl`` or ``vk`` are oriented
    into it; ``vl -> vk`` is the mirror case. An orientation that would close a directed cycle is
    skipped, since the causal graph is acyclic. After an orientation the pairs are visited again;
    ``n`` grows until no undirected pair has that many candidates. Returns an updated copy.
    """
    pdag = pdag.copy()
    measure = _ModuleDependence(data, context)
    size = 0
    while True:
        oriented, any_large = _orient_one(pdag, sorted(changing), size, measure, threshold)
        if oriented:
            continue
        if not any_large:
            return pdag
        size += 1


def _orient_one(
    pdag: np.ndarray,
    changing: list[int],
    size: int,
    measure: _ModuleDependence,
    threshold: float,
) -> tuple[bool, bool]:
    """Orient the first undirected pair that the tests at set size ``size`` decide.

    Returns whether a pair was oriented, and whether any pair visited has ``|Z1| + |Z2| >= size``.
    """
    n_features = measure.n_features
    any_large = False
    for a, b in combinations(changing, 2):
        if not (pdag[a, b] == TAIL and pdag[b, a] == TAIL):
            continue
        for vl, vk in ((a, b), (b, a)):
            z1, z2 = _deconfounding_sets(pdag, vl, vk, n_features)
            if len(z1) + len(z2) < size:
                continue
            any_large = True
            if size < len(z1):
                continue
            for extra in combinations(z2, size - len(z1)):
                cond = z1 + list(extra)
                k_to_l = measure(vk, vl, cond)
                l_to_k = measure(vl, vk, cond)
                if k_to_l <= threshold < l_to_k and not _reaches_directed(pdag, vl, vk):
                    _orient_pair(pdag, vk, vl, extra)
                    return True, True
                if l_to_k <= threshold < k_to_l and not _reaches_directed(pdag, vk, vl):
                    _orient_pair(pdag, vl, vk, extra)
                    return True, True
    return False, any_large


def _orient_pair(pdag: np.ndarray, cause: int, effect: int, extra: tuple[int, ...]) -> None:
    """Orient ``cause -> effect``, and each ``z -- cause`` or ``z -- effect`` with ``z`` in
    ``extra`` into that end unless ``z -> end`` would close a directed cycle."""
    pdag[cause, effect], pdag[effect, cause] = TAIL, ARROW
    for z in extra:
        for end in (cause, effect):
            undirected = pdag[z, end] == TAIL and pdag[end, z] == TAIL
            if undirected and not _reaches_directed(pdag, end, z):
                pdag[z, end], pdag[end, z] = TAIL, ARROW


def _deconfounding_sets(
    pdag: np.ndarray, vl: int, vk: int, n_features: int
) -> tuple[list[int], list[int]]:
    """Return ``(Z1, Z2)`` for the pair ``(vl, vk)``, members drawn from the neighbors of ``vl``.

    See the module docstring for the rule. Only data variables (indices below ``n_features``)
    are candidates, and walks never pass through other nodes.
    """
    descendants = _descendants(pdag, (vl, vk))
    z1: list[int] = []
    z2: list[int] = []
    for z in range(n_features):
        if z in (vl, vk) or z in descendants or pdag[z, vl] == NULL:
            continue
        into_l = _is_directed(pdag, z, vl)
        ends = _walk_ends(pdag, z, vl, vk, n_features)
        if into_l and "arrow" in ends:
            z1.append(z)
        elif ends:
            z2.append(z)
    return z1, z2


def _descendants(pdag: np.ndarray, sources: tuple[int, ...]) -> set[int]:
    """Nodes reached from ``sources`` along directed edges, the sources excluded."""
    seen: set[int] = set()
    queue = deque(sources)
    while queue:
        node = queue.popleft()
        for child in np.nonzero(pdag[node] != NULL)[0].tolist():
            if _is_directed(pdag, node, child) and child not in seen and child not in sources:
                seen.add(child)
                queue.append(child)
    return seen


def _walk_ends(pdag: np.ndarray, start: int, vl: int, vk: int, n_features: int) -> set[str]:
    """How walks from ``start`` (entered from ``vl``) reach ``vk`` with no definite collider.

    Returns ``"arrow"`` when some walk ends ``u -> vk`` and ``"undirected"`` when some walk ends
    ``u -- vk``; a walk ending ``vk -> u`` leaves ``vk`` through an arrow and counts for neither.
    The walks avoid ``vl`` and every node at or above ``n_features``.
    """
    ends: set[str] = set()
    seen = {(start, False)}
    queue = deque(seen)
    while queue:
        node, arrived_into = queue.popleft()
        for nxt in np.nonzero(pdag[node] != NULL)[0].tolist():
            if nxt == vl or nxt >= n_features:
                continue
            if arrived_into and pdag[node, nxt] == ARROW:  # a definite collider at ``node``
                continue
            if nxt == vk:
                if pdag[vk, node] == ARROW:
                    ends.add("arrow")
                elif pdag[node, vk] == TAIL:
                    ends.add("undirected")
                continue
            state = (nxt, bool(pdag[nxt, node] == ARROW))
            if state not in seen:
                seen.add(state)
                queue.append(state)
    return ends


class _ModuleDependence:
    """The module dependence measure of Eqs. 15-17, cached by ``(cause, effect, Z)``.

    ``self(cause, effect, cond)`` is the normalized HSIC between the Gram matrix of the embeddings
    of ``P(cause, cond)`` (Eq. 12) and that of ``P(effect | cause, cond)`` (Eq. 6) over the rows'
    index values. With the 0/1 index kernel ``Kc = E E^T`` (``E`` one-hot by index value), Eq. 12
    is ``E D (E^T Kx E) D E^T`` with ``D = diag(1 / (n_c + lambda))`` and Eq. 6 is
    ``E E^T (Kx^3 * A^-1 Ky A^-1) E E^T`` with ``A = Kx * Kc + lambda I`` block diagonal by index
    value; the constant factors cancel in Eq. 15.
    """

    def __init__(self, data: np.ndarray, context: np.ndarray) -> None:
        self.n_features = data.shape[1]
        spread = data.std(axis=0)
        self._data = (data - data.mean(axis=0)) / np.where(spread > 0, spread, 1.0)
        values, codes = np.unique(context, return_inverse=True)
        self._onehot = np.eye(len(values))[codes]
        self._blocks = [np.nonzero(codes == c)[0] for c in range(len(values))]
        self._shrink = 1.0 / (self._onehot.sum(axis=0) + _RIDGE)
        self._kernels: dict[tuple[int, ...], np.ndarray] = {}
        self._cache: dict[tuple[int, int, tuple[int, ...]], float] = {}

    def __call__(self, cause: int, effect: int, cond: list[int]) -> float:
        key = (cause, effect, tuple(sorted(cond)))
        if key not in self._cache:
            given = (cause, *key[2])
            kx = self._kernel(given)
            marginal = self._expand(self._shrink[:, None] * self._sum(kx) * self._shrink[None, :])
            self._cache[key] = _normalized_hsic(
                marginal, self._conditional(self._kernel((effect,)), kx)
            )
        return self._cache[key]

    def _kernel(self, columns: tuple[int, ...]) -> np.ndarray:
        if columns not in self._kernels:
            block = self._data[:, list(columns)]
            n = block.shape[0]
            sigma = 0.8 if n < 200 else 0.5 if n < 1200 else 0.3
            self._kernels[columns] = _gaussian_kernel(block, 1.0 / (sigma**2 * block.shape[1]))
        return self._kernels[columns]

    def _sum(self, matrix: np.ndarray) -> np.ndarray:
        """``E^T M E``: the sums of ``M`` over each pair of index values."""
        return self._onehot.T @ matrix @ self._onehot

    def _expand(self, core: np.ndarray) -> np.ndarray:
        """``E C E^T``: a per-index-value matrix spread over the rows."""
        return self._onehot @ core @ self._onehot.T

    def _conditional(self, ky: np.ndarray, kx: np.ndarray) -> np.ndarray:
        """Eq. 6 up to a constant: ``Kc (Kx^3 * A^-1 Ky A^-1) Kc``."""
        left = np.empty_like(ky)
        for rows in self._blocks:  # A^-1 Ky, one index value at a time
            block = kx[np.ix_(rows, rows)] + _RIDGE * np.eye(len(rows))
            left[rows] = np.linalg.solve(block, ky[rows])
        middle = np.empty_like(ky)
        for rows in self._blocks:  # A^-1 (A^-1 Ky)^T = A^-1 Ky A^-1
            block = kx[np.ix_(rows, rows)] + _RIDGE * np.eye(len(rows))
            middle[rows] = np.linalg.solve(block, left.T[rows])
        return self._expand(self._sum((kx @ kx @ kx) * middle))


def _normalized_hsic(first: np.ndarray, second: np.ndarray) -> float:
    """Eq. 15: ``tr(M1 H M2 H) / (tr(M1 H) tr(M2 H))``; 0 when either module does not change."""
    a, b = _center_kernel(first), _center_kernel(second)
    scale_a, scale_b = np.trace(a), np.trace(b)
    if scale_a <= 1e-12 * np.trace(first) or scale_b <= 1e-12 * np.trace(second):
        return 0.0
    return float(np.sum(a * b) / (scale_a * scale_b))
