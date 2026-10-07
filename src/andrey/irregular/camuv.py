"""CAM-UV: causal additive models with unobserved variables.

CAM-UV finds each observed variable's direct parents when some common causes are unobserved. It
assumes each variable is a sum of nonlinear functions of its parents plus independent noise. Over
variable subsets of growing size, it regresses one member on the others and its known parents
(an additive cubic-spline fit), and accepts the others as parents when the residual is independent
of them (gamma HSIC) and they stay dependent on the child's residual; the subset size then starts
over. A last pass drops each parent whose residual is independent of the child's. A pair with no
edge that stays dependent after both are regressed on their parents shares an unobserved cause.

References
----------
Maeda, Shimizu. "Causal additive models with unobserved variables." UAI 37, 2021.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

from andrey.core.independence import hsic_test


def _column(values: np.ndarray) -> np.ndarray:
    """Return ``values`` as a float64 ``(n, 1)`` column."""
    return np.asarray(values, dtype=np.float64).reshape(-1, 1)


def _regress_residual(inputs: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the residual of an additive nonlinear fit of ``target`` on ``inputs``.

    ``inputs`` is an ``(n, k)`` matrix of explanatory columns; with ``k == 0`` the residual is the
    target itself. Each explanatory column is expanded into a cubic B-spline basis and the bases are
    concatenated, so a linear least-squares fit over them is an additive model (no cross terms) --
    the causal-additive assumption. The residual is ``target`` minus the fitted values.
    """
    y = np.asarray(target, dtype=np.float64).ravel()
    X = np.asarray(inputs, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[1] == 0:
        return y
    model = make_pipeline(
        SplineTransformer(n_knots=5, degree=3, include_bias=False),
        LinearRegression(),
    )
    model.fit(X, y)
    return y - model.predict(X)


def _residual(data: np.ndarray, child: int, parents: list[int]) -> np.ndarray:
    """Return variable ``child``'s residual after regressing it on ``parents`` (raw columns)."""
    if len(parents) == 0:
        return np.asarray(data[:, child], dtype=np.float64)
    return _regress_residual(data[:, parents], data[:, child])


def _neighborhoods(data: np.ndarray, alpha: float) -> list[set[int]]:
    """Return, per variable, the set of variables it is marginally dependent on (HSIC p < alpha)."""
    d = data.shape[1]
    neighbors: list[set[int]] = [set() for _ in range(d)]
    for i in range(d):
        for j in range(i + 1, d):
            if hsic_test(_column(data[:, i]), _column(data[:, j])) < alpha:
                neighbors[i].add(j)
                neighbors[j].add(i)
    return neighbors


def _is_identifiable(subset: tuple[int, ...], parents: list[set[int]]) -> bool:
    """True when no two members of ``subset`` already stand in a parent relation."""
    members = list(subset)
    for pos, i in enumerate(members):
        for j in members[pos + 1 :]:
            if j in parents[i] or i in parents[j]:
                return False
    return True


def _all_dependent(child: int, candidates: set[int], neighbors: list[set[int]]) -> bool:
    """True when every candidate parent is a marginal neighbor of ``child``."""
    return all(candidate in neighbors[child] for candidate in candidates)


def _select_child(
    data: np.ndarray,
    subset: tuple[int, ...],
    parents: list[set[int]],
    neighbors: list[set[int]],
    residuals: np.ndarray,
) -> tuple[int | None, float]:
    """Return the subset member whose residual is most independent of the other members.

    For each candidate child the other members are its candidate parents; the child is regressed on
    those candidates together with its committed parents and the residual is scored for independence
    against the candidates' current residual columns. The child with the largest independence
    p-value wins (ties resolve to the lower index via the sorted scan).
    """
    best_child: int | None = None
    best_independence = 0.0
    for child in sorted(subset):
        candidates = set(subset) - {child}
        if not _all_dependent(child, candidates, neighbors):
            continue
        explanatory = sorted(candidates | parents[child])
        residual = _residual(data, child, explanatory)
        independence = hsic_test(_column(residual), residuals[:, sorted(candidates)])
        if best_independence < independence:
            best_independence = independence
            best_child = child
    return best_child, best_independence


def _candidates_stay_dependent(
    candidates: set[int], child: int, alpha: float, residuals: np.ndarray
) -> bool:
    """True when the child's residual column stays dependent (HSIC p <= alpha) on each candidate."""
    for candidate in candidates:
        if hsic_test(_column(residuals[:, child]), _column(residuals[:, candidate])) > alpha:
            return False
    return True


def _find_parents(
    data: np.ndarray,
    alpha: float,
    max_explanatory_vars: int,
    neighbors: list[set[int]],
) -> list[set[int]]:
    """Search subsets of growing size for accepted (child, parents) additive relations.

    A subset yields a parent set when its selected child's residual is independent of the candidate
    parents (p > alpha) yet those parents stay dependent on the child's running residual. Committing
    a parent replaces the child's residual column and resets the subset size; when a full sweep adds
    nothing the size grows, up to ``max_explanatory_vars``. A closing pass prunes any parent whose
    residual is independent of the child's residual.
    """
    d = data.shape[1]
    parents: list[set[int]] = [set() for _ in range(d)]
    residuals = np.array(data, dtype=np.float64)
    size = 2
    while True:
        changed = False
        for subset in combinations(range(d), size):
            if not _is_identifiable(subset, parents):
                continue
            child, independence = _select_child(data, subset, parents, neighbors, residuals)
            if child is None or not independence > alpha:
                continue
            candidates = set(subset) - {child}
            if not _candidates_stay_dependent(candidates, child, alpha, residuals):
                continue
            for parent in candidates:
                parents[child].add(parent)
                changed = True
                residuals[:, child] = _residual(data, child, sorted(parents[child]))
        if changed:
            size = 2
        else:
            size += 1
            if size > max_explanatory_vars:
                break

    for child in range(d):
        spurious: set[int] = set()
        for parent in parents[child]:
            residual_child = _residual(data, child, sorted(parents[child] - {parent}))
            residual_parent = _residual(data, parent, sorted(parents[parent]))
            if hsic_test(_column(residual_child), _column(residual_parent)) > alpha:
                spurious.add(parent)
        parents[child] -= spurious
    return parents


def _find_confounders(
    data: np.ndarray,
    alpha: float,
    parents: list[set[int]],
    neighbors: list[set[int]],
) -> list[list[int]]:
    """Return observed index pairs whose parent-adjusted residuals stay dependent (a latent cause).

    A pair qualifies when neither variable is the other's parent and the two are marginal neighbors;
    if their residuals -- each variable regressed on its own committed parents -- remain dependent
    (HSIC p < alpha), the pair shares an unobserved confounder.
    """
    d = data.shape[1]
    confounded: list[list[int]] = []
    for i in range(d):
        for j in range(i + 1, d):
            if i in parents[j] or j in parents[i]:
                continue
            if i not in neighbors[j] or j not in neighbors[i]:
                continue
            residual_i = _residual(data, i, sorted(parents[i]))
            residual_j = _residual(data, j, sorted(parents[j]))
            if hsic_test(_column(residual_i), _column(residual_j)) < alpha:
                confounded.append([i, j])
    return confounded


def camuv(
    data: np.ndarray,
    *,
    alpha: float = 0.01,
    max_explanatory_vars: int = 3,
) -> tuple[list[list[int]], list[list[int]]]:
    """Return ``(parents, confounded_pairs)`` for CAM-UV on the observed matrix ``data``.

    ``data`` is an ``(n_samples, n_vars)`` float matrix. ``parents[i]`` is the sorted list of direct
    parents of variable ``i`` (edge ``j -> i`` for ``j`` in ``parents[i]``); ``confounded_pairs`` is
    the sorted ``[i, j]`` index lists that share an unobserved confounder. ``alpha`` is the HSIC
    independence level and ``max_explanatory_vars`` caps the parent-search subset size (the paper's
    ``d``). Structural output; no edge weights are estimated.
    """
    matrix = np.asarray(data, dtype=np.float64)
    neighbors = _neighborhoods(matrix, alpha)
    parents = _find_parents(matrix, alpha, max_explanatory_vars, neighbors)
    confounded = _find_confounders(matrix, alpha, parents, neighbors)
    return [sorted(parent_set) for parent_set in parents], confounded
