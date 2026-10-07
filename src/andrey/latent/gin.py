"""GIN, the engine behind :func:`andrey.gin`.

The algorithm, its settings, and its references are described on :func:`andrey.gin`.

For a candidate group ``X`` of observed variables and the rest ``Z``, ``omega`` is the right
singular vector of the smallest singular value of ``cov(Z, X)``, and ``e = X @ omega`` stands in
for the group's noise. ``e`` is independent of every variable in ``Z`` when ``X`` is a pure
cluster, one whose members share a single latent parent; the per-variable gamma-HSIC p-values are
pooled by Fisher's method. Clusters grow by size and merge when they overlap. The latents are then
ordered by repeatedly taking as root the cluster whose ``e``, built against the other clusters and
given the ones already ordered, is least dependent.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.special import gammainc

from andrey.core.independence import hsic_gamma_test
from andrey.core.output import StructureOutput
from andrey.core.structure import ARROW, LATENT, OBSERVED, TAIL, GraphStructure


@dataclass(frozen=True, slots=True)
class GINResult:
    """Recovered latent structure of a linear non-Gaussian latent-variable model.

    ``clusters`` groups the observed-variable indices by shared latent parent (an unordered
    partition). ``causal_order`` lists those same clusters from the earliest latent to the latest;
    any clusters whose order could not be resolved trail after the ordered ones.
    """

    clusters: list[list[int]]
    causal_order: list[list[int]]


def _fisher_combine(pvals: list[float]) -> float:
    """Pool p-values by Fisher's method, returning the combined right-tail p-value."""
    clipped = np.clip(np.asarray(pvals, dtype=np.float64), 1e-5, None)
    statistic = -2.0 * float(np.sum(np.log(clipped)))
    return float(1.0 - gammainc(len(clipped), 0.5 * statistic))


def _surrogate(data: np.ndarray, cov: np.ndarray, x: list[int], z: list[int]) -> np.ndarray:
    """Return the GIN residual surrogate ``data[:, x] @ omega`` for group ``x`` against set ``z``.

    ``omega`` is the right singular vector of the smallest singular value of the ``cov[z, x]``
    block.
    """
    block = cov[np.ix_(z, x)]
    _, _, vh = np.linalg.svd(block)
    omega = vh[-1]
    return data[:, x] @ omega


def _independence_pvalue(data: np.ndarray, cov: np.ndarray, x: list[int], z: list[int]) -> float:
    """Fisher-pooled p-value that the surrogate of ``x`` is independent of every member of ``z``."""
    e = _surrogate(data, cov, x, z)
    pvals = [hsic_gamma_test(data[:, j], e)[1] for j in z]
    return _fisher_combine(pvals)


def _residual_dependence(data: np.ndarray, cov: np.ndarray, x: list[int], z: list[int]) -> float:
    """Mean gamma-HSIC statistic between the surrogate of ``x`` and each variable in ``z``.

    A larger value means more leftover dependence; the root cluster minimizes it.
    """
    e = _surrogate(data, cov, x, z)
    stats = [hsic_gamma_test(data[:, j], e)[0] for j in z]
    return float(np.mean(stats))


def _merge_overlapping(groups: list[tuple[int, ...]]) -> list[list[int]]:
    """Merge groups that share any variable into maximal connected clusters (sorted ascending)."""
    remaining = [set(g) for g in groups]
    merged: list[set[int]] = []
    while remaining:
        current = remaining.pop()
        changed = True
        while changed:
            changed = False
            rest = []
            for other in remaining:
                if current & other:
                    current |= other
                    changed = True
                else:
                    rest.append(other)
            remaining = rest
        merged.append(current)
    return [sorted(cluster) for cluster in merged]


def _find_clusters(data: np.ndarray, cov: np.ndarray, alpha: float) -> list[list[int]]:
    """Discover causal clusters by growing the group size and keeping GIN-passing groups."""
    n = data.shape[1]
    available = set(range(n))
    clusters: list[list[int]] = []
    size = 2
    while size < len(available):
        passing: list[tuple[int, ...]] = []
        for group in combinations(sorted(available), size):
            remaining = sorted(available - set(group))
            if not remaining:
                continue
            pval = _independence_pvalue(data, cov, list(group), remaining)
            if pval >= alpha:
                passing.append(group)
        if passing:
            new_clusters = _merge_overlapping(passing)
            clusters.extend(new_clusters)
            for cluster in new_clusters:
                available -= set(cluster)
        size += 1
    return clusters


def _find_root(
    data: np.ndarray, cov: np.ndarray, clusters: list[list[int]], ordered: list[list[int]]
) -> list[int]:
    """Return the cluster with the least residual dependence given the already-ordered prefix."""
    if len(clusters) == 1:
        return clusters[0]
    root = clusters[0]
    best = np.inf
    for candidate in clusters:
        for other in clusters:
            if candidate is other:
                continue
            x = [candidate[0], other[0]]
            z = list(candidate[1:])
            for prefix in ordered:
                x.append(prefix[0])
                z.append(prefix[1])
            dependence = _residual_dependence(data, cov, x, z)
            if dependence < best:
                best = dependence
                root = candidate
    return root


def _order_clusters(
    data: np.ndarray, cov: np.ndarray, clusters: list[list[int]]
) -> list[list[int]]:
    """Order clusters from earliest to latest latent by iteratively peeling the root."""
    pending = [list(cluster) for cluster in clusters]
    ordered: list[list[int]] = []
    while pending:
        root = _find_root(data, cov, pending, ordered)
        ordered.append(root)
        pending.remove(root)
    return ordered


def gin(data: np.ndarray, *, alpha: float = 0.05) -> GINResult:
    """Fit GIN to observed data ``(n_samples, n_variables)``; return clusters and their order.

    ``alpha`` is the significance level of the pooled independence test used to discover clusters.
    Returns a :class:`GINResult` whose ``clusters`` partition the observed variables by latent
    parent and whose ``causal_order`` lists those clusters from the earliest latent to the latest.
    """
    data = np.asarray(data, dtype=np.float64)
    cov = np.cov(data, rowvar=False)
    clusters = _find_clusters(data, cov, alpha)
    ordered = _order_clusters(data, cov, clusters)
    return GINResult(clusters=clusters, causal_order=ordered)


def _structure_from_result(
    n_variables: int, result: GINResult, labels: tuple[str, ...] | None
) -> StructureOutput:
    """Assemble the latent-plus-observed graph from a fitted :class:`GINResult`.

    Observed variables keep their column index ``0..n_variables-1``; one latent node is appended per
    ordered cluster. Each latent points to every observed variable in its cluster, and the latents
    form a complete DAG in causal order (an earlier latent points to every later one), so the graph
    is a faithful directed model over both node kinds. Observed variables that fell into no cluster
    remain isolated observed nodes.
    """
    ordered = result.causal_order
    n_latent = len(ordered)
    n_nodes = n_variables + n_latent
    marks = np.zeros((n_nodes, n_nodes), dtype=np.int8)

    for position, cluster in enumerate(ordered):
        latent = n_variables + position
        for child in cluster:
            marks[latent, child] = TAIL
            marks[child, latent] = ARROW
        for later in range(position + 1, n_latent):
            marks[latent, n_variables + later] = TAIL
            marks[n_variables + later, latent] = ARROW

    node_types = np.full(n_nodes, OBSERVED, dtype=np.int8)
    node_types[n_variables:] = LATENT

    observed_labels = (
        tuple(labels) if labels is not None else tuple(f"X{i + 1}" for i in range(n_variables))
    )
    all_labels = observed_labels + tuple(f"L{k + 1}" for k in range(n_latent))

    structure = GraphStructure.from_numpy(
        marks, kind="dag", labels=all_labels, node_types=node_types
    )
    metadata = {
        "algorithm": "GIN",
        "n_latents": n_latent,
        "clusters": [list(c) for c in result.clusters],
        "causal_order": [list(c) for c in ordered],
    }
    return StructureOutput.new(structure, metadata=metadata)


def gin_structure(
    data: np.ndarray, *, alpha: float = 0.05, labels: tuple[str, ...] | None = None
) -> StructureOutput:
    """Fit GIN and return the recovered latent-variable model as a :class:`StructureOutput`.

    A convenience over :func:`gin` that materializes the clustering and latent order as a directed
    graph over observed *and* latent nodes: observed variables occupy indices ``0..n_variables-1``
    (optionally named by ``labels``), latents follow in causal order named ``L1..Lm``, and
    ``node_types`` flags which nodes are latent. Each latent points to its observed children and to
    every later latent; the clustering and order also ride along in ``metadata``.
    """
    data = np.asarray(data, dtype=np.float64)
    return _structure_from_result(data.shape[1], gin(data, alpha=alpha), labels)
