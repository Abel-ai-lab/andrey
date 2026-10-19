"""GIN, the engine behind :func:`andrey.gin`.

The algorithm, its settings, and its references are described on :func:`andrey.gin`.

For a candidate group ``X`` of observed variables and the rest ``Z``, ``omega`` is the right
singular vector of the smallest singular value of ``cov(Z, X)``, and ``e = X @ omega`` stands in
for the group's noise. ``e`` is independent of every variable in ``Z`` when ``X`` is a pure
cluster, one whose members share a single latent parent; the per-variable gamma-HSIC p-values are
pooled by Fisher's method. Clusters grow by size and merge when they overlap. The latents are then
ordered one root at a time (Xie et al. 2020, Proposition 4 and Algorithm 2): a cluster is a root
when its ``e``, built with each other remaining cluster in turn and given the clusters already
ordered, passes the pooled test at ``alpha`` for every one of them. Each of those tests runs once
per member, so that every member of every cluster involved enters ``e``, and pools the runs by
Fisher's method. When several candidates pass, the one whose lowest pooled p-value is highest is
taken, and of lowest p-values equal to ``_KEY_DP`` decimals, the one holding the lowest-ranked
column. When none passes, the ordering stops and the remaining clusters stay unordered: the paper
adds a latent to the order only when the condition holds.

The search runs on the columns sorted by :func:`~andrey.search.hc.tie_ranks`, a rank computed from
each column's own data, so every column order of the same data gives the same clusters and order.

Each column's HSIC kernel matrix is built once per fit and reused by every test, so a fit holds one
``n x n`` matrix per column for ``n`` rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.special import gammaincc

from andrey.core.independence import gamma_gram, hsic_gamma_from_grams
from andrey.core.output import StructureOutput
from andrey.core.structure import ARROW, LATENT, OBSERVED, TAIL, GraphStructure
from andrey.search._parallel import _KEY_DP
from andrey.search.hc import tie_ranks

_Gram = tuple[np.ndarray, float]  # a column's centered HSIC kernel matrix and its mean


@dataclass(frozen=True, slots=True)
class GINResult:
    """Recovered latent structure of a linear non-Gaussian latent-variable model.

    ``clusters`` groups the observed-variable indices by shared latent parent, in disjoint groups
    that can leave variables out, listed by smallest member. ``causal_order`` lists the clusters
    whose latents were ordered, from the earliest latent to the latest, and ``unordered`` the rest,
    by smallest member: every latent in ``causal_order`` comes before every unordered one.
    """

    clusters: list[list[int]]
    causal_order: list[list[int]]
    unordered: list[list[int]]


def _fisher_combine(pvals: list[float]) -> float:
    """Pool p-values by Fisher's method, returning the combined right-tail p-value.

    The statistic ``-2 sum(log p)`` over ``c`` p-values is chi-square with ``2c`` degrees of
    freedom under independence. A p-value below the smallest positive normal float64 is raised to
    it, so its logarithm stays finite.
    """
    clipped = np.clip(np.asarray(pvals, dtype=np.float64), np.finfo(np.float64).tiny, None)
    statistic = -2.0 * float(np.sum(np.log(clipped)))
    return float(gammaincc(len(clipped), 0.5 * statistic))


def _surrogate(data: np.ndarray, cov: np.ndarray, x: list[int], z: list[int]) -> np.ndarray:
    """Return the GIN residual surrogate ``data[:, x] @ omega`` for group ``x`` against set ``z``.

    ``omega`` is the right singular vector of the smallest singular value of the ``cov[z, x]``
    block.
    """
    block = cov[np.ix_(z, x)]
    _, _, vh = np.linalg.svd(block)
    omega = vh[-1]
    return data[:, x] @ omega


def _independence_pvalue(
    data: np.ndarray, cov: np.ndarray, grams: list[_Gram], x: list[int], z: list[int]
) -> float:
    """Fisher-pooled p-value that the surrogate of ``x`` is independent of every member of ``z``.

    ``grams`` holds each column's :func:`~andrey.core.independence.gamma_gram`, so only the
    surrogate's is built here.
    """
    e_gram = gamma_gram(_surrogate(data, cov, x, z))
    pvals = [hsic_gamma_from_grams(grams[j], e_gram, data.shape[0])[1] for j in z]
    return _fisher_combine(pvals)


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


def _find_clusters(
    data: np.ndarray, cov: np.ndarray, grams: list[_Gram], alpha: float
) -> list[list[int]]:
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
            pval = _independence_pvalue(data, cov, grams, list(group), remaining)
            if pval >= alpha:
                passing.append(group)
        if passing:
            new_clusters = _merge_overlapping(passing)
            clusters.extend(new_clusters)
            for cluster in new_clusters:
                available -= set(cluster)
        size += 1
    return clusters


def _root_pvalue(
    data: np.ndarray,
    cov: np.ndarray,
    grams: list[_Gram],
    candidate: list[int],
    other: list[int],
    ordered: list[list[int]],
) -> float:
    """Fisher-pooled p-value that ``candidate`` is a root against ``other``, given ``ordered``.

    The test runs once per member, as many times as the largest of these clusters has members.
    Run ``k`` builds the surrogate from member ``k`` of the candidate, of ``other``, and of every
    ordered cluster, counting modulo each cluster's size, against the candidate's other members and
    member ``k + 1`` of every ordered cluster. Each run's pooled p-value enters Fisher's method.
    """
    runs = max(len(cluster) for cluster in [candidate, other, *ordered])
    pvals = []
    for k in range(runs):
        member = candidate[k % len(candidate)]
        x = [member, other[k % len(other)]] + [prefix[k % len(prefix)] for prefix in ordered]
        z = [m for m in candidate if m != member]
        z += [prefix[(k + 1) % len(prefix)] for prefix in ordered]
        pvals.append(_independence_pvalue(data, cov, grams, x, z))
    return _fisher_combine(pvals)


def _find_root(
    data: np.ndarray,
    cov: np.ndarray,
    grams: list[_Gram],
    clusters: list[list[int]],
    ordered: list[list[int]],
    alpha: float,
) -> list[int] | None:
    """Return the root among ``clusters`` given the ``ordered`` ones, or ``None`` if none passes.

    A candidate passes when :func:`_root_pvalue` is at least ``alpha`` against every other cluster;
    of the passing candidates, the one with the highest lowest p-value is the root. Lowest p-values
    equal to ``_KEY_DP`` decimals go to the cluster with the smallest member: :func:`gin` passes the
    columns sorted by :func:`~andrey.search.hc.tie_ranks`, so a column's index is its rank. A single
    remaining cluster is the root.
    """
    if len(clusters) == 1:
        return clusters[0]
    best, best_key = None, None
    for candidate in clusters:
        lowest = min(
            _root_pvalue(data, cov, grams, candidate, other, ordered)
            for other in clusters
            if other is not candidate
        )
        key = (-round(lowest, _KEY_DP), min(candidate))
        if lowest >= alpha and (best_key is None or key < best_key):
            best, best_key = candidate, key
    return best


def _order_clusters(
    data: np.ndarray, cov: np.ndarray, grams: list[_Gram], clusters: list[list[int]], alpha: float
) -> tuple[list[list[int]], list[list[int]]]:
    """Order clusters from the earliest latent by peeling roots; return ``(ordered, unordered)``.

    The peeling stops at the first step where no candidate passes, and the clusters left then are
    returned unordered, sorted by smallest member.
    """
    pending = [list(cluster) for cluster in clusters]
    ordered: list[list[int]] = []
    while pending:
        root = _find_root(data, cov, grams, pending, ordered, alpha)
        if root is None:
            break
        ordered.append(root)
        pending.remove(root)
    return ordered, sorted(pending, key=min)


def gin(data: np.ndarray, *, alpha: float = 0.05) -> GINResult:
    """Fit GIN to observed data ``(n_samples, n_variables)``; return clusters and their order.

    ``alpha`` is the significance level of the pooled independence test that finds the clusters and
    then the root at each ordering step. Returns a :class:`GINResult` whose ``clusters`` group
    observed variables by latent parent, disjointly, whose ``causal_order`` lists the ordered
    clusters from the earliest latent, and whose ``unordered`` lists the clusters left when no root
    passed.

    The search runs on the columns sorted by :func:`~andrey.search.hc.tie_ranks`, so it orders its
    groups and members, and breaks its ties, by each column's data rather than its position. The
    sorted data are the same bytes for every column order, so reordering the columns of ``data``
    relabels the result and changes nothing else, unless two columns are identical.
    """
    data = np.asarray(data, dtype=np.float64)
    by_rank = np.argsort(tie_ranks(data))
    ranked = np.ascontiguousarray(data[:, by_rank])
    cov = np.cov(ranked, rowvar=False)
    grams = [gamma_gram(ranked[:, j]) for j in range(ranked.shape[1])]
    clusters = _find_clusters(ranked, cov, grams, alpha)
    ordered, unordered = _order_clusters(ranked, cov, grams, clusters, alpha)

    def columns(groups: list[list[int]]) -> list[list[int]]:
        return [sorted(int(by_rank[i]) for i in group) for group in groups]

    return GINResult(
        clusters=sorted(columns(clusters), key=min),
        causal_order=columns(ordered),
        unordered=sorted(columns(unordered), key=min),
    )


def _structure_from_result(
    n_variables: int, result: GINResult, labels: tuple[str, ...] | None
) -> StructureOutput:
    """Assemble the latent-plus-observed graph from a fitted :class:`GINResult`.

    Observed variables keep their column index ``0..n_variables-1``; one latent node is appended per
    cluster, the ordered clusters first, in causal order, then the unordered ones. Each latent
    points to every observed variable in its cluster. An ordered latent points to every later latent
    and to every unordered one, and unordered latents have no edge between them. GIN finds the
    order only, so an edge between latents does not mean a direct effect was found. Observed
    variables that fell into no cluster remain isolated observed nodes.
    """
    ordered = result.causal_order
    latents = ordered + result.unordered
    n_latent = len(latents)
    n_nodes = n_variables + n_latent
    marks = np.zeros((n_nodes, n_nodes), dtype=np.int8)

    for position, cluster in enumerate(latents):
        latent = n_variables + position
        for child in cluster:
            marks[latent, child] = TAIL
            marks[child, latent] = ARROW
        if position >= len(ordered):
            continue
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
        "unordered": [list(c) for c in result.unordered],
    }
    return StructureOutput.new(structure, metadata=metadata)


def gin_structure(
    data: np.ndarray, *, alpha: float = 0.05, labels: tuple[str, ...] | None = None
) -> StructureOutput:
    """Fit GIN and return the recovered latent-variable model as a :class:`StructureOutput`.

    A convenience over :func:`gin` that materializes the clustering and latent order as a directed
    graph over observed *and* latent nodes: observed variables occupy indices ``0..n_variables-1``
    (optionally named by ``labels``), latents follow named ``L1..Lm``, the ordered ones first, and
    ``node_types`` flags which nodes are latent. Each latent points to its observed children, and an
    ordered latent to every later latent; the clustering and order also ride along in ``metadata``.
    """
    data = np.asarray(data, dtype=np.float64)
    return _structure_from_result(data.shape[1], gin(data, alpha=alpha), labels)
