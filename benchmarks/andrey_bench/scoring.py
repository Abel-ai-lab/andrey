"""Scoring utilities for benchmark estimates.

``score_result`` scores through ``andrey.metrics``, against the truth's PAG for a PAG estimate, and
lag by lag for a temporal estimate (:func:`temporal_score_columns`).

``confounded_pair_scores`` scores the pairs an estimate attributes to a hidden common cause.

``sortability`` computes optional variance- and R2-sortability diagnostics
for generated datasets.

"""

from __future__ import annotations

import hashlib
from itertools import combinations
from typing import Any

import numpy as np

import andrey.metrics as M
from andrey import GraphStructure, TemporalStructure
from andrey.core import LATENT
from andrey.data.latent import marginal
from andrey_bench.contracts import links_from_temporal

# Andrey endpoint-mark convention (mirrors contracts.py): mark AT node i on edge i-j is marks[i, j].
_TAIL, _ARROW, _CIRCLE = 1, 2, 3


def structure_hash(structure: GraphStructure | TemporalStructure) -> str:
    """A hash of the normalized graph, comparable across packages.

    A temporal structure hashes its links as sorted ``(lag, cause, effect)`` triples.
    """
    if isinstance(structure, TemporalStructure):
        links = sorted(
            (int(lag) + 1, int(i), int(j))
            for lag, i, j in zip(*np.nonzero(links_from_temporal(structure)), strict=True)
        )
        return hashlib.sha1(f"temporal:{structure.n_nodes}:{links}".encode()).hexdigest()
    edges = sorted(tuple(int(v) for v in e) for e in structure.to_edges())
    return hashlib.sha1(f"{structure.kind}:{structure.n_nodes}:{edges}".encode()).hexdigest()


def score_result(
    estimate: GraphStructure | TemporalStructure, truth: GraphStructure | TemporalStructure
) -> dict[str, Any]:
    """Score an estimated graph against the ground truth.

    A PAG estimate is scored against the truth's PAG - the class FCI and RFCI recover, whose circles
    mark what no observational method can orient. Scoring against the DAG charges each circle as an
    error, even for the true PAG. Every other estimate goes to ``andrey.metrics.score`` as is, which
    scores a CPDAG through the truth's essential graph.

    A truth with hidden nodes is scored as its observed PAG for every estimate kind. The truth DAG
    includes nodes absent from the data. ``andrey.metrics.score`` would treat that DAG as a latent
    clustering.

    A temporal truth or estimate is scored by :func:`temporal_score_columns`, which raises
    ``TypeError`` unless both are temporal.
    """
    if isinstance(truth, TemporalStructure) or isinstance(estimate, TemporalStructure):
        return temporal_score_columns(estimate, truth)
    if truth.kind == "dag" and (estimate.kind == "pag" or _has_hidden(truth)):
        truth = _pag_of(truth)
    return dict(M.score(estimate, truth))


def temporal_score_columns(estimate: object, truth: object) -> dict[str, Any]:
    """Score a lagged estimate lag by lag through ``andrey.metrics.temporal_scores``.

    Returns flat columns: ``family`` (``"temporal"``); ``lag<k>_shd``, ``lag<k>_precision``,
    ``lag<k>_recall``, and ``lag<k>_f1`` for every lag either structure holds; ``temporal_shd``,
    their SHDs summed, the lag-resolved SHD; ``lagged_*``, the lags >= 1 merged into one graph; and
    ``summary_*``, every scored lag merged with self-loops left out. Each lag is a directed graph,
    so a link in the wrong direction counts 2 in its SHD. A lag with no true link has recall 1 and a
    lag with no estimated link precision 1 (``andrey.metrics``' convention), so read them beside
    the SHD. The CPDAG and PAG columns stay empty: their units are not these.
    """
    scores = M.temporal_scores(estimate, truth)
    columns: dict[str, Any] = {"family": "temporal"}
    per_lag = scores["per_lag"]
    for lag, stats in sorted(per_lag.items()):
        columns.update({f"lag{lag}_{stat}": float(value) for stat, value in stats.items()})
    columns["temporal_shd"] = float(sum(stats["shd"] for stats in per_lag.values()))
    for block in ("lagged", "summary"):
        if block in scores:
            columns.update({f"{block}_{stat}": float(v) for stat, v in scores[block].items()})
    return columns


def _has_hidden(truth: GraphStructure) -> bool:
    types = truth.node_types
    return types is not None and bool((types == LATENT).any())


#: Cache the last truth and its PAG by identity; `GraphStructure` is unhashable. Retaining the truth
#: prevents ID reuse. A dataset's solutions and repeats share one truth object. Projection time
#: grows about 8x per doubling of `d`: 15 s at `d = 400`, 16 min at 1600. It runs uncapped in the
#: parent, overriding `marginal`'s size guard to score every completed fit regardless of cost.
_PAG_OF: dict[int, tuple[GraphStructure, GraphStructure]] = {}


def _pag_of(truth: GraphStructure) -> GraphStructure:
    hit = _PAG_OF.get(id(truth))
    if hit is None or hit[0] is not truth:
        _PAG_OF.clear()
        hit = _PAG_OF[id(truth)] = (truth, marginal(truth, "pag", max_nodes=truth.n_nodes))
    return hit[1]


def _directed_adjacency(truth: GraphStructure) -> np.ndarray:
    """Return binary adjacency where ``A[i, j] = 1`` represents ``i -> j``."""
    marks = np.asarray(truth.to_numpy())
    return ((marks == _TAIL) & (marks.T == _ARROW)).astype(np.int8)


#: The mark at both ends of an edge that an estimate uses for a pair with a hidden common cause:
#: ``<->`` for a latent confounder, ``o-o`` for a pair whose order the confounder leaves open.
CONFOUNDED_MARKS = {"bidirected": _ARROW, "circle": _CIRCLE}

#: What counts as a true confounded pair. ``pag``: the observed pairs joined by ``<->`` in the
#: observed PAG. A PAG keeps ``<->`` only where conditional independences fix both arrowheads, so a
#: sparse graph has few. ``hidden_cause``: the observed pairs reached from one hidden node by
#: directed paths that pass through hidden nodes only, the target of RCD and BottomUpParceLiNGAM.
#: ``unobserved_path``: CAM-UV's target, the pairs joined by an unobserved causal or backdoor path
#: (:func:`unobserved_path_pairs`). Each method marks its pairs whether or not one node is the
#: other's parent.
CONFOUNDED_TARGETS = ("pag", "hidden_cause", "unobserved_path")

#: The columns :func:`confounded_pair_scores` returns, in order: the number of estimated pairs, then
#: per target the number of true pairs, precision, recall, and F1.
CONFOUNDED_COLUMNS = (
    "confounded_n_est",
    *(
        f"confounded_{target}_{stat}"
        for target in CONFOUNDED_TARGETS
        for stat in ("n_true", "precision", "recall", "f1")
    ),
)


def _marked_pairs(marks: np.ndarray, mark: int) -> set[tuple[int, int]]:
    """Return the pairs ``i < j`` with ``mark`` at both ends of the edge between them."""
    both = np.triu((marks == mark) & (marks.T == mark), k=1)
    return {(int(i), int(j)) for i, j in zip(*np.nonzero(both), strict=True)}


def _hidden_cause_pairs(truth: GraphStructure) -> set[tuple[int, int]]:
    """Return the observed pairs ``i < j`` that one hidden node reaches through hidden nodes only."""
    directed = _directed_adjacency(truth).astype(bool)
    hidden = np.asarray(truth.node_types) == LATENT
    pairs: set[tuple[int, int]] = set()
    for source in np.flatnonzero(hidden):
        seen, stack, reached = {int(source)}, [int(source)], set()
        while stack:
            for child in map(int, np.flatnonzero(directed[stack.pop()])):
                if not hidden[child]:
                    reached.add(child)
                elif child not in seen:
                    seen.add(child)
                    stack.append(child)
        pairs.update(combinations(sorted(reached), 2))
    return pairs


def projected_directed(truth: GraphStructure) -> set[tuple[int, int]]:
    """Return the directed edges ``(i, j)`` of the latent projection of a DAG `truth`.

    Observed ``i -> j`` holds when a directed path runs from ``i`` to ``j`` with every node between
    them hidden, a direct edge included.
    """
    directed = _directed_adjacency(truth).astype(bool)
    hidden = np.asarray(truth.node_types) == LATENT
    edges: set[tuple[int, int]] = set()
    for source in np.flatnonzero(~hidden):
        seen, stack = {int(source)}, [int(source)]
        while stack:
            for child in map(int, np.flatnonzero(directed[stack.pop()])):
                if not hidden[child]:
                    edges.add((int(source), child))
                elif child not in seen:
                    seen.add(child)
                    stack.append(child)
    return edges


def unobserved_path_pairs(truth: GraphStructure) -> set[tuple[int, int]]:
    """Return the observed pairs ``i < j`` of a DAG `truth` joined by an unobserved path.

    CAM-UV's definitions (Maeda and Shimizu, 2021): an unobserved causal path from ``i`` to ``j`` is
    a directed path that enters ``j`` from a hidden parent; an unobserved backdoor path enters both
    ``i`` and ``j`` from hidden parents that share an ancestor (either may be that ancestor). Nodes
    inside either path may be observed.
    """
    directed = _directed_adjacency(truth).astype(bool)
    hidden = np.asarray(truth.node_types) == LATENT
    reach = directed | np.eye(len(hidden), dtype=bool)  # reach[a, b]: a is b or an ancestor of b
    for k in range(len(hidden)):
        reach |= reach[:, [k]] & reach[[k], :]
    hidden_parents = [np.flatnonzero(directed[:, v] & hidden) for v in range(len(hidden))]
    observed = np.flatnonzero(~hidden)
    pairs: set[tuple[int, int]] = set()
    for i, j in combinations(map(int, observed), 2):
        causal = any(reach[i, h] for h in hidden_parents[j]) or any(
            reach[j, h] for h in hidden_parents[i]
        )
        backdoor = any(
            (reach[:, a] & reach[:, b]).any() for a in hidden_parents[i] for b in hidden_parents[j]
        )
        if causal or backdoor:
            pairs.add((i, j))
    return pairs


def direct_edges(truth: GraphStructure) -> set[tuple[int, int]]:
    """Return the edges ``(i, j)`` of a DAG `truth` that CAM-UV can identify as direct.

    These join two observed nodes with no unobserved path between them (:func:`unobserved_path_pairs`):
    CAM-UV marks a pair with such a path as unresolved even when one node is the other's parent.
    """
    hidden = np.asarray(truth.node_types) == LATENT
    directed = _directed_adjacency(truth).astype(bool) & ~hidden[:, None] & ~hidden[None, :]
    unresolved = unobserved_path_pairs(truth)
    return {
        (int(i), int(j))
        for i, j in zip(*np.nonzero(directed), strict=True)
        if (min(i, j), max(i, j)) not in unresolved
    }


def true_confounded_pairs(truth: GraphStructure, target: str) -> set[tuple[int, int]]:
    """Return the true confounded pairs ``i < j`` of a DAG `truth` with hidden nodes.

    `target` is one of :data:`CONFOUNDED_TARGETS`. Observed nodes come first, so a pair indexes the
    data columns.
    """
    if target == "pag":
        return _marked_pairs(np.asarray(_pag_of(truth).to_numpy()), _ARROW)
    if target == "hidden_cause":
        return _hidden_cause_pairs(truth)
    if target == "unobserved_path":
        return unobserved_path_pairs(truth)
    raise ValueError(f"unknown target {target!r}; known: {list(CONFOUNDED_TARGETS)}")


def confounded_pair_scores(
    estimate: GraphStructure, truth: GraphStructure, mark: str
) -> dict[str, float]:
    """Score the pairs `estimate` joins with the `mark` edge against each true confounded pair set.

    `mark` is a key of :data:`CONFOUNDED_MARKS`. Returns the number of estimated pairs and, per
    target in :data:`CONFOUNDED_TARGETS`, the number of true pairs with precision, recall, and F1.
    A ratio with no denominator is ``NaN``, not 1: precision when nothing is claimed and recall when
    nothing is true. F1 is ``NaN`` only when both sets are empty, and ``0`` when one set is empty.
    """
    claimed = _marked_pairs(np.asarray(estimate.to_numpy()), CONFOUNDED_MARKS[mark])
    scores: dict[str, float] = {"confounded_n_est": len(claimed)}
    for target in CONFOUNDED_TARGETS:
        true = true_confounded_pairs(truth, target)
        hits = len(claimed & true)
        scores[f"confounded_{target}_n_true"] = len(true)
        scores[f"confounded_{target}_precision"] = hits / len(claimed) if claimed else np.nan
        scores[f"confounded_{target}_recall"] = hits / len(true) if true else np.nan
        both = len(claimed) + len(true)
        scores[f"confounded_{target}_f1"] = 2 * hits / both if both else np.nan
    return scores


#: What counts as a true directed edge. ``projection``: the latent projection's edges
#: (:func:`projected_directed`) whose endpoints share no hidden cause, what RCD and
#: BottomUpParceLiNGAM estimate; a pair with a hidden cause is marked, not oriented. ``direct``: the edges
#: between two observed nodes with no unobserved path between them (:func:`direct_edges`), what
#: CAM-UV estimates; it marks a pair with such a path as unresolved instead.
DIRECTED_TARGETS = ("projection", "direct")

#: The columns :func:`directed_edge_scores` returns, in order.
DIRECTED_COLUMNS = (
    "directed_n_est",
    *(
        f"directed_{target}_{stat}"
        for target in DIRECTED_TARGETS
        for stat in ("n_true", "precision", "recall", "f1", "errors")
    ),
)


def directed_edge_scores(estimate: GraphStructure, truth: GraphStructure) -> dict[str, float]:
    """Score the estimate's directed edges against each target in :data:`DIRECTED_TARGETS`.

    A hidden-confounder method claims ``i -> j`` only for a direct cause, so its directed edges are
    scored apart from its confounded pairs. ``errors`` counts the edges in one set and not the
    other; a reversed edge counts twice. Ratios with no denominator are ``NaN``, as in
    :func:`confounded_pair_scores`.
    """
    adjacency = _directed_adjacency(estimate)
    claimed = {(int(i), int(j)) for i, j in zip(*np.nonzero(adjacency), strict=True)}
    scores: dict[str, float] = {"directed_n_est": len(claimed)}
    confounded = _hidden_cause_pairs(truth)
    projection = {e for e in projected_directed(truth) if tuple(sorted(e)) not in confounded}
    for target, true in (("projection", projection), ("direct", direct_edges(truth))):
        hits = len(claimed & true)
        both = len(claimed) + len(true)
        scores[f"directed_{target}_n_true"] = len(true)
        scores[f"directed_{target}_precision"] = hits / len(claimed) if claimed else np.nan
        scores[f"directed_{target}_recall"] = hits / len(true) if true else np.nan
        scores[f"directed_{target}_f1"] = 2 * hits / both if both else np.nan
        scores[f"directed_{target}_errors"] = len(claimed ^ true)
    return scores


def sortability(data: np.ndarray, truth: GraphStructure | TemporalStructure) -> dict[str, float]:
    """Compute variance- and R²-sortability under the ground-truth DAG.

    Returns an empty dictionary when the optional sortability implementation
    is unavailable, or for a temporal truth: sortability is defined for i.i.d. rows, not a series.

    """
    if isinstance(truth, TemporalStructure):
        return {}
    try:
        import andrey.data as _data  # noqa: WPS433 (deliberately lazy — optional dependency probe)

        sortmod = getattr(_data, "sortability", None)
        varsort = getattr(sortmod, "varsortability", None)
        r2sort = getattr(sortmod, "r2sortability", None)
        if varsort is None or r2sort is None:
            return {}
    except Exception:
        return {}

    x = np.asarray(data)
    # Hidden nodes come last and have no column, so only edges between observed nodes count.
    n_obs = x.shape[1]
    adj = _directed_adjacency(truth)[:n_obs, :n_obs]
    return {
        "varsortability": float(varsort(x, adj)),
        "r2_sortability": float(r2sort(x, adj)),
    }
