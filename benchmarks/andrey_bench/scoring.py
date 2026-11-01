"""Scoring utilities for benchmark estimates.

``score_result`` scores through ``andrey.metrics``, against the truth's PAG for a PAG estimate, lag
by lag for a temporal estimate (:func:`temporal_score_columns`), link by link for a window graph
(:func:`window_scores`), and coefficient by coefficient for a coefficient stack
(:func:`coefficient_scores`).

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
from andrey_bench import window as W
from andrey_bench.coefficients import CoefficientStack
from andrey_bench.contracts import links_from_temporal

# Andrey endpoint-mark convention (mirrors contracts.py): mark AT node i on edge i-j is marks[i, j].
_TAIL, _ARROW, _CIRCLE = 1, 2, 3


def structure_hash(structure: GraphStructure | TemporalStructure | W.WindowGraph) -> str:
    """A hash of the normalized graph, comparable across packages.

    A temporal structure hashes its links as sorted ``(lag, cause, effect)`` triples, and a window
    graph its nonzero marks as sorted ``(lag, i, j, mark)`` tuples, conflicts included, and a
    coefficient stack the positions of its links (nonzero, finite coefficients), not their values.
    """
    if isinstance(structure, CoefficientStack):
        links = sorted(tuple(int(v) for v in idx) for idx in np.argwhere(structure.support()))
        shape = structure.stack.shape
        return hashlib.sha1(f"coefficients:{structure.kind}:{shape}:{links}".encode()).hexdigest()
    if isinstance(structure, W.WindowGraph):
        marks = sorted(
            (int(k), int(i), int(j), int(structure.marks[k, i, j]))
            for k, i, j in zip(*np.nonzero(structure.marks), strict=True)
        )
        return hashlib.sha1(
            f"window:{structure.lag0}:{structure.n_nodes}:{marks}".encode()
        ).hexdigest()
    if isinstance(structure, TemporalStructure):
        links = sorted(
            (int(lag) + 1, int(i), int(j))
            for lag, i, j in zip(*np.nonzero(links_from_temporal(structure)), strict=True)
        )
        return hashlib.sha1(f"temporal:{structure.n_nodes}:{links}".encode()).hexdigest()
    edges = sorted(tuple(int(v) for v in e) for e in structure.to_edges())
    return hashlib.sha1(f"{structure.kind}:{structure.n_nodes}:{edges}".encode()).hexdigest()


def score_result(
    estimate: GraphStructure | TemporalStructure | W.WindowGraph,
    truth: GraphStructure | TemporalStructure | W.WindowGraph,
) -> dict[str, Any]:
    """Score an estimated graph against the ground truth.

    A PAG estimate is scored against the truth's PAG - the class FCI and RFCI recover, whose circles
    mark what no observational method can orient. Scoring against the DAG charges each circle as an
    error, even for the true PAG. Every other estimate goes to ``andrey.metrics.score`` as is, which
    scores a CPDAG through the truth's essential graph.

    A truth with hidden nodes is scored as its observed PAG for every estimate kind. The truth DAG
    includes nodes absent from the data. ``andrey.metrics.score`` would treat that DAG as a latent
    clustering. An estimate with latent nodes of its own, such as GIN's, is that clustering, so it
    is scored against the truth's latents instead: ``andrey.metrics.score`` returns the clustering's
    adjusted Rand index and both latent counts (:func:`measurement_scores` adds the rest).

    A temporal truth or estimate is scored by :func:`temporal_score_columns`, which raises
    ``TypeError`` unless both are temporal, and a window graph by :func:`window_scores`, which
    raises ``TypeError`` unless both are window graphs.
    """
    if isinstance(truth, CoefficientStack) or isinstance(estimate, CoefficientStack):
        return coefficient_scores(estimate, truth)
    if isinstance(truth, W.WindowGraph) or isinstance(estimate, W.WindowGraph):
        return window_scores(estimate, truth)
    if isinstance(truth, TemporalStructure) or isinstance(estimate, TemporalStructure):
        return temporal_score_columns(estimate, truth)
    if _has_hidden(estimate):
        return dict(M.score(estimate, truth))
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


def _scored_marks(graph: W.WindowGraph) -> np.ndarray:
    """The window marks as scoring reads them, in the domain of the graph's lag-0 kind.

    A ``cpdag`` has no circles: an unoriented (``o-o``) or conflicting (``x-x``) same-step link is
    undirected, a tail at both ends, as Andrey's PCMCI+ returns it. A ``pag`` reads a conflict mark
    as a circle, as Andrey's LPCMCI returns it.
    """
    marks = graph.marks.copy()
    if graph.lag0 == "cpdag":
        lag0 = marks[0]
        unoriented = np.isin(lag0, (W.CIRCLE, W.CONFLICT)) | np.isin(lag0.T, (W.CIRCLE, W.CONFLICT))
        lag0[unoriented & (lag0 != W.NULL)] = W.TAIL
    marks[marks == W.CONFLICT] = W.CIRCLE
    return marks


def window_scores(estimate: object, truth: object) -> dict[str, Any]:
    """Score a window graph link by link against a window truth.

    A link is a same-step pair ``{i, j}`` or a lagged pair ``(X_i(t-k), X_j(t))``, ``k >= 1``,
    self-links included. Columns, all over every link of lags ``0..L``:

    - ``endpoint_shd``: the marks that differ from the truth's, two for a missing or an extra link
      and one for each wrong end of a link both hold. A circle differs from the true tail or
      arrowhead it leaves open, so an unoriented link costs its open ends. ``lag0_endpoint_shd`` and
      ``lagged_endpoint_shd`` split it.
    - ``adjacency_precision``, ``adjacency_recall``, ``adjacency_f1``: the links, whatever their
      marks; ``n_true_links`` and ``n_est_links`` count them.
    - ``wrong_marks``: tails and arrowheads, on links the truth holds, that contradict it; the ends
      of an unoriented, conflicting, or circle-marked link claim nothing and are not counted.
    - ``lagged_tail``, ``lagged_circle``, ``lagged_arrow``: the true lagged links found, by the mark
      at their earlier end (a tail is the true one).

    An estimate whose lag 0 is a DAG, scored against a truth without unobserved variables, also gets
    the edge columns of :func:`_edge_scores`, the units of the DYNOTEARS paper.

    Conflicts are read by :func:`_scored_marks`. A truth with unobserved variables is scored as its
    true PAG over the observed ones at the estimate's ``tau_max``
    (:func:`andrey_bench.window_pag.true_window_pag`), the class LPCMCI recovers: a circle in both
    is a match, and a tail or an arrowhead where the true PAG has a circle differs from it but is
    not a ``wrong_marks`` claim. Raises ``TypeError`` unless both are window graphs and
    ``ValueError`` unless the truth's lag 0 is a DAG and it covers the estimate's variables.
    """
    if not isinstance(estimate, W.WindowGraph) or not isinstance(truth, W.WindowGraph):
        raise TypeError("window_scores needs a window estimate and a window truth")
    if truth.lag0 != "dag":
        raise ValueError("window_scores needs a truth whose lag 0 is a DAG")
    directed = estimate.lag0 == "dag" and not truth.latents
    if truth.latents:
        truth = _window_pag_of(truth, estimate.tau_max)
    if estimate.n_nodes != truth.n_nodes:
        raise ValueError("window_scores needs a truth over the estimate's variables")
    lags = max(estimate.tau_max, truth.tau_max) + 1
    n = truth.n_nodes
    raw, est, true_ = (
        np.concatenate([m, np.zeros((lags - len(m), n, n), np.int8)])
        for m in (estimate.marks, _scored_marks(estimate), truth.marks)
    )
    off = ~np.eye(n, dtype=bool)
    upper = np.triu(off)
    est0, true0 = est[0] != W.NULL, true_[0] != W.NULL
    est_adj = np.concatenate([(est0 | est0.T)[upper], (est[1:] != W.NULL).ravel()])
    true_adj = np.concatenate([(true0 | true0.T)[upper], (true_[1:] != W.NULL).ravel()])
    tp = int((est_adj & true_adj).sum())
    fp = int((est_adj & ~true_adj).sum())
    fn = int((~est_adj & true_adj).sum())

    lag0_shd = int((est[0] != true_[0])[off].sum())
    lagged_est, lagged_true = est[1:] != W.NULL, true_[1:] != W.NULL
    both = lagged_est & lagged_true
    lagged_shd = 2 * int((lagged_est ^ lagged_true).sum()) + int(
        (both & (est[1:] != true_[1:])).sum()
    )

    # A claim is a tail or an arrowhead the solution returned; an unoriented link claims nothing,
    # and a claim where the true PAG has a circle contradicts nothing.
    definite = np.isin(raw, (W.TAIL, W.ARROW)) & np.isin(true_, (W.TAIL, W.ARROW))
    held0 = (true0 | true0.T) & off
    wrong0 = int((definite[0] & held0 & (est[0] != true_[0])).sum())
    wrong_lagged = int((definite[1:] & both & (est[1:] != true_[1:])).sum())

    found = est[1:][both]
    # A ratio with no denominator scores 1, as andrey.metrics scores it.
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    edges = _edge_scores(est, true_) if directed else {}
    return {
        **edges,
        "family": "temporal_window",
        "endpoint_shd": float(lag0_shd + lagged_shd),
        "lag0_endpoint_shd": float(lag0_shd),
        "lagged_endpoint_shd": float(lagged_shd),
        "adjacency_precision": precision,
        "adjacency_recall": recall,
        "adjacency_f1": f1,
        "n_true_links": float(tp + fn),
        "n_est_links": float(tp + fp),
        "wrong_marks": float(wrong0 + wrong_lagged),
        "lagged_tail": float((found == W.TAIL).sum()),
        "lagged_circle": float((found == W.CIRCLE).sum()),
        "lagged_arrow": float((found == W.ARROW).sum()),
    }


def _f1(tp: int, n_est: int, n_true: int) -> float:
    """F1 of ``tp`` matches among ``n_est`` estimated and ``n_true`` true edges.

    A ratio with no denominator scores 1, as :func:`window_scores` scores it.
    """
    precision = tp / n_est if n_est else 1.0
    recall = tp / n_true if n_true else 1.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _edge_scores(est: np.ndarray, true_: np.ndarray) -> dict[str, float]:
    """Edge SHD and directed-edge F1 of window marks whose lag 0 is a DAG, padded to equal lags.

    An edge is a directed same-step pair ``i -> j`` or a lagged link. ``lag0_shd`` counts the
    same-step pairs whose state differs (no edge, ``i -> j``, ``j -> i``): a missing, an extra, and
    a reversed edge each cost 1, the structural Hamming distance of Pamfil et al. (2020),
    Appendix B.7. ``lagged_shd`` counts the lagged links in one graph and not the other, and
    ``shd`` is their sum. ``lag0_f1``, ``lagged_f1``, and ``f1`` are the F1 of the directed edges, a
    reversed edge one false and one missed, the paper's F1 per block, and over both blocks pooled.
    """
    est0 = (est[0] == W.TAIL) & (est[0].T == W.ARROW)
    true0 = (true_[0] == W.TAIL) & (true_[0].T == W.ARROW)
    lag0_shd = int(np.triu((est0 != true0) | (est0.T != true0.T), 1).sum())
    est_l, true_l = est[1:] != W.NULL, true_[1:] != W.NULL
    tp0, tpl = int((est0 & true0).sum()), int((est_l & true_l).sum())
    n0, nl = (int(est0.sum()), int(true0.sum())), (int(est_l.sum()), int(true_l.sum()))
    lagged_shd = int((est_l ^ true_l).sum())
    return {
        "shd": float(lag0_shd + lagged_shd),
        "lag0_shd": float(lag0_shd),
        "lagged_shd": float(lagged_shd),
        "f1": _f1(tp0 + tpl, n0[0] + nl[0], n0[1] + nl[1]),
        "lag0_f1": _f1(tp0, *n0),
        "lagged_f1": _f1(tpl, *nl),
    }


#: The last window truth, its ``tau_max``, and its true PAG; one truth serves all solutions of a dataset.
_WINDOW_PAG_OF: dict[int, tuple[W.WindowGraph, int, W.WindowGraph]] = {}


def _window_pag_of(truth: W.WindowGraph, tau_max: int) -> W.WindowGraph:
    from andrey_bench.window_pag import true_window_pag

    hit = _WINDOW_PAG_OF.get(id(truth))
    if hit is None or hit[0] is not truth or hit[1] != tau_max:
        _WINDOW_PAG_OF.clear()
        hit = _WINDOW_PAG_OF[id(truth)] = (truth, tau_max, true_window_pag(truth, tau_max))
    return hit[2]


def coefficient_scores(estimate: object, truth: object) -> dict[str, Any]:
    """Score a coefficient stack against the true one.

    The blocks scored are the instantaneous and lag-1 autoregressive blocks: ``B0`` and ``B1`` of a
    ``varma`` stack, and ``B(t,t)`` and ``B(t,t-1)`` at every time point but the first of a
    ``panel`` stack. A link is a nonzero coefficient; its block gives its lag, its row the effect,
    and its column the cause. Columns:

    - ``precision``, ``recall``, ``f1``: the directed cross-links (cause and effect differ) of
      those blocks, a link counted when its lag, cause, and effect all match;
      ``instantaneous_*`` and ``lagged_*`` split them by block.
    - ``instantaneous_rmse`` and ``lagged_rmse``: the root mean square error of every entry of
      those blocks, as the LiNGAM papers report their estimates; ``ma_rmse`` that of the
      moving-average block ``M1`` of a ``varma`` stack.

    Raises ``TypeError`` unless both are coefficient stacks, and ``ValueError`` unless they have
    the same kind and shape.
    """
    if not isinstance(estimate, CoefficientStack) or not isinstance(truth, CoefficientStack):
        raise TypeError("coefficient_scores needs a coefficient estimate and a coefficient truth")
    if estimate.kind != truth.kind or estimate.stack.shape != truth.stack.shape:
        raise ValueError("coefficient_scores needs stacks of the same kind and shape")
    if truth.kind == "varma":
        est = estimate.stack[None, :2]
        true_ = truth.stack[None, :2]
    else:
        est = estimate.stack[1:]
        true_ = truth.stack[1:]
    cross = ~np.eye(truth.n_nodes, dtype=bool)
    est_links = np.isfinite(est) & (est != 0) & cross
    true_links = (true_ != 0) & cross

    def prf(prefix: str, e: np.ndarray, t: np.ndarray) -> dict[str, float]:
        tp, fp, fn = int((e & t).sum()), int((e & ~t).sum()), int((~e & t).sum())
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {f"{prefix}precision": precision, f"{prefix}recall": recall, f"{prefix}f1": f1}

    columns: dict[str, Any] = {"family": "temporal_coefficients"}
    columns.update(prf("", est_links, true_links))
    columns.update(prf("instantaneous_", est_links[:, 0], true_links[:, 0]))
    columns.update(prf("lagged_", est_links[:, 1], true_links[:, 1]))
    for block, name in ((0, "instantaneous_rmse"), (1, "lagged_rmse")):
        columns[name] = float(np.sqrt(np.mean((est[:, block] - true_[:, block]) ** 2)))
    if truth.kind == "varma":
        error = estimate.stack[2] - truth.stack[2]
        columns["ma_rmse"] = float(np.sqrt(np.mean(error**2)))
    return columns


def _window_links(structure: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The links of a time-series graph as ``(lagged, adjacent, directed)`` boolean stacks.

    ``lagged[s, k - 1, i, j]`` is the link from variable ``i`` at ``t - k`` to ``j`` at ``t``;
    ``adjacent[s, i, j]``, symmetric, is a same-step link between ``i`` and ``j`` whatever its
    marks, and ``directed[s, i, j]`` a same-step link oriented ``i -> j``. ``s`` indexes the time
    points a panel is scored at (from its second), and is a single slot otherwise.
    """
    if isinstance(structure, TemporalStructure):
        lagged = links_from_temporal(structure).astype(bool)[None]
        n = structure.n_nodes
        none = np.zeros((1, n, n), dtype=bool)
        return lagged, none, none
    if isinstance(structure, W.WindowGraph):
        marks = structure.marks
        lag0 = marks[0]
        directed = (lag0 == W.TAIL) & (lag0.T == W.ARROW)
        return (
            (marks[1:] != W.NULL)[None],
            ((lag0 != W.NULL) | (lag0.T != W.NULL))[None],
            directed[None],
        )
    if isinstance(structure, CoefficientStack):
        # Blocks are [effect, cause]: transpose to [cause, effect]. A panel is scored from t = 1.
        stack = structure.stack[None, :2] if structure.kind == "varma" else structure.stack[1:]
        links = np.isfinite(stack) & (stack != 0)
        directed = links[:, 0].transpose(0, 2, 1)
        lagged = links[:, 1:].transpose(0, 1, 3, 2)
        return lagged, directed | directed.transpose(0, 2, 1), directed
    raise TypeError(f"no time-series links for {type(structure).__name__}")


def cross_link_scores(estimate: object, truth: object) -> dict[str, float]:
    """Precision, recall, and F1 of the window causal graph over its cross-links.

    A cross-link joins two different variables: a lagged link from ``i`` at ``t - k`` to ``j`` at
    ``t``, ``i != j``, or a same-step link between ``i`` and ``j``. Autodependencies (``i`` at
    ``t - k`` to ``i`` at ``t``) are left out. A lagged link is found when the estimate holds it at
    the same lag; time order orients it. A same-step link is found when the estimate joins the same
    two variables (``cross_*``), and, in the ``cross_oriented_*`` columns, only when it also points
    the same way: an unoriented same-step link then finds nothing. A panel pools its time points
    from the second. ``cross_n_true``, ``cross_n_est``, and ``cross_tp`` count the ``cross_*``
    links. A ratio with no denominator is 1, as :func:`window_scores` scores it. Raises
    ``ValueError`` when a truth has unobserved variables: its estimate covers the observed ones,
    and the true graph over them is not computed here.
    """
    if isinstance(truth, W.WindowGraph) and truth.latents:
        raise ValueError("cross_link_scores needs a truth without unobserved variables")
    est_lag, est_adj, est_dir = _window_links(estimate)
    true_lag, true_adj, true_dir = _window_links(truth)
    lags = max(est_lag.shape[1], true_lag.shape[1])
    est_lag, true_lag = (
        np.concatenate([a, np.zeros((a.shape[0], lags - a.shape[1], *a.shape[2:]), bool)], axis=1)
        for a in (est_lag, true_lag)
    )
    n = true_adj.shape[-1]
    off = ~np.eye(n, dtype=bool)
    upper = np.triu(off)

    def prf(prefix: str, pairs: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, float]:
        tp = sum(int((e & t).sum()) for e, t in pairs)
        n_est = sum(int(e.sum()) for e, _ in pairs)
        n_true = sum(int(t.sum()) for _, t in pairs)
        precision = tp / n_est if n_est else 1.0
        recall = tp / n_true if n_true else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        columns = {f"{prefix}precision": precision, f"{prefix}recall": recall, f"{prefix}f1": f1}
        if prefix == "cross_":
            columns.update(cross_n_true=float(n_true), cross_n_est=float(n_est), cross_tp=float(tp))
        return columns

    lagged = (est_lag & off, true_lag & off)
    scores = prf("cross_", [lagged, (est_adj & upper, true_adj & upper)])
    scores.update(prf("cross_oriented_", [lagged, (est_dir & off, true_dir & off)]))
    return scores


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
#: directed paths that pass through hidden nodes only, the target of RCD and BottomUpParceLiNGAM,
#: which mark a pair whether or not one node is the other's parent.
CONFOUNDED_TARGETS = ("pag", "hidden_cause")

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


def true_confounded_pairs(truth: GraphStructure, target: str) -> set[tuple[int, int]]:
    """Return the true confounded pairs ``i < j`` of a DAG `truth` with hidden nodes.

    `target` is one of :data:`CONFOUNDED_TARGETS`. Observed nodes come first, so a pair indexes the
    data columns.
    """
    if target == "pag":
        return _marked_pairs(np.asarray(_pag_of(truth).to_numpy()), _ARROW)
    if target == "hidden_cause":
        return _hidden_cause_pairs(truth)
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
#: BottomUpParceLiNGAM estimate; a pair with a hidden cause is marked, not oriented.
DIRECTED_TARGETS = ("projection",)

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
    hits = len(claimed & projection)
    both = len(claimed) + len(projection)
    scores["directed_projection_n_true"] = len(projection)
    scores["directed_projection_precision"] = hits / len(claimed) if claimed else np.nan
    scores["directed_projection_recall"] = hits / len(projection) if projection else np.nan
    scores["directed_projection_f1"] = 2 * hits / both if both else np.nan
    scores["directed_projection_errors"] = len(claimed ^ projection)
    return scores


#: The columns :func:`measurement_scores` returns, in order.
MEASUREMENT_COLUMNS = (
    "measurement_cluster_ari",
    "measurement_n_latents",
    "measurement_clusters_found",
    "measurement_order_accuracy",
)


def _latent_clusters(graph: GraphStructure) -> dict[int, frozenset[int]]:
    """Map each latent node of `graph` to the observed nodes it points into."""
    if graph.node_types is None:
        return {}
    latent = np.asarray(graph.node_types) == LATENT
    directed = _directed_adjacency(graph).astype(bool)
    return {
        int(node): frozenset(int(c) for c in np.flatnonzero(directed[node] & ~latent))
        for node in np.flatnonzero(latent)
    }


def measurement_scores(estimate: GraphStructure, truth: GraphStructure) -> dict[str, float]:
    """Score the latent clusters and the latent order of `estimate` against a measurement model.

    `truth` is a DAG whose latents each point into their own observed indicators and into the
    latents after them. A latent's cluster is the observed nodes it points into; an estimate with
    no latents, such as the empty graph, has none. Returns the adjusted Rand index of the two
    clusterings of the observed nodes (an unclustered node is a cluster of its own), the number of
    estimated latents, the share of true clusters the estimate has exactly, and, over the pairs of
    true latents whose clusters it has exactly, the share it orders as the truth does: its latent
    of the earlier one points into its latent of the later one. A ratio with no pair is ``NaN``.
    """
    from andrey.metrics import latent_cluster_ari

    true = _latent_clusters(truth)
    estimated = _latent_clusters(estimate)
    found = {members: node for node, members in estimated.items()}
    matched = {node: found[members] for node, members in true.items() if members in found}
    true_directed = _directed_adjacency(truth).astype(bool)
    est_directed = _directed_adjacency(estimate).astype(bool)
    pairs = correct = 0
    for a, b in combinations(sorted(matched), 2):
        if not (true_directed[a, b] or true_directed[b, a]):
            continue
        first, second = (a, b) if true_directed[a, b] else (b, a)
        pairs += 1
        correct += bool(est_directed[matched[first], matched[second]])
    return {
        "measurement_cluster_ari": float(latent_cluster_ari(estimate, truth)),
        "measurement_n_latents": len(estimated),
        "measurement_clusters_found": len(matched) / len(true) if true else np.nan,
        "measurement_order_accuracy": correct / pairs if pairs else np.nan,
    }


def sortability(data: np.ndarray, truth: GraphStructure | TemporalStructure) -> dict[str, float]:
    """Compute variance- and R²-sortability under the ground-truth DAG.

    Returns an empty dictionary when the optional sortability implementation
    is unavailable, or for a temporal truth: sortability is defined for i.i.d. rows, not a series.

    """
    if isinstance(truth, (TemporalStructure, W.WindowGraph, CoefficientStack)):
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
