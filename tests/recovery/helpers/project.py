"""Baseline output projection shared by capture and recovery tests."""

from __future__ import annotations

from typing import Any

import numpy as np

from andrey.core import ARROW, TAIL
from andrey.core.structure import GraphStructure

# A graph candidate also carries the scalar objective score (from metadata).
_SCORE_GRAPH = frozenset({"GES", "GIES", "XGES", "HC", "Tabu", "MMHC", "RSMAX2"})
# Graph-output algorithms: candidate = {"graph"[, "score"]}.
_GRAPH = (
    frozenset(
        {
            "PC",
            "AGES",
            "FCI",
            "GFCI",
            "CDNOD",
            "IAMB",
            "InterIAMB",
            "GS",
            "FastIAMB",
            "MMPC",
            "HITONPC",
            "PCMB",
            "ExactSearch",
            "BOSS",
            "BOSSFCI",
            "GRaSP",
            "GRaSPFCI",
            "CALM",
        }
    )
    | _SCORE_GRAPH
)
# LiNGAM: candidate = ordering + weighted_adjacency, transposing native W back to the B layout.
_LINGAM = frozenset({"DirectLiNGAM", "ICALiNGAM"})
# Hidden-confounder PAGs: candidate = graph + weighted_adjacency as returned.
_CONFOUNDED = frozenset({"RCD", "BottomUpParceLiNGAM"})
# CAM-UV: candidate = the graph's marks and the marked pairs.
_MARKED_PAIRS = frozenset({"CAMUV"})
# DAG and causal-order algorithms: candidate = the DAG and the order.
_ORDERED_DAG = frozenset({"CAM", "RESIT", "ScoreMatching", "NoGAM"})
# Pairwise direction tests: candidate = the two-node graph, both p-values, and both statistics.
_PAIRWISE = frozenset({"ANM", "PNL", "LOCI"})
# The recorded fields of the pairwise tests, by metadata key and direction.
_PAIRWISE_FIELDS = {
    "pval_forward": ("p_values", "forward"),
    "pval_backward": ("p_values", "backward"),
    "stat_forward": ("statistics", "forward"),
    "stat_backward": ("statistics", "backward"),
}
# LOCI's added fields: the two models' mean log-likelihoods.
_LIKELIHOOD_FIELDS = {
    "ll_forward": ("log_likelihoods", "forward"),
    "ll_backward": ("log_likelihoods", "backward"),
}
# PCMCI family: candidate = the per-lag graphs (LPCMCI's links with both marks), the conflicts,
# and the test p-values and statistics.
_PCMCI = frozenset({"PCMCI", "PCMCIplus", "LPCMCI"})
# ICP: candidate = the DAG into the target, the accepted sets, and each column's p-value, ``None``
# at the target.
_INVARIANT = frozenset({"ICP"})
# Weighted temporal fits: candidate = the per-lag graphs and their weights.
_WEIGHTED_LAGS = frozenset({"DYNOTEARS", "Granger"})
# Weighted DAGs: candidate = the DAG and its weights, ``weighted_adjacency[i, j]`` for ``i -> j``.
_WEIGHTED_DAG = frozenset({"NOTEARS"})
# VARMA-LiNGAM: candidate = the causal order and both coefficient stacks in the native B layout.
_VARMA = frozenset({"VARMALiNGAM", "VARMALiNGAMPruned"})


# Y-structures: candidate = the edgeless graph, the claimed ``[treatment, outcome]`` pairs by
# descending score, and their scores.
_CLAIMS = frozenset({"YStructures"})


def found_dag(out: Any) -> GraphStructure:
    """The DAG an exact search found, which ``metadata["dag"]`` lists as ``[i, j]`` for ``i -> j``;
    the output's own structure is the DAG's CPDAG."""
    marks = np.zeros((out.structure.n_nodes,) * 2, dtype=np.int8)
    for i, j in out.metadata["dag"]:
        marks[i, j], marks[j, i] = TAIL, ARROW
    return GraphStructure.from_numpy(marks, kind="dag", labels=out.structure.labels)


def project_output(out: Any, algorithm: str) -> dict:
    """Project a ``StructureOutput`` (or a list for multi-group) to the baseline field layout."""
    if algorithm == "ExactSearch":  # the recorded graph is the DAG the search found
        return {"graph": found_dag(out).to_numpy()}
    if algorithm == "IGSP":  # the graph and both parts of its objective
        md = out.metadata
        return {
            "graph": out.structure.to_numpy(),
            "score": md["score"],
            "n_contradicting": md["n_contradicting"],
        }
    if algorithm in _GRAPH:
        candidate: dict = {"graph": out.structure.to_numpy()}
        if algorithm in _SCORE_GRAPH:
            candidate["score"] = out.metadata["score"]
        return candidate
    if algorithm == "GnIES":  # the graph, its score, and the learned targets
        return {
            "graph": out.structure.to_numpy(),
            "score": out.metadata["score"],
            "targets": list(out.metadata["changing_variables"]),
        }
    if algorithm in _LINGAM:
        return {
            "causal_order": np.asarray(out.ordering),
            "weighted_adjacency": np.asarray(out.weighted_adjacency).T,
        }
    if algorithm == "FCIJCI":  # the PAG over the data, and the index's edges
        return {"graph": out.structure.to_numpy(), "index_edges": out.metadata["index_edges"]}
    if algorithm in _CONFOUNDED:
        candidate = {
            "graph": out.structure.to_numpy(),
            "weighted_adjacency": np.asarray(out.weighted_adjacency),
        }
        if algorithm == "BottomUpParceLiNGAM":  # the order, unordered variables nested in a list
            candidate["causal_order"] = out.metadata["partial_order"]
        return candidate
    if algorithm in _MARKED_PAIRS:
        pairs = np.asarray(out.metadata["latent_pairs"], dtype=np.int64).reshape(-1, 2)
        return {"graph": out.structure.to_numpy(), "latent_pairs": pairs}
    if algorithm in _ORDERED_DAG:
        return {"graph": out.structure.to_numpy(), "causal_order": np.asarray(out.ordering)}
    if algorithm in _CLAIMS:
        claims = out.metadata["effects"]
        return {
            "graph": out.structure.to_numpy(),
            "claims": [[c["treatment"], c["outcome"]] for c in claims],
            "scores": [c["score"] for c in claims],
        }
    if algorithm in _INVARIANT:
        return {
            "graph": out.structure.to_numpy(),
            "accepted_sets": out.metadata["accepted_sets"],
            "p_values": out.metadata["p_values"],
        }
    if algorithm in _PAIRWISE:
        md = out.metadata
        fields = _PAIRWISE_FIELDS | (_LIKELIHOOD_FIELDS if algorithm == "LOCI" else {})
        return {"graph": out.structure.to_numpy()} | {
            field: md[key][direction] for field, (key, direction) in fields.items()
        }
    if algorithm == "GIN":  # the graph over indicators and latents, and the clusters in order
        return {"graph": out.structure.to_numpy(), "causal_order": out.metadata["cluster_order"]}
    if algorithm == "UT-IGSP":  # the graph, its objective, and each environment's targets
        targets = out.metadata["targets"]
        return {
            "graph": out.structure.to_numpy(),
            "score": out.metadata["score"],
            "targets": [targets[key] for key in sorted(targets)],
        }
    if algorithm == "JPCMCIplus":  # PCMCI+'s fields, and the context and dummy links
        s, md = out.structure, out.metadata
        return {
            "graph": np.stack([s.lag(lag).to_numpy() for lag in s.lags]),
            "p_values": np.asarray(md["p_values"]),
            "statistics": np.asarray(md["statistics"]),
            "conflicts": md["conflicts"],
            "context_links": md["context_links"],
            "index_edges": md["index_edges"],
            "time_index_edges": md["time_index_edges"],
        }
    if algorithm in _PCMCI:
        s = out.structure
        candidate = {
            "p_values": np.asarray(out.metadata["p_values"]),
            "statistics": np.asarray(out.metadata["statistics"]),
        }
        if algorithm == "LPCMCI":
            candidate["links"] = out.metadata["links"]
        else:
            candidate["graph"] = np.stack([s.lag(lag).to_numpy() for lag in s.lags])
        if algorithm != "PCMCI":
            candidate["conflicts"] = out.metadata["conflicts"]
        return candidate
    if algorithm in _WEIGHTED_LAGS:
        s = out.structure
        return {
            "graph": np.stack([s.lag(lag).to_numpy() for lag in s.lags]),
            "lag_weights": np.asarray(s.lag_weights),
        }
    if algorithm in _WEIGHTED_DAG:
        return {
            "graph": out.structure.to_numpy(),
            "weighted_adjacency": np.asarray(out.weighted_adjacency),
        }
    if algorithm == "LongitudinalLiNGAM":
        return {
            "causal_orders": [list(order) for order in out.metadata["causal_orders"]],
            "time_weights": np.asarray(out.structure.time_weights),
        }
    if algorithm == "MultiGroupDirectLiNGAM":
        if not out:
            raise ValueError("MultiGroupDirectLiNGAM projection needs a non-empty list of outputs")
        order = np.asarray(out[0].ordering)
        if not all(np.array_equal(np.asarray(o.ordering), order) for o in out):
            raise AssertionError("MultiGroupDirectLiNGAM groups disagree on the causal order")
        return {
            "causal_order": order,
            "group_weighted_adjacency": np.stack([np.asarray(o.weighted_adjacency).T for o in out]),
        }
    if algorithm in _VARMA:
        s = out.structure
        return {
            "causal_order": np.asarray(out.ordering),
            "psis": np.asarray(s.lag_weights).transpose(0, 2, 1),
            "omegas": np.asarray(s.lag_weights_ma).transpose(0, 2, 1),
        }
    raise KeyError(f"no output projection registered for {algorithm!r}")
