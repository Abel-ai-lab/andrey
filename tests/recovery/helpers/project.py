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
            "FCI",
            "GFCI",
            "CDNOD",
            "IAMB",
            "InterIAMB",
            "MMPC",
            "ExactSearch",
            "BOSS",
            "GRaSP",
            "CALM",
        }
    )
    | _SCORE_GRAPH
)
# LiNGAM: candidate = ordering + weighted_adjacency, transposing native W back to the B layout.
_LINGAM = frozenset({"DirectLiNGAM", "ICALiNGAM"})
# Hidden-confounder PAGs: candidate = graph + weighted_adjacency as returned.
_CONFOUNDED = frozenset({"RCD", "BottomUpParceLiNGAM"})
# Pairwise direction tests: candidate = the two-node graph, both p-values, and both statistics.
_PAIRWISE = frozenset({"ANM", "PNL"})
# The recorded fields of the pairwise tests, by metadata key and direction.
_PAIRWISE_FIELDS = {
    "pval_forward": ("p_values", "forward"),
    "pval_backward": ("p_values", "backward"),
    "stat_forward": ("statistics", "forward"),
    "stat_backward": ("statistics", "backward"),
}
# PCMCI family: candidate = the per-lag graphs (LPCMCI's links with both marks), the conflicts,
# and the test p-values and statistics.
_PCMCI = frozenset({"PCMCI", "PCMCIplus", "LPCMCI"})


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
    if algorithm in _GRAPH:
        candidate: dict = {"graph": out.structure.to_numpy()}
        if algorithm in _SCORE_GRAPH:
            candidate["score"] = out.metadata["score"]
        return candidate
    if algorithm in _LINGAM:
        return {
            "causal_order": np.asarray(out.ordering),
            "weighted_adjacency": np.asarray(out.weighted_adjacency).T,
        }
    if algorithm in _CONFOUNDED:
        candidate = {
            "graph": out.structure.to_numpy(),
            "weighted_adjacency": np.asarray(out.weighted_adjacency),
        }
        if algorithm == "BottomUpParceLiNGAM":  # the order, unordered variables nested in a list
            candidate["causal_order"] = out.metadata["partial_order"]
        return candidate
    if algorithm in _PAIRWISE:
        md = out.metadata
        return {"graph": out.structure.to_numpy()} | {
            field: md[key][direction] for field, (key, direction) in _PAIRWISE_FIELDS.items()
        }
    if algorithm == "GIN":  # the graph over indicators and latents, and the clusters in order
        return {"graph": out.structure.to_numpy(), "causal_order": out.metadata["cluster_order"]}
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
    if algorithm == "VARLiNGAM":
        return {
            "causal_order": np.asarray(out.ordering),
            "adjacency_matrices": np.asarray(out.structure.lag_weights).transpose(0, 2, 1),
        }
    if algorithm == "VARMALiNGAM":
        s = out.structure
        return {
            "causal_order": np.asarray(out.ordering),
            "psis": np.asarray(s.lag_weights).transpose(0, 2, 1),
            "omegas": np.asarray(s.lag_weights_ma).transpose(0, 2, 1),
        }
    raise KeyError(f"no output projection registered for {algorithm!r}")
