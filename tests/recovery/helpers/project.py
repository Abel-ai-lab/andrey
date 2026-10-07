"""Baseline output projection shared by capture and recovery tests."""

from __future__ import annotations

from typing import Any

import numpy as np

# A graph candidate also carries the scalar objective score (from metadata).
_SCORE_GRAPH = frozenset({"GES", "GIES", "HC"})
# Graph-output algorithms: candidate = {"graph"[, "score"]}.
_GRAPH = frozenset({"PC", "FCI", "GFCI", "CDNOD", "ExactSearch", "BOSS", "GRaSP"}) | _SCORE_GRAPH
# LiNGAM: candidate = ordering + weighted_adjacency, transposing native W back to the B layout.
_LINGAM = frozenset({"DirectLiNGAM", "ICALiNGAM"})


def project_output(out: Any, algorithm: str) -> dict:
    """Project a ``StructureOutput`` (or a list for multi-group) to the baseline field layout."""
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
    raise KeyError(
        f"no output projection registered for {algorithm!r} "
        "(lossy layouts like Granger/CAMUV/RCD add their own in-slice)"
    )
