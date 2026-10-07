"""Scoring utilities for benchmark estimates.

``score_result`` scores through ``andrey.metrics``, against the truth's PAG for a PAG estimate.

``sortability`` computes optional variance- and R2-sortability diagnostics
for generated datasets.

"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

import andrey.metrics as M
from andrey import GraphStructure
from andrey.core import LATENT
from andrey.data.latent import marginal

# Andrey endpoint-mark convention (mirrors contracts.py): mark AT node i on edge i-j is marks[i, j].
_TAIL, _ARROW = 1, 2


def structure_hash(structure: GraphStructure) -> str:
    """A hash of the normalized graph, comparable across packages."""
    edges = sorted(tuple(int(v) for v in e) for e in structure.to_edges())
    return hashlib.sha1(f"{structure.kind}:{structure.n_nodes}:{edges}".encode()).hexdigest()


def score_result(estimate: GraphStructure, truth: GraphStructure) -> dict[str, Any]:
    """Score an estimated graph against the ground truth.

    A PAG estimate is scored against the truth's PAG - the class FCI and RFCI recover, whose circles
    mark what no observational method can orient. Scoring against the DAG charges each circle as an
    error, even for the true PAG. Every other estimate goes to ``andrey.metrics.score`` as is, which
    scores a CPDAG through the truth's essential graph.

    A truth with hidden nodes is scored as its observed PAG for every estimate kind. The truth DAG
    includes nodes absent from the data. ``andrey.metrics.score`` would treat that DAG as a latent
    clustering.
    """
    if truth.kind == "dag" and (estimate.kind == "pag" or _has_hidden(truth)):
        truth = _pag_of(truth)
    return dict(M.score(estimate, truth))


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


def sortability(data: np.ndarray, truth: GraphStructure) -> dict[str, float]:
    """Compute variance- and R²-sortability under the ground-truth DAG.

    Returns an empty dictionary when the optional sortability implementation
    is unavailable.

    """
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
