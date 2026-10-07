"""The DGP product types: a sampled dataset and its quality report.

``andrey.data`` returns a :class:`CausalDataset` -- observational ``data`` paired with the
ground-truth causal ``graph`` (an Andrey :class:`~andrey.core.structure.GraphStructure`), the drawn
structural ``params``, a ``provenance`` record sufficient to regenerate it, and a ``report`` of
honesty/QA diagnostics. All three container types are frozen ``kw_only`` dataclasses, matching the
core ``Structure`` style, so fields grow without ossifying a positional contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from andrey.core.structure import LATENT, GraphStructure


@dataclass(frozen=True, kw_only=True)
class QAReport:
    """Honesty / quality diagnostics for a sampled dataset (all ``None`` until computed).

    Populated by ``andrey.data.qa``; the field names are a stable contract downstream consumers
    bind to.

    Parameters
    ----------
    varsortability : float or None, default=None
        Fraction of directed paths ordered by increasing marginal variance (Reisach et al. 2021);
        ``0.5`` is uninformative. ``None`` until computed.
    r2sortability : float or None, default=None
        The R^2-based companion (Reisach et al. 2023); ``None`` when uncomputed or undefined
        (for example, ``n <= d``, where the correlation matrix is singular).
    faithfulness_accept_rate : float or None, default=None
        Accept rate of the optional faithfulness screen; ``None`` when no screen ran.
    scale : str or None, default=None
        The scale/normalization mode used (``"raw"`` / ``"standardize"`` / ``"rescale"``).
    """

    varsortability: float | None = None
    r2sortability: float | None = None
    faithfulness_accept_rate: float | None = None
    scale: str | None = None


@dataclass(frozen=True, kw_only=True)
class SCMParams:
    """The drawn structural parameters, aligned to the truth graph's edge list.

    Part of the return contract so weighted metrics and parity comparison read the exact realized
    coefficients rather than regenerating them.

    Parameters
    ----------
    edges : np.ndarray of shape (m, 2)
        Directed edges ``(parent, child)`` in node-label space; row ``k`` pairs with ``weights[k]``.
    weights : np.ndarray of shape (m,)
        Structural coefficient of each edge (empty for parameter-free functionals).
    noise_scales : np.ndarray of shape (d,)
        Per-node exogenous-noise standard deviation.
    functional : str
        Functional-form name (for example, ``"linear"``).
    noise : str
        Noise-family name (for example, ``"gaussian"``).
    """

    edges: np.ndarray
    weights: np.ndarray
    noise_scales: np.ndarray
    functional: str
    noise: str


@dataclass(frozen=True, kw_only=True)
class CausalDataset:
    """Observational data paired with its ground-truth causal graph.

    Parameters
    ----------
    data : np.ndarray of shape (n, n_obs)
        Observational sample, ``float64`` by default (see ``SCM.sample(dtype=...)``). Columns
        are the **observed** variables, indexed to match the observed nodes of ``graph``
        (observed-first: for a causally-sufficient SCM ``n_obs == graph.n_nodes`` and column
        ``j`` is node ``j``).
    graph : GraphStructure
        Ground-truth causal graph over all ``d`` nodes (``kind="dag"``); latent confounders, when
        present, are flagged via ``graph.node_types`` and dropped from ``data``.
    report : QAReport
        Honesty / QA diagnostics (fields are ``None`` until computed).
    provenance : dict
        Regeneration record: root entropy, per-component seed layout, SCM config, package versions.
    params : SCMParams
        The drawn structural coefficients and noise scales.

    Notes
    -----
    Use :attr:`observed_graph` -- never ``graph`` directly -- when scoring a discovery estimate, so
    the graph's node set matches ``data``'s columns for latent SCMs.
    """

    data: np.ndarray
    graph: GraphStructure
    report: QAReport
    provenance: dict
    params: SCMParams

    @property
    def observed_graph(self) -> GraphStructure:
        """The truth graph restricted to observed nodes, aligned to ``data``'s columns.

        Equal to ``graph`` for a causally-sufficient SCM. For a latent SCM it is the subgraph
        induced on the observed nodes (indices ``0 .. n_obs-1``, observed-first) -- a DAG over
        exactly ``data``'s columns, so ``shd(estimate, observed_graph)`` is well-shaped. This
        induced DAG drops the confounding paths through latents; the confounding-aware observed
        truth is the latent-projected MAG (``andrey.data.latent.marginal``).
        """
        types = self.graph.node_types
        if types is None or not np.any(types == LATENT):
            return self.graph
        from .graphs import dag_truth

        n_obs = self.data.shape[1]
        e = self.params.edges
        keep = (e[:, 0] < n_obs) & (e[:, 1] < n_obs)
        return dag_truth(e[keep, 0], e[keep, 1], n_obs)
