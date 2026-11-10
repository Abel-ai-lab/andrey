"""The DGP product types: a sampled dataset and its quality report.

``andrey.data`` returns a :class:`CausalDataset` -- observational ``data`` paired with the
ground-truth causal ``graph`` (an Andrey :class:`~andrey.core.structure.GraphStructure`), the drawn
structural ``params``, a ``provenance`` record sufficient to regenerate it, and a ``report`` of
honesty/QA diagnostics. All three container types are frozen ``kw_only`` dataclasses, matching the
core ``Structure`` style, so fields grow without ossifying a positional contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
        The scale mode used (``"standardize"``, ``"unitless"``, ``"rescale"``, or ``"raw"``).
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
    weights : np.ndarray of shape (m,) or (m, 3, n_features)
        Structural coefficient of each edge (empty for parameter-free functionals). Under the
        location-scale functional, the weight of each edge in the location. Under the
        random-features functional, the frequencies, phases, and amplitudes of each edge's
        function.
    noise_scales : np.ndarray of shape (d,)
        Per-node exogenous-noise standard deviation. Under the location-scale functional the noise
        of a node is this constant times a scale that depends on its parents. A measurement
        column's entry is the standard deviation of its measurement noise.
    functional : str
        Functional-form name (for example, ``"linear"``).
    noise : str
        Noise-family name (for example, ``"gaussian"``).
    scale_weights : np.ndarray of shape (m,) or (0,), default=empty
        Under the location-scale functional, the weight of each edge in the scale, row ``k``
        pairing with ``edges[k]``; empty for every other functional.

    Notes
    -----
    The edges are those of the structural causal model. The edge from a variable into its
    measurement column is in the dataset's ``graph`` and in ``measurements.parents``, not here.
    """

    edges: np.ndarray
    weights: np.ndarray
    noise_scales: np.ndarray
    functional: str
    noise: str
    scale_weights: np.ndarray = field(default_factory=lambda: np.empty(0))


@dataclass(frozen=True, kw_only=True)
class Missingness:
    """The missing values drawn over a dataset, with the parents of each missingness indicator.

    The missingness indicator ``R_i`` of column ``i`` is 1 where its value is missing. The m-graph
    adds every indicator to the causal graph, with an edge into it from each of its parents
    (Mohan, Pearl, and Tian 2013); :mod:`andrey.data.missingness` draws them.

    Parameters
    ----------
    mask : np.ndarray of shape (n, d)
        ``True`` where a value is missing; the data hold ``NaN`` there.
    parents : dict of int to list of int
        Each column that can go missing, mapped to the columns its indicator depends on, in
        ascending order: ``[]`` under ``"mcar"``, always-observed columns under ``"mar"``, other
        columns that can go missing under ``"mnar"``, and ``[i]`` for column ``i`` under
        ``"self_masking"``. A column it does not list is always observed.
    mechanism : str
        ``"mcar"``, ``"mar"``, ``"mnar"``, or ``"self_masking"``.
    form : str or None
        How an indicator depends on its parents, ``"logistic"`` or ``"quantile"``; ``None`` under
        ``"mcar"``.
    rate : float
        The expected share of missing values in each column that can go missing.

    References
    ----------
    Mohan, K., Pearl, J., and Tian, J. (2013). Graphical models for inference with missing data.
    Advances in Neural Information Processing Systems 26.
    """

    mask: np.ndarray
    parents: dict[int, list[int]]
    mechanism: str
    form: str | None
    rate: float


@dataclass(frozen=True, kw_only=True)
class Measurements:
    """The measurement columns added to a dataset, each a variable plus independent noise.

    :func:`andrey.data.measurement.noisy_children` describes them and ``measurements=`` on
    :meth:`~andrey.data.SCM.sample` draws them.

    Parameters
    ----------
    parents : dict of int to int
        Each measurement column mapped to the column it measures, both as columns of the
        returned data, in ascending order of the measurement columns. In the true graph the
        measured column is the measurement column's only parent.
    noise : str
        The family of the measurement noise (for example, ``"gaussian"``).
    noise_scale : float
        The standard deviation of the measurement noise, as a multiple of the standard deviation
        of the measured column.
    """

    parents: dict[int, int]
    noise: str
    noise_scale: float


@dataclass(frozen=True, kw_only=True)
class CausalDataset:
    """Simulated data paired with its ground-truth causal graph.

    Parameters
    ----------
    data : np.ndarray of shape (n, n_obs)
        The sample, ``float64`` by default (see ``SCM.sample(dtype=...)``): observational rows,
        each environment's rows when ``env`` is set, or one row per time step for a time-series
        model. Columns are the **observed** variables, indexed to match the observed nodes of
        ``graph`` (observed-first: for a causally-sufficient SCM ``n_obs == graph.n_nodes`` and
        column ``j`` is node ``j``).
    graph : GraphStructure or TemporalStructure
        Ground-truth causal graph. For an i.i.d. model, a ``GraphStructure`` over all ``d`` nodes
        (``kind="dag"``); latent confounders, when present, are flagged via ``graph.node_types``
        and dropped from ``data``. For a time-series model, a ``TemporalStructure``, one graph per
        lag.
    report : QAReport
        Honesty / QA diagnostics (fields are ``None`` until computed).
    provenance : dict
        Regeneration record: root entropy, per-component seed layout, SCM config, package versions.
    params : SCMParams
        The drawn structural coefficients and noise scales.
    env : np.ndarray of shape (n,) or None, default=None
        The environment of each row, ``0`` to ``K - 1``, its rows together in that order, when the
        data were drawn under interventions; ``None`` otherwise.
    targets : dict of int to list of int, or None, default=None
        Each environment's intervened columns, after the column shuffle: ``{0: [], 1: [3]}`` when
        environment 0 has no intervention and environment 1 intervenes on column 3. ``None`` when
        ``env`` is ``None``.
    missing : Missingness or None, default=None
        The missing values drawn by ``missing=`` on :meth:`~andrey.data.SCM.sample`, their mask
        and the parents of each missingness indicator; ``None`` when the data are complete.
    measurements : Measurements or None, default=None
        The measurement columns added by ``measurements=`` on
        :meth:`~andrey.data.SCM.sample`, each with the column it measures; ``None`` without them.
        They are columns of ``data``, shuffled with the others when the sample shuffles its
        columns, and nodes of ``graph``, each a child of the column it measures.

    Notes
    -----
    For an i.i.d. model, use :attr:`observed_graph` -- never ``graph`` directly -- when scoring a
    discovery estimate, so the graph's node set matches ``data``'s columns for latent SCMs.
    """

    data: np.ndarray
    graph: GraphStructure
    report: QAReport
    provenance: dict
    params: SCMParams
    env: np.ndarray | None = None
    targets: dict[int, list[int]] | None = None
    missing: Missingness | None = None
    measurements: Measurements | None = None

    @property
    def observed_graph(self) -> GraphStructure:
        """The truth graph restricted to observed nodes, aligned to ``data``'s columns.

        Equal to ``graph`` for a causally-sufficient SCM. For a latent SCM it is the subgraph
        induced on the observed nodes (indices ``0 .. n_obs-1``, observed-first) -- a DAG over
        exactly ``data``'s columns, measurement columns included, so
        ``shd(estimate, observed_graph)`` is well-shaped. This
        induced DAG drops the confounding paths through latents; the confounding-aware observed
        truth is the latent-projected MAG (``andrey.data.latent.marginal``).
        """
        types = self.graph.node_types
        if types is None or not np.any(types == LATENT):
            return self.graph
        from .graphs import dag_truth

        n_obs = self.data.shape[1]
        e = self.params.edges
        e = e[(e[:, 0] < n_obs) & (e[:, 1] < n_obs)]
        if self.measurements is not None:
            measured = [(parent, column) for column, parent in self.measurements.parents.items()]
            e = np.concatenate([e, np.array(measured, dtype=e.dtype).reshape(-1, 2)])
        return dag_truth(e[:, 0], e[:, 1], n_obs)
