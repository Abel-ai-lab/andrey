"""The structural causal model: compose a graph, functional forms, and noise; sample data.

An :class:`SCM` pairs a lazy graph spec with functional and noise specs. ``sample(n, seed)`` derives
**independent, statelessly-keyed** RNG streams from the seed -- one each for the graph, the
weights, the noise, and the column shuffle -- so a dataset is byte-reproducible from
``(root_entropy, spawn_key)`` alone and a change to one axis cannot shift another's stream. Every
draw takes an explicit ``np.random.Generator``; the module never touches ``np.random.*`` or
``ANDREY_SEED``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

import andrey
from andrey.core.structure import LATENT, OBSERVED

from . import qa
from .dataset import CausalDataset, QAReport, SCMParams
from .functional import Functional, additive_noise, linear, post_nonlinear, var, varma
from .graphs import (
    GraphSpec,
    dag_truth,
    erdos_renyi,
    hub,
    params_for_density,
    scale_free,
    small_world,
)
from .latent import latent_order
from .noise import Noise, exponential, gaussian, gumbel, laplace, uniform
from .qa import FaithfulnessScreen
from .sortability import standardize
from .temporal import relabel_truth

_SCALES = ("raw", "standardize", "rescale")
_TEMPORAL_BURN_IN = 200  # discarded prefix so the returned window is stationary

# Per-component sub-keys off the root SeedSequence: the graph, the weights, the noise, the latent
# selection, the temporal lag matrices, and the column shuffle each get an independent stream, so
# a change to one component's draws never shifts another's.
_GRAPH, _WEIGHTS, _NOISE, _LATENT, _LAG, _COLUMNS = 0, 1, 2, 3, 4, 5
_SEED_SCHEME = 1  # bump if the seed-derivation layout ever changes (recorded in provenance)


def _component_rng(root: np.random.SeedSequence, component: int) -> np.random.Generator:
    """Derive an independent ``Generator`` for one pipeline component, statelessly from ``root``.

    Uses ``spawn_key`` extension rather than ``root.spawn()`` (which mutates a counter), so the
    stream for task ``i`` component ``c`` is a pure function of ``(root.entropy, (*root.spawn_key,
    c))`` -- reproducible regardless of submission order or worker count.
    """
    seq = np.random.SeedSequence(entropy=root.entropy, spawn_key=(*root.spawn_key, component))
    return np.random.default_rng(seq)


def _column_order(root: np.random.SeedSequence, n_observed: int, shuffle: bool) -> np.ndarray:
    """Return the drawn index of each output column; the identity unless ``shuffle``.

    The permutation comes from the column stream of ``root``, so no other draw depends on it.
    """
    if not shuffle:
        return np.arange(n_observed)
    return _component_rng(root, _COLUMNS).permutation(n_observed)


@dataclass(frozen=True, kw_only=True)
class SCM:
    """A structural causal model: a causal graph, functional forms, and exogenous noise.

    Parameters
    ----------
    graph : GraphSpec
        A lazy graph spec (for example, ``andrey.data.graphs.erdos_renyi(...)``). A concrete
        ``GraphStructure`` is not accepted.
    functional : Functional
        The structural-equation family (for example, ``andrey.data.functional.linear()``).
    noise : Noise
        The exogenous-noise family (for example, ``andrey.data.noise.gaussian()``).

    Examples
    --------
    >>> import andrey.data as data
    >>> scm = data.SCM(graph=data.graphs.erdos_renyi(d=8, avg_degree=2),
    ...                functional=data.functional.linear(),
    ...                noise=data.noise.gaussian())
    >>> ds = scm.sample(n=100, seed=0)
    >>> ds.data.shape
    (100, 8)
    >>> ds.graph.kind
    'dag'
    """

    graph: GraphSpec
    functional: Functional
    noise: Noise
    latents: int = 0

    def sample(
        self,
        n: int,
        seed: int,
        *,
        scale: str = "raw",
        screen: FaithfulnessScreen | None = None,
        dtype: npt.DTypeLike = np.float64,
        shuffle: bool = True,
    ) -> CausalDataset:
        """Draw ``n`` observational rows; return a :class:`~andrey.data.dataset.CausalDataset`.

        Parameters
        ----------
        n : int
            Number of samples (rows).
        seed : int
            Root seed; the graph, weights, and noise streams are derived from it statelessly.
        scale : {"raw", "standardize", "rescale"}, default="raw"
            Honesty mode. ``"raw"`` leaves the generated variances (high varsortability).
            ``"standardize"`` z-scores columns post-hoc -- varsortability becomes 0.5 but
            **R^2-sortability is unchanged** (standardized synthetic data is not scale-hard).
            ``"rescale"`` standardizes each node *during generation* (ancestral rescaling), killing
            varsortability by construction while preserving the recovered structure.
        screen : FaithfulnessScreen or None, default=None
            If given, reject-resample near-unfaithful draws (bounded; raises after
            ``max_attempts``); the accept rate is recorded in ``report.faithfulness_accept_rate``.
        dtype : npt.DTypeLike, default=np.float64
            Output dtype of ``data``. The structural equations are always solved in ``float64``;
            this only controls the final cast. Every consumer upcasts to ``float64``, so the
            default avoids a lossy round-trip; ``np.float32`` gives a half-footprint array.
        shuffle : bool, default=True
            Shuffle the observed columns with a permutation from the seed's own column stream, and
            relabel the true graph and the per-node parameters to match; latent nodes stay last.
            The graph, weights, and noise are the same either way, so the dataset differs from
            ``shuffle=False`` only in its column order. ``provenance["column_order"][j]`` is the
            index column ``j`` has with ``shuffle=False``, where the order follows how the graph
            was drawn (a scale-free graph's hubs come first, a small-world graph keeps its ring
            order).

        Returns
        -------
        CausalDataset
        """
        if scale not in _SCALES:
            raise ValueError(f"scale must be one of {_SCALES}, got {scale!r}")
        if not isinstance(self.graph, GraphSpec):
            raise NotImplementedError(
                "SCM.sample currently supports a GraphSpec; concrete-graph input is not supported"
            )
        root = np.random.SeedSequence(seed)
        if screen is None:
            return self._sample_once(n, root, scale=scale, dtype=dtype, shuffle=shuffle)
        for attempt in range(screen.max_attempts):
            attempt_root = np.random.SeedSequence(
                entropy=root.entropy, spawn_key=(*root.spawn_key, attempt)
            )
            ds = self._sample_once(n, attempt_root, scale=scale, dtype=dtype, shuffle=shuffle)
            edges = ds.params.edges
            n_obs = ds.data.shape[1]
            obs_edges = edges[
                (edges[:, 0] < n_obs) & (edges[:, 1] < n_obs)
            ]  # latents dropped from data
            if screen.accepts(qa.faithfulness_margin(ds.data, obs_edges)):
                return replace(
                    ds, report=replace(ds.report, faithfulness_accept_rate=1.0 / (attempt + 1))
                )
        raise RuntimeError(
            f"faithfulness screen (min_abs_corr={screen.min_abs_corr}) "
            f"failed after {screen.max_attempts} attempts"
        )

    def _sample_once(
        self,
        n: int,
        root: np.random.SeedSequence,
        *,
        scale: str,
        dtype: npt.DTypeLike,
        shuffle: bool,
    ) -> CausalDataset:
        _, draw = self.graph.materialize(_component_rng(root, _GRAPH))
        weights = self.functional.draw_weights(draw, _component_rng(root, _WEIGHTS))
        noise_rng = _component_rng(root, _NOISE)
        scales = self.noise.draw_scales(draw.d, noise_rng)
        if self.functional.is_temporal:
            return self._sample_temporal(
                n, draw, weights, scales, noise_rng, root, scale, dtype, shuffle
            )
        innovations = self.noise.draw(n, scales, noise_rng)
        values = self.functional.evaluate(draw, weights, innovations, rescale=(scale == "rescale"))
        if scale == "standardize":
            values = standardize(values)
        data = values.astype(dtype)
        if not np.all(np.isfinite(data)):  # deep weighted paths can overflow float32
            raise OverflowError(
                "generated data has non-finite values (variance grew past float32 down a "
                "deep causal path); reduce the weight range / density, or use "
                "scale='rescale' to bound it"
            )
        # ``order[j]`` is the drawn label of output node ``j``: the observed nodes in column order,
        # then the latents. Edge rows keep their order, so ``weights`` stays aligned with ``edges``.
        edges = np.stack([draw.parents, draw.children], axis=1)
        n_obs = draw.d - self.latents
        order = np.arange(draw.d)
        node_types = None
        if self.latents > 0:
            order = latent_order(edges, draw.d, self.latents, _component_rng(root, _LATENT))
            node_types = np.full(draw.d, OBSERVED, dtype=np.int8)
            node_types[n_obs:] = LATENT
        columns = _column_order(root, n_obs, shuffle)
        order[:n_obs] = order[columns]
        data = np.take(data, order[:n_obs], axis=1)  # C order; data[:, order] is Fortran order
        edges = np.argsort(order)[edges]
        truth = dag_truth(edges[:, 0], edges[:, 1], draw.d, node_types=node_types)
        params = SCMParams(
            edges=edges,
            weights=weights,
            noise_scales=scales[order],
            functional=self.functional.name,
            noise=self.noise.name,
        )
        provenance = {
            "root_entropy": str(root.entropy),
            "spawn_key": list(root.spawn_key),
            "seed_scheme": _SEED_SCHEME,
            "column_order": columns.tolist(),
            "config": {
                "graph": {
                    "model": self.graph.name,
                    "num_nodes": draw.d,
                    "density": self.graph.density,
                    "params": self.graph.params,
                },
                "functional": self.functional.name,
                "noise": self.noise.name,
                "n": int(n),
                "scale": scale,
                "dtype": np.dtype(dtype).name,
                "latents": int(self.latents),
                "shuffle": bool(shuffle),
            },
            "versions": {"andrey": andrey.__version__, "numpy": np.__version__},
        }
        # Sortability is a property of the observed data over observed-observed edges.
        obs_mask = (edges[:, 0] < n_obs) & (edges[:, 1] < n_obs)
        report = qa.report(data, edges[obs_mask], n_obs, scale=scale)
        return CausalDataset(
            data=data, graph=truth, report=report, provenance=provenance, params=params
        )

    def _sample_temporal(self, n, draw, weights, scales, noise_rng, root, scale, dtype, shuffle):
        """Generate a VAR/VARMA series; ``n`` is the number of time steps, truth temporal."""
        if self.latents > 0:
            raise NotImplementedError(
                "latent confounders are not supported for temporal functionals"
            )
        if scale != "raw":
            raise ValueError("temporal functionals support only scale='raw'")
        innovations = self.noise.draw(n + _TEMPORAL_BURN_IN, scales, noise_rng)
        series, temporal_truth = self.functional.generate_temporal(  # ty: ignore[unresolved-attribute]  # gated by is_temporal (VAR/VARMA only)
            draw, weights, innovations, _component_rng(root, _LAG), burn_in=_TEMPORAL_BURN_IN
        )
        data = series.astype(dtype)
        if not np.all(np.isfinite(data)):
            raise OverflowError(
                "temporal generation produced non-finite values (unstable process); "
                "reduce lag_density or the weight range"
            )
        # The shuffle relabels the lag-0 DAG and every lag block on both axes.
        order = _column_order(root, draw.d, shuffle)
        data = np.take(data, order, axis=1)
        temporal_truth = relabel_truth(temporal_truth, order)
        edges = np.stack([draw.parents, draw.children], axis=1)  # the instantaneous (lag-0) DAG
        params = SCMParams(
            edges=np.argsort(order)[edges],
            weights=weights,
            noise_scales=scales[order],
            functional=self.functional.name,
            noise=self.noise.name,
        )
        provenance = {
            "root_entropy": str(root.entropy),
            "spawn_key": list(root.spawn_key),
            "seed_scheme": _SEED_SCHEME,
            "column_order": order.tolist(),
            "config": {
                "graph": {
                    "model": self.graph.name,
                    "num_nodes": draw.d,
                    "density": self.graph.density,
                    "params": self.graph.params,
                },
                "functional": self.functional.name,
                "noise": self.noise.name,
                "n": int(n),
                "scale": scale,
                "dtype": np.dtype(dtype).name,
                "temporal": True,
                "n_lags": self.functional.n_lags,  # ty: ignore[unresolved-attribute]  # gated by is_temporal (VAR/VARMA only)
                "ma_lags": getattr(
                    self.functional, "ma_lags", None
                ),  # VARMA MA order (None for VAR)
                "shuffle": bool(shuffle),
            },
            "versions": {"andrey": andrey.__version__, "numpy": np.__version__},
        }
        # Sortability targets iid data; it does not apply to a time series.
        report = QAReport(scale=scale)
        return CausalDataset(
            data=data,
            graph=temporal_truth,  # ty: ignore[invalid-argument-type]  # temporal truth is a TemporalStructure
            report=report,
            provenance=provenance,
            params=params,
        )


_GRAPH_BUILDERS = {
    "erdos_renyi": erdos_renyi,
    "scale_free": scale_free,
    "small_world": small_world,
    "hub": hub,
}


def named_graph(graph: str, d: int, density: float) -> GraphSpec:
    """Return the graph spec ``graph`` names, on ``d`` nodes at mean total degree ``density``.

    Raises ``ValueError`` for an unknown name, listing the known ones.
    """
    if graph not in _GRAPH_BUILDERS:
        raise ValueError(f"unknown graph {graph!r}; supported: {sorted(_GRAPH_BUILDERS)}")
    return _GRAPH_BUILDERS[graph](d, **params_for_density(graph, d, density))


def sample_scm(
    *,
    graph: str = "erdos_renyi",
    functional: str = "linear",
    noise: str = "gaussian",
    d: int,
    n: int,
    seed: int,
    density: float = 4.0,
    scale: str = "raw",
    latents: int = 0,
    dtype: npt.DTypeLike = np.float64,
    shuffle: bool = True,
) -> CausalDataset:
    """One-shot convenience: build an :class:`SCM` from string axes and sample it.

    Parameters
    ----------
    graph, functional, noise : str
        Axis selectors; see ``graphs`` / ``functional`` / ``noise`` for the available names.
    d, n : int
        Node count and sample count.
    seed : int
        Root seed.
    density : float, default=4.0
        Mean total degree (``2 * |E| / d``), for every ``graph``. Each builder rounds it to the
        nearest value it can draw.
    scale : {"raw", "standardize", "rescale"}, default="raw"
        How the columns are scaled; forwarded to :meth:`SCM.sample`. Temporal functionals take
        only ``"raw"``.
    latents : int, default=0
        How many of the ``d`` nodes to hide as latent confounders, chosen among nodes with at least
        two children: ``data`` drops them, and the true graph lists them last. Temporal
        functionals take only ``0``.
    dtype : npt.DTypeLike, default=np.float64
        Output dtype of the sampled ``data``; forwarded to :meth:`SCM.sample`.
    shuffle : bool, default=True
        Shuffle the column order with the true graph; forwarded to :meth:`SCM.sample`.

    Returns
    -------
    CausalDataset
    """
    functional_specs = {
        "linear": linear,
        "additive_noise": additive_noise,
        "post_nonlinear": post_nonlinear,
        "var": var,
        "varma": varma,
    }
    noise_specs = {
        "gaussian": gaussian,
        "uniform": uniform,
        "laplace": laplace,
        "exponential": exponential,
        "gumbel": gumbel,
    }
    graph_spec = named_graph(graph, d, density)
    if functional not in functional_specs:
        raise ValueError(
            f"unknown functional {functional!r}; supported: {sorted(functional_specs)}"
        )
    if noise not in noise_specs:
        raise ValueError(f"unknown noise {noise!r}; supported: {sorted(noise_specs)}")
    scm = SCM(
        graph=graph_spec,
        functional=functional_specs[functional](),
        noise=noise_specs[noise](),
        latents=latents,
    )
    return scm.sample(n=n, seed=seed, scale=scale, dtype=dtype, shuffle=shuffle)
