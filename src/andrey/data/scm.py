"""The structural causal model: compose a graph, functional forms, and noise; sample data.

An :class:`SCM` pairs a lazy graph spec with functional and noise specs. ``sample(n, seed)`` derives
three **independent, statelessly-keyed** RNG streams from the seed -- one each for the graph, the
weights, and the noise -- so a dataset is byte-reproducible from ``(root_entropy, spawn_key)`` alone
and a change to one axis cannot shift another's stream. Every draw takes an explicit
``np.random.Generator``; the module never touches ``np.random.*`` or ``ANDREY_SEED``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

import andrey

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
from .latent import with_latents
from .noise import Noise, exponential, gaussian, gumbel, laplace, uniform
from .qa import FaithfulnessScreen
from .sortability import standardize

_SCALES = ("raw", "standardize", "rescale")
_TEMPORAL_BURN_IN = 200  # discarded prefix so the returned window is stationary

# Per-component sub-keys off the root SeedSequence: the graph, the weights, the noise, the latent
# selection, and the temporal lag matrices each get an independent stream so refactoring one axis
# never perturbs another's bytes.
_GRAPH, _WEIGHTS, _NOISE, _LATENT, _LAG = 0, 1, 2, 3, 4
_SEED_SCHEME = 1  # bump if the seed-derivation layout ever changes (recorded in provenance)


def _component_rng(root: np.random.SeedSequence, component: int) -> np.random.Generator:
    """Derive an independent ``Generator`` for one pipeline component, statelessly from ``root``.

    Uses ``spawn_key`` extension rather than ``root.spawn()`` (which mutates a counter), so the
    stream for task ``i`` component ``c`` is a pure function of ``(root.entropy, (*root.spawn_key,
    c))`` -- reproducible regardless of submission order or worker count.
    """
    seq = np.random.SeedSequence(entropy=root.entropy, spawn_key=(*root.spawn_key, component))
    return np.random.default_rng(seq)


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
            this only controls the final cast. Every consumer upcasts to ``float64`` anyway, so the
            default avoids a lossy round-trip; pass ``np.float32`` to opt into a
            half-footprint array.

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
            return self._sample_once(n, root, scale=scale, dtype=dtype)
        for attempt in range(screen.max_attempts):
            attempt_root = np.random.SeedSequence(
                entropy=root.entropy, spawn_key=(*root.spawn_key, attempt)
            )
            ds = self._sample_once(n, attempt_root, scale=scale, dtype=dtype)
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
        self, n: int, root: np.random.SeedSequence, *, scale: str, dtype: npt.DTypeLike
    ) -> CausalDataset:
        truth, draw = self.graph.materialize(_component_rng(root, _GRAPH))
        weights = self.functional.draw_weights(draw, _component_rng(root, _WEIGHTS))
        noise_rng = _component_rng(root, _NOISE)
        scales = self.noise.draw_scales(draw.d, noise_rng)
        if self.functional.is_temporal:
            return self._sample_temporal(
                n, draw, truth, weights, scales, noise_rng, root, scale, dtype
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
        edges = np.stack([draw.parents, draw.children], axis=1)
        if self.latents > 0:
            data, edges, node_types = with_latents(
                data, edges, draw.d, self.latents, _component_rng(root, _LATENT)
            )
            truth = dag_truth(edges[:, 0], edges[:, 1], draw.d, node_types=node_types)
        n_obs = data.shape[1]
        params = SCMParams(
            edges=edges,
            weights=weights,
            noise_scales=scales,
            functional=self.functional.name,
            noise=self.noise.name,
        )
        provenance = {
            "root_entropy": str(root.entropy),
            "spawn_key": list(root.spawn_key),
            "seed_scheme": _SEED_SCHEME,
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
            },
            "versions": {"andrey": andrey.__version__, "numpy": np.__version__},
        }
        # Sortability is a property of the observed data over observed-observed edges.
        obs_mask = (edges[:, 0] < n_obs) & (edges[:, 1] < n_obs)
        report = qa.report(data, edges[obs_mask], n_obs, scale=scale)
        return CausalDataset(
            data=data, graph=truth, report=report, provenance=provenance, params=params
        )

    def _sample_temporal(self, n, draw, truth, weights, scales, noise_rng, root, scale, dtype):
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
        edges = np.stack(
            [draw.parents, draw.children], axis=1
        )  # the instantaneous (lag-0) structure
        params = SCMParams(
            edges=edges,
            weights=weights,
            noise_scales=scales,
            functional=self.functional.name,
            noise=self.noise.name,
        )
        provenance = {
            "root_entropy": str(root.entropy),
            "spawn_key": list(root.spawn_key),
            "seed_scheme": _SEED_SCHEME,
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
            },
            "versions": {"andrey": andrey.__version__, "numpy": np.__version__},
        }
        # Sortability targets iid data; it does not apply to a time series.
        report = QAReport(scale=scale)
        return CausalDataset(
            data=data, graph=temporal_truth, report=report, provenance=provenance, params=params
        )


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
    dtype : npt.DTypeLike, default=np.float64
        Output dtype of the sampled ``data``; forwarded to :meth:`SCM.sample`.

    Returns
    -------
    CausalDataset
    """
    graph_specs = {
        "erdos_renyi": lambda: erdos_renyi(d, **params_for_density("erdos_renyi", d, density)),
        "scale_free": lambda: scale_free(d, **params_for_density("scale_free", d, density)),
        "small_world": lambda: small_world(d, **params_for_density("small_world", d, density)),
        "hub": lambda: hub(d, **params_for_density("hub", d, density)),
    }
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
    if graph not in graph_specs:
        raise ValueError(f"unknown graph {graph!r}; supported: {sorted(graph_specs)}")
    if functional not in functional_specs:
        raise ValueError(
            f"unknown functional {functional!r}; supported: {sorted(functional_specs)}"
        )
    if noise not in noise_specs:
        raise ValueError(f"unknown noise {noise!r}; supported: {sorted(noise_specs)}")
    scm = SCM(
        graph=graph_specs[graph](),
        functional=functional_specs[functional](),
        noise=noise_specs[noise](),
        latents=latents,
    )
    return scm.sample(n=n, seed=seed, scale=scale, dtype=dtype)
