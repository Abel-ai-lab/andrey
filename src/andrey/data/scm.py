"""The structural causal model: compose a graph, functional forms, and noise; sample data.

An :class:`SCM` pairs a lazy graph spec with functional and noise specs. ``sample(n, seed)`` derives
**independent, statelessly-keyed** RNG streams from the seed -- one each for the graph, the
weights, the noise, and the column shuffle -- so a dataset is byte-reproducible from
``(root_entropy, spawn_key)`` alone and a change to one axis cannot shift another's stream. Every
draw takes an explicit ``np.random.Generator``; the module never touches ``np.random.*`` or
``ANDREY_SEED``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

import andrey
from andrey.core.structure import LATENT, OBSERVED

from . import qa
from .dataset import CausalDataset, QAReport, SCMParams
from .functional import (
    Functional,
    Linear,
    additive_noise,
    linear,
    location_scale,
    nonmonotone,
    post_nonlinear,
    var,
    varma,
)
from .graphs import (
    DAGDraw,
    GraphSpec,
    dag_truth,
    erdos_renyi,
    hub,
    params_for_density,
    scale_free,
    small_world,
)
from .interventions import (
    Environment,
    HardIntervention,
    Intervention,
    as_interventions,
    check_columns,
    check_continuous_soft,
    continuous_hard_values,
    env_labels,
    env_targets,
    pooled_rows,
    record,
)
from .latent import latent_order
from .noise import Noise, exponential, gaussian, gumbel, laplace, uniform
from .qa import FaithfulnessScreen
from .sortability import standardize
from .temporal import relabel_truth

__all__ = ["SCM", "sample_scm"]

_SCALES = ("raw", "standardize", "rescale", "unitless")
_TEMPORAL_SCALES = ("raw", "standardize")
_TEMPORAL_BURN_IN = 200  # discarded prefix so the returned window is stationary

# Per-component sub-keys off the root SeedSequence: the graph, the weights, the noise, the latent
# selection, the temporal lag matrices, the column shuffle, and the intervention draws each get an
# independent stream, so a change to one component's draws never shifts another's. Environment
# ``k >= 1`` draws its noise from ``(NOISE, k)`` and environment ``k`` its intervention values from
# ``(INTERVENTION, k)`` and its redrawn weights from ``(WEIGHTS, k)``; environment 0 draws from
# ``NOISE``, as data without interventions do. The model's streams (graph, weights, noise scales,
# latents, lags, columns) come off the model seed's root, the rows' streams off the seed's root;
# without a model seed the two are one root.
GRAPH, WEIGHTS, NOISE, _LATENT, _LAG, _COLUMNS, INTERVENTION = 0, 1, 2, 3, 4, 5, 6
_SEED_SCHEME = 1  # bump if the seed-derivation layout ever changes (recorded in provenance)


def component_rng(root: np.random.SeedSequence, *component: int) -> np.random.Generator:
    """Derive an independent ``Generator`` for one pipeline component, statelessly from ``root``.

    Uses ``spawn_key`` extension rather than ``root.spawn()`` (which mutates a counter), so the
    stream for task ``i`` component ``c`` is a pure function of ``(root.entropy, (*root.spawn_key,
    c))`` -- reproducible regardless of submission order or worker count. A component of several
    keys, such as ``(NOISE, k)``, names a stream of its own.
    """
    seq = np.random.SeedSequence(entropy=root.entropy, spawn_key=(*root.spawn_key, *component))
    return np.random.default_rng(seq)


def column_order(root: np.random.SeedSequence, n_observed: int, shuffle: bool) -> np.ndarray:
    """Return the drawn index of each output column; the identity unless ``shuffle``.

    The permutation comes from the column stream of ``root``, so no other draw depends on it.
    """
    if not shuffle:
        return np.arange(n_observed)
    return component_rng(root, _COLUMNS).permutation(n_observed)


def dataset_provenance(
    root: np.random.SeedSequence,
    model: np.random.SeedSequence,
    columns: np.ndarray,
    graph: GraphSpec,
    d: int,
    config: dict,
) -> dict:
    """Return the provenance of a sampled dataset: its seeds, column order, settings, and versions.

    ``columns`` is the drawn index of each output column, ``graph`` the spec the ``d``-node graph
    was drawn from, and ``config`` the sampler's own settings, listed after the graph's. The model's
    seed is recorded when it is not the root's.
    """
    return {
        "root_entropy": str(root.entropy),
        "spawn_key": list(root.spawn_key),
        **({} if model is root else {"model_entropy": str(model.entropy)}),
        "seed_scheme": _SEED_SCHEME,
        "column_order": columns.tolist(),
        "config": {
            "graph": {
                "model": graph.name,
                "num_nodes": d,
                "density": graph.density,
                "params": graph.params,
            },
            **config,
        },
        "versions": {"andrey": andrey.__version__, "numpy": np.__version__},
    }


def _options(functional: Functional, noise: Noise) -> dict:
    """Return the functional and noise options the provenance names beside the two names.

    Each option appears only when it is set, so a record of a model without them is unchanged.
    """
    return {
        **({"positive": True} if functional.positive else {}),
        **({"heteroscedastic": True} if noise.heteroscedastic else {}),
        **({} if noise.variance_range is None else {"variance_range": list(noise.variance_range)}),
    }


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
    latents : int, default=0
        How many of the ``d`` nodes to hide as latent confounders, chosen among nodes with at least
        two children: ``data`` drops them, and the true graph lists them last. Temporal
        functionals take only ``0``.
    interventions : tuple of (int, HardIntervention or SoftIntervention) pairs, default=()
        The interventions :meth:`do` applied, by column, in column order.

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
    interventions: tuple[tuple[int, Intervention], ...] = ()

    def do(self, interventions: Mapping[int, Intervention | float]) -> SCM:
        """Return the model under interventions; its :meth:`sample` draws from it.

        Parameters
        ----------
        interventions : mapping of int to HardIntervention, SoftIntervention, or float
            Each intervened column of the data :meth:`sample` returns, after the column shuffle,
            with its intervention: :func:`~andrey.data.interventions.hard`, which cuts the variable
            from its parents, :func:`~andrey.data.interventions.soft`, which changes its mechanism
            and keeps them, or a number, which stands for ``hard(number)``. A column this model
            already intervenes on takes the new intervention.

        Returns
        -------
        SCM
            The same graph, functional, and noise under the interventions. Its ``sample(n, seed)``
            draws the graph, the weights, and the noise as this model's does, so the two samples
            are equal in every column the interventions do not reach, and ``targets`` is
            ``{0: [the intervened columns]}``. The true graph is the graph without interventions.

        Raises
        ------
        TypeError
            If a column is not a whole number or an intervention is not one of the three forms.
        ValueError
            If a column is negative.

        Examples
        --------
        >>> import andrey.data as data
        >>> from andrey.data.interventions import hard
        >>> scm = data.SCM(graph=data.graphs.erdos_renyi(d=8, avg_degree=2),
        ...                functional=data.functional.linear(),
        ...                noise=data.noise.gaussian())
        >>> ds = scm.do({3: hard(2.0)}).sample(n=100, seed=0)
        >>> bool((ds.data[:, 3] == 2.0).all()), ds.targets
        (True, {0: [3]})
        """
        merged = {**dict(self.interventions), **as_interventions(interventions)}
        return replace(self, interventions=tuple(sorted(merged.items())))

    def sample(
        self,
        n: int | Sequence[int],
        seed: int,
        *,
        model_seed: int | None = None,
        envs: Sequence[Mapping[int, Intervention | float]] | None = None,
        scale: str = "standardize",
        screen: FaithfulnessScreen | None = None,
        dtype: npt.DTypeLike = np.float64,
        shuffle: bool = True,
    ) -> CausalDataset:
        """Draw ``n`` rows; return a :class:`~andrey.data.dataset.CausalDataset`.

        Parameters
        ----------
        n : int or sequence of int
            Number of samples (rows). With ``envs``, the rows of each environment: ``n`` each, or
            ``n[k]`` for environment ``k``.
        seed : int
            Root seed of the rows, and of the model when ``model_seed`` is None; the streams it
            seeds are derived from it statelessly.
        model_seed : int or None, default=None
            The seed of the model: the graph, the weights, the noise scales, the latent variables,
            the column order, the lag matrices of a temporal functional, and the weights
            ``soft(weights="redraw")`` draws. ``seed`` then draws only the rows: the noise and the
            values interventions draw for each row. Samples with one ``model_seed`` and different
            seeds are datasets of one model, with the same true graph, ``params``, and column
            order. ``None`` takes ``seed``, as ``model_seed=seed`` does.
        envs : sequence of mappings of int to intervention or float, or None, default=None
            One mapping per environment, ``{column: intervention}`` as for :meth:`do`; ``{}`` is
            an environment without interventions, and the first environment is the observational
            one by convention. Every environment shares the graph and the weights, and draws its
            own noise; environment 0 draws the noise a sample without ``envs`` draws. The rows
            come pooled, environment 0 first, with ``env`` and ``targets`` set. ``None`` draws
            from this model: observational data, or, after :meth:`do`, data under its
            interventions.
        scale : {"standardize", "unitless", "rescale", "raw"}, default="standardize"
            How the generated values are scaled; no mode changes the graph. In raw linear data a
            variable's variance grows down the causal order, so sorting the columns by variance
            recovers much of that order (var-sortability, Reisach et al. 2021). The default
            removes it for every functional at any size, so no method gains from the units the
            generator happens to produce.

            - ``"standardize"`` z-scores each column after generation: every column has mean 0
              and variance 1, and var-sortability is exactly 0.5. R^2-sortability, the share of a
              variable's variance the other variables explain, is the same under any rescaling of
              the columns, so it stays as high as in the raw data (Reisach et al. 2023).
            - ``"unitless"`` draws the weights and noise scales of the linear model so that every
              node has variance 1 in the population, with each node's parameters depending only on
              its number of parents and no bound on the ratio of a parent's share of the variance
              to the noise's (Algorithm 1 of Herman et al. 2025;
              :meth:`~andrey.data.functional.Linear.draw_unitless`). Var-sortability and
              R^2-sortability are near 0.5, R^2-sortability a little below it on denser graphs,
              and the sample variances differ from 1 by sampling error. It takes the linear
              functional without heteroscedastic noise, does not use the functional's
              ``weight_range``, and keeps a dense ``(d, d)`` correlation matrix, so it takes at
              most 1,000 nodes (``andrey.data.functional.UNITLESS_MAX_NODES``).
            - ``"rescale"`` divides each node by its sample standard deviation as it is generated,
              before its children use it, so variance does not accumulate down the causal order:
              var-sortability is exactly 0.5 and R^2-sortability near 0.5 on sparse graphs. The
              model then depends on the sample, and the weights keep their ``weight_range``
              relative to the noise.
            - ``"raw"`` returns the generated values as they are.

            Temporal functionals take ``"standardize"`` and ``"raw"``. Under interventions, every
            mode scales the data as it would without them: ``"standardize"`` takes each column's
            mean and standard deviation, and ``"rescale"`` each node's divisor, from the model
            without interventions on the same noise, and applies them to every row.
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
            Under interventions, ``env`` holds each row's environment and ``targets`` each
            environment's intervened columns; the true graph is the graph without interventions.

        Raises
        ------
        ValueError
            If ``scale`` is not one of the modes, a temporal functional is given a mode other than
            ``"standardize"`` or ``"raw"``, ``"unitless"`` is given a functional other than linear,
            heteroscedastic noise, or more than 1,000 nodes; or, under interventions, if the
            functional is temporal, ``screen`` is given, this model is under :meth:`do` and
            ``envs`` is given, ``n`` does not give each environment at least one row, a target is
            not an observed column, ``hard()`` on a variable has no ``value``, or ``soft()``
            changes nothing. Also if ``"unitless"`` is given a positive-weight functional, noise
            with a ``variance_range``, or ``soft(weights="redraw")``, or ``screen`` is given with
            ``model_seed``.

        References
        ----------
        Reisach, A. G., Seiler, C., and Weichwald, S. (2021). Beware of the Simulated DAG! Causal
        Discovery Benchmarks May Be Easy to Game. Advances in Neural Information Processing
        Systems 34.

        Reisach, A. G., Tami, M., Seiler, C., Chambaz, A., and Weichwald, S. (2023). A
        Scale-Invariant Sorting Criterion to Find a Causal Order in Additive Noise Models.
        Advances in Neural Information Processing Systems 36.

        Herman, R. J., Wahl, J., Ninad, U., and Runge, J. (2025). Unitless Unrestricted
        Markov-Consistent SCM Generation: Better Benchmark Datasets for Causal Discovery.
        Proceedings of the Fourth Conference on Causal Learning and Reasoning, PMLR 275,
        1506-1531.
        """
        if scale not in _SCALES:
            raise ValueError(f"scale must be one of {_SCALES}, got {scale!r}")
        if self.functional.is_temporal and scale not in _TEMPORAL_SCALES:
            raise ValueError(
                f"temporal functionals take scale in {_TEMPORAL_SCALES}, got {scale!r}"
            )
        if scale == "unitless" and not isinstance(self.functional, Linear):
            raise ValueError(
                f"scale='unitless' takes the linear functional, got {self.functional.name!r}"
            )
        if scale == "unitless" and (
            self.noise.heteroscedastic or self.noise.variance_range is not None
        ):
            raise ValueError(
                "scale='unitless' draws the noise scales itself; use noise without "
                "heteroscedastic=True or a variance_range"
            )
        if scale == "unitless" and self.functional.positive:
            raise ValueError(
                "scale='unitless' draws the weights itself; use linear() without positive=True"
            )
        if screen is not None and model_seed is not None:
            raise ValueError("screen redraws the model, which model_seed fixes; give one of them")
        if not isinstance(self.graph, GraphSpec):
            raise NotImplementedError(
                "SCM.sample currently supports a GraphSpec; concrete-graph input is not supported"
            )
        if self.interventions and envs is not None:
            raise ValueError(
                "a model under do(...) takes no envs; give each environment its interventions"
            )
        if self.interventions:
            envs = [dict(self.interventions)]
        if envs is not None and self.functional.is_temporal:
            raise ValueError("temporal functionals take no interventions")
        if envs is not None and screen is not None:
            raise ValueError("screen is not supported under interventions")
        rows, pooled = pooled_rows(envs, n)
        root = np.random.SeedSequence(seed)
        if screen is None:
            model = root if model_seed is None else np.random.SeedSequence(model_seed)
            return self._sample_once(
                rows, root, model, scale=scale, dtype=dtype, shuffle=shuffle, envs=pooled
            )
        for attempt in range(screen.max_attempts):
            attempt_root = np.random.SeedSequence(
                entropy=root.entropy, spawn_key=(*root.spawn_key, attempt)
            )
            ds = self._sample_once(
                rows, attempt_root, attempt_root, scale=scale, dtype=dtype, shuffle=shuffle
            )
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
        model: np.random.SeedSequence,
        *,
        scale: str,
        dtype: npt.DTypeLike,
        shuffle: bool,
        envs: list[Environment] | None = None,
    ) -> CausalDataset:
        """Draw one dataset: the model from the streams of ``model``, the rows from ``root``'s."""
        _, draw = self.graph.materialize(component_rng(model, GRAPH))
        noise_rng = component_rng(root, NOISE)
        scale_weights = np.empty(0)
        if scale == "unitless":
            weights, scales = self.functional.draw_unitless(  # ty: ignore[unresolved-attribute]  # checked in sample: Linear only
                draw, component_rng(model, WEIGHTS)
            )
        else:
            weight_rng = component_rng(model, WEIGHTS)
            weights = self.functional.draw_weights(draw, weight_rng)
            # The scale weights follow the location weights in their stream, so they leave the
            # location weights and every other stream as they are.
            scale_weights = self.functional.draw_scale_weights(draw, weight_rng)
            scales = self.noise.draw_scales(draw.d, noise_rng)
            if model is not root:  # the rows' stream moves past its own scales, as without a model
                scales = self.noise.draw_scales(draw.d, component_rng(model, NOISE))
        if self.functional.is_temporal:
            return self._sample_temporal(
                n, draw, weights, scales, noise_rng, root, model, scale, dtype, shuffle
            )
        # ``order[j]`` is the drawn label of output node ``j``: the observed nodes in column order,
        # then the latents. Edge rows keep their order, so ``weights`` stays aligned with ``edges``.
        edges = np.stack([draw.parents, draw.children], axis=1)
        n_obs = draw.d - self.latents
        order = np.arange(draw.d)
        node_types = None
        if self.latents > 0:
            order = latent_order(edges, draw.d, self.latents, component_rng(model, _LATENT))
            node_types = np.full(draw.d, OBSERVED, dtype=np.int8)
            node_types[n_obs:] = LATENT
        columns = column_order(model, n_obs, shuffle)
        order[:n_obs] = order[columns]
        if envs is None:
            innovations = self.noise.draw(n, scales, noise_rng)
            values = self.functional.evaluate(
                draw,
                weights,
                innovations,
                rescale=(scale == "rescale"),
                scale_weights=scale_weights,
            )
            if scale == "standardize":
                values = standardize(values)
        else:
            check_columns(envs, n_obs)
            values = self._evaluate_envs(
                draw, weights, scale_weights, scales, noise_rng, root, model, envs, scale, order
            )
        data = values.astype(dtype)
        if not np.all(np.isfinite(data)):  # deep weighted paths can overflow float32
            raise OverflowError(
                "generated data has non-finite values (variance grew past float32 down a "
                "deep causal path); reduce the weight range / density, or use "
                "scale='rescale' to bound it"
            )
        data = np.take(data, order[:n_obs], axis=1)  # C order; data[:, order] is Fortran order
        edges = np.argsort(order)[edges]
        truth = dag_truth(edges[:, 0], edges[:, 1], draw.d, node_types=node_types)
        params = SCMParams(
            edges=edges,
            weights=weights,
            scale_weights=scale_weights,
            noise_scales=scales[order],
            functional=self.functional.name,
            noise=self.noise.name,
        )
        provenance = dataset_provenance(
            root,
            model,
            columns,
            self.graph,
            draw.d,
            {
                "functional": self.functional.name,
                "noise": self.noise.name,
                **_options(self.functional, self.noise),
                "n": int(n),
                "scale": scale,
                "dtype": np.dtype(dtype).name,
                "latents": int(self.latents),
                "shuffle": bool(shuffle),
                **({} if envs is None else {"envs": record(envs)}),
            },
        )
        # Sortability is a property of the observed data over observed-observed edges.
        obs_mask = (edges[:, 0] < n_obs) & (edges[:, 1] < n_obs)
        report = qa.report(data, edges[obs_mask], n_obs, scale=scale)
        return CausalDataset(
            data=data,
            graph=truth,
            report=report,
            provenance=provenance,
            params=params,
            env=None if envs is None else env_labels(envs),
            targets=None if envs is None else env_targets(envs),
        )

    def _evaluate_envs(
        self,
        draw: DAGDraw,
        weights: np.ndarray,
        scale_weights: np.ndarray,
        scales: np.ndarray,
        noise_rng: np.random.Generator,
        root: np.random.SeedSequence,
        model: np.random.SeedSequence,
        envs: list[Environment],
        scale: str,
        order: np.ndarray,
    ) -> np.ndarray:
        """Evaluate each environment on its own noise; return the pooled rows, scaled.

        The reference is the model without interventions on every row's noise. Its column means
        and standard deviations (``"standardize"``) and its divisors (``"rescale"``) scale every
        environment, and a ``hard`` value or a ``soft`` shift, given in the units of the returned
        data, is converted to the model's units with them. ``order[j]`` is the node of column ``j``.
        Redrawn weights come from ``model``'s streams, every other intervention draw from
        ``root``'s.
        """
        noise = np.concatenate(
            [
                self.noise.draw(
                    size, scales, noise_rng if k == 0 else component_rng(root, NOISE, k)
                )
                for k, (_, size) in enumerate(envs)
            ]
        )
        rescale = scale == "rescale"
        values, divisors = self.functional.propagate(
            draw, weights, noise, rescale=rescale, scale_weights=scale_weights
        )
        loc, unit = np.zeros(draw.d), np.ones(draw.d)
        if scale == "standardize":
            loc, unit = values.mean(axis=0), values.std(axis=0)
            unit = np.where(unit == 0, 1.0, unit)
        set_values = []  # (rows, node, values in the units of the returned data)
        end = 0
        for k, (spec, size) in enumerate(envs):
            rows, end = slice(end, end + size), end + size
            if not spec:
                continue  # its rows are the reference's
            env_weights, env_scale_weights = weights.copy(), scale_weights.copy()
            env_noise, fixed = noise[rows].copy(), {}
            draws = component_rng(root, INTERVENTION, k)
            redraws = component_rng(model, WEIGHTS, k)
            for column, intervention in spec.items():
                c = int(order[column])
                if isinstance(intervention, HardIntervention):
                    target = continuous_hard_values(intervention, column, size, draws)
                    set_values.append((rows, c, target))
                    fixed[c] = loc[c] + unit[c] * target
                    continue
                check_continuous_soft(intervention, column)
                into = draw.children == c
                if intervention.weights == "redraw":
                    if scale == "unitless":
                        raise ValueError(
                            f"column {column}: soft(weights='redraw') draws from the functional's "
                            "weight_range, which scale='unitless' does not use"
                        )
                    env_weights[into] = self.functional.draw_weights(draw, redraws)[into]
                    if env_scale_weights.size:
                        env_scale_weights[into] = self.functional.draw_scale_weights(draw, redraws)[
                            into
                        ]
                env_weights[into] *= intervention.weight_scale
                if env_scale_weights.size:
                    env_scale_weights[into] *= intervention.weight_scale
                noise_scale = intervention.noise_scale
                if isinstance(noise_scale, tuple):  # one factor per row
                    noise_scale = draws.uniform(*noise_scale, size=size)
                env_noise[:, c] = (
                    env_noise[:, c] * noise_scale + intervention.shift * unit[c] * divisors[c]
                )
            values[rows] = self.functional.propagate(
                draw,
                env_weights,
                env_noise,
                rescale=rescale,
                scale_weights=env_scale_weights,
                divisors=divisors,
                fixed=fixed,
            )[0]
        if scale == "standardize":
            values = (values - loc) / unit
        for rows, c, target in set_values:  # exact, free of the rounding of the scaling
            values[rows, c] = target
        return values

    def _sample_temporal(
        self, n, draw, weights, scales, noise_rng, root, model, scale, dtype, shuffle
    ):
        """Generate a VAR/VARMA series; ``n`` is the number of time steps, truth temporal."""
        if self.latents > 0:
            raise NotImplementedError(
                "latent confounders are not supported for temporal functionals"
            )
        innovations = self.noise.draw(n + _TEMPORAL_BURN_IN, scales, noise_rng)
        series, temporal_truth = self.functional.generate_temporal(  # ty: ignore[unresolved-attribute]  # gated by is_temporal (VAR/VARMA only)
            draw, weights, innovations, component_rng(model, _LAG), burn_in=_TEMPORAL_BURN_IN
        )
        if scale == "standardize":
            series = standardize(series)
        data = series.astype(dtype)
        if not np.all(np.isfinite(data)):
            raise OverflowError(
                "temporal generation produced non-finite values (unstable process); "
                "reduce lag_density or the weight range"
            )
        # The shuffle relabels the lag-0 DAG and every lag block on both axes.
        order = column_order(model, draw.d, shuffle)
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
        provenance = dataset_provenance(
            root,
            model,
            order,
            self.graph,
            draw.d,
            {
                "functional": self.functional.name,
                "noise": self.noise.name,
                **_options(self.functional, self.noise),
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
        )
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
    scale: str = "standardize",
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
    scale : {"standardize", "unitless", "rescale", "raw"}, default="standardize"
        How the generated values are scaled; forwarded to :meth:`SCM.sample`, which describes each
        mode. ``"standardize"`` z-scores the columns; ``"unitless"`` (linear functional, at most
        1,000 nodes) gives every node variance 1 by its draw of weights and noise scales, with
        R^2-sortability near 0.5; ``"rescale"`` divides each node by its sample standard deviation
        as it is generated; ``"raw"`` keeps the generated values. Temporal functionals take
        ``"standardize"`` and ``"raw"``.
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
        "nonmonotone": nonmonotone,
        "location_scale": location_scale,
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
