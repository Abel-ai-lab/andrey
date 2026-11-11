"""Functional forms for the structural equations (the ``f`` in ``X_j = f(pa_j, noise_j)``).

A functional spec draws per-edge parameters and then evaluates the SCM in topological order over the
sparse edge list -- no dense adjacency. ``linear`` is the causally-sufficient default;
``additive_noise`` (ANM) and ``post_nonlinear`` (PNL) are the nonlinear identifiable families,
``nonmonotone`` an additive-noise family whose effects partial correlation does not see, and
``location_scale`` (LSNM) a family whose noise scale is a function of the parents, and
``random_features`` an additive-noise family whose functions are approximate Gaussian-process
draws. All six share one topological-evaluation loop and differ only in the per-node combine rule.
:meth:`Linear.draw_unitless` draws the weights and noise scales of ``scale="unitless"``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from scipy.special import expit

from .graphs import DAGDraw

#: The most nodes :meth:`Linear.draw_unitless` takes. It keeps the dense ``(d, d)`` correlation
#: matrix of the nodes, 8 MB in float64 at this size, and its time grows with ``d`` times the edges.
UNITLESS_MAX_NODES = 1000

#: The noise scales of the location-scale benchmark in Immer et al. (2023), Supplementary Material
#: E: ``0.1 (4 pi - |x|) + 0.2`` over ``x ~ U[-4 pi, 4 pi]``, from 0.2 at the borders to
#: ``0.2 + 0.4 pi`` at the center.
LOCATION_SCALE_RANGE = (0.2, 0.2 + 0.4 * np.pi)


@dataclass(frozen=True, kw_only=True)
class Functional:
    """A lazy structural-equation specification (shared topological evaluation)."""

    name: str
    weight_range: tuple[float, float] = (0.5, 2.0)
    positive: bool = False
    is_temporal: ClassVar[bool] = False  # temporal families set this and provide generate_temporal

    def draw_weights(self, draw: DAGDraw, rng: np.random.Generator) -> np.ndarray:
        """Per-edge weights, magnitude ``U[weight_range]`` with a random sign (``(m,)``).

        Under ``positive``, each weight is the magnitude alone: the signs are drawn and dropped,
        so the weights are those of the signed draw on the same stream, made positive.
        """
        m = draw.parents.shape[0]
        lo, hi = self.weight_range
        weights = rng.uniform(lo, hi, size=m) * rng.choice(np.array([-1.0, 1.0]), size=m)
        return np.abs(weights) if self.positive else weights

    def draw_scale_weights(self, draw: DAGDraw, rng: np.random.Generator) -> np.ndarray:
        """Per-edge weights of the noise scale; empty, with no draws, unless a subclass has one."""
        return np.empty(0)

    @property
    def options(self) -> dict:
        """The family's own settings, as a dataset's provenance records them; empty without."""
        return {}

    def scale_effect(self, weights: np.ndarray, factor: float) -> np.ndarray:
        """Return ``weights`` with ``factor`` applied as ``soft(weight_scale=factor)`` applies it.

        Every weight is multiplied by ``factor`` unless a subclass scales its own part.
        """
        return weights * factor

    def _node(
        self, parent_values: np.ndarray, weights: np.ndarray, noise_c: np.ndarray
    ) -> np.ndarray:
        """Combine a child's parent values, its edge weights, and its noise into its column."""
        raise NotImplementedError

    def evaluate(
        self,
        draw: DAGDraw,
        weights: np.ndarray,
        noise: np.ndarray,
        *,
        rescale: bool = False,
        scale_weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """Evaluate ``X`` (shape ``(n, d)``, node-label columns) in topological order.

        Source nodes (no parents) equal their exogenous noise; every other node is combined by
        :meth:`_node` once all its parents are finalized (guaranteed by the topological order).

        Parameters
        ----------
        rescale : bool, default=False
            If ``True``, standardize each node's column to unit variance immediately after it is
            computed (before its children use it) -- ancestral rescaling. This stops variance from
            accumulating down the causal order, so varsortability sits at ~0.5 by construction; the
            recovered *structure* is unchanged (a per-column scaling of a DAG).
        scale_weights : np.ndarray of shape (m,) or None, default=None
            The per-edge weights of the noise scale, from :meth:`draw_scale_weights`. When not
            empty, :meth:`_node` receives each child's ``(k, 2)`` block of location and scale
            weights.
        """
        return self.propagate(draw, weights, noise, rescale=rescale, scale_weights=scale_weights)[0]

    def propagate(
        self,
        draw: DAGDraw,
        weights: np.ndarray,
        noise: np.ndarray,
        *,
        rescale: bool = False,
        scale_weights: np.ndarray | None = None,
        divisors: np.ndarray | None = None,
        fixed: Mapping[int, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate ``X`` as :meth:`evaluate` does, with set nodes and given rescale divisors.

        Parameters
        ----------
        divisors : np.ndarray of shape (d,) or None, default=None
            Under ``rescale``, each node is divided by its entry here rather than by its own sample
            standard deviation, so a model under intervention keeps the divisors of the model
            without one.
        fixed : mapping of int to np.ndarray of shape (n,), or None, default=None
            Nodes set from outside: each listed node takes its column as given, its parents unread
            and no divisor applied.

        Returns
        -------
        x : np.ndarray of shape (n, d)
            The values.
        divisors : np.ndarray of shape (d,)
            The divisor of each node; ``1`` for every node without ``rescale``, and for a node set
            by ``fixed`` or with a sample standard deviation of ``0``.
        """
        x = noise.astype(np.float64, copy=True)
        fixed = fixed or {}
        used = np.ones(noise.shape[1])
        if scale_weights is not None and scale_weights.size:
            weights = np.column_stack([weights, scale_weights])
        child_block = _parents_by_child(draw.children, draw.parents, weights)
        for c in draw.topo_order:
            c = int(c)
            if c in fixed:
                x[:, c] = fixed[c]
                continue
            block = child_block.get(c)
            if block is not None:
                pa, w = block
                x[:, c] = self._node(x[:, pa], w, noise[:, c])
            if rescale:
                sd = x[:, c].std() if divisors is None else divisors[c]
                if sd > 0:
                    x[:, c] /= sd
                    used[c] = sd
        return x, used


@dataclass(frozen=True, kw_only=True)
class Linear(Functional):
    """Linear structural equations ``X_c = sum_p w_{p,c} X_p + noise_c``."""

    name: str = "linear"

    def _node(self, parent_values, weights, noise_c):
        return noise_c + parent_values @ weights

    def draw_unitless(
        self, draw: DAGDraw, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Draw edge weights and noise scales that give every node unit variance.

        Algorithm 1 of Herman et al. (2025). In topological order, a node with ``k`` parents
        draws a uniform point of the unit ``k``-ball: a direction ``a ~ N(0, I_k)`` and a radius
        ``r = u ** (1 / k)`` with ``u ~ U(0, 1)``. Its weights start as ``r * a / ||a||`` and its
        noise scale as ``sqrt(1 - r**2)``; both are then divided by the node's standard deviation
        under the correlation matrix of its parents, so the node has variance 1 however its
        parents are correlated. The draw of a node depends only on its number of parents. A node
        without parents has noise scale 1. ``weight_range`` is not used.

        Parameters
        ----------
        draw : DAGDraw
            The drawn graph.
        rng : np.random.Generator
            The stream of the draws: one normal per edge, then one uniform per node.

        Returns
        -------
        weights : np.ndarray of shape (m,)
            The weight of each edge, in the order of ``draw.parents`` and ``draw.children``.
        scales : np.ndarray of shape (d,)
            The noise standard deviation of each node.

        Raises
        ------
        ValueError
            If the graph has more than :data:`UNITLESS_MAX_NODES` nodes.

        References
        ----------
        Herman, R. J., Wahl, J., Ninad, U., and Runge, J. (2025). Unitless Unrestricted
        Markov-Consistent SCM Generation: Better Benchmark Datasets for Causal Discovery.
        Proceedings of the Fourth Conference on Causal Learning and Reasoning, PMLR 275,
        1506-1531.
        """
        d = draw.d
        if d > UNITLESS_MAX_NODES:
            raise ValueError(
                f"scale='unitless' keeps a dense (d, d) correlation matrix and takes at most "
                f"{UNITLESS_MAX_NODES} nodes, got {d}"
            )
        m = draw.parents.shape[0]
        directions = rng.standard_normal(m)
        radii = rng.uniform(size=d)
        weights = np.zeros(m)
        scales = np.ones(d)
        corr = np.eye(d)
        edges_by_child = _parents_by_child(draw.children, draw.parents, np.arange(m))
        for c in draw.topo_order:
            c = int(c)
            block = edges_by_child.get(c)
            if block is None:
                continue
            pa, edge = block
            r = radii[c] ** (1.0 / pa.size)
            a = r * directions[edge] / np.linalg.norm(directions[edge])
            s = np.sqrt(1.0 - r * r)
            sd = np.sqrt(a @ corr[np.ix_(pa, pa)] @ a + s * s)
            weights[edge] = a / sd
            scales[c] = s / sd
            # Cov(X_c, X_j) = sum_p w_p Cov(X_p, X_j) for every node j before c. The entries of the
            # nodes after c are still zero here and are set when those nodes are drawn.
            row = weights[edge] @ corr[pa, :]
            corr[c, :] = row
            corr[:, c] = row
            corr[c, c] = 1.0
        return weights, scales


@dataclass(frozen=True, kw_only=True)
class AdditiveNoise(Functional):
    """Additive-noise model ``X_c = sum_p w_{p,c} g(X_p) + noise_c`` with a nonlinear ``g``.

    The nonlinearity ``g(z) = z + sin(z)`` is nonlinear yet non-degenerate (never flattens the
    signal), so the additive-noise direction is identifiable.
    """

    name: str = "additive_noise"

    def _node(self, parent_values, weights, noise_c):
        return noise_c + _nl(parent_values) @ weights


@dataclass(frozen=True, kw_only=True)
class PostNonlinear(Functional):
    """Post-nonlinear model ``X_c = g2(sum_p w_{p,c} X_p + noise_c)`` with invertible ``g2``.

    The outer transform ``g2`` is strictly increasing on the real line, the requirement of the
    post-nonlinear model (Zhang and Hyvarinen 2009), and ``outer`` names it:

    - ``"tanh"``, the default: ``g2(z) = z + 0.5 tanh(z)``, with a bounded nonlinear term so the
      transform does not amplify signal down the causal order into overflow.
    - ``"cube"``: ``g2(z) = z**3``, the transform of the post-nonlinear data of Montagna et al.
      (2023). It triples the exponent at every generation, so under ``scale="standardize"`` and
      ``scale="raw"``, which evaluate the model on raw values, a path of more than a few edges
      overflows and the sample raises; ``scale="rescale"`` bounds it.
    - ``"cube_root"``: ``g2(z) = z**(1/3)``, the inverse of ``"cube"``, which compresses large
      values and has an unbounded slope at 0.
    - ``"sigmoid"``: ``g2(z) = 1 / (1 + exp(-z))``, the logistic transform of the post-nonlinear
      data of Lachapelle et al. (2020); the simulation of Zhang and Hyvarinen (2009) takes a
      ``tanh``, the same shape. Its values lie in ``(0, 1)``, and in float64 it returns exactly
      0 or 1 for an argument beyond about 37 in absolute value.

    Parameters
    ----------
    weight_range : tuple of float, default=(0.5, 2.0)
        Magnitude range of the edge weights, each drawn with a random sign.
    outer : {"tanh", "cube", "cube_root", "sigmoid"}, default="tanh"
        The outer transform ``g2``. It draws nothing, so the weights and the noise are those of
        the default on the same seed.

    Raises
    ------
    ValueError
        If ``outer`` is not one of the transforms.

    References
    ----------
    Zhang, K., and Hyvarinen, A. (2009). On the identifiability of the post-nonlinear causal
    model. Proceedings of the 25th Conference on Uncertainty in Artificial Intelligence, 647-655.

    Lachapelle, S., Brouillard, P., Deleu, T., and Lacoste-Julien, S. (2020). Gradient-based
    neural DAG learning. International Conference on Learning Representations.

    Montagna, F., Mastakouri, A. A., Eulig, E., Noceti, N., Rosasco, L., Janzing, D., Aragam, B.,
    and Locatello, F. (2023). Assumption violations in causal discovery and the robustness of
    score matching. Advances in Neural Information Processing Systems 36.
    """

    name: str = "post_nonlinear"
    outer: str = "tanh"

    def __post_init__(self) -> None:
        """Check that ``outer`` names a transform."""
        if self.outer not in OUTER_TRANSFORMS:
            raise ValueError(f"outer must be one of {tuple(OUTER_TRANSFORMS)}, got {self.outer!r}")

    @property
    def options(self) -> dict:
        """The outer transform, when it is not the default."""
        return {} if self.outer == "tanh" else {"outer": self.outer}

    def _node(self, parent_values, weights, noise_c):
        return OUTER_TRANSFORMS[self.outer](parent_values @ weights + noise_c)


@dataclass(frozen=True, kw_only=True)
class NonMonotone(Functional):
    """Non-monotone additive-noise model ``X_c = sum_p w_{p,c} cos(X_p) + noise_c``.

    ``cos`` is even, so a child is uncorrelated with a parent whose distribution is symmetric about
    zero though it depends on it: partial correlation misses the edge where a nonlinear
    conditional-independence test finds it. ``cos`` is bounded, so no variance accumulates down
    the causal order.
    """

    name: str = "nonmonotone"

    def _node(self, parent_values, weights, noise_c):
        return noise_c + np.cos(parent_values) @ weights


@dataclass(frozen=True, kw_only=True)
class LocationScale(Functional):
    """Location-scale noise model ``X_c = sum_p w_{p,c} g(X_p) + s(pa_c) noise_c``.

    The location is the additive-noise mean, with ``g(z) = z + sin(z)``. The scale is a function of
    the parents, ``s(pa_c) = lo + (hi - lo) sigmoid(sum_p v_{p,c} X_p)`` with
    ``(lo, hi) = scale_range``, so it is strictly positive and bounded, and the conditional variance
    of a child given its parents changes with them (Immer et al. 2023, Definition 1). The scale
    weights ``v`` are drawn independently of the location weights ``w``, from the same range, after
    them in the weight stream, so the location weights are those of :class:`AdditiveNoise` on the
    same seed; ``params.scale_weights`` records them. The default ``scale_range`` is the range of
    the noise scale ``0.1 (4 pi - |x|) + 0.2`` over ``x ~ U[-4 pi, 4 pi]`` in the location-scale
    benchmark of Immer et al. (2023, Supplementary Material E), a ratio of about 7.3.
    ``scale_range=(1.0, 1.0)`` gives a constant scale and the data of :class:`AdditiveNoise`.

    The scale argument grows with its parents' variance. Under ``scale="standardize"`` and
    ``scale="raw"``, which evaluate the model on raw values, a node deep in the causal order has a
    large argument, so its scale is near ``lo`` or ``hi`` by the sign of the argument and its
    location outgrows its noise. ``scale="rescale"`` gives every parent unit variance before its
    children read it, which keeps the scale a smooth function of the parents at every depth.
    Heteroscedastic noise multiplies the scale by the node's constant noise scale, the one
    ``params.noise_scales`` records.

    Parameters
    ----------
    weight_range : tuple of float, default=(0.5, 2.0)
        Magnitude range of the edge weights, each drawn with a random sign.
    scale_range : tuple of float, default=(0.2, 0.2 + 0.4 * pi)
        The bounds ``(lo, hi)`` of the scale, with ``0 < lo <= hi``
        (:data:`LOCATION_SCALE_RANGE`).

    Raises
    ------
    ValueError
        If ``scale_range`` is not finite with ``0 < lo <= hi``.

    References
    ----------
    Immer, A., Schultheiss, C., Vogt, J. E., Scholkopf, B., Buhlmann, P., and Marx, A. (2023). On
    the Identifiability and Estimation of Causal Location-Scale Noise Models. Proceedings of the
    40th International Conference on Machine Learning, PMLR 202.
    """

    name: str = "location_scale"
    scale_range: tuple[float, float] = LOCATION_SCALE_RANGE

    def __post_init__(self) -> None:
        """Check that the scale is strictly positive and its bounds ordered."""
        lo, hi = self.scale_range
        if not (np.isfinite(lo) and np.isfinite(hi) and 0 < lo <= hi):
            raise ValueError(
                f"scale_range must be finite with 0 < lo <= hi, got {self.scale_range!r}"
            )

    def draw_scale_weights(self, draw: DAGDraw, rng: np.random.Generator) -> np.ndarray:
        """Per-edge weights of the scale, drawn as the location weights are (``(m,)``)."""
        return self.draw_weights(draw, rng)

    def _node(self, parent_values, weights, noise_c):
        lo, hi = self.scale_range
        location = np.ascontiguousarray(weights[:, 0])
        scale_weights = np.ascontiguousarray(weights[:, 1])
        # sigmoid(z) = (1 + tanh(z / 2)) / 2, which does not overflow for a large |z|.
        scale = lo + (hi - lo) * 0.5 * (1.0 + np.tanh(0.5 * (parent_values @ scale_weights)))
        return _nl(parent_values) @ location + scale * noise_c


@dataclass(frozen=True, kw_only=True)
class RandomFeatures(Functional):
    """Additive-noise model ``X_c = sum_p f_{p,c}(X_p) + noise_c`` with Gaussian-process links.

    Each ``f_{p,c}`` is one approximate draw of a Gaussian process with mean 0 and the kernel
    ``exp(-(u - v)**2 / (2 length_scale**2))``, built from ``n_features`` random Fourier features
    (Rahimi and Recht 2007):
    ``f(u) = sqrt(2 / n_features) sum_k a_k cos(omega_k u + b_k)`` with frequencies
    ``omega_k ~ N(0, 1 / length_scale**2)``, phases ``b_k ~ U[0, 2 pi]``, and amplitudes
    ``a_k ~ N(0, 1)``. Over the draws, ``f`` has mean 0, variance 1 at every point, and that
    kernel as its covariance, and it approaches a Gaussian process as ``n_features`` grows. One
    function per parent, summed, is the additive model of the simulations of Buhlmann, Peters,
    and Ernest (2014), which draw each function from a Gaussian process of bandwidth 1; the cost is
    ``n * n_features`` per edge where an exact draw costs ``n**3``. Each function is bounded, so no
    variance accumulates down the causal order.

    ``params.weights`` has shape ``(m, 3, n_features)``: for each edge its frequencies, phases,
    and amplitudes. ``weight_range`` is not used. ``soft(weight_scale=...)`` multiplies the
    amplitudes of the edges into its target, and ``soft(weights="redraw")`` draws their functions
    anew. The length scale is in the units the parents have when their children read them: raw
    values under ``scale="standardize"`` and ``scale="raw"``, unit variance under
    ``scale="rescale"``.

    Parameters
    ----------
    n_features : int, default=100
        The number of random Fourier features of each function, at least 1.
    length_scale : float, default=1.0
        The length scale (bandwidth) of the kernel, finite and positive.

    Raises
    ------
    ValueError
        If ``n_features`` is not a whole number of at least 1 or ``length_scale`` is not finite
        and positive.

    References
    ----------
    Rahimi, A., and Recht, B. (2007). Random features for large-scale kernel machines. Advances
    in Neural Information Processing Systems 20.

    Buhlmann, P., Peters, J., and Ernest, J. (2014). CAM: Causal additive models,
    high-dimensional order search and penalized regression. The Annals of Statistics 42(6),
    2526-2556.
    """

    name: str = "random_features"
    n_features: int = 100
    length_scale: float = 1.0

    def __post_init__(self) -> None:
        """Check the number of features and the length scale."""
        if isinstance(self.n_features, bool) or not isinstance(self.n_features, int | np.integer):
            raise ValueError(f"n_features must be a whole number >= 1, got {self.n_features!r}")
        if self.n_features < 1:
            raise ValueError(f"n_features must be a whole number >= 1, got {self.n_features!r}")
        if not (np.isfinite(self.length_scale) and self.length_scale > 0):
            raise ValueError(f"length_scale must be finite and positive, got {self.length_scale!r}")
        object.__setattr__(self, "n_features", int(self.n_features))
        object.__setattr__(self, "length_scale", float(self.length_scale))

    @property
    def options(self) -> dict:
        """The number of features and the length scale."""
        return {"n_features": self.n_features, "length_scale": self.length_scale}

    def draw_weights(self, draw: DAGDraw, rng: np.random.Generator) -> np.ndarray:
        """Per-edge frequencies, phases, and amplitudes, shape ``(m, 3, n_features)``."""
        shape = (draw.parents.shape[0], self.n_features)
        frequencies = rng.standard_normal(shape) / self.length_scale
        phases = rng.uniform(0.0, 2.0 * np.pi, size=shape)
        amplitudes = rng.standard_normal(shape)
        return np.stack([frequencies, phases, amplitudes], axis=1)

    def scale_effect(self, weights: np.ndarray, factor: float) -> np.ndarray:
        """Return ``weights`` with the amplitudes multiplied by ``factor``."""
        scaled = weights.copy()
        scaled[:, 2] *= factor
        return scaled

    def _node(self, parent_values, weights, noise_c):
        signal = np.zeros(parent_values.shape[0])
        for values, (frequencies, phases, amplitudes) in zip(parent_values.T, weights):
            signal += np.cos(np.outer(values, frequencies) + phases) @ amplitudes
        return noise_c + np.sqrt(2.0 / self.n_features) * signal


@dataclass(frozen=True, kw_only=True)
class VAR(Functional):
    """Vector-autoregressive temporal model: instantaneous DAG mixing ``B0`` plus lag matrices.

    ``x_t = B0 x_t + sum_k A_k x_{t-k} + e_t``. The sampled DAG is the instantaneous ``B0``;
    ``n_lags`` autoregressive matrices ``A_k`` (with diagonal self-terms) are drawn and stabilized.
    A modest-d subsystem (dense weight-stack truth) -- see :mod:`andrey.data.temporal`.
    """

    name: str = "var"
    is_temporal: ClassVar[bool] = True
    n_lags: int = 1
    lag_density: float = 0.3

    def generate_temporal(self, draw, weights, innovations, lag_rng, *, burn_in):
        """Generate the VAR time series and its ``TemporalStructure`` truth."""
        from . import temporal

        temporal.check_size(draw.d)
        b0 = temporal.build_b0(draw, weights)
        a_stack = temporal.draw_lag_stack(
            draw.d, self.n_lags, self.lag_density, self.weight_range, lag_rng
        )
        a_stack, ib_inv = temporal.stabilize(b0, a_stack)
        data = temporal.generate(ib_inv, a_stack, None, innovations, burn_in)
        return data, temporal.build_truth(draw, b0, a_stack, None)


@dataclass(frozen=True, kw_only=True)
class VARMA(VAR):
    """Vector ARMA temporal model: VAR plus ``ma_lags`` moving-average matrices on the innovations.

    ``x_t = B0 x_t + sum_k A_k x_{t-k} + e_t + sum_q M_q e_{t-q}``.
    """

    name: str = "varma"
    ma_lags: int = 1
    ma_density: float = 0.3

    def generate_temporal(self, draw, weights, innovations, lag_rng, *, burn_in):
        """Generate the VARMA time series and its ``TemporalStructure`` truth."""
        from . import temporal

        temporal.check_size(draw.d)
        b0 = temporal.build_b0(draw, weights)
        a_stack = temporal.draw_lag_stack(
            draw.d, self.n_lags, self.lag_density, self.weight_range, lag_rng
        )
        a_stack, ib_inv = temporal.stabilize(b0, a_stack)
        m_stack = temporal.draw_lag_stack(
            draw.d, self.ma_lags, self.ma_density, (0.1, 0.4), lag_rng
        )
        data = temporal.generate(ib_inv, a_stack, m_stack, innovations, burn_in)
        return data, temporal.build_truth(draw, b0, a_stack, m_stack)


def linear(weight_range: tuple[float, float] = (0.5, 2.0), *, positive: bool = False) -> Linear:
    """Build a linear structural-equation spec with per-edge weights from ``+/- weight_range``.

    Parameters
    ----------
    weight_range : tuple of float, default=(0.5, 2.0)
        Magnitude range of the edge weights, each drawn with a random sign.
    positive : bool, default=False
        Make every weight positive: its magnitude, with the sign dropped. A sample then has the
        weights of the signed model on the same seed, in absolute value, and every other draw
        unchanged. ``scale="unitless"`` draws its own weights and does not take it.

    Returns
    -------
    Linear
    """
    return Linear(weight_range=weight_range, positive=positive)


def additive_noise(weight_range: tuple[float, float] = (0.5, 2.0)) -> AdditiveNoise:
    """Build an additive-noise (ANM) spec: a nonlinear function of parents plus additive noise."""
    return AdditiveNoise(weight_range=weight_range)


def post_nonlinear(
    weight_range: tuple[float, float] = (0.5, 2.0), *, outer: str = "tanh"
) -> PostNonlinear:
    """Build a post-nonlinear (PNL) spec.

    An invertible nonlinearity of a linear combination plus noise.

    Parameters
    ----------
    weight_range : tuple of float, default=(0.5, 2.0)
        Magnitude range of the edge weights, each drawn with a random sign.
    outer : {"tanh", "cube", "cube_root", "sigmoid"}, default="tanh"
        The outer transform, strictly increasing on the real line; see :class:`PostNonlinear` for
        each one and its source.

    Returns
    -------
    PostNonlinear

    Raises
    ------
    ValueError
        If ``outer`` is not one of the transforms.
    """
    return PostNonlinear(weight_range=weight_range, outer=outer)


def nonmonotone(weight_range: tuple[float, float] = (0.5, 2.0)) -> NonMonotone:
    """Non-monotone additive-noise structural equations (see :class:`NonMonotone`)."""
    return NonMonotone(weight_range=weight_range)


def location_scale(
    weight_range: tuple[float, float] = (0.5, 2.0),
    scale_range: tuple[float, float] = LOCATION_SCALE_RANGE,
) -> LocationScale:
    """Build a location-scale noise (LSNM) spec: a noise scale that depends on the parents.

    See :class:`LocationScale` for the model, ``scale_range``, and the reference.
    """
    return LocationScale(weight_range=weight_range, scale_range=scale_range)


def random_features(n_features: int = 100, length_scale: float = 1.0) -> RandomFeatures:
    """Build an additive-noise spec whose functions are approximate Gaussian-process draws.

    Parameters
    ----------
    n_features : int, default=100
        The number of random Fourier features of each edge's function, at least 1.
    length_scale : float, default=1.0
        The length scale of the Gaussian-process kernel, finite and positive.

    Returns
    -------
    RandomFeatures
        See :class:`RandomFeatures` for the model, its parameters, and the references.

    Raises
    ------
    ValueError
        If ``n_features`` is not a whole number of at least 1 or ``length_scale`` is not finite
        and positive.
    """
    return RandomFeatures(n_features=n_features, length_scale=length_scale)


def var(n_lags: int = 1, lag_density: float = 0.3) -> VAR:
    """Build a vector-autoregressive VAR(``n_lags``) temporal spec (modest d)."""
    return VAR(n_lags=n_lags, lag_density=lag_density)


def varma(n_lags: int = 1, ma_lags: int = 1) -> VARMA:
    """Build a vector ARMA VARMA(``n_lags``, ``ma_lags``) temporal spec (modest d)."""
    return VARMA(n_lags=n_lags, ma_lags=ma_lags)


def _nl(z: np.ndarray) -> np.ndarray:
    """Apply the ANM nonlinearity: nonlinear but non-degenerate."""
    return z + np.sin(z)


def _g2(z: np.ndarray) -> np.ndarray:
    """Map the PNL outer nonlinearity: strictly increasing (invertible), bounded nonlinear term."""
    return z + 0.5 * np.tanh(z)


def _cube(z: np.ndarray) -> np.ndarray:
    """Map ``z`` to ``z**3``."""
    return z * z * z


#: The outer transforms of :class:`PostNonlinear` by name, each strictly increasing on the real
#: line.
OUTER_TRANSFORMS = {"tanh": _g2, "cube": _cube, "cube_root": np.cbrt, "sigmoid": expit}


def _parents_by_child(
    children: np.ndarray, parents: np.ndarray, weights: np.ndarray
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Group edges by child: ``child -> (parent_labels, weights)`` (parentless children omitted)."""
    if children.size == 0:
        return {}
    order = np.argsort(children, kind="stable")
    ch, pa, w = children[order], parents[order], weights[order]
    bounds = np.concatenate([[0], np.flatnonzero(np.diff(ch)) + 1, [ch.size]])
    return {int(ch[a]): (pa[a:b], w[a:b]) for a, b in zip(bounds[:-1], bounds[1:])}
