"""Functional forms for the structural equations (the ``f`` in ``X_j = f(pa_j, noise_j)``).

A functional spec draws per-edge parameters and then evaluates the SCM in topological order over the
sparse edge list -- no dense adjacency. ``linear`` is the causally-sufficient default;
``additive_noise`` (ANM) and ``post_nonlinear`` (PNL) are the nonlinear identifiable families,
``nonmonotone`` an additive-noise family whose effects partial correlation does not see, and
``location_scale`` (LSNM) a family whose noise scale is a function of the parents. All five share
one topological-evaluation loop and differ only in the per-node combine rule.
:meth:`Linear.draw_unitless` draws the weights and noise scales of ``scale="unitless"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from .graphs import DAGDraw

#: The most nodes :meth:`Linear.draw_unitless` takes. It keeps the dense ``(d, d)`` correlation
#: matrix of the nodes, 8 MB in float64 at this size, and its time grows with ``d`` times the edges.
UNITLESS_MAX_NODES = 1000


@dataclass(frozen=True, kw_only=True)
class Functional:
    """A lazy structural-equation specification (shared topological evaluation)."""

    name: str
    weight_range: tuple[float, float] = (0.5, 2.0)
    is_temporal: ClassVar[bool] = False  # temporal families set this and provide generate_temporal

    def draw_weights(self, draw: DAGDraw, rng: np.random.Generator) -> np.ndarray:
        """Per-edge weights, magnitude ``U[weight_range]`` with a random sign (``(m,)``)."""
        m = draw.parents.shape[0]
        lo, hi = self.weight_range
        return rng.uniform(lo, hi, size=m) * rng.choice(np.array([-1.0, 1.0]), size=m)

    def _node(
        self, parent_values: np.ndarray, weights: np.ndarray, noise_c: np.ndarray
    ) -> np.ndarray:
        """Combine a child's parent values, its edge weights, and its noise into its column."""
        raise NotImplementedError

    def evaluate(
        self, draw: DAGDraw, weights: np.ndarray, noise: np.ndarray, *, rescale: bool = False
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
        """
        x = noise.astype(np.float64, copy=True)
        child_block = _parents_by_child(draw.children, draw.parents, weights)
        for c in draw.topo_order:
            c = int(c)
            block = child_block.get(c)
            if block is not None:
                pa, w = block
                x[:, c] = self._node(x[:, pa], w, noise[:, c])
            if rescale:
                sd = x[:, c].std()
                if sd > 0:
                    x[:, c] /= sd
        return x


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

    ``g2(z) = z + 0.5 tanh(z)`` is strictly increasing (invertible), the PNL requirement, with a
    bounded nonlinear term so the transform does not amplify signal down the causal order into
    overflow.
    """

    name: str = "post_nonlinear"

    def _node(self, parent_values, weights, noise_c):
        return _g2(parent_values @ weights + noise_c)


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
    the parents, ``s(pa_c) = lo + (hi - lo) sigmoid(sum_p w_{p,c} X_p)`` with
    ``(lo, hi) = scale_range``, so it is strictly positive and bounded, and the conditional variance
    of a child given its parents changes with them (Immer et al. 2023, Definition 1). The scale
    reuses the edge weights of the location. ``scale_range=(1.0, 1.0)`` gives a constant scale and
    the data of :class:`AdditiveNoise`.

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
    scale_range : tuple of float, default=(0.2, 2.0)
        The bounds ``(lo, hi)`` of the scale, with ``0 < lo <= hi``.

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
    scale_range: tuple[float, float] = (0.2, 2.0)

    def __post_init__(self) -> None:
        """Check that the scale is strictly positive and its bounds ordered."""
        lo, hi = self.scale_range
        if not (np.isfinite(lo) and np.isfinite(hi) and 0 < lo <= hi):
            raise ValueError(
                f"scale_range must be finite with 0 < lo <= hi, got {self.scale_range!r}"
            )

    def _node(self, parent_values, weights, noise_c):
        lo, hi = self.scale_range
        # sigmoid(z) = (1 + tanh(z / 2)) / 2, which does not overflow for a large |z|.
        scale = lo + (hi - lo) * 0.5 * (1.0 + np.tanh(0.5 * (parent_values @ weights)))
        return _nl(parent_values) @ weights + scale * noise_c


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


def linear(weight_range: tuple[float, float] = (0.5, 2.0)) -> Linear:
    """Build a linear structural-equation spec with per-edge weights from ``+/- weight_range``."""
    return Linear(weight_range=weight_range)


def additive_noise(weight_range: tuple[float, float] = (0.5, 2.0)) -> AdditiveNoise:
    """Build an additive-noise (ANM) spec: a nonlinear function of parents plus additive noise."""
    return AdditiveNoise(weight_range=weight_range)


def post_nonlinear(weight_range: tuple[float, float] = (0.5, 2.0)) -> PostNonlinear:
    """Build a post-nonlinear (PNL) spec.

    An invertible nonlinearity of a linear combination plus noise.
    """
    return PostNonlinear(weight_range=weight_range)


def nonmonotone(weight_range: tuple[float, float] = (0.5, 2.0)) -> NonMonotone:
    """Non-monotone additive-noise structural equations (see :class:`NonMonotone`)."""
    return NonMonotone(weight_range=weight_range)


def location_scale(
    weight_range: tuple[float, float] = (0.5, 2.0), scale_range: tuple[float, float] = (0.2, 2.0)
) -> LocationScale:
    """Build a location-scale noise (LSNM) spec: a noise scale that depends on the parents.

    See :class:`LocationScale` for the model, ``scale_range``, and the reference.
    """
    return LocationScale(weight_range=weight_range, scale_range=scale_range)


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
