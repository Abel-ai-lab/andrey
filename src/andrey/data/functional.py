"""Functional forms for the structural equations (the ``f`` in ``X_j = f(pa_j, noise_j)``).

A functional spec draws per-edge parameters and then evaluates the SCM in topological order over the
sparse edge list -- no dense adjacency. ``linear`` is the causally-sufficient default;
``additive_noise`` (ANM) and ``post_nonlinear`` (PNL) are the nonlinear identifiable families. All
three share one topological-evaluation loop and differ only in the per-node combine rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from .graphs import DAGDraw


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
