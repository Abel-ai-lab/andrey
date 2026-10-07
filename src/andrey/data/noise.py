"""Exogenous-noise families for structural causal models.

A noise spec draws per-node scales and then a unit-variance innovation array; both from an explicit
RNG so a dataset is regenerable from its seed. Gaussian is the causally-sufficient default;
non-Gaussian families (uniform / Laplace / exponential / Gumbel) drive LiNGAM identifiability. Every
family is normalized to unit variance so ``scales`` is the sole variance knob.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EULER_GAMMA = 0.5772156649015329
_GUMBEL_STD = np.pi / np.sqrt(6.0)


@dataclass(frozen=True, kw_only=True)
class Noise:
    """A lazy exogenous-noise specification.

    Parameters
    ----------
    heteroscedastic : bool, default=False
        If ``True``, draw per-node scales from ``U[0.5, 2.0]``; otherwise every scale is ``1.0``.
    """

    name: str
    heteroscedastic: bool = False

    def draw_scales(self, d: int, rng: np.random.Generator) -> np.ndarray:
        """Per-node noise standard deviations, shape ``(d,)``."""
        if self.heteroscedastic:
            return rng.uniform(0.5, 2.0, size=d)
        return np.ones(d, dtype=np.float64)

    def _standard(self, shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
        """Draw a unit-variance, zero-mean iid innovation array of the given shape."""
        raise NotImplementedError

    def draw(self, n: int, scales: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Draw an ``(n, d)`` innovation array with column ``j`` scaled by ``scales[j]``."""
        return self._standard((n, scales.shape[0]), rng) * scales


@dataclass(frozen=True, kw_only=True)
class Gaussian(Noise):
    """Gaussian (normal) exogenous noise -- the causally-sufficient default."""

    name: str = "gaussian"

    def _standard(self, shape, rng):
        return rng.standard_normal(shape)


@dataclass(frozen=True, kw_only=True)
class Uniform(Noise):
    """Uniform exogenous noise (non-Gaussian; the classic LiNGAM innovation)."""

    name: str = "uniform"

    def _standard(self, shape, rng):
        return rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), size=shape)  # unit variance


@dataclass(frozen=True, kw_only=True)
class Laplace(Noise):
    """Laplace (double-exponential) exogenous noise (heavy-tailed, non-Gaussian)."""

    name: str = "laplace"

    def _standard(self, shape, rng):
        return rng.laplace(0.0, 1.0 / np.sqrt(2.0), size=shape)  # var = 2 b^2 = 1


@dataclass(frozen=True, kw_only=True)
class Exponential(Noise):
    """Centered exponential exogenous noise (skewed, non-Gaussian)."""

    name: str = "exponential"

    def _standard(self, shape, rng):
        return rng.standard_exponential(size=shape) - 1.0  # mean 0, var 1


@dataclass(frozen=True, kw_only=True)
class Gumbel(Noise):
    """Centered Gumbel exogenous noise (skewed, non-Gaussian)."""

    name: str = "gumbel"

    def _standard(self, shape, rng):
        return (rng.gumbel(0.0, 1.0, size=shape) - _EULER_GAMMA) / _GUMBEL_STD  # mean 0, var 1


def gaussian(heteroscedastic: bool = False) -> Gaussian:
    """Build a Gaussian noise spec; ``heteroscedastic`` randomizes per-node scales."""
    return Gaussian(heteroscedastic=heteroscedastic)


def uniform(heteroscedastic: bool = False) -> Uniform:
    """Build a uniform (non-Gaussian) noise spec."""
    return Uniform(heteroscedastic=heteroscedastic)


def laplace(heteroscedastic: bool = False) -> Laplace:
    """Build a Laplace (non-Gaussian) noise spec."""
    return Laplace(heteroscedastic=heteroscedastic)


def exponential(heteroscedastic: bool = False) -> Exponential:
    """Build a centered-exponential (non-Gaussian) noise spec."""
    return Exponential(heteroscedastic=heteroscedastic)


def gumbel(heteroscedastic: bool = False) -> Gumbel:
    """Build a centered-Gumbel (non-Gaussian) noise spec."""
    return Gumbel(heteroscedastic=heteroscedastic)
