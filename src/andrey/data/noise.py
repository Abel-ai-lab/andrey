"""Exogenous-noise families for structural causal models.

A noise spec draws per-node scales and then a unit-variance innovation array; both from an explicit
RNG so a dataset is regenerable from its seed. Gaussian is the causally-sufficient default;
non-Gaussian families (uniform / Laplace / exponential / Gumbel) drive LiNGAM identifiability, and
:func:`student_t` and :func:`gaussian_mixture` add heavy tails and two modes. Every family is
normalized to zero mean and unit variance so ``scales`` is the sole variance knob.
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
    variance_range : tuple of float or None, default=None
        The bounds ``(lo, hi)`` of the per-node noise variances: each node draws its variance from
        ``U[lo, hi]``, and its scale is the square root. Takes ``0 < lo <= hi`` and
        ``heteroscedastic=False``.

    Raises
    ------
    ValueError
        If ``variance_range`` is not a finite pair with ``0 < lo <= hi``, or is given with
        ``heteroscedastic=True``.
    """

    name: str
    heteroscedastic: bool = False
    variance_range: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        """Check the variance range and store it as a pair of floats."""
        if self.variance_range is None:
            return
        bounds = tuple(float(v) for v in self.variance_range)
        if len(bounds) != 2 or not (np.all(np.isfinite(bounds)) and 0 < bounds[0] <= bounds[1]):
            raise ValueError(
                f"variance_range must be finite with 0 < lo <= hi, got {self.variance_range!r}"
            )
        if self.heteroscedastic:
            raise ValueError("noise takes heteroscedastic=True or a variance_range, not both")
        object.__setattr__(self, "variance_range", bounds)

    @property
    def options(self) -> dict:
        """The family's own parameters, as a dataset's provenance records them; empty without."""
        return {}

    def draw_scales(self, d: int, rng: np.random.Generator) -> np.ndarray:
        """Per-node noise standard deviations, shape ``(d,)``."""
        if self.variance_range is not None:
            return np.sqrt(rng.uniform(*self.variance_range, size=d))
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


@dataclass(frozen=True, kw_only=True)
class StudentT(Noise):
    """Student-t exogenous noise with ``df`` degrees of freedom, scaled to unit variance.

    A Student-t variable with ``df > 2`` degrees of freedom has variance ``df / (df - 2)``, so
    each draw is multiplied by ``sqrt((df - 2) / df)``. The tails are heavier the smaller ``df``
    is: the excess kurtosis is ``6 / (df - 4)`` for ``df > 4`` and infinite for ``df <= 4``, and
    the family approaches the Gaussian as ``df`` grows.

    Parameters
    ----------
    df : float, default=5.0
        The degrees of freedom, finite and greater than 2, where the variance is finite.
    heteroscedastic, variance_range
        The noise scales; see :class:`Noise`.

    Raises
    ------
    ValueError
        If ``df`` is not finite and greater than 2.

    References
    ----------
    Student (1908). The probable error of a mean. Biometrika 6(1), 1-25.
    """

    name: str = "student_t"
    df: float = 5.0

    def __post_init__(self) -> None:
        """Check the degrees of freedom, then the scale arguments."""
        if not (np.isfinite(self.df) and self.df > 2):
            raise ValueError(
                f"student_t takes finite df > 2, where the variance is finite, got {self.df!r}"
            )
        object.__setattr__(self, "df", float(self.df))
        super().__post_init__()

    @property
    def options(self) -> dict:
        """The degrees of freedom."""
        return {"df": self.df}

    def _standard(self, shape, rng):
        return rng.standard_t(self.df, size=shape) * np.sqrt((self.df - 2.0) / self.df)


@dataclass(frozen=True, kw_only=True)
class GaussianMixture(Noise):
    """Two-component Gaussian-mixture exogenous noise, centered and scaled to unit variance.

    A draw comes from ``N(means[0], sds[0]**2)`` with probability ``weight`` and from
    ``N(means[1], sds[1]**2)`` otherwise; the mixture's mean is then subtracted and the result
    divided by the mixture's standard deviation, so only the shape of the mixture reaches the data.
    The default is the bimodal density of Marron and Wand (1992, density 6):
    ``0.5 N(-1, (2/3)**2) + 0.5 N(1, (2/3)**2)``. Unequal ``means`` with an unequal ``weight`` or
    unequal ``sds`` give a skewed density.

    Parameters
    ----------
    weight : float, default=0.5
        The probability of the first component, in ``(0, 1)``.
    means : tuple of float, default=(-1.0, 1.0)
        The two component means, finite.
    sds : tuple of float, default=(2/3, 2/3)
        The two component standard deviations, finite and positive.
    heteroscedastic, variance_range
        The noise scales; see :class:`Noise`.

    Raises
    ------
    ValueError
        If ``weight`` is not in ``(0, 1)``, ``means`` is not a pair of finite numbers, or ``sds``
        is not a pair of finite positive numbers.

    References
    ----------
    Marron, J. S., and Wand, M. P. (1992). Exact mean integrated squared error. The Annals of
    Statistics 20(2), 712-736.

    McLachlan, G., and Peel, D. (2000). Finite Mixture Models. Wiley.
    """

    name: str = "gaussian_mixture"
    weight: float = 0.5
    means: tuple[float, float] = (-1.0, 1.0)
    sds: tuple[float, float] = (2.0 / 3.0, 2.0 / 3.0)

    def __post_init__(self) -> None:
        """Check the mixture's parameters, then the scale arguments."""
        if not (np.isfinite(self.weight) and 0 < self.weight < 1):
            raise ValueError(f"gaussian_mixture takes a weight in (0, 1), got {self.weight!r}")
        means = tuple(float(v) for v in self.means)
        sds = tuple(float(v) for v in self.sds)
        if len(means) != 2 or not np.all(np.isfinite(means)):
            raise ValueError(f"gaussian_mixture takes two finite means, got {self.means!r}")
        if len(sds) != 2 or not (np.all(np.isfinite(sds)) and min(sds) > 0):
            raise ValueError(f"gaussian_mixture takes two finite positive sds, got {self.sds!r}")
        object.__setattr__(self, "weight", float(self.weight))
        object.__setattr__(self, "means", means)
        object.__setattr__(self, "sds", sds)
        super().__post_init__()

    @property
    def options(self) -> dict:
        """The weight, means, and standard deviations of the two components."""
        return {"weight": self.weight, "means": list(self.means), "sds": list(self.sds)}

    def _standard(self, shape, rng):
        means, sds = np.array(self.means), np.array(self.sds)
        probs = np.array([self.weight, 1.0 - self.weight])
        mean = probs @ means
        sd = np.sqrt(probs @ (sds**2 + means**2) - mean**2)
        component = (rng.random(shape) >= self.weight).astype(np.intp)  # 0 with probability weight
        return (means[component] + sds[component] * rng.standard_normal(shape) - mean) / sd


def gaussian(
    heteroscedastic: bool = False, variance_range: tuple[float, float] | None = None
) -> Gaussian:
    """Build a Gaussian noise spec; ``heteroscedastic`` and ``variance_range`` set the scales.

    See :class:`Noise` for the two scale arguments.
    """
    return Gaussian(heteroscedastic=heteroscedastic, variance_range=variance_range)


def uniform(
    heteroscedastic: bool = False, variance_range: tuple[float, float] | None = None
) -> Uniform:
    """Build a uniform (non-Gaussian) noise spec."""
    return Uniform(heteroscedastic=heteroscedastic, variance_range=variance_range)


def laplace(
    heteroscedastic: bool = False, variance_range: tuple[float, float] | None = None
) -> Laplace:
    """Build a Laplace (non-Gaussian) noise spec."""
    return Laplace(heteroscedastic=heteroscedastic, variance_range=variance_range)


def exponential(
    heteroscedastic: bool = False, variance_range: tuple[float, float] | None = None
) -> Exponential:
    """Build a centered-exponential (non-Gaussian) noise spec."""
    return Exponential(heteroscedastic=heteroscedastic, variance_range=variance_range)


def gumbel(
    heteroscedastic: bool = False, variance_range: tuple[float, float] | None = None
) -> Gumbel:
    """Build a centered-Gumbel (non-Gaussian) noise spec."""
    return Gumbel(heteroscedastic=heteroscedastic, variance_range=variance_range)


def student_t(
    df: float = 5.0,
    heteroscedastic: bool = False,
    variance_range: tuple[float, float] | None = None,
) -> StudentT:
    """Build a Student-t (heavy-tailed) noise spec of unit variance.

    Parameters
    ----------
    df : float, default=5.0
        The degrees of freedom, finite and greater than 2.
    heteroscedastic : bool, default=False
        Draw per-node scales from ``U[0.5, 2.0]``; see :class:`Noise`.
    variance_range : tuple of float or None, default=None
        The bounds of the per-node noise variances; see :class:`Noise`.

    Returns
    -------
    StudentT
        See :class:`StudentT` for the scaling and the reference.

    Raises
    ------
    ValueError
        If ``df`` is not finite and greater than 2.
    """
    return StudentT(df=df, heteroscedastic=heteroscedastic, variance_range=variance_range)


def gaussian_mixture(
    weight: float = 0.5,
    means: tuple[float, float] = (-1.0, 1.0),
    sds: tuple[float, float] = (2.0 / 3.0, 2.0 / 3.0),
    heteroscedastic: bool = False,
    variance_range: tuple[float, float] | None = None,
) -> GaussianMixture:
    """Build a two-component Gaussian-mixture noise spec of zero mean and unit variance.

    Parameters
    ----------
    weight : float, default=0.5
        The probability of the first component, in ``(0, 1)``.
    means : tuple of float, default=(-1.0, 1.0)
        The two component means.
    sds : tuple of float, default=(2/3, 2/3)
        The two component standard deviations, positive.
    heteroscedastic : bool, default=False
        Draw per-node scales from ``U[0.5, 2.0]``; see :class:`Noise`.
    variance_range : tuple of float or None, default=None
        The bounds of the per-node noise variances; see :class:`Noise`.

    Returns
    -------
    GaussianMixture
        See :class:`GaussianMixture` for the centering, the default, and the references.

    Raises
    ------
    ValueError
        If ``weight`` is not in ``(0, 1)``, ``means`` is not a pair of finite numbers, or ``sds``
        is not a pair of finite positive numbers.
    """
    return GaussianMixture(
        weight=weight,
        means=means,
        sds=sds,
        heteroscedastic=heteroscedastic,
        variance_range=variance_range,
    )
