"""The simulated data of the VAR-LiNGAM and longitudinal LiNGAM papers' experiments.

Both models are linear with non-Gaussian noise, and both truths are coefficient stacks in the
``[block, effect, cause]`` layout the LiNGAM estimators report: ``stack[k][i, j]`` weights the effect
of variable ``j`` on variable ``i``.

``varma2010`` follows Hyvarinen, Zhang, Shimizu, and Hoyer (2010), JMLR 11, Sect. 5.4: one lag,
a strictly lower-triangular instantaneous matrix ``B0`` and a lag matrix ``B1`` with about 60% of
their entries zero, the others uniform in magnitude on [0.05, 0.5] with a random sign, and
super-Gaussian noise: standard Gaussian draws through a power nonlinearity with an exponent between
1.5 and 2.0 that keeps their sign. At the paper's ``N = 5`` that is 4 of the 10 entries of ``B0``'s
lower triangle and 10 of the 25 entries of ``B1``; at every ``N`` the model keeps those counts per
variable, ``0.8 N`` and ``2 N``, so the graph does not grow denser with ``N``. VARMA-LiNGAM also
fits a moving-average part, which the paper's VAR model does not have: the series adds one
moving-average matrix ``M1`` on the noise, drawn as ``B1`` is,
``x_t = B0 x_t + B1 x_{t-1} + e_t + M1 e_{t-1}``. A draw whose autoregressive or moving-average
part is not stable is drawn again; if none of the :data:`MAX_DRAWS` is stable, the first draw is kept
and each part that is not stable is scaled until it is (:func:`_rescale`). Its columns are then
listed in an order drawn from a stream of their own, with the stack permuted the same way, so the
column order does not follow ``B0``'s causal order.

``panel2013`` follows Kadowaki, Shimizu, and Washio's longitudinal LiNGAM experiments as their JSAI
2013 paper describes them (Sect. 6.1; the MLSP 2013 paper is not openly available): ``M`` units
observed at ``T`` time points, a fully dense strictly lower-triangular ``B(t,t)`` at every time
point with entries uniform on [-1.5, -0.5] and [0.5, 1.5], one lag with ``B(t,t-1)`` entries
uniform on [-0.5, 0.5], every matrix drawn independently, noise variances uniform on [1, 2], and at
the first time point a hidden common cause ``f``, uniform on [0, 1], entering every variable with a
weight uniform on [0, 1]. The variables are then put in a random order. The paper draws the noise
from the 18 densities of Bach and Jordan (2002); that paper specifies 5 of them in full (Student t
with 3 and 5 degrees of freedom, double exponential, uniform, and exponential) and shows the
mixtures only as plots, so each variable's noise here is one of those 5, scaled to zero mean and
unit variance before its own variance. Each time point draws its own lag matrix, so a panel is not
stationary, and its scale grows with the product of those matrices unless each is small. A time
point whose lag matrix through the same-step part, ``(I - B(t,t))^-1 B(t,t-1)``, has spectral norm
1 or more, which is almost every time point from ``N = 10`` on, has ``B(t,t-1)`` scaled down until
that norm is 0.95 (:func:`_contract_lags`). Every step then contracts, and the panel stays within a
fixed multiple of one time point's scale for any ``T``. The norm bounds the spectral radius, which
alone does not bound a product of different, non-normal matrices. A time point below norm 1 keeps
its draw.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import asdict, dataclass

import numpy as np

from andrey_bench.temporal_models import STATIONARY_TARGET

#: Time steps simulated and discarded before a series starts.
BURN_IN = 500

#: Draws of a model before rescaling the first one to stability.
MAX_DRAWS = 1000

#: The last seed coordinate of the column-order stream, which no other draw uses.
COLUMN_STREAM = zlib.crc32(b"columns")


@dataclass(frozen=True)
class VarmaModel:
    """Hyvarinen et al. (2010) Sect. 5.4, with one moving-average matrix; see the module docstring."""

    instantaneous_per_variable: float  # nonzero entries of B0 per variable
    lagged_per_variable: float  # nonzero entries of B1, and of M1, per variable
    coefficient: tuple[float, float]  # |b| is uniform on this range
    exponent: tuple[float, float]  # the noise's power-nonlinearity exponent range


@dataclass(frozen=True)
class PanelModel:
    """Kadowaki et al. (2013), JSAI Sect. 6.1; see the module docstring."""

    instantaneous: tuple[float, float]  # |B(t,t)| entries are uniform on this range
    lagged: float  # B(t,t-1) entries are uniform on [-lagged, lagged]
    noise_variance: tuple[float, float]
    lag_norm: float  # a time point's reduced-form lag matrix of norm 1 or more is scaled to this


MODELS: dict[str, VarmaModel | PanelModel] = {
    "varma2010": VarmaModel(
        instantaneous_per_variable=0.8,
        lagged_per_variable=2.0,
        coefficient=(0.05, 0.5),
        exponent=(1.5, 2.0),
    ),
    "panel2013": PanelModel(
        instantaneous=(0.5, 1.5),
        lagged=0.5,
        noise_variance=(1.0, 2.0),
        lag_norm=STATIONARY_TARGET,
    ),
}

#: The noise densities of Bach and Jordan (2002) given in full, each with zero mean, unit variance.
DENSITIES = ("student_t3", "double_exponential", "uniform", "student_t5", "exponential")


def model_params(name: str) -> dict:
    """A model's setup as plain data, for the dataset's generation call and digest."""
    return asdict(MODELS[name])


def _rng(name: str, *coordinates: int) -> np.random.Generator:
    return np.random.default_rng([zlib.crc32(name.encode()), *coordinates])


def column_order(name: str, d: int, seed: int, shuffle: bool = True) -> np.ndarray:
    """Return the drawn variable each of `d` columns holds: the identity unless `shuffle`.

    The permutation comes from the column-order stream of ``(name, d, seed)``, which no draw of the
    data or the truth uses, so turning the shuffle off leaves every other value as it was.
    """
    return _rng(name, d, seed, COLUMN_STREAM).permutation(d) if shuffle else np.arange(d)


def _sparse(
    rng: np.random.Generator, slots: np.ndarray, count: int, magnitude: tuple[float, float]
) -> np.ndarray:
    """Values on ``count`` of the boolean ``slots``, uniform in magnitude with a random sign."""
    out = np.zeros(slots.shape)
    flat = np.flatnonzero(slots)
    chosen = rng.choice(flat, size=min(count, len(flat)), replace=False)
    values = rng.uniform(*magnitude, size=len(chosen)) * rng.choice([-1.0, 1.0], size=len(chosen))
    out.flat[chosen] = values
    return out


def _radius(matrix: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(matrix))))


def _rescale(inverse: np.ndarray, b1: np.ndarray, m1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scale ``B1`` and ``M1`` each down to the stationary target, if it is not stable.

    ``inverse`` is ``(I - B0)^-1``. Scaling a single-lag matrix by ``s`` scales its eigenvalues by
    ``s``, so one ``s`` lands the radius of ``inverse @ B1``, or of ``M1``, on the target exactly; it
    is the rescale of ``andrey.data.temporal.stabilize`` at one lag. A positive scale leaves every
    link where it was, and a part that is stable stays as drawn.
    """
    ar, ma = _radius(inverse @ b1), _radius(m1)
    if ar >= 1.0:
        b1 = b1 * (STATIONARY_TARGET / ar)
    if ma >= 1.0:
        m1 = m1 * (STATIONARY_TARGET / ma)
    return b1, m1


def sample_varma(
    name: str, d: int, n: int, seed: int, *, shuffle: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(data, truth, column_order)``: ``n`` steps of ``d`` variables, the ``[B0, B1, M1]``
    stack, and the drawn variable each column holds.

    With ``shuffle``, the columns are permuted with a permutation from the seed's column-order
    stream, and both axes of every block of the stack with them; without it, ``column_order`` is the
    identity and ``B0`` is strictly lower-triangular.
    """
    model = MODELS[name]
    assert isinstance(model, VarmaModel)
    rng = _rng(name, d, seed)
    lower = np.tril(np.ones((d, d), dtype=bool), k=-1)
    everywhere = np.ones((d, d), dtype=bool)
    first = None
    for _ in range(MAX_DRAWS):
        b0 = _sparse(rng, lower, round(model.instantaneous_per_variable * d), model.coefficient)
        b1 = _sparse(rng, everywhere, round(model.lagged_per_variable * d), model.coefficient)
        m1 = _sparse(rng, everywhere, round(model.lagged_per_variable * d), model.coefficient)
        inverse = np.linalg.inv(np.eye(d) - b0)
        if _radius(inverse @ b1) < 1.0 and _radius(m1) < 1.0:
            break
        if first is None:
            first = (b0, b1, m1)
    else:
        b0, b1, m1 = first
        inverse = np.linalg.inv(np.eye(d) - b0)
        b1, m1 = _rescale(inverse, b1, m1)
    exponent = rng.uniform(*model.exponent, size=d)
    z = rng.standard_normal((n + BURN_IN, d))
    noise = np.sign(z) * np.abs(z) ** exponent
    x = np.zeros((n + BURN_IN, d))
    for t in range(1, n + BURN_IN):
        x[t] = inverse @ (b1 @ x[t - 1] + noise[t] + m1 @ noise[t - 1])
    columns = column_order(name, d, seed, shuffle)
    stack = np.stack([b0, b1, m1])[:, columns][:, :, columns]
    return np.ascontiguousarray(x[BURN_IN:, columns]), np.ascontiguousarray(stack), columns


def _standard_noise(rng: np.random.Generator, density: str, size: tuple[int, ...]) -> np.ndarray:
    """Draws of one of :data:`DENSITIES`, with zero mean and unit variance."""
    if density == "student_t3":
        return rng.standard_t(3, size) / math.sqrt(3.0)
    if density == "student_t5":
        return rng.standard_t(5, size) / math.sqrt(5.0 / 3.0)
    if density == "double_exponential":
        return rng.laplace(0.0, 1.0 / math.sqrt(2.0), size)
    if density == "uniform":
        return rng.uniform(-math.sqrt(3.0), math.sqrt(3.0), size)
    if density == "exponential":
        return rng.exponential(1.0, size) - 1.0
    raise ValueError(f"unknown density {density!r}")


def _contract_lags(truth: np.ndarray, target: float) -> None:
    """Scale each ``B(t,t-1)`` in ``truth`` so its reduced-form lag matrix has norm ``target``, if needed.

    The reduced-form lag matrix of time point ``t`` is ``A_t = (I - B(t,t))^-1 B(t,t-1)``, so
    ``x_t = A_t x_{t-1} + (I - B(t,t))^-1 e_t``. Scaling ``B(t,t-1)`` by ``s`` scales ``A_t`` by
    ``s``, so one ``s`` puts ``A_t``'s spectral norm on ``target`` exactly. A norm below 1 makes the
    step a contraction, which bounds the series for any number of time points, and bounds the radius
    too. A positive scale leaves every link where it was, and a time point already below norm 1
    stays as drawn. No random numbers are drawn.
    """
    identity = np.eye(truth.shape[-1])
    for t in range(1, len(truth)):
        norm = float(np.linalg.norm(np.linalg.solve(identity - truth[t, 0], truth[t, 1]), 2))
        if norm >= 1.0:
            truth[t, 1] *= target / norm


def sample_panel(
    name: str, d: int, times: int, m: int, seed: int, *, scale: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(data, truth)``: ``m`` units of ``d`` variables at ``times`` time points.

    ``data`` has shape ``(times, m, d)``. ``truth`` has shape ``(times, 2, d, d)``: ``truth[t, 0]``
    is ``B(t,t)`` and ``truth[t, 1]`` is ``B(t,t-1)``, zero at the first time point, scaled where
    :func:`_contract_lags` says. ``scale`` multiplies every drawn ``B(t,t)`` and ``B(t,t-1)`` entry
    and leaves every other draw as it is.
    """
    model = MODELS[name]
    assert isinstance(model, PanelModel)
    rng = _rng(name, d, times, m, seed)
    lower = np.tril(np.ones((d, d), dtype=bool), k=-1)
    truth = np.zeros((times, 2, d, d))
    for t in range(times):
        magnitude = rng.uniform(*model.instantaneous, size=(d, d))
        sign = rng.choice([-1.0, 1.0], size=(d, d))
        truth[t, 0] = np.where(lower, magnitude * sign * scale, 0.0)
        if t:
            truth[t, 1] = rng.uniform(-model.lagged, model.lagged, size=(d, d)) * scale
    _contract_lags(truth, model.lag_norm)
    noise_scale = np.sqrt(rng.uniform(*model.noise_variance, size=d))
    densities = rng.choice(len(DENSITIES), size=d)
    noise = (
        np.stack([_standard_noise(rng, DENSITIES[k], (times, m)) for k in densities], axis=-1)
        * noise_scale
    )
    hidden = rng.uniform(0.0, 1.0, size=m)
    weights = rng.uniform(0.0, 1.0, size=d)
    x = np.zeros((times, m, d))
    for t in range(times):
        drive = noise[t] + (np.outer(hidden, weights) if t == 0 else x[t - 1] @ truth[t, 1].T)
        x[t] = np.linalg.solve(np.eye(d) - truth[t, 0], drive.T).T
    # The random variable order makes the panel order-free, so it takes no column-order stream.
    order = rng.permutation(d)
    return (
        np.ascontiguousarray(x[:, :, order]),
        np.ascontiguousarray(truth[:, :, order][:, :, :, order]),
    )
