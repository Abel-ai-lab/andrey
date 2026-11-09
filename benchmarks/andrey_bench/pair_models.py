"""The synthetic cause-effect pair sets of Tagasovska et al. (2020), which the LOCI paper uses.

Tagasovska et al. (2020), Sect. 3.2, draw five sets of 100 pairs of ``n = 1000`` rows each, from a
cause ``X`` and an effect ``Y``, each pair listed in an order a coin flip draws; Immer et al.
(2023), Sect. 6 and Supplementary Material D, evaluate LOCI on them with every column standardized
to mean 0 and variance 1. The sets, by the ``functional`` label their datasets carry:

- ``an2020`` (AN): additive noise, ``Y = f(X) + E``, ``f`` a Gaussian-process draw.
- ``ans2020`` (AN-s): the same with ``f`` a sigmoid, which is injective.
- ``ls2020`` (LS): location-scale noise, ``Y = f(X) + g(X) E``, ``f`` a Gaussian-process draw.
- ``lss2020`` (LS-s): the same with ``f`` a sigmoid.
- ``mnu2020`` (MN-U): multiplicative noise, ``Y = f(X) U``, ``f`` a sigmoid, ``U`` uniform.

The values the paper states:

- Cause: ``X ~ N(0, sqrt(2))``, read as a standard deviation, as for the noise below.
- Noise: ``E ~ N(0, sigma)`` with ``sigma ~ U[1/5, sqrt(2)/5]``, one ``sigma`` per pair. The range
  is that of a standard deviation, the reading the authors' generator confirms.
- Gaussian process: a Gaussian kernel of bandwidth one, ``k(u, v) = exp(-(u - v)^2 / 2)``.
- Sigmoid: "as in Buhlmann et al. (2014)", whose Sect. 6.1.4 gives
  ``f(x) = a b (x + c) / (1 + |b (x + c)|)`` with ``a ~ Exp(4) + 1``,
  ``b ~ U([-2, -0.5] U [0.5, 2])``, and ``c ~ U([-2, 2])``.
- MN-U: ``U ~ U(0, 1)``.

The constants the paper leaves open, and where each comes from:

- The Gaussian process's variance: 1, the value of the authors' generator
  (``tagas/bQCD``, ``R/utils/data_generators.R``).
- ``Exp(4)``, rate or mean: rate 4, so mean 1/4, the authors' generator's ``rexp(rate = 4)``.
- The scale ``g`` of LS and LS-s: the paper says only that the two sets "correspond to the Gaussian
  processes and sigmoids described for AN and AN-s". The authors' generator takes the mechanism
  itself shifted to a minimum of zero, ``g(X) = f(X) - min_i f(X_i)`` over the pair's rows, and so
  does this module (:attr:`PairModel.mechanism`, :attr:`PairModel.noise_model`). Its smallest
  value is 0, at one row.
- The factorization of the Gaussian-process covariance: Cholesky with :attr:`PairModel.jitter`
  times the variance added to its diagonal. The authors' generator draws through an
  eigendecomposition, whose eigenvector signs vary between linear-algebra libraries; a Cholesky
  factor is unique, so a draw repeats across machines.

Where the authors' generator departs from the paper's text, this module follows the text: that
generator draws the variance of ``X`` and of ``E`` per row, from ``U(1, 2)`` and
``0.04 U(1, 2)``, a scale mixture of Gaussians, where the paper draws ``X`` Gaussian and one
``sigma`` per pair.

A pair's draws come, in this order, from the stream of ``(model, seed)``: ``X``, ``sigma``, the
mechanism (the Gaussian-process vector, or the sigmoid's ``a``, ``b``, ``c``), and the noise
(``E``, or ``U`` for MN-U). Its column order comes from a stream of its own, shared by the five sets
(:data:`ORDER_STREAM`, :func:`andrey_bench.lingam_models.column_order`): the coin flip.
``standardize`` (:data:`SCALES`) is ``"standardized"``, every column z-scored as the LOCI paper
does, or ``"raw"``, the values as drawn.

The Gaussian-process draw factors an ``n``-by-``n`` covariance, so its time grows with ``n^3``.

References
----------
Tagasovska, N., Chavez-Demoulin, V., and Vatter, T. (2020). Distinguishing cause from effect using
quantiles: bivariate quantile causal discovery. ICML, PMLR 119.

Immer, A., Schultheiss, C., Vogt, J. E., Scholkopf, B., Buhlmann, P., and Marx, A. (2023). On the
identifiability and estimation of causal location-scale noise models. ICML, PMLR 202.

Buhlmann, P., Peters, J., and Ernest, J. (2014). CAM: causal additive models, high-dimensional
order search and penalized regression. Annals of Statistics 42(6), 2526-2556.
"""

from __future__ import annotations

import zlib
from dataclasses import asdict, dataclass

import numpy as np

from andrey_bench.lingam_models import column_order

#: The ``standardize`` labels a dataset takes; see the module docstring.
SCALES = ("raw", "standardized")

#: The column-order stream every set draws its coin flip from.
ORDER_STREAM = "anlsmn2020"

# Andrey's endpoint marks: the mark at node i on the edge i-j is marks[i, j].
_TAIL, _ARROW = 1, 2


@dataclass(frozen=True)
class PairModel:
    """One pair set of Tagasovska et al. (2020), Sect. 3.2; see the module docstring."""

    mechanism: str  # "gp" or "sigmoid"
    noise_model: str  # "additive", "location_scale", or "multiplicative"
    noise: str  # the key's noise label: the family of E, or of U for MN-U
    cause_sd: float = float(np.sqrt(2.0))
    noise_sd: tuple[float, float] = (0.2, float(np.sqrt(2.0)) / 5)
    gp_bandwidth: float = 1.0
    gp_variance: float = 1.0
    jitter: float = 1e-6  # added to the covariance's diagonal, in units of its variance
    sigmoid_amplitude_rate: float = 4.0  # a ~ Exp(rate) + 1
    sigmoid_slope: tuple[float, float] = (0.5, 2.0)  # |b|, with a random sign
    sigmoid_shift: tuple[float, float] = (-2.0, 2.0)  # c
    multiplier: tuple[float, float] = (0.0, 1.0)  # U of MN-U


#: Each set, keyed by the ``functional`` label its datasets carry.
MODELS: dict[str, PairModel] = {
    "an2020": PairModel(mechanism="gp", noise_model="additive", noise="gaussian"),
    "ans2020": PairModel(mechanism="sigmoid", noise_model="additive", noise="gaussian"),
    "ls2020": PairModel(mechanism="gp", noise_model="location_scale", noise="gaussian"),
    "lss2020": PairModel(mechanism="sigmoid", noise_model="location_scale", noise="gaussian"),
    "mnu2020": PairModel(mechanism="sigmoid", noise_model="multiplicative", noise="uniform"),
}


def model_params(model: str) -> dict:
    """Return `model`'s setup, recorded in each dataset's generation call."""
    return asdict(MODELS[model])


def gp_draw(x: np.ndarray, spec: PairModel, rng: np.random.Generator) -> np.ndarray:
    """One draw at the rows of `x` of the Gaussian process; see the module docstring."""
    diff = x[:, None] - x[None, :]
    cov = spec.gp_variance * np.exp(-(diff * diff) / (2.0 * spec.gp_bandwidth**2))
    cov[np.diag_indices_from(cov)] += spec.jitter * spec.gp_variance
    return np.linalg.cholesky(cov) @ rng.standard_normal(x.shape[0])


def sigmoid_draw(x: np.ndarray, spec: PairModel, rng: np.random.Generator) -> np.ndarray:
    """One sigmoid of Buhlmann et al. (2014) at the rows of `x`; see the module docstring."""
    a = rng.exponential(1.0 / spec.sigmoid_amplitude_rate) + 1.0
    b = rng.choice([-1.0, 1.0]) * rng.uniform(*spec.sigmoid_slope)
    c = rng.uniform(*spec.sigmoid_shift)
    u = b * (x + c)
    return a * u / (1.0 + np.abs(u))


def sample(
    model: str, *, n: int, seed: int, standardize: str, shuffle: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw one pair of `model`; return ``(data, dag_marks, column_order)``.

    The truth is ``X -> Y``, node 0 to node 1 before the column order. ``column_order[j]`` is the
    drawn variable column ``j`` holds: the identity unless `shuffle`. Raises ``ValueError`` for a
    scale outside :data:`SCALES`.
    """
    if standardize not in SCALES:
        raise ValueError(f"{model} takes a scale in {SCALES}, got {standardize!r}")
    spec = MODELS[model]
    rng = np.random.default_rng([zlib.crc32(model.encode()), seed])
    x = spec.cause_sd * rng.standard_normal(n)
    sigma = rng.uniform(*spec.noise_sd)
    f = gp_draw(x, spec, rng) if spec.mechanism == "gp" else sigmoid_draw(x, spec, rng)
    if spec.noise_model == "multiplicative":
        y = f * rng.uniform(*spec.multiplier, size=n)
    else:
        e = sigma * rng.standard_normal(n)
        y = f + e if spec.noise_model == "additive" else f + (f - f.min()) * e
    pair = np.column_stack([x, y])
    if standardize == "standardized":
        pair = (pair - pair.mean(axis=0)) / pair.std(axis=0)
    marks = np.array([[0, _TAIL], [_ARROW, 0]], dtype=np.int8)
    columns = column_order(ORDER_STREAM, 2, seed, shuffle)
    data = np.ascontiguousarray(pair[:, columns])
    return data, np.ascontiguousarray(marks[np.ix_(columns, columns)]), columns
