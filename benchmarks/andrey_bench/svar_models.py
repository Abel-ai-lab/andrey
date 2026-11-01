"""The structural vector autoregressions of the DYNOTEARS paper's simulations.

Pamfil et al. (2020), AISTATS, Sect. 2.1 and Appendix B.4, write the model with row vectors:

    x_t = x_t W + x_{t-1} A + z_t

``W[i, j]`` weights ``x_i(t) -> x_j(t)`` and ``A[i, j]`` weights ``x_i(t-1) -> x_j(t)``. Every
model here has one lag, the order of the paper's Figure 3; the paper's weight decay ``eta`` scales
lags beyond the first only, so it does not enter.

- Same-step graph: Erdos-Renyi, a Bernoulli draw of probability ``k / (d - 1)`` on every entry below
  the diagonal, so the mean degree (in and out) is ``k``; or Barabasi-Albert, each new node sending
  ``k / 2`` edges to existing nodes drawn in proportion to their degree, as the authors' CausalNex
  draws it. Weights uniform in magnitude on [0.5, 2] with a random sign.
- Lagged graph: Erdos-Renyi, a Bernoulli draw of probability ``k / d`` on every entry, self-lags
  included, so the mean in-degree is ``k``; or a stochastic block model with two equal blocks,
  ``p_out / p_in = 0.3``, and the same mean in-degree. Weights uniform in magnitude on [0.3, 0.5]
  with a random sign.
- Noise: standard Gaussian, or exponential of scale 1 (:data:`NOISES`). Sect. 2.1 takes noise of
  mean zero, so ``"exponential"`` subtracts the mean 1 and ``"exponential_uncentered"`` keeps it.

The nodes are then relabeled by one permutation drawn from the seed, so the column order carries no
information about either graph. The paper assumes a stationary series and states no check; a draw
whose reduced-form lag matrix ``A (I - W)^-1`` has spectral radius 1 or more is drawn again, up to
:data:`MAX_DRAWS` times. The Barabasi-Albert and block-model graphs need more draws than that at 100
variables for some seeds, so no draw is rescaled and such a size raises instead.

``dynotears2020_sin_tanh`` keeps the linear model's graphs and weights and replaces each same-step
parent ``x_i(t)`` by ``sin(x_i(t))`` and each lagged one by ``tanh(x_i(t-1))``: a nonlinear series
the paper's linear model does not describe. Its terms are bounded, so it needs no stationarity draw.

A series has ``n + 1`` steps, so the fit regresses ``n`` steps on the step before each, the paper's
``n``. ``standardize="standardized"`` z-scores every column of the same series.
"""

from __future__ import annotations

import zlib
from dataclasses import asdict, dataclass

import numpy as np

from andrey_bench.window import marks_from_edges

#: Time steps simulated and discarded before a series starts.
BURN_IN = 1000

#: Draws of a linear model's graphs and weights before a size is given up as not stationary.
MAX_DRAWS = 1000

#: The noise kinds a key may name.
NOISES = ("gaussian", "exponential", "exponential_uncentered")

#: The scales a key may name: the series as simulated, or every column z-scored.
SCALES = ("raw", "standardized")


@dataclass(frozen=True)
class SVARModel:
    """One simulation setup of Pamfil et al. (2020), Appendix B.4; see the module docstring."""

    same_step_graph: str  # "er" or "ba"
    same_step_degree: float  # k of the same-step graph: mean degree, in and out
    lagged_graph: str  # "er" or "sbm"
    lagged_degree: float  # mean in-degree of the lagged graph, self-lags included
    same_step_weight: tuple[float, float]  # |W| entries are uniform on this range
    lagged_weight: tuple[float, float]  # |A| entries are uniform on this range
    block_ratio: float  # p_out / p_in of the block model
    link: str  # "linear", or "sin_tanh" for sin of same-step and tanh of lagged parents


_PAPER = {
    "same_step_weight": (0.5, 2.0),
    "lagged_weight": (0.3, 0.5),
    "block_ratio": 0.3,
}

MODELS: dict[str, SVARModel] = {
    "dynotears2020": SVARModel("er", 2.0, "er", 2.0, link="linear", **_PAPER),
    "dynotears2020_ba_sbm": SVARModel("ba", 4.0, "sbm", 2.0, link="linear", **_PAPER),
    "dynotears2020_sin_tanh": SVARModel("er", 2.0, "er", 2.0, link="sin_tanh", **_PAPER),
}

#: Each model's dataset topology label: its two graph models.
TOPOLOGIES = {
    name: "er" if m.same_step_graph == "er" else f"{m.same_step_graph}_{m.lagged_graph}"
    for name, m in MODELS.items()
}


def model_params(name: str) -> dict:
    """A model's setup as plain data, for the dataset's generation call and digest."""
    return asdict(MODELS[name])


def _signed(
    rng: np.random.Generator, flags: np.ndarray, magnitude: tuple[float, float]
) -> np.ndarray:
    """Weights uniform in magnitude on `magnitude` with a random sign, where `flags` is set."""
    weights = rng.uniform(*magnitude, flags.shape) * rng.choice([-1.0, 1.0], flags.shape)
    return np.where(flags, weights, 0.0)


def _same_step(model: SVARModel, d: int, rng: np.random.Generator) -> np.ndarray:
    """The same-step weights ``W`` in construction order: ``W[i, j]`` weights ``i -> j``."""
    k = model.same_step_degree
    if model.same_step_graph == "er":
        flags = np.tril(rng.random((d, d)) < k / (d - 1), -1)
    else:  # Barabasi-Albert: each new node links to earlier ones drawn by degree
        flags = np.zeros((d, d), dtype=bool)
        bag = [0]
        for i in range(1, d):
            targets = rng.choice(bag, size=int(round(k / 2)))
            flags[i, targets] = True
            bag.append(i)
            bag.extend(targets.tolist())
    return _signed(rng, flags, model.same_step_weight)


def _lagged(model: SVARModel, d: int, rng: np.random.Generator) -> np.ndarray:
    """The lag-1 weights ``A``: ``A[i, j]`` weights ``x_i(t-1) -> x_j(t)``."""
    k = model.lagged_degree
    if model.lagged_graph == "er":
        prob = np.full((d, d), k / d)
    else:  # two blocks; n_same * p_in + n_other * p_out = k for every effect j
        block = np.arange(d) < d // 2
        same = block[:, None] == block[None, :]
        n_same = same.sum(axis=0)
        p_in = k / (n_same + model.block_ratio * (d - n_same))
        prob = np.where(same, p_in[None, :], model.block_ratio * p_in[None, :])
    return _signed(rng, rng.random((d, d)) < prob, model.lagged_weight)


def spectral_radius(w: np.ndarray, a: np.ndarray) -> float:
    """Spectral radius of the reduced-form lag matrix ``A (I - W)^-1``; below 1 is stationary."""
    inverse = np.linalg.inv(np.eye(len(w)) - w)
    return float(np.max(np.abs(np.linalg.eigvals(a @ inverse))))


def draw(name: str, d: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.random.Generator]:
    """Return ``(W, A, rng)`` of model `name`: relabeled weights and the stream left for the noise.

    Raises ``RuntimeError`` when none of :data:`MAX_DRAWS` linear draws is stationary.
    """
    model = MODELS[name]
    rng = np.random.default_rng([zlib.crc32(name.encode()), d, seed])
    for _ in range(MAX_DRAWS):
        w, a = _same_step(model, d, rng), _lagged(model, d, rng)
        if model.link != "linear" or spectral_radius(w, a) < 1.0:
            break
    else:
        raise RuntimeError(f"{name}: no stationary draw in {MAX_DRAWS} at d={d}, seed={seed}")
    order = rng.permutation(d)
    return w[np.ix_(order, order)], a[np.ix_(order, order)], rng


def _noise(kind: str, shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    if kind == "gaussian":
        return rng.standard_normal(shape)
    if kind == "exponential":
        return rng.exponential(1.0, shape) - 1.0
    if kind == "exponential_uncentered":
        return rng.exponential(1.0, shape)
    raise ValueError(f"unknown noise {kind!r}; known: {NOISES}")


def _parents_first(w: np.ndarray) -> list[int]:
    """The nodes of the same-step DAG ``w`` in a causal order."""
    indegree = (w != 0).sum(axis=0)
    ready = [j for j in range(len(w)) if indegree[j] == 0]
    order = []
    while ready:
        i = ready.pop()
        order.append(i)
        for j in np.flatnonzero(w[i]):
            indegree[j] -= 1
            if indegree[j] == 0:
                ready.append(int(j))
    return order


def sample(
    name: str, d: int, n: int, seed: int, noise: str, standardize: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(data, window_marks)``: a series of ``n + 1`` steps of model `name` and its truth."""
    if standardize not in SCALES:
        raise ValueError(f"unknown scale {standardize!r}; known: {SCALES}")
    w, a, rng = draw(name, d, seed)
    steps = n + 1 + BURN_IN
    z = _noise(noise, (steps, d), rng)
    x = np.zeros((steps, d))
    if MODELS[name].link == "linear":
        inverse = np.linalg.inv(np.eye(d) - w)
        for t in range(steps):
            drive = z[t] + (x[t - 1] @ a if t else 0.0)
            x[t] = drive @ inverse
    else:
        order = _parents_first(w)
        for t in range(steps):
            base = z[t] + (np.tanh(x[t - 1]) @ a if t else 0.0)
            for j in order:  # a node's same-step parents come before it, so x[t] holds them
                x[t, j] = base[j] + np.sin(x[t]) @ w[:, j]
    series = x[BURN_IN:]
    if standardize == "standardized":
        series = (series - series.mean(axis=0)) / series.std(axis=0)
    return np.ascontiguousarray(series), marks_from_edges(w, a[None])
