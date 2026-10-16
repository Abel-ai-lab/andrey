"""The simulated time series of the PCMCI, PCMCI+, and LPCMCI papers' numerical experiments.

Each paper draws a linear model of the form

    x_t^j = a_j x_{t-1}^j + sum_i c_i x_{t - tau_i}^{k_i} + eta_t^j

with an autodependency ``a_j`` for every variable and ``L`` cross-links ``k_i -> j``, ``k_i != j``,
where ``L`` grows with the number of variables ``d``, so each variable has the same expected number
of cross-links at every size. A cross-link at ``tau_i = 0`` acts within a time step; the same-step
links follow one random causal order, so they form a DAG. A model whose reduced-form process is not
stationary is drawn again, as the papers keep stationary models only. A share of the variables can
be left unobserved: they are generated, then dropped from the data, and placed last in the truth.

The three models, with each paper's section:

- ``pcmci2019``: Runge et al. (2019), Science Advances 5(11), eaau4996, Supplementary Sect. S4.1
  and Table S3: ``L = d`` cross-links (cross-in-degree 1), lags drawn from {1, 2}, no same-step
  links, ``|c| = 0.2`` for every link with a random sign (the T = 300 rows of Table S3), ``a_j``
  drawn from {0, 0.2, 0.4, 0.6, 0.8, 0.9} for even seeds and from {0.6, 0.8, 0.9, 0.95} for odd
  ones (the two halves of the ensemble), and N(0, 1) noise.
- ``pcmciplus2020``: Runge (2020), UAI, Sect. 4: ``L = floor(1.5 d)`` cross-links (cross-in-degree
  1.5), 30% of them at lag 0 and the rest at lags drawn from [1, 5], ``|c|`` uniform on [0.1, 0.5]
  with a random sign, ``a_j`` uniform on [max(0, a - 0.3), a] with the default a = 0.95, and
  Gaussian noise whose standard deviation is uniform on [0.5, 2].
- ``lpcmci2020``: Gerhardus and Runge (2020), NeurIPS, Sect. 4 and Figure 2: ``L = d`` cross-links,
  30% at lag 0 and the rest at lags drawn from [1, 3], ``|c|`` uniform on [0.2, 0.8], ``a_j`` as
  for PCMCI+ with a = 0.95, the same noise, and ``d - ceil(0.7 d)`` variables unobserved
  (lambda = 0.3).

The truth is returned as window marks (:mod:`andrey_bench.window`) over lags ``0..max lag``: a tail
at the cause and an arrowhead at the effect of each same-step link, a tail at the earlier end of each
lagged link, and the autodependencies as lag-1 self-links (none where ``a_j = 0``).
"""

from __future__ import annotations

import math
import zlib
from dataclasses import asdict, dataclass

import numpy as np

from andrey_bench.window import ARROW, TAIL

#: Time steps simulated and discarded before the series starts.
BURN_IN = 1000

#: Draws of a model's links and coefficients before giving up on a stationary one.
MAX_DRAWS = 1000


@dataclass(frozen=True)
class Model:
    """One paper's simulation setup; see the module docstring for the sources."""

    cross_degree: float  # cross-links per variable: L = floor(cross_degree * d)
    same_step: float  # share of the cross-links at lag 0
    lags: tuple[int, int]  # the lag range of the lagged cross-links, both ends included
    coefficient: tuple[float, float]  # |c| is uniform on this range
    autocorrelation_sets: tuple[tuple[float, ...], ...]  # a_j from one set, by seed, if given
    autocorrelation: tuple[float, float]  # else a_j is uniform on this range
    noise_sd: tuple[float, float]  # the noise standard deviation is uniform on this range
    hidden: float  # share of the variables left unobserved

    @property
    def max_lag(self) -> int:
        """The largest lag of any true link, autodependencies included."""
        return max(1, self.lags[1])


MODELS = {
    "pcmci2019": Model(
        cross_degree=1.0,
        same_step=0.0,
        lags=(1, 2),
        coefficient=(0.2, 0.2),
        autocorrelation_sets=((0.0, 0.2, 0.4, 0.6, 0.8, 0.9), (0.6, 0.8, 0.9, 0.95)),
        autocorrelation=(0.0, 0.0),
        noise_sd=(1.0, 1.0),
        hidden=0.0,
    ),
    "pcmciplus2020": Model(
        cross_degree=1.5,
        same_step=0.3,
        lags=(1, 5),
        coefficient=(0.1, 0.5),
        autocorrelation_sets=(),
        autocorrelation=(0.65, 0.95),
        noise_sd=(0.5, 2.0),
        hidden=0.0,
    ),
    "lpcmci2020": Model(
        cross_degree=1.0,
        same_step=0.3,
        lags=(1, 3),
        coefficient=(0.2, 0.8),
        autocorrelation_sets=(),
        autocorrelation=(0.65, 0.95),
        noise_sd=(0.5, 2.0),
        hidden=0.3,
    ),
}


def hidden_count(name: str, d: int) -> int:
    """The number of the ``d`` generated variables a model leaves unobserved."""
    return d - math.ceil((1 - MODELS[name].hidden) * d)


def model_params(name: str) -> dict:
    """A model's setup as plain data, for the dataset's generation call and digest."""
    return asdict(MODELS[name])


def _spectral_radius(lag0: np.ndarray, lagged: np.ndarray) -> float:
    """Spectral radius of the reduced-form VAR's companion matrix; below 1 is stationary.

    ``lag0`` and ``lagged[k - 1]`` are ``[effect, cause]`` coefficient matrices.
    """
    d, p = lag0.shape[0], lagged.shape[0]
    inverse = np.linalg.inv(np.eye(d) - lag0)
    companion = np.zeros((p * d, p * d))
    companion[:d] = np.concatenate([inverse @ lagged[k] for k in range(p)], axis=1)
    companion[d:, : (p - 1) * d] = np.eye((p - 1) * d)
    return float(np.max(np.abs(np.linalg.eigvals(companion))))


def _draw_model(model: Model, d: int, seed: int, rng: np.random.Generator) -> np.ndarray:
    """Draw links and coefficients until the process is stationary.

    Returns the ``(max_lag + 1, d, d)`` ``[lag, effect, cause]`` coefficient stack.
    """
    n_links = math.floor(model.cross_degree * d)
    n_same = round(model.same_step * n_links)
    for _ in range(MAX_DRAWS):
        if model.autocorrelation_sets:
            choices = model.autocorrelation_sets[seed % len(model.autocorrelation_sets)]
            auto = rng.choice(np.asarray(choices), size=d)
        else:
            auto = rng.uniform(*model.autocorrelation, size=d)
        flat = rng.choice(d * (d - 1), size=n_links, replace=False)
        cause, effect = np.divmod(flat, d - 1)
        effect = effect + (effect >= cause)  # skip the diagonal: cause != effect
        lags = rng.integers(model.lags[0], model.lags[1] + 1, size=n_links)
        lags[rng.choice(n_links, size=n_same, replace=False)] = 0
        # Same-step links follow one causal order, so they form a DAG.
        rank = rng.permutation(d)
        backward = (lags == 0) & (rank[cause] > rank[effect])
        cause[backward], effect[backward] = effect[backward], cause[backward]
        coefficient = rng.uniform(*model.coefficient, size=n_links) * rng.choice(
            [-1.0, 1.0], size=n_links
        )
        stack = np.zeros((model.max_lag + 1, d, d))
        stack[1][np.arange(d), np.arange(d)] = auto
        np.add.at(stack, (lags, effect, cause), coefficient)
        same_pairs = {(min(c, e), max(c, e)) for c, e in zip(cause[lags == 0], effect[lags == 0])}
        if len(same_pairs) < n_same:  # a pair drawn both ways at lag 0 would be a cycle
            continue
        if _spectral_radius(stack[0], stack[1:]) < 1.0:
            return stack
    raise RuntimeError(f"no stationary draw in {MAX_DRAWS} tries for d={d}, seed={seed}")


def sample(name: str, d: int, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(data, window_marks)`` for one series of ``n`` steps of model ``name``.

    ``data`` has ``d - hidden_count(name, d)`` columns, the observed variables; the window marks
    cover all ``d`` variables, the unobserved ones last.
    """
    model = MODELS[name]
    rng = np.random.default_rng([zlib.crc32(name.encode()), d, seed])
    stack = _draw_model(model, d, seed, rng)
    sd = rng.uniform(*model.noise_sd, size=d)
    noise = rng.standard_normal((n + BURN_IN, d)) * sd
    inverse = np.linalg.inv(np.eye(d) - stack[0])
    x = np.zeros((n + BURN_IN, d))
    for t in range(n + BURN_IN):
        drive = noise[t].copy()
        for k in range(1, min(t, model.max_lag) + 1):
            drive += stack[k] @ x[t - k]
        x[t] = inverse @ drive
    # [lag, effect, cause] coefficients -> [lag, cause, effect] marks at the causes' ends.
    links = np.transpose(stack != 0, (0, 2, 1))
    marks = np.where(links, TAIL, 0).astype(np.int8)
    marks[0][links[0].T] = ARROW  # a same-step link's effect end
    hidden = hidden_count(name, d)
    order = np.arange(d)
    if hidden:
        unobserved = rng.choice(d, size=hidden, replace=False)
        order = np.concatenate([np.setdiff1d(order, unobserved), np.sort(unobserved)])
    data = np.ascontiguousarray(x[BURN_IN:][:, order[: d - hidden]])
    return data, marks[:, order][:, :, order]
