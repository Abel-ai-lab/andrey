"""The simulation of the SCORE paper, which the NoGAM paper repeats.

Rolland et al. (2022), Sect. 4.1, draw a nonlinear additive noise model, ``x_j = f_j(pa_j) + e_j``,
and Montagna et al. (2023), Sect. 6, draw the same one, "mimicking the setting of Rolland et al.":

- Graph: Erdos-Renyi from ``andrey.data`` at the key's mean degree, which fixes the edge count:
  2 for the papers' ER1 (``d`` edges) and 8 for ER4 (``4 d``).
- Links: each ``f_j`` is one draw of a Gaussian process over the vector of ``j``'s parents, with the
  RBF kernel ``exp(-||u - u'||^2 / 2)`` of unit bandwidth (both papers) and unit variance (the
  papers state none; the authors' code takes 1), evaluated at the ``n`` rows: a Gaussian vector
  with that covariance. The draw is joint over the parents, not a sum of one function per parent.
  The covariance is factored by Cholesky with :attr:`Score2022.jitter` added to its diagonal, a
  constant the papers leave open: ``1e-6`` of the kernel variance, the first jitter GPy's Cholesky
  tries.
- Noise: ``e_j = sigma_j z_j`` with ``sigma_j^2`` uniform on ``[0.4, 0.8]`` for every variable
  (Rolland et al., Sect. 4.1) and ``z_j`` of mean 0 and variance 1, Gaussian, Laplace, or Gumbel
  (:data:`NOISES`; ``andrey.data``'s families). NoGAM's model takes noise of mean 0 (its Eq. 1), so
  the Gumbel noise is centered.
- Rows: ``n``, 1,000 in both papers.

``standardize`` (:data:`SCALES`) sets the scale: ``"raw"`` keeps the values as drawn, the papers'
scale; ``"standardized"`` z-scores every column of the same draw; ``"ancestral"`` divides each
variable by its standard deviation before its children read it, as ``andrey.data`` does for
``scale="rescale"``, so the links see inputs of unit variance.

The variables are then listed in an order drawn from the seed's column-order stream
(:func:`andrey_bench.lingam_models.column_order`), with the truth permuted the same way: the
authors' generator lists the variables in causal order, which this harness does not inherit.

References
----------
Rolland, P., Cevher, V., Kleindessner, M., Russell, C., Janzing, D., Scholkopf, B., and Locatello,
F. (2022). Score matching enables causal discovery of nonlinear additive noise models. ICML, PMLR
162.

Montagna, F., Noceti, N., Rosasco, L., Zhang, K., and Locatello, F. (2023). Causal discovery with
score matching on additive models with arbitrary noise. CLeaR, PMLR 213.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from andrey.data import noise as noise_families
from andrey.data.graphs import erdos_renyi, params_for_density
from andrey_bench.lingam_models import column_order

#: The noise families, each of mean 0 and variance 1 before its scale.
NOISES = ("gaussian", "laplace", "gumbel")

#: The ``standardize`` labels a dataset takes; see the module docstring.
SCALES = ("raw", "standardized", "ancestral")

# Andrey's endpoint marks: the mark at node i on the edge i-j is marks[i, j].
_TAIL, _ARROW = 1, 2


@dataclass(frozen=True)
class Score2022:
    """The SCORE paper's simulation, Rolland et al. (2022), Sect. 4.1; see the module docstring."""

    bandwidth: float = 1.0  # of the RBF kernel
    kernel_variance: float = 1.0
    noise_variance: tuple[float, float] = (0.4, 0.8)
    jitter: float = 1e-6  # added to the kernel's diagonal, in units of its variance


#: Each model, keyed by the ``functional`` label its datasets carry.
MODELS: dict[str, Score2022] = {"score2022": Score2022()}


def model_params(model: str) -> dict:
    """Return `model`'s setup, recorded in each dataset's generation call."""
    return asdict(MODELS[model])


def gp_draw(u: np.ndarray, spec: Score2022, rng: np.random.Generator) -> np.ndarray:
    """One draw at the rows of `u` of the Gaussian process over its columns; see the module."""
    sq = np.sum(u * u, axis=1)
    dist = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (u @ u.T), 0.0)
    cov = spec.kernel_variance * np.exp(-dist / (2.0 * spec.bandwidth**2))
    cov[np.diag_indices_from(cov)] += spec.jitter * spec.kernel_variance
    return np.linalg.cholesky(cov) @ rng.standard_normal(u.shape[0])


def sample(
    model: str,
    *,
    d: int,
    n: int,
    seed: int,
    density: float,
    noise: str,
    standardize: str,
    shuffle: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw one dataset of `model`; return ``(data, dag_marks, column_order)``.

    `density` is the Erdos-Renyi mean degree, `noise` one of :data:`NOISES`, and `standardize` one
    of :data:`SCALES`. ``column_order[j]`` is the drawn variable column ``j`` holds: the identity
    unless `shuffle`. Raises ``ValueError`` for another noise or scale.
    """
    if noise not in NOISES or standardize not in SCALES:
        raise ValueError(f"{model} takes noise in {NOISES} and a scale in {SCALES}")
    spec = MODELS[model]
    rng = np.random.default_rng(seed)
    _, graph = erdos_renyi(d, **params_for_density("erdos_renyi", d, density)).materialize(rng)
    family = getattr(noise_families, noise)(variance_range=spec.noise_variance)
    innovations = family.draw(n, family.draw_scales(d, rng), rng)
    x = np.zeros((n, d))
    for j in graph.topo_order:
        parents = graph.parents[graph.children == j]
        x[:, j] = innovations[:, j]
        if parents.size:
            x[:, j] += gp_draw(x[:, parents], spec, rng)
        if standardize == "ancestral":
            x[:, j] /= x[:, j].std()
    if standardize == "standardized":
        x = (x - x.mean(axis=0)) / x.std(axis=0)
    marks = np.zeros((d, d), dtype=np.int8)
    marks[graph.parents, graph.children] = _TAIL
    marks[graph.children, graph.parents] = _ARROW
    columns = column_order(model, d, seed, shuffle)
    data = np.ascontiguousarray(x[:, columns])
    return data, np.ascontiguousarray(marks[np.ix_(columns, columns)]), columns
