"""Data for the methods whose input is not one plain sample: domains, groups, and latent indicators.

Each model draws a dataset of the usual ``(n, columns)`` layout, with its truth as endpoint marks,
so the harness stores, fits, and scores it like any other dataset:

- ``domain_shift`` (CD-NOD): ``n`` rows pooled from :data:`N_DOMAINS` domains in contiguous equal
  blocks, domain ``k`` in rows ``k * n / N_DOMAINS`` onward. A linear-Gaussian Erdos-Renyi DAG on
  ``d`` variables at the key's mean degree; one variable in three, at least one, has a mean that
  moves with the domain, by ``s * k / (N_DOMAINS - 1)`` in domain ``k``, with ``s`` uniform on
  :attr:`DomainShift.shift`. The shift grows linearly over the domains, as a drift over time
  blocks does, so a test of linear correlation with the domain index detects it. The truth is the
  CPDAG CD-NOD can identify: the augmented DAG, with the domain index as a source pointing into the
  shifted variables, turned into its CPDAG; the index's edges are then directed out of it, Meek's
  rules applied, and the index dropped (:func:`domain_cpdag`). An edge that only the index orients
  is directed in this truth, where the plain CPDAG of the DAG would leave it undirected.
- ``multi_group`` (MultiGroupDirectLiNGAM): :data:`N_GROUPS` groups of ``n`` rows each over the same
  ``d`` variables, side by side: group ``g`` is columns ``g * d`` to ``(g + 1) * d``. Every group
  has the same Erdos-Renyi DAG, so the same causal order, with its own weights and its own
  uniform noise. Groups are independent, so the ``(n, N_GROUPS * d)`` matrix is a sample of the
  block-diagonal DAG of the groups' copies, which is the stored truth. A graph is scored over all
  groups at once: SHD is the sum of the groups' SHDs.
- ``measurement`` (GIN): ``m`` latent variables, every earlier one a cause of every later one, and
  :attr:`Measurement.indicators` observed indicators per latent, each with that latent as its only
  parent; linear with uniform noise. The indicators come in a random column order; the latents
  are the last ``m`` nodes of the truth and have no data column. The key's ``d`` counts both.

Every variable is rescaled to unit variance in causal order before its children are drawn, as
``andrey.data`` does for ``scale="rescale"``, so values stay bounded on long paths. Weights are
uniform in magnitude on :data:`WEIGHT_RANGE` with a random sign. The domain shift is added after a
variable's rescaling, in units of its standard deviation.

``domain_shift`` and ``multi_group`` then list their variables in an order drawn from the seed's
column-order stream (:func:`andrey_bench.lingam_models.column_order`), with the truth permuted the
same way, so the column order does not follow how the graph was drawn. The rows keep their order, so
each domain keeps its block of rows; a multi-group dataset repeats one order of its ``d`` variables
in every group, so each group keeps its block of columns and lists the variables as the others do.
``measurement`` already draws its column order and takes no such step.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from andrey_bench.lingam_models import column_order

#: Domains in a ``domain_shift`` dataset; the CD-NOD adapters rebuild the domain index from it.
N_DOMAINS = 4

#: Groups in a ``multi_group`` dataset; the MultiGroupDirectLiNGAM adapters split the columns by it.
N_GROUPS = 3

#: Magnitude range of every edge weight and indicator loading; the sign is random.
WEIGHT_RANGE = (0.5, 2.0)


@dataclass(frozen=True)
class DomainShift:
    """CD-NOD's regime; see the module docstring."""

    n_domains: int = N_DOMAINS
    shifted_per_variable: float = 1 / 3  # share of variables whose mean moves with the domain
    shift: tuple[float, float] = (0.5, 1.5)  # |s| range, in the variable's standard deviations


@dataclass(frozen=True)
class MultiGroup:
    """MultiGroupDirectLiNGAM's regime; see the module docstring."""

    n_groups: int = N_GROUPS


@dataclass(frozen=True)
class Measurement:
    """GIN's regime; see the module docstring."""

    indicators: int = 3  # observed indicators per latent
    indicator_noise_sd: float = 0.5  # before the indicator's rescaling; the latent has variance 1


#: Each model, keyed by the ``functional`` label its datasets carry.
MODELS: dict[str, DomainShift | MultiGroup | Measurement] = {
    "domain_shift": DomainShift(),
    "multi_group": MultiGroup(),
    "measurement": Measurement(),
}

#: Noise draws with variance 1.
_NOISE = {
    "gaussian": lambda rng, n: rng.standard_normal(n),
    "uniform": lambda rng, n: rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), n),
}

# Andrey's endpoint marks: the mark at node i on the edge i-j is marks[i, j].
_TAIL, _ARROW = 1, 2


def domain_index(n: int, n_domains: int) -> np.ndarray:
    """Return the ``(n, 1)`` domain index of a ``domain_shift`` dataset: ``k`` in domain ``k``."""
    if n % n_domains:
        raise ValueError(f"n = {n} rows do not split into {n_domains} equal domains")
    return np.repeat(np.arange(n_domains, dtype=np.float64), n // n_domains).reshape(-1, 1)


def split_groups(data: np.ndarray, n_groups: int) -> list[np.ndarray]:
    """Return the groups of a ``multi_group`` dataset, its ``n_groups`` equal blocks of columns."""
    if data.shape[1] % n_groups:
        raise ValueError(f"{data.shape[1]} columns do not split into {n_groups} equal groups")
    return [np.ascontiguousarray(block) for block in np.split(data, n_groups, axis=1)]


def block_diagonal(adjacencies: list[np.ndarray]) -> np.ndarray:
    """Return the 0/1 adjacency of the groups' graphs side by side, as a multi-group truth is."""
    d = adjacencies[0].shape[0]
    out = np.zeros((len(adjacencies) * d,) * 2, dtype=np.int8)
    for g, adjacency in enumerate(adjacencies):
        out[g * d : (g + 1) * d, g * d : (g + 1) * d] = np.asarray(adjacency) != 0
    return out


def model_params(model: str) -> dict:
    """Return `model`'s setup, recorded in each dataset's generation call."""
    return {**asdict(MODELS[model]), "weight_range": list(WEIGHT_RANGE)}


def latents_for(model: str, d: int) -> int:
    """Return the hidden nodes of a `model` dataset with `d` nodes: the latents of ``measurement``.

    Raises ``ValueError`` when `d` is not a whole number of latents with their indicators.
    """
    spec = MODELS[model]
    if not isinstance(spec, Measurement):
        return 0
    per = spec.indicators + 1
    if d < 2 * per or d % per:
        raise ValueError(f"a measurement model on {d} nodes needs d = m * {per} with m >= 2")
    return d // per


def sample(
    model: str, *, d: int, n: int, seed: int, density: float, noise: str, shuffle: bool = True
) -> tuple[np.ndarray, np.ndarray, str, np.ndarray | None]:
    """Draw one dataset of `model`; return ``(data, truth_marks, truth_kind, column_order)``.

    For ``domain_shift`` and ``multi_group``, `d` is the variable count and `density` the
    Erdos-Renyi mean degree; for ``measurement``, `d` counts the latents and their indicators and
    `density` is unused. `noise` names the noise family (:data:`_NOISE`).

    ``column_order[j]`` is the column that column ``j`` holds with `shuffle` off, so the identity
    then. With `shuffle`, ``domain_shift`` and ``multi_group`` permute their variables, and the truth
    with them, as the module docstring says; a multi-group order covers every group's columns.
    ``measurement`` takes no shuffle step and returns ``None``.
    """
    rng = np.random.default_rng(seed)
    draw = _NOISE[noise]
    if model == "measurement":
        return (*_measurement(MODELS[model], d, n, draw, rng), None)
    if model == "domain_shift":
        x, marks, kind = _domain_shift(MODELS[model], d, n, density, draw, rng)
    elif model == "multi_group":
        x, marks, kind = _multi_group(MODELS[model], d, n, density, draw, rng)
    else:
        raise KeyError(f"unknown model {model!r}; known: {sorted(MODELS)}")
    variables = column_order(model, d, seed, shuffle)
    columns = np.concatenate([block * d + variables for block in range(x.shape[1] // d)])
    data = np.ascontiguousarray(np.take(x, columns, axis=1))
    return data, np.ascontiguousarray(marks[np.ix_(columns, columns)]), kind, columns


def _er_dag(d: int, density: float, rng: np.random.Generator):
    """An Erdos-Renyi DAG from ``andrey.data``: ``(parents, children, topological order)``."""
    from andrey.data.graphs import erdos_renyi, params_for_density

    _, draw = erdos_renyi(d, **params_for_density("erdos_renyi", d, density)).materialize(rng)
    return draw.parents, draw.children, draw.topo_order


def _weights(m: int, rng: np.random.Generator) -> np.ndarray:
    """``m`` weights, uniform in magnitude on :data:`WEIGHT_RANGE`, each with a random sign."""
    return rng.uniform(*WEIGHT_RANGE, m) * rng.choice([-1.0, 1.0], m)


def _linear(parents, children, order, weights, n, draw, rng, offset=None) -> np.ndarray:
    """Sample a linear SEM in causal `order`, rescaling each variable before its children.

    ``offset[:, j]``, when given, is added to variable ``j`` after its rescaling.
    """
    d = len(order)
    x = np.zeros((n, d))
    for j in order:
        into = children == j
        value = x[:, parents[into]] @ weights[into] + draw(rng, n)
        x[:, j] = value / value.std()
        if offset is not None:
            x[:, j] += offset[:, j]
    return x


def _marks(d: int, parents: np.ndarray, children: np.ndarray) -> np.ndarray:
    """Endpoint marks of the DAG ``parents[k] -> children[k]`` on `d` nodes."""
    marks = np.zeros((d, d), dtype=np.int8)
    marks[parents, children] = _TAIL
    marks[children, parents] = _ARROW
    return marks


def _domain_shift(spec: DomainShift, d, n, density, draw, rng):
    level = domain_index(n, spec.n_domains)[:, 0] / (spec.n_domains - 1)
    parents, children, order = _er_dag(d, density, rng)
    weights = _weights(len(parents), rng)
    shifted = rng.choice(d, max(1, round(d * spec.shifted_per_variable)), replace=False)
    offset = np.zeros((n, d))
    offset[:, shifted] = np.outer(level, rng.uniform(*spec.shift, len(shifted)))
    x = _linear(parents, children, order, weights, n, draw, rng, offset)
    return x, domain_cpdag(_marks(d, parents, children), shifted), "cpdag"


def domain_cpdag(dag_marks: np.ndarray, shifted: np.ndarray) -> np.ndarray:
    """Return the CPDAG CD-NOD identifies from a DAG whose `shifted` variables move with the domain.

    The domain index joins as a last node with an edge into each shifted variable. The augmented
    DAG becomes its CPDAG, the index's edges are directed out of it, and Meek's rules complete the
    orientation before the index is dropped. These are ``andrey.core.orient``'s helpers, the ones
    Andrey's CD-NOD orients with.
    """
    from andrey.core.orient import dag2cpdag, meek

    d = len(dag_marks)
    augmented = np.zeros((d + 1, d + 1), dtype=np.int8)
    augmented[:d, :d] = dag_marks
    augmented[d, shifted] = _TAIL
    augmented[shifted, d] = _ARROW
    cpdag = dag2cpdag(augmented)
    cpdag[d, shifted] = _TAIL
    cpdag[shifted, d] = _ARROW
    return np.ascontiguousarray(meek(cpdag)[:d, :d])


def _multi_group(spec: MultiGroup, d, n, density, draw, rng):
    parents, children, order = _er_dag(d, density, rng)
    blocks = [
        _linear(parents, children, order, _weights(len(parents), rng), n, draw, rng)
        for _ in range(spec.n_groups)
    ]
    adjacency = np.zeros((d, d), dtype=np.int8)
    adjacency[parents, children] = 1
    joint = block_diagonal([adjacency] * spec.n_groups)
    edges = np.nonzero(joint)
    return np.hstack(blocks), _marks(len(joint), *edges), "dag"


def _measurement(spec: Measurement, d, n, draw, rng):
    m = d // (spec.indicators + 1)
    observed = m * spec.indicators
    latent = np.zeros((n, m))
    for j in range(m):
        value = latent[:, :j] @ _weights(j, rng) + draw(rng, n)
        latent[:, j] = value / value.std()
    # Column c of the data measures latent cluster[c]. The columns come in a uniformly random order
    # and the latents have no column, so the column order carries nothing of the graph.
    cluster = rng.permutation(np.repeat(np.arange(m), spec.indicators))
    loadings = _weights(observed, rng)
    noise = np.column_stack([draw(rng, n) for _ in range(observed)])
    x = latent[:, cluster] * loadings + spec.indicator_noise_sd * noise
    x /= x.std(axis=0)
    parents = np.concatenate([observed + cluster, *[observed + np.arange(j) for j in range(m)]])
    children = np.concatenate([np.arange(observed), *[np.full(j, observed + j) for j in range(m)]])
    return x, _marks(d, parents, children), "dag"
