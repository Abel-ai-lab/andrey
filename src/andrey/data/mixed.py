"""Mixed networks: categorical and continuous variables in a conditional linear Gaussian network.

:func:`sample_mixed` draws a DAG and makes its first variables in causal order categorical, so no
categorical variable has a continuous parent, as a conditional linear Gaussian network requires. A
categorical variable has a conditional probability table, one Dirichlet-distributed row per
configuration of its parents, as in :func:`~andrey.data.sample_discrete`. A continuous variable is
a linear function of its continuous parents, plus a shift for each categorical parent's category,
plus standard Gaussian noise. The data are the input of the mixed-data tests and scores
(``indep_test="dg"`` or ``"cg"``, ``score_func="local_score_DG"`` or ``"local_score_CG"``).

As in :mod:`andrey.data.scm`, the graph, the parameters, and the samples each come from their own
stream of the seed, so a change to one leaves the others' draws unchanged.
"""

from __future__ import annotations

import numpy as np

import andrey

from .dataset import CausalDataset, QAReport, SCMParams
from .discrete import _parents_of
from .scm import _GRAPH, _NOISE, _SEED_SCHEME, _WEIGHTS, _component_rng, named_graph

__all__ = ["sample_mixed"]

# Magnitudes of a continuous parent's weight and of a categorical parent's shifts.
_WEIGHT_RANGE = (0.5, 2.0)


def sample_mixed(
    *,
    graph: str = "erdos_renyi",
    d: int,
    n: int,
    seed: int,
    density: float = 2.0,
    categorical: float = 0.5,
    categories: tuple[int, int] = (2, 3),
    concentration: float = 1.0,
) -> CausalDataset:
    """Sample a conditional linear Gaussian network of categorical and continuous variables.

    Parameters
    ----------
    graph : {"erdos_renyi", "scale_free", "small_world", "hub"}, default="erdos_renyi"
        The random graph the DAG is drawn from, as for :func:`~andrey.data.sample_scm`.
    d : int
        Node count.
    n : int
        Sample count.
    seed : int
        Root seed.
    density : float, default=2.0
        Mean total degree (``2 * |E| / d``) of the graph.
    categorical : float, default=0.5
        The share of the variables that are categorical, from ``0`` to ``1``: the first
        ``round(categorical * d)`` in causal order.
    categories : tuple of int, default=(2, 3)
        The least and the most categories a categorical variable has; each one's count is drawn
        uniformly between them, both included. The least is at least 2.
    concentration : float, default=1.0
        Concentration of the symmetric Dirichlet each table row is drawn from, as for
        :func:`~andrey.data.sample_discrete`.

    Returns
    -------
    CausalDataset
        ``data``, shape ``(n, d)``, holds the category codes ``0..k-1`` of the categorical
        variables and the values of the continuous ones; ``graph`` is the true DAG.
        ``provenance["config"]["categorical"]`` lists the categorical columns and
        ``provenance["config"]["categories"]`` their numbers of categories, in the same order.
        ``params.weights`` holds each edge's weight into a continuous variable: the coefficient of
        a continuous parent, or the largest shift of a categorical one; an edge into a categorical
        variable has ``nan``, its table being its parameters. ``params.noise_scales`` is ``1`` for
        a continuous variable and ``nan`` for a categorical one.

    Raises
    ------
    ValueError
        If ``categorical`` is not in ``[0, 1]``, ``categories`` is not a pair with
        ``2 <= least <= most``, ``concentration`` is not positive, or ``graph`` names no graph.

    Notes
    -----
    A continuous variable ``x`` with continuous parents ``p`` and categorical parents ``c`` is
    ``x = sum_p w_p * p + sum_c s_c[category of c] + e``, with ``e`` standard Gaussian. Each
    ``|w_p|`` is uniform on ``[0.5, 2]`` with a random sign. The shifts ``s_c`` of a parent with
    ``k`` categories are ``k`` evenly spaced values from ``-m`` to ``m``, in a random order, with
    ``m`` uniform on ``[0.5, 2]``, so every category moves ``x`` by a different amount.

    Examples
    --------
    >>> from andrey.data import sample_mixed
    >>> ds = sample_mixed(d=6, n=200, seed=0)
    >>> ds.data.shape, ds.graph.kind, ds.provenance["config"]["categorical"]
    ((200, 6), 'dag', [1, 2, 5])
    """
    whole = len(categories) == 2 and all(
        isinstance(k, int | np.integer) and not isinstance(k, bool) for k in categories
    )
    low, high = (int(k) for k in categories) if whole else (0, 0)
    if not 2 <= low <= high:
        raise ValueError(
            f"categories must be (least, most) with 2 <= least <= most, got {categories}"
        )
    if not 0.0 <= categorical <= 1.0:
        raise ValueError(f"categorical must be a share from 0 to 1, got {categorical}")
    if not (np.isfinite(concentration) and concentration > 0):
        raise ValueError(f"concentration must be finite and positive, got {concentration}")
    spec = named_graph(graph, d, density)
    root = np.random.SeedSequence(seed)
    truth, draw = spec.materialize(_component_rng(root, _GRAPH))
    is_categorical = np.zeros(draw.d, dtype=bool)
    is_categorical[draw.topo_order[: round(categorical * draw.d)]] = True

    rng = _component_rng(root, _WEIGHTS)
    cards = np.where(is_categorical, rng.integers(low, high + 1, size=draw.d), 0)
    parents = _parents_of(draw)
    tables: dict[int, np.ndarray] = {}
    shifts: dict[tuple[int, int], np.ndarray] = {}
    weights = np.full(draw.parents.shape[0], np.nan)
    for c in range(draw.d):
        if is_categorical[c]:
            rows = int(np.prod(cards[parents[c]], dtype=np.int64))
            tables[c] = rng.dirichlet(np.full(int(cards[c]), float(concentration)), size=rows)
            continue
        for k in np.flatnonzero(draw.children == c).tolist():
            p = int(draw.parents[k])
            magnitude = rng.uniform(*_WEIGHT_RANGE)
            if is_categorical[p]:
                shifts[(p, c)] = rng.permutation(np.linspace(-magnitude, magnitude, cards[p]))
                weights[k] = magnitude
            else:
                weights[k] = magnitude * rng.choice(np.array([-1.0, 1.0]))

    noise = _component_rng(root, _NOISE)
    uniforms = noise.random((n, draw.d))
    gaussians = noise.standard_normal((n, draw.d))
    values = np.zeros((n, draw.d))
    for c in draw.topo_order.tolist():
        if is_categorical[c]:
            row = np.zeros(n, dtype=np.int64)
            for p in parents[c].tolist():
                row = row * cards[p] + values[:, p].astype(np.int64)
            cdf = np.cumsum(tables[c][row], axis=1)
            values[:, c] = np.minimum((uniforms[:, [c]] > cdf).sum(axis=1), cards[c] - 1)
            continue
        column = gaussians[:, c].copy()
        for k in np.flatnonzero(draw.children == c).tolist():
            p = int(draw.parents[k])
            if is_categorical[p]:
                column += shifts[(p, c)][values[:, p].astype(np.int64)]
            else:
                column += weights[k] * values[:, p]
        values[:, c] = column

    params = SCMParams(
        edges=np.stack([draw.parents, draw.children], axis=1),
        weights=weights,
        noise_scales=np.where(is_categorical, np.nan, 1.0),
        functional="conditional_linear_gaussian",
        noise="gaussian",
    )
    columns = np.flatnonzero(is_categorical)
    provenance = {
        "root_entropy": str(root.entropy),
        "spawn_key": list(root.spawn_key),
        "seed_scheme": _SEED_SCHEME,
        "config": {
            "graph": {
                "model": spec.name,
                "num_nodes": draw.d,
                "density": spec.density,
                "params": spec.params,
            },
            "functional": "conditional_linear_gaussian",
            "categorical": columns.tolist(),
            "categories": cards[columns].tolist(),
            "concentration": float(concentration),
            "n": int(n),
        },
        "versions": {"andrey": andrey.__version__, "numpy": np.__version__},
    }
    return CausalDataset(
        data=values, graph=truth, report=QAReport(scale="raw"), provenance=provenance, params=params
    )
