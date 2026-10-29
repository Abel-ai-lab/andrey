"""Discrete Bayesian networks: random conditional probability tables on a random DAG.

:func:`sample_discrete` draws a DAG, gives each node a number of categories and a conditional
probability table with one Dirichlet-distributed row per configuration of its parents, and samples
the network forward in causal order. The data hold each variable's category code, ``0..k-1``, as
``float64``, the input the discrete tests and scores read (``indep_test="chisq"`` or ``"gsq"``,
``score_func="bdeu"``).

As in :mod:`andrey.data.scm`, the graph, the tables, and the samples each come from their own
stream of the seed, so a change to one leaves the others' draws unchanged.
"""

from __future__ import annotations

import numpy as np

import andrey

from .dataset import CausalDataset, QAReport, SCMParams
from .graphs import DAGDraw, dag_truth
from .scm import (
    _GRAPH,
    _NOISE,
    _SEED_SCHEME,
    _WEIGHTS,
    _column_order,
    _component_rng,
    named_graph,
)

__all__ = ["sample_discrete"]


def sample_discrete(
    *,
    graph: str = "erdos_renyi",
    d: int,
    n: int,
    seed: int,
    density: float = 2.0,
    categories: tuple[int, int] = (2, 3),
    concentration: float = 1.0,
    shuffle: bool = True,
) -> CausalDataset:
    """Sample a discrete Bayesian network with random conditional probability tables.

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
    categories : tuple of int, default=(2, 3)
        The least and the most categories a node has; each node's count is drawn uniformly
        between them, both included. The least is at least 2.
    concentration : float, default=1.0
        Concentration of the symmetric Dirichlet each table row is drawn from. ``1.0`` draws rows
        uniformly over the probability simplex; a smaller value draws peakier rows, so the
        variables depend on their parents more strongly.
    shuffle : bool, default=True
        Shuffle the columns with a permutation from the seed's own column stream, and relabel the
        true graph and the categories to match, as for :func:`~andrey.data.sample_scm`.
        ``provenance["column_order"][j]`` is the index column ``j`` has with ``shuffle=False``.

    Returns
    -------
    CausalDataset
        ``data`` holds the category codes, shape ``(n, d)``; ``graph`` is the true DAG;
        ``provenance["config"]["categories"]`` lists each column's number of categories.
        ``params.weights`` and ``params.noise_scales`` are empty: the tables are the parameters.

    Raises
    ------
    ValueError
        If ``categories`` is not a pair with ``2 <= least <= most``, ``concentration`` is not
        positive, or ``graph`` names no graph.

    Examples
    --------
    >>> from andrey.data import sample_discrete
    >>> ds = sample_discrete(d=5, n=200, seed=0)
    >>> ds.data.shape, ds.graph.kind
    ((200, 5), 'dag')
    """
    whole = len(categories) == 2 and all(
        isinstance(k, int | np.integer) and not isinstance(k, bool) for k in categories
    )
    low, high = (int(k) for k in categories) if whole else (0, 0)
    if not 2 <= low <= high:
        raise ValueError(
            f"categories must be (least, most) with 2 <= least <= most, got {categories}"
        )
    if not (np.isfinite(concentration) and concentration > 0):
        raise ValueError(f"concentration must be finite and positive, got {concentration}")
    spec = named_graph(graph, d, density)
    root = np.random.SeedSequence(seed)
    _, draw = spec.materialize(_component_rng(root, _GRAPH))
    table_rng = _component_rng(root, _WEIGHTS)
    cards = table_rng.integers(low, high + 1, size=draw.d)
    parents = _parents_of(draw)
    tables = [
        table_rng.dirichlet(
            np.full(int(cards[c]), float(concentration)),
            size=int(np.prod(cards[parents[c]], dtype=np.int64)),
        )
        for c in range(draw.d)
    ]
    values = _forward_sample(
        draw.topo_order.tolist(), parents, cards, tables, n, _component_rng(root, _NOISE)
    )
    order = _column_order(root, draw.d, shuffle)  # the drawn node of each column
    edges = np.argsort(order)[np.stack([draw.parents, draw.children], axis=1)]
    params = SCMParams(
        edges=edges,
        weights=np.empty(0),
        noise_scales=np.empty(0),
        functional="categorical",
        noise="dirichlet",
    )
    provenance = {
        "root_entropy": str(root.entropy),
        "spawn_key": list(root.spawn_key),
        "seed_scheme": _SEED_SCHEME,
        "column_order": order.tolist(),
        "config": {
            "graph": {
                "model": spec.name,
                "num_nodes": draw.d,
                "density": spec.density,
                "params": spec.params,
            },
            "functional": "categorical",
            "categories": cards[order].tolist(),
            "concentration": float(concentration),
            "n": int(n),
            "shuffle": bool(shuffle),
        },
        "versions": {"andrey": andrey.__version__, "numpy": np.__version__},
    }
    return CausalDataset(
        data=np.take(values, order, axis=1),
        graph=dag_truth(edges[:, 0], edges[:, 1], draw.d),
        report=QAReport(scale="raw"),
        provenance=provenance,
        params=params,
    )


def _parents_of(draw: DAGDraw) -> list[np.ndarray]:
    """Each node's parents, ascending; the first is the most significant digit of a table row."""
    return [np.sort(draw.parents[draw.children == c]) for c in range(draw.d)]


def _forward_sample(
    order: list[int],
    parents: list[np.ndarray],
    cards: np.ndarray,
    tables: list[np.ndarray],
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample ``n`` rows in causal ``order``: each node by inverse CDF from its table row.

    Row ``i`` of a node's table is its parents' configuration ``i`` in mixed radix, the first parent
    in ``parents`` the most significant digit.
    """
    values = np.zeros((n, len(order)), dtype=np.int64)
    uniforms = rng.random((n, len(order)))
    for c in order:
        row = np.zeros(n, dtype=np.int64)
        for p in parents[c].tolist():
            row = row * cards[p] + values[:, p]
        cdf = np.cumsum(tables[c][row], axis=1)
        values[:, c] = np.minimum((uniforms[:, [c]] > cdf).sum(axis=1), cards[c] - 1)
    return values.astype(np.float64)
