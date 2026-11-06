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

from collections.abc import Mapping, Sequence

import numpy as np

from .dataset import CausalDataset, QAReport, SCMParams
from .graphs import DAGDraw, dag_truth
from .interventions import (
    HardIntervention,
    Intervention,
    SoftIntervention,
    check_columns,
    env_labels,
    env_targets,
    pooled_rows,
    record,
)
from .scm import (
    _GRAPH,
    _INTERVENTION,
    _NOISE,
    _WEIGHTS,
    _column_order,
    _component_rng,
    dataset_provenance,
    named_graph,
)

__all__ = ["sample_discrete"]


def sample_discrete(
    *,
    graph: str = "erdos_renyi",
    d: int,
    n: int | Sequence[int],
    seed: int,
    model_seed: int | None = None,
    density: float = 2.0,
    categories: tuple[int, int] = (2, 3),
    concentration: float = 1.0,
    shuffle: bool = True,
    envs: Sequence[Mapping[int, Intervention | float]] | None = None,
) -> CausalDataset:
    """Sample a discrete Bayesian network with random conditional probability tables.

    Parameters
    ----------
    graph : {"erdos_renyi", "scale_free", "small_world", "hub"}, default="erdos_renyi"
        The random graph the DAG is drawn from, as for :func:`~andrey.data.sample_scm`.
    d : int
        Node count.
    n : int or sequence of int
        Sample count. With ``envs``, the rows of each environment: ``n`` each, or ``n[k]`` for
        environment ``k``.
    seed : int
        Root seed.
    model_seed : int or None, default=None
        The seed of the model: the graph, the categories, the tables, the column order, and the
        tables ``soft()`` draws anew. ``seed`` then draws only the rows. Samples with one
        ``model_seed`` and different seeds are datasets of one network. ``None`` takes ``seed``,
        as ``model_seed=seed`` does.
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
    envs : sequence of mappings of int to intervention or int, or None, default=None
        One mapping per environment, ``{column: intervention}``, as for
        :meth:`SCM.sample <andrey.data.SCM.sample>`; ``{}`` is an environment without
        interventions. ``hard(category)`` sets a column to one category, ``hard(probs=...)``
        draws it from one distribution for every configuration of its parents, and ``hard()``
        from the uniform one; a whole number stands for ``hard(number)``. ``soft()`` draws every
        row of the column's table anew, from the same Dirichlet. Every environment shares the
        graph and the tables, and draws its own samples; environment 0 draws those a sample
        without ``envs`` draws. The rows come pooled, environment 0 first, with ``env`` and
        ``targets`` set.

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
        positive, or ``graph`` names no graph; or, with ``envs``, if ``n`` does not give each
        environment at least one row, a target is not a column, a category is not one of the
        column's, ``probs`` does not have one entry per category, or an intervention takes the
        arguments of a continuous variable.

    Examples
    --------
    >>> from andrey.data import sample_discrete
    >>> ds = sample_discrete(d=5, n=200, seed=0)
    >>> ds.data.shape, ds.graph.kind
    ((200, 5), 'dag')
    """
    low, high = check_cpt_args(categories, concentration)
    rows, pooled = pooled_rows(envs, n)
    spec = named_graph(graph, d, density)
    root = np.random.SeedSequence(seed)
    model = root if model_seed is None else np.random.SeedSequence(model_seed)
    _, draw = spec.materialize(_component_rng(model, _GRAPH))
    table_rng = _component_rng(model, _WEIGHTS)
    cards = table_rng.integers(low, high + 1, size=draw.d)
    parents = _parents_of(draw)
    tables = [
        table_rng.dirichlet(
            np.full(int(cards[c]), float(concentration)),
            size=int(np.prod(cards[parents[c]], dtype=np.int64)),
        )
        for c in range(draw.d)
    ]
    order = _column_order(model, draw.d, shuffle)  # the drawn node of each column
    if pooled is None:
        values = _forward_sample(
            draw.topo_order.tolist(), parents, cards, tables, rows, _component_rng(root, _NOISE)
        )
    else:
        check_columns(pooled, draw.d)
        values = np.concatenate(
            [
                _forward_sample(
                    draw.topo_order.tolist(),
                    parents,
                    cards,
                    intervened_tables(
                        tables,
                        interventions,
                        order,
                        concentration,
                        _component_rng(model, _INTERVENTION, k),
                    ),
                    size,
                    _component_rng(root, _NOISE) if k == 0 else _component_rng(root, _NOISE, k),
                )
                for k, (interventions, size) in enumerate(pooled)
            ]
        )
    edges = np.argsort(order)[np.stack([draw.parents, draw.children], axis=1)]
    params = SCMParams(
        edges=edges,
        weights=np.empty(0),
        noise_scales=np.empty(0),
        functional="categorical",
        noise="dirichlet",
    )
    provenance = dataset_provenance(
        root,
        model,
        order,
        spec,
        draw.d,
        {
            "functional": "categorical",
            "categories": cards[order].tolist(),
            "concentration": float(concentration),
            "n": rows,
            "shuffle": bool(shuffle),
            **({} if pooled is None else {"envs": record(pooled)}),
        },
    )
    return CausalDataset(
        data=np.take(values, order, axis=1),
        graph=dag_truth(edges[:, 0], edges[:, 1], draw.d),
        report=QAReport(scale="raw"),
        provenance=provenance,
        params=params,
        env=None if pooled is None else env_labels(pooled),
        targets=None if pooled is None else env_targets(pooled),
    )


def check_cpt_args(categories: tuple[int, int], concentration: float) -> tuple[int, int]:
    """Return the least and the most categories of ``categories``.

    Raises ``ValueError`` unless ``categories`` is a pair of whole numbers with
    ``2 <= least <= most`` and ``concentration`` is finite and positive.
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
    return low, high


def intervened_tables(
    tables: list[np.ndarray],
    spec: Mapping[int, Intervention],
    order: np.ndarray,
    concentration: float,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Each node's conditional probability table under one environment's interventions.

    ``spec`` names columns; ``order[j]`` is the node of column ``j``.
    """
    tables = list(tables)
    for column, intervention in spec.items():
        c = int(order[column])
        tables[c] = intervened_table(tables[c], intervention, column, concentration, rng)
    return tables


def intervened_table(
    table: np.ndarray,
    intervention: Intervention,
    column: int,
    concentration: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a node's conditional probability table under an intervention on its ``column``.

    A ``hard`` intervention gives every configuration of the parents one row; ``soft()`` draws
    every row anew from ``rng``, from the symmetric Dirichlet of ``concentration``.
    """
    rows, k = table.shape
    if isinstance(intervention, SoftIntervention):
        if intervention != SoftIntervention():
            raise ValueError(
                f"column {column}: soft() on a discrete variable takes no arguments; it draws "
                "the variable's table anew"
            )
        return rng.dirichlet(np.full(k, float(concentration)), size=rows)
    return np.tile(categorical_probs(intervention, column, k), (rows, 1))


def categorical_probs(intervention: HardIntervention, column: int, k: int) -> np.ndarray:
    """Return the distribution a ``hard`` intervention sets a column of ``k`` categories to."""
    if intervention.sd:
        raise ValueError(f"column {column}: hard() on a discrete variable takes no sd")
    if intervention.probs is not None:
        if len(intervention.probs) != k:
            raise ValueError(
                f"column {column} has {k} categories; probs has {len(intervention.probs)} entries"
            )
        return np.asarray(intervention.probs)
    if intervention.value is None:
        return np.full(k, 1.0 / k)
    if not (float(intervention.value).is_integer() and 0 <= intervention.value < k):
        raise ValueError(
            f"column {column} has the categories 0..{k - 1}; hard() got {intervention.value}"
        )
    return np.eye(k)[int(intervention.value)]


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
        own = parents[c]
        values[:, c] = draw_categories(
            tables[c], [values[:, p] for p in own.tolist()], cards[own], uniforms[:, c]
        )
    return values.astype(np.float64)


def draw_categories(
    table: np.ndarray,
    parents: Sequence[np.ndarray],
    cards: Sequence[int] | np.ndarray,
    uniform: np.ndarray,
) -> np.ndarray:
    """Draw one node's category in each row by inverse CDF of ``uniform`` from its table row.

    ``parents`` holds the parents' category codes and ``cards`` their numbers of categories. Row
    ``i`` of ``table`` is the parents' configuration ``i`` in mixed radix, the first parent the most
    significant digit.
    """
    row = np.zeros(uniform.shape[0], dtype=np.int64)
    for values, k in zip(parents, cards, strict=True):
        row = row * k + values.astype(np.int64, copy=False)
    cdf = np.cumsum(table[row], axis=1)
    return np.minimum((uniform[:, None] > cdf).sum(axis=1), table.shape[1] - 1)
