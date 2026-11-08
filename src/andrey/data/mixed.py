"""Mixed networks: categorical and continuous variables in a conditional linear Gaussian network.

:func:`sample_mixed` draws a DAG and makes its first variables in causal order categorical, so no
categorical variable has a continuous parent, as a conditional linear Gaussian network requires. A
categorical variable has a conditional probability table, one Dirichlet-distributed row per
configuration of its parents, as in :func:`~andrey.data.sample_discrete`. A continuous variable is a
linear function of its continuous parents, plus a shift for each categorical parent's category, plus
standard Gaussian noise. The data are the input of the mixed-data tests and scores
(``indep_test="degen_gauss"`` or ``"cond_gauss"``, ``score_func="degen_gauss"`` or
``"cond_gauss"``).

As in :mod:`andrey.data.scm`, the graph, the parameters, and the samples each come from their own
stream of the seed, so a change to one leaves the others' draws unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

import numpy as np

from .dataset import CausalDataset, QAReport, SCMParams
from .discrete import _parents_of, check_cpt_args, draw_categories, intervened_table
from .graphs import DAGDraw, dag_truth
from .interventions import (
    HardIntervention,
    Intervention,
    check_columns,
    check_continuous_soft,
    continuous_hard_values,
    env_labels,
    env_targets,
    pooled_rows,
    record,
)
from .scm import (
    GRAPH,
    INTERVENTION,
    NOISE,
    WEIGHTS,
    column_order,
    component_rng,
    dataset_provenance,
    named_graph,
)

__all__ = ["sample_mixed"]

# Magnitudes of a continuous parent's weight and of a categorical parent's shifts.
_WEIGHT_RANGE = (0.5, 2.0)


def sample_mixed(
    *,
    graph: str = "erdos_renyi",
    d: int,
    n: int | Sequence[int],
    seed: int,
    model_seed: int | None = None,
    density: float = 2.0,
    categorical_share: float = 0.5,
    categories: tuple[int, int] = (2, 3),
    concentration: float = 1.0,
    shuffle: bool = True,
    envs: Sequence[Mapping[int, Intervention | float]] | None = None,
) -> CausalDataset:
    """Sample a conditional linear Gaussian network of categorical and continuous variables.

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
        The seed of the model: the graph, the categories, the tables, the weights and shifts, the
        column order, and the tables, weights, and shifts ``soft()`` draws anew. ``seed`` then
        draws only the rows. Samples with one ``model_seed`` and different seeds are datasets of
        one network. ``None`` takes ``seed``, as ``model_seed=seed`` does.
    density : float, default=2.0
        Mean total degree (``2 * |E| / d``) of the graph.
    categorical_share : float, default=0.5
        The share of the variables that are categorical, from ``0`` to ``1``: the first
        ``round(categorical_share * d)`` in causal order.
    categories : tuple of int, default=(2, 3)
        The least and the most categories a categorical variable has; each one's count is drawn
        uniformly between them, both included. The least is at least 2.
    concentration : float, default=1.0
        Concentration of the symmetric Dirichlet each table row is drawn from, as for
        :func:`~andrey.data.sample_discrete`.
    shuffle : bool, default=True
        Shuffle the columns with a permutation from the seed's own column stream, and relabel the
        true graph, the categorical columns, and the parameters to match, as for
        :func:`~andrey.data.sample_scm`. ``provenance["column_order"][j]`` is the index column
        ``j`` has with ``shuffle=False``.
    envs : sequence of mappings of int to intervention or float, or None, default=None
        One mapping per environment, ``{column: intervention}``, as for
        :meth:`SCM.sample <andrey.data.SCM.sample>`; ``{}`` is an environment without
        interventions. On a categorical column an intervention acts as for
        :func:`~andrey.data.sample_discrete`: ``hard(category)``, ``hard(probs=...)``, and
        ``hard()`` set its distribution, and ``soft()`` draws its table anew. On a continuous
        column it acts as for :meth:`SCM.sample <andrey.data.SCM.sample>`: ``hard(value, sd=...)``
        sets its values, and ``soft`` changes its noise and the effects of its parents (see
        Notes). A number stands for ``hard(number)``. Every environment shares the graph and the
        parameters, and draws its own samples; environment 0 with ``n`` rows draws those a sample
        without ``envs`` draws. The rows come pooled, environment 0 first, with ``env`` and
        ``targets`` set.

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
        a continuous variable and ``nan`` for a categorical one. Both hold the network without
        interventions.

    Raises
    ------
    ValueError
        If ``categorical_share`` is not in ``[0, 1]``, ``categories`` is not a pair with
        ``2 <= least <= most``, ``concentration`` is not positive, or ``graph`` names no graph; or,
        with ``envs``, if ``n`` does not give each environment at least one row, a target is not a
        column, an intervention on a categorical column is not one
        :func:`~andrey.data.sample_discrete` takes, or one on a continuous column is ``hard()``
        without a value or ``soft()`` without an argument.

    Notes
    -----
    A continuous variable ``x`` with continuous parents ``p`` and categorical parents ``c`` is
    ``x = sum_p w_p * p + sum_c s_c[category of c] + e``, with ``e`` standard Gaussian. Each
    ``|w_p|`` is uniform on ``[0.5, 2]`` with a random sign. The shifts ``s_c`` of a parent with
    ``k`` categories are ``k`` evenly spaced values from ``-m`` to ``m``, in a random order, with
    ``m`` uniform on ``[0.5, 2]``, so every category moves ``x`` by a different amount.

    ``soft(shift=a, noise_scale=b, weight_scale=c)`` on a continuous variable multiplies each
    ``w_p`` and each ``s_c`` by ``c`` and replaces ``e`` with ``b * e + a``; a pair ``b`` multiplies
    the noise of each row by its own draw, as for :meth:`SCM.sample <andrey.data.SCM.sample>`.
    ``soft(weights="redraw")`` draws each ``w_p`` and each ``s_c`` anew, as above.

    Examples
    --------
    >>> from andrey.data import sample_mixed
    >>> ds = sample_mixed(d=6, n=200, seed=0)
    >>> ds.data.shape, ds.graph.kind, ds.provenance["config"]["categorical"]
    ((200, 6), 'dag', [1, 2, 5])
    """
    low, high = check_cpt_args(categories, concentration)
    if not 0.0 <= categorical_share <= 1.0:
        raise ValueError(f"categorical_share must be from 0 to 1, got {categorical_share}")
    rows, pooled = pooled_rows(envs, n)
    spec = named_graph(graph, d, density)
    root = np.random.SeedSequence(seed)
    model = root if model_seed is None else np.random.SeedSequence(model_seed)
    _, draw = spec.materialize(component_rng(model, GRAPH))
    is_categorical = np.zeros(draw.d, dtype=bool)
    is_categorical[draw.topo_order[: round(categorical_share * draw.d)]] = True

    rng = component_rng(model, WEIGHTS)
    cards = np.where(is_categorical, rng.integers(low, high + 1, size=draw.d), 0)
    parents = _parents_of(draw)
    tables: dict[int, np.ndarray] = {}
    shifts: dict[int, np.ndarray] = {}  # by edge, an index into draw.parents
    weights = np.full(draw.parents.shape[0], np.nan)
    for c in range(draw.d):
        if is_categorical[c]:
            rows_of_table = int(np.prod(cards[parents[c]], dtype=np.int64))
            tables[c] = rng.dirichlet(
                np.full(int(cards[c]), float(concentration)), size=rows_of_table
            )
            continue
        for e in np.flatnonzero(draw.children == c).tolist():
            weights[e], shift = _draw_effect(rng, int(cards[draw.parents[e]]))
            if shift is not None:
                shifts[e] = shift
    network = _Network(draw, parents, is_categorical, cards, tables, weights, shifts)
    order = column_order(model, draw.d, shuffle)  # the drawn node of each column

    if pooled is None:
        values = network.sample(rows, component_rng(root, NOISE))
    else:
        check_columns(pooled, draw.d)
        values = np.concatenate(
            [
                network.intervened(
                    environment,
                    size,
                    order=order,
                    concentration=concentration,
                    draws=component_rng(root, INTERVENTION, k),
                    redraws=component_rng(model, WEIGHTS, k),
                ).sample(
                    size,
                    component_rng(root, NOISE) if k == 0 else component_rng(root, NOISE, k),
                )
                for k, (environment, size) in enumerate(pooled)
            ]
        )

    edges = np.argsort(order)[np.stack([draw.parents, draw.children], axis=1)]
    is_categorical, cards = is_categorical[order], cards[order]
    params = SCMParams(
        edges=edges,
        weights=weights,
        noise_scales=np.where(is_categorical, np.nan, 1.0),
        functional="conditional_linear_gaussian",
        noise="gaussian",
    )
    columns = np.flatnonzero(is_categorical)
    provenance = dataset_provenance(
        root,
        model,
        order,
        spec,
        draw.d,
        {
            "functional": "conditional_linear_gaussian",
            "categorical": columns.tolist(),
            "categories": cards[columns].tolist(),
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


def _draw_effect(rng: np.random.Generator, k: int) -> tuple[float, np.ndarray | None]:
    """Draw the effect of one parent on a continuous variable: its weight and its shifts.

    A categorical parent of ``k`` categories has the largest shift as its weight; a continuous
    parent has ``k = 0``, a weight with a random sign, and no shifts.
    """
    magnitude = rng.uniform(*_WEIGHT_RANGE)
    if k:
        return magnitude, rng.permutation(np.linspace(-magnitude, magnitude, k))
    return magnitude * rng.choice(np.array([-1.0, 1.0])), None


@dataclass(frozen=True)
class _Network:
    """A drawn network, with the changes of one environment's interventions.

    ``tables`` holds each categorical node's table, ``weights`` each edge's weight, an edge being an
    index into ``draw.parents``, and ``shifts`` the shifts of each edge from a categorical parent.
    ``fixed`` holds the values a ``hard`` intervention sets a continuous node to, and ``noise`` the
    factor and the shift a ``soft`` one applies to a continuous node's noise.
    """

    draw: DAGDraw
    parents: list[np.ndarray]
    is_categorical: np.ndarray
    cards: np.ndarray
    tables: dict[int, np.ndarray]
    weights: np.ndarray
    shifts: dict[int, np.ndarray]
    fixed: dict[int, np.ndarray] = field(default_factory=dict)
    noise: dict[int, tuple[float | np.ndarray, float]] = field(default_factory=dict)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` rows in causal order, one column per drawn node.

        ``rng`` draws an ``(n, d)`` block of uniforms and then one of standard Gaussians; a
        categorical node reads its column of uniforms and a continuous node its column of Gaussians.
        """
        d = self.draw.d
        uniforms = rng.random((n, d))
        gaussians = rng.standard_normal((n, d))
        values = np.zeros((n, d))
        for c in self.draw.topo_order.tolist():
            if c in self.fixed:
                values[:, c] = self.fixed[c]
            elif self.is_categorical[c]:
                own = self.parents[c]
                values[:, c] = draw_categories(
                    self.tables[c],
                    [values[:, p] for p in own.tolist()],
                    self.cards[own],
                    uniforms[:, c],
                )
            else:
                column = gaussians[:, c].copy()
                if c in self.noise:
                    factor, shift = self.noise[c]
                    column = column * factor + shift
                for e in np.flatnonzero(self.draw.children == c).tolist():
                    p = int(self.draw.parents[e])
                    if self.is_categorical[p]:
                        column += self.shifts[e][values[:, p].astype(np.int64)]
                    else:
                        column += self.weights[e] * values[:, p]
                values[:, c] = column
        return values

    def intervened(
        self,
        environment: dict[int, Intervention],
        n: int,
        *,
        order: np.ndarray,
        concentration: float,
        draws: np.random.Generator,
        redraws: np.random.Generator,
    ) -> _Network:
        """Return the network under ``environment``, for ``n`` rows.

        ``environment`` names columns, ``order[j]`` being the node of column ``j``. ``draws`` draws
        the random values of a ``hard`` intervention and the noise factors of a ``soft`` one, and
        ``redraws`` the tables, weights, and shifts a ``soft`` intervention draws anew.
        """
        tables, weights, shifts = dict(self.tables), self.weights.copy(), dict(self.shifts)
        fixed: dict[int, np.ndarray] = {}
        noise: dict[int, tuple[float | np.ndarray, float]] = {}
        for column, intervention in environment.items():
            c = int(order[column])
            if self.is_categorical[c]:
                tables[c] = intervened_table(
                    tables[c], intervention, column, concentration, redraws
                )
                continue
            if isinstance(intervention, HardIntervention):
                fixed[c] = continuous_hard_values(intervention, column, n, draws)
                continue
            check_continuous_soft(intervention, column)
            for e in np.flatnonzero(self.draw.children == c).tolist():
                if intervention.weights == "redraw":
                    weights[e], shift = _draw_effect(redraws, int(self.cards[self.draw.parents[e]]))
                    if shift is not None:
                        shifts[e] = shift
                weights[e] *= intervention.weight_scale
                if e in shifts:
                    shifts[e] = shifts[e] * intervention.weight_scale
            factor = intervention.noise_scale
            if isinstance(factor, tuple):  # one factor per row
                factor = draws.uniform(*factor, size=n)
            noise[c] = (factor, intervention.shift)
        return replace(
            self, tables=tables, weights=weights, shifts=shifts, fixed=fixed, noise=noise
        )
