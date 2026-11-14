"""Linear-Gaussian models with a selection node, for the Y-structure lane.

The simulations of Versteeg, Zhang, and Mooij (2022, Sect. 4.2 and Appendix D), drawn here by
rejection. A model is a DAG over ``p`` system variables and one context variable, the last column
(the JCI context node of the paper): the context is a source of the graph, and the system
variables are the others. Weights have a random sign and a magnitude uniform on
:data:`WEIGHT_RANGE`; each variable's column of weights is divided by
``sqrt(sum of the squared weights from system variables + 1)``, the paper's rescaling against
accumulating variance, which the authors' code applies without the context's weight in the sum;
every noise term is standard Gaussian. Two datasets come from one model: ``unbiased``, ``n``
plain rows, and ``biased``, ``n`` rows kept when the sum of the selection variables, a few system
variables, falls in :data:`INTERVAL`, drawn in chunks until ``n`` are kept.

:data:`SETTINGS` holds the paper's three: ``fixed``, the 7-node graph of its Figure 5(a) (the edges
of the authors' ``simulate.R``), and ``small`` and ``large``, random graphs with ``p = 8`` and
``p = 16`` system variables (:func:`random_graph`). A random graph follows Appendix D: every ordered pair of the ``p + 1`` nodes has an edge with a
fixed probability, in a causal order that makes it acyclic; a graph with too few colliders (nodes
with two or more parents) is discarded, to raise the share of spurious correlations selection
creates; the context is a source with children, drawn uniformly; and the selection variables come
by one of two rules, :data:`RULES`. ``authors``, the default, is the rule of the authors'
``simulate.R``: the first selection variable is drawn among the colliders and their descendants,
the others among all other nodes, the context included, and a graph needs more than
``min_colliders`` colliders. ``text`` is the rule the paper's text describes: every selection
variable is drawn among the leaves of the colliders' descendants, and a graph needs at least
``min_colliders`` colliders; it is the harder setting. The system columns are then listed in an
order drawn from the seed, with the truth permuted the same way, and the context stays last.

A model is a function of the setting and the seed; the rows are a function of the model and ``n``.
:attr:`SelectionSample.ancestral` is the truth a claim is scored against: ``ancestral[i, j]`` when
``i`` is a strict ancestor of ``j`` in the model, over all columns.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np

#: Magnitude range of an edge weight; the sign is random.
WEIGHT_RANGE = (0.5, 1.5)

#: The sum of the selection variables a row must fall in (open), the paper's ``[2, 2.5]``.
INTERVAL = (2.0, 2.5)

#: The rules that draw a random graph's selection variables (:func:`random_graph`).
RULES = ("authors", "text")

#: The generator a run's records come from; a run recorded with another cannot be rescored.
SAMPLER = (
    "selection_models.sample(rule), numpy.random.SeedSequence([crc32(setting), seed, n]), "
    "selection variables by the rule"
)

#: Draws of a random graph before giving up, as the authors' code does, and the biased draw's
#: chunks: ``CHUNK_FACTOR * n`` rows each, at most ``MAX_CHUNKS`` of them.
MAX_GRAPHS = 100_000
CHUNK_FACTOR = 20
MAX_CHUNKS = 1000


@dataclass(frozen=True)
class Setting:
    """One of the paper's three simulation settings."""

    system: int  # system variables, without the context
    edge_prob: float | None  # None: the fixed graph
    min_colliders: int
    selection: int  # selection variables


SETTINGS: dict[str, Setting] = {
    "fixed": Setting(system=6, edge_prob=None, min_colliders=0, selection=2),
    "small": Setting(system=8, edge_prob=0.15, min_colliders=3, selection=1),
    "large": Setting(system=16, edge_prob=0.09, min_colliders=5, selection=3),
}

#: The fixed graph's edges ``(cause, effect)`` over columns 0 to 5 and the context 6, from the
#: authors' ``simulate.R`` (its 1-based ``X1`` to ``X6`` and ``C``), and its selection variables.
FIXED_EDGES = ((6, 0), (6, 3), (5, 3), (3, 4), (2, 1))
FIXED_SELECTION = (0, 1)


@dataclass(frozen=True)
class SelectionGraph:
    """A model's graph: ``adjacency[i, j] = 1`` for ``i -> j`` over the system columns and the
    context, last, and the columns whose sum is selected."""

    adjacency: np.ndarray
    selection: tuple[int, ...]

    @property
    def context(self) -> int:
        return self.adjacency.shape[0] - 1


@dataclass(frozen=True)
class SelectionSample:
    """One model's two datasets, its graph, and the truth claims are scored against."""

    setting: str
    n: int
    seed: int
    biased: np.ndarray
    unbiased: np.ndarray
    graph: SelectionGraph
    weights: np.ndarray
    ancestral: np.ndarray
    acceptance: float

    @property
    def system(self) -> np.ndarray:
        """The system columns: every column but the context."""
        return np.arange(self.graph.adjacency.shape[0] - 1)


def ancestral_matrix(adjacency: np.ndarray) -> np.ndarray:
    """``out[i, j]`` for a directed path from ``i`` to ``j`` of one or more edges."""
    reach = np.asarray(adjacency, dtype=bool)
    steps = max(1, int(np.ceil(np.log2(max(reach.shape[0], 2)))) + 1)
    for _ in range(steps):  # squaring doubles the path length the closure covers
        reach = reach | ((reach.astype(np.int64) @ reach.astype(np.int64)) > 0)
    return reach


def fixed_graph() -> SelectionGraph:
    """The 7-node graph of the paper's Figure 5(a), the context last."""
    adjacency = np.zeros((7, 7), dtype=np.int8)
    for cause, effect in FIXED_EDGES:
        adjacency[cause, effect] = 1
    return SelectionGraph(adjacency, FIXED_SELECTION)


def random_graph(
    setting: Setting, rng: np.random.Generator, rule: str = "authors"
) -> SelectionGraph:
    """A random graph of ``setting``, its selection variables drawn by ``rule`` (:data:`RULES`).

    Raises ``RuntimeError`` when :data:`MAX_GRAPHS` draws hold no graph of the setting.
    """
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}, not {rule!r}")
    nodes = setting.system + 1
    upper = np.triu(np.ones((nodes, nodes), dtype=bool), 1)
    for _ in range(MAX_GRAPHS):
        adjacency = (rng.random((nodes, nodes)) < setting.edge_prob) & upper
        colliders = np.flatnonzero(adjacency.sum(0) >= 2)
        sources = np.flatnonzero((adjacency.sum(0) == 0) & (adjacency.sum(1) > 0))
        enough = (
            len(colliders) > setting.min_colliders
            if rule == "authors"
            else (len(colliders) >= setting.min_colliders)
        )
        if not enough or not len(sources):
            continue
        if rule == "text":
            below = ancestral_matrix(adjacency)[colliders].any(0) | np.isin(
                np.arange(nodes), colliders
            )
            leaves = np.flatnonzero(below & (adjacency.sum(1) == 0))
            if len(leaves) < setting.selection:
                continue
        context = int(rng.choice(sources))
        # The context moves to the last column; the other nodes keep their order.
        order = np.r_[[i for i in range(nodes) if i != context], context]
        position = {int(old): new for new, old in enumerate(order)}
        moved = adjacency[np.ix_(order, order)].astype(np.int8)
        if rule == "text":
            selection = np.sort(rng.choice(leaves, size=setting.selection, replace=False))
            return SelectionGraph(moved, tuple(position[int(i)] for i in selection))
        colliders = np.flatnonzero(moved.sum(0) >= 2)
        below = ancestral_matrix(moved)[colliders].any(0) | np.isin(np.arange(nodes), colliders)
        first = int(rng.choice(np.flatnonzero(below)))
        rest = [i for i in range(nodes) if i != first]
        chosen = rng.choice(rest, size=setting.selection - 1, replace=False)
        return SelectionGraph(moved, tuple(sorted([first, *(int(i) for i in chosen)])))
    raise RuntimeError(f"no graph of {setting} in {MAX_GRAPHS} draws")


def _streams(name: str, seed: int, n: int) -> tuple[np.random.Generator, ...]:
    """The model's generator, then the biased and the unbiased rows' generators."""
    key = zlib.crc32(name.encode())
    return (
        np.random.default_rng(np.random.SeedSequence([key, seed])),
        np.random.default_rng(np.random.SeedSequence([key, seed, n, 1])),
        np.random.default_rng(np.random.SeedSequence([key, seed, n, 2])),
    )


def sample(
    setting: str, *, n: int, seed: int, shuffle: bool = True, rule: str = "authors"
) -> SelectionSample:
    """Draw one model of ``setting`` and its biased and unbiased datasets of ``n`` rows, a random
    graph's selection variables by ``rule`` (:data:`RULES`).

    Raises ``RuntimeError`` when :data:`MAX_CHUNKS` chunks of the biased draw keep fewer than ``n``
    rows: a selection interval the model rarely reaches.
    """
    spec = SETTINGS[setting]
    model, biased_rows, unbiased_rows = _streams(setting, seed, n)
    graph = fixed_graph() if spec.edge_prob is None else random_graph(spec, model, rule)
    p = graph.adjacency.shape[0]
    system = p - 1

    # The draws below come before the shuffle, so they do not depend on it.
    low, high = WEIGHT_RANGE
    magnitude = low + (high - low) * model.random((p, p))
    weights = graph.adjacency * magnitude * model.choice([-1.0, 1.0], (p, p))
    scale = np.sqrt((weights[:system] ** 2).sum(0) + 1.0)
    weights = weights / scale

    order = np.r_[model.permutation(system) if shuffle else np.arange(system), system]
    inverse = np.argsort(order)
    adjacency = graph.adjacency[np.ix_(order, order)]
    weights = weights[np.ix_(order, order)]
    graph = SelectionGraph(adjacency, tuple(sorted(int(inverse[i]) for i in graph.selection)))

    inverse_map = np.linalg.inv(np.eye(p) - weights)
    unbiased = unbiased_rows.standard_normal((n, p)) @ inverse_map

    kept, seen, drawn = [], 0, 0
    for _ in range(MAX_CHUNKS):
        chunk = biased_rows.standard_normal((CHUNK_FACTOR * n, p)) @ inverse_map
        total = chunk[:, list(graph.selection)].sum(1)
        keep = chunk[(total > INTERVAL[0]) & (total < INTERVAL[1])]
        kept.append(keep)
        seen += len(keep)
        drawn += len(chunk)
        if seen >= n:
            break
    if seen < n:
        raise RuntimeError(f"{setting} seed {seed}: {seen} of {n} rows kept in {drawn} draws")
    return SelectionSample(
        setting=setting,
        n=n,
        seed=seed,
        biased=np.ascontiguousarray(np.concatenate(kept)[:n]),
        unbiased=np.ascontiguousarray(unbiased),
        graph=graph,
        weights=weights,
        ancestral=ancestral_matrix(adjacency),
        acceptance=seen / drawn,
    )
