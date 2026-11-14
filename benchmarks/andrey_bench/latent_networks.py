"""A published network as a linear-Gaussian model with hidden nodes, for the target and latent lanes.

:func:`sample` draws the simulation of the MMB-by-MMB paper (Xie et al., 2024, Sect. 5.1) on any
:class:`~andrey_bench.networks.Network`: the network's DAG is the model's graph; the weight of each
arc has a random sign and a magnitude uniform on ``weight_range`` (the paper's
:math:`U([-1, -0.5] \\cup [0.5, 1])`); every noise term is standard Gaussian; and ``hidden`` nodes,
drawn at random among the nodes that are not targets, leave the data. The paper hides random nodes,
which :func:`andrey.data.latent.latent_order` does not (it hides nodes with two or more children),
so a hidden leaf simply disappears here. With ``hidden = 0`` the model is the plain linear-Gaussian
replay of the network that the methods for causally sufficient data use, and :func:`truth` is the
CPDAG.

Targets are named, such as the paper's own (:data:`PAPER_TARGETS`), or else the nodes with the most
neighbors in the network, ties in file order (:func:`default_targets`); they depend on the network
alone, so every seed of a network has the same targets, and a target is never hidden. The observed columns are in
an order drawn from the seed, with the hidden nodes last in the truth, as every latent dataset of
the harness is. The model (weights, hidden nodes, column order) is a function of the network and
the seed; the rows are a function of the model and ``n``, so ``n = 1000`` and ``n = 5000`` of one
seed are samples of one model.

The data are not rescaled: a variable's variance grows along the paths into it, which the tests
that read correlations do not see, and :func:`sample` reports the largest variance it drew in
``max_variance``.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np

from andrey_bench.networks import Network

#: Magnitude range of an arc's weight; the sign is random.
WEIGHT_RANGE = (0.5, 1.0)

#: Hidden nodes per network in the paper's four: MILDEW, ALARM, WIN95PTS, and ANDES.
PAPER_HIDDEN = {"mildew": 4, "alarm": 4, "win95pts": 6, "andes": 10}

#: The paper's targets in its four networks, by the networks' node names, in the order of its
#: Tables 7-10 (App. E; Tables 1-4 show two of each). The paper writes ``dm_1`` as "dm-1",
#: ``SNode_27`` as "SNode-27", ``RApp3`` as "RApp3(V5)", and ``PrtData`` as "PrData", a name no
#: node has (``PrtData`` is one letter away).
PAPER_TARGETS = {
    "mildew": ("dm_2", "dm_1", "foto_4", "dm_4"),
    "alarm": ("LVEDVOLUME", "VENTTUBE", "CATECHOL", "STROKEVOLUME"),
    "win95pts": ("Problem5", "HrglssDrtnAftrPrnt", "Problem1", "GDIOUT", "PrtData"),
    "andes": ("RApp3", "SNode_27", "SNode_21", "RApp4", "SNode_4", "SNode_47", "SNode_24"),
}

#: The generator a run's records come from; a run recorded with another cannot be rescored.
SAMPLER = "latent_networks.sample, numpy.random.SeedSequence([crc32(network), seed, n])"

# Andrey endpoint marks.
_TAIL, _ARROW = 1, 2


@dataclass(frozen=True)
class LatentSample:
    """One dataset: the rows, the model's DAG over all nodes, and the targets.

    ``dag`` holds the DAG's endpoint marks with the observed nodes first, in column order, and the
    hidden nodes last; ``data`` has the ``d - hidden`` observed columns, named by ``columns``.
    ``weights[i, j]`` is the weight of ``i -> j`` in the same order.
    """

    network: str
    n: int
    seed: int
    data: np.ndarray
    dag: np.ndarray
    weights: np.ndarray
    hidden: int
    columns: tuple[str, ...]
    hidden_names: tuple[str, ...]
    targets: tuple[int, ...]
    max_variance: float

    @property
    def d(self) -> int:
        """Nodes of the model, hidden included."""
        return self.dag.shape[0]


def default_targets(network: Network, k: int = 2) -> tuple[str, ...]:
    """The ``k`` nodes with the most neighbors, parents and children, ties in file order."""
    order = np.argsort(-network.degrees, kind="stable")[:k]
    return tuple(network.nodes[i] for i in order)


def _streams(name: str, seed: int, n: int) -> tuple[np.random.Generator, np.random.Generator]:
    """The model's generator, a function of ``(network, seed)``, and the rows' of ``n`` too."""
    key = zlib.crc32(name.encode())
    return (
        np.random.default_rng(np.random.SeedSequence([key, seed])),
        np.random.default_rng(np.random.SeedSequence([key, seed, n])),
    )


def sample(
    network: Network,
    *,
    n: int,
    seed: int,
    hidden: int,
    weight_range: tuple[float, float] = WEIGHT_RANGE,
    targets: tuple[str, ...] | None = None,
    n_targets: int = 2,
    shuffle: bool = True,
) -> LatentSample:
    """Draw ``n`` rows of ``network``'s linear-Gaussian model with ``hidden`` nodes left out.

    ``targets`` names the target nodes, by default the :func:`default_targets` ``n_targets``. With
    ``shuffle`` off the observed columns keep the file's order.

    Raises ``ValueError`` when more nodes are asked hidden than the non-target nodes hold.
    """
    names = list(network.nodes)
    chosen = tuple(targets) if targets is not None else default_targets(network, n_targets)
    unknown = [t for t in chosen if t not in names]
    if unknown:
        raise ValueError(f"{network.name} has no nodes {unknown}")
    candidates = [i for i, v in enumerate(names) if v not in chosen]
    if not 0 <= hidden <= len(candidates):
        raise ValueError(f"cannot hide {hidden} of the {len(candidates)} nodes that are no target")

    model, noise = _streams(network.name, seed, n)
    low, high = weight_range
    adjacency = network.adjacency
    # Weights first, in arc order, so they do not depend on the hidden count or the shuffle.
    weights = adjacency.astype(np.float64) * (
        model.uniform(low, high, adjacency.shape) * model.choice([-1.0, 1.0], adjacency.shape)
    )
    hide = (
        np.sort(model.choice(candidates, size=hidden, replace=False))
        if hidden
        else np.empty(0, int)
    )
    keep = np.setdiff1d(np.arange(len(names)), hide)
    observed = model.permutation(keep) if shuffle else keep
    layout = np.r_[observed, hide].astype(int)

    weights = weights[np.ix_(layout, layout)]
    d = len(names)
    # X (I - W) = E, so X = E (I - W)^-1, one row per draw.
    full = np.linalg.solve((np.eye(d) - weights).T, noise.standard_normal((n, d)).T).T
    m = d - hidden
    data = np.ascontiguousarray(full[:, :m])

    dag = np.zeros((d, d), dtype=np.int8)
    dag[weights != 0] = _TAIL
    dag.T[weights != 0] = _ARROW
    columns = tuple(names[i] for i in observed)
    return LatentSample(
        network=network.name,
        n=n,
        seed=seed,
        data=data,
        dag=dag,
        weights=weights,
        hidden=hidden,
        columns=columns,
        hidden_names=tuple(names[i] for i in hide),
        targets=tuple(columns.index(t) for t in chosen),
        max_variance=float(full.var(axis=0).max()),
    )


def truth(draw: LatentSample) -> np.ndarray:
    """The endpoint marks the observed columns' methods are scored against.

    The PAG of the observed nodes when ``draw`` hides nodes, else the DAG's CPDAG.
    """
    from andrey import GraphStructure
    from andrey.core import LATENT, OBSERVED
    from andrey.core.orient import dag2cpdag
    from andrey.data.latent import marginal

    if not draw.hidden:
        return np.asarray(dag2cpdag(draw.dag.copy()), dtype=np.int8)
    types = np.full(draw.d, OBSERVED, dtype=np.int8)
    types[draw.d - draw.hidden :] = LATENT
    graph = GraphStructure.from_numpy(draw.dag, kind="dag", node_types=types)
    return np.asarray(marginal(graph, "pag", max_nodes=draw.d).to_numpy(), dtype=np.int8)
