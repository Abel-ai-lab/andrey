"""The planted models of the adjustment lane, each as the paper of its method describes it.

A model is a linear-Gaussian structural model (or, for the discrete cells, a table sampler) with the
treatment-outcome pairs the method is asked about. The module draws no data by itself: a model is
a function of its setting, its parameters, and the seed, and its rows are a function of the model,
the sample size, and a stream, so the *fresh sample* the effect estimates are made on
(:mod:`andrey_bench.adjustment_lane`) comes from the same structural model as the data the methods
see.

Conventions every generator follows, because the scorer reads them:

- The columns are the observed variables in an order drawn from the seed, never a causal order; the
  hidden variables come after them, so position ``i < p`` is column ``i`` of the data and a position
  ``>= p`` is hidden. :attr:`LinearModel.dag` is the true DAG over all the nodes in that order.
- ``weights[i, j]`` is the weight of the arc ``i -> j``; ``x = (I - B^T)^{-1} e`` with independent
  noise ``e``. The true effect of a pair is read from the model, in population-standardized units:
  the total effect is ``[(I - B)^{-1}][x, y]``, the direct effect ``B[x, y]``, each times
  ``sd(x) / sd(y)`` under the model's covariance, hidden variables included. The effect estimates
  of the lane standardize with the fresh sample's standard deviations.
- A setting is registered in :data:`SETTINGS` with its paper, its estimand, and its default
  parameters; a cell of the lane is ``setting@n,key=value,...`` (:func:`parse_cell`).

Where a paper leaves a constant open the generator uses the smallest valid value and says so in its
docstring; where the authors' code differs from the paper, the paper is followed and the difference
is named.
"""

from __future__ import annotations

import ast
import zlib
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

#: The generator a run's records come from; a run recorded with another cannot be rescored.
SAMPLER = "adjustment_models, numpy.random.SeedSequence([crc32(setting), seed, stream, n])"

#: The streams of a model: the model itself, the data the methods see, and the fresh sample.
MODEL, DATA, FRESH = 0, 1, 2

#: Most random graphs a generator draws before it gives up on a setting that asks for a property.
MAX_TRIES = 10_000


@dataclass(frozen=True)
class Pair:
    """A treatment-outcome pair, by column, and the effect it asks about."""

    treatment: int
    outcome: int
    estimand: str = "total"


def rng_for(setting: str, seed: int, stream: int, n: int = 0) -> np.random.Generator:
    """The generator of ``stream`` for ``setting``, ``seed`` and ``n``."""
    return np.random.default_rng(
        np.random.SeedSequence([zlib.crc32(setting.encode()), seed, stream, n])
    )


@dataclass(frozen=True)
class LinearModel:
    """A linear-Gaussian structural model with hidden variables and the pairs asked about.

    ``weights`` and ``noise_sd`` cover every node, the ``p`` observed ones first in column order and
    the hidden ones after them. ``intercepts`` (default zero) shift the means and change no effect.
    ``meta`` holds what the setting records about its draw (roles of the variables, for example).
    """

    setting: str
    seed: int
    weights: np.ndarray
    noise_sd: np.ndarray
    p: int
    pairs: tuple[Pair, ...]
    intercepts: np.ndarray | None = None
    params: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    #: ``"gaussian"``, or ``"binary"`` for the logistic model: ``weights`` are then the logistic
    #: coefficients, the rows are 0/1, and no linear effect is defined.
    family: str = "gaussian"
    #: The conditional-independence test the lane gives the methods for these rows.
    indep_test: str = "fisherz"

    @property
    def discrete(self) -> bool:
        """Whether the rows are categories, so no effect is estimated by least squares."""
        return self.family != "gaussian"

    @property
    def d(self) -> int:
        """The number of nodes, hidden ones included."""
        return self.weights.shape[0]

    @property
    def hidden(self) -> tuple[int, ...]:
        """The positions of the hidden nodes."""
        return tuple(range(self.p, self.d))

    @property
    def dag(self) -> np.ndarray:
        """The true DAG as a directed ``int8`` adjacency over every node."""
        return (self.weights != 0).astype(np.int8)

    def rows(self, n: int, stream: int) -> np.ndarray:
        """``n`` rows of the observed columns from ``stream`` (:data:`DATA` or :data:`FRESH`)."""
        rng = rng_for(self.setting, self.seed, stream, n)
        if self.family == "binary":
            return self._binary_rows(n, rng)
        noise = rng.standard_normal((n, self.d)) * self.noise_sd
        if self.intercepts is not None:
            noise = (
                noise + self.intercepts
            )  # x = (c + e)(I - B)^{-1}: a node's intercept is its own
        return (noise @ np.linalg.inv(np.eye(self.d) - self.weights))[:, : self.p]

    def _binary_rows(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Rows of the logistic model, drawn node by node along a causal order: ``P(V_i = 1) =
        1 / (1 + exp(-sum_j a_ji V_j))``, no intercept, Loranchet and Assaad (2026), App. D.1."""
        parents = self.weights != 0
        order, done = [], np.zeros(self.d, dtype=bool)
        while len(order) < self.d:
            ready = [v for v in range(self.d) if not done[v] and done[parents[:, v]].all()]
            order.extend(ready)
            done[ready] = True
        rows = np.zeros((n, self.d))
        for v in order:
            prob = 1.0 / (1.0 + np.exp(-rows @ self.weights[:, v]))
            rows[:, v] = rng.random(n) < prob
        return rows[:, : self.p]

    def covariance(self) -> np.ndarray:
        """The population covariance of every node."""
        inv = np.linalg.inv(np.eye(self.d) - self.weights)
        return inv.T @ np.diag(self.noise_sd**2) @ inv

    def true_effect(self, x: int, y: int, estimand: str) -> float:
        """The population-standardized effect of ``x`` on ``y``."""
        if estimand == "total":
            raw = np.linalg.inv(np.eye(self.d) - self.weights)[x, y]
        else:
            raw = self.weights[x, y]
        var = np.diag(self.covariance())
        return float(raw * np.sqrt(var[x] / var[y]))

    def effects(self) -> dict[tuple[int, int], float | None]:
        """The true effect of every ordered pair the lane can ask about: each pair and its reverse,
        in the pair's estimand; none for a discrete model."""
        out: dict[tuple[int, int], float | None] = {}
        if self.discrete:
            return out
        for pair in self.pairs:
            for x, y in ((pair.treatment, pair.outcome), (pair.outcome, pair.treatment)):
                out[(x, y)] = self.true_effect(x, y, pair.estimand)
        return out


def finish(
    setting: str,
    seed: int,
    weights: np.ndarray,
    noise_sd: np.ndarray,
    hidden: list[int],
    pairs: list[tuple[int, int, str]],
    *,
    params: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
    intercepts: np.ndarray | None = None,
    shuffle: bool = True,
) -> LinearModel:
    """A :class:`LinearModel` from a graph in the generator's own node order.

    ``hidden`` lists the hidden nodes and ``pairs`` the ``(treatment, outcome, estimand)`` triples
    in the generator's order. The observed nodes are put in an order drawn from the seed and the
    hidden nodes last; the pairs and every position in ``meta["roles"]`` (a dict of lists of nodes,
    if given) follow the permutation.
    """
    d = weights.shape[0]
    rng = rng_for(setting, seed, MODEL, 1)
    observed = [v for v in range(d) if v not in set(hidden)]
    if shuffle:
        observed = [observed[i] for i in rng.permutation(len(observed))]
    order = [*observed, *hidden]
    place = {old: new for new, old in enumerate(order)}
    meta = dict(meta or {})
    if "names" in meta:
        meta["names"] = [meta["names"][v] for v in order]
    if "roles" in meta:
        meta["roles"] = {k: [place[v] for v in vs] for k, vs in meta["roles"].items()}
    return LinearModel(
        setting=setting,
        seed=seed,
        weights=weights[np.ix_(order, order)],
        noise_sd=noise_sd[order],
        p=len(observed),
        pairs=tuple(Pair(place[x], place[y], est) for x, y, est in pairs),
        intercepts=None if intercepts is None else intercepts[order],
        params=dict(params or {}),
        meta=meta,
    )


def signed_uniform(rng: np.random.Generator, size: Any, low: float, high: float) -> np.ndarray:
    """Draws with a random sign and a magnitude uniform on ``[low, high]``."""
    return rng.uniform(low, high, size) * rng.choice([-1.0, 1.0], size)


# --- Entner, Hoyer and Spirtes (2013) ----------------------------------------------------------


def entner(
    seed: int, *, task: str, w: int = 10, u: int = 5, edge_prob: float = 0.25
) -> LinearModel:
    """Entner, Hoyer, and Spirtes (2013), Sect. 6: pretreatment covariates, some of them hidden.

    A random linear-Gaussian model over the treatment ``x``, the outcome ``y``, ``w`` observed
    covariates (the paper's 10) and ``u`` hidden ones (5), under the paper's partial order: ``y`` is
    no ancestor of ``x``, and ``x`` and ``y`` are no ancestors of any covariate, so ``x`` precedes
    ``y`` and the covariates, hidden ones too, precede both. The four tasks of its Table 1 are
    ``task`` in ``effect-admissible`` (a nonzero effect and an admissible set among the observed
    covariates), ``effect-none`` (a nonzero effect, no admissible set), ``zero-admissible`` and
    ``zero-none``; "admissible" is decided on the graph as the paper does, by whether some set of
    observed covariates is a valid adjustment set (:func:`andrey.metrics.effect_oracle` with
    ``identified_from="dag"``), and models are drawn until one has the task's property.

    The paper does not state the edge density, the weights, the noise, or how a model is brought to
    a task. This generator draws each possible arc with probability ``edge_prob`` (0.25), weights
    with a random sign and a magnitude uniform on ``[0.5, 1]``, and unit noise standard deviations,
    and rejects models that miss the task. The covariates, hidden ones included, come in a random
    causal order before ``x`` and ``y``. Nothing is taken from the authors' code, which is not
    available.
    """
    from andrey.metrics import effect_oracle

    tasks = {
        "effect-admissible": (True, True),
        "effect-none": (True, False),
        "zero-admissible": (False, True),
        "zero-none": (False, False),
    }
    if task not in tasks:
        raise ValueError(f"unknown task {task!r}; the tasks are {sorted(tasks)}")
    want_effect, want_set = tasks[task]
    rng = rng_for("entner", seed, MODEL)
    pre, x, y = w + u, w + u, w + u + 1  # covariates first, then the treatment, then the outcome
    n = pre + 2
    hidden = list(range(w, w + u))
    for _ in range(MAX_TRIES):
        # The covariates, hidden ones too, in a random causal order; both x and y come after them.
        rank = np.empty(pre, dtype=int)
        rank[rng.permutation(pre)] = np.arange(pre)
        arcs = np.zeros((n, n), dtype=bool)
        arcs[:pre, :pre] = (rng.random((pre, pre)) < edge_prob) & (rank[:, None] < rank[None, :])
        arcs[:pre, x] = rng.random(pre) < edge_prob
        arcs[:pre, y] = rng.random(pre) < edge_prob
        arcs[x, y] = want_effect
        record = effect_oracle(
            arcs.astype(np.int8), x, y, set_kind="valid", identified_from="dag", latent=hidden
        )
        if record["identifiable"] == want_set:
            break
    else:
        raise RuntimeError(f"no model of task {task!r} in {MAX_TRIES} draws (seed {seed})")
    weights = arcs * signed_uniform(rng, (n, n), 0.5, 1.0)
    return finish(
        f"entner-{task}",
        seed,
        weights,
        np.ones(n),
        hidden,
        [(x, y, "total")],
        params={"task": task, "w": w, "u": u, "edge_prob": edge_prob},
    )


# --- Loranchet and Assaad (2026): LocPC-CDE ----------------------------------------------------


def locpc(seed: int, *, identifiable: bool, v: int = 50, family: str = "gaussian") -> LinearModel:
    """Loranchet and Assaad (2026), Sect. 7.1 and App. D.1: random DAGs with a pair whose controlled
    direct effect is identifiable, or not, in the true CPDAG.

    An Erdos-Renyi graph on ``v`` nodes with edge probability ``2 / (v - 1)`` (two neighbors on
    average), oriented by a random permutation, redrawn until some pair ``X -> Y`` of the DAG has
    every edge at ``Y`` oriented in its CPDAG (``identifiable``) or at least one unoriented
    (otherwise), the paper's rule; the outcome is the first node of a random order that has such a
    pair, the treatment a random parent of it. Linear Gaussian: arc weights uniform on ``[-1, 1]``
    with magnitude at least 0.2, and noise *variances* uniform on ``[0.8, 1]`` as the paper says (the
    authors' script draws the standard deviations from that range). The binary family (``family=
    "binary"``) is the paper's logistic model, ``P(V_i = 1) = 1 / (1 + exp(-sum_j a_ji V_j))`` with
    ``|a|`` uniform on ``[0.2, 5]`` and a random sign, no intercept, rows drawn along a causal
    order; it has no linear effect, so the lane scores its answers and no effect error. The paper's
    experiment draws 100 DAGs per size with ``v`` from 10 to 200 and n = 5000.
    """
    from andrey.core import adjustment as adj
    from andrey.metrics import to_cpdag

    if family not in ("gaussian", "binary"):
        raise ValueError(f"family must be 'gaussian' or 'binary', got {family!r}")
    rng = rng_for("locpc", seed, MODEL)
    for _ in range(MAX_TRIES):
        order = rng.permutation(v)
        undirected = np.triu(rng.random((v, v)) < 2.0 / (v - 1), 1)
        arcs = np.zeros((v, v), dtype=bool)
        i, j = np.nonzero(undirected)
        lo_first = order[i] < order[j]
        arcs[np.where(lo_first, i, j), np.where(lo_first, j, i)] = True
        cpdag = to_cpdag(arcs.astype(np.int8)).to_numpy()
        unoriented = adj.undirected(cpdag).any(axis=1)  # an undirected edge at each node
        for y in rng.permutation(v):
            parents = np.flatnonzero(arcs[:, y])
            if len(parents) and bool(unoriented[y]) != identifiable:
                x = int(rng.choice(parents))
                break
        else:
            continue
        break
    else:
        raise RuntimeError(f"no DAG with a pair of identifiable={identifiable} (seed {seed})")
    if family == "gaussian":
        magnitudes = rng.uniform(0.2, 1.0, (v, v))
        weights = arcs * magnitudes * rng.choice([-1.0, 1.0], (v, v))
        noise_sd = np.sqrt(rng.uniform(0.8, 1.0, v))
    else:
        weights = arcs * rng.uniform(0.2, 5.0, (v, v)) * rng.choice([-1.0, 1.0], (v, v))
        noise_sd = np.ones(v)
    model = finish(
        f"locpc-{'identifiable' if identifiable else 'nonidentifiable'}-{family}",
        seed,
        weights,
        noise_sd,
        [],
        [(x, int(y), "direct")],
        params={"identifiable": identifiable, "v": v, "family": family},
    )
    return model if family == "gaussian" else replace(model, family="binary", indep_test="gsq")


# --- from an edge list ------------------------------------------------------------------------


def from_edges(
    setting: str,
    seed: int,
    edges: list[tuple[str, str]],
    isolated: tuple[str, ...] = (),
    *,
    coefficient: Callable[[np.random.Generator], float] | float = 1.0,
    special: dict[tuple[str, str], float] | None = None,
    pair: tuple[str, str, str] = ("X", "Y", "total"),
    hide: tuple[str, ...] = (),
    params: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
) -> LinearModel:
    """A unit-noise linear-Gaussian model on a named graph.

    ``edges`` are ``(parent, child)`` pairs; ``coefficient`` is one value for every arc, or a
    function of the model's generator drawn once per arc, in the order of ``edges``; ``special``
    overrides the coefficient of named arcs. ``hide`` names the hidden nodes.
    """
    rng = rng_for(setting, seed, MODEL)
    names = list(dict.fromkeys([v for e in edges for v in e] + list(isolated)))
    index = {v: i for i, v in enumerate(names)}
    weights = np.zeros((len(names), len(names)))
    for a, b in edges:
        c = coefficient(rng) if callable(coefficient) else coefficient
        weights[index[a], index[b]] = (special or {}).get((a, b), c)
    unknown = [v for v in (*hide, pair[0], pair[1]) if v not in index]
    if unknown:
        raise ValueError(f"{setting} has no nodes {unknown}; its nodes are {names}")
    if pair[0] in hide or pair[1] in hide:
        raise ValueError("the treatment and the outcome cannot be hidden")
    meta = {**(meta or {}), "names": names}
    return finish(
        setting,
        seed,
        weights,
        np.ones(len(names)),
        [index[v] for v in hide],
        [(index[pair[0]], index[pair[1]], pair[2])],
        params=params,
        meta={**meta, "roles": {"hidden": [index[v] for v in hide]}},
    )


# --- Maasch et al.: LDP (2024) and LD3 (2025) ---------------------------------------------------

#: The 10-node DAG of the LDP paper's Figure 3, one node per partition: Z1 confounder, Z2 collider,
#: Z3 mediator, Z4 cause of the outcome only, Z5 instrument, Z6 child of the outcome, Z7 child of
#: the treatment, Z8 unconnected. The arc X -> Y is the (possibly null) effect.
LDP_EDGES = [
    ("Z1", "X"),
    ("Z1", "Y"),
    ("Z5", "X"),
    ("Z4", "Y"),
    ("X", "Y"),
    ("X", "Z3"),
    ("Z3", "Y"),
    ("X", "Z2"),
    ("Y", "Z2"),
    ("X", "Z7"),
    ("Y", "Z6"),
]

#: The 18-node DAG of the LDP paper's Figure E.5, the 10-node DAG with an M-structure (M1, M2, M3),
#: a butterfly (B1, B2, B3), and a second variable behind Z4 and Z5, whose dashed nodes are the
#: common causes the paper hides.
LDP_LATENT_EDGES = [
    ("B1", "B3"),
    ("B2", "B3"),
    ("B1", "X"),
    ("B3", "X"),
    ("B3", "Y"),
    ("B2", "Y"),
    ("Z1", "X"),
    ("Z1", "Y"),
    ("Z5a", "X"),
    ("Z5a", "Z5b"),
    ("Z4a", "Y"),
    ("Z4a", "Z4b"),
    ("X", "Y"),
    ("X", "Z7"),
    ("X", "Z3"),
    ("Z3", "Y"),
    ("X", "Z2"),
    ("Y", "Z2"),
    ("Y", "Z6"),
    ("Z6", "Z2"),
    ("M1", "X"),
    ("M1", "M3"),
    ("M2", "M3"),
    ("M2", "Y"),
]


def ldp(seed: int, *, effect: float = 2.75, latent: bool = False, hide: str = "") -> LinearModel:
    """Maasch et al. (2024, UAI), Sect. 6 and Figure 3: the 10-node DAG with one node per partition.

    Linear Gaussian, every coefficient 1 except the effect of ``X`` on ``Y``, ``effect`` (the
    paper's 2.75, so the total effect is 3.75 through the mediator), and unit-variance noise (the
    paper's Figure 6 model and the authors' ``data_generation.py``, whose default effect is 1). With
    ``latent`` the graph is the 18-node DAG of Figure E.5, whose dashed nodes ``B1 B2 B3 Z1 Z5a Z4a
    M1 M2`` the paper hides (``hide`` names the ones to hide, joined by ``+``); the paper's data
    for that figure are discrete, which this generator replaces by the linear-Gaussian instance.
    ``effect=0`` removes the arc, the null effect of the paper's pair of processes. The paper's
    sizes are n = 1000 to 7500 with the Fisher-z test at ``alpha = 0.01``.
    """
    edges = [e for e in (LDP_LATENT_EDGES if latent else LDP_EDGES) if effect or e != ("X", "Y")]
    isolated = ("Z8",)
    return from_edges(
        "ldp",
        seed,
        edges,
        isolated,
        special={("X", "Y"): effect},
        hide=tuple(v for v in hide.split("+") if v),
        params={"effect": effect, "latent": latent, "hide": hide},
    )


#: The 18-node DAG of the LD3 paper's Figure D.1 with the nodes the authors' code adds (``Z7``,
#: ``Z8``). Its parents of ``Y`` are ``X Z1 Z3c Z3d Z4b M2 B2 B3``.
LD3_EDGES = [
    ("B1", "B3"),
    ("B2", "B3"),
    ("B1", "X"),
    ("B3", "X"),
    ("B3", "Y"),
    ("B2", "Y"),
    ("Z1", "X"),
    ("Z1", "Y"),
    ("Z5", "X"),
    ("X", "Y"),
    ("X", "Z3a"),
    ("Z3a", "Z3b"),
    ("Z3b", "Z3c"),
    ("Z3c", "Y"),
    ("X", "Z3d"),
    ("Z4a", "Z3d"),
    ("Z3d", "Y"),
    ("Z4a", "Z4b"),
    ("Z4b", "Y"),
    ("M1", "X"),
    ("M1", "M3"),
    ("M2", "M3"),
    ("M2", "Y"),
    ("X", "Z7"),
]


def ld3_dag(seed: int, *, effect: float = 1.25, hide: str = "") -> LinearModel:
    """Maasch et al. (2025, AAAI), Appendix D.1 and Figure D.1: the 18-node DAG with a direct
    effect of ``X`` on ``Y``.

    Linear Gaussian with unit-variance noise, the direct effect ``effect`` (the paper's 1.25) and
    every other coefficient uniform on ``[0.75, 1.1]``, the range of the authors' notebook (their
    ``data_generation.py`` defaults to ``[1.1, 1.25]``). The figure draws 16 nodes and the code
    adds ``Z7`` and ``Z8``; the arc ``Z4a -> Z4b`` is in the figure and not in the code, and the
    figure is followed. ``hide`` names hidden nodes, joined by ``+``, from ``B1 B2 B3 Z1 Z3c Z4a M1
    M2``, the paper's one-at-a-time latent variants. The estimand is the direct effect. The paper's
    sizes are n = 1000 to 10000 with the Fisher-z test at ``alpha = 0.01``.
    """
    return from_edges(
        "ld3-dag",
        seed,
        LD3_EDGES,
        ("Z8",),
        coefficient=lambda rng: float(rng.uniform(0.75, 1.1)),
        special={("X", "Y"): effect},
        pair=("X", "Y", "direct"),
        hide=tuple(v for v in hide.split("+") if v),
        params={"effect": effect, "hide": hide},
    )


def ld3_er(seed: int, *, nodes: int = 50, ratio: float = 1.5) -> LinearModel:
    """Maasch et al. (2025), Sect. 5 and Appendix D.3: a random DAG whose outcome is a sink.

    The authors' notebook loads graphs with a fixed number of arcs, ``ratio * nodes`` (their files
    run from 5 nodes with 5 arcs to 500 nodes with 700), drawn at random in a fixed topological
    order. The outcome is the leaf with the most parents (the first on ties) and the treatment a
    random parent of it, so the outcome has no descendants (the paper's assumption A1) and every
    parent is observed. The paper gives no data model for these oracle graphs; the arcs here carry
    weights with a random sign and a magnitude uniform on ``[0.5, 1]`` and unit-variance noise.
    The estimand is the direct effect.
    """
    rng = rng_for("ld3-er", seed, MODEL)
    pairs = [(i, j) for i in range(nodes) for j in range(i + 1, nodes)]
    arcs = np.zeros((nodes, nodes), dtype=bool)
    for k in rng.choice(len(pairs), size=min(int(round(ratio * nodes)), len(pairs)), replace=False):
        arcs[pairs[k]] = True
    leaves = [v for v in range(nodes) if arcs[:, v].any() and not arcs[v].any()]
    if not leaves:
        raise RuntimeError(f"no leaf with a parent (seed {seed})")
    y = max(leaves, key=lambda v: (arcs[:, v].sum(), -v))
    x = int(rng.choice(np.flatnonzero(arcs[:, y])))
    weights = arcs * signed_uniform(rng, (nodes, nodes), 0.5, 1.0)
    return finish(
        "ld3-er",
        seed,
        weights,
        np.ones(nodes),
        [],
        [(x, y, "direct")],
        params={"nodes": nodes, "ratio": ratio},
    )


# --- Gupta, Childers and Lipton (2023): LDECC ------------------------------------------------


def ldecc(seed: int, *, covariates: int = 20, mediators: int = 3) -> LinearModel:
    """Gupta, Childers, and Lipton (2023), Appendix C.1: a linear-Gaussian graph with covariates
    and mediators.

    ``covariates`` covariates ``W`` (the paper's 20), the treatment ``X``, ``mediators`` mediators
    ``M`` (3), and the outcome ``Y``, in that causal order. Each arc is drawn with probability 0.1
    (covariate-covariate, covariate-treatment, covariate-outcome, treatment-mediator,
    mediator-mediator, mediator-outcome) or 0.05 (covariate-mediator), the paper's values. Weights
    have a random sign, independent per arc, and a magnitude uniform on ``[0.25, 1]``; the noise
    variances are uniform on ``[0.1, 0.2]``. The paper does not say how a graph is kept connected;
    the generator follows the authors' notebook, which redraws until each covariate after the first
    has a covariate parent, the treatment and the outcome each have a covariate parent, and each
    mediator after the first has an earlier mediator as a parent, and which always adds the arcs
    ``X -> M1`` and ``M3 -> Y``, so every effect of ``X`` on ``Y`` passes through a mediator. The
    code draws one sign for a whole graph and gives ``X -> M2`` and ``X -> M3`` probability 0.05;
    the paper is followed. The paper's sizes are 8000 to 32000 rows with the Fisher-z test and a
    level of ``0.01 * 32000 / n``.
    """
    rng = rng_for("ldecc", seed, MODEL)
    c, m = covariates, mediators
    x, first_m, y = c, c + 1, c + m + 1
    n = y + 1
    arcs = np.zeros((n, n), dtype=bool)

    def parents(count: int, p: float) -> np.ndarray:
        """Bernoulli draws, redrawn until at least one is set."""
        while True:
            drawn = rng.random(count) < p
            if drawn.any():
                return drawn

    for i in range(1, c):
        arcs[:i, i] = parents(i, 0.1)
    arcs[:c, x] = parents(c, 0.1)
    arcs[:c, y] = parents(c, 0.1)
    arcs[x, first_m] = True
    arcs[x, first_m + 1 : first_m + m] = rng.random(m - 1) < 0.1
    for j in range(first_m + 1, first_m + m):
        arcs[first_m:j, j] = parents(j - first_m, 0.1)
    arcs[:c, first_m : first_m + m] = rng.random((c, m)) < 0.05
    arcs[first_m : first_m + m - 1, y] = rng.random(m - 1) < 0.1
    arcs[first_m + m - 1, y] = True
    weights = arcs * rng.uniform(0.25, 1.0, (n, n)) * rng.choice([-1.0, 1.0], (n, n))
    return finish(
        "ldecc",
        seed,
        weights,
        np.sqrt(rng.uniform(0.1, 0.2, n)),
        [],
        [(x, y, "total")],
        params={"covariates": c, "mediators": m},
    )


# --- Schubert, Claassen and Magliacane: SNAP (2025) and LOAD (2026) ------------------------------


def random_graph(
    seed: int,
    *,
    nodes: int = 100,
    degree: float = 2.0,
    max_degree: int = 10,
    weight_low: float = 0.5,
    weight_high: float = 3.0,
) -> LinearModel:
    """A random graph with a target pair, as the SNAP and LOAD papers draw them.

    Schubert, Claassen, and Magliacane (2025), Sect. 5, and (2026), Sect. 5: ``nodes`` nodes with
    expected degree ``degree`` and maximum degree ``max_degree`` (10), weights with a random sign
    and a magnitude uniform on ``[0.5, 3]``, and standard Gaussian noise. Each pair of nodes is
    adjacent with probability ``degree / (nodes - 1)`` (the LOAD paper's Erdos-Renyi graphs; the
    SNAP authors' code draws a degree sequence instead, which the paper does not describe), a
    node over the maximum degree loses random incident arcs, and the arcs follow a random order.
    The target pair is drawn as the LOAD paper has it: an explicit ancestor of the other (a
    directed path between them in the CPDAG of the true DAG, which the methods are not told) with
    the effect identifiable from the CPDAG; the paper does not say how a qualifying pair is drawn,
    and this generator takes a node uniformly, then one of its explicit descendants uniformly,
    until the effect is identifiable. The pair is the treatment and the outcome in that order; the
    methods that do not know the order answer both directions.
    """
    rng = rng_for("random-graph", seed, MODEL)
    for _ in range(MAX_TRIES):
        order = rng.permutation(nodes)
        adjacent = np.triu(rng.random((nodes, nodes)) < degree / (nodes - 1), 1)
        adjacent = adjacent | adjacent.T
        for v in np.flatnonzero(adjacent.sum(1) > max_degree):
            while adjacent[v].sum() > max_degree:
                u = rng.choice(np.flatnonzero(adjacent[v]))
                adjacent[v, u] = adjacent[u, v] = False
        i, j = np.nonzero(np.triu(adjacent, 1))
        forward = order[i] < order[j]
        arcs = np.zeros((nodes, nodes), dtype=bool)
        arcs[np.where(forward, i, j), np.where(forward, j, i)] = True
        pair = explicit_ancestor_pair(rng, arcs)
        if pair is not None:
            break
    else:
        raise RuntimeError(f"no graph with a qualifying pair in {MAX_TRIES} draws (seed {seed})")
    weights = arcs * signed_uniform(rng, (nodes, nodes), weight_low, weight_high)
    return finish(
        "random-graph",
        seed,
        weights,
        np.ones(nodes),
        [],
        [(*pair, "total")],
        params={"nodes": nodes, "degree": degree, "max_degree": max_degree},
    )


def explicit_ancestor_pair(
    rng: np.random.Generator, arcs: np.ndarray, tries: int = 200
) -> tuple[int, int] | None:
    """A pair ``(x, y)`` where ``x`` is an explicit ancestor of ``y`` and the total effect is
    identifiable from the CPDAG of the DAG ``arcs``, or ``None`` after ``tries`` draws.

    An explicit ancestor has a directed path to the other in the CPDAG (Schubert et al., 2026,
    Sect. 2); identifiable is as :func:`andrey.metrics.effect_oracle` defines it for the class.
    """
    from andrey.core import adjustment as adj
    from andrey.metrics import to_cpdag

    cpdag = to_cpdag(arcs.astype(np.int8))
    compelled = adj.directed(cpdag.to_numpy())
    for _ in range(tries):
        x = int(rng.integers(len(arcs)))
        below = sorted(adj.reach(compelled, [x]) - {x})
        if not below:
            continue
        y = int(rng.choice(below))
        if adj.adjustment_set(cpdag, x, y) is not None:
            return x, y
    return None


# --- registry ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Setting:
    """A registered setting: the generator, the effect it asks about, and its paper."""

    name: str
    make: Callable[..., LinearModel]
    estimand: str
    paper: str
    defaults: dict[str, Any]
    #: Whether the generator can draw hidden variables, so the methods that identify from the
    #: Markov equivalence class (no hidden variables) skip it.
    hidden: bool = False


SETTINGS: dict[str, Setting] = {}


def register(setting: Setting) -> None:
    """Add ``setting`` to :data:`SETTINGS`."""
    SETTINGS[setting.name] = setting


for _task in ("effect-admissible", "effect-none", "zero-admissible", "zero-none"):
    register(
        Setting(
            f"entner-{_task}",
            lambda seed, _task=_task, **kw: entner(seed, task=_task, **kw),
            "total",
            "Entner, Hoyer, and Spirtes (2013)",
            {},
            hidden=True,
        )
    )
for _ident in (True, False):
    register(
        Setting(
            f"locpc-{'identifiable' if _ident else 'nonidentifiable'}",
            lambda seed, _ident=_ident, **kw: locpc(seed, identifiable=_ident, **kw),
            "direct",
            "Loranchet and Assaad (2026)",
            {"v": 50},
        )
    )


for _name, _make, _estimand, _paper, _defaults, _hidden in (
    ("ldp", ldp, "total", "Maasch et al. (2024)", {}, True),
    ("ld3-dag", ld3_dag, "direct", "Maasch et al. (2025)", {}, True),
    ("ld3-er", ld3_er, "direct", "Maasch et al. (2025)", {}, False),
    ("ldecc", ldecc, "total", "Gupta, Childers, and Lipton (2023)", {}, False),
    ("random-graph", random_graph, "total", "Schubert et al. (2025, 2026)", {}, False),
):
    register(Setting(_name, _make, _estimand, _paper, _defaults, hidden=_hidden))


@dataclass(frozen=True)
class Cell:
    """One cell of the lane: a setting, its sample size, and the parameters it was given."""

    setting: str
    n: int
    params: tuple[tuple[str, Any], ...] = ()

    @property
    def key(self) -> str:
        """The cell as text: ``setting@n,key=value,...``, parameters sorted."""
        extra = "".join(f",{k}={v}" for k, v in self.params)
        return f"{self.setting}@{self.n}{extra}"

    def model(self, seed: int) -> LinearModel:
        """The cell's model for ``seed``."""
        spec = SETTINGS[self.setting]
        return spec.make(seed, **{**spec.defaults, **dict(self.params)})


def parse_cell(text: str) -> Cell:
    """The cell ``setting@n,key=value,...``; values are numbers where they parse, else text."""
    head, *rest = text.split(",")
    name, sep, n = head.partition("@")
    if name not in SETTINGS or not sep or not n.isdigit():
        raise ValueError(f"a cell is setting@n[,key=value...] with a setting of {sorted(SETTINGS)}")
    params = []
    for item in rest:
        key, eq, raw = item.partition("=")
        if not eq:
            raise ValueError(f"{item!r} in cell {text!r} is not key=value")
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            value = raw
        params.append((key, value))
    return Cell(name, int(n), tuple(sorted(params)))


# The settings that read a published network register themselves; the import comes last because
# that module builds on everything above.
from andrey_bench import network_models  # noqa: E402,F401
