"""Check that a change returns the same answers, bit for bit, by hashing each method's full output.

Every supported method of the quality gate (:mod:`qa.quality_gate`), and each experimental method of
:data:`EXPERIMENTAL_DATASETS`, is fitted on the first draws of its gate datasets and of one larger
dataset (:data:`LARGER`), on the numpy backend with one worker and one BLAS thread. A fit's hash
covers its whole output: the graph, the edge weights, the causal order, and every metadata value,
such as a score, a p-value, or a list of nodes. Floats enter as their bytes, never as printed text,
NaN as one bit pattern, and mappings in key order, so equal hashes mean equal bits. A second hash
covers only the endpoint marks of the graphs, so a difference reads as a changed graph or as changed
bits under the same graph.

The manifest holds one row per fit, keyed by method, dataset, seed, permutation, and worker count,
under a header naming the commit, the imported Andrey package's directory and commit, the Python,
NumPy, and SciPy versions, the loaded BLAS, the CPU, and the thread settings. ``--compare`` counts
the rows of two manifests that differ. A fit that
queries a CI test or a local score also records two deterministic counts (:class:`Counts`), and a
compare names every row where one changed and fails on any change, a fall as well as a rise, unless
``--allow-fewer-queries`` accepts a fall.

``--shuffle K`` refits each draw on ``K`` seeded column orders and maps each answer back to the
original columns before hashing it; ``--workers N`` refits the methods that run worker processes
or threads (:data:`WORKER_METHODS`) on ``N`` workers, with every GES pass and every skeleton pass of
conditioning size two or more sent to the pool. Both report each refit whose answer differs from
the first fit.

    uv run python -m qa.identity -o before.json             # fit the corpus, write a manifest
    uv run python -m qa.identity --compare before.json after.json  # exit 1 on any difference
    uv run python -m qa.identity --shuffle 3                # exit 1 if a column order changes one
    uv run python -m qa.identity GES GFCI --workers 4       # exit 1 if 4 workers change one
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import importlib
import json
import os
import platform
import struct
import subprocess
import sys
from collections import defaultdict
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import threadpoolctl
from threadpoolctl import threadpool_limits

import andrey
from andrey import GraphStructure, StructureOutput, TemporalStructure
from andrey.core import ci, ci_cmiknn, ci_gpdc, ci_kernel, score, score_delta
from andrey.search import hc, xges
from qa import quality_gate as qg

# ==================================================================================================
# The corpus -- each method's public call, on the gate's datasets and one larger dataset.
# ==================================================================================================

#: Draws per dataset, the first of the gate's ten seeds.
SEEDS = 2
#: The significance level of the CI-test methods, passed to each; the counts read it.
ALPHA = 0.05


@dataclass(frozen=True)
class Fit:
    """A method's public call on one draw, returning its full output.

    ``blocks`` is the number of side-by-side copies of the variables in the data: GIES's samples
    and intervention indicators, MultiGroupDirectLiNGAM's groups; a column order moves each copy
    the same way. ``alpha`` is the significance level of a CI-test method.
    """

    call: Callable[[np.ndarray], Any]
    blocks: int = 1
    alpha: float | None = None


_CDNOD = qg.AndreyCDNOD().params()
_GROUPS = qg.AndreyMultiGroupDirectLiNGAM().params()["n_groups"]
_VARMA = qg.AndreyVarmaLiNGAM().params()


def _domains(x: np.ndarray) -> Any:
    index = qg.regime_models.domain_index(len(x), _CDNOD["n_domains"])
    return andrey.cdnod(x, index, alpha=_CDNOD["alpha"], indep_test=_CDNOD["indep_test"])


#: The gate's settings, with each method's full output: the benchmark adapters' parameters for the
#: methods the gate fits through an adapter, which returns only the graph.
FITS: dict[str, Fit] = {
    "PC": Fit(functools.partial(andrey.pc, alpha=ALPHA), alpha=ALPHA),
    "FCI": Fit(functools.partial(andrey.fci, alpha=ALPHA), alpha=ALPHA),
    "GFCI": Fit(functools.partial(andrey.gfci, alpha=ALPHA), alpha=ALPHA),
    "CDNOD": Fit(_domains, alpha=_CDNOD["alpha"]),
    "GES": Fit(andrey.ges),
    "GIES": Fit(lambda x: andrey.gies(*qg.unstack(x)), blocks=2),
    "XGES": Fit(andrey.xges),
    "HC": Fit(andrey.hc),
    "Tabu": Fit(andrey.tabu),
    "IAMB": Fit(functools.partial(andrey.iamb, alpha=ALPHA), alpha=ALPHA),
    "InterIAMB": Fit(functools.partial(andrey.inter_iamb, alpha=ALPHA), alpha=ALPHA),
    "MMPC": Fit(functools.partial(andrey.mmpc, alpha=ALPHA), alpha=ALPHA),
    "MMHC": Fit(functools.partial(andrey.mmhc, alpha=ALPHA), alpha=ALPHA),
    "RSMAX2": Fit(
        functools.partial(andrey.rsmax2, restrict="pc", maximize="tabu", alpha=ALPHA), alpha=ALPHA
    ),
    "ExactSearch": Fit(andrey.exact_search),
    "CALM": Fit(functools.partial(andrey.calm, seed=0)),
    "BOSS": Fit(functools.partial(andrey.boss, seed=0)),
    "GRaSP": Fit(functools.partial(andrey.grasp, seed=0)),
    "DirectLiNGAM": Fit(andrey.direct_lingam),
    "ICA-LiNGAM": Fit(functools.partial(andrey.ica_lingam, random_state=0)),
    "MultiGroupDirectLiNGAM": Fit(
        lambda x: andrey.multi_group_direct_lingam(qg.regime_models.split_groups(x, _GROUPS)),
        blocks=_GROUPS,
    ),
    "RCD": Fit(andrey.rcd),
    "BottomUpParceLiNGAM": Fit(andrey.bottom_up_parce_lingam),
    "GIN": Fit(andrey.gin),
    "ANM": Fit(andrey.anm),
    "PNL": Fit(andrey.pnl),
    "PCMCI": Fit(
        lambda x: andrey.pcmci(x, **qg.AndreyPCMCI("pcmci", "student_t", tau_max=2).params())
    ),
    "PCMCIplus": Fit(lambda x: andrey.pcmciplus(x, **qg.andrey_pcmciplus_adapter(5).params())),
    "LPCMCI": Fit(lambda x: andrey.lpcmci(x, **qg.andrey_lpcmci_adapter(3).params())),
    "VARMA-LiNGAM": Fit(
        lambda x: andrey.varma_lingam(
            x, order=tuple(_VARMA["order"]), criterion=_VARMA["criterion"], prune=_VARMA["prune"]
        )
    ),
    "LongitudinalLiNGAM": Fit(
        lambda x: andrey.longitudinal_lingam(list(x), **qg.AndreyLongitudinalLiNGAM().params())
    ),
    "Granger": Fit(lambda x: andrey.granger(x, maxlag=qg.GRANGER_LAGS, cv=5)),
}

#: The methods that run on ``andrey.config(num_workers=...)``: GES, GFCI's GES phase, and the
#: skeleton searches of PC, FCI, GFCI, CD-NOD, and RSMAX2's PC phase in worker processes, Granger's
#: regressions and PNL's two directions on threads.
WORKER_METHODS = ("PC", "FCI", "GFCI", "CDNOD", "RSMAX2", "GES", "Granger", "PNL")

#: The gate datasets of the experimental methods, which the quality gate does not fit.
EXPERIMENTAL_DATASETS: dict[str, tuple[str, ...]] = dict.fromkeys(
    ("IAMB", "InterIAMB", "MMPC", "MMHC", "RSMAX2"), qg.CPDAG_DATA
)

#: The factor on the longitudinal LiNGAM panel's coefficients at ``d = 10``. At the published
#: ranges the panel grows several-fold per time point; at this factor the spectral radius of every
#: step's lagged dynamics stays below 1 and the panel stays bounded.
PANEL_SCALE = 0.3

#: One dataset per method beyond the gate's sizes, drawn as the gate draws its regime, at a size
#: that keeps the whole corpus quick on one core; the panel's coefficients take
#: :data:`PANEL_SCALE`.
LARGER_DATASETS: dict[str, Callable[[], tuple[qg.Draw, ...]]] = {
    "linear_gauss_er_d100": functools.partial(qg.regime_draws, "linear_gauss_er", 100, 5000),
    "latent_gauss_er_d100": functools.partial(qg.regime_draws, "latent_gauss_er", 100, 5000),
    "lingam_sf_d70": functools.partial(qg.regime_draws, "lingam_sf", 70, 3500),
    "latent_lingam_er_d17": functools.partial(qg.regime_draws, "latent_lingam_er", 17, 2000),
    "linear_gauss_er_d16": functools.partial(qg.regime_draws, "linear_gauss_er", 16, 800),
    "interventional_gauss_er_d100": functools.partial(qg.intervention_draws, 100, 5000),
    "domain_shift_er_d70": functools.partial(qg.model_draws, "domain_shift_er", 70, 2000),
    "multi_group_er_d30": functools.partial(qg.model_draws, "multi_group_er", 30, 1000),
    "latent_measurement_d16": functools.partial(qg.model_draws, "latent_measurement", 16, 1000),
    "anm_pair_n1500": functools.partial(qg.pair_draws, "anm_pair", 1500),
    "pnl_pair_n1500": functools.partial(qg.pair_draws, "pnl_pair", 1500),
    "pcmci_paper_d20_n1000": functools.partial(qg.series_draws, "pcmci_paper", 20, 1000),
    "pcmciplus_paper_d40_n1000": functools.partial(qg.series_draws, "pcmciplus_paper", 40, 1000),
    "lpcmci_paper_d20_n1000": functools.partial(qg.series_draws, "lpcmci_paper", 20, 1000),
    "varma_lingam_paper_d8_n300": functools.partial(qg.series_draws, "varma_lingam_paper", 8, 300),
    "longitudinal_lingam_stable_d10_n20_m500": functools.partial(
        qg.panel_draws, 10, 20, 500, scale=PANEL_SCALE
    ),
    "var2_lagged_d10_n1000": functools.partial(qg.granger_draws, 10, 1000),
}
LARGER: dict[str, str] = {
    **dict.fromkeys(
        (
            "PC",
            "GES",
            "XGES",
            "HC",
            "Tabu",
            "IAMB",
            "InterIAMB",
            "MMPC",
            "MMHC",
            "RSMAX2",
            "BOSS",
            "GRaSP",
        ),
        "linear_gauss_er_d100",
    ),
    **dict.fromkeys(("FCI", "GFCI"), "latent_gauss_er_d100"),
    **dict.fromkeys(("DirectLiNGAM", "ICA-LiNGAM"), "lingam_sf_d70"),
    **dict.fromkeys(("RCD", "BottomUpParceLiNGAM"), "latent_lingam_er_d17"),
    "ExactSearch": "linear_gauss_er_d16",
    "CALM": "linear_gauss_er_d15",
    "CDNOD": "domain_shift_er_d70",
    "GIES": "interventional_gauss_er_d100",
    "MultiGroupDirectLiNGAM": "multi_group_er_d30",
    "GIN": "latent_measurement_d16",
    "ANM": "anm_pair_n1500",
    "PNL": "pnl_pair_n1500",
    "PCMCI": "pcmci_paper_d20_n1000",
    "PCMCIplus": "pcmciplus_paper_d40_n1000",
    "LPCMCI": "lpcmci_paper_d20_n1000",
    "VARMA-LiNGAM": "varma_lingam_paper_d8_n300",
    "LongitudinalLiNGAM": "longitudinal_lingam_stable_d10_n20_m500",
    "Granger": "var2_lagged_d10_n1000",
}


@functools.cache
def draws(dataset: str) -> tuple[qg.Draw, ...]:
    """Return the draws of a gate or larger dataset, generated once per process."""
    return LARGER_DATASETS[dataset]() if dataset in LARGER_DATASETS else qg.draws(dataset)


def corpus(methods: Iterable[str], seeds: int) -> Iterator[tuple[str, str, int, np.ndarray]]:
    """Yield ``(method, dataset, seed, data)`` for the first ``seeds`` draws of each dataset."""
    for method in methods:
        gate = EXPERIMENTAL_DATASETS.get(method) or qg.SUPPORTED[method].datasets
        for dataset in (*gate, LARGER[method]):
            for seed, draw in enumerate(draws(dataset)[:seeds]):
                yield method, dataset, seed, np.ascontiguousarray(draw.data)


# ==================================================================================================
# Hashes -- the full output, its graphs, and the output mapped back from a column order.
# ==================================================================================================


def digest(obj: Any) -> str:
    """Return the SHA-256 of ``obj``'s bytes: see :func:`_feed`."""
    h = hashlib.sha256()
    _feed(h, obj)
    return h.hexdigest()


def _feed(h: Any, obj: Any) -> None:
    """Feed ``obj`` to ``h`` as tagged, length-prefixed bytes.

    A float or an array enters as its little-endian bytes after its type and shape, with every NaN
    as one bit pattern; an integer as its decimal digits; a mapping in key order; a dataclass, such
    as a result or a graph, as its class name and every field. Any other type raises
    ``TypeError``, so nothing enters by its printed form.
    """
    if obj is None:
        h.update(b"N")
    elif isinstance(obj, (bool, np.bool_)):
        h.update(b"T" if obj else b"F")
    elif isinstance(obj, (int, np.integer)):
        _sized(h, b"I", str(int(obj)).encode())
    elif isinstance(obj, (float, np.floating, np.ndarray)):
        a = np.asarray(obj)
        if a.dtype.hasobject:
            raise TypeError("cannot hash an object array")
        a = np.ascontiguousarray(a, dtype=a.dtype.newbyteorder("<"))
        if a.dtype.kind in "fc":
            a = np.where(np.isnan(a), np.nan, a).astype(a.dtype)
        _sized(h, b"A", f"{a.dtype.str}{a.shape}".encode())
        _sized(h, b"", a.tobytes())
    elif isinstance(obj, str):
        _sized(h, b"S", obj.encode())
    elif isinstance(obj, Mapping):
        h.update(b"M" + struct.pack("<Q", len(obj)))
        for key in sorted(obj):
            _feed(h, key)
            _feed(h, obj[key])
    elif isinstance(obj, (list, tuple)):
        h.update(b"L" + struct.pack("<Q", len(obj)))
        for item in obj:
            _feed(h, item)
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        _sized(h, b"C", type(obj).__qualname__.encode())
        for field in dataclasses.fields(obj):
            _sized(h, b"K", field.name.encode())
            _feed(h, getattr(obj, field.name))
    else:
        raise TypeError(f"cannot hash a {type(obj).__name__}")


def _sized(h: Any, tag: bytes, payload: bytes) -> None:
    h.update(tag + struct.pack("<Q", len(payload)) + payload)


def graphs(obj: Any) -> Iterator[GraphStructure]:
    """Yield every graph in ``obj``: a result's graph, a time series's graph at each lag and
    occasion, and each group's graph in a list of results."""
    if isinstance(obj, GraphStructure):
        yield obj
    elif isinstance(obj, (StructureOutput, TemporalStructure)):
        for field in dataclasses.fields(obj):
            yield from graphs(getattr(obj, field.name))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from graphs(item)


def graph_digest(out: Any) -> str:
    """Return the digest of the kind, endpoint marks, and node types of every graph in ``out``."""
    return digest([(g.kind, g.to_numpy(), g.node_types) for g in graphs(out)])


def parts(out: Any) -> dict[str, str]:
    """Return the digest of each part of ``out``: its graphs, its structure with any lag weights,
    its causal order, its edge weights, and each metadata value; a list of results prefixes each
    result's parts with its position."""
    if isinstance(out, list):
        return {f"{i}.{name}": d for i, o in enumerate(out) for name, d in parts(o).items()}
    return {
        "graph": graph_digest(out),
        "structure": digest(out.structure),
        "ordering": digest(out.ordering),
        "weights": digest(out.weighted_adjacency),
        **{f"metadata.{key}": digest(value) for key, value in out.metadata.items()},
    }


#: Metadata with a value for each lag and ordered pair of variables, ``[k][i][j]``.
LAG_MATRICES = ("p_values", "statistics", "links")
#: Metadata that lists node indices among other values. The column-shuffle comparison leaves it
#: out: each method lays out its lists in its own way.
NODE_LISTS = (
    "ambiguous_triples",
    "causal_order",
    "causal_orders",
    "clusters",
    "confounded_pairs",
    "conflicts",
    "parents",
    "unordered",
)
#: Metadata of a pair's two directions, first column to second and back, which a swap exchanges.
DIRECTIONS = (("pval_forward", "pval_backward"), ("stat_forward", "stat_backward"))


def permuted(x: np.ndarray, order: np.ndarray, blocks: int) -> np.ndarray:
    """Return ``x`` with each of its ``blocks`` copies of the variables in column ``order``, in C
    order as the corpus is: some fits' bits depend on the memory layout."""
    copies = [b[..., order] for b in np.split(x, blocks, axis=-1)]
    return np.ascontiguousarray(np.concatenate(copies, axis=-1))


def unpermute(out: Any, order: np.ndarray) -> Any:
    """Return ``out``, fitted on the data's columns in ``order``, with each node moved back to its
    original column: the graphs, the time series's weights, the causal order, the edge weights,
    and the metadata in :data:`LAG_MATRICES`. Hidden nodes after the observed ones keep their
    places, a swapped pair's :data:`DIRECTIONS` trade places, and the metadata in
    :data:`NODE_LISTS` is left out."""
    if isinstance(out, list):
        return [unpermute(o, order) for o in out]
    order = np.concatenate([order, np.arange(len(order), out.structure.n_nodes)])
    back = np.argsort(order)
    weights = out.weighted_adjacency
    metadata = {
        key: np.asarray(value)[:, back][:, :, back].tolist() if key in LAG_MATRICES else value
        for key, value in out.metadata.items()
        if key not in NODE_LISTS
    }
    if len(order) == 2 and order[0] == 1:
        for forward, backward in DIRECTIONS:
            if forward in metadata:
                metadata[forward], metadata[backward] = metadata[backward], metadata[forward]
    return StructureOutput.new(
        _moved(out.structure, back),
        ordering=None if out.ordering is None else [int(order[i]) for i in out.ordering],
        weighted_adjacency=None if weights is None else weights[np.ix_(back, back)],
        metadata=metadata,
    )


def _moved(value: Any, back: np.ndarray) -> Any:
    """Return ``value`` with node ``back[i]`` as node ``i``: a graph's marks and node types, and a
    time series's graphs and ``(..., n, n)`` weights. Labels stay: an array's columns are named by
    position."""
    if isinstance(value, GraphStructure):
        types = value.node_types
        return GraphStructure.from_numpy(
            value.to_numpy()[np.ix_(back, back)],
            kind=value.kind,
            labels=value.labels,
            node_types=None if types is None else np.asarray(types)[back],
            allow_self_loops=True,
        )
    if isinstance(value, TemporalStructure):
        moved = {f.name: _moved(getattr(value, f.name), back) for f in dataclasses.fields(value)}
        return dataclasses.replace(value, **moved)
    if isinstance(value, tuple):
        return tuple(_moved(v, back) for v in value)
    if isinstance(value, np.ndarray) and value.ndim >= 2 and value.shape[-2:] == (len(back),) * 2:
        return value[..., back, :][..., back]
    return value


# ==================================================================================================
# Query counts -- the CI-test and local-score queries one fit makes.
# ==================================================================================================


class Counts:
    """Deterministic counts of the CI-test and local-score queries made inside :func:`counting`.

    ``ci_after_separation`` counts the CI-test queries of a pair made after a query of that pair
    returned ``p > alpha``, its first separating set; it needs ``alpha``. ``ci_sets`` and
    ``score_sets`` count the distinct conditioning sets of each size: a CI test's conditioning set
    and a local score's parent set. ``ci_queries`` and ``score_queries`` count every query, one
    per p-value or per parent set scored.
    """

    def __init__(self, alpha: float | None) -> None:
        self.alpha = alpha
        self.busy = False  # inside a counted call, whose own queries are not counted again
        self.ci_queries = self.after_separation = self.score_queries = 0
        self.separated: set[tuple[int, int]] = set()
        self.ci_sets: defaultdict[int, set[frozenset[int]]] = defaultdict(set)
        self.score_sets: defaultdict[int, set[frozenset[int]]] = defaultdict(set)

    def ci(self, queries: Iterable[tuple[int, int, Iterable[int], float]]) -> None:
        for x, y, condition_set, p in queries:
            pair = (min(int(x), int(y)), max(int(x), int(y)))
            self.ci_queries += 1
            self.after_separation += pair in self.separated
            if self.alpha is not None and p > self.alpha:
                self.separated.add(pair)
            conditions = frozenset(int(z) for z in condition_set)
            self.ci_sets[len(conditions)].add(conditions)

    def score(self, parent_sets: Iterable[Iterable[int]]) -> None:
        for parent_set in parent_sets:
            parents = frozenset(int(p) for p in parent_set)
            self.score_queries += 1
            self.score_sets[len(parents)].add(parents)

    def summary(self) -> dict[str, Any]:
        """Return the counts with a query, the sets by size as ``{"size": count}``."""
        out: dict[str, Any] = {}
        if self.ci_queries:
            out["ci_queries"] = self.ci_queries
            if self.alpha is not None:
                out["ci_after_separation"] = self.after_separation
            out["ci_sets"] = {str(k): len(v) for k, v in sorted(self.ci_sets.items())}
        if self.score_queries:
            out["score_queries"] = self.score_queries
            out["score_sets"] = {str(k): len(v) for k, v in sorted(self.score_sets.items())}
        return out


def _moves(base: Iterable[int], moves: Iterable[tuple[str, int]]) -> list[list[int]]:
    """The parent set ``base`` and ``base`` after each ``("add" | "remove", u)`` move."""
    base = list(base)
    return [base, *([*base, u] if op == "add" else [p for p in base if p != u] for op, u in moves)]


#: The CI tests' query methods, each with the ``(x, y, conditioning set, p-value)`` of every test
#: one call makes, read from the call's result, the test, and the call's arguments.
CI_QUERIES: dict[str, Callable[..., Iterable[tuple[int, int, Any, float]]]] = {
    "__call__": lambda p, test, x, y, conditions=(): [(x, y, conditions, p)],
    "_bounded_pvalue": lambda p, test, x, y, conditions, *settings, **options: [
        (x, y, conditions, p)
    ],
    "batched_call": lambda ps, test, x, y, sets: [(x, y, s, p) for s, p in zip(sets, ps)],
    "first_order_pvalues": lambda ps, test, x, y, zs: [(x, y, (z,), p) for z, p in zip(zs, ps)],
    "_canonical_pvalues": lambda ps, test, xs, ys, sets: list(zip(xs, ys, sets, ps)),
    "marginal_pvalues": lambda ps, test: [
        (x, y, (), ps[x, y]) for x, y in zip(*np.triu_indices(len(ps), 1))
    ],
}
CI_CLASSES = (
    ci.PartialCorrelationTest,
    ci.ChiSquare,
    ci.DGTest,
    ci.CGTest,
    ci_kernel.KCI,
    ci_kernel.RCoT,
    ci_cmiknn.CMIknn,
    ci_gpdc.GPDC,
)
#: The local scores' query methods and the score-delta functions, each with the parent sets one
#: call scores, read from the call's result, the score, and the call's arguments. A delta scores
#: a base set and each move from it.
SCORE_QUERIES: dict[str, Callable[..., Iterable[Iterable[int]]]] = {
    "score": lambda result, s, node, parents: [parents],
    "score_many": lambda result, s, node, parent_sets: parent_sets,
    "delta": lambda result, s, node, base, u, op: _moves(base, [(op, u)]),
    "delta_many_with_base": lambda result, s, node, base, moves, **options: _moves(base, moves),
    "score_many_with_base": lambda result, s, node, base, moves, **options: _moves(base, moves),
    "add_deltas": lambda result, s, node, base, **options: _moves(
        base, [("add", u) for u in range(len(s.cov)) if u != node and u not in set(base)]
    ),
    "remove_deltas": lambda result, s, node, base, **options: _moves(
        base, [("remove", u) for u in base]
    ),
}
SCORE_CLASSES = (
    importlib.import_module("andrey.search.exact")._GramBICScore,
    score.BICScore,
    score.BDeuScore,
    score.CategoricalBICScore,
    score.DGScore,
    score.CGScore,
    score_delta.DeltaBICScore,
    score_delta.FullScoreDelta,
)
#: The modules that name the score-delta functions ``add_deltas`` and ``remove_deltas``.
DELTA_MODULES = (score_delta, hc, xges)

#: The counts of the running :func:`counting` block, or ``None``.
_active: Counts | None = None


@contextmanager
def counting(alpha: float | None) -> Generator[Counts]:
    """Count the CI-test and local-score queries made inside the block.

    Each query method of :data:`CI_CLASSES` and :data:`SCORE_CLASSES`, and each name of a
    score-delta function in :data:`DELTA_MODULES`, is wrapped for the block and restored after it.
    A call made inside another counted call is part of it and is not counted again. Only this
    process counts, so the queries of worker processes are not seen.
    """
    global _active
    patches = list(_patches())
    for owner, name, wrapped in patches:
        setattr(owner, name, _counted(wrapped, *_RECORDERS[name]))
    _active = Counts(alpha)
    try:
        yield _active
    finally:
        _active = None
        for owner, name, wrapped in patches:
            setattr(owner, name, wrapped)


_RECORDERS: dict[str, tuple[Callable[..., Iterable[Any]], Callable[[Counts, Any], None]]] = {
    **{name: (queries, Counts.ci) for name, queries in CI_QUERIES.items()},
    **{name: (queries, Counts.score) for name, queries in SCORE_QUERIES.items()},
}


def _patches() -> Iterator[tuple[Any, str, Callable[..., Any]]]:
    """Yield ``(owner, name, function)`` for each query method or function to count, once each:
    the class that defines a method, and each module that names a score-delta function."""
    owners = set()
    for classes, names in ((CI_CLASSES, CI_QUERIES), (SCORE_CLASSES, SCORE_QUERIES)):
        for cls in classes:
            for name in names:
                owner = next((c for c in cls.__mro__ if name in vars(c)), object)
                if owner is not object:
                    owners.add((owner, name))
    for owner, name in owners:
        yield owner, name, vars(owner)[name]
    for module in DELTA_MODULES:
        for name in ("add_deltas", "remove_deltas"):
            if hasattr(module, name):
                yield module, name, getattr(module, name)


def _counted(
    function: Callable[..., Any],
    queries: Callable[..., Iterable[Any]],
    record: Callable[[Counts, Any], None],
) -> Callable[..., Any]:
    """Wrap ``function`` to record the queries of each outermost call inside :func:`counting`. An
    iterator argument is read into a list first, so the queries can read it too."""

    @functools.wraps(function)
    def counted(*args: Any, **kwargs: Any) -> Any:
        counts = _active
        if counts is None or counts.busy:
            return function(*args, **kwargs)
        args = tuple(list(a) if isinstance(a, Iterator) else a for a in args)
        counts.busy = True
        try:
            result = function(*args, **kwargs)
        finally:
            counts.busy = False
        record(counts, queries(result, *args, **kwargs))
        return result

    return counted


# ==================================================================================================
# The run and the manifest
# ==================================================================================================

#: The environment variables that set thread and worker counts, recorded in the header.
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "ANDREY_NUM_WORKERS",
    "ANDREY_GES_PARALLEL_MIN_WORK",
)
ROOT = Path(__file__).resolve().parents[1]


def commit() -> dict[str, Any]:
    """Return the checked-out commit, whether tracked files differ from it, and the imported
    Andrey package's directory with the commit of its git work tree, or ``None`` outside one."""
    package = Path(andrey.__file__).resolve().parent
    return {
        "commit": _git("rev-parse", "HEAD") or "unknown",
        "uncommitted_changes": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "andrey_path": str(package),
        "andrey_commit": _git("rev-parse", "HEAD", cwd=package) or None,
    }


def header(code: dict[str, Any], seeds: int, shuffle: int, workers: int) -> dict[str, Any]:
    """Return ``code`` (:func:`commit`, read before the fits), the versions, the BLAS and OpenMP
    libraries loaded by now, the CPU, and the thread settings."""
    blas = [
        {key: value for key, value in pool.items() if key != "filepath"}
        for pool in threadpoolctl.threadpool_info()
    ]
    return {
        **code,
        "andrey": andrey.__version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "blas": sorted(blas, key=lambda pool: json.dumps(pool, sort_keys=True)),
        "cpu": _cpu_model(),
        "machine": f"{platform.system()} {platform.machine()}",
        "threads": {
            "backend": "numpy",
            "blas_threads": 1,
            "num_workers": 1,
            **{name: os.environ.get(name) for name in THREAD_VARIABLES},
        },
        "corpus": {"seeds": seeds, "shuffle": shuffle, "workers": workers},
        # A refit on more workers sends every GES pass and every skeleton pass to the worker pool.
        "refit_ges_parallel_min_work": "0" if workers > 1 else None,
        "refit_skeleton_min_pool_queries": 0 if workers > 1 else None,
    }


def _git(*args: str, cwd: Path = ROOT) -> str:
    try:
        done = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    except OSError:
        return ""
    return done.stdout.strip() if done.returncode == 0 else ""


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


@contextmanager
def _environ(name: str, value: str) -> Generator[None]:
    old = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if old is None:
            del os.environ[name]
        else:
            os.environ[name] = old


def row(method: str, dataset: str, seed: int, permutation: int, workers: int, out: Any) -> dict:
    return {
        "method": method,
        "dataset": dataset,
        "seed": seed,
        "permutation": permutation,
        "workers": workers,
        "hash": digest(out),
        "graph_hash": graph_digest(out),
    }


def run(
    methods: list[str], seeds: int, shuffle: int = 0, workers: int = 1
) -> tuple[list[dict], list[tuple[str, str]]]:
    """Fit ``methods`` on the corpus; return the manifest's rows and the refits that differ.

    Each draw is fitted once with its queries counted. ``shuffle`` refits it on that many column
    orders, the k-th drawn from ``(seed, k)``, and hashes each answer mapped back
    (:func:`unpermute`). ``workers`` above 1 refits :data:`WORKER_METHODS` on that many workers,
    with every GES pass and every skeleton pass of conditioning size two or more sent to the worker
    pool. A refit differs when one of its :func:`parts`
    differs from the first fit's, mapped back the same way for a column order; each difference is
    reported as ``("graph" | "bits", description)``, naming the parts that changed.
    """
    rows, findings = [], []
    with andrey.config(backend="numpy", num_workers=1), threadpool_limits(limits=1):
        for method, dataset, seed, x in corpus(methods, seeds):
            fit = FITS[method]
            with counting(fit.alpha) as counts:
                out = fit.call(x)
            rows.append({**row(method, dataset, seed, 0, 1, out), **counts.summary()})
            print(f"{method:24s} {dataset:40s} seed {seed}  {_brief(rows[-1])}", flush=True)
            refits = []  # (permutation, workers, the first fit hashed as the refit is, the refit)
            n_vars = x.shape[-1] // fit.blocks
            for k in range(1, shuffle + 1):
                order = np.random.default_rng([seed, k]).permutation(n_vars)
                refit = unpermute(fit.call(permuted(x, order, fit.blocks)), order)
                refits.append((k, 1, unpermute(out, np.arange(n_vars)), refit))
            if workers > 1 and method in WORKER_METHODS:
                with (
                    andrey.config(num_workers=workers),
                    _environ("ANDREY_GES_PARALLEL_MIN_WORK", "0"),
                    _environ("ANDREY_SKELETON_PARALLEL_MIN_WORK", "0"),
                ):
                    refits.append((0, workers, out, fit.call(x)))
            base = rows[-1]
            for permutation, n_workers, first, refit in refits:
                rows.append(row(method, dataset, seed, permutation, n_workers, refit))
                a, b = parts(first), parts(refit)
                changed = sorted(name for name in a.keys() | b.keys() if a.get(name) != b.get(name))
                if changed:
                    kind = "graph" if rows[-1]["graph_hash"] != base["graph_hash"] else "bits"
                    findings.append((kind, f"{_label(rows[-1])}: {', '.join(changed)}"))
    return rows, findings


def _label(r: dict) -> str:
    refit = f"permutation {r['permutation']}" if r["permutation"] else f"{r['workers']} workers"
    return f"{r['method']} {r['dataset']} seed {r['seed']} {refit}"


def _brief(r: dict) -> str:
    keys = ("ci_queries", "ci_after_separation", "score_queries")
    return ", ".join(f"{key} {r[key]}" for key in keys if key in r)


def write(path: Path, head: dict, rows: list[dict]) -> None:
    """Write the manifest, one row per line, so a diff shows one line per changed fit."""
    lines = ",\n".join(f"    {json.dumps(r)}" for r in rows)
    head_text = json.dumps(head, indent=2).replace("\n", "\n  ")
    path.write_text(f'{{\n  "header": {head_text},\n  "rows": [\n{lines}\n  ]\n}}\n')


# ==================================================================================================
# Compare
# ==================================================================================================

KEY = ("method", "dataset", "seed", "permutation", "workers")


def _counters(r: dict) -> dict[str, int]:
    """Flatten a row's counts: each total, and each set size as ``ci_sets[2]``."""
    out = {}
    for key, value in r.items():
        if isinstance(value, dict):
            out.update({f"{key}[{size}]": n for size, n in value.items()})
        elif key in ("ci_queries", "ci_after_separation", "score_queries"):
            out[key] = value
    return out


def compare(before: dict, after: dict, *, allow_fewer: bool = False) -> int:
    """Print how manifest ``after`` differs from ``before``; return 1 if a row's hash differs, a row
    is in one only, or a count rose or fell, else 0. A count that fell passes with ``allow_fewer``:
    it is either a real algorithmic change or a query the counting no longer sees."""
    for key in sorted(before["header"].keys() | after["header"].keys()):
        if before["header"].get(key) != after["header"].get(key):
            print(f"header {key}: {before['header'].get(key)} -> {after['header'].get(key)}")
    old = {tuple(r[k] for k in KEY): r for r in before["rows"]}
    new = {tuple(r[k] for k in KEY): r for r in after["rows"]}
    graph, bits, raised, lowered = [], [], [], []
    for key in sorted(old.keys() & new.keys(), key=str):
        a, b = old[key], new[key]
        label = " ".join(map(str, key))
        if a["graph_hash"] != b["graph_hash"]:
            graph.append(label)
        elif a["hash"] != b["hash"]:
            bits.append(label)
        ca, cb = _counters(a), _counters(b)
        for name in sorted(ca.keys() | cb.keys()):
            change = f"{label} {name}: {ca.get(name, 0)} -> {cb.get(name, 0)}"
            if cb.get(name, 0) > ca.get(name, 0):
                raised.append(change)
            elif cb.get(name, 0) < ca.get(name, 0):
                lowered.append(change)
    only = sorted(" ".join(map(str, k)) for k in old.keys() ^ new.keys())
    for title, lines in (
        ("graph differs", graph),
        ("only bits differ", bits),
        ("in one manifest only", only),
        ("count rose", raised),
        ("count fell", lowered),
    ):
        for line in lines:
            print(f"{title}: {line}")
    print(
        f"\n{len(old.keys() & new.keys())} rows in both: {len(graph)} with a different graph, "
        f"{len(bits)} with only different bits; {len(only)} in one only; "
        f"{len(raised)} counts rose, {len(lowered)} fell."
    )
    return 1 if graph or bits or only or raised or (lowered and not allow_fewer) else 0


# ==================================================================================================
# Command line
# ==================================================================================================


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Hash every supported method's full output.")
    ap.add_argument("methods", nargs="*", help="methods to fit (default: all supported)")
    ap.add_argument("-o", "--out", type=Path, help="write the manifest to this file")
    ap.add_argument("--seeds", type=int, default=SEEDS, help="draws per dataset")
    ap.add_argument("--shuffle", type=int, default=0, metavar="K", help="refit on K column orders")
    ap.add_argument("--workers", type=int, default=1, metavar="N", help="refit on N workers")
    ap.add_argument("--compare", nargs=2, type=Path, metavar=("BEFORE", "AFTER"))
    ap.add_argument(
        "--allow-fewer-queries", action="store_true", help="with --compare, pass when a count fell"
    )
    args = ap.parse_args(argv)

    if args.compare:
        before, after = (json.loads(path.read_text()) for path in args.compare)
        return compare(before, after, allow_fewer=args.allow_fewer_queries)
    unknown = sorted(set(args.methods) - set(FITS))
    if unknown:
        ap.error(f"unknown methods {unknown}; known: {list(FITS)}")
    code = commit()
    rows, findings = run(args.methods or list(FITS), args.seeds, args.shuffle, args.workers)
    if args.out:
        write(args.out, header(code, args.seeds, args.shuffle, args.workers), rows)
    for kind, line in findings:
        print(f"{'graph differs' if kind == 'graph' else 'only bits differ'}: {line}")
    refits = sum(r["permutation"] > 0 or r["workers"] > 1 for r in rows)
    graph = sum(kind == "graph" for kind, _ in findings)
    print(
        f"\n{len(rows)} fits hashed. Of {refits} refits, {graph} have a different graph and "
        f"{len(findings) - graph} only different bits."
    )
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
