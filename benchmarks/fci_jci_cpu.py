#!/usr/bin/env python
"""Benchmark FCI-JCI on CPU: Andrey against pcalg's FCI-JCI, on the JCI paper's simulation.

Each cell of :data:`CELLS` draws its datasets with
:func:`~andrey_bench.regime_models.jci_diagonal`, the linear-Gaussian simulation of Mooij,
Magliacane, and Claassen (2020, Sec. 5.2): ``p`` system variables, hidden confounders with exactly
two children each, and context variables in a diagonal design, one regime of ``n_per`` rows per
context. The environment is one column, the context variable (``q = 1``), or, in the merged cells,
the regime number of ``q`` context variables. Cells marked ``claim="out"`` leave the claimed setting
(several context variables merged into one index, other row counts, 20 and 50 variables) or the
reference's reach (100 variables).

Every arm fits the same rows, one after another, each in a fresh interpreter on one core with one
BLAS thread (:func:`andrey_bench.runner.run_task`), at alpha 0.01 and 0.05 (:data:`ARMS`):

- Andrey's ``fci_jci`` at ``confounded_index=True`` (JCI Assumption 1) and ``False`` (Assumptions
  1-3), the gated arms;
- pcalg's ``fci`` with ``jci = "1"`` and ``jci = "123"``, the gated references, each paired with the
  Andrey arm that makes the same assumptions;
- as context: Tetrad's FCI with the environment in the first knowledge tier (tetrad-lib 7.6.10 with
  the rules for selection bias off, and causal-cmd 1.12.0, whose FCI keeps them on), Andrey's
  ``fci`` on the pooled rows without the environment and on the observational rows, Andrey's
  ``cdnod``, Andrey's ``fci_jci`` with the columns in reverse order, and, in the merged cells,
  Andrey's ``fci_jci`` with the mixed-data test, which reads the environment as categorical.

The truth of each assumption set is FCI-JCI's PAG at the true distribution, built as
:func:`andrey.data.latent.marginal` builds the plain PAG, with FCI-JCI's marks on the environment's
edges (:func:`projected_pag`). Each dataset checks it against the true MAG and the plain PAG: the
same adjacencies, every mark that is not a circle the MAG's, and every mark the plain PAG sets kept;
up to 10 variables, also that Andrey's engine with :class:`~andrey.data.latent.DSepOracle` in place
of its tests returns it. ``--check-pcalg`` also checks the plain PAG against pcalg's ``dag2pag`` (on
the DAG numbered in a topological order, which it needs) and each truth against pcalg's ``fci``
with ``jci`` and a d-separation oracle; ``truth_ok`` on the dataset line says whether every check
passed.

A fit past its cap is a ``timeout`` with no scores, never an empty graph. :func:`scores` computes,
against the truth of the arm's assumptions, the endpoint SHD over the data variables and the
environment (``shd``) and over the data variables alone (``shd_data``, the only scope of the arms
without the environment), and the precision, recall, and F1 of the adjacencies, the arrowheads,
and the tails; the empty graph is scored the same way. Each Andrey FCI fit is repeated untimed in
the parent to record its largest possible-d-separation set (``pdsep_max``) and to check that the
repeat returns the same marks.

One JSON line per dataset goes to ``<out>/datasets.jsonl`` and one per (cell, seed, arm, alpha) to
``<out>/records.jsonl``; ``--resume`` skips what is on disk. ``summarize`` pairs the records of one
or more runs into the gated rows: per row, the percentile bootstrap 95% interval (10,000 resamples)
of the mean paired difference in ``shd``, Andrey minus pcalg, judged only where pcalg's own interval
against the empty graph lies below 0.

    $ANDREY_BENCH_PYTHON benchmarks/fci_jci_cpu.py run --out outputs/fci-jci --cells S4 L10 \\
        --seeds 0-19 --check-pcalg
    $ANDREY_BENCH_PYTHON benchmarks/fci_jci_cpu.py summarize --runs outputs/fci-jci \\
        --out outputs/fci-jci-summary.json
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import platform
import socket
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from andrey_bench.regime_models import JCIDraw, jci_diagonal, merged_index_dag
from andrey_bench.runner import run_task

# Andrey's endpoint marks: the mark at node i on the edge i-j is marks[i, j].
NULL, TAIL, ARROW, CIRCLE = 0, 1, 2, 3

ALPHAS = (0.01, 0.05)

#: Wall cap of one fit, imports and warm-up included.
CAP_WALL_S = 1800.0

#: Memory cap of one fit's process tree.
CAP_MEM_MB = 16000.0

BOOTSTRAP = 10_000
MIN_PAIRS = 10

#: The largest cell whose pcalg fits take a warm-up fit before the timed one; above it the warm-up
#: would double a fit that may run to the cap. Every other arm always takes one.
WARMUP_MAX_P = 10


@dataclass(frozen=True)
class Cell:
    """One setting of :func:`~andrey_bench.regime_models.jci_diagonal`; see the module docstring."""

    name: str
    p: int
    q: int
    eps: float
    n_per: int
    claim: str
    surgical: bool = False
    alphas: tuple[float, ...] = ALPHAS
    context: bool = False  # run the context arms
    reversed: bool = False  # run the column-order arm
    reference: bool = True  # run pcalg

    @property
    def merged(self) -> bool:
        """Whether the environment merges several context variables into one regime number."""
        return self.q > 1


CELLS = {
    cell.name: cell
    for cell in (
        Cell("S4", 4, 1, 0.5, 500, "in", context=True),
        Cell("S4p", 4, 1, 0.5, 500, "in", surgical=True),
        Cell("L10", 10, 1, 0.25, 500, "in", context=True, reversed=True),
        Cell("X100", 100, 1, 0.02, 100, "in", context=True),
        Cell("S4m", 4, 2, 0.5, 500, "out"),
        Cell("L10m", 10, 10, 0.25, 500, "out"),
        Cell("X100m", 100, 10, 0.02, 100, "out", alphas=(0.01,)),
        Cell("L10n100", 10, 1, 0.25, 100, "out"),
        Cell("L10n2000", 10, 1, 0.25, 2000, "out"),
        Cell("P20", 20, 1, 2.25 / 19, 500, "out", alphas=(0.01,), context=True),
        Cell("P50", 50, 1, 2.25 / 49, 500, "out", alphas=(0.01,), context=True),
    )
}


@dataclass(frozen=True)
class Arm:
    """One solution with its settings: ``variant`` names the truth it is scored against, ``"1"``
    or ``"123"``, and ``scope`` whether its graph covers the environment (``"full"``) or the data
    variables alone (``"data"``)."""

    name: str
    variant: str
    scope: str
    role: str  # "andrey", "reference", or "context"
    fci: bool  # an Andrey FCI fit: repeated untimed for its possible-d-sep sets


ARMS = {
    arm.name: arm
    for arm in (
        Arm("andrey.jci1", "1", "full", "andrey", True),
        Arm("andrey.jci123", "123", "full", "andrey", True),
        Arm("pcalg.jci1", "1", "full", "reference", False),
        Arm("pcalg.jci123", "123", "full", "reference", False),
        Arm("tetrad.tier", "1", "full", "context", False),
        Arm("causal-cmd.tier", "1", "full", "context", False),
        Arm("andrey.fci.pooled", "1", "data", "context", True),
        Arm("andrey.fci.obs", "1", "data", "context", True),
        Arm("andrey.cdnod", "1", "data", "context", False),
        Arm("andrey.jci1.reversed", "1", "full", "context", True),
        Arm("andrey.jci1.cond_gauss", "1", "full", "context", True),
    )
}

#: The gated pairs: Andrey's arm and the reference arm that makes the same JCI assumptions.
PAIRS = (("andrey.jci1", "pcalg.jci1"), ("andrey.jci123", "pcalg.jci123"))


def cell_arms(cell: Cell) -> list[str]:
    """The arms a cell runs (:data:`ARMS`)."""
    arms = ["andrey.jci1", "andrey.jci123"]
    if cell.reference:
        arms += ["pcalg.jci1", "pcalg.jci123"]
    if cell.context:
        arms += ["tetrad.tier", "causal-cmd.tier"]
        if cell.q == 1 and cell.p <= 10:
            arms += ["andrey.fci.pooled", "andrey.fci.obs", "andrey.cdnod"]
    if cell.reversed:
        arms.append("andrey.jci1.reversed")
    if cell.merged and cell.p <= 10:
        arms.append("andrey.jci1.cond_gauss")
    return arms


# --- data and truth --------------------------------------------------------------------------------


@dataclass
class Dataset:
    """A cell's draw for one seed, with the environment column and the DAG the truth reads."""

    cell: Cell
    seed: int
    draw: JCIDraw
    env: np.ndarray  # (n,): the context variable, or the regime number in a merged cell
    dag: np.ndarray  # the DAG over the data, the environment at position p, then the hidden nodes

    @property
    def full(self) -> np.ndarray:
        """The data with the environment as the last column."""
        return np.column_stack([self.draw.data, self.env])


def draw(cell: Cell, seed: int) -> Dataset:
    """Draw `cell`'s dataset for `seed`."""
    d = jci_diagonal(cell.p, cell.q, cell.eps, cell.eps, cell.n_per, seed, cell.surgical)
    if cell.merged:
        return Dataset(
            cell, seed, d, d.regime.astype(float), merged_index_dag(d.adjacency, cell.p, cell.q)
        )
    return Dataset(cell, seed, d, d.context[:, 0].copy(), d.adjacency)


def _digraph(dag: np.ndarray):
    import networkx as nx

    g = nx.DiGraph()
    g.add_nodes_from(range(dag.shape[0]))
    g.add_edges_from((int(u), int(v)) for u, v in np.argwhere(dag != 0))
    return g


@contextlib.contextmanager
def possible_dsep_sizes():
    """Record the size of every possible-d-separation set Andrey's FCI step forms.

    Yields a dict with ``max`` and ``calls``. The engine reads ``_possible_dsep`` from
    :mod:`andrey.constraint.fci` at each call, so wrapping it there reaches ``fci`` and ``fci_jci``.
    """
    module = importlib.import_module("andrey.constraint.fci")
    real = module._possible_dsep
    seen = {"max": 0, "calls": 0}

    def wrapped(*args, **kwargs):
        found = real(*args, **kwargs)
        seen["max"] = max(seen["max"], len(found))
        seen["calls"] += 1
        return found

    module._possible_dsep = wrapped
    try:
        yield seen
    finally:
        module._possible_dsep = real


def oracle_pag(dag: np.ndarray, p: int, confounded_index: bool) -> tuple[np.ndarray, dict]:
    """FCI-JCI's PAG over the data and the environment (at `p`) under d-separation in `dag`.

    Returns the marks and the largest possible-d-separation set the oracle run formed.
    """
    from andrey.constraint.fci_jci import fci_jci as engine
    from andrey.core import ci
    from andrey.data.latent import DSepOracle
    from andrey_bench.adapters.andrey_fci_jci import with_index

    graph = _digraph(dag)
    ci._CI_REGISTRY["fci_jci_lane_oracle"] = lambda _data: DSepOracle(graph)
    with possible_dsep_sizes() as seen:
        pag, index_edges = engine(
            np.zeros((2, p)),
            np.array([0.0, 1.0]),
            indep_test="fci_jci_lane_oracle",
            possible_dsep="fast",
            confounded_index=confounded_index,
        )
    return with_index(np.asarray(pag.to_numpy()), index_edges, p), dict(seen)


def dag_marks(dag: np.ndarray) -> np.ndarray:
    """Endpoint marks of a DAG adjacency."""
    marks = np.zeros(dag.shape, dtype=np.int8)
    for u, v in np.argwhere(dag != 0):
        marks[u, v], marks[v, u] = TAIL, ARROW
    return marks


#: The largest cell whose truth is also built by Andrey's FCI-JCI engine with d-separation in
#: place of its tests, a check of the projection; the engine's possible-d-separation step tests every
#: subset of each set, too many past this size.
ENGINE_CHECK_MAX_P = 10


def projected_pag(graph, p: int, confounded_index: bool) -> tuple[np.ndarray, int]:
    """FCI-JCI's PAG over the observed nodes of `graph`, the environment at `p`, by projection.

    The steps of :func:`andrey.data.latent.marginal`'s PAG, with FCI-JCI's marks on the
    environment's edges where ``marginal`` sets background knowledge, before the colliders, and the
    rules without R5-R7, as FCI-JCI runs them: the adjacencies of the true MAG, the separating sets
    of the observed ancestors, and the discriminating-path rule answered by d-separation. Also
    returns the largest possible-d-separation set FCI forms on the true graph at the step where it
    forms them, after the colliders.
    """
    from andrey.constraint.fci_jci import _mark_index
    from andrey.data import latent

    fci = importlib.import_module("andrey.constraint.fci")
    n_obs, dag, anc = latent._observed_dag(graph, max_nodes=graph.n_nodes)
    observed = range(n_obs)
    support = latent._ancestral_adjacency(dag, observed, ())
    sepsets = latent._AncestorSepsets(support, observed, anc, set())
    marks = fci._circle_pag(support)
    _mark_index(marks, p, confounded_index)
    fci._orient_colliders(marks, sepsets)
    neighbors, reach = fci._neighbor_lists(marks), fci._semidirected_reach(marks)
    largest = max(
        (
            len(fci._possible_dsep(marks, x, y, neighbors, reach))
            for a, b in zip(*np.nonzero(np.triu(marks)))
            for x, y in ((int(a), int(b)), (int(b), int(a)))
        ),
        default=0,
    )
    fci._fci_orient(marks, latent.DSepOracle(dag, observed, ()), 0.5, sepsets, selection_bias=False)
    return np.asarray(marks, dtype=np.int8), largest


def truth(data: Dataset) -> dict[str, Any]:
    """Both truths (:func:`projected_pag`), the plain PAG and the MAG of
    :func:`andrey.data.latent.marginal`, the checks that relate them, and the largest
    possible-d-separation set on the true graph."""
    from andrey import GraphStructure
    from andrey.core.structure import LATENT, OBSERVED
    from andrey.data.latent import marginal

    p, dag = data.cell.p, data.dag
    types = np.array([OBSERVED] * (p + 1) + [LATENT] * (dag.shape[0] - p - 1), dtype=np.int8)
    graph = GraphStructure.from_numpy(dag_marks(dag), kind="dag", node_types=types)
    t0 = time.perf_counter()
    out: dict[str, Any] = {
        "plain": np.asarray(marginal(graph, "pag", max_nodes=p + 1).to_numpy()),
        "mag": np.asarray(marginal(graph, "mag", max_nodes=p + 1).to_numpy()),
    }
    largest = {}
    for variant, confounded in (("1", True), ("123", False)):
        out[variant], largest[variant] = projected_pag(graph, p, confounded)
    out["seconds"] = time.perf_counter() - t0
    out["pdsep_max_true"] = max(largest.values())
    checks = {}
    known = (out["plain"] != NULL) & (out["plain"] != CIRCLE)
    for variant, confounded in (("1", True), ("123", False)):
        found = out[variant]
        resolved = (found != NULL) & (found != CIRCLE)
        checks[f"jci{variant}_skeleton_is_the_mag"] = bool(
            np.array_equal(found != NULL, out["mag"] != NULL)
        )
        checks[f"jci{variant}_marks_are_the_mag"] = bool(
            np.array_equal(found[resolved], out["mag"][resolved])
        )
        checks[f"jci{variant}_keeps_the_plain_marks"] = bool(
            np.array_equal(found[known], out["plain"][known])
        )
        if p <= ENGINE_CHECK_MAX_P:
            engine, _ = oracle_pag(dag, p, confounded)
            checks[f"jci{variant}_is_the_engines_oracle_pag"] = bool(np.array_equal(engine, found))
    out["checks"] = checks
    return out


#: pcalg's two checks of the truth, on the DAG with the observed nodes first: ``dag2pag`` (on the
#: DAG in a topological order, since its ancestor search reads parents among lower numbers only)
#: and ``fci`` with ``jci`` and a d-separation oracle. The oracle answers by the moral graph of the
#: ancestral set (Lauritzen et al. 1990), memoized, the same answers as ``pcalg::dsepTest`` faster;
#: ``andrey_oracle_mismatches`` compares the two on random queries.
_R_TRUTH = """
andrey_fast_dsep <- function(A, jp) {
    n <- nrow(A)
    pa <- lapply(seq_len(n), function(j) which(A[, j] == 1))
    ch <- lapply(seq_len(n), function(i) which(A[i, ] == 1))
    anc <- is.finite(jp)
    cache <- new.env(hash = TRUE)
    function(x, y, S = NULL, suffStat = NULL) {
        if (x == y || x %in% S || y %in% S) return(0)
        key <- paste(min(x, y), max(x, y), paste(sort(S), collapse = ","))
        v <- cache[[key]]
        if (!is.null(v)) return(v)
        inAn <- rowSums(anc[, c(x, y, S), drop = FALSE]) > 0
        blocked <- logical(n); blocked[S] <- TRUE
        seen <- logical(n); seen[x] <- TRUE
        stack <- x; found <- FALSE
        while (length(stack) > 0 && !found) {
            u <- stack[length(stack)]; stack <- stack[-length(stack)]
            nb <- pa[[u]]
            for (c in ch[[u]]) if (inAn[c]) nb <- c(nb, c, pa[[c]])
            if (y %in% nb) { found <- TRUE; break }
            nb <- unique(nb[!seen[nb] & !blocked[nb]])
            seen[nb] <- TRUE
            stack <- c(stack, nb)
        }
        v <- if (found) 0 else 1
        assign(key, v, envir = cache)
        v
    }
}

andrey_graph <- function(A) {
    g <- methods::as(A, "graphNEL")
    list(g = g, jp = RBGL::johnson.all.pairs.sp(g))
}

andrey_oracle_mismatches <- function(A, k, queries) {
    suff <- andrey_graph(A)
    oracle <- andrey_fast_dsep(A, suff$jp)
    set.seed(1)
    wrong <- 0
    for (i in seq_len(queries)) {
        xy <- sample.int(k, 2); rest <- setdiff(seq_len(k), xy)
        size <- sample.int(min(6, length(rest)) + 1, 1) - 1
        S <- if (size > 0) rest[sample.int(length(rest), size)] else integer(0)
        if (oracle(xy[1], xy[2], S) != pcalg::dsepTest(xy[1], xy[2], S, suff)) wrong <- wrong + 1
    }
    wrong
}

andrey_dag2pag <- function(A, k) {
    perm <- match(RBGL::tsort(methods::as(A, "graphNEL")), colnames(A))
    A2 <- A[perm, perm]
    stopifnot(all(A2[lower.tri(A2)] == 0))
    suff2 <- andrey_graph(A2)
    obs <- perm[perm <= k]
    fit <- pcalg::dag2pag(suff2, indepTest = pcalg::dsepTest, graph = suff2$g,
                          L = which(perm > k), alpha = 0.5)
    amat <- matrix(0, k, k, dimnames = list(colnames(A)[1:k], colnames(A)[1:k]))
    amat[obs, obs] <- fit@amat
    amat
}

andrey_fci_jci_oracle <- function(A, k, jci, context) {
    suff <- andrey_graph(A)
    fit <- pcalg::fci(suff, indepTest = andrey_fast_dsep(A, suff$jp), alpha = 0.5,
                      labels = colnames(A)[1:k], type = "normal", jci = jci,
                      contextVars = context, selectionBias = FALSE)
    fit@amat
}
"""


def pcalg_check(data: Dataset, found: dict[str, Any], queries: int = 200) -> dict[str, Any]:
    """Compare the plain PAG with pcalg's ``dag2pag`` and each truth with pcalg's oracle FCI-JCI."""
    import rpy2.robjects as ro

    from andrey_bench import rsession
    from andrey_bench.adapters.r_convert import marks_from_amat_pag

    session = rsession.start("pcalg", "graph", "RBGL")
    session.define(_R_TRUTH)
    p, dag = data.cell.p, data.dag
    k = p + 1
    names = rsession.columns(dag.shape[0])
    A = session.matrix(dag.astype(float))
    ro.r.assign("andrey_A", A)
    A = ro.r("rownames(andrey_A) <- colnames(andrey_A); andrey_A")
    t0 = time.perf_counter()
    out: dict[str, Any] = {
        "oracle_mismatches": int(session.function("andrey_oracle_mismatches")(A, k, queries)[0]),
        "oracle_queries": queries,
    }
    plain = marks_from_amat_pag(
        session.square_matrix(session.function("andrey_dag2pag")(A, k), names=names[:k])
    )
    out["dag2pag_equals_plain"] = bool(np.array_equal(plain, found["plain"]))
    for variant in ("1", "123"):
        amat = session.function("andrey_fci_jci_oracle")(A, k, variant, ro.IntVector([k]))
        theirs = marks_from_amat_pag(session.square_matrix(amat, names=names[:k]))
        out[f"fci_jci{variant}_equals_truth"] = bool(np.array_equal(theirs, found[variant]))
        if not out[f"fci_jci{variant}_equals_truth"]:
            out[f"fci_jci{variant}_pcalg"] = edge_list(theirs)
    out["seconds"] = time.perf_counter() - t0
    return out


# --- arms ------------------------------------------------------------------------------------------


def adapter(arm: str):
    """A fresh adapter for `arm`."""
    if arm.startswith("andrey.jci"):
        from andrey_bench.adapters.andrey_fci_jci import AndreyFCIJCI

        return AndreyFCIJCI()
    if arm.startswith("pcalg."):
        from andrey_bench.adapters.pcalg_fci_jci import PcalgFCIJCI

        return PcalgFCIJCI()
    if arm == "tetrad.tier":
        from andrey_bench.adapters.tetrad_search import TetradFCIJCI

        return TetradFCIJCI()
    if arm == "causal-cmd.tier":
        from andrey_bench.adapters.tetrad_pag import TetradCmdFCIJCI

        return TetradCmdFCIJCI()
    if arm.startswith("andrey.fci."):
        from andrey_bench.adapters.andrey_fci import AndreyFCI

        return AndreyFCI()
    if arm == "andrey.cdnod":
        from andrey_bench.adapters.andrey_cdnod import AndreyCDNOD

        return AndreyCDNOD()
    raise KeyError(arm)


def inputs(arm: str, data: Dataset, alpha: float) -> tuple[np.ndarray, dict[str, Any]]:
    """The array `arm` fits and its ``params``."""
    p, q = data.cell.p, data.cell.q
    full, context = data.full, [p]
    if arm == "andrey.jci1":
        return full, {"alpha": alpha, "confounded_index": True, "context_vars": context}
    if arm == "andrey.jci123":
        return full, {"alpha": alpha, "confounded_index": False, "context_vars": context}
    if arm == "andrey.jci1.reversed":
        return full[:, ::-1].copy(), {"alpha": alpha, "confounded_index": True, "context_vars": [0]}
    if arm == "andrey.jci1.cond_gauss":
        params = {"alpha": alpha, "indep_test": "cond_gauss", "context_vars": context}
        return full, params
    if arm in ("pcalg.jci1", "pcalg.jci123"):
        jci = arm.removeprefix("pcalg.jci")
        return full, {"alpha": alpha, "jci": jci, "context_vars": context}
    if arm in ("tetrad.tier", "causal-cmd.tier"):
        return full, {"alpha": alpha, "context_vars": context}
    if arm == "andrey.fci.pooled":
        return data.draw.data, {"alpha": alpha, "indep_test": "fisherz"}
    if arm == "andrey.fci.obs":
        return data.draw.data[data.draw.regime == 0], {"alpha": alpha, "indep_test": "fisherz"}
    if arm == "andrey.cdnod":
        return data.draw.data, {"alpha": alpha, "indep_test": "fisherz", "n_domains": q + 1}
    raise KeyError(arm)


def estimate(arm: str, native: np.ndarray) -> np.ndarray:
    """The marks of `arm`'s graph in the dataset's column order."""
    marks = np.asarray(native, dtype=np.int8)
    if arm == "andrey.jci1.reversed":
        return marks[::-1, ::-1].copy()
    return marks


# --- scores ----------------------------------------------------------------------------------------


def edge_list(marks: np.ndarray) -> list[list[int]]:
    """``[i, j, mark at i, mark at j]`` for each adjacent pair ``i < j``."""
    m = np.asarray(marks)
    return [[int(i), int(j), int(m[i, j]), int(m[j, i])] for i, j in zip(*np.nonzero(np.triu(m)))]


def from_edge_list(edges: list[list[int]], d: int) -> np.ndarray:
    """Inverse of :func:`edge_list` for `d` nodes."""
    marks = np.zeros((d, d), dtype=np.int8)
    for i, j, mi, mj in edges:
        marks[i, j], marks[j, i] = mi, mj
    return marks


def _prf(est: set, true: set, name: str) -> dict[str, float | None]:
    hit = len(est & true)
    return {
        f"{name}_precision": hit / len(est) if est else None,
        f"{name}_recall": hit / len(true) if true else None,
        f"{name}_f1": 2 * hit / (len(est) + len(true)) if est or true else None,
    }


def endpoint_shd(est: np.ndarray, true: np.ndarray) -> int:
    """Pairs whose two marks differ: a missing or extra edge costs both of its marks."""
    from andrey import GraphStructure
    from andrey.metrics import shd

    return int(
        shd(
            GraphStructure.from_numpy(est, kind="pag"),
            GraphStructure.from_numpy(true, kind="pag"),
            endpoint_aware=True,
        )
    )


def scores(est: np.ndarray, true: np.ndarray) -> dict[str, Any]:
    """Endpoint SHD and the adjacency, arrowhead, and tail precision, recall, and F1.

    An arrowhead (tail) is an endpoint, a node on one of its edges; it counts as found when the
    estimate has the same edge with the same mark at that node.
    """
    est, true = np.asarray(est), np.asarray(true)
    out: dict[str, Any] = {"shd": endpoint_shd(est, true)}

    def pairs(m):
        return {(int(i), int(j)) for i, j in zip(*np.nonzero(np.triu(m | m.T)))}

    out |= _prf(pairs(est), pairs(true), "adjacency")
    for name, mark in (("arrowhead", ARROW), ("tail", TAIL)):
        found = {(int(i), int(j)) for i, j in np.argwhere(est == mark)}
        wanted = {(int(i), int(j)) for i, j in np.argwhere(true == mark)}
        out |= _prf(found, wanted, name)
    out["edges"] = len(pairs(est))
    return out


def score_arm(arm: str, est: np.ndarray, true: dict[str, Any], p: int) -> dict[str, Any]:
    """Scores of `arm`'s graph against the truth of its assumptions, in its scope."""
    spec = ARMS[arm]
    target = true[spec.variant]
    if spec.scope == "full":
        out = scores(est, target)
        out["shd_data"] = endpoint_shd(est[:p, :p], target[:p, :p])
        return out
    out = {f"{k}_data" if k == "shd" else k: v for k, v in scores(est, target[:p, :p]).items()}
    return out | {"shd": None}


# --- run -------------------------------------------------------------------------------------------


def _cpu_model() -> str:
    with contextlib.suppress(OSError):
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _append(path: Path, record: dict) -> None:
    with path.open("a") as fh:
        fh.write(json.dumps(record, default=_json_default) + "\n")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value))


def _seeds(text: str) -> list[int]:
    seeds: list[int] = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        seeds += list(range(int(lo), int(hi or lo) + 1))
    return seeds


def refit(arm: str, x: np.ndarray, params: dict) -> tuple[np.ndarray, dict]:
    """Repeat an Andrey FCI fit in this process, untimed, recording its possible-d-sep sets."""
    solution = adapter(arm)
    solution.setup()
    with possible_dsep_sizes() as seen:
        marks = np.ascontiguousarray(solution.fit(x, params))
    return marks, dict(seen)


def fit_arm(arm: str, data: Dataset, alpha: float, true: dict, args) -> dict[str, Any]:
    """Fit `arm` on `data` at `alpha` in a fresh interpreter; return its record."""
    x, params = inputs(arm, data, alpha)
    solution = adapter(arm)
    warmup = 1 if data.cell.p <= WARMUP_MAX_P or ARMS[arm].role != "reference" else 0
    params = {**solution.params(), **params}
    result = run_task(
        solution,
        np.ascontiguousarray(x),
        params,
        cap_wall_s=args.cap,
        cap_mem_mb=args.cap_mem_mb,
        repeats=1,
        warmup=warmup,
        threads=1,
    )[0]
    record: dict[str, Any] = {
        "arm": arm,
        "alpha": alpha,
        "variant": ARMS[arm].variant,
        "scope": ARMS[arm].scope,
        "role": ARMS[arm].role,
        "solution": solution.name,
        "package_version": str(getattr(solution, "package_version", "")),
        "params": params,
        "status": result["status"],
        "cap_wall_s": args.cap,
        "warmup": warmup,
    }
    for key in ("wall_s", "cpu_s", "process_s", "peak_rss_mb", "elapsed_at_kill_s", "exit_code"):
        if key in result:
            record[key] = result[key]
    for key in ("error_text", "stderr"):
        if result.get(key):
            record[key] = str(result[key])[-1500:]
    if result["status"] != "ok":
        return record
    est = estimate(arm, result["adj"])
    record["marks"] = edge_list(est)
    record["marks_sha1"] = hashlib.sha1(est.tobytes()).hexdigest()
    record |= score_arm(arm, est, true, data.cell.p)
    if ARMS[arm].fci:
        again, seen = refit(arm, x, params)
        record["pdsep_max"], record["pdsep_calls"] = seen["max"], seen["calls"]
        record["refit_same"] = hashlib.sha1(again.tobytes()).hexdigest() == result["output_hash"]
    return record


def dataset_record(data: Dataset, true: dict, check: dict | None) -> dict[str, Any]:
    """The dataset line: its setting, truth, checks, and the empty graph's scores."""
    p = data.cell.p
    empty = np.zeros((p + 1, p + 1), dtype=np.int8)
    ok = all(true["checks"].values())
    if check is not None:
        ok = ok and check.get("oracle_mismatches") == 0 and check.get("dag2pag_equals_plain")
        ok = ok and check.get("fci_jci1_equals_truth") and check.get("fci_jci123_equals_truth")
    return {
        "cell": data.cell.name,
        "seed": data.seed,
        "setting": asdict(data.cell),
        "n": int(data.draw.data.shape[0]),
        "targets": data.draw.targets.tolist(),
        "hidden": data.draw.hidden,
        "true_edges": int(np.count_nonzero(np.triu(true["1"] | true["1"].T))),
        "truth": {v: edge_list(true[v]) for v in ("1", "123", "plain")},
        "empty": {
            v: {
                "shd": endpoint_shd(empty, true[v]),
                "shd_data": endpoint_shd(empty[:p, :p], true[v][:p, :p]),
            }
            for v in ("1", "123")
        },
        "truth_checks": true["checks"],
        "pcalg_check": check,
        "truth_ok": bool(ok),
        "truth_s": true["seconds"],
        "pdsep_max_true": true["pdsep_max_true"],
    }


def run(args) -> int:
    """Fit every arm of each cell on each seed; append to ``datasets.jsonl`` and ``records.jsonl``."""
    import andrey

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rec_path, ds_path = out / "records.jsonl", out / "datasets.jsonl"
    done = (
        {(r["cell"], r["seed"], r["arm"], r["alpha"]) for r in _read(rec_path)}
        if args.resume
        else set()
    )
    have = {(r["cell"], r["seed"]) for r in _read(ds_path)} if args.resume else set()
    if not args.resume:
        for path in (rec_path, ds_path):
            path.unlink(missing_ok=True)
    meta = {
        "argv": vars(args) | {"func": None},
        "andrey": andrey.__version__,
        "andrey_path": str(Path(andrey.__file__).resolve().parent),
        "numpy": np.__version__,
        "python": platform.python_version(),
        "cpu_model": _cpu_model(),
        "machine": args.machine_id or socket.gethostname(),
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _append(out / "provenance.jsonl", meta)
    counts: dict[str, int] = {}
    for name in args.cells:
        cell = CELLS[name]
        arms = [a for a in cell_arms(cell) if not args.arms or a in args.arms]
        alphas = [a for a in cell.alphas if not args.alphas or a in args.alphas]
        for seed in args.seeds:
            data = draw(cell, seed)
            true = truth(data)
            if (name, seed) not in have:
                check = None
                if args.check_pcalg:
                    try:
                        check = pcalg_check(data, true)
                    except Exception as exc:  # recorded; the dataset's truth_ok is then False
                        check = {"error": f"{type(exc).__name__}: {exc}"}
                _append(ds_path, dataset_record(data, true, check))
            for arm in arms:
                for alpha in alphas:
                    if (name, seed, arm, alpha) in done:
                        continue
                    t0 = time.perf_counter()
                    record = {"cell": name, "seed": seed} | fit_arm(arm, data, alpha, true, args)
                    record["cpu_model"] = meta["cpu_model"]
                    _append(rec_path, record)
                    counts[record["status"]] = counts.get(record["status"], 0) + 1
                    print(
                        f"{name} seed {seed} {arm} alpha {alpha}: {record['status']} "
                        f"shd {record.get('shd', record.get('shd_data'))} "
                        f"({time.perf_counter() - t0:.1f} s)",
                        flush=True,
                    )
    _append(
        out / "done.jsonl",
        {"finished": datetime.now(timezone.utc).isoformat(timespec="seconds"), "counts": counts},
    )
    return 0


# --- summarize -------------------------------------------------------------------------------------


def bootstrap(values: list[float], resamples: int = BOOTSTRAP) -> list[float] | None:
    """Percentile bootstrap 95% interval of the mean, ``default_rng(0)``."""
    if not values:
        return None
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(0)
    means = v[rng.integers(0, len(v), size=(resamples, len(v)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return [float(lo), float(hi)]


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def paired_row(
    cell: Cell, alpha: float, a: str, r: str, recs: dict, empty: dict, metric: str = "shd"
) -> dict[str, Any]:
    """The gated row of the pair (`a`, `r`) in `cell` at `alpha`."""
    variant = ARMS[a].variant
    seeds = sorted(
        {s for (c, s, arm, al) in recs if c == cell.name and al == alpha and arm in (a, r)}
    )
    mine = {s: recs.get((cell.name, s, a, alpha)) for s in seeds}
    theirs = {s: recs.get((cell.name, s, r, alpha)) for s in seeds}
    ok = lambda rec: rec is not None and rec["status"] == "ok"  # noqa: E731
    paired = [s for s in seeds if ok(mine[s]) and ok(theirs[s])]
    d = [mine[s][metric] - theirs[s][metric] for s in paired]
    e = {s: empty[(cell.name, s)][variant][metric] for s in seeds if (cell.name, s) in empty}
    a_empty = [mine[s][metric] - e[s] for s in seeds if ok(mine[s]) and s in e]
    r_empty = [theirs[s][metric] - e[s] for s in seeds if ok(theirs[s]) and s in e]
    interval = bootstrap(d)
    floor_a, floor_r = bootstrap(a_empty), bootstrap(r_empty)
    applies = floor_r is not None and floor_r[1] < 0
    gated = cell.claim == "in" and applies and len(paired) >= MIN_PAIRS
    verdict = None
    if interval is not None:
        verdict = "worse" if interval[0] > 0 else "better" if interval[1] < 0 else "tie"
    same = sum(mine[s]["marks_sha1"] == theirs[s]["marks_sha1"] for s in paired)
    ratios = [mine[s]["wall_s"] / theirs[s]["wall_s"] for s in paired if theirs[s].get("wall_s")]
    claim = "in" if gated else "out"
    row = {
        "regime": f"{cell.name}|jci={variant}|alpha={alpha}",
        "cell": cell.name,
        "variant": variant,
        "alpha": alpha,
        "method": "fci_jci",
        "comparator": f"pcalg-jci{variant}",
        "metric": metric,
        "claim": claim,
        "seeds": seeds,
        "paired": len(paired),
        "andrey_shd_mean": _mean([mine[s][metric] for s in paired]),
        "comparator_shd_mean": _mean([theirs[s][metric] for s in paired]),
        "empty_shd_mean": _mean([e[s] for s in paired if s in e]),
        "mean_D": _mean(d),
        "bootstrap_95": interval,
        "above_zero": interval is not None and interval[0] > 0,
        "verdict": verdict,
        "same_pag": same,
        "speed_ratio_median": float(statistics.median(ratios)) if ratios else None,
        "pdsep_max": max((mine[s].get("pdsep_max", 0) for s in seeds if ok(mine[s])), default=None),
        "coverage": {
            "datasets": len(seeds),
            "andrey_short": sum(ok(theirs[s]) and not ok(mine[s]) for s in seeds),
            "reference_short": sum(ok(mine[s]) and not ok(theirs[s]) for s in seeds),
            "andrey_status": _statuses(mine.values()),
            "reference_status": _statuses(theirs.values()),
        },
        "floor": {
            "andrey_minus_empty": floor_a,
            "reference_minus_empty": floor_r,
            "applies": applies,
            "pass": (floor_a is not None and floor_a[1] < 0) if applies else None,
        },
    }
    if cell.claim == "in" and not gated:
        row["reason"] = (
            f"fewer than {MIN_PAIRS} paired datasets"
            if len(paired) < MIN_PAIRS and applies
            else "reference finished no dataset"
            if floor_r is None
            else "reference not better than the empty graph"
        )
    return row


def _statuses(records) -> dict[str, int]:
    counts: dict[str, int] = {}
    for rec in records:
        key = "missing" if rec is None else rec["status"]
        counts[key] = counts.get(key, 0) + 1
    return counts


def context_rows(cell: Cell, alpha: float, recs: dict, empty: dict) -> list[dict[str, Any]]:
    """Each arm's mean scores in `cell` at `alpha`, and its paired difference in ``shd_data``
    with Andrey's default arm."""
    rows = []
    seeds = sorted({s for (c, s, _, al) in recs if c == cell.name and al == alpha})
    base = {s: recs.get((cell.name, s, "andrey.jci1", alpha)) for s in seeds}
    for arm in cell_arms(cell):
        got = [recs.get((cell.name, s, arm, alpha)) for s in seeds]
        ok = [g for g in got if g is not None and g["status"] == "ok"]
        row: dict[str, Any] = {
            "cell": cell.name,
            "alpha": alpha,
            "arm": arm,
            "datasets": len(seeds),
        }
        row["status"] = _statuses(got)
        for key in (
            "shd",
            "shd_data",
            "adjacency_f1",
            "arrowhead_f1",
            "tail_f1",
            "tail_precision",
            "wall_s",
        ):
            vals = [g[key] for g in ok if g.get(key) is not None]
            row[f"{key}_mean"] = _mean(vals)
        row["pdsep_max"] = max((g.get("pdsep_max", 0) for g in ok), default=None)
        if ARMS[arm].fci:
            row["refit_same"] = sum(bool(g.get("refit_same")) for g in ok)
        pairs = [
            g["shd_data"] - base[g["seed"]]["shd_data"]
            for g in ok
            if base.get(g["seed"]) and base[g["seed"]]["status"] == "ok"
        ]
        row["shd_data_minus_andrey_default"] = {
            "mean": _mean(pairs),
            "bootstrap_95": bootstrap(pairs),
        }
        rows.append(row)
    empties = [empty[(cell.name, s)]["1"] for s in seeds if (cell.name, s) in empty]
    rows.append(
        {
            "cell": cell.name,
            "alpha": alpha,
            "arm": "empty",
            "datasets": len(empties),
            "shd_mean": _mean([e["shd"] for e in empties]),
            "shd_data_mean": _mean([e["shd_data"] for e in empties]),
        }
    )
    return rows


def summarize(args) -> int:
    """Pair the records of ``--runs`` into rows; ``--replications`` adds each row's rerun."""

    def load(dirs: list[str]) -> tuple[dict, dict, list]:
        """The records by (cell, seed, arm, alpha) and the datasets by (cell, seed): runs split
        into parts by seeds or arms repeat a dataset's line, with the same truth."""
        recs, empty, datasets = {}, {}, {}
        for d in dirs:
            for path in sorted(Path(d).rglob("records.jsonl")):
                for r in _read(path):
                    recs[(r["cell"], r["seed"], r["arm"], r["alpha"])] = r
            for path in sorted(Path(d).rglob("datasets.jsonl")):
                for r in _read(path):
                    empty[(r["cell"], r["seed"])] = r["empty"]
                    datasets[(r["cell"], r["seed"])] = r
        return recs, empty, list(datasets.values())

    recs, empty, datasets = load(args.runs)
    again = load(args.replications) if args.replications else None
    cells = [c for c in CELLS.values() if any(k[0] == c.name for k in recs)]
    rows, context = [], []
    for cell in cells:
        for alpha in cell.alphas:
            if cell.reference:
                for a, r in PAIRS:
                    row = paired_row(cell, alpha, a, r, recs, empty)
                    if again is not None and any(k[0] == cell.name for k in again[0]):
                        rep = paired_row(cell, alpha, a, r, again[0], again[1])
                        row["replication"] = {
                            k: rep[k]
                            for k in (
                                "seeds",
                                "paired",
                                "andrey_shd_mean",
                                "comparator_shd_mean",
                                "mean_D",
                                "bootstrap_95",
                                "above_zero",
                                "verdict",
                                "coverage",
                                "floor",
                            )
                        }
                    rows.append(row)
            context += context_rows(cell, alpha, recs, empty)
    failed = [
        {
            "cell": d["cell"],
            "seed": d["seed"],
            "checks": d["truth_checks"],
            "pcalg": d["pcalg_check"],
        }
        for d in datasets
        if not d["truth_ok"]
    ]
    summary = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "lane": "fci_jci",
        "parity_rule": "paired bootstrap",
        "bootstrap": {
            "resamples": BOOTSTRAP,
            "interval": "percentile 95%",
            "rng": "default_rng(0)",
        },
        "min_pairs": MIN_PAIRS,
        "datasets": len(datasets),
        "truth_checked_by_pcalg": sum(d["pcalg_check"] is not None for d in datasets),
        "truth_failures": failed,
        "pdsep_max": max((r.get("pdsep_max") or 0 for r in recs.values()), default=None),
        "pdsep_max_true": max((d["pdsep_max_true"] for d in datasets), default=None),
        "status": _statuses(recs.values()),
        "rows": rows,
        "context": context,
    }
    Path(args.out).write_text(json.dumps(summary, indent=1, default=_json_default) + "\n")
    for row in rows:
        lo_hi = row["bootstrap_95"]
        print(
            f"{row['regime']:28s} {row['claim']:3s} n={row['paired']:2d} "
            f"A={row['andrey_shd_mean']} R={row['comparator_shd_mean']} "
            f"D={row['mean_D']} {lo_hi} {row['verdict']} same={row['same_pag']}"
        )
    print(f"truth failures: {len(failed)}; status {summary['status']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="fit every arm of the cells on the seeds")
    r.add_argument("--out", required=True)
    r.add_argument("--cells", nargs="+", choices=list(CELLS), required=True)
    r.add_argument("--seeds", type=_seeds, default=_seeds("0-19"), help="e.g. 0-19 or 0,5,7-9")
    r.add_argument("--arms", nargs="+", choices=list(ARMS), help="only these arms")
    r.add_argument("--alphas", nargs="+", type=float, help="only these levels")
    r.add_argument("--cap", type=float, default=CAP_WALL_S, help="wall cap per fit, seconds")
    r.add_argument("--cap-mem-mb", type=float, default=CAP_MEM_MB)
    r.add_argument("--check-pcalg", action="store_true", help="check each truth against pcalg's")
    r.add_argument("--resume", action="store_true")
    r.add_argument("--machine-id", default=None)
    r.set_defaults(func=run)
    s = sub.add_parser("summarize", help="pair the records into rows")
    s.add_argument("--runs", nargs="+", required=True)
    s.add_argument("--replications", nargs="+")
    s.add_argument("--out", required=True)
    s.set_defaults(func=summarize)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
