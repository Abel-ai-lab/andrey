#!/usr/bin/env python
"""Benchmark path and confounder claims in background knowledge: exactness and no-worse lanes.

``Knowledge`` takes three claims beyond edges and tiers: ``required_paths`` (``a`` is an ancestor
of ``b``), ``forbidden_paths`` (``a`` is not), and ``confounded`` (a hidden common cause of ``a``
and ``b``). The lanes check them against exact answers, against the same fit without the claims,
and against reference packages given the same knowledge.

``oracle`` runs the exactness lanes, with d-separation in place of the tests:

- ``cpdag``: every DAG on 4 and 5 nodes and random DAGs on 6 to 8. PC with d-separation returns
  the true CPDAG, and the step that ``pc``, ``ges``, and ``boss`` share (``restrict_paths``)
  restricts it by claims drawn from the truth, true and false. The MPDAG must direct an edge
  exactly when every DAG of the class that meets the claims directs it alike, the MPDAG with its
  clauses must keep exactly those DAGs, and when no DAG meets every claim, some must be dropped.
  A claim set some claim of which is dropped is counted, not checked further: ``tests/unit``
  checks which claims are kept against the enumeration.
- ``pag``: every DAG on 4 observed nodes with one hidden parent of two or more of them, and random
  DAGs on 5 to 8 observed and 1 to 3 hidden. ``fci``, and ``gfci`` and ``boss_fci`` with the score
  phase at its large-sample limit, take true claims; every tail and arrowhead they set must be the
  true MAG's, the adjacencies the MAG's, and every confounded pair adjacent. Where the class is
  small enough to list (at most :data:`MAX_CIRCLES` circles), the MAGs of the true PAG that meet
  the claims are enumerated and each mark they share is compared with the fit's: completeness,
  reported and not judged.

``run`` runs the no-worse lanes on simulated data in each method's setting, with claims drawn from
the truth at several fractions. Each dataset is fitted without claims (``bare``), with true claims
at each fraction (``true``), and with false claims (``false``, context only). In ``pag``,
``lpcmci``, and ``lingam``, ``errors`` counts the pairs adjacent in one graph and not the other
plus every tail or arrowhead, on a pair adjacent in both, that differs from the true MAG over the
observed variables (``pag``), the true window MAG (``lpcmci``), or the true DAG (``lingam``); a
circle claims nothing and is never wrong. In ``cpdag`` a fit is scored against the class the truth
and its claims define (:func:`class_errors`): ``errors`` is the structural Hamming distance to the
true CPDAG for a fit without true claims and, for a fit with them, to the MPDAG of the DAGs of the
true class that meet them, so an edge the claims direct as the true DAG does is no error and an
edge left undirected where they imply a direction is one; ``dag_errors`` is the endpoint-aware
distance to the true DAG, the same measure for every arm, in which an edge directed the right way
costs 0, an undirected one 1, and a reversed, missing, or extra one 2.
``empty_errors`` is the empty graph's count, the floor. A fit drops the claims its result cannot
meet and records how many as ``unmet``; a fit that raises ``KnowledgeError`` (knowledge no graph
meets) is recorded as ``knowledge_error``, never as a graph.
A seed whose graph has too few nodes with two children to hide the cell's hidden nodes is recorded
once as ``no_dataset`` and left out.

- ``cpdag``: ``pc``, ``ges``, ``boss``; linear-Gaussian, standardized; ``d`` 10, 20, 50; ``n`` 500,
  2000; expected degree 2 and 4; claims on 5, 10, 30 % of ordered pairs, ancestor and non-ancestor
  facts in equal parts.
- ``pag``: ``fci``, ``gfci``, ``boss_fci``; linear-Gaussian with hidden nodes; 10 and 20 observed
  with 2 and 3 hidden; ``n`` 1000, 5000; claims on 10 and 30 % of ordered observed pairs and of
  the confounded pairs.
- ``lpcmci``: the simulation of Gerhardus and Runge (2020), ``tau_max = 3``; 5 and 10 series;
  ``T`` 500, 1000; lagged claims on 10 and 30 % of the window's ordered pairs and confounded links.
- ``lingam``: ``direct_lingam``, and lingam's DirectLiNGAM with the same claims as its
  ``prior_knowledge`` (context); linear with uniform noise, standardized; ``d`` 10, 20, 50; ``n``
  500, 2000; claims on 10 and 30 % of ordered pairs.

The other methods that take the path parts run beside these (:data:`EXTRA_METHODS`, :data:`KIND`):
on the ``cpdag`` cells with ``--methods`` (``hc``, ``tabu``, ``mmhc``, ``rsmax2``, ``grasp``,
``xges``, ``exact_search``, ``iamb``, ``inter_iamb``, ``mmpc`` returning its CPDAG, and the plain
BIC variants ``hc_bic`` and ``tabu_bic``) and in ``pag`` (``grasp_fci``), and in lanes of their own
where the data differ. ``cdnod`` draws the domain-shift data of CD-NOD over four domains and scores
a fit against the CPDAG the domain index identifies, which the claims restrict as they restrict a
CPDAG, and against the true DAG. ``jci`` draws the simulation of Mooij et al. (2020) with one
context variable and hidden confounders, the columns permuted, and scores ``fci_jci`` over the
system variables against their true MAG, as ``pag`` does. ``multigroup`` draws three groups that
share one DAG and scores ``multi_group_direct_lingam`` by the sum of its groups' errors. The default
methods of the first lanes are unchanged, and ``--fractions`` keeps only some claim fractions.

``run --methods`` also fits the reference packages (:data:`REFERENCES`), each unmodified in its own
environment, without knowledge and with the same true knowledge. tigramite's LPCMCI takes the
claims as ``link_assumptions`` and lingam's DirectLiNGAM as ``prior_knowledge``. pcalg and Tetrad
take edges and tiers, and no path claim between variables that are not adjacent, so each set of
true claims is translated to the edges it implies (:func:`edge_claims`): a forbidden path
``a -> b`` forbids the edge ``a -> b``, a required path on a pair adjacent in the true graph is a
required edge, and a required path on any other pair, or a confounded pair, has no edge form and is
counted. Tetrad's PC, FGES, BOSS, FCI, GFCI, and BOSS-FCI take the edges as ``Knowledge``; pcalg's
``pc`` keeps the required pairs adjacent with ``fixedEdges`` and directs its CPDAG with
``addBgKnowledge``; pcalg's ``ges`` has no knowledge argument and is not run. Each method of the
``cpdag`` and ``pag`` lanes takes the same edges as ``required`` and ``forbidden`` in its ``edges``
arm, the like-for-like comparison. The references need ``ANDREY_BENCH_JAVA``,
``ANDREY_BENCH_TETRAD_GIN_JAR`` (tetrad-lib 7.6.10), ``ANDREY_BENCH_TETRAD_JAR`` (causal-cmd, for
GFCI), ``Rscript`` with pcalg on ``PATH``, and ``ANDREY_BENCH_TIGRAMITE_PYTHON``. bnlearn's searches take the edges as ``whitelist`` and
``blacklist`` (``andrey_bench/knowledge_bnlearn.R``), causal-learn's CD-NOD as
``background_knowledge``, Tetrad's GRaSP-FCI as ``Knowledge``, and lingam's MultiGroupDirectLiNGAM
the path claims as ``prior_knowledge``.

``summarize`` also judges each method against each of its references on the same datasets
(``reference_rows``, :func:`against_references`): the method's fit minus the reference's under the
same knowledge, in ``dag_errors`` in ``cpdag`` and in ``errors`` elsewhere, with the same interval
and the same rule for a worse cell; a reference's own change with knowledge is a row of ``rows``.

``summarize`` pairs each ``true`` fit with the ``bare`` fit of its dataset: the mean difference in
``errors``, and in ``cpdag`` in ``dag_errors`` too, with its 95 % interval over :data:`RESAMPLES`
bootstrap resamples. A cell is ``worse`` when the interval lies above 0; a worse cell counts only
when the same cell on fresh seeds (``--replicate``) is worse too. A cell outside a method's stated
setting (:data:`OUT_OF_SETTING`) is reported with ``in_setting`` false and left out of its verdict.

    $ANDREY_BENCH_PYTHON benchmarks/knowledge_paths.py oracle --lane cpdag --out outputs/kp-oracle
    $ANDREY_BENCH_PYTHON benchmarks/knowledge_paths.py run --lane cpdag --cell d20-n500-deg2 \\
        --seeds 0 20 --out outputs/kp-cpdag-d20-n500-deg2
    $ANDREY_BENCH_PYTHON benchmarks/knowledge_paths.py summarize --runs outputs/kp-* \\
        --replicate outputs/kp-rep-* --out outputs/kp-summary.json

References
----------
Fang, Z., Zhao, R., Liu, Y., He, Y. (2025). On the representation of pairwise causal background
knowledge and its applications in causal inference. JMLR 26(229):1-73.
Gerhardus, A., Runge, J. (2020). High-recall causal discovery for autocorrelated time series with
latent confounders. NeurIPS 33.
Richardson, T., Spirtes, P. (2002). Ancestral graph Markov models. Annals of Statistics 30(4).
Shimizu, S. et al. (2011). DirectLiNGAM: a direct method for learning a linear non-Gaussian
structural equation model. JMLR 12:1225-1248.
Zhang, J. (2008). On the completeness of orientation rules for causal discovery in the presence of
latent confounders and selection bias. Artificial Intelligence 172(16-17):1873-1896.
"""

from __future__ import annotations

import argparse
import inspect
import itertools
import json
import multiprocessing
import os
import signal
import subprocess
import tempfile
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

RECORDS, DONE, ORACLE = "records.jsonl", "done.json", "oracle.json"
#: Bootstrap resamples of a paired difference.
RESAMPLES = 10_000
#: The largest number of circle endpoints whose MAG class the PAG oracle lists.
MAX_CIRCLES = 12
#: Seconds one fit may take before it is recorded as a timeout.
FIT_CAP_S = 1800

NULL, TAIL, ARROW, CIRCLE = 0, 1, 2, 3

LANES = {
    "cpdag": {
        "methods": ("pc", "ges", "boss"),
        "fractions": (0.05, 0.1, 0.3),
        "cells": {
            f"d{d}-n{n}-deg{g}": {"d": d, "n": n, "degree": g}
            for d in (10, 20, 50)
            for n in (500, 2000)
            for g in (2, 4)
        },
    },
    "pag": {
        "methods": ("fci", "gfci", "boss_fci"),
        "fractions": (0.1, 0.3),
        "cells": {
            f"obs{o}-n{n}": {"d": o + h, "latents": h, "n": n, "degree": 2}
            for o, h in ((10, 2), (20, 3))
            for n in (1000, 5000)
        },
    },
    "lpcmci": {
        "methods": ("lpcmci",),
        "fractions": (0.1, 0.3),
        "cells": {
            f"d{d}-T{t}": {"d": d, "n": t, "tau_max": 3} for d in (5, 10) for t in (500, 1000)
        },
    },
    "lingam": {
        "methods": ("direct_lingam", "lingam.direct_lingam"),
        "fractions": (0.1, 0.3),
        "cells": {
            f"d{d}-n{n}": {"d": d, "n": n, "degree": 2} for d in (10, 20, 50) for n in (500, 2000)
        },
    },
    # The lanes of the methods that need data of their own kind (see KIND).
    "cdnod": {
        "methods": ("cdnod",),
        "fractions": (0.05, 0.1, 0.3),
        "cells": {
            f"d{d}-n{n}-deg{g}": {"d": d, "n": n, "degree": g}
            for d in (10, 20, 50)
            for n in (500, 2000)
            for g in (2, 4)
        },
    },
    "jci": {
        "methods": ("fci_jci",),
        "fractions": (0.1, 0.3),
        "cells": {
            f"p{p}-n{n}": {"p": p, "eps": 2.25 / (p - 1), "n_per": n}
            for p in (10, 20)
            for n in (500, 2000)
        },
    },
    "multigroup": {
        "methods": ("multi_group_direct_lingam",),
        "fractions": (0.1, 0.3),
        "cells": {
            f"d{d}-n{n}": {"d": d, "n": n, "degree": 2} for d in (10, 20, 50) for n in (500, 2000)
        },
    },
}
#: What each lane's fits and scores are: ``cpdag`` (a CPDAG scored against the class the truth and
#: the claims define and against the true DAG), ``pag`` and ``dag`` (marks against the true MAG or
#: DAG), ``window`` (lagged marks). The new lanes are ``cdnod`` (the domain-shift data of CD-NOD,
#: its class the CPDAG the domain index identifies), ``jci`` (the simulation of Mooij et al. 2020
#: with hidden confounders and one context variable, over the system variables, columns permuted),
#: and ``multigroup`` (three groups sharing one DAG; a fit's marks are the groups' graphs side by
#: side, so its errors are the sum over the groups).
KIND = {
    "cpdag": "cpdag",
    "pag": "pag",
    "lpcmci": "window",
    "lingam": "dag",
    "cdnod": "cpdag",
    "jci": "pag",
    "multigroup": "dag",
}
#: Andrey methods a lane runs with ``--methods`` beside its default ones, on the lane's own data.
#: The defaults of a published lane stay as they are, so its record is reproducible.
EXTRA_METHODS = {
    "cpdag": (
        "hc",
        "tabu",
        "mmhc",
        "rsmax2",
        "grasp",
        "xges",
        "exact_search",
        "iamb",
        "inter_iamb",
        "mmpc",
    ),
    "pag": ("grasp_fci",),
}
#: The reference packages each method is judged against under the same knowledge, each run
#: unmodified in its own environment. pcalg's ``ges`` has no knowledge argument, so GES has one
#: reference.
REFERENCES = {
    "pc": ("pcalg.pc", "tetrad.pc"),
    "ges": ("tetrad.ges",),
    "ges_bic": ("tetrad.ges",),
    "boss": ("tetrad.boss",),
    "fci": ("tetrad.fci",),
    "gfci": ("tetrad.gfci",),
    "boss_fci": ("tetrad.boss_fci",),
    "lpcmci": ("tigramite.lpcmci",),
    "direct_lingam": ("lingam.direct_lingam",),
    "hc": ("bnlearn.hc",),
    "hc_bic": ("bnlearn.hc",),
    "tabu": ("bnlearn.tabu",),
    "tabu_bic": ("bnlearn.tabu",),
    "mmhc": ("bnlearn.mmhc",),
    "rsmax2": ("bnlearn.rsmax2",),
    "iamb": ("bnlearn.iamb",),
    "inter_iamb": ("bnlearn.inter_iamb",),
    "grasp_fci": ("tetrad.grasp_fci",),
    "cdnod": ("causal_learn.cdnod",),
    "multi_group_direct_lingam": ("lingam.multi_group_direct_lingam",),
}
#: The references that take a path claim as it is; the others take the edges it implies
#: (:func:`edge_claims`).
TAKES_PATHS = ("tigramite.lpcmci", "lingam.direct_lingam", "lingam.multi_group_direct_lingam")
#: The references a lane runs with ``--methods``, beside the methods it runs by default.
LANE_REFERENCES = {
    "cpdag": ("pcalg.pc", "tetrad.pc", "tetrad.ges", "tetrad.boss"),
    "pag": ("tetrad.fci", "tetrad.gfci", "tetrad.boss_fci"),
    "lpcmci": ("tigramite.lpcmci",),
    "lingam": (),
    "cdnod": ("causal_learn.cdnod",),
    "jci": (),
    "multigroup": ("lingam.multi_group_direct_lingam",),
}
LANE_REFERENCES["cpdag"] += (
    "bnlearn.hc",
    "bnlearn.tabu",
    "bnlearn.mmhc",
    "bnlearn.rsmax2",
    "bnlearn.iamb",
    "bnlearn.inter_iamb",
)
LANE_REFERENCES["pag"] += ("tetrad.grasp_fci",)
#: Andrey methods at settings other than their defaults, run with ``--methods``. ``ges_bic`` is
#: ``ges`` with the plain BIC (``ebic_gamma=0``), the score of the FGES reference arm; ``hc_bic``
#: and ``tabu_bic`` are the same score for the searches whose reference (bnlearn's ``bic-g``)
#: scores with it. ``mmpc`` returns its skeleton by default, which takes no path claim; its variant
#: returns the CPDAG, which does.
VARIANTS = {
    "ges_bic": ("ges", {"ebic_gamma": 0.0}),
    "hc_bic": ("hc", {"ebic_gamma": 0.0}),
    "tabu_bic": ("tabu", {"ebic_gamma": 0.0}),
    "mmpc": ("mmpc", {"structure": "cpdag"}),
}


def own_methods(lane: str) -> tuple[str, ...]:
    """Andrey's methods in ``lane``: its defaults, its extras, and their variants."""
    base = (*LANES[lane]["methods"], *EXTRA_METHODS.get(lane, ()))
    return (*base, *(v for v, (b, _) in VARIANTS.items() if b in base and v not in base))


ARMS = ("bare", "true", "edges", "false")
#: The interpreter of tigramite's environment (``benchmarks/start_tigramite.sh``).
TIGRAMITE_PYTHON = "ANDREY_BENCH_TIGRAMITE_PYTHON"
#: The share of ordered pairs the false-claims arm states wrongly.
FALSE_FRACTION = 0.1
#: The cells outside a method's stated setting, by lane and method. GES suits sparse graphs: its
#: release checks run at expected degree 2, and at degree 4 its fit with the default settings does
#: not beat the empty graph with or without claims, so those cells say nothing about the claims.
OUT_OF_SETTING = {
    ("cpdag", "ges"): lambda cell: cell["degree"] >= 4,
    ("cpdag", "ges_bic"): lambda cell: cell["degree"] >= 4,
    # The score searches that suit the sparse graphs GES suits (their docstrings say so).
    **{
        ("cpdag", m): lambda cell: cell["degree"] >= 4
        for m in ("hc", "hc_bic", "tabu", "tabu_bic", "xges")
    },
}


def _log(msg: str) -> None:
    print(msg, flush=True)


# ------------------------------------------------------------------------------------------------
# truth, claims, and errors
# ------------------------------------------------------------------------------------------------


def dag_reach(marks: np.ndarray) -> np.ndarray:
    """``reach[a, b]``: a directed path leads from ``a`` to ``b`` in the DAG ``marks``."""
    reach = (marks == TAIL) & (marks.T == ARROW)
    for k in range(len(marks)):
        reach |= reach[:, [k]] & reach[[k], :]
    return reach


def hidden_cause_pairs(marks: np.ndarray, hidden: np.ndarray) -> set[tuple[int, int]]:
    """The observed pairs ``a < b`` that one hidden node reaches through hidden nodes only."""
    child = (marks == TAIL) & (marks.T == ARROW)
    pairs: set[tuple[int, int]] = set()
    for source in np.flatnonzero(hidden):
        seen, stack, reached = {int(source)}, [int(source)], set()
        while stack:
            for c in map(int, np.flatnonzero(child[stack.pop()])):
                if not hidden[c]:
                    reached.add(c)
                elif c not in seen:
                    seen.add(c)
                    stack.append(c)
        pairs.update(itertools.combinations(sorted(reached), 2))
    return pairs


def draw_claims(
    rng: np.random.Generator,
    facts: dict[tuple, bool],
    fraction: float,
    confounded: list[tuple],
    *,
    true: bool = True,
) -> dict[str, list]:
    """Path claims on ``fraction`` of the pairs ``facts`` maps to whether the first is an ancestor
    of the second, ancestor and non-ancestor facts in equal parts, and ``fraction`` of the
    ``confounded`` pairs (at least one when there are any). ``true=False`` states each path fact
    wrongly and leaves out the confounded pairs."""
    pairs = list(facts)
    m = max(1, round(fraction * len(pairs)))
    if not true:
        chosen = [pairs[i] for i in rng.choice(len(pairs), min(m, len(pairs)), replace=False)]
        return {
            "required_paths": [list(p) for p in chosen if not facts[p]],
            "forbidden_paths": [list(p) for p in chosen if facts[p]],
            "confounded": [],
        }
    out: dict[str, list] = {"confounded": []}
    for part, value in (("required_paths", True), ("forbidden_paths", False)):
        side = [p for p in pairs if facts[p] == value]
        k = min(len(side), (m + 1) // 2 if value else m // 2)
        out[part] = [list(side[i]) for i in sorted(rng.choice(len(side), k, replace=False))]
    if confounded:
        k = max(1, round(fraction * len(confounded)))
        out["confounded"] = [
            list(confounded[i]) for i in sorted(rng.choice(len(confounded), k, replace=False))
        ]
    return out


def knowledge_of(claims: dict[str, list]):
    from andrey import Knowledge

    return Knowledge(**{part: [tuple(c) for c in pairs] for part, pairs in claims.items() if pairs})


def edge_claims(claims: dict[str, list], truth: np.ndarray) -> dict:
    """The edge knowledge the true ``claims`` imply on the true graph ``truth``.

    A forbidden path ``a -> b`` forbids the edge ``a -> b``. A required path on a pair adjacent in
    ``truth`` is the required edge ``a -> b``; on any other pair it has no edge form, and neither
    has a confounded pair. ``no_edge_form`` and ``confounded_no_edge_form`` count those.
    """
    adjacent = truth != NULL
    required = [list(p) for p in claims.get("required_paths", []) if adjacent[p[0], p[1]]]
    return {
        "required": required,
        "forbidden": [list(p) for p in claims.get("forbidden_paths", [])],
        "no_edge_form": len(claims.get("required_paths", [])) - len(required),
        "confounded_no_edge_form": len(claims.get("confounded", [])),
    }


def link_assumptions(claims: dict[str, list], d: int, tau_max: int) -> list[list]:
    """tigramite's ``link_assumptions`` for lagged ``claims``, as ``[j, i, lag, link]`` rows.

    Every link of the window is listed, since LPCMCI reads a missing one as absent: ``o?>`` at a
    lag and ``o?o`` at lag 0 claim nothing. A required path ``[a, b, lag]`` puts a tail at ``a``
    and an arrowhead at ``b``, a forbidden path an arrowhead at ``a``, and a confounded link drops
    the ``?``, which states that the link is there.
    """
    left = {(a, b, k): "o" for a in range(d) for b in range(d) for k in range(tau_max + 1)}
    sure = set()
    for a, b, k in claims.get("required_paths", []):
        left[a, b, k] = "-"
        if k == 0:
            left[b, a, 0] = "<"
    for a, b, k in claims.get("forbidden_paths", []):
        left[a, b, k] = "<"
    for a, b, k in claims.get("confounded", []):
        sure |= {(a, b, k), (b, a, k)} if k == 0 else {(a, b, k)}
    flip = {"<": ">", "-": "-", "o": "o"}
    rows = []
    for (a, b, k), mark in left.items():
        if a == b and k == 0:
            continue
        right = flip[left[b, a, 0]] if k == 0 else ">"
        rows.append([b, a, -k, mark + ("-" if (a, b, k) in sure else "?") + right])
    return rows


def static_errors(est: np.ndarray, truth: np.ndarray, *, undirected_open: bool) -> dict[str, int]:
    """Adjacency errors and wrong tails or arrowheads of ``est`` against ``truth``, both endpoint-
    mark matrices over the same nodes. With ``undirected_open``, a tail-tail edge claims nothing."""
    adj_e, adj_t = est != NULL, truth != NULL
    upper = np.triu(np.ones_like(adj_e, dtype=bool), 1)
    adjacency = int((upper & (adj_e != adj_t)).sum())
    definite = (est == TAIL) | (est == ARROW)
    if undirected_open:
        definite &= ~((est == TAIL) & (est.T == TAIL))
    both = adj_e & adj_t
    counted = definite & both
    wrong = int((counted & (est != truth)).sum())
    return {
        "adjacency_errors": adjacency,
        "wrong_marks": wrong,
        "errors": adjacency + wrong,
        "definite_marks": int(counted.sum()),
        "true_adjacencies": int((upper & adj_t).sum()),
        "empty_errors": int((upper & adj_t).sum()),
    }


def class_errors(
    est: np.ndarray, dag: np.ndarray, claims: dict | None, base: np.ndarray | None = None
) -> dict[str, int]:
    """The counts of a CPDAG fit ``est`` against the true ``dag`` and the true ``claims`` it took.

    ``errors`` is the structural Hamming distance (:func:`andrey.metrics.shd`) to the class the
    truth and the claims define: the true CPDAG without claims and, with them, the MPDAG of the
    DAGs of the true class that meet them (:func:`andrey.core.ancestral.restrict` on the true
    CPDAG, the step the ``cpdag`` oracle checks against the listed class). A pair costs 1 when its
    edge is missing, extra, reversed, or directed where the target leaves it undirected or the
    reverse. ``dag_errors`` is the endpoint-aware distance to the true DAG, which does not depend
    on the claims: an edge directed as the DAG directs it costs 0, an undirected one 1, and a
    reversed, missing, or extra one 2. ``empty_errors`` and ``empty_dag_errors`` are the empty
    graph's counts. The other counts are those of :func:`static_errors` against the true DAG.
    ``base`` is the class the truth defines when it is not the true DAG's CPDAG: the CPDAG the
    domain index identifies in the ``cdnod`` lane, which the claims restrict as they do the CPDAG.
    """
    from andrey import GraphStructure, metrics
    from andrey.core.ancestral import restrict
    from andrey.core.orient import dag2cpdag

    d = len(dag)
    masks = {part: np.zeros((d, d), dtype=bool) for part in ("required_paths", "forbidden_paths")}
    for part, mask in masks.items():
        for a, b in (claims or {}).get(part, []):
            mask[a, b] = True
    base = dag2cpdag(dag) if base is None else base
    target, _, dropped = restrict(base, masks["required_paths"], masks["forbidden_paths"])
    assert not dropped, "a claim drawn from the truth holds in a DAG of the true class"

    def graph(marks: np.ndarray, kind: str) -> GraphStructure:
        return GraphStructure.from_numpy(marks, kind=kind)

    counts = static_errors(est, dag, undirected_open=True)
    return {
        **counts,
        "errors": metrics.shd(graph(est, "cpdag"), graph(target, "cpdag")),
        "dag_errors": metrics.shd(graph(est, "cpdag"), graph(dag, "dag"), endpoint_aware=True),
        "empty_dag_errors": 2 * counts["true_adjacencies"],
    }


def window_errors(est: np.ndarray, truth: np.ndarray) -> dict[str, int]:
    """:func:`static_errors` on ``(tau_max + 1, n, n)`` window marks: lag 0 as a graph, and at lag
    ``k >= 1`` each link ``(X_i(t - k), X_j(t))`` with the mark at its earlier end."""
    out = static_errors(est[0], truth[0], undirected_open=False)
    for k in range(1, len(truth)):
        e, t = est[k], truth[k]
        adj_e, adj_t = e != NULL, t != NULL
        counted = ((e == TAIL) | (e == ARROW)) & adj_e & adj_t
        wrong = int((counted & (e != t)).sum())
        adjacency = int((adj_e != adj_t).sum())
        out["adjacency_errors"] += adjacency
        out["wrong_marks"] += wrong
        out["errors"] += adjacency + wrong
        out["definite_marks"] += int(counted.sum())
        out["true_adjacencies"] += int(adj_t.sum())
        out["empty_errors"] += int(adj_t.sum())
    return out


# ------------------------------------------------------------------------------------------------
# datasets
# ------------------------------------------------------------------------------------------------


def dataset_key(lane: str, cell: dict, seed: int):
    from andrey_bench import datasets
    from andrey_bench.datasets import DatasetKey

    if lane == "lpcmci":
        slice_ = dict(
            zip(
                ("topology", "functional", "noise", "standardize"), datasets.REGIMES["lpcmci_paper"]
            )
        )
        return DatasetKey(
            **slice_,
            density=datasets.TEMPORAL_REGIMES["lpcmci_paper"],
            d=cell["d"],
            n=cell["n"],
            seed=seed,
            latents=datasets.latents_for("lpcmci_paper", cell["d"]),
        )
    noise = "uniform" if lane == "lingam" else "gaussian"
    return DatasetKey(
        topology="er",
        functional="linear",
        noise=noise,
        standardize="standardized",
        density=float(cell["degree"]),
        d=cell["d"],
        n=cell["n"],
        seed=seed,
        latents=cell.get("latents", 0),
    )


def window_truth(truth, tau_max: int):
    """The window nodes, the window MAG's marks, each window node's window ancestors, and the
    unrolled DAG, with as much past as :func:`andrey_bench.window_pag.true_window_pag` takes."""
    from andrey_bench.window_pag import _MAX_ROUNDS, _window_mag, unrolled_dag

    step = tau_max + truth.tau_max
    found = []
    for round_ in range(1, _MAX_ROUNDS + 1):
        nodes, mag, anc = _window_mag(truth, tau_max, round_ * step)
        found.append(mag)
        if len(found) >= 3 and all(np.array_equal(mag, f) for f in found[-3:-1]):
            return nodes, mag, anc, unrolled_dag(truth, tau_max + 1 + round_ * step)
    raise RuntimeError("the window MAG did not settle")


def _claims_for(out: dict, lane: str, seed: int, facts: dict, confounded: list) -> dict:
    """``out`` with the true claims at every fraction of ``lane`` and the false claims."""
    out["claims"] = {
        f: draw_claims(np.random.default_rng([seed, 9124, round(f * 100)]), facts, f, confounded)
        for f in LANES[lane]["fractions"]
    }
    rng = np.random.default_rng([seed, 9124])
    out["false_claims"] = draw_claims(rng, facts, FALSE_FRACTION, confounded, true=False)
    return out


def _regime_key(name: str, cell: dict, seed: int):
    """The key of a regime the harness defines (``datasets.REGIMES``) at the cell's size."""
    from andrey_bench import datasets
    from andrey_bench.datasets import DatasetKey

    slice_ = dict(zip(("topology", "functional", "noise", "standardize"), datasets.REGIMES[name]))
    return DatasetKey(**slice_, density=float(cell["degree"]), d=cell["d"], n=cell["n"], seed=seed)


def problem_cdnod(cell: dict, seed: int) -> dict:
    """CD-NOD's domain-shift data: ``truth`` is the true DAG, ``base`` the CPDAG the domain index
    identifies (the harness's truth for this data), and ``env`` the domain of each row."""
    from andrey_bench import datasets, regime_models

    key = _regime_key("domain_shift_er", cell, seed)
    data, base, _, _ = datasets.generate_one(key)
    dag = regime_models.domain_shift_dag(
        d=key.d, n=key.n, seed=seed, density=key.density, noise=key.noise, shuffle=key.shuffle
    )
    env = regime_models.domain_index(len(data), regime_models.N_DOMAINS)[:, 0]
    reach = dag_reach(dag)
    facts = {(a, b): bool(reach[a, b]) for a in range(key.d) for b in range(key.d) if a != b}
    out = {"key": key, "data": data, "dataset_id": key.digest, "truth": dag, "base": base}
    return _claims_for({**out, "env": env}, "cdnod", seed, facts, [])


def problem_jci(cell: dict, seed: int) -> dict:
    """The simulation of Mooij, Magliacane, and Claassen (2020) with one context variable: the
    system variables in a column order drawn from the seed, ``env`` the context's regime, and
    ``truth`` the MAG over the system variables, the hidden confounders marginalized."""
    from andrey.core import LATENT, OBSERVED, GraphStructure
    from andrey.data.latent import marginal
    from andrey_bench import regime_models

    p = cell["p"]
    draw = regime_models.jci_diagonal(p, 1, cell["eps"], cell["eps"], cell["n_per"], seed)
    h = draw.hidden
    # System variables, then the confounders; the context variable (index p) points into one
    # system variable and is dropped, which leaves the system variables' MAG as it is.
    order = np.random.default_rng([seed, 4177]).permutation(p)
    keep = [*order, *range(p + 1, p + 1 + h)]
    adjacency = draw.adjacency[np.ix_(keep, keep)]
    marks = np.zeros(adjacency.shape, dtype=np.int8)
    marks[adjacency != 0], marks.T[adjacency != 0] = TAIL, ARROW
    hidden = np.arange(p + h) >= p
    types = np.where(hidden, LATENT, OBSERVED).astype(np.int8)
    truth = marginal(GraphStructure.from_numpy(marks, kind="dag", node_types=types), "mag")
    reach = dag_reach(marks)
    facts = {(a, b): bool(reach[a, b]) for a in range(p) for b in range(p) if a != b}
    out = {
        "data": np.ascontiguousarray(draw.data[:, order]),
        "dataset_id": f"jci-p{p}-n{cell['n_per']}-s{seed}",
        "truth": truth.to_numpy(),
        "env": draw.regime.astype(float),
    }
    return _claims_for(out, "jci", seed, facts, sorted(hidden_cause_pairs(marks, hidden)))


def problem_multigroup(cell: dict, seed: int) -> dict:
    """Three groups that share one DAG: ``truth`` is the groups' copies side by side, and the claims
    name the variables the groups share."""
    from andrey_bench import datasets

    key = _regime_key("multi_group_er", cell, seed)
    data, marks, _, _ = datasets.generate_one(key)
    d = key.d
    reach = dag_reach(marks[:d, :d])
    facts = {(a, b): bool(reach[a, b]) for a in range(d) for b in range(d) if a != b}
    out = {"key": key, "data": data, "dataset_id": key.digest, "truth": marks}
    return _claims_for(out, "multigroup", seed, facts, [])


def problem(lane: str, cell: dict, seed: int) -> dict:
    """The data, the truth the errors are counted against, and the claims of every arm."""
    from andrey_bench import datasets

    if lane in ("cdnod", "jci", "multigroup"):
        make = {"cdnod": problem_cdnod, "jci": problem_jci, "multigroup": problem_multigroup}
        return make[lane](cell, seed)
    key = dataset_key(lane, cell, seed)
    data, marks, kind, _ = datasets.generate_one(key)
    out = {"key": key, "data": data, "dataset_id": key.digest}
    if lane == "lpcmci":
        from andrey_bench.window import WindowGraph
        from andrey_bench.window_pag import _window_marks

        tau_max = cell["tau_max"]
        truth = WindowGraph(marks, lag0="dag", latents=key.latents)
        nodes, mag, anc, dag = window_truth(truth, tau_max)
        n_obs = truth.n_nodes - truth.latents
        out["truth"] = _window_marks(nodes, mag, n_obs, tau_max)
        facts = {
            (i, j, k): (i, k) in anc[(j, 0)]
            for k in range(tau_max + 1)
            for i in range(n_obs)
            for j in range(n_obs)
            if i != j or k
        }
        confounded = set()
        for source in (v for v in dag.nodes if v[0] >= n_obs):
            seen, stack, reached = {source}, [source], set()
            while stack:
                for c in dag.successors(stack.pop()):
                    if c[0] < n_obs:
                        reached.add(c)
                    elif c not in seen:
                        seen.add(c)
                        stack.append(c)
            for (i, si), (j, sj) in itertools.combinations(sorted(reached), 2):
                lag = abs(si - sj)
                if lag > tau_max or (i == j and lag == 0):
                    continue
                early, late = ((i, j) if si > sj else (j, i)) if lag else tuple(sorted((i, j)))
                confounded.add((early, late, lag))
        confounded = sorted(confounded)
    else:
        d_obs = cell["d"] - key.latents
        hidden = np.zeros(len(marks), dtype=bool)
        hidden[d_obs:] = True
        reach = dag_reach(marks)
        if key.latents:
            from andrey.core import LATENT, OBSERVED, GraphStructure
            from andrey.data.latent import marginal

            types = np.where(hidden, LATENT, OBSERVED).astype(np.int8)
            dag = GraphStructure.from_numpy(marks, kind="dag", node_types=types)
            out["truth"] = marginal(dag, "mag").to_numpy()
        else:
            out["truth"] = marks
        facts = {(a, b): bool(reach[a, b]) for a in range(d_obs) for b in range(d_obs) if a != b}
        confounded = sorted(hidden_cause_pairs(marks, hidden))
    return _claims_for(out, lane, seed, facts, confounded)


# ------------------------------------------------------------------------------------------------
# fits
# ------------------------------------------------------------------------------------------------


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout


def fit(
    method: str,
    data: np.ndarray,
    claims: dict | None,
    cell: dict,
    seed: int,
    edges: dict | None = None,
    env: np.ndarray | None = None,
) -> dict:
    """One fit: its status, its marks, and the clauses or unmet claims its metadata lists.

    With ``edges`` (:func:`edge_claims`), the fit takes those required and forbidden edges and not
    the path ``claims``. ``env`` is the domain of each row, for the methods that take one.
    """
    import andrey
    from andrey.core.knowledge import KnowledgeError

    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(FIT_CAP_S)
    start = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if method == "lingam.direct_lingam":
                marks = _lingam_reference(data, claims)
                meta = {}
            elif method == "tigramite.lpcmci":
                marks = _tigramite_reference(data, claims, cell["tau_max"])
                meta = {}
            elif method == "lingam.multi_group_direct_lingam":
                marks = _lingam_reference(data, claims, groups=True)
                meta = {}
            elif method == "causal_learn.cdnod":
                marks = _causal_learn_cdnod(data, env, edges)
                meta = {}
            elif method.startswith("bnlearn."):
                marks = _bnlearn_reference(method, data, edges)
                meta = {}
            elif method == "pcalg.pc":
                marks, unmet = _pcalg_reference(data, edges)
                meta = {"unmet_knowledge": [None] * unmet}
            elif method.startswith("tetrad."):
                marks = _tetrad_reference(method, data, edges)
                meta = {}
            else:
                name, settings = VARIANTS.get(method, (method, {}))
                fn = getattr(andrey, name)
                kwargs = dict(settings)
                if claims:
                    kwargs["knowledge"] = knowledge_of(claims)
                if edges is not None:
                    kwargs["knowledge"] = andrey.Knowledge(
                        required=[tuple(e) for e in edges["required"]],
                        forbidden=[tuple(e) for e in edges["forbidden"]],
                    )
                if method == "lpcmci":
                    kwargs["tau_max"] = cell["tau_max"]
                if "seed" in inspect.signature(fn).parameters:
                    kwargs["seed"] = seed
                if method in ("cdnod", "fci_jci"):
                    out = fn(data, env, **kwargs)
                elif method == "multi_group_direct_lingam":
                    from andrey_bench.regime_models import N_GROUPS, split_groups

                    out = fn(split_groups(data, N_GROUPS), **kwargs)
                else:
                    out = fn(data, **kwargs)
                meta = out[0].metadata if isinstance(out, list) else out.metadata
                if isinstance(out, list):
                    marks = _side_by_side([o.structure.to_numpy() for o in out])
                elif method == "lpcmci":
                    from andrey_bench.adapters.andrey_pcmci import lpcmci_link_strings
                    from andrey_bench.window import marks_from_link_strings

                    marks = marks_from_link_strings(lpcmci_link_strings(out))
                else:
                    marks = np.asarray(out.structure.to_numpy(), dtype=np.int8)
        result = {"status": "ok", "marks": marks}
        result["clauses"] = len(meta.get("ancestor_clauses", []))
        result["unmet"] = len(meta.get("unmet_knowledge", []))
    except KnowledgeError as exc:
        result = {"status": "knowledge_error", "message": str(exc)[:300]}
    except _Timeout:
        result = {"status": "timeout"}
    except Exception as exc:  # noqa: BLE001 - a failed fit is a record, not a crash
        result = {"status": "error", "message": f"{type(exc).__name__}: {str(exc)[:300]}"}
    finally:
        signal.alarm(0)
    result["wall_s"] = time.perf_counter() - start
    return result


def _side_by_side(graphs: list[np.ndarray]) -> np.ndarray:
    """The groups' endpoint marks as one block-diagonal matrix."""
    d = len(graphs[0])
    out = np.zeros((d * len(graphs), d * len(graphs)), dtype=np.int8)
    for g, marks in enumerate(graphs):
        out[g * d : (g + 1) * d, g * d : (g + 1) * d] = marks
    return out


def _lingam_reference(data: np.ndarray, claims: dict | None, groups: bool = False) -> np.ndarray:
    """lingam's DirectLiNGAM with the claims as ``prior_knowledge``: entry ``[b, a]`` is 1 when
    ``a`` is an ancestor of ``b``, 0 when it is not, and -1 when unknown. With ``groups``, the
    columns hold the groups of a ``multigroup`` dataset side by side and the reference is
    MultiGroupDirectLiNGAM, whose result is the groups' graphs side by side."""
    import lingam

    if groups:
        from andrey_bench.regime_models import N_GROUPS, split_groups

        parts = split_groups(data, N_GROUPS)
        d = parts[0].shape[1]
    else:
        d = data.shape[1]
    prior = None
    if claims:
        prior = -np.ones((d, d), dtype=int)
        for a, b in claims.get("required_paths", []):
            prior[b, a] = 1
        for a, b in claims.get("forbidden_paths", []):
            prior[b, a] = 0
    if groups:
        model = lingam.MultiGroupDirectLiNGAM(prior_knowledge=prior).fit(parts)
        stack = model.adjacency_matrices_
    else:
        model = lingam.DirectLiNGAM(prior_knowledge=prior).fit(data)
        stack = [model.adjacency_matrix_]
    graphs = []
    for weights in stack:  # weights[b, a] != 0: a -> b
        marks = np.zeros((d, d), dtype=np.int8)
        for b, a in np.argwhere(weights != 0):
            marks[a, b], marks[b, a] = TAIL, ARROW
        graphs.append(marks)
    return _side_by_side(graphs) if groups else graphs[0]


def _causal_learn_cdnod(data: np.ndarray, env: np.ndarray, edges: dict | None) -> np.ndarray:
    """causal-learn's CD-NOD (the harness adapter's settings), with the edges as its
    ``background_knowledge``: the CPDAG over the data's variables."""
    from causallearn.graph.GraphNode import GraphNode
    from causallearn.search.ConstraintBased.CDNOD import cdnod
    from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

    from andrey_bench.adapters.causal_learn_convert import generalgraph_to_adjacency

    knowledge = None
    if edges and (edges["required"] or edges["forbidden"]):
        knowledge = BackgroundKnowledge()
        name = lambda j: GraphNode(f"X{j + 1}")  # noqa: E731 - causal-learn's default node names
        for a, b in edges["required"]:
            knowledge.add_required_by_node(name(a), name(b))
        for a, b in edges["forbidden"]:
            knowledge.add_forbidden_by_node(name(a), name(b))
    X = np.asarray(data, dtype=np.float64)
    cg = cdnod(
        X,
        np.asarray(env, dtype=np.float64).reshape(-1, 1),
        alpha=0.05,
        indep_test="fisherz",
        stable=True,
        uc_rule=0,
        uc_priority=2,
        background_knowledge=knowledge,
        show_progress=False,
    )
    d = X.shape[1]
    return _cpdag_marks(generalgraph_to_adjacency(np.asarray(cg.G.graph)[:d, :d]))


def _bnlearn_reference(method: str, data: np.ndarray, edges: dict | None) -> np.ndarray:
    """bnlearn's search (``andrey_bench/knowledge_bnlearn.R``) in a fresh R process, with the edges
    as ``whitelist`` and ``blacklist``: the CPDAG of its result."""
    script = Path(__file__).parent / "andrey_bench" / "knowledge_bnlearn.R"
    with tempfile.TemporaryDirectory(prefix="andrey_bench_bnlearn_") as tmp:
        work = Path(tmp)
        header = ",".join(f"x{j}" for j in range(data.shape[1]))
        np.savetxt(work / "data.csv", data, delimiter=",", fmt="%.17g", header=header, comments="")
        for part in ("required", "forbidden"):
            rows = (edges or {}).get(part, [])
            (work / f"{part}.csv").write_text("".join(f"{a},{b}\n" for a, b in rows))
        files = [str(work / f) for f in ("data.csv", "required.csv", "forbidden.csv", "out.csv")]
        done = subprocess.run(
            ["Rscript", "--vanilla", str(script), method.split(".", 1)[1], *files, "0.05"],
            capture_output=True,
            text=True,
        )
        if done.returncode != 0:
            raise RuntimeError(
                f"knowledge_bnlearn.R exited {done.returncode}: {done.stderr[-300:]}"
            )
        adjacency = np.loadtxt(work / "out.csv", delimiter=",", ndmin=2)
    return _cpdag_marks(adjacency != 0)


def _cpdag_marks(adjacency: np.ndarray) -> np.ndarray:
    from andrey_bench.contracts import structure_from_adjacency

    graph = structure_from_adjacency(np.asarray(adjacency, dtype=np.int8), kind="cpdag")
    return np.asarray(graph.to_numpy(), dtype=np.int8)


def _tetrad_reference(method: str, data: np.ndarray, edges: dict | None) -> np.ndarray:
    """Tetrad's search with the edges as its ``Knowledge``. Its FCI steps orient by knowledge only
    without the rules for selection bias, so FCI and BOSS-FCI leave them out in every arm."""
    from andrey_bench.adapters import tetrad_gfci, tetrad_search

    adapter = {
        "tetrad.pc": tetrad_search.TetradPC,
        "tetrad.ges": tetrad_search.TetradFGES,
        "tetrad.boss": tetrad_search.TetradBOSS,
        "tetrad.fci": tetrad_search.TetradFCI,
        "tetrad.boss_fci": tetrad_search.TetradBossFCI,
        "tetrad.grasp_fci": tetrad_search.TetradGraspFCI,
        "tetrad.gfci": tetrad_gfci.TetradGFCI,
    }[method]()
    params = dict(adapter.params())
    if method in ("tetrad.fci", "tetrad.boss_fci", "tetrad.grasp_fci"):
        params["excludeSelectionBias"] = True
    if edges:
        params.update(required=edges["required"], forbidden=edges["forbidden"])
    native = adapter.fit(data, params)
    return native if adapter.output_type == "pag" else _cpdag_marks(native)


def _pcalg_reference(data: np.ndarray, edges: dict | None) -> tuple[np.ndarray, int]:
    """pcalg's ``pc`` and ``addBgKnowledge`` (``andrey_bench/knowledge_pcalg.R``) in a fresh R
    process: the CPDAG's marks and the number of edges pcalg could not apply."""
    script = Path(__file__).parent / "andrey_bench" / "knowledge_pcalg.R"
    with tempfile.TemporaryDirectory(prefix="andrey_bench_pcalg_") as tmp:
        work = Path(tmp)
        header = ",".join(f"x{j}" for j in range(data.shape[1]))
        np.savetxt(work / "data.csv", data, delimiter=",", fmt="%.17g", header=header, comments="")
        for part in ("required", "forbidden"):
            rows = (edges or {}).get(part, [])
            (work / f"{part}.csv").write_text("".join(f"{a},{b}\n" for a, b in rows))
        files = [str(work / f) for f in ("data.csv", "required.csv", "forbidden.csv", "out.csv")]
        done = subprocess.run(
            ["Rscript", "--vanilla", str(script), *files, "0.05"], capture_output=True, text=True
        )
        if done.returncode != 0:
            raise RuntimeError(f"knowledge_pcalg.R exited {done.returncode}: {done.stderr[-300:]}")
        adjacency = np.loadtxt(work / "out.csv", delimiter=",")
    return _cpdag_marks(adjacency != 0), int(done.stdout.split()[-1])


_TIGRAMITE_FIT = """
import json, sys
import numpy as np
from tigramite import data_processing
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.lpcmci import LPCMCI

data, tau_max = np.load(sys.argv[1]), int(sys.argv[2])
links = json.load(open(sys.argv[3]))
assumptions = None
if links is not None:
    assumptions = {j: {} for j in range(data.shape[1])}
    for j, i, lag, link in links:
        assumptions[j][(i, lag)] = link
search = LPCMCI(dataframe=data_processing.DataFrame(data),
                cond_ind_test=ParCorr(significance="analytic"), verbosity=0)
out = search.run_lpcmci(link_assumptions=assumptions, tau_min=0, tau_max=tau_max, pc_alpha=0.05,
                        n_preliminary_iterations=1)
print(json.dumps(out["graph"].tolist()))
"""


def _tigramite_reference(data: np.ndarray, claims: dict | None, tau_max: int) -> np.ndarray:
    """tigramite's LPCMCI, at the settings of the harness's adapter, with the claims as its
    ``link_assumptions`` (:func:`link_assumptions`), in tigramite's own environment."""
    from andrey_bench.window import marks_from_link_strings

    links = link_assumptions(claims, data.shape[1], tau_max) if claims else None
    with tempfile.TemporaryDirectory(prefix="andrey_bench_tigramite_") as tmp:
        np.save(Path(tmp) / "data.npy", np.asarray(data, dtype=np.float64))
        links_file = Path(tmp) / "links.json"
        links_file.write_text(json.dumps(links))
        command = [os.environ[TIGRAMITE_PYTHON], "-c", _TIGRAMITE_FIT, f"{tmp}/data.npy"]
        done = subprocess.run(
            [*command, str(tau_max), str(links_file)], capture_output=True, text=True
        )
    if done.returncode != 0:
        raise RuntimeError(f"tigramite exited {done.returncode}: {done.stderr[-300:]}")
    return marks_from_link_strings(np.array(json.loads(done.stdout.splitlines()[-1])))


def scored(
    lane: str, result: dict, truth: np.ndarray, claims: dict | None = None, base=None
) -> dict:
    """``result`` with its marks replaced by its error counts against ``truth``; ``claims`` are the
    true claims the fit took, which set the class a ``cpdag`` fit is scored against, and ``base``
    the class the truth defines when it is not the true DAG's CPDAG (:func:`class_errors`)."""
    marks = result.pop("marks", None)
    if marks is None:
        return result
    if KIND[lane] == "window":
        return {**result, **window_errors(marks, truth)}
    d = len(truth)
    marks = marks[:d, :d]
    if KIND[lane] == "cpdag":
        return {**result, **class_errors(marks, truth, claims, base)}
    return {**result, **static_errors(marks, truth, undirected_open=False)}


def run_unit(task: tuple) -> list[dict]:
    """Every fit of one dataset, for the methods and arms asked for.

    A method of the lane is fitted without claims (``bare``), with true claims at each fraction
    (``true``), with the edges those claims imply (``edges``, in ``cpdag`` and ``pag``), and with
    false claims (``false``). A reference is fitted without knowledge and with the same true
    knowledge: the claims themselves when it takes them (:data:`TAKES_PATHS`, arm ``true``), the
    edges they imply otherwise (arm ``edges``). With a ``part``, the true claims are that part's
    alone and there is no ``false`` arm.
    """
    lane, cell_name, seed, part, methods, wanted, *rest = task
    only = rest[0] if rest else None  # the fractions to fit; all when absent
    cell = LANES[lane]["cells"][cell_name]
    try:
        prob = problem(lane, cell, seed)
    except ValueError as exc:  # the generator cannot hide the key's hidden nodes on this graph
        skipped = {"lane": lane, "cell": cell_name, "seed": seed, "part": part or "all"}
        return [{**skipped, "arm": "none", "status": "no_dataset", "message": str(exc)[:300]}]
    truth = prob["truth"]
    true_claims = {
        f: claims if part is None else {p: c if p == part else [] for p, c in claims.items()}
        for f, claims in prob["claims"].items()
    }
    rows = []
    for method in methods:
        reference = method not in own_methods(lane)
        arms = [("bare", None, None)]
        if not reference or method in TAKES_PATHS:
            arms += [("true", f, claims) for f, claims in true_claims.items()]
        if KIND[lane] in ("cpdag", "pag") and method not in TAKES_PATHS:
            arms += [("edges", f, claims) for f, claims in true_claims.items()]
        if part is None and not reference:
            arms.append(("false", FALSE_FRACTION, prob["false_claims"]))
        for arm, fraction, claims in arms:
            if arm not in wanted or (only and fraction not in only and arm != "bare"):
                continue
            if method.startswith("lingam.") and claims is not None and claims.get("confounded"):
                claims = {**claims, "confounded": []}
            edges = edge_claims(claims, truth) if arm == "edges" else None
            # An edge arm's class: the true CPDAG with the implied edges, which on a pair adjacent
            # in the truth say what the path claims they come from say.
            target = claims if arm == "true" else None
            if edges is not None:
                adjacent = [e for e in edges["forbidden"] if truth[e[0], e[1]] != NULL]
                target = {"required_paths": edges["required"], "forbidden_paths": adjacent}
            result = scored(
                lane,
                fit(method, prob["data"], claims, cell, seed, edges, prob.get("env")),
                truth,
                target,
                prob.get("base"),
            )
            counts = {f"n_{p}": len((claims or {}).get(p, [])) for p in PARTS}
            if edges is not None:
                counts.update({f"n_{key}": len(edges[key]) for key in ("required", "forbidden")})
                counts.update({f"n_{key}": edges[key] for key in edges if key.endswith("form")})
            rows.append(
                {
                    "lane": lane,
                    "cell": cell_name,
                    "seed": seed,
                    "part": part or "all",
                    "dataset_id": prob["dataset_id"],
                    "method": method,
                    "arm": arm,
                    "fraction": fraction,
                    **counts,
                    **result,
                }
            )
    return rows


PARTS = ("required_paths", "forbidden_paths", "confounded")


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def cmd_run(args: argparse.Namespace) -> int:
    from andrey_bench import rundir

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lane = args.lane
    cells = args.cell or sorted(LANES[lane]["cells"])
    seeds = range(*args.seeds)
    import andrey

    methods = tuple(args.methods or LANES[lane]["methods"])
    known = {*own_methods(lane), *LANE_REFERENCES[lane]}
    unknown = set(methods) - known
    if unknown:
        raise SystemExit(f"{lane} runs no {sorted(unknown)}")
    arms = tuple(args.arms or ARMS)
    meta = {
        "andrey": andrey.__version__,
        "lane": lane,
        "cells": cells,
        "seeds": list(args.seeds),
        "methods": methods,
        "arms": arms,
        "fractions": args.fractions or LANES[lane]["fractions"],
        "false_fraction": FALSE_FRACTION,
        "fit_cap_s": FIT_CAP_S,
        "workers": args.workers,
    }
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2))
    prov = rundir.provenance_once(out, timestamp=datetime.now(timezone.utc).isoformat())
    done = {(r["cell"], r["seed"]) for r in _read(out / RECORDS)}
    tasks = [
        (lane, c, s, args.part, methods, arms, args.fractions)
        for c in cells
        for s in seeds
        if (c, s) not in done
    ]
    _log(f"=== {lane}: {len(tasks)} datasets -> {out} (git {prov['git_sha']})")
    start = time.monotonic()
    with multiprocessing.get_context("forkserver").Pool(args.workers, maxtasksperchild=1) as pool:
        with (out / RECORDS).open("a") as fh:
            for rows in pool.imap_unordered(run_unit, tasks):
                for row in rows:
                    fh.write(json.dumps(row) + "\n")
                fh.flush()
                _log(f"{rows[0]['cell']} seed {rows[0]['seed']}: {len(rows)} fits")
    (out / DONE).write_text(
        json.dumps({"datasets": len(tasks), "wall_s": time.monotonic() - start})
    )
    return 0


# ------------------------------------------------------------------------------------------------
# oracle lanes
# ------------------------------------------------------------------------------------------------


def all_dags(d: int):
    """Every DAG on ``d`` labeled nodes, as endpoint marks."""
    pairs = list(itertools.combinations(range(d), 2))
    for choice in itertools.product((0, 1, 2), repeat=len(pairs)):
        marks = np.zeros((d, d), dtype=np.int8)
        for (a, b), c in zip(pairs, choice, strict=True):
            if c == 1:
                marks[a, b], marks[b, a] = TAIL, ARROW
            elif c == 2:
                marks[b, a], marks[a, b] = TAIL, ARROW
        if not np.diag(dag_reach(marks)).any():
            yield marks


def random_dag(rng: np.random.Generator, d: int) -> np.ndarray:
    order = rng.permutation(d)
    p = rng.uniform(0.3, 0.8)
    marks = np.zeros((d, d), dtype=np.int8)
    for a in range(d):
        for b in range(a + 1, d):
            if rng.random() < p:
                marks[order[a], order[b]], marks[order[b], order[a]] = TAIL, ARROW
    return marks


def class_members(cpdag: np.ndarray) -> list[np.ndarray]:
    """The DAGs of a CPDAG's class: each acyclic orientation of its undirected edges with the same
    CPDAG."""
    from andrey.core.orient import dag2cpdag

    lines = np.argwhere(np.triu((cpdag == TAIL) & (cpdag.T == TAIL))).tolist()
    members = []
    for bits in itertools.product((0, 1), repeat=len(lines)):
        dag = cpdag.copy()
        for (i, j), bit in zip(lines, bits, strict=True):
            a, b = (i, j) if bit else (j, i)
            dag[a, b], dag[b, a] = TAIL, ARROW
        if not np.diag(dag_reach(dag)).any() and np.array_equal(dag2cpdag(dag), cpdag):
            members.append(dag)
    return members


def _register_oracle(dag, d: int) -> np.ndarray:
    """Register d-separation in ``dag`` as the test ``dsep_oracle`` and return data whose columns
    name the nodes ``0..d-1``, in whatever order a search reads them."""
    from andrey.core import ci
    from andrey.data.latent import DSepOracle

    x = np.random.default_rng(0).standard_normal((30, d))

    def oracle(columns):
        nodes = [next(j for j in range(d) if np.array_equal(c, x[:, j])) for c in columns.T]
        return DSepOracle(dag, nodes=nodes)

    ci._CI_REGISTRY["dsep_oracle"] = oracle
    return x


def _nx_dag(marks: np.ndarray):
    import networkx as nx

    g = nx.DiGraph()
    g.add_nodes_from(range(len(marks)))
    g.add_edges_from(map(tuple, np.argwhere((marks == TAIL) & (marks.T == ARROW)).tolist()))
    return g


def _cpdag_claim_sets(rng, truth: np.ndarray) -> list[tuple[str, np.ndarray, np.ndarray]]:
    d = len(truth)
    reach = dag_reach(truth)
    facts = {(a, b): bool(reach[a, b]) for a in range(d) for b in range(d) if a != b}
    sets = []
    for label, fraction, true in (
        ("true-0.2", 0.2, True),
        ("true-0.5", 0.5, True),
        ("true-1.0", 1.0, True),
        ("false-0.2", 0.2, False),
    ):
        claims = draw_claims(rng, facts, fraction, [], true=True)
        if not true:
            a, b = list(facts)[rng.integers(len(facts))]
            part = "forbidden_paths" if facts[(a, b)] else "required_paths"
            claims[part].append([a, b])
        req, forb = np.zeros((d, d), dtype=bool), np.zeros((d, d), dtype=bool)
        for a, b in claims["required_paths"]:
            req[a, b] = True
        for a, b in claims["forbidden_paths"]:
            forb[a, b] = True
        sets.append((label, req, forb))
    return sets


def oracle_cpdag(seed: int, graphs: int, exhaustive: tuple[int, ...]) -> dict:
    """The CPDAG oracle lane: see the module docstring."""
    from andrey.constraint import pc as pc_engine
    from andrey.core.ancestral import restrict
    from andrey.core.orient import dag2cpdag, meek

    rng = np.random.default_rng(seed)
    tally: dict[str, int] = {
        "graphs": 0,
        "claim_sets": 0,
        "exact": 0,
        "dropped_as_expected": 0,
        "pc_true_cpdag": 0,
        "with_clauses": 0,
    }
    failures = []

    def directed(m):
        return {tuple(e) for e in np.argwhere((m == TAIL) & (m.T == ARROW)).tolist()}

    def meets(dag, req, forb):
        r = dag_reach(dag)
        return bool((r[req]).all() and not (r[forb]).any())

    def holds(dag, clauses):
        return all(
            any(dag[t, h] == TAIL and dag[h, t] == ARROW for h in heads) for t, heads in clauses
        )

    def truths():
        for d in exhaustive:
            by_class: dict[bytes, list] = {}
            dags = list(all_dags(d))
            for dag in dags:
                by_class.setdefault(dag2cpdag(dag).tobytes(), []).append(dag)
            for dag in dags:
                yield dag, by_class[dag2cpdag(dag).tobytes()]
        k = 0
        while k < graphs:
            dag = random_dag(rng, int(rng.integers(6, 9)))
            cpdag = dag2cpdag(dag)
            if ((cpdag == TAIL) & (cpdag.T == TAIL)).sum() // 2 > 11:
                continue
            k += 1
            yield dag, class_members(cpdag)

    for dag, members in truths():
        d = len(dag)
        tally["graphs"] += 1
        cpdag = dag2cpdag(dag)
        x = _register_oracle(_nx_dag(dag), d)
        found = pc_engine.pc(x, indep_test="dsep_oracle")
        found = np.asarray(found.to_numpy() if hasattr(found, "to_numpy") else found, dtype=np.int8)
        tally["pc_true_cpdag"] += bool(np.array_equal(found, cpdag))
        for label, req, forb in _cpdag_claim_sets(rng, dag):
            tally["claim_sets"] += 1
            kept = [m for m in members if meets(m, req, forb)]
            mpdag, clauses, unmet = restrict(found, req, forb)
            if unmet:
                if kept:
                    failures.append({"why": "dropped", "dag": dag.tolist(), "claims": label})
                else:
                    tally["dropped_as_expected"] += 1
                continue
            if not kept:
                failures.append({"why": "did not drop", "dag": dag.tolist(), "claims": label})
                continue
            common = set.intersection(*(directed(m) for m in kept))
            expected = cpdag.copy()
            for a, b in common:
                expected[a, b], expected[b, a] = TAIL, ARROW
            within = [m for m in members if directed(mpdag) <= directed(m)]
            ok = (
                np.array_equal(mpdag, expected)
                and np.array_equal(meek(mpdag, rule4=True), mpdag)
                and [holds(m, clauses) for m in within] == [meets(m, req, forb) for m in within]
            )
            tally["exact"] += ok
            tally["with_clauses"] += bool(clauses)
            if not ok and len(failures) < 50:
                failures.append({"why": "mismatch", "dag": dag.tolist(), "claims": label})
    return {
        "lane": "cpdag",
        "seed": seed,
        **tally,
        "failures": len(failures),
        "examples": failures[:20],
    }


def latent_dags(rng: np.random.Generator, graphs: int, exhaustive: bool):
    """Every DAG on 4 observed nodes with one hidden parent of two or more, then ``graphs`` random
    DAGs on 5 to 8 observed and 1 to 3 hidden nodes, each hidden one a parent of two or three."""
    for obs in all_dags(4) if exhaustive else ():
        for size in (2, 3, 4):
            for children in itertools.combinations(range(4), size):
                marks = np.zeros((5, 5), dtype=np.int8)
                marks[:4, :4] = obs
                for c in children:
                    marks[4, c], marks[c, 4] = TAIL, ARROW
                yield marks, 4
    for _ in range(graphs):
        d, h = int(rng.integers(5, 9)), int(rng.integers(1, 4))
        marks = np.zeros((d + h, d + h), dtype=np.int8)
        order = rng.permutation(d)
        for i in range(d):
            for j in range(i + 1, d):
                if rng.random() < 2.0 / (d - 1):
                    marks[order[i], order[j]], marks[order[j], order[i]] = TAIL, ARROW
        for hidden in range(d, d + h):
            for c in rng.choice(d, int(rng.integers(2, 4)), replace=False):
                marks[hidden, c], marks[c, hidden] = TAIL, ARROW
        yield marks, d


def mag_class(pag: np.ndarray) -> list[np.ndarray]:
    """The MAGs whose PAG is ``pag``: each assignment of a tail or an arrowhead to its circles that
    is ancestral (no directed or almost directed cycle) and has the same PAG."""
    from andrey.core import GraphStructure
    from andrey.core._convert import pag_of

    circles = np.argwhere(pag == CIRCLE).tolist()
    found = []
    for bits in itertools.product((TAIL, ARROW), repeat=len(circles)):
        mag = pag.copy()
        for (i, j), mark in zip(circles, bits, strict=True):
            mag[i, j] = mark
        directed = (mag == TAIL) & (mag.T == ARROW)
        if ((mag == TAIL) & (mag.T == TAIL)).any():
            continue
        reach = directed.copy()
        for k in range(len(mag)):
            reach |= reach[:, [k]] & reach[[k], :]
        if np.diag(reach).any() or (reach & (mag == ARROW) & (mag.T == ARROW)).any():
            continue
        try:
            again = pag_of(GraphStructure.from_numpy(mag, kind="mag"), None, []).to_numpy()
        except ValueError:  # not maximal
            continue
        if np.array_equal(again, pag):
            found.append((mag, reach))
    return found


def oracle_pag(
    seed: int, graphs: int, exhaustive: tuple[int, ...], max_circles: int = MAX_CIRCLES
) -> dict:
    """The PAG oracle lane: see the module docstring."""
    from andrey.constraint import fci as fci_engine
    from andrey.core import LATENT, OBSERVED, GraphStructure
    from andrey.core.knowledge import resolve
    from andrey.core.orient import dag2cpdag, to_structure
    from andrey.data.latent import DSepOracle, marginal
    from andrey.search import gfci as engines

    rng = np.random.default_rng(seed)
    tally: dict[str, dict[str, int]] = {}
    failures = []
    originals = {name: getattr(engines, name) for name in ("ges", "boss")}

    def bump(method, key, by=1):
        tally.setdefault(method, {}).setdefault(key, 0)
        tally[method][key] += by

    for marks, d in latent_dags(rng, graphs, 4 in exhaustive):
        n = len(marks)
        types = np.array([OBSERVED] * d + [LATENT] * (n - d), dtype=np.int8)
        mag = marginal(
            GraphStructure.from_numpy(marks, kind="dag", node_types=types), "mag"
        ).to_numpy()
        nx_dag = _nx_dag(marks)
        x = _register_oracle(nx_dag, d)
        reach = dag_reach(marks)[:d, :d]
        hidden = np.arange(n) >= d
        confounded = sorted(hidden_cause_pairs(marks, hidden))
        facts = {(a, b): bool(reach[a, b]) for a in range(d) for b in range(d) if a != b}
        # The pattern a score search finds in the large-sample limit: the minimal I-map's CPDAG in
        # a causal order of the DAG.
        order = [v for v in _topological(marks) if v < d]
        separated = DSepOracle(nx_dag)
        imap = np.zeros((d, d), dtype=np.int8)
        for k, child in enumerate(order):
            for parent in order[:k]:
                if not separated(child, parent, [v for v in order[:k] if v != parent]):
                    imap[parent, child], imap[child, parent] = TAIL, ARROW
        pattern = to_structure(dag2cpdag(imap), kind="cpdag")
        for name in originals:
            setattr(engines, name, lambda *a, p=pattern, **k: (p, 0.0))
        pag = fci_engine.fci(x, indep_test="dsep_oracle").to_numpy()
        listed = None
        if (pag == CIRCLE).sum() <= max_circles:
            listed = mag_class(pag)
        for fraction in (0.3, 1.0):
            claims = draw_claims(
                np.random.default_rng([seed, d, round(fraction * 10)]), facts, fraction, confounded
            )
            claims["confounded"] = [list(p) for p in confounded]
            req, forb = np.zeros((d, d), dtype=bool), np.zeros((d, d), dtype=bool)
            for a, b in claims["required_paths"]:
                req[a, b] = True
            for a, b in claims["forbidden_paths"]:
                forb[a, b] = True
            expected = None
            if listed is not None:
                kept = [m for m, r in listed if r[req].all() and not r[forb].any()]
                if kept:
                    expected = kept[0].copy()
                    for m in kept[1:]:
                        expected[(m != expected)] = CIRCLE
            for method in ("fci", "gfci", "boss_fci"):
                constraints = resolve(
                    knowledge_of(claims), d, None, method=method, ancestral_tiers=True
                )
                engine = fci_engine.fci if method == "fci" else getattr(engines, method)
                found = engine(x, indep_test="dsep_oracle", constraints=constraints).to_numpy()
                bump(method, "fits")
                wrong = (found != NULL) & (found != CIRCLE) & (found != mag)
                sound = (
                    np.array_equal(found != NULL, mag != NULL)
                    and not wrong.any()
                    and all(found[a, b] != NULL for a, b in confounded)
                )
                bump(method, "sound", sound)
                if not sound and len(failures) < 50:
                    failures.append(
                        {"method": method, "dag": marks.tolist(), "d": d, "claims": claims}
                    )
                if expected is not None:
                    bump(method, "listed")
                    bump(method, "complete", bool(np.array_equal(found, expected)))
                    bump(
                        method,
                        "circles_left",
                        int(((found == CIRCLE) & (expected != CIRCLE)).sum()),
                    )
    for name, fn in originals.items():
        setattr(engines, name, fn)
    return {
        "lane": "pag",
        "seed": seed,
        "methods": tally,
        "failures": len(failures),
        "examples": failures[:20],
    }


def _topological(marks: np.ndarray) -> list[int]:
    import networkx as nx

    return list(nx.topological_sort(_nx_dag(marks)))


def cmd_oracle(args: argparse.Namespace) -> int:
    from andrey_bench import rundir

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prov = rundir.provenance_once(out, timestamp=datetime.now(timezone.utc).isoformat())
    _log(f"=== oracle {args.lane} -> {out} (git {prov['git_sha']})")
    start = time.monotonic()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if args.lane == "cpdag":
            result = oracle_cpdag(args.seed, args.graphs, tuple(args.exhaustive))
        else:
            result = oracle_pag(args.seed, args.graphs, tuple(args.exhaustive), args.max_circles)
    result["wall_s"] = time.monotonic() - start
    (out / f"oracle-{args.lane}.json").write_text(json.dumps(result, indent=2, default=int))
    (out / DONE).write_text(json.dumps({"lane": args.lane, "failures": result["failures"]}))
    _log(json.dumps({k: v for k, v in result.items() if k != "examples"}, default=int))
    return 0


# ------------------------------------------------------------------------------------------------
# summarize
# ------------------------------------------------------------------------------------------------


def bootstrap_95(values: np.ndarray) -> list[float]:
    """The 95 % percentile interval of the mean of ``values`` over :data:`RESAMPLES` resamples."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(0)
    means = values[rng.integers(0, values.size, size=(RESAMPLES, values.size))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return [float(lo), float(hi)]


def _verdict(interval: list[float]) -> str:
    return "worse" if interval[0] > 0 else "better" if interval[1] < 0 else "tie"


def _outside(lane: str, method: str, cell: str) -> bool:
    rule = OUT_OF_SETTING.get((lane, method))
    return rule is not None and cell in LANES[lane]["cells"] and rule(LANES[lane]["cells"][cell])


def compare(records: list[dict]) -> list[dict]:
    """Each (lane, method, cell, arm, fraction) against the bare fits of the same datasets."""
    bare = {(r["method"], r["dataset_id"]): r for r in records if r["arm"] == "bare"}
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        if r["arm"] not in ("bare", "none"):
            groups.setdefault(
                (r["lane"], r["method"], r["cell"], r.get("part", "all"), r["arm"], r["fraction"]),
                [],
            ).append(r)
    rows = []
    for (lane, method, cell, part, arm, fraction), rs in sorted(groups.items(), key=str):
        pairs = [(r, bare.get((method, r["dataset_id"]))) for r in rs]
        pairs = [(r, b) for r, b in pairs if b is not None and b["status"] == "ok"]
        ok = [(r, b) for r, b in pairs if r["status"] == "ok"]
        row = {
            "lane": lane,
            "method": method,
            "cell": cell,
            "part": part,
            "arm": arm,
            "fraction": fraction,
            "in_setting": not _outside(lane, method, cell),
            "datasets": len(rs),
            "paired": len(ok),
            "knowledge_error": sum(r["status"] == "knowledge_error" for r, _ in pairs),
            "failed": sum(r["status"] in ("error", "timeout") for r, _ in pairs),
            "claims": float(np.mean([sum(r[f"n_{p}"] for p in PARTS) for r in rs])),
        }
        if ok:
            D = np.array([r["errors"] - b["errors"] for r, b in ok], dtype=float)
            interval = bootstrap_95(D)
            row.update(
                errors_with=float(np.mean([r["errors"] for r, _ in ok])),
                errors_without=float(np.mean([b["errors"] for _, b in ok])),
                mean_D=float(D.mean()),
                bootstrap_95=interval,
                fewer=int((D < 0).sum()),
                more=int((D > 0).sum()),
                verdict=_verdict(interval),
                definite_with=float(np.mean([r["definite_marks"] for r, _ in ok])),
                definite_without=float(np.mean([b["definite_marks"] for _, b in ok])),
                floor_with=bootstrap_95(np.array([r["errors"] - r["empty_errors"] for r, _ in ok])),
                floor_without=bootstrap_95(
                    np.array([b["errors"] - b["empty_errors"] for _, b in ok])
                ),
            )
        if ok and all("dag_errors" in r for pair in ok for r in pair):
            D = np.array([r["dag_errors"] - b["dag_errors"] for r, b in ok], dtype=float)
            interval = bootstrap_95(D)
            row.update(
                dag_errors_with=float(np.mean([r["dag_errors"] for r, _ in ok])),
                dag_errors_without=float(np.mean([b["dag_errors"] for _, b in ok])),
                dag_mean_D=float(D.mean()),
                dag_bootstrap_95=interval,
                dag_verdict=_verdict(interval),
                reversed_with=float(np.mean([r["wrong_marks"] for r, _ in ok])) / 2,
                reversed_without=float(np.mean([b["wrong_marks"] for _, b in ok])) / 2,
            )
        rows.append(row)
    return rows


def against_references(records: list[dict]) -> list[dict]:
    """Each method against each of its references (:data:`REFERENCES`) on the same datasets.

    A row pairs the method's fits of one arm with the reference's fits under the same knowledge:
    the method's ``edges`` arm and its ``true`` arm against the reference's knowledge arm (the
    like-for-like comparison and the path claims against the edges they imply; one row where the
    reference takes the claims), and ``bare`` against ``bare``. The measure is ``dag_errors`` in
    ``cpdag``, the same for every arm, and ``errors`` elsewhere; ``mean_D`` is the method's minus
    the reference's, and ``floor`` the interval of the method's count minus the empty graph's.
    """
    fits: dict[tuple, dict] = {}
    for r in records:
        if r.get("status") == "ok":
            part = "all" if r["arm"] == "bare" else r.get("part", "all")
            fits[r["method"], r["dataset_id"], part, r["arm"], r["fraction"]] = r
    groups: dict[tuple, list] = {}
    for (method, dataset, part, arm, fraction), r in fits.items():
        for reference in REFERENCES.get(method, ()):
            takes = "true" if reference in TAKES_PATHS else "edges"
            other = fits.get(
                (reference, dataset, part, "bare" if arm == "bare" else takes, fraction)
            )
            if other is not None and arm in ("bare", "true", "edges"):
                key = (r["lane"], method, reference, r["cell"], part, arm, fraction)
                groups.setdefault(key, []).append((r, other))
    rows = []
    for (lane, method, reference, cell, part, arm, fraction), pairs in sorted(
        groups.items(), key=str
    ):
        measure = "dag_errors" if KIND[lane] == "cpdag" else "errors"
        own = np.array([r[measure] for r, _ in pairs], dtype=float)
        ref = np.array([o[measure] for _, o in pairs], dtype=float)
        empty = np.array([r["empty_" + measure] for r, _ in pairs], dtype=float)
        interval = bootstrap_95(own - ref)
        rows.append(
            {
                "lane": lane,
                "method": method,
                "reference": reference,
                "cell": cell,
                "part": part,
                "arm": arm,
                "fraction": fraction,
                "in_setting": not _outside(lane, method, cell),
                "paired": len(pairs),
                "measure": measure,
                "errors_method": float(own.mean()),
                "errors_reference": float(ref.mean()),
                "mean_D": float((own - ref).mean()),
                "bootstrap_95": interval,
                "verdict": _verdict(interval),
                "floor": bootstrap_95(own - empty),
                "no_edge_form": float(np.mean([o.get("n_no_edge_form", 0) for _, o in pairs])),
                "reference_unmet": float(np.mean([o.get("unmet", 0) for _, o in pairs])),
            }
        )
    return rows


def cmd_summarize(args: argparse.Namespace) -> int:
    def pool(dirs):
        return [r for d in dirs for r in _read(Path(d) / RECORDS)]

    records, again_records = pool(args.runs), pool(args.replicate or [])
    reference_rows = against_references(records)
    names = ("lane", "method", "reference", "cell", "part", "arm", "fraction")
    again = {tuple(r[k] for k in names): r for r in against_references(again_records)}
    for row in reference_rows:
        if row["verdict"] == "worse":
            repeat = again.get(tuple(row[k] for k in names))
            row["replicate"] = repeat and {
                k: repeat[k] for k in ("paired", "mean_D", "bootstrap_95", "verdict")
            }
            row["counts"] = repeat is not None and repeat["verdict"] == "worse"
    main_rows = compare(records)
    replicate = {
        (r["lane"], r["method"], r["cell"], r["part"], r["arm"], r["fraction"]): r
        for r in compare(again_records)
    }
    kept = ("paired", "mean_D", "bootstrap_95", "verdict")
    kept += ("dag_mean_D", "dag_bootstrap_95", "dag_verdict")
    for row in main_rows:
        worse = [v for v in ("verdict", "dag_verdict") if row.get(v) == "worse"]
        if worse and row["arm"] in ("true", "edges"):
            again = replicate.get(
                (row["lane"], row["method"], row["cell"], row["part"], row["arm"], row["fraction"])
            )
            row["replicate"] = None if again is None else {k: again[k] for k in kept if k in again}
            row["counts"] = again is not None and any(again.get(v) == "worse" for v in worse)
    oracle = {
        f"{Path(d).name}/{p.name}": json.loads(p.read_text())
        for d in (args.oracle or [])
        for p in Path(d).glob("oracle-*.json")
    }
    Path(args.out).write_text(
        json.dumps(
            {"rows": main_rows, "reference_rows": reference_rows, "oracle": oracle},
            indent=2,
            default=int,
        )
    )
    _log(f"{len(main_rows)} rows, {len(reference_rows)} against references -> {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    o = sub.add_parser("oracle", help="the exactness lanes, with d-separation")
    o.add_argument("--lane", choices=("cpdag", "pag"), required=True)
    o.add_argument("--out", required=True)
    o.add_argument("--seed", type=int, default=0)
    o.add_argument(
        "--graphs", type=int, default=600, help="random graphs after the exhaustive ones"
    )
    o.add_argument("--max-circles", type=int, default=MAX_CIRCLES, help="largest PAG listed")
    o.add_argument("--exhaustive", type=int, nargs="*", default=[4, 5], help="sizes listed in full")
    r = sub.add_parser("run", help="the no-worse lanes, on simulated data")
    r.add_argument("--lane", choices=sorted(LANES), required=True)
    r.add_argument("--cell", action="append", help="a cell of the lane; all when absent")
    r.add_argument("--seeds", type=int, nargs=2, default=(0, 20), metavar=("FIRST", "STOP"))
    r.add_argument("--workers", type=int, default=max(1, len(os.sched_getaffinity(0))))
    r.add_argument("--part", choices=PARTS, help="true claims of this part alone")
    r.add_argument(
        "--methods", nargs="+", help="the lane's methods, variants, or references; default: methods"
    )
    r.add_argument("--arms", nargs="+", choices=ARMS, help="the arms to fit; default: all")
    r.add_argument(
        "--fractions", type=float, nargs="+", help="fit the claim arms at these fractions only"
    )
    r.add_argument("--out", required=True)
    s = sub.add_parser("summarize", help="paired verdicts from finished runs")
    s.add_argument("--runs", nargs="+", required=True)
    s.add_argument("--replicate", nargs="*", help="runs of the same cells on fresh seeds")
    s.add_argument("--oracle", nargs="*", help="oracle run directories")
    s.add_argument("--out", required=True)
    args = p.parse_args(argv)
    return {"oracle": cmd_oracle, "run": cmd_run, "summarize": cmd_summarize}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
