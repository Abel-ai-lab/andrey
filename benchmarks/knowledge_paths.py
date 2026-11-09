#!/usr/bin/env python
"""Benchmark path and confounder claims in background knowledge: exactness and no-worse lanes.

``Knowledge`` takes three claims beyond edges and tiers: ``required_paths`` (``a`` is an ancestor
of ``b``), ``forbidden_paths`` (``a`` is not), and ``confounded`` (a hidden common cause of ``a``
and ``b``). No other package implements these rules for the CPDAG and PAG searches, so the lanes
check them against exact answers and against the same fit without the claims.

``oracle`` runs the exactness lanes, with d-separation in place of the tests:

- ``cpdag``: every DAG on 4 and 5 nodes and random DAGs on 6 to 8. PC with d-separation returns
  the true CPDAG, and the step that ``pc``, ``ges``, and ``boss`` share (``restrict_paths``)
  restricts it by claims drawn from the truth, true and false. The MPDAG must direct an edge
  exactly when every DAG of the class that meets the claims directs it alike, the MPDAG with its
  clauses must keep exactly those DAGs, and claims no DAG meets must raise.
- ``pag``: every DAG on 4 observed nodes with one hidden parent of two or more of them, and random
  DAGs on 5 to 8 observed and 1 to 3 hidden. ``fci``, and ``gfci`` and ``boss_fci`` with the score
  phase at its large-sample limit, take true claims; every tail and arrowhead they set must be the
  true MAG's, the adjacencies the MAG's, and every confounded pair adjacent. Where the class is
  small enough to list (at most :data:`MAX_CIRCLES` circles), the MAGs of the true PAG that meet
  the claims are enumerated and each mark they share is compared with the fit's: completeness,
  reported and not judged.

``run`` runs the no-worse lanes on simulated data in each method's setting, with claims drawn from
the truth at several fractions. Each dataset is fitted without claims (``bare``), with true claims
at each fraction (``true``), and with false claims (``false``, context only). ``errors`` counts
the pairs adjacent in one graph and not the other plus every tail or arrowhead, on a pair adjacent
in both, that differs from the true DAG (``cpdag``, ``lingam``), the true MAG over the observed
variables (``pag``), or the true window MAG (``lpcmci``); an undirected edge or a circle claims
nothing and is never wrong, so a claim that orients an edge correctly lowers ``errors``.
``empty_errors`` is the empty graph's count, the floor. A fit that raises ``KnowledgeError``
because its result cannot meet the claims is recorded as ``knowledge_error``, never as a graph.
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

``summarize`` pairs each ``true`` fit with the ``bare`` fit of its dataset: the mean difference in
``errors`` and its 95 % interval over :data:`RESAMPLES` bootstrap resamples. A cell is ``worse``
when the interval lies above 0; a worse cell counts only when the same cell on fresh seeds
(``--replicate``) is worse too.

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
}
#: The share of ordered pairs the false-claims arm states wrongly.
FALSE_FRACTION = 0.1


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


def problem(lane: str, cell: dict, seed: int) -> dict:
    """The data, the truth the errors are counted against, and the claims of every arm."""
    from andrey_bench import datasets

    key = dataset_key(lane, cell, seed)
    data, marks, kind, _ = datasets.generate_one(key)
    rng = np.random.default_rng([seed, 9124])
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
    out["claims"] = {
        f: draw_claims(np.random.default_rng([seed, 9124, round(f * 100)]), facts, f, confounded)
        for f in LANES[lane]["fractions"]
    }
    out["false_claims"] = draw_claims(rng, facts, FALSE_FRACTION, confounded, true=False)
    return out


# ------------------------------------------------------------------------------------------------
# fits
# ------------------------------------------------------------------------------------------------


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout


def fit(method: str, data: np.ndarray, claims: dict | None, cell: dict, seed: int) -> dict:
    """One fit: its status, its marks, and the clauses or unmet claims its metadata lists."""
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
            else:
                fn = getattr(andrey, method)
                kwargs = {"knowledge": knowledge_of(claims)} if claims else {}
                if method == "lpcmci":
                    kwargs["tau_max"] = cell["tau_max"]
                if "seed" in inspect.signature(fn).parameters:
                    kwargs["seed"] = seed
                out = fn(data, **kwargs)
                meta = out.metadata
                if method == "lpcmci":
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


def _lingam_reference(data: np.ndarray, claims: dict | None) -> np.ndarray:
    """lingam's DirectLiNGAM with the claims as ``prior_knowledge``: entry ``[b, a]`` is 1 when
    ``a`` is an ancestor of ``b``, 0 when it is not, and -1 when unknown."""
    import lingam

    d = data.shape[1]
    prior = None
    if claims:
        prior = -np.ones((d, d), dtype=int)
        for a, b in claims.get("required_paths", []):
            prior[b, a] = 1
        for a, b in claims.get("forbidden_paths", []):
            prior[b, a] = 0
    model = lingam.DirectLiNGAM(prior_knowledge=prior).fit(data)
    weights = model.adjacency_matrix_  # weights[b, a] != 0: a -> b
    marks = np.zeros((d, d), dtype=np.int8)
    for b, a in np.argwhere(weights != 0):
        marks[a, b], marks[b, a] = TAIL, ARROW
    return marks


def scored(lane: str, result: dict, truth: np.ndarray) -> dict:
    """``result`` with its marks replaced by its error counts against ``truth``."""
    marks = result.pop("marks", None)
    if marks is None:
        return result
    if lane == "lpcmci":
        return {**result, **window_errors(marks, truth)}
    d = len(truth)
    marks = marks[:d, :d]
    return {**result, **static_errors(marks, truth, undirected_open=lane == "cpdag")}


def run_unit(task: tuple) -> list[dict]:
    """Every fit of one dataset: each method without claims, with true claims at each fraction,
    and with false claims; with a ``part``, with true claims of that part alone instead."""
    lane, cell_name, seed, part = task
    cell = LANES[lane]["cells"][cell_name]
    try:
        prob = problem(lane, cell, seed)
    except ValueError as exc:  # the generator cannot hide the key's hidden nodes on this graph
        skipped = {"lane": lane, "cell": cell_name, "seed": seed, "part": part or "all"}
        return [{**skipped, "arm": "none", "status": "no_dataset", "message": str(exc)[:300]}]
    truth = prob["truth"]
    rows = []
    arms = [("bare", None, None)]
    if part is None:
        arms += [("true", f, prob["claims"][f]) for f in LANES[lane]["fractions"]]
        arms.append(("false", FALSE_FRACTION, prob["false_claims"]))
    else:
        arms += [
            ("true", f, {p: c if p == part else [] for p, c in prob["claims"][f].items()})
            for f in LANES[lane]["fractions"]
        ]
    for method in LANES[lane]["methods"]:
        for arm, fraction, claims in arms:
            if method.startswith("lingam.") and claims is not None and claims.get("confounded"):
                claims = {**claims, "confounded": []}
            result = scored(lane, fit(method, prob["data"], claims, cell, seed), truth)
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
                    **{f"n_{part}": len((claims or {}).get(part, [])) for part in PARTS},
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

    meta = {
        "andrey": andrey.__version__,
        "lane": lane,
        "cells": cells,
        "seeds": list(args.seeds),
        "methods": LANES[lane]["methods"],
        "fractions": LANES[lane]["fractions"],
        "false_fraction": FALSE_FRACTION,
        "fit_cap_s": FIT_CAP_S,
        "workers": args.workers,
    }
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2))
    prov = rundir.provenance_once(out, timestamp=datetime.now(timezone.utc).isoformat())
    done = {(r["cell"], r["seed"]) for r in _read(out / RECORDS)}
    tasks = [(lane, c, s, args.part) for c in cells for s in seeds if (c, s) not in done]
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
    from andrey.core.knowledge import KnowledgeError
    from andrey.core.orient import dag2cpdag, meek

    rng = np.random.default_rng(seed)
    tally: dict[str, int] = {
        "graphs": 0,
        "claim_sets": 0,
        "exact": 0,
        "raised_as_expected": 0,
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
            try:
                mpdag, clauses = restrict(found, req, forb)
            except KnowledgeError:
                if kept:
                    failures.append({"why": "raised", "dag": dag.tolist(), "claims": label})
                else:
                    tally["raised_as_expected"] += 1
                continue
            if not kept:
                failures.append({"why": "did not raise", "dag": dag.tolist(), "claims": label})
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
                verdict="worse" if interval[0] > 0 else "better" if interval[1] < 0 else "tie",
                definite_with=float(np.mean([r["definite_marks"] for r, _ in ok])),
                definite_without=float(np.mean([b["definite_marks"] for _, b in ok])),
                floor_with=bootstrap_95(np.array([r["errors"] - r["empty_errors"] for r, _ in ok])),
                floor_without=bootstrap_95(
                    np.array([b["errors"] - b["empty_errors"] for _, b in ok])
                ),
            )
        rows.append(row)
    return rows


def cmd_summarize(args: argparse.Namespace) -> int:
    def pool(dirs):
        return [r for d in dirs for r in _read(Path(d) / RECORDS)]

    main_rows = compare(pool(args.runs))
    replicate = {
        (r["lane"], r["method"], r["cell"], r["part"], r["arm"], r["fraction"]): r
        for r in compare(pool(args.replicate or []))
    }
    for row in main_rows:
        if row.get("verdict") == "worse" and row["arm"] == "true":
            again = replicate.get(
                (row["lane"], row["method"], row["cell"], row["part"], row["arm"], row["fraction"])
            )
            row["replicate"] = (
                None
                if again is None
                else {k: again.get(k) for k in ("paired", "mean_D", "bootstrap_95", "verdict")}
            )
            row["counts"] = again is not None and again.get("verdict") == "worse"
    oracle = {
        p.name: json.loads(p.read_text())
        for d in (args.oracle or [])
        for p in Path(d).glob("oracle-*.json")
    }
    Path(args.out).write_text(
        json.dumps({"rows": main_rows, "oracle": oracle}, indent=2, default=int)
    )
    _log(f"{len(main_rows)} rows -> {args.out}")
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
