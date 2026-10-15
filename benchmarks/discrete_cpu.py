#!/usr/bin/env python
"""Compare the packages on categorical data: published networks and random discrete networks.

The published networks (ASIA, ALARM, HEPAR2, ANDES) are sampled from their own probability tables
with ``andrey.data.load_dataset(name, discrete=True)``; the random networks are
``andrey.data.discrete.DiscreteBN`` on an Erdos-Renyi graph of mean degree 2, with 2 to 4 states
per variable. Each method runs every package with matched settings:

- PC: the chi-square and the G-square test at ``alpha = 0.05``, with degrees of freedom adjusted for
  empty rows and columns (bnlearn's ``x2-adf`` and ``mi-adf``).
- GES, hill climbing, Tabu search, BOSS: BDeu with equivalent sample size 1 and the structure prior
  that makes each other variable a parent with probability ``1/(d - 1)`` (bnlearn's ``vsp``);
  GES caps parents at ``d/2``, Tabu search keeps a list of 10.
- Hill climbing also runs the multinomial BIC, against pgmpy and bnlearn.

Every network is downloaded once, here in the parent, before any fit starts. Each fit runs isolated
after one warm-up, with one numeric thread, under a per-repeat wall cap; a fit past the cap is
recorded as ``timeout``. Every estimate is scored against the network's DAG through its equivalence
class; ``cpdag_hash`` names that class, so a DAG and a CPDAG result compare on it. A sample in
which a variable takes one value is refused by the methods, so it is recorded as ``skipped``. One JSON line per timed repeat goes to ``<out>/records.jsonl``.

    $ANDREY_BENCH_PYTHON benchmarks/discrete_cpu.py --out outputs/discrete --seeds 0 1 2

``--smoke`` runs every solution once on ASIA and a 10-variable random network.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from andrey.data import load_dataset
from andrey.data.discrete import DiscreteBN
from andrey.data.graphs import erdos_renyi
from andrey.metrics import to_cpdag
from andrey_bench import scoring
from andrey_bench.adapters.andrey_boss import AndreyBOSS
from andrey_bench.adapters.andrey_ges import andrey_ges_adapters
from andrey_bench.adapters.andrey_hc import andrey_hc_adapters
from andrey_bench.adapters.andrey_pc import AndreyPC
from andrey_bench.adapters.andrey_tabu import andrey_tabu_adapters
from andrey_bench.adapters.baseline_empty import BaselineEmpty
from andrey_bench.adapters.causal_learn_boss import CausalLearnBOSS
from andrey_bench.adapters.causal_learn_ges import CausalLearnGES
from andrey_bench.adapters.causal_learn_pc import CausalLearnPC
from andrey_bench.adapters.pgmpy_hc import PgmpyHC
from andrey_bench.runner import run_task

NETWORKS = ("asia", "alarm", "hepar2", "andes")
METHODS = ("PC-chisq", "PC-gsq", "GES-BDeu", "HC-BDeu", "HC-BIC", "Tabu-BDeu", "BOSS-BDeu", "Empty")
BDEU = "local_score_BDeu"


def solutions(method: str, d: int, *, with_r: bool) -> list[tuple[object, dict]]:
    """Each package's adapter for ``method`` with its matched parameters, Andrey first."""
    andrey_hc = andrey_hc_adapters()[0]
    rows = {
        "PC-chisq": [
            (AndreyPC(), {"alpha": 0.05, "indep_test": "chisq"}),
            (CausalLearnPC(), {"alpha": 0.05, "indep_test": "chisq"}),
        ],
        "PC-gsq": [
            (AndreyPC(), {"alpha": 0.05, "indep_test": "gsq"}),
            (CausalLearnPC(), {"alpha": 0.05, "indep_test": "gsq"}),
        ],
        "GES-BDeu": [
            (
                andrey_ges_adapters()[0],
                {"score_func": BDEU, "lambda_value": 1.0, "max_parents": d / 2},
            ),
            (CausalLearnGES(), {"score_func": BDEU, "max_parents": d / 2}),
        ],
        "HC-BDeu": [(andrey_hc, {"score_func": BDEU, "lambda_value": 1.0})],
        "HC-BIC": [
            (andrey_hc, {"score_func": "local_score_BIC_discrete", "lambda_value": 1.0}),
            (PgmpyHC(), {"score": "bic"}),
        ],
        "Tabu-BDeu": [
            (
                andrey_tabu_adapters()[0],
                {"score_func": BDEU, "lambda_value": 1.0, "tabu_length": 10, "max_tabu": 10},
            )
        ],
        "BOSS-BDeu": [
            (AndreyBOSS(), {"score_func": BDEU, "lambda_value": 1.0, "seed": 0}),
            (CausalLearnBOSS(), {"score_func": BDEU, "seed": 0}),
        ],
        "Empty": [(BaselineEmpty(), {})],
    }[method]
    if with_r:
        from andrey_bench.adapters.bnlearn_discrete import (
            BnlearnDiscreteHC,
            BnlearnDiscretePC,
            BnlearnDiscreteTabu,
        )

        rows += {
            "PC-chisq": [(BnlearnDiscretePC(), {"alpha": 0.05, "test": "x2-adf"})],
            "PC-gsq": [(BnlearnDiscretePC(), {"alpha": 0.05, "test": "mi-adf"})],
            "HC-BDeu": [(BnlearnDiscreteHC(), {"score": "bde"})],
            "HC-BIC": [(BnlearnDiscreteHC(), {"score": "bic"})],
            "Tabu-BDeu": [(BnlearnDiscreteTabu(), {"score": "bde", "tabu": 10})],
        }.get(method, [])
    return rows


def datasets(args: argparse.Namespace):
    """Yield ``(name, n, seed, data, truth)`` for every requested dataset and seed."""
    for name in args.datasets:
        for n in args.n:
            for seed in args.seeds:
                ds = load_dataset(name, n=n, seed=seed, discrete=True, as_frame=False)
                yield name, n, seed, np.asarray(ds.data, dtype=np.float64), ds.graph
    for d in args.ladder:
        bn = DiscreteBN(graph=erdos_renyi(d=d, avg_degree=2.0), levels=(2, 4))
        for seed in args.seeds:
            ds = bn.sample(n=args.er_n, seed=seed)
            yield f"er{d}", args.er_n, seed, ds.data, ds.graph


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--datasets", nargs="*", default=["asia", "alarm", "hepar2"], choices=NETWORKS)
    ap.add_argument("--n", type=int, nargs="+", default=[1000, 5000], help="rows per network")
    ap.add_argument("--ladder", type=int, nargs="*", default=[10, 20, 50], help="random-network d")
    ap.add_argument("--er-n", type=int, default=2000, help="rows per random network")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    ap.add_argument("--repeats", type=int, default=3, help="timed repeats per fit")
    ap.add_argument("--cap-wall-s", type=float, default=1800.0)
    ap.add_argument("--no-r", action="store_true", help="leave out the bnlearn solutions")
    ap.add_argument("--smoke", action="store_true", help="one quick pass over every solution")
    args = ap.parse_args(argv)
    if args.smoke:
        args.datasets, args.n, args.ladder, args.seeds, args.repeats = ["asia"], [500], [10], [0], 1
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        load_dataset(name, n=2, discrete=True)  # one download per network, before any fit
    errors = 0
    with (args.out / "records.jsonl").open("a") as sink:
        for name, n, seed, x, truth in datasets(args):
            constant = np.flatnonzero(np.ptp(x, axis=0) == 0)
            for method in args.methods:
                for adapter, params in solutions(method, x.shape[1], with_r=not args.no_r):
                    base = {
                        "dataset": name,
                        "n": n,
                        "seed": seed,
                        "method": method,
                        "solution": adapter.name,
                        "params": params,
                    }
                    if constant.size:
                        sink.write(
                            json.dumps(
                                {
                                    **base,
                                    "status": "skipped",
                                    "reason": f"constant columns {constant.tolist()}",
                                }
                            )
                            + "\n"
                        )
                        continue
                    measures = run_task(
                        adapter,
                        x,
                        params,
                        cap_wall_s=args.cap_wall_s,
                        repeats=args.repeats,
                        warmup=1,
                        threads=1,
                    )
                    for repeat, m in enumerate(measures):
                        record = {
                            **base,
                            "repeat": repeat,
                            "status": m.get("status"),
                            "wall_s": m.get("wall_s"),
                        }
                        if m.get("status") == "ok" and m.get("adj") is not None:
                            est = adapter.to_structure(m["adj"])
                            record.update(scoring.score_result(est, truth))
                            record["structure_hash"] = scoring.structure_hash(est)
                            # A DAG and a CPDAG compare through the DAG's equivalence class.
                            cpdag = to_cpdag(est) if est.kind == "dag" else est
                            record["cpdag_hash"] = scoring.structure_hash(cpdag)
                        elif m.get("status") == "error":
                            errors += 1
                            record["error_text"] = m.get("error_text")
                        sink.write(json.dumps(record, default=str) + "\n")
                        sink.flush()
                    print(
                        name,
                        n,
                        seed,
                        method,
                        adapter.name,
                        [m.get("status") for m in measures],
                        flush=True,
                    )
    print(f"errors: {errors}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
