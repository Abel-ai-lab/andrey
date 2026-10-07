#!/usr/bin/env python
"""Compare the packages on the public datasets: Sachs, and data simulated on four bnlearn networks.

Loads each dataset with ``andrey.data.load_dataset`` and fits every package's adapter to the same
rows with matched settings: Fisher-z at ``alpha = 0.05`` for PC and FCI; Andrey's
``lambda_value = 1.0`` (causal-learn's 0.5, the same BIC penalty on its scale) for GES, BOSS,
and GRaSP, with at most ``d/2`` parents for GES and search seed 0; GRaSP depth 3. The published
networks' simulated data are Gaussian, so the LiNGAM methods run on measured data only.

Each fit runs isolated after one warm-up, with one numeric thread, under a per-repeat wall cap;
a fit past the cap is recorded as ``timeout``. Every estimate is scored against the dataset's
reference DAG (CPDAG, PAG, or DAG, by the method's output). FCI compares on ``shd_endpoint``, the
endpoint SHD; its ``shd`` is the plain SHD. One JSON line per timed repeat goes to
``<out>/records.jsonl``.

    ANDREY_BENCH_PYTHON=... $ANDREY_BENCH_PYTHON benchmarks/public_datasets.py \\
        --out outputs/public-datasets --seeds 0 1 2 --repeats 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from andrey.data import list_datasets, load_dataset
from andrey_bench import scoring
from andrey_bench.adapters.andrey_boss import AndreyBOSS
from andrey_bench.adapters.andrey_direct_lingam import andrey_direct_lingam_adapters
from andrey_bench.adapters.andrey_fci import AndreyFCI
from andrey_bench.adapters.andrey_ges import andrey_ges_adapters
from andrey_bench.adapters.andrey_grasp import AndreyGRaSP
from andrey_bench.adapters.andrey_ica_lingam import AndreyICALiNGAM
from andrey_bench.adapters.andrey_pc import AndreyPC
from andrey_bench.adapters.causal_learn_boss import CausalLearnBOSS
from andrey_bench.adapters.causal_learn_direct_lingam import CausalLearnDirectLiNGAM
from andrey_bench.adapters.causal_learn_fci import CausalLearnFCI
from andrey_bench.adapters.causal_learn_ges import CausalLearnGES
from andrey_bench.adapters.causal_learn_grasp import CausalLearnGRaSP
from andrey_bench.adapters.causal_learn_ica_lingam import CausalLearnICALiNGAM
from andrey_bench.adapters.causal_learn_pc import CausalLearnPC
from andrey_bench.adapters.gcastle_ges import GCastleGES
from andrey_bench.adapters.gcastle_pc import GCastlePC
from andrey_bench.adapters.lingam_direct import LingamDirect
from andrey_bench.runner import run_task

GAUSSIAN = ("PC", "FCI", "GES", "BOSS", "GRaSP")
NON_GAUSSIAN = ("DirectLiNGAM", "ICA-LiNGAM")


def solutions(method: str, d: int) -> list[tuple[object, dict]]:
    """Each package's adapter for ``method`` with its matched parameters, Andrey first."""
    fz = {"alpha": 0.05, "indep_test": "fisherz"}
    andrey_bic = {"score_func": "local_score_BIC_from_cov", "lambda_value": 1.0, "seed": 0}
    other_bic = {"score_func": "local_score_BIC_from_cov", "lambda_value": 0.5, "seed": 0}
    return {
        "PC": [(AndreyPC(), fz), (CausalLearnPC(), fz), (GCastlePC(), fz)],
        "FCI": [(AndreyFCI(), fz), (CausalLearnFCI(), fz)],
        "GES": [
            (andrey_ges_adapters()[0], {"lambda_value": 1.0, "max_parents": d / 2}),
            (
                CausalLearnGES(),
                {"score_func": "local_score_BIC", "lambda_value": 0.5, "max_parents": d / 2},
            ),
            (GCastleGES(), {"criterion": "bic", "method": "scatter"}),
        ],
        "BOSS": [(AndreyBOSS(), andrey_bic), (CausalLearnBOSS(), other_bic)],
        "GRaSP": [
            (AndreyGRaSP(), {**andrey_bic, "depth": 3}),
            (CausalLearnGRaSP(), {**other_bic, "depth": 3}),
        ],
        "DirectLiNGAM": [
            (andrey_direct_lingam_adapters()[0], {"measure": "pwling"}),
            (CausalLearnDirectLiNGAM(), {"measure": "pwling"}),
            (LingamDirect(), {}),
        ],
        "ICA-LiNGAM": [
            (AndreyICALiNGAM(), {"random_state": 0, "max_iter": 1000}),
            (CausalLearnICALiNGAM(), {"random_state": 0, "max_iter": 1000}),
        ],
    }[method]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--datasets", nargs="+", default=list(list_datasets()))
    ap.add_argument("--n", type=int, default=1000, help="rows simulated on a published network")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--repeats", type=int, default=3, help="timed repeats per fit")
    ap.add_argument("--cap-wall-s", type=float, default=1800.0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "records.jsonl").open("a") as sink:
        for name in args.datasets:
            probe = load_dataset(name, as_frame=False) if name == "sachs" else None
            measured = probe is not None
            seeds = [None] if measured else args.seeds
            for seed in seeds:
                ds = probe or load_dataset(name, n=args.n, seed=seed, as_frame=False)
                x, truth = np.asarray(ds.data, dtype=np.float64), ds.graph
                methods = GAUSSIAN + (NON_GAUSSIAN if measured else ())
                for method in methods:
                    for adapter, params in solutions(method, x.shape[1]):
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
                                "dataset": name,
                                "n": int(x.shape[0]),
                                "seed": seed,
                                "method": method,
                                "solution": adapter.name,
                                "repeat": repeat,
                                "status": m.get("status"),
                                "wall_s": m.get("wall_s"),
                                "params": params,
                            }
                            if m.get("status") == "ok" and m.get("adj") is not None:
                                est = adapter.to_structure(m["adj"])
                                record.update(scoring.score_result(est, truth))
                                record["structure_hash"] = scoring.structure_hash(est)
                            sink.write(json.dumps(record, default=str) + "\n")
                            sink.flush()
                        print(
                            name,
                            seed,
                            method,
                            adapter.name,
                            [m.get("status") for m in measures],
                            flush=True,
                        )


if __name__ == "__main__":
    main()
