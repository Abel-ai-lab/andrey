#!/usr/bin/env python
"""Fixed cases: Andrey's IGSP and the reference implementation against the true graph.

Each case is a dataset from ``andrey.data`` in the single-target design of Yang, Katcoff, and Uhler
(2018, Section 5) at 6 to 8 variables: an Erdos-Renyi DAG with mean degree 1.5, weights uniform on
[0.25, 1] in absolute value, standard normal noise, an observational environment and 3 settings of
100,000 rows each, each on one distinct variable. Each setting cuts its target from its causes and
gives it the noise N(0, 0.2), the intervention of Wang et al. (2017, Section 5.1), which changes
the distribution of every target, roots included. A case is kept only when the reference returns
the true interventional essential graph on 20 of 20 seeds of Python's ``random`` and NumPy's
global generator, so the truth is an answer the reference finds whatever its draws. The
comparison is with the truth, not with the reference's other answers: the two searches differ by
design (the visiting order, the backtracking, and the levels of their tests).

    benchmarks/start_utigsp.sh
    export ANDREY_BENCH_UTIGSP_PYTHON=$PWD/benchmarks/.venv-utigsp/bin/python
    $ANDREY_BENCH_PYTHON benchmarks/igsp_cases.py record
    $ANDREY_BENCH_PYTHON benchmarks/igsp_cases.py compare

``record`` runs the reference on each candidate and writes ``igsp_cases.json`` beside this file:
per case the generator settings, each column's mean and standard deviation, NumPy's version, and
the true essential graph (endpoint marks). The moments, not the data's bytes, identify the data:
the generator's last bits differ between CPUs. The reference runs in its own interpreter
(``child`` mode), which reads only the stored data and imports NumPy and the reference packages.
``compare`` regenerates each case, checks its moments to a relative 1e-9, fits ``andrey.igsp`` on
seeds 0-4, and prints where it returns the truth; it exits with 1 when a fit does not.

Both sides test at ``alpha = 1e-3``, Andrey's default and a level of the authors' sweep. The
reference runs at its library defaults (depth 4, 5 runs, the start from its threshold graph).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RECORD = HERE / "igsp_cases.json"
ALPHA = 1e-3
N_ROWS = 100_000
N_SETTINGS = 3
VARIANCE = 0.2
N_CASES = 10
N_SEEDS = 20
ANDREY_SEEDS = range(5)


def candidates():
    """The candidate cases in the order they are tried: seed and variables."""
    for i in range(60):
        yield {"seed": i, "p": 6 + i % 3}


def dataset(case: dict):
    """The case's data, each row's environment, and the targets by environment."""
    import andrey.data as data
    from andrey.data.interventions import hard

    p, seed = case["p"], case["seed"]
    scm = data.SCM(
        graph=data.graphs.erdos_renyi(d=p, avg_degree=1.5),
        functional=data.functional.linear(weight_range=(0.25, 1.0)),
        noise=data.noise.gaussian(),
    )
    chosen = np.random.default_rng([seed, 655]).choice(p, size=N_SETTINGS, replace=False)
    envs = [{}, *({int(v): hard(0.0, sd=VARIANCE**0.5)} for v in chosen)]
    ds = scm.sample(n=[N_ROWS] * (N_SETTINGS + 1), seed=seed, scale="raw", envs=envs)
    return ds, {0: [], **{k + 1: [int(v)] for k, v in enumerate(chosen)}}


def moments(X: np.ndarray) -> list[float]:
    """Each column's mean and standard deviation, which identify the data on any CPU."""
    return np.concatenate([X.mean(axis=0), X.std(axis=0)]).tolist()


def essential(dag: np.ndarray, targets: dict[int, list[int]]) -> list[list[int]]:
    """The endpoint marks of the interventional essential graph of the DAG's marks ``dag`` under
    the observational setting and each setting's targets."""
    from andrey.search.gies import cut_pairs, replace_unprotected

    family = np.zeros((len(targets), dag.shape[0]), dtype=bool)
    for k, variables in targets.items():
        family[k, variables] = True
    return replace_unprotected(dag.astype(np.int8), cut_pairs(family)).tolist()


def reference_runs(python: str, ds, targets: dict) -> list[list]:
    """Run the reference child on the case's stored data; the arcs it returns on each seed."""
    p = ds.data.shape[1]
    mask = np.zeros((N_SETTINGS + 1, p), dtype=np.int8)
    for k, variables in targets.items():
        mask[k, variables] = 1
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "case.npz"
        np.savez(path, X=ds.data, env=ds.env, targets=mask)
        done = subprocess.run(
            [python, str(Path(__file__).resolve()), "child", str(path)],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
        )
    return json.loads(done.stdout)


def record(python: str) -> None:
    from andrey.core import ARROW, TAIL

    cases = []
    for case in candidates():
        ds, targets = dataset(case)
        p = ds.data.shape[1]
        truth = essential(ds.graph.to_numpy(), targets)
        found = 0
        for arcs in reference_runs(python, ds, targets):
            dag = np.zeros((p, p), dtype=np.int8)
            for i, j in arcs:
                dag[i, j], dag[j, i] = TAIL, ARROW
            found += essential(dag, targets) == truth
        print(f"seed {case['seed']} p {p}: the reference finds the truth on {found} of {N_SEEDS}")
        if found < N_SEEDS:
            continue
        cases.append(
            {
                **case,
                "n": N_ROWS,
                "settings": N_SETTINGS,
                "variance": VARIANCE,
                "moments": moments(ds.data),
                "numpy": np.__version__,
                "marks": truth,
            }
        )
        if len(cases) == N_CASES:
            break
    lines = ",\n".join(json.dumps(case) for case in cases)
    RECORD.write_text(f'{{"alpha": {ALPHA}, "cases": [\n{lines}\n]}}\n')
    print(f"wrote {len(cases)} cases to {RECORD.name}")


def compare() -> list[dict]:
    """Fit Andrey on every recorded case and seed; return one row per case."""
    import warnings

    import andrey

    rows = []
    for case in json.loads(RECORD.read_text())["cases"]:
        ds, targets = dataset(case)
        if not np.allclose(moments(ds.data), case["moments"], rtol=1e-9, atol=1e-12):
            raise ValueError(
                f"case seed {case['seed']}: the data differ from the recorded case; re-record"
            )
        found = 0
        for seed in ANDREY_SEEDS:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", andrey.ExperimentalWarning)
                out = andrey.igsp(ds.data, ds.env, targets=targets, alpha=ALPHA, seed=seed)
            found += out.structure.to_numpy().tolist() == case["marks"]
        rows.append({"seed": case["seed"], "p": case["p"], "found": found})
    return rows


def child(path: str) -> None:
    """Run the reference on the stored case over seeds; print the arcs of each run."""
    import random

    import graphical_model_learning as gml
    from conditional_independence import (
        MemoizedCI_Tester,
        MemoizedInvarianceTester,
        gauss_invariance_suffstat,
        gauss_invariance_test,
        partial_correlation_suffstat,
        partial_correlation_test,
    )

    stored = np.load(path)
    X, env, mask = stored["X"], stored["env"], stored["targets"]
    p, settings = X.shape[1], int(env.max())
    obs, ivs = X[env == 0], [X[env == k] for k in range(1, settings + 1)]
    setting_list = [
        {"interventions": [int(v) for v in np.flatnonzero(mask[k])]} for k in range(1, settings + 1)
    ]
    runs = []
    for seed in range(N_SEEDS):
        random.seed(seed)
        np.random.seed(seed)
        ci = MemoizedCI_Tester(
            partial_correlation_test, partial_correlation_suffstat(obs), alpha=ALPHA
        )
        inv = MemoizedInvarianceTester(
            gauss_invariance_test, gauss_invariance_suffstat(obs, ivs), alpha=ALPHA
        )
        dag = gml.igsp(setting_list, set(range(p)), ci, inv)
        runs.append(sorted((int(i), int(j)) for i, j in dag.arcs))
    print(json.dumps(runs))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("mode", choices=("record", "compare", "child"))
    parser.add_argument("path", nargs="?", help="child mode: the stored case")
    args = parser.parse_args()
    if args.mode == "child":
        child(args.path)
        return 0
    if args.mode == "record":
        python = os.environ.get("ANDREY_BENCH_UTIGSP_PYTHON")
        if not python:
            sys.exit(
                "set ANDREY_BENCH_UTIGSP_PYTHON to the reference interpreter (start_utigsp.sh)"
            )
        record(python)
        return 0
    rows = compare()
    for row in rows:
        print(
            f"seed {row['seed']:>2} p {row['p']}: Andrey finds the truth on {row['found']} of "
            f"{len(ANDREY_SEEDS)} seeds"
        )
    agree = all(row["found"] == len(ANDREY_SEEDS) for row in rows)
    print(f"{sum(row['found'] == len(ANDREY_SEEDS) for row in rows)} of {len(rows)} cases agree")
    return 0 if agree else 1


if __name__ == "__main__":
    sys.exit(main())
