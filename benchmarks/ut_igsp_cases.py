#!/usr/bin/env python
"""Fixed cases: Andrey's UT-IGSP against the reference implementation on small datasets.

Each case is a dataset from ``andrey.data`` in the design of Squires, Wang, and Uhler (2020,
Section 5.1) at 6 to 10 variables: an Erdos-Renyi DAG with mean degree 1.5, weights uniform on
[0.25, 1] in absolute value, standard normal noise, an observational environment and 5 settings of
5,000 rows each. Each setting shifts the noise mean of one known target and of 0 or 1 unknown ones
by 2, the shift of the authors' simulations. A case is kept only when the reference returns one
interventional essential graph and one target list over 10 seeds of Python's ``random`` and 6
column orders, so its answer does not depend on a draw.

    benchmarks/start_utigsp.sh
    export ANDREY_BENCH_UTIGSP_PYTHON=$PWD/benchmarks/.venv-utigsp/bin/python
    $ANDREY_BENCH_PYTHON benchmarks/ut_igsp_cases.py record
    $ANDREY_BENCH_PYTHON benchmarks/ut_igsp_cases.py compare

``record`` runs the reference on each candidate and writes ``ut_igsp_cases.json`` beside this file:
per case the generator settings, the SHA-256 of the data's float64 bytes, NumPy's version, and the
reference's essential graph (endpoint marks) and targets. The reference runs in its own
interpreter (``child`` mode), which reads only the stored data and imports NumPy and the reference
packages. ``compare`` regenerates each case, checks its hash, fits ``andrey.ut_igsp`` at its
defaults, and prints where the two agree; it exits with 1 when they differ on a case.

Both sides test at ``alpha = 1e-5``, the paper's level. The reference runs at its library
defaults (depth 4, 5 runs); its targets are the known ones and the ones it learns, as the authors
score them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RECORD = HERE / "ut_igsp_cases.json"
ALPHA = 1e-5
N_ROWS = 5000
N_SETTINGS = 5
SHIFT = 2.0
N_CASES = 8
N_SEEDS = 10
N_ORDERS = 6


def candidates():
    """The candidate cases in the order they are tried: seed, variables, unknown targets."""
    for i in range(60):
        yield {"seed": i, "p": 6 + i % 5, "off": (i // 5) % 2}


def dataset(case: dict):
    """The case's data, each row's environment, and the known targets by environment."""
    import andrey.data as data
    from andrey.data.interventions import soft

    p, off, seed = case["p"], case["off"], case["seed"]
    scm = data.SCM(
        graph=data.graphs.erdos_renyi(d=p, avg_degree=1.5),
        functional=data.functional.linear(weight_range=(0.25, 1.0)),
        noise=data.noise.gaussian(),
    )
    designed = np.random.default_rng([seed, 654]).choice(p, size=N_SETTINGS, replace=False)
    envs: list[dict] = [{}]
    for k, target in enumerate(designed):
        others = [v for v in range(p) if v != target]
        unknown = np.random.default_rng([seed, 654, off, k]).choice(others, off, replace=False)
        envs.append({int(v): soft(shift=SHIFT) for v in (target, *unknown)})
    ds = scm.sample(n=[N_ROWS] * (N_SETTINGS + 1), seed=seed, scale="raw", envs=envs)
    known = {0: [], **{k + 1: [int(t)] for k, t in enumerate(designed)}}
    return ds, known


def digest(X: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(X, dtype=np.float64).tobytes()).hexdigest()


def essential(p: int, arcs: list, targets: list[list[int]]) -> list[list[int]]:
    """The endpoint marks of the interventional essential graph of the DAG ``arcs`` under the
    observational setting and ``targets``."""
    from andrey.core import ARROW, TAIL
    from andrey.search.gies import cut_pairs, replace_unprotected

    dag = np.zeros((p, p), dtype=np.int8)
    for i, j in arcs:
        dag[i, j], dag[j, i] = TAIL, ARROW
    family = np.zeros((len(targets) + 1, p), dtype=bool)
    for row, variables in enumerate(targets, start=1):
        family[row, variables] = True
    return replace_unprotected(dag, cut_pairs(family)).tolist()


def reference_runs(python: str, ds, known: dict) -> list[dict]:
    """Run the reference child on the case's stored data; its runs over seeds and column orders."""
    p = ds.data.shape[1]
    mask = np.zeros((N_SETTINGS + 1, p), dtype=np.int8)
    for k, variables in known.items():
        mask[k, variables] = 1
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "case.npz"
        np.savez(path, X=ds.data, env=ds.env, known=mask)
        done = subprocess.run(
            [python, str(Path(__file__).resolve()), "child", str(path)],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
        )
    return json.loads(done.stdout)


def record(python: str) -> None:
    cases = []
    for case in candidates():
        ds, known = dataset(case)
        p = ds.data.shape[1]
        runs = reference_runs(python, ds, known)
        answers = {json.dumps([essential(p, r["arcs"], r["targets"]), r["targets"]]) for r in runs}
        print(f"seed {case['seed']} p {p} off {case['off']}: {len(answers)} distinct answers")
        if len(answers) != 1:
            continue
        marks, targets = json.loads(answers.pop())
        cases.append(
            {
                **case,
                "n": N_ROWS,
                "settings": N_SETTINGS,
                "shift": SHIFT,
                "sha256": digest(ds.data),
                "numpy": np.__version__,
                "marks": marks,
                "targets": targets,
            }
        )
        if len(cases) == N_CASES:
            break
    lines = ",\n".join(json.dumps(case) for case in cases)
    RECORD.write_text(f'{{"alpha": {ALPHA}, "cases": [\n{lines}\n]}}\n')
    print(f"wrote {len(cases)} cases to {RECORD.name}")


def compare() -> list[dict]:
    """Fit Andrey on every recorded case; return one row per case."""
    import warnings

    import andrey

    rows = []
    for case in json.loads(RECORD.read_text())["cases"]:
        ds, known = dataset(case)
        if digest(ds.data) != case["sha256"]:
            raise ValueError(
                f"case seed {case['seed']}: the data differ from the recorded case; re-record"
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", andrey.ExperimentalWarning)
            out = andrey.ut_igsp(ds.data, ds.env, targets=known, alpha=ALPHA)
        marks = out.structure.to_numpy()
        reference = np.array(case["marks"])
        upper = np.triu_indices(marks.shape[0], 1)
        differ = (marks[upper] != reference[upper]) | (marks.T[upper] != reference.T[upper])
        targets = [out.metadata["targets"][str(k)] for k in range(1, N_SETTINGS + 1)]
        rows.append(
            {
                "seed": case["seed"],
                "p": case["p"],
                "off": case["off"],
                "shd_between": int(differ.sum()),
                "same_targets": targets == case["targets"],
            }
        )
    return rows


def child(path: str) -> None:
    """Run the reference on the stored case over seeds and column orders; print its runs."""
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
    X, env, mask = stored["X"], stored["env"], stored["known"]
    p, settings = X.shape[1], int(env.max())
    orders = [np.arange(p)] + [
        np.random.default_rng([654, k]).permutation(p) for k in range(N_ORDERS - 1)
    ]
    runs = []
    for perm in orders:  # column c of the permuted data is column perm[c] of the stored data
        position = np.argsort(perm)
        Xp = X[:, perm]
        obs, ivs = Xp[env == 0], [Xp[env == k] for k in range(1, settings + 1)]
        known = [
            sorted(int(position[v]) for v in np.flatnonzero(mask[k]))
            for k in range(1, settings + 1)
        ]
        for seed in range(N_SEEDS):
            random.seed(seed)
            np.random.seed(seed)
            ci = MemoizedCI_Tester(
                partial_correlation_test, partial_correlation_suffstat(obs), alpha=ALPHA
            )
            inv = MemoizedInvarianceTester(
                gauss_invariance_test, gauss_invariance_suffstat(obs, ivs), alpha=ALPHA
            )
            dag, learned = gml.unknown_target_igsp(
                [{"known_interventions": k} for k in known], set(range(p)), ci, inv
            )
            runs.append(
                {
                    "arcs": sorted((int(perm[i]), int(perm[j])) for i, j in dag.arcs),
                    "targets": [
                        sorted(int(perm[v]) for v in set(k) | set(t))
                        for k, t in zip(known, learned, strict=True)
                    ],
                }
            )
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
            f"seed {row['seed']:>2} p {row['p']:>2} off {row['off']}: "
            f"SHD between {row['shd_between']}, same targets {row['same_targets']}"
        )
    agree = all(row["shd_between"] == 0 and row["same_targets"] for row in rows)
    print(f"{sum(r['shd_between'] == 0 and r['same_targets'] for r in rows)} of {len(rows)} agree")
    return 0 if agree else 1


if __name__ == "__main__":
    sys.exit(main())
