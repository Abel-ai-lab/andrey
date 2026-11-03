#!/usr/bin/env python
"""Andrey's GnIES and gnies 0.3.3 on six small fixed cases: an agreement table.

Each case draws linear-Gaussian data with ``andrey.data``: an Erdos-Renyi DAG at mean degree 2,
weights of either sign with magnitude in ``[0.5, 1]``, unit noise variance, an observational
environment, and two or three environments that each multiply one variable's noise variance by
``U[5, 10] / U[1, 2]``, with 1,000 rows per environment, standardized on the pooled rows. Both
packages fit the same rows: the reference at every default (``gnies.fit(data)``, the standard BIC),
and Andrey at the same objective, ``penalty_discount=1.0, ebic_gamma=0.0``, and at its defaults.
The table reports whether the two returned the same interventional essential graph and the same
targets, and both totals in Andrey's deviance units (lower is better). The reference runs in its
own interpreter (``start_gnies.sh``, ``ANDREY_BENCH_GNIES_PYTHON``) through ``gnies_worker.py``.

    $ANDREY_BENCH_PYTHON benchmarks/gnies_cases.py

prints the table as Markdown.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

import andrey
from andrey.core import ARROW, TAIL
from andrey.data import SCM, graphs
from andrey.data.functional import linear
from andrey.data.interventions import soft
from andrey.data.noise import gaussian

PYTHON_ENV_VAR = "ANDREY_BENCH_GNIES_PYTHON"
WORKER = Path(__file__).resolve().parent / "gnies_worker.py"
#: ``(variables, interventional environments, seed)`` of each case.
CASES = [(5, 2, 0), (5, 2, 1), (5, 3, 2), (6, 2, 3), (6, 3, 4), (6, 3, 5)]
ROWS = 1000


def draw(d: int, n_targets: int, seed: int, rows: int = ROWS) -> tuple:
    """The rows, each row's environment code, and the targets of one case."""
    scm = SCM(
        graph=graphs.erdos_renyi(d, avg_degree=2.0),
        functional=linear(weight_range=(0.5, 1.0)),
        noise=gaussian(),
    )
    rng = np.random.default_rng([d, n_targets, seed])
    targets = sorted(int(j) for j in rng.choice(d, size=n_targets, replace=False))
    ratios = [rng.uniform(5, 10) / rng.uniform(1, 2) for _ in targets]
    envs = [{}] + [{j: soft(noise_scale=float(np.sqrt(r)))} for j, r in zip(targets, ratios)]
    ds = scm.sample(n=rows, seed=seed, envs=envs, scale="raw")
    X = (ds.data - ds.data.mean(axis=0)) / ds.data.std(axis=0)
    return X, np.asarray(ds.env, dtype=np.intp), targets


def reference(X: np.ndarray, env: np.ndarray, mode: str, **fields: object) -> dict:
    """Run ``gnies_worker.py`` under ``ANDREY_BENCH_GNIES_PYTHON`` and return its answer."""
    python = os.environ.get(PYTHON_ENV_VAR)
    if not python:
        raise RuntimeError(f"set {PYTHON_ENV_VAR}; benchmarks/start_gnies.sh builds it")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "request.npz"
        np.savez(path, X=X, env=env, mode=mode, **fields)
        env_vars = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
        done = subprocess.run(
            [python, "-I", str(WORKER), str(path)],
            capture_output=True,
            text=True,
            check=True,
            env=env_vars,
        )
    return json.loads(done.stdout)


def marks(icpdag: list[list[int]]) -> np.ndarray:
    """The reference's 0/1 matrix as Andrey's endpoint marks: ``A[i, j]`` alone is ``i -> j``,
    both directions an undirected edge."""
    A = np.asarray(icpdag) != 0
    out = np.zeros(A.shape, dtype=np.int8)
    out[A | A.T] = TAIL
    out[A.T & ~A] = ARROW
    return out


def _fit(X: np.ndarray, env: np.ndarray, **kwargs: float) -> andrey.StructureOutput:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", andrey.ExperimentalWarning)
        return andrey.gnies(X, env, **kwargs)


def _targets(out: andrey.StructureOutput) -> list[int]:
    return [int(v) for v, _ in out.metadata["index_edges"]]


def main() -> None:
    rows = [
        "| Case | Variables | Environments | True targets | Reference targets | Andrey targets "
        "(BIC) | Same graph | Same targets | Reference total | Andrey total (BIC) | Andrey "
        "defaults: same graph and targets |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for case, (d, k, seed) in enumerate(CASES, start=1):
        X, env, truth = draw(d, k, seed)
        ref = reference(X, env, "fit")
        bic = _fit(X, env, ebic_gamma=0.0)
        default = _fit(X, env)
        same_graph = bool(np.array_equal(bic.structure.to_numpy(), marks(ref["icpdag"])))
        same_targets = _targets(bic) == ref["targets"]
        defaults_agree = bool(
            np.array_equal(default.structure.to_numpy(), bic.structure.to_numpy())
        ) and _targets(default) == _targets(bic)
        rows.append(
            f"| {case} | {d} | obs + {k} | {truth} | {ref['targets']} | {_targets(bic)} | "
            f"{'yes' if same_graph else 'no'} | {'yes' if same_targets else 'no'} | "
            f"{ref['deviance']:.3f} | {bic.metadata['score']:.3f} | "
            f"{'yes' if defaults_agree else 'no'} |"
        )
    sys.stdout.write("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
