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

import sys

import numpy as np

import andrey
from andrey.data import SCM, graphs
from andrey.data.functional import linear
from andrey.data.interventions import soft
from andrey.data.noise import gaussian
from andrey_bench.adapters.gnies_gnies import marks, request

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
    """Run one reference request and return its answer; raises unless it finished."""
    answer = request(X, env, mode, **fields)
    if answer["status"] != "ok":
        raise RuntimeError(f"gnies {mode}: {answer['status']}: {answer.get('error_text')}")
    return answer


def _targets(out: andrey.StructureOutput) -> list[int]:
    return [int(v) for v in out.metadata["changing_variables"]]


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
        bic = andrey.gnies(X, env, ebic_gamma=0.0)
        default = andrey.gnies(X, env)
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
