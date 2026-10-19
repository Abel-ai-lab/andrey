"""Answer conditional-independence queries with a reference package's test, in its own environment.

    python ci_query_worker.py <package> <test> <draws.npz> <out.jsonl>

``draws.npz`` holds one ``(n, 2 + k)`` array per draw, named ``d0``, ``d1``, ...; each query tests
column 0 against column 1 given the others. One JSON line per draw: ``p``, ``statistic``, and
``seconds``, the time of the test alone (building it and answering the query), not of the file
input and output. The worker imports NumPy and the reference package only, so it runs in that
package's environment (tigramite's is GPL-licensed and kept out of bench-env). The settings match
Andrey's defaults:

- tigramite ``cmiknn``: ``CMIknn(knn=0.1, shuffle_neighbors=5, sig_samples=500,
  significance="shuffle_test", transform="ranks", sig_blocklength=1, workers=1)``;
- tigramite ``gpdc``: ``GPDC(significance="analytic")``;
- causal-learn ``kci``: ``CIT(data, "kci")``;
- causal-learn ``rcot``: ``CIT(data, "rcit", rcit=False, num_f=100, num_f2=5, approx="hbe")``;
- causal-learn ``fastkci``: ``CIT(data, "fastkci")``, a different algorithm, for time and accuracy.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np


def _tigramite(test: str):
    """A function ``data -> (statistic, p)`` running tigramite's ``test`` on columns 0 and 1."""
    if test == "cmiknn":
        from tigramite.independence_tests.cmiknn import CMIknn

        ci = CMIknn(
            knn=0.1,
            shuffle_neighbors=5,
            sig_samples=500,
            significance="shuffle_test",
            transform="ranks",
            sig_blocklength=1,
            workers=1,
        )
    elif test == "gpdc":
        from tigramite.independence_tests.gpdc import GPDC

        ci = GPDC(significance="analytic")
    else:
        raise ValueError(f"tigramite has no test {test!r} here")

    def run(data: np.ndarray) -> tuple[float, float]:
        z = data[:, 2:] if data.shape[1] > 2 else None
        statistic, p = ci.run_test_raw(data[:, :1], data[:, 1:2], z)[:2]
        return float(statistic), float(p)

    return run


def _causal_learn(test: str):
    """A function ``data -> (statistic, p)`` running causal-learn's ``test`` on columns 0 and 1."""
    from causallearn.utils.cit import CIT

    settings = {
        "kci": ("kci", {}),
        "rcot": ("rcit", {"rcit": False, "num_f": 100, "num_f2": 5, "approx": "hbe"}),
        "fastkci": ("fastkci", {}),
    }[test]

    def run(data: np.ndarray) -> tuple[float, float]:
        method, kwargs = settings
        p = CIT(data, method, **kwargs)(0, 1, list(range(2, data.shape[1])))
        return float("nan"), float(p)

    return run


def main(package: str, test: str, draws_path: str, out_path: str) -> None:
    run = {"tigramite": _tigramite, "causal-learn": _causal_learn}[package](test)
    draws = np.load(draws_path)
    with open(out_path, "w") as out:
        for i in range(len(draws.files)):
            data = draws[f"d{i}"]
            start = time.perf_counter()
            statistic, p = run(data)
            seconds = time.perf_counter() - start
            out.write(json.dumps({"draw": i, "p": p, "statistic": statistic, "seconds": seconds}))
            out.write("\n")
            out.flush()


if __name__ == "__main__":
    main(*sys.argv[1:5])
