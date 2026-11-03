"""Run gnies 0.3.3 in its own environment: read one ``.npz`` request, print one JSON answer.

``gnies_cases.py`` and ``tests/test_gnies_lane.py`` start it with ``ANDREY_BENCH_GNIES_PYTHON``,
the interpreter ``start_gnies.sh`` builds; it imports NumPy and gnies alone. The request holds the
pooled rows ``X``, each row's environment ``env`` (codes ``0..K-1``), and ``mode``:

- ``fit``: ``gnies.fit`` with every default (BIC, greedy, forward and backward phases);
- ``known``: ``gnies.fit(known_targets=I, phases=[])``, one inner search at the targets ``I``;
- ``score``: ``GnIESScore(data, I, max_iter=M).full_score(A)`` for each ``M`` in ``iters``, the DAG
  ``A`` given as a 0/1 matrix with ``A[i, j] = 1`` for ``i -> j``.

Every score is converted to Andrey's deviance units, ``-2 * score - N p - p log N`` (lower is
better). A fit answers ``{"deviance", "icpdag", "targets", "seconds"}``, where ``icpdag[i][j] != 0``
alone means ``i -> j`` and both directions mean ``i - j``; ``score`` answers ``{"deviance": [...]}``.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np


def main(path: str) -> None:
    import gnies
    from gnies.scores.gnies_score import GnIESScore

    request = np.load(path)
    X, env, mode = request["X"], request["env"], str(request["mode"])
    data = [X[env == k] for k in range(int(env.max()) + 1)]
    N, p = X.shape

    def deviance(score: float) -> float:
        return float(-2.0 * score - N * p - p * np.log(N))

    if mode == "score":
        targets = {int(j) for j in request["I"]}
        A = request["A"]
        values = [
            deviance(GnIESScore(data, targets, max_iter=int(m)).full_score(A))
            for m in request["iters"]
        ]
        print(json.dumps({"deviance": values}))
        return
    start = time.perf_counter()
    if mode == "fit":
        score, icpdag, targets = gnies.fit(data)
    else:
        known = {int(j) for j in request["I"]}
        score, icpdag, targets = gnies.fit(data, known_targets=known, phases=[])
    seconds = time.perf_counter() - start
    answer = {
        "deviance": deviance(score),
        "icpdag": np.asarray(icpdag).astype(int).tolist(),
        "targets": sorted(int(j) for j in targets),
        "seconds": seconds,
    }
    print(json.dumps(answer))


if __name__ == "__main__":
    main(sys.argv[1])
