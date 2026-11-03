"""Run gnies 0.3.3 in its own environment: read one ``.npz`` request, print one JSON answer.

:mod:`andrey_bench.adapters.gnies_gnies` starts it with ``ANDREY_BENCH_GNIES_PYTHON``, the
interpreter ``start_gnies.sh`` builds; it imports NumPy and gnies alone. The request holds the
pooled rows ``X``, each row's environment ``env`` (codes ``0..K-1``), and ``mode``:

- ``fit``: ``gnies.fit`` with every default (BIC, greedy, forward and backward phases);
- ``known``: ``gnies.fit(known_targets=I, phases=[])``, one inner search at the targets ``I``;
- ``score``: ``GnIESScore(data, I, max_iter=M).full_score(A)`` for each ``M`` in ``iters``, the DAG
  ``A`` given as a 0/1 matrix with ``A[i, j] = 1`` for ``i -> j``.

Every score is converted to Andrey's deviance units, ``-2 * score - N p - p log N`` (lower is
better). A fit answers ``{"deviance", "icpdag", "targets", "seconds", "counts"}``, where
``icpdag[i][j] != 0`` alone means ``i -> j`` and both directions mean ``i - j``, and ``seconds`` is
the wall time of the fit call alone. A fit request may also carry the true DAG ``A_true`` and
targets ``I_true``: the answer then adds ``truth_deviance``, the package's own score of the truth at
its default 10 iterations. ``score`` answers ``{"deviance": [...]}``.

``counts`` come from a wrapper around the score's local computation, which runs once per family and
set of targets: ``nan``, ``posinf``, ``neginf`` (non-finite local scores), ``target_one_dof``
(target families with ``min_e n_e - 2`` parents, one residual degree of freedom left in the
smallest environment), and ``target_saturated`` (target families with ``min_e n_e - 1`` parents or
more). A fit with a NaN local score or a non-finite total exits with an error, since its result is
not a fit of the model.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np


def _count_local_scores(counts: dict[str, int]) -> None:
    """Wrap ``GnIESScore._compute_local_score`` so each computed family updates ``counts``."""
    from gnies.scores.gnies_score import GnIESScore

    compute = GnIESScore._compute_local_score

    def counted(self, j, pa):
        value = compute(self, j, pa)
        if np.isnan(value):
            counts["nan"] += 1
        elif value == np.inf:
            counts["posinf"] += 1
        elif value == -np.inf:
            counts["neginf"] += 1
        if j in self.I:
            extra = len(pa) - (int(min(self.n_obs)) - 2)
            if extra == 0:
                counts["target_one_dof"] += 1
            elif extra > 0:
                counts["target_saturated"] += 1
        return value

    GnIESScore._compute_local_score = counted


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
    counts = dict.fromkeys(("nan", "posinf", "neginf", "target_one_dof", "target_saturated"), 0)
    _count_local_scores(counts)
    start = time.perf_counter()
    if mode == "fit":
        score, icpdag, targets = gnies.fit(data)
    else:
        known = {int(j) for j in request["I"]}
        score, icpdag, targets = gnies.fit(data, known_targets=known, phases=[])
    seconds = time.perf_counter() - start
    if counts["nan"] or not np.isfinite(score):
        raise RuntimeError(f"not a fit of the model: total {score}, local score counts {counts}")
    answer = {
        "deviance": deviance(score),
        "icpdag": np.asarray(icpdag).astype(int).tolist(),
        "targets": sorted(int(j) for j in targets),
        "seconds": seconds,
        "counts": counts,
    }
    if "A_true" in request.files:
        truth = GnIESScore(data, {int(j) for j in request["I_true"]})
        answer["truth_deviance"] = deviance(truth.full_score(request["A_true"]))
    print(json.dumps(answer))


if __name__ == "__main__":
    main(sys.argv[1])
