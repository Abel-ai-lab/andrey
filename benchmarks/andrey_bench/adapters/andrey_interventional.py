"""Run Andrey's interventional methods: one request in, one JSON line per draw out.

``ut_igsp_cpu.py`` and ``igsp_cpu.py`` start this file with ``ANDREY_BENCH_PYTHON`` as
``python andrey_interventional.py <request.npz> <out.jsonl>``. The request is the one the
reference child reads (:mod:`andrey_bench.adapters.gml_interventional`): the pooled rows ``X``,
each row's environment ``env`` (0 the observational one, 1..K the settings), ``known``, the 0/1 mask
of each row's known targets, and ``fits``, a JSON list. Each fit names an ``algorithm``, a
``config``, its keyword arguments ``kwargs``, and ``draws``:

- ``ut_igsp``: ``andrey.ut_igsp(X, env, targets={0: [], k: known_k}, seed=draw, **kwargs)``, the
  settings with no known targets left out of the mapping, which reads them as unknown. A fit's
  ``objective`` sets ``andrey.search.igsp.UT_IGSP_OBJECTIVE`` for its calls. Its targets are
  ``metadata["targets"]`` per setting, and its line keeps the level used, ``metadata["alpha"]``.
- ``igsp``: ``andrey.igsp(X, env, targets={0: [], k: known_k}, seed=draw, **kwargs)``.
- ``gies``: ``andrey.gies(X, env, targets={0: [], k: known_k})``, at its defaults.
- ``grasp``: ``andrey.grasp(X[env == 0], **kwargs)``, the observational rows alone.
- ``cdnod``: ``andrey.cdnod(X, env=env, **kwargs)``; no targets.

Each line holds the graph's endpoint marks. Before the first draw of each fit, one untimed warm-up
runs the same call on the first ``warmup_rows`` rows of each environment. Each draw appends its
line as it finishes, so a killed child keeps the draws it finished.
"""

from __future__ import annotations

import json
import resource
import sys
import time
import traceback
import warnings

import numpy as np


def run_fit(fit: dict, X: np.ndarray, env: np.ndarray, known: dict[int, list[int]], seed: int):
    """One call of the fit's method on ``X``: its marks and, for ``ut_igsp``, its targets."""
    import andrey
    import andrey.search.igsp as igsp

    algorithm, kwargs = fit["algorithm"], dict(fit.get("kwargs", {}))
    if algorithm == "ut_igsp":
        mapping = {0: [], **{k: v for k, v in known.items() if v}}
        previous = igsp.UT_IGSP_OBJECTIVE
        igsp.UT_IGSP_OBJECTIVE = fit.get("objective", previous)
        try:
            out = andrey.ut_igsp(X, env, targets=mapping, seed=seed, **kwargs)
        finally:
            igsp.UT_IGSP_OBJECTIVE = previous
        targets = [sorted(out.metadata["targets"][str(k)]) for k in sorted(known)]
        return {
            "marks": out.structure.to_numpy().astype(int).tolist(),
            "targets": targets,
            "score": out.metadata.get("score"),
            "alpha": out.metadata.get("alpha"),
        }
    if algorithm == "igsp":
        out = andrey.igsp(X, env, targets={0: [], **known}, seed=seed, **kwargs)
        return {
            "marks": out.structure.to_numpy().astype(int).tolist(),
            "score": out.metadata.get("score"),
            "n_contradicting": out.metadata.get("n_contradicting"),
        }
    if algorithm == "gies":
        out = andrey.gies(X, env, targets={0: [], **known}, **kwargs)
        return {"marks": out.structure.to_numpy().astype(int).tolist()}
    if algorithm == "grasp":
        out = andrey.grasp(X[env == 0], **kwargs)
        return {"marks": out.structure.to_numpy().astype(int).tolist()}
    if algorithm == "cdnod":
        out = andrey.cdnod(X, env=env, **kwargs)
        return {"marks": out.structure.to_numpy().astype(int).tolist()}
    raise ValueError(f"unknown algorithm {algorithm!r}")


def fit_request(request, sink) -> None:
    import andrey

    X, env, mask = request["X"], request["env"], request["known"]
    known = {
        k: [int(v) for v in np.flatnonzero(mask[np.flatnonzero(env == k)[0]])]
        for k in range(1, int(env.max()) + 1)
    }
    head = np.concatenate(
        [np.flatnonzero(env == k)[: int(request["warmup_rows"])] for k in np.unique(env)]
    )
    for fit in json.loads(str(request["fits"])):
        warmup_s, warmup_error = None, None
        t0 = time.perf_counter()
        try:
            run_fit(fit, X[head], env[head], known, seed=0)
            warmup_s = time.perf_counter() - t0
        except Exception as exc:  # a warm-up failure is recorded; the timed draws still run
            warmup_error = f"{type(exc).__name__}: {exc}"
        for draw in fit["draws"]:
            line = {"algorithm": fit["algorithm"], "config": fit["config"], "draw": draw}
            t0 = time.perf_counter()
            try:
                result = run_fit(fit, X, env, known, seed=draw)
                line.update(status="ok", wall_s=time.perf_counter() - t0, **result)
            except Exception:
                line.update(status="error", error_text=traceback.format_exc()[-2000:])
            line.update(
                warmup_s=warmup_s,
                warmup_error=warmup_error,
                peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
                versions={"andrey": andrey.__version__},
            )
            sink.write(json.dumps(line) + "\n")
            sink.flush()


def main(argv: list[str]) -> int:
    warnings.simplefilter("ignore")
    request = np.load(argv[1])
    with open(argv[2], "a") as sink:
        fit_request(request, sink)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
