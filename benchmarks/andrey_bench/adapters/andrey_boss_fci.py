"""Run Andrey's BOSS-FCI, GRaSP-FCI, GFCI, and FCI: one request in, one JSON line per fit out.

``boss_grasp_fci_cpu.py`` starts this file with ``ANDREY_BENCH_PYTHON`` as
``python andrey_boss_fci.py <request.npz> <out.jsonl>``. The request holds the data ``X`` and
``fits``, a JSON list; each fit names an ``algorithm`` (``boss_fci``, ``grasp_fci``, ``gfci``, or
``fci``), a ``config``, its ``params`` in the harness's names, a ``trace`` or ``None``, and
``draws``. The names map to Andrey's:

====================================  ==========================================================
harness                               Andrey
====================================  ==========================================================
``alpha``, ``indep_test``             ``alpha``, ``indep_test``
``score_func="local_score_BIC"``      ``score_func="bic"``
``lambda_value``                      ``penalty_discount``: both score ``n log s2 + lambda k log n``
``depth``                             ``max_depth``; ``-1`` or absent is no cap
``grasp_depth``                       ``grasp_depth`` (GRaSP-FCI only)
the draw                              ``seed`` (BOSS-FCI, GRaSP-FCI, and GFCI)
====================================  ==========================================================

Every other setting is the facade's default: the extended-BIC weight 0, one start, the
``"sepsets"`` collider rule, and the order-free possible-d-separation step.

Each line records ``possible_dsep_max``, the most candidates any pair's possible-d-separation search
had (the list ``andrey.constraint.fci._possible_dsep`` returns, both endpoints left out), and
``possible_dsep_searches``, the number of searches; both are ``None`` and ``0`` when the step
searched no pair. The trace ``"no_possible_dsep"`` skips that step's removals for the fit
(``andrey.constraint.fci._remove_by_possible_dsep`` does nothing): the step still turns every kept
edge into ``o-o``, so the colliders and rules run as after a step that removed no edge. Both read
private functions, so each line keeps the Andrey version that ran.

Before the first draw of each fit, one untimed warm-up runs the same call on the first
``warmup_rows`` rows. Each draw appends its line as it finishes, so a killed child keeps the draws
it finished.
"""

from __future__ import annotations

import json
import resource
import sys
import time
import traceback
import warnings
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np

#: The score names of the harness that Andrey calls by another name.
_SCORE = {"local_score_BIC": "bic"}
#: The traces a fit may name.
TRACES = (None, "no_possible_dsep")


def andrey_kwargs(algorithm: str, params: dict[str, Any], draw: int) -> dict[str, Any]:
    """The facade's keyword arguments for ``params`` in the harness's names (module docstring)."""
    depth = params.get("depth")
    kwargs: dict[str, Any] = {
        "alpha": float(params["alpha"]),
        "indep_test": params.get("indep_test", "fisherz"),
        "max_depth": None if depth is None or int(depth) < 0 else int(depth),
    }
    if algorithm != "fci":
        score = params.get("score_func", "local_score_BIC")
        kwargs |= {
            "score_func": _SCORE.get(score, score),
            "penalty_discount": float(params["lambda_value"]),
            "seed": draw,
        }
    if algorithm == "grasp_fci" and "grasp_depth" in params:
        kwargs["grasp_depth"] = int(params["grasp_depth"])
    return kwargs


@contextmanager
def instrumented(trace: str | None) -> Iterator[dict[str, Any]]:
    """Count the possible-d-separation searches and their largest candidate set; apply ``trace``.

    Yields the counts, which fill in as the fit runs; the private functions come back on exit.
    """
    import andrey.constraint.fci as fci

    if trace not in TRACES:
        raise ValueError(f"trace must be one of {TRACES}, not {trace!r}")
    seen: dict[str, Any] = {"possible_dsep_max": None, "possible_dsep_searches": 0}
    candidates, removals = fci._possible_dsep, fci._remove_by_possible_dsep

    def counted(*args, **kwargs):
        found = candidates(*args, **kwargs)
        seen["possible_dsep_searches"] += 1
        seen["possible_dsep_max"] = max(seen["possible_dsep_max"] or 0, len(found))
        return found

    fci._possible_dsep = counted
    if trace == "no_possible_dsep":
        fci._remove_by_possible_dsep = lambda *args, **kwargs: None
    try:
        yield seen
    finally:
        fci._possible_dsep, fci._remove_by_possible_dsep = candidates, removals


def run_fit(fit: dict, X: np.ndarray, draw: int) -> dict[str, Any]:
    """One call of the fit's facade on ``X``: its PAG's endpoint marks and the step's counts."""
    import andrey

    method = getattr(andrey, fit["algorithm"])
    kwargs = andrey_kwargs(fit["algorithm"], fit["params"], draw)
    with instrumented(fit.get("trace")) as seen:
        out = method(X, **kwargs)
    return {"marks": out.structure.to_numpy().astype(int).tolist(), **seen}


def fit_request(request, sink) -> None:
    import andrey

    X = np.asarray(request["X"], dtype=np.float64)
    head = X[: int(request["warmup_rows"])]
    for fit in json.loads(str(request["fits"])):
        warmup_s, warmup_error = None, None
        t0 = time.perf_counter()
        try:
            run_fit(fit, head, draw=0)
            warmup_s = time.perf_counter() - t0
        except Exception as exc:  # a warm-up failure is recorded; the timed draws still run
            warmup_error = f"{type(exc).__name__}: {exc}"
        for draw in fit["draws"]:
            line = {"algorithm": fit["algorithm"], "config": fit["config"], "draw": draw}
            t0 = time.perf_counter()
            try:
                result = run_fit(fit, X, draw)
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
