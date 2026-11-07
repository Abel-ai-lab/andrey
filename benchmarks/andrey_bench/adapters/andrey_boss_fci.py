"""Run Andrey's BOSS-FCI, GRaSP-FCI, GFCI, and FCI: one request in, one JSON line per fit out.

``boss_grasp_fci_cpu.py`` starts this file with ``ANDREY_BENCH_PYTHON`` as
``python andrey_boss_fci.py <request.npz> <out.jsonl>``. The request holds the data ``X`` and
``fits``, a JSON list; each fit names an ``algorithm`` (``boss_fci``, ``grasp_fci``, ``gfci``, or
``fci``), a ``config``, its ``params`` in the harness's names, and ``draws``. The names map to Andrey's:

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
``"sepsets"`` collider rule, and, for GFCI and FCI, the order-free possible-d-separation step.

Each line records ``possible_dsep_max``, the most candidates any pair's possible-d-separation search
had (the list ``andrey.constraint.fci._possible_dsep`` returns, both endpoints left out), and
``possible_dsep_searches``, the number of searches; both are ``None`` and ``0`` when the step
searched no pair, as for BOSS-FCI and GRaSP-FCI, which have no such step. The counts read a private
function, so each line keeps the Andrey version that ran.

Before the first draw of each fit, one untimed warm-up runs the same call on the first
``warmup_rows`` rows. Each call stops at ``cap_s`` seconds (:func:`capped`): a draw past it is a
``"timeout"``, and a warm-up past it makes every draw of its fit one without running it. Each draw
appends its line as it finishes, so a killed child keeps the draws it finished.
"""

from __future__ import annotations

import json
import resource
import signal
import sys
import time
import traceback
import warnings
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np


class FitTimeout(Exception):
    """A call ran past its cap."""


@contextmanager
def capped(seconds: float) -> Iterator[None]:
    """Raise :class:`FitTimeout` in the block once ``seconds`` pass; main thread only.

    The alarm interrupts Python code between bytecodes; ``subprocess.run`` kills its child when
    the exception reaches it.
    """

    def expire(signum, frame):
        raise FitTimeout(f"past the {seconds:g} s cap")

    previous = signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


#: The score names of the harness that Andrey calls by another name.
_SCORE = {"local_score_BIC": "bic"}


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
def instrumented() -> Iterator[dict[str, Any]]:
    """Count the possible-d-separation searches and their largest candidate set.

    Yields the counts, which fill in as the fit runs; the private function comes back on exit.
    """
    import andrey.constraint.fci as fci

    seen: dict[str, Any] = {"possible_dsep_max": None, "possible_dsep_searches": 0}
    candidates = fci._possible_dsep

    def counted(*args, **kwargs):
        found = candidates(*args, **kwargs)
        seen["possible_dsep_searches"] += 1
        seen["possible_dsep_max"] = max(seen["possible_dsep_max"] or 0, len(found))
        return found

    fci._possible_dsep = counted
    try:
        yield seen
    finally:
        fci._possible_dsep = candidates


def run_fit(fit: dict, X: np.ndarray, draw: int) -> dict[str, Any]:
    """One call of the fit's facade on ``X``: its PAG's endpoint marks and the step's counts."""
    import andrey

    method = getattr(andrey, fit["algorithm"])
    kwargs = andrey_kwargs(fit["algorithm"], fit["params"], draw)
    with instrumented() as seen:
        out = method(X, **kwargs)
    return {"marks": out.structure.to_numpy().astype(int).tolist(), **seen}


def fit_request(request, sink) -> None:
    import andrey

    X = np.asarray(request["X"], dtype=np.float64)
    head = X[: int(request["warmup_rows"])]
    cap = float(request["cap_s"])
    for fit in json.loads(str(request["fits"])):
        warmup_s, warmup_error, expired = None, None, False
        t0 = time.perf_counter()
        try:
            with capped(cap):
                run_fit(fit, head, draw=0)
            warmup_s = time.perf_counter() - t0
        except FitTimeout as exc:
            warmup_error, expired = str(exc), True
        except Exception as exc:  # a warm-up failure is recorded; the timed draws still run
            warmup_error = f"{type(exc).__name__}: {exc}"
        for draw in fit["draws"]:
            line = {"algorithm": fit["algorithm"], "config": fit["config"], "draw": draw}
            t0 = time.perf_counter()
            try:
                if expired:
                    raise FitTimeout(f"the warm-up ran past the {cap:g} s cap")
                with capped(cap):
                    result = run_fit(fit, X, draw)
                line.update(status="ok", wall_s=time.perf_counter() - t0, **result)
            except FitTimeout as exc:
                line.update(status="timeout", wall_s=time.perf_counter() - t0, error_text=str(exc))
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
