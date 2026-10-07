"""Andrey hill-climbing adapters for greedy score-based search.

Hill climbing moves through DAG space one edge at a time (add, remove, reverse), with an
`O(d^2)` per-step neighborhood. GES searches CPDAG space over subsets of a neighbor set, with an
`O(d^2 * 2^|N|)` neighborhood. Comparing bnlearn's `hc` with Andrey's hill climbing measures
implementation differences; comparing it with GES measures search differences. bnlearn's `tabu`
is also a hill-climbing counterpart.

Andrey canonicalizes its DAG to a CPDAG and declares `output_type="cpdag"`. bnlearn returns
the DAG itself, so compare bnlearn's `mec_shd` and `mec_arrowhead_f1` with this adapter's `shd`
and `arrowhead_f1`.

Two settings keep the pair on the same search:

* `max_iter` caps *accepted moves*, and the shipped 200 binds from about `d = 200` up, where the
  true graph already has that many edges. bnlearn's `max.iter` is unbounded, so this is set high
  enough never to bind and the value is declared in `params`.
* Andrey has no parent cap, so bnlearn's `maxp` is left at its own unbounded default rather than
  pinned to a number Andrey cannot honor.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

_DEFAULT_SCORE_FUNC = "local_score_BIC_from_cov"
#: Deviance units, matching bnlearn's `bic-g` at `k = log(n)/2` and pcalg's `0.5*log(n)`.
_DEFAULT_LAMBDA = 1.0
#: High enough never to bind: the search stops when no move improves. The shipped default is 200.
_DEFAULT_MAX_ITER = 100_000
# Worker count for the parallel variant when `ANDREY_NUM_WORKERS` is unset or unreadable. An explicit
# 1 is kept: the runner pins the core budget there, and the record reports it.
_PARALLEL_FALLBACK_WORKERS = 2


def _andrey_version() -> str:
    from importlib import metadata

    try:
        return metadata.version("andrey")
    except metadata.PackageNotFoundError:  # No distribution metadata in editable/dev checkout.
        import andrey

        return getattr(andrey, "__version__", "unknown")


def _parallel_workers() -> int:
    try:
        n = int(os.environ.get("ANDREY_NUM_WORKERS", "").strip())
    except ValueError:
        return _PARALLEL_FALLBACK_WORKERS
    return n if n == -1 or n >= 1 else _PARALLEL_FALLBACK_WORKERS


@dataclass(frozen=True)
class AndreyHC:
    """Benchmark adapter for one Andrey hill-climbing execution mode."""

    name: str
    backend: str
    mode: str
    package: str = "andrey"
    package_version: str = ""
    algorithm: str = "hc"
    output_type: str = "cpdag"

    def setup(self) -> None:
        """Import Andrey and its hill-climbing implementation before timing.

        The facade loads `andrey.search.hc` on its first call. Explicit setup excludes that import
        from `fit`, including the first timed fit at `--warmup 0`.
        """
        import andrey  # noqa: F401
        from andrey.search.hc import hc  # noqa: F401

    def params(self) -> dict[str, Any]:
        """BIC in deviance units and an accepted-move budget high enough not to limit the search."""
        return {
            "score_func": _DEFAULT_SCORE_FUNC,
            "lambda_value": _DEFAULT_LAMBDA,
            "max_iter": _DEFAULT_MAX_ITER,
        }

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run Andrey hill climbing and return CPDAG endpoint marks. The only timed region.

        Calls the search directly to set `max_iter`. The `andrey.hc` facade fixes it at the
        shipped default, which would limit this search while bnlearn's remains unbounded.
        """
        import andrey
        from andrey.search.hc import hc as _hc

        score_func = params.get("score_func", _DEFAULT_SCORE_FUNC)
        lambda_value = float(params.get("lambda_value", _DEFAULT_LAMBDA))
        max_iter = int(params.get("max_iter", _DEFAULT_MAX_ITER))

        workers = 1 if self.mode == "serial" else _parallel_workers()
        X = np.asarray(data, dtype=np.float64)
        with andrey.config(backend=self.backend, num_workers=workers):
            cpdag, _score = _hc(
                X, score_func=score_func, lambda_value=lambda_value, max_iter=max_iter
            )
        return np.asarray(cpdag.to_numpy())

    def to_structure(self, native: np.ndarray) -> Any:
        """Rebuild a ``GraphStructure`` from endpoint marks; untimed."""
        from andrey import GraphStructure

        return GraphStructure.from_numpy(native, kind=self.output_type)


def andrey_hc_adapters() -> list[AndreyHC]:
    """Return serial and parallel HC contestants.

    Declare ``ANDREY_HC_PARALLEL_MIN_WORK`` for each campaign; see ``benchmarks/README.md``
    for calibration measurements.
    """
    version = _andrey_version()
    specs = (
        ("andrey.hc.numpy.serial", "numpy", "serial"),
        ("andrey.hc.numpy.parallel", "numpy", "parallel"),
    )
    return [
        AndreyHC(name=name, backend=backend, mode=mode, package_version=version)
        for name, backend, mode in specs
    ]
