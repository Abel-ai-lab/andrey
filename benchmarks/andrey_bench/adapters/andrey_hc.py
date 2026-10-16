"""Andrey hill-climbing adapter for greedy score-based search.

Hill climbing moves through DAG space one edge at a time (add, remove, reverse), with an
`O(d^2)` per-step neighborhood. GES searches CPDAG space over subsets of a neighbor set, with an
`O(d^2 * 2^|N|)` neighborhood. Comparing bnlearn's `hc` with Andrey's hill climbing measures
implementation differences; comparing it with GES measures search differences. bnlearn's `tabu`
is also a hill-climbing counterpart.

Andrey canonicalizes its DAG to a CPDAG and declares `output_type="cpdag"`. bnlearn returns
the DAG itself, so compare bnlearn's `mec_shd` and `mec_arrowhead_f1` with this adapter's `shd`
and `arrowhead_f1`.

Two settings keep the pair on the same search:

* `max_iter` caps *accepted moves*. The shipped default sets no limit, and bnlearn's `max.iter` is
  unbounded; this is set high enough never to bind and the value is declared in `params`.
* Andrey has no parent cap, so bnlearn's `maxp` is left at its own unbounded default rather than
  pinned to a number Andrey cannot honor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

_DEFAULT_SCORE_FUNC = "local_score_BIC_from_cov"
#: Deviance units: the penalty of bnlearn's `bic-g` at `k = log(n)/2` and pcalg's `0.5*log(n)`.
#: `bic-g` divides the residual sum of squares by n - p - 1, so its likelihood differs.
_DEFAULT_LAMBDA = 1.0
#: High enough never to bind: the search stops when no move improves. The shipped default is None.
_DEFAULT_MAX_ITER = 100_000


def _andrey_version() -> str:
    from importlib import metadata

    try:
        return metadata.version("andrey")
    except metadata.PackageNotFoundError:  # No distribution metadata in editable/dev checkout.
        import andrey

        return getattr(andrey, "__version__", "unknown")


@dataclass(frozen=True)
class AndreyHC:
    """Benchmark adapter for Andrey hill climbing."""

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

        Calls the search directly with the declared `max_iter`, high enough never to bind, so the
        search is as unbounded as bnlearn's.
        """
        import andrey
        from andrey.search.hc import hc as _hc

        score_func = params.get("score_func", _DEFAULT_SCORE_FUNC)
        lambda_value = float(params.get("lambda_value", _DEFAULT_LAMBDA))
        max_iter = int(params.get("max_iter", _DEFAULT_MAX_ITER))

        X = np.asarray(data, dtype=np.float64)
        with andrey.config(backend=self.backend):
            cpdag, _score = _hc(
                X, score_func=score_func, lambda_value=lambda_value, max_iter=max_iter
            )
        return np.asarray(cpdag.to_numpy())

    def to_structure(self, native: np.ndarray) -> Any:
        """Rebuild a ``GraphStructure`` from endpoint marks; untimed."""
        from andrey import GraphStructure

        return GraphStructure.from_numpy(native, kind=self.output_type)


def andrey_hc_adapters() -> list[AndreyHC]:
    """Return the HC contestant."""
    return [
        AndreyHC(
            name="andrey.hc.numpy.serial",
            backend="numpy",
            mode="serial",
            package_version=_andrey_version(),
        )
    ]
