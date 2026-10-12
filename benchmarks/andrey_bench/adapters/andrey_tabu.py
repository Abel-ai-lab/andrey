"""Andrey tabu-search adapter for greedy score-based search.

Tabu search is hill climbing that keeps going past a local optimum: when no move improves the score
it takes the best non-improving move, never returning to one of the `tabu_length` DAGs it visited
last, and it returns the best DAG it visited. bnlearn's `tabu` is the counterpart; Andrey's `hc`
measures what the extra search buys.

Andrey canonicalizes its DAG to a CPDAG and declares `output_type="cpdag"`. bnlearn returns
the DAG itself, so compare bnlearn's `mec_shd` and `mec_arrowhead_f1` with this adapter's `shd`
and `arrowhead_f1`.

Three settings keep the pair on the same search:

* `tabu_length=10` and `max_tabu=10` are bnlearn's `tabu` and `max.tabu` defaults. bnlearn 5.2.1
  stops after `tabu` non-improving moves whatever `max.tabu` is, so the two must stay equal for the
  pair to stop by the same rule.
* `max_iter` caps all moves, improving or not. The shipped default sets no limit, and bnlearn's
  `max.iter` is unbounded; this is set high enough never to bind and the value is declared in
  `params`.
* Andrey has no parent cap, so bnlearn's `maxp` stays at its unbounded default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from andrey_bench.adapters.andrey_hc import _andrey_version

_DEFAULT_SCORE_FUNC = "local_score_BIC_from_cov"
#: Deviance units, bnlearn's `bic-g` at `k = log(n)/2` up to its residual variance, which
#: divides by n - p - 1 where Andrey divides by n: an O(p^2 / n) difference per node.
_DEFAULT_LAMBDA = 1.0
#: bnlearn's defaults: a tabu list of 10 DAGs, and 10 non-improving moves without a new best.
_DEFAULT_TABU_LENGTH = 10
_DEFAULT_MAX_TABU = 10
#: High enough never to bind: the search stops by the tabu rule. The shipped default is None.
_DEFAULT_MAX_ITER = 100_000


@dataclass(frozen=True)
class AndreyTabu:
    """Benchmark adapter for Andrey tabu search."""

    name: str
    backend: str
    mode: str
    package: str = "andrey"
    package_version: str = ""
    algorithm: str = "tabu"
    output_type: str = "cpdag"

    def setup(self) -> None:
        """Import Andrey and its tabu search before timing, outside `fit`."""
        import andrey  # noqa: F401
        from andrey.search.tabu import tabu  # noqa: F401

    def params(self) -> dict[str, Any]:
        """BIC in deviance units, bnlearn's tabu settings, and a move budget that does not bind."""
        return {
            "score_func": _DEFAULT_SCORE_FUNC,
            "lambda_value": _DEFAULT_LAMBDA,
            "tabu_length": _DEFAULT_TABU_LENGTH,
            "max_tabu": _DEFAULT_MAX_TABU,
            "max_iter": _DEFAULT_MAX_ITER,
        }

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run Andrey tabu search and return CPDAG endpoint marks. The only timed region.

        Calls the search directly, as the hill-climbing adapter does, so the facade's experimental
        warning and input checks stay outside the timing.
        """
        import andrey
        from andrey.search.tabu import tabu as _tabu

        X = np.asarray(data, dtype=np.float64)
        with andrey.config(backend=self.backend):
            cpdag, _score = _tabu(
                X,
                score_func=params.get("score_func", _DEFAULT_SCORE_FUNC),
                lambda_value=float(params.get("lambda_value", _DEFAULT_LAMBDA)),
                tabu_length=int(params.get("tabu_length", _DEFAULT_TABU_LENGTH)),
                max_tabu=int(params.get("max_tabu", _DEFAULT_MAX_TABU)),
                max_iter=int(params.get("max_iter", _DEFAULT_MAX_ITER)),
            )
        return np.asarray(cpdag.to_numpy())

    def to_structure(self, native: np.ndarray) -> Any:
        """Rebuild a ``GraphStructure`` from endpoint marks; untimed."""
        from andrey import GraphStructure

        return GraphStructure.from_numpy(native, kind=self.output_type)


def andrey_tabu_adapters() -> list[AndreyTabu]:
    """Return the tabu contestant."""
    return [
        AndreyTabu(
            name="andrey.tabu.numpy.serial",
            backend="numpy",
            mode="serial",
            package_version=_andrey_version(),
        )
    ]
