"""Andrey XGES benchmark adapter.

Wraps ``andrey.search.xges.xges``, the search behind ``andrey.xges``. ``extended_search=False`` is
XGES without the extended search, so the one class covers both arms. ``lambda_value`` is the xges
package's ``alpha``: the two scores differ by a factor of ``-2`` and by terms that cancel in every
move, so equal values search under equal penalties.

**One BLAS thread.** The search runs inside ``threadpool_limits(limits=1)`` so that every score,
and every tie between moves, is reproducible. The record's ``threads`` is the budget the runner
allotted; this solution uses one thread of it whatever that budget is.

**No parent cap.** Like the xges package, XGES has no ``maxP``; ``params`` records
``max_parents: None``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from andrey_bench.adapters.andrey_ges import _andrey_version

_SCORE_FUNC = "local_score_BIC"


@dataclass(frozen=True)
class AndreyXGES:
    """Benchmark adapter for Andrey's XGES; ``extended_search=False`` drops the extended search.

    Satisfies :class:`andrey_bench.contracts.SolutionAdapter`. Each search setting needs a
    ``name`` of its own, because ``params`` are excluded from ``run_id``.
    """

    name: str = "andrey.xges.numpy"
    package: str = "andrey"
    package_version: str = field(default_factory=_andrey_version)
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "xges"
    output_type: str = "cpdag"
    extended_search: bool = True
    lambda_value: float = 1.0

    def setup(self) -> None:
        """Import Andrey and the search before timing; see `contracts.SetupAdapter`."""
        import andrey  # noqa: F401
        from andrey.search.xges import xges  # noqa: F401

    def params(self) -> dict[str, Any]:
        """The search settings, including the parent cap XGES does not have."""
        return {
            "score_func": _SCORE_FUNC,
            "lambda_value": float(self.lambda_value),
            "extended_search": bool(self.extended_search),
            "max_parents": None,
        }

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Return the CPDAG endpoint array. The only timed region."""
        from andrey.search.xges import xges

        cpdag, _score = xges(
            np.asarray(data, dtype=np.float64),
            score_func=params.get("score_func", _SCORE_FUNC),
            lambda_value=float(params.get("lambda_value", self.lambda_value)),
            extended_search=bool(params.get("extended_search", self.extended_search)),
        )
        return np.asarray(cpdag.to_numpy())

    def to_structure(self, native: np.ndarray) -> Any:
        """Rebuild a ``GraphStructure`` from endpoint marks; untimed."""
        from andrey import GraphStructure

        return GraphStructure.from_numpy(native, kind=self.output_type)
