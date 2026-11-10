"""Andrey SCORE solution adapter.

Wraps ``andrey.score_matching`` at its defaults: the Stein estimate of the Hessian's diagonal picks
the leaves of the causal order, and smooth-term tests prune the order's complete DAG at
``alpha = 0.001``, the cutoff of the paper's experiments (Rolland et al., 2022). The facade returns
a DAG and a causal order; the adapter keeps the DAG. :class:`AndreyScoreMatchingOrder` keeps the
order instead, for the Stein-score lane.

The fits run in the interpreter named by ``$ANDREY_BENCH_CAM_PYTHON``, an environment holding the
build under test, like the other Andrey solutions of that lane.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from andrey_bench.adapters.andrey_cam import _andrey_version, _marks

_DEFAULTS: dict[str, Any] = {"alpha": 0.001}


class AndreyScoreMatching:
    """Solution adapter for Andrey's SCORE.

    Satisfies :class:`andrey_bench.contracts.SolutionAdapter`.
    """

    name: str = "andrey.score_matching.numpy"
    package: str = "andrey"
    package_version: str = _andrey_version()
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "score_matching"
    output_type: str = "dag"
    python_env: str = "ANDREY_BENCH_CAM_PYTHON"

    def setup(self) -> None:
        """Fit a tiny table before timing, so lazy imports stay out of the first timed fit."""
        from andrey_bench.adapters.fcm_warm import tiny_table

        self.fit(tiny_table(), {})

    def params(self) -> dict[str, Any]:
        """Andrey's default: the pruning level ``alpha=0.001``."""
        return dict(_DEFAULTS)

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run Andrey SCORE; return the DAG's endpoint marks. The only timed region."""
        import andrey

        out = andrey.score_matching(np.asarray(data, dtype=np.float64), **{**_DEFAULTS, **params})
        return np.asarray(out.structure.to_numpy())

    def to_structure(self, native: np.ndarray) -> Any:
        """Rebuild the DAG from its endpoint marks outside the timed region."""
        from andrey import GraphStructure

        return GraphStructure.from_numpy(native, kind=self.output_type)


class AndreyScoreMatchingOrder(AndreyScoreMatching):
    """Andrey's SCORE order alone: the full DAG of ``ordering``, the order before pruning.

    The DAG it returns joins each variable to every later one
    (:mod:`~andrey_bench.adapters.order_dag`), so its arrowhead recall is the share of true edges
    the order puts in the right direction.
    """

    name: str = "andrey.score_matching.order.numpy"

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run Andrey SCORE; return the endpoint marks of its order's full DAG. Timed."""
        import andrey
        from andrey_bench.adapters.order_dag import order_dag

        out = andrey.score_matching(np.asarray(data, dtype=np.float64), **{**_DEFAULTS, **params})
        return _marks(order_dag(out.ordering))
