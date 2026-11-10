"""Andrey NoGAM solution adapter.

Wraps ``andrey.nogam`` at its defaults: the Stein estimate of the score, predicted from each
variable's regression residual, picks the leaves of the causal order, and smooth-term tests prune
the order's complete DAG at ``alpha = 0.001``, the cutoff of the paper's experiments (Montagna et
al., 2023). The facade returns a DAG and a causal order; the adapter keeps the DAG.
:class:`AndreyNoGAMOrder` keeps the order instead, for the Stein-score lane.

The fits run in the interpreter named by ``$ANDREY_BENCH_CAM_PYTHON``, an environment holding the
build under test, like the other Andrey solutions of that lane.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from andrey_bench.adapters.andrey_cam import _andrey_version, _marks

_DEFAULTS: dict[str, Any] = {"alpha": 0.001}


class AndreyNoGAM:
    """Solution adapter for Andrey's NoGAM.

    Satisfies :class:`andrey_bench.contracts.SolutionAdapter`.
    """

    name: str = "andrey.nogam.numpy"
    package: str = "andrey"
    package_version: str = _andrey_version()
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "nogam"
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
        """Run Andrey NoGAM; return the DAG's endpoint marks. The only timed region."""
        import andrey

        out = andrey.nogam(np.asarray(data, dtype=np.float64), **{**_DEFAULTS, **params})
        return np.asarray(out.structure.to_numpy())

    def to_structure(self, native: np.ndarray) -> Any:
        """Rebuild the DAG from its endpoint marks outside the timed region."""
        from andrey import GraphStructure

        return GraphStructure.from_numpy(native, kind=self.output_type)


class AndreyNoGAMOrder(AndreyNoGAM):
    """Andrey's NoGAM order alone: the full DAG of ``ordering``, the order before pruning.

    The DAG it returns joins each variable to every later one
    (:mod:`~andrey_bench.adapters.order_dag`), so its arrowhead recall is the share of true edges
    the order puts in the right direction.
    """

    name: str = "andrey.nogam.order.numpy"

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run Andrey NoGAM; return the endpoint marks of its order's full DAG. Timed."""
        import andrey
        from andrey_bench.adapters.order_dag import order_dag

        out = andrey.nogam(np.asarray(data, dtype=np.float64), **{**_DEFAULTS, **params})
        return _marks(order_dag(out.ordering))
