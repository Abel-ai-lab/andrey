"""DAGMA adapter for the authors' package (Bello et al., 2022), run in its own environment.

The package is ``dagma`` :data:`VERSION` (Apache-2.0), the authors' release:
``DagmaLinear(loss_type="l2").fit`` solves the least-squares fit with the log-determinant
acyclicity function along a central path, then sets the weights below ``w_threshold`` to zero.
``benchmarks/start_notears.sh`` installs it beside the NOTEARS reference. Its linear model imports
NumPy, SciPy, and tqdm; PyTorch serves the nonlinear model alone and stays out of the environment.

A fit is ``DagmaLinear(loss_type="l2").fit(X, lambda1=0.03)`` with the method's other defaults
(``w_threshold=0.3``, ``T=5``, ``s=[1.0, 0.9, 0.8, 0.7, 0.6]``, ``warm_iter=3e4``, ``max_iter=6e4``,
``lr=3e-4``); ``BelloDAGMA(lambda1=0.05)`` is ``bello.dagma.paper``, the same fit at the penalty of
the paper's experiments. The thresholded weights can hold a cycle, which
:func:`~andrey_bench.adapters.weights_dag.dag_of_weights` repairs outside the package; each repair
is logged.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

import numpy as np

from andrey_bench.adapters.weights_dag import dag_of_weights

logger = logging.getLogger(__name__)

#: The pinned release of ``dagma``; ``start_notears.sh`` installs it.
VERSION = "1.1.1"

_DEFAULTS: dict[str, Any] = {"lambda1": 0.03, "loss_type": "l2", "w_threshold": 0.3}


class BelloDAGMA:
    """The authors' ``DagmaLinear`` with the least-squares loss.

    Satisfies :class:`~andrey_bench.contracts.SolutionAdapter` and
    :class:`~andrey_bench.contracts.SetupAdapter`. Fits run in ``$ANDREY_BENCH_NOTEARS_PYTHON``.
    `lambda1` other than the package default names the arm ``bello.dagma.paper``.
    """

    name: str = "bello.dagma"
    package: str = "dagma"
    package_version: str = VERSION
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "dagma"
    output_type: str = "dag"
    python_env: str = "ANDREY_BENCH_NOTEARS_PYTHON"

    def __init__(self, lambda1: float = _DEFAULTS["lambda1"]) -> None:
        self.lambda1 = lambda1
        if lambda1 != _DEFAULTS["lambda1"]:
            self.name = "bello.dagma.paper"

    def setup(self) -> None:
        """Import the package outside fit timing."""
        from dagma.linear import DagmaLinear  # noqa: F401  # ty: ignore[unresolved-import]

    def params(self) -> dict[str, Any]:
        """The penalty, the loss, the threshold, and the pinned release."""
        return {**_DEFAULTS, "lambda1": self.lambda1, "version": VERSION}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run ``DagmaLinear.fit`` and return a 0/1 DAG adjacency. The only timed region."""
        from dagma.linear import DagmaLinear  # ty: ignore[unresolved-import]

        weights = DagmaLinear(loss_type=str(params["loss_type"])).fit(
            np.array(data, dtype=np.float64),
            lambda1=float(params["lambda1"]),
            w_threshold=float(params["w_threshold"]),
        )
        adjacency, dropped = dag_of_weights(weights)
        if dropped:
            logger.warning("%s: cyclic output, dropped %d edge(s)", self.name, dropped)
        return adjacency

    def to_structure(self, native: np.ndarray) -> Any:
        """0/1 DAG adjacency -> Andrey ``GraphStructure``; untimed."""
        from andrey_bench.contracts import structure_from_adjacency

        return structure_from_adjacency(native, kind=self.output_type)
