"""NOTEARS adapter for the authors' code (Zheng et al., 2018), run unmodified in its own environment.

The code is ``notears/linear.py`` of ``xunzheng/notears`` at commit :data:`COMMIT` (Apache-2.0):
``notears_linear`` solves the least-squares fit under the matrix-exponential acyclicity constraint
with an augmented Lagrangian and L-BFGS-B, then sets the weights below ``w_threshold`` to zero. It
is a measurement reference only: ``benchmarks/start_notears.sh`` downloads the file, checks its
SHA-256, and builds the environment; the benchmark loads the file from the directory
``$ANDREY_BENCH_NOTEARS`` names and copies none of it.

A fit is ``notears_linear(X, lambda1=0.1, loss_type="l2")`` with the function's other defaults
(``max_iter=100``, ``h_tol=1e-8``, ``rho_max=1e16``, ``w_threshold=0.3``). The function has no
default for ``lambda1``; 0.1 is the value of the repository's example and the paper's. The
thresholded weights can hold a cycle, which :func:`~andrey_bench.adapters.weights_dag.dag_of_weights`
repairs outside the authors' code; each repair is logged.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from andrey_bench.adapters.weights_dag import dag_of_weights

logger = logging.getLogger(__name__)

#: Names the directory holding the downloaded file.
CODE_ENV = "ANDREY_BENCH_NOTEARS"

#: The pinned commit of ``xunzheng/notears``; ``start_notears.sh`` fetches it.
COMMIT = "4a9ab19fe502e392503da331773c5223d82d3666"

_DEFAULTS: dict[str, Any] = {"lambda1": 0.1, "loss_type": "l2", "w_threshold": 0.3}


def code_file() -> Path:
    """The authors' ``linear.py``, or ``SystemExit`` naming what is missing."""
    raw = os.environ.get(CODE_ENV, "").strip()
    if not raw:
        raise SystemExit(
            f"${CODE_ENV} is unset: the authors' NOTEARS code is a measurement reference that the "
            "benchmark reads from disk. Run benchmarks/start_notears.sh and export what it prints."
        )
    path = Path(raw) / "linear.py"
    if not path.is_file():
        raise SystemExit(f"${CODE_ENV}={raw!r} lacks linear.py")
    return path


def _notears_linear():
    """The authors' ``notears_linear``, loaded from :func:`code_file`."""
    spec = importlib.util.spec_from_file_location("zheng_notears_linear", code_file())
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.notears_linear


class ZhengNOTEARS:
    """The authors' ``notears_linear`` with the least-squares loss.

    Satisfies :class:`~andrey_bench.contracts.SolutionAdapter` and
    :class:`~andrey_bench.contracts.SetupAdapter`. Fits run in ``$ANDREY_BENCH_NOTEARS_PYTHON``.
    """

    name: str = "zheng.notears"
    package: str = "notears"
    package_version: str = COMMIT[:8]
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "notears"
    output_type: str = "dag"
    python_env: str = "ANDREY_BENCH_NOTEARS_PYTHON"

    def setup(self) -> None:
        """Load the authors' file outside fit timing."""
        self._fit = _notears_linear()

    def params(self) -> dict[str, Any]:
        """The penalty, the loss, the threshold, and the pinned commit."""
        return {**_DEFAULTS, "commit": COMMIT}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run ``notears_linear`` and return a 0/1 DAG adjacency. The only timed region."""
        weights = self._fit(
            np.array(data, dtype=np.float64),
            lambda1=float(params["lambda1"]),
            loss_type=str(params["loss_type"]),
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
