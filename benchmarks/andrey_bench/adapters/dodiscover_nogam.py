"""NoGAM adapter for the authors' code in dodiscover (Montagna et al., 2023), in its own environment.

The first author added NoGAM to dodiscover (MIT) as ``dodiscover.toporder.NoGAM``; dodiscover has
no release with it, so ``benchmarks/start_stein.sh`` installs commit :data:`COMMIT`. A fit is the
class's ``learn_graph`` at its defaults, which are the paper's (Sect. 6): the Stein gradient
estimator with ``eta_G = 0.001``, residuals and score entries predicted by kernel ridge regression
with ``alpha = 0.01`` and five-fold cross-validation, no preliminary neighbor search, and CAM
pruning by spline GAMs, with the paper's cutoff ``0.001`` in place of the class's ``0.05``. The
paper's kernel bandwidth ``gamma = 0.1`` is the class's ``ridge_gamma``, but the code builds every
kernel ridge with ``gamma=ridge_alpha``, so the fits use ``0.01``; the adapter leaves that as the
code has it. The data go in as a data frame of ``float64`` columns.

:class:`DodiscoverNoGAMOrder` sets ``prune=False``, which returns the full DAG of the order
(:mod:`~andrey_bench.adapters.order_dag`).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

#: The pinned commit of ``py-why/dodiscover``; ``start_stein.sh`` installs it.
COMMIT = "6e64456198a5b66e515f250a8f4b143ebc31c9a6"

_DEFAULTS: dict[str, Any] = {
    "n_crossval": 5,
    "ridge_alpha": 0.01,
    "ridge_gamma": 0.1,
    "eta_G": 0.001,
    "eta_H": 0.001,
    "alpha": 0.001,
    "n_splines": 10,
    "splines_degree": 3,
    "pns": False,
}


class DodiscoverNoGAM:
    """The authors' NoGAM: the order from score and residual regressions, then CAM pruning.

    Satisfies :class:`~andrey_bench.contracts.SolutionAdapter` and
    :class:`~andrey_bench.contracts.SetupAdapter`.
    """

    name: str = "dodiscover.nogam"
    package: str = "dodiscover"
    package_version: str = COMMIT[:8]
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "nogam"
    output_type: str = "dag"
    python_env: str = "ANDREY_BENCH_STEIN_PYTHON"
    prune: bool = True

    def setup(self) -> None:
        """Fit a tiny table before timing, so imports and first-call costs stay out of the fit."""
        from andrey_bench.adapters.fcm_warm import tiny_table

        self.fit(tiny_table(), self.params())

    def params(self) -> dict[str, Any]:
        """The class's defaults with the paper's cutoff, and the pinned commit."""
        return {**_DEFAULTS, "prune": self.prune, "commit": COMMIT}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run the authors' NoGAM and return a 0/1 DAG adjacency. The only timed region."""
        import networkx as nx
        import pandas as pd
        from dodiscover import make_context  # ty: ignore[unresolved-import]
        from dodiscover.toporder import NoGAM  # ty: ignore[unresolved-import]

        frame = pd.DataFrame(np.asarray(data, dtype=np.float64))
        model = NoGAM(**{k: v for k, v in params.items() if k != "commit"})
        model.learn_graph(frame, make_context().variables(data=frame).build())
        nodes = list(frame.columns)
        return nx.to_numpy_array(model.graph_, nodelist=nodes).astype(np.int8)

    def to_structure(self, native: np.ndarray) -> Any:
        """0/1 DAG adjacency -> Andrey ``GraphStructure``; untimed."""
        from andrey_bench.contracts import structure_from_adjacency

        return structure_from_adjacency(native, kind=self.output_type)


class DodiscoverNoGAMOrder(DodiscoverNoGAM):
    """The authors' NoGAM order alone, as its full DAG."""

    name: str = "dodiscover.nogam.order"
    prune: bool = False
