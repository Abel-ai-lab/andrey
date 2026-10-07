"""gCastle GES adapter — the second score-based competitor.

Calls ``castle.algorithms.GES(criterion="bic", method="scatter").learn(X)``. gCastle exposes no
penalty or max-parents knob (verified against gCastle 1.0.4): its ``k``/``N`` arguments are the
discrete BDeu prior and equivalent sample size, its ``BICScore`` hardcodes the ``(k+1)·log n``
penalty (a plain BIC, ``λ=1``), and its forward/backward search takes no max-parents cap. Its score
family is still linear-Gaussian BIC, so it competes in the GES lane on its own terms — which is what
its recorded ``params`` say.

Orientation (verified empirically, gCastle 1.0.4): ``causal_matrix[i,j]=1`` means the directed edge
``i->j`` (row = source), Andrey's own adjacency orientation, so no transpose is needed.
Undirected CPDAG edges come back as symmetric ``1``s, the form ``structure_from_adjacency``
reconstructs into TAIL-TAIL marks.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

# The two knobs gCastle actually exposes for linear-Gaussian GES.
_DEFAULT_CRITERION = "bic"  # linear-Gaussian BIC score family
_DEFAULT_METHOD = "scatter"  # covariance-form BIC (gCastle's native default; alt: "r2")


def _gcastle_version() -> str:
    """The installed gCastle version (distribution metadata; the ``castle`` module exposes one too)."""
    from importlib import metadata

    try:
        return metadata.version("gcastle")
    except metadata.PackageNotFoundError:  # pragma: no cover - installed in bench-env
        return "unknown"


class GCastleGES:
    """Solution adapter for gCastle's GES (score-based lane).

    Satisfies :class:`andrey_bench.contracts.SolutionAdapter`.
    """

    name: str = "gcastle.ges"
    package: str = "gcastle"
    package_version: str = _gcastle_version()
    backend: str = "native"
    mode: str = "serial"
    algorithm: str = "ges"
    output_type: str = "cpdag"

    def setup(self) -> None:
        """Import gCastle before timing; see `contracts.SetupAdapter`.

        The adapter is unpickled in the child, and `fit` imports lazily. Setup excludes that
        import from the first timed fit at `--warmup 0`.
        """
        from castle.algorithms import GES  # noqa: F401

    def params(self) -> dict[str, Any]:
        """gCastle's GES pins. It exposes no penalty or max-parent knob, so neither appears."""
        return {"criterion": _DEFAULT_CRITERION, "method": _DEFAULT_METHOD}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run gCastle GES; return a plain 0/1 CPDAG adjacency. The timed region.

        Only the two knobs gCastle exposes are read from ``params``: ``criterion`` (default ``bic``,
        the linear-Gaussian BIC score family) and ``method`` (default ``scatter``).
        """
        from castle.algorithms import GES as _GES

        criterion = params.get("criterion", _DEFAULT_CRITERION)
        method = params.get("method", _DEFAULT_METHOD)

        X = np.asarray(data, dtype=np.float64)
        model = _GES(criterion=criterion, method=method)
        model.learn(X)

        # gCastle's causal_matrix uses adj[i,j]=1 for i->j (directed) with undirected CPDAG edges as
        # symmetric 1s — already Andrey's orientation, so a straight 0/1 reduction (no transpose).
        adj = (np.asarray(model.causal_matrix) != 0).astype(np.int8)
        return adj

    def to_structure(self, native: np.ndarray) -> Any:
        """0/1 CPDAG adjacency -> Andrey ``GraphStructure``. Runs in bench-env (parent); untimed."""
        from andrey_bench.contracts import structure_from_adjacency

        return structure_from_adjacency(native, kind=self.output_type)
