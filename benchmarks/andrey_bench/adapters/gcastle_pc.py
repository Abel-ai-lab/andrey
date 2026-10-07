"""gCastle PC adapter — the third constraint-based competitor.

Calls ``castle.algorithms.PC(alpha=0.05, ci_test="fisherz").learn(X)`` — gCastle's own defaults,
and both are real constructor arguments (verified against gCastle 1.0.4).

Orientation (verified empirically, gCastle 1.0.4): ``causal_matrix[i,j]=1`` means the directed edge
``i->j`` (row = source), Andrey's own adjacency orientation, so no transpose is needed.
Undirected CPDAG edges come back as symmetric ``1``s, the form ``structure_from_adjacency``
reconstructs into TAIL-TAIL marks; a one-sided entry becomes TAIL/ARROW.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

# Equalized constraint-lane pins (Fisher-Z CI test at alpha=0.05, shared with the Andrey and
# causal-learn PC adapters); both are gCastle PC constructor knobs, so they apply. Runner params
# override them.
_DEFAULT_ALPHA = 0.05
_DEFAULT_INDEP_TEST = "fisherz"


def _gcastle_version() -> str:
    """The installed gCastle version (distribution metadata; the ``castle`` module exposes one too)."""
    from importlib import metadata

    try:
        return metadata.version("gcastle")
    except metadata.PackageNotFoundError:  # pragma: no cover - installed in bench-env
        return "unknown"


class GCastlePC:
    """Solution adapter for gCastle's PC (constraint-based lane).

    Satisfies :class:`andrey_bench.contracts.SolutionAdapter`. Unlike
    :class:`~andrey_bench.adapters.gcastle_ges.GCastleGES`, gCastle's PC exposes both knobs
    (``alpha`` / ``ci_test``) as real constructor arguments.
    """

    name: str = "gcastle.pc"
    package: str = "gcastle"
    package_version: str = _gcastle_version()
    backend: str = "native"
    mode: str = "serial"
    algorithm: str = "pc"
    output_type: str = "cpdag"

    def setup(self) -> None:
        """Import gCastle before timing; see `contracts.SetupAdapter`.

        The adapter is unpickled in the child, and `fit` imports lazily. Setup excludes that
        import from the first timed fit at `--warmup 0`.
        """
        from castle.algorithms import PC  # noqa: F401

    def params(self) -> dict[str, Any]:
        """Fisher-Z at ``alpha=0.05`` — gCastle's own defaults."""
        return {"alpha": _DEFAULT_ALPHA, "indep_test": _DEFAULT_INDEP_TEST}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run gCastle PC; return a plain 0/1 CPDAG adjacency. The timed region.

        ``alpha`` / ``indep_test`` come from ``params``, defaulting to ``0.05`` / ``"fisherz"`` and
        passed straight to gCastle's ``PC(alpha=..., ci_test=...)``.
        """
        from castle.algorithms import PC as _PC

        alpha = float(params.get("alpha", _DEFAULT_ALPHA))
        indep_test = params.get("indep_test", params.get("ci_test", _DEFAULT_INDEP_TEST))

        X = np.asarray(data, dtype=np.float64)
        model = _PC(alpha=alpha, ci_test=indep_test)
        model.learn(X)

        # gCastle's causal_matrix uses adj[i,j]=1 for i->j (directed) with undirected CPDAG edges as
        # symmetric 1s — already Andrey's orientation, so a straight 0/1 reduction (no transpose).
        adj = (np.asarray(model.causal_matrix) != 0).astype(np.int8)
        return adj

    def to_structure(self, native: np.ndarray) -> Any:
        """0/1 CPDAG adjacency -> Andrey ``GraphStructure``. Runs in bench-env (parent); untimed."""
        from andrey_bench.contracts import structure_from_adjacency

        return structure_from_adjacency(native, kind=self.output_type)
