"""causal-learn PC adapter — the reference constraint-based competitor.

Calls ``pc(X, alpha=0.05, indep_test="fisherz", show_progress=False)``; those pins are also
causal-learn's own defaults. Unlike the GES adapter this path needs no ``np.mat`` shim — it never
imports ``ScoreUtils``.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from andrey_bench.adapters.causal_learn_convert import generalgraph_to_adjacency

_DEFAULT_ALPHA = 0.05
_DEFAULT_INDEP_TEST = "fisherz"


def _causal_learn_version() -> str:
    """The installed causal-learn version (distribution metadata; the module exposes none)."""
    from importlib import metadata

    try:
        return metadata.version("causal-learn")
    except metadata.PackageNotFoundError:  # pragma: no cover - installed in bench-env
        return "unknown"


class CausalLearnPC:
    """Solution adapter for causal-learn's PC (constraint-based lane).

    Satisfies :class:`andrey_bench.contracts.SolutionAdapter`.
    """

    name: str = "causal-learn.pc"
    package: str = "causal-learn"
    package_version: str = _causal_learn_version()
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "pc"
    output_type: str = "cpdag"

    def setup(self) -> None:
        """Import causal-learn before timing; see `contracts.SetupAdapter`.

        The adapter is unpickled in the child, and `fit` imports lazily. Setup excludes that
        import from the first timed fit at `--warmup 0`.
        """
        from causallearn.search.ConstraintBased.PC import pc  # noqa: F401

    def params(self) -> dict[str, Any]:
        """Fisher-Z at ``alpha=0.05`` — causal-learn's own defaults."""
        return {"alpha": _DEFAULT_ALPHA, "indep_test": _DEFAULT_INDEP_TEST}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run causal-learn PC; return a plain 0/1 CPDAG adjacency. The timed region.

        ``alpha`` / ``indep_test`` come from ``params``, defaulting to ``0.05`` / ``"fisherz"``. The
        resulting ``CausalGraph.G`` is reduced by
        :func:`~andrey_bench.adapters.causal_learn_convert.generalgraph_to_adjacency`.
        """
        from causallearn.search.ConstraintBased.PC import pc as _pc

        alpha = float(params.get("alpha", _DEFAULT_ALPHA))
        indep_test = params.get("indep_test", _DEFAULT_INDEP_TEST)

        X = np.asarray(data, dtype=np.float64)
        cg = _pc(X, alpha=alpha, indep_test=indep_test, show_progress=False)
        return generalgraph_to_adjacency(cg.G.graph)

    def to_structure(self, native: np.ndarray) -> Any:
        """0/1 CPDAG adjacency -> Andrey ``GraphStructure``. Runs in bench-env (parent); untimed."""
        from andrey_bench.contracts import structure_from_adjacency

        return structure_from_adjacency(native, kind=self.output_type)
