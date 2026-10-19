"""causal-learn PC adapter — the reference constraint-based competitor.

Calls ``pc(X, alpha=0.05, indep_test="fisherz", show_progress=False)``; those pins are also
causal-learn's own defaults. :class:`CausalLearnPCChisq` and :class:`CausalLearnPCGsq` run its
chi-square and G-square tests, for categorical data. Unlike the GES adapter this path needs no
``np.mat`` shim — it never imports ``ScoreUtils``.

``CausalLearnPC("kci")`` and ``CausalLearnPC("rcot")`` are the references for Andrey's KCI and
RCoT, named ``causal-learn.pc.<indep_test>``. Their settings map to Andrey's as follows:

| Andrey | causal-learn |
|---|---|
| ``indep_test="kci"`` (gamma approximation) | ``indep_test="kci"`` (its default ``approx=True``) |
| ``indep_test="rcot"`` | ``indep_test="rcit"`` with ``rcit=False`` |
| RCoT ``n_features=5`` | ``num_f2=5`` |
| RCoT ``n_features_cond=100`` | ``num_f=100`` |
| RCoT's Hall-Buckley-Eagleson p-value | ``approx="hbe"`` |

Each pin is causal-learn's own default except ``approx``, whose default is ``"lpd4"``; the adapter
sets ``"hbe"`` so both packages compute the p-value the same way.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from andrey_bench.adapters.causal_learn_convert import generalgraph_to_adjacency

_DEFAULT_ALPHA = 0.05
_DEFAULT_INDEP_TEST = "fisherz"

#: Andrey's ``indep_test`` name -> causal-learn's name for the same test and the settings that
#: match Andrey's defaults (see the module docstring).
CI_TESTS: dict[str, tuple[str, dict[str, Any]]] = {
    "fisherz": ("fisherz", {}),
    "chisq": ("chisq", {}),
    "gsq": ("gsq", {}),
    "kci": ("kci", {}),
    "rcot": ("rcit", {"rcit": False, "num_f2": 5, "num_f": 100, "approx": "hbe"}),
}


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
    #: The CI test this solution runs; a variant with another test has its own ``name``.
    indep_test: str = _DEFAULT_INDEP_TEST

    def __init__(self, indep_test: str | None = None) -> None:
        if indep_test is None:  # the class's own test
            return
        if indep_test not in CI_TESTS:
            raise ValueError(f"Unsupported indep_test: {indep_test!r}")
        self.indep_test = indep_test
        if indep_test != _DEFAULT_INDEP_TEST:
            self.name = f"causal-learn.pc.{indep_test}"

    def setup(self) -> None:
        """Import causal-learn before timing; see `contracts.SetupAdapter`.

        The adapter is unpickled in the child, and `fit` imports lazily. Setup excludes that
        import from the first timed fit at `--warmup 0`.
        """
        from causallearn.search.ConstraintBased.PC import pc  # noqa: F401

    def params(self) -> dict[str, Any]:
        """``alpha=0.05`` and the adapter's CI test, in Andrey's names."""
        return {"alpha": _DEFAULT_ALPHA, "indep_test": self.indep_test}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run causal-learn PC; return a plain 0/1 CPDAG adjacency. The timed region.

        ``alpha`` / ``indep_test`` come from ``params``, defaulting to ``0.05`` and the adapter's
        test, which :data:`CI_TESTS` maps to causal-learn's name and settings. The resulting
        ``CausalGraph.G`` is reduced by
        :func:`~andrey_bench.adapters.causal_learn_convert.generalgraph_to_adjacency`.
        """
        from causallearn.search.ConstraintBased.PC import pc as _pc

        alpha = float(params.get("alpha", _DEFAULT_ALPHA))
        indep_test, settings = CI_TESTS[params.get("indep_test", self.indep_test)]

        X = np.asarray(data, dtype=np.float64)
        cg = _pc(X, alpha=alpha, indep_test=indep_test, show_progress=False, **settings)
        return generalgraph_to_adjacency(cg.G.graph)

    def to_structure(self, native: np.ndarray) -> Any:
        """0/1 CPDAG adjacency -> Andrey ``GraphStructure``. Runs in bench-env (parent); untimed."""
        from andrey_bench.contracts import structure_from_adjacency

        return structure_from_adjacency(native, kind=self.output_type)


class CausalLearnPCChisq(CausalLearnPC):
    """causal-learn's PC with its chi-square test, for categorical data."""

    name: str = "causal-learn.pc.chisq"
    indep_test: str = "chisq"


class CausalLearnPCGsq(CausalLearnPC):
    """causal-learn's PC with its G-square test, for categorical data."""

    name: str = "causal-learn.pc.gsq"
    indep_test: str = "gsq"
