"""Andrey PC solution adapter — the home team's constraint-based entry.

Calls the public facade ``andrey.pc(X, alpha=0.05, indep_test=...)``. ``alpha=0.05`` and
``indep_test="fisherz"`` are Andrey's own defaults; ``AndreyPC("kci")`` and the other nonlinear
tests (``"rcot"``, ``"cmiknn"``, ``"gpdc"``) are separate solutions, named
``andrey.pc.<indep_test>.numpy``, each at the test's default settings.

PC has no accelerated backend axis — its CI tests are numpy and its Meek orientation is pure
Python — so one ``numpy``/``serial`` variant per test is the whole lane entry.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

# Equalized constraint-lane pins (Fisher-Z CI test at alpha=0.05); runner params override them.
_DEFAULT_ALPHA = 0.05
_DEFAULT_INDEP_TEST = "fisherz"

#: The module each nonlinear test lives in, imported in ``setup`` so the first fit does not pay for
#: it (GPDC's pulls in scikit-learn's Gaussian processes).
NONLINEAR_TEST_MODULES = {
    "kci": "andrey.core.ci_kernel",
    "rcot": "andrey.core.ci_kernel",
    "cmiknn": "andrey.core.ci_cmiknn",
    "gpdc": "andrey.core.ci_gpdc",
}


def _andrey_version() -> str:
    """The installed Andrey version (distribution metadata, else the module attribute)."""
    from importlib import metadata

    try:
        return metadata.version("andrey")
    except metadata.PackageNotFoundError:  # editable/dev checkout with no dist metadata
        import andrey

        return getattr(andrey, "__version__", "unknown")


class AndreyPC:
    """Solution adapter for Andrey's PC (constraint-based lane), one solution per CI test.

    Satisfies :class:`andrey_bench.contracts.SolutionAdapter`. PC has no backend·mode product to
    enumerate, so this is a plain class with class-level metadata; the CI test is the one axis.
    """

    name: str = "andrey.pc.numpy"
    package: str = "andrey"
    package_version: str = _andrey_version()
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "pc"
    output_type: str = "cpdag"

    def __init__(self, indep_test: str = _DEFAULT_INDEP_TEST) -> None:
        if indep_test != _DEFAULT_INDEP_TEST and indep_test not in NONLINEAR_TEST_MODULES:
            raise ValueError(f"Unsupported indep_test: {indep_test!r}")
        self.indep_test = indep_test
        if indep_test != _DEFAULT_INDEP_TEST:
            self.name = f"andrey.pc.{indep_test}.numpy"

    def setup(self) -> None:
        """Import Andrey, its PC implementation, and the CI test's module before timing.

        The facade loads `andrey.constraint.pc` on its first call, and the registry the module of
        a nonlinear test when it first builds it. Setup excludes those imports from `fit`,
        including the first timed fit at `--warmup 0`. See `contracts.SetupAdapter`.
        """
        import importlib

        import andrey  # noqa: F401
        from andrey.constraint.pc import pc  # noqa: F401

        if self.indep_test in NONLINEAR_TEST_MODULES:
            importlib.import_module(NONLINEAR_TEST_MODULES[self.indep_test])

    def params(self) -> dict[str, Any]:
        """``alpha=0.05`` and the adapter's CI test, at that test's default settings."""
        return {"alpha": _DEFAULT_ALPHA, "indep_test": self.indep_test}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run Andrey PC; return the CPDAG's endpoint-mark array. The only timed region.

        ``alpha`` and ``indep_test`` come from ``params``, defaulting to ``0.05`` and the
        adapter's test; the public ``andrey.pc`` facade forwards both. The nonlinear tests'
        ``ExperimentalWarning`` is silenced. Its ``structure.to_numpy()`` is the endpoint-mark
        matrix (``NULL``/``TAIL``/``ARROW``) returned here.
        """
        import warnings

        import andrey

        alpha = float(params.get("alpha", _DEFAULT_ALPHA))
        indep_test = params.get("indep_test", self.indep_test)

        X = np.asarray(data, dtype=np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", andrey.ExperimentalWarning)
            out = andrey.pc(X, alpha=alpha, indep_test=indep_test)
        return np.asarray(out.structure.to_numpy())

    def to_structure(self, native: np.ndarray) -> Any:
        """Endpoint-mark array -> ``GraphStructure``, a direct ``from_numpy`` rebuild; untimed.

        No adjacency remap — ``structure_from_adjacency`` is only for the 0/1 competitors.
        """
        from andrey import GraphStructure

        return GraphStructure.from_numpy(native, kind=self.output_type)
