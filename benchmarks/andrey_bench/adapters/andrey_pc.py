"""Andrey PC solution adapter — the home team's constraint-based entry.

Calls the public facade ``andrey.pc(X, alpha=0.05, indep_test="fisherz")``. Both values are
Andrey's own defaults; ``fisherz`` is the only CI test Andrey supports.

PC has no accelerated backend axis — its CI test is pure-numpy partial correlation and its Meek
orientation is pure Python — so the single ``numpy``/``serial`` variant is the whole lane entry.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

# Equalized constraint-lane pins (Fisher-Z CI test at alpha=0.05); runner params override them.
_DEFAULT_ALPHA = 0.05
_DEFAULT_INDEP_TEST = "fisherz"


def _andrey_version() -> str:
    """The installed Andrey version (distribution metadata, else the module attribute)."""
    from importlib import metadata

    try:
        return metadata.version("andrey")
    except metadata.PackageNotFoundError:  # editable/dev checkout with no dist metadata
        import andrey

        return getattr(andrey, "__version__", "unknown")


class AndreyPC:
    """Solution adapter for Andrey's PC (constraint-based lane).

    Satisfies :class:`andrey_bench.contracts.SolutionAdapter`. A single fixed variant — PC has no
    backend·mode product to enumerate — so this is a plain class with class-level metadata rather
    than a factory of dataclass instances.
    """

    name: str = "andrey.pc.numpy"
    package: str = "andrey"
    package_version: str = _andrey_version()
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "pc"
    output_type: str = "cpdag"

    def setup(self) -> None:
        """Import Andrey and its PC implementation before timing.

        The facade loads `andrey.constraint.pc` on its first call. Setup excludes
        that import from `fit`, including the first timed fit at `--warmup 0`.
        See `contracts.SetupAdapter`.
        """
        import andrey  # noqa: F401
        from andrey.constraint.pc import pc  # noqa: F401

    def params(self) -> dict[str, Any]:
        """Fisher-Z at ``alpha=0.05`` — Andrey's own defaults."""
        return {"alpha": _DEFAULT_ALPHA, "indep_test": _DEFAULT_INDEP_TEST}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run Andrey PC; return the CPDAG's endpoint-mark array. The only timed region.

        ``alpha`` and ``indep_test`` come from ``params``, defaulting to ``0.05`` / ``"fisherz"``;
        the public ``andrey.pc`` facade forwards both. Its ``structure.to_numpy()`` is the
        endpoint-mark matrix (``NULL``/``TAIL``/``ARROW``) returned here.
        """
        import andrey

        alpha = float(params.get("alpha", _DEFAULT_ALPHA))
        indep_test = params.get("indep_test", _DEFAULT_INDEP_TEST)

        X = np.asarray(data, dtype=np.float64)
        out = andrey.pc(X, alpha=alpha, indep_test=indep_test)
        return np.asarray(out.structure.to_numpy())

    def to_structure(self, native: np.ndarray) -> Any:
        """Endpoint-mark array -> ``GraphStructure``, a direct ``from_numpy`` rebuild; untimed.

        No adjacency remap — ``structure_from_adjacency`` is only for the 0/1 competitors.
        """
        from andrey import GraphStructure

        return GraphStructure.from_numpy(native, kind=self.output_type)
