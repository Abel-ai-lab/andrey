"""Andrey RESIT solution adapter.

Wraps ``andrey.resit`` at its defaults: regression by an additive model of thin plate regression
splines with GCV smoothing, the gamma HSIC test, and ``alpha = 0.05``, the settings of Peters,
Mooij, Janzing, and Scholkopf (2014). It returns a DAG and a causal order; the adapter keeps the
DAG.

The fits run in the interpreter named by ``$ANDREY_BENCH_RESIT_PYTHON``, an environment holding the
build under test, so a build that is not the driver's can be measured against the references.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

_DEFAULTS: dict[str, Any] = {"alpha": 0.05}


def _andrey_version() -> str:
    """The installed Andrey version (distribution metadata, else the module attribute)."""
    from importlib import metadata

    try:
        return metadata.version("andrey")
    except metadata.PackageNotFoundError:  # editable/dev checkout with no dist metadata
        import andrey

        return getattr(andrey, "__version__", "unknown")


class AndreyRESIT:
    """Solution adapter for Andrey's RESIT.

    Satisfies :class:`andrey_bench.contracts.SolutionAdapter`.
    """

    name: str = "andrey.resit.numpy"
    package: str = "andrey"
    package_version: str = _andrey_version()
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "resit"
    output_type: str = "dag"
    python_env: str = "ANDREY_BENCH_RESIT_PYTHON"

    def setup(self) -> None:
        """Fit a tiny table before timing, so lazy imports stay out of the first timed fit."""
        from andrey_bench.adapters.fcm_warm import tiny_table

        self.fit(tiny_table(), {})

    def params(self) -> dict[str, Any]:
        """Andrey's default: ``alpha=0.05``."""
        return dict(_DEFAULTS)

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run Andrey RESIT; return the DAG's endpoint marks. The only timed region."""
        import andrey

        out = andrey.resit(np.asarray(data, dtype=np.float64), **{**_DEFAULTS, **params})
        return np.asarray(out.structure.to_numpy())

    def to_structure(self, native: np.ndarray) -> Any:
        """Rebuild the DAG from its endpoint marks outside the timed region."""
        from andrey import GraphStructure

        return GraphStructure.from_numpy(native, kind=self.output_type)
