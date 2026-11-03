"""Andrey CAM solution adapter.

Wraps ``andrey.cam`` at its defaults: the paper's three steps (boosting preselects up to 10
possible parents per variable, a greedy search adds edges, smooth-term tests prune at
``p <= 0.001``), with no parent cap. The facade returns a DAG and a causal order; the adapter keeps
the DAG.

The fits run in the interpreter named by ``$ANDREY_BENCH_CAM_PYTHON``, an environment holding the
build under test, so a build that is not the driver's can be measured against the reference.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

_DEFAULTS: dict[str, Any] = {"alpha": 0.001, "max_parents": None}


def _andrey_version() -> str:
    """The installed Andrey version (distribution metadata, else the module attribute)."""
    from importlib import metadata

    try:
        return metadata.version("andrey")
    except metadata.PackageNotFoundError:  # editable/dev checkout with no dist metadata
        import andrey

        return getattr(andrey, "__version__", "unknown")


class AndreyCAM:
    """Solution adapter for Andrey's CAM.

    Satisfies :class:`andrey_bench.contracts.SolutionAdapter`.
    """

    name: str = "andrey.cam.numpy"
    package: str = "andrey"
    package_version: str = _andrey_version()
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "cam"
    output_type: str = "dag"
    python_env: str = "ANDREY_BENCH_CAM_PYTHON"

    def setup(self) -> None:
        """Fit a tiny table before timing, so lazy imports stay out of the first timed fit."""
        from andrey_bench.adapters.fcm_warm import tiny_table

        self.fit(tiny_table(), {})

    def params(self) -> dict[str, Any]:
        """Andrey's defaults: ``alpha=0.001`` and no parent cap."""
        return dict(_DEFAULTS)

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run Andrey CAM; return the DAG's endpoint marks. The only timed region."""
        import andrey

        out = andrey.cam(np.asarray(data, dtype=np.float64), **{**_DEFAULTS, **params})
        return np.asarray(out.structure.to_numpy())

    def to_structure(self, native: np.ndarray) -> Any:
        """Rebuild the DAG from its endpoint marks outside the timed region."""
        from andrey import GraphStructure

        return GraphStructure.from_numpy(native, kind=self.output_type)
