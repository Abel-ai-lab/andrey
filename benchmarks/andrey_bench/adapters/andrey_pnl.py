"""Andrey PNL solution adapter for cause-effect pairs.

Wraps ``andrey.pnl``, which decides the direction of a two-column table and returns a two-node DAG:
``0 -> 1``, ``1 -> 0``, or no edge. ``alpha=None`` draws the direction with the larger p-value, the
rule the causal-learn adapter applies to causal-learn's p-values.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

_DEFAULTS: dict[str, Any] = {"alpha": None}


def _andrey_version() -> str:
    """The installed Andrey version (distribution metadata, else the module attribute)."""
    from importlib import metadata

    try:
        return metadata.version("andrey")
    except metadata.PackageNotFoundError:  # editable/dev checkout with no dist metadata
        import andrey

        return getattr(andrey, "__version__", "unknown")


class AndreyPNL:
    """Solution adapter for Andrey's PNL.

    Satisfies :class:`andrey_bench.contracts.SolutionAdapter`.
    """

    name: str = "andrey.pnl.numpy"
    package: str = "andrey"
    package_version: str = _andrey_version()
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "pnl"
    output_type: str = "dag"

    def setup(self) -> None:
        """Import Andrey and its PNL implementation before timing.

        The facade loads `andrey.irregular.pnl` on its first call. Setup excludes that import from
        `fit`, including the first timed fit at `--warmup 0`. See `contracts.SetupAdapter`.
        """
        import andrey  # noqa: F401
        from andrey.irregular.pnl import pnl  # noqa: F401

    def params(self) -> dict[str, Any]:
        """Return the decision rule: ``alpha=None`` picks the direction with the larger p-value."""
        return dict(_DEFAULTS)

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run Andrey PNL; return the two-node DAG's endpoint marks. The only timed region."""
        import andrey

        out = andrey.pnl(np.asarray(data, dtype=np.float64), **{**_DEFAULTS, **params})
        return np.asarray(out.structure.to_numpy())

    def to_structure(self, native: np.ndarray) -> Any:
        """Rebuild the two-node DAG from its endpoint marks outside the timed region."""
        from andrey import GraphStructure

        return GraphStructure.from_numpy(native, kind=self.output_type)
