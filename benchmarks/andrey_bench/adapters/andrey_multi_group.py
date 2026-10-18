"""Andrey MultiGroupDirectLiNGAM solution adapter for several groups over the same variables.

Calls ``andrey.multi_group_direct_lingam(groups)``. A ``multi_group`` dataset holds its groups side
by side, in equal blocks of columns (:mod:`andrey_bench.regime_models`), so ``fit`` splits the
columns by ``n_groups`` and returns the groups' DAGs side by side, the layout of the truth.
Andrey's ``random_state`` is ignored by the method and is not passed.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from andrey_bench.regime_models import N_GROUPS, block_diagonal, split_groups

_DEFAULTS = {"n_groups": N_GROUPS}


def _andrey_version() -> str:
    """The installed Andrey version (distribution metadata, else the module attribute)."""
    from importlib import metadata

    try:
        return metadata.version("andrey")
    except metadata.PackageNotFoundError:  # editable/dev checkout with no dist metadata
        import andrey

        return getattr(andrey, "__version__", "unknown")


class AndreyMultiGroupDirectLiNGAM:
    """Solution adapter for Andrey's MultiGroupDirectLiNGAM.

    Satisfies :class:`andrey_bench.contracts.SolutionAdapter`.
    """

    name: str = "andrey.multi_group_direct_lingam.numpy"
    package: str = "andrey"
    package_version: str = _andrey_version()
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "multi_group_direct_lingam"
    output_type: str = "dag"

    def setup(self) -> None:
        """Import Andrey and the method before timing; see `contracts.SetupAdapter`."""
        import andrey  # noqa: F401
        from andrey.lingam.multi_group import multi_group_direct_lingam  # noqa: F401

    def params(self) -> dict[str, Any]:
        """Return the number of groups in the columns."""
        return dict(_DEFAULTS)

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run Andrey's method on the groups; return their 0/1 DAGs side by side. Timed."""
        import andrey

        groups = split_groups(
            np.asarray(data, dtype=np.float64), int({**_DEFAULTS, **params}["n_groups"])
        )
        outs = andrey.multi_group_direct_lingam(groups)
        # weighted_adjacency[i, j] weights i -> j, the orientation of the 0/1 adjacency.
        return block_diagonal([out.weighted_adjacency for out in outs])

    def to_structure(self, native: np.ndarray) -> Any:
        """0/1 adjacency -> Andrey ``GraphStructure``; untimed."""
        from andrey_bench.contracts import structure_from_adjacency

        return structure_from_adjacency(native, kind=self.output_type)
