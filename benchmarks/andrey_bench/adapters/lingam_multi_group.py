"""lingam MultiGroupDirectLiNGAM adapter, in an environment of its own.

Calls ``lingam.MultiGroupDirectLiNGAM().fit(groups)`` on the groups of a ``multi_group`` dataset,
split as :mod:`~andrey_bench.adapters.andrey_multi_group` splits them, and returns the groups' DAGs
side by side. ``adjacency_matrices_[g][i, j]`` weights ``j -> i``, so each is transposed. Its fit
does not read ``random_state``, so none is passed. The fits run in the environment
:mod:`~andrey_bench.adapters.lingam_env` describes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from andrey_bench.adapters.lingam_env import PYTHON_ENV_VAR, env_version
from andrey_bench.regime_models import N_GROUPS, block_diagonal, split_groups


@dataclass(frozen=True)
class LingamMultiGroupDirectLiNGAM:
    """Adapter for ``lingam.MultiGroupDirectLiNGAM``; build it with :func:`lingam_multi_group_adapter`."""

    name: str = "lingam.multi_group_direct_lingam"
    package: str = "lingam"
    package_version: str = ""
    backend: str = "native"
    mode: str = "serial"
    algorithm: str = "multi_group_direct_lingam"
    output_type: str = "dag"
    python_env: str = PYTHON_ENV_VAR

    def setup(self) -> None:
        """Import lingam before timing; see `contracts.SetupAdapter`."""
        import lingam  # noqa: F401

    def params(self) -> dict[str, Any]:
        """Return the number of groups in the columns."""
        return {"n_groups": N_GROUPS}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run lingam's method on the groups; return their 0/1 DAGs side by side. Timed."""
        import lingam

        settings = {**self.params(), **params}
        groups = split_groups(np.asarray(data, dtype=np.float64), int(settings["n_groups"]))
        model = lingam.MultiGroupDirectLiNGAM().fit(groups)
        return block_diagonal([np.asarray(B).T for B in model.adjacency_matrices_])

    def to_structure(self, native: np.ndarray) -> Any:
        """0/1 adjacency -> Andrey ``GraphStructure``; runs in the parent, untimed."""
        from andrey_bench.contracts import structure_from_adjacency

        return structure_from_adjacency(native, kind=self.output_type)


def lingam_multi_group_adapter() -> LingamMultiGroupDirectLiNGAM:
    """A :class:`LingamMultiGroupDirectLiNGAM` carrying its environment's lingam version."""
    return LingamMultiGroupDirectLiNGAM(package_version=env_version("lingam"))
