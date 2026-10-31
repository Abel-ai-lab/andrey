"""lingam RESIT adapter.

Calls ``lingam.RESIT(regressor=RandomForestRegressor(max_depth=4, random_state=0), alpha=0.01)``:
lingam's default ``alpha`` and the regressor its documentation uses, since ``regressor`` has no
default. lingam offers no additive-model regression, so the regression is a difference from Andrey
and from the authors' code, not a tuning. The other differences are named, not tuned away:

- the next sink is the variable with the smallest HSIC statistic, not the largest p-value;
- the edge-removal tests run at ``alpha = 0.01``, where Andrey and the authors' code use ``0.05``;
- the kernel bandwidth is the median distance between samples.

``adjacency_matrix_`` is a 0/1 matrix with ``B[i, j] = 1`` for ``j -> i`` (row = effect, column =
cause), the opposite indexing from Andrey's, so the reduction transposes it as ``lingam_direct``
does. The output is acyclic.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

_RANDOM_STATE = 0
_ALPHA = 0.01
_MAX_DEPTH = 4


def _lingam_version() -> str:
    """The installed lingam version (distribution metadata)."""
    from importlib import metadata

    try:
        return metadata.version("lingam")
    except metadata.PackageNotFoundError:  # pragma: no cover - installed in bench-env
        return "unknown"


class LingamRESIT:
    """Solution adapter for lingam's RESIT.

    Satisfies :class:`andrey_bench.contracts.SolutionAdapter`.
    """

    name: str = "lingam.resit"
    package: str = "lingam"
    package_version: str = _lingam_version()
    backend: str = "native"
    mode: str = "serial"
    algorithm: str = "resit"
    output_type: str = "dag"

    def setup(self) -> None:
        """Fit a tiny table before timing, so lazy imports stay out of the first timed fit."""
        from andrey_bench.adapters.fcm_warm import tiny_table

        self.fit(tiny_table(), {})

    def params(self) -> dict[str, Any]:
        """lingam's default ``alpha`` and the regressor of its documentation."""
        return {
            "regressor": f"RandomForestRegressor(max_depth={_MAX_DEPTH}, random_state={_RANDOM_STATE})",
            "alpha": _ALPHA,
            "random_state": _RANDOM_STATE,
        }

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run lingam RESIT; return a plain 0/1 DAG adjacency. The timed region."""
        import lingam
        from sklearn.ensemble import RandomForestRegressor

        regressor = RandomForestRegressor(max_depth=_MAX_DEPTH, random_state=_RANDOM_STATE)
        model = lingam.RESIT(regressor=regressor, random_state=_RANDOM_STATE, alpha=_ALPHA)
        model.fit(np.asarray(data, dtype=np.float64))
        # B[i, j] = 1 means j -> i; Andrey wants adj[s, t] = 1 to mean s -> t, so transpose.
        b = np.asarray(model.adjacency_matrix_)
        return np.ascontiguousarray((b != 0).astype(np.int8).T)

    def to_structure(self, native: np.ndarray) -> Any:
        """0/1 DAG adjacency -> Andrey ``GraphStructure``; untimed."""
        from andrey_bench.contracts import structure_from_adjacency

        return structure_from_adjacency(native, kind=self.output_type)
