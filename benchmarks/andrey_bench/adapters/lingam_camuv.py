"""lingam CAM-UV adapter for nonlinear additive data with hidden variables, in its own environment.

Calls ``lingam.CAMUV(alpha=0.01, num_explanatory_vals=3).fit(X)``, the settings of
:mod:`~andrey_bench.adapters.andrey_camuv` in lingam's names:

| Andrey (:func:`andrey.camuv`) | lingam 1.13.0 ``CAMUV`` |
|---|---|
| ``alpha=0.01`` | ``alpha=0.01`` |
| ``max_depth=2``, sets of up to 3 variables | ``num_explanatory_vals=3``, the largest set, the variable itself included |

``adjacency_matrix_`` has ``B[i, j] = 1`` for each parent ``j`` of ``i`` and ``NaN`` in both cells of
a pair joined through a hidden variable. ``fit`` reads the parents and the pairs back and builds the
marks of :func:`camuv_marks`, which draws a pair ``<->``, where Andrey draws ``o-o``.

**It runs in an environment of its own.** lingam's CAM-UV fits its smooth functions with pygam, and
lingam 1.13.0 and pygam 0.12.0 do not install beside the bench environment's scipy, so the fits run
in the environment :mod:`~andrey_bench.adapters.lingam_env` describes. Only ``fit`` runs there; this
module imports nothing but numpy and that module at import time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

from andrey_bench.adapters.lingam_env import PYTHON_ENV_VAR, env_version

#: The settings in Andrey's names; ``fit`` passes ``max_depth + 1`` as ``num_explanatory_vals``.
DEFAULTS = {"alpha": 0.01, "max_depth": 2}

# Andrey's endpoint marks: the mark at node i on the edge i-j is marks[i, j].
_TAIL, _ARROW = 1, 2


def camuv_marks(
    parents: Iterable[Iterable[int]], pairs: Iterable[Iterable[int]], d: int
) -> np.ndarray:
    """Return the PAG marks of a CAM-UV result of the reference implementations.

    ``parents[i]`` holds the parents of variable ``i``, each an edge ``j -> i``; each of `pairs`
    joins two variables through a hidden variable and becomes ``<->``, in place of any directed edge
    between them.
    """
    marks = np.zeros((d, d), dtype=np.int8)
    for i, members in enumerate(parents):
        for j in members:
            marks[int(j), i], marks[i, int(j)] = _TAIL, _ARROW
    for pair in pairs:
        a, b = (int(v) for v in pair)
        marks[a, b] = marks[b, a] = _ARROW
    return marks


@dataclass(frozen=True)
class LingamCAMUV:
    """Solution adapter for lingam's CAM-UV; build it with :func:`lingam_camuv_adapter`.

    Satisfies :class:`andrey_bench.contracts.SolutionAdapter`.
    """

    name: str = "lingam.camuv"
    package: str = "lingam"
    package_version: str = ""
    backend: str = "native"
    mode: str = "serial"
    algorithm: str = "camuv"
    output_type: str = "pag"
    #: The endpoint marks that carry the pairs this method attributes to a hidden variable.
    confounded_mark: str = "bidirected"
    python_env: str = PYTHON_ENV_VAR

    def setup(self) -> None:
        """Import lingam before timing; see `contracts.SetupAdapter`."""
        import lingam  # noqa: F401

    def params(self) -> dict[str, Any]:
        """Return the settings in Andrey's names."""
        return dict(DEFAULTS)

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run lingam CAM-UV; return the PAG's endpoint-mark array. The only timed region.

        The returned matrix uses ``NULL=0``, ``TAIL=1``, ``ARROW=2``, and ``CIRCLE=3``.
        """
        import lingam

        X = np.asarray(data, dtype=np.float64)
        model = lingam.CAMUV(
            alpha=float(params["alpha"]), num_explanatory_vals=int(params["max_depth"]) + 1
        )
        B = np.asarray(model.fit(X).adjacency_matrix_)
        unknown = np.isnan(B)
        parents = [np.flatnonzero((B[i] != 0) & ~unknown[i]) for i in range(len(B))]
        pairs = zip(*np.nonzero(np.triu(unknown & unknown.T, k=1)), strict=True)
        return camuv_marks(parents, pairs, len(B))

    def to_structure(self, native: np.ndarray) -> Any:
        """Rebuild the PAG from its endpoint marks outside the timed region."""
        from andrey import GraphStructure

        return GraphStructure.from_numpy(native, kind=self.output_type)


def lingam_camuv_adapter() -> LingamCAMUV:
    """A :class:`LingamCAMUV` carrying its environment's lingam version."""
    return LingamCAMUV(package_version=env_version("lingam"))
