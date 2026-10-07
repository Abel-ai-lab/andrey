"""lingam DirectLiNGAM adapter — the LiNGAM-lane competitor.

Calls ``lingam.DirectLiNGAM().fit(X)``, which returns a fully oriented DAG. It takes no search
parameters and is deterministic given the data, so no seed applies either.

Orientation (verified empirically, lingam 1.12.2): ``adjacency_matrix_`` is the weighted ``B`` of
the structural equation ``x = B x + e``, so ``B[i,j] != 0`` means ``j`` is a direct cause of ``i``
(edge ``j->i``, row = effect, column = cause) — the *opposite* indexing from Andrey's, where
``adj[s,t]=1`` means ``s->t``. The reduction therefore thresholds to nonzero and **transposes**.
DirectLiNGAM's output is acyclic and single-directional, so the result is a genuine DAG with no
symmetric entries.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def _lingam_version() -> str:
    """The installed lingam version (distribution metadata; the module exposes one too)."""
    from importlib import metadata

    try:
        return metadata.version("lingam")
    except metadata.PackageNotFoundError:  # pragma: no cover - installed in bench-env
        return "unknown"


class LingamDirect:
    """Solution adapter for lingam's DirectLiNGAM (LiNGAM lane).

    Satisfies :class:`andrey_bench.contracts.SolutionAdapter`.
    """

    name: str = "lingam.direct_lingam"
    package: str = "lingam"
    package_version: str = _lingam_version()
    backend: str = "native"
    mode: str = "serial"
    algorithm: str = "lingam"
    output_type: str = "dag"

    def setup(self) -> None:
        """Import lingam before timing; see `contracts.SetupAdapter`.

        The adapter is unpickled in the child, and `fit` imports lazily. Setup excludes that
        import from the first timed fit at `--warmup 0`.
        """
        import lingam  # noqa: F401

    def params(self) -> dict[str, Any]:
        """No knob changes the search — pip lingam's DirectLiNGAM is deterministic."""
        return {}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run DirectLiNGAM; return a plain 0/1 DAG adjacency. The timed region.

        DirectLiNGAM takes no search parameters and is deterministic, so ``params`` is
        unused. The weighted ``adjacency_matrix_`` (``B[i,j]≠0`` means the edge ``j->i``) is
        thresholded to nonzero and transposed into Andrey orientation, where ``adj[s,t]=1`` means
        ``s->t``.
        """
        import lingam

        X = np.asarray(data, dtype=np.float64)
        model = lingam.DirectLiNGAM()
        model.fit(X)

        # B[i,j]!=0 means j->i (row=effect, col=cause); Andrey wants adj[s,t]=1 to mean s->t, so
        # transpose.
        b = np.asarray(model.adjacency_matrix_)
        adj = (b != 0).astype(np.int8).T
        return np.ascontiguousarray(adj)

    def to_structure(self, native: np.ndarray) -> Any:
        """0/1 DAG adjacency -> Andrey ``GraphStructure``. Runs in bench-env (parent); untimed."""
        from andrey_bench.contracts import structure_from_adjacency

        return structure_from_adjacency(native, kind=self.output_type)
