"""Andrey DirectLiNGAM solution adapters.

Wraps ``andrey.direct_lingam(data, measure="pwling")``, which returns a fully oriented DAG. Two
variants ship, numpy and ``torch-cuda``: the CUDA point is real here and absent from GES because the
``pwling`` order search's Hyvarinen entropy is DirectLiNGAM's one primitive with a torch kernel.

``weighted_adjacency[i, j]`` already weights edge ``i -> j`` (row = source), so the 0/1 reduction is
a bare nonzero threshold with **no transpose** -- unlike the pip lingam competitor. The device
pin is what makes the GPU run: ``entropy`` auto-selects torch only at batch ``B >= 1000`` and
DirectLiNGAM scores in far smaller batches, but a pinned backend bypasses that threshold. Without
CUDA the pin warns once and falls back to numpy rather than failing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

# Bench backend metadata -> Andrey device preference passed to ``andrey.config(backend=...)``. The
# runner exports the GPU variant's device as ``ANDREY_DEVICE=cuda`` in the child env; fit re-asserts
# the same choice via ``andrey.config`` (which outranks env), so the pin holds standalone too.
_NUMPY = "numpy"
_TORCH_CUDA = "torch-cuda"
_DEVICE_FOR_BACKEND = {_NUMPY: "numpy", _TORCH_CUDA: "cuda"}

# Pairwise likelihood-ratio independence measure — Andrey's own default, deterministic.
_DEFAULT_MEASURE = "pwling"


def _andrey_version() -> str:
    """The installed Andrey version (distribution metadata, else the module attribute)."""
    from importlib import metadata

    try:
        return metadata.version("andrey")
    except metadata.PackageNotFoundError:  # editable/dev checkout with no dist metadata
        import andrey

        return getattr(andrey, "__version__", "unknown")


@dataclass(frozen=True)
class AndreyDirectLiNGAM:
    """Solution adapter for one Andrey DirectLiNGAM ``backend`` variant (LiNGAM lane).

    Satisfies :class:`andrey_bench.contracts.SolutionAdapter`. ``backend`` is the runner-visible
    metadata (``numpy`` / ``torch-cuda``); :attr:`device` is the Andrey device preference
    (``numpy`` / ``cuda``) that :meth:`fit` pins. Build the variants with
    :func:`andrey_direct_lingam_adapters`.
    """

    name: str
    backend: str
    device: str
    package: str = "andrey"
    package_version: str = ""
    mode: str = "serial"
    algorithm: str = "lingam"
    output_type: str = "dag"

    def setup(self) -> None:
        """Import Andrey and its DirectLiNGAM implementation before timing.

        The facade loads `andrey.lingam.direct` on its first call. Setup excludes
        that import from `fit`, including the first timed fit at `--warmup 0`.
        See `contracts.SetupAdapter`.
        """
        import andrey  # noqa: F401
        from andrey.lingam.direct import direct_lingam  # noqa: F401

    def params(self) -> dict[str, Any]:
        """DirectLiNGAM's independence measure."""
        return {"measure": _DEFAULT_MEASURE}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run Andrey DirectLiNGAM under this variant's device; return the DAG's 0/1 adjacency.

        The only timed region. ``andrey.config(backend=self.device)`` pins the entropy kernel and
        outranks the runner's ``ANDREY_DEVICE`` env; the pin is also what bypasses the entropy size
        threshold, so the CUDA variant actually reaches the GPU. ``measure`` comes from ``params``
        (default ``pwling``, deterministic, so no seed applies). ``weighted_adjacency`` is already
        in canonical ``i -> j`` orientation, so the 0/1 reduction needs no transpose.
        """
        import andrey

        X = np.asarray(data, dtype=np.float64)
        measure = params.get("measure", _DEFAULT_MEASURE)
        with andrey.config(backend=self.device):
            out = andrey.direct_lingam(X, measure=measure)

        # weighted_adjacency[i, j] weights edge i -> j (canonical, row=source) == Andrey adj[s, t].
        w = out.weighted_adjacency
        if w is None:  # unweighted envelope: fall back to the structure's own edge marks
            adj = np.asarray(out.structure.to_numpy()) != 0
        else:
            adj = np.asarray(w) != 0
        return np.ascontiguousarray(adj.astype(np.int8))

    def to_structure(self, native: np.ndarray) -> Any:
        """0/1 DAG adjacency -> Andrey ``GraphStructure`` via ``structure_from_adjacency``; untimed."""
        from andrey_bench.contracts import structure_from_adjacency

        return structure_from_adjacency(native, kind=self.output_type)


def andrey_direct_lingam_adapters() -> list[AndreyDirectLiNGAM]:
    """The Andrey DirectLiNGAM variants a benchmark puts on its solution list, one per backend.

    A ``numpy`` reference (pinned pure-numpy entropy) and a ``torch-cuda`` variant (pinned CUDA
    entropy). Both are serial: DirectLiNGAM's public facade exposes no worker axis.
    """
    version = _andrey_version()
    specs = (
        ("andrey.direct_lingam.numpy", _NUMPY),
        ("andrey.direct_lingam.torch-cuda", _TORCH_CUDA),
    )
    return [
        AndreyDirectLiNGAM(
            name=name,
            backend=backend,
            device=_DEVICE_FOR_BACKEND[backend],
            package_version=version,
        )
        for name, backend in specs
    ]
