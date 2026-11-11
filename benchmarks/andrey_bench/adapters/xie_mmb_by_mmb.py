"""MMB-by-MMB adapter for the authors' code (Xie et al., 2024), run unmodified in its own environment.

The code is ``fengxie009/MMB-by-MMB`` at commit :data:`COMMIT`, a repository with no license file.
It is a measurement reference only: ``benchmarks/start_mmb.sh`` downloads the files a fit imports,
checks each one's SHA-256, and builds the environment; the benchmark imports them from the
directory ``$ANDREY_BENCH_MMB`` names and copies none of them.

A fit calls ``MMB_by_MMB(data, target, alpha, p, maxK).mmb_by_mmb()`` as the repository's own
``__main__`` does, on the data as given, with the code's example settings: ``alpha = 0.01`` and
``maxK = 10``. The code runs Fisher's z test only, learns each blanket by total conditioning at
``alpha / (10 * floor(p / 10))`` for ``p >= 10``, and orients with Zhang's rules R1 to R4. It
draws no random numbers.

The answer is the target's row and column of the PAG the object holds when the call returns,
``run.pag``, which the code's own outputs (parents, children, bidirected neighbors, and the
undetermined) summarize. The code's encoding is ``pag[i, j]`` for the mark at ``j`` of the edge
between ``i`` and ``j``: ``1`` arrowhead, ``-1`` tail, ``2`` circle, ``0`` no edge. The adapter
turns it into Andrey's (:mod:`andrey_bench.contracts`) and keeps the target's edges. The number of
tests is the code's ``ci_test``: every test of every blanket and every adjacency search.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from andrey_bench.local_lane import TARGET_PARAM, pack_answer, unpack_answer

#: Names the directory holding the downloaded code.
CODE_ENV = "ANDREY_BENCH_MMB"

#: The pinned commit of ``fengxie009/MMB-by-MMB``; ``start_mmb.sh`` fetches it.
COMMIT = "2c63c890726fbe6f46e68850c4de52873810927f"

#: The files a fit imports, relative to the code directory.
REQUIRED_FILES = (
    "MMB_by_MMB.py",
    "Utils/CI_test.py",
    "Utils/MMB_TC_Z.py",
    "Utils/subsets1.py",
    "Utils/zhang_orient.py",
)

#: The code's example settings: the level of every test, and the largest conditioning set of the
#: adjacency search (``maxK``, which the repository calls the maximal degree of any variable).
DEFAULTS = {"alpha": 0.01, "max_k": 10}

# The code's marks of ``pag[i, j]``, the mark at ``j``, as Andrey's marks (tail, arrowhead, circle).
_FROM_CODE = {0: 0, -1: 1, 1: 2, 2: 3}


def _code_dir() -> Path:
    raw = os.environ.get(CODE_ENV, "").strip()
    directory = Path(raw)
    if not raw or not all((directory / name).is_file() for name in REQUIRED_FILES):
        raise FileNotFoundError(
            f"${CODE_ENV}={raw!r} does not hold the MMB-by-MMB files; run benchmarks/start_mmb.sh"
        )
    return directory


def _algorithm() -> Any:
    """The repository's ``MMB_by_MMB`` class, imported from the downloaded directory."""
    directory = str(_code_dir())
    if directory not in sys.path:
        sys.path.insert(0, directory)
    import MMB_by_MMB  # the repository's module of that name

    return MMB_by_MMB.MMB_by_MMB


def andrey_marks(code_pag: np.ndarray) -> np.ndarray:
    """The code's ``pag`` as Andrey's endpoint marks: ``marks[i, j]`` is the mark at ``i``."""
    out = np.zeros(code_pag.shape, dtype=np.int8)
    for code, mark in _FROM_CODE.items():
        out[code_pag.T == code] = mark
    return out


def at_target(marks: np.ndarray, target: int) -> np.ndarray:
    """``marks`` with every edge but the target's removed."""
    out = np.zeros_like(marks)
    out[target], out[:, target] = marks[target], marks[:, target]
    return out


class XieMMBbyMMB:
    """The authors' MMB-by-MMB at one target.

    Satisfies :class:`~andrey_bench.contracts.SolutionAdapter`,
    :class:`~andrey_bench.contracts.SetupAdapter`, and :class:`~andrey_bench.local_lane.LocalAdapter`.
    """

    name: str = "mmb-by-mmb.authors"
    package: str = "mmb-by-mmb"
    package_version: str = COMMIT[:12]
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "mmb_by_mmb"
    output_type: str = "pag"
    local: bool = True
    python_env: str = "ANDREY_BENCH_MMB_PYTHON"

    def setup(self) -> None:
        """Import the code outside fit timing."""
        _algorithm()

    def params(self) -> dict[str, Any]:
        """The code's example settings (:data:`DEFAULTS`)."""
        return dict(DEFAULTS)

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run the code at ``params["target"]``; return the packed marks and test count."""
        X = np.asarray(data, dtype=np.float64)
        target = int(params[TARGET_PARAM])
        settings = {**DEFAULTS, **params}
        run = _algorithm()(X, target, float(settings["alpha"]), X.shape[1], int(settings["max_k"]))
        *_, tests = run.mmb_by_mmb()
        return pack_answer(at_target(andrey_marks(np.asarray(run.pag)), target), tests)

    def to_structure(self, native: np.ndarray) -> Any:
        """The marks at the target as a PAG ``GraphStructure``; untimed."""
        from andrey import GraphStructure

        return GraphStructure.from_numpy(unpack_answer(native)[0], kind=self.output_type)
