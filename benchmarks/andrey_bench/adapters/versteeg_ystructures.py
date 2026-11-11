"""Y-structure adapters for the authors' R code (Versteeg, Zhang, and Mooij, 2022), run unmodified.

The code is ``philipversteeg/sbcd`` at commit :data:`COMMIT`, under the BSD 3-clause license; the
benchmark runs it as a comparator and copies none of it. ``benchmarks/start_sbcd_r.sh`` downloads
the repository at the pinned commit, checks its SHA-256, and installs the R packages it lists
beside pcalg and RBGL; ``$ANDREY_BENCH_SBCD_R`` names the extracted repository. The adapter
``source()``s ``R/wrapYStructures.R``, the file ``simulate.R`` sources for the method, and calls its
``ystructures`` as ``simulate.R`` does: ``test = "gaussCItest"`` (pcalg's Fisher z test on the
correlation matrix), ``alpha = 0.01``, ``beta = NULL`` (the acceptance level is ``alpha``), and
``fullYStruc = TRUE`` for the Y-structure and ``FALSE`` for the extended one. The code draws no
random numbers.

``ystructures`` reads the number of variables ``p`` and of rows ``N`` from the global environment,
where ``simulate.R`` defines them, and ignores its ``systemVars`` argument; the entry point assigns
both from the data before every call, and passes the first ``p - 1`` columns as the system
variables and the last as the context, the layout of the paper's simulation
(:mod:`andrey_bench.selection_models`).

The answer is the code's ``arel``: ``arel[i, j]`` is the score of the claim that ``i`` is an
ancestor of ``j`` (the best ``min(-log p_uj, -log p_vj)`` over the patterns found), ``0`` for no
claim. The code counts no tests, so the answer carries none.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from andrey_bench import rsession
from andrey_bench.adapters.r_adapter import RAdapter
from andrey_bench.local_lane import pack_answer, unpack_answer

#: Names the extracted repository.
CODE_ENV = "ANDREY_BENCH_SBCD_R"

#: The pinned commit of ``philipversteeg/sbcd``; ``start_sbcd_r.sh`` fetches it.
COMMIT = "4bae8afb885382b7045cf2d67c25a88fd1c013e8"

#: The file a fit sources, relative to the repository.
SOURCE = "R/wrapYStructures.R"

#: The code's settings in ``simulate.R``: the level of the tests.
DEFAULTS = {"alpha": 0.01}

_R_SOURCE = """
andrey_sbcd_ystructures <- function(X, alpha, full) {
    assign("p", ncol(X), envir = .GlobalEnv)
    assign("N", nrow(X), envir = .GlobalEnv)
    fit <- suppressWarnings(ystructures(
        data = X, systemVars = seq_len(ncol(X) - 1L), contextVars = ncol(X), alpha = alpha,
        beta = NULL, test = "gaussCItest", fullYStruc = full, verbose = 0))
    out <- fit$arel
    dimnames(out) <- list(colnames(X), colnames(X))
    out
}
"""

_SOURCED = False


def _start() -> rsession.RSession:
    """Start R, source the repository's file, and define the entry point.

    Untimed and idempotent: a process sources the file once.
    """
    global _SOURCED
    raw = os.environ.get(CODE_ENV, "").strip()
    path = Path(raw) / SOURCE
    if not raw or not path.is_file():
        raise FileNotFoundError(
            f"${CODE_ENV}={raw!r} is not the sbcd repository; run benchmarks/start_sbcd_r.sh"
        )
    session = rsession.start("pcalg")
    if not _SOURCED:
        session.define(f'source("{path.as_posix()}")')
        session.define(_R_SOURCE)
        _SOURCED = True
    return session


class VersteegYStructures(RAdapter):
    """The authors' Y-structure method, in its two forms.

    ``extended=False`` is the Y-structure (YSt), ``True`` the extended one (YSt-Ext). Satisfies
    :class:`~andrey_bench.contracts.SolutionAdapter`,
    :class:`~andrey_bench.contracts.SetupAdapter`, and
    :class:`~andrey_bench.local_lane.ClaimAdapter`.
    """

    package: str = "sbcd"
    package_version: str = COMMIT[:12]
    backend: str = "r"
    mode: str = "serial"
    algorithm: str = "y_structures"
    output_type: str = "claims"
    r_package: str = "pcalg"
    r_source: str = _R_SOURCE
    r_function: str = "andrey_sbcd_ystructures"

    def __init__(self, extended: bool = False) -> None:
        self.extended = extended
        self.name = "sbcd.yst_ext" if extended else "sbcd.yst"

    def setup(self) -> None:
        """Start R and source the repository outside fit timing."""
        _start()

    def params(self) -> dict[str, Any]:
        """The code's settings in ``simulate.R`` (:data:`DEFAULTS`)."""
        return dict(DEFAULTS)

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run the code on ``data``, the context in the last column; return the claims' scores."""
        settings = {**DEFAULTS, **params}
        scores = self.call(data, float(settings["alpha"]), not self.extended)
        return pack_answer(np.asarray(scores, dtype=np.float64), None)

    def to_structure(self, native: np.ndarray) -> Any:
        """The empty graph over the columns: a claim is ancestry, not an edge; untimed."""
        from andrey import GraphStructure

        d = unpack_answer(native)[0].shape[0]
        return GraphStructure.from_numpy(np.zeros((d, d), dtype=np.int8), kind="digraph")
