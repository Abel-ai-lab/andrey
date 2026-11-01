"""RESIT adapter for the authors' R code (Peters, Mooij, Janzing, and Scholkopf, 2014).

The code is supplementary material to the paper, distributed with a two-clause BSD license as
``onlineCodeANM.zip``. It is a measurement reference only: the benchmark ``source()``s the files
from a directory the environment names and copies none of them. ``benchmarks/start_resit_r.sh``
downloads the archive, checks its SHA-256, and prints the export.

``ICML`` is the code's name for RESIT. The adapter calls it as the paper's experiments do:
``model = train_gam`` (``mgcv::gam`` with ``k = 10`` basis functions per term, GCV smoothing),
``indtest = indtestHsic`` with its default ``method = "ExactFastTrace"`` (the gamma approximation of
the null, a Gaussian kernel with the median-heuristic bandwidth), and ``alpha = 0.05``. The function's
own default regression is ``train_linear``.

Differences from Andrey's RESIT are named, not tuned away:

- the code's ``indtestHsic`` draws a random subsample of 1000 rows when ``n > 1000``, so
  ``n = 1000`` keeps it deterministic and the benchmark stays there;
- the code compares the p-values themselves, where Andrey compares their logarithms, so p-values
  below the smallest double tie at zero and the first variable wins;
- the kernels read the columns as they are, with the bandwidth from the median of all pairwise
  squared distances; Andrey reads them centered and divided by their standard deviations, with the
  median over pairs at a positive distance;
- smoothing parameters come from mgcv's GCV optimizer, Andrey's from its own, so near-tied fits can
  land on different local optima.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from andrey_bench.adapters.r_adapter import RAdapter

#: Names the ``code`` directory of the extracted archive (``codeANM/code``).
CODE_ENV = "ANDREY_BENCH_RESIT_R"

#: Files ``source()`` reads, relative to the code directory.
REQUIRED_FILES = (
    "startups/startupICML.R",
    "inferDAG/ICML.R",
    "fitting/fitting.R",
    "indtests/indtestHsic.R",
)

_R_SOURCE = """
source(file.path(Sys.getenv("ANDREY_BENCH_RESIT_R"), "startups", "startupICML.R"), chdir = TRUE)
andrey_peters_resit <- function(X) {
    # `C[i, j] = 1` for i -> j, already the orientation the benchmark scores.
    C <- ICML(X, alpha = 0.05, model = train_gam, indtest = indtestHsic, output = FALSE)
    C <- matrix(as.numeric(C == 1), nrow(C), ncol(C))
    dimnames(C) <- list(colnames(X), colnames(X))
    C
}
"""


def code_dir() -> Path:
    """The authors' ``code`` directory, or ``SystemExit`` naming what is missing."""
    raw = os.environ.get(CODE_ENV, "").strip()
    if not raw:
        raise SystemExit(
            f"${CODE_ENV} is unset: the authors' RESIT code is a measurement reference that the "
            "benchmark reads from disk. Run benchmarks/start_resit_r.sh and export what it prints."
        )
    root = Path(raw)
    missing = [f for f in REQUIRED_FILES if not (root / f).is_file()]
    if missing:
        raise SystemExit(f"${CODE_ENV}={raw!r} lacks {', '.join(missing)}")
    return root


def archive_sha256() -> str:
    """The SHA-256 of ``onlineCodeANM.zip`` that ``start_resit_r.sh`` verified, or ``unknown``."""
    raw = os.environ.get(CODE_ENV, "").strip()
    marker = Path(raw).parent / "ARCHIVE_SHA256" if raw else None
    return marker.read_text().strip() if marker is not None and marker.is_file() else "unknown"


class PetersRESIT(RAdapter):
    """The authors' R implementation of RESIT through the embedded R session.

    Satisfies :class:`~andrey_bench.contracts.SolutionAdapter` and
    :class:`~andrey_bench.contracts.SetupAdapter`.
    """

    name: str = "peters.resit"
    package: str = "onlineCodeANM"
    package_version: str = "2014 supplementary code"
    backend: str = "native"
    mode: str = "serial"
    algorithm: str = "resit"
    output_type: str = "dag"

    r_package: str = "mgcv"
    r_source: str = _R_SOURCE
    r_function: str = "andrey_peters_resit"

    def setup(self) -> None:
        """Start R, then fit a tiny table before timing, so first-call costs stay out of the fit."""
        from andrey_bench.adapters.fcm_warm import tiny_table

        super().setup()
        self.call(tiny_table())

    def params(self) -> dict[str, Any]:
        """The paper's experiment settings."""
        return {
            "archive_sha256": archive_sha256(),
            "alpha": 0.05,
            "model": "train_gam",
            "numBasisFcts": 10,
            "indtest": "indtestHsic",
            "indtest_method": "ExactFastTrace",
        }

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run the authors' RESIT and return a 0/1 DAG adjacency. The only timed region."""
        return self.call(data).astype(np.int8)

    def to_structure(self, native: np.ndarray) -> Any:
        """0/1 DAG adjacency -> Andrey ``GraphStructure``; untimed."""
        from andrey_bench.contracts import structure_from_adjacency

        return structure_from_adjacency(native, kind=self.output_type)
