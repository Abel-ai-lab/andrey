"""SCORE adapter for the authors' code (Rolland et al., 2022), run unmodified in its own environment.

The code is ``paulrolland1307/SCORE`` at commit :data:`COMMIT` (Apache-2.0): ``stein.py`` estimates
the diagonal of the Hessian of ``log p`` with the Stein identities and orders the variables, and
four R files taken from the CAM package prune the full DAG of that order with ``mgcv``. It is a
measurement reference only: ``benchmarks/start_stein.sh`` downloads the files, checks each one's
SHA-256, and builds the environment; the benchmark imports them from the directory
``$ANDREY_BENCH_SCORE`` names and copies none of them.

A fit does what the code's ``SCORE`` does at the defaults its ``demo_SCORE.py`` uses:
``compute_top_order`` with ``eta_G = eta_H = 0.001``, the variance of each Hessian column as the
leaf criterion, no division of a column by its mean (``SCORE`` passes ``normalize_var=False``), then
``cam_pruning.R`` at the cutoff ``0.001``, the paper's (Sect. 4.1). The code's ``cam_pruning``
reads the R script from a path on the authors' machine; the adapter runs the same script the same
way, through the code's own ``np_to_csv`` and the R launcher of the Causal Discovery Toolbox it
imports, with the script's placeholders filled in and R started in the code directory, where the
script ``source()``s its R helpers. The data go in as ``float32``, the type of the tensors the
authors' generator returns.

:class:`RollandSCOREOrder` stops after the order and returns its full DAG
(:mod:`~andrey_bench.adapters.order_dag`).
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

#: Names the directory holding the downloaded files.
CODE_ENV = "ANDREY_BENCH_SCORE"

#: The pinned commit of ``paulrolland1307/SCORE``; ``start_stein.sh`` fetches it.
COMMIT = "c2c2b8e0645b1a2a3970b5a16710ee776c4ed801"

#: The files a fit reads, relative to the code directory.
REQUIRED_FILES = ("stein.py", "utils.py", "cam_pruning.R", "pruning.R", "selGam.R", "train_gam.R")

_DEFAULTS: dict[str, Any] = {
    "eta_G": 0.001,
    "eta_H": 0.001,
    "normalize_var": False,
    "dispersion": "var",
    "cutoff": 0.001,
}


def code_dir() -> Path:
    """The directory of the authors' files, or ``SystemExit`` naming what is missing."""
    raw = os.environ.get(CODE_ENV, "").strip()
    if not raw:
        raise SystemExit(
            f"${CODE_ENV} is unset: the authors' SCORE code is a measurement reference that the "
            "benchmark reads from disk. Run benchmarks/start_stein.sh and export what it prints."
        )
    root = Path(raw)
    missing = [f for f in REQUIRED_FILES if not (root / f).is_file()]
    if missing:
        raise SystemExit(f"${CODE_ENV}={raw!r} lacks {', '.join(missing)}")
    return root


def _stein():
    """The authors' ``stein`` module, imported from :func:`code_dir`."""
    root = str(code_dir())
    if root not in sys.path:
        sys.path.insert(0, root)
    import stein  # ty: ignore[unresolved-import]  # the downloaded code, on sys.path above

    return stein


def cam_pruning(adjacency: np.ndarray, x: np.ndarray, cutoff: float) -> np.ndarray:
    """Run the authors' ``cam_pruning.R`` on `adjacency` as their ``cam_pruning`` does.

    The CSV files go to a directory of their own, so concurrent fits never share one; R starts in
    the code directory, where ``cam_pruning.R`` finds the files it sources.
    """
    import pandas as pd
    from cdt.utils.R import launch_R_script  # ty: ignore[unresolved-import]

    stein = _stein()
    root = code_dir()
    with tempfile.TemporaryDirectory(prefix="score-pruning-") as work:
        save_path = os.path.join(work, "")
        arguments = {
            "{PATH_DATA}": stein.np_to_csv(np.asarray(x), save_path),
            "{PATH_DAG}": stein.np_to_csv(adjacency, save_path),
            "{PATH_RESULTS}": os.path.join(work, "results.csv"),
            "{ADJFULL_RESULTS}": os.path.join(work, "adjfull.csv"),
            "{CUTOFF}": str(cutoff),
            "{VERBOSE}": "TRUE",
        }
        before = os.getcwd()
        os.chdir(root)
        try:
            return launch_R_script(
                str(root / "cam_pruning.R"),
                arguments,
                output_function=lambda: pd.read_csv(arguments["{PATH_RESULTS}"]).values,
            )
        finally:
            os.chdir(before)


class RollandSCORE:
    """The authors' SCORE: the Stein-Hessian order, then CAM pruning in R.

    Satisfies :class:`~andrey_bench.contracts.SolutionAdapter` and
    :class:`~andrey_bench.contracts.SetupAdapter`.
    """

    name: str = "rolland.score"
    package: str = "SCORE"
    package_version: str = COMMIT[:8]
    backend: str = "torch-cpu"
    mode: str = "serial"
    algorithm: str = "score"
    output_type: str = "dag"
    python_env: str = "ANDREY_BENCH_STEIN_PYTHON"
    prune: bool = True

    def setup(self) -> None:
        """Import the code and fit a tiny table before timing, so first-call costs stay out."""
        from andrey_bench.adapters.fcm_warm import tiny_table

        self.fit(tiny_table(), self.params())

    def params(self) -> dict[str, Any]:
        """The defaults of the code's ``SCORE``, the paper's cutoff, and the pinned commit."""
        return {**_DEFAULTS, "commit": COMMIT, "dtype": "float32", "prune": self.prune}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run the authors' SCORE and return a 0/1 DAG adjacency. The only timed region."""
        import torch

        stein = _stein()
        x = torch.tensor(np.asarray(data), dtype=torch.float32)
        order = stein.compute_top_order(
            x, params["eta_G"], params["eta_H"], params["normalize_var"], params["dispersion"]
        )
        full = stein.full_DAG(order)
        if not params["prune"]:
            return full.astype(np.int8)
        return np.asarray(cam_pruning(full, x.numpy(), params["cutoff"]), dtype=np.int8)

    def to_structure(self, native: np.ndarray) -> Any:
        """0/1 DAG adjacency -> Andrey ``GraphStructure``; untimed."""
        from andrey_bench.contracts import structure_from_adjacency

        return structure_from_adjacency(native, kind=self.output_type)


class RollandSCOREOrder(RollandSCORE):
    """The authors' SCORE order alone, as its full DAG."""

    name: str = "rolland.score.order"
    prune: bool = False
