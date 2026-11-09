"""MVPC adapters and data: the authors' R code (Tu et al. 2019), run unmodified.

``benchmarks/start_mvpc_r.sh`` downloads the repository at a pinned commit, checks its SHA-256, and
installs the R packages it loads beside pcalg; ``$ANDREY_BENCH_MVPC_R`` names its ``src/``. The
repository has no license file, so the benchmark ``source()``s its files from there and copies none
of them. Every call follows the repository's ``demo.R``:

- :class:`MvpcR` with ``variant="permc"`` runs ``mvpc(list(data = X), gaussCItest.td,
  gaussCItest.permc, alpha, p)``: test-wise deletion PC, the parents of each missingness indicator,
  and the permutation-based correction (PermC). The parents come from the repository's
  ``detection_prt_m(X, gaussCItest.td, alpha, p)``, passed to ``mvpc`` as ``prt_m``, which is the
  call ``mvpc`` makes itself without one, so the graph is the same and the parents are returned.
- ``variant="testwise"`` runs pcalg's ``pc(list(data = X), gaussCItest.td, alpha, p)``, the
  test-wise deletion PC of the repository.

``fit`` returns an int8 ``(2, d, d)`` stack: the CPDAG's 0/1 adjacency, row -> column, and the
detected parents, ``[1][j, i] = 1`` when column ``j`` is a parent of column ``i``'s missingness
indicator (all 0 for the test-wise variant). ``R``'s generator is seeded with ``params["seed"]``
before each fit, since PermC shuffles rows with ``sample()``.

:func:`paper_data` draws one dataset with the repository's ``gen_data``, the generator of the
paper's simulations (``src/SyntheticDataGeneration.R``), for the ``mvpc2019`` keys of
:mod:`andrey_bench.datasets`.

References
----------
.. [1] Tu, R., Zhang, C., Ackermann, P., Mohan, K., Kjellstrom, H., and Zhang, K. (2019). Causal
   discovery in the presence of missing data. Proceedings of Machine Learning Research 89
   (AISTATS), 1762-1770.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from andrey_bench import rsession

#: The variable naming the repository's ``src/`` directory.
CODE_ENV = "ANDREY_BENCH_MVPC_R"
#: The commit ``start_mvpc_r.sh`` pins; the adapters report it as their version.
COMMIT = "e95160c00987f540e47af776bbd7b6a1d9c25d5b"
#: The files ``demo.R`` sources, in its order.
SOURCES = ("CITest.R", "Evaluation.R", "MissingValuePC.R", "SyntheticDataGeneration.R")
#: The variants: MVPC with PermC, and test-wise deletion PC.
VARIANTS = ("permc", "testwise")
#: ``gen_data``'s ``(num_extra_e, num_m)`` by number of variables, from the repository's
#: ``exps/conti/NoV.R`` (20 and 50 variables); ``exps/conti/NoS_mar.R`` uses the first at 20.
PAPER_SIZES = {20: (5, 10), 50: (8, 16)}

# The entry points. `amat.cpdag` writes `i -> j` at [j, i], so the transpose reads row -> column.
_R_SOURCE = """
andrey_mvpc_r <- function(X, alpha, variant, seed) {
    set.seed(seed)
    p <- ncol(X)
    names <- colnames(X)
    data <- unname(X)
    prt <- matrix(0, p, p)
    if (variant == "testwise") {
        fit <- pc(list(data = data), gaussCItest.td, alpha = alpha, p = p)
    } else {
        prt_m <- detection_prt_m(data, gaussCItest.td, alpha, p)
        fit <- mvpc(list(data = data), gaussCItest.td, gaussCItest.permc, alpha = alpha, p = p,
                    prt_m = prt_m)
        for (k in seq_along(prt_m$m)) for (j in prt_m$prt[[k]]) prt[j, prt_m$m[[k]]] <- 1
    }
    amat <- t(as(fit, "amat")) * 1
    out <- rbind(amat, prt)
    dimnames(out) <- list(c(names, paste0(names, ".R")), names)
    out
}

andrey_mvpc_gen <- function(num_sample, mode, num_var, num_extra_e, num_m, seed) {
    g <- gen_data(num_sample = num_sample, mode = mode, num_var = num_var,
                  num_extra_e = num_extra_e, num_m = num_m, seed = seed)
    dag <- as(g$ground_truth$dag, "matrix")
    list(data_m = as.matrix(g$data_m), data_complete = as.matrix(g$data_complete),
         data_ref = as.matrix(g$data_ref), dag = (dag != 0) * 1,
         m_ind = as.integer(g$ground_truth$m_ind),
         parent_m_ind = as.integer(g$ground_truth$parent_m_ind))
}
"""


_SOURCED = False


def _start() -> rsession.RSession:
    """Start R, source the repository's files as ``demo.R`` does, and define the entry points.

    Untimed and idempotent: a process sources the files once.
    """
    global _SOURCED
    raw = os.environ.get(CODE_ENV, "").strip()
    src = Path(raw)
    if not raw or not (src / "MissingValuePC.R").is_file():
        raise FileNotFoundError(f"${CODE_ENV}={raw!r} is not MVPC's src/; run start_mvpc_r.sh")
    session = rsession.start("pcalg")
    if not _SOURCED:
        for name in SOURCES:
            session.define(f'source("{(src / name).as_posix()}")')
        session.define(_R_SOURCE)
        _SOURCED = True
    return session


class MvpcR:
    """The authors' R MVPC with PermC, or their test-wise deletion PC.

    Satisfies :class:`~andrey_bench.contracts.SolutionAdapter` and
    :class:`~andrey_bench.contracts.SetupAdapter`.
    """

    package: str = "mvpc-r"
    package_version: str = COMMIT[:12]
    backend: str = "r"
    mode: str = "serial"
    algorithm: str = "pc"
    output_type: str = "cpdag"

    def __init__(self, variant: str = "permc") -> None:
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        self.variant = variant
        self.name = f"mvpc-r.{variant}"

    def setup(self) -> None:
        """Start R and source the repository outside fit timing."""
        _start()

    def params(self) -> dict[str, Any]:
        """``alpha = 0.01``, the value of the repository's ``demo.R``, and seed 0."""
        return {"alpha": 0.01, "seed": 0}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run the variant on `data`, ``NaN`` where a value is missing; return the ``(2, d, d)`` stack."""
        session = rsession.session()
        X = np.asarray(data, dtype=np.float64)
        d = X.shape[1]
        names = rsession.columns(d)
        out = session.function("andrey_mvpc_r")(
            session.matrix(X), float(params["alpha"]), self.variant, int(params["seed"])
        )
        stacked = np.asarray(out)
        if stacked.shape != (2 * d, d):
            raise ValueError(f"expected a {2 * d}x{d} matrix from R, got {stacked.shape}")
        rows, cols = session._dimnames(out) or ([], [])
        if list(cols) != names or list(rows[:d]) != names:
            raise ValueError("R returned the variables in another order")
        return np.stack([stacked[:d], stacked[d:]]).astype(np.int8)

    def to_structure(self, native: np.ndarray) -> Any:
        """The CPDAG slab to a ``GraphStructure``; untimed."""
        from andrey_bench.contracts import structure_from_adjacency

        return structure_from_adjacency(np.asarray(native)[0], kind=self.output_type)

    def detected_parents(self, native: np.ndarray) -> np.ndarray | None:
        """The detected parents of the missingness indicators; ``None`` for test-wise PC."""
        return np.asarray(native)[1] if self.variant == "permc" else None


@dataclass(frozen=True)
class PaperDraw:
    """One draw of the paper's generator.

    ``data`` holds ``NaN`` where the mechanism removed a value, ``reference`` the same column counts
    removed completely at random (the paper's MCAR "target"), and ``complete`` the values before
    either. ``dag[i, j] = 1`` for an edge ``i -> j``; ``parents`` maps each column that can go
    missing to the parents of its indicator, by 0-based column.
    """

    data: np.ndarray
    reference: np.ndarray
    complete: np.ndarray
    dag: np.ndarray
    parents: dict[int, list[int]]


def paper_data(*, n: int, mode: str, d: int, num_extra_e: int, num_m: int, seed: int) -> PaperDraw:
    """Draw one dataset with the repository's ``gen_data``, starting R if needed.

    ``mode`` is ``"mar"`` or ``"mnar"``; ``num_extra_e`` and ``num_m`` are the generator's
    ``num_extra_e`` (indicators placed at a collider's parent) and ``num_m`` (indicators drawn).
    The generator's other arguments keep their defaults: ``p_missing_h = 0.9`` and
    ``p_missing_l = 0.1``, the probabilities of a missing value when the indicator's parent is
    below its quantile and when it is not.
    """
    if mode not in ("mar", "mnar"):
        raise ValueError(f"mode must be 'mar' or 'mnar', got {mode!r}")
    session = _start()
    g = session.function("andrey_mvpc_gen")(n, mode, d, num_extra_e, num_m, seed)
    get = dict(zip(g.names, g))
    parents: dict[int, list[int]] = {}
    for m, pa in zip(np.asarray(get["m_ind"]), np.asarray(get["parent_m_ind"])):
        parents.setdefault(int(m) - 1, [])
        if int(pa) - 1 not in parents[int(m) - 1]:
            parents[int(m) - 1].append(int(pa) - 1)

    def array(name: str) -> np.ndarray:
        return np.ascontiguousarray(np.asarray(get[name], dtype=np.float64))

    return PaperDraw(
        data=array("data_m"),
        reference=array("data_ref"),
        complete=array("data_complete"),
        dag=np.asarray(get["dag"]).astype(np.int8),
        parents=parents,
    )
