"""Adapter for the XGES score-based search.

Extremely Greedy Equivalence Search (Nazaret and Blei, UAI 2024), Python/Numba package. The
separate C++ implementation is a different artifact and is not measured here.

**It runs in an environment of its own.** ``xges 0.1.6`` pins ``numpy<2`` and ``numba<0.60``, which
bench-env (numpy 2, Python 3.13) cannot satisfy, so this adapter declares ``python_env`` and its
fits are spawned in ``.venv-xges`` instead (``benchmarks/start_xges.sh``). Only ``fit`` crosses that
line: it returns a plain adjacency array, and :meth:`to_structure` runs back in the parent, so
there is still one graph model and one scorer over the whole table.

Penalty. ``alpha`` scales the same quantity as Andrey's ``lambda_value``. XGES's local score is
``-0.5*n*(1 + log sigma) - 0.5*log(n)*(k+1)*alpha`` in log-likelihood units; Andrey's is
``n*log(R) + log(n)*k*lambda_value`` in deviance units, which is the same expression times ``-0.5``
apart from terms that do not vary with the parent set (the ``-0.5*n``, the ``+1`` in ``(k+1)``, and
the covariance ``ddof``). Those cancel in every insert, delete, and reverse delta, so matching the
two knobs numerically matches the two penalties. XGES's own default is ``alpha=2.0``, twice
Andrey's; :data:`DEFAULT_ALPHA` here is the matched value, not the package's.

**No parent cap.** XGES has no ``maxP`` and no equivalent - its search is bounded by the score
alone, while Andrey's GES adapter passes ``maxP=4``. ``params`` records ``max_parents: None`` so
the asymmetry is in every row rather than in a footnote. It is a difference between the algorithms
as offered, not a setting left unmatched.

Orientation (``PDAG.to_adjacency_matrix``, verified against the source): ``adj[x,y]=1`` means the
directed edge ``x->y`` (row = source), and an undirected CPDAG edge is a symmetric pair of ``1``s.
That is Andrey's own orientation, so ``structure_from_adjacency`` reconstructs it without a
transpose.
"""

from __future__ import annotations

import functools
import os
import subprocess
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

#: Name of the environment variable holding the interpreter XGES fits run in. The runner reads the
#: variable; the adapter only names it, because the path is a fact about the machine.
PYTHON_ENV_VAR = "ANDREY_BENCH_XGES_PYTHON"

#: BIC penalty weight, on Andrey's ``lambda_value`` scale (see the module docstring). This is the
#: value that matches ``andrey.ges``'s shipped ``lambda_value=1.0``.
DEFAULT_ALPHA = 1.0

#: What ``pip install xges`` searches under, which is twice the penalty Andrey ships.
SHIPPED_ALPHA = 2.0

#: The two search settings the adapter fixes. ``extended_search=False`` would be XGES-0, the
#: paper's ablation; ``use_fast_numba=False`` would be the pure-Python scorer.
DEFAULT_EXTENDED_SEARCH = True
DEFAULT_USE_FAST_NUMBA = True

#: What ``use_fast_numba=True`` must actually select. XGES catches the ``ImportError`` from its own
#: Numba scorer and falls back to ``BICScorer`` with a log warning. A missing ``scipy``, which that
#: module imports but the package does not declare, would otherwise select a different scorer than
#: the record names.
_FAST_SCORER = "BICScorerFast"


@functools.lru_cache(maxsize=None)
def _xges_version(python: str) -> str:
    """``xges.__version__`` as reported by the interpreter that will run the fits.

    Read out of the child environment, not this one: the parent is bench-env, where xges is not
    installed and ``importlib.metadata`` would answer for the wrong environment or not at all.
    """
    try:
        out = subprocess.run(
            [python, "-c", "import xges; print(xges.__version__)"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() if out.returncode == 0 else "unknown"


def requested_threads() -> int:
    """The thread budget this fit was allotted, as the harness declared it.

    ``OMP_NUM_THREADS`` is what :func:`~andrey_bench.contracts.thread_env` pinned from
    ``split_cores``, so it is the budget rather than the machine. Reading the affinity mask instead
    would hand XGES every core the job holds while every other solution kept its split, and the
    table would compare two core budgets.
    """
    try:
        return max(1, int(os.environ["OMP_NUM_THREADS"]))
    except (KeyError, ValueError):
        return 1


def configure_numba_threads() -> int:
    """Pin ``NUMBA_NUM_THREADS`` to the allotted budget and return it.

    Must run before anything imports numba: the launch-time thread count is read once, when
    ``numba.config`` is first imported, and cannot be raised afterwards. XGES's fast scorer carries
    one ``parallel=True`` kernel, so an unpinned import would size that pool from the whole node.
    """
    threads = requested_threads()
    os.environ["NUMBA_NUM_THREADS"] = str(threads)
    return threads


@dataclass(frozen=True)
class XgesGES:
    """Benchmark adapter for the XGES Python/Numba package.

    Satisfies :class:`andrey_bench.contracts.SolutionAdapter`, plus the optional ``python_env``.
    """

    name: str = "xges.ges"
    package: str = "xges"
    package_version: str = ""
    backend: str = "numba"
    mode: str = "serial"
    algorithm: str = "ges"
    output_type: str = "cpdag"
    python_env: str = PYTHON_ENV_VAR
    alpha: float = DEFAULT_ALPHA

    def params(self) -> dict[str, Any]:
        """The search settings, including the parent cap XGES does not have."""
        return {
            "alpha": float(self.alpha),
            "extended_search": DEFAULT_EXTENDED_SEARCH,
            "use_fast_numba": DEFAULT_USE_FAST_NUMBA,
            # Not a default that happens to be unset: XGES offers no parent cap at all, while the
            # Andrey rows beside these carry max_parents=4.
            "max_parents": None,
        }

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run XGES and return a 0/1 CPDAG adjacency. Timed, and the only part in ``.venv-xges``.

        Both settings the campaign varies are checked against what the run actually got: numba's
        thread count against the allotted budget, and the scorer class against ``use_fast_numba``.
        Either mismatch raises, so the fit fails loudly instead of publishing a row that describes a
        path it did not take.
        """
        threads = configure_numba_threads()  # before the first numba import, via xges below

        import numba
        from xges import XGES

        if numba.get_num_threads() != threads:
            raise RuntimeError(
                f"numba runs {numba.get_num_threads()} threads, not the {threads} this fit was "
                "given: numba was imported before NUMBA_NUM_THREADS was set"
            )

        alpha = float(params.get("alpha", DEFAULT_ALPHA))
        extended_search = bool(params.get("extended_search", DEFAULT_EXTENDED_SEARCH))
        use_fast_numba = bool(params.get("use_fast_numba", DEFAULT_USE_FAST_NUMBA))

        if use_fast_numba:
            # Imported before the search because XGES catches this ImportError and continues with
            # ``BICScorer``. Checked only after the fit, a broken environment would first run the
            # slow search, which at a large size hits the wall cap and reads as a timeout.
            from xges.bic_scorer_fast import BICScorerFast  # noqa: F401

        X = np.asarray(data, dtype=np.float64)
        model = XGES(alpha=alpha)
        pdag = model.fit(
            X,
            extended_search=extended_search,
            use_fast_numba=use_fast_numba,
            verbose=0,
        )

        scorer = type(model.scorer).__name__
        if use_fast_numba and scorer != _FAST_SCORER:
            raise RuntimeError(
                f"use_fast_numba=True selected {scorer}, not {_FAST_SCORER}: xges falls back "
                "to the pure-Python scorer when its numba module will not import, and scipy is "
                "the dependency it needs but does not declare"
            )

        return (np.asarray(pdag.to_adjacency_matrix()) != 0).astype(np.int8)

    def to_structure(self, native: np.ndarray) -> Any:
        """0/1 CPDAG adjacency -> Andrey ``GraphStructure``. Runs in bench-env (parent); untimed."""
        from andrey_bench.contracts import structure_from_adjacency

        return structure_from_adjacency(native, kind=self.output_type)


def xges_adapter(alpha: float = DEFAULT_ALPHA, name: str = "xges.ges") -> XgesGES:
    """An :class:`XgesGES` carrying the version its own interpreter reports.

    Resolving that version needs a subprocess, so it happens when a benchmark builds its solution
    list rather than at import.

    ``name`` exists because ``params`` are **not** part of a ``run_id``: two penalties under one
    name would write to one part file and one of the two measurements would be lost to a resume.
    Every arm that changes a search setting needs a name of its own.
    """
    python = os.environ.get(PYTHON_ENV_VAR, "").strip()
    version = _xges_version(python) if python else "unknown"
    return XgesGES(name=name, package_version=version, alpha=alpha)
