"""Andrey GES benchmark adapters."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

_DEFAULT_SCORE_FUNC = "local_score_BIC"
#: Matches causal-learn's 0.5 penalty after conversion to deviance units.
_DEFAULT_LAMBDA = 1.0
_DEFAULT_MAX_PARENTS = 4
# Worker count for the parallel variant when ANDREY_NUM_WORKERS is unset or unreadable. An explicit 1
# is kept: the runner pins the core budget there, and the record reports it.
_PARALLEL_FALLBACK_WORKERS = 2


def _andrey_version() -> str:
    from importlib import metadata

    try:
        return metadata.version("andrey")
    except metadata.PackageNotFoundError:  # editable/dev checkout with no dist metadata
        import andrey

        return getattr(andrey, "__version__", "unknown")


def _parallel_workers() -> int:
    try:
        n = int(os.environ.get("ANDREY_NUM_WORKERS", "").strip())
    except ValueError:
        return _PARALLEL_FALLBACK_WORKERS
    return n if n == -1 or n >= 1 else _PARALLEL_FALLBACK_WORKERS


@dataclass(frozen=True)
class AndreyGES:
    """Benchmark adapter for one GES backend and execution mode."""

    name: str
    backend: str
    mode: str
    package: str = "andrey"
    package_version: str = ""
    algorithm: str = "ges"
    output_type: str = "cpdag"
    #: BIC complexity weight. Defaults to the shipped value, so a benchmark that does not mention it
    #: measures what a user gets. A benchmark comparing penalties passes it - and must give each
    #: value its own ``name``, because ``params`` are excluded from ``run_id`` and two weights under
    #: one name would collide in the part files.
    lambda_value: float = _DEFAULT_LAMBDA

    def setup(self) -> None:
        """Import Andrey before timing; see `contracts.SetupAdapter`.

        The adapter is unpickled in a child that may not have Andrey loaded. Setup excludes
        its lazy `fit` import from timing at `--warmup 0`, matching adapters that load libraries
        in `setup`.
        """
        import andrey  # noqa: F401
        from andrey.search.ges import ges  # noqa: F401

    def params(self) -> dict[str, Any]:
        return {
            "score_func": _DEFAULT_SCORE_FUNC,
            "lambda_value": float(self.lambda_value),
            "max_parents": _DEFAULT_MAX_PARENTS,
        }

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Return the CPDAG endpoint array for this backend and mode. The only timed region."""
        import andrey
        from andrey.search.ges import ges as _ges

        score_func = params.get("score_func", _DEFAULT_SCORE_FUNC)
        lambda_value = float(params.get("lambda_value", _DEFAULT_LAMBDA))
        max_parents = params.get("max_parents", params.get("maxP", _DEFAULT_MAX_PARENTS))

        workers = 1 if self.mode == "serial" else _parallel_workers()
        X = np.asarray(data, dtype=np.float64)
        with andrey.config(backend=self.backend, num_workers=workers):
            cpdag, _score = _ges(
                X, score_func=score_func, lambda_value=lambda_value, maxP=max_parents
            )
        return np.asarray(cpdag.to_numpy())

    def to_structure(self, native: np.ndarray) -> Any:
        """Rebuild a ``GraphStructure`` from endpoint marks."""
        from andrey import GraphStructure

        return GraphStructure.from_numpy(native, kind=self.output_type)


def andrey_ges_adapters(lambda_value: float = _DEFAULT_LAMBDA, suffix: str = "") -> list[AndreyGES]:
    """The three shipped GES builds: numpy serial, numpy parallel, numba serial.

    ``lambda_value`` and ``suffix`` exist together and only together: a second penalty needs a
    second set of names, since ``params`` are excluded from ``run_id``. Called with no arguments,
    it returns the shipped weight under the shipped names.
    """
    if lambda_value != _DEFAULT_LAMBDA and not suffix:
        # Without a suffix the second weight reuses the first's names, so its part files are the
        # first's. A resume then reads them as already measured and skips the whole weight, and the
        # table publishes one penalty under two headings.
        raise ValueError(
            f"lambda_value={lambda_value} needs a name suffix: params are excluded from run_id, so "
            "two weights under one solution name collide in the part files"
        )
    version = _andrey_version()
    specs = (
        ("andrey.ges.numpy.serial", "numpy", "serial"),
        ("andrey.ges.numpy.parallel", "numpy", "parallel"),
        ("andrey.ges.numba.serial", "numba", "serial"),
    )
    return [
        AndreyGES(
            name=name + suffix,
            backend=backend,
            mode=mode,
            package_version=version,
            lambda_value=lambda_value,
        )
        for name, backend, mode in specs
    ]
