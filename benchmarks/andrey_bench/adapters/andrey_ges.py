"""Andrey GES benchmark adapters."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from andrey_bench.adapters.bic_penalty import ANDREY_EBIC_GAMMA, andrey_score

_DEFAULT_SCORE_FUNC = "local_score_BIC"
#: Matches causal-learn's 0.5 penalty after conversion to deviance units.
_DEFAULT_LAMBDA = 1.0
_DEFAULT_MAX_PARENTS = 4
# Worker count for the parallel variant when ANDREY_N_JOBS is unset or unreadable. An explicit 1
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
        n = int(os.environ.get("ANDREY_N_JOBS", "").strip())
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
    #: The local score. A variant with another score needs its own ``name``, as a weight does.
    score_func: str = _DEFAULT_SCORE_FUNC
    #: The limit on a node's parents, applied as ``parent_rule`` says; ``None`` sets none and keeps
    #: the default gate of half the variables, as the public ``andrey.ges`` does. A benchmark that
    #: changes it gives the arm its own ``name``.
    max_parents: float | None = _DEFAULT_MAX_PARENTS
    #: How ``max_parents`` applies. ``"cap"`` is the public ``andrey.ges(max_parents=k)``: no DAG of
    #: the CPDAG's class gives a node more than ``k`` parents. ``"gate"`` is the rule of
    #: the default of half the variables and of causal-learn's ``maxP``: a node that already holds
    #: more than ``k`` directed parents gains no new ones in the forward phase. A lane that limits
    #: the reference's parents too sets ``"gate"``, so both packages apply one rule; a lane that
    #: limits only Andrey's keeps ``"cap"``.
    parent_rule: str = "cap"

    def setup(self) -> None:
        """Import Andrey before timing; see `contracts.SetupAdapter`.

        The adapter is unpickled in a child that may not have Andrey loaded. Setup excludes
        its lazy `fit` import from timing at `--warmup 0`, matching adapters that load libraries
        in `setup`.
        """
        import andrey  # noqa: F401
        from andrey.search.ges import ges  # noqa: F401

    def params(self) -> dict[str, Any]:
        """The score, its BIC weight (BDeu has none), and the parent limit with its rule."""
        params: dict[str, Any] = {
            "score_func": self.score_func,
            "max_parents": self.max_parents,
        }
        if self.max_parents is not None:
            params["parent_rule"] = self.parent_rule
        if self.score_func != "local_score_BDeu":
            params["lambda_value"] = float(self.lambda_value)
        if self.score_func in ("local_score_BIC", "local_score_BIC_from_cov"):
            params["ebic_gamma"] = ANDREY_EBIC_GAMMA
        return params

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Return the CPDAG endpoint array for this backend and mode. The only timed region."""
        import andrey
        from andrey.search.ges import ges as _ges

        score_func = params.get("score_func", _DEFAULT_SCORE_FUNC)
        lambda_value = float(params.get("lambda_value", _DEFAULT_LAMBDA))
        max_parents = params.get("max_parents", params.get("maxP", _DEFAULT_MAX_PARENTS))
        parent_rule = params.get("parent_rule", self.parent_rule)

        workers = 1 if self.mode == "serial" else _parallel_workers()
        X = np.asarray(data, dtype=np.float64)
        with andrey.config(backend=self.backend, n_jobs=workers):
            cpdag, _score = _ges(
                X,
                score_func=andrey_score(score_func),
                penalty_discount=lambda_value,
                ebic_gamma=float(params.get("ebic_gamma", ANDREY_EBIC_GAMMA)),
                max_parents=max_parents,
                _parent_rule=parent_rule,
            )
        return np.asarray(cpdag.to_numpy())

    def to_structure(self, native: np.ndarray) -> Any:
        """Rebuild a ``GraphStructure`` from endpoint marks."""
        from andrey import GraphStructure

        return GraphStructure.from_numpy(native, kind=self.output_type)


def andrey_ges_adapters(
    lambda_value: float = _DEFAULT_LAMBDA, suffix: str = "", parent_rule: str = "cap"
) -> list[AndreyGES]:
    """Two GES benchmark configurations using the ``numpy`` backend: serial and parallel.

    ``lambda_value`` sets the BIC penalty weight for both configurations. ``suffix`` is appended
    to their names and must be nonempty for a nondefault weight: ``params`` are excluded from
    ``run_id``, so distinct weights need distinct names to keep their results separate.
    ``parent_rule`` is :attr:`AndreyGES.parent_rule`: ``"gate"`` in a lane whose reference limits
    its parents too. With no arguments, the function returns both configurations with the default
    weight, names, and rule.
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
    )
    return [
        AndreyGES(
            name=name + suffix,
            backend=backend,
            mode=mode,
            package_version=version,
            lambda_value=lambda_value,
            parent_rule=parent_rule,
        )
        for name, backend, mode in specs
    ]


def andrey_ges_bdeu() -> AndreyGES:
    """GES with the BDeu score for categorical data, using the ``numpy`` backend serially.

    BDeu moves are scored in the calling process, so a parallel variant would run the same code.
    Its reference, causal-learn's BDeu GES, limits parents too, so Andrey applies its limit as a
    gate.
    """
    return AndreyGES(
        name="andrey.ges.bdeu",
        backend="numpy",
        mode="serial",
        package_version=_andrey_version(),
        score_func="local_score_BDeu",
        parent_rule="gate",
    )
