"""Andrey GnIES adapter: the facade, called with ``env=`` and ``targets=`` as a user calls it.

Each arm is a setting of :func:`andrey.gnies`; :data:`ARMS` lists the lane's. The known-target
arms pass the full true mapping, the observational environment included with no targets, so the
targets are fixed and one inner search runs. During the call a counting handler sits on the
``andrey.search.gnies`` and ``andrey.search.gies`` loggers at DEBUG, so the record says how often
the score's guards made a family unavailable, how often the alternating fit reached its step cap,
how many target families kept one residual degree of freedom, how many inner searches ran, how
many outer steps were ties, and how many inner steps the tie ranks decided.
"""

from __future__ import annotations

import contextlib
import logging
import time
import warnings
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from andrey_bench.adapters.andrey_hc import _andrey_version

#: Each arm's settings: facade arguments, plus ``known`` (pass the true targets) and ``tie_break``.
ARMS: dict[str, dict[str, Any]] = {
    "andrey.gnies": {},
    "andrey.gnies.bic": {"ebic_gamma": 0.0},
    "andrey.gnies.known_bic": {"ebic_gamma": 0.0, "known": True},
    "andrey.gnies.known": {"known": True},
    "andrey.gnies.position": {"tie_break": "position"},
}

#: The loggers whose DEBUG records are counted.
LOGGERS = ("andrey.search.gnies", "andrey.search.gies")

#: Count keys, each with a phrase of the DEBUG record it counts.
RECORDS = {
    "unavailable": "unavailable",
    "fit_cap": "alternating fit reached",
    "one_dof": "one residual degree of freedom",
    "inner_searches": "inner search with targets",
    "outer_ties": "outer step tie",
    "inner_ties": "two moves tie",
}

#: Andrey's version, for the record.
VERSION = _andrey_version()


class _Counting(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.counts: Counter[str] = Counter()

    def emit(self, record: logging.LogRecord) -> None:
        text = str(record.msg)
        key = next((k for k, phrase in RECORDS.items() if phrase in text), "other")
        self.counts[key] += 1


def fit(
    X: np.ndarray,
    env: np.ndarray,
    arm: str,
    true_targets: Sequence[Sequence[int]] | None = None,
) -> dict[str, Any]:
    """Run one arm and return its marks, learned targets, own total, seconds, and counts.

    ``env`` holds each row's environment code; ``true_targets[k]`` lists environment ``k``'s true
    targets and is read only by the known-target arms. ``deviance`` is the result's total under
    the arm's own settings (``metadata["score"]``), lower is better.
    """
    import andrey

    settings = dict(ARMS[arm])
    known = settings.pop("known", False)
    tie_break = settings.pop("tie_break", None)
    if known:
        if true_targets is None:
            raise ValueError(f"{arm} needs the true targets")
        settings["targets"] = {k: [int(j) for j in t] for k, t in enumerate(true_targets)}
    handler = _Counting()
    loggers = [logging.getLogger(name) for name in LOGGERS]
    levels = [logger.level for logger in loggers]
    for logger in loggers:
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", andrey.ExperimentalWarning)
            with andrey.config(tie_break=tie_break) if tie_break else contextlib.nullcontext():
                start = time.perf_counter()
                out = andrey.gnies(X, env, **settings)
                seconds = time.perf_counter() - start
    finally:
        for logger, level in zip(loggers, levels, strict=True):
            logger.removeHandler(handler)
            logger.setLevel(level)
    return {
        "marks": np.asarray(out.structure.to_numpy(), dtype=np.int8),
        "targets": sorted(int(j) for j in out.metadata["changing_variables"]),
        "deviance": float(out.metadata["score"]),
        "seconds": seconds,
        "counts": dict(handler.counts),
    }


def settings(arm: str) -> Mapping[str, float]:
    """The arm's ``penalty_discount`` and ``ebic_gamma``, the facade's defaults where unset."""
    chosen = ARMS[arm]
    return {
        "penalty_discount": float(chosen.get("penalty_discount", 1.0)),
        "ebic_gamma": float(chosen.get("ebic_gamma", 1.0)),
    }
