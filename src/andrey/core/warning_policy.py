"""Andrey's warning categories and the one frequency rule they share.

Every warning Andrey raises is an :class:`AndreyWarning`, so one filter silences or escalates all of
them; each category below it can be filtered on its own. A warning never changes a result.

Each distinct message warns once per process, pointing at the first line outside the package. A
message is recorded only after :func:`warnings.warn` returns, so an ``error`` filter fails every
call rather than only the first.
"""

from __future__ import annotations

import os
import sys
import threading
import warnings
from collections.abc import Callable
from typing import TypeVar

_T = TypeVar("_T")

_PACKAGE_DIR = os.path.dirname(os.path.dirname(__file__)) + os.sep

_warned: set[tuple[type[Warning], str]] = set()
# Reentrant: a warning handler that warns again from the same thread must not deadlock.
_lock = threading.RLock()


class AndreyWarning(UserWarning):
    """Base class of every warning Andrey raises."""


class ExperimentalWarning(AndreyWarning):
    """A method without a published benchmark, whose API may change without deprecation."""


class PerformanceWarning(AndreyWarning):
    """A call that will run correctly but slowly.

    PC warns before a pass of ten million or more tests.
    """


class BackendFallbackWarning(AndreyWarning):
    """A requested compute backend is unavailable, so the call runs on another one."""


class SearchLimitWarning(AndreyWarning):
    """An iterative fit stopped at its iteration cap rather than on its own stopping rule, so a
    longer run might find a better graph.

    Every method with ``max_iter`` warns when it stops on the cap (:func:`warn_search_limit`), and
    a scikit-learn ``ConvergenceWarning`` from FastICA or LARS is raised again as this warning.
    """


class SampleSizeWarning(AndreyWarning):
    """A test runs below its minimum rows, so it may reject a true independence more often than
    ``alpha``.

    RCoT warns on a conditional query below 500 rows.
    """


class DataTypeWarning(AndreyWarning):
    """How a call reads its columns' types: inferred, missing, or possibly mistaken.

    A mixed-data test or score warns when it infers the categorical columns or finds none, and so
    runs as its linear-Gaussian form; a continuous one warns once when columns look like category
    codes. A score search left at ``score_func=None`` warns once when it picks BDeu because every
    column is categorical, or the conditional Gaussian BIC because some are.
    """


class BoundaryWarning(AndreyWarning):
    """A fit larger than the largest size the method finished within the caps of the published
    boundary lane, at the nearest measured setting.

    Each method warns once per process, whatever the size.
    """


class DeprecatedArgumentWarning(AndreyWarning):
    """A former argument name or value, which works for one more release as the current one."""


def warn_once(message: str, category: type[AndreyWarning], *, key: str | None = None) -> None:
    """Warn with ``message`` the first time this process sees it, at the caller's line.

    With ``key``, the warning is seen once per ``key`` instead, whatever the message.
    """
    seen = (category, message if key is None else key)
    with _lock:
        if seen in _warned:
            return
        warnings.warn(message, category, stacklevel=_caller_stacklevel())
        _warned.add(seen)


def warn_search_limit(method: str, parameter: str, value: int, unit: str) -> None:
    """Warn once that ``method`` stopped at ``parameter=value`` ``unit`` (its iteration cap)."""
    warn_once(
        f"{method} stopped at its limit of {parameter}={value} {unit}; a longer search may find a "
        f"better graph, so raise {parameter}",
        SearchLimitWarning,
    )


def convergence_as_search_limit(source: str, fit: Callable[[], _T]) -> _T:
    """Run ``fit``, one scikit-learn call, and return its result; raise each scikit-learn
    ``ConvergenceWarning`` it gives again as a :class:`SearchLimitWarning`, its text after
    ``source``. Other warnings pass through unchanged.

    One call only: :func:`warnings.catch_warnings` is not thread-safe.
    """
    from sklearn.exceptions import ConvergenceWarning

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        result = fit()
    for item in caught:
        if issubclass(item.category, ConvergenceWarning):
            warn_once(f"{source}: {item.message}", SearchLimitWarning)
        else:
            warnings.warn_explicit(item.message, item.category, item.filename, item.lineno)
    return result


def _reset_after_fork() -> None:
    """Replace the inherited lock in a forked child.

    A child forked while another thread is warning inherits the lock held, with no thread left to
    release it. The warned messages are kept: the child does not repeat what the parent showed.
    """
    global _lock
    _lock = threading.RLock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)


def _caller_stacklevel() -> int:
    """The ``stacklevel``, counted from ``warn_once``, of the first frame outside the package."""
    frame = sys._getframe(2)
    level = 2
    while frame is not None and frame.f_code.co_filename.startswith(_PACKAGE_DIR):
        frame = frame.f_back
        level += 1
    return level
