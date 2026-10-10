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
    """A search stopped at its move limit, so a longer search might find a better graph.

    ``hc`` warns when it takes ``max_iter`` moves.
    """


def warn_once(message: str, category: type[AndreyWarning]) -> None:
    """Warn with ``message`` the first time this process sees it, at the caller's line."""
    key = (category, message)
    with _lock:
        if key in _warned:
            return
        warnings.warn(message, category, stacklevel=_caller_stacklevel())
        _warned.add(key)


def _reset_after_fork() -> None:
    """Replace the inherited lock in a forked child.

    A child forked while another thread is warning inherits the lock held, with no thread left to
    release it. The warned messages are kept: the child does not repeat what the parent showed.
    """
    global _lock
    _lock = threading.RLock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)


def reset() -> None:
    """Forget which messages have warned (test isolation)."""
    with _lock:
        _warned.clear()


def _caller_stacklevel() -> int:
    """The ``stacklevel``, counted from ``warn_once``, of the first frame outside the package."""
    frame = sys._getframe(2)
    level = 2
    while frame is not None and frame.f_code.co_filename.startswith(_PACKAGE_DIR):
        frame = frame.f_back
        level += 1
    return level
