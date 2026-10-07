"""The warning an experimental method gives on its first call in a process."""

from __future__ import annotations

from andrey.core.warning_policy import ExperimentalWarning, warn_once

__all__ = ["ExperimentalWarning", "warn_experimental"]


def warn_experimental(name: str) -> None:
    """Warn on the first call of experimental method ``name`` in this process."""
    warn_once(
        f"andrey.{name} is experimental: it has no published benchmark, and its API may change "
        "without deprecation.",
        ExperimentalWarning,
    )
