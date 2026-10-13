"""Coefficient stacks: the weighted lag blocks VARMA-LiNGAM and longitudinal LiNGAM return.

Both packages report their estimates as matrices in the ``[effect, cause]`` layout:
``block[i, j]`` weights the effect of variable ``j`` on variable ``i``, and a zero is no link.

- ``varma``: an ``(3, n, n)`` stack ``[B0, B1, M1]``, the instantaneous block, the lag-1
  autoregressive block, and the lag-1 moving-average block.
- ``panel``: a ``(T, 2, n, n)`` stack, ``[t, 0]`` the instantaneous block ``B(t,t)`` and
  ``[t, 1]`` the lag block ``B(t,t-1)`` at time point ``t``; the first time point is not estimated
  and holds ``NaN`` in an estimate.

This module imports nothing but numpy, because the worker in the lingam environment imports it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

KINDS = ("varma", "panel")


@dataclass(frozen=True, eq=False)
class CoefficientStack:
    """A coefficient stack and its kind; see the module docstring."""

    stack: np.ndarray
    kind: str

    def __post_init__(self) -> None:
        stack = np.asarray(self.stack, dtype=np.float64)
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
        wanted = 3 if self.kind == "varma" else 4
        if stack.ndim != wanted or stack.shape[-1] != stack.shape[-2]:
            raise ValueError(f"a {self.kind} stack has {wanted} axes, got shape {stack.shape}")
        object.__setattr__(self, "stack", stack)

    @property
    def n_nodes(self) -> int:
        """The number of variables."""
        return int(self.stack.shape[-1])

    def support(self) -> np.ndarray:
        """The links: nonzero, finite coefficients."""
        return np.isfinite(self.stack) & (self.stack != 0)
