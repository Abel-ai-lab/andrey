"""The environment and target arguments of ``gies`` and ``cdnod``.

``env`` gives each row's environment as a label, a string or a number. ``targets`` gives the
variables intervened on in each environment, in one of three forms: a mapping from each label of
``env`` to its variables, one cell per row (a DataFrame column the wrapper took out of the data,
:func:`~andrey.api._method.take_columns`), or an ``(n_samples, n_variables)`` 0/1 matrix. Each form
becomes one internal form, a code per row and one row of intervened variables per environment, and
``gies`` reads the row mask ``mask[i] = targets[env[i]]``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

from andrey.core.knowledge import column_position

if TYPE_CHECKING:
    import numpy.typing as npt


def env_codes(env: npt.ArrayLike, n: int, method: str) -> tuple[np.ndarray, list]:
    """Return each row's environment as a code, and the labels in the order of their codes.

    The codes are ``0..K-1`` for the ``K`` labels, in order of first appearance. Raises
    ``ValueError`` when ``env`` does not hold one label per row, as shape ``(n,)`` or ``(n, 1)``,
    when a label is missing (``None`` or ``NaN``), when a label is neither a string nor a number, or
    when ``env`` mixes strings and numbers.
    """
    import pandas as pd

    values = np.asarray(env, dtype=object)
    if values.shape not in ((n,), (n, 1)):
        raise ValueError(
            f"{method}: env must hold one label per row of data, as shape ({n},) or ({n}, 1); "
            f"got shape {values.shape}"
        )
    values = values.reshape(-1)
    try:
        codes, labels = pd.factorize(values)
    except TypeError:
        raise ValueError(f"{method}: env labels must be strings or numbers") from None
    if (codes < 0).any():
        row = int(np.flatnonzero(codes < 0)[0])
        raise ValueError(f"{method}: env has no label at row {row}; every row needs one")
    labels = list(labels)
    text = {isinstance(label, str) for label in labels}
    if not all(isinstance(label, (str, int, float, np.number)) for label in labels):
        raise ValueError(f"{method}: env labels must be strings or numbers")
    if len(text) > 1:
        raise ValueError(f"{method}: env mixes strings and numbers; give its labels one type")
    return codes.astype(np.intp), labels


def cdnod_index(env: npt.ArrayLike, n: int) -> npt.ArrayLike:
    """Return ``env`` as ``cdnod``'s index: numbers unchanged, string labels as codes.

    The codes are ``0.0, 1.0, ...`` in order of first appearance (:func:`env_codes`).
    """
    values = np.asarray(env)
    if values.dtype.kind in "US" or (
        values.dtype.kind == "O" and any(isinstance(v, str) for v in values.ravel())
    ):
        return env_codes(values, n, "cdnod")[0].astype(np.float64)
    return env


def intervened_rows(
    env: npt.ArrayLike | None,
    targets: Mapping | npt.ArrayLike | None,
    X: np.ndarray,
    labels: tuple[str, ...] | None,
) -> np.ndarray | None:
    """Return ``gies``'s intervention mask, or ``None`` for observational data.

    The mask is a boolean ``(n_samples, n_variables)`` matrix, ``True`` where the row's value of the
    variable was set by an intervention. ``targets`` names variables by label (a string, among
    ``labels``) or by 0-based position. A mapping needs ``env``, and every label of ``env`` needs an
    entry that is not ``None``. One cell per row, or a 0/1 matrix, gives each row's variables, and
    its distinct rows are the environments when ``env`` is ``None``; with ``env``, every row of an
    environment must give the same variables. Raises ``ValueError`` when ``env`` comes without
    ``targets``, when a mapping comes without ``env`` or names a label ``env`` lacks, when an
    environment's targets are unknown or disagree within it, and as
    :func:`~andrey.search.gies.intervention_mask` raises on the mask.
    """
    from andrey.search.gies import intervention_mask

    n, d = X.shape
    if targets is None:
        if env is not None:
            raise ValueError(
                "gies: env needs targets, the variables intervened on in each environment; pass "
                "targets as a mapping from each label of env to its variables, [] for an "
                "observational one"
            )
        return None
    if isinstance(targets, Mapping):
        if env is None:
            raise ValueError(
                "gies: a targets mapping is keyed by the labels of env; pass env, each row's "
                "environment"
            )
        codes, envs = env_codes(env, n, "gies")
        rows = _mapped_rows(targets, envs, labels, d)
        return intervention_mask(rows[codes], X)
    if np.ndim(targets) == 1:
        mask = _cell_rows(np.asarray(targets, dtype=object), n, labels, d)
    else:
        mask = intervention_mask(targets, X)
    rows, inverse = np.unique(mask, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    if env is None:
        return intervention_mask(rows[inverse], X)
    codes, envs = env_codes(env, n, "gies")
    pairs = np.unique(np.column_stack([codes, inverse]), axis=0)
    split = np.flatnonzero(np.bincount(pairs[:, 0], minlength=len(envs)) > 1)
    if split.size:
        raise ValueError(
            f"gies: targets gives different variables to rows of one environment, "
            f"{[envs[k] for k in split]}; give every row of an environment the same variables"
        )
    _, first = np.unique(codes, return_index=True)
    return intervention_mask(rows[inverse[first]][codes], X)


def _mapped_rows(
    targets: Mapping, envs: list, labels: tuple[str, ...] | None, d: int
) -> np.ndarray:
    """Return the ``(K, d)`` intervened variables of the environments ``envs`` from a mapping."""
    index = {label: k for k, label in enumerate(envs)}
    unknown = [key for key in targets if key not in index]
    if unknown:
        raise ValueError(
            f"gies: targets names environments {unknown} that env does not have; its labels "
            f"are {envs}"
        )
    rows = np.zeros((len(envs), d), dtype=bool)
    known = np.zeros(len(envs), dtype=bool)
    for key, value in targets.items():
        if value is not None:
            rows[index[key], _positions(value, labels, d)] = True
            known[index[key]] = True
    if not known.all():
        raise ValueError(
            f"gies: targets gives no variables for environments "
            f"{[envs[k] for k in np.flatnonzero(~known)]}; GIES needs the intervened variables of "
            "every environment, [] for an observational one"
        )
    return rows


def _cell_rows(cells: np.ndarray, n: int, labels: tuple[str, ...] | None, d: int) -> np.ndarray:
    """Return the ``(n, d)`` mask of one cell per row.

    A cell is a variable, a sequence of them, or empty (``None``, ``NaN``, ``""``, or an empty
    sequence) for an observational row.
    """
    if cells.shape != (n,):
        raise ValueError(f"gies: targets must hold one cell per row of data, got {cells.shape}")
    mask = np.zeros((n, d), dtype=bool)
    for i, cell in enumerate(cells):
        if cell is None or (isinstance(cell, str) and not cell) or _is_nan(cell):
            continue
        mask[i, _positions(cell, labels, d)] = True
    return mask


def _is_nan(value: object) -> bool:
    """Whether ``value`` is a float ``NaN``, the empty cell of a numeric column."""
    return isinstance(value, float) and np.isnan(value)


def _positions(value: object, labels: tuple[str, ...] | None, d: int) -> list[int]:
    """Return the 0-based positions of the variables ``value`` names.

    ``value`` is one label or position, or a sequence of them.
    """
    items = [value] if isinstance(value, (str, int, np.integer)) else value
    try:
        items = list(items)  # ty: ignore[invalid-argument-type]  # a TypeError is caught below
    except TypeError:
        items = [value]
    positions = []
    for item in items:
        if isinstance(item, (bool, np.bool_)) or not isinstance(item, (str, int, np.integer)):
            raise ValueError(
                f"gies: targets names variables by label or 0-based position, got {item!r}"
            )
        node = item if isinstance(item, str) else int(item)
        positions.append(column_position(node, labels, d, "gies: targets"))
    return positions
