"""Environments and intervention targets: one normal form for every method that reads them.

Data pooled from several environments give each row an environment, ``env``, and may say which
variables each environment intervened on, its ``targets``. :func:`normalize_env_targets` reads
every accepted form of the two arguments and returns one form: the variables, a code per row, the
environments' labels, and each environment's targets as a set of column positions, or ``None``
where they are unknown. A method then builds what it reads from that form, such as GIES's row mask
``mask[i] = targets_by_code[env_codes[i]]``.

``env`` is ``None``, one label per row, or the name of a column of a DataFrame ``data``.
``targets`` is ``None``; a mapping from each label of ``env`` to its variables; one cell per row,
given as the name of a column of a DataFrame ``data`` or as a 1-D array; or an
``(n_samples, n_variables)`` 0/1 matrix. A named column is not a variable: it leaves the data, and
every position counts the columns that remain. Variables are named by column label (a string) or
by 0-based position (an integer).

The module is a building block for the methods; ``andrey`` does not export it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from andrey.core.knowledge import column_position
from andrey.core.structure import node_labels

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = ["env_codes", "normalize_env_targets", "take_columns"]


def normalize_env_targets(
    data: Any,
    env: npt.ArrayLike | str | None,
    targets: Mapping | npt.ArrayLike | str | None,
    *,
    require_known_targets: bool,
    method: str | None = None,
) -> tuple[Any, np.ndarray, list, dict[int, frozenset[int] | None]]:
    """Return the variables, each row's environment, and each environment's targets.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_columns), or a DataFrame
        The data. A column that ``env`` or ``targets`` names leaves it.
    env : array-like of shape (n_samples,) or (n_samples, 1), str, or None
        Each row's environment: a label per row, strings or numbers but not both, or the name of a
        column of the DataFrame ``data`` that holds them. ``None`` takes the environments from
        ``targets``.
    targets : mapping, str, array-like, or None
        The variables intervened on in each environment, by column label or 0-based position:

        - ``None``: with ``env``, every environment's targets are unknown; without it, the rows
          are one observational environment.
        - A mapping ``{label: variables}`` from labels of ``env``: a list of variables, or one
          label or position, with ``[]`` for an observational environment, or ``None`` for unknown
          targets. A label of ``env`` the mapping leaves out has unknown targets. It needs ``env``.
        - A string, the name of a column of the DataFrame ``data``, or a 1-D array: one cell per
          row, a label or a position, a list of them, or an empty cell (``None``, ``NaN``, ``""``,
          or an empty list) for an observational row.
        - An ``(n_samples, n_variables)`` 0/1 matrix, 1 where the row's value of the variable was
          set by an intervention.

        With a cell per row or a matrix, every row of one environment of ``env`` must give the
        same targets; without ``env``, the environments are the distinct targets of the rows.
    require_known_targets : bool
        ``True`` raises ``ValueError`` when some environment's targets are unknown, naming each
        such environment; ``False`` returns them as ``None``.
    method : str or None, default=None
        The method that calls, which opens each error message.

    Returns
    -------
    X : ndarray or DataFrame of shape (n_samples, n_variables)
        The variables: ``data`` without the columns ``env`` and ``targets`` name. A DataFrame stays
        a DataFrame, so its column types and names reach the caller's data checks; any other input
        becomes an ndarray, its values unchanged.
    env_codes : ndarray of int of shape (n_samples,)
        Each row's environment as a code ``0..K-1``. The codes follow the sorted labels, text or
        numbers alike, so they do not depend on the row order. Without ``env``, they follow the
        distinct rows of the targets' 0/1 matrix as :func:`numpy.unique` sorts them.
    env_labels : list
        The ``K`` labels in code order, as ``env`` holds them, so ``env_labels[k]`` is the label of
        code ``k``. Without ``env``, each label is that environment's targets, a ``frozenset`` of
        positions, and rows with neither ``env`` nor ``targets`` are one environment labeled
        ``frozenset()``.
    targets_by_code : dict of int to frozenset of int or None
        Each code's targets, as 0-based positions among the columns of ``X``; ``frozenset()`` for an
        observational environment and ``None`` where they are unknown.

    Raises
    ------
    ValueError
        If ``env`` does not hold one label per row, misses a label (``None`` or ``NaN``), holds a
        label that is neither a string nor a number, or mixes strings and numbers; if ``env`` or
        ``targets`` names a column ``data`` lacks, or both name one column; if a mapping comes
        without ``env`` or has a key that is not a label of ``env``; if a cell or a mapping names a
        variable ``data`` lacks, or a value that is neither a label nor a position; if a matrix is
        not ``(n_samples, n_variables)`` or holds a value other than 0 or 1; if two rows of one
        environment give different targets; or, with ``require_known_targets``, if some
        environment's targets are unknown.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from andrey.core.environments import normalize_env_targets
    >>> data = pd.DataFrame({"x": [0.1, 0.4, 0.2, 0.9], "y": [1.0, 0.3, 0.5, 0.2]})
    >>> env = ["observed", "set y", "observed", "set y"]
    >>> X, codes, labels, by_code = normalize_env_targets(
    ...     data, env, {"observed": [], "set y": ["y"]}, require_known_targets=True
    ... )
    >>> codes.tolist(), labels, by_code
    ([0, 1, 0, 1], ['observed', 'set y'], {0: frozenset(), 1: frozenset({1})})
    """
    prefix = f"{method}: " if method else ""
    named = {key: v for key, v in (("env", env), ("targets", targets)) if isinstance(v, str)}
    variables, values = take_columns(data, named, method)
    env, targets = values.get("env", env), values.get("targets", targets)
    X = variables if hasattr(variables, "columns") else np.asarray(variables)
    n, d = np.shape(X)[0], (np.shape(X)[1] if np.ndim(X) == 2 else 0)
    labels = _column_labels(X)
    if targets is None:
        if env is None:
            return X, np.zeros(n, dtype=np.intp), [frozenset()], {0: frozenset()}
        codes, envs = env_codes(env, n, method)
        by_code: dict[int, frozenset[int] | None] = dict.fromkeys(range(len(envs)))
        if require_known_targets:
            raise ValueError(
                f"{prefix}env needs targets, the variables intervened on in each environment; "
                "pass targets as a mapping from each label of env to its variables, [] for an "
                "observational one"
            )
        return X, codes, envs, by_code
    if isinstance(targets, Mapping):
        if env is None:
            raise ValueError(
                f"{prefix}a targets mapping is keyed by the labels of env; pass env, each row's "
                "environment"
            )
        codes, envs = env_codes(env, n, method)
        by_code = _mapped(targets, envs, labels, d, prefix)
    else:
        mask = _row_cells(targets, n, labels, d, prefix) if np.ndim(targets) == 1 else None
        mask = _matrix(targets, n, d, prefix) if mask is None else mask
        rows, inverse = np.unique(mask, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        sets = [frozenset(int(j) for j in np.flatnonzero(row)) for row in rows]
        if env is None:
            return X, inverse.astype(np.intp), sets, dict(enumerate(sets))
        codes, envs = env_codes(env, n, method)
        pairs = np.unique(np.column_stack([codes, inverse]), axis=0)
        split = np.flatnonzero(np.bincount(pairs[:, 0], minlength=len(envs)) > 1)
        if split.size:
            raise ValueError(
                f"{prefix}targets gives different variables to rows of one environment, "
                f"{[envs[k] for k in split]}; give every row of an environment the same variables"
            )
        _, first = np.unique(codes, return_index=True)
        by_code = {k: sets[inverse[i]] for k, i in enumerate(first)}
    unknown = [envs[k] for k, s in by_code.items() if s is None]
    if require_known_targets and unknown:
        raise ValueError(
            f"{prefix}targets gives no variables for environments {unknown}; every environment "
            "needs its intervened variables, [] for an observational one"
        )
    return X, codes, envs, by_code


def env_codes(env: npt.ArrayLike, n: int, method: str | None = None) -> tuple[np.ndarray, list]:
    """Return each row's environment as a code, and the labels in code order.

    The codes are ``0..K-1`` for the ``K`` distinct labels in sorted order, text or numbers alike,
    so they do not depend on the row order. Raises ``ValueError`` when ``env`` does not hold one
    label per row, as shape ``(n,)`` or ``(n, 1)``, when a label is missing (``None`` or ``NaN``),
    when a label is neither a string nor a number, or when ``env`` mixes strings and numbers.
    """
    import pandas as pd

    prefix = f"{method}: " if method else ""
    values = np.asarray(env, dtype=object)
    if values.shape not in ((n,), (n, 1)):
        raise ValueError(
            f"{prefix}env must hold one value per row of data, as shape ({n},) or ({n}, 1); got "
            f"shape {values.shape}"
        )
    values = values.reshape(-1)
    missing = [i for i, v in enumerate(values) if v is None or (_is_number(v) and v != v)]
    if missing:
        raise ValueError(
            f"{prefix}env contains NaN or None at row {missing[0]}; every row needs a label"
        )
    kinds = {isinstance(v, str) for v in values}
    if not all(isinstance(v, str) or _is_number(v) for v in values):
        raise ValueError(f"{prefix}env labels must be strings or numbers")
    if len(kinds) > 1:
        raise ValueError(f"{prefix}env mixes strings and numbers; give its labels one type")
    codes, labels = pd.factorize(values, sort=True)
    return codes.astype(np.intp), list(labels)


def take_columns(
    data: Any, named: Mapping[str, str], method: str | None = None
) -> tuple[Any, dict[str, np.ndarray]]:
    """Return ``data`` without the columns ``named`` names, and each such column's values by key.

    ``named`` maps an argument's name, such as ``"env"``, to the column it names. Raises
    ``ValueError`` when ``data`` is not a DataFrame with each column, or two keys name one column.
    """
    if not named:
        return data, {}
    prefix = f"{method}: " if method else ""
    columns = list(getattr(data, "columns", ()))
    for key, column in named.items():
        if column not in columns:
            raise ValueError(
                f"{prefix}{key}={column!r} names a column, but data has no column {column!r}; "
                f"pass a DataFrame with that column, or {key}'s values"
            )
    if len(set(named.values())) < len(named):
        raise ValueError(f"{prefix}{' and '.join(named)} name one column; name two columns")
    values = {key: data[column].to_numpy() for key, column in named.items()}
    return data.drop(columns=list(named.values())), values


def _column_labels(X: Any) -> tuple[str, ...] | None:
    """A DataFrame's column names as text, ``None`` for an array or the default names ``0..d-1``,
    as the methods label their nodes."""
    columns = getattr(X, "columns", None)
    if columns is None:
        return None
    names = list(columns)
    return None if names == list(range(len(names))) else node_labels(names)


def _mapped(
    targets: Mapping, envs: list, labels: tuple[str, ...] | None, d: int, prefix: str
) -> dict[int, frozenset[int] | None]:
    """Return each code's targets from a mapping keyed by the labels ``envs``."""
    index = {label: k for k, label in enumerate(envs)}
    unknown = [key for key in targets if key not in index]
    if unknown:
        raise ValueError(
            f"{prefix}targets names environments {unknown} that env does not have; its labels "
            f"are {envs}"
        )
    by_code: dict[int, frozenset[int] | None] = dict.fromkeys(range(len(envs)))
    for key, value in targets.items():
        if value is not None:
            by_code[index[key]] = frozenset(_positions(value, labels, d, prefix))
    return by_code


def _row_cells(
    cells: npt.ArrayLike, n: int, labels: tuple[str, ...] | None, d: int, prefix: str
) -> np.ndarray:
    """Return the ``(n, d)`` 0/1 matrix of one cell per row.

    A cell is a variable, a sequence of them, or empty (``None``, ``NaN``, ``""``, or an empty
    sequence) for an observational row.
    """
    values = np.asarray(cells, dtype=object)
    if values.shape != (n,):
        raise ValueError(f"{prefix}targets must hold one cell per row of data, got {values.shape}")
    mask = np.zeros((n, d), dtype=bool)
    for i, cell in enumerate(values):
        if cell is None or (isinstance(cell, str) and not cell) or _is_nan(cell):
            continue
        mask[i, _positions(cell, labels, d, prefix)] = True
    return mask


def _matrix(targets: npt.ArrayLike, n: int, d: int, prefix: str) -> np.ndarray:
    """Return the 0/1 matrix ``targets`` as booleans, checked against the ``(n, d)`` variables."""
    try:
        values = np.asarray(targets, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{prefix}a targets matrix must hold 0/1 or booleans: {error}") from error
    if values.shape != (n, d):
        raise ValueError(
            f"{prefix}a targets matrix must have the shape of data, {(n, d)}, got {values.shape}"
        )
    if not np.isin(values, (0.0, 1.0)).all():
        raise ValueError(f"{prefix}a targets matrix must hold 0/1 or booleans")
    return values.astype(bool)


def _is_number(value: object) -> bool:
    """Whether ``value`` is a real number: a Python or NumPy int, float, or bool."""
    return isinstance(value, (int, float, np.number, np.bool_)) and not isinstance(
        value, (complex, np.complexfloating)
    )


def _is_nan(value: object) -> bool:
    """Whether ``value`` is a float ``NaN``, the empty cell of a numeric column."""
    return isinstance(value, float) and np.isnan(value)


def _positions(value: object, labels: tuple[str, ...] | None, d: int, prefix: str) -> list[int]:
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
                f"{prefix}targets names variables by label or 0-based position, got {item!r}"
            )
        node = item if isinstance(item, str) else int(item)
        positions.append(column_position(node, labels, d, f"{prefix}targets"))
    return positions
