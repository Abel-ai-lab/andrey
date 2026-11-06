"""Environments and intervention targets: one normal form for every method that reads them.

Data pooled from several environments give each row an environment, ``env``, and may say which
variables each environment intervened on, its ``targets``. :func:`normalize_env_targets` reads
every accepted form of the two arguments and returns one form: the variables, a code per row, the
environments' labels, and each environment's targets as a set of column positions, or ``None``
where they are unknown. A method then builds what it reads from that form, such as GIES's row mask
``mask[i] = targets_by_code[env_codes[i]]``.

``data`` is one table, or each environment's rows apart: a list of tables, labeled by position
``0..K-1``, or a mapping from each label to its table (:func:`pool_environments`).
``env`` is ``None``, one label per row, or the name of a column of a DataFrame ``data``.
``targets`` is ``None``; a mapping from each label of ``env`` to its variables; one cell per row,
given as the name of a column of a DataFrame ``data`` or as a 1-D array; or an
``(n_samples, n_variables)`` 0/1 matrix. A mapping's key is a label of ``env``; when the labels are
numbers, a text key that spells one also names it, so a mapping written to JSON, whose keys are
text, names the same environments. A named column is not a variable: it leaves the data, and
every position counts the columns that remain. Variables are named by column label (a string) or
by 0-based position (an integer).

The module is a building block for the methods; ``andrey`` does not export it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from andrey.core.knowledge import column_position
from andrey.core.structure import column_labels

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = [
    "check_constant_columns",
    "check_environment_rows",
    "env_codes",
    "normalize_env_targets",
    "pool_environments",
    "take_columns",
]


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
    data : array-like of shape (n_samples, n_columns), a DataFrame, or a list or mapping of them
        The data. A column that ``env`` or ``targets`` names leaves it. A list or mapping holds
        each environment's rows apart, labeled by position or key (:func:`pool_environments`),
        with ``env`` ``None`` and ``targets`` a mapping from those labels or ``None``.
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
          A key names the label it equals. When the labels are numbers, a text key names the label
          it spells, read by ``int`` or else by ``float``, finite: ``"1"``, ``"1.0"``, and
          ``"01"`` name the label ``1``. When the labels are booleans, ``"true"`` and ``"false"``
          name them. Text labels are never read as numbers.
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
        If ``data`` holds each environment apart and :func:`pool_environments` raises; if
        ``env`` does not hold one label per row, misses a label (``None`` or ``NaN``), holds a
        label that is neither a string nor a number, or mixes strings and numbers; if ``env`` or
        ``targets`` names a column ``data`` lacks, or both name one column; if a mapping comes
        without ``env``, has a key that names no label of ``env``, or has two keys that name one
        label, such as ``1`` and ``"1"``; if a cell or a mapping names a
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
    data, env = pool_environments(data, env, targets, method)
    named = {key: v for key, v in (("env", env), ("targets", targets)) if isinstance(v, str)}
    variables, values = take_columns(data, named, method)
    env, targets = values.get("env", env), values.get("targets", targets)
    X = variables if hasattr(variables, "columns") else np.asarray(variables)
    n, d = np.shape(X)[0], (np.shape(X)[1] if np.ndim(X) == 2 else 0)
    labels = column_labels(X)
    if targets is None:
        if env is None:
            return X, np.zeros(n, dtype=np.intp), [frozenset()], {0: frozenset()}
        codes, envs = env_codes(env, n, method)
        by_code: dict[int, frozenset[int] | None] = dict.fromkeys(range(len(envs)))
        if require_known_targets:
            raise ValueError(
                f"{prefix}each environment needs its targets, the variables intervened on in "
                "it; pass targets as a mapping from each environment's label to its variables, [] "
                "for an observational one"
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


def pool_environments(
    data: Any, env: Any, targets: Any = None, method: str | None = None, missing: bool = False
) -> tuple[Any, Any]:
    """Return ``data`` as one table and ``env`` as each row's label, when ``data`` holds each
    environment's rows apart.

    ``data`` holds them apart as a list or tuple, empty or with a table among its members, each
    member labeled by its position ``0..K-1``, or as a mapping from each label, a string or a finite
    number, to its table; a table is a DataFrame or a 2-D array. The rows are stacked in the order
    ``data`` holds them, so the pooled table and ``env``, the label of each row, are what a call
    with one table passes. DataFrames keep their columns, in the order of the first, and their
    column types; a column of the ``category`` type in every DataFrame keeps it, with the union of
    their categories. Any other ``data`` returns unchanged, with ``env``.

    Raises ``ValueError``, naming ``method``, when ``env`` is also given, ``targets`` is neither
    ``None`` nor a mapping, ``data`` holds no environment, an environment has no rows or is not a
    table, the tables mix DataFrames and arrays, a DataFrame repeats a column name, the
    DataFrames' columns differ, the arrays' numbers of columns differ, the labels are not strings
    or finite numbers, or mix them, or an environment holds ``inf``, or ``NaN`` without
    ``missing``, which a method that accepts missing values passes; the message names the
    environment and its own row.
    """
    if isinstance(data, Mapping):
        labels, tables = list(data.keys()), list(data.values())
    elif isinstance(data, (list, tuple)) and (not data or any(_is_table(t) for t in data)):
        labels, tables = list(range(len(data))), list(data)
    else:
        return data, env
    prefix = f"{method}: " if method else ""
    if env is not None:
        raise ValueError(
            f"{prefix}data holds each environment's rows apart, which gives each row's "
            "environment; leave env out, or pass data as one table with env"
        )
    if targets is not None and not isinstance(targets, Mapping):
        raise ValueError(
            f"{prefix}with each environment's rows apart, targets maps each environment's "
            "position in the list, or key in the mapping, to its variables"
        )
    if not tables:
        raise ValueError(f"{prefix}data holds no environment; give each environment's rows")
    if not all(isinstance(k, str) or (_is_number(k) and math.isfinite(k)) for k in labels):
        raise ValueError(
            f"{prefix}the keys of data label the environments: strings or finite numbers"
        )
    if len({isinstance(k, str) for k in labels}) > 1:
        raise ValueError(f"{prefix}the keys of data mix strings and numbers; give them one type")
    other = [k for k, table in zip(labels, tables, strict=True) if not _is_table(table)]
    if other:
        raise ValueError(
            f"{prefix}each environment's rows are a DataFrame or a 2-D array; environments "
            f"{other} are not"
        )
    frames = [hasattr(table, "columns") for table in tables]
    if any(frames) and not all(frames):
        raise ValueError(
            f"{prefix}data mixes DataFrames and arrays; give every environment the same kind"
        )
    sizes = [np.shape(table)[0] for table in tables]
    empty = [k for k, size in zip(labels, sizes, strict=True) if size == 0]
    if empty:
        raise ValueError(f"{prefix}environments {empty} have no rows; leave them out")
    import pandas as pd

    for k, table in zip(labels, tables, strict=True):
        cells = pd.DataFrame(table)
        bad = cells.isin([np.inf, -np.inf]).to_numpy()
        if not missing:
            bad = bad | cells.isna().to_numpy()
        if bad.any():
            row, j = (int(i) for i in np.argwhere(bad)[0])
            where = f"{prefix}environment {k!r} of data, at row {row}, column {cells.columns[j]!r},"
            if not pd.isna(cells.iat[row, j]):
                raise ValueError(f"{where} holds inf; drop or replace it before the fit")
            raise ValueError(
                f"{where} holds NaN, and {method or 'this method'} does not accept missing "
                "values; drop the rows that hold one (see Missing values in the getting-started "
                "guide)"
            )
    if all(frames):
        repeated = [
            k for k, table in zip(labels, tables, strict=True) if table.columns.has_duplicates
        ]
        if repeated:
            raise ValueError(
                f"{prefix}the DataFrames of environments {repeated} repeat a column name; give "
                "each column one name"
            )
        columns = list(tables[0].columns)
        differ = [
            k for k, table in zip(labels, tables, strict=True) if set(table.columns) != set(columns)
        ]
        if differ:
            raise ValueError(
                f"{prefix}the DataFrames of environments {differ} have other columns than "
                f"those of environment {labels[0]!r}, {columns}; give each the same columns"
            )
        tables = [table[columns] for table in tables]
        union = {}
        for column in columns:
            kinds = [table[column].dtype for table in tables]
            if all(isinstance(kind, pd.CategoricalDtype) for kind in kinds) and len(set(kinds)) > 1:
                categories = kinds[0].categories
                for kind in kinds[1:]:
                    categories = categories.union(kind.categories, sort=False)
                union[column] = pd.CategoricalDtype(categories)
        pooled = pd.concat([table.astype(union) for table in tables], ignore_index=True)
    else:
        arrays = [np.asarray(table) for table in tables]
        widths = [a.shape[1] for a in arrays]
        if len(set(widths)) > 1:
            raise ValueError(
                f"{prefix}the environments' arrays have {widths} columns; give each the same "
                "columns"
            )
        pooled = np.concatenate(arrays)
    return pooled, np.repeat(np.array(labels, dtype=object), sizes)


def env_codes(env: npt.ArrayLike, n: int, method: str | None = None) -> tuple[np.ndarray, list]:
    """Return each row's environment as a code, and the labels in code order.

    The codes are ``0..K-1`` for the ``K`` distinct labels in sorted order, text or numbers alike,
    so they do not depend on the row order. Raises ``ValueError`` when ``env`` does not hold one
    label per row, as shape ``(n,)`` or ``(n, 1)``, when a label is missing (``None`` or ``NaN``),
    when a label is neither a string nor a number, or when ``env`` mixes strings and numbers.
    """
    import pandas as pd

    if isinstance(env, np.ndarray) and env.shape in ((n,), (n, 1)) and env.dtype.kind in "iuU":
        # Integer or text labels: none can be missing or of the other kind, so no row is checked.
        codes, labels = pd.factorize(np.asarray(env).reshape(-1), sort=True)
        return codes.astype(np.intp), labels.tolist()
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


def check_environment_rows(
    codes: np.ndarray, names: Sequence[object], method: str, min_rows: int
) -> None:
    """Raise ``ValueError`` unless every environment has at least ``min_rows`` rows.

    ``codes`` gives each row's environment as a code ``0..K-1``, and ``names[k]`` names
    environment ``k`` in the message, which lists every environment with fewer rows.
    """
    sizes = np.bincount(codes, minlength=len(names))
    small = [names[k] for k in np.flatnonzero(sizes < min_rows)]
    if small:
        raise ValueError(
            f"{method} needs at least {min_rows} rows in every environment; {small} have fewer"
        )


def check_constant_columns(
    X: np.ndarray, name: Callable[[int], str], method: str, where: str
) -> None:
    """Raise ``ValueError`` when a column of ``X``, the rows ``where`` names, is constant on them.

    ``name`` names a column by its position in the message, which names every constant column.
    """
    constant = [name(int(j)) for j in np.flatnonzero(np.ptp(X, axis=0) == 0)]
    if not constant:
        return
    if len(constant) == 1:
        columns, verb, pronoun = f"column {constant[0]}", "is", "it"
    else:
        columns, verb, pronoun = f"columns {', '.join(constant)}", "are", "them"
    raise ValueError(
        f"{method}: data {columns} {verb} constant on {where}; drop {pronoun} before the fit"
    )


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


def _is_table(value: object) -> bool:
    """Whether ``value`` is one environment's rows: a DataFrame or a 2-D array."""
    if hasattr(value, "columns"):
        return True
    try:
        return np.ndim(value) == 2  # ty: ignore[invalid-argument-type]  # a ValueError is caught
    except ValueError:  # a ragged nested list
        return False


def _mapped(
    targets: Mapping, envs: list, labels: tuple[str, ...] | None, d: int, prefix: str
) -> dict[int, frozenset[int] | None]:
    """Return each code's targets from a mapping keyed by the labels ``envs``."""
    index = {label: k for k, label in enumerate(envs)}
    codes = {key: _code(key, index, envs) for key in targets}
    unknown = [key for key, code in codes.items() if code is None]
    if unknown:
        number = next((key for key in unknown if not isinstance(key, str)), None)
        text = number is not None and isinstance(envs[0], str)
        hint = f"; the labels are text, so write the key {str(number)!r}" if text else ""
        raise ValueError(
            f"{prefix}targets names environments {unknown} that env does not have; its labels "
            f"are {envs}{hint}"
        )
    resolved = {key: code for key, code in codes.items() if code is not None}
    keys: dict[int, list] = {}
    for key, code in resolved.items():
        keys.setdefault(code, []).append(key)
    shared = [f"{group} name {envs[code]!r}" for code, group in keys.items() if len(group) > 1]
    if shared:
        raise ValueError(
            f"{prefix}targets gives one environment more than one key: {'; '.join(shared)}; keep "
            "one key per label of env"
        )
    by_code: dict[int, frozenset[int] | None] = dict.fromkeys(range(len(envs)))
    for key, value in targets.items():
        if value is not None:
            by_code[resolved[key]] = frozenset(_positions(value, labels, d, prefix))
    return by_code


def _code(key: object, index: dict, envs: list) -> int | None:
    """Return the code of the label ``key`` names, or ``None`` when it names none.

    A key names the label it equals. When the labels are numbers, a text key names the label it
    spells: by ``int``, so a label beyond ``2**53`` matches exactly, or else by ``float``, finite.
    When they are booleans, the text ``"true"`` or ``"false"``.
    """
    if key in index:
        return index[key]
    if not isinstance(key, str) or not envs or isinstance(envs[0], str):
        return None
    if all(isinstance(label, (bool, np.bool_)) for label in envs):
        return index.get({"true": True, "false": False}.get(key))
    try:
        return index.get(int(key))
    except ValueError:
        pass
    try:
        number = float(key)
    except ValueError:
        return None
    return index.get(number) if math.isfinite(number) else None


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
