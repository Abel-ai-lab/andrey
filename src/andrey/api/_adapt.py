"""Shared adapters for adjacency arrays, column names, JSON-safe metadata, and run envelopes."""

from __future__ import annotations

import contextlib
import dataclasses
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

from andrey.core import ARROW, TAIL, GraphStructure, Structure, StructureOutput
from andrey.core.ci import DISCRETE_INDEP_TESTS, MIXED_INDEP_TESTS, SingularCorrelationError
from andrey.core.score import DISCRETE_SCORES, MIXED_SCORES, CopiedColumnsError, copies_message
from andrey.core.structure import relabel
from andrey.core.warning_policy import DataTypeWarning, warn_once

if TYPE_CHECKING:
    import numpy.typing as npt

    from andrey.core.output import JSONValue


def data_matrix(data: npt.ArrayLike, *, name: str = "data", min_rows: int = 2) -> np.ndarray:
    """Return ``data`` as a float64 ``(n_samples, n_features)`` matrix, checked before a fit.

    Raises ``ValueError``, naming ``name``, when the values are not numeric, the array is not 2-D,
    it has fewer than ``min_rows`` rows, it holds ``NaN`` or ``inf``, or a column is constant: no
    method can estimate a variance or a dependence from such data. The message names every
    constant column.
    """
    try:
        X = np.asarray(data, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must hold numbers: {error}") from error
    if X.ndim != 2:
        hint = "; for one variable, reshape it with .reshape(-1, 1)" if X.ndim == 1 else ""
        raise ValueError(
            f"{name} must be a 2-D (n_samples, n_features) matrix, got shape {X.shape}{hint}"
        )
    if X.shape[0] < min_rows:
        raise ValueError(f"{name} needs at least {min_rows} rows (samples), got {X.shape[0]}")
    bad = np.argwhere(~np.isfinite(X))
    if bad.size:
        row, col = (int(k) for k in bad[0])
        cells = "1 cell" if len(bad) == 1 else f"{len(bad)} cells, the first"
        raise ValueError(
            f"{name} holds NaN or inf in {cells} at row {row}, column {col}; drop or impute "
            "them before the fit"
        )
    constant = [_column_name(data, int(j)) for j in np.flatnonzero(np.ptp(X, axis=0) == 0)]
    if constant:
        if len(constant) == 1:
            columns, verb, pronoun = f"column {constant[0]}", "is", "it"
        else:
            columns = f"columns {', '.join(constant[:-1])} and {constant[-1]}"
            verb, pronoun = "are", "them"
        raise ValueError(
            f"{name} {columns} {verb} constant; a constant column holds no information about "
            f"the other variables, so drop {pronoun} before the fit"
        )
    return X


def check_categories(values: np.ndarray, data: object, setting: str, *, name: str = "data") -> None:
    """Check that ``values`` holds whole numbers only, the codes of a categorical setting.

    A setting that reads categories, such as ``indep_test="chisq"``, takes each distinct value of a
    column as one category, so a fractional value most likely means continuous data. ``setting``
    names the option in the message and ``data`` supplies column names. Raises ``ValueError``,
    naming ``name``, ``setting``, and the first fractional cell, or the first code of ``2**53`` or
    more in magnitude, where whole numbers stop being distinct as floats; ``NaN`` and ``inf`` are
    left to :func:`data_matrix`.
    """
    huge = np.argwhere(np.isfinite(values) & (np.abs(values) >= 2.0**53))
    if huge.size:
        cell = tuple(int(k) for k in huge[0])
        raise ValueError(
            f"{setting} reads {name} as categories, but {name} holds {float(values[cell])!r} at "
            f"row {cell[0]}; codes of 2**53 or more in magnitude may merge, so recode them"
        )
    fractional = np.argwhere(np.isfinite(values) & (values != np.round(values)))
    if fractional.size:
        cell = tuple(int(k) for k in fractional[0])
        where = f"row {cell[0]}"
        if len(cell) > 1:
            where += f", column {_column_name(data, cell[1])}"
        raise ValueError(
            f"{setting} reads {name} as categories coded by whole numbers, but {name} holds "
            f"{float(values[cell])!r} at {where}; code each category as a whole number"
        )


def data_kind(name: str) -> str:
    """Return the data the test or score ``name`` reads: categorical, mixed, or continuous.

    The value is ``"categorical"``, ``"mixed"``, or ``"continuous"``. Any name the categorical and
    mixed-data lists leave out, a registered test included, reads continuous data.
    """
    if name in DISCRETE_INDEP_TESTS or name in DISCRETE_SCORES:
        return "categorical"
    if name in MIXED_INDEP_TESTS or name in MIXED_SCORES:
        return "mixed"
    return "continuous"


#: The default ``max_categories``: the most distinct whole numbers a column may hold to be inferred
#: as categorical.
MAX_CATEGORIES = 10


def default_score(
    data: npt.ArrayLike,
    score_func: str | None,
    linear: str,
    *,
    categorical: Sequence[str | int] | None = None,
    max_categories: int = MAX_CATEGORIES,
    mixed: bool = True,
) -> tuple[str, Sequence[str | int] | None]:
    """Return the score of a score search on ``data`` and the ``categorical`` it reads.

    A given ``score_func`` returns as it is, with ``categorical``. For ``None``, the columns are
    inferred by the rule :func:`mixed_matrix` infers with: a DataFrame's ``category``, ``bool``,
    ``object``, and ``string`` columns, and every column of at most ``max_categories`` distinct
    whole numbers, are categorical. When every column is, the score is ``"local_score_BDeu"``; when
    some are, ``"local_score_cond_gauss"``, the conditional Gaussian BIC, returned with the inferred
    columns by 0-based position so that :func:`checked_data` reads them without inferring again.
    Either switch warns once with a :class:`~andrey.DataTypeWarning` naming ``linear``, the
    linear-Gaussian BIC the search uses otherwise, since columns of counts or ratings meet the rule
    too. ``mixed=False``, for a search without a mixed-data score, keeps ``linear`` on mixed data. A
    given ``categorical`` keeps ``linear``, as does data that is not a finite numeric table apart
    from its typed columns, so that :func:`checked_data` reports it.

    Raises ``TypeError`` or ``ValueError`` when ``max_categories`` is not an int of at least 2.
    """
    if score_func is not None:
        return score_func, categorical
    max_categories = int_in_range(max_categories, "max_categories", 2)
    inferred = None if categorical is not None else _inferred_columns(data, max_categories)
    if inferred is None:
        return linear, categorical
    typed, counts = inferred
    mask = typed | (counts > 0)
    rule = f"a category, bool, or text type, or at most max_categories={max_categories} distinct"
    if mask.all():
        warn_once(
            f"every column of data is categorical ({rule} whole numbers), so the score defaults "
            "to score_func='local_score_BDeu'; if the columns are counts or ratings to read as "
            f"numbers, pass score_func={linear!r}, the linear-Gaussian BIC (see Data types in the "
            "getting-started guide)",
            DataTypeWarning,
        )
        return "local_score_BDeu", None
    if not mask.any() or not mixed:
        return linear, None
    columns = np.flatnonzero(mask).tolist()
    warn_once(
        f"data columns {_listed_columns(data, columns, typed, counts)} are categorical ({rule}"
        " whole numbers) and the others continuous, so the score defaults to "
        "score_func='local_score_cond_gauss', the conditional Gaussian BIC; pass "
        f"score_func={linear!r} to keep the linear-Gaussian BIC, which reads every column as "
        "numbers, or score_func='local_score_cond_gauss' with categorical=[...] to name the "
        "categorical columns (see Data types in the getting-started guide)",
        DataTypeWarning,
    )
    return "local_score_cond_gauss", columns


def _inferred_columns(
    data: npt.ArrayLike, max_categories: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return the typed columns of ``data`` and each other column's count of category codes.

    ``typed`` marks a DataFrame's ``category``, ``bool``, ``object``, and ``string`` columns;
    ``counts`` holds each other column's count of distinct whole numbers when at most
    ``max_categories``, else ``0``. ``None`` when ``data`` has no columns, or its untyped columns
    are not a finite numeric table of at least 2 rows with no constant column, which
    :func:`data_matrix` rejects.
    """
    if getattr(data, "columns", None) is None:
        numeric, typed = data, None
    else:
        import pandas as pd

        typed = _categorical_dtypes(data)
        numeric = pd.DataFrame(data).iloc[:, ~typed]
    try:
        X = np.asarray(numeric, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if X.ndim != 2 or X.shape[0] < 2 or not np.isfinite(X).all() or (np.ptp(X, axis=0) == 0).any():
        return None
    if typed is None:
        typed = np.zeros(X.shape[1], dtype=bool)
    if typed.size == 0:
        return None
    counts = np.zeros(typed.size, dtype=np.int64)
    counts[~typed] = _category_counts(X, max_categories)
    return typed, counts


def checked_data(
    data: npt.ArrayLike,
    categorical: object,
    option: str,
    value: str,
    *,
    max_categories: int = MAX_CATEGORIES,
    index_categorical: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return ``data`` checked for the setting ``option=value``, and its categorical-column mask.

    The test or score ``value`` names decides what is checked (:func:`data_kind`): a mixed-data one
    reads ``categorical`` through :func:`mixed_matrix` and returns its mask; a categorical one
    numbers the values of columns that do not hold numbers, such as text and pandas ``category``
    columns of text (:func:`categories_as_numbers`), and checks every value is a whole number; a
    continuous one gets :func:`data_matrix`'s checks and a
    one-time :class:`~andrey.DataTypeWarning` naming the columns that look like category codes:
    at most ``max_categories`` distinct whole numbers. The mask is ``None`` unless the data are
    mixed. ``index_categorical`` passes to :func:`mixed_matrix`.

    Raises ``TypeError`` or ``ValueError`` when ``max_categories`` is not an int of at least 2, and
    ``ValueError`` when ``categorical`` is given for a setting that reads no column types.
    """
    setting = f"{option}={value!r}"
    kind = data_kind(value)
    max_categories = int_in_range(max_categories, "max_categories", 2)
    if kind == "mixed":
        return mixed_matrix(
            data,
            categorical,
            setting,
            max_categories=max_categories,
            index_categorical=index_categorical,
        )
    if categorical is not None:
        reads = "every column as categorical" if kind == "categorical" else "continuous data"
        raise ValueError(
            "categorical names the categorical columns for a mixed-data test or score, but "
            f"{setting} reads {reads}; leave categorical as None"
        )
    if kind == "categorical":
        X = data_matrix(categories_as_numbers(data, setting))
        check_categories(X, data, setting)
        return X, None
    X = data_matrix(data)
    codes = np.flatnonzero(_category_counts(X, max_categories))
    if codes.size:
        mixed = (
            "indep_test='cond_gauss' or 'degen_gauss'"
            if option == "indep_test"
            else "score_func='local_score_cond_gauss' or 'local_score_degen_gauss'"
        )
        warn_once(
            f"data columns {', '.join(_column_name(data, int(j)) for j in codes)} hold at most "
            f"{max_categories} distinct whole numbers each, so they may be category codes, which "
            f"{setting} reads as numbers; if they are categories, choose {mixed} (see Data types "
            "in the getting-started guide)",
            DataTypeWarning,
        )
    return X, None


def categories_as_numbers(
    data: npt.ArrayLike, setting: str, *, name: str = "data"
) -> npt.ArrayLike:
    """Return ``data`` with each column that does not hold numbers numbered by its values.

    For a categorical setting, every column is categorical, so a text, ``string``, or pandas
    ``category`` column of text has each distinct value numbered as one category, as
    :func:`mixed_matrix` numbers its categorical columns. Data that convert to numbers, booleans
    included, are returned unchanged, as is anything that is not a 2-D table, which
    :func:`data_matrix` then rejects. Raises ``ValueError``, naming ``setting``, on a missing value
    in a numbered column.
    """
    try:
        np.asarray(data, dtype=np.float64)
    except (TypeError, ValueError):
        if np.ndim(data) == 2:
            return _encode_categories(data, np.ones(np.shape(data)[1], dtype=bool), setting, name)
    return data


def mixed_matrix(
    data: npt.ArrayLike,
    categorical: object,
    setting: str,
    *,
    name: str = "data",
    max_categories: int = MAX_CATEGORIES,
    index_categorical: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return mixed data as a float64 matrix and a boolean mask of its categorical columns.

    ``categorical`` lists the categorical columns by DataFrame column name or 0-based position;
    ``[]`` lists none. ``None`` infers them: a pandas DataFrame's ``category``, ``bool``,
    ``object``, and ``string`` columns, and every column whose values, over all its rows, are at
    most ``max_categories`` distinct whole numbers. A :class:`~andrey.DataTypeWarning` names the
    inferred columns and why. A categorical column that does not hold numbers has its values
    numbered, each distinct value one category; a numeric one must hold whole numbers, as
    :func:`check_categories` checks. Then :func:`data_matrix` checks the whole matrix.

    With no categorical column, the setting runs as its linear-Gaussian form, and a
    :class:`~andrey.DataTypeWarning` says so, unless ``index_categorical``: a setting that adds a
    categorical column of its own, such as CD-NOD's domain index.

    Raises ``TypeError`` when ``categorical`` is not a list of names and positions, and
    ``ValueError``, naming ``setting``, when it names a column the data lack, when a categorical
    column has a missing value, or when a column that is not categorical does not hold numbers.
    """
    columns = getattr(data, "columns", None)
    names = None if columns is None else list(columns)
    if categorical is None:
        typed = np.zeros(0, dtype=bool) if names is None else _categorical_dtypes(data)
        table = data if names is None else _encode_categories(data, typed, setting, name)
        X = data_matrix(table, name=name)
        counts = _category_counts(X, max_categories)
        mask = (counts > 0) | (typed if typed.size else False)
        if mask.any():
            listed = _listed_columns(data, np.flatnonzero(mask).tolist(), typed, counts)
            warn_once(
                f"{setting} reads {name} columns {listed} as categorical, inferred from a "
                "category, bool, or text type or from at most "
                f"max_categories={max_categories} distinct whole numbers; name the categorical "
                "columns with categorical=[...], or pass categorical=[] for none",
                DataTypeWarning,
            )
    elif names is None:
        X = data_matrix(data, name=name)
        mask = _column_mask(categorical, None, X.shape[1])
    else:
        mask = _column_mask(categorical, names, len(names))
        X = data_matrix(_encode_categories(data, mask, setting, name), name=name)
    check_categories(np.where(mask, X, 0.0), data, setting, name=name)
    if not mask.any() and not index_categorical:
        what = "test" if setting.startswith("indep_test") else "score"
        if categorical is None:
            reason = (
                f"found no categorical column in {name}: none has a category, bool, or text type "
                f"or at most max_categories={max_categories} distinct whole numbers"
            )
        else:
            reason = f"reads every column of {name} as continuous, with categorical=[]"
        warn_once(
            f"{setting} {reason}, so it runs as the linear-Gaussian {what}; name categorical "
            "columns with categorical=[...] (see Data types in the getting-started guide)",
            DataTypeWarning,
        )
    return X, mask


def categorical_metadata(mask: np.ndarray | None, data: object) -> dict[str, list]:
    """Return ``{"categorical": [...]}`` naming the columns ``mask`` marks; ``{}`` for no mask.

    A DataFrame's columns are named, an array's numbered from 0.
    """
    if mask is None:
        return {}
    labels = column_labels(data)
    marked = np.flatnonzero(mask).tolist()
    return {"categorical": [labels[j] for j in marked] if labels is not None else marked}


def _category_counts(X: np.ndarray, max_categories: int) -> np.ndarray:
    """Count each column's values when they are at most ``max_categories`` whole numbers.

    The count reads all the rows; any other column counts ``0``.
    """
    counts = np.zeros(X.shape[1], dtype=np.int64)
    for j in np.flatnonzero(np.all(X == np.round(X), axis=0)).tolist():
        k = np.unique(X[:, j]).size
        if k <= max_categories:
            counts[j] = k
    return counts


def _listed_columns(data: object, columns: list[int], typed: np.ndarray, counts: np.ndarray) -> str:
    """Name each of ``columns`` with why it is categorical: its type, or its count of values."""
    return ", ".join(
        f"{_column_name(data, j)} ("
        + (_dtype_word(data, j) if typed.size and typed[j] else f"{counts[j]} values")
        + ")"
        for j in columns
    )


def _dtype_word(data: object, j: int) -> str:
    """Name a DataFrame column's categorical type: ``category``, ``bool``, or ``text``."""
    import pandas as pd

    dtype = pd.DataFrame(data).dtypes.iloc[j]
    if isinstance(dtype, pd.CategoricalDtype):
        return "category"
    return "bool" if pd.api.types.is_bool_dtype(dtype) else "text"


def _categorical_dtypes(data: object) -> np.ndarray:
    """Mark a pandas DataFrame's ``category``, ``bool``, ``object``, and ``string`` columns."""
    import pandas as pd

    def categorical(dtype: object) -> bool:
        return (
            isinstance(dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(dtype)
            or pd.api.types.is_object_dtype(dtype)
            or pd.api.types.is_string_dtype(dtype)
        )

    dtypes = pd.DataFrame(data).dtypes
    return np.array([categorical(dtype) for dtype in dtypes], dtype=bool)


def _column_mask(categorical: object, names: list | None, d: int) -> np.ndarray:
    """Mark the columns ``categorical`` lists by name or 0-based position."""
    if isinstance(categorical, (str, bytes)) or not isinstance(
        categorical, (list, tuple, np.ndarray)
    ):
        raise TypeError(
            "categorical must be a list of column names or positions, got "
            f"{type(categorical).__name__}"
        )
    mask = np.zeros(d, dtype=bool)
    for item in categorical:
        if isinstance(item, (int, np.integer)) and not isinstance(item, (bool, np.bool_)):
            if not 0 <= item < d:
                raise ValueError(f"categorical lists column {item}, but data has {d} columns")
            mask[int(item)] = True
        elif isinstance(item, str):
            if names is None:
                raise ValueError(
                    f"categorical names column {item!r}, but data has no column names; give a "
                    "DataFrame or the column's 0-based position"
                )
            if item not in names:
                raise ValueError(
                    f"categorical names column {item!r}, which data does not have; its columns "
                    f"are {[str(n) for n in names]}"
                )
            mask[names.index(item)] = True
        else:
            raise TypeError(
                "categorical must list column names or 0-based positions, got "
                f"{type(item).__name__}"
            )
    return mask


def _encode_categories(data: object, mask: np.ndarray, setting: str, name: str) -> npt.ArrayLike:
    """Replace the values of each categorical column that does not hold numbers by numbers.

    A column that does not hold numbers must be categorical. Raises ``ValueError`` on a missing
    value in a numbered column, or on such a column left out of ``mask``.
    """
    import pandas as pd

    frame = pd.DataFrame(data)
    out = {}
    for j, column in enumerate(frame.columns):
        values = frame.iloc[:, j]
        if pd.api.types.is_numeric_dtype(values.dtype):
            out[column] = values
            continue
        if not mask[j]:
            raise ValueError(
                f"{name} column {column!r} does not hold numbers; list it in categorical, or code "
                "it as numbers"
            )
        codes, _ = pd.factorize(values)
        missing = np.flatnonzero(codes < 0)
        if missing.size:
            raise ValueError(
                f"{name} column {column!r} has a missing value at row {int(missing[0])}; drop or "
                f"impute it before the fit ({setting})"
            )
        out[column] = codes.astype(np.float64)
    return pd.DataFrame(out, index=frame.index)


def _column_name(data: object, j: int) -> str:
    """Column ``j``'s name as quoted text for a DataFrame, else its index."""
    names = getattr(data, "columns", None)
    return repr(str(list(names)[j])) if names is not None else str(j)


def int_in_range(value: object, name: str, low: int, high: int | None = None) -> int:
    """Return ``value`` as an ``int`` from ``low`` to ``high`` inclusive; ``high=None`` has no top.

    Raises ``TypeError`` for a non-integer (``bool`` included) and ``ValueError`` outside the range,
    each naming ``name``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < low or (high is not None and value > high):
        accepted = f"at least {low}" if high is None else f"from {low} to {high}"
        raise ValueError(f"{name} must be an int {accepted}, got {value}")
    return int(value)


def flag(value: object, name: str) -> bool:
    """Return ``value`` as a ``bool``.

    Raises ``TypeError``, naming ``name``, for anything but ``True`` or ``False``; ``0`` and ``1``
    included.
    """
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be True or False, got {type(value).__name__}")
    return bool(value)


def open_unit_interval(value: float, name: str) -> float:
    """Return ``value`` as a ``float`` in the open interval ``(0, 1)``.

    Raises ``ValueError``, naming ``name``, outside it.
    """
    if not 0 < value < 1:
        raise ValueError(f"{name} must lie in the open interval (0, 1), got {value}")
    return float(value)


def dag_from_adjacency(adjacency: npt.ArrayLike) -> GraphStructure:
    """Build a ``dag`` structure from an adjacency where any ``A[i, j]`` != 0 is edge ``i -> j``.

    For algorithms that hand back a bare adjacency rather than an endpoint matrix: a 0/1 DAG
    (ExactSearch) or a weighted one already in Andrey's canonical row-is-source orientation (the
    LiNGAM adapters, via :func:`andrey.api.lingam._canonical_weights`). Only the support matters --
    marks are set from the nonzero pattern.
    """
    A = np.asarray(adjacency)
    d = A.shape[0]
    M = np.zeros((d, d), dtype=np.int8)
    rows, cols = np.nonzero(A)  # A[i, j] != 0  =>  edge i -> j
    M[rows, cols] = TAIL  # tail at the source i
    M[cols, rows] = ARROW  # arrow at the target j
    return GraphStructure.from_numpy(M, kind="dag")


def json_safe(value: object) -> JSONValue:
    """Coerce a result value (score, p-values, metadata) into a JSON-safe metadata value.

    numpy scalars/arrays collapse to Python ``int`` / ``float`` / ``bool`` / ``list``; mappings and
    sequences recurse. Raises ``TypeError`` on a value with no JSON-safe form, mirroring the
    fail-fast ``StructureOutput`` metadata check.
    """
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    raise TypeError(f"cannot coerce {type(value).__name__} to a JSON-safe metadata value")


def structure_output(
    structure: Structure,
    *,
    labels: tuple[str, ...] | None = None,
    ordering: npt.ArrayLike | None = None,
    weighted_adjacency: npt.ArrayLike | None = None,
    metadata: Mapping[str, object] | None = None,
) -> StructureOutput:
    """Assemble the run envelope, coercing ordering / weights / metadata into the store types.

    Metadata values pass through :func:`json_safe` (so a raw ``np.float64`` score is accepted);
    ``ordering`` becomes a plain ``int`` tuple and ``weighted_adjacency`` is cast to ``float64``.
    ``labels`` names the structure's nodes. Delegates to ``StructureOutput.new``, which validates
    the whole envelope.
    """
    if labels is not None:
        structure = relabel(structure, labels)
    meta = None if metadata is None else {str(k): json_safe(v) for k, v in metadata.items()}
    order = None if ordering is None else tuple(int(x) for x in np.asarray(ordering).tolist())
    weights = (
        None if weighted_adjacency is None else np.asarray(weighted_adjacency, dtype=np.float64)
    )
    return StructureOutput.new(structure, ordering=order, weighted_adjacency=weights, metadata=meta)


def node_labels(names: Iterable[object]) -> tuple[str, ...]:
    """Turn column names into node labels: strings, unique so that each names one node."""
    labels = tuple(str(name) for name in names)
    if len(set(labels)) != len(labels):
        raise ValueError(f"column names must be unique to name the nodes, got {list(labels)}")
    return labels


def column_labels(data: object) -> tuple[str, ...] | None:
    """Return a DataFrame's column names as node labels; ``None`` for an array.

    Reads any table with a ``columns`` attribute (pandas, polars), so pandas stays optional. Default
    integer columns ``0..n-1`` name nothing, so such a table numbers its nodes like an array.
    """
    columns = getattr(data, "columns", None)
    if columns is None:
        return None
    names = list(columns)
    return None if names == list(range(len(names))) else node_labels(names)


def column_namer(labels: tuple[str, ...] | None) -> Callable[[int], str]:
    """Name a column position by its label in ``labels``, quoted, or by its number.

    A position past the labels, such as a column a method adds to the data, and every position
    without ``labels`` keep their number.
    """
    if labels is None:
        return str
    return lambda j: repr(labels[j]) if j < len(labels) else str(j)


@contextlib.contextmanager
def copies_named(
    labels: tuple[str, ...] | None, name: Callable[[int], str] | None = None
) -> Generator[None]:
    """Name the columns of a copies error raised in the block by ``labels``, or by ``name``.

    A :class:`~andrey.core.score.CopiedColumnsError`, or a
    :class:`~andrey.core.ci.SingularCorrelationError` that found copies, names columns by position;
    the message names them by :func:`column_namer` of ``labels`` instead, or by ``name`` when it is
    given.
    """
    try:
        yield
    except (CopiedColumnsError, SingularCorrelationError) as error:
        if error.groups and (labels is not None or name is not None):
            raised = error.column_name if isinstance(error, CopiedColumnsError) else str
            named = copies_message(error.groups, name or column_namer(labels))
            error.args = (str(error).replace(copies_message(error.groups, raised), named),)
        raise


def shared_labels(label_sets: Iterable[tuple[str, ...] | None]) -> tuple[str, ...] | None:
    """Return the column names several datasets share; ``None`` when none of them is named.

    Raises ``ValueError`` when two datasets name their columns differently.
    """
    named = list(dict.fromkeys(labels for labels in label_sets if labels is not None))
    if len(named) > 1:
        raise ValueError(f"datasets name their columns differently: {named[0]} and {named[1]}")
    return named[0] if named else None


def with_labels(out: StructureOutput, labels: tuple[str, ...] | None) -> StructureOutput:
    """Return ``out`` with its structure's nodes named ``labels``; ``out`` itself for ``None``."""
    if labels is None:
        return out
    return dataclasses.replace(out, _structure=relabel(out.structure, labels))
