"""The ``effects`` record: a method's answer for an ordered treatment-outcome pair.

A method that answers for pairs of variables, rather than with a graph alone, lists one record per
ordered pair it answers in ``metadata["effects"]``. Every record holds the same seven fields, so
one reader serves every such method:

- ``treatment``, ``outcome``: the pair, by column label or 0-based position.
- ``estimand``: ``"total"`` for the total effect of the treatment on the outcome, ``"direct"`` for
  the direct effect.
- ``nonzero``: whether the effect is nonzero, ``None`` when the method does not decide.
- ``identifiable``: whether one adjustment set is valid in every graph the method's answer allows,
  ``None`` when the method does not decide.
- ``adjustment_sets``: the sets found, each a list of columns that holds neither the treatment nor
  the outcome; ``[]`` when none is found and ``[[]]`` when the empty set is the one found.
- ``set_kind``: what the sets are (:data:`SET_KINDS`), ``None`` exactly when there are none.

A method may add fields of its own after these, such as a claim's ``score``. :func:`effect_record`
builds a record from column positions, and :func:`check_effects` checks a list of records.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Literal

#: The effect a record answers for.
ESTIMANDS: tuple[str, ...] = ("total", "direct")
#: What a record's adjustment sets are: ``"optimal"``, the valid set with the smallest asymptotic
#: variance of the effect estimate; ``"valid"``, sets that each satisfy the adjustment criterion;
#: ``"parents_of_outcome"``, the outcome's parents but the treatment, for a direct effect; and
#: ``"local_parents"``, one set per parent set of the treatment the answer allows, as the IDA
#: approach takes them, when no one set is valid in every allowed graph.
SET_KINDS: tuple[str, ...] = ("optimal", "valid", "parents_of_outcome", "local_parents")
#: The fields every record holds, in order.
EFFECT_FIELDS: tuple[str, ...] = (
    "treatment",
    "outcome",
    "estimand",
    "nonzero",
    "identifiable",
    "adjustment_sets",
    "set_kind",
)

Node = str | int


def effect_record(
    treatment: int,
    outcome: int,
    *,
    estimand: Literal["total", "direct"],
    nonzero: bool | None = None,
    identifiable: bool | None = None,
    adjustment_sets: Iterable[Iterable[int]] = (),
    set_kind: str | None = None,
    name: Callable[[int], Node] = int,
    **extra: Any,
) -> dict[str, Any]:
    """Build one ``effects`` record from column positions and check it.

    Parameters
    ----------
    treatment, outcome : int
        The pair's 0-based column positions.
    estimand : {"total", "direct"}
        The effect the record answers for.
    nonzero, identifiable : bool or None, default=None
        The method's decisions, ``None`` where it does not decide.
    adjustment_sets : iterable of iterables of int, default=()
        The sets found, by column position. Each set is listed in ascending column order and the
        sets by their lists, so the record does not depend on the order the method found them in.
    set_kind : str or None, default=None
        One of :data:`SET_KINDS`, or ``None`` when there are no sets.
    name : callable, default=int
        Maps a position to the name the record holds, such as the column label.
    **extra
        The method's own fields, which follow the seven shared ones.

    Returns
    -------
    dict
        The record, its fields in :data:`EFFECT_FIELDS` order and then ``extra``.

    Raises
    ------
    ValueError
        If the record breaks a rule :func:`check_effects` checks.
    """
    sets = sorted({tuple(sorted(int(v) for v in s)) for s in adjustment_sets})
    record: dict[str, Any] = {
        "treatment": name(int(treatment)),
        "outcome": name(int(outcome)),
        "estimand": estimand,
        "nonzero": nonzero,
        "identifiable": identifiable,
        "adjustment_sets": [[name(v) for v in s] for s in sets],
        "set_kind": set_kind,
        **extra,
    }
    check_effects([record])
    return record


def check_effects(records: object, columns: Sequence[Node] | None = None) -> None:
    """Check that ``records`` is a list of ``effects`` records.

    Each record is a dict holding every field of :data:`EFFECT_FIELDS`; the treatment and the
    outcome are two different columns, named by a string or an int; ``estimand`` is one of
    :data:`ESTIMANDS`; ``nonzero`` and ``identifiable`` are ``True``, ``False``, or ``None``;
    ``adjustment_sets`` is a list of lists of columns, no set repeating a column or holding the
    treatment or the outcome, and no set listed twice; and ``set_kind`` is ``None`` exactly when
    there are no sets, else one of :data:`SET_KINDS`. With ``columns``, every column a record names
    is one of them. The check reads the records only; whether a set is valid in a graph is
    another question.

    Raises
    ------
    ValueError
        Naming the first record and field that break a rule.
    """
    if not isinstance(records, list):
        raise ValueError(f"effects must be a list of records, got {type(records).__name__}")
    known = None if columns is None else set(columns)
    for i, record in enumerate(records):
        where = f"effects[{i}]"
        if not isinstance(record, Mapping):
            raise ValueError(f"{where} must be a dict, got {type(record).__name__}")
        missing = [f for f in EFFECT_FIELDS if f not in record]
        if missing:
            raise ValueError(f"{where} lacks {missing}")
        pair = (record["treatment"], record["outcome"])
        for node in pair:
            _check_node(node, known, f"{where}: treatment and outcome")
        if pair[0] == pair[1]:
            raise ValueError(f"{where}: treatment and outcome are both {pair[0]!r}")
        if record["estimand"] not in ESTIMANDS:
            raise ValueError(
                f"{where}: estimand must be one of {ESTIMANDS}, got {record['estimand']!r}"
            )
        for field in ("nonzero", "identifiable"):
            if not isinstance(record[field], bool | None):
                raise ValueError(f"{where}: {field} must be True, False, or None")
        sets = record["adjustment_sets"]
        if not isinstance(sets, list) or not all(isinstance(s, list) for s in sets):
            raise ValueError(f"{where}: adjustment_sets must be a list of lists")
        seen: set[frozenset] = set()
        for s in sets:
            for node in s:
                _check_node(node, known, f"{where}: adjustment_sets")
            if len(set(s)) != len(s) or set(s) & set(pair):
                raise ValueError(
                    f"{where}: the set {s} repeats a column or holds the treatment or the outcome"
                )
            if frozenset(s) in seen:
                raise ValueError(f"{where}: the set {s} is listed twice")
            seen.add(frozenset(s))
        kind = record["set_kind"]
        if (kind is None) != (not sets) or (kind is not None and kind not in SET_KINDS):
            raise ValueError(
                f"{where}: set_kind must be None exactly when there are no sets, else one of "
                f"{SET_KINDS}; got {kind!r} with {len(sets)} sets"
            )


def _check_node(node: object, known: set[Node] | None, what: str) -> None:
    """Raise ``ValueError`` unless ``node`` names a column: a string or an int, among ``known``."""
    if isinstance(node, bool) or not isinstance(node, str | int):
        raise ValueError(f"{what} name columns by label or position, got {node!r}")
    if known is not None and node not in known:
        raise ValueError(f"{what} name {node!r}, which is not a column")
