"""Columns and answers shared by the methods that answer for one variable or a pair of variables.

A target-centered method answers for one column, ``target``; a treatment-outcome method for an
ordered pair, ``treatment`` and ``outcome``; a method that answers for every ordered pair of a few
columns takes them unordered as ``variables``. The helpers here turn those arguments into column
positions with one set of messages, and :func:`target_metadata` writes a target-centered answer's
sets under the keys of :data:`TARGET_SETS`. The pair methods' answer is the ``effects`` record of
:mod:`andrey.core.effects`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence

from andrey.core.knowledge import as_node, column_position

#: The metadata key of each set a target-centered answer lists, and what puts a column in it. A
#: set learner lists one set (``markov_blanket`` or ``parents_and_children``); a method that
#: orients the target's edges lists the edge of each adjacent column under one of the others.
TARGET_SETS: dict[str, str] = {
    "markov_blanket": "the target's Markov blanket: its parents, its children, and its "
    "children's other parents",
    "parents_and_children": "adjacent to the target, by a learner of parents and children",
    "parents": "a direct cause of the target: x -> target",
    "children": "a direct effect of the target: target -> x",
    "undirected": "adjacent to the target with the direction undecided: x - target in a CPDAG",
    "bidirected": "x <-> target in a PAG: neither is an ancestor of the other",
    "undetermined": "adjacent to the target with a circle mark at the target's end in a PAG",
}


def target_column(
    target: str | int | None, labels: Sequence[str] | None, d: int, method: str
) -> int:
    """Return the position of the column ``target`` names, for a method that requires one.

    Raises
    ------
    ValueError
        If ``target`` is ``None`` or names a column the data lack.
    TypeError
        If ``target`` is neither a column label nor a position.
    """
    if target is None:
        raise ValueError(
            f"{method}: pass target, the variable to answer for, as a column label or a 0-based "
            "position"
        )
    return column_position(as_node(target, f"{method}: target"), labels, d, f"{method}: target")


def pair_columns(
    treatment: str | int | None,
    outcome: str | int | None,
    labels: Sequence[str] | None,
    d: int,
    method: str,
) -> tuple[int, int]:
    """Return the positions of ``treatment`` and ``outcome``, two different columns.

    Raises
    ------
    ValueError
        If either is ``None``, names a column the data lack, or both name one column.
    TypeError
        If either is neither a column label nor a position.
    """
    if treatment is None or outcome is None:
        raise ValueError(
            f"{method}: pass treatment and outcome, two columns, each as a column label or a "
            "0-based position"
        )
    t = column_position(
        as_node(treatment, f"{method}: treatment"), labels, d, f"{method}: treatment"
    )
    o = column_position(as_node(outcome, f"{method}: outcome"), labels, d, f"{method}: outcome")
    if t == o:
        raise ValueError(f"{method}: treatment and outcome name the same column, {treatment!r}")
    return t, o


def variable_columns(
    variables: Iterable[str | int] | None,
    labels: Sequence[str] | None,
    d: int,
    method: str,
    *,
    minimum: int = 2,
    maximum: int | None = None,
) -> list[int]:
    """Return the positions of the columns ``variables`` names, ascending.

    ``variables`` names at least ``minimum`` and at most ``maximum`` different columns, in any
    order.

    Raises
    ------
    ValueError
        If ``variables`` is ``None``, names a column the data lack or one column twice, or names
        too few or too many columns.
    TypeError
        If ``variables`` is a string, or a member is neither a column label nor a position.
    """
    if variables is None:
        raise ValueError(
            f"{method}: pass variables, a list of columns, each as a column label or a 0-based "
            "position"
        )
    if isinstance(variables, str | bytes):
        raise TypeError(f"{method}: variables must be a list of columns, got {variables!r}")
    what = f"{method}: variables"
    found = [column_position(as_node(v, what), labels, d, what) for v in variables]
    if len(set(found)) != len(found):
        raise ValueError(f"{what} names a column twice: {list(variables)}")
    if len(found) < minimum or (maximum is not None and len(found) > maximum):
        if maximum is None:
            count = f"at least {minimum}"
        else:
            count = str(minimum) if maximum == minimum else f"{minimum} to {maximum}"
        raise ValueError(f"{what} must name {count} columns, got {len(found)}")
    return sorted(found)


def target_metadata(
    target: int, sets: Mapping[str, Iterable[int]], name: Callable[[int], str | int]
) -> dict[str, object]:
    """``{"target": name(target), key: [name(j), ...]}`` for each set of a target-centered answer.

    Each key of ``sets`` is a key of :data:`TARGET_SETS`, and its members are listed in the order
    given; ``name`` maps a position to the name metadata holds (``label_of``).

    Raises
    ------
    ValueError
        If a key is not in :data:`TARGET_SETS` or a set holds the target.
    """
    out: dict[str, object] = {"target": name(target)}
    for key, members in sets.items():
        if key not in TARGET_SETS:
            raise ValueError(f"{key!r} is not a target set; the sets are {list(TARGET_SETS)}")
        listed = [int(j) for j in members]
        if target in listed:
            raise ValueError(f"the set {key!r} holds the target")
        out[key] = [name(j) for j in listed]
    return out
