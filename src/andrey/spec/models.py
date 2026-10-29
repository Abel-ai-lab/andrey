"""Immutable Pydantic models for method contracts and the environment variables.

Models validate at construction and reject unknown fields with ``extra="forbid"``. Tests check
method specs against live facade signatures. See :mod:`andrey.spec` for the generated interfaces.
"""

from __future__ import annotations

import math
import reprlib
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

Family = Literal["constraint", "score", "permutation", "lingam", "temporal", "latent", "pairwise"]
Determinism = Literal["deterministic", "stochastic"]
Accel = Literal["numpy", "torch"]
# supported: benchmarked and gated before each release. experimental: warns once; API may change.
Status = Literal["supported", "experimental"]
# What reordering the input columns does to the result, with the default ``tie_break="rank"``:
# relabels it and changes nothing else; relabels it where FastICA converges; or can change it.
ColumnOrder = Literal["relabels", "relabels_if_converged", "can_change"]

# How an array input reaches a method -- the CLI wires each to a --flag, the schema to a type.
InputKind = Literal[
    "matrix",  # one (n_samples, n_variables) 2-D array
    "panel",  # a sequence of 2-D arrays, one per group/time point (repeated flag)
    "context",  # an auxiliary (n_samples, k) matrix aligned to the data rows (cdnod)
]
# Hyperparameter value types -- map to a JSON-schema type and a CLI option.
# knowledge: an object with required/forbidden/allowed edge lists and tiers (andrey.Knowledge).
ParamType = Literal["float", "int", "str", "bool", "int_pair", "str_list", "knowledge"]
# The parts of an andrey.Knowledge a method honors.
KnowledgePart = Literal["required", "forbidden", "allowed", "tiers"]

EnvGroup = Literal["backend", "reproducibility", "determinism", "data"]

_FROZEN = ConfigDict(frozen=True, extra="forbid")


class Spec(BaseModel):
    """Immutable spec model that rejects unknown fields."""

    model_config = _FROZEN


class InputSpec(Spec):
    """One array input to a method (matches the facade parameter name)."""

    name: str
    kind: InputKind
    doc: str
    required: bool = True

    @property
    def flag(self) -> str:
        """The CLI flag for this input (``data`` -> ``--data``)."""
        return "--" + self.name.replace("_", "-")


class DataSpec(Spec):
    """The data contract: the array inputs plus their expected layout."""

    inputs: tuple[InputSpec, ...]
    layout: str  # human/agent-readable shape, for example "(n_samples, n_variables), row = sample"
    notes: str = ""
    missing: bool = False  # accepts NaN as a missing value, its tests deleting rows per test


class ParamSpec(Spec):
    """One keyword hyperparameter (matches the facade keyword and its default)."""

    name: str
    type: ParamType
    default: Any  # JSON-safe default; ``None`` allowed when ``nullable``
    doc: str
    choices: tuple[str, ...] | None = None  # the supported values, when enumerable
    nullable: bool = False  # accepts ``None`` (for example, seed, criterion, labels)
    # Bounds of an int or float value; an exclusive bound is itself out of range.
    minimum: float | None = None
    maximum: float | None = None
    exclusive_minimum: bool = False
    exclusive_maximum: bool = False
    # Former names, accepted for one release after a rename.
    aliases: tuple[str, ...] = ()
    # Former values, accepted for one release: (old value, current value) pairs.
    choice_aliases: tuple[tuple[str, str], ...] = ()
    # The package-wide concept this parameter implements (andrey.spec.concepts), if any.
    concept: str | None = None

    @model_validator(mode="after")
    def _check_aliases(self) -> ParamSpec:
        if self.name in self.aliases:
            raise ValueError(f"{self.name}: a parameter is not its own alias")
        for old, new in self.choice_aliases:
            if not self.choices or new not in self.choices or old in self.choices:
                raise ValueError(f"{self.name}: choice alias {old!r} -> {new!r} is not a rename")
        return self

    @property
    def flag(self) -> str:
        """The CLI flag for this parameter (``max_iter`` -> ``--max-iter``)."""
        return "--" + self.name.replace("_", "-")

    def out_of_range(self, value: float) -> str | None:
        """Why ``value`` is outside the bounds, or ``None`` when it is inside them."""
        low, high = self.minimum, self.maximum
        below = low is not None and (value <= low if self.exclusive_minimum else value < low)
        above = high is not None and (value >= high if self.exclusive_maximum else value > high)
        if not (below or above):
            return None
        return f"{self.name} must be {self.range_text()}, got {value}"

    def check(
        self, value: object, method: str, *, choices: tuple[str, ...] | None = None
    ) -> object:
        """Return ``value`` as the parameter's type; raise ``TypeError`` when it is another type and
        ``ValueError`` when it is out of bounds, not a finite number, or not one of the choices.

        Checks the type, the bounds, and the choices: an int is an ``int`` or a NumPy integer, not
        a ``bool``; a float is any real number but ``NaN`` and infinity; a bool is ``True`` or
        ``False``. ``None`` passes when the parameter is nullable. Column labels or positions,
        which only the data can check, pass unchanged: a ``str_list``, a ``knowledge``, and a
        ``str`` without choices. ``choices`` replaces the spec's choices when the accepted values
        change at run time. The message names ``method``, the parameter, what it accepts
        (:meth:`accepted`), and the value.
        """
        if choices is not None:
            return self.model_copy(update={"choices": choices}).check(value, method)
        if value is None and self.nullable:
            return None
        if self.type in ("str_list", "knowledge") or (self.type == "str" and not self.choices):
            return value
        typed = _typed(self.type, value)
        message = f"{method}: {self.name} must be {self.accepted()}, got {reprlib.repr(value)}"
        nonfinite = (
            self.type == "float"
            and isinstance(value, (float, np.floating))
            and not math.isfinite(value)
        )
        if typed is None and not nonfinite:
            raise TypeError(message)
        numbers = typed if isinstance(typed, tuple) else (typed,)
        if (
            typed is None
            or (self.choices is not None and typed not in self.choices)
            or any(isinstance(x, (int, float)) and self.out_of_range(x) for x in numbers)
        ):
            raise ValueError(message)
        return typed

    def accepted(self) -> str:
        """What the parameter accepts, in words: ``one of 'a', 'b'``, ``an int at least 1``."""
        if self.choices:
            text = "one of " + ", ".join(repr(c) for c in self.choices)
        else:
            bounded = self.minimum is not None or self.maximum is not None
            kind = {
                "int": "an int",
                "float": "a number"
                if self.minimum is not None and self.maximum is not None
                else "a finite number",
                "bool": "True or False",
                "str": "a string",
                "int_pair": "a pair of ints, each",
            }[self.type]
            text = f"{kind} {self.range_text()}" if bounded else kind.removesuffix(", each")
        return f"{text}, or None" if self.nullable else text

    def range_text(self) -> str:
        """The bounds in words: ``in (0, 1)``, ``at least 0``, ``from 0 to 4294967295``."""
        low, high = _number(self.minimum), _number(self.maximum)
        if self.minimum is not None and self.maximum is not None:
            if not (self.exclusive_minimum or self.exclusive_maximum):
                return f"from {low} to {high}"
            left = "(" if self.exclusive_minimum else "["
            right = ")" if self.exclusive_maximum else "]"
            return f"in {left}{low}, {high}{right}"
        if self.minimum is not None:
            return f"above {low}" if self.exclusive_minimum else f"at least {low}"
        if self.maximum is not None:
            return f"below {high}" if self.exclusive_maximum else f"at most {high}"
        return "any value"


def _typed(kind: ParamType, value: object) -> object:
    """``value`` as a ``kind`` value (``int``, ``float``, ``bool``, ``str``, or a pair of ints), or
    ``None`` when it is not one."""
    if kind == "int_pair":
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            return None
        pair = tuple(_typed("int", x) for x in value)
        return None if None in pair else pair
    if kind == "bool":
        return bool(value) if isinstance(value, (bool, np.bool_)) else None
    if kind == "str":
        return str(value) if isinstance(value, str) else None
    if isinstance(value, (bool, np.bool_)):
        return None
    if kind == "int":
        return int(value) if isinstance(value, (int, np.integer)) else None
    if isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(value):
        return float(value)
    return None


def _number(value: float | None) -> str:
    return str(int(value)) if value is not None and float(value).is_integer() else str(value)


class OutputSpec(Spec):
    """The shape of the returned ``StructureOutput`` (what the result envelope carries)."""

    structure_type: Literal["graph", "temporal"]
    graph_kind: Literal["dag", "cpdag", "pag", "mag", "undirected"] | None  # None for temporal
    weighted: bool = False  # carries a weighted_adjacency
    has_ordering: bool = False  # carries a discovered causal order
    has_latents: bool = False  # flags latent nodes via node_types (GIN)
    multi: bool = False  # returns list[StructureOutput], one per group
    # Every key a fit's metadata can hold; a fit holds a subset (some keys depend on the settings).
    metadata: tuple[str, ...] = ()
    doc: str = ""


class MethodSpec(Spec):
    """The complete typed contract for one public method."""

    name: str  # the public callable; must equal an entry of andrey.api.__all__
    family: Family
    summary: str  # bounded "when to reach for this"
    data: DataSpec
    output: OutputSpec
    determinism: Determinism
    accel: Accel
    status: Status
    column_order: ColumnOrder
    tie_break: bool = False  # follows andrey.config.tie_break (tie_ranks; collider_order)
    order_note: str = ""  # the column-order guide's note: what still follows the column order
    params: tuple[ParamSpec, ...] = ()
    seeded: bool = False  # takes a seed (the param tagged ``seed``), which ANDREY_SEED fills
    # Former arguments the method no longer has, ignored with a DeprecatedArgumentWarning for one
    # release.
    dropped_args: tuple[str, ...] = ()
    references: tuple[str, ...] = ()
    knowledge: tuple[KnowledgePart, ...] = ()  # the Knowledge parts its knowledge= honors


class EnvVarSpec(Spec):
    """One ``ANDREY_*`` environment variable in the shared configuration list."""

    name: str
    values: str  # accepted values, human-readable
    default: str  # default when unset
    effect: str  # bounded description
    group: EnvGroup
