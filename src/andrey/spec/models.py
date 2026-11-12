"""Immutable Pydantic models for method contracts and the environment variables.

Models validate at construction and reject unknown fields with ``extra="forbid"``. Tests check
method specs against live facade signatures. See :mod:`andrey.spec` for the generated interfaces.
"""

from __future__ import annotations

import math
import reprlib
from typing import Any, Literal, get_args

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from andrey.core.ci import CONTINUOUS_INDEP_TESTS, DISCRETE_INDEP_TESTS, MIXED_INDEP_TESTS
from andrey.core.ci_missing import EM, EM_SUPPORTED
from andrey.core.knowledge import EDGE_PARTS, KnowledgePart
from andrey.core.score import DISCRETE_SCORES, LINEAR_SCORES, MIXED_SCORES

Family = Literal[
    "constraint", "score", "permutation", "functional", "temporal", "latent", "pairwise"
]
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
    "context",  # an auxiliary input aligned to the data rows: one value per row (env)
]
# Hyperparameter value types -- map to a JSON-schema type and a CLI option.
# knowledge: an object with required/forbidden/allowed edge lists and tiers (andrey.Knowledge).
# targets: an object mapping each environment's label to its intervened variables (gies).
ParamType = Literal["float", "int", "str", "bool", "int_pair", "str_list", "knowledge", "targets"]

EnvGroup = Literal["backend", "reproducibility", "determinism", "data"]
# The kinds of data a method reads (MethodSpec.data_kinds), in the order the pages list them:
# column types, missing values, data from several environments or datasets, and the settings its
# result implies.
DataKind = Literal[
    "continuous",
    "categorical",
    "mixed",
    "missing",
    "environments",
    "datasets",
    "hidden_causes",
    "time_series",
    "two_variables",
]
DATA_KINDS: tuple[DataKind, ...] = get_args(DataKind)

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
    # Former names, accepted for one release after a rename.
    aliases: tuple[str, ...] = ()

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
    steps: bool = False  # a panel's rows are time steps, so a resample keeps them in order


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
        which only the data can check, pass unchanged: a ``str_list``, a ``knowledge``, a
        ``targets``, and a ``str`` without choices. ``choices`` replaces the spec's choices when
        the accepted values change at run time. The message names ``method``, the parameter, what
        it accepts (:meth:`accepted`), and the value.
        """
        if choices is not None:
            return self.model_copy(update={"choices": choices}).check(value, method)
        if value is None and self.nullable:
            return None
        if self.type in ("str_list", "knowledge", "targets") or (
            self.type == "str" and not self.choices
        ):
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
    accel: Accel
    status: Status
    column_order: ColumnOrder
    # The published algorithm reads the columns' units, so rescaling a column can change the
    # default fit's graph; every other method returns the same graph whatever factor rescales each
    # column.
    reads_units: bool = False
    tie_break: bool = False  # follows andrey.config.tie_break (tie_ranks; collider_order)
    order_note: str = ""  # the column-order guide's note: what still follows the column order
    params: tuple[ParamSpec, ...] = ()
    # Former arguments the method no longer has, ignored with a DeprecatedArgumentWarning for one
    # release.
    dropped_args: tuple[str, ...] = ()
    references: tuple[str, ...] = ()
    # The parts of andrey.Knowledge the method honors beyond the edge parts every method with
    # knowledge honors, and the parts it honors experimentally (ExperimentalWarning on use).
    knowledge_extra: tuple[KnowledgePart, ...] = ()
    knowledge_experimental: tuple[KnowledgePart, ...] = ()
    # Whether required_paths and forbidden_paths restrict the CPDAG the method returns, after the
    # fit, rather than reaching the method's search.
    restricts_paths: bool = False

    @property
    def seeded(self) -> bool:
        """Whether the method takes a seed, the param tagged ``seed``, which ANDREY_SEED fills."""
        return any(p.concept == "seed" for p in self.params)

    @property
    def determinism(self) -> Determinism:
        """``stochastic`` for a seeded method, else ``deterministic``."""
        return "stochastic" if self.seeded else "deterministic"

    @property
    def data_kinds(self) -> tuple[DataKind, ...]:
        """The kinds of data the method reads, in :data:`DATA_KINDS` order, derived from the spec.

        - ``continuous``: a method without a test or score models continuous columns; one with
          a test or score reads them when it offers a continuous test or the linear-Gaussian BIC.
        - ``categorical`` and ``mixed``: it offers a categorical or a mixed-data test or score.
        - ``missing``: it accepts missing values in a supported way, by test-wise deletion
          (``data.missing``) or by structural EM where EM is supported for it.
        - ``environments``: it takes ``env``, the environment of each row; ``datasets``: it takes
          a list of datasets over the same variables (a panel input).
        - ``hidden_causes``: its graph allows hidden common causes: a PAG or a MAG, latent nodes,
          or the ``confounded`` part of background knowledge.
        - ``time_series``: it returns a graph over time; ``two_variables``: it decides the
          direction between two variables (the pairwise family).
        """
        tests = [p for p in self.params if p.name in ("indep_test", "score_func")]
        offered = {choice for p in tests for choice in p.choices or ()}
        missing = next((p.choices or () for p in self.params if p.name == "missing"), None)
        em = missing is not None and EM in missing and self.name in EM_SUPPORTED
        found = {
            "continuous": not tests or bool(offered & {*CONTINUOUS_INDEP_TESTS, *LINEAR_SCORES}),
            "categorical": bool(offered & {*DISCRETE_INDEP_TESTS, *DISCRETE_SCORES}),
            "mixed": bool(offered & {*MIXED_INDEP_TESTS, *MIXED_SCORES}),
            "missing": missing is not None and (self.data.missing or em),
            "environments": any(i.name == "env" for i in self.data.inputs),
            "datasets": any(i.kind == "panel" for i in self.data.inputs),
            "hidden_causes": self.output.graph_kind in ("pag", "mag")
            or self.output.has_latents
            or "confounded" in self.knowledge,
            "time_series": self.output.structure_type == "temporal",
            "two_variables": self.family == "pairwise",
        }
        return tuple(kind for kind in DATA_KINDS if found[kind])

    @property
    def knowledge(self) -> tuple[KnowledgePart, ...]:
        """The parts of an ``andrey.Knowledge`` the method honors: the edge parts and
        ``knowledge_extra`` when it takes ``knowledge``, else none."""
        if not any(p.type == "knowledge" for p in self.params):
            return ()
        return (*EDGE_PARTS, *self.knowledge_extra)


class EnvVarSpec(Spec):
    """One ``ANDREY_*`` environment variable in the shared configuration list."""

    name: str
    values: str  # accepted values, human-readable
    default: str  # default when unset
    effect: str  # bounded description
    group: EnvGroup
