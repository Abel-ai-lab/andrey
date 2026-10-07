"""Immutable Pydantic models for method contracts and the environment variables.

Models validate at construction and reject unknown fields with ``extra="forbid"``. Tests check
method specs against live facade signatures. See :mod:`andrey.spec` for the generated interfaces.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

Family = Literal["constraint", "score", "permutation", "lingam", "temporal", "latent", "pairwise"]
Determinism = Literal["deterministic", "stochastic"]
Accel = Literal["numpy", "numba", "torch"]
# supported: benchmarked and gated before each release. experimental: warns once; API may change.
Status = Literal["supported", "experimental"]

# How an array input reaches a method -- the CLI wires each to a --flag, the schema to a type.
InputKind = Literal[
    "matrix",  # one (n_samples, n_variables) 2-D array
    "panel",  # a sequence of 2-D arrays, one per group/time point (repeated flag)
    "context",  # an auxiliary (n_samples, k) matrix aligned to the data rows (cdnod)
]
# Hyperparameter value types -- map to a JSON-schema type and a CLI option.
ParamType = Literal["float", "int", "str", "bool", "int_pair", "str_list"]

EnvGroup = Literal["backend", "reproducibility", "determinism", "data"]

_FROZEN = ConfigDict(frozen=True, extra="forbid")


class Spec(BaseModel):
    """Immutable spec model that rejects unknown fields."""

    model_config = _FROZEN


class InputSpec(Spec):
    """One array input to a method (matches the facade parameter name)."""

    name: str  # facade parameter name: "data", "c_indx", "x", "y", "data_groups", "data_list"
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


class ParamSpec(Spec):
    """One keyword hyperparameter (matches the facade keyword and its default)."""

    name: str
    type: ParamType
    default: Any  # JSON-safe default; ``None`` allowed when ``nullable``
    doc: str
    choices: tuple[str, ...] | None = None  # the supported values, when enumerable
    nullable: bool = False  # accepts ``None`` (for example, seed, criterion, labels)

    @property
    def flag(self) -> str:
        """The CLI flag for this parameter (``max_iter`` -> ``--max-iter``)."""
        return "--" + self.name.replace("_", "-")


class OutputSpec(Spec):
    """The shape of the returned ``StructureOutput`` (what the result envelope carries)."""

    structure_type: Literal["graph", "temporal"]
    graph_kind: Literal["dag", "cpdag", "pag"] | None  # None for temporal
    weighted: bool = False  # carries a weighted_adjacency
    has_ordering: bool = False  # carries a discovered causal order
    has_latents: bool = False  # flags latent nodes via node_types (GIN)
    multi: bool = False  # returns list[StructureOutput], one per group
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
    params: tuple[ParamSpec, ...] = ()
    seeded: bool = False  # honors a seed / random_state (and so ANDREY_SEED)
    passthrough: bool = False  # accepts extra **params (CALM optimizer knobs)
    references: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()


class EnvVarSpec(Spec):
    """One ``ANDREY_*`` environment variable in the shared configuration list."""

    name: str
    values: str  # accepted values, human-readable
    default: str  # default when unset
    effect: str  # bounded description
    group: EnvGroup
