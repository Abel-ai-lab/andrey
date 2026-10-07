"""Typed method registry and generated agent interfaces.

One :class:`MethodSpec` per public method (``andrey.api.__all__``) supplies the JSON tool schema,
``llms.txt``, generated ``SKILL.md``, and ``andrey`` CLI. The shared ``ANDREY_*`` environment
variables are included. The Pydantic specs require only base dependencies.
"""

from __future__ import annotations

from .environment import ENV_VARS, PRECEDENCE
from .generate import (
    EXIT_CODES,
    SCHEMA_VERSION,
    help_schema,
    render_llms_txt,
    render_skill,
    to_json_schema,
)
from .models import (
    DataSpec,
    EnvVarSpec,
    InputSpec,
    MethodSpec,
    OutputSpec,
    ParamSpec,
)
from .registry import REGISTRY, get_spec, list_specs

__all__ = [
    "MethodSpec",
    "DataSpec",
    "InputSpec",
    "ParamSpec",
    "OutputSpec",
    "EnvVarSpec",
    "REGISTRY",
    "get_spec",
    "list_specs",
    "ENV_VARS",
    "PRECEDENCE",
    "SCHEMA_VERSION",
    "EXIT_CODES",
    "to_json_schema",
    "help_schema",
    "render_llms_txt",
    "render_skill",
]
