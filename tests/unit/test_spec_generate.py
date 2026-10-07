"""The generated agent surfaces are complete, serializable, and consistent with the registry."""

from __future__ import annotations

import json

import pytest

from andrey.spec import (
    ENV_VARS,
    REGISTRY,
    SCHEMA_VERSION,
    get_spec,
    help_schema,
    render_llms_txt,
    render_skill,
    to_json_schema,
)

_SPEC_NAMES = sorted(REGISTRY)


def test_help_schema_is_json_serializable_and_complete():
    ctx = help_schema()
    json.dumps(ctx)  # must not raise
    assert set(ctx["methods"]) == set(REGISTRY)
    assert ctx["schema_version"] == SCHEMA_VERSION
    assert len(ctx["environment"]["variables"]) == len(ENV_VARS)
    assert ctx["environment"]["precedence"]
    assert ctx["cli"]["exit_codes"]["0"]


def test_help_schema_single_method_carries_env_and_header():
    ctx = help_schema("pc")
    assert ctx["method"]["name"] == "pc"
    assert "methods" not in ctx
    assert ctx["environment"]["variables"]  # env travels with a single-method context too


@pytest.mark.parametrize("name", _SPEC_NAMES)
def test_method_schema_matches_spec(name):
    spec = get_spec(name)
    schema = to_json_schema(spec)
    json.dumps(schema)

    # Param properties are exactly the spec params, and passthrough drives additionalProperties.
    props = schema["params"]["properties"]
    assert set(props) == {p.name for p in spec.params}
    assert schema["params"]["additionalProperties"] is spec.passthrough

    for p in spec.params:
        entry = props[p.name]
        assert "default" in entry and "description" in entry
        if p.choices:
            expected = [*p.choices, None] if p.nullable else list(p.choices)
            assert entry["enum"] == expected
        if p.nullable and isinstance(entry["type"], list):
            assert "null" in entry["type"]

    assert schema["status"] == spec.status

    # Data inputs surface a flag; the CLI example runs this method.
    assert [i["name"] for i in schema["data"]["inputs"]] == [i.name for i in spec.data.inputs]
    assert schema["cli"].startswith(f"andrey run {name}")


def test_llms_txt_lists_every_method_and_env_var():
    txt = render_llms_txt()
    for name, spec in REGISTRY.items():
        tag = " (experimental)" if spec.status == "experimental" else ""
        assert f"- `{name}`{tag}: " in txt
    for v in ENV_VARS:
        assert v.name in txt


def test_skill_lists_every_method_and_env_var():
    skill = render_skill()
    for name, spec in REGISTRY.items():
        kind = spec.output.graph_kind or spec.output.structure_type
        status = ", experimental" if spec.status == "experimental" else ""
        assert f"- `{name}` ({kind}{status}): " in skill
    for v in ENV_VARS:
        assert v.name in skill
    # The guide names the executable discovery and introspection commands.
    assert "andrey config" in skill
    assert "andrey list" in skill and "--help --json" in skill


def test_skill_is_an_installable_skill_that_explains_every_edge_type():
    import yaml

    from andrey.core.structure import EDGE_GLYPHS
    from andrey.spec.generate import EDGE_MEANINGS

    skill = render_skill()
    front = yaml.safe_load(skill.split("---\n")[1])
    assert front["name"] == "andrey" and front["description"].strip()
    # Every edge type a result can carry has a meaning, and the guide shows each with its symbol.
    assert set(EDGE_MEANINGS) == set(EDGE_GLYPHS)
    for name, glyph in EDGE_GLYPHS.items():
        assert name in skill and glyph in skill
