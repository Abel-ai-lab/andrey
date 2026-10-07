"""Drift guards: the MethodSpec registry must stay locked to the live public API.

These tests are the reason the registry can be trusted as the single source of truth. They assert
the registry and ``andrey.api.__all__`` name the same methods, that every spec's inputs / params /
defaults match the real facade signature, that shared concepts stay consistent, that a supported
method is benchmarked, and that every ``ANDREY_*`` variable the source reads has a spec.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import pathlib
import re
import typing
import warnings

import pytest

import andrey
from andrey.api import __all__ as API_ALL
from andrey.spec import ENV_VARS, PRECEDENCE, REGISTRY, get_spec
from andrey.spec.models import Status

_SPEC_NAMES = sorted(REGISTRY)


def test_registry_matches_api_all_both_ways():
    """No method is in one list but not the other -- the registry cannot silently drift."""
    assert set(REGISTRY) == set(API_ALL)


@pytest.mark.parametrize("name", _SPEC_NAMES)
def test_signature_agreement(name):
    """Data inputs, keyword params, defaults, and ``**params`` passthrough match the live signature.

    Convention across every facade: array inputs are positional-or-keyword (before ``*``);
    hyperparameters are keyword-only; a ``**params`` passthrough maps to ``spec.passthrough``.
    """
    spec = get_spec(name)
    sig = inspect.signature(getattr(andrey, name))
    params = list(sig.parameters.values())

    data_names = [p.name for p in params if p.kind is p.POSITIONAL_OR_KEYWORD]
    kw_params = {p.name: p for p in params if p.kind is p.KEYWORD_ONLY}
    has_var_kw = any(p.kind is p.VAR_KEYWORD for p in params)
    assert not any(p.kind is p.VAR_POSITIONAL for p in params), "no facade uses *args"

    # Array inputs: same names, same order.
    assert [i.name for i in spec.data.inputs] == data_names

    # Keyword hyperparameters: same set, and each default matches the signature exactly.
    spec_params = {p.name: p for p in spec.params}
    assert set(spec_params) == set(kw_params)
    for pname, sig_param in kw_params.items():
        assert spec_params[pname].default == sig_param.default, f"{name}.{pname} default drift"

    # A **params passthrough is declared iff the facade accepts one.
    assert spec.passthrough is has_var_kw


def test_shared_concepts_are_consistent():
    """One concept, one type -- for example, ``alpha`` is always a float, seeds are always
    nullable ints."""
    for spec in REGISTRY.values():
        for p in spec.params:
            if p.name == "alpha":
                assert p.type == "float"
            if p.name in ("seed", "random_state"):
                assert p.type == "int" and p.nullable


def test_choices_include_the_default():
    for spec in REGISTRY.values():
        for p in spec.params:
            if p.choices and p.default is not None:
                assert p.default in p.choices


def _type_matches(ptype: str, default: object) -> bool:
    if ptype == "bool":
        return isinstance(default, bool)
    if ptype == "int":  # bool is an int subclass -- exclude it
        return isinstance(default, int) and not isinstance(default, bool)
    if ptype == "float":
        return isinstance(default, (int, float)) and not isinstance(default, bool)
    if ptype == "str":
        return isinstance(default, str)
    if ptype == "int_pair":
        return isinstance(default, tuple) and all(isinstance(x, int) for x in default)
    if ptype == "str_list":
        return isinstance(default, (tuple, list))
    return False


def test_param_type_is_consistent_with_default():
    """A spec's declared ``type`` must match its default's Python type (catches alpha typed str)."""
    for spec in REGISTRY.values():
        for p in spec.params:
            if p.default is None:
                assert p.nullable, f"{spec.name}.{p.name}: None default but not nullable"
                continue
            assert _type_matches(p.type, p.default), (
                f"{spec.name}.{p.name}: default {p.default!r} disagrees with type {p.type!r}"
            )


def test_param_and_input_names_avoid_cli_reserved():
    """A param may not shadow a CLI/reserved flag or a grammar literal; inputs and params are
    disjoint per method; and the CLI's input flags cover every input name in the registry."""
    from andrey.cli import _INPUT_FLAGS, _RESERVED_PARAM_NAMES

    literals = {"null", "true", "false"}
    for spec in REGISTRY.values():
        param_names = {p.name for p in spec.params}
        input_names = {i.name for i in spec.data.inputs}
        assert not (param_names & _RESERVED_PARAM_NAMES), (
            f"{spec.name}: param shadows a reserved name {param_names & _RESERVED_PARAM_NAMES}"
        )
        assert not (param_names & literals), f"{spec.name}: a param shadows a grammar literal"
        assert not (param_names & input_names), f"{spec.name}: an input and param share a name"
        assert input_names <= set(_INPUT_FLAGS), (
            f"{spec.name}: input {input_names - set(_INPUT_FLAGS)} has no CLI flag"
        )
        for p in spec.params:
            assert p.name.isidentifier() and not any(ch in p.name for ch in ". []=,")
            assert p.name == p.name.lower(), f"{spec.name}.{p.name}: a param name is lowercase"


def _docstring_param_names(doc: str) -> list[str]:
    """Names declared in a NumPy-style ``Parameters`` section (the ``name : type`` entries)."""
    lines = doc.splitlines()
    names: list[str] = []
    in_params = False
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if (
            stripped == "Parameters"
            and idx + 1 < len(lines)
            and set(lines[idx + 1].strip()) == {"-"}
        ):
            in_params = True
            continue
        if not in_params:
            continue
        if stripped in ("Returns", "Raises", "Examples", "Notes", "See Also", "Yields"):
            break
        # getdoc dedents, so a param entry sits at column 0 (`name : type`); its description prose
        # is indented, so it never matches -- that is how names and prose stay apart.
        m = re.match(r"^([A-Za-z_]\w*) : ", line)
        if m:
            names.append(m.group(1))
    return names


@pytest.mark.parametrize("name", sorted(set(REGISTRY) & set(API_ALL)))
def test_facade_docstring_documents_every_parameter(name):
    """Every facade docstring names exactly the signature's arguments in ``Parameters``, plus a
    summary line and ``Returns`` / ``Examples`` sections. A ``**params`` passthrough (calm) is not a
    signature parameter, so it drops out of ``sig_names``; its ``**params`` doc entry is likewise
    skipped -- the parser matches only ``name :`` lines."""
    fn = getattr(andrey, name)
    doc = inspect.getdoc(fn)
    assert doc and doc.splitlines()[0].strip(), f"{name}: missing docstring summary line"
    assert re.search(r"^Returns\n-{3,}", doc, re.MULTILINE), f"{name}: no Returns section"
    # Examples are half the standard (they feed the gallery); the doctest runner only checks the
    # ones that exist, so enforce presence here or a facade could silently drop its example.
    assert re.search(r"^Examples\n-{3,}", doc, re.MULTILINE), f"{name}: no Examples section"
    sig_names = [
        p.name for p in inspect.signature(fn).parameters.values() if p.kind is not p.VAR_KEYWORD
    ]
    doc_names = _docstring_param_names(doc)
    assert doc_names == sig_names, f"{name}: Parameters {doc_names} != signature {sig_names}"


_PUBLIC_CORE = [
    "Structure",
    "GraphStructure",
    "TemporalStructure",
    "SummaryGraph",
    "StructureOutput",
]


def _own_public_methods(cls):
    """Public methods defined on ``cls`` itself (not inherited, not properties/attrs)."""
    for name, member in inspect.getmembers(cls):
        if name.startswith("_") or name not in cls.__dict__:
            continue
        if isinstance(inspect.getattr_static(cls, name, None), property) or not callable(member):
            continue
        yield name, member


@pytest.mark.parametrize("cls_name", _PUBLIC_CORE)
def test_public_core_methods_document_their_parameters(cls_name):
    """The public core API Sphinx autodoc renders reads consistently with the facades: every
    public method carries a docstring, and an argument-taking one names its arguments in a
    ``Parameters`` section. Properties stay concise prose, so they are exempt."""
    import andrey.core as core

    cls = getattr(core, cls_name)
    for name, member in _own_public_methods(cls):
        sig = inspect.signature(member)
        doc = inspect.getdoc(member)
        assert doc and doc.splitlines()[0].strip(), f"{cls_name}.{name}: missing docstring"
        args = [
            p.name
            for p in sig.parameters.values()
            if p.name not in ("self", "cls") and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)
        ]
        if args:
            doc_names = _docstring_param_names(doc)
            assert doc_names == args, f"{cls_name}.{name}: Parameters {doc_names} != args {args}"
        # A value-returning method needs a Returns section; a None-returning one (validate) is not.
        if (
            sig.return_annotation is not inspect.Signature.empty
            and str(sig.return_annotation) != "None"
        ):
            assert re.search(r"^Returns\n-{3,}", doc, re.MULTILINE), (
                f"{cls_name}.{name}: returns a value but has no Returns section"
            )


def test_cmd_run_wires_every_input_flag():
    """Every `_INPUT_FLAGS` entry is an actual `cmd_run` option, so an input added to the registry
    and `_INPUT_FLAGS` but not to `cmd_run` fails here instead of being unreachable at runtime."""
    from andrey.cli import _INPUT_FLAGS, cmd_run

    params = set(inspect.signature(cmd_run).parameters)
    missing = set(_INPUT_FLAGS) - params
    assert not missing, f"input flags with no cmd_run option: {missing}"


def test_seeded_methods_expose_a_seed_param():
    for spec in REGISTRY.values():
        if spec.seeded:
            names = {p.name for p in spec.params}
            assert names & {"seed", "random_state"}, f"{spec.name} is seeded but has no seed param"


def test_multi_output_only_in_family_that_supports_it():
    for spec in REGISTRY.values():
        if spec.output.multi:
            assert spec.name == "multi_group_direct_lingam"


# --- environment variables ------------------------------------------------------------------------


def test_env_vars_are_named_and_ordered():
    assert ENV_VARS
    for v in ENV_VARS:
        assert v.name.startswith("ANDREY_")
        assert v.effect and v.values
    assert PRECEDENCE


def test_env_vars_match_source_vars_both_ways():
    """Every ``ANDREY_*`` name in the package has an ``EnvVarSpec``, and code outside
    ``core/env.py`` reads every setting. ``tests/gpu/conftest.py`` reads ``ANDREY_REQUIRE_GPU``."""
    from andrey.core import env

    root = pathlib.Path(__file__).resolve().parents[2]
    src = root / "src" / "andrey"
    pattern = re.compile(r"ANDREY_[A-Z_]+")
    named = {var for py in src.rglob("*.py") for var in pattern.findall(py.read_text())}
    documented = {v.name for v in ENV_VARS}
    assert named <= documented, f"undocumented env vars: {named - documented}"
    readers = [py for py in src.rglob("*.py") if py != src / "core" / "env.py"]
    text = "\n".join(py.read_text() for py in [*readers, root / "tests" / "gpu" / "conftest.py"])
    unread = {
        setting.name
        for attr, setting in vars(env).items()
        if isinstance(setting, env.Setting) and not re.search(rf"\benv\.{attr}\b", text)
    }
    assert not unread, f"settings no code reads: {unread}"


def test_every_env_var_has_one_parser():
    from andrey.core.env import SETTINGS

    assert set(SETTINGS) == {v.name for v in ENV_VARS}


def test_indep_test_choices_track_the_shipped_ci_registry():
    """Every method's ``indep_test`` choices are the shipped CI-registry names, so the CLI/JSON
    schema the registry renders cannot drift from what the facades actually dispatch."""
    from andrey.core.ci import SHIPPED_INDEP_TESTS

    params = {s.name: p for s in REGISTRY.values() for p in s.params if p.name == "indep_test"}
    assert params
    for name, p in params.items():
        assert p.choices == SHIPPED_INDEP_TESTS, name


# --- support status -------------------------------------------------------------------------------
#
# Supported means every criterion in docs/docs/code/index.md's "Experimental methods" section holds.

ROOT = pathlib.Path(__file__).resolve().parents[2]
SUPPORTED = sorted(n for n, s in REGISTRY.items() if s.status == "supported")
EXPERIMENTAL = sorted(n for n, s in REGISTRY.items() if s.status == "experimental")
GRADUATE = "mark it experimental until it meets the criteria in docs/docs/code/index.md"


def _normalized(name: str) -> str:
    """``ICA-LiNGAM``, ``ICALiNGAM``, and ``ica_lingam`` all read ``icalingam``."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _published() -> set[str]:
    """Methods in the summary the homepage and README publish (``SUMMARY`` in ``site/build.py``)."""
    spec = importlib.util.spec_from_file_location("site_build", ROOT / "site" / "build.py")
    assert spec and spec.loader
    site = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(site)
    return {_normalized(m["method"]) for m in json.loads(site.SUMMARY.read_text())["methods"]}


def _calls(source: str) -> set[str]:
    """Methods called as ``andrey.<name>(...)``, or by a name imported from ``andrey[.api]``."""
    tree = ast.parse(source)
    imported = {
        alias.asname or alias.name: alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and re.fullmatch(r"andrey(\.api(\.\w+)?)?", node.module or "")
        for alias in node.names
    }
    called = set()
    for node in ast.walk(tree):
        f = node.func if isinstance(node, ast.Call) else None
        if isinstance(f, ast.Attribute) and getattr(f.value, "id", None) == "andrey":
            called.add(f.attr)
        elif isinstance(f, ast.Name) and f.id in imported:
            called.add(imported[f.id])
    return called


def _notebook_code(path: pathlib.Path) -> str:
    """A notebook's code cells as one module, each shell or magic line replaced by ``pass``."""
    cells = json.loads(path.read_text())["cells"]
    lines = (
        line
        for cell in cells
        if cell["cell_type"] == "code"
        for line in "".join(cell["source"]).splitlines()
    )
    return "\n".join(
        line[: len(line) - len(line.lstrip())] + "pass"
        if line.lstrip().startswith(("!", "%"))
        else line
        for line in lines
    )


def test_every_method_has_a_status():
    assert {spec.status for spec in REGISTRY.values()} <= set(typing.get_args(Status))
    assert SUPPORTED and EXPERIMENTAL


def test_supported_means_published():
    """Exactly the supported methods appear in the published benchmark summary."""
    assert _published() == {_normalized(n) for n in SUPPORTED}, GRADUATE


@pytest.mark.parametrize("name", SUPPORTED)
def test_supported_method_has_a_benchmarks_section(name):
    page = (ROOT / "docs" / "docs" / "benchmarks.md").read_text()
    sections = {
        _normalized(part)
        for heading in re.findall(r"^### (.+)$", page, re.MULTILINE)
        for part in re.split(r",| and ", heading)
    }
    missing = f"{name}: no section on docs/docs/benchmarks.md; {GRADUATE}"
    assert _normalized(name) in sections, missing


@pytest.mark.parametrize("name", SUPPORTED)
def test_supported_method_has_a_recovery_baseline(name):
    recorded = {
        _normalized(json.loads(path.read_text())["algorithm"])
        for path in (ROOT / "tests" / "recovery" / "baselines").glob("*/*.json")
    }
    assert _normalized(name) in recorded, f"{name}: no recovery baseline; {GRADUATE}"


def test_quality_gate_checks_exactly_the_supported_methods():
    from qa import quality_gate

    gated = {_normalized(n) for n in quality_gate.SUPPORTED}
    assert gated == {_normalized(n) for n in SUPPORTED}, GRADUATE


@pytest.mark.parametrize("name", SUPPORTED)
def test_supported_method_has_an_example(name):
    notebooks = (ROOT / "examples").glob("*.ipynb")
    called = set().union(*(_calls(_notebook_code(path)) for path in notebooks))
    assert name in called, f"{name}: no example notebook calls it; {GRADUATE}"


@pytest.mark.parametrize("name", EXPERIMENTAL)
def test_experimental_method_has_a_recovery_test(name):
    tests = (ROOT / "tests" / "recovery").glob("test_*.py")
    assert any(name in _calls(path.read_text()) for path in tests), f"no recovery test calls {name}"


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_facade_warns_first_exactly_when_experimental(name):
    """An experimental facade warns before it reads its inputs; a supported one never warns.

    Placeholder inputs fail validation, so a warning placed after validation never fires here.
    """
    fn = getattr(andrey, name)
    experimental = get_spec(name).status == "experimental"
    with warnings.catch_warnings():
        warnings.simplefilter("error", andrey.ExperimentalWarning)
        with pytest.raises(Exception) as raised:
            fn(*[None] * len(get_spec(name).data.inputs))
    assert (raised.type is andrey.ExperimentalWarning) is experimental, raised.value
    assert ("ExperimentalWarning" in (inspect.getdoc(fn) or "")) is experimental


def test_api_reference_lists_each_method_under_its_status():
    page = (ROOT / "docs" / "docs" / "code" / "index.md").read_text()
    supported, rest = page.split("### Supported methods", 1)[1].split("### Experimental methods", 1)
    experimental = rest.split("\n## ", 1)[0]

    def listed(text: str) -> list[str]:
        return sorted(re.findall(r"^   (\w+)$", text, re.MULTILINE))

    assert listed(supported) == SUPPORTED
    assert listed(experimental) == EXPERIMENTAL
