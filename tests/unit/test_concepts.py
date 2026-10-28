"""Every package-wide concept against every method (``andrey.spec.concepts``).

A suitable method's cell is the concept's template unless a deviation is recorded; an exempt
method lacks the concept. Each recorded deviation is checked to still hold, so the change that
removes one fails here until it deletes the record.
"""

from __future__ import annotations

import functools
import inspect
import pathlib
import re
import warnings

import numpy as np
import pandas as pd
import pytest

import andrey
from andrey.api import _method as method_wrapper
from andrey.core import ci, warning_policy
from andrey.core.knowledge import KNOWLEDGE_PARTS
from andrey.spec import REGISTRY
from andrey.spec.concepts import CONCEPTS, Concept

API_MD = pathlib.Path(__file__).resolve().parents[2] / "docs" / "docs" / "api.md"
#: Public callables outside the registry that a concept may name.
UTILITIES = ("bootstrap",)

PARAMETERS = [c for c in CONCEPTS.values() if c.param is not None]
BEHAVIORS = [c for c in CONCEPTS.values() if c.param is None]


def cells(concepts: list[Concept]) -> list:
    return [pytest.param(c, m, id=f"{c.key}-{m}") for c in concepts for m in REGISTRY]


def signature(method: str) -> dict[str, inspect.Parameter]:
    return dict(inspect.signature(getattr(andrey, method)).parameters)


def tagged(concept: Concept, method: str):
    """The method's parameter that implements the concept, if any."""
    return next((p for p in REGISTRY[method].params if p.concept == concept.key), None)


def lacks(concept: Concept, method: str) -> None:
    """No parameter of the method implements the concept or takes its name for another one,
    unless that other concept records the name as a deviation."""
    assert tagged(concept, method) is None
    name = concept.param.name  # ty: ignore[possibly-missing-attribute]
    other = next((p for p in REGISTRY[method].params if p.name == name), None)
    if other is not None:
        assert other.concept in CONCEPTS, f"{method}.{name} implements no concept"
        deviation = CONCEPTS[other.concept].deviation(method)
        assert deviation is not None and deviation.name == name, f"{method}.{name}: two meanings"
    elif name in signature(method):
        pytest.fail(f"{method} takes {name} without a spec")


@pytest.mark.parametrize("concept", list(CONCEPTS.values()), ids=lambda c: c.key)
def test_every_method_is_suitable_or_exempt(concept):
    """A new method fails here until each concept lists it as suitable or exempt."""
    listed = set(concept.suitable) | {e.method for e in concept.exempt}
    assert listed <= set(REGISTRY)
    if not concept.others_exempt:
        assert listed == set(REGISTRY), sorted(set(REGISTRY) - listed)
    assert all(e.reason for e in concept.exempt)
    for deviation in concept.deviations:
        assert deviation.method in (*REGISTRY, *UTILITIES)
        assert deviation.reason
        if deviation.method not in concept.suitable:  # an exempt method that has it anyway
            assert deviation.name is not None


@pytest.mark.parametrize(("concept", "method"), cells(PARAMETERS))
def test_parameter_cell(concept, method):
    template, deviation = concept.param, concept.deviation(method)
    live = signature(method)
    param = tagged(concept, method)
    if deviation is not None and deviation.name is None:
        lacks(concept, method)
    elif deviation is not None:
        assert param is not None, f"{method} has no {deviation.name}"
        assert (param.name, param.default, param.choices) == (
            deviation.name,
            deviation.default,
            deviation.choices,
        )
        assert live[deviation.name].default == deviation.default
    elif method in concept.suitable:
        assert param is not None, f"{method} lacks {template.name}"
        same = ("name", "type", "choices", "minimum", "maximum")
        same += ("exclusive_minimum", "exclusive_maximum")
        if not concept.default_rule:
            same += ("default", "nullable")
        assert {f: getattr(param, f) for f in same} == {f: getattr(template, f) for f in same}
        assert live[param.name].default == param.default
    else:
        lacks(concept, method)
    if concept.key == "knowledge" and param is not None:
        parts = deviation.observed.split(",") if deviation else KNOWLEDGE_PARTS
        assert REGISTRY[method].knowledge == tuple(parts)


@pytest.mark.parametrize(
    ("concept", "deviation"),
    [
        pytest.param(c, d, id=f"{c.key}-{d.method}")
        for c in CONCEPTS.values()
        for d in c.deviations
        if d.method in UTILITIES
    ],
)
def test_utility_cell(concept, deviation):
    live = signature(deviation.method)
    assert deviation.name in live and live[deviation.name].default == deviation.default


def docstring_entries(doc: str) -> dict[str, str]:
    """Each ``Parameters`` entry's description, lowercase, with markup and line breaks removed."""
    entries: dict[str, list[str]] = {}
    section = doc.split("Parameters\n----------\n", 1)[1]
    section = re.split(r"\n(Returns|Raises|Notes|Examples|See Also)\n-{3,}", section)[0]
    name = None
    for line in section.splitlines():
        match = re.match(r"^([A-Za-z_]\w*) : ", line)
        if match:
            name = match.group(1)
            entries[name] = []
        elif name:
            entries[name].append(line)
    return {k: " ".join(" ".join(v).replace("`", "").lower().split()) for k, v in entries.items()}


@pytest.mark.parametrize(
    ("concept", "method"),
    [
        pytest.param(c, m, id=f"{c.key}-{m}")
        for c in PARAMETERS
        for m in REGISTRY
        if any(p.concept == c.key for p in REGISTRY[m].params)
    ],
)
def test_docstring_entry_states_the_concept(concept, method):
    deviation = concept.deviation(method)
    fragment = concept.fragment
    if deviation is not None and deviation.fragment is not None:
        fragment = deviation.fragment
    name = tagged(concept, method).name  # ty: ignore[possibly-missing-attribute]
    entry = docstring_entries(inspect.getdoc(getattr(andrey, method)) or "")[name]
    assert fragment.lower() in entry


# --- behaviors ------------------------------------------------------------------------------------

N = 200


def frame(seed: int, *, codes: bool) -> pd.DataFrame:
    """A chain x -> y -> z; with ``codes``, x holds four whole numbers that look like categories."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(-1, 1, size=(N, 3)) ** 3
    x = rng.integers(0, 4, N).astype(float) if codes else u[:, 0]
    y = 0.8 * x + u[:, 1]
    return pd.DataFrame({"x": x, "y": y, "z": 0.7 * y + u[:, 2]})


def inputs(method: str, *, codes: bool) -> list:
    spec = REGISTRY[method]
    frames = [frame(seed, codes=codes) for seed in range(3)]
    args = []
    for inp in spec.data.inputs:
        if not inp.required:
            continue
        if inp.kind == "matrix":
            args.append(frames[0].iloc[:, :2] if spec.family == "pairwise" else frames[0])
        elif inp.kind == "panel":
            args.append(frames)
        else:
            args.append(np.repeat([0.0, 1.0], N // 2).reshape(-1, 1))
    return args


#: Settings that shorten a slow fit, and the settings that stop each loop at its cap.
SHORT = {"calm": {"max_outer": 1, "subproblem_iter": 20}}
AT_CAP = {"calm": {"max_outer": 1, "subproblem_iter": 1}, "dagma": {"stages": 1, "warm_iter": 1}}


def fit(method: str, *, codes: bool, **kwargs):
    if method == "calm":
        pytest.importorskip("torch")
    warning_policy._warned.clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = getattr(andrey, method)(*inputs(method, codes=codes), **kwargs)
    return (out[0] if isinstance(out, list) else out), [w.category for w in caught]


@functools.cache
def coded_fit(method: str):
    return fit(method, codes=True, **SHORT.get(method, {}))


def stopped_at_cap(method: str) -> str:
    cap = AT_CAP.get(method, {"max_iter": 1})
    _, caught = fit(method, codes=False, **{**SHORT.get(method, {}), **cap})
    names = {c.__name__ for c in caught}
    return next((n for n in ("SearchLimitWarning", "ConvergenceWarning") if n in names), "")


def non_value_errors(method: str) -> str:
    """The parameters whose unknown value raises something other than ValueError."""
    raised = []
    for param in REGISTRY[method].params:
        if not param.choices:
            continue
        try:
            getattr(andrey, method)(*inputs(method, codes=False), **{param.name: "unknown"})
        except ValueError:
            continue
        except Exception:  # noqa: BLE001 - the kind of error is the observation
            raised.append(param.name)
        else:
            raised.append(param.name)
    return ",".join(sorted(raised))


#: How each behavior is observed, and the value a conforming method shows.
OBSERVE = {
    "missing": (lambda m: REGISTRY[m].data.missing, True),
    "data": (lambda m: REGISTRY[m].data.inputs[0].name, "data"),
    "graph_kind": (lambda m: REGISTRY[m].output.graph_kind, None),
    "algorithm": (lambda m: coded_fit(m)[0].metadata["algorithm"], "{method}"),
    "category_codes": (
        lambda m: "DataTypeWarning" * any(c is andrey.DataTypeWarning for c in coded_fit(m)[1]),
        "DataTypeWarning",
    ),
    "cap_warning": (stopped_at_cap, "SearchLimitWarning"),
    "invalid_choice": (non_value_errors, ""),
}


def test_every_behavior_has_an_observation():
    assert set(OBSERVE) == {c.key for c in BEHAVIORS}


@pytest.mark.parametrize(
    ("concept", "method"),
    [
        pytest.param(c, m, id=f"{c.key}-{m}")
        for c in BEHAVIORS
        for m in REGISTRY
        if m in c.suitable or c.deviation(m) or c.key == "missing"
    ],
)
def test_behavior_cell(concept, method):
    observe, expected = OBSERVE[concept.key]
    deviation = concept.deviation(method)
    observed = observe(method)
    if deviation is not None:
        assert observed == (deviation.observed if deviation.name is None else deviation.name)
    elif method not in concept.suitable:
        assert observed is False
    elif expected is not None:
        assert observed == (
            expected.format(method=method) if isinstance(expected, str) else expected
        )


# --- the wrapper every method runs in ------------------------------------------------------------


def bad_values(param) -> list:
    """Values ``param`` refuses: another type, each side of its bounds, an unknown choice."""
    if param.choices:
        return ["unknown", 1]
    values = {"int": [1.5, True], "float": ["0.5", float("nan")], "bool": [1, "yes"]}.get(
        param.type, [[1.5, 1]] if param.type == "int_pair" else []
    )
    step = 1 if param.type in ("int", "int_pair") else 0.5
    if param.minimum is not None:
        low = param.minimum if param.exclusive_minimum else param.minimum - step
        values.append(low if param.type != "int_pair" else (int(low), 1))
    if param.maximum is not None:
        high = param.maximum if param.exclusive_maximum else param.maximum + step
        values.append(high)
    return values


@pytest.mark.parametrize(
    ("method", "param", "value"),
    [
        pytest.param(m, p, v, id=f"{m}-{p.name}-{v!r}")
        for m, spec in REGISTRY.items()
        for p in spec.params
        for v in bad_values(p)
    ],
)
def test_a_bad_value_raises_one_value_error_shape(method, param, value):
    """Every parameter of every method refuses a bad value with one message: the method, the
    parameter, what it accepts, and the value."""
    with pytest.raises(ValueError, match=rf"^{method}: {param.name} must be .+, got "):
        getattr(andrey, method)(*inputs(method, codes=False), **{param.name: value})


SEEDED = [m for m, spec in REGISTRY.items() if spec.seeded]


@pytest.mark.parametrize("method", SEEDED)
def test_a_seeded_fit_records_its_seed(method, monkeypatch):
    seed = tagged(CONCEPTS["seed"], method).name  # ty: ignore[possibly-missing-attribute]
    out, _ = fit(method, codes=False, **SHORT.get(method, {}), **{seed: 7})
    assert out.metadata["seed"] == 7
    monkeypatch.setenv("ANDREY_SEED", "11")
    out, _ = fit(method, codes=False, **SHORT.get(method, {}))
    assert out.metadata["seed"] == 11


def test_an_unseeded_fit_records_no_seed():
    assert all("seed" not in coded_fit(m)[0].metadata for m in REGISTRY if m not in SEEDED)


def renamed(monkeypatch, method: str, name: str, **update) -> None:
    """Give ``method``'s parameter ``name`` the former names or values in ``update``."""
    spec = REGISTRY[method]
    params = tuple(p.model_copy(update=update) if p.name == name else p for p in spec.params)
    monkeypatch.setattr(
        method_wrapper, "spec_of", lambda _: spec.model_copy(update={"params": params})
    )


def test_a_former_name_maps_to_the_current_one_with_one_warning(monkeypatch):
    renamed(monkeypatch, "pc", "alpha", aliases=("significance",))
    data = inputs("pc", codes=False)
    expected = andrey.pc(*data, alpha=0.01)
    warning_policy._warned.clear()
    with pytest.warns(andrey.DeprecatedArgumentWarning, match="significance is now alpha"):
        out = andrey.pc(*data, significance=0.01)
    assert out.structure.to_numpy().tolist() == expected.structure.to_numpy().tolist()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        andrey.pc(*data, significance=0.01)  # the message warns once
    with pytest.raises(ValueError, match="pass only alpha"):
        andrey.pc(*data, alpha=0.01, significance=0.01)


def test_a_former_value_maps_to_the_current_one_with_one_warning(monkeypatch):
    renamed(monkeypatch, "pc", "indep_test", choice_aliases=(("fisher", "fisherz"),))
    data = inputs("pc", codes=False)
    warning_policy._warned.clear()
    with pytest.warns(andrey.DeprecatedArgumentWarning, match="indep_test='fisher' is now"):
        out = andrey.pc(*data, indep_test="fisher")
    assert out.structure == andrey.pc(*data, indep_test="fisherz").structure


def test_an_indep_test_takes_the_registered_tests(monkeypatch):
    monkeypatch.setitem(ci._CI_REGISTRY, "fisherz_copy", ci._CI_REGISTRY["fisherz"])
    data = inputs("pc", codes=False)
    out = andrey.pc(*data, indep_test="fisherz_copy")
    assert out.structure == andrey.pc(*data, indep_test="fisherz").structure


# --- surfaces -------------------------------------------------------------------------------------


def test_documented_method_lists_match_the_table():
    """A paragraph of docs/docs/api.md marked ``% concept: <key>`` names exactly the methods that
    take the concept today, under any name."""
    lines = API_MD.read_text(encoding="utf-8").splitlines()
    marked = [
        (i, line.split(":", 1)[1].strip())
        for i, line in enumerate(lines)
        if line.startswith("% concept:")
    ]
    assert marked
    for i, key in marked:
        paragraph = []
        for line in lines[i + 1 :]:
            if not line.strip():
                break
            paragraph.append(line)
        named = set(re.findall(r"`(\w+)`", " ".join(paragraph))) & set(REGISTRY)
        if key == "missing":
            takes = {m for m, spec in REGISTRY.items() if spec.data.missing}
        else:
            takes = {m for m in REGISTRY if tagged(CONCEPTS[key], m) is not None}
        assert named == takes, f"{key}: documented {sorted(named)}, table {sorted(takes)}"


def test_schema_carries_bounds_aliases_and_concepts():
    from andrey.spec import help_schema

    schema = help_schema()
    for name, spec in REGISTRY.items():
        properties = schema["methods"][name]["params"]["properties"]
        for p in spec.params:
            entry = properties[p.name]
            assert entry["concept"] == p.concept
            assert entry["aliases"] == list(p.aliases)
            assert entry["choice_aliases"] == dict(p.choice_aliases)
            low = "exclusiveMinimum" if p.exclusive_minimum else "minimum"
            assert entry.get(low) == p.minimum
            high = "exclusiveMaximum" if p.exclusive_maximum else "maximum"
            assert entry.get(high) == p.maximum
    for key, concept in schema["concepts"].items():
        assert concept["methods"] == {
            m: p.name for m in REGISTRY for p in REGISTRY[m].params if p.concept == key
        }
