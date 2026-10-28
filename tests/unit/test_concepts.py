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
from andrey.core import warning_policy
from andrey.core.knowledge import KNOWLEDGE_PARTS
from andrey.spec import REGISTRY
from andrey.spec.concepts import CONCEPTS, Concept
from tests.recovery.helpers import datagen

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


#: Each method's data for the units check: the first ``UNIT_ROWS`` rows of a case of
#: ``tests/recovery/helpers/datagen`` whose model the method fits.
UNIT_CASES = {
    **dict.fromkeys(
        (
            *("pc", "fci", "gfci", "cdnod", "iamb", "inter_iamb", "mmpc", "mmhc", "rsmax2"),
            *("ges", "gies", "xges", "hc", "tabu", "exact_search", "boss", "grasp"),
        ),
        "gauss_5v",
    ),
    "direct_lingam": "lingam_5v_uniform",
    "multi_group_direct_lingam": "mgdl_2groups",
    "varma_lingam": "varma_3v",
    "longitudinal_lingam": "panel_4v",
    "pcmci": "var_3v_lag2",
    "pcmciplus": "var_4v_stable",
    "lpcmci": "var_6v_hidden",
    "granger": "var_4v_stable",
    "gin": "measurement_chain_9v",
    "rcd": "confounded_fork_4v",
    "bottom_up_parce_lingam": "confounded_fork_4v",
    "anm": "anm_pair_cubic",
    "pnl": "anm_pair_cubic",
}
UNIT_ROWS = 1000
#: Two draws of units: variable ``j`` of every case is multiplied by ``2**k[j]``, an exact product.
UNIT_EXPONENTS = [np.random.default_rng(seed).integers(-20, 21, size=16) for seed in (9, 22)]


def unit_inputs(method: str, exponents: np.ndarray) -> list:
    """The method's case with variable ``j`` multiplied by ``2**exponents[j]``; a domain index
    stays as it is."""
    case = UNIT_CASES[method]
    factors = np.ldexp(1.0, exponents)
    args = []
    for inp in REGISTRY[method].data.inputs:
        if not inp.required:
            continue
        if inp.kind == "panel":
            groups = datagen.generate_groups(case)
            args.append([x[:UNIT_ROWS] * factors[: x.shape[1]] for x in groups])
        elif inp.kind == "matrix":
            x = datagen.generate(case)[:UNIT_ROWS]
            args.append(x * factors[: x.shape[1]])
        else:
            args.append(datagen.domain_index(case)[:UNIT_ROWS])
    return args


def graphs_and_orders(out) -> list:
    """The endpoint marks of every graph (each lag, occasion, and group) and every causal order."""
    if isinstance(out, list):
        return [graphs_and_orders(o) for o in out]
    s = out.structure
    if isinstance(s, andrey.TemporalStructure):
        graphs = [s.at(t, k) for t in s.times for k in s.lags] if s.n_times else map(s.lag, s.lags)
    else:
        graphs = [s]
    orders = {k: out.metadata.get(k) for k in ("causal_order", "causal_orders", "unordered")}
    return [[g.to_numpy().tolist() for g in graphs], out.ordering, orders]


def unit_free(method: str) -> bool:
    """Whether every graph and order stays the same with each variable in other units."""
    fits = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for exponents in [np.zeros(16, dtype=int), *UNIT_EXPONENTS]:
            out = getattr(andrey, method)(*unit_inputs(method, exponents))
            fits.append(graphs_and_orders(out))
    return all(fit == fits[0] for fit in fits[1:])


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
    "units": (unit_free, True),
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
