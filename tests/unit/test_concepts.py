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
from andrey.spec.concepts import CI_SEARCHES, CONCEPTS, PCMCI_FAMILY, Concept
from tests.recovery.helpers import datagen
from tests.recovery.helpers.ci_tests import PValueOnly

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


@pytest.mark.parametrize("utility", UTILITIES)
def test_a_utility_takes_an_execution_setting_only_as_a_recorded_deviation(utility):
    concept = CONCEPTS["num_workers"]
    assert (concept.deviation(utility) is not None) == (concept.param.name in signature(utility))  # ty: ignore[possibly-missing-attribute]


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


def frame(seed: int, *, codes: bool, gaps: bool = False) -> pd.DataFrame:
    """A chain x -> y -> z; with ``codes``, x holds four whole numbers that look like categories,
    and with ``gaps``, a few values of x and y are missing."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(-1, 1, size=(N, 3)) ** 3
    x = rng.integers(0, 4, N).astype(float) if codes else u[:, 0]
    y = 0.8 * x + u[:, 1]
    data = pd.DataFrame({"x": x, "y": y, "z": 0.7 * y + u[:, 2]})
    if gaps:
        data.iloc[::13, 0] = np.nan
        data.iloc[5::11, 1] = np.nan
    return data


def inputs(method: str, *, codes: bool, gaps: bool = False) -> list:
    spec = REGISTRY[method]
    frames = [frame(seed, codes=codes, gaps=gaps) for seed in range(3)]
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
SHORT = {"calm": {"max_iter": 1, "subproblem_iter": 20}}
AT_CAP = {"calm": {"max_iter": 1, "subproblem_iter": 1}, "dagma": {"stages": 1, "max_iter": 1}}


def fit(method: str, *, codes: bool, gaps: bool = False, **kwargs):
    if method == "calm":
        pytest.importorskip("torch")
    warning_policy._warned.clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = getattr(andrey, method)(*inputs(method, codes=codes, gaps=gaps), **kwargs)
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


# --- the wrapper every method runs in ------------------------------------------------------------


def bad_values(param) -> list[tuple[object, type[Exception]]]:
    """Values ``param`` refuses, each with its error: another type raises ``TypeError``; each side
    of the bounds, a non-finite number, and an unknown choice raise ``ValueError``."""
    if param.choices:
        return [("unknown", ValueError), (1, TypeError)]
    values = {
        "int": [(1.5, TypeError), (True, TypeError)],
        "float": [("0.5", TypeError), (float("nan"), ValueError)],
        "bool": [(1, TypeError), ("yes", TypeError)],
    }.get(param.type, [([1.5, 1], TypeError)] if param.type == "int_pair" else [])
    step = 1 if param.type in ("int", "int_pair") else 0.5
    cast = int if param.type in ("int", "int_pair") else float
    if param.minimum is not None:
        low = cast(param.minimum if param.exclusive_minimum else param.minimum - step)
        values.append((low if param.type != "int_pair" else (low, 1), ValueError))
    if param.maximum is not None:
        high = cast(param.maximum if param.exclusive_maximum else param.maximum + step)
        values.append((high if param.type != "int_pair" else (high, 1), ValueError))
    return values


@pytest.mark.parametrize(
    ("method", "param", "value", "error"),
    [
        pytest.param(m, p, v, e, id=f"{m}-{p.name}-{v!r}")
        for m, spec in REGISTRY.items()
        for p in spec.params
        for v, e in bad_values(p)
    ],
)
def test_a_bad_value_raises_one_message_shape(method, param, value, error):
    """Every parameter of every method refuses a bad value with one message: the method, the
    parameter, what it accepts, and the value."""
    with pytest.raises(error, match=rf"^{method}: {param.name} must be .+, got "):
        getattr(andrey, method)(*inputs(method, codes=False), **{param.name: value})


SEEDED = [m for m, spec in REGISTRY.items() if spec.seeded]
#: The settings that make a CI search draw random numbers.
RANDOMIZED = {m: {"indep_test": "rcot"} for m in (*CI_SEARCHES, *PCMCI_FAMILY)}


@pytest.mark.parametrize("method", SEEDED)
def test_a_seeded_fit_records_its_seed(method, monkeypatch):
    seed = tagged(CONCEPTS["seed"], method).name  # ty: ignore[possibly-missing-attribute]
    settings = {**SHORT.get(method, {}), **RANDOMIZED.get(method, {})}
    out, _ = fit(method, codes=False, **settings, **{seed: 7})
    assert out.metadata["seed"] == 7
    monkeypatch.setenv("ANDREY_SEED", "11")
    out, _ = fit(method, codes=False, **settings)
    assert out.metadata["seed"] == 11


def test_a_fit_that_draws_no_random_numbers_records_no_seed():
    """A method without a seed, and a CI search under a deterministic test, record none."""
    assert all("seed" not in coded_fit(m)[0].metadata for m in REGISTRY if m not in SEEDED)
    for method in (*CI_SEARCHES, *PCMCI_FAMILY):
        out, _ = fit(method, codes=False, seed=7)
        assert "seed" not in out.metadata, method


# --- randomness -----------------------------------------------------------------------------------

STOCHASTIC = ci.STOCHASTIC_INDEP_TESTS


def takes_a_seed(test: str) -> bool:
    """Whether the shipped test builds with a ``seed`` setting."""
    codes = np.random.default_rng(0).integers(0, 3, (60, 3)).astype(float)
    args = (codes, None) if test in ci.MIXED_INDEP_TESTS else (codes,)
    try:
        ci._CI_REGISTRY[test](*args, seed=1)
    except TypeError:
        return False
    return True


def test_the_stochastic_tests_are_the_ones_that_take_a_seed():
    assert {t for t in ci.SHIPPED_INDEP_TESTS if takes_a_seed(t)} == set(STOCHASTIC)


def test_seeded_stochastic_and_suitable_for_the_seed_concept_agree():
    suitable = CONCEPTS["seed"].suitable
    for method, spec in REGISTRY.items():
        tests = next((p.choices for p in spec.params if p.name == "indep_test"), None) or ()
        assert spec.seeded == (method in suitable) == (spec.determinism == "stochastic"), method
        assert not set(tests) & set(STOCHASTIC) or spec.seeded, method
        names = [p.name for p in spec.params if p.concept == "seed"]
        assert names == (["seed"] if spec.seeded else []), method


@pytest.mark.filterwarnings("ignore::andrey.SampleSizeWarning")
def test_the_stochastic_tests_give_other_p_values_for_another_seed():
    data = inputs("pc", codes=False)[0].to_numpy()
    queries = [(0, 2, ()), (0, 1, ()), (0, 2, (1,)), (1, 2, (0,))]
    for name in STOCHASTIC:
        draws = {
            seed: [ci.make_indep_test(name, data, seed=seed)(*q) for q in queries]
            for seed in (7, 8)
        }
        assert draws[7] == [ci.make_indep_test(name, data, seed=7)(*q) for q in queries], name
        assert draws[7] != draws[8], name


#: Settings that make a CI search build each test it has: the one over the data, the one in rank
#: order, and the one that votes on colliders.
CI_CASES = {
    "pc": [{}, {"collider_rule": "conservative"}],
    "fci": [{}, {"possible_dsep": "fast", "collider_rule": "majority"}],
    "gfci": [{}, {"possible_dsep": "fast"}],
    "cdnod": [{}, {"collider_rule": "conservative"}],
    "iamb": [{}, {"target": 0}],
    "inter_iamb": [{}],
    "mmpc": [{}, {"target": 0}],
    "mmhc": [{}],
    "rsmax2": [{"restrict": "pc"}, {"restrict": "iamb"}],
    "pcmci": [{}],
    "pcmciplus": [{}],
    "lpcmci": [{}],
}


def test_every_search_with_a_ci_test_has_cases():
    assert set(CI_CASES) == {*CI_SEARCHES, *PCMCI_FAMILY}


@pytest.mark.parametrize(
    ("method", "settings", "gaps"),
    [
        pytest.param(m, s, g, id=f"{m}-{'+'.join(s) or 'default'}-{'gaps' if g else 'complete'}")
        for m, cases in CI_CASES.items()
        for s in cases
        for g in (False, True)
        if not g or REGISTRY[m].data.missing
    ],
)
def test_every_search_with_a_ci_test_accepts_a_registered_test(method, settings, gaps, monkeypatch):
    """A test registered with the single-query call alone runs every search that takes
    ``indep_test``, on complete data and, where the method takes missing values, on data with gaps,
    and gives the graph of Fisher's z-test it wraps."""
    expected, _ = fit(method, codes=False, gaps=gaps, **settings)
    monkeypatch.setitem(ci._CI_REGISTRY, "p_value_only", PValueOnly)
    out, _ = fit(method, codes=False, gaps=gaps, indep_test="p_value_only", **settings)
    assert out.structure == expected.structure


@pytest.fixture
def built_seeds(monkeypatch):
    """The seed each randomized test a fit builds receives; ``None`` is none passed."""
    seeds = []
    for name in STOCHASTIC:

        def build(data, *args, _factory=ci._CI_REGISTRY[name], **options):
            seeds.append(options.get("seed"))
            return _factory(data, *args, **options)

        monkeypatch.setitem(ci._CI_REGISTRY, name, build)
    return seeds


@pytest.mark.parametrize(
    ("method", "settings", "test"),
    [
        pytest.param(m, s, t, id=f"{m}-{t}-{'+'.join(s)}")
        for m, cases in CI_CASES.items()
        for s in cases
        for t in STOCHASTIC
    ],
)
def test_the_seed_reaches_every_randomized_test_a_search_builds(
    method, settings, test, built_seeds, monkeypatch
):
    monkeypatch.setenv("ANDREY_SEED", "11")
    out, _ = fit(method, codes=False, indep_test=test, seed=7, **settings)
    assert built_seeds
    assert set(built_seeds) == {7}
    assert out.metadata["seed"] == 7


@pytest.mark.parametrize("method", (*CI_SEARCHES, *PCMCI_FAMILY))
def test_the_global_seed_does_not_change_a_fit_that_passes_a_seed(method, monkeypatch):
    results = []
    for global_seed in ("0", "11"):
        monkeypatch.setenv("ANDREY_SEED", global_seed)
        out, _ = fit(method, codes=False, indep_test="rcot", seed=7)
        results.append((out.structure, out.metadata))
    assert results[0][0] == results[1][0]
    assert repr(results[0][1]) == repr(results[1][1])


def test_a_former_seed_name_maps_to_seed_for_one_release():
    data = inputs("ica_lingam", codes=False)
    expected = andrey.ica_lingam(*data, seed=3)
    warning_policy._warned.clear()
    with pytest.warns(andrey.DeprecatedArgumentWarning, match="random_state is now seed"):
        out = andrey.ica_lingam(*data, random_state=3)
    assert out.metadata["seed"] == 3 and out.ordering == expected.ordering


DROPPED = [(m, arg) for m, spec in REGISTRY.items() for arg in spec.dropped_args]


@pytest.mark.parametrize(("method", "arg"), DROPPED)
def test_a_dropped_argument_is_ignored_with_one_warning(method, arg):
    assert arg not in signature(method)
    assert all(p.name != arg for p in REGISTRY[method].params)
    data = inputs(method, codes=False)
    first = lambda out: out[0] if isinstance(out, list) else out  # noqa: E731
    expected = first(getattr(andrey, method)(*data))
    warning_policy._warned.clear()
    with pytest.warns(andrey.DeprecatedArgumentWarning, match=f"{arg} is ignored"):
        out = first(getattr(andrey, method)(*data, **{arg: 3}))
    assert out.ordering == expected.ordering
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        getattr(andrey, method)(*data, **{arg: 3})  # the message warns once


ALIASED = [
    pytest.param(m, a.name, old, id=f"{m}-{old}")
    for m, spec in REGISTRY.items()
    for a in (*spec.data.inputs, *spec.params)
    for old in a.aliases
]


@pytest.mark.parametrize(("method", "name", "old"), ALIASED)
def test_each_former_name_in_the_registry_maps_to_the_current_one_with_one_warning(
    method, name, old
):
    spec = REGISTRY[method]
    argument = next(a for a in (*spec.data.inputs, *spec.params) if a.name == name)
    as_input = argument in spec.data.inputs
    data = inputs(method, codes=False)
    settings = dict(SHORT.get(method, {}))
    value = data[0] if as_input else settings.pop(name, argument.default)

    def call(**named):
        out = getattr(andrey, method)(*(() if as_input else data), **settings, **named)
        return [o.structure for o in (out if isinstance(out, list) else [out])]

    expected = call(**{name: value})
    warning_policy._warned.clear()
    with pytest.warns(andrey.DeprecatedArgumentWarning, match=f"{old} is now {name}"):
        assert call(**{old: value}) == expected
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        call(**{old: value})  # the message warns once
    with pytest.raises(ValueError, match=f"pass only {name}"):
        call(**{old: value, name: value})


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
        assert schema["methods"][name]["dropped_args"] == list(spec.dropped_args)
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
