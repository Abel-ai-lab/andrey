"""An experimental method warns once per process; a supported method never does."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import andrey
from andrey.api._experimental import warn_experimental
from andrey.spec import REGISTRY


@pytest.fixture(scope="module")
def chain():
    """x -> y -> z with uniform noise, which suits every supported method."""
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, 1000)
    y = 0.8 * x + rng.uniform(-1, 1, 1000)
    z = 0.8 * y + rng.uniform(-1, 1, 1000)
    return np.column_stack([x, y, z])


def experimental(caught: list[warnings.WarningMessage]) -> list[warnings.WarningMessage]:
    return [w for w in caught if w.category is andrey.ExperimentalWarning]


# An experimental facade calls ``warn_experimental`` first (DEVELOPMENT.md, Add a method), so these
# tests make that call for a method name of their own.


def test_warns_once_on_the_first_call():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_experimental("new_method")
        warn_experimental("new_method")
    [warning] = experimental(caught)
    assert str(warning.message).startswith("andrey.new_method is experimental")
    assert "may change without deprecation" in str(warning.message)
    assert warning.filename == __file__  # the caller's line, not the package's


def test_each_method_warns_on_its_own_first_call():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_experimental("first_method")
        warn_experimental("second_method")
        warn_experimental("first_method")
    assert [str(w.message).split()[0] for w in experimental(caught)] == [
        "andrey.first_method",
        "andrey.second_method",
    ]


def test_an_error_filter_fails_every_call():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for _ in range(2):  # a call the filter stopped does not use up the warning
            with pytest.raises(andrey.ExperimentalWarning, match="andrey.new_method"):
                warn_experimental("new_method")


def test_is_a_user_warning():
    assert issubclass(andrey.ExperimentalWarning, UserWarning)


#: Settings that shorten a slow fit; the facade path stays the same.
SHORT = {"calm": {"max_iter": 1, "subproblem_iter": 20}}


@pytest.mark.parametrize("name", sorted(n for n, s in REGISTRY.items() if s.status == "supported"))
def test_supported_methods_do_not_warn_experimental(name, chain):
    """Only ``ExperimentalWarning`` fails here: a numerical warning from a fit is not a status. A
    pairwise direction test takes the pair ``x -> y``, a panel method three time points or groups,
    and a required context, such as CD-NOD's domain index, two domains of equal size."""
    spec = REGISTRY[name]
    if spec.accel == "torch":
        pytest.importorskip("torch")
    data = chain[:200, :2] if spec.family == "pairwise" else chain
    if spec.data.inputs[0].kind == "panel":
        data = [chain[:300], chain[300:600], chain[600:900]]
    contexts = [
        np.repeat([0.0, 1.0], len(chain) // 2)
        for i in spec.data.inputs[1:]
        if i.kind == "context" and i.required
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error", andrey.ExperimentalWarning)
        getattr(andrey, name)(data, *contexts, **SHORT.get(name, {}))
