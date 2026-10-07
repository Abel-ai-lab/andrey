"""An experimental method warns once per process; a supported method never does."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import andrey
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


def test_warns_once_on_the_first_call(chain):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        andrey.hc(chain)
        andrey.hc(chain)
    [warning] = experimental(caught)
    assert str(warning.message).startswith("andrey.hc is experimental")
    assert "may change without deprecation" in str(warning.message)
    assert warning.filename == __file__  # the caller's line, not the package's


def test_each_method_warns_on_its_own_first_call(chain):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        andrey.hc(chain)
        andrey.gies(chain)
        andrey.hc(chain)
    assert [str(w.message).split()[0] for w in experimental(caught)] == ["andrey.hc", "andrey.gies"]


def test_an_error_filter_fails_every_call(chain):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for _ in range(2):  # a call the filter stopped does not use up the warning
            with pytest.raises(andrey.ExperimentalWarning, match="andrey.hc"):
                andrey.hc(chain)


def test_is_a_user_warning():
    assert issubclass(andrey.ExperimentalWarning, UserWarning)


@pytest.mark.parametrize("name", sorted(n for n, s in REGISTRY.items() if s.status == "supported"))
def test_supported_methods_do_not_warn_experimental(name, chain):
    """Only ``ExperimentalWarning`` fails here: a numerical warning from a fit is not a status."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", andrey.ExperimentalWarning)
        getattr(andrey, name)(chain)
