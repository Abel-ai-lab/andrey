"""Every method's docstring follows the docstring standard in ``DEVELOPMENT.md``.

The sentences shared by many methods match the table in ``scripts/docstring_text.py``, the sections
come in the standard's order, Warns names each of ``DataTypeWarning``, ``SearchLimitWarning``,
``ExperimentalWarning``, and ``BoundaryWarning`` a fit can raise, Notes ends with Limitations, and
an example whose fit draws random numbers passes ``seed=0``.
"""

from __future__ import annotations

import inspect
import re

import pytest

import andrey
import scripts.docstring_text as shared
from andrey.api._method import draws_random_numbers
from andrey.spec import REGISTRY
from andrey.spec.concepts import CONCEPTS
from tests.unit.test_concepts import docstring_section as section

#: The sections of a method's docstring, in order.
SECTIONS = ["Parameters", "Returns", "Raises", "Warns", "Notes", "References", "Examples"]


def docstring(method: str) -> str:
    return inspect.getdoc(getattr(andrey, method)) or ""


def test_every_shared_sentence_matches_the_table():
    assert shared.run(write=False) == []


def test_writing_restores_a_shared_sentence_that_drifted():
    source = (shared.API / "constraint.py").read_text()
    drifted = source.replace("is the configured seed", "is a configured seed")
    assert drifted != source
    assert shared.write_source(drifted) == (source, {})


@pytest.mark.parametrize("method", list(REGISTRY))
def test_sections_come_in_the_standard_order(method):
    titles = re.findall(r"^([A-Z][A-Za-z ]+)\n-{3,}$", docstring(method), flags=re.MULTILINE)
    assert titles == SECTIONS


@pytest.mark.parametrize("method", list(REGISTRY))
def test_notes_say_how_missing_values_are_read_and_end_with_limitations(method):
    paragraphs = section(docstring(method), "Notes").strip().split("\n\n")
    assert any(p.startswith("**Missing values.**") for p in paragraphs)
    assert paragraphs[-1].startswith("**Limitations.**")


@pytest.mark.parametrize("method", list(REGISTRY))
def test_warns_lists_the_data_cap_experimental_and_boundary_warnings(method):
    listed = set(re.findall(r"^(\w+Warning)$", section(docstring(method), "Warns"), re.MULTILINE))
    raised = {
        "DataTypeWarning": method in CONCEPTS["category_codes"].suitable,
        "SearchLimitWarning": method in CONCEPTS["cap_warning"].suitable,
        "ExperimentalWarning": REGISTRY[method].status == "experimental",
    }
    assert {name for name, raises in raised.items() if raises} <= listed
    assert ("BoundaryWarning" in listed) == shared.can_pass_boundary(method)


def draws_by_default(method: str) -> bool:
    spec = REGISTRY[method]
    parameters = inspect.signature(getattr(andrey, method)).parameters.values()
    defaults = {p.name: p.default for p in parameters if p.default is not inspect.Parameter.empty}
    return spec.seeded and draws_random_numbers(spec, defaults)


@pytest.mark.parametrize("method", [m for m in REGISTRY if draws_by_default(m)])
def test_an_example_whose_fit_draws_random_numbers_passes_seed_0(method):
    assert "seed=0" in section(docstring(method), "Examples")
