"""The notebook check compares stored outputs with a fresh run, except in cells tagged as CPU-bound.

The script needs the docs group (nbformat, nbclient, a Jupyter kernel); the docs workflow runs this
file there, and the other jobs skip it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
pytest.importorskip("nbclient")

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location(
        "execute_notebooks", ROOT / "scripts" / "execute_notebooks.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cell(printed: str, tags: tuple[str, ...] = ()):
    """A code cell whose stored output is ``printed``."""
    output = nbformat.v4.new_output("stream", name="stdout", text=printed)
    code = nbformat.v4.new_code_cell(f"print({printed!r})", outputs=[output])
    code.metadata["tags"] = list(tags)
    return code


def streamed(*chunks: tuple[str, str]):
    """A code cell whose stored outputs are the ``(stream name, text)`` ``chunks``, in order."""
    outputs = [nbformat.v4.new_output("stream", name=n, text=t) for n, t in chunks]
    return nbformat.v4.new_code_cell("print('streamed')", outputs=outputs)


def notebook(*cells):
    return nbformat.v4.new_notebook(cells=list(cells))


def test_an_untagged_cell_must_print_what_is_stored(script):
    stored = notebook(cell("11 of 12"))
    assert script.differences(stored, notebook(cell("11 of 12"))) == []
    assert script.differences(stored, notebook(cell("10 of 12")))


def test_a_stream_split_into_chunks_equals_the_same_text_whole(script):
    stored = notebook(streamed(("stdout", "errors: 11\n"), ("stdout", "done\n")))
    split = notebook(
        streamed(("stdout", "errors:"), ("stdout", " 11"), ("stdout", "\n"), ("stdout", "done\n"))
    )
    assert script.differences(stored, split) == []
    assert script.differences(split, stored) == []


def test_a_stream_split_into_chunks_still_fails_when_its_text_changes(script):
    stored = notebook(streamed(("stdout", "errors: 11\n")))
    split = notebook(streamed(("stdout", "errors:"), ("stdout", " 12\n")))
    assert script.differences(stored, split)


def test_chunks_join_only_within_one_stream_and_in_order(script):
    out_err = notebook(streamed(("stdout", "a\n"), ("stderr", "b\n")))
    assert (
        script.differences(out_err, notebook(streamed(("stdout", "a\n"), ("stderr", "b\n")))) == []
    )
    assert script.differences(out_err, notebook(streamed(("stderr", "a\n"), ("stderr", "b\n"))))
    assert script.differences(out_err, notebook(streamed(("stderr", "b\n"), ("stdout", "a\n"))))
    assert script.differences(out_err, notebook(streamed(("stdout", "a\nb\n"))))


def test_an_extra_empty_stdout_line_is_equal(script):
    stored = notebook(streamed(("stdout", "errors: 11\n")))
    trailing = notebook(streamed(("stdout", "errors: 11\n"), ("stdout", "\n")))
    assert script.differences(stored, trailing) == []
    assert script.differences(stored, notebook(cell("errors: 11"))) == []

    warned = notebook(streamed(("stdout", "a\n"), ("stderr", "warn\n")))
    after_warning = notebook(streamed(("stdout", "a\n"), ("stderr", "warn\n"), ("stdout", "\n")))
    assert script.differences(warned, after_warning) == []
    assert script.differences(after_warning, warned) == []


def test_an_empty_line_between_texts_is_a_difference(script):
    stored = notebook(streamed(("stdout", "a\nb\n")))
    assert script.differences(stored, notebook(streamed(("stdout", "a\n\nb\n"))))


def test_a_changed_number_fails_in_a_stream_of_chunks(script):
    stored = notebook(streamed(("stdout", "errors:"), ("stdout", " 11\n")))
    assert script.differences(stored, notebook(streamed(("stdout", "errors: 10\n"))))
    assert script.differences(stored, notebook(streamed(("stdout", "errors:"), ("stdout", " 1\n"))))


def test_a_tagged_cell_may_print_something_else(script):
    tag = (script.CPU_TAG,)
    stored, fresh = notebook(cell("11 of 12", tag)), notebook(cell("10 of 12", tag))
    assert script.differences(stored, fresh) == []


def test_a_tagged_cell_hides_no_change_beside_it(script):
    tag = (script.CPU_TAG,)
    stored = notebook(cell("11 of 12", tag), cell("3 edges"))
    fresh = notebook(cell("10 of 12", tag), cell("4 edges"))
    assert script.differences(stored, fresh)


def test_a_tagged_cell_still_runs_and_still_fails_when_it_raises(script, tmp_path):
    tag = [script.CPU_TAG]
    ran = nbformat.v4.new_code_cell("print('ran')")
    ran.metadata["tags"] = tag
    path = tmp_path / "ran.ipynb"
    nbformat.write(notebook(ran), path)
    assert script.execute(path).cells[0].outputs[0].text == "ran\n"

    failing = nbformat.v4.new_code_cell("raise RuntimeError('broken')")
    failing.metadata["tags"] = tag
    path = tmp_path / "failing.ipynb"
    nbformat.write(notebook(failing), path)
    with pytest.raises(Exception, match="broken"):
        script.execute(path)


def test_the_contributor_guide_names_the_tag(script):
    assert script.CPU_TAG in (ROOT / "docs" / "DEVELOPMENT.md").read_text()
