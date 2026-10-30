"""``andrey markov-check``: the same check as ``andrey.markov_check``, from a result file."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import andrey
from andrey.cli import _apply_output_cap, main, render_markov_check_human
from andrey.spec import SCHEMA_VERSION, help_schema, render_skill
from andrey.spec.concepts import MARKOV_CHECK_PARAMS


def _run(argv, capsys):
    code = main(argv)
    out, err = capsys.readouterr()
    return code, out, err


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    a = rng.standard_normal(600)
    b = 0.9 * a + 0.5 * rng.standard_normal(600)
    c = 0.9 * b + 0.5 * rng.standard_normal(600)
    d = 0.9 * a + 0.5 * rng.standard_normal(600)
    return pd.DataFrame({"a": a, "b": b, "c": c, "d": d})


@pytest.fixture
def csv_path(tmp_path, data):
    path = tmp_path / "data.csv"
    data.to_csv(path, index=False)
    return path


@pytest.fixture
def graph_path(tmp_path, csv_path, capsys):
    path = tmp_path / "pc.json"
    assert main(["run", "pc", "--data", str(csv_path), "-o", str(path)]) == 0
    capsys.readouterr()
    return path


def test_the_envelope_holds_the_api_check_of_the_graph_in_the_file(
    graph_path, csv_path, data, capsys
):
    code, out, _ = _run(
        ["markov-check", str(graph_path), str(csv_path), "--seed", "4", "--json"], capsys
    )
    env = json.loads(out)
    assert code == 0 and env["schema_version"] == SCHEMA_VERSION
    check = andrey.markov_check(andrey.pc(data), data, seed=4)
    result = env["markov_check"]
    assert result["passed"] == check.passed and result["verdict"] == check.verdict
    assert result["n_tests"] == check.n_tests == len(result["tests"])
    assert result["uniformity_p_value"] == pytest.approx(check.uniformity_p_value)
    assert [t["p_value"] for t in result["tests"]] == pytest.approx(check.p_values.tolist())
    assert result["labels"] == ["a", "b", "c", "d"] and result["seed"] == 4
    assert env["run"]["n_samples"] == len(data) and env["run"]["seed"] == 4
    assert {"x", "y", "given", "p_value"} == set(result["tests"][0])


def test_a_saved_result_is_read_like_the_run_output(tmp_path, graph_path, csv_path, data, capsys):
    saved = tmp_path / "saved.json"
    andrey.pc(data).save(saved)
    argv = ["--seed", "2", "--json"]
    _, from_run, _ = _run(["markov-check", str(graph_path), str(csv_path), *argv], capsys)
    _, from_save, _ = _run(["markov-check", str(saved), str(csv_path), *argv], capsys)
    assert json.loads(from_run)["markov_check"] == json.loads(from_save)["markov_check"]


def test_a_graph_or_option_that_cannot_be_checked_is_a_usage_error(
    tmp_path, graph_path, csv_path, capsys
):
    pag = tmp_path / "fci.json"
    assert main(["run", "fci", "--data", str(csv_path), "-o", str(pag)]) == 0
    capsys.readouterr()
    for argv, code in (
        ([str(pag), str(csv_path)], "unsupported_value"),
        ([str(graph_path), str(csv_path), "--alpha", "2"], "bad_value"),
        ([str(graph_path), str(csv_path), "--sample-fraction", "0"], "bad_value"),
        ([str(graph_path), str(csv_path), "--categorical", "zz"], "unknown_column"),
        ([str(tmp_path / "missing.json"), str(csv_path)], "input_not_found"),
    ):
        status, _, err = _run(["markov-check", *argv], capsys)
        assert status == 2 and json.loads(err)["error"]["code"] == code


def test_a_capped_table_keeps_the_smallest_p_values_and_says_so():
    tests = [
        {"x": k, "y": 0, "given": [], "p_value": p} for k, p in enumerate([0.9, 0.01, 0.5, 0.2])
    ]
    env = {"markov_check": {"tests": tests}}
    _apply_output_cap(env, 2)
    assert [t["p_value"] for t in env["markov_check"]["tests"]] == [0.01, 0.2]
    assert env["markov_check"]["tests_truncated"] and "capped" in env["note"]


def test_the_terminal_summary_gives_the_verdict_and_the_smallest_p_values(data):
    check = andrey.markov_check(andrey.pc(data), data, seed=0)
    text = render_markov_check_human(
        check,
        {
            "warnings": [],
            "n_samples": 1,
            "n_variables": 1,
            "elapsed_s": 0,
            "versions": {"andrey": "x"},
        },
    )
    assert check.verdict in text
    smallest = min(check.table(), key=lambda row: row["p_value"])
    assert f"{smallest['p_value']:.3g}" in text


def test_the_schema_and_the_agent_guide_describe_the_command():
    schema = help_schema()["cli"]["markov_check"]
    assert set(schema["parameters"]) == set(MARKOV_CHECK_PARAMS)
    assert "andrey markov-check" in render_skill()
