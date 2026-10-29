"""``andrey bootstrap``: the result envelope, its flags, and what the agent guide says about it."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import andrey
from andrey.cli import main, render_method_help
from andrey.spec import REGISTRY, SCHEMA_VERSION, help_schema, render_skill


def _run(argv, capsys):
    code = main(argv)
    out, err = capsys.readouterr()
    return code, out, err


def chain(n=200, seed=0, columns=("a", "b", "c", "d")):
    rng = np.random.default_rng(seed)
    a = rng.uniform(-1, 1, n)
    b = 0.9 * a + 0.4 * rng.uniform(-1, 1, n)
    c = 0.9 * b + 0.4 * rng.uniform(-1, 1, n)
    return pd.DataFrame(dict(zip(columns, (a, b, c, rng.uniform(-1, 1, n)))))


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "chain.csv"
    chain().to_csv(path, index=False)
    return path


def test_the_envelope_is_the_runs_with_the_averaged_graph_and_the_frequencies(csv_path, capsys):
    argv = ["bootstrap", "pc", "--data", str(csv_path), "--n-boot", "6", "--seed", "3", "--json"]
    code, out, _ = _run([*argv, "alpha=0.01"], capsys)
    env = json.loads(out)
    assert code == 0
    assert env["schema_version"] == SCHEMA_VERSION
    assert (env["method"], env["status"]) == ("pc", "supported")
    assert env["bootstrap"] == {
        "n_boot": 6,
        "threshold": 0.5,
        "seed": 3,
        "resampling": "subsample",
        "sample_fraction": 0.5,
        "block_length": None,
    }
    assert env["structure"]["labels"] == ["a", "b", "c", "d"]
    assert env["structure"]["kind"] == "cpdag" and env["structure"]["n_edges"] == 2
    assert env["run"]["params"]["alpha"] == 0.01 and "seed" not in env["run"]["params"]
    api = andrey.bootstrap("pc", pd.read_csv(csv_path), n_boot=6, seed=3, alpha=0.01)
    assert env["frequencies"] == api.table(index=True)
    assert {"source", "target", "type", "count", "frequency"} == set(env["frequencies"][0])
    freqs = [row["frequency"] for row in env["frequencies"]]
    assert freqs == sorted(freqs, reverse=True)
    # a repeat with the same seed gives the same frequencies and the same graph
    _, again, _ = _run([*argv, "alpha=0.01"], capsys)
    assert json.loads(again)["frequencies"] == env["frequencies"]
    assert json.loads(again)["structure"] == env["structure"]


def test_the_threshold_flag_sets_the_averaged_graph(csv_path, capsys):
    base = ["bootstrap", "hc", "--data", str(csv_path), "--n-boot", "5", "--seed", "1", "--json"]
    _, loose, _ = _run([*base, "--threshold", "0.2"], capsys)
    _, strict, _ = _run([*base, "--threshold", "1"], capsys)
    loose, strict = json.loads(loose), json.loads(strict)
    assert loose["bootstrap"]["threshold"] == 0.2 and strict["bootstrap"]["threshold"] == 1.0
    assert strict["structure"]["n_edges"] <= loose["structure"]["n_edges"]
    assert loose["frequencies"] == strict["frequencies"]


def test_the_resampling_flags_reach_the_fits_and_the_envelope(csv_path, capsys):
    base = ["bootstrap", "hc", "--data", str(csv_path), "--n-boot", "5", "--seed", "1", "--json"]
    data = pd.read_csv(csv_path)
    for flags, kwargs in (
        ([], {}),
        (["--sample-fraction", "0.8"], {"sample_fraction": 0.8}),
        (["--resampling", "bootstrap"], {"resampling": "bootstrap"}),
    ):
        code, out, _ = _run([*base, *flags], capsys)
        env = json.loads(out)
        api = andrey.bootstrap("hc", data, n_boot=5, seed=1, **kwargs)
        assert code == 0 and env["frequencies"] == api.table(index=True)
        assert env["bootstrap"]["resampling"] == api.metadata["resampling"]
        assert env["bootstrap"]["sample_fraction"] == api.metadata["sample_fraction"]


def test_the_bootstrap_seeds_a_seeded_method_and_rejects_its_own_seed(csv_path, capsys):
    base = ["bootstrap", "boss", "--data", str(csv_path), "--n-boot", "3", "--json"]
    code, _out, err = _run([*base, "seed=2"], capsys)
    assert code == 2 and json.loads(err)["error"]["code"] == "bad_value"
    code, out, _ = _run([*base, "--seed", "2"], capsys)
    assert code == 0 and json.loads(out)["bootstrap"]["seed"] == 2


@pytest.mark.parametrize(
    "flags",
    [
        ["--n-boot", "0"],
        ["--threshold", "0"],
        ["--threshold", "1.5"],
        ["--seed", "-1"],
        ["--sample-fraction", "0"],
        ["--sample-fraction", "1.5"],
        ["--resampling", "jackknife"],
    ],
)
def test_a_flag_out_of_range_is_a_usage_error(csv_path, capsys, flags):
    code, _out, err = _run(["bootstrap", "pc", "--data", str(csv_path), *flags], capsys)
    assert code == 2 and json.loads(err)["error"]["code"] == "bad_value"


def test_a_method_that_cannot_be_bootstrapped_is_an_unsupported_value(csv_path, capsys):
    code, _out, err = _run(["bootstrap", "gin", "--data", str(csv_path)], capsys)
    error = json.loads(err)["error"]
    assert code == 2 and error["code"] == "unsupported_value" and "gin" in error["message"]
    code, _out, err = _run(
        ["bootstrap", "pc", "--data", str(csv_path), "--block-length", "5"], capsys
    )
    error = json.loads(err)["error"]
    assert code == 2 and error["code"] == "bad_value" and "--block-length" in error["message"]


def test_it_resolves_the_method_like_run(capsys):
    code, _out, err = _run(["bootstrap", "nope"], capsys)
    assert code == 2 and json.loads(err)["error"]["code"] == "unknown_method"
    code, _out, err = _run(["bootstrap"], capsys)
    assert code == 2 and json.loads(err)["error"]["code"] == "missing_method"


def test_a_time_series_names_the_lag_of_each_frequency(tmp_path, capsys):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((300, 2))
    for t in range(1, 300):
        x[t, 1] += 0.9 * x[t - 1, 0]
    path = tmp_path / "series.csv"
    pd.DataFrame(x, columns=["u", "v"]).to_csv(path, index=False)
    argv = ["bootstrap", "pcmci", "--data", str(path), "--n-boot", "3", "--json"]
    code, out, _ = _run(argv, capsys)
    env = json.loads(out)
    assert code == 0 and env["structure"]["type"] == "temporal"
    assert (
        env["bootstrap"]["resampling"] == "subsample" and env["bootstrap"]["block_length"] is None
    )
    assert env["frequencies"][0]["lag"] == 1 and env["frequencies"][0]["frequency"] == 1.0
    blocks = ["--resampling", "bootstrap", "--block-length", "30"]
    code, out, _ = _run([*argv, *blocks], capsys)
    drawn = json.loads(out)["bootstrap"]
    assert code == 0 and drawn["resampling"] == "bootstrap" and drawn["block_length"] == 30
    code, _out, err = _run([*argv, "--block-length", "30"], capsys)
    assert code == 2 and "--block-length" in json.loads(err)["error"]["message"]


def test_a_method_with_a_result_per_group_gives_a_result_per_group(tmp_path, capsys):
    paths = []
    for k in range(2):
        paths.append(tmp_path / f"g{k}.csv")
        chain(150, k).to_csv(paths[-1], index=False)
    argv = ["bootstrap", "multi_group_direct_lingam", "--n-boot", "2", "--json"]
    for path in paths:
        argv += ["--data", str(path)]
    code, out, _ = _run(argv, capsys)
    env = json.loads(out)
    assert code == 0 and env["multi"] is True and len(env["results"]) == 2
    assert all({"structure", "frequencies"} <= set(r) for r in env["results"])


def test_a_context_input_is_read_from_its_flag(tmp_path, capsys):
    data, domain = tmp_path / "d.csv", tmp_path / "domain.csv"
    chain(150).to_csv(data, index=False)
    pd.DataFrame({"domain": np.repeat([0, 1, 2], 50)}).to_csv(domain, index=False)
    argv = ["bootstrap", "cdnod", "--data", str(data), "--domain-index", str(domain)]
    code, out, _ = _run([*argv, "--n-boot", "2", "--json"], capsys)
    assert code == 0 and json.loads(out)["structure"]["labels"] == ["a", "b", "c", "d"]


def test_a_long_table_is_capped_on_stdout_and_whole_in_a_file(
    csv_path, tmp_path, capsys, monkeypatch
):
    monkeypatch.setattr("andrey.cli.EDGE_LIMIT", 1)
    argv = ["bootstrap", "pc", "--data", str(csv_path), "--n-boot", "6", "--seed", "3", "--json"]
    _, out, _ = _run(argv, capsys)
    env = json.loads(out)
    assert len(env["frequencies"]) == 1 and env["frequencies_truncated"] is True
    assert env["structure"]["edges_truncated"] and "capped" in env["note"]
    _, full, _ = _run([*argv, "--full"], capsys)
    assert len(json.loads(full)["frequencies"]) > 1
    dest = tmp_path / "result.json"
    code, out, _ = _run([*argv, "-o", str(dest)], capsys)
    assert code == 0 and out.strip() == ""
    saved = json.loads(dest.read_text())
    assert saved["frequencies"] == json.loads(full)["frequencies"]
    assert "frequencies_truncated" not in saved


def test_a_terminal_gets_the_averaged_graph_and_its_most_frequent_edges(
    csv_path, capsys, monkeypatch
):
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    argv = ["bootstrap", "pc", "--data", str(csv_path), "--n-boot", "4", "--seed", "1"]
    code, out, _ = _run(argv, capsys)
    assert code == 0 and out.startswith("pc  |  4 fits  |  averaged graph at threshold 0.5")
    assert "most frequent edges:" in out and "a -- b  1.00" in out and "200 samples" in out


def test_the_help_and_the_contract_describe_the_command(capsys):
    code, out, _ = _run(["bootstrap", "pc", "--help"], capsys)
    assert code == 0 and out.startswith("andrey bootstrap pc -- ")
    assert "--n-boot" in out and "andrey bootstrap pc --data data.csv" in out
    assert out == render_method_help(REGISTRY["pc"], command="bootstrap") + "\n"
    code, out, _ = _run(["bootstrap", "pc", "--help", "--json"], capsys)
    assert json.loads(out)["method"]["name"] == "pc"
    cli = help_schema()["cli"]["bootstrap"]
    assert cli["invocation"].startswith("andrey bootstrap <method> --data")
    assert "frequencies" in cli["envelope"] and "gin" in cli["what"]
    for flag in ("--resampling", "--sample-fraction"):
        assert flag in cli["invocation"] and flag in out
    params = cli["parameters"]
    assert params["resampling"]["enum"][:2] == ["subsample", "bootstrap"]
    assert params["sample_fraction"]["exclusiveMinimum"] == 0
    assert params["sample_fraction"]["maximum"] == 1 and params["sample_fraction"]["default"] == 0.5


def test_the_agent_guide_explains_when_to_bootstrap_and_how_to_read_it():
    guide = render_skill()
    assert "## How stable an edge is" in guide
    assert "andrey bootstrap <method> --data" in guide and "`frequencies`" in guide
