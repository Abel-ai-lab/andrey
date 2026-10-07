"""End-to-end CLI behavior: introspection, the result envelope, and the error contract."""

from __future__ import annotations

import io
import json
import warnings

import numpy as np
import pytest

from andrey.cli import graph_view, main
from andrey.spec import REGISTRY, SCHEMA_VERSION


@pytest.fixture
def csv_path(tmp_path):
    """A small linear chain a -> b -> c (plus an isolated d)."""
    rng = np.random.default_rng(0)
    n = 300
    a = rng.standard_normal(n)
    b = 0.8 * a + 0.3 * rng.standard_normal(n)
    c = 0.6 * b + 0.3 * rng.standard_normal(n)
    d = rng.standard_normal(n)
    path = tmp_path / "data.csv"
    np.savetxt(path, np.column_stack([a, b, c, d]), delimiter=",")
    return path


@pytest.fixture
def nongaussian_path(tmp_path):
    rng = np.random.default_rng(1)
    n = 300
    u = rng.uniform(-1, 1, size=(n, 3)) ** 3
    x0 = u[:, 0]
    x1 = 1.2 * x0 + u[:, 1]
    x2 = 0.7 * x1 + u[:, 2]
    path = tmp_path / "ng.csv"
    np.savetxt(path, np.column_stack([x0, x1, x2]), delimiter=",")
    return path


def _run(argv, capsys):
    code = main(argv)
    out, err = capsys.readouterr()
    return code, out, err


# --- introspection --------------------------------------------------------------------------------


def test_list_is_json_when_piped(capsys, monkeypatch):
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    code, out, _ = _run(["list"], capsys)
    rows = json.loads(out)
    assert code == 0
    assert {r["name"]: r["status"] for r in rows} == {n: s.status for n, s in REGISTRY.items()}


def test_list_on_a_terminal_names_its_columns(capsys, monkeypatch):
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    _, out, _ = _run(["list"], capsys)
    header, *rows = out.splitlines()
    assert header.split() == ["method", "family", "status", "summary"]
    assert len(rows) == len(REGISTRY)


def test_root_skill_and_llms(capsys):
    _, skill, _ = _run(["--skill"], capsys)
    assert "andrey run <method> --help --json" in skill and "andrey config" in skill
    _, llms, _ = _run(["--llms"], capsys)
    assert "ANDREY_BACKEND" in llms


def test_bare_invocation_prints_help(capsys):
    code, out, _ = _run([], capsys)
    assert code == 0
    assert "usage" in out.lower()


# --- run: the result envelope ---------------------------------------------------------------------


def test_run_pc_envelope(capsys, csv_path):
    code, out, _ = _run(["run", "pc", "--data", str(csv_path)], capsys)
    env = json.loads(out)
    assert code == 0
    assert env["schema_version"] == SCHEMA_VERSION and env["method"] == "pc"
    assert env["structure"]["kind"] == "cpdag"
    assert env["structure"]["n_nodes"] == 4
    assert env["structure"]["labels"] is None  # no header row
    run = env["run"]
    assert run["n_samples"] == 300 and run["n_variables"] == 4
    assert run["params"]["alpha"] == 0.05 and run["params"]["indep_test"] == "fisherz"  # defaults
    assert set(run["versions"]) == {"andrey", "numpy", "python"}


def test_run_direct_lingam_weights_and_ordering(capsys, nongaussian_path):
    code, out, _ = _run(["run", "direct_lingam", "--data", str(nongaussian_path)], capsys)
    env = json.loads(out)
    assert code == 0
    assert env["structure"]["ordering"] is not None
    directed = [e for e in env["structure"]["edges"] if e["type"] == "directed"]
    assert directed and all("weight" in e for e in directed)


def test_run_stdin(capsys, monkeypatch, csv_path):
    monkeypatch.setattr("sys.stdin", io.StringIO(csv_path.read_text()))
    code, out, _ = _run(["run", "ges", "--data", "-"], capsys)
    env = json.loads(out)
    assert code == 0 and env["method"] == "ges"


def test_run_writes_output_file(capsys, tmp_path, csv_path):
    dest = tmp_path / "result.json"
    code, out, _ = _run(["run", "pc", "--data", str(csv_path), "-o", str(dest)], capsys)
    assert code == 0 and out.strip() == ""  # stdout stays clean when -o is given
    assert json.loads(dest.read_text())["method"] == "pc"


def test_run_multi_group(capsys, nongaussian_path):
    code, out, _ = _run(
        [
            "run",
            "multi_group_direct_lingam",
            "--data-groups",
            str(nongaussian_path),
            "--data-groups",
            str(nongaussian_path),
        ],
        capsys,
    )
    env = json.loads(out)
    assert code == 0 and env["multi"] is True
    assert len(env["results"]) == 2


# --- run: the error contract ----------------------------------------------------------------------


def test_missing_file_is_usage_error(capsys):
    code, out, err = _run(["run", "pc", "--data", "/no/such/file.csv"], capsys)
    assert code == 2 and out.strip() == ""
    assert json.loads(err)["error"]["code"] == "input_not_found"


def test_unknown_method_enumerates_valid_options(capsys):
    code, _, err = _run(["run", "does-not-exist", "--help", "--json"], capsys)
    payload = json.loads(err)["error"]
    assert code == 2
    assert "pc" in payload["valid_options"] and payload["next"] == ["andrey list"]


def test_too_few_groups_is_usage_error(capsys, nongaussian_path):
    code, _, err = _run(
        ["run", "multi_group_direct_lingam", "--data-groups", str(nongaussian_path)], capsys
    )
    assert code == 2 and json.loads(err)["error"]["code"] == "too_few_groups"


def test_nan_data_is_data_error(capsys, tmp_path):
    path = tmp_path / "nan.csv"
    np.savetxt(path, np.full((50, 3), np.nan), delimiter=",")
    code, out, err = _run(["run", "pc", "--data", str(path)], capsys)
    assert code == 3 and out.strip() == ""
    assert json.loads(err)["error"]["code"] == "data_error"


# --- run grammar: key=value coercion (unit) -------------------------------------------------------


def test_coerce_scalars_and_choices():
    from andrey.cli import CliError, _coerce
    from andrey.spec import get_spec

    pc = {p.name: p for p in get_spec("pc").params}
    assert _coerce("0.05", pc["alpha"]) == 0.05
    assert _coerce("1e-3", pc["alpha"]) == 0.001  # scientific notation
    assert _coerce("fisherz", pc["indep_test"]) == "fisherz"
    with pytest.raises(CliError):  # bad choice
        _coerce("kci", pc["indep_test"])
    with pytest.raises(CliError):  # not a number
        _coerce("abc", pc["alpha"])


def test_coerce_bool_pair_list_and_null():
    from andrey.cli import CliError, _coerce
    from andrey.spec import get_spec

    varma = {p.name: p for p in get_spec("varma_lingam").params}
    labels = next(p for p in get_spec("gin").params if p.name == "labels")
    assert _coerce("true", varma["prune"]) is True
    assert _coerce("false", varma["prune"]) is False
    with pytest.raises(CliError):  # bool is true/false only
        _coerce("1", varma["prune"])
    assert _coerce("[1,1]", varma["order"]) == (1, 1)
    with pytest.raises(CliError):  # int_pair requires brackets
        _coerce("1,1", varma["order"])
    assert _coerce("[a,b,c]", labels) == ("a", "b", "c")
    assert _coerce('["a","b"]', labels) == ("a", "b")  # JSON-quoted items are unquoted
    assert _coerce("null", varma["criterion"]) is None  # None spelling
    with pytest.raises(CliError):  # empty value is not None
        _coerce("", varma["criterion"])


def test_coerce_quoted_is_literal_string():
    from andrey.cli import CliError, _coerce
    from andrey.spec import get_spec

    pc = {p.name: p for p in get_spec("pc").params}
    assert _coerce('"fisherz"', pc["indep_test"]) == "fisherz"  # quoted -> literal str
    with pytest.raises(CliError):  # a quoted value is not valid for a non-str param
        _coerce('"0.05"', pc["alpha"])


def test_coerce_passthrough_sniffs():
    from andrey.cli import CliError, _coerce_passthrough

    assert _coerce_passthrough("0.1") == 0.1
    assert _coerce_passthrough("5") == 5
    assert _coerce_passthrough("true") is True
    assert _coerce_passthrough("null") is None
    assert _coerce_passthrough("foo") == "foo"
    assert _coerce_passthrough('"true"') == "true"  # quoted stays a string
    assert _coerce_passthrough("[1,2]") == [1, 2]
    with pytest.raises(CliError):  # an empty passthrough value is an error, not ""
        _coerce_passthrough("")


# --- run grammar: overrides, config, self-healing (integration) -----------------------------------


def test_run_key_value_override(capsys, csv_path):
    code, out, _ = _run(["run", "pc", "--data", str(csv_path), "alpha=0.01"], capsys)
    env = json.loads(out)
    assert code == 0 and env["run"]["params"]["alpha"] == 0.01


def test_run_config_then_override(capsys, tmp_path, csv_path):
    cfg = tmp_path / "pc.yaml"
    cfg.write_text("alpha: 0.2\nindep_test: fisherz\n")
    code, out, _ = _run(
        ["run", "pc", "--data", str(csv_path), "--config", str(cfg), "alpha=0.3"], capsys
    )
    params = json.loads(out)["run"]["params"]
    assert code == 0 and params["alpha"] == 0.3  # CLI override > config > default


def test_run_config_rejects_wrong_type(capsys, tmp_path, csv_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text('alpha: "0.05"\n')  # a string where a float is expected
    code, _out, err = _run(["run", "pc", "--data", str(csv_path), "--config", str(cfg)], capsys)
    assert code == 2 and json.loads(err)["error"]["code"] == "invalid_params"


def test_unknown_param_reports_details(capsys, csv_path):
    code, _out, err = _run(["run", "pc", "--data", str(csv_path), "bogus=1"], capsys)
    payload = json.loads(err)["error"]
    assert code == 2 and payload["code"] == "invalid_params" and payload["details"]


def test_stale_flag_gets_did_you_mean(capsys, csv_path):
    code, _out, err = _run(["run", "pc", "--data", str(csv_path), "--alpha", "0.05"], capsys)
    payload = json.loads(err)["error"]
    assert code == 2 and "alpha=" in payload.get("did_you_mean", "")


def test_repeated_key_is_error(capsys, csv_path):
    code, _out, err = _run(["run", "pc", "--data", str(csv_path), "alpha=0.1", "alpha=0.2"], capsys)
    assert code == 2 and json.loads(err)["error"]["code"] == "invalid_params"


def test_missing_input_error(capsys):
    code, _out, err = _run(["run", "pc"], capsys)
    assert code == 2 and json.loads(err)["error"]["code"] == "missing_input"


def test_unexpected_input_error(capsys, csv_path):
    argv = ["run", "pc", "--data", str(csv_path), "--c-indx", str(csv_path)]
    code, _out, err = _run(argv, capsys)
    assert code == 2 and json.loads(err)["error"]["code"] == "unexpected_input"


def test_method_help_defaults_are_grammar_not_repr():
    """A default renders in key=value grammar (order=[1,1], prune=false), not a Python repr that
    would fail to parse if copy-pasted back."""
    from andrey.cli import render_method_help
    from andrey.spec import get_spec

    text = render_method_help(get_spec("varma_lingam"))
    assert "order=[1,1]" in text and "order=(1, 1)" not in text
    assert "prune=false" in text and "prune=False" not in text


def test_input_name_override_is_usage_error(capsys, csv_path):
    """A `data=...` override on a passthrough method is caught in validation (exit 2), not passed to
    the facade positionally where it would raise a raw TypeError (exit 1)."""
    code, _out, err = _run(["run", "calm", "--data", str(csv_path), "data=other.csv"], capsys)
    assert code == 2 and json.loads(err)["error"]["code"] == "invalid_params"


def test_param_error_precedes_input_load(capsys):
    """Params validate before any input I/O: a bad param surfaces even with no `--data` given."""
    code, _out, err = _run(["run", "pc", "alpha=abc"], capsys)
    assert code == 2 and json.loads(err)["error"]["code"] == "invalid_params"


def test_per_method_help(capsys):
    code, out, _ = _run(["run", "pc", "--help"], capsys)
    assert code == 0
    assert "andrey run pc" in out and "alpha=" in out and "andrey run pc --help --json" in out
    assert "--config" in out  # run-level flags are listed in per-method help
    assert "Experimental" not in out
    _, out, _ = _run(["run", "gfci", "--help"], capsys)
    assert "Experimental: no published benchmark" in out


# --- edge orientation + mark constants ------------------------------------------------------------


def test_edges_are_the_oriented_edges_for_every_mark_pair():
    from andrey.cli import graph_view
    from andrey.core import ARROW, CIRCLE, TAIL, GraphStructure

    marks = (TAIL, ARROW, CIRCLE)
    graphs = [
        GraphStructure.from_numpy(np.array([[0, a], [b, 0]]), kind="pag", labels=("u", "v"))
        for a in marks
        for b in marks
    ]
    loop = np.array([[ARROW, TAIL], [ARROW, 0]])  # a self-loop at 0, and 0 -> 1
    graphs.append(GraphStructure.from_numpy(loop, kind="digraph", allow_self_loops=True))
    for graph in graphs:
        edges = [(e["source"], e["target"], e["type"]) for e in graph_view(graph)["edges"]]
        assert edges == graph.oriented_edges(index=True)  # indices, even with labels


def test_two_cycles_in_graph_and_temporal_json():
    from andrey.cli import graph_view, temporal_view
    from andrey.core import ARROW, GraphStructure, TemporalStructure

    M = np.array([[0, ARROW], [ARROW, 0]], dtype=np.int8)
    graph = GraphStructure.from_numpy(M, kind="digraph", labels=("a", "b"))
    expected = [
        {"source": 0, "target": 1, "type": "directed"},
        {"source": 1, "target": 0, "type": "directed"},
    ]
    view = graph_view(graph, np.array([[0, 0.8], [-1.25, 0]]))
    assert view["n_edges"] == 2 and view["edge_types"] == {"directed": 2}
    assert view["edges"] == [{**expected[0], "weight": 0.8}, {**expected[1], "weight": -1.25}]
    temporal = TemporalStructure.from_lag_graphs([graph], lags=[1], labels=("a", "b"))
    view = temporal_view(temporal)
    assert view["n_edges"] == 2 and view["edge_types"] == {"directed": 2}
    assert view["lags"][0]["edges"] == expected


# --- the generated-option machinery (int_pair / bool / enum / temporal) ---------------------


def test_run_varma_temporal_envelope(capsys, csv_path):
    code, out, _ = _run(["run", "varma_lingam", "--data", str(csv_path), "order=[1,1]"], capsys)
    env = json.loads(out)
    assert code == 0
    s = env["structure"]
    assert s["type"] == "temporal" and s["n_lags"] >= 1
    assert "lag_weights" in s  # the coefficient stack promised by the output spec


def test_run_bool_flag_recorded(capsys, csv_path):
    code, out, _ = _run(["run", "varma_lingam", "--data", str(csv_path), "prune=true"], capsys)
    env = json.loads(out)
    assert code == 0 and env["run"]["params"]["prune"] is True


def test_run_enum_rejects_bad_value(capsys, csv_path):
    code, _out, err = _run(["run", "pc", "--data", str(csv_path), "indep_test=bogus"], capsys)
    assert code == 2
    assert json.loads(err)["error"]["code"] == "invalid_params"


def test_run_unsupported_value_is_usage_error(capsys, csv_path):
    code, _out, err = _run(["run", "grasp", "--data", str(csv_path), "depth=2"], capsys)
    assert code == 2
    assert json.loads(err)["error"]["code"] == "unsupported_value"


def test_run_pnl_on_a_named_pair(capsys, tmp_path):
    rng = np.random.default_rng(2)
    x = rng.uniform(-1, 1, 300)
    pair = tmp_path / "pair.csv"
    pair.write_text("cause,effect\n")
    with pair.open("a") as fh:
        np.savetxt(
            fh, np.column_stack([x, np.tanh(x**3 + 0.3 * rng.uniform(-1, 1, 300))]), delimiter=","
        )
    code, out, _ = _run(["run", "pnl", "--data", str(pair)], capsys)
    env = json.loads(out)
    s = env["structure"]
    assert code == 0 and env["run"]["n_variables"] == 2 and env["run"]["params"]["alpha"] is None
    assert s["labels"] == ["cause", "effect"]
    assert s["edges"] == [{"source": 0, "target": 1, "type": "directed"}]
    assert {"pval_forward", "pval_backward"} <= set(env["metadata"])
    code, out, _ = _run(["run", "pnl", "--data", str(pair), "alpha=0.05"], capsys)
    assert code == 0 and json.loads(out)["run"]["params"]["alpha"] == 0.05


def test_pnl_rejects_a_table_that_is_not_a_pair(capsys, tmp_path):
    rng = np.random.default_rng(4)
    bad = tmp_path / "three.csv"
    np.savetxt(bad, rng.standard_normal((100, 3)), delimiter=",")
    code, _out, err = _run(["run", "pnl", "--data", str(bad)], capsys)
    assert code == 3 and json.loads(err)["error"]["code"] == "data_error"


def test_seeded_run_reports_resolved_seed(capsys, csv_path):
    code, out, _ = _run(["run", "boss", "--data", str(csv_path)], capsys)
    env = json.loads(out)
    assert code == 0
    assert env["run"]["seed"] == 0  # no --seed -> ANDREY_SEED default, resolved (not null)


@pytest.mark.parametrize("name", ["pc", "direct_lingam", "varma_lingam", "gin"])
def test_output_envelope_matches_spec(name, capsys, csv_path, nongaussian_path):
    """The result envelope reflects the method's OutputSpec claims (kind / ordering / weighted)."""
    from andrey.spec import get_spec

    data = str(nongaussian_path if name in ("direct_lingam", "gin") else csv_path)
    code, out, _ = _run(["run", name, "--data", data], capsys)
    env = json.loads(out)
    spec = get_spec(name)
    assert code == 0
    assert env["status"] == spec.status
    s = env["structure"]
    assert s["type"] == spec.output.structure_type
    if spec.output.graph_kind:
        assert s["kind"] == spec.output.graph_kind
    if spec.output.has_ordering:
        assert "ordering" in s
    if spec.output.weighted:
        assert any("weight" in e for e in s["edges"])
    if spec.output.has_latents:
        assert "latent_nodes" in s


@pytest.mark.parametrize("filter_action", ["default", "error"])
def test_fit_warnings_go_in_the_envelope_not_stderr(capsys, csv_path, filter_action):
    """An error filter (``PYTHONWARNINGS=error``) does not turn a fit's warning into a failure."""
    with warnings.catch_warnings():
        warnings.simplefilter(filter_action)
        code, out, err = _run(["run", "gfci", "--data", str(csv_path)], capsys)
    assert code == 0 and err == ""
    [warning] = json.loads(out)["run"]["warnings"]
    assert warning["category"] == "ExperimentalWarning"
    assert warning["message"].startswith("andrey.gfci is experimental")


def test_supported_run_has_no_warnings(capsys, csv_path):
    code, out, _ = _run(["run", "pc", "--data", str(csv_path)], capsys)
    assert code == 0 and json.loads(out)["run"]["warnings"] == []


# --- shell-aware output ---------------------------------------------------------------------------


def test_human_summary_ends_with_the_fit_warnings(capsys, monkeypatch, csv_path):
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    _, out, err = _run(["run", "gfci", "--data", str(csv_path)], capsys)
    assert err == ""
    assert out.rstrip().splitlines()[-1].startswith("  ExperimentalWarning: andrey.gfci is")


def test_run_prints_human_summary_on_tty(capsys, monkeypatch, csv_path):
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    code, out, _ = _run(["run", "pc", "--data", str(csv_path)], capsys)
    assert code == 0
    assert "schema_version" not in out  # not the JSON envelope
    assert out.startswith("pc  cpdag")  # the human header


def test_json_flag_forces_json_on_tty(capsys, monkeypatch, csv_path):
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    code, out, _ = _run(["run", "pc", "--data", str(csv_path), "--json"], capsys)
    assert code == 0 and json.loads(out)["schema_version"] == SCHEMA_VERSION


# --- bounded output (large-graph threshold) -------------------------------------------------------


def test_edge_types_breakdown_is_complete(capsys, csv_path):
    code, out, _ = _run(["run", "pc", "--data", str(csv_path)], capsys)
    s = json.loads(out)["structure"]
    assert code == 0 and sum(s["edge_types"].values()) == s["n_edges"]


def test_edges_capped_on_stdout(capsys, monkeypatch, csv_path):
    monkeypatch.setattr("andrey.cli.EDGE_LIMIT", 1)
    code, out, _ = _run(["run", "pc", "--data", str(csv_path)], capsys)  # non-TTY -> JSON
    env = json.loads(out)
    s = env["structure"]
    assert code == 0
    assert s["edges_truncated"] is True and len(s["edges"]) == 1
    assert s["n_edges"] > 1  # the full count survives the cap
    assert sum(s["edge_types"].values()) == s["n_edges"]  # and so does the breakdown
    assert "note" in env


def test_full_flag_disables_cap(capsys, monkeypatch, csv_path):
    monkeypatch.setattr("andrey.cli.EDGE_LIMIT", 1)
    code, out, _ = _run(["run", "pc", "--data", str(csv_path), "--full"], capsys)
    s = json.loads(out)["structure"]
    assert code == 0
    assert "edges_truncated" not in s and len(s["edges"]) == s["n_edges"]


def test_output_file_is_never_capped(capsys, monkeypatch, tmp_path, csv_path):
    monkeypatch.setattr("andrey.cli.EDGE_LIMIT", 1)
    dest = tmp_path / "graph.json"
    code, out, _ = _run(["run", "pc", "--data", str(csv_path), "-o", str(dest)], capsys)
    assert code == 0 and out.strip() == ""
    s = json.loads(dest.read_text())["structure"]
    assert "edges_truncated" not in s and len(s["edges"]) == s["n_edges"]


# --- temporal envelope: schema fields + bounded dense weights -----------------------------------


def test_temporal_envelope_carries_summary_fields(capsys, csv_path):
    code, out, _ = _run(["run", "varma_lingam", "--data", str(csv_path), "order=[1,1]"], capsys)
    s = json.loads(out)["structure"]
    assert code == 0 and s["type"] == "temporal"
    assert "n_edges" in s and "edge_types" in s  # aggregate summary, like a graph structure
    assert all("edge_types" in lag for lag in s["lags"])  # per-lag summary
    assert s["n_edges"] == sum(lag["n_edges"] for lag in s["lags"])


def test_temporal_dense_weights_capped_on_stdout(capsys, monkeypatch, csv_path):
    monkeypatch.setattr("andrey.cli.EDGE_LIMIT", 1)
    code, out, _ = _run(["run", "varma_lingam", "--data", str(csv_path), "order=[1,1]"], capsys)
    env = json.loads(out)
    s = env["structure"]
    assert code == 0
    assert s["lag_weights"] is None and s["lag_weights_shape"]  # dense stack dropped, shape kept
    assert s["weights_truncated"] is True
    assert "note" in env


def test_temporal_full_keeps_dense_weights(capsys, monkeypatch, csv_path):
    monkeypatch.setattr("andrey.cli.EDGE_LIMIT", 1)
    code, out, _ = _run(
        ["run", "varma_lingam", "--data", str(csv_path), "order=[1,1]", "--full"], capsys
    )
    s = json.loads(out)["structure"]
    assert code == 0 and s["lag_weights"] is not None and "weights_truncated" not in s


def test_temporal_edge_cap_is_shared_across_lags():
    """The inline-edge budget is per result, not per lag: many lags each under the limit must not
    collectively exceed it. Three lags of two edges at limit 2 keep two edges total, not six."""
    from andrey.cli import _truncate_output

    two_edges = [{"source": 0, "target": 1}, {"source": 1, "target": 2}]
    struct = {"type": "temporal", "lags": [{"lag": k, "edges": list(two_edges)} for k in range(3)]}
    cut = _truncate_output(struct, limit=2)
    total = sum(len(lag["edges"]) for lag in struct["lags"])
    assert cut is True and total == 2  # aggregate budget, not 2 per lag (would be 6)
    assert any(lag.get("edges_truncated") for lag in struct["lags"])


# --- input parsing: a malformed first row is not a header -----------------------------------


def test_parse_text_rejects_malformed_first_row():
    from andrey.cli import CliError, _parse_text

    with pytest.raises(CliError) as exc:
        _parse_text("1,2,bad\n3,4,5\n6,7,8\n")
    assert exc.value.code == "bad_data"


def test_parse_text_skips_a_real_header():
    from andrey.cli import _parse_text

    arr, labels = _parse_text('a, "b",c\n1,2,3\n4,5,6\n')
    assert arr.shape == (2, 3) and labels == ("a", "b", "c")
    arr, labels = _parse_text("a\tb\n1\t2\n")  # TSV
    assert arr.shape == (1, 2) and labels == ("a", "b")


def test_parse_text_keeps_every_numeric_row():
    from andrey.cli import _parse_text

    arr, labels = _parse_text("1,2,3\n4,5,6\n7,8,9\n")
    assert arr.shape == (3, 3) and labels is None  # no row silently dropped


@pytest.mark.parametrize(("header", "message"), [("a,b", "names 2 columns"), ("a,a,b", "unique")])
def test_parse_text_rejects_a_header_that_cannot_name_the_columns(header, message):
    from andrey.cli import CliError, _parse_text

    with pytest.raises(CliError, match=message) as exc:
        _parse_text(f"{header}\n1,2,3\n4,5,6\n")
    assert exc.value.code == "bad_data"


# --- a header row names the nodes -----------------------------------------------------------------


@pytest.fixture
def named_path(tmp_path, csv_path):
    path = tmp_path / "named.csv"
    path.write_text("a,b,c,d\n" + csv_path.read_text())
    return path


def test_header_names_the_nodes(capsys, monkeypatch, named_path):
    from andrey.core.structure import EDGE_GLYPHS

    code, out, _ = _run(["run", "pc", "--data", str(named_path)], capsys)
    s = json.loads(out)["structure"]
    assert code == 0 and s["labels"] == ["a", "b", "c", "d"] and s["edges"]
    assert {e["source"] for e in s["edges"]} <= {0, 1, 2, 3}  # edges stay indices
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    code, out, _ = _run(["run", "pc", "--data", str(named_path)], capsys)
    for e in s["edges"]:
        assert f"  {'abcd'[e['source']]} {EDGE_GLYPHS[e['type']]} {'abcd'[e['target']]}\n" in out


def test_header_names_gin_observed_nodes_unless_labels_given(capsys, named_path):
    code, out, _ = _run(["run", "gin", "--data", str(named_path)], capsys)
    env = json.loads(out)
    assert code == 0 and env["structure"]["labels"][:4] == ["a", "b", "c", "d"]
    assert env["run"]["params"]["labels"] == ["a", "b", "c", "d"]  # the params the run used
    code, out, _ = _run(["run", "gin", "--data", str(named_path), "labels=[w,x,y,z]"], capsys)
    assert code == 0 and json.loads(out)["structure"]["labels"][:4] == ["w", "x", "y", "z"]


def test_panel_headers_name_every_group(capsys, tmp_path, named_path):
    argv = ["run", "multi_group_direct_lingam", "--data-groups", str(named_path)]
    code, out, _ = _run([*argv, "--data-groups", str(named_path)], capsys)
    results = json.loads(out)["results"]
    assert code == 0 and [r["structure"]["labels"] for r in results] == [["a", "b", "c", "d"]] * 2
    other = tmp_path / "other.csv"
    other.write_text(named_path.read_text().replace("a,b,c,d", "a,b,c,e", 1))
    code, _out, err = _run([*argv, "--data-groups", str(other)], capsys)
    assert code == 3 and json.loads(err)["error"]["code"] == "bad_data"


@pytest.mark.parametrize("method", ["fci", "gfci"])
def test_run_majority_colliders(method, csv_path, capsys):
    import andrey

    code, out, _ = _run(["run", method, "--data", str(csv_path), "collider_rule=majority"], capsys)
    assert code == 0
    result = json.loads(out)
    assert result["run"]["params"]["collider_rule"] == "majority"
    expected = getattr(andrey, method)(
        np.loadtxt(csv_path, delimiter=","), collider_rule="majority"
    ).structure
    assert result["structure"] == graph_view(expected)


def test_cli_rejects_unknown_collider_rule(csv_path, capsys):
    code, _, err = _run(["run", "fci", "--data", str(csv_path), "collider_rule=invalid"], capsys)
    assert code != 0
    assert "collider" in err


@pytest.mark.parametrize(
    "args",
    [
        ["run", "hc", "--help", "--json"],
        ["run", "hc", "--json", "--help"],
        ["run", "--help", "--json", "hc"],
        ["run", "--json", "hc", "-h"],
    ],
)
def test_method_contract_flag_order_and_no_io(args, capsys, monkeypatch, tmp_path):
    from andrey import cli
    from andrey.spec import help_schema

    def forbidden(*args, **kwargs):
        pytest.fail("Introspection must not read run inputs or execute a method")

    for name in ("_load_config", "_load_method_inputs", "execute_run"):
        monkeypatch.setattr(cli, name, forbidden)
    output = tmp_path / "unused.json"
    args += ["--data", "-", "--config", "missing.yaml", "-o", str(output), "unknown=bad"]
    code, out, err = _run(args, capsys)
    assert code == 0 and not err
    assert json.loads(out) == help_schema("hc")
    assert not output.exists()


@pytest.mark.parametrize("args", [["--help", "--json"], ["--json", "-h"]])
def test_package_contract_matches_generator(args, capsys):
    from andrey.spec import help_schema

    code, out, err = _run(args, capsys)
    assert code == 0 and not err
    assert json.loads(out) == help_schema()


@pytest.mark.parametrize("args", [["run", "hc", "--help", "--json"], ["run", "--json", "hc", "-h"]])
def test_direct_typer_entry_returns_method_contract(args):
    from typer.testing import CliRunner

    from andrey.cli import app
    from andrey.spec import help_schema

    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == help_schema("hc")


def test_json_without_help_still_requires_run_data(capsys):
    code, out, err = _run(["run", "hc", "--json"], capsys)
    assert code == 2 and not out
    error = json.loads(err)["error"]
    assert error["code"] == "missing_input"
    assert error["next"] == ["andrey run hc --help --json"]


@pytest.mark.parametrize("args", [["config", "--help", "--json"], ["config", "--json", "-h"]])
def test_config_help_uses_the_specs_without_backend_probe(args, capsys, monkeypatch):
    from andrey.spec.generate import env_dict

    def forbidden():
        pytest.fail("Configuration help must not probe optional backends")

    monkeypatch.setattr("andrey.describe", forbidden)
    monkeypatch.setenv("ANDREY_BACKEND", "invalid")
    code, out, err = _run(args, capsys)
    assert code == 0 and not err
    report = json.loads(out)
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["environment"] == env_dict()
    assert "configuration" not in report


def test_config_report_separates_preference_availability_and_raw_values(capsys, monkeypatch):
    from andrey.core.backend import Backend
    from andrey.spec.generate import env_dict

    state = Backend(backend="auto", num_workers=-1, numba=False, torch=True, cuda=False, mps=True)
    monkeypatch.setattr("andrey.describe", lambda: state)
    monkeypatch.setenv("ANDREY_NUM_WORKERS", "-1")
    monkeypatch.setenv("ANDREY_GES_PARALLEL_MIN_WORK", "90000")
    monkeypatch.delenv("ANDREY_BACKEND", raising=False)
    code, out, err = _run(["config", "--json"], capsys)
    assert code == 0 and not err
    report = json.loads(out)
    assert report["configuration"] == {"backend": "auto", "num_workers": -1}
    assert report["available_backends"] == {
        "numpy": True,
        "numba": False,
        "cpu": True,
        "cuda": False,
        "mps": True,
    }
    variables = {v["name"]: v for v in report["environment"]["variables"]}
    assert variables["ANDREY_NUM_WORKERS"]["value"] == "-1"
    assert variables["ANDREY_BACKEND"]["value"] is None
    assert variables["ANDREY_GES_PARALLEL_MIN_WORK"]["value"] == "90000"
    for actual, expected in zip(
        report["environment"]["variables"], env_dict()["variables"], strict=True
    ):
        assert {k: v for k, v in actual.items() if k != "value"} == expected


def test_config_report_is_readable_when_piped(capsys, monkeypatch):
    from andrey.core.backend import Backend

    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    monkeypatch.setattr(
        "andrey.describe",
        lambda: Backend(
            backend="numpy", num_workers=1, numba=False, torch=False, cuda=False, mps=False
        ),
    )
    code, out, err = _run(["config"], capsys)
    assert code == 0 and not err
    assert "Backend preference: numpy" in out
    assert "Available backends: numpy" in out
    assert "  numba: the [numba] extra is not installed" in out
    assert "Platform: " in out and "usable cores" in out
    assert "Extras:" in out
    assert "andrey config --help" in out


def test_config_report_describes_the_machine_and_how_to_fill_its_gaps(capsys, monkeypatch):
    # A faked machine: torch installed without a GPU, viz half installed, not macOS.
    from andrey import cli
    from andrey.core.backend import Backend

    monkeypatch.setattr(
        "andrey.describe",
        lambda: Backend(
            backend="auto", num_workers=1, numba=False, torch=True, cuda=False, mps=False
        ),
    )
    monkeypatch.setattr(
        cli,
        "_extra_packages",
        lambda: {
            "numba": ["numba"],
            "torch": ["torch"],
            "viz": ["matplotlib", "pygraphviz"],
            "data": ["safetensors"],
        },
    )
    monkeypatch.setattr(cli, "_version", {"torch": "2.8.0", "matplotlib": "3.9.0"}.get)
    monkeypatch.setattr(cli.sys, "platform", "linux")
    # The cores ANDREY_NUM_WORKERS=-1 would use: the same count, cgroup quota included.
    monkeypatch.setattr(cli, "usable_cpus", lambda: 2)
    code, out, err = _run(["config", "--json"], capsys)
    assert code == 0 and not err
    system = json.loads(out)["system"]
    assert system["platform"] == "linux" and system["support"] == "supported"
    assert system["usable_cores"] == 2
    extras = {extra["name"]: extra for extra in system["extras"]}
    assert list(extras) == list(cli.EXTRAS)
    assert extras["torch"]["installed"] and extras["torch"]["packages"] == {"torch": "2.8.0"}
    assert not extras["viz"]["installed"]  # pygraphviz is missing
    assert extras["viz"]["install"] == 'pip install "andrey-core[viz]"'
    assert system["unavailable_backends"] == {
        "numba": "the [numba] extra is not installed",
        "cuda": "torch finds no CUDA device",
        "mps": "Apple MPS needs macOS",
    }


def test_the_reported_extras_are_the_packages_extras():
    import tomllib
    from pathlib import Path

    from andrey import cli

    pyproject = tomllib.loads((Path(__file__).parents[2] / "pyproject.toml").read_text())
    extras = set(pyproject["project"]["optional-dependencies"]) - {"all"}
    assert set(cli.EXTRAS) == extras
    # The installed metadata names each extra's packages.
    assert extras <= set(cli._extra_packages())


@pytest.mark.parametrize(("torch", "cuda"), [(False, False), (True, False), (True, True)])
def test_config_and_describe_report_one_availability_shape(torch, cuda, capsys, monkeypatch):
    # A faked machine: `andrey config --json` reports exactly what describe().as_dict() does, in
    # the ANDREY_BACKEND vocabulary, where `cpu` is torch on the CPU.
    from andrey.core import backend

    installed = {"numba": False, "torch": torch}
    monkeypatch.setattr(backend, "_installed", lambda name: installed.get(name, False))
    monkeypatch.setattr(backend, "_torch_gpu_build", lambda: True)
    monkeypatch.setattr(backend, "_is_available", lambda device: cuda and device == "cuda")
    monkeypatch.delenv("ANDREY_BACKEND", raising=False)
    monkeypatch.delenv("ANDREY_DEVICE", raising=False)
    code, out, err = _run(["config", "--json"], capsys)
    assert code == 0 and not err
    reported = json.loads(out)["available_backends"]
    expected = {"numpy": True, "numba": False, "cpu": torch, "cuda": cuda, "mps": False}
    assert reported == backend.describe().as_dict()["available_backends"] == expected


# One bad value per parser: backend name, worker count, integer, seed, flag.
BAD_SETTINGS = [
    ("ANDREY_BACKEND", "quantum"),
    ("ANDREY_NUM_WORKERS", "-2"),
    ("ANDREY_COV_GPU_THRESHOLD", "abc"),
    ("ANDREY_SEED", "4294967296"),
    ("ANDREY_GPU_CALIBRATE", "maybe"),
]


@pytest.mark.parametrize(("name", "value"), BAD_SETTINGS)
def test_bad_setting_fails_config_and_run_but_not_help(name, value, capsys, monkeypatch):
    monkeypatch.setenv(name, value)
    for args in (["config"], ["config", "--json"], ["run", "pc", "--data", "missing.csv"]):
        code, out, err = _run(args, capsys)
        assert code == 2 and not out, args
        error = json.loads(err)["error"]
        assert error["code"] == "invalid_configuration"
        assert f"{name} must be" in error["message"] and repr(value) in error["message"]
        assert error["next"] == ["andrey config --help"]
    for args in (["config", "--help"], ["config", "--help", "--json"], ["run", "pc", "--help"]):
        code, _, err = _run(args, capsys)
        assert code == 0 and not err, args


def test_invalid_configuration_names_every_bad_setting(capsys, monkeypatch):
    monkeypatch.setenv("ANDREY_SEED", "abc")
    monkeypatch.setenv("ANDREY_REQUIRE_GPU", "maybe")
    code, _, err = _run(["config", "--json"], capsys)
    message = json.loads(err)["error"]["message"]
    assert code == 2 and "ANDREY_SEED" in message and "ANDREY_REQUIRE_GPU" in message


@pytest.mark.parametrize(
    "args",
    [
        ["--json"],
        ["--help", "--skill"],
        ["--skill", "--llms"],
        ["run"],
        ["run", "--help", "--json"],
        ["agent-context", "hc"],
    ],
)
def test_invalid_introspection_requests_have_structured_errors(args, capsys):
    code, out, err = _run(args, capsys)
    assert code == 2 and not out
    assert json.loads(err)["error"]["code"] in {"usage_error", "missing_method"}


def test_list_json_on_terminal(capsys, monkeypatch):
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    code, out, err = _run(["list", "--json"], capsys)
    assert code == 0 and not err
    assert {row["name"] for row in json.loads(out)} == set(REGISTRY)


def test_root_help_keeps_run_override_syntax(capsys):
    code, out, _ = _run(["--help"], capsys)
    assert code == 0 and "[key=value ...]" in out


def test_root_json_without_help_points_to_command_json(capsys):
    code, out, err = _run(["--json", "list"], capsys)
    assert code == 2 and not out
    assert "andrey list --json" in json.loads(err)["error"]["next"]


@pytest.mark.parametrize(
    ("args", "hint"),
    [(["run"], "andrey list"), (["run", "--help", "--json"], "andrey --help --json")],
)
def test_missing_method_suggests_next_command(args, hint, capsys):
    code, _, err = _run(args, capsys)
    error = json.loads(err)["error"]
    assert code == 2 and error["code"] == "missing_method" and error["next"] == [hint]


@pytest.mark.parametrize(
    "path",
    [
        "DEVELOPMENT.md",
        "docs/docs/code/index.md",
    ],
)
def test_documented_schema_version_matches_code(path):
    import re
    from pathlib import Path

    text = (Path(__file__).parents[2] / path).read_text()
    assert set(re.findall(r'`schema_version`?:? \(?`?"(\d+)"', text)) == {SCHEMA_VERSION}
