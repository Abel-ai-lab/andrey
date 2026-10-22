"""The release scripts flag what they guard against and pass what they should not."""

from __future__ import annotations

import importlib.util
import io
import json
import re
import runpy
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def dist_check():
    return runpy.run_path(str(ROOT / "scripts" / "check_dist.py"))


def _wheel(path: Path, names: list[str], version: str = "0.1.0") -> None:
    contents = {
        "METADATA": "Name: andrey-core\nVersion: 0.1.0\n\nbody\n",
        "andrey/__init__.py": f'__version__ = "{version}"\n',
    }
    with zipfile.ZipFile(path, "w") as archive:
        for name in names:
            key = "METADATA" if name.endswith("METADATA") else name
            archive.writestr(name, contents.get(key, ""))


def _sdist(path: Path, names: list[str]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name in names:
            archive.addfile(tarfile.TarInfo(f"andrey_core-0.1.0/{name}"), io.BytesIO(b""))


_INFO = "andrey_core-0.1.0.dist-info"
_WHEEL_OK = [
    "andrey/__init__.py",
    "andrey/core/score.py",
    *(
        f"{_INFO}/{n}"
        for n in ("METADATA", "WHEEL", "RECORD", "entry_points.txt", "licenses/LICENSE")
    ),
]
_SDIST_OK = [
    "src/andrey/__init__.py",
    "src/andrey/core/score.py",
    *("pyproject.toml", "README.md", "LICENSE", "PKG-INFO", ".gitignore"),
]


def test_dist_check_passes_the_allowlist(tmp_path, dist_check):
    _wheel(tmp_path / "andrey_core-0.1.0-py3-none-any.whl", _WHEEL_OK)
    _sdist(tmp_path / "andrey_core-0.1.0.tar.gz", _SDIST_OK)
    assert dist_check["main"]([str(tmp_path)]) == 0


def test_dist_check_flags_files_outside_the_allowlist(tmp_path, dist_check, capsys):
    _wheel(tmp_path / "a-0.1.0-py3-none-any.whl", [*_WHEEL_OK, "andrey/data/table.json"])
    _sdist(
        tmp_path / "a-0.1.0.tar.gz",
        [*_SDIST_OK, "tests/recovery/baselines/pc.json", "qa/README.md", "src/andrey/viz/new.py"],
    )
    assert dist_check["main"]([str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "4 packaging problem(s)" in err
    for line in (
        "wheel: andrey/data/table.json",
        "sdist: tests/recovery/baselines/pc.json",
        "sdist: qa/README.md",
        "in one artifact only: andrey/viz/new.py",
    ):
        assert line in err


def test_dist_check_flags_a_version_that_differs_from_the_metadata(tmp_path, dist_check, capsys):
    _wheel(tmp_path / "andrey_core-0.1.0-py3-none-any.whl", _WHEEL_OK, version="0.0.0")
    _sdist(tmp_path / "andrey_core-0.1.0.tar.gz", _SDIST_OK)
    assert dist_check["main"]([str(tmp_path)]) == 1
    assert "metadata version 0.1.0 but andrey.__version__ 0.0.0" in capsys.readouterr().err


def test_readme_links_must_be_absolute(dist_check):
    markdown = "\n".join(
        [
            '<img src="https://andrey.abel.ai/a.svg" srcset="assets/dark.svg 2x, https://x.io/b.svg">',
            "[docs](https://andrey.abel.ai/docs/) and [guide](docs/guide.md) and [top](#install)",
            "[ref]: ../CONTRIBUTING.md",
            "[mail](mailto:team@example.org)",
            "```python",
            "table[0](x)  # code, not a link",
            "```",
        ]
    )
    assert dist_check["relative_links"](markdown) == [
        "docs/guide.md",
        "#install",
        "../CONTRIBUTING.md",
        "assets/dark.svg",
    ]


QUALITY_GATE = "./.github/workflows/quality-gate.yml"


def ungated_publish_jobs(workflow: dict) -> list[str]:
    """Return the jobs that publish to PyPI without needing a job that runs the quality gate."""
    jobs = workflow.get("jobs") or {}
    gates = {name for name, job in jobs.items() if job.get("uses") == QUALITY_GATE}
    ungated = []
    for name, job in jobs.items():
        steps = job.get("steps") or []
        publishes = any(
            "pypa/gh-action-pypi-publish" in str(step.get("uses", ""))
            or any(cmd in str(step.get("run", "")) for cmd in ("twine upload", "uv publish"))
            for step in steps
        )
        needs = job.get("needs") or []
        if publishes and not gates & ({needs} if isinstance(needs, str) else set(needs)):
            ungated.append(name)
    return ungated


def test_every_pypi_publish_job_needs_the_quality_gate():
    for path in sorted((ROOT / ".github" / "workflows").glob("*.yml")):
        assert ungated_publish_jobs(yaml.safe_load(path.read_text())) == [], path.name


def test_the_publish_guard_flags_a_job_without_the_gate():
    action = {"steps": [{"uses": "pypa/gh-action-pypi-publish@release/v1"}]}
    command = {"steps": [{"run": "uv publish dist/*"}]}
    assert ungated_publish_jobs({"jobs": {"a": action, "b": command}}) == ["a", "b"]
    gated = {"gate": {"uses": QUALITY_GATE}, "a": {**action, "needs": ["build", "gate"]}}
    assert ungated_publish_jobs({"jobs": {**gated, "b": {**command, "needs": "gate"}}}) == []


def _workflow(name: str) -> dict:
    # BaseLoader keeps every value a string and the `on` key a string, not `True`.
    path = ROOT / ".github" / "workflows" / name
    return yaml.load(path.read_text(), Loader=yaml.BaseLoader)


def _evaluate(condition: str, needs: str | list[str], **context: dict) -> bool:
    """Evaluate a job's ``if`` for the ``results``, ``outputs``, and ``inputs`` given.

    ``!cancelled()`` is true: the run is not cancelled. Every job the condition reads must be in
    the job's ``needs``, as GitHub resolves ``needs.<job>`` only for those.
    """
    needed = {needs} if isinstance(needs, str) else set(needs)
    read = set(re.findall(r"needs\.([\w-]+)\.", condition))
    assert read <= needed, read - needed
    expression = re.sub(r"^\$\{\{|\}\}$", "", condition.strip()).replace("!cancelled()", "True")
    expression = re.sub(r"needs\.([\w-]+)\.result", r'results["\1"]', expression)
    expression = re.sub(r"needs\.([\w-]+)\.outputs\.(\w+)", r'outputs["\1.\2"]', expression)
    expression = re.sub(r"inputs\.(\w+)", r'inputs["\1"]', expression)
    expression = expression.replace("&&", " and ").replace("||", " or ")
    return eval(expression, {"__builtins__": {}}, context)


def test_the_quality_gate_runs_on_pull_requests_into_main():
    triggers = _workflow("quality-gate.yml")["on"]
    assert triggers["pull_request"]["branches"] == ["main"]
    assert "workflow_call" in triggers


def test_the_publish_gate_runs_unless_the_lookup_found_a_run():
    jobs = _workflow("publish.yml")["jobs"]
    gate = jobs["quality-gate"]
    assert gate["uses"] == QUALITY_GATE
    # A failed or empty lookup is not a reuse, so the gate runs.
    for reuse, runs in (("true", False), ("false", True), ("", True)):
        outputs = {"gate-lookup.reuse": reuse}
        assert _evaluate(gate["if"], gate["needs"], outputs=outputs) is runs, reuse
    lookup = jobs["gate-lookup"]
    assert lookup["permissions"]["actions"] == "read"
    assert lookup["steps"][-1]["continue-on-error"] == "true"


@pytest.mark.parametrize(
    ("job", "index"), [("publish-testpypi", "testpypi"), ("publish-pypi", "pypi")]
)
def test_publish_needs_a_gate_that_passed_or_was_reused(job, index):
    publish = _workflow("publish.yml")["jobs"][job]
    assert "cancelled()" in publish["if"]  # else GitHub adds success(), which a skipped gate fails

    def publishes(gate, reuse, build="success", install="success", chosen=index):
        results = {"build": build, "install": install, "quality-gate": gate}
        outputs = {"gate-lookup.reuse": reuse}
        inputs = {"index": chosen}
        return _evaluate(
            publish["if"], publish["needs"], results=results, outputs=outputs, inputs=inputs
        )

    assert publishes("success", "")
    assert publishes("skipped", "true")
    for gate, reuse in (
        ("skipped", ""),
        ("skipped", "false"),
        ("failure", "true"),
        ("cancelled", ""),
    ):
        assert not publishes(gate, reuse), (gate, reuse)
    assert not publishes("success", "", build="failure")
    assert not publishes("success", "", install="skipped")
    assert not publishes("success", "", chosen="elsewhere")


@pytest.fixture(scope="module")
def gate_reuse():
    spec = importlib.util.spec_from_file_location("gate_reuse", ROOT / "scripts" / "gate_reuse.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _gate_run(tree: str | None, conclusion: str = "success", run_id: int = 1) -> dict:
    return {
        "id": run_id,
        "conclusion": conclusion,
        "html_url": f"https://github.com/example/runs/{run_id}",
        "head_commit": {"tree_id": tree},
    }


def test_a_passing_run_on_the_published_tree_is_reused(gate_reuse):
    runs = [_gate_run("other", run_id=1), _gate_run("tree", run_id=2), _gate_run("tree", run_id=3)]
    assert gate_reuse.passing_run(runs, "tree")["id"] == 2


@pytest.mark.parametrize(
    ("runs", "tree"),
    [
        ([_gate_run("other")], "tree"),  # a change landed after the gate ran
        ([_gate_run("tree", "failure"), _gate_run("tree", "cancelled")], "tree"),
        ([_gate_run("tree", "skipped"), _gate_run("tree", None)], "tree"),
        (
            [{"id": 1, "conclusion": "success"}, {"conclusion": "success", "head_commit": None}],
            "tree",
        ),
        ([_gate_run(None)], None),  # no tree to compare is never a match
        ([_gate_run("")], ""),
        ([], "tree"),
    ],
)
def test_a_tree_mismatch_or_an_unfinished_run_forces_the_gate(gate_reuse, runs, tree):
    assert gate_reuse.passing_run(runs, tree) is None


def _reuse(gate_reuse, monkeypatch, tmp_path, output) -> str:
    output_file = tmp_path / "output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
    monkeypatch.setenv("GITHUB_REPOSITORY", "example/andrey")
    monkeypatch.setattr(gate_reuse, "_output", output)
    assert gate_reuse.main() == 0
    return output_file.read_text().strip()


def _fake_github(tree: str, runs: list[dict]):
    def output(command: list[str]) -> str:
        if command[0] == "git":
            return tree
        assert command[:2] == ["gh", "api"] and "quality-gate.yml/runs" in command[2]
        return json.dumps({"workflow_runs": runs})

    return output


def test_the_lookup_reuses_the_gate_only_for_the_checked_out_tree(
    gate_reuse, monkeypatch, tmp_path
):
    runs = [_gate_run("tree-a", run_id=1), _gate_run("tree-b", run_id=2)]
    same = _fake_github("tree-b", runs)
    assert _reuse(gate_reuse, monkeypatch, tmp_path, same) == "reuse=true"
    (tmp_path / "output").unlink()
    changed = _fake_github("tree-c", runs)
    assert _reuse(gate_reuse, monkeypatch, tmp_path, changed) == "reuse=false"


@pytest.mark.parametrize(
    "failure", [subprocess.CalledProcessError(1, "gh"), OSError("no gh"), ValueError("bad json")]
)
def test_a_failed_lookup_runs_the_gate_instead_of_failing(
    gate_reuse, monkeypatch, tmp_path, failure
):
    def output(command):
        raise failure

    assert _reuse(gate_reuse, monkeypatch, tmp_path, output) == "reuse=false"


def test_an_unusable_api_response_runs_the_gate(gate_reuse, monkeypatch, tmp_path):
    def output(command):
        return "git-tree" if command[0] == "git" else json.dumps({"message": "Not Found"})

    assert _reuse(gate_reuse, monkeypatch, tmp_path, output) == "reuse=false"


@pytest.mark.parametrize("lane", ["preview", "prod"])
def test_website_workflows_request_deployment_for_exact_application_commits(lane):
    path = ROOT / ".github" / "workflows" / f"deploy-{lane}.yml"
    workflow = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
    job = workflow["jobs"]["request-build"]
    step = job["steps"][0]
    # The server and the credentials come from secrets, which the logs mask; a variable would print.
    for key in ("DEPLOY_URL", "DEPLOY_USER", "DEPLOY_TOKEN"):
        assert re.fullmatch(r"\$\{\{ secrets\.\w+ \}\}", step["env"][key]), key
    assert "vars." not in path.read_text()
    assert step["env"]["COMMIT_SHA"] == "${{ github.sha }}"
    assert '"$DEPLOY_URL")' in step["run"]
    assert "pull_request" not in workflow["on"]
    if lane == "preview":
        assert workflow["on"]["push"]["branches"] == ["main"]
        assert "github.ref == 'refs/heads/main'" in job["if"]
    else:
        assert workflow["on"]["push"]["tags"] == ["v*"]
        assert set(job["needs"]) == {"ci", "docs"}
        assert workflow["jobs"]["ci"]["uses"] == "./.github/workflows/ci.yml"
        assert workflow["jobs"]["docs"]["uses"] == "./.github/workflows/docs.yml"


def _summary(*rows: dict, comparator: str | None = "ref") -> dict:
    return {"methods": [{"method": "M", "comparator": comparator, "rows": list(rows)}]}


def _entry(**changes) -> dict:
    entry = {
        "summary": "s",
        "method": "M",
        "comparator": "ref",
        "row": {"d": 50},
        "andrey_errors": 5.0,
        "comparator_errors": 4.0,
        "cause": "correct-by-paper",
        "reason": "the paper's rule",
        "citation": "Author (2000).",
    }
    return {**entry, **changes}


def _failures(*rows: dict, allowlist: tuple[dict, ...] = ()) -> list[str]:
    from qa import reference_parity

    return reference_parity.check(reference_parity.paired_rows("s", _summary(*rows)), [*allowlist])[
        0
    ]


@pytest.mark.parametrize(
    ("row", "fails"),
    [
        ({"d": 50, "andrey_shd": 5.0, "comparator_shd": 4.99}, True),
        ({"d": 50, "andrey_shd": 5.0, "comparator_shd": 5.0}, False),
        ({"d": 50, "andrey_directed_errors": 3.0, "comparator_directed_errors": 2.0}, True),
        # Per-series counts decide the row alone, whatever the central errors say.
        (
            {
                "d": 50,
                "andrey_fewer_errors": 1,
                "comparator_fewer_errors": 2,
                "andrey_shd": 1.0,
                "comparator_shd": 9.0,
            },
            True,
        ),
        (
            {
                "d": 50,
                "andrey_fewer_errors": 2,
                "comparator_fewer_errors": 1,
                "andrey_shd": 9.0,
                "comparator_shd": 1.0,
            },
            False,
        ),
        ({"n": 200, "andrey_right_comparator_wrong": 0, "comparator_right_andrey_wrong": 1}, True),
        # A side without a result is not a paired row.
        ({"d": 50, "andrey_shd": 5.0, "comparator_shd": None}, False),
    ],
)
def test_parity_fails_a_row_where_the_reference_has_fewer_errors(row, fails):
    assert bool(_failures(row)) is fails


def test_parity_reports_each_failing_row_with_its_numbers():
    rows = (
        {"d": 50, "andrey_shd": 5.0, "comparator_shd": 4.67},
        {"d": 100, "andrey_shd": 7.0, "comparator_shd": 6.0},
    )
    failures = _failures(*rows)
    assert len(failures) == 2
    assert "d=50" in failures[0] and "5" in failures[0] and "4.67" in failures[0]


def test_parity_rejects_a_paired_row_without_an_error_field():
    from qa import reference_parity

    with pytest.raises(ValueError):
        reference_parity.paired_rows("s", _summary({"d": 50, "andrey_f1": 0.5}))
    assert reference_parity.paired_rows("s", _summary({"d": 50}, comparator=None)) == []


def test_parity_allowlist_exempts_only_a_cited_cause_with_the_row_numbers():
    row = {"d": 50, "andrey_shd": 5.0, "comparator_shd": 4.0}
    assert _failures(row, allowlist=(_entry(),)) == []
    for bad in (
        _entry(cause="column-order"),
        _entry(citation=""),
        _entry(reason=" "),
        _entry(andrey_errors=6.0),
    ):
        assert len(_failures(row, allowlist=(bad,))) == 1, bad
    assert len(_failures(row, allowlist=(_entry(), _entry()))) == 1


def test_parity_allowlist_exempts_a_tie_break_only_when_the_reversed_order_passes():
    # Series counts: the reference has fewer errors on 4 series, Andrey on 0.
    row = {"d": 50, "andrey_fewer_errors": 0, "comparator_fewer_errors": 4}
    reversed_order = {"column_order": "reversed", "andrey_errors": 1, "comparator_errors": 1}

    def tie_break(evidence):
        return _entry(cause="tie-break", andrey_errors=4, comparator_errors=0, evidence=evidence)

    assert _failures(row, allowlist=(tie_break(reversed_order),)) == []
    for evidence in (
        None,
        "pgmpy wins on ties",
        {**reversed_order, "column_order": "random"},
        {"column_order": "reversed", "andrey_errors": 1},
        {**reversed_order, "andrey_errors": 2},
    ):
        failures = _failures(row, allowlist=(tie_break(evidence),))
        assert len(failures) == 1 and "allowlist entry rejected" in failures[0], evidence


def test_parity_fails_an_allowlist_entry_without_a_failing_row():
    row = {"d": 50, "andrey_shd": 4.0, "comparator_shd": 5.0}
    assert len(_failures(row, allowlist=(_entry(),))) == 1
    assert len(_failures(row, allowlist=(_entry(row={"d": 20}),))) == 1


def test_parity_judges_a_method_by_its_newest_summary(tmp_path):
    from qa import reference_parity

    for name, generated, shd in (("old", "2026-09-01", 9.0), ("new", "2026-10-01", 4.0)):
        (tmp_path / name).mkdir()
        summary = {
            "generated": generated,
            **_summary({"d": 50, "andrey_shd": shd, "comparator_shd": 5.0}),
        }
        (tmp_path / name / "summary.json").write_text(json.dumps(summary))
    rows = reference_parity.published_rows(tmp_path)
    assert [r.summary for r in rows] == ["new"]
    assert reference_parity.check(rows, [])[0] == []


def _column_row(andrey: list, reference: list, **changes) -> dict:
    """A row of an order-dependent reference whose order-free default passes: Andrey ahead on 2."""
    row = {
        "d": 10,
        "andrey_fewer_errors": 2,
        "comparator_fewer_errors": 0,
        "reference_order_dependent": True,
        "andrey_column_shd_by_seed": andrey,
        "comparator_shd_by_seed": reference,
    }
    return {**row, **changes}


def test_parity_judges_an_order_dependent_reference_by_andrey_in_column_order():
    from qa import reference_parity

    # Column order: the reference has fewer errors on seeds 0 and 2, Andrey on seed 3; seed 4 has no
    # result on one side and is not a series.
    row = _column_row([3, 1, 2, 0, None], [2, 1, 1, 1, 5])
    [parsed] = reference_parity.paired_rows("s", _summary(row))
    assert (parsed.rule, parsed.andrey, parsed.reference, parsed.default) == (
        "column-order series",
        2,
        1,
        (0, 2),
    )
    [failure] = _failures(row)
    assert "column-order mode" in failure and "order-free default (not judged)" in failure
    # A default that fails is not judged when the column-order arm passes.
    assert _failures(_column_row([1, 1], [2, 1], comparator_fewer_errors=5)) == []


def test_parity_judges_an_order_dependent_reference_without_the_arm_by_the_default():
    row = {"d": 10, "andrey_fewer_errors": 0, "comparator_fewer_errors": 1}
    assert len(_failures({**row, "reference_order_dependent": True})) == 1
    assert _failures({**row, "andrey_fewer_errors": 1, "reference_order_dependent": True}) == []


@pytest.mark.parametrize(
    "row",
    [
        _column_row([1, 2], [1, 2], reference_order_dependent=False),
        {k: v for k, v in _column_row([1], [1]).items() if k != "reference_order_dependent"},
        {k: v for k, v in _column_row([1], [1]).items() if k != "comparator_shd_by_seed"},
        _column_row([1, 2], [1]),
    ],
)
def test_parity_rejects_a_column_order_arm_it_cannot_judge(row):
    from qa import reference_parity

    with pytest.raises(ValueError):
        reference_parity.paired_rows("s", _summary(row))


def test_parity_pins_the_column_order_comparison_of_an_allowlisted_row():
    row = _column_row([3, 1, 2], [2, 1, 1])  # the reference has fewer errors on 2 series, Andrey 0
    reversed_order = {"column_order": "reversed", "andrey_errors": 0, "comparator_errors": 1}
    entry = _entry(
        row={"d": 10},
        cause="order-dependent-reference",
        andrey_errors=2,
        comparator_errors=0,
        evidence=reversed_order,
    )
    assert _failures(row, allowlist=(entry,)) == []
    # The order-free default's counts (Andrey ahead on 2) are not what the entry pins.
    assert len(_failures(row, allowlist=({**entry, "andrey_errors": 0, "comparator_errors": 2},)))


def test_parity_keys_a_row_by_regime_when_a_method_has_one_entry_per_regime():
    from qa import reference_parity

    row = {"d": 10, "n": 500, "andrey_fewer_errors": 0, "comparator_fewer_errors": 1}
    summary = {
        "methods": [
            {"method": "M", "comparator": "ref", "regime": regime, "rows": [row]}
            for regime in ("additive", "nonmonotone")
        ]
    }
    rows = reference_parity.paired_rows("s", summary)
    assert [r.key for r in rows] == [
        {"regime": "additive", "d": 10, "n": 500},
        {"regime": "nonmonotone", "d": 10, "n": 500},
    ]
    entry = _entry(row={"regime": "additive", "d": 10, "n": 500}, cause="experimental-test")
    failures = reference_parity.check(
        rows, [{**entry, "andrey_errors": 1, "comparator_errors": 0}]
    )[0]
    assert len(failures) == 1 and "regime=nonmonotone" in failures[0]
    for method in summary["methods"]:
        del method["regime"]
    with pytest.raises(ValueError):
        reference_parity.paired_rows("s", summary)


def test_parity_allowlist_checks_the_evidence_of_each_cause():
    row = {"d": 50, "andrey_fewer_errors": 0, "comparator_fewer_errors": 3}

    def entry(cause, evidence=None):
        return _entry(cause=cause, andrey_errors=3, comparator_errors=0, evidence=evidence)

    draws = {"draws": [{"seed": 1, "andrey_errors": 1, "comparator_errors": 3}] * 2}
    reversed_order = {"column_order": "reversed", "andrey_errors": 4, "comparator_errors": 12}
    accepted = (
        entry("experimental-test"),
        entry("random-draw", draws),
        entry("order-dependent-reference", reversed_order),
    )
    for good in accepted:
        assert _failures(row, allowlist=(good,)) == [], good
    one_draw = {"draws": draws["draws"][:1]}
    failing_draw = {
        "draws": [*draws["draws"], {"seed": 3, "andrey_errors": 2, "comparator_errors": 1}]
    }
    rejected = (
        entry("random-draw"),
        entry("random-draw", one_draw),
        entry("random-draw", failing_draw),
        entry("order-dependent-reference"),
        entry("order-dependent-reference", {**reversed_order, "andrey_errors": 13}),
    )
    for bad in rejected:
        failures = _failures(row, allowlist=(bad,))
        assert len(failures) == 1 and "allowlist entry rejected" in failures[0], bad
