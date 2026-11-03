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


@pytest.mark.parametrize(
    "paths",
    [
        [],  # the same tree
        ["CHANGELOG.md"],
        ["README.md", "AGENTS.md", "CHANGELOG.md"],
        ["docs/docs/ci.md", "docs/blog/20261023-post.md", "docs/conf.py"],
        ["site/brand/art-launch.jpg", "site/home/index.html"],
        ["docs/docs/ci.md", "site/brand/art-launch.jpg", "CHANGELOG.md"],
    ],
)
def test_a_change_the_gate_does_not_read_keeps_its_result(gate_reuse, paths):
    assert all(gate_reuse.unread_by_gate(path) for path in paths)


@pytest.mark.parametrize(
    "path",
    [
        "src/andrey/search/ges.py",
        "qa/quality_gate.py",
        "qa/README.md",  # a Markdown file is unread only at the top level
        "benchmarks/published/0.9.0/summary.json",  # reference parity reads the summaries
        "benchmarks/README.md",
        "tests/unit/test_release_gates.py",
        "scripts/gate_reuse.py",
        "pyproject.toml",
        "uv.lock",
        ".github/workflows/quality-gate.yml",
        "docs",  # a file named like the directory
        "documents/guide.md",
        "sitemap.xml",
        "CHANGELOG.txt",
    ],
)
def test_a_change_the_gate_may_read_runs_it(gate_reuse, path):
    assert not gate_reuse.unread_by_gate(path)
    assert not all(gate_reuse.unread_by_gate(p) for p in ["docs/docs/ci.md", path])  # mixed


def _ancestor_runs(*runs: tuple[str, str]) -> list[dict]:
    return [
        {"id": i, "head_sha": sha, "conclusion": conclusion, "html_url": f"https://example/{i}"}
        for i, (sha, conclusion) in enumerate(runs, start=1)
    ]


def test_the_first_passing_ancestor_with_only_unread_changes_is_reused(gate_reuse):
    changes = {
        "new": ["docs/docs/ci.md"],  # newest run, but it failed
        "src": ["docs/docs/ci.md", "src/andrey/search/ges.py"],  # mixed
        "docs": ["docs/docs/ci.md", "CHANGELOG.md"],
        "older": [],
    }
    runs = _ancestor_runs(
        ("new", "failure"),
        ("gone", "success"),
        ("src", "success"),
        ("docs", "success"),
        ("older", "success"),
    )
    found = gate_reuse.inert_run(runs, changes.get)  # "gone" is not an ancestor: None
    assert found is not None
    run, paths = found
    assert run["head_sha"] == "docs"
    assert paths == ["docs/docs/ci.md", "CHANGELOG.md"]
    assert gate_reuse.inert_run(runs[:3], changes.get) is None
    assert gate_reuse.inert_run([{"conclusion": "success"}], changes.get) is None
    assert gate_reuse.inert_run([], changes.get) is None


def _git_history(ancestors: dict[str, str]):
    """A fake ``_output`` for ``git``: ``ancestors`` maps each ancestor of HEAD to its diff."""

    def output(command: list[str]) -> str:
        assert command[0] == "git"
        sha = command[command.index("HEAD") - 1]
        if command[1] == "merge-base":
            if sha not in ancestors:
                raise subprocess.CalledProcessError(1 if sha == "other" else 128, command)
            return ""
        assert command[1:4] == ["diff", "--name-only", "--no-renames"] and "-z" in command
        return ancestors[sha]

    return output


def _ancestor_lookup(gate_reuse, monkeypatch, tmp_path, ancestors, runs, failure=None):
    git = _git_history(ancestors)

    def output(command: list[str]) -> str:
        if command[0] == "git":
            return git(command)
        if failure:
            raise failure
        assert command[:2] == ["gh", "api"] and "quality-gate.yml/runs" in command[2]
        return json.dumps({"workflow_runs": runs})

    files = {name: tmp_path / name for name in ("output", "summary")}
    monkeypatch.setenv("GITHUB_OUTPUT", str(files["output"]))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(files["summary"]))
    monkeypatch.setenv("GITHUB_REPOSITORY", "example/andrey")
    monkeypatch.setattr(gate_reuse, "_output", output)
    assert gate_reuse.main(["--ancestor"]) == 0
    read = {name: path.read_text() if path.exists() else "" for name, path in files.items()}
    for path in files.values():
        path.unlink(missing_ok=True)
    return read


def test_an_ancestor_run_is_reused_when_only_unread_paths_changed_since(
    gate_reuse, monkeypatch, tmp_path
):
    runs = _ancestor_runs(("other", "success"), ("src", "success"), ("docs", "success"))
    history = {"src": "src/andrey/pc.py\0docs/a.md", "docs": "docs/a.md\0CHANGELOG.md\0"}
    read = _ancestor_lookup(gate_reuse, monkeypatch, tmp_path, history, runs)
    assert read["output"].splitlines() == ["reuse=true", "run=https://example/3", "commit=docs"]
    assert "https://example/3" in read["summary"] and "`docs`" in read["summary"]
    assert "`CHANGELOG.md`" in read["summary"]
    unchanged = _ancestor_lookup(gate_reuse, monkeypatch, tmp_path, {"docs": ""}, runs)
    assert unchanged["output"].startswith("reuse=true")  # the same tree


def test_an_ancestor_lookup_runs_the_gate_unless_every_changed_path_is_unread(
    gate_reuse, monkeypatch, tmp_path
):
    runs = _ancestor_runs(("a", "success"), ("b", "success"), ("c", "failure"))
    history = {"a": "docs/a.md\0qa/quality_baselines.json", "b": "src/andrey/__init__.py", "c": ""}
    read = _ancestor_lookup(gate_reuse, monkeypatch, tmp_path, history, runs)
    assert read == {"output": "reuse=false\n", "summary": ""}


@pytest.mark.parametrize(
    "failure", [subprocess.CalledProcessError(1, "gh"), OSError("no gh"), ValueError("bad json")]
)
def test_a_failed_ancestor_lookup_runs_the_gate(gate_reuse, monkeypatch, tmp_path, failure):
    runs = _ancestor_runs(("docs", "success"))
    read = _ancestor_lookup(gate_reuse, monkeypatch, tmp_path, {"docs": ""}, runs, failure)
    assert read == {"output": "reuse=false\n", "summary": ""}


def test_the_gate_job_runs_no_step_after_a_reused_run():
    workflow = _workflow("quality-gate.yml")
    assert list(workflow["jobs"]) == ["quality-gate"]
    steps = workflow["jobs"]["quality-gate"]["steps"]
    assert steps[0]["with"]["fetch-depth"] == "0"  # the lookup needs the earlier commits
    first = next(i for i, step in enumerate(steps) if step.get("id") == "reuse")
    lookup = steps[first]
    assert lookup["continue-on-error"] == "true"  # an empty output runs the gate
    assert "scripts/gate_reuse.py --ancestor" in lookup["run"]
    for step in steps[first + 1 :]:
        assert step["if"] == "steps.reuse.outputs.reuse != 'true'", step
    assert "qa.quality_gate" in steps[-1]["run"]
    # A called workflow gets at most what its caller grants.
    asked = workflow["permissions"]
    granted = _workflow("publish.yml")["jobs"]["quality-gate"]["permissions"]
    assert asked["actions"] == "read" and asked.items() <= granted.items()


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
        assert job["environment"] == "production-site"  # waits for the owner's approval
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


def _paired_row(interval: list, **changes) -> dict:
    """A row of a ``paired bootstrap`` summary: 20 paired datasets, the reference fewer on 6."""
    row = {
        "d": 10,
        "paired": 20,
        "andrey_fewer_errors": 1,
        "comparator_fewer_errors": 6,
        "andrey_shd_mean": 2.5,
        "comparator_shd_mean": 2.0,
        "mean_D": 0.5,
        "bootstrap_95": interval,
        "above_zero": interval[0] > 0,
    }
    return {**row, **changes}


def _paired(*rows: dict) -> list:
    from qa import reference_parity

    summary = {**_summary(*rows), "parity_rule": reference_parity.PAIRED_BOOTSTRAP}
    return reference_parity.paired_rows("s", summary)


def _paired_failures(*rows: dict) -> list[str]:
    from qa import reference_parity

    return reference_parity.check(_paired(*rows), [])[0]


def test_parity_fails_a_paired_row_whose_interval_lies_above_zero():
    [failure] = _paired_failures(_paired_row([0.05, 0.95]))
    assert "mean D +0.50" in failure and "[+0.05, +0.95]" in failure and "20 paired" in failure


def test_parity_passes_a_near_tie_that_the_count_rule_would_fail():
    # The reference has fewer errors on 6 datasets and Andrey on 1, but the interval reaches 0.
    assert _failures(_paired_row([-0.05, 1.05])) != []
    assert _paired_failures(_paired_row([-0.05, 1.05])) == []
    assert _paired_failures(_paired_row([0.0, 1.05])) == []


def test_parity_reports_a_row_under_ten_pairs_without_gating_it():
    from qa import reference_parity

    # Five pairs, the reference fewer on 2 and Andrey on 0, and no interval: reported, not judged.
    row = {"d": 10, "paired": 5, "andrey_fewer_errors": 0, "comparator_fewer_errors": 2}
    [parsed] = _paired(row)
    assert parsed.reported and not parsed.failed
    assert "reported, not gated" in parsed.numbers()
    assert _paired_failures(row) == []
    # A row with no paired dataset has no result and is skipped.
    assert _paired({**row, "paired": 0}) == []
    rows = (_paired_row([0.05, 0.95], paired=reference_parity.MIN_PAIRS - 1),)
    assert _paired_failures(*rows) == []


def test_parity_judges_a_randomized_test_by_its_draws():
    multi = {
        "draws": 4,
        "paired": 20,
        "andrey_shd_mean": 1.0,
        "comparator_shd_mean": 1.2,
        "mean_D": -0.2,
        "bootstrap_95": [-0.5, 0.1],
        "above_zero": False,
    }
    # One draw: the interval lies above 0; four draws per dataset: it reaches below 0.
    row = _paired_row([0.05, 0.95], multi_draw=multi)
    [parsed] = _paired(row)
    assert (parsed.draws, parsed.mean_d, parsed.interval) == (4, -0.2, (-0.5, 0.1))
    assert _paired_failures(row) == []
    above = {**multi, "bootstrap_95": [0.1, 0.4], "above_zero": True}
    [failure] = _paired_failures(_paired_row([-0.5, 0.95], multi_draw=above))
    assert "the mean of 4 draws each" in failure


def _replication(interval: list, **changes) -> dict:
    """A rerun on fresh seeds: 10 paired datasets, with the fields of a judged row."""
    rerun = {
        "seeds": "20-29",
        "paired": 10,
        "andrey_shd_mean": 2.0,
        "comparator_shd_mean": 2.1,
        "mean_D": -0.1,
        "bootstrap_95": interval,
        "above_zero": interval[0] > 0,
    }
    return {**rerun, **changes}


def test_parity_judges_a_row_by_its_replication_on_fresh_seeds():
    # The run was worse; the rerun's interval reaches 0, so the row does not fail ...
    row = _paired_row([0.05, 0.95], replication=_replication([-0.3, 0.0]))
    [parsed] = _paired(row)
    assert (parsed.replicated, parsed.pairs, parsed.mean_d) == (True, 10, -0.1)
    assert _paired_failures(row) == []
    # ... and when the rerun is also worse, it fails with the rerun's numbers.
    row = _paired_row([0.05, 0.95], replication=_replication([0.2, 0.9], mean_D=0.5))
    [failure] = _paired_failures(row)
    assert failure.startswith("FAIL") and "replication on fresh seeds" in failure
    assert "[+0.20, +0.90]" in failure and "10 paired" in failure
    # A replication is the whole verdict: a row that was not worse can fail by it as well.
    assert _paired_failures(
        _paired_row([-0.5, 0.95], replication=_replication([0.2, 0.9], mean_D=0.5))
    )


def test_parity_reports_a_replication_under_ten_pairs_without_gating_it():
    from qa import reference_parity

    short = reference_parity.MIN_PAIRS - 1
    row = _paired_row([0.05, 0.95], replication=_replication([0.2, 0.9], paired=short))
    [parsed] = _paired(row)
    assert parsed.reported and parsed.replicated and not parsed.failed
    assert f"{short} paired datasets, under" in parsed.numbers()
    assert _paired_failures(row) == []


@pytest.mark.parametrize(
    "replication",
    [
        _replication([-0.3, 0.0], seeds=""),
        _replication([-0.3, 0.0], above_zero=True),
        _replication([-0.3, 0.0], paired=None),
        {k: v for k, v in _replication([-0.3, 0.0]).items() if k != "mean_D"},
        _replication([-0.3, 0.0], paired=3, andrey_shd_mean=None),
    ],
)
def test_parity_rejects_a_replication_it_cannot_judge(replication):
    with pytest.raises(ValueError):
        _paired(_paired_row([0.05, 0.95], replication=replication))


def test_parity_rejects_a_replication_without_the_paired_bootstrap_rule():
    from qa import reference_parity

    row = {
        "d": 10,
        "andrey_shd": 1.0,
        "comparator_shd": 1.0,
        "replication": _replication([-0.3, 0.0]),
    }
    with pytest.raises(ValueError):
        reference_parity.paired_rows("s", _summary(row))


def test_parity_reports_a_row_out_of_claim_without_gating_it():
    from qa import reference_parity

    # The interval lies above 0: in claim the row fails, marked claim "out" it is reported.
    worse = _paired_row([0.05, 0.95])
    assert _paired_failures(worse) != []
    [parsed] = _paired({**worse, "claim": "out"})
    assert parsed.reported and parsed.out_of_claim and not parsed.failed
    assert "out of the method's claim: reported, not gated" in parsed.numbers()
    assert "mean D +0.50" in parsed.numbers()
    assert _paired_failures({**worse, "claim": "out"}) == []
    # The mark holds for the other kinds of row: a replicated one and one under ten pairs.
    rerun = _replication([0.2, 0.9], mean_D=0.5)
    assert _paired_failures({**worse, "claim": "out", "replication": rerun}) == []
    assert _paired_failures({**worse, "replication": rerun}) != []
    # Only the marked row is exempt: its neighbor in claim still fails.
    rows = [{**worse, "claim": "out"}, {**worse, "d": 20}]
    assert len(_paired_failures(*rows)) == 1
    # A row marked "in" is judged. The mark may sit on the entry, and a row's own mark overrides it.
    assert len(_paired_failures({**worse, "claim": "in"})) == 1
    entry = {
        **_summary(worse, {**worse, "d": 20, "claim": "in"}),
        "parity_rule": "paired bootstrap",
    }
    entry["methods"][0]["claim"] = "out"
    failures, _ = reference_parity.check(reference_parity.paired_rows("s", entry), [])
    assert len(failures) == 1 and "d=20" in failures[0]


def test_parity_rejects_a_claim_it_does_not_know():
    for claim in ("outside", True, ""):
        with pytest.raises(ValueError, match="claim"):
            _paired({**_paired_row([0.05, 0.95]), "claim": claim})


def test_parity_keeps_the_count_rule_for_a_summary_without_a_parity_rule():
    # The same row: the interval reaches 0, but the reference has fewer errors on more datasets.
    row = _paired_row([-0.05, 1.05])
    assert _paired_failures(row) == []
    [failure] = _failures(row)
    assert "fewer errors on 6 series, Andrey on 1" in failure


@pytest.mark.parametrize(
    "row",
    [
        {k: v for k, v in _paired_row([0.05, 0.95]).items() if k != "bootstrap_95"},
        _paired_row([0.05, 0.95], above_zero=False),
        {k: v for k, v in _paired_row([0.05, 0.95]).items() if k != "paired"},
        _paired_row([-0.5, 0.95], multi_draw={"draws": 4, "paired": 20, "mean_D": 0.1}),
    ],
)
def test_parity_rejects_a_paired_row_it_cannot_judge(row):
    with pytest.raises(ValueError):
        _paired(row)


def test_parity_rejects_an_unknown_parity_rule():
    from qa import reference_parity

    with pytest.raises(ValueError):
        reference_parity.paired_rows("s", {**_summary(), "parity_rule": "bootstrap"})


def _column_row(andrey: list, reference: list, **changes) -> dict:
    """A paired row whose interval reaches 0, with Andrey's column-order mode beside it."""
    row = {
        **_paired_row([-0.05, 1.05]),
        "reference_order_dependent": True,
        "andrey_column_shd_by_seed": andrey,
        "comparator_shd_by_seed": reference,
    }
    return {**row, **changes}


def test_parity_prints_the_column_order_arm_as_context_only():
    # Column order: the reference has fewer errors on seeds 0 and 2, Andrey on seed 3; seed 4 has no
    # result on one side and is not a series.
    row = _column_row([3, 1, 2, 0, None], [2, 1, 1, 1, 5])
    [parsed] = _paired(row)
    assert parsed.column == (2, 1) and not parsed.failed
    [failure] = _paired_failures({**row, "bootstrap_95": [0.05, 0.95], "above_zero": True})
    assert "column-order mode (context, not judged)" in failure
    assert "fewer errors on 2 series, Andrey on 1" in failure


@pytest.mark.parametrize(
    "row",
    [
        _column_row([1, 2], [1, 2], reference_order_dependent=False),
        {k: v for k, v in _column_row([1], [1]).items() if k != "reference_order_dependent"},
        {k: v for k, v in _column_row([1], [1]).items() if k != "comparator_shd_by_seed"},
        _column_row([1, 2], [1]),
    ],
)
def test_parity_rejects_a_malformed_column_order_arm(row):
    with pytest.raises(ValueError):
        _paired(row)


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
    entry = _entry(row={"regime": "additive", "d": 10, "n": 500})
    failures = reference_parity.check(
        rows, [{**entry, "andrey_errors": 1, "comparator_errors": 0}]
    )[0]
    assert len(failures) == 1 and "regime=nonmonotone" in failures[0]
    for method in summary["methods"]:
        del method["regime"]
    with pytest.raises(ValueError):
        reference_parity.paired_rows("s", summary)
