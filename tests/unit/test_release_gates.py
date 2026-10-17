"""The release scripts flag what they guard against and pass what they should not."""

from __future__ import annotations

import io
import json
import re
import runpy
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
