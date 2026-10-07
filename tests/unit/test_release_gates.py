"""The release scripts flag what they guard against and pass what they should not."""

from __future__ import annotations

import io
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
