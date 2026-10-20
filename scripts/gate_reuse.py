"""Release gate: find a passing quality-gate run on the tree being published.

The quality gate runs on the release pull request into ``main``. A squash merge keeps the tree of
the pull request's head, so a passing run on a commit with the published commit's tree already
checked the published files, and ``publish.yml`` skips its own gate. A run on any other tree does
not count, and neither does a failed run or a lookup that errors: the publish then runs the gate.

Run it in a checkout of the commit to publish, with ``GITHUB_REPOSITORY`` set and ``GH_TOKEN``
allowed to read Actions. It exits 0 and writes ``reuse=true`` or ``reuse=false`` to
``$GITHUB_OUTPUT``:

    uv run --no-project python scripts/gate_reuse.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

WORKFLOW = "quality-gate.yml"
RUNS_PER_LOOKUP = 50


def passing_run(runs: list[dict], tree: str | None) -> dict | None:
    """Return the first run that passed on a commit whose tree is ``tree``, else ``None``.

    A run is the Actions API's workflow run: ``head_commit.tree_id`` is the tree of the commit the
    run started on. A missing tree or a run without one never matches.
    """
    if not tree:
        return None
    for run in runs:
        head = run.get("head_commit") or {}
        if run.get("conclusion") == "success" and head.get("tree_id") == tree:
            return run
    return None


def _output(command: list[str]) -> str:
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()


def lookup(repository: str) -> dict | None:
    """Return the passing gate run on the tree of the checked-out commit, if there is one."""
    tree = _output(["git", "rev-parse", "HEAD^{tree}"])
    path = f"repos/{repository}/actions/workflows/{WORKFLOW}/runs"
    runs = json.loads(_output(["gh", "api", f"{path}?status=success&per_page={RUNS_PER_LOOKUP}"]))
    return passing_run(runs["workflow_runs"], tree)


def main() -> int:
    """Write ``reuse=true`` or ``reuse=false`` to ``$GITHUB_OUTPUT``, and exit 0 either way."""
    try:
        run = lookup(os.environ["GITHUB_REPOSITORY"])
    except Exception as error:  # Reuse only skips work, so any failure to look runs the gate.
        print(f"::warning::Quality-gate lookup failed, so the gate will run: {error}")
        run = None
    if run:
        print(
            f"::notice::Reusing quality-gate run {run.get('html_url')}: same tree as this commit."
        )
    else:
        print("::notice::No passing quality-gate run has this commit's tree; the gate will run.")
    line = f"reuse={'true' if run else 'false'}"
    if output := os.environ.get("GITHUB_OUTPUT"):
        with open(output, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
