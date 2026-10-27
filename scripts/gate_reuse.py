"""Release gate: find a passing quality-gate run that already checked what is about to be checked.

The quality gate runs on the release pull request into ``main``. A squash merge keeps the tree of
the pull request's head, so a passing run on a commit with the published commit's tree already
checked the published files, and ``publish.yml`` skips its own gate. A run on any other tree does
not count, and neither does a failed run or a lookup that errors: the publish then runs the gate.

With ``--ancestor``, the quality-gate workflow also accepts a passing run on an earlier commit of
the checked-out history when every path that changed since is one the gate does not read
(:func:`unread_by_gate`): documentation, the website's assets, and top-level Markdown files. Such a
change cannot alter any result, so the new commit has the passing run's result. A run reused this
way is itself a pass on its own tree, and each commit in a chain of reuses changed only paths the
gate does not read, so the chain's last commit has the result of the run that fitted the methods.
``publish.yml`` keeps the same-tree rule.

Run it in a checkout of the commit to check, with the history of the commits to compare against,
``GITHUB_REPOSITORY`` set, and ``GH_TOKEN`` allowed to read Actions. It exits 0 and writes
``reuse=true`` or ``reuse=false`` to ``$GITHUB_OUTPUT``, with the reused run's ``run`` and
``commit`` when it is true:

    uv run --no-project python scripts/gate_reuse.py [--ancestor]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable, Sequence

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


def _passing_runs(repository: str) -> list[dict]:
    path = f"repos/{repository}/actions/workflows/{WORKFLOW}/runs"
    runs = json.loads(_output(["gh", "api", f"{path}?status=success&per_page={RUNS_PER_LOOKUP}"]))
    return runs["workflow_runs"]


def lookup(repository: str) -> dict | None:
    """Return the passing gate run on the tree of the checked-out commit, if there is one."""
    tree = _output(["git", "rev-parse", "HEAD^{tree}"])
    return passing_run(_passing_runs(repository), tree)


#: Directories whose files the gate does not read.
UNREAD_DIRECTORIES = ("docs/", "site/")


def unread_by_gate(path: str) -> bool:
    """Return whether the quality gate's result cannot depend on the repository file ``path``.

    The gate reads the package (``src/``), the benchmark harness and the published summaries
    (``benchmarks/``), its own files (``qa/``), and the locked environment (``pyproject.toml`` and
    ``uv.lock``). It reads nothing under ``docs/`` or ``site/``, and no top-level Markdown file
    (``CHANGELOG.md``, ``README.md``, ...). Any other path, including a Markdown file in a
    directory, may change a result, so it runs the gate.
    """
    return path.startswith(UNREAD_DIRECTORIES) or ("/" not in path and path.endswith(".md"))


def inert_run(
    runs: list[dict], changed_since: Callable[[str], list[str] | None]
) -> tuple[dict, list[str]] | None:
    """Return the first passing run whose commit is an ancestor of the checked-out commit and
    differs from it only in paths the gate does not read, with those paths, else ``None``.

    ``changed_since(sha)`` returns the paths that differ between commit ``sha`` and the checked-out
    commit, and ``None`` when ``sha`` is not one of its ancestors. A run whose commit left the
    history, as after a force-push, never matches, and neither does a run that did not succeed.
    """
    for run in runs:
        sha = run.get("head_sha")
        if run.get("conclusion") != "success" or not sha:
            continue
        paths = changed_since(sha)
        if paths is not None and all(unread_by_gate(path) for path in paths):
            return run, paths
    return None


def changed_since(sha: str) -> list[str] | None:
    """Return the paths that differ between commit ``sha`` and ``HEAD``, or ``None`` when ``sha``
    is not an ancestor of ``HEAD`` or not in this checkout."""
    try:
        _output(["git", "merge-base", "--is-ancestor", sha, "HEAD"])
    except subprocess.CalledProcessError:
        return None
    # No rename detection: a file moved out of ``src/`` into ``docs/`` lists both paths.
    names = _output(["git", "diff", "--name-only", "--no-renames", "-z", sha, "HEAD"])
    return [name for name in names.split("\0") if name]


def lookup_ancestor(repository: str) -> tuple[dict, list[str]] | None:
    """Return a passing gate run on an ancestor of the checked-out commit, or on its tree, with the
    paths that changed since, when none of them is read by the gate."""
    return inert_run(_passing_runs(repository), changed_since)


def _summary(run: dict, paths: list[str]) -> str:
    shown = "".join(f"- `{path}`\n" for path in paths[:20])
    more = f"- and {len(paths) - 20} more\n" if len(paths) > 20 else ""
    return (
        f"### Quality gate reused\n\n[This run]({run.get('html_url')}) passed on "
        f"`{run['head_sha']}`. {len(paths)} paths changed since, none of them read by the gate:\n\n"
        f"{shown}{more}"
    )


def main(argv: Sequence[str] = ()) -> int:
    """Write ``reuse=true`` or ``reuse=false`` to ``$GITHUB_OUTPUT``, and exit 0 either way."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--ancestor",
        action="store_true",
        help="also reuse a passing run on an ancestor commit when only unread paths changed since",
    )
    args = parser.parse_args(argv)
    run, paths = None, []
    try:
        repository = os.environ["GITHUB_REPOSITORY"]
        if args.ancestor:
            run, paths = lookup_ancestor(repository) or (None, [])
        else:
            run = lookup(repository)
    except Exception as error:  # Reuse only skips work, so any failure to look runs the gate.
        print(f"::warning::Quality-gate lookup failed, so the gate will run: {error}")
        run = None
    lines = [f"reuse={'true' if run else 'false'}"]
    if run and args.ancestor:
        url, sha = run.get("html_url"), run["head_sha"]
        print(
            f"::notice::Reusing quality-gate run {url} on {sha}: {len(paths)} paths changed since."
        )
        lines += [f"run={url}", f"commit={sha}"]
        if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(summary, "a", encoding="utf-8") as handle:
                handle.write(_summary(run, paths))
    elif run:
        print(
            f"::notice::Reusing quality-gate run {run.get('html_url')}: same tree as this commit."
        )
    else:
        print("::notice::No passing quality-gate run has this commit's tree; the gate will run.")
    if output := os.environ.get("GITHUB_OUTPUT"):
        with open(output, "a", encoding="utf-8") as handle:
            handle.write("".join(line + "\n" for line in lines))
    print(lines[0])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
