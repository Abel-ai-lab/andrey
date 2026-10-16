"""Run the example notebooks and store their outputs, which the site renders without running them.

``uv run --group docs python scripts/execute_notebooks.py`` runs every notebook in ``examples/`` but
``speed_on_your_machine.ipynb``, whose timings belong to the reader's machine, and writes its
outputs into the file. With ``--check`` it writes nothing and exits 1 when a notebook fails or an
output differs from the stored one.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
SKIPPED = {"speed_on_your_machine.ipynb"}
TIMEOUT = 600
ADDRESS = re.compile(r"0x[0-9a-f]+")


def notebooks() -> list[Path]:
    """Return the notebooks whose outputs are stored."""
    return sorted(p for p in EXAMPLES.glob("*.ipynb") if p.name not in SKIPPED)


def execute(path: Path) -> nbformat.NotebookNode:
    """Run the notebook at ``path`` in an empty directory and return it with its outputs."""
    notebook = nbformat.read(path, as_version=4)
    with tempfile.TemporaryDirectory() as cwd:
        NotebookClient(
            notebook,
            timeout=TIMEOUT,
            kernel_name="python3",
            record_timing=False,
            resources={"metadata": {"path": cwd}},
        ).execute()
    return notebook


def outputs(notebook: nbformat.NotebookNode) -> list[str]:
    """Return each code cell's outputs as text lines, for comparing two runs.

    An object's memory address changes from run to run, so it reads ``0x...``.
    """
    lines = []
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            kind = output.get("name") or output["output_type"]
            parts = {kind: output["text"]} if "text" in output else output.get("data", {})
            for mime, text in sorted(parts.items()):
                text = text if isinstance(text, str) else json.dumps(text, sort_keys=True)
                for line in ADDRESS.sub("0x...", text).splitlines():
                    lines.append(f"cell {index} {mime}: {line}")
    return lines


def main() -> int:
    """Store the outputs, or with ``--check`` compare them; return the exit status."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="compare, do not write")
    check = parser.parse_args().check
    stale = []
    for path in notebooks():
        fresh = execute(path)
        if not check:
            nbformat.write(fresh, path)
            print(f"wrote {path.relative_to(ROOT)}")
            continue
        stored = outputs(nbformat.read(path, as_version=4))
        diff = list(difflib.unified_diff(stored, outputs(fresh), "stored", "fresh", lineterm=""))
        if diff:
            stale.append(path.name)
            print("\n".join(diff[:80]))
    if stale:
        print(f"Outputs differ in {', '.join(stale)}; run scripts/execute_notebooks.py.")
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
