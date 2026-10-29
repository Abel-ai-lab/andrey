"""Generate the boundary warning's table from the newest published boundary lane.

Reads ``boundary.ladders`` of the newest ``benchmarks/published/<release>/summary.json`` that has
them and writes ``src/andrey/api/_boundary_table.py``: per method, one entry per ladder with its
setting (label, rows or rows per variable, mean degree), the largest size its fits finished within
the caps, and the stop that ended it. The size is counted as the method's data see it: the
columns, the observed ones where the generator hides some nodes, or the rows for the pair methods,
whose ladders step rows. ``andrey.api._boundary`` picks the warning's threshold from these entries. A summary whose lane
carries ``carried_over_from`` holds an earlier release's lane unchanged, so the table keeps the
release that measured it.

    uv run python scripts/boundary_table.py

``tests/unit/test_boundary.py`` fails until the table is regenerated after a new summary.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ROOT / "benchmarks" / "published"
TABLE = ROOT / "src" / "andrey" / "api" / "_boundary_table.py"

#: Generators whose ``d`` counts hidden nodes: the hidden count of a rung, and the text of the
#: summary's generator entry that states the same rule, so a changed generator stops the script.
HIDDEN = {
    "latent_lingam_er": (lambda rung: max(1, round(0.2 * rung["d"])), "max(1, round(0.2 * d))"),
    # benchmarks/andrey_bench/temporal_models.py: hidden_count
    "lpcmci_paper": (
        lambda rung: rung["d"] - math.ceil((1 - 0.3) * rung["d"]),
        "30% of the series unobserved",
    ),
    "latent_measurement": (lambda rung: rung["latents"], "3 observed indicators each"),
}
HIDDEN_WORDS = ("hidden", "latent", "unobserved")
PER_VARIABLE = re.compile(r"deg\d+-n(\d+)d")


def newest() -> Path:
    """Return the summary of the newest release that measured a boundary lane."""
    releases = []
    for path in PUBLISHED.glob("*/summary.json"):
        version = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", path.parent.name)
        lane = json.loads(path.read_text()).get("boundary") or {}
        if version and "ladders" in lane and "carried_over_from" not in lane:
            releases.append((tuple(map(int, version.groups())), path))
    if not releases:
        raise SystemExit(f"no summary under {PUBLISHED} has boundary.ladders")
    return max(releases)[1]


def caps(ladders: list[dict]) -> tuple[str, str, int]:
    """Return the time cap, memory cap, and cores every ladder shares, as the warning names them."""
    shared = {(lad["cap_s"], lad["cap_mem_mb"], lad["cores"]) for lad in ladders}
    if len(shared) != 1:
        raise SystemExit(f"the ladders do not share one set of caps: {sorted(shared)}")
    [(seconds, megabytes, cores)] = shared
    hours, gigabytes = seconds / 3600, megabytes / 1000
    if not (hours.is_integer() and gigabytes.is_integer()):
        raise SystemExit(f"caps of {seconds} s and {megabytes} MB are not whole hours and GB")
    return f"{hours:.0f} h", f"{gigabytes:.0f} GB", int(cores)


def hidden_rule(name: str, generator: dict):
    """Return the hidden count of a rung for generator ``name``, ``None`` when it hides none."""
    text = f"{generator['call']} {generator['description']}"
    if name in HIDDEN:
        count, rule = HIDDEN[name]
        if rule not in text:
            raise SystemExit(f"generator {name} no longer states {rule!r}: {text}")
        return count
    if any(word in text for word in HIDDEN_WORDS):
        raise SystemExit(f"generator {name} may hide nodes; add its rule to HIDDEN: {text}")
    return None


def entry(ladder: dict, generators: dict) -> tuple:
    """Return one ladder as a table entry.

    The entry is ``(label, rows, per_variable, mean_degree, unit, last_through, last_through_d,
    stop)``.
    """
    stop = ladder["end"]["stop"]
    if ladder["axis"] == "n":
        return (ladder["label"], None, False, None, "rows", ladder["last_through_n"], None, stop)
    rungs = {rung["d"]: rung for rung in ladder["rungs"]}
    d = ladder["last_through_d"]
    rung = rungs[d] if d is not None else ladder["rungs"][0]
    per_variable = PER_VARIABLE.fullmatch(ladder["regime"])
    rows = int(per_variable.group(1)) if per_variable else rung["n"]
    count = hidden_rule(ladder["generator"], generators[ladder["generator"]])
    observed = None if d is None else d - (count(rung) if count else 0)
    degree = ladder["mean_degree"]
    return (ladder["label"], rows, bool(per_variable), degree, "variables", observed, d, stop)


def table(summary: dict) -> dict:
    """Return the table's constants: the release, the caps, and the ladders by method."""
    lane = summary["boundary"]
    time_cap, memory_cap, cores = caps(lane["ladders"])
    ladders: dict[str, list[tuple]] = {}
    for ladder in lane["ladders"]:
        ladders.setdefault(ladder["method"], []).append(entry(ladder, lane["generators"]))
    return {
        "RELEASE": summary["release"],
        "TIME_CAP": time_cap,
        "MEMORY_CAP": memory_cap,
        "CORES": cores,
        "LADDERS": {method: tuple(rows) for method, rows in ladders.items()},
    }


def source(constants: dict, summary_path: Path) -> str:
    """Return the table module's source text, before formatting."""
    lines = [
        f'"""The boundary lane of the {constants["RELEASE"]} benchmark, one entry per ladder.',
        "",
        "Generated by ``scripts/boundary_table.py`` from",
        f"``{summary_path.relative_to(ROOT).as_posix()}``; do not edit. An entry's fields are",
        "named by ``andrey.api._boundary.Ladder``.",
        '"""',
        "",
    ]
    for name in ("RELEASE", "TIME_CAP", "MEMORY_CAP", "CORES"):
        lines.append(f"{name} = {constants[name]!r}")
    lines += ["", "LADDERS = {"]
    for method, rows in constants["LADDERS"].items():
        lines.append(f"    {method!r}: (")
        lines += [f"        {row!r}," for row in rows]
        lines.append("    ),")
    lines.append("}")
    return "\n".join(lines) + "\n"


def main() -> int:
    """Write the table from the newest summary and format it."""
    path = newest()
    TABLE.write_text(source(table(json.loads(path.read_text())), path))
    subprocess.run([sys.executable, "-m", "ruff", "format", "-q", str(TABLE)], check=True)
    print(f"{TABLE.relative_to(ROOT)} <- {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
