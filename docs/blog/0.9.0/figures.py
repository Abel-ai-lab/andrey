"""The 0.9.0 post's speed table, saved as HTML beside this script for inclusion in the post.

- ``speedup.html``: each sped-up method against Andrey 0.8.0, grouped by family: the range over
  the measured sizes of 0.8.0's median fit time over 0.9.0's, with one worker, and whether every
  output matched.

The numbers come from the 0.9.0 release card's summary (``SUMMARY``), its ``speed`` lane: one row
per method, regime, size, and worker count. The table reuses the 0.8.0 post's ``table``. The script
runs with ``uv run python docs/blog/0.9.0/figures.py [summary.json]``.
"""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

from andrey.spec import list_specs

HERE = Path(__file__).parent
SUMMARY = HERE.parents[2] / "benchmarks" / "published" / "0.9.0" / "summary.json"
_080 = runpy.run_path(str(HERE.parent / "0.8.0" / "figures.py"))
table, floored = _080["table"], _080["floored"]
FAMILIES = {
    "constraint": "Constraint",
    "score": "Score",
    "permutation": "Permutation",
    "lingam": "LiNGAM",
    "pairwise": "Pairwise",
    "temporal": "Temporal",
    "latent": "Latent",
}
# Each method as the post names it.
NAMES = {
    "pc": "PC", "fci": "FCI", "gfci": "GFCI", "cdnod": "CD-NOD", "ges": "GES", "xges": "XGES",
    "exact_search": "Exact search", "boss": "BOSS", "grasp": "GRaSP",
    "direct_lingam": "DirectLiNGAM", "ica_lingam": "ICA-LiNGAM", "anm": "ANM", "pnl": "PNL",
    "lpcmci": "LPCMCI",
}  # fmt: skip


def speedup_table(summary: dict) -> str:
    """One row per method with one worker: the range of its speedups and whether outputs matched."""
    family = {s.name: s.family for s in list_specs()}
    speeds: dict[str, list[float]] = {}
    same: dict[str, bool] = {}
    for row in summary["speed"]:
        if row.get("workers", 1) != 1 or not row.get("speedup"):
            continue
        speeds.setdefault(row["method"], []).append(row["speedup"])
        same[row["method"]] = same.get(row["method"], True) and bool(row["same_output"])
    groups = tuple(
        (
            label,
            tuple(
                (
                    NAMES.get(m, m),
                    "",
                    (
                        f"{floored(min(speeds[m]))}-{max(speeds[m]):.2g}x",
                        "Yes" if same[m] else "No",
                    ),
                )
                for m in speeds
                if family.get(m) == key
            ),
        )
        for key, label in FAMILIES.items()
    )
    return table(
        "Each method against Andrey 0.8.0",
        ("Faster by", "Same output"),
        tuple((label, rows) for label, rows in groups if rows),
        rank=lambda cell: None,
        note="The range over sizes of 0.8.0's median fit time over 0.9.0's, with one worker, on "
        "the same datasets and machine.",
    )


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else SUMMARY
    summary = json.loads(path.read_text(encoding="utf-8"))
    (HERE / "speedup.html").write_text(speedup_table(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
