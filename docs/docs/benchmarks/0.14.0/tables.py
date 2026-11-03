"""The 0.14.0 release card's tables, saved as HTML beside this script.

``gnies.html`` and ``ut-igsp.html``: for each setting in the method's claim, the mean errors of
Andrey, the authors' code, and a baseline, for the structural Hamming distance (SHD) and the target
error, with the mean paired difference, its 95% interval, and the verdict. The settings are grouped
by the kind of experiment, with the rows per environment beside them. A comparison is better only
where a rerun on fresh seeds agreed; one lower or higher on the first seeds alone is a tie, with a
note.

The numbers come from the published lanes ``gnies.json`` and ``ut-igsp.json``. The script runs with
``uv run python docs/docs/benchmarks/0.14.0/tables.py [gnies.json ut-igsp.json]``.
"""

from __future__ import annotations

import json
import sys
from html import escape
from pathlib import Path

HERE = Path(__file__).parent
PUBLISHED = HERE.parents[3] / "benchmarks" / "published" / "0.14.0"

# GnIES's settings by regime: the kind of experiment each regime varies.
GNIES_GROUPS = {
    "A": "3 experiments, 1 target each",
    "B": "Denser graph, mean degree 4",
    "C": "3 targets per experiment",
    "D": "Smaller change, variance x3 to x4",
    "E": "No observational environment",
    "F": "9 experiments",
    "G": "1,000 observed rows",
    "H": "Fewer experiments",
    "S": "Weights of either sign",
}
# UT-IGSP's arms by the intervention each makes.
UT_GROUPS = {
    ("G1", "shift"): "Mean shift of 1",
    ("G2", "perfect"): "Perfect intervention",
    ("G3", "shift2"): "Mean shift of 2",
    ("G3", "perfect_code"): "Perfect, variance 0.01",
    ("G4", "variance"): "Noise doubled",
    ("G5", "shift"): "Mean shift of 1, no targets given",
    ("G6", "weight"): "Parent weights halved",
}


def interval(mean: float, low: float, high: float) -> str:
    """A mean difference with its 95% interval."""
    return f"{mean:+.2f}<small>[{low:+.2f}, {high:+.2f}]</small>"


def verdict(first: str, rerun: str | None) -> str:
    """Better or worse only where the rerun agreed; a lone first-seed difference is noted."""
    if first == "tie":
        return ""
    if first == rerun:
        return f'<small class="ct-verdict ct-{first}">{first}</small>'
    return f"<small>{'lower' if first == 'better' else 'higher'} on the first seeds only</small>"


def table(
    title: str, reference: str, baselines: tuple[str, str], groups: list, caption: str
) -> str:
    """One table: SHD and target error side by side, a row per setting under its group's label."""
    sub = (
        '<th scope="col" class="ct-ours ct-start">Andrey</th>'
        f'<th scope="col">{escape(reference)}</th><th scope="col">{{base}}</th>'
        '<th scope="col" class="ct-ours">Difference</th>'
    )
    head = (
        '<tr><td rowspan="2"></td>'
        '<th scope="colgroup" colspan="4" class="ct-start">SHD</th>'
        '<th scope="colgroup" colspan="4" class="ct-start">Target error</th></tr>'
        f"<tr>{sub.format(base=baselines[0])}{sub.format(base=baselines[1])}</tr>"
    )
    body = ""
    for label, rows in groups:
        body += f'<tr class="ct-section"><th scope="rowgroup" colspan="9">{escape(label)}</th></tr>'
        for row_label, cells in rows:
            tds = ""
            for andrey, ref, base, diff, note in cells:
                tds += (
                    f'<td class="ct-ours ct-start">{andrey:.2f}</td><td>{ref:.2f}</td>'
                    f'<td>{base:.2f}</td><td class="ct-ours">{diff}{note}</td>'
                )
            body += f'<tr><th scope="row">{row_label}</th>{tds}</tr>'
    return (
        f'<figure class="ct"><p class="ct-title">{escape(title)}</p><div class="ct-scroll">'
        f'<table class="ct-table ct-nested"><thead>{head}</thead><tbody>{body}</tbody></table>'
        f"</div><figcaption>{caption}</figcaption></figure>\n"
    )


def gnies(lane: dict) -> str:
    """GnIES against gnies 0.3.3, in its claim."""
    grouped: dict[str, list] = {}
    for c in lane["cells"]:
        if c["claim"] != "in":
            continue
        regime = c["cell"][0]
        if regime == "H":
            row = f"{len(c['sizes']) - 1} experiment{'s' if len(c['sizes']) > 2 else ''}"
        elif regime == "G":
            row = f"experiments of {min(c['sizes']):,} rows"
        else:
            row = f"{c['n']:,} rows each"
        rerun = c.get("replication") or {}
        cells = []
        for error, base in (
            ("shd", c["empty_shd_mean"]),
            ("target_error", c["empty_target_error_mean"]),
        ):
            e = c[error]
            low, high = e["bootstrap_95"]
            cells.append(
                (
                    e["andrey_mean"],
                    e["comparator_mean"],
                    base,
                    interval(e["mean_D"], low, high),
                    verdict(e["verdict"], (rerun.get(error) or {}).get("verdict")),
                )
            )
        grouped.setdefault(regime, []).append((row, cells))
    groups = [(GNIES_GROUPS[r], rows) for r, rows in grouped.items()]
    return table(
        "GnIES against gnies 0.3.3, targets unknown",
        "gnies",
        ("Empty graph", "No targets"),
        groups,
        "Means over 20 datasets, seeds 0 to 19, on 10 variables; each experiment multiplies its "
        "targets' noise variance by 5 to 10 unless noted. The difference is Andrey's mean error "
        "minus gnies's, with its 95% interval, marked better only where seeds 20 to 39 agreed; "
        "unmarked is a tie. No targets is the target error of naming none: the number of true "
        "targets.",
    )


def ut_igsp(lane: dict) -> str:
    """UT-IGSP against the authors' library, in its claim, at Andrey's defaults."""
    grouped: dict[tuple, list] = {}
    for c in lane["in_claim"]["cells"].values():
        off = f"{c['l']} off-target{'' if c['l'] == 1 else 's'}"
        row = f"{off}<small>{c['n']:,} rows each</small>"
        cells = []
        for error, base in (("shd", c["empty_shd_mean"]), ("target_error", c["target_floor_mean"])):
            e = c["J2"][error]
            low, high = e["ci95"]
            first = {"below 0": "better", "above 0": "worse"}.get(e["side"], "tie")
            cells.append(
                (
                    e["andrey_mean"],
                    e["comparator_mean"],
                    base,
                    interval(e["D"], low, high),
                    verdict(first, e["verdict"]),
                )
            )
        grouped.setdefault((c["arm"], c["kind"]), []).append(((c["l"], c["n"]), row, cells))
    groups = [
        (UT_GROUPS[key], [(row, cells) for _, row, cells in sorted(rows)])
        for key, rows in grouped.items()
    ]
    return table(
        "UT-IGSP against the authors' library, at Andrey's defaults",
        "Library",
        ("Empty graph", "Given targets"),
        groups,
        "Means over 20 datasets, seeds 0 to 19, 5 draws each, on 20 variables with an "
        "observational environment and 5 experiments. Each experiment intervenes on one target it "
        "names and on off-targets it does not. The difference is Andrey's mean error minus the "
        "library's, with its 95% interval, marked better only where a rerun agreed; unmarked is "
        "a tie. "
        "Given targets is the target error of the named targets alone.",
    )


if __name__ == "__main__":
    paths = [Path(p) for p in sys.argv[1:]] or [
        PUBLISHED / "gnies.json",
        PUBLISHED / "ut-igsp.json",
    ]
    (HERE / "gnies.html").write_text(gnies(json.loads(paths[0].read_text())))
    (HERE / "ut-igsp.html").write_text(ut_igsp(json.loads(paths[1].read_text())))
