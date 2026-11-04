"""The 0.14.0 release card's tables, saved as HTML beside this script.

``gnies.html`` and ``ut-igsp.html``: for each setting in the method's claim, the mean errors of
Andrey, the authors' code, and a baseline, for the structural Hamming distance (SHD) and the target
error, with the mean paired difference, its 95% interval, and the verdict. The settings are grouped
by the kind of experiment, with the rows per environment beside them. A comparison is better only
where a rerun on fresh seeds agreed; one lower or higher on the first seeds alone is a tie, with a
note.

``camuv.html``: CAM-UV's table has one error, the directed-edge errors, against two references,
lingam and causal-learn.

``icp.html``: ICP against R's InvariantCausalPrediction on the same fits, the errors in the target's
parents, the familywise error rate (FWER), and the fits that accepted the same sets; then ICP's FWER
over every fit, in its claim and out of it.

The numbers come from the published lanes ``gnies.json``, ``ut-igsp.json``, ``camuv.json``, and
``icp.json``, from ``benchmarks/published/0.14.0`` or from the paths given, in that order. The
script runs with ``uv run python docs/docs/benchmarks/0.14.0/tables.py [paths]``.
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


def signed(value: float) -> str:
    """A signed number to two decimals, with no negative zero."""
    return f"{value + 0.0:+.2f}".replace("-0.00", "+0.00")


def interval(mean: float, low: float, high: float) -> str:
    """A mean difference with its 95% interval."""
    return f"{signed(mean)}<small>[{signed(low)}, {signed(high)}]</small>"


def verdict(first: str, rerun: str | None) -> str:
    """Better or worse only where the rerun agreed; a lone first-seed difference is noted."""
    if first == "tie":
        return ""
    if first == rerun:
        return f'<small class="ct-verdict ct-{first}">{first}</small>'
    return f"<small>{'lower' if first == 'better' else 'higher'}, first seeds only</small>"


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


# CAM-UV's regimes, in the comparison page's words.
CAMUV_GROUPS = {
    "latent_anm_er": "One in five hidden",
    "latent_anm_er_tenth": "One in ten hidden",
    "camuv_paper": "The paper's simulation",
    "anm_er": "Nothing hidden",
}


def camuv(lane: dict) -> str:
    """CAM-UV against lingam and causal-learn, in its claim: directed-edge errors."""
    refs = (("lingam", "lingam"), ("causal_learn", "causal-learn"))
    head = (
        '<tr><td rowspan="2"></td>'
        '<th scope="colgroup" colspan="4" class="ct-start">Directed-edge errors</th>'
        '<th scope="colgroup" colspan="2" class="ct-start">Andrey minus</th></tr><tr>'
        '<th scope="col" class="ct-ours ct-start">Andrey</th><th scope="col">lingam</th>'
        '<th scope="col">causal-learn</th><th scope="col">Empty graph</th>'
        + "".join(
            f'<th scope="col" class="ct-ours{" ct-start" if k == 0 else ""}">{escape(name)}</th>'
            for k, (_, name) in enumerate(refs)
        )
        + "</tr>"
    )
    body = ""
    for m in lane["methods"]:
        if m["claim"] != "in":
            continue
        reruns = {(r["observed"], r["latents"], r["n"]): r for r in m["rows"] if r["replication"]}
        body += (
            f'<tr class="ct-section"><th scope="rowgroup" colspan="7">'
            f"{escape(CAMUV_GROUPS[m['regime']])}</th></tr>"
        )
        for r in m["rows"]:
            if r["replication"]:
                continue
            hidden = f" + {r['latents']} hidden" if r["latents"] else ""
            label = f"{r['observed']} observed{hidden}<small>{r['n']:,} rows</small>"
            rerun = reruns.get((r["observed"], r["latents"], r["n"]))
            tds = (
                f'<td class="ct-ours ct-start">{r["andrey_direct_errors_mean"]:.2f}</td>'
                f"<td>{r['lingam_direct_errors_mean']:.2f}</td>"
                f"<td>{r['causal_learn_direct_errors_mean']:.2f}</td>"
                f"<td>{r['empty_direct_errors_mean']:.2f}</td>"
            )
            for k, (key, _) in enumerate(refs):
                v = r[f"vs_{key}"]
                low, high = v["bootstrap_95_direct_errors"]
                first = v["verdict"]
                again = (
                    (rerun or {}).get(f"vs_{key}", {}).get("verdict") if first != "tie" else first
                )
                tds += (
                    f'<td class="ct-ours{" ct-start" if k == 0 else ""}">'
                    f"{interval(v['mean_D_direct_errors'], low, high)}{verdict(first, again)}</td>"
                )
            body += f'<tr><th scope="row">{label}</th>{tds}</tr>'
    return (
        '<figure class="ct"><p class="ct-title">CAM-UV against lingam and causal-learn</p>'
        '<div class="ct-scroll"><table class="ct-table ct-nested"><thead>'
        f"{head}</thead><tbody>{body}</tbody></table></div><figcaption>"
        "Mean directed-edge errors over 20 datasets, seeds 0 to 19: a missing or extra edge costs "
        "1 and a reversed one 2, against the edges CAM-UV can identify. lingam and causal-learn "
        "search sets of up to 3 variables, as Andrey does; lingam at its default set size of 2 "
        "ties Andrey in every setting too. The difference is Andrey's mean errors minus the "
        "reference's, with its 95% interval, marked better only where seeds 20 to 39 agreed; "
        "unmarked is a tie."
        "</figcaption></figure>\n"
    )


# ICP's cells, in plain words.
ICP_CELLS = {
    "A": "5 to 12 variables",
    "C": "Do-interventions, 6 or 10 variables",
    "D": "16 or 21 variables, at most 3 parents",
    "D40": "40 variables, at most 3 parents",
    "B, target intervened": "Interventions that hit the target",
    "E-hidden": "A hidden parent of the target",
    "E-nonlinear": "A nonlinear parent of the target",
}


def icp(lane: dict) -> str:
    """ICP against R on the same fits, then ICP's FWER over every fit."""
    lanes = {m["lane"]: m for m in lane["methods"]}
    head = (
        '<tr><td rowspan="2"></td>'
        '<th scope="colgroup" colspan="4" class="ct-start">Errors in the parents</th>'
        '<th scope="colgroup" colspan="2" class="ct-start">FWER</th>'
        '<th scope="col" rowspan="2" class="ct-start">Same accepted sets</th></tr><tr>'
        '<th scope="col" class="ct-ours ct-start">Andrey</th><th scope="col">R</th>'
        '<th scope="col">Empty set</th><th scope="col" class="ct-ours">Difference</th>'
        '<th scope="col" class="ct-ours ct-start">Andrey</th><th scope="col">R</th></tr>'
    )
    body = ""
    for key, name in (("icp.approximate", "Approximate test"), ("icp.exact", "Exact test")):
        body += f'<tr class="ct-section"><th scope="rowgroup" colspan="8">{name}</th></tr>'
        for r in lanes[key]["rows"]:
            if r["cell"].startswith("speed"):
                continue
            k = r["environments"]
            envs = f"{k[0]} environments" if len(k) == 1 else f"{k[0]} or {k[-1]} environments"
            low, high = r["bootstrap_95"]
            body += (
                f'<tr><th scope="row">{escape(ICP_CELLS[r["cell"]])}<small>{envs}</small></th>'
                f'<td class="ct-ours ct-start">{r["andrey_shd_mean"]:.3f}</td>'
                f"<td>{r['comparator_shd_mean']:.3f}</td><td>{r['empty_shd_mean']:.3f}</td>"
                f'<td class="ct-ours">{interval(r["mean_D"], low, high)}'
                f"{verdict(r['verdict'], r['verdict'])}</td>"
                f'<td class="ct-ours ct-start">{r["andrey_fwer"]:.4f}</td>'
                f"<td>{r['comparator_fwer']:.4f}</td>"
                f'<td class="ct-start">{r["same_accepted_sets"]:,} of {r["paired"]:,}</td></tr>'
            )
    paired = (
        '<figure class="ct"><p class="ct-title">ICP against InvariantCausalPrediction 0.8, '
        'the same fits</p><div class="ct-scroll"><table class="ct-table ct-nested">'
        f"<thead>{head}</thead><tbody>{body}</tbody></table></div><figcaption>"
        "Means over the paired fits at alpha 0.05: 50 datasets of each of 100 scenarios with 5 to "
        "12 variables, 50 of each of 20 with do-interventions, and 20 of each of 10 with 16 or 21 "
        "variables. The errors count the variables missing from or added to the target's true "
        "parents; the FWER is the share of fits that name a variable outside them. The difference "
        "is Andrey's mean errors minus R's, with its 95% interval; unmarked is a tie."
        "</figcaption></figure>\n"
    )
    head = (
        '<tr><td rowspan="2"></td>'
        '<th scope="colgroup" colspan="3" class="ct-start">Approximate test</th>'
        '<th scope="colgroup" colspan="3" class="ct-start">Exact test</th></tr><tr>'
        + '<th scope="col" class="ct-ours ct-start">FWER</th><th scope="col">95% interval</th>'
        '<th scope="col">Success</th>' * 2 + "</tr>"
    )
    rows = {
        test: {r["cell"]: r for r in lanes[f"icp.{test}.claim"]["rows"]}
        for test in ("approximate", "exact")
    }
    body = ""
    for section, cells in (
        ("In the claim", ("A", "C", "D", "D40")),
        ("Out of the claim", ("B, target intervened", "E-hidden", "E-nonlinear")),
    ):
        body += f'<tr class="ct-section"><th scope="rowgroup" colspan="7">{section}</th></tr>'
        for cell in cells:
            tds = ""
            for test in ("approximate", "exact"):
                r = rows[test][cell]
                low, high = r["andrey_fwer_95"]
                tds += (
                    f'<td class="ct-ours ct-start">{r["andrey_fwer"]:.4f}</td>'
                    f"<td>{low:.4f} to {high:.4f}</td><td>{r['andrey_success']:.3f}</td>"
                )
            fits = rows["approximate"][cell]["fits"]
            body += (
                f'<tr><th scope="row">{escape(ICP_CELLS[cell])}<small>{fits:,} fits</small></th>'
                f"{tds}</tr>"
            )
    rates = (
        '<figure class="ct"><p class="ct-title">ICP\'s FWER at alpha 0.05, every fit</p>'
        '<div class="ct-scroll"><table class="ct-table ct-nested">'
        f"<thead>{head}</thead><tbody>{body}</tbody></table></div><figcaption>"
        "The FWER is the share of fits that name a variable outside the target's true parents, "
        "with its 95% Clopper-Pearson interval; success is the share whose estimate equals them. "
        "In the claim the environments leave the target's own mechanism alone; out of it they do "
        "not, or a hidden or nonlinear parent breaks the linear model."
        "</figcaption></figure>\n"
    )
    return paired + rates


if __name__ == "__main__":
    paths = [Path(p) for p in sys.argv[1:]] or [
        PUBLISHED / "gnies.json",
        PUBLISHED / "ut-igsp.json",
    ]
    (HERE / "gnies.html").write_text(gnies(json.loads(paths[0].read_text())))
    (HERE / "ut-igsp.html").write_text(ut_igsp(json.loads(paths[1].read_text())))
    (HERE / "camuv.html").write_text(camuv(json.loads(paths[2].read_text())))
    (HERE / "icp.html").write_text(icp(json.loads(paths[3].read_text())))
