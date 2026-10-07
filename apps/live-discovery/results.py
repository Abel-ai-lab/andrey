"""Score package fits and lay out the Race table; no Gradio import, so tests can load it."""

from __future__ import annotations

import sys
from html import escape
from pathlib import Path

import numpy as np
from runners import KINDS, structure

from andrey import GraphStructure
from andrey.metrics import score

# A deployed Space has site/formatting.py beside this file; a checkout finds it under site/.
sys.path.append(str(Path(__file__).resolve().parents[2] / "site"))
from formatting import ratio_text  # noqa: E402


def measure(pool, method, package, observations, truth=None, labels=None):
    """Score after a bounded fit, using PAG endpoint SHD for FCI."""
    result = pool.run(package, method, observations)
    result.update(package=package, shd=None, empty_shd=None)
    metric = "shd_endpoint" if method == "FCI" else "shd"
    if truth is not None:
        empty = GraphStructure.from_numpy(
            np.zeros((truth.n_nodes, truth.n_nodes)), kind=KINDS[method], labels=truth.labels
        )
        result["empty_shd"] = score(empty, truth)[metric]
    if result["status"] != "ok":
        return result
    graph = structure(result.pop("marks"), method, labels)
    result["graph"] = graph
    if truth is not None:
        result["shd"] = score(graph, truth)[metric]
    return result


def status_text(result):
    """Describe a fit's state for visitors, including the message of a failed fit."""
    label = {
        "ok": "Finished",
        "running": "Running",
        "queued": "Waiting",
        "timeout": "Did not finish in 60 s",
        "setup_timeout": "Worker startup timed out",
        "error": "Fit failed",
    }[result["status"]]
    return f"{label}: {result['error']}" if result.get("error") else label


def results_html(results, method):
    """Show every package; a speed ratio and bar need both fits to finish, whatever their SHD."""
    metric = "Endpoint SHD" if method == "FCI" else "SHD"
    ours = next((r for r in results if r["package"] == "Andrey"), None)

    def valid(row):
        return row is not None and row["status"] == "ok"

    comparable = [r for r in results if valid(ours) and valid(r)]
    longest = max((r["seconds"] for r in comparable), default=1)
    rows = []
    for result in results:
        status = result["status"]
        seconds = f"{result['seconds']:.4g} s" if status == "ok" else "-"
        shd = "-" if result["shd"] is None else str(result["shd"])
        floor = "-" if result["empty_shd"] is None else str(result["empty_shd"])
        ratio, bar = "-", ""
        if valid(ours) and valid(result):
            ratio = ratio_text(result["seconds"] / ours["seconds"])
            bar = (
                f'<div class="bar-track"><div class="bar-fill" '
                f'style="width:{100 * result["seconds"] / longest:.2f}%"></div></div>'
            )
        label = status_text(result)
        row_class = ' class="ours"' if result["package"] == "Andrey" else ""
        rows.append(
            f"<tr{row_class}><td><strong>{escape(result['package'])}</strong>{bar}</td>"
            f"<td>{seconds}</td><td>{shd}</td><td>{floor}</td><td>{ratio}</td>"
            f"<td>{escape(label)}</td></tr>"
        )
    return (
        '<div class="table-scroll"><table class="race-table"><thead><tr><th>Package</th>'
        f"<th>Fit time</th><th>{metric}</th><th>Empty graph</th><th>Time / Andrey</th>"
        "<th>Result</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
        '<p class="quiet">SHD counts graph errors; lower is better. Ratios apply to this '
        "dataset. A timing ratio is shown whenever both fits finish; each graph's SHD is "
        "beside it.</p>"
    )
