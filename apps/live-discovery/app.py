"""Andrey's live demo: live discovery, controlled results, and teaching."""

from __future__ import annotations

import json
import os
from html import escape
from pathlib import Path

import gradio as gr
from data import explore_preset, fingerprint, new_seed, preset, read_csv, upload_path
from learn import HOST_SCRIPT, drawing, dsep_html, estimate_and_truth, iframe, preset_dag
from results import measure, results_html, status_text
from runners import PACKAGES
from style import css, theme
from workers import Workers

from andrey import GraphStructure

HERE = Path(__file__).resolve().parent
INITIAL = {"PC": 50, "FCI": 50, "DirectLiNGAM": 20}


def hardware_label():
    """Describe the actual host without presenting local timings as Space results."""
    host = os.environ.get("ANDREY_DEMO_HARDWARE")
    if not host:
        host = (
            "this Space's 2 shared vCPUs" if os.environ.get("SPACE_ID") else "this preview server"
        )
    return f"live on {host}; controlled results are in the Benchmarks tab"


def limits():
    """Load caps measured on this Space; local previews use conservative sizes."""
    path = HERE / "data" / "calibration.json"
    if not path.exists():
        return {m: {"default": d, "cap": d} for m, d in INITIAL.items()}
    return json.loads(path.read_text())["methods"]


def race(pool, method, size, seed):
    """Yield progress between sequential package fits on identical observations."""
    if method not in PACKAGES:
        raise gr.Error("Choose a listed method.")
    try:
        observations, truth, seed = preset(method, size, seed, limits()[method]["cap"])
    except ValueError as error:
        raise gr.Error(str(error)) from error
    packages = [p for p in PACKAGES[method] if p in pool.available]
    rows = [
        dict(package=p, status="queued", seconds=None, shd=None, empty_shd=None) for p in packages
    ]
    config = "Fisher-Z, alpha = 0.05" if method in ("PC", "FCI") else "pwling"
    detail = (
        f"{method} · {observations.shape[1]} variables · {len(observations)} observations · "
        f"seed {seed} · data {fingerprint(observations)}. {config}; "
        "one numeric thread per package. Imports and warm-up excluded."
    )
    graphs = ""
    yield results_html(rows, method), detail, graphs
    for index, package in enumerate(packages):
        rows[index]["status"] = "running"
        yield results_html(rows, method), detail, graphs
        rows[index] = measure(pool, method, package, observations, truth)
        row = rows[index]
        if row["status"] == "ok":
            graphs += (
                f'<details class="graph-pair"><summary>{escape(package)} graph '
                f"({escape(row['version'])})</summary>{drawing(row['graph'])}</details>"
            )
        yield results_html(rows, method), detail, graphs


def explore(pool, method, upload, seed):
    """Recover a graph from a preset, drawn beside its truth, or from a validated CSV."""
    if method not in (*PACKAGES, "GES"):
        raise gr.Error("Choose a listed method.")
    try:
        if upload:
            observations, labels = read_csv(upload_path(upload))
            truth = None
        else:
            observations, truth, description, seed = explore_preset(method, seed)
            labels = None
        result = measure(pool, method, "Andrey", observations, truth, labels)
    except ValueError as error:
        raise gr.Error(str(error)) from error
    if result["status"] != "ok":
        return "", f"{status_text(result)}."
    fit = f"Fit: {result['seconds']:.4g} s."
    if truth is None:
        return drawing(result["graph"]), (
            f"{observations.shape[1]} variables, {len(observations):,} observations. {fit} "
            "Accuracy is unavailable without a known truth graph."
        )
    metric = "Endpoint SHD" if method == "FCI" else "SHD"
    return estimate_and_truth(result["graph"], truth), (
        f"{description} {observations.shape[1]} variables, seed {seed}. {fit} "
        f"{metric}: {result['shd']}; empty graph: {result['empty_shd']}."
    )


def build_app(pool):
    """Build four tabs with one shared compute queue across all expensive events."""
    measured = limits()
    with gr.Blocks(title="Andrey | Causal discovery, fast") as demo:
        logos = "".join(
            f'<span class="wordmark wordmark-{mode}">'
            + (HERE / "brand" / f"wordmark-{mode}.svg").read_text()
            + "</span>"
            for mode in ("light", "dark")
        )
        gr.HTML(
            logos + "<h1>Causal discovery, fast.</h1><p>Analyze the same data with "
            "Andrey, causal-learn, and lingam. Compare fit times and recovered graphs.</p>"
            "<p>Over 100x faster on PC, and faster on most other supported methods. "
            '<a href="https://andrey.abel.ai/docs/benchmarks.html">Controlled benchmarks</a>; '
            "live results depend on this server and the dataset.</p>",
            elem_id="hero",
        )
        gr.HTML(f'<p class="live-label">{escape(hardware_label())}</p>')
        with gr.Tabs():
            with gr.Tab("Race"):
                with gr.Row():
                    method = gr.Dropdown(list(PACKAGES), value="PC", label="Method")
                    size = gr.Slider(
                        5,
                        measured["PC"]["cap"],
                        value=measured["PC"]["default"],
                        step=1,
                        label="Variables",
                    )
                    seed = gr.Number(value=7, precision=0, label="Dataset seed")
                with gr.Row():
                    run = gr.Button("Run comparison", variant="primary")
                    fresh = gr.Button("New dataset")
                detail = gr.Markdown("PC · seed 7. Choose a size and run the comparison.")
                results = gr.HTML()
                with gr.Accordion("Recovered graphs", open=False):
                    graphs = gr.HTML()

                def change_method(selected):
                    cap = limits()[selected]
                    return gr.Slider(
                        minimum=50 if selected == "FCI" else 5,
                        maximum=cap["cap"],
                        value=cap["default"],
                        step=1,
                    )

                method.change(change_method, method, size, queue=False, api_name=False)
                fresh.click(new_seed, None, seed, queue=False, api_name=False)

                def run_race(m, d, s):
                    yield from race(pool, m, d, s)

                run.click(
                    run_race,
                    [method, size, seed],
                    [results, detail, graphs],
                    concurrency_id="compute",
                    concurrency_limit=1,
                    api_name="race",
                )
            with gr.Tab("Benchmarks"):
                gr.HTML(
                    iframe(
                        (HERE / "data" / "benchmarks.html").read_text(),
                        "Controlled benchmarks",
                        1500,
                    )
                )
            with gr.Tab("Explore"):
                gr.Markdown(
                    "Try a reproducible preset or upload a numeric CSV with column names. "
                    "Up to 5 MB, 50 columns, and 20,000 rows. Remove constant or duplicate "
                    "columns. At least three more rows than columns are required."
                )
                with gr.Row():
                    explore_method = gr.Dropdown([*PACKAGES, "GES"], value="PC", label="Method")
                    upload = gr.File(label="CSV (optional)", file_types=[".csv"], type="filepath")
                    explore_seed = gr.Number(value=7, precision=0, label="Preset seed")
                recover = gr.Button("Recover graph", variant="primary")
                explore_detail, graph = gr.Markdown(), gr.HTML()
                recover.click(
                    lambda m, f, s: explore(pool, m, f, s),
                    [explore_method, upload, explore_seed],
                    [graph, explore_detail],
                    concurrency_id="compute",
                    concurrency_limit=1,
                    api_name="explore",
                )
            with gr.Tab("Learn"):
                gr.HTML(iframe((HERE / "data" / "pc.html").read_text(), "PC step by step", 900))
                gr.HTML(iframe((HERE / "data" / "meek.html").read_text(), "Meek's rules"))
                gr.Markdown("## Explore d-separation\nChoose two nodes and a conditioning set.")
                gr.HTML(drawing(GraphStructure.from_networkx(preset_dag())))
                with gr.Row():
                    x = gr.Dropdown(list(range(5)), value=1, label="From")
                    y = gr.Dropdown(list(range(5)), value=2, label="To")
                    conditioned = gr.CheckboxGroup(list(range(5)), label="Condition on")
                check = gr.Button("Check paths")
                answer = gr.HTML()

                def check_paths(a, b, z):
                    try:
                        return dsep_html(a, b, z)
                    except ValueError as error:
                        raise gr.Error(str(error)) from error

                # Path checks need no worker, so they never wait behind a fit.
                check.click(
                    check_paths,
                    [x, y, conditioned],
                    answer,
                    concurrency_id="d_separation",
                    concurrency_limit=4,
                    api_name="d_separation",
                )
        gr.Markdown(
            "[Documentation](https://andrey.abel.ai/docs/) · "
            "[Controlled benchmarks](https://andrey.abel.ai/docs/benchmarks.html) · "
            "Alpha: the API may change."
        )
    return demo.queue(default_concurrency_limit=1, max_size=12)


if __name__ == "__main__":
    pool = Workers()
    pool.warm()
    if (os.environ.get("SPACE_ID") or os.environ.get("ANDREY_DEMO_CALIBRATE") == "1") and not (
        HERE / "data" / "calibration.json"
    ).exists():
        from probe import calibrate

        calibrate(pool, HERE / "data" / "calibration.json")
    build_app(pool).launch(
        theme=theme(),
        css=css(),
        head=f"<script>{HOST_SCRIPT}</script>",
        max_file_size="5mb",
        server_name=os.environ.get(
            "GRADIO_SERVER_NAME", "0.0.0.0" if os.environ.get("SPACE_ID") else "127.0.0.1"
        ),
        root_path=os.environ.get("GRADIO_ROOT_PATH", ""),
        show_error=False,
    )
