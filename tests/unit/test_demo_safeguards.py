"""Test CSV handling, path explanations, and worker deadlines in the demo."""

from __future__ import annotations

import importlib.util
import itertools
import sys
import time
import xml.dom.minidom as minidom
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "apps" / "live-discovery"


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


data = load("demo_data", APP / "data.py")
learn = load("demo_learn", APP / "learn.py")
workers = load("demo_workers", APP / "workers.py")


def test_path_classifier_agrees_with_oracle_on_all_four_node_dags():
    edges = list(itertools.combinations(range(4), 2))
    for mask in range(2 ** len(edges)):
        dag = nx.DiGraph()
        dag.add_nodes_from(range(4))
        dag.add_edges_from(edge for i, edge in enumerate(edges) if mask & (1 << i))
        for x, y in itertools.combinations(range(4), 2):
            others = set(dag) - {x, y}
            for size in range(len(others) + 1):
                for z in itertools.combinations(others, size):
                    answer = learn.classify_paths(dag, x, y, z)
                    assert not answer["truncated"]
                    assert any(p["active"] for p in answer["paths"]) != answer["separated"]


def test_descendant_opens_collider_and_path_listing_is_capped():
    dag = nx.DiGraph([(0, 2), (1, 2), (2, 3)])
    assert learn.classify_paths(dag, 0, 1, [])["separated"]
    assert not learn.classify_paths(dag, 0, 1, [3])["separated"]
    dense = nx.DiGraph(itertools.combinations(range(7), 2))
    result = learn.classify_paths(dense, 0, 6, [], limit=3)
    assert result["truncated"] and len(result["paths"]) == 3
    assert not result["separated"]


def test_all_meek_examples_have_real_firings():
    from andrey.core.orient import meek

    for rule, initial in learn.meek_inputs().items():
        trace = []
        meek(initial, _trace=trace)
        assert any(s["rule"] == rule for s in trace)


def test_meek_frames_keep_one_layout_and_ring_each_changed_endpoint():
    frames = learn.meek_frames()
    assert {f["replay"] for f in frames} == set(learn.MEEK_INPUTS)
    layouts, previous = {}, {}
    for frame in frames:
        document = minidom.parseString(frame["svg"])
        circles = document.getElementsByTagName("circle")
        nodes = np.array(
            [
                (float(c.getAttribute("cx")), float(c.getAttribute("cy")))
                for c in circles
                if not c.getAttribute("class") and float(c.getAttribute("r")) == 13
            ]
        )
        np.testing.assert_array_equal(layouts.setdefault(frame["replay"], nodes), nodes)
        # No edge runs through a node it does not join.
        for line in document.getElementsByTagName("line"):
            a = np.array([float(line.getAttribute(k)) for k in ("x1", "y1")])
            b = np.array([float(line.getAttribute(k)) for k in ("x2", "y2")])
            ends = {int(np.argmin(np.hypot(*(nodes - e).T))) for e in (a, b)}
            for k, node in enumerate(nodes):
                t = np.clip(np.dot(node - a, b - a) / np.dot(b - a, b - a), 0, 1)
                assert k in ends or np.hypot(*(a + t * (b - a) - node)) > 13
        assert "<path" not in frame["svg"]
        rings = [c for c in circles if c.getAttribute("class") == "highlight"]
        before = previous.get(frame["replay"])
        previous[frame["replay"]] = frame["marks"]
        if before is None:
            assert not rings
            continue
        ((head, tail),) = np.argwhere(frame["marks"] != before)
        assert frame["marks"][head, tail] == 2  # the arrowhead sits at ``head``
        (ring,) = rings
        center = np.array([float(ring.getAttribute(k)) for k in ("cx", "cy")])
        assert int(np.argmin(np.hypot(*(nodes - center).T))) == head


def final_marks(payload):
    n = payload["n"]
    marks = payload["layers"][payload["frames"][-1]["now"]]["marks"]
    return np.array([int(m) for m in marks]).reshape(n, n)


def test_pc_replays_end_on_the_graph_andrey_pc_returns():
    import andrey

    X, labels = learn.sprinkler_data()
    small = learn.pc_walkthrough(X, labels)
    np.testing.assert_array_equal(final_marks(small), andrey.pc(X).structure.to_numpy())
    _, trace = learn.trace_pc(X, 0.05)
    tests = [r for r in trace if r["step"] == "test"]
    assert sum(f["lines"] in ([6], [7]) for f in small["frames"]) == len(tests)
    assert all("svg" in layer for layer in small["layers"])

    large_X = learn.matrix_example()
    large = learn.pc_by_depth(large_X)
    assert 30 <= large["n"] <= 50 and large["labels"] is None
    assert not any("svg" in layer for layer in large["layers"])
    np.testing.assert_array_equal(final_marks(large), andrey.pc(large_X).structure.to_numpy())
    depths = {r["depth"] for r in learn.trace_pc(large_X, 0.05)[1] if r["step"] == "test"}
    assert sum(f["lines"] == [8] for f in large["frames"]) == len(depths) >= 3


def test_pc_replay_page_is_identical_across_builds():
    X, labels = learn.sprinkler_data()
    assert learn.pc_replay(X, labels) == learn.pc_replay(X.copy(), labels)


@pytest.fixture
def results(monkeypatch):
    monkeypatch.syspath_prepend(str(APP))
    return load("demo_results", APP / "results.py")


def race_row(package, seconds, shd, empty_shd=16):
    return dict(package=package, status="ok", seconds=seconds, shd=shd, empty_shd=empty_shd)


def test_race_ratios_use_the_homepage_format(results):
    rows = [
        race_row("Andrey", 0.01, 3),
        race_row("causal-learn", 2.18, 4),
        race_row("lingam", 0.0431, 5),
    ]
    html = results.results_html(rows, "DirectLiNGAM")
    assert "218x" in html and "4.3x" in html
    assert "e+" not in html


def test_race_and_explore_show_a_failed_fit_s_message(results):
    failed = dict(package="lingam", status="error", seconds=None, shd=None, empty_shd=16)
    failed["error"] = "ValueError: <bad> input"
    html = results.results_html([race_row("Andrey", 0.01, 3), failed], "DirectLiNGAM")
    assert "Fit failed: ValueError: &lt;bad&gt; input" in html
    assert results.status_text(dict(status="timeout")) == "Did not finish in 60 s"


def test_race_shows_a_ratio_whenever_both_fits_finish_whatever_the_shd(results):
    rows = [
        race_row("Andrey", 0.01, 3),
        race_row("causal-learn", 2.0, 16),  # SHD equal to the empty graph's
        race_row("gCastle", 1.0, 2),
    ]
    html = results.results_html(rows, "PC")
    assert all(f"<strong>{r['package']}</strong>" in html for r in rows)
    cells = html.split("<tr")[3:]  # header, Andrey, then causal-learn
    assert "200x" in cells[0] and "speed claim" not in html
    assert "100x" in cells[1]
    # Andrey's graph no better than the empty graph still gets its speed ratio.
    loser = [race_row("Andrey", 0.01, 16), race_row("gCastle", 1.0, 2)]
    assert "100x" in results.results_html(loser, "PC")
    timeout = [race_row("Andrey", 0.01, 3), dict(race_row("gCastle", 1.0, 2), status="timeout")]
    assert "100x" not in results.results_html(timeout, "PC")


def test_estimate_and_truth_share_node_positions():
    truth = data.explore_preset("DirectLiNGAM", 7)[1]
    estimate = truth.__class__.from_numpy(np.zeros((truth.n_nodes,) * 2), kind="dag")
    svgs = learn.estimate_and_truth(estimate, truth).split("<svg")[1:]
    centers = [
        sorted(
            (c.getAttribute("cx"), c.getAttribute("cy"))
            for c in minidom.parseString(
                "<svg" + svg.split("</svg>")[0] + "</svg>"
            ).getElementsByTagName("circle")
            if c.getAttribute("r") == "13.00"
        )
        for svg in svgs
    ]
    assert len(centers) == 2 and len(centers[0]) == truth.n_nodes
    assert centers[0] == centers[1]


def test_embedded_pages_follow_the_host_theme_and_report_their_height(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "apps"))
    build_assets = load("demo_build_assets", ROOT / "apps" / "build_assets.py")
    build_assets.build_assets(tmp_path)
    for name in ("pc.html", "meek.html", "d-separation.html", "benchmarks.html"):
        page = (tmp_path / "data" / name).read_text()
        assert learn.FRAME_SCRIPT in page, name
        assert ':root[data-theme="dark"]' in page or ":root[data-theme=dark]" in page, name
    # The hosts: the Space's page, and each docs page that shows an interactive example.
    assert 'head=f"<script>{HOST_SCRIPT}</script>"' in (APP / "app.py").read_text()
    conf = (ROOT / "docs" / "conf.py").read_text()
    assert '"example-frame.js", build_assets.HOST_SCRIPT' in conf
    assert '["examples.css"], ["example-frame.js"]' in conf
    # The docs give each example its own title, so the embedded page leaves its heading out.
    pages = (
        build_assets.pc_page(heading=False),
        learn.meek_replay(heading=False),
        learn.dsep_document(heading=False),
    )
    for page in pages:
        assert "<h1>" not in page and learn.FRAME_SCRIPT in page
    # The benchmark page drives its belts, so they can be dragged and swiped as on the homepage.
    benchmarks = (tmp_path / "data" / "benchmarks.html").read_text()
    assert 'class="tc-belt"' in benchmarks
    assert (ROOT / "site" / "assets" / "figures.js").read_text() in benchmarks


def csv_file(tmp_path, values):
    path = tmp_path / "data.csv"
    np.savetxt(path, values, delimiter=",", header="a,b,c", comments="")
    return path


@pytest.mark.parametrize("defect", ["constant", "duplicate", "infinity", "few_rows"])
def test_csv_rejects_invalid_data(tmp_path, defect):
    values = np.random.default_rng(4).normal(size=(10, 3))
    if defect == "constant":
        values[:, 0] = 1
    elif defect == "duplicate":
        values[:, 1] = values[:, 0]
    elif defect == "infinity":
        values[0, 0] = np.inf
    else:
        values = values[:3]
    with pytest.raises(ValueError):
        data.read_csv(csv_file(tmp_path, values))


@pytest.mark.parametrize("seed", [None, "7", 1.5, float("nan"), True])
def test_seed_validation_names_the_problem(seed):
    with pytest.raises(ValueError, match="whole number"):
        data.explore_preset("PC", seed)


def test_csv_limits_checked_before_body(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text(",".join(f"x{i}" for i in range(51)) + "\nnot numeric")
    with pytest.raises(ValueError, match="50 columns"):
        data.read_csv(path)
    with path.open("wb") as handle:
        handle.truncate(data.MAX_BYTES + 1)
    with pytest.raises(ValueError, match="5 MB"):
        data.read_csv(path)


def test_csv_accepts_valid_numeric_headered_data(tmp_path):
    values = np.random.default_rng(4).normal(size=(10, 3))
    actual, names = data.read_csv(csv_file(tmp_path, values))
    np.testing.assert_array_equal(actual, values)
    assert names == ("a", "b", "c")


# A fake worker's preamble: the same private protocol pipe that worker.py writes to.
PROTOCOL = (
    "import json, os, pathlib, sys, time, numpy as np\n"
    "channel=os.fdopen(int(os.environ['ANDREY_DEMO_PROTOCOL_FD']),'w')\n"
    "def emit(message): channel.write(json.dumps(message)+'\\n'); channel.flush()\n"
)


def test_stdout_writes_never_reach_the_protocol():
    script = (
        PROTOCOL
        + "print('library chatter', flush=True); os.write(1, b'{not json\\n')\n"
        + "emit({'event':'ready','version':'test'})\n"
        + "request=json.loads(sys.stdin.readline())\n"
        + "os.write(1, b'native kernel output\\n'); emit({'event':'fit'})\n"
        + "np.save(request['output'],np.eye(3,dtype=np.uint8))\n"
        + 'os.write(1, b\'{"event": "error", "error": "forged"}\\n\')\n'
        + "emit({'event':'result','seconds':0.01})\n"
    )
    worker = workers.Worker("test", [sys.executable, "-u", "-c", script])
    try:
        result = worker.run("PC", np.zeros((5, 3)), timeout=5)
        assert result["status"] == "ok"
        np.testing.assert_array_equal(result["marks"], np.eye(3))
    finally:
        worker.close()


def test_fit_error_reports_its_message_and_keeps_the_worker():
    worker = workers.Worker("Andrey")
    try:
        failed = worker.run("Nope", np.zeros((5, 3)))
        assert failed["status"] == "error"
        assert failed["error"] == "KeyError: 'Nope'"
        process = worker.process
        assert process is not None
        data = np.random.default_rng(0).normal(size=(200, 4))
        assert worker.run("PC", data)["status"] == "ok"
        assert worker.process is process
    finally:
        worker.close()


def test_worker_timeout_kills_and_reaps_process(tmp_path):
    pid_file = tmp_path / "pid"
    script = (
        PROTOCOL
        + f"open({str(pid_file)!r},'w').write(str(os.getpid()))\n"
        + "emit({'event':'ready','version':'test'})\n"
        + "sys.stdin.readline()\n"
        + "emit({'event':'fit'})\n"
        + "time.sleep(60)\n"
    )
    worker = workers.Worker("test", [sys.executable, "-u", "-c", script])
    start = time.monotonic()
    result = worker.run("PC", np.zeros((5, 3)), timeout=0.05)
    assert result["status"] == "timeout"
    assert time.monotonic() - start < 5
    assert worker.process is None
    import os

    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)


def test_setup_timeout_is_not_a_fit_timeout(monkeypatch):
    monkeypatch.setattr(workers, "SETUP_TIMEOUT", 0.05)
    worker = workers.Worker("test", [sys.executable, "-c", "import time; time.sleep(60)"])
    assert worker.run("PC", np.zeros((5, 3)))["status"] == "setup_timeout"
    assert worker.process is None


def test_timed_out_worker_can_serve_the_next_request(tmp_path):
    marker = tmp_path / "already_started"
    script = (
        PROTOCOL
        + f"marker=pathlib.Path({str(marker)!r})\n"
        + "first=not marker.exists(); marker.touch()\n"
        + "emit({'event':'ready','version':'test'})\n"
        + "request=json.loads(sys.stdin.readline())\n"
        + "emit({'event':'fit'})\n"
        + "if first: time.sleep(60)\n"
        + "np.save(request['output'],np.zeros((3,3),dtype=np.uint8))\n"
        + "emit({'event':'result','seconds':0.01})\n"
    )
    worker = workers.Worker("test", [sys.executable, "-u", "-c", script])
    try:
        assert worker.run("PC", np.zeros((5, 3)), timeout=0.05)["status"] == "timeout"
        result = worker.run("PC", np.zeros((5, 3)), timeout=5)
        assert result["status"] == "ok"
        np.testing.assert_array_equal(result["marks"], np.zeros((3, 3)))
    finally:
        worker.close()


def test_signed_endpoints_and_weighted_adjacency_preserve_edge_direction():
    runners = load("demo_runners", APP / "runners.py")
    # 0 -> 1, 1 o-> 2: row i stores the mark at i, not at the opposite end.
    signed = np.array([[0, -1, 0], [1, 0, 2], [0, 1, 0]])
    np.testing.assert_array_equal(runners.signed_marks(signed), [[0, 1, 0], [2, 0, 3], [0, 2, 0]])
    # LiNGAM stores weights as B[child, parent]; adapters transpose before conversion.
    weights = np.array([[0, 0, 0], [-0.8, 0, 0], [0, 1.2, 0]])
    np.testing.assert_array_equal(
        runners.adjacency_marks(weights.T), [[0, 1, 0], [2, 0, 1], [0, 2, 0]]
    )


def test_worker_environment_pins_all_numeric_threads():
    assert workers.THREAD_ENV["OMP_NUM_THREADS"] == "1"
    assert workers.THREAD_ENV["ANDREY_NUM_WORKERS"] == "1"
    assert workers.THREAD_ENV["CUDA_VISIBLE_DEVICES"] == ""
    assert all(
        workers.THREAD_ENV[name] == "1"
        for name in (
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "NUMBA_NUM_THREADS",
        )
    )


def test_calibration_names_why_the_first_size_failed(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(APP))
    probe = load("demo_probe", APP / "probe.py")
    failure = dict(status="error", seconds=None, version="x", error="ImportError: no torch")
    success = dict(status="ok", seconds=1.0, version="y")
    pool = SimpleNamespace(run=lambda package, *_: failure if package == "Andrey" else success)
    with pytest.raises(
        RuntimeError, match="d = 50, Andrey: ImportError: no torch; causal-learn: ok"
    ):
        probe.calibrate(pool, tmp_path / "calibration.json")


def test_compute_lock_serializes_visitors():
    import threading
    from concurrent.futures import ThreadPoolExecutor

    active, peak = 0, 0
    lock = threading.Lock()

    class FakeWorker:
        def run(self, *args):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.02)
            with lock:
                active -= 1

        def close(self):
            pass

    pool = workers.Workers()
    pool.workers = {"a": FakeWorker(), "b": FakeWorker()}
    with ThreadPoolExecutor(max_workers=2) as executor:
        jobs = [
            executor.submit(pool.run, package, "PC", np.zeros((5, 3))) for package in ("a", "b")
        ]
        for job in jobs:
            job.result()
    assert peak == 1
