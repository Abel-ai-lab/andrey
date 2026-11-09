"""The LOCI lane: its pair regimes, the adapters, and the authors' code at the reference boundary.

Facts about the regimes and Andrey's adapters are checked in bench-env. The authors' adapter is
checked there against a stand-in for its code, and by running the code itself in ``.venv-loci``,
skipped when that environment has not been built.
"""

from __future__ import annotations

import dataclasses
import json
import pickle
import sys
from pathlib import Path

import loci_pairs
import numpy as np
import pytest

from andrey import GraphStructure
from andrey.data.functional import LOCATION_SCALE_RANGE
from andrey_bench import datasets, pair_models, runner
from andrey_bench.adapters import immer_loci
from andrey_bench.adapters.andrey_loci import AndreyLOCI, AndreyLOCILikelihood
from andrey_bench.adapters.immer_loci import ImmerLOCI, ImmerLOCILikelihood, edge_marks

LOCI_PYTHON = Path(runner.BENCH_ROOT) / ".venv-loci" / "bin" / "python"
LOCI_CODE = Path(runner.BENCH_ROOT) / ".loci"
needs_loci_env = pytest.mark.skipif(
    not (LOCI_PYTHON.exists() and (LOCI_CODE / "COMMIT").is_file()),
    reason="loci env not built - run benchmarks/start_loci.sh",
)

ANDREY = [AndreyLOCI(), AndreyLOCILikelihood()]
AUTHORS = [ImmerLOCI(), ImmerLOCILikelihood()]
PAPER_SETS = [r for r in loci_pairs.REGIMES if r.startswith("tagasovska_")]
LSNM_PAIRS = [r for r in loci_pairs.REGIMES if r.startswith("lsnm_")]


def _ls_pair(n: int = 400, seed: int = 0) -> np.ndarray:
    """A location-scale pair ``x -> y`` whose noise scale changes with the cause."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    return np.column_stack(
        [x, np.sin(2 * x) + x + (0.6 + 0.5 * np.tanh(x)) * rng.standard_normal(n)]
    )


def _edges(marks: np.ndarray) -> list[tuple[int, int]]:
    return [(i, j) for i, j, _ in GraphStructure.from_numpy(marks, kind="dag").oriented_edges()]


# --- regimes ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("regime", loci_pairs.REGIMES)
def test_every_regime_draws_two_columns_and_one_edge_either_way(regime: str) -> None:
    directions = set()
    for key in loci_pairs.regime_keys(regime, n=200, seeds=8):
        data, marks, kind, columns = datasets.generate_one(key)
        assert data.shape == (200, 2) and kind == "dag" and sorted(columns) == [0, 1]
        [edge] = _edges(marks)
        directions.add(edge)
    assert directions == {(0, 1), (1, 0)}  # the seed draws the direction


def test_the_location_scale_pairs_are_the_generator_at_the_papers_scale_range() -> None:
    """Gaussian, Gumbel, and Laplace noise, z-scored or rescaled, at the paper's scale bounds."""
    assert LOCATION_SCALE_RANGE == pytest.approx((0.2, 0.2 + 0.4 * np.pi))
    cells = set()
    for regime in LSNM_PAIRS:
        (key,) = loci_pairs.regime_keys(regime, n=500, seeds=1)
        call = datasets.generation_kwargs(key)
        assert (call["functional"], call["d"], call["density"]) == ("location_scale", 2, 1.0)
        cells.add((call["noise"], call["scale"]))
        data, _, _, _ = datasets.generate_one(key)
        if call["scale"] == "standardize":
            np.testing.assert_allclose(data.std(axis=0), 1.0)
    assert cells == {
        (noise, scale)
        for noise in ("gaussian", "gumbel", "laplace")
        for scale in ("standardize", "rescale")
    }


@pytest.mark.parametrize("regime", PAPER_SETS)
def test_a_paper_set_is_one_thousand_rows_of_standardized_columns(regime: str) -> None:
    assert loci_pairs.SIZES[regime] == [1000]
    (key,) = loci_pairs.regime_keys(regime, n=1000, seeds=1)
    data, _, _, _ = datasets.generate_one(key)
    assert data.shape == (1000, 2)
    np.testing.assert_allclose(data.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(data.std(axis=0), 1.0)


def test_a_paper_key_is_one_pair_with_its_sets_noise() -> None:
    (key,) = loci_pairs.regime_keys("tagasovska_mnu_pair", n=100, seeds=1)
    assert key.noise == "uniform"
    for change in ({"d": 3}, {"density": 2.0}, {"noise": "gaussian"}, {"standardize": "ancestral"}):
        with pytest.raises(ValueError, match="one pair"):
            datasets.generation_kwargs(dataclasses.replace(key, **change))


def test_the_paper_sets_share_one_coin_flip_per_seed_and_both_orders_occur() -> None:
    orders = {
        regime: [
            tuple(datasets.generate_one(k)[3])
            for k in loci_pairs.regime_keys(regime, n=50, seeds=40)
        ]
        for regime in PAPER_SETS
    }
    first = orders[PAPER_SETS[0]]
    assert all(o == first for o in orders.values())
    assert 10 <= first.count((1, 0)) <= 30


def test_the_cause_has_the_papers_standard_deviation() -> None:
    """``X ~ N(0, sqrt(2))``: over 20 raw pairs of 1,000 rows the cause's spread is sqrt(2)."""
    sds = []
    for seed in range(20):
        data, marks, columns = pair_models.sample("ans2020", n=1000, seed=seed, standardize="raw")
        cause = int(np.flatnonzero(marks == 1)[0] // 2)  # the tail sits in the cause's row
        sds.append(data[:, cause].std())
    assert np.mean(sds) == pytest.approx(np.sqrt(2), rel=0.02)


@pytest.mark.parametrize(
    ("model", "mechanism", "noise"),
    [
        ("an2020", lambda x: np.zeros_like(x), "additive"),
        ("ls2020", lambda x: x, "location_scale"),
        ("mnu2020", lambda x: np.ones_like(x), "multiplicative"),
    ],
)
def test_each_set_combines_its_mechanism_and_noise_as_the_paper_says(
    monkeypatch, model, mechanism, noise
) -> None:
    """With the mechanism fixed, the noise left over is the paper's: Gaussian with one standard
    deviation per pair in ``[1/5, sqrt(2)/5]``, scaled by ``f - min f`` for LS, uniform on (0, 1)
    times ``f`` for MN-U."""
    assert pair_models.MODELS[model].noise_model == noise
    monkeypatch.setattr(pair_models, "gp_draw", lambda x, spec, rng: mechanism(x))
    monkeypatch.setattr(pair_models, "sigmoid_draw", lambda x, spec, rng: mechanism(x))
    lo, hi = pair_models.MODELS[model].noise_sd
    for seed in range(5):
        data, _, columns = pair_models.sample(
            model, n=4000, seed=seed, standardize="raw", shuffle=False
        )
        x, y = data.T
        f = mechanism(x)
        if noise == "additive":
            e = y - f
        elif noise == "location_scale":
            keep = f > f.min()
            e = (y - f)[keep] / (f - f.min())[keep]
        else:
            assert 0 < y.min() and y.max() < 1
            assert abs(y.mean() - 0.5) < 0.02 and abs(y.var() - 1 / 12) < 0.01
            continue
        assert lo * 0.95 < e.std() < hi * 1.05
        assert abs(e.mean()) < 0.02 and abs(np.mean(e**4) / e.var() ** 2 - 3) < 0.3  # Gaussian


def test_the_s_sets_and_mnu_draw_bounded_monotone_sigmoids() -> None:
    """``f(x) = a b (x + c) / (1 + |b (x + c)|)`` is injective, with ``|f| < a`` and ``a >= 1``."""
    assert {m for m, s in pair_models.MODELS.items() if s.mechanism == "sigmoid"} == {
        "ans2020",
        "lss2020",
        "mnu2020",
    }
    grid = np.linspace(-6, 6, 2001)
    spec = pair_models.MODELS["ans2020"]
    amplitudes = []
    for seed in range(200):
        f = pair_models.sigmoid_draw(grid, spec, np.random.default_rng(seed))
        steps = np.diff(f)
        assert (steps > 0).all() or (steps < 0).all()
        amplitudes.append(np.abs(f).max())
    amplitudes = np.array(amplitudes)
    assert amplitudes.max() < 1 + 8.0  # a = Exp(rate 4) + 1 stays small
    assert 1.0 < np.median(amplitudes) < 1.6


def test_the_gp_draw_has_the_unit_kernel() -> None:
    """At three points the draws' covariance is ``exp(-(u - v)^2 / 2)``, variance 1."""
    points = np.array([0.0, 0.5, 2.0])
    spec = pair_models.MODELS["an2020"]
    rng = np.random.default_rng(0)
    draws = np.array([pair_models.gp_draw(points, spec, rng) for _ in range(20000)])
    expected = np.exp(-((points[:, None] - points[None, :]) ** 2) / 2)
    np.testing.assert_allclose(np.cov(draws.T), expected, atol=0.03)


def test_a_paper_set_repeats_its_draw_and_records_its_setup() -> None:
    (key,) = loci_pairs.regime_keys("tagasovska_ls_pair", n=300, seeds=1)
    first, second = datasets.generate_one(key), datasets.generate_one(key)
    np.testing.assert_array_equal(first[0], second[0])
    call = datasets.generation_kwargs(key)
    assert call["setup"] == dataclasses.asdict(pair_models.MODELS["ls2020"])
    assert call["generator"] == "andrey_bench.pair_models.sample"


def test_the_likelihood_criterion_is_misspecified_where_the_noise_is_not_gaussian() -> None:
    misspecified = {r for r in loci_pairs.REGIMES if loci_pairs.likelihood_misspecified(r)}
    assert misspecified == {
        "lsnm_gumbel_pair",
        "lsnm_laplace_pair",
        "lsnm_gumbel_rescaled_pair",
        "lsnm_laplace_rescaled_pair",
        "tagasovska_mnu_pair",
    }


# --- adapters --------------------------------------------------------------------------------------


@pytest.mark.parametrize("adapter", ANDREY + AUTHORS, ids=lambda a: a.name)
def test_an_adapter_survives_the_record_and_the_worker(adapter) -> None:
    assert json.loads(json.dumps(adapter.params(), sort_keys=True)) == adapter.params()
    assert pickle.loads(pickle.dumps(adapter)).name == adapter.name
    assert (adapter.algorithm, adapter.output_type) == ("loci", "dag")


def test_each_andrey_arm_has_the_authors_arm_of_its_criterion() -> None:
    pairs = {a.name: b.name for a, b in zip(ANDREY, AUTHORS)}
    assert loci_pairs.COMPARATOR == pairs
    for mine, theirs in zip(ANDREY, AUTHORS):
        assert (mine.params()["criterion"] == "independence") == theirs.params()[
            "independence_test"
        ]
        assert theirs.params()["neural_network"] is False


@pytest.mark.parametrize("adapter", ANDREY, ids=lambda a: a.name)
def test_andreys_arm_finds_the_planted_direction(adapter) -> None:
    adapter.setup()
    for data, edge in ((_ls_pair(), (0, 1)), (_ls_pair()[:, ::-1], (1, 0))):
        native = np.asarray(adapter.fit(data, adapter.params()))
        structure = adapter.to_structure(native)
        assert (structure.kind, structure.n_nodes) == ("dag", 2)
        assert structure.oriented_edges() == [(*edge, "directed")]


@pytest.mark.parametrize(
    ("score", "edges"), [(0.3, [(0, 1)]), (-1e-9, [(1, 0)]), (0.0, []), (float("nan"), [])]
)
def test_the_authors_score_decides_the_edge_by_its_sign(score, edges) -> None:
    assert _edges(edge_marks(score)) == edges


def _stand_in(root: Path) -> Path:
    """A ``causa`` package whose ``loci`` records its call and scores ``x`` against ``y``."""
    package = root / "causa"
    package.mkdir()
    for name in immer_loci.REQUIRED_FILES:
        (root / name).write_text("")
    (package / "loci.py").write_text(
        "import numpy as np\n"
        "CALLS = []\n"
        "def loci(x, y, independence_test=True, neural_network=True, return_function=False,\n"
        "         n_steps=None):\n"
        "    CALLS.append((x, y, independence_test, neural_network, n_steps))\n"
        "    return float(np.mean(x) - np.mean(y)) + (1.0 if independence_test else -1.0)\n"
    )
    return root


@pytest.mark.parametrize("adapter", AUTHORS, ids=lambda a: a.name)
def test_the_authors_arm_standardizes_and_sets_two_arguments(adapter, monkeypatch, tmp_path):
    monkeypatch.setenv(immer_loci.CODE_ENV, str(_stand_in(tmp_path)))
    for name in [m for m in sys.modules if m == "causa" or m.startswith("causa.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.syspath_prepend(str(tmp_path))
    rng = np.random.default_rng(1)
    data = np.column_stack([5 + 3 * rng.standard_normal(50), -2 + 0.1 * rng.standard_normal(50)])
    adapter.setup()
    marks = adapter.fit(data, adapter.params())
    from causa.loci import CALLS  # ty: ignore[unresolved-import]  # the stand-in above

    [warm_up, (x, y, independence_test, neural_network, n_steps)] = CALLS
    assert warm_up[2:] == (adapter.independence_test, False, immer_loci.WARMUP_STEPS)
    for column in (x, y):
        assert column.ndim == 1 and abs(column.mean()) < 1e-12 and column.std() == pytest.approx(1)
    assert (independence_test, neural_network, n_steps) == (adapter.independence_test, False, None)
    assert _edges(marks) == ([(0, 1)] if adapter.independence_test else [(1, 0)])


def test_the_authors_arm_names_a_missing_code_directory(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv(immer_loci.CODE_ENV, raising=False)
    with pytest.raises(SystemExit, match="start_loci.sh"):
        immer_loci.code_dir()
    monkeypatch.setenv(immer_loci.CODE_ENV, str(tmp_path))
    with pytest.raises(SystemExit, match="causa/loci.py"):
        immer_loci.code_dir()


@needs_loci_env
@pytest.mark.parametrize("adapter", AUTHORS, ids=lambda a: a.name)
def test_the_authors_code_runs_in_its_environment_and_finds_the_planted_direction(
    adapter, monkeypatch
) -> None:
    """On the pair of 300 rows that ``start_loci.sh`` fits, which both criteria orient correctly
    (on 400 rows of the same model, LOCI-H names the reverse direction)."""
    monkeypatch.setenv(adapter.python_env, str(LOCI_PYTHON))
    monkeypatch.setenv(immer_loci.CODE_ENV, str(LOCI_CODE))
    assert (LOCI_CODE / "COMMIT").read_text().strip() == immer_loci.COMMIT
    pair = _ls_pair(n=300)
    for data, edge in ((pair, (0, 1)), (pair[:, ::-1], (1, 0))):
        (measures,) = runner.run_task(
            adapter, data, adapter.params(), cap_wall_s=600.0, repeats=1, warmup=0
        )
        assert measures["status"] == "ok", measures.get("stderr")
        assert adapter.to_structure(measures["adj"]).oriented_edges() == [(*edge, "directed")]
