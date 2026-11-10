"""The ``m2-tetrad`` lane: its cells and arms, the settings the driver reads, the summary's pairs,
and fits.

The fits run where ``ANDREY_BENCH_TETRAD_MISSING_JAR`` names tetrad-lib's shaded jar, 8.0.0 or
later (``benchmarks/start_tetrad_missing.sh``), and ``java`` has the ``jdk.compiler`` module.
"""

from __future__ import annotations

import itertools
import json
import os
import pickle
import re
import shutil
import subprocess

import missing_cpu
import missing_summary
import numpy as np
import pandas as pd
import pytest

from andrey_bench.adapters import tetrad_missing
from andrey_bench.adapters.tetrad_cmd import JAVA_ENV
from andrey_bench.adapters.tetrad_missing import JAR_ENV, TetradMissing
from andrey_bench.contracts import SolutionAdapter

SEARCHES = ("fges", "boss", "grasp")
ARMS = [TetradMissing(s, p) for s in SEARCHES for p in tetrad_missing.POLICIES]
ESS_ARMS = [TetradMissing(s, "em_covariance", ess="mean_pairwise") for s in SEARCHES]
DRIVER = tetrad_missing.DRIVER.read_text()


def _unavailable() -> str | None:
    """Why no fit can run here, or ``None`` when one can."""
    jar = os.environ.get(JAR_ENV, "")
    java = os.environ.get(JAVA_ENV) or shutil.which("java")
    if not jar or not os.path.isfile(jar):
        return f"no tetrad-lib jar (${JAR_ENV})"
    if not java:
        return "no java"
    modules = subprocess.run([java, "--list-modules"], capture_output=True, text=True)
    if modules.returncode != 0 or "jdk.compiler" not in modules.stdout:
        return f"{java} has no jdk.compiler module"
    return None


needs_tetrad = pytest.mark.skipif(_unavailable() is not None, reason=_unavailable() or "")


# --- the lane ---------------------------------------------------------------------------------


def _cells() -> set[tuple]:
    """The masked cells of ``m2-tetrad``, written out: MissDAG's three graphs at 15 variables and
    100 rows (MCAR at 0.1 to 0.5, logistic MAR at 0.1, 0.3, 0.5), and ER2 and ER4 at 10, 20, and
    50 variables and 100 to 2,000 rows (MCAR and logistic MAR at 0.1 to 0.3)."""
    graphs = (("er", 1.0), ("er", 2.0), ("scale_free", 2.0))
    paper = {
        (*g, 15, 100, "raw", "mcar", "", r) for g in graphs for r in (0.1, 0.2, 0.3, 0.4, 0.5)
    } | {(*g, 15, 100, "raw", "mar", "logistic", r) for g in graphs for r in (0.1, 0.3, 0.5)}
    generic = {
        ("er", degree, d, n, "standardized", *mask, r)
        for degree, d, n, mask, r in itertools.product(
            (2.0, 4.0),
            (10, 20, 50),
            (100, 500, 2000),
            (("mcar", ""), ("mar", "logistic")),
            (0.1, 0.2, 0.3),
        )
    }
    return paper | generic


def _cell(key) -> tuple:
    return (
        key.topology,
        key.density,
        key.d,
        key.n,
        key.standardize,
        key.missing,
        key.missing_form,
        key.missing_rate,
    )


def test_the_lane_runs_the_claimed_cells_of_the_three_searches_once_each():
    units = missing_cpu.plan("m2-tetrad", seeds=2)
    masked = [u.key for u in units if u.key.missing]
    assert {_cell(k) for k in masked} == _cells()
    assert len(masked) == 2 * len(_cells()) == 2 * 132
    keys = [u.key for u in units]
    assert len(keys) == len(set(keys))
    # The datasets are those of the lanes the cells come from: one key, one file.
    for lane in ("m2-b", "m2-c", "m2-generic"):
        theirs = {u.key for u in missing_cpu.plan(lane, seeds=2)}
        assert {k for k in keys if k in theirs}


def test_every_unit_fits_tetrad_beside_andrey_at_the_standard_bic():
    units = missing_cpu.plan("m2-tetrad", seeds=1)
    masked = {tuple(a.name for a in u.arms) for u in units if u.key.missing}
    complete = {tuple(a.name for a in u.arms) for u in units if not u.key.missing}
    assert len(masked) == len(complete) == 1
    [masked_arms], [complete_arms] = masked, complete
    plain = [f"andrey.{m}.default.ebic_gamma0" for m in ("ges", "boss", "grasp")]
    assert sorted(masked_arms) == sorted(
        [
            "andrey.pc.default",
            *[f"andrey.{m}.em.ebic_gamma0" for m in ("ges", "boss", "grasp")],
            *[f"listwise.{p}" for p in plain],
            *[f"impute_mean.{p}" for p in plain],
            *[f"tetrad.{s}.em_covariance" for s in SEARCHES],
            *[f"tetrad.{s}.em_covariance.mean_pairwise" for s in SEARCHES],
            *[f"tetrad.{s}.testwise" for s in SEARCHES],
            "baseline.empty",
        ]
    )
    assert sorted(complete_arms) == sorted(
        [*plain, *[f"tetrad.{s}.complete" for s in SEARCHES], "baseline.empty"]
    )
    # One objective on both sides: Andrey's penalty_discount default at ebic_gamma 0.0.
    for unit in units[:1]:
        for arm in unit.arms:
            if arm.package == "tetrad":
                assert arm.params()["penaltyDiscount"] == 1.0
                assert arm.params()["structurePrior"] == 0.0
            elif arm.name.startswith("andrey.") and ".pc." not in arm.name:
                assert arm.params()["ebic_gamma"] == 0.0 and "penalty_discount" not in arm.params()


# --- the adapters -----------------------------------------------------------------------------


@pytest.mark.parametrize("arm", [*ARMS, *ESS_ARMS], ids=lambda a: a.name)
def test_an_arm_meets_the_harness_contract(arm):
    assert isinstance(arm, SolutionAdapter)
    json.dumps(arm.params(), sort_keys=True)
    clone = pickle.loads(pickle.dumps(arm))
    assert (clone.name, clone.params()) == (arm.name, arm.params())
    assert arm.output_type == "cpdag"
    assert arm.algorithm == {"fges": "ges"}.get(arm.search, arm.search)


def test_the_arms_are_named_for_their_search_and_policy():
    assert [a.name for a in ARMS[:3]] == [
        "tetrad.fges.em_covariance",
        "tetrad.fges.testwise",
        "tetrad.fges.complete",
    ]
    assert TetradMissing("boss").params()["missingDataPolicy"] == "em_covariance"
    assert TetradMissing("boss", "fail").params()["missingDataPolicy"] == "fail"
    with pytest.raises(ValueError, match="search"):
        TetradMissing("pc")
    with pytest.raises(ValueError, match="policy"):
        TetradMissing("fges", "em")


def test_the_mean_pairwise_arm_differs_from_the_em_arm_in_its_effective_sample_size_only():
    assert [a.name for a in ESS_ARMS] == [
        f"tetrad.{s}.em_covariance.mean_pairwise" for s in SEARCHES
    ]
    for search in SEARCHES:
        full, mean = TetradMissing(search), TetradMissing(search, ess="mean_pairwise")
        assert full.name == f"tetrad.{search}.em_covariance"  # the recorded arm keeps its name
        assert full.params()["essMode"] == "full_n" and mean.params()["essMode"] == "mean_pairwise"
        assert {**mean.params(), "essMode": "full_n"} == full.params()
        assert mean.package_version == full.package_version
    assert (
        TetradMissing("fges", ess="min_pairwise").name == "tetrad.fges.em_covariance.min_pairwise"
    )


def test_an_effective_sample_size_is_for_the_policy_that_reads_it():
    for policy in ("testwise", "fail"):
        assert TetradMissing("boss", policy, ess="full_n").params()["essMode"] == "full_n"
        with pytest.raises(ValueError, match="em_covariance only"):
            TetradMissing("boss", policy, ess="mean_pairwise")
    with pytest.raises(ValueError, match="ess must be one of"):
        TetradMissing("boss", ess="mean")


def _read_by_driver() -> dict[str, set[str]]:
    """The setting names the driver reads for each algorithm, from its source."""
    body = DRIVER.split("static Graph run(", 1)[1].split("static String upper(", 1)[0]
    score = DRIVER.split("static SemBicScore score(", 1)[1].split("static DataSet read(", 1)[0]
    shared = set(re.findall(r's\.\w+\("(\w+)"\)', score))
    read = {}
    for case in re.split(r'\n            case "', body)[1:]:
        read[case.split('"', 1)[0]] = set(re.findall(r's\.\w+\("(\w+)"\)', case)) | shared
    return read


@pytest.mark.parametrize("arm", [*ARMS, *ESS_ARMS], ids=lambda a: a.name)
def test_an_arm_passes_exactly_the_settings_the_driver_reads(arm, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    assert set(arm.settings({})) == _read_by_driver()[arm.search]


def test_the_em_settings_are_tetrads_own_and_written_out():
    params = TetradMissing("fges").params()
    assert {k: params[k] for k in ("emRidge", "emTolerance", "emMaxIterations", "essMode")} == {
        "emRidge": 0.0,
        "emTolerance": 1e-6,
        "emMaxIterations": 1000,
        "essMode": "full_n",
    }


# --- the summary ------------------------------------------------------------------------------


def test_the_summary_pairs_each_search_with_tetrads_em_arms_and_its_testwise_baseline():
    arms = missing_summary.arms_of
    for method, search in missing_cpu.TETRAD_SEARCHES.items():
        got = arms("m2-tetrad", f"andrey.{method}.em.ebic_gamma0")
        plain = f"andrey.{method}.default.ebic_gamma0"
        assert got == {
            "method": f"andrey.{method}.em.ebic_gamma0",
            "testwise": "andrey.pc.default",
            "complete": plain,
            "reference": f"tetrad.{search}.em_covariance",
            "reference_testwise": f"tetrad.{search}.testwise",
            "reference_mean_pairwise": f"tetrad.{search}.em_covariance.mean_pairwise",
            "listwise": f"listwise.{plain}",
            "mean": f"impute_mean.{plain}",
        }
        assert (
            missing_summary.COMPLETE_REFERENCE["m2-tetrad"][method] == f"tetrad.{search}.complete"
        )
    assert arms("m2-tetrad", "tetrad.fges.em_covariance") is None
    assert arms("m2-tetrad", "tetrad.fges.em_covariance.mean_pairwise") is None
    # The lanes without a reference for the three searches keep none.
    assert "reference" not in arms("m2-b", "andrey.ges.em.ebic_gamma0")
    assert "reference_testwise" not in arms("m2-bnlearn", "andrey.hc.em.ebic_gamma0")
    assert "reference_mean_pairwise" not in arms("m2-b", "andrey.ges.em.ebic_gamma0")
    assert missing_summary.shd_column("m2-tetrad") == "shd"


def _records(arm: str, seeds: int, shd: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": arm,
            "data_seed": range(seeds),
            "status": "ok",
            "output_type": "cpdag",
            "shd": shd,
            "mec_shd": np.nan,
            "skeleton_f1": 0.8,
            "wall_s": 1.0,
            "complete_rows": 60.0,
        }
    )


def _table(method_shd: float, d: int, n: int, mechanism: str = "mcar") -> pd.DataFrame:
    """One ``m2-tetrad`` cell for ``ges``: Tetrad's EM arm at 19, with its effective sample size at
    the mean pairwise count at 17, its test-wise arm at 25, the empty graph at 20, and the
    complete-data fits at 2 (Andrey) and 3 (Tetrad)."""
    masked = pd.concat(
        [
            _records("andrey.ges.em.ebic_gamma0", 12, method_shd),
            _records("tetrad.fges.em_covariance", 12, 19.0),
            _records("tetrad.fges.em_covariance.mean_pairwise", 12, 17.0),
            _records("tetrad.fges.testwise", 12, 25.0),
            _records("baseline.empty", 12, 20.0),
        ]
    ).assign(mechanism=mechanism, form="", rate=0.2)
    complete = pd.concat(
        [
            _records("andrey.ges.default.ebic_gamma0", 12, 2.0),
            _records("tetrad.fges.complete", 12, 3.0),
            _records("baseline.empty", 12, 20.0),
        ]
    ).assign(mechanism="complete", form="", rate=np.nan)
    return pd.concat([masked, complete], ignore_index=True).assign(
        lane="m2-tetrad", graph="er", density=2.0, d=d, n=n
    )


def test_a_row_is_judged_against_tetrads_em_arm_and_reports_its_testwise_arm():
    [row] = missing_summary.all_rows(_table(18.0, 20, 500))
    assert row["judged"] and row["result"] == "better"
    assert row["reference"]["arm"] == "tetrad.fges.em_covariance"
    assert row["reference"]["mean_D"] == -1.0
    assert row["reference_testwise"]["arm"] == "tetrad.fges.testwise"
    assert row["reference_testwise"]["verdict"] == "better"
    assert row["control"] == row["control"] | {"other": "tetrad.fges.complete", "mean_D": -1.0}
    [worse] = missing_summary.all_rows(_table(21.0, 20, 500))
    assert worse["result"] == "worse" and worse["rerun"]
    # The test-wise arm decides nothing: the method is better than it and still worse.
    assert worse["reference_testwise"]["verdict"] == "better"
    # Nor does the arm at the mean pairwise count: the method is worse than it and still better.
    [row] = missing_summary.all_rows(_table(18.0, 20, 500))
    assert row["result"] == "better" and not row["rerun"]
    mean = row["reference_mean_pairwise"]
    assert (mean["arm"], mean["mean_D"], mean["verdict"], mean["minus_empty"]) == (
        "tetrad.fges.em_covariance.mean_pairwise",
        1.0,
        "worse",
        [-3.0, -3.0],
    )
    [out] = missing_summary.all_rows(_table(21.0, 20, 500, mechanism="self_masking"))
    assert not out["judged"] and out["result"] == "not judged"


def test_the_floor_decides_the_paper_size_cells_only():
    """Tied with Tetrad at the empty graph's distance where Tetrad is below it: the cell of 15
    variables and 100 rows fails; a 100-row cell of the generic grid keeps its verdict."""
    [paper] = missing_summary.all_rows(_table(20.0, 15, 100))
    assert paper["floor"]["applies"] and paper["result"].startswith("fail")
    [generic] = missing_summary.all_rows(_table(20.0, 20, 100))
    assert not generic["floor"]["applies"] and generic["result"] == "worse"


# --- fits -------------------------------------------------------------------------------------


def _chain(
    n: int = 400, d: int = 5, rate: float = 0.2, weight: float = 0.9
) -> tuple[np.ndarray, np.ndarray]:
    """A linear-Gaussian chain ``x0 -> ... -> x{d-1}``, complete and with values missing at
    random."""
    rng = np.random.default_rng(0)
    x = np.zeros((n, d))
    x[:, 0] = rng.normal(size=n)
    for j in range(1, d):
        x[:, j] = weight * x[:, j - 1] + rng.normal(size=n)
    masked = x.copy()
    masked[rng.random(x.shape) < rate] = np.nan
    return x, masked


@needs_tetrad
@pytest.mark.parametrize("search", SEARCHES)
def test_tetrad_recovers_a_chain_from_data_with_missing_values(search):
    """EM covariance returns the chain's CPDAG; test-wise deletion keeps the chain's edges, and may
    add one, since each local score reads its own rows."""
    _, masked = _chain()
    chain = np.zeros((5, 5), dtype=np.int8)
    for j in range(4):
        chain[j, j + 1] = chain[j + 1, j] = 1  # a chain's CPDAG is undirected
    em = TetradMissing(search, "em_covariance")
    em.setup()
    adjacency = em.fit(masked, em.params())
    assert np.array_equal(adjacency, chain)
    assert em.search_s is not None and em.search_s > 0
    assert em.to_structure(adjacency).kind == "cpdag"
    testwise = TetradMissing(search, "testwise")
    skeleton = testwise.fit(masked, testwise.params())
    assert all(skeleton[j, j + 1] or skeleton[j + 1, j] for j in range(4))
    assert re.fullmatch(r"tetrad-lib \d+\.\d+\.\d+, Java .+", em.package_version)


@needs_tetrad
@pytest.mark.parametrize("search", SEARCHES)
def test_the_mean_pairwise_arm_recovers_a_chain_from_data_with_missing_values(search):
    _, masked = _chain()
    chain = np.zeros((5, 5), dtype=np.int8)
    for j in range(4):
        chain[j, j + 1] = chain[j + 1, j] = 1
    arm = TetradMissing(search, ess="mean_pairwise")
    arm.setup()
    assert np.array_equal(arm.fit(masked, arm.params()), chain)


@needs_tetrad
@pytest.mark.parametrize("search", SEARCHES)
def test_the_mean_pairwise_count_makes_the_penalty_heavier_when_half_the_values_are_missing(search):
    """The chain of 8 variables has 7 edges. Half the values are missing, so the mean pairwise count
    is about a quarter of the rows: with every row as the sample size the EM covariance's weak
    links pass the penalty, with the count they do not."""
    _, masked = _chain(d=8, rate=0.5, weight=0.25)
    edges = {}
    for ess in ("full_n", "mean_pairwise"):
        arm = TetradMissing(search, ess=ess)
        adjacency = arm.fit(masked, arm.params())
        edges[ess] = int(((adjacency + adjacency.T) > 0).sum() // 2)
    assert edges["full_n"] > 7 >= edges["mean_pairwise"]


@needs_tetrad
def test_on_complete_data_the_effective_sample_size_is_the_rows():
    complete, _ = _chain(n=200, d=8, weight=0.5)
    fits = []
    for ess in tetrad_missing.ESS_MODES:
        arm = TetradMissing("boss", ess=ess)
        fits.append(arm.fit(complete, arm.params()))
    assert np.array_equal(fits[0], fits[1]) and np.array_equal(fits[0], fits[2])


@needs_tetrad
def test_the_complete_arm_fits_complete_data_and_stops_on_a_missing_value():
    complete, masked = _chain()
    arm = TetradMissing("fges", "fail")
    assert arm.fit(complete, arm.params()).sum() == 8
    with pytest.raises(RuntimeError, match="missing"):
        arm.fit(masked, arm.params())


@needs_tetrad
def test_on_complete_data_every_policy_is_the_plain_score():
    complete, _ = _chain(n=60, d=8)
    fits = []
    for policy in tetrad_missing.POLICIES:
        arm = TetradMissing("boss", policy)
        fits.append(arm.fit(complete, arm.params()))
    assert np.array_equal(fits[0], fits[1]) and np.array_equal(fits[0], fits[2])


@needs_tetrad
def test_the_driver_stops_on_an_unknown_setting():
    _, masked = _chain()
    arm = TetradMissing("grasp")
    with pytest.raises(RuntimeError, match="unknown setting --extra"):
        arm.fit(masked, {**arm.params(), "extra": 1})
