"""Missing values: test-wise deletion in the CI tests, and the methods that accept or refuse NaN.

A test on data with ``NaN`` reads the rows observed in all of its variables, and gives the p-value
the same test gives on that block. The methods whose spec sets ``data.missing`` accept ``NaN``;
every other method raises a ``ValueError`` that names it.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest

import andrey
from andrey.cli import main
from andrey.core import ci
from andrey.core.ci import (
    FisherZ,
    StudentT,
    make_indep_test,
    register_indep_test,
)
from andrey.core.ci_missing import rows_metadata, rows_tested
from andrey.core.score import CopiedColumnsError
from andrey.spec import REGISTRY
from andrey.temporal.pcmci import design_test, lagged_design

SUPPORTED = [name for name, spec in REGISTRY.items() if spec.data.missing]
REFUSED = [name for name, spec in REGISTRY.items() if not spec.data.missing]
SERIES = [name for name in SUPPORTED if REGISTRY[name].family == "temporal"]


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", andrey.ExperimentalWarning)
        warnings.simplefilter("ignore", andrey.DataTypeWarning)
        warnings.simplefilter("ignore", andrey.SampleSizeWarning)
        yield


def _gaps(x: np.ndarray, rate: float, seed: int = 1) -> np.ndarray:
    """``x`` with each cell missing completely at random with probability ``rate``."""
    out = np.array(x, dtype=np.float64)
    out[np.random.default_rng(seed).random(out.shape) < rate] = np.nan
    return out


def _continuous(n: int = 300, seed: int = 0) -> np.ndarray:
    """A chain 0 -> 2 -> 3 with independent columns 1 and 4."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 5))
    x[:, 2] += x[:, 0]
    x[:, 3] += x[:, 2]
    return x


def _categorical(n: int = 400, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 3, (n, 4)).astype(np.float64)
    x[:, 1] = np.where(rng.random(n) < 0.7, x[:, 0], x[:, 1])
    x[:, 2] = np.where(rng.random(n) < 0.7, x[:, 1], x[:, 2])
    return x


def _mixed(n: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    x = _continuous(n, seed)
    x[:, 0] = np.random.default_rng(seed + 1).integers(0, 3, n)
    x[:, 2] += x[:, 0]
    return x, np.array([True, False, False, False, False])


# Each shipped test, with the data it reads and its categorical mask.
TESTS = {
    "fisherz": (_continuous, None),
    "student_t": (_continuous, None),
    "kci": (lambda: _continuous(120), None),
    "rcot": (_continuous, None),
    "cmiknn": (lambda: _continuous(120), None),
    "gpdc": (lambda: _continuous(60), None),
    "chisq": (_categorical, None),
    "gsq": (_categorical, None),
    "degen_gauss": (lambda: _mixed()[0], _mixed()[1]),
    "cond_gauss": (lambda: _mixed()[0], _mixed()[1]),
}
QUERIES = [(0, 3, ()), (3, 0, (2,)), (0, 3, (2, 1)), (1, 2, (3,))]


def _block(x: np.ndarray, columns: list[int]) -> np.ndarray:
    rows = ~np.isnan(x[:, columns]).any(axis=1)
    return x[np.ix_(rows, columns)]


@pytest.mark.parametrize("name", list(TESTS))
def test_a_test_on_missing_values_is_the_test_on_the_rows_its_variables_share(name):
    make, mask = TESTS[name]
    x = _gaps(make(), 0.1)
    test = make_indep_test(name, x, categorical=mask)
    for a, b, cond in QUERIES:
        columns = sorted({a, b, *cond})
        at = {c: i for i, c in enumerate(columns)}
        sub_mask = None if mask is None else mask[columns]
        alone = make_indep_test(name, _block(x, columns), categorical=sub_mask)
        assert test(a, b, cond) == alone(at[a], at[b], [at[c] for c in cond])


@pytest.mark.parametrize("name", ["fisherz", "student_t"])
def test_a_partial_correlation_on_missing_values_reports_its_block_statistic(name):
    x = _gaps(_continuous(), 0.15)
    test = make_indep_test(name, x)
    statistic, p = test.statistic_pvalue(3, 0, [2])
    alone = make_indep_test(name, _block(x, [0, 2, 3]))
    assert (statistic, p) == alone.statistic_pvalue(0, 2, [1])
    r, ps = test.statistic_pvalues(np.array([0, 3]), np.array([3, 1]), np.array([[2], [2]]))
    assert (r[0], ps[0]) == (statistic, p)
    assert (r[1], ps[1]) == test.statistic_pvalue(1, 3, [2])


def test_a_registered_test_is_built_on_each_query_block():
    seen = []

    def factory(block):
        seen.append(block)
        return FisherZ(block)

    register_indep_test("seen", factory)
    try:
        x = _gaps(_continuous(), 0.1)
        make_indep_test("seen", x)(4, 1, [3])
    finally:
        ci._CI_REGISTRY.pop("seen")
    assert len(seen) == 1 and np.array_equal(seen[0], _block(x, [1, 3, 4]))


@pytest.mark.parametrize("name", ["fisherz", "chisq", "cond_gauss"])
def test_a_query_gives_one_p_value_whatever_its_argument_order_and_history(name):
    make, mask = TESTS[name]
    x = _gaps(make(), 0.1)
    first = make_indep_test(name, x, categorical=mask)
    later = make_indep_test(name, x, categorical=mask)
    for a, b, cond in QUERIES[::-1]:  # a different history of queries
        later(a, b, cond)
    for a, b, cond in QUERIES:
        p = first(a, b, cond)
        assert p == first(b, a, tuple(reversed(cond))) == later(a, b, cond)


def test_complete_data_get_the_test_itself():
    x = _continuous()
    assert type(make_indep_test("fisherz", x)) is FisherZ
    assert type(make_indep_test("student_t", x)) is StudentT


def test_rows_tested_reports_the_fewest_and_the_median_rows():
    assert rows_metadata({}) == {}
    assert rows_metadata({7: 1}) == {"rows_tested": {"min": 7, "median": 7.0}}
    assert rows_metadata({3: 1, 9: 1}) == {"rows_tested": {"min": 3, "median": 6.0}}
    assert rows_metadata({3: 1, 5: 2, 9: 2}) == {"rows_tested": {"min": 3, "median": 5.0}}
    x = _continuous(100)
    x[:10, 0] = np.nan  # column 0 is observed on 90 rows, the others on all 100
    test = make_indep_test("fisherz", x)
    with rows_tested() as rows:
        test(0, 1)
        test(1, 2)
        test(0, 2, [3])
    assert dict(rows) == {90: 2, 100: 1}


# --- edge cases -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "size", "least"), [("fisherz", 2, 5), ("student_t", 1, 4), ("chisq", 1, 2)]
)
def test_a_test_with_too_few_rows_raises_naming_its_minimum(name, size, least):
    make, _ = TESTS[name]
    x = make()[:40]
    x[least - 1 :, 0] = np.nan  # column 0 observed on least - 1 rows
    test = make_indep_test(name, x)
    with pytest.raises(ValueError, match=rf"on the {least - 1} rows .* needs at least {least}"):
        test(0, 1, list(range(2, 2 + size)))
    x[least - 1, 0] = 1.0  # one row more: the test runs
    assert 0.0 <= make_indep_test(name, x)(0, 1, list(range(2, 2 + size))) <= 1.0


def test_a_column_constant_on_a_test_rows_raises_for_a_continuous_test():
    x = _continuous(60)
    x[:30, 1] = 2.0
    x[30:, 0] = np.nan  # the rows where column 0 is observed hold column 1 constant
    with pytest.raises(ValueError, match="constant on the 30 rows"):
        make_indep_test("fisherz", x)(0, 1)
    assert 0.0 <= make_indep_test("fisherz", x)(1, 2) <= 1.0


def test_a_category_alone_on_a_test_rows_is_left_to_the_test():
    x = _categorical(60)
    x[:30, 1] = 2.0
    x[30:, 0] = np.nan
    assert make_indep_test("chisq", x)(0, 1) == 1.0  # no degree of freedom


def test_copies_raise_naming_the_data_columns():
    x = _continuous()
    x[:, 4] = 3.0 * x[:, 1] - 1.0
    x = _gaps(x, 0.1)
    with pytest.raises(CopiedColumnsError, match="4 repeats 1") as error:
        make_indep_test("fisherz", x)(1, 4)
    assert error.value.groups == ((1, 4),)
    frame = pd.DataFrame(x, columns=list("abcde"))
    with pytest.raises(CopiedColumnsError, match="'e' repeats 'b'"):
        andrey.pc(frame)


@pytest.mark.parametrize("name", SUPPORTED)
def test_a_column_with_no_observed_value_raises(name):
    x = _gaps(_continuous(), 0.1)
    x[:, 1] = np.nan
    frame = pd.DataFrame(x, columns=list("abcde"))
    with pytest.raises(ValueError, match="column 'b' holds no observed value"):
        _fit(name, frame)


def test_a_column_constant_on_its_observed_values_raises():
    x = _gaps(_continuous(), 0.1)
    x[~np.isnan(x[:, 2]), 2] = 5.0
    with pytest.raises(ValueError, match="column 2 is constant"):
        andrey.pc(x)


@pytest.mark.parametrize("name", SUPPORTED)
def test_inf_still_raises(name):
    x = _gaps(_continuous(), 0.1)
    x[4, 0] = np.inf
    with pytest.raises(ValueError, match=r"holds inf in 1 cell at row 4, column 0"):
        _fit(name, x)


@pytest.mark.parametrize("indep_test", ["chisq", "gsq", "cond_gauss"])
def test_a_missing_category_is_a_missing_value(indep_test):
    pd_frame = pd.DataFrame(_categorical(), columns=list("abcd"))
    pd_frame["b"] = pd_frame["b"].map({0.0: "low", 1.0: "mid", 2.0: "high"})
    pd_frame.loc[[3, 17, 40], "b"] = None
    pd_frame.loc[[5, 9], "a"] = np.nan
    out = andrey.pc(pd_frame, indep_test=indep_test)
    assert out.metadata["rows_tested"]["min"] < len(pd_frame)
    rows = pd_frame.notna().all(axis=1).to_numpy()
    assert out.metadata["rows_tested"]["min"] >= rows.sum()


def test_categories_are_inferred_from_the_observed_values():
    x, _ = _mixed()
    x = _gaps(x, 0.1)
    out = andrey.pc(pd.DataFrame(x, columns=list("abcde")), indep_test="cond_gauss")
    assert out.metadata["categorical"] == ["a"]


def test_cdnod_reads_a_complete_domain_index_and_refuses_the_independent_change_step():
    x = _gaps(_continuous(), 0.1)
    domains = np.repeat([0.0, 1.0], 150)
    assert "rows_tested" in andrey.cdnod(x, domains).metadata
    gappy = domains.copy()
    gappy[3] = np.nan
    with pytest.raises(ValueError, match="domain_index contains NaN"):
        andrey.cdnod(x, gappy)
    with pytest.raises(ValueError, match="dependence_threshold step .* does not accept missing"):
        andrey.cdnod(x, domains, dependence_threshold=0.1)


def test_a_lagged_design_reads_the_steps_its_lagged_variables_share():
    x = _gaps(_continuous(400), 0.05)
    design = lagged_design(x, 2)
    test = design_test("student_t", x, 2)
    columns = [0, 6, 13]  # variable 0 now, variable 1 one step back, variable 3 two steps back
    rows = ~np.isnan(design[:, columns]).any(axis=1)
    alone = StudentT(design[np.ix_(rows, columns)])
    statistic, p = test.statistic_pvalues(np.array([0]), np.array([13]), np.array([[6]]))
    assert (statistic[0], p[0]) == alone.statistic_pvalue(0, 2, [1])


# --- the methods ----------------------------------------------------------------------------------


def _fit(name: str, x, **kwargs):
    fit = getattr(andrey, name)
    if name == "cdnod":
        return fit(x, np.repeat([0.0, 1.0], len(x) // 2 + 1)[: len(x)], **kwargs)
    return fit(x, **kwargs)


@pytest.mark.parametrize("name", SUPPORTED)
def test_a_supported_method_runs_on_missing_values_and_repeats(name):
    x = _gaps(_continuous(400), 0.1)
    first, again = _fit(name, x), _fit(name, x)
    assert first.structure == again.structure
    assert first.metadata == again.metadata
    rows = first.metadata["rows_tested"]
    assert 0 < rows["min"] <= rows["median"] <= len(x)
    assert "rows_tested" not in _fit(name, _continuous(400)).metadata


@pytest.mark.parametrize("name", ["pc", "fci", "cdnod"])
def test_more_workers_give_the_same_result_on_missing_values(name, monkeypatch):
    # Every pass with a conditioning set goes to the pool where a pool runs.
    monkeypatch.setenv("ANDREY_SKELETON_PARALLEL_MIN_WORK", "1")
    x = _gaps(_continuous(400), 0.1)
    serial = _fit(name, x)
    with andrey.config(num_workers=2):
        assert _fit(name, x) == serial


def _refused_inputs(name: str) -> list:
    spec = REGISTRY[name]
    x = _gaps(_continuous(200), 0.05)
    args: list = []
    for inp in spec.data.inputs:
        if not inp.required:
            continue
        if inp.kind == "panel":
            args.append([x, _continuous(200, 1)])
        elif inp.kind == "context":
            args.append(np.repeat([0.0, 1.0], 100))
        elif spec.family == "pairwise":
            args.append(x[:, :2])
        else:
            args.append(x)
    return args


@pytest.mark.parametrize("name", REFUSED)
def test_every_other_method_refuses_missing_values_naming_itself(name):
    if name == "calm":
        pytest.importorskip("torch")  # without the extra, calm raises ImportError first
    pattern = (
        rf"^{name} does not accept missing values, and data\S* holds NaN .*row \d+, column \d+"
    )
    with pytest.raises(ValueError, match=pattern) as error:
        getattr(andrey, name)(*_refused_inputs(name))
    if name == "longitudinal_lingam":  # a panel: a unit's rows go together
        assert "drop the units that hold one from every dataset of data_list" in str(error.value)
    elif REGISTRY[name].family == "temporal":  # one series: a step cannot be dropped
        assert "fill in the missing steps first" in str(error.value)
        assert "use pcmci, pcmciplus, or lpcmci" in str(error.value)
    else:
        assert "drop the rows that hold one" in str(error.value)
        assert "pc, fci, cdnod, iamb, inter_iamb, or mmpc" in str(error.value)


def test_a_refused_text_category_names_the_method():
    frame = pd.DataFrame(_categorical(), columns=list("abcd"))
    frame["b"] = frame["b"].map({0.0: "low", 1.0: "mid", 2.0: "high"})
    frame.loc[3, "b"] = None
    with pytest.raises(ValueError, match=r"^hc does not accept .* column 'b' has a missing value"):
        andrey.hc(frame, score_func="local_score_BDeu")


def test_the_refusal_lists_exactly_the_supported_methods():
    from andrey.api._adapt import TESTWISE_METHODS, TESTWISE_SERIES_METHODS

    assert [*TESTWISE_METHODS, *TESTWISE_SERIES_METHODS] == SUPPORTED


@pytest.mark.parametrize("name", ["pc", "fci"])
def test_bootstrap_resamples_rows_with_missing_values(name):
    x = _gaps(_continuous(300), 0.1)
    boot = andrey.bootstrap(name, x, n_boot=4, seed=0)
    assert boot.n_boot == 4


# --- the command line -----------------------------------------------------------------------------


def _csv(tmp_path) -> str:
    x = _gaps(_continuous(300), 0.1)
    path = tmp_path / "gaps.csv"
    rows = [",".join("" if np.isnan(v) else repr(float(v)) for v in row) for row in x]
    path.write_text("a,b,c,d,e\n" + "\n".join(rows) + "\n")
    return str(path)


def test_the_cli_reads_an_empty_cell_as_a_missing_value(tmp_path, capsys):
    code = main(["run", "pc", "--data", _csv(tmp_path), "--json"])
    out = json.loads(capsys.readouterr().out)
    assert code == 0 and out["metadata"]["rows_tested"]["min"] < 300
    assert out["structure"]["labels"] == list("abcde")


def test_the_cli_reports_a_refusal_as_a_data_error(tmp_path, capsys):
    code = main(["run", "ges", "--data", _csv(tmp_path), "--json"])
    err = json.loads(capsys.readouterr().err)
    assert code == 3 and err["error"]["code"] == "data_error"
    assert "ges does not accept missing values" in err["error"]["message"]


def test_the_cli_reads_an_empty_cell_in_the_first_data_row(tmp_path, capsys):
    path = tmp_path / "first.csv"
    rows = _continuous(200)
    text = "\n".join(",".join(repr(float(v)) for v in row) for row in rows)
    path.write_text("," + text[text.index(",") + 1 :] + "\n")  # the first cell is empty
    code = main(["run", "pc", "--data", str(path), "--json"])
    assert code == 0 and json.loads(capsys.readouterr().out)["structure"]["labels"] is None


@pytest.mark.parametrize("name", SERIES)
def test_a_series_method_refits_a_series_with_missing_steps(name):
    x = _gaps(_continuous(400), 0.05)
    out = getattr(andrey, name)(x, tau_max=1)
    assert out.metadata["rows_tested"]["min"] < 400 - 2


@pytest.mark.parametrize("name", ["varma_lingam", "granger"])
def test_the_other_series_methods_refuse_missing_steps(name):
    with pytest.raises(ValueError, match=rf"^{name} does not accept missing values"):
        getattr(andrey, name)(_gaps(_continuous(400), 0.05))
