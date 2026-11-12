"""The test counter: every shipped test counts each query it computes once, on every path."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

import andrey
from andrey.constraint.blanket import local_set, local_structure
from andrey.core.ci import (
    SHIPPED_INDEP_TESTS,
    count_tests,
    counting_tests,
    make_indep_test,
    register_indep_test,
)
from andrey.core.ci_missing import rows_tested
from andrey.data.latent import DSepOracle

N, D = 300, 6
#: Conditioning sets of one size, none repeated, for the pair (0, 1).
SETS = [[2, 3], [2, 4], [3, 5], [4, 5]]


def data_for(name: str, *, gaps: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
    """A chain over ``D`` columns that suits test ``name``, and its categorical mask."""
    rng = np.random.default_rng(0)
    x = np.empty((N, D))
    x[:, 0] = rng.standard_normal(N)
    for j in range(1, D):
        x[:, j] = 0.6 * x[:, j - 1] + rng.standard_normal(N)
    mask = None
    if name in ("chisq", "gsq"):
        x = np.digitize(x, [-0.5, 0.5]).astype(float)
    elif name in ("degen_gauss", "cond_gauss"):
        x[:, 2] = np.digitize(x[:, 2], [0.0])
        mask = np.arange(D) == 2
    if gaps:
        x[rng.random(x.shape) < 0.05] = np.nan
    return x, mask


def built(name: str, *, gaps: bool = False):
    x, mask = data_for(name, gaps=gaps)
    return make_indep_test(name, x, categorical=mask, seed=0)


def counted(call) -> int:
    with counting_tests() as tests:
        call()
    return tests.n


#: Each many-query path, how to call it on ``SETS``, and how many queries that asks.
PATHS = {
    "batched_call": (lambda t: t.batched_call(0, 1, SETS), len(SETS)),
    "statistic_pvalues": (
        lambda t: t.statistic_pvalues(
            np.zeros(len(SETS), dtype=np.intp), np.ones(len(SETS), dtype=np.intp), np.array(SETS)
        ),
        len(SETS),
    ),
    "pvalues_given": (lambda t: t.pvalues_given(0, [1, 4, 5], [2, 3]), 3),
    "first_order_pvalues": (lambda t: t.first_order_pvalues(0, 1, [2, 3, 4]), 3),
    "_canonical_pvalues": (
        lambda t: t._canonical_pvalues(
            np.zeros(len(SETS), dtype=np.intp), np.ones(len(SETS), dtype=np.intp), np.array(SETS)
        ),
        len(SETS),
    ),
    "residual_statistic_pvalues": (
        lambda t: t.residual_statistic_pvalues(
            data_for("fisherz")[0], np.array([0, 0]), np.array([1, 2]), np.array([[3], [4]])
        ),
        2,
    ),
    "marginal_pvalues": (lambda t: t.marginal_pvalues(), D * (D - 1) // 2),
}


@pytest.mark.filterwarnings("ignore::andrey.SampleSizeWarning")
@pytest.mark.parametrize("name", SHIPPED_INDEP_TESTS)
@pytest.mark.parametrize("gaps", [False, True], ids=["complete", "missing"])
def test_each_path_counts_each_query_once(name, gaps):
    if gaps and name in ("kci", "gpdc", "cmiknn"):
        pytest.skip("slow on missing values; the wrapper is the one the other tests take")
    test = built(name, gaps=gaps)
    assert counted(lambda: test(0, 1, [2])) == 1
    if hasattr(test, "statistic_pvalue"):
        assert counted(lambda: test.statistic_pvalue(0, 1, [3])) == 1
    for path, (call, queries) in PATHS.items():
        if hasattr(test, path):
            assert counted(lambda call=call: call(test)) == queries, path


@pytest.mark.parametrize("name", SHIPPED_INDEP_TESTS)
def test_separating_prefix_counts_the_queries_computed(name):
    """A batch computes every set, the scalar loop stops at the first separating one."""
    test = built(name)
    if not hasattr(test, "separating_prefix"):
        pytest.skip("no separating_prefix")
    with counting_tests() as tests:
        found = test.separating_prefix(0, 2, [[1], [3], [4]], 0.05)
    assert tests.n == (3 if hasattr(test, "batched_call") else len(found))


@pytest.mark.parametrize("name", ["chisq", "gsq"])
def test_a_kept_marginal_answer_is_not_a_new_test(name):
    test = built(name)
    assert counted(lambda: test(0, 3)) == 1
    assert counted(lambda: (test(3, 0), test.pvalues_given(0, [3, 4]))) == 1


def test_a_registered_test_counts_what_it_reports():
    """A registered test that runs a shipped one counts that test's queries; one that runs none
    counts what it reports."""
    fisherz = make_indep_test("fisherz", data_for("fisherz")[0])

    class Plain:
        def __call__(self, x, y, condition_set=()):
            return fisherz.batched_call(x, y, [list(condition_set), list(condition_set)])[0]

    register_indep_test("plain_counted", lambda data: Plain())
    assert counted(lambda: make_indep_test("plain_counted", np.zeros((3, 3)))(0, 1)) == 2

    class Quiet:
        def __call__(self, x, y, condition_set=()):
            return 0.5

    class Reports(Quiet):
        def __call__(self, x, y, condition_set=()):
            count_tests()
            return 0.5

    assert counted(lambda: Quiet()(0, 1)) == 0
    assert counted(lambda: Reports()(0, 1)) == 1


def test_blocks_nest_and_outside_nothing_counts():
    test = built("fisherz")
    test(0, 1, [2])  # outside every block: nothing to count into, and no error
    with counting_tests() as outer:
        test(0, 1, [2])
        with counting_tests() as inner:
            test.batched_call(0, 1, SETS)
        assert inner.n == len(SETS) and outer.n == 1 + len(SETS)
        outer.add(10)
    assert outer.n == 11 + len(SETS)


def test_the_oracle_counts_each_query_it_answers_afresh():
    oracle = DSepOracle(nx.DiGraph([(0, 1), (1, 2)]))
    assert counted(lambda: (oracle(0, 2, [1]), oracle(0, 2, [1]), oracle(0, 2))) == 2


@pytest.mark.parametrize("name", ["fisherz", "student_t"])
@pytest.mark.parametrize(("method", "structure"), [("iamb", "cpdag"), ("mmpc", "cpdag")])
def test_with_missing_values_every_query_is_one_test(name, method, structure):
    """Test-wise deletion counts each query's rows once, which counts the queries apart."""
    x, _ = data_for(name, gaps=True)
    with rows_tested() as rows, counting_tests() as tests:
        local_structure(x, method, structure, alpha=0.05, indep_test=name)
    assert tests.n == sum(rows.values()) > 0


@pytest.mark.parametrize("method", ["iamb", "inter_iamb", "gs", "fast_iamb"])
def test_a_blanket_graph_counts_each_variable_s_search(method):
    """The moral graph runs each variable's search once, the tests of a run with that target."""
    x, _ = data_for("fisherz")
    whole = getattr(andrey, method)(x, structure="moral").metadata["n_tests"]
    assert whole == sum(getattr(andrey, method)(x, target=t).metadata["n_tests"] for t in range(D))


def test_iamb_under_the_oracle_counts_each_new_query():
    """0 -> 1 -> 2 under d-separation: IAMB on 0 asks 0, 1 and 0, 2 marginally, then 0, 2 given
    1, and its shrink asks 0, 1 again, which the oracle answers from what it kept."""
    oracle = DSepOracle(nx.DiGraph([(0, 1), (1, 2)]))
    register_indep_test("dsep_count", lambda data: oracle)
    with counting_tests() as tests:
        found = local_set(np.zeros((10, 3)), "iamb", 0, alpha=0.05, indep_test="dsep_count")
    assert found == [1] and tests.n == 3


@pytest.mark.parametrize("method", ["iamb", "inter_iamb", "gs", "fast_iamb", "mmpc", "hiton_pc"])
def test_the_count_does_not_depend_on_the_column_order(method):
    x, _ = data_for("fisherz")
    order = np.random.default_rng(1).permutation(D)
    fit = getattr(andrey, method)
    assert fit(x).metadata["n_tests"] == fit(x[:, order]).metadata["n_tests"]
    t = int(np.flatnonzero(order == 2)[0])
    assert fit(x, target=2).metadata["n_tests"] == fit(x[:, order], target=t).metadata["n_tests"]
