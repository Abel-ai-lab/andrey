"""The categorical primitives: the chi-square and G-square tests, the BDeu and multinomial BIC
scores, and the codes.

Each statistic is checked against an independent computation: SciPy's contingency-table test for
one table, a per-configuration loop for a conditional test, the BDeu formula written out with
``math.lgamma``, and the multinomial BIC written out per cell.
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pytest
from scipy.special import chdtrc
from scipy.stats import chi2, chi2_contingency

from andrey.core import categorical
from andrey.core.ci import ChiSquare
from andrey.core.score import BDeuScore, CategoricalBICScore, make_score


def _codes(seed: int = 0, n: int = 400, cards=(2, 3, 4, 3)) -> np.ndarray:
    """Categorical columns where column 1 copies column 0 in most rows, so some pairs depend."""
    rng = np.random.default_rng(seed)
    x = np.column_stack([rng.integers(0, k, n) for k in cards])
    x[:, 1] = np.where(rng.random(n) < 0.6, x[:, 0], x[:, 1])
    return x.astype(np.float64)


def _table(x: np.ndarray, a: int, b: int) -> np.ndarray:
    """The observed table of columns ``a`` and ``b``, without empty rows or columns."""
    va, ia = np.unique(x[:, a], return_inverse=True)
    vb, ib = np.unique(x[:, b], return_inverse=True)
    table = np.zeros((va.size, vb.size))
    np.add.at(table, (ia, ib), 1)
    return table


@pytest.mark.parametrize(("g_square", "lambda_"), [(False, None), (True, "log-likelihood")])
def test_a_marginal_test_matches_scipy(g_square, lambda_):
    x = _codes()
    test = ChiSquare(x, g_square=g_square)
    for a, b in combinations(range(x.shape[1]), 2):
        stat, p, dof, _ = chi2_contingency(_table(x, a, b), correction=False, lambda_=lambda_)
        assert test(a, b) == pytest.approx(p, rel=1e-9, abs=1e-300)


@pytest.mark.parametrize(("g_square", "lambda_"), [(False, None), (True, "log-likelihood")])
def test_a_conditional_test_sums_one_table_per_configuration(g_square, lambda_):
    """The statistic and degrees of freedom add up over the conditioning configurations, each
    table counting only the values present in it."""
    x = _codes(seed=1, n=600)
    statistic = dof = 0.0
    for config in np.unique(x[:, [2, 3]], axis=0):
        rows = x[(x[:, [2, 3]] == config).all(axis=1)]
        table = _table(rows, 0, 1)
        if min(table.shape) > 1:
            s, _, k, _ = chi2_contingency(table, correction=False, lambda_=lambda_)
            statistic, dof = statistic + s, dof + k
    assert ChiSquare(x, g_square=g_square)(0, 1, (2, 3)) == pytest.approx(
        chi2.sf(statistic, dof), rel=1e-9
    )


def test_a_pair_with_no_degrees_of_freedom_has_p_value_one():
    """When ``y`` takes one value in every configuration of the conditioning set, no table has a
    degree of freedom, and the test cannot reject independence."""
    x = _codes()
    x[:, 3] = x[:, 2]  # y is a copy of the conditioning column
    for g_square in (False, True):
        assert ChiSquare(x, g_square=g_square)(0, 3, (2,)) == 1.0


def test_dependence_and_independence_are_told_apart():
    x = _codes(n=2000)
    test = ChiSquare(x)
    assert test(0, 1) < 1e-10  # column 1 copies column 0
    assert test(2, 3) > 0.01  # independent draws


def test_p_values_do_not_depend_on_codes_or_argument_order():
    x = _codes()
    recoded = x * np.array([3.0, -2.0, 1.0, 7.0]) + 5.0
    for g in (False, True):
        a, b = ChiSquare(x, g_square=g), ChiSquare(recoded, g_square=g)
        assert a(1, 0, (3, 2, 3)) == a(0, 1, (2, 3))
        assert b(0, 1, (2, 3)) == pytest.approx(a(0, 1, (2, 3)), rel=1e-12)


def test_a_sparse_table_counts_only_the_occupied_cells(monkeypatch):
    """Above the dense-table limit the cells that occur are counted directly; same p-value."""
    from andrey.core import ci

    x = _codes(n=300, cards=(4, 5, 6, 6))
    dense = ChiSquare(x)(0, 1, (2, 3))
    monkeypatch.setattr(ci, "_DENSE_CELLS", 0)
    assert ChiSquare(x)(0, 1, (2, 3)) == pytest.approx(dense, rel=1e-12)


def _fast_path_data(case: str) -> np.ndarray:
    """Categorical columns of one of three kinds of table.

    ``"dependent"``: tables with every cell filled. ``"sparse"``: few rows and many categories,
    so most cells are empty and many configurations hold one row. ``"one value"``: a column with
    one category, and a pair whose first column takes one value in some configurations, so a test
    has zero degrees of freedom.
    """
    if case == "dependent":
        return _codes(n=400, cards=(2, 3, 4, 3, 5))
    if case == "sparse":
        return _codes(seed=2, n=15, cards=(6, 5, 7, 4, 6))
    x = _codes(seed=3, n=60, cards=(2, 3, 3, 2, 4))
    x[:, 3] = 1.0
    x[:, 0] = np.where(x[:, 2] == 0, 0.0, x[:, 0])
    return x


@pytest.mark.parametrize("g_square", [False, True], ids=["chisq", "gsq"])
@pytest.mark.parametrize("case", ["dependent", "sparse", "one value"])
@pytest.mark.parametrize(
    ("batch", "rows", "table"),
    [(1 << 22, 4, 1 << 16), (64, 4, 1 << 16), (1 << 22, 0, 40)],
    ids=["one count", "a count per set", "large tables on the scalar path"],
)
def test_every_fast_path_returns_the_scalar_p_values(
    monkeypatch, g_square, case, batch, rows, table
):
    """The batched, first-order, and all-pairs marginal paths equal the scalar call bit for bit:
    whatever the argument order, with every table in one count or one count per set and the
    marginal tables in one Gram block or several, and with tables over the size limit on the
    scalar path."""
    from andrey.core import ci

    monkeypatch.setattr(ci, "_BATCH_CELLS", batch)
    monkeypatch.setattr(ci, "_TABLE_ROWS", rows)
    monkeypatch.setattr(ci, "_TABLE_MIN", table)
    x = _fast_path_data(case)
    d = x.shape[1]
    test = ChiSquare(x, g_square=g_square)
    marginal = test.marginal_pvalues()
    for a, b in combinations(range(d), 2):
        assert marginal[a, b] == marginal[b, a] == test(a, b)
        others = [c for c in range(d) if c not in (a, b)]
        sets = [s[::-1] for size in range(4) for s in combinations(others, size)][::-1]
        assert test.batched_call(b, a, sets) == [test(a, b, s) for s in sets]
        assert test.first_order_pvalues(b, a, others).tolist() == [test(a, b, (c,)) for c in others]


def _dense_tables(data: np.ndarray, x: int, y: int, conds: list[tuple[int, ...]], g_square: bool):
    """Statistic, degrees of freedom, and p-value of each set, from one table per possible
    configuration of the set, counted together in one ``bincount``, the empty ones dropped.

    Every set has the same size. The arithmetic after the counting is the batched path's: each
    set's statistic is one ``np.sum`` over its occupied cells, in configuration, ``x``, ``y`` order.
    """
    codes, cards = categorical.encode(data)
    n = codes.shape[0]
    kx, ky = int(cards[x]), int(cards[y])
    cells = kx * ky
    sets = np.array(conds, dtype=np.intp).reshape(len(conds), -1)
    radix = cards[sets]
    q = radix.prod(axis=1)
    offset = np.cumsum(q) - q
    total = int(q.sum())
    index = np.repeat(offset[:, None], n, axis=1)
    for j in range(sets.shape[1]):
        stride = np.prod(radix[:, j + 1 :], axis=1)
        index += codes[:, sets[:, j]].T * stride[:, None]
    index = index * cells + codes[:, x] * ky + codes[:, y]
    tables = np.bincount(index.ravel(), minlength=total * cells).reshape(total, kx, ky)
    occurs = tables.reshape(total, cells).any(axis=1)
    bounds = np.concatenate(([0], np.cumsum(np.add.reduceat(occurs.astype(np.int64), offset))))
    tables = tables[occurs]
    n_sx = tables.sum(axis=2)
    n_sy = tables.sum(axis=1)
    n_s = n_sx.sum(axis=1)
    flat = tables.reshape(-1)
    occupied = np.flatnonzero(flat)
    observed = flat[occupied]
    s, rest = np.divmod(occupied, cells)
    sx, sy = np.divmod(rest, ky)
    expected = n_sx[s, sx] * n_sy[s, sy] / n_s[s]
    if g_square:
        values = observed * np.log(observed / expected)
    else:
        values = observed * (observed / expected)
    cuts = np.searchsorted(occupied, bounds * cells).tolist()
    sums = np.array([np.sum(values[a:b]) for a, b in zip(cuts[:-1], cuts[1:])])
    statistic = 2.0 * sums if g_square else sums - float(n)
    per_table = ((n_sx > 0).sum(axis=1) - 1) * ((n_sy > 0).sum(axis=1) - 1)
    dof = np.add.reduceat(per_table, bounds[:-1])
    pvalue = np.ones(len(dof))
    live = dof != 0
    pvalue[live] = chdtrc(dof[live], np.maximum(statistic[live], 0.0))
    return statistic, dof, pvalue


def _occupied_tables_data(case: str) -> np.ndarray:
    """Categorical columns whose conditioning sets leave many configurations empty.

    ``"empty configurations"``: more configurations than rows. ``"one value per configuration"``:
    column 0 takes one value wherever column 2 is 0, so those tables have an empty row and fewer
    degrees of freedom. ``"one category"``: column 1 has one category. ``"many categories"``:
    twelve categories per conditioning column, so a set of four has 20,736 configurations against
    150 rows. ``"two rows"``: every table has one cell per row, so no degree of freedom.
    """
    if case == "empty configurations":
        return _codes(seed=4, n=200, cards=(2, 3, 4, 3, 5, 2))
    if case == "one value per configuration":
        x = _codes(seed=5, n=300, cards=(3, 3, 3, 4, 2, 3))
        x[:, 0] = np.where(x[:, 2] == 0, 0.0, x[:, 0])
        return x
    if case == "one category":
        x = _codes(seed=6, n=120, cards=(3, 2, 4, 3, 2))
        x[:, 1] = 1.0
        return x
    if case == "many categories":
        return _codes(seed=7, n=150, cards=(3, 3, 12, 12, 12, 12))
    return _codes(seed=8, n=2, cards=(2, 2, 3, 3))


@pytest.mark.parametrize("g_square", [False, True], ids=["chisq", "gsq"])
@pytest.mark.parametrize(
    "case",
    [
        "empty configurations",
        "one value per configuration",
        "one category",
        "many categories",
        "two rows",
    ],
)
@pytest.mark.parametrize("batch", [1 << 22, 64], ids=["one count", "a count per set"])
def test_batched_counts_equal_the_tables_of_every_configuration(monkeypatch, g_square, case, batch):
    """Counting only the configurations that occur gives every set the statistic, degrees of
    freedom, and p-value, bit for bit, of counting a table for every possible configuration and
    dropping the empty ones, whatever the set's size."""
    from andrey.core import ci

    monkeypatch.setattr(ci, "_BATCH_CELLS", batch)
    monkeypatch.setattr(ci, "_TABLE_MIN", 1 << 24)  # every set on the batched path
    recorded: list[tuple[np.ndarray, np.ndarray]] = []

    def recording_chdtrc(dof, statistic):
        recorded.append((np.asarray(dof).copy(), np.asarray(statistic).copy()))
        return chdtrc(dof, statistic)

    monkeypatch.setattr(ci, "chdtrc", recording_chdtrc)
    x = _occupied_tables_data(case)
    d = x.shape[1]
    test = ChiSquare(x, g_square=g_square)
    for a, b in [(0, 1), (1, d - 1), (0, d - 1)]:
        others = [c for c in range(d) if c not in (a, b)]
        for size in range(min(len(others), 4) + 1):
            sets = list(combinations(others, size))
            if len(sets) == 1:
                sets = sets * 2  # a batch holds at least two sets
            statistic, dof, pvalue = _dense_tables(x, a, b, sets, g_square)
            recorded.clear()
            got = np.array(test.batched_call(b, a, sets))
            assert np.array_equal(got.view(np.int64), pvalue.view(np.int64))
            live = dof != 0
            got_dof = np.concatenate([r[0] for r in recorded]) if recorded else np.array([])
            got_statistic = np.concatenate([r[1] for r in recorded]) if recorded else np.array([])
            assert sorted(zip(got_dof.tolist(), got_statistic.view(np.int64).tolist())) == sorted(
                zip(dof[live].tolist(), np.maximum(statistic[live], 0.0).view(np.int64).tolist())
            )
    if case == "two rows":
        assert (test.batched_call(0, 1, [(2,), (3,), (2, 3)]) == np.ones(3)).all()


def _bdeu(counts: np.ndarray, q: int, a: float = 1.0) -> float:
    """Log BDeu marginal likelihood of ``counts`` (one row per parent configuration)."""
    r = counts.shape[1]
    total = 0.0
    for row in counts:
        total += math.lgamma(a / q) - math.lgamma(row.sum() + a / q)
        for c in row:
            total += math.lgamma(c + a / (r * q)) - math.lgamma(a / (r * q))
    return total


def test_bdeu_matches_the_formula():
    """Three variables, so the structure prior gives each other variable probability 1/2."""
    x = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 1, 1], [1, 1, 0], [0, 1, 1]], float)
    score = BDeuScore(x)
    half = math.log(0.5)
    # Node 0 with no parent: counts (4, 2).
    assert score.score(0, []) == pytest.approx(-2 * (_bdeu(np.array([[4, 2]]), 1) + 2 * half))
    # Node 0 given node 1: (2, 0) where node 1 is 0, (2, 2) where it is 1.
    assert score.score(0, [1]) == pytest.approx(
        -2 * (_bdeu(np.array([[2, 0], [2, 2]]), 2) + 2 * half)
    )


def test_bdeu_structure_prior_penalizes_each_parent():
    """From five variables up, p = 1 / (d - 1) makes each parent cost 2 log((1 - p) / p) more."""
    x = _codes(n=300, cards=(2, 3, 2, 3, 2))
    with_prior = BDeuScore(x)
    p = 1 / 4
    penalty = 2 * math.log((1 - p) / p)
    likelihood_gain = BDeuScore(x[:, :3]).score(0, [1]) - BDeuScore(x[:, :3]).score(0, [])
    # At d = 3 the prior is flat (p = 1/2), so the difference there is the likelihood's alone.
    assert with_prior.score(0, [1]) - with_prior.score(0, []) == pytest.approx(
        likelihood_gain + penalty
    )


def test_bdeu_is_score_equivalent():
    """``x -> y`` and ``y -> x`` describe one equivalence class, so they score the same."""
    x = _codes(n=500)
    s = BDeuScore(x)
    for a, b in combinations(range(x.shape[1]), 2):
        assert s.score(a, [b]) + s.score(b, []) == pytest.approx(s.score(b, [a]) + s.score(a, []))


@pytest.mark.parametrize("d", [1, 2])
def test_bdeu_leaves_out_an_undefined_structure_prior(d):
    """With two or fewer variables, structure_prior / (d - 1) is not below 1: no prior term."""
    x = _codes()[:, :d]
    score = BDeuScore(x)
    counts = np.bincount(x[:, 0].astype(int))[None, :]
    assert score.score(0, []) == pytest.approx(-2 * _bdeu(counts, 1))
    if d == 2:
        assert np.isfinite(score.score(0, [1]))


def test_bdeu_does_not_depend_on_the_codes():
    x = _codes()
    a, b = BDeuScore(x), BDeuScore(x * -3.0 + 2.0)
    assert b.score(1, [2, 0]) == pytest.approx(a.score(1, [0, 2]), rel=1e-12)


@pytest.mark.parametrize(("name", "value"), [("sample_prior", 0.0), ("structure_prior", -1.0)])
def test_bdeu_priors_must_be_positive(name, value):
    with pytest.raises(ValueError, match=name):
        BDeuScore(_codes(), **{name: value})


def test_configurations_number_the_observed_rows_in_code_order():
    """Many high-cardinality columns overflow a plain mixed-radix code; the configurations that
    occur are still numbered 0..m-1 in lexicographic order, at most one per row."""
    rng = np.random.default_rng(0)
    x = rng.integers(0, 50, (1000, 12)).astype(np.float64)
    x[:500, :6] = 0.0  # repeated rows, so some configurations occur more than once
    codes, cards = categorical.encode(x)
    columns = list(range(12))
    index, m = categorical.configurations(codes, cards, columns)
    _, expected = np.unique(codes[:, columns], axis=0, return_inverse=True)
    assert m == expected.max() + 1 <= len(x)
    assert np.array_equal(index, expected.reshape(-1))


def test_no_columns_is_one_configuration():
    codes, cards = categorical.encode(_codes())
    index, m = categorical.configurations(codes, cards, [])
    assert m == 1 and not index.any()


@pytest.mark.parametrize("categories", [(2.9, 2.1), (2, 3.5), (True, 3), (3, 2), (1, 4), (2,)])
def test_the_sampler_rejects_category_bounds_that_are_not_ordered_whole_numbers(categories):
    from andrey.data import sample_discrete

    with pytest.raises(ValueError, match="categories must be"):
        sample_discrete(d=3, n=20, seed=0, categories=categories)


def _multinomial_loglik(counts: np.ndarray) -> float:
    """Maximized multinomial log-likelihood of ``counts`` (one row per parent configuration)."""
    total = 0.0
    for row in counts:
        for c in row:
            if c:
                total += c * math.log(c / row.sum())
    return total


def test_categorical_bic_matches_the_formula():
    x = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 1, 1], [1, 1, 0], [0, 1, 1]], float)
    score = CategoricalBICScore(x)
    log_n = math.log(6)
    # Node 0 with no parent: counts (4, 2), one free parameter.
    assert score.score(0, []) == pytest.approx(-2 * _multinomial_loglik(np.array([[4, 2]])) + log_n)
    # Node 0 given nodes 1 and 2: four configurations, each with one free parameter.
    counts = np.array([[1, 0], [1, 0], [1, 1], [1, 1]])
    assert score.score(0, [2, 1]) == pytest.approx(-2 * _multinomial_loglik(counts) + 4 * log_n)


def test_categorical_bic_counts_the_parameters_of_unseen_configurations():
    """Two parents of 2 and 3 categories give q = 6 configurations; the 3-category node has 2
    free parameters in each, whether or not a row falls in it."""
    x = np.array([[0, 0, 0], [1, 1, 1], [2, 0, 2], [0, 1, 0], [1, 0, 2], [2, 1, 1]], float)
    x = np.vstack([x, [[0, 0, 1], [0, 0, 2]]])  # parent configuration (0, 0) seen; (1, 2) never
    score = CategoricalBICScore(x, penalty_discount=0.0)
    assert score.score(0, [1, 2]) - CategoricalBICScore(x).score(0, [1, 2]) == pytest.approx(
        -2 * 3 * 2 * math.log(len(x))
    )


def test_categorical_bic_without_a_penalty_is_minus_twice_the_log_likelihood():
    x = _codes()
    score = CategoricalBICScore(x, penalty_discount=0.0)
    counts = np.zeros((2, 3))
    np.add.at(counts, (x[:, 0].astype(int), x[:, 1].astype(int)), 1)
    assert score.score(1, [0]) == pytest.approx(-2 * _multinomial_loglik(counts), rel=1e-12)


def test_categorical_bic_penalty_scales_with_penalty_discount():
    x = _codes()
    plain, double = CategoricalBICScore(x), CategoricalBICScore(x, penalty_discount=2.0)
    free = CategoricalBICScore(x, penalty_discount=0.0)
    for parents in ([], [0], [0, 2]):
        penalty = plain.score(1, parents) - free.score(1, parents)
        assert double.score(1, parents) - free.score(1, parents) == pytest.approx(2 * penalty)


def test_categorical_bic_is_score_equivalent():
    """``x -> y`` and ``y -> x`` describe one equivalence class, so they score the same."""
    x = _codes(n=500)
    s = CategoricalBICScore(x)
    for a, b in combinations(range(x.shape[1]), 2):
        assert s.score(a, [b]) + s.score(b, []) == pytest.approx(s.score(b, [a]) + s.score(a, []))


def test_categorical_bic_does_not_depend_on_the_codes():
    x = _codes()
    a, b = CategoricalBICScore(x), CategoricalBICScore(x * -3.0 + 2.0)
    assert b.score(1, [2, 0]) == pytest.approx(a.score(1, [0, 2]), rel=1e-12)


def test_categorical_bic_and_bdeu_agree_on_which_parent_explains_a_copy():
    """Column 1 copies column 0 in most rows: both scores improve most by that parent."""
    x = _codes(n=2000)
    for score in (BDeuScore(x), CategoricalBICScore(x)):
        gains = {p: score.score(1, [p]) - score.score(1, []) for p in (0, 2, 3)}
        assert min(gains, key=gains.get) == 0 and gains[0] < 0


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_categorical_bic_penalty_discount_must_be_finite_and_non_negative(value):
    with pytest.raises(ValueError, match="penalty_discount"):
        CategoricalBICScore(_codes(), penalty_discount=value)


def test_make_score_builds_the_categorical_bic_with_its_weight():
    x = _codes()
    built = make_score("discrete_bic", x, penalty_discount=2.0)
    assert isinstance(built, CategoricalBICScore)
    assert built.score(1, [0]) == CategoricalBICScore(x, penalty_discount=2.0).score(1, [0])


def _configuration_counts(x: np.ndarray, node: int, parents: tuple[int, ...]) -> np.ndarray:
    codes, cards = categorical.encode(x)
    config, m = categorical.configurations(codes, cards, parents)
    r = int(cards[node])
    return np.bincount(config * r + codes[:, node], minlength=m * r).reshape(m, r)


def test_batched_table_counts_equal_one_numbering_per_set(monkeypatch):
    """A batch counts each set's table as :func:`configurations` numbers it, small or large, in one
    bincount or several."""
    x = _codes(n=300, cards=(2, 3, 4, 3, 5, 4))
    codes, cards = categorical.encode(x)
    sets = [(), (1,), (2, 1), (3, 4, 5), (1, 2, 3, 4, 5), (4,)]
    expected = [_configuration_counts(x, 0, s) for s in sets]
    for batch in (1 << 22, 1):  # every set in one bincount, then each in its own
        monkeypatch.setattr(categorical, "_BATCH_CELLS", batch)
        for dense in (1 << 16, 0):  # counted over every configuration, then numbered
            monkeypatch.setattr(categorical, "_DENSE_MIN", dense)
            monkeypatch.setattr(categorical, "_DENSE_ROWS", 4 if dense else 0)
            got = categorical.table_counts(codes, cards, 0, sets)
            for a, b in zip(got, expected, strict=True):
                np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("cls", [BDeuScore, CategoricalBICScore])
def test_a_score_is_the_same_alone_or_in_a_batch(cls):
    x = _codes(n=300)
    sets = [[], [1], [2, 1], [3, 1, 2]]
    alone = [cls(x).score(0, s) for s in sets]
    assert cls(x).score_many(0, sets) == alone
