"""ICP's invariance tests and outputs equal a direct computation, and its arguments behave as
documented.

The reference fits every regression on the rows themselves: the approximate test by Welch's t-test
and the F-test on the pooled residuals, the exact test by Chow's statistic with its full
covariance matrix (Peters, Buhlmann and Meinshausen 2016, eq. 17), and the intervals by the
regression's covariance. ICP computes the same numbers from Gram matrices.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import andrey
from andrey.core.score import CopiedColumnsError


def fit(data, env, **kwargs) -> andrey.StructureOutput:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", andrey.ExperimentalWarning)
        return andrey.icp(data, env, **kwargs)


def _data(sizes: tuple[int, ...], seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """``x0 -> x1 -> y -> x3`` and ``x0 -> y``, with environment ``k`` scaling the noise of ``x0``
    and shifting ``x1`` by ``k``."""
    rng = np.random.default_rng(seed)
    env = np.repeat(np.arange(len(sizes)), sizes)
    n = env.size
    x0 = rng.standard_normal(n) * (1.0 + env)
    x1 = 0.6 * x0 + rng.standard_normal(n) + 0.8 * env
    y = 0.5 * x0 - 0.7 * x1 + rng.standard_normal(n)
    x3 = 0.9 * y + rng.standard_normal(n)
    return np.column_stack([x0, x1, y, x3]), env


def _residuals(X, y):
    design = np.column_stack([np.ones(len(y)), X])
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    return y - design @ coef, coef, design


def _set_p_value(X, y, env, test):
    """The p-value of one set's invariance test, from the rows."""
    labels = np.unique(env)
    p = []
    for e in labels:
        inside = env == e
        if test == "approximate":
            r, _, _ = _residuals(X, y)
            a, b = r[inside], r[~inside]
            p_mean = stats.ttest_ind(a, b, equal_var=False).pvalue
            ratio = a.var(ddof=1) / b.var(ddof=1)
            f = stats.f(len(a) - 1, len(b) - 1)
            p.append(2 * min(p_mean, 2 * min(f.cdf(ratio), f.sf(ratio))))
        else:
            r, coef, design = _residuals(X[~inside], y[~inside])
            out_design = design
            in_design = np.column_stack([np.ones(inside.sum()), X[inside]])
            D = y[inside] - in_design @ coef
            cov = np.eye(inside.sum()) + in_design @ np.linalg.solve(
                out_design.T @ out_design, in_design.T
            )
            df = (~inside).sum() - in_design.shape[1]
            statistic = D @ np.linalg.solve(cov, D) / (inside.sum() * (r @ r / df))
            p.append(stats.f.sf(statistic, inside.sum(), df))
    factor = 1 if (test == "approximate" and len(labels) == 2) else len(labels)
    return min(1.0, factor * min(p))


def _reference(X, env, target, test, alpha, max_parents=None):
    """Every output of ICP from the per-set p-values and regressions on the rows."""
    others = [j for j in range(X.shape[1]) if j != target]
    y = X[:, target]
    largest = len(others) if max_parents is None else max_parents
    tested = [S for s in range(largest + 1) for S in itertools.combinations(others, s)]
    p = {S: _set_p_value(X[:, list(S)], y, env, test) for S in tested}
    accepted = [S for S in tested if p[S] > alpha]
    p_values = [
        None if j == target else max(v for S, v in p.items() if j not in S)
        for j in range(X.shape[1])
    ]
    low, high = np.full(X.shape[1], np.inf), np.full(X.shape[1], -np.inf)
    for S in accepted:
        r, coef, design = _residuals(X[:, list(S)], y)
        n, k = design.shape
        se = np.sqrt(r @ r / (n - k) * np.diag(np.linalg.inv(design.T @ design)))
        q = stats.t.ppf(1 - alpha / (2 * len(S)), n - k) if S else 0.0
        for j in others:
            at = S.index(j) + 1 if j in S else None
            centre, half = (coef[at], q * se[at]) if at else (0.0, 0.0)
            low[j], high[j] = min(low[j], centre - half), max(high[j], centre + half)
    parents = sorted(set(others).intersection(*map(set, accepted))) if accepted else []
    ones = [None if j == target else 1.0 for j in range(X.shape[1])]
    return {
        "parents": parents,
        "accepted_sets": [list(S) for S in accepted],
        "p_values": p_values if accepted else ones,
        "best_p_value": max(p.values()),
        "intervals": [None if j == target else [low[j], high[j]] for j in range(X.shape[1])],
    }


@pytest.mark.parametrize("max_parents", [None, 1])
@pytest.mark.parametrize("invariance_test", ["approximate", "exact"])
@pytest.mark.parametrize("sizes", [(70, 50), (40, 60, 50)], ids=["two envs", "three envs"])
def test_every_output_equals_the_direct_computation(sizes, invariance_test, max_parents):
    X, env = _data(sizes)
    out = fit(X, env, target=2, invariance_test=invariance_test, max_parents=max_parents)
    want = _reference(X, env, 2, invariance_test, 0.05, max_parents)
    meta = out.metadata
    assert meta["accepted_sets"] == want["accepted_sets"] and meta["accepted_sets"]
    assert meta["parents"] == want["parents"]
    assert meta["p_values"][2] is None
    np.testing.assert_allclose(
        [v for v in meta["p_values"] if v is not None],
        [v for v in want["p_values"] if v is not None],
        rtol=1e-9,
        atol=1e-14,
    )
    np.testing.assert_allclose(meta["best_p_value"], want["best_p_value"], rtol=1e-9)
    got = [v for v in meta["confidence_intervals"] if v is not None]
    np.testing.assert_allclose(got, [v for v in want["intervals"] if v is not None], rtol=1e-9)
    assert meta["model_rejected"] is False
    edges = [(j, 2, "directed") for j in want["parents"]]
    assert out.structure.oriented_edges() == edges


def test_two_environments_compare_once_under_the_approximate_test():
    """With two environments, each one against the other is the same test, so the factor is 1."""
    X, env = _data((70, 50))
    y, rows = X[:, 2], X[:, [0]]
    r, _, _ = _residuals(rows, y)
    first = stats.ttest_ind(r[env == 0], r[env == 1], equal_var=False).pvalue
    second = stats.ttest_ind(r[env == 1], r[env == 0], equal_var=False).pvalue
    assert first == pytest.approx(second, rel=1e-12)
    assert _set_p_value(rows, y, env, "approximate") < 1.0


def test_no_invariant_set_rejects_the_model():
    X, env = _data((200, 200))
    X[env == 1, 2] += 3.0  # an intervention on the target
    out = fit(X, env, target=2)
    meta = out.metadata
    assert meta["model_rejected"] and meta["parents"] == [] and meta["accepted_sets"] == []
    assert meta["p_values"] == [1.0, 1.0, None, 1.0]
    assert meta["confidence_intervals"] is None
    assert meta["best_p_value"] <= 0.05
    assert out.structure.oriented_edges() == []


def test_targets_leave_out_the_environments_that_intervened_on_the_target():
    X, env = _data((80, 80, 80))
    X[env == 1, 2] += 3.0
    frame = pd.DataFrame(X, columns=["a", "b", "y", "c"])
    labels = np.array(["obs", "hit", "other"])[env]
    out = fit(frame, labels, target="y", targets={"obs": [], "hit": ["y", "a"]})
    kept = env != 1
    alone = fit(frame[kept], labels[kept], target="y")
    assert out.metadata["environments_left_out"] == ["hit"]
    assert {**out.metadata, "environments_left_out": []} == alone.metadata


def test_a_dataframe_names_the_outputs_and_an_env_column_leaves_the_data():
    X, env = _data((60, 60))
    frame = pd.DataFrame(X, columns=["a", "b", "y", "c"]).assign(site=env)
    out = fit(frame, "site", target="y", max_parents=1)
    assert out.structure.labels == ("a", "b", "y", "c")
    assert out.metadata["target"] == "y"
    assert all(isinstance(j, str) for s in out.metadata["accepted_sets"] for j in s)
    by_position = fit(X, env, target=2, max_parents=1).metadata
    names = ["a", "b", "y", "c"]
    assert out.metadata["parents"] == [names[j] for j in by_position["parents"]]
    assert out.metadata["p_values"] == by_position["p_values"]


def _one_row_env():
    X, env = _data((60, 60))
    return X, np.r_[env[:-2], 2, 2]


ERRORS = {
    "no target": (lambda X, env: fit(X, env), r"icp: pass target"),
    "a target data lack": (lambda X, env: fit(X, env, target=7), r"target"),
    "one environment": (lambda X, env: fit(X, np.zeros(len(X)), target=2), r"at least two"),
    "a small environment": (lambda X, env: fit(*_one_row_env(), target=2), r"at least 3"),
    "every other environment left out": (
        lambda X, env: fit(X, env, target=2, targets={0: [2], 1: []}),
        r"intervenes on target 2 .*left out.*1 environment remains",
    ),
    "a predictor constant outside an environment": (
        lambda X, env: fit(
            np.column_stack([X, np.where(env == 0, np.arange(len(X)), 1.0)]),
            env,
            target=2,
            invariance_test="exact",
        ),
        r"columns 4 are constant on the rows outside 0",
    ),
    "too few rows for the sets": (
        lambda X, env: fit(
            np.random.default_rng(0).standard_normal((8, 9)), np.arange(8) % 2, target=0
        ),
        r"sets of 8 variables need at least 10 rows",
    ),
}


@pytest.mark.parametrize("case", list(ERRORS))
def test_an_unusable_setting_raises_a_value_error_naming_it(case):
    X, env = _data((60, 60))
    call, message = ERRORS[case]
    with pytest.raises(ValueError, match=message):
        call(X, env)


def test_a_copy_of_a_column_raises_before_the_fit():
    X, env = _data((60, 60))
    frame = pd.DataFrame(np.column_stack([X, 2 * X[:, 0] + 1]), columns=["a", "b", "y", "c", "a2"])
    with pytest.raises(CopiedColumnsError, match="'a2' repeats 'a'"):
        fit(frame, env, target="y")
