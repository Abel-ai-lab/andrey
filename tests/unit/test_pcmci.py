"""PCMCI engine: the lagged design, the PC1 rounds, and the MCI conditioning sets on hand cases.

A scripted test answers each query from a table, so the expected parents and conditioning sets
follow from the algorithm alone, with no data.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from andrey.temporal.pcmci import lagged_design, mci, pc1, pcmci


class Scripted:
    """A test that answers ``(x, sorted conditions)`` from a table and records every query."""

    def __init__(self, answers: dict[tuple[int, tuple[int, ...]], tuple[float, float]]) -> None:
        self.answers = answers
        self.queries: list[tuple[int, int, tuple[int, ...]]] = []

    def statistic_pvalue(self, x, y, condition_set=()):
        conditions = tuple(sorted(condition_set))
        self.queries.append((x, y, conditions))
        return self.answers.get((x, conditions), (0.0, 1.0))


def test_lagged_design_puts_variable_i_at_lag_k_in_column_k_n_plus_i():
    data = np.arange(30.0).reshape(10, 3)
    design = lagged_design(data, tau_max=2)
    assert design.shape == (10 - 4, 3 * 5)
    for lag in range(5):
        for i in range(3):
            np.testing.assert_array_equal(design[:, lag * 3 + i], data[4 - lag : 10 - lag, i])


def test_pc1_conditions_on_the_strongest_others_and_drops_after_each_round():
    # Three candidates for target 0 at tau_max=1: A=(0, 1), B=(1, 1), C=(2, 1), in columns 3, 4, 5.
    A, B, C = 3, 4, 5
    test = Scripted(
        {
            (A, ()): (0.2, 0.001),
            (B, ()): (0.6, 0.001),
            (C, ()): (0.4, 0.001),
            # Round 1 runs in strength order B, C, A, each given the strongest other candidate.
            (B, (C,)): (0.05, 0.6),  # B tests independent: dropped after the round
            (C, (B,)): (0.3, 0.001),  # still conditioned on B, the drop is not yet applied
            (A, (B,)): (0.35, 0.01),  # A's strength stays 0.2, its smallest so far
        }
    )
    parents = pc1(test, n_vars=3, target=0, tau_max=1, pc_alpha=0.05)
    assert parents == ((2, 1), (0, 1))  # C (0.3) before A (0.2)
    assert test.queries == [
        (A, 0, ()),
        (B, 0, ()),
        (C, 0, ()),
        (B, 0, (C,)),
        (C, 0, (B,)),
        (A, 0, (B,)),
    ]  # round 2 would need two others, but only C and A remain: PC1 stops


def test_pc1_keeps_a_candidate_whose_p_value_equals_pc_alpha():
    test = Scripted({(1, ()): (0.5, 0.05)})
    assert pc1(test, n_vars=1, target=0, tau_max=1, pc_alpha=0.05) == ((0, 1),)


def test_mci_conditions_on_both_ends_parents_with_the_cause_shifted_by_the_lag():
    # Two variables, tau_max=2: column lag * 2 + i. Parents of 0: (0, 1); of 1: (1, 1), (0, 2).
    parents = (((0, 1),), ((1, 1), (0, 2)))
    answers = {
        (2, (3, 4)): (0.7, 0.001),  # (0, 1) -> 1 given (1, 1), (0, 2) and (0, 2) shifted from 0
        (5, (3, 4, 7, 8)): (-0.1, 0.3),  # (1, 2) -> 1 given 1's parents, and them shifted by 2
    }
    test = Scripted(answers)
    statistics, p_values = mci(test, n_vars=2, parents=parents, tau_max=2)
    conditions = {(x, y): c for x, y, c in test.queries}
    assert conditions[(2, 1)] == (3, 4)
    assert conditions[(5, 1)] == (3, 4, 7, 8)
    assert conditions[(2, 0)] == (4,)  # (0, 1) -> 0: its own parent dropped, (0, 2) from the shift
    assert conditions[(3, 0)] == (2, 5, 6)  # (1, 1) -> 0: (0, 1), then (1, 2) and (0, 3)
    assert len(test.queries) == 2 * 2 * 2
    assert statistics[0, 0, 1] == 0.7 and p_values[0, 0, 1] == 0.001
    assert statistics[1, 1, 1] == -0.1 and p_values[1, 1, 1] == 0.3


def _chain(n_steps: int = 600, seed: int = 0) -> np.ndarray:
    """X0 -> X1 at lag 1 and X1 -> X2 at lag 2, each series autocorrelated."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((n_steps + 50, 3))
    x = np.zeros_like(e)
    for t in range(2, len(x)):
        x[t, 0] = 0.7 * x[t - 1, 0] + e[t, 0]
        x[t, 1] = 0.5 * x[t - 1, 1] + 0.6 * x[t - 1, 0] + e[t, 1]
        x[t, 2] = 0.4 * x[t - 1, 2] - 0.6 * x[t - 2, 1] + e[t, 2]
    return x[50:]


def test_pcmci_finds_the_chain_links_and_the_self_dependence():
    result = pcmci(_chain(), tau_max=2, pc_alpha=0.2, alpha_level=0.01)
    links = {(int(i), int(j), int(k) + 1) for k, i, j in np.argwhere(result.graph)}
    assert links == {(0, 0, 1), (1, 1, 1), (2, 2, 1), (0, 1, 1), (1, 2, 2)}
    assert set(result.parents[2]) >= {(2, 1), (1, 2)}
    assert result.graph.shape == result.p_values.shape == result.statistics.shape == (2, 3, 3)


@pytest.mark.parametrize("alpha_level", [0.01, 0.2])
def test_the_graph_is_the_p_values_at_most_alpha_level(alpha_level):
    result = pcmci(_chain(300), tau_max=2, alpha_level=alpha_level)
    np.testing.assert_array_equal(result.graph, result.p_values <= alpha_level)


def _partial_correlation(design, x, y, conditions):
    """Correlation of the least-squares residuals of columns ``x`` and ``y`` on the conditions."""
    z = np.column_stack([np.ones(len(design)), design[:, conditions]])
    residuals = [design[:, k] - z @ np.linalg.lstsq(z, design[:, k])[0] for k in (x, y)]
    return np.corrcoef(*residuals)[0, 1]


def test_default_p_values_are_the_student_t_tail_of_each_links_partial_correlation():
    data, tau_max, n_vars = _chain(300), 2, 3
    result = pcmci(data, tau_max=tau_max, pc_alpha=0.2)
    design = lagged_design(data, tau_max)
    for lag in range(1, tau_max + 1):
        for i in range(n_vars):
            for j in range(n_vars):
                conditions = {c for c in result.parents[j] if c != (i, lag)}
                conditions.update((k, s + lag) for k, s in result.parents[i])
                columns = sorted(s * n_vars + k for k, s in conditions)
                r = _partial_correlation(design, lag * n_vars + i, j, columns)
                df = len(design) - len(columns) - 2
                p = 2 * stats.t.sf(abs(r) * np.sqrt(df / (1 - r * r)), df)
                assert result.statistics[lag - 1, i, j] == pytest.approx(r, abs=1e-10)
                assert result.p_values[lag - 1, i, j] == pytest.approx(p, rel=1e-6, abs=1e-300)


# Fisher-Z results recorded on _chain(300) before the Student-t test existed:
# {(lag - 1, cause, effect): p_value}.
FISHER_Z_BASELINE = {
    (0, 1, 0): 0.06821704538462758,
    (0, 1, 1): 4.9960036108132044e-14,
    (0, 2, 1): 0.02602403693306332,
    (1, 0, 1): 0.19187550024330813,
    (1, 1, 1): 0.5452899840069427,
    (1, 2, 2): 0.7140314703736454,
}


def test_fisherz_reproduces_the_recorded_results():
    result = pcmci(_chain(300), tau_max=2, pc_alpha=0.2, alpha_level=0.01, indep_test="fisherz")
    assert result.parents == (
        ((0, 1), (2, 1), (1, 1)),
        ((0, 1), (1, 1), (0, 2), (2, 2)),
        ((1, 2), (2, 1)),
    )
    for index, p_value in FISHER_Z_BASELINE.items():
        assert result.p_values[index] == pytest.approx(p_value, abs=1e-12)


@pytest.mark.parametrize(("n_steps", "alpha_level"), [(600, 0.01), (300, 0.01), (300, 0.2)])
def test_both_tests_give_the_same_graph_on_the_fixed_cases(n_steps, alpha_level):
    data = _chain(n_steps)
    student_t, fisherz = (
        pcmci(data, tau_max=2, alpha_level=alpha_level, indep_test=name)
        for name in ("student_t", "fisherz")
    )
    np.testing.assert_array_equal(student_t.graph, fisherz.graph)
    assert student_t.parents == fisherz.parents
    assert not np.array_equal(student_t.p_values, fisherz.p_values)


def test_an_unknown_indep_test_raises_naming_the_choices():
    with pytest.raises(ValueError, match=r"indep_test must be one of \['fisherz', 'student_t'\]"):
        pcmci(_chain(100), indep_test="spearman")
