"""J-PCMCI+ recovery: planted context and dummy links, the links they explain, and the null case."""

from __future__ import annotations

import functools

import numpy as np
import pytest

import andrey
from andrey.core import ARROW, NULL, TAIL
from andrey.data import sample_context_series
from tests.recovery.helpers import datagen
from tests.recovery.helpers.harness import assert_recorded_output

N_DATASETS = 10


def _lagged(out: andrey.StructureOutput) -> set[tuple[int, int, int]]:
    """Every ``(cause, effect, lag)`` link at a positive lag: an arrowhead at the effect."""
    return {
        (int(i), int(j), lag)
        for lag in out.structure.lags
        if lag > 0
        for j, i in np.argwhere(out.structure.lag(lag).to_numpy() == ARROW)
    }


def _same_step(out: andrey.StructureOutput) -> set[tuple[int, int, str]]:
    """Every lag-0 link as ``(i, j, "->")`` for ``i -> j``, or ``(i < j, "--")``."""
    m = out.structure.lag(0).to_numpy()
    links = set()
    for i, j in zip(*np.nonzero(np.triu(m != NULL))):
        i, j = int(i), int(j)
        if m[i, j] == TAIL and m[j, i] == TAIL:
            links.add((i, j, "--"))
        else:
            links.add((i, j, "->") if m[j, i] == ARROW else (j, i, "->"))
    return links


def _noise(rng: np.random.Generator, n_steps: int, n_columns: int = 1) -> np.ndarray:
    return rng.standard_normal((N_DATASETS, n_steps, n_columns))


def test_the_recorded_output_is_reproduced():
    out = andrey.jpcmciplus(datagen.generate_groups("context_var_4v"), tau_max=1)
    assert_recorded_output(out, "JPCMCIplus", case="context_var_4v")


def test_independent_series_have_no_links():
    data = list(np.random.default_rng(1).standard_normal((N_DATASETS, 60, 3)))
    out = andrey.jpcmciplus(data)
    assert not _same_step(out) and not _lagged(out)
    assert not out.metadata["index_edges"] and not out.metadata["time_index_edges"]
    assert not out.metadata["context_links"]


def test_space_dummy_removes_the_link_a_hidden_space_context_adds():
    rng = np.random.default_rng(1)
    hidden = 3.0 * rng.standard_normal((N_DATASETS, 1, 1))
    data = list(hidden + _noise(rng, 100, 2))
    pooled = andrey.pcmciplus(np.vstack(data))
    assert (0, 1, "--") in _same_step(pooled)
    out = andrey.jpcmciplus(data, time_dummy=False)
    assert not _same_step(out)
    assert out.metadata["index_edges"] == [[0, "directed"], [1, "directed"]]


def test_time_dummy_removes_the_link_a_hidden_time_context_adds():
    rng = np.random.default_rng(2)
    hidden = 2.0 * rng.standard_normal((1, 60, 1))
    data = list(hidden + _noise(rng, 60, 2))
    assert (0, 1, "--") in _same_step(andrey.jpcmciplus(data, time_dummy=False))
    out = andrey.jpcmciplus(data)
    assert not _same_step(out)
    assert out.metadata["time_index_edges"] == [[0, "directed"], [1, "directed"]]


@functools.cache
def _context_fit() -> andrey.StructureOutput:
    """A space context into ``x``, ``x -> y`` at lag 0, and a time context into ``y`` at lag 1."""
    rng = np.random.default_rng(3)
    space = rng.standard_normal(N_DATASETS)
    time = rng.standard_normal(200)
    data = []
    for level in space:
        x = level + rng.standard_normal(200)
        y = 0.8 * x + 0.8 * np.r_[0.0, time[:-1]] + rng.standard_normal(200)
        data.append(np.column_stack([x, time, y, np.full(200, level)]))
    return andrey.jpcmciplus(
        data, time_contexts=[1], space_contexts=[3], time_dummy=False, alpha=0.01
    )


def test_context_links_are_found_at_their_lags():
    assert sorted(_context_fit().metadata["context_links"]) == [[1, 1, 1], [3, 0, 0]]


def test_a_context_link_orients_the_link_between_system_variables():
    # PCMCI+ cannot orient x - y from x and y alone; C -> x - y, with C and y apart, gives x -> y.
    assert _same_step(_context_fit()) == {(0, 1, "->")}


def test_the_structure_holds_the_system_variables_only():
    out = _context_fit()
    assert out.structure.lag(0).to_numpy().shape == (2, 2)
    assert np.shape(out.metadata["p_values"]) == (2, 2, 2)


def test_a_dataset_level_is_no_autocorrelation():
    rng = np.random.default_rng(4)
    levels = 2.0 * rng.standard_normal((N_DATASETS, 1, 1))
    data = list(levels + _noise(rng, 100, 2))
    assert (0, 0, 1) in _lagged(andrey.pcmciplus(np.vstack(data)))
    assert not _lagged(andrey.jpcmciplus(data, time_dummy=False))


def _seasons(n_steps: int = 240) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """A sine, a cosine, and two noisy variables that share the sine, for each of 6 datasets."""
    rng = np.random.default_rng(0)
    angle = 2 * np.pi * np.arange(n_steps) / 12
    sine, cosine = np.sin(angle), np.cos(angle)
    noise = rng.standard_normal((6, n_steps, 2))
    return sine, cosine, [2 * sine[:, None] + block for block in noise]


def test_a_context_its_lags_determine_keeps_its_links():
    sine, cosine, blocks = _seasons()
    data = [np.column_stack([block, sine, cosine]) for block in blocks]
    out = andrey.jpcmciplus(data, time_contexts=[2, 3], time_dummy=False)
    links = out.metadata["context_links"]
    assert [2, 0, 0] in links and [2, 0, 1] in links
    assert {variable for _, _, variable in links} == {0, 1}
    assert not _same_step(out)


@pytest.mark.parametrize("tau_max", [1, 2])
def test_a_sine_alone_keeps_its_links_at_any_lag(tau_max):
    sine, _, blocks = _seasons()
    data = [np.column_stack([block, sine]) for block in blocks]
    out = andrey.jpcmciplus(data, time_contexts=[2], tau_max=tau_max, time_dummy=False)
    assert {variable for _, _, variable in out.metadata["context_links"]} == {0, 1}
    assert not _same_step(out)


def test_a_sine_its_parents_determine_is_read_without_them():
    sine, _, blocks = _seasons()
    data = [np.column_stack([block, sine]) for block in blocks]
    out = andrey.jpcmciplus(data, time_contexts=[2], time_dummy=False)
    assert out.metadata["context_links"] == [[2, 0, 0], [2, 0, 1]]


@pytest.mark.parametrize("rule", ["conservative", "majority"])
@pytest.mark.parametrize("cap", [None, 2, 1])
def test_a_context_its_dummy_fixes_leaves_a_link_undirected(rule, cap):
    # The space context drives x alone, and hidden levels give x and y dummy links, which fix the
    # context in every test of the pair that the vote reads: the triple through x is ambiguous,
    # however many subsets a cap leaves.
    space = np.array([1, 1, 1, 1, -1, -1, -1, -1.0])
    for_y = np.array([1, 1, -1, -1, 1, 1, -1, -1.0])
    for_x = np.array([1, -1, 1, -1, 1, -1, 1, -1.0])
    rng = np.random.default_rng(0)
    data = []
    for m in range(8):
        x = space[m] + 2 * for_x[m] + rng.standard_normal(300)
        y = 0.8 * x + 2 * for_y[m] + rng.standard_normal(300)
        z = 0.8 * y + rng.standard_normal(300)
        data.append(np.column_stack([x, y, z, np.full(300, space[m])]))
    out = andrey.jpcmciplus(
        data, space_contexts=[3], collider_rule=rule, max_p_global=cap, time_dummy=False
    )
    assert (0, 1, "--") in _same_step(out)


@pytest.mark.parametrize("rule", ["sepsets", "conservative", "majority"])
def test_every_collider_rule_orients_a_collider(rule):
    rng = np.random.default_rng(5)
    data = []
    for _ in range(N_DATASETS):
        x, z = rng.standard_normal((2, 100))
        y = 0.8 * x + 0.8 * z + rng.standard_normal(100)
        data.append(np.column_stack([x, y, z]))
    assert _same_step(andrey.jpcmciplus(data, collider_rule=rule, time_dummy=False)) == {
        (0, 1, "->"),
        (2, 1, "->"),
    }


def test_a_hidden_context_leaves_the_links_of_a_series_with_lags():
    series = sample_context_series(
        n_system=4, n_time=1, n_space=1, n_steps=150, tau_max=1, observed=0.0, seed=1
    )
    out = andrey.jpcmciplus(list(series.datasets), tau_max=1, alpha=0.05)
    found = {(i, j, lag) for i, j, lag in _lagged(out)}
    found |= {(i, j, 0) for i, j, _ in _same_step(out)}
    undirected = {frozenset((i, j)) for i, j, lag in found if lag == 0}
    true_same_step = {frozenset((i, j)) for i, j, lag in series.links if lag == 0}
    assert undirected == true_same_step
    assert {link for link in found if link[2]} == {link for link in series.links if link[2]}


def _ordered(link: tuple[int, int, str]) -> tuple[int, int, str]:
    i, j, kind = link
    return min(i, j), max(i, j), kind


def _true_links(series) -> set[tuple[int, int, int]]:
    """The system links of ``series``, a lag-0 link without its direction."""
    return {(i, j, lag) if lag else (min(i, j), max(i, j), 0) for i, j, lag in series.links}


def _found_links(out: andrey.StructureOutput) -> set[tuple[int, int, int]]:
    """The system links of ``out``, a lag-0 link without its direction."""
    return _lagged(out) | {(i, j, 0) for i, j, _ in map(_ordered, _same_step(out))}


@functools.cache
def _scores() -> tuple[float, float, float, float]:
    """Recall and precision of the system links and of the context links, over seeds."""
    hits = misses = extras = context_hits = context_links = context_found = 0
    for seed in range(5):
        series = sample_context_series(n_datasets=10, n_steps=100, seed=seed)
        out = andrey.jpcmciplus(
            list(series.datasets),
            time_contexts=list(series.time_contexts),
            space_contexts=list(series.space_contexts),
            tau_max=2,
            alpha=0.05,
        )
        true, found = _true_links(series), _found_links(out)
        hits += len(true & found)
        misses += len(true - found)
        extras += len(found - true)
        links = {tuple(link) for link in out.metadata["context_links"]}
        context_hits += len(links & series.context_links)
        context_links += len(series.context_links)
        context_found += len(links)
    return (
        hits / (hits + misses),
        hits / (hits + extras),
        context_hits / context_links,
        context_hits / context_found,
    )


def test_it_recovers_the_links_of_the_model_the_paper_describes():
    recall, precision, context_recall, context_precision = _scores()
    assert recall >= 0.9
    assert precision >= 0.9
    # The lagged phase can lose a context link an autocorrelated target hides.
    assert context_recall >= 0.8
    assert context_precision >= 0.8


def test_hidden_contexts_add_fewer_false_links_than_pooling():
    # Half the contexts are hidden. pcmciplus on the same columns, the datasets pooled without
    # their index, finds more false links between the system variables over the seeds.
    false_links = false_pooled = 0
    for seed in range(5):
        series = sample_context_series(
            n_system=5, n_datasets=10, n_steps=100, observed=0.5, seed=seed
        )
        out = andrey.jpcmciplus(
            list(series.datasets),
            time_contexts=list(series.time_contexts),
            space_contexts=list(series.space_contexts),
            tau_max=2,
            alpha=0.05,
        )
        pooled = andrey.pcmciplus(np.vstack(series.datasets), tau_max=2, alpha=0.05)
        system = {link for link in _found_links(pooled) if max(link[:2]) < 5}
        false_links += len(_found_links(out) - _true_links(series))
        false_pooled += len(system - _true_links(series))
    assert false_links < false_pooled


def test_a_fit_repeats_and_does_not_read_the_order_of_the_datasets():
    series = sample_context_series(n_system=3, n_time=1, n_space=1, n_steps=60, seed=2)
    kept = {
        "time_contexts": list(series.time_contexts),
        "space_contexts": list(series.space_contexts),
        "tau_max": 2,
    }
    first = andrey.jpcmciplus(list(series.datasets), **kept)
    rng = np.random.default_rng(0)
    orders = [
        np.arange(len(series.datasets)),
        *(rng.permutation(len(series.datasets)) for _ in range(3)),
    ]
    for order in orders:
        out = andrey.jpcmciplus([series.datasets[k] for k in order], **kept)
        assert _same_step(out) == _same_step(first) and _lagged(out) == _lagged(first)
        assert out.metadata["context_links"] == first.metadata["context_links"]
        assert out.metadata["index_edges"] == first.metadata["index_edges"]
        assert out.metadata["time_index_edges"] == first.metadata["time_index_edges"]
