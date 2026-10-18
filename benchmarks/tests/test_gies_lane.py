"""The GIES lane: its interventional data, the side-by-side encoding, and the two adapters.

A fit takes the samples and the 0/1 intervention matrix side by side, so an adapter that read one
half as the other would still return a graph. The checks below pin what each half holds and that
an intervention's orientation survives into each package's output, in the right direction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import needs_r

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gies_cpu  # noqa: E402

from andrey_bench.adapters.andrey_gies import AndreyGIES, stack, unstack  # noqa: E402
from andrey_bench.adapters.pcalg_gies import PcalgGIES  # noqa: E402


def _xy_with_y_intervened() -> np.ndarray:
    """``x -> y``; the last third of the rows set ``y`` from outside."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(900)
    y = x + 0.5 * rng.standard_normal(900)
    y[600:] = rng.normal(2.0, 0.2, 300)
    mask = np.zeros((900, 2), dtype=bool)
    mask[600:, 1] = True
    return stack(np.column_stack([x, y]), mask)


def test_the_generating_dags_point_to_earlier_and_later_columns():
    """The causal order is drawn, not the column order, so the lane takes no shuffle step."""
    forward = backward = 0
    for seed in range(3):
        _, _, dag = gies_cpu.sample(10, seed, rows_per_variable=10, target_share=0.2)
        parent, child = np.nonzero((dag == 1) & (dag.T == 2))
        forward += int((parent < child).sum())
        backward += int((parent > child).sum())
    assert forward and backward


def test_unstack_returns_what_stack_was_given():
    rng = np.random.default_rng(1)
    samples, mask = rng.standard_normal((20, 3)), rng.random((20, 3)) < 0.3
    back_samples, back_mask = unstack(stack(samples, mask))
    assert np.array_equal(back_samples, samples) and np.array_equal(back_mask, mask)
    with pytest.raises(ValueError, match="side by side"):
        unstack(np.zeros((5, 3)))


def test_the_simulation_intervenes_on_its_targets_in_their_rows_only():
    d = 10
    X, mask, dag = gies_cpu.sample(d, 0, rows_per_variable=10, target_share=0.4)
    assert X.shape == mask.shape == (100, d) and dag.shape == (d, d)
    targets = np.flatnonzero(mask.any(axis=0))
    assert len(targets) == 4
    assert (mask.sum(axis=1) <= 1).all()  # one variable per target
    assert (~mask.any(axis=1)).sum() == 20  # the rows without an intervention, one setting in five
    intervened = X[mask]
    assert abs(intervened.mean() - gies_cpu.LEVEL[0]) < 0.1


def test_a_variable_intervened_on_is_cut_from_its_parents():
    """The essential graph orients every edge at a target, so the truth carries those arrows."""
    X, mask, dag = gies_cpu.sample(12, 3, rows_per_variable=10, target_share=0.5)
    truth = gies_cpu.essential_graph(dag, mask)
    for j in np.flatnonzero(mask.any(axis=0)):
        for i in np.flatnonzero(dag[j] != 0):  # every neighbor of the target in the DAG
            assert truth[i, j] == dag[i, j] and truth[j, i] == dag[j, i]


def test_andrey_orients_the_intervened_edge():
    adapter = AndreyGIES()
    adapter.setup()
    native = adapter.fit(_xy_with_y_intervened(), adapter.params())
    assert adapter.to_structure(native).oriented_edges(index=True) == [(0, 1, "directed")]


@needs_r()
def test_pcalg_orients_the_intervened_edge():
    adapter = PcalgGIES()
    adapter.setup()
    native = adapter.fit(_xy_with_y_intervened(), adapter.params())
    assert adapter.to_structure(native).oriented_edges(index=True) == [(0, 1, "directed")]
