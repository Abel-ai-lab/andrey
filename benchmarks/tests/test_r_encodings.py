"""R output encodings checked against known graphs.

pcalg and bnlearn assign `amat[i, j]` marks to different endpoints. pcalg also differs between
`amat.cpdag` and `amat.pag`. Reversing an encoding produces a well-formed transposed graph that
downstream checks accept. Each adapter's decoder is tested against a known fit:

* a **collider** `0 -> 2 <- 1`, whose v-structure survives into the CPDAG and the PAG, so every
  arrowhead has a known place;
* a **two-variable** model `0 -> 1`, which no observational method can orient, so the undirected
  and circle encodings are exercised too.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_r

pytestmark = needs_r()

# Andrey endpoint marks.
NULL, TAIL, ARROW, CIRCLE = 0, 1, 2, 3


def collider_data(n: int = 4000) -> np.ndarray:
    """`0 -> 2 <- 1`, with 3 independent of everything."""
    rng = np.random.default_rng(0)
    x0 = rng.standard_normal(n)
    x1 = rng.standard_normal(n)
    x2 = x0 + x1 + 0.3 * rng.standard_normal(n)
    x3 = rng.standard_normal(n)
    data = np.column_stack([x0, x1, x2, x3])
    return (data - data.mean(0)) / data.std(0)


def pair_data(n: int = 4000) -> np.ndarray:
    """`0 -> 1`: one edge, and no observational method can orient it."""
    rng = np.random.default_rng(2)
    x0 = rng.standard_normal(n)
    x1 = x0 + 0.4 * rng.standard_normal(n)
    data = np.column_stack([x0, x1])
    return (data - data.mean(0)) / data.std(0)


def collider_with_child_data(n: int = 4000) -> np.ndarray:
    """`0 -> 2 <- 1` with `2 -> 3`: a collider and an edge whose PAG endpoint is a tail.

    The collider and unorientable pair produce no tails. This fixture tests tail decoding;
    a tail-only mapping error passes the other fixtures.
    """
    rng = np.random.default_rng(0)
    x0 = rng.standard_normal(n)
    x1 = rng.standard_normal(n)
    x2 = x0 + x1 + 0.3 * rng.standard_normal(n)
    x3 = x2 + 0.3 * rng.standard_normal(n)
    data = np.column_stack([x0, x1, x2, x3])
    return (data - data.mean(0)) / data.std(0)


def collider_marks(mark_at_parent: int) -> np.ndarray:
    """The 4-node collider in Andrey marks, with ``mark_at_parent`` at the 0 and 1 ends."""
    marks = np.zeros((4, 4), dtype=np.uint8)
    for parent in (0, 1):
        marks[parent, 2] = mark_at_parent
        marks[2, parent] = ARROW
    return marks


def pair_marks(mark: int) -> np.ndarray:
    """The single unorientable edge, with ``mark`` at both ends."""
    return np.array([[NULL, mark], [mark, NULL]], dtype=np.uint8)


def fitted(adapter, data: np.ndarray) -> np.ndarray:
    """Run one adapter end to end and return the Andrey marks it produced."""
    adapter.setup()
    return np.asarray(adapter.to_structure(adapter.fit(data, adapter.params())).to_numpy())


# --------------------------------------------------------------------------------------------------
# CPDAG-producing solutions: a v-structure stays directed, a lone edge does not.
# --------------------------------------------------------------------------------------------------


def _cpdag_adapters():
    from andrey_bench.adapters.bnlearn_pc_stable import BnlearnPCStable
    from andrey_bench.adapters.pcalg_ges import PcalgGES
    from andrey_bench.adapters.pcalg_pc import PcalgPC

    return [PcalgGES(), PcalgPC(), BnlearnPCStable()]


@pytest.mark.parametrize("adapter", _cpdag_adapters(), ids=lambda a: a.name)
def test_a_cpdag_solution_puts_the_arrowheads_on_the_collider(adapter):
    assert np.array_equal(fitted(adapter, collider_data()), collider_marks(TAIL))


@pytest.mark.parametrize("adapter", _cpdag_adapters(), ids=lambda a: a.name)
def test_a_cpdag_solution_leaves_an_unorientable_edge_undirected(adapter):
    assert np.array_equal(fitted(adapter, pair_data()), pair_marks(TAIL))


@pytest.mark.parametrize("adapter", _cpdag_adapters(), ids=lambda a: a.name)
def test_a_cpdag_solution_orients_the_edge_out_of_a_collider(adapter):
    """Meek's R1 orients `2 -> 3`. The collider fixture has no such edge, so a decode or an R call
    that kept only the v-structures would pass every other CPDAG test."""
    expected = collider_marks(TAIL)
    expected[2, 3], expected[3, 2] = TAIL, ARROW
    assert np.array_equal(fitted(adapter, collider_with_child_data()), expected)


# --------------------------------------------------------------------------------------------------
# PAG-producing solutions: circles where the data cannot fix an endpoint.
# --------------------------------------------------------------------------------------------------


def _pag_adapters():
    from andrey_bench.adapters.pcalg_fci import PcalgFCI
    from andrey_bench.adapters.pcalg_rfci import PcalgRFCI

    return [PcalgFCI(), PcalgRFCI()]


@pytest.mark.parametrize("adapter", _pag_adapters(), ids=lambda a: a.name)
def test_a_pag_solution_reads_arrowheads_at_the_collider_and_circles_away_from_it(adapter):
    # `0 o-> 2 <-o 1`: pcalg marks the opposite end from Andrey. Without transposing,
    # decoding would put arrowheads at 0 and 1.
    assert np.array_equal(fitted(adapter, collider_data()), collider_marks(CIRCLE))


@pytest.mark.parametrize("adapter", _pag_adapters(), ids=lambda a: a.name)
def test_a_pag_solution_leaves_an_unorientable_edge_circled(adapter):
    assert np.array_equal(fitted(adapter, pair_data()), pair_marks(CIRCLE))


@pytest.mark.parametrize("adapter", _pag_adapters(), ids=lambda a: a.name)
def test_a_pag_solution_puts_a_tail_on_the_edge_out_of_a_collider(adapter):
    """The tail code is covered alongside none, circle, and arrowhead.

    pcalg encodes none / circle / arrowhead / tail as 0 / 1 / 2 / 3. The collider and pair
    fixtures produce only 0 / 1 / 2.
    """
    expected = collider_marks(CIRCLE)
    # `2 -> 3`: an edge from the collider, with a PAG tail at 2.
    expected[2, 3], expected[3, 2] = TAIL, ARROW
    assert np.array_equal(fitted(adapter, collider_with_child_data()), expected)


# --------------------------------------------------------------------------------------------------
# DAG-producing solutions: bnlearn returns one member of the equivalence class.
# --------------------------------------------------------------------------------------------------


def _dag_adapters():
    from andrey_bench.adapters.bnlearn_hc import BnlearnHC
    from andrey_bench.adapters.bnlearn_tabu import BnlearnTabu

    return [BnlearnHC(), BnlearnTabu()]


@pytest.mark.parametrize("adapter", _dag_adapters(), ids=lambda a: a.name)
def test_a_dag_solution_orients_the_collider(adapter):
    assert np.array_equal(fitted(adapter, collider_data()), collider_marks(TAIL))


@pytest.mark.parametrize("adapter", _dag_adapters(), ids=lambda a: a.name)
def test_a_dag_solution_orients_the_edge_it_cannot_identify(adapter):
    """DAG search assigns one arrowhead to the observationally unorientable pair.

    The direction is arbitrary, which makes its `shd` differ from a CPDAG's. Decoding must
    preserve exactly one arrowhead.
    """
    marks = fitted(adapter, pair_data())
    assert sorted(
        marks[np.triu_indices(2, 1)].tolist() + marks[np.tril_indices(2, -1)].tolist()
    ) == [
        TAIL,
        ARROW,
    ]


# --------------------------------------------------------------------------------------------------
# The output families the analysis groups on.
# --------------------------------------------------------------------------------------------------


def test_the_output_families_are_what_the_analysis_assumes():
    """DAG and CPDAG scores use different units.

    Tables group by `output_type`; an incorrect declaration compares those scores directly.
    """
    families = {
        a.name: a.output_type for a in _cpdag_adapters() + _pag_adapters() + _dag_adapters()
    }
    assert families == {
        "pcalg.ges": "cpdag",
        "pcalg.pc": "cpdag",
        "bnlearn.pc-stable": "cpdag",
        "pcalg.fci": "pag",
        "pcalg.rfci": "pag",
        "bnlearn.hc": "dag",
        "bnlearn.tabu": "dag",
    }
