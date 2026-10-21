"""The identity harness hashes every bit of a result, maps columns back, and counts queries."""

from __future__ import annotations

import contextlib
import importlib
import pkgutil
import struct
import sys

import numpy as np
import pytest

import andrey
from andrey import GraphStructure, StructureOutput
from andrey.core import ci, score_delta
from andrey.core.ci import make_indep_test
from qa import identity


@pytest.fixture(scope="module")
def x() -> np.ndarray:
    """A chain ``0 -> 1 -> 2`` with an independent fourth column."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal(500)
    b = a + rng.standard_normal(500)
    c = b + rng.standard_normal(500)
    return np.column_stack([a, b, c, rng.standard_normal(500)])


def test_a_result_hashes_the_same_on_every_fit(x: np.ndarray) -> None:
    first, second = andrey.ges(x), andrey.ges(x)
    assert identity.digest(first) == identity.digest(second)
    assert identity.graph_digest(first) == identity.graph_digest(second)


def test_one_float_bit_changes_the_hash_and_not_the_graph_hash(x: np.ndarray) -> None:
    out = andrey.ges(x)
    score = float(out.metadata["score"])
    nudged = StructureOutput.new(
        out.structure, metadata={**out.metadata, "score": float(np.nextafter(score, np.inf))}
    )
    assert identity.digest(nudged) != identity.digest(out)
    assert identity.graph_digest(nudged) == identity.graph_digest(out)


def test_a_changed_edge_changes_the_graph_hash() -> None:
    marks = np.zeros((3, 3), dtype=np.int8)
    marks[0, 1], marks[1, 0] = 1, 2  # 0 -> 1
    out = StructureOutput.new(GraphStructure.from_numpy(marks, kind="dag"))
    marks[1, 2], marks[2, 1] = 1, 2  # and 1 -> 2
    changed = StructureOutput.new(GraphStructure.from_numpy(marks, kind="dag"))
    assert identity.graph_digest(changed) != identity.graph_digest(out)


def test_floats_enter_as_bits_with_one_nan() -> None:
    other_nan = np.frombuffer(struct.pack("<Q", 0x7FF8000000000001), dtype=np.float64)
    assert identity.digest(other_nan) == identity.digest(np.array([np.nan]))
    assert identity.digest(-0.0) != identity.digest(0.0)
    assert identity.digest(np.float32(1.0)) != identity.digest(1.0)
    assert identity.digest({"b": 1, "a": 2}) == identity.digest({"a": 2, "b": 1})
    with pytest.raises(TypeError):
        identity.digest({1, 2})


def test_a_column_order_maps_back_to_the_original_nodes() -> None:
    marks = np.zeros((4, 4), dtype=np.int8)
    marks[0, 1], marks[1, 0] = 1, 2  # 0 -> 1
    marks[1, 2], marks[2, 1] = 1, 2  # 1 -> 2
    weights = np.zeros((4, 4))
    weights[0, 1], weights[1, 2] = 0.5, -1.5
    out = StructureOutput.new(
        GraphStructure.from_numpy(marks, kind="dag"),
        ordering=[3, 0, 1, 2],
        weighted_adjacency=weights,
        metadata={"algorithm": "test", "p_values": [np.arange(16.0).reshape(4, 4).tolist()]},
    )
    order = np.array([2, 0, 3, 1])  # permuted column k is original column order[k]
    back = np.argsort(order)
    fitted = StructureOutput.new(  # the same answer, as a fit on x[:, order] gives it
        GraphStructure.from_numpy(marks[np.ix_(order, order)], kind="dag"),
        ordering=[int(back[i]) for i in out.ordering],
        weighted_adjacency=weights[np.ix_(order, order)],
        metadata={
            "algorithm": "test",
            "p_values": [np.arange(16.0).reshape(4, 4)[np.ix_(order, order)].tolist()],
        },
    )
    assert identity.digest(identity.unpermute(fitted, order)) == identity.digest(
        identity.unpermute(out, np.arange(4))
    )
    assert identity.graph_digest(identity.unpermute(fitted, order)) == identity.graph_digest(out)


def test_a_swapped_pair_trades_its_directions() -> None:
    marks = np.array([[0, 1], [2, 0]], dtype=np.int8)  # 0 -> 1
    fitted = StructureOutput.new(  # 1 -> 0 on the swapped columns
        GraphStructure.from_numpy(marks.T, kind="dag"),
        metadata={"pval_forward": 0.5, "pval_backward": 0.01},
    )
    back = identity.unpermute(fitted, np.array([1, 0]))
    assert back.structure.to_numpy().tolist() == marks.tolist()
    assert dict(back.metadata) == {"pval_forward": 0.01, "pval_backward": 0.5}


def test_permuted_moves_each_block_of_variables_alike() -> None:
    x = np.arange(12.0).reshape(1, 12)  # three groups of four columns
    order = np.array([3, 1, 0, 2])
    expected = [3, 1, 0, 2, 7, 5, 4, 6, 11, 9, 8, 10]
    assert identity.permuted(x, order, 3).ravel().tolist() == expected


def test_counts_separation_and_distinct_conditioning_sets() -> None:
    counts = identity.Counts(alpha=0.05)
    counts.ci([(0, 2, (), 0.001), (2, 0, (1,), 0.4), (0, 2, (3,), 0.01), (2, 0, (1, 3), 0.9)])
    counts.ci([(1, 3, (1,), 0.5)])
    counts.score([(0, 1), (1, 0), (), (2,)])
    assert counts.summary() == {
        "ci_queries": 5,
        "ci_after_separation": 2,  # the two queries of (0, 2) after p = 0.4
        "ci_sets": {"0": 1, "1": 2, "2": 1},
        "score_queries": 4,
        "score_sets": {"0": 1, "1": 1, "2": 1},
    }


def test_counting_records_each_outermost_query_and_restores_the_methods(x: np.ndarray) -> None:
    test = make_indep_test("fisherz", x)
    original = ci.PartialCorrelationTest.__dict__["batched_call"]
    with identity.counting(alpha=0.05) as counts:
        test(0, 2, (1,))
        test.batched_call(0, 2, iter([(), (1, 3)]))
        test.first_order_pvalues(0, 1, [2, 3])
    assert ci.PartialCorrelationTest.__dict__["batched_call"] is original
    assert counts.summary()["ci_queries"] == 5
    assert counts.summary()["ci_sets"] == {"0": 1, "1": 3, "2": 1}


def test_counting_wraps_every_name_of_the_score_delta_functions() -> None:
    for module in pkgutil.walk_packages(andrey.__path__, "andrey."):
        with contextlib.suppress(ImportError):  # an optional extra
            importlib.import_module(module.name)
    functions = (score_delta.add_deltas, score_delta.remove_deltas)
    naming = {
        name
        for name, module in sys.modules.items()
        if name.startswith("andrey.")
        and any(v is f for v in vars(module).values() for f in functions)
    }
    assert naming == {module.__name__ for module in identity.DELTA_MODULES}


def test_counting_leaves_the_result_unchanged(x: np.ndarray) -> None:
    for fit in (andrey.pc, andrey.hc):
        with identity.counting(alpha=0.05) as counts:
            counted = fit(x)
        assert counts.summary()
        assert identity.digest(counted) == identity.digest(fit(x))


def _manifest(**changes: object) -> dict:
    row = {
        "method": "PC",
        "dataset": "d",
        "seed": 0,
        "permutation": 0,
        "workers": 1,
        "hash": "h",
        "graph_hash": "g",
        "ci_queries": 10,
    }
    return {"header": {"commit": "a"}, "rows": [{**row, **changes}]}


@pytest.mark.parametrize(
    ("changes", "status", "message"),
    [
        ({}, 0, "0 with a different graph, 0 with only different bits"),
        ({"hash": "h2"}, 1, "0 with a different graph, 1 with only different bits"),
        ({"hash": "h2", "graph_hash": "g2"}, 1, "1 with a different graph"),
        ({"ci_queries": 11}, 1, "1 counts rose"),
        ({"ci_queries": 9}, 0, "1 fell"),
        ({"seed": 1}, 1, "2 in one only"),
    ],
)
def test_compare_counts_the_rows_that_differ(changes, status, message, capsys) -> None:
    assert identity.compare(_manifest(), _manifest(**changes)) == status
    assert message in capsys.readouterr().out
