from __future__ import annotations

import numpy as np

import andrey
from andrey.core import ARROW, TAIL
from tests.recovery.helpers import datagen
from tests.recovery.helpers.slice_fixture import assert_recorded_output, assert_recorded_output_mec

# --- LiNGAM family (causal order + weighted adjacency) --------------------------
#
# The LiNGAM recorded outputs are checked in test_direct_lingam.py and test_lingam_lane.py.


_BASELINE_LAMBDA = 2.0


def test_lingam_weighted_orientation_matches_structure():
    out = andrey.direct_lingam(datagen.generate("lingam_5v_uniform"))
    marks = out.structure.to_numpy()
    rows, cols = np.nonzero(out.weighted_adjacency)
    weighted_edges = set(zip(rows.tolist(), cols.tolist()))
    graph_edges = {
        (i, j)
        for i in range(marks.shape[0])
        for j in range(marks.shape[0])
        if marks[i][j] == TAIL and marks[j][i] == ARROW  # tail at source i, arrow at target j
    }
    assert weighted_edges
    assert weighted_edges == graph_edges


# --- constraint-based (PAG / CPDAG endpoint matrices) --------------------------


def test_pc_recovery():
    out = andrey.pc(datagen.generate("gauss_5v"))
    assert_recorded_output(out, "PC", case="gauss_5v")


def test_fci_recovery():
    out = andrey.fci(datagen.generate("gauss_5v"))
    assert_recorded_output(out, "FCI", case="gauss_5v")


def test_gfci_recovery():
    out = andrey.gfci(datagen.generate("gauss_5v"), lambda_value=_BASELINE_LAMBDA)
    assert_recorded_output(out, "GFCI", case="gauss_5v")


def test_cdnod_recovery():
    out = andrey.cdnod(datagen.generate("gauss_5v"), datagen.domain_index("gauss_5v"))
    assert_recorded_output(out, "CDNOD", case="gauss_5v")


# --- score-based ---------------------------------------------------------------


def test_ges_recovery():
    out = andrey.ges(datagen.generate("gauss_5v"), lambda_value=_BASELINE_LAMBDA)
    assert_recorded_output(out, "GES", case="gauss_5v")


def test_gies_recovery():
    out = andrey.gies(datagen.generate("gauss_5v"), lambda_value=_BASELINE_LAMBDA)
    assert_recorded_output(out, "GIES", case="gauss_5v")


def test_hc_recovery():
    out = andrey.hc(datagen.generate("gauss_5v"))
    assert_recorded_output(out, "HC", case="gauss_5v")


def test_exact_search_recovery():
    # The optimal DAG is Markov-equivalence-class-identified, so compare at the CPDAG level.
    out = andrey.exact_search(datagen.generate("gauss_5v"))
    assert_recorded_output_mec(out, "ExactSearch", case="gauss_5v")


# --- permutation-based ---------------------------------------------------------


def test_boss_recovery():
    out = andrey.boss(datagen.generate("gauss_5v"), lambda_value=_BASELINE_LAMBDA)
    assert_recorded_output(out, "BOSS", case="gauss_5v")


def test_grasp_recovery():
    out = andrey.grasp(datagen.generate("gauss_5v"), lambda_value=_BASELINE_LAMBDA)
    assert_recorded_output(out, "GRaSP", case="gauss_5v")
