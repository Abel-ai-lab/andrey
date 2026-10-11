"""Recovery on categorical data: the chi-square and G-square tests and the BDeu score.

The fixed cases have one answer: ``discrete_5v`` is the network ``x0 -> x2 <- x1``, ``x2 -> x3 ->
x4``, whose CPDAG has every edge oriented, and ``discrete_independent_4v`` has no edges. The
ground-truth case is a random network from :func:`andrey.data.sample_discrete`, scored against its
CPDAG with a margin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import andrey
from andrey.core import ARROW
from andrey.data import sample_discrete
from andrey.metrics import shd, to_cpdag
from tests.recovery.helpers import datagen

pytestmark = pytest.mark.filterwarnings("ignore::andrey.ExperimentalWarning")

TRUTH = {(0, 2), (1, 2), (2, 3), (3, 4)}

# Each fit that learns a CPDAG from categorical data, by its categorical setting.
CPDAG_FITS = {
    "pc-chisq": lambda x: andrey.pc(x, indep_test="chisq"),
    "pc-gsq": lambda x: andrey.pc(x, indep_test="gsq"),
    "ges-bdeu": lambda x: andrey.ges(x, score_func="local_score_BDeu"),
    "hc-bdeu": lambda x: andrey.hc(x, score_func="local_score_BDeu"),
    "cdnod-chisq": lambda x: andrey.cdnod(x, np.zeros(len(x)), indep_test="chisq"),
}
PAG_FITS = {
    "fci-chisq": lambda x: andrey.fci(x, indep_test="chisq"),
    "fci-gsq": lambda x: andrey.fci(x, indep_test="gsq"),
    "gfci-bdeu-chisq": lambda x: andrey.gfci(x, score_func="local_score_BDeu", indep_test="chisq"),
}


def _edges(out, kinds):
    return {(a, b) for a, b, kind in out.structure.oriented_edges() if kind in kinds}


@pytest.mark.parametrize("fit", list(CPDAG_FITS))
def test_cpdag_methods_recover_the_categorical_network(fit):
    out = CPDAG_FITS[fit](datagen.generate("discrete_5v"))
    assert out.structure.kind == "cpdag"
    assert _edges(out, {"directed"}) == TRUTH
    assert _edges(out, {"undirected"}) == set()


@pytest.mark.parametrize("fit", list(PAG_FITS))
def test_pag_methods_find_the_skeleton_and_the_collider(fit):
    out = PAG_FITS[fit](datagen.generate("discrete_5v"))
    assert out.structure.kind == "pag"
    marks = out.structure.to_numpy()
    assert {(a, b) for a, b, _ in out.structure.oriented_edges()} == TRUTH
    # Arrowheads at x2 from both parents: the collider every method must see.
    assert marks[2, 0] == ARROW and marks[2, 1] == ARROW  # M[i, j] is the mark at i


@pytest.mark.parametrize("fit", [*CPDAG_FITS, *PAG_FITS])
def test_independent_columns_give_no_edges(fit):
    out = {**CPDAG_FITS, **PAG_FITS}[fit](datagen.generate("discrete_independent_4v"))
    assert out.structure.oriented_edges() == []


@pytest.mark.parametrize("fit", list(CPDAG_FITS))
def test_the_codes_need_not_start_at_zero(fit):
    """Each distinct value is a category, so recoding a column leaves the graph unchanged."""
    x = datagen.generate("discrete_5v")
    recoded = x * np.array([3.0, -1.0, 2.0, 5.0, 1.0]) + np.array([-7.0, 4.0, 0.0, 11.0, 100.0])
    assert np.array_equal(
        CPDAG_FITS[fit](recoded).structure.to_numpy(), CPDAG_FITS[fit](x).structure.to_numpy()
    )


def test_a_dataframe_names_the_nodes():
    x = datagen.generate("discrete_5v")
    frame = pd.DataFrame(x, columns=list("abcde"))
    out = andrey.pc(frame, indep_test="chisq")
    assert set(out.structure.oriented_edges()) == {
        ("a", "c", "directed"),
        ("b", "c", "directed"),
        ("c", "d", "directed"),
        ("d", "e", "directed"),
    }


@pytest.mark.parametrize("d", [1, 2])
@pytest.mark.parametrize("fit", ["ges", "hc"])
def test_bdeu_fits_one_or_two_columns(fit, d):
    """With two or fewer variables the structure prior is left out, so the score stays finite."""
    x = datagen.generate("discrete_5v")[:, [2, 3][:d]]
    out = getattr(andrey, fit)(x, score_func="local_score_BDeu")
    assert np.isfinite(out.metadata["score"])
    assert len(out.structure.oriented_edges()) == d - 1


def test_cdnod_chisq_finds_a_shift_in_one_domain_of_three():
    """The categorical index finds a shift in the middle domain, which Fisher-Z's linear read of the
    codes ``0, 1, 2`` misses; the shift orients ``x0 -> x1``."""
    rng = np.random.default_rng(20240621)
    n = 3000
    domain = np.repeat([0, 1, 2], n // 3)
    x0 = (rng.random(n) < np.where(domain == 1, 0.8, 0.3)).astype(int)
    x1 = np.where(rng.random(n) < 0.8, x0, rng.integers(0, 2, n))
    x = np.column_stack([x0, x1])
    assert andrey.cdnod(x, domain, indep_test="chisq").structure.oriented_edges() == [
        (0, 1, "directed")
    ]
    assert andrey.cdnod(x, domain).structure.oriented_edges() == [(0, 1, "undirected")]


# A random network: 10 variables, 10 edges, 5000 rows. On this draw every method finds the true
# CPDAG (SHD 0); the margin allows for a change in tie-breaking without hiding a broken build,
# which scores near the empty graph's 10.
_RANDOM = {"d": 10, "n": 5000, "seed": 5}
_MARGIN = 2


@pytest.fixture(scope="module")
def random_network():
    return sample_discrete(**_RANDOM)


@pytest.mark.parametrize("fit", ["pc-chisq", "pc-gsq", "ges-bdeu"])
def test_a_random_network_is_recovered_within_the_margin(random_network, fit):
    truth = to_cpdag(random_network.graph)
    assert shd(np.zeros((_RANDOM["d"],) * 2), truth) == 10
    assert shd(CPDAG_FITS[fit](random_network.data), truth) <= _MARGIN
