from __future__ import annotations

import numpy as np
import pytest

import andrey
from andrey import data
from andrey.constraint.pc import _discover_skeleton, _orient_colliders
from tests.recovery.helpers import datagen
from tests.recovery.helpers.slice_fixture import assert_recorded_output


@pytest.fixture(autouse=True)
def _force_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANDREY_DEVICE", "numpy")


def test_fci_recovery():
    out = andrey.fci(datagen.generate("gauss_5v"))
    assert_recorded_output(out, "FCI", case="gauss_5v")


def test_cdnod_recovery():
    out = andrey.cdnod(datagen.generate("gauss_5v"), datagen.domain_index("gauss_5v"))
    assert_recorded_output(out, "CDNOD", case="gauss_5v")


def test_cdnod_is_invariant_to_column_order():
    # Two variables shift between the domains. Colliders of the augmented skeleton conflict on
    # this seed, so the rank order decides which are kept.
    x = data.sample_scm(d=12, n=600, seed=0, density=3.0, scale="standardize").data
    context = np.repeat([0.0, 1.0], 300)
    x[context == 1, 0] += 1.5
    x[context == 1, 6] += 1.5
    augmented = np.column_stack([x, context])
    skeleton, sepsets, pvalues = _discover_skeleton(augmented, 0.05)
    trace: list[dict] = []
    _orient_colliders(skeleton, sepsets, pvalues, _trace=trace)
    assert any(
        r["step"] == "collider" and not r["oriented"] and r["triple"][1] not in r["sepset"]
        for r in trace
    )
    marks = andrey.cdnod(x, context).structure.to_numpy()
    for seed in range(3):
        order = np.random.default_rng(seed).permutation(x.shape[1])
        back = np.argsort(order)
        permuted = andrey.cdnod(x[:, order], context).structure.to_numpy()
        assert np.array_equal(permuted[np.ix_(back, back)], marks)


def test_fci_output_contract():
    assert andrey.fci(datagen.generate("gauss_5v")).structure.kind == "pag"


def test_cdnod_output_contract():
    out = andrey.cdnod(datagen.generate("gauss_5v"), datagen.domain_index("gauss_5v"))
    assert out.structure.kind == "cpdag"


def test_fci_rejects_non_fisherz():
    with pytest.raises(NotImplementedError):
        andrey.fci(datagen.generate("gauss_5v"), indep_test="kci")


def test_cdnod_rejects_non_fisherz():
    x, c = datagen.generate("gauss_5v"), datagen.domain_index("gauss_5v")
    with pytest.raises(NotImplementedError):
        andrey.cdnod(x, c, indep_test="kci")


def test_cdnod_rejects_mismatched_context():
    x = datagen.generate("gauss_5v")
    with pytest.raises(ValueError, match="one value per row"):
        andrey.cdnod(x, np.zeros((len(x) + 1, 1)))


def test_cdnod_rejects_an_index_with_the_right_size_and_the_wrong_shape():
    """A two-column index with as many cells as rows would be flattened onto the wrong rows."""
    x = datagen.generate("gauss_5v")
    x = x[: len(x) // 2 * 2]
    with pytest.raises(ValueError, match="one value per row"):
        andrey.cdnod(x, np.zeros((len(x) // 2, 2)))
    column = datagen.domain_index("gauss_5v")[: len(x)]
    flat = andrey.cdnod(x, column).structure
    assert andrey.cdnod(x, column.reshape(-1, 1)).structure == flat


# --- general-input behavior ----------------------------------------------------------------------
#
# Random latent-confounded models check repeatability and well-formed output beyond the fixed
# baseline. FCI/GFCI column-order sensitivity is measured by qa.column_order.


def _latent_confounded_sample(seed: int, d: int) -> np.ndarray:
    """An observed sample from a random DAG with one hidden confounder."""
    from andrey import data

    return data.sample_scm(graph="erdos_renyi", d=d, n=1200, seed=seed, density=2.0, latents=1).data


def _fit(learner: str, x: np.ndarray):
    if learner == "cdnod":
        context = np.zeros((len(x), 1))
        context[len(x) // 2 :] = 1.0
        return andrey.cdnod(x, context).structure
    return getattr(andrey, learner)(x).structure


@pytest.mark.parametrize(
    ("learner", "kind"),
    [("fci", "pag"), ("gfci", "pag"), ("cdnod", "cpdag")],
)
def test_constraint_family_is_deterministic_and_well_formed(learner, kind):
    for seed in range(4):
        for d in (6, 8):
            x = _latent_confounded_sample(seed, d)
            observed = x.shape[1]  # d nodes minus the hidden confounder
            first = _fit(learner, x)
            assert first.kind == kind
            assert first.n_nodes == observed
            first.validate()
            repeat = _fit(learner, x).to_numpy()
            assert np.array_equal(repeat, first.to_numpy()), f"{learner} is not deterministic"

            if learner == "cdnod":
                # CDNOD checks skeleton invariance; its collider conflicts can change marks.
                order = np.random.default_rng(seed).permutation(observed)
                back = np.argsort(order)
                permuted = _fit(learner, x[:, order]).to_numpy()[np.ix_(back, back)]
                assert np.array_equal(permuted != 0, first.to_numpy() != 0), (
                    f"{learner} selects different edges when the columns are reordered"
                )
