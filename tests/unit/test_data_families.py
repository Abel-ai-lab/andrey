"""Student-t and Gaussian-mixture noise, random-feature links, and the outer transforms.

The distribution checks draw large samples on fixed seeds and compare them with the family's
distribution at a level of ``1e-3``.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

import andrey.data as data
from andrey.data import missingness
from andrey.data.functional import OUTER_TRANSFORMS

ALPHA = 1e-3
N = 200_000


def _standard(noise, n=N, seed=0):
    """Draws of ``noise`` at scale 1, through the public ``draw``."""
    return noise.draw(n, np.ones(1), np.random.default_rng(seed))[:, 0]


def _pair(functional, noise=None, n=4000, seed=0, **kwargs):
    """A two-variable sample with its one edge, unshuffled and raw: ``(dataset, parent, child)``."""
    scm = data.SCM(
        graph=data.graphs.erdos_renyi(d=2, avg_degree=1),
        functional=functional,
        noise=noise or data.noise.gaussian(),
    )
    ds = scm.sample(n=n, seed=seed, scale="raw", shuffle=False, **kwargs)
    ((parent, child),) = ds.params.edges.tolist()
    return ds, parent, child


# ---- the draws of seeds without the new options ------------------------------------------------


def test_samples_without_the_new_options_keep_their_recorded_values():
    """Three small samples equal their recorded values, and name no new option."""
    linear = data.sample_scm(d=3, n=4, seed=0, density=2)
    np.testing.assert_allclose(
        linear.data,
        [
            [0.4392173525397, 0.0234951555036, -0.2309454552122],
            [1.4273919536274, -1.054892075581, -1.51957919511],
            [-0.8159905937956, -0.569373169664, 1.1047998937779],
            [-1.0506187123715, 1.6007700897415, 0.6457247565442],
        ],
        rtol=1e-11,
    )
    assert linear.params.edges.tolist() == [[2, 0], [2, 1], [0, 1]]
    np.testing.assert_allclose(
        linear.params.weights, [-1.5157952854627, -0.8644801228142, -1.4176456944827], rtol=1e-11
    )
    assert linear.provenance["column_order"] == [0, 2, 1]

    pnl = data.sample_scm(
        functional="post_nonlinear", noise="laplace", d=3, n=4, seed=1, density=2, scale="raw"
    )
    np.testing.assert_allclose(
        pnl.data,
        [
            [2.3857248075117, -0.3983283819651, -1.6716010425459],
            [-0.8629144736982, 0.8015909463889, 0.4191672255174],
            [-0.0804417861558, -1.0077427729131, 0.1393777118686],
            [-2.0945574732685, 1.4708815302149, 0.9198314490809],
        ],
        rtol=1e-11,
    )

    masked = data.sample_scm(d=3, n=6, seed=2, density=2, missing=missingness.mcar(0.3))
    assert masked.missing.mask.astype(int).tolist() == [
        [0, 0, 1],
        [1, 0, 0],
        [1, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
        [0, 0, 0],
    ]
    for ds in (linear, pnl, masked):
        assert ds.measurements is None
        assert not {"functional_params", "noise_params", "measurements"} & set(
            ds.provenance["config"]
        )


@pytest.mark.parametrize("noise", ["student_t", "gaussian_mixture"])
def test_a_new_noise_family_leaves_the_model_streams_as_they_are(noise):
    gaussian = data.sample_scm(d=12, n=50, seed=3)
    ds = data.sample_scm(noise=noise, d=12, n=50, seed=3)
    assert np.array_equal(ds.params.edges, gaussian.params.edges)
    assert np.array_equal(ds.params.weights, gaussian.params.weights)
    assert ds.provenance["column_order"] == gaussian.provenance["column_order"]
    assert not np.array_equal(ds.data, gaussian.data)


# ---- noise -------------------------------------------------------------------------------------


@pytest.mark.parametrize("df", [3.0, 5.0, 30.0])
def test_student_t_noise_is_a_t_variable_scaled_to_unit_variance(df):
    x = _standard(data.noise.student_t(df=df))
    reference = stats.t(df, scale=np.sqrt((df - 2.0) / df))
    assert stats.kstest(x, reference.cdf).pvalue > ALPHA
    assert abs(x.mean()) < 0.02
    if df > 4:  # the sample variance has a finite variance of its own
        assert x.var() == pytest.approx(1.0, rel=0.05)


def test_student_t_tails_grow_as_df_falls():
    kurtosis = [stats.kurtosis(_standard(data.noise.student_t(df=df))) for df in (30.0, 8.0, 5.0)]
    assert 0 < kurtosis[0] < kurtosis[1] < kurtosis[2]
    assert kurtosis[1] == pytest.approx(6.0 / (8.0 - 4.0), rel=0.2)


def _mixture_cdf(weight, means, sds):
    """The CDF of the mixture after its centering and scaling."""
    means, sds = np.array(means), np.array(sds)
    probs = np.array([weight, 1 - weight])
    mean = probs @ means
    sd = np.sqrt(probs @ (sds**2 + means**2) - mean**2)
    return lambda x: sum(
        p * stats.norm.cdf(x * sd + mean, loc=m, scale=s) for p, m, s in zip(probs, means, sds)
    )


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"weight": 0.2, "means": (0.0, 3.0), "sds": (1.0, 0.5)},
        {"weight": 0.9, "means": (0.0, 0.0), "sds": (1.0, 5.0)},
    ],
)
def test_gaussian_mixture_noise_is_the_standardized_mixture(params):
    spec = data.noise.gaussian_mixture(**params)
    x = _standard(spec)
    assert stats.kstest(x, _mixture_cdf(spec.weight, spec.means, spec.sds)).pvalue > ALPHA
    assert abs(x.mean()) < 0.01
    assert x.var() == pytest.approx(1.0, rel=0.03)


def test_the_default_mixture_is_bimodal_and_an_unequal_one_is_skewed():
    bimodal = _standard(data.noise.gaussian_mixture())
    density = np.histogram(bimodal, bins=[-1.2, -0.6, -0.2, 0.2, 0.6, 1.2], density=True)[0]
    assert density[2] < density[0] and density[2] < density[4]
    assert abs(stats.skew(bimodal)) < 0.02
    skewed = _standard(data.noise.gaussian_mixture(weight=0.2, means=(0.0, 3.0), sds=(1.0, 0.5)))
    assert stats.skew(skewed) < -1.0


@pytest.mark.parametrize(
    ("build", "match"),
    [
        (lambda: data.noise.student_t(df=2.0), "df > 2"),
        (lambda: data.noise.student_t(df=np.inf), "df > 2"),
        (lambda: data.noise.student_t(df=np.nan), "df > 2"),
        (lambda: data.noise.gaussian_mixture(weight=1.0), r"weight in \(0, 1\)"),
        (lambda: data.noise.gaussian_mixture(means=(0.0, np.inf)), "two finite means"),
        (lambda: data.noise.gaussian_mixture(sds=(1.0, 0.0)), "two finite positive sds"),
        (lambda: data.noise.gaussian_mixture(sds=(1.0,)), "two finite positive sds"),
        (lambda: data.noise.student_t(variance_range=(1.0, 0.5)), "variance_range"),
    ],
)
def test_a_noise_family_rejects_parameters_outside_its_range(build, match):
    with pytest.raises(ValueError, match=match):
        build()


@pytest.mark.parametrize(
    ("noise", "recorded"),
    [
        (data.noise.student_t(df=7), {"df": 7.0}),
        (
            data.noise.gaussian_mixture(weight=0.3),
            {"weight": 0.3, "means": [-1.0, 1.0], "sds": [2 / 3, 2 / 3]},
        ),
    ],
)
def test_a_sample_records_its_noise_parameters_and_is_reproducible(noise, recorded):
    scm = data.SCM(
        graph=data.graphs.erdos_renyi(d=8, avg_degree=2),
        functional=data.functional.linear(),
        noise=noise,
    )
    ds = scm.sample(n=300, seed=4)
    assert ds.params.noise == noise.name
    assert ds.provenance["config"]["noise"] == noise.name
    assert ds.provenance["config"]["noise_params"] == recorded
    assert np.array_equal(ds.params.noise_scales, np.ones(8))
    assert np.array_equal(ds.data, scm.sample(n=300, seed=4).data)
    np.testing.assert_allclose(ds.data.std(axis=0), 1.0)


def test_new_noise_families_take_the_scale_arguments_and_every_scale_mode():
    spec = data.noise.student_t(df=6, variance_range=(0.4, 0.8))
    x = spec.draw(N, spec.draw_scales(3, np.random.default_rng(0)), np.random.default_rng(1))
    assert np.all((x.var(axis=0) > 0.36) & (x.var(axis=0) < 0.88))
    for scale in ("standardize", "unitless", "rescale", "raw"):
        ds = data.sample_scm(noise="gaussian_mixture", d=6, n=200, seed=0, scale=scale)
        assert np.all(np.isfinite(ds.data))
    series = data.sample_scm(functional="var", noise="student_t", d=4, n=200, seed=0, density=2)
    assert series.data.shape == (200, 4)


def test_sample_scm_names_the_allowed_families():
    with pytest.raises(ValueError, match="gaussian_mixture.*student_t"):
        data.sample_scm(noise="cauchy", d=8, n=10, seed=0)
    with pytest.raises(ValueError, match="random_features"):
        data.sample_scm(functional="gp", d=8, n=10, seed=0)


# ---- random-feature links ----------------------------------------------------------------------


def _functions(weights, u):
    """Each edge's function at the points ``u``: ``(m, len(u))``, from the recorded weights."""
    frequencies, phases, amplitudes = weights[:, 0], weights[:, 1], weights[:, 2]
    features = np.cos(frequencies[:, None, :] * u[None, :, None] + phases[:, None, :])
    return np.sqrt(2.0 / weights.shape[2]) * np.einsum("mud,md->mu", features, amplitudes)


@pytest.mark.parametrize("length_scale", [1.0, 2.0])
def test_random_feature_functions_have_the_gaussian_process_covariance(length_scale):
    spec = data.functional.random_features(n_features=50, length_scale=length_scale)
    _, draw = data.graphs.erdos_renyi(d=400, avg_degree=40).materialize(np.random.default_rng(0))
    weights = spec.draw_weights(draw, np.random.default_rng(1))
    assert weights.shape == (draw.parents.shape[0], 3, 50)
    u = np.array([0.0, 0.5, 1.0, 3.0])
    f = _functions(weights, u)
    kernel = np.exp(-((u[:, None] - u[None, :]) ** 2) / (2.0 * length_scale**2))
    np.testing.assert_allclose(f.T @ f / f.shape[0], kernel, atol=0.05)
    assert np.all(np.abs(f.mean(axis=0)) < 0.05)


def test_a_random_features_child_is_its_recorded_function_plus_unit_noise():
    ds, parent, child = _pair(data.functional.random_features(), n=20_000)
    assert ds.params.weights.shape == (1, 3, 100)
    assert ds.params.functional == "random_features"
    assert ds.provenance["config"]["functional_params"] == {
        "n_features": 100,
        "length_scale": 1.0,
    }
    x, y = ds.data[:, parent], ds.data[:, child]
    frequencies, phases, amplitudes = ds.params.weights[0]
    signal = np.sqrt(2.0 / 100) * np.cos(np.outer(x, frequencies) + phases) @ amplitudes
    noise = y - signal
    assert signal.std() > 0.1
    assert noise.std() == pytest.approx(1.0, rel=0.03)
    assert stats.kstest(noise, "norm").pvalue > ALPHA
    assert abs(np.corrcoef(noise, x)[0, 1]) < 0.03


def test_random_features_samples_are_reproducible_finite_and_valid():
    scm = data.SCM(
        graph=data.graphs.erdos_renyi(d=30, avg_degree=4),
        functional=data.functional.random_features(n_features=20),
        noise=data.noise.gaussian(),
    )
    for scale in ("standardize", "rescale", "raw"):
        a, b = scm.sample(n=200, seed=5, scale=scale), scm.sample(n=200, seed=5, scale=scale)
        assert np.array_equal(a.data, b.data)
        assert np.array_equal(a.params.weights, b.params.weights)
        assert np.all(np.isfinite(a.data))
        a.graph.validate()
    with pytest.raises(ValueError, match="linear functional"):
        scm.sample(n=50, seed=0, scale="unitless")


def test_a_soft_intervention_scales_the_amplitudes_or_redraws_the_function():
    from andrey.data.interventions import soft

    spec = data.functional.random_features(n_features=10)
    weights = np.random.default_rng(0).standard_normal((4, 3, 10))
    scaled = spec.scale_effect(weights, 0.0)
    assert np.array_equal(scaled[:, :2], weights[:, :2])
    assert np.all(scaled[:, 2] == 0)

    scm = data.SCM(
        graph=data.graphs.erdos_renyi(d=2, avg_degree=1),
        functional=spec,
        noise=data.noise.gaussian(),
    )
    plain = scm.sample(n=[300, 300], seed=0, scale="raw", shuffle=False, envs=[{}, {}])
    ((parent, child),) = plain.params.edges.tolist()
    rows = plain.env == 1
    cut = scm.sample(
        n=[300, 300], seed=0, scale="raw", shuffle=False, envs=[{}, {child: soft(weight_scale=0.0)}]
    )
    anew = scm.sample(
        n=[300, 300], seed=0, scale="raw", shuffle=False, envs=[{}, {child: soft(weights="redraw")}]
    )
    frequencies, phases, amplitudes = plain.params.weights[0]
    x = plain.data[rows, parent]
    signal = np.sqrt(2.0 / 10) * np.cos(np.outer(x, frequencies) + phases) @ amplitudes
    np.testing.assert_allclose(cut.data[rows, child], plain.data[rows, child] - signal)
    assert np.array_equal(cut.data[~rows], plain.data[~rows])
    assert not np.allclose(anew.data[rows, child], plain.data[rows, child])
    assert np.array_equal(anew.data[:, parent], plain.data[:, parent])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_features": 0}, "n_features must be a whole number >= 1"),
        ({"n_features": 2.5}, "n_features must be a whole number >= 1"),
        ({"length_scale": 0.0}, "length_scale must be finite and positive"),
        ({"length_scale": np.inf}, "length_scale must be finite and positive"),
    ],
)
def test_random_features_rejects_parameters_outside_their_range(kwargs, match):
    with pytest.raises(ValueError, match=match):
        data.functional.random_features(**kwargs)


# ---- post-nonlinear outer transforms -----------------------------------------------------------


@pytest.mark.parametrize("outer", sorted(OUTER_TRANSFORMS))
def test_an_outer_transform_is_strictly_increasing(outer):
    z = np.concatenate([np.linspace(-20.0, 20.0, 40_001), np.linspace(-1e-3, 1e-3, 2001)])
    z = np.unique(z)
    values = OUTER_TRANSFORMS[outer](z)
    assert np.all(np.isfinite(values))
    assert np.all(np.diff(values) > 0)


@pytest.mark.parametrize("outer", ["cube", "cube_root", "sigmoid"])
def test_an_outer_transform_changes_only_the_shape_of_the_child(outer):
    """The inner argument is the default's, so the child keeps the default's order of the rows."""
    default, parent, child = _pair(data.functional.post_nonlinear(), data.noise.laplace())
    ds, _, _ = _pair(data.functional.post_nonlinear(outer=outer), data.noise.laplace())
    assert np.array_equal(ds.params.edges, default.params.edges)
    assert np.array_equal(ds.params.weights, default.params.weights)
    assert np.array_equal(ds.data[:, parent], default.data[:, parent])
    assert np.array_equal(np.argsort(ds.data[:, child]), np.argsort(default.data[:, child]))
    assert not np.allclose(ds.data[:, child], default.data[:, child])
    assert ds.provenance["config"]["functional_params"] == {"outer": outer}
    assert "functional_params" not in default.provenance["config"]


def test_the_outer_transforms_are_the_documented_functions():
    z = np.array([-8.0, -0.5, 0.0, 0.5, 8.0])
    np.testing.assert_allclose(OUTER_TRANSFORMS["tanh"](z), z + 0.5 * np.tanh(z))
    np.testing.assert_allclose(OUTER_TRANSFORMS["cube"](z), z**3)
    np.testing.assert_allclose(OUTER_TRANSFORMS["cube_root"](z) ** 3, z, atol=1e-12)
    np.testing.assert_allclose(OUTER_TRANSFORMS["sigmoid"](z), 1.0 / (1.0 + np.exp(-z)))


@pytest.mark.parametrize("outer", ["cube", "cube_root", "sigmoid"])
def test_outer_transforms_stay_finite_on_a_graph_under_rescale(outer):
    scm = data.SCM(
        graph=data.graphs.erdos_renyi(d=30, avg_degree=3),
        functional=data.functional.post_nonlinear(outer=outer),
        noise=data.noise.laplace(),
    )
    ds = scm.sample(n=300, seed=0, scale="rescale")
    assert np.all(np.isfinite(ds.data))
    assert np.array_equal(ds.data, scm.sample(n=300, seed=0, scale="rescale").data)


def test_an_unknown_outer_transform_names_the_allowed_ones():
    with pytest.raises(ValueError, match=r"outer must be one of \('tanh', 'cube', 'cube_root'"):
        data.functional.post_nonlinear(outer="exp")
