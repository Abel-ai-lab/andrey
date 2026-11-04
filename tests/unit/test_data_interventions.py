"""Interventions in ``andrey.data``: ``SCM.do``, ``envs=`` pooling, and the discrete sampler.

The checks that compare one sample with another are exact: ``SCM.do(...).sample(n, seed)`` draws
the noise ``sample(n, seed)`` draws, so a column an intervention does not reach is equal in the two.
The checks on distributions use fixed seeds, large samples, and a level of ``1e-3`` split over
the columns tested.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from scipy import stats

import andrey.data as data
from andrey.data.interventions import HardIntervention, hard, soft

FUNCTIONALS = ["linear", "additive_noise", "post_nonlinear", "nonmonotone", "location_scale"]
SCALES = ["standardize", "rescale", "raw"]
ALPHA = 1e-3


def _scm(functional="linear", d=10, density=3.0, latents=0):
    return data.SCM(
        graph=data.graphs.erdos_renyi(d=d, avg_degree=density),
        functional=getattr(data.functional, functional)(),
        noise=data.noise.gaussian(),
        latents=latents,
    )


def _digraph(ds) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_nodes_from(range(ds.graph.n_nodes))
    g.add_edges_from(ds.params.edges.tolist())
    return g


def _with_parents_and_children(ds) -> int:
    """An observed column with both a parent and a child."""
    g = _digraph(ds)
    return next(j for j in range(ds.data.shape[1]) if g.in_degree(j) and g.out_degree(j))


@pytest.mark.parametrize("scale", [*SCALES, "unitless"])
def test_values_and_shifts_are_in_the_units_of_the_returned_data(scale):
    scm = _scm()
    fixed = scm.do({4: hard(2.0)}).sample(n=500, seed=3, scale=scale)
    assert np.all(fixed.data[:, 4] == 2.0)
    drawn = scm.do({4: hard(1.5, sd=0.3)}).sample(n=3000, seed=3, scale=scale).data[:, 4]
    assert stats.kstest(drawn, stats.norm(1.5, 0.3).cdf).pvalue > ALPHA
    plain = scm.sample(n=500, seed=3, scale=scale)
    shifted = scm.do({4: soft(shift=0.7)}).sample(n=500, seed=3, scale=scale)
    assert np.allclose(shifted.data[:, 4] - plain.data[:, 4], 0.7)


@pytest.mark.parametrize("functional", FUNCTIONALS)
@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize(
    "intervention",
    [
        hard(1.0, sd=0.5),
        soft(shift=1.0, noise_scale=2.0),
        soft(noise_scale=(0.5, 3.0)),
        soft(weights="redraw"),
    ],
)
def test_an_intervention_changes_only_its_column_and_descendants(functional, scale, intervention):
    scm = _scm(functional, latents=2 if functional == "linear" else 0)
    plain = scm.sample(n=400, seed=1, scale=scale)
    target = _with_parents_and_children(plain)
    under = scm.do({target: intervention}).sample(n=400, seed=1, scale=scale)
    reached = nx.descendants(_digraph(plain), target) | {target}
    for j in range(plain.data.shape[1]):
        if j not in reached:
            assert np.array_equal(under.data[:, j], plain.data[:, j])
    assert not np.array_equal(under.data[:, target], plain.data[:, target])
    assert under.graph == plain.graph


@pytest.mark.parametrize("scale", ["raw", "standardize"])
@pytest.mark.parametrize("intervention", [soft(shift=0.7), hard(2.0, sd=0.5)])
def test_descendants_move_by_the_total_effect(scale, intervention):
    """In linear data, a change at the target moves column ``j`` by the target's total effect on it.

    The change is in the units of the returned data, so under ``"standardize"`` it is converted to
    the model's units with the standard deviations of the sample without interventions.
    """
    scm = _scm(d=12)
    raw = scm.sample(n=200, seed=5, scale="raw")
    target = _with_parents_and_children(raw)
    weights = np.zeros((12, 12))
    weights[raw.params.edges[:, 0], raw.params.edges[:, 1]] = raw.params.weights
    total = np.linalg.inv(np.eye(12) - weights)  # total[i, j]: the effect of i on j
    units = raw.data.std(axis=0) if scale == "standardize" else np.ones(12)
    plain = scm.sample(n=200, seed=5, scale=scale)
    under = scm.do({target: intervention}).sample(n=200, seed=5, scale=scale)
    change = under.data[:, target] - plain.data[:, target]
    if isinstance(intervention, HardIntervention):
        assert stats.kstest(under.data[:, target], stats.norm(2.0, 0.5).cdf).pvalue > ALPHA
    else:
        assert np.allclose(change, 0.7)
    expected = np.outer(change * units[target], total[target]) / units
    assert np.allclose(under.data - plain.data, expected)


def test_a_pooled_descendant_shifts():
    scm = _scm(d=12)
    raw = scm.sample(n=10, seed=5, scale="raw")
    target = _with_parents_and_children(raw)
    child = next(iter(_digraph(raw).successors(target)))
    pooled = scm.sample(n=2000, seed=5, envs=[{}, {target: soft(shift=1.0)}, {target: 2.0}])
    for k in (1, 2):
        rows = [pooled.data[pooled.env == e, child] for e in (0, k)]
        assert stats.ks_2samp(*rows).pvalue < 1e-6


@pytest.mark.parametrize("functional", FUNCTIONALS)
def test_pooled_non_descendants_keep_their_marginals(functional):
    scm = _scm(functional, d=12)
    probe = scm.sample(n=10, seed=11)
    graph = _digraph(probe)
    first = _with_parents_and_children(probe)
    second = next(j for j in range(12) if j != first and graph.in_degree(j))
    envs = [{}, {first: hard(2.0, sd=0.5)}, {second: soft(shift=1.5, weight_scale=0.5)}]
    pooled = scm.sample(n=3000, seed=11, envs=envs)
    observational = pooled.data[pooled.env == 0]
    for k, target in ((1, first), (2, second)):
        under = pooled.data[pooled.env == k]
        reached = nx.descendants(graph, target) | {target}
        rest = [j for j in range(12) if j not in reached]
        for j in rest:
            p = stats.ks_2samp(observational[:, j], under[:, j]).pvalue
            assert p > ALPHA / len(rest), (k, j, p)
        assert stats.ks_2samp(observational[:, target], under[:, target]).pvalue < 1e-6


def test_soft_keeps_the_parents_and_hard_cuts_them():
    scm = _scm(d=12)
    probe = scm.sample(n=10, seed=2, scale="raw")
    target = _with_parents_and_children(probe)
    into = probe.params.edges[:, 1] == target
    parents, weights = probe.params.edges[into, 0], probe.params.weights[into]
    envs = [{}, {target: soft(shift=1.0)}, {target: soft(weight_scale=0.5)}, {target: 1.0}]
    pooled = scm.sample(n=5000, seed=2, scale="raw", envs=envs)

    def coefficients(k):
        rows = pooled.data[pooled.env == k]
        design = np.column_stack([rows[:, parents], np.ones(len(rows))])
        return np.linalg.lstsq(design, rows[:, target], rcond=None)[0][:-1]

    assert np.allclose(coefficients(1), weights, atol=0.1)
    assert np.allclose(coefficients(2), 0.5 * weights, atol=0.1)
    assert np.all(pooled.data[pooled.env == 3, target] == 1.0)


def _parent_weights(ds, target):
    """The parents of ``target`` and the weights of their edges into it."""
    into = ds.params.edges[:, 1] == target
    return ds.params.edges[into, 0], ds.params.weights[into]


def _coefficients(ds, k, parents, target):
    """The least-squares coefficients of ``target`` on ``parents`` in environment ``k``."""
    rows = ds.data[ds.env == k]
    design = np.column_stack([rows[:, parents], np.ones(len(rows))])
    return np.linalg.lstsq(design, rows[:, target], rcond=None)[0][:-1]


@pytest.mark.parametrize("positive", [False, True])
def test_redraw_draws_each_parent_weight_on_its_own(positive):
    scm = data.SCM(
        graph=data.graphs.erdos_renyi(d=12, avg_degree=4.0),
        functional=data.functional.linear(weight_range=(0.5, 1.0), positive=positive),
        noise=data.noise.gaussian(),
    )
    probe = scm.sample(n=10, seed=2, scale="raw")
    target = max(range(12), key=lambda j: int(np.sum(probe.params.edges[:, 1] == j)))
    parents, weights = _parent_weights(probe, target)
    assert len(parents) >= 3
    envs = [{}, {target: soft(weights="redraw")}, {target: soft(weights="redraw")}]
    pooled = scm.sample(n=20000, seed=2, scale="raw", envs=envs)
    assert np.allclose(_coefficients(pooled, 0, parents, target), weights, atol=0.05)
    new = [_coefficients(pooled, k, parents, target) for k in (1, 2)]
    for w in new:
        assert np.all((np.abs(w) > 0.45) & (np.abs(w) < 1.05))
        assert np.ptp(w / weights) > 0.2  # not one factor on every weight
        assert not positive or np.all(w > 0)
    assert not np.allclose(new[0], new[1], atol=0.1)  # each environment draws its own


def test_redrawn_weights_are_part_of_the_model():
    scm = _scm(d=12, density=4.0)
    probe = scm.sample(n=10, seed=7, scale="raw")
    target = _with_parents_and_children(probe)
    parents, weights = _parent_weights(probe, target)
    envs = [{}, {target: soft(weights="redraw")}]
    fits = [
        _coefficients(
            scm.sample(n=20000, seed=seed, model_seed=7, scale="raw", envs=envs), 1, parents, target
        )
        for seed in (1, 2)
    ]
    assert np.allclose(fits[0], fits[1], atol=0.05)
    assert not np.allclose(fits[0], weights, atol=0.1)


def test_a_noise_scale_range_multiplies_each_row_by_its_own_draw():
    """The target's noise is its Gaussian noise times a ``U[1, 3]`` draw per row: a scale mixture
    with variance ``13 / 3``, not a Gaussian of that variance."""
    scm = _scm(d=10)
    probe = scm.sample(n=10, seed=6, scale="raw")
    target = _with_parents_and_children(probe)
    parents, weights = _parent_weights(probe, target)
    envs = [{}, {target: soft(noise_scale=(1.0, 3.0))}]
    pooled = scm.sample(n=20000, seed=6, scale="raw", envs=envs)
    rows = pooled.data[pooled.env == 1]
    noise = (rows[:, target] - rows[:, parents] @ weights) / probe.params.noise_scales[target]
    factors = np.linspace(1.0, 3.0, 201)[:-1] + 0.005  # midpoints of U[1, 3]

    def mixture(x):
        return stats.norm.cdf(np.asarray(x)[:, None] / factors).mean(axis=1)

    assert stats.kstest(noise, mixture).pvalue > ALPHA
    assert stats.kstest(noise, stats.norm(0.0, np.sqrt(13 / 3)).cdf).pvalue < 1e-6


def test_pooled_env_and_targets_match_the_rows():
    scm = _scm(d=8, latents=2)
    envs = [{}, {1: hard(-3.0)}, {0: hard(5.0), 2: soft(shift=1.0)}, {}]
    pooled = scm.sample(n=[50, 20, 30, 10], seed=4, envs=envs)
    assert pooled.data.shape == (110, 6)
    assert pooled.env.tolist() == [0] * 50 + [1] * 20 + [2] * 30 + [3] * 10
    assert pooled.targets == {0: [], 1: [1], 2: [0, 2], 3: []}
    for k, spec in enumerate(envs):
        for column, intervention in spec.items():
            if isinstance(intervention, HardIntervention):
                values = pooled.data[:, column]
                assert np.all(values[pooled.env == k] == intervention.value)
                assert not np.any(values[pooled.env != k] == intervention.value)
    assert [e["n"] for e in pooled.provenance["config"]["envs"]] == [50, 20, 30, 10]
    assert pooled.provenance["config"]["n"] == 110


def test_targets_name_the_shuffled_columns():
    """Column ``j`` of a shuffled sample is node ``column_order[j]`` of the unshuffled one."""
    scm = _scm(d=9)
    shuffled = scm.sample(n=100, seed=8, envs=[{}, {2: hard(1.0)}, {5: soft(shift=1.0)}])
    order = shuffled.provenance["column_order"]
    unshuffled = scm.sample(
        n=100,
        seed=8,
        shuffle=False,
        envs=[{}, {order[2]: hard(1.0)}, {order[5]: soft(shift=1.0)}],
    )
    assert np.array_equal(shuffled.data, unshuffled.data[:, order])
    assert np.array_equal(shuffled.env, unshuffled.env)


def test_samples_repeat_with_the_seed():
    scm = _scm()
    envs = [{}, {3: hard(1.0, sd=0.2)}, {6: soft(noise_scale=3.0)}]
    a = scm.sample(n=[100, 40, 40], seed=9, envs=envs)
    b = scm.sample(n=[100, 40, 40], seed=9, envs=envs)
    assert np.array_equal(a.data, b.data) and np.array_equal(a.env, b.env)
    assert a.targets == b.targets
    assert not np.array_equal(a.data, scm.sample(n=[100, 40, 40], seed=10, envs=envs).data)
    plain = scm.sample(n=100, seed=9)
    assert np.array_equal(scm.sample(n=100, seed=9, envs=[{}]).data, plain.data)
    assert plain.env is None and plain.targets is None
    assert np.array_equal(
        scm.do(envs[1]).sample(n=40, seed=9).data,
        scm.sample(n=40, seed=9, envs=[envs[1]]).data,
    )


def test_an_environment_draws_its_own_noise():
    """In raw units, an environment's rows depend on the seed, its place, and its row count only."""
    scm = _scm()
    envs = [{}, {3: hard(1.0, sd=0.2)}, {6: soft(noise_scale=3.0)}]
    a = scm.sample(n=[100, 40, 40], seed=9, envs=envs, scale="raw")
    b = scm.sample(n=[300, 40, 40], seed=9, envs=envs, scale="raw")
    assert np.array_equal(a.data[a.env != 0], b.data[b.env != 0])
    twins = scm.sample(n=40, seed=9, envs=[{}, {}, {}], scale="raw").data.reshape(3, 40, -1)
    assert not np.array_equal(twins[0], twins[1]) and not np.array_equal(twins[1], twins[2])


def test_do_merges_and_records_its_interventions():
    scm = _scm().do({3: hard(1.0)}).do({3: 2.0, 1: soft(shift=1.0)})
    assert scm.interventions == ((1, soft(shift=1.0)), (3, hard(2.0)))
    ds = scm.sample(n=50, seed=0)
    assert ds.targets == {0: [1, 3]} and ds.env.tolist() == [0] * 50
    assert ds.provenance["config"]["envs"][0]["interventions"]["3"]["value"] == 2.0


def test_discrete_hard_soft_and_untouched_columns():
    kwargs = {"d": 10, "density": 3.0, "seed": 6, "categories": (3, 3), "concentration": 0.3}
    plain = data.sample_discrete(n=4000, **kwargs)
    graph = _digraph(plain)
    target = _with_parents_and_children(plain)
    parent = next(iter(graph.predecessors(target)))
    child = next(iter(graph.successors(target)))
    envs = [{target: hard(1)}, {target: hard()}, {target: soft()}, {target: hard(probs=[0, 0, 1])}]
    pooled = data.sample_discrete(n=4000, envs=envs, **kwargs)
    rows = [pooled.data[pooled.env == k] for k in range(4)]
    assert np.all(rows[0][:, target] == 1) and np.all(rows[3][:, target] == 2)
    counts = np.bincount(rows[1][:, target].astype(int), minlength=3)
    assert stats.chisquare(counts).pvalue > ALPHA

    def dependence(x):
        table = np.ones((3, 3))
        np.add.at(table, (x[:, parent].astype(int), x[:, target].astype(int)), 1)
        return stats.chi2_contingency(table).pvalue

    assert dependence(plain.data) < 1e-6 and dependence(rows[2]) < 1e-6
    assert dependence(rows[1]) > ALPHA
    # Environment 0 draws the samples of a sample without envs.
    reached = nx.descendants(graph, target) | {target}
    for j in range(10):
        if j not in reached:
            assert np.array_equal(rows[0][:, j], plain.data[:, j])
    shift = np.zeros((2, 3))
    for i, x in enumerate((plain.data, rows[3])):
        shift[i] = np.bincount(x[:, child].astype(int), minlength=3)
    assert stats.chi2_contingency(shift + 1).pvalue < 1e-6
    assert pooled.targets == {0: [target], 1: [target], 2: [target], 3: [target]}


def test_discrete_without_envs_is_unchanged():
    plain = data.sample_discrete(d=6, n=200, seed=2)
    same = data.sample_discrete(d=6, n=200, seed=2, envs=[{}])
    assert np.array_equal(plain.data, same.data)
    assert plain.env is None and same.targets == {0: []}


@pytest.mark.parametrize(
    ("envs", "match"),
    [
        ([{10: hard(1.0)}], "not one of the observed columns"),
        ([{-1: hard(1.0)}], "column number >= 0"),
        ([{8: hard(1.0)}], "not one of the observed columns"),  # a latent node
        ([{True: hard(1.0)}], "column number"),
        ([{1: "high"}], "an intervention is"),
        ([{1: hard()}], "takes a value"),
        ([{1: hard(probs=[0.5, 0.5])}], "takes a value"),
        ([{1: soft()}], "changes nothing"),
        ({1: hard(1.0)}, "one per environment"),
        ([], "non-empty"),
    ],
)
def test_bad_continuous_envs_raise(envs, match):
    with pytest.raises((ValueError, TypeError), match=match):
        _scm(latents=2).sample(n=20, seed=0, envs=envs)


def test_bad_sample_arguments_raise():
    scm = _scm()
    with pytest.raises(ValueError, match="row counts"):
        scm.sample(n=[10, 10], seed=0, envs=[{}])
    with pytest.raises(ValueError, match="at least one row|>= 1"):
        scm.sample(n=[10, 0], seed=0, envs=[{}, {}])
    with pytest.raises(TypeError, match="one row count"):
        scm.sample(n=[10], seed=0)
    with pytest.raises(ValueError, match="takes no envs"):
        scm.do({1: 1.0}).sample(n=10, seed=0, envs=[{}])
    with pytest.raises(ValueError, match="screen"):
        scm.sample(n=10, seed=0, envs=[{}], screen=data.FaithfulnessScreen(min_abs_corr=0.1))
    temporal = data.SCM(
        graph=data.graphs.erdos_renyi(d=4, avg_degree=1),
        functional=data.functional.var(),
        noise=data.noise.gaussian(),
    )
    with pytest.raises(ValueError, match="temporal"):
        temporal.sample(n=10, seed=0, envs=[{}])
    with pytest.raises(ValueError, match="model_seed"):
        scm.sample(n=10, seed=0, model_seed=1, screen=data.FaithfulnessScreen(min_abs_corr=0.1))
    with pytest.raises(ValueError, match="unitless"):
        scm.sample(n=10, seed=0, scale="unitless", envs=[{1: soft(weights="redraw")}])


@pytest.mark.parametrize(
    ("intervention", "match"),
    [
        (hard(3), "categories 0..2"),
        (hard(0.5), "categories 0..2"),
        (hard(probs=[0.5, 0.5]), "3 categories"),
        (hard(1.0, sd=0.2), "no sd"),
        (soft(shift=1.0), "takes no arguments"),
    ],
)
def test_bad_discrete_interventions_raise(intervention, match):
    with pytest.raises(ValueError, match=match):
        data.sample_discrete(d=5, n=20, seed=0, categories=(3, 3), envs=[{1: intervention}])


@pytest.mark.parametrize(
    ("build", "match"),
    [
        (lambda: hard(np.inf), "finite value"),
        (lambda: hard(1.0, sd=-1.0), "sd >= 0"),
        (lambda: hard(1.0, probs=[0.5, 0.5]), "not both"),
        (lambda: hard(probs=[0.5, 0.6]), "sum to 1"),
        (lambda: soft(shift=np.nan), "finite"),
        (lambda: soft(noise_scale=-1.0), "noise_scale >= 0"),
        (lambda: soft(noise_scale=(2.0, 1.0)), "pair 0 <= low <= high"),
        (lambda: soft(noise_scale=(1.0, 2.0, 3.0)), "pair 0 <= low <= high"),
        (lambda: soft(weights="new"), "'keep' or 'redraw'"),
        (lambda: soft(weights="redraw", weight_scale=2.0), "not both"),
    ],
)
def test_bad_specs_raise(build, match):
    with pytest.raises(ValueError, match=match):
        build()
