import andrey
from andrey.data import sample_scm
from andrey.metrics import shd


def draw(functional):
    """1,000 rows of ten variables, each a nonlinear function of its parents plus Gaussian noise."""
    return sample_scm(
        graph="erdos_renyi",
        functional=functional,
        noise="gaussian",
        d=10,
        n=1000,
        seed=0,
        density=2.0,
    )


def in_order(fit, truth):
    """The number of true edges whose cause comes before its effect in the causal order."""
    place = {variable: i for i, variable in enumerate(fit.ordering)}
    return sum(place[cause] < place[effect] for cause, effect, _ in truth.graph.oriented_edges())


if __name__ == "__main__":  # worker processes re-import this script
    # "random_features" draws each edge's function from a Gaussian process, the kind of function
    # the SCORE and NoGAM papers simulate. "additive_noise" gives the data of the example above.
    for functional in ("random_features", "additive_noise"):
        truth = draw(functional)
        print(f"{functional}: {len(truth.graph.oriented_edges())} edges")
        for method in (andrey.score_matching, andrey.nogam, andrey.cam):
            fit = method(truth.data)
            errors, placed = shd(fit.structure, truth.graph), in_order(fit, truth)
            name = method.__name__ + ":"
            print(f"  {name:<16}errors {errors:>2}, true edges in causal order {placed:>2}")
