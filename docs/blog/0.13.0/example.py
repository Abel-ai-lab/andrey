import andrey
from andrey.data import sample_scm
from andrey.metrics import shd


def draw(d, noise="gaussian"):
    """1,000 rows of d variables, each a nonlinear function of its parents plus noise."""
    return sample_scm(
        graph="erdos_renyi",
        functional="additive_noise",
        noise=noise,
        d=d,
        n=1000,
        seed=0,
        density=2.0,
    )


if __name__ == "__main__":  # worker processes re-import this script
    truth = draw(10)
    print(f"true graph: {len(truth.graph.oriented_edges())} edges")

    # PC, with a kernel test for nonlinear dependence, leaves an edge undirected when the
    # independences alone cannot orient it.
    pc = andrey.pc(truth.data, indep_test="kci")
    undirected = sum(kind == "undirected" for *_, kind in pc.structure.oriented_edges())
    print(f"pc:            {shd(pc.structure, truth.graph)} errors, {undirected} edges undirected")

    # DirectLiNGAM orients every edge, but assumes linear effects.
    lingam = andrey.direct_lingam(truth.data)
    print(f"direct_lingam: {shd(lingam.structure, truth.graph)} errors")

    # CAM assumes nonlinear additive effects, under which only one direction fits.
    cam = andrey.cam(truth.data)
    print(f"cam:           {shd(cam.structure, truth.graph)} errors, causal order {cam.ordering}")

    # RESIT fits the same model with noise of any distribution, for a handful of variables.
    small = draw(5, noise="gumbel")
    resit = andrey.resit(small.data)
    print(f"resit:         {shd(resit.structure, small.graph)} errors on 5 variables, Gumbel noise")
