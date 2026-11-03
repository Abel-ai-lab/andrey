import andrey
import andrey.data as data
from andrey.data.interventions import soft
from andrey.metrics import shd

if __name__ == "__main__":  # worker processes re-import this script
    # Ten variables on a known graph, observed in environment 0 and changed by one intervention
    # each in environments 1 to 3: the noise of one variable grows threefold.
    scm = data.SCM(
        graph=data.graphs.erdos_renyi(d=10, avg_degree=2),
        functional=data.functional.linear(),
        noise=data.noise.gaussian(),
    )
    envs = [{}, {2: soft(noise_scale=3.0)}, {5: soft(noise_scale=3.0)}, {7: soft(noise_scale=3.0)}]
    truth = scm.sample(n=[1000] * 4, seed=0, envs=envs)
    print(f"true graph: {len(truth.graph.oriented_edges())} edges, targets {truth.targets}")

    def undirected(result):
        return sum(kind == "undirected" for *_, kind in result.structure.oriented_edges())

    # GES on the observational rows alone returns the equivalence class.
    observed = truth.data[truth.env == 0]
    ges = andrey.ges(observed)
    print(f"ges:     {shd(ges.structure, truth.graph)} errors, {undirected(ges)} edges undirected")

    # GIES is told each environment's targets.
    gies = andrey.gies(truth.data, truth.env, targets=truth.targets)
    print(
        f"gies:    {shd(gies.structure, truth.graph)} errors, {undirected(gies)} edges undirected"
    )

    # GnIES and UT-IGSP are not told the targets, and learn them. GnIES learns the variables
    # whose noise changes in some environment; UT-IGSP learns each environment's targets.
    gnies = andrey.gnies(truth.data, truth.env)
    learned = [v for v, _ in gnies.metadata["index_edges"]]
    print(
        f"gnies:   {shd(gnies.structure, truth.graph)} errors, {undirected(gnies)} edges "
        f"undirected, targets {learned}"
    )
    unknown = {0: [], 1: None, 2: None, 3: None}
    ut_igsp = andrey.ut_igsp(truth.data, truth.env, targets=unknown)
    print(
        f"ut_igsp: {shd(ut_igsp.structure, truth.graph)} errors, {undirected(ut_igsp)} edges "
        f"undirected, targets {ut_igsp.metadata['targets']}"
    )
