import numpy as np
import pandas as pd

import andrey
from andrey.core.structure import GraphStructure
from andrey.data import sample_scm
from andrey.metrics import shd

# Ten variables on a known graph, each column the exponential of a Gaussian one: log-normal, as
# many lab values are.
truth = sample_scm(d=10, n=4000, seed=2, density=3.0)
names = [f"x{i}" for i in range(10)]
data = np.exp(pd.DataFrame(truth.data, columns=names))
graph = GraphStructure.from_edges(truth.graph.to_edges(), n_nodes=10, kind="dag", labels=names)
learn, held_out = data.iloc[:2000], data.iloc[2000:]  # learn on one half, check on the other


if __name__ == "__main__":  # worker processes re-import this script
    # The test: Fisher's z assumes Gaussian columns; normal_scores reads only their ranks.
    for test in ["fisherz", "normal_scores"]:
        fit = andrey.pc(learn, indep_test=test)
        print(f"{test}: {shd(fit.structure, graph)} errors against the true graph")

    # The check: does the held-out half contradict a graph?
    for test in ["fisherz", "normal_scores"]:
        check = andrey.markov_check(graph, held_out, indep_test=test)
        print(
            f"true graph, checked with {test}: p = {check.uniformity_p_value:.3f}, "
            f"{'passed' if check.passed else 'rejected'}"
        )
    edges = graph.to_edges()
    missing_one = GraphStructure.from_edges(edges[1:], n_nodes=10, kind="dag", labels=names)
    check = andrey.markov_check(missing_one, held_out, indep_test="normal_scores")
    print(
        f"one edge missing: p = {check.uniformity_p_value:.3f}, "
        f"{'passed' if check.passed else 'rejected'}"
    )

    # How stable: 100 fits on half-size subsamples, four at a time, and their averaged graph.
    with andrey.config(n_jobs=4):
        boot = andrey.bootstrap("pc", learn, n_resamples=100, seed=0, indep_test="normal_scores")
    pairs = {frozenset(e[:2]) for e in graph.oriented_edges()}
    stable = boot.table(min_frequency=0.9)
    true = sum(frozenset((e["source"], e["target"])) in pairs for e in stable)
    print(f"edges in at least 9 fits in 10: {len(stable)}, {true} of them in the true graph")
    averaged = boot.structure
    print(f"averaged graph: {boot.metadata['averaged_graph']}, a {averaged.kind}")

    # Ask the graph: what screens x0 off from the rest?
    print(f"Markov blanket of x0: {averaged.markov_blanket('x0')}")
