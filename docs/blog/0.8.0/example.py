import warnings

import numpy as np

import andrey
from andrey.data import sample_scm

warnings.simplefilter("ignore", andrey.ExperimentalWarning)

data = sample_scm(
    d=8, n=500, functional="nonmonotone", density=2.0, scale="standardize", seed=0
)
true = data.graph.adjacency != 0
true |= true.T

for test in ("fisherz", "kci", "rcot", "cmiknn", "gpdc"):
    fit = andrey.pc(data.data, alpha=0.05, indep_test=test)
    found = fit.structure.adjacency != 0
    found |= found.T
    hits, extra = (found & true).sum() // 2, (found & ~true).sum() // 2
    print(f"{test:>7}: {hits} of {true.sum() // 2} true edges, {extra} extra")
