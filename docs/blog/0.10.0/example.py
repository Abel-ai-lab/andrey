import numpy as np

import andrey
from andrey.data import sample_scm
from andrey.metrics import shd

# PC on 50 variables and 500 rows: a sparse graph, then a dense one.
for degree in (2, 10):
    data = sample_scm(d=50, n=500, density=degree, scale="standardize", seed=0)
    errors = shd(andrey.pc(data.data).structure, data.graph)
    empty = shd(np.zeros_like(data.graph.adjacency), data.graph)
    print(f"mean degree {degree}: {errors} errors; the empty graph makes {empty}")
