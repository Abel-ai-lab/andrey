import numpy as np

import andrey
from andrey.data import sample_scm
from andrey.metrics import shd

# GES on 50 variables and 2,000 rows, with the map's settings: a sparse graph, then a dense one.
for degree in (2, 8):
    data = sample_scm(d=50, n=2000, density=degree, scale="standardize", seed=0)
    errors = shd(andrey.ges(data.data, max_parents=4).structure, data.graph)
    empty = shd(np.zeros_like(data.graph.adjacency), data.graph)
    print(f"mean degree {degree}: {errors} errors; the empty graph makes {empty}")
