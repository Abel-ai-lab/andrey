import numpy as np

import andrey
from andrey.data import sample_scm

if __name__ == "__main__":  # worker processes re-import this script
    data = sample_scm(d=200, n=2000, density=2.0, seed=0)
    true = data.graph.adjacency != 0

    # The same fit with one worker and with four.
    graphs = {}
    for workers in (1, 4):
        with andrey.config(num_workers=workers):
            graphs[workers] = andrey.ges(data.data).structure.adjacency

    found = graphs[1] != 0
    print(f"{(found & true).sum()} of {true.sum()} true edges found")
    print("same graph with 1 and 4 workers:", np.array_equal(graphs[1], graphs[4]))
