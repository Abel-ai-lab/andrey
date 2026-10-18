import numpy as np

import andrey
from andrey.data import sample_scm

data = sample_scm(d=50, n=500, density=4.0, scale="standardize", seed=0).data
rng = np.random.default_rng(0)
orders = [rng.permutation(data.shape[1]) for _ in range(10)]

for rule in ("column", "p_value"):
    graphs = set()
    for order in orders:
        fit = andrey.pc(data[:, order], alpha=0.05, collider_order=rule)
        back = np.argsort(order)  # put the columns back in their first order
        graphs.add(fit.structure.adjacency[np.ix_(back, back)].tobytes())
    count = len(graphs)
    print(f"collider_order={rule!r}: {count} graph{'s' if count > 1 else ''} from 10 column orders")
