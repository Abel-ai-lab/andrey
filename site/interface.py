import numpy as np
import pandas as pd

import andrey

# The sprinkler network: season drives rain and the sprinkler, both wet the grass, and wet grass
# is slippery. Non-Gaussian noise lets DirectLiNGAM orient every edge.
rng = np.random.default_rng(0)
n = 5000
season = rng.laplace(size=n)
rain = 0.8 * season + rng.laplace(size=n)
sprinkler = 0.6 * season + rng.laplace(size=n)
wet = rain + sprinkler + rng.laplace(size=n)
slippery = wet + rng.laplace(size=n)
X = pd.DataFrame(
    {"season": season, "rain": rain, "sprinkler": sprinkler, "wet": wet, "slippery": slippery}
)

graphs = {}
for learn in (andrey.pc, andrey.fci, andrey.direct_lingam, andrey.mmpc):
    graph = learn(X).structure  # a GraphStructure, whichever method ran
    edges = len(graph.to_edges())
    print(f"{learn.__name__:<15}{graph.kind:<11}{edges} edges")
    graphs[learn.__name__] = graph

# The same lawn, day by day: the rain wets it that day, the sprinkler the next day, and a lawn
# that was dry turns the sprinkler on the next day. VARMA-LiNGAM returns a graph for each lag.
# The columns' order is the collapsed graph's, left to right.
rain = rng.laplace(size=n)
sprinkler, wet = np.zeros(n), np.zeros(n)
for t in range(1, n):
    sprinkler[t] = -0.6 * wet[t - 1] + rng.laplace()
    wet[t] = 0.8 * rain[t] + 0.7 * sprinkler[t - 1] + rng.laplace()
days = pd.DataFrame({"sprinkler": sprinkler, "wet": wet, "rain": rain})
over_time = andrey.varma_lingam(days, order=(1, 0), prune=True).structure
