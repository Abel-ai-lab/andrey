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
for learn in (andrey.pc, andrey.fci, andrey.direct_lingam):
    graph = learn(X).structure  # a GraphStructure, whichever method ran
    edges = len(graph.to_edges())
    print(f"{learn.__name__:<15}{graph.kind:<7}{edges} edges")
    graphs[learn.__name__] = graph
