import numpy as np
import pandas as pd

import andrey

# x causes y in both pairs; e is noise.
rng = np.random.default_rng(0)
x = rng.laplace(size=500)
e = rng.laplace(size=500)

# The additive pair: y = x + 0.3 * x**3 + 0.5 * e.
additive = pd.DataFrame({"x": x, "y": x + 0.3 * x**3 + 0.5 * e})
# The saturating pair, which a sensor reads: y = tanh(x + 0.5 * e).
saturating = pd.DataFrame({"x": x, "y": np.tanh(x + 0.5 * e)})

print("Additive pair\n")
print(andrey.anm(additive), andrey.pnl(additive), sep="\n\n")
print("\nSaturating pair\n")
print(andrey.anm(saturating), andrey.pnl(saturating), sep="\n\n")

# For comparison: ANM on the saturating pair with a significance level.
print()
print(andrey.anm(saturating, alpha=0.05))
