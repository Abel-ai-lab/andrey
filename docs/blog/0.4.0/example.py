import numpy as np
import pandas as pd

import andrey

# x causes y, which a saturating sensor reads: y = tanh(x + e).
rng = np.random.default_rng(0)
x = rng.laplace(size=500)
y = np.tanh(x + 0.5 * rng.laplace(size=500))
df = pd.DataFrame({"x": x, "y": y})

print(andrey.anm(df), andrey.pnl(df), sep="\n\n")

# For comparison: ANM with a significance level.
print()
print(andrey.anm(df, alpha=0.05))
