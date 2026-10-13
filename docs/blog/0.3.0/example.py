import numpy as np
import pandas as pd

import andrey

# x0 causes x1, and x0, x1, and x2 cause x3.
rng = np.random.default_rng(0)
e0, e1, e2, e3 = rng.standard_normal((4, 500))
x0, x2 = e0, e2
x1 = -0.8 * x0 + e1
x3 = 0.8 * x0 - x1 + x2 + e3
df = pd.DataFrame({"x0": x0, "x1": x1, "x2": x2, "x3": x3})

methods = (andrey.ges, andrey.hc, andrey.tabu, andrey.xges)
print(*(method(df) for method in methods), sep="\n\n")

# For comparison: XGES without its extended search.
print()
print(andrey.xges(df, extended_search=False))
