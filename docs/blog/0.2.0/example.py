import numpy as np
import pandas as pd

import andrey

# h is a common cause of x0 and x1, and it is not in the data.
rng = np.random.default_rng(0)
h, e0, e1, e2, e3 = rng.uniform(-1, 1, (5, 2000))
x0, x1 = 1.5 * h + e0, 1.5 * h + e1
df = pd.DataFrame({"x0": x0, "x1": x1, "x2": 0.6 * x1 + e2, "x3": 0.6 * x0 + e3})

methods = (andrey.fci, andrey.rcd, andrey.bottom_up_parce_lingam)
print(*(method(df) for method in methods), sep="\n\n")

# For comparison: DirectLiNGAM assumes every common cause is in the data.
print()
print(andrey.direct_lingam(df))
