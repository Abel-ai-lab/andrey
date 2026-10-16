import numpy as np
import pandas as pd

import andrey

# Three daily series. Rain slows traffic the same day; traffic drives the next day's sales; each
# series also carries over from the day before.
rng = np.random.default_rng(0)
days = 1000
e = rng.uniform(-1, 1, size=(days, 3))
x = np.zeros((days, 3))
for t in range(1, days):
    rain = 0.6 * x[t - 1, 0] + e[t, 0]
    traffic = 0.5 * x[t - 1, 1] - 0.6 * rain + e[t, 1]
    sales = 0.4 * x[t - 1, 2] + 0.7 * x[t - 1, 1] + e[t, 2]
    x[t] = rain, traffic, sales
data = pd.DataFrame(x, columns=["rain", "traffic", "sales"])

# Links from the past only, then links within a day as well.
print(andrey.pcmci(data, tau_max=1), end="\n\n")
print(andrey.pcmciplus(data, tau_max=1))
