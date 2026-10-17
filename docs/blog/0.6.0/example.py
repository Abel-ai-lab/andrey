import numpy as np
import pandas as pd

import andrey

# Customers of a service. Their plan and their region each change how many hours they use it,
# and the hours set the monthly bill. Plan and region are text; hours and bill are numbers.
rng = np.random.default_rng(0)
n = 2000
plan = rng.choice(["basic", "pro"], size=n)
region = rng.choice(["north", "south", "west"], size=n)
shift = {"north": 0.0, "south": 3.0, "west": -3.0}
hours = 10 + 12 * (plan == "pro") + np.vectorize(shift.get)(region) + rng.normal(0, 3, n)
bill = 5 + 1.5 * hours + rng.normal(0, 5, n)
data = pd.DataFrame({"plan": plan, "region": region, "hours": hours, "bill": bill})

# The conditional Gaussian test and score handle both kinds of column.
categorical = ["plan", "region"]
print(andrey.pc(data, indep_test="cond_gauss", categorical=categorical), end="\n\n")
print(andrey.ges(data, score_func="local_score_cond_gauss", categorical=categorical))
