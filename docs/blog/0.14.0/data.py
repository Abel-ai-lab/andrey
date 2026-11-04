import pandas as pd

import andrey
import andrey.data as data
from andrey.data.interventions import hard, soft

if __name__ == "__main__":  # worker processes re-import this script
    scm = data.SCM(
        graph=data.graphs.erdos_renyi(d=4, avg_degree=2),
        functional=data.functional.linear(),
        noise=data.noise.gaussian(),
    )
    # Three environments: observed, do(x1 = 2), and the noise of x3 tripled.
    envs = [{}, {1: hard(2.0)}, {3: soft(noise_scale=3.0)}]
    ds = scm.sample(n=[500, 500, 500], seed=0, envs=envs)

    # One table: a column per variable, and a column naming each row's environment.
    names = {0: "observed", 1: "do(x1 = 2)", 2: "noise x3"}
    df = pd.DataFrame(ds.data, columns=["x0", "x1", "x2", "x3"]).round(2)
    df["env"] = [names[e] for e in ds.env]
    print(df.groupby("env", sort=False).head(2).to_string())

    # And each environment's targets, by column name.
    targets = {"observed": [], "do(x1 = 2)": ["x1"], "noise x3": ["x3"]}
    result = andrey.gies(df, "env", targets=targets)
    print(result.structure.oriented_edges())
