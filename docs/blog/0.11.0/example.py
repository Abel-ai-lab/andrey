import numpy as np
import pandas as pd

import andrey
from andrey.metrics import shd

# NHANES 2017-2018 (CDC, public domain): six files joined on the participant's number, adults 20
# and over. Survey weights are not used.
URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/{}.xpt"
COLUMNS = {
    "RIDAGEYR": "age", "RIAGENDR": "sex", "INDFMPIR": "income", "BMXBMI": "bmi",
    "BMXWAIST": "waist", "BPXSY1": "systolic", "BPXDI1": "diastolic", "BPXPLS": "pulse",
    "LBXGH": "hba1c", "LBXTC": "cholesterol", "LBDHDD": "hdl",
}  # fmt: skip
files = ["DEMO_J", "BMX_J", "BPX_J", "GHB_J", "TCHOL_J", "HDL_J"]
table = pd.read_sas(URL.format(files[0]), format="xport")
for name in files[1:]:
    table = table.merge(pd.read_sas(URL.format(name), format="xport"), on="SEQN", how="left")
data = table[table.RIDAGEYR >= 20][list(COLUMNS)].rename(columns=COLUMNS)
data["sex"] = data["sex"].map({1: "male", 2: "female"})
print(f"{len(data):,} adults; {len(data.dropna()):,} with no value missing")


def edges(fit):
    a = np.asarray(fit.structure.adjacency)
    return int(((a != 0) | (a.T != 0)).sum() // 2)


# Gaps: each test reads the rows where its variables are observed.
numeric = data.assign(sex=(data["sex"] == "female").astype(float))
every_row = andrey.pc(numeric)
complete_rows = andrey.pc(numeric.dropna())
print(f"gaps: {edges(every_row)} edges from every row, {edges(complete_rows)} from the complete")
print(f"  rows; they differ in {shd(every_row.structure, complete_rows.structure)} places")

# Units: blood pressure in kPa and cholesterol in mmol/L give the same graph.
metric = numeric.assign(
    systolic=numeric["systolic"] * 0.133322,
    diastolic=numeric["diastolic"] * 0.133322,
    cholesterol=numeric["cholesterol"] * 0.02586,
    hdl=numeric["hdl"] * 0.02586,
)
print(f"units: {shd(andrey.pc(metric).structure, every_row.structure)} differences")

# Mixed types and what you already know: sex is a category, and nothing causes age or sex.
known = andrey.Knowledge(tiers=[["age", "sex"], [c for c in data if c not in ("age", "sex")]])
settings = dict(categorical=["sex"], indep_test="degen_gauss", knowledge=known)

# How sure each edge is: 100 fits on resampled rows.
boot = andrey.bootstrap("pc", data, n_boot=100, seed=0, **settings)
print("edges in at least half the fits:")
MARK = {"directed": "->", "undirected": "--"}
for edge in boot.table(min_frequency=0.5):
    mark = MARK.get(edge["type"], edge["type"])
    print(f"  {edge['source']} {mark} {edge['target']}: {edge['frequency']:.2f}")

# Hand it on: the averaged graph as Graphviz DOT, which DoWhy reads.
print(boot.structure.to_dot().splitlines()[0])
