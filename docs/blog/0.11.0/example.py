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


def edges(fit):
    a = np.asarray(fit.structure.adjacency)
    return int(((a != 0) | (a.T != 0)).sum() // 2)


def into_age_or_sex(fit):
    oriented = fit.structure.oriented_edges()
    return sum(kind == "directed" and target in ("age", "sex") for _, target, kind in oriented)


if __name__ == "__main__":  # worker processes re-import this script
    files = ["DEMO_J", "BMX_J", "BPX_J", "GHB_J", "TCHOL_J", "HDL_J"]
    table = pd.read_sas(URL.format(files[0]), format="xport")
    for name in files[1:]:
        table = table.merge(pd.read_sas(URL.format(name), format="xport"), on="SEQN", how="left")
    data = table[table.RIDAGEYR >= 20][list(COLUMNS)].rename(columns=COLUMNS)
    # Sex is a category, so each test is one for mixed data.
    data["sex"] = data["sex"].map({1: "male", 2: "female"}).astype("category")
    print(f"{len(data):,} adults; {len(data.dropna()):,} with no value missing")

    # Gaps: each test reads the samples where its variables are observed.
    every_adult = andrey.pc(data)
    complete = andrey.pc(data.dropna())
    differ = shd(every_adult.structure, complete.structure)
    print(f"gaps: {edges(every_adult)} edges from every adult, {edges(complete)} from the")
    print(f"  adults with no value missing; the two differ in {differ} places")
    print(f"test: {every_adult.metadata['indep_test']}")

    # Units: blood pressure in kPa and cholesterol in mmol/L give the same graph.
    metric = data.assign(
        systolic=data["systolic"] * 0.133322,
        diastolic=data["diastolic"] * 0.133322,
        cholesterol=data["cholesterol"] * 0.02586,
        hdl=data["hdl"] * 0.02586,
    )
    print(f"units: {shd(andrey.pc(metric).structure, every_adult.structure)} differences")

    # What you already know: nothing causes age or sex.
    known = andrey.Knowledge(tiers=[["age", "sex"], [c for c in data if c not in ("age", "sex")]])
    guided = andrey.pc(data, knowledge=known)
    before, after = into_age_or_sex(every_adult), into_age_or_sex(guided)
    print(f"arrows into age or sex: {before} without it, {after} with it")

    # How sure each edge is: 100 fits on bootstrap samples, four at a time.
    with andrey.config(num_workers=4):
        boot = andrey.bootstrap("pc", data, n_boot=100, seed=0, knowledge=known)
    print("edges in at least half the fits:")
    MARK = {"directed": "->", "undirected": "--"}
    for edge in boot.table(min_frequency=0.5):
        mark = MARK.get(edge["type"], edge["type"])
        print(f"  {edge['source']} {mark} {edge['target']}: {edge['frequency']:.2f}")

    # Hand it on: the averaged graph as Graphviz DOT, which DoWhy reads.
    print(boot.structure.to_dot().splitlines()[0])
