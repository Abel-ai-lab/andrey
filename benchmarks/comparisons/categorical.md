# Categorical data

`discrete_cpu.py` compares Andrey's PC with the chi-square and the G-square test, GES, BOSS, and
GRaSP with the BDeu score, and hill climbing with the BDeu score and the multinomial BIC against the
same method in causal-learn, bnlearn, and pgmpy, with the empty-graph baseline. Andrey's BOSS and
GRaSP with the multinomial BIC run alone, since no package here takes that score for them. It runs
on `discrete_er`: category codes from a discrete network on an Erdos-Renyi graph, each node with 2
or 3 categories and each row of its conditional probability table drawn from a flat Dirichlet
(`datasets.DISCRETE_SETTINGS`, recorded in each dataset's generator call). The codes are never
rescaled. Every solution is serial. Andrey's GES runs on the numpy backend; BDeu moves are scored
in one process.

| method | Andrey | causal-learn | bnlearn | pgmpy |
|---|---|---|---|---|
| PC, chi-square | `andrey.pc.chisq` | `causal-learn.pc.chisq` | `bnlearn.pc-stable.chisq` | |
| PC, G-square | `andrey.pc.gsq` | `causal-learn.pc.gsq` | `bnlearn.pc-stable.gsq` | |
| GES, BDeu | `andrey.ges.bdeu` | `causal-learn.ges.bdeu` | | |
| BOSS, BDeu | `andrey.boss.bdeu` | `causal-learn.boss.bdeu` | | |
| GRaSP, BDeu | `andrey.grasp.bdeu` | `causal-learn.grasp.bdeu` | | |
| BOSS, BIC | `andrey.boss.bic_categorical` | | | |
| GRaSP, BIC | `andrey.grasp.bic_categorical` | | | |
| HC, BDeu | `andrey.hc.BDeu` | | `bnlearn.hc.BDeu` | `pgmpy.hc.BDeu` |
| HC, BIC | `andrey.hc.BIC_categorical` | | `bnlearn.hc.BIC_categorical` | `pgmpy.hc.BIC_categorical` |

A run can keep to some of these solutions, by name, with `baseline.empty`. Each pair runs one
method with one test or score and the same settings, each set explicitly and listed in `params`.
Where a package's default differs, the adapter overrides it:

| setting | Andrey | bnlearn | pgmpy 1.1.2 |
|---|---|---|---|
| PC test | `chisq`, `gsq` | `x2-adf`, `mi-adf` | |
| PC level, conditioning-set size | `alpha=0.05`, no limit | `alpha=0.05`, no limit (`max.sx`) | |
| BDeu equivalent sample size | `equivalent_sample_size=1` | `iss=1` | `equivalent_sample_size=1` (default 10) |
| BDeu structure prior | each other variable a parent with probability `1 / (d - 1)` | `prior="vsp"`, `beta = 1 / (d - 1)` | the same, added through `structure_prior_ratio` (pgmpy has none) |
| BIC penalty | `penalty_discount=1`: `log(n)` per parameter, in deviance units | `k = log(n) / 2`, in log-likelihood units | `log(n) / 2`, fixed |
| parent cap | none | `maxp=Inf` | `max_indegree=None` |
| move budget | `max_iter=100000` | `max.iter=100000` | `max_iter=100000` |
| restarts, tabu list | none | `restart=0` | `tabu_length=0` (default 100) |

Andrey's scores are `-2` times the log-likelihood scale of bnlearn and pgmpy. Hill climbing stops
at the first pass with no improving move; pgmpy's threshold is its default `epsilon=1e-4`, and
Andrey's is `1e-9` in deviance units, both far below the gain of a real move. pgmpy is pinned to
1.1.2, which installs on Python 3.13 with NumPy 2.5 and pandas 3 (`start.sh` installs it; the
driver stops when another version is installed). bnlearn is the R
library of [The R lane](../README.md#the-r-lane); its version is in `run_meta.json`.

bnlearn names Pearson's chi-square test `x2` and the G-square test `mi`, the mutual information
times `2 n`, which is the G statistic. Each has an `-adf` form that takes the empty rows and
columns of a table out of the degrees of freedom, as Andrey's tests do, so the `-adf` forms are the
ones paired with Andrey's: the plain `x2` and `mi` differ from Andrey's p-values on tables with an
empty row or column. The `-adf` p-values are Andrey's to rounding, with one exception: when a table
has fewer than 5 observations per cell on average (`n < 5 * r * c * q`, for `r` and `c` categories
in the two variables and `q` configurations of the conditioning set), bnlearn returns `p = 1` and
Andrey's tests have no such rule. At 2000 rows and three-category variables that starts at four
conditioning variables, where bnlearn's PC removes the edge and Andrey's PC tests it.

The defaults are 2000 rows, `d` of 10, 20, and 40, five seeds, one repeat after one warm-up fit, and
`--cap-wall-s` of 1800. Each estimate is scored against the true DAG's CPDAG: `shd` and the
skeleton and arrowhead columns. Andrey's solutions and causal-learn's return CPDAGs. bnlearn's and
pgmpy's hill climbing return DAGs, whose comparable columns are `mec_shd` and `mec_arrowhead_f1`,
scored through the DAG's essential graph. The summary prints each solution's median wall time and
error per `d`, with the F1 columns in the same comparable form, then each pair's speed-up: the other
package's median wall time over Andrey's on the datasets where both finished. A smoke run covers
every solution on one small dataset, and a rescore recomputes the score columns from the stored
estimates; pass the first run's flags. The bnlearn rows need the R lane and the pgmpy rows need
pgmpy; a run that names neither needs neither:

```shell
$ANDREY_BENCH_PYTHON benchmarks/discrete_cpu.py --out outputs/discrete-smoke --smoke
$ANDREY_BENCH_PYTHON benchmarks/discrete_cpu.py --out outputs/discrete-cpu --rescore
```

To run only some solutions, name them. `--resume` requires the first run's solution list, so a run
of other solutions takes a fresh `--out`; pointing `ANDREY_BENCH_DATA` at the first run's `data/`
gives it the same datasets, and `andrey_bench.aggregate.pool` joins the two runs' records:

```shell
ANDREY_BENCH_DATA=outputs/discrete-cpu/data $ANDREY_BENCH_PYTHON benchmarks/discrete_cpu.py \
    --out outputs/discrete-comparators --solutions andrey.hc.BDeu andrey.hc.BIC_categorical \
    bnlearn.pc-stable.chisq bnlearn.pc-stable.gsq bnlearn.hc.BDeu bnlearn.hc.BIC_categorical \
    pgmpy.hc.BDeu pgmpy.hc.BIC_categorical
```

The p-values and BDeu scores of Andrey and causal-learn follow the same definitions (Andrey's BDeu
is `-2` times causal-learn's), and on the datasets of `benchmarks/tests/test_adapters_discrete.py`
the two packages' PC and GES return the same graph, with Andrey's PC on the column-order collider
rule that causal-learn's PC runs; PC's default p-value order can orient conflicting colliders
differently. BOSS and GRaSP run on search seed 0 in both packages, but each package draws its orders
from its own random streams, so their graphs can differ at equal settings.
Each comparator's objective is checked against Andrey's in
`benchmarks/tests/test_adapters_discrete_comparators.py`, where R or pgmpy is installed: the scores
agree on a fixed DAG, bnlearn's hill climbing reaches the score Andrey's does on those datasets, and
each declared setting takes effect. pgmpy's search ends in a different local optimum on some
datasets: each package takes the first of equal moves in its own scan order, and the graphs can
differ at equal settings. pgmpy breaks ties by the iteration order of a set of column-name pairs, so
its columns are named by integer position, which keeps the graph the same under any
`PYTHONHASHSEED`. Read the error columns beside the speed-up: where the graphs differ, the
difference is a result to report.
