# CAM-UV

`camuv_cpu.py` compares Andrey's CAM-UV with lingam's `CAMUV` and causal-learn's `CAMUV`, with the
empty graph as the floor. The claim: on data from CAM-UV's model class, nonlinear additive effects
with hidden variables, at the paper's settings, Andrey's graphs make no more directed-edge errors
than either reference's.

## The model

CAM-UV (Maeda and Shimizu, 2021) models each variable, observed or hidden, as a sum of nonlinear
functions of its direct causes plus independent noise, over a DAG (Sect. 2). A pair joined through
a hidden variable, by an unobserved causal path (a directed path entering one of them from a hidden
parent) or an unobserved backdoor path (entering both from hidden parents that share an ancestor),
has a relation regression cannot identify, and the method marks it instead of orienting it. Its
simulation (Sect. 5.1) draws 10 observed variables, each pair joined with probability 0.3, two
hidden common causes into two observed variables each, and two hidden intermediates, each inserted
between a pair of observed variables. Each effect is `(v + a)^c + b`, with `c` 2 or 3, the noise
is uniform, and every variable is divided by its standard deviation. It runs 100 datasets at each of
100 to 1,000 rows, with a significance level of 0.01 and sets of up to 3 variables, and scores the
directed edges (precision, recall, and F1) and the marked pairs.

The driver runs three settings in claim and two out of claim, `d` counting the hidden nodes:

| regime | effects | hidden | `d` (observed + hidden) | rows | claim |
|---|---|---|---|---|---|
| `latent_anm_er` | `w (z + sin z)`, Gaussian noise | one node in five | 5 (4 + 1), 10 (8 + 2), 15 (12 + 3) | 2,000 | in |
| `latent_anm_er_tenth` | `w (z + sin z)`, Gaussian noise | one node in ten | 11 (10 + 1), 22 (20 + 2), 55 (50 + 5) | 1,000 | in |
| `camuv_paper` | the paper's simulation | two common causes, two intermediates | 14 (10 + 4) | 500, 1,000 | in |
| `anm_er` | `w (z + sin z)`, Gaussian noise | none | 20 | 1,000 | out |
| `latent_gauss_er` | linear, Gaussian noise | one node in ten | 22 (20 + 2) | 1,000 | out |

The first two regimes draw additive-noise data on an Erdos-Renyi graph of mean degree 2 with
`andrey.data`, standardized, and hide nodes chosen among those with two or more children, so a
hidden node is a common cause and, when it has a parent, an intermediate too. They lie in the
model class: the effects are nonlinear, the noise independent, and some causes hidden. They differ
from the paper's simulation in the effects, close to linear where `z` is large, the Gaussian noise,
the sparser graph, and the share of hidden nodes. `camuv_paper` is the paper's simulation
(`andrey_bench/regime_models.py`): the paper inserts each intermediate "between each variable pair"
without saying whether the pair keeps its edge, and here the intermediate takes the place of an
observed edge drawn at random. Out of claim, `anm_er` hides nothing, so every marked pair is false,
and `latent_gauss_er` has linear effects, where a path through a hidden variable can be regressed
away, which the method assumes it cannot.

## Settings

Every package runs the paper's settings, which are Andrey's defaults:

| Andrey (`andrey.camuv`) | lingam 1.13.0 `CAMUV` | causal-learn 0.1.4.8 `CAMUV.execute` |
|---|---|---|
| `alpha=0.01` | `alpha=0.01` | `alpha=0.01` |
| `max_depth=2`, sets of up to 3 variables | `num_explanatory_vals=3` | `num_explanatory_vals=3` |
| a pair marked `o-o` | a pair marked `<->` (`NaN` in `adjacency_matrix_`) | a pair marked `<->` (its `U`) |

lingam and causal-learn fit their regressions with pygam, which does not install beside
`.venv-bench`'s SciPy, so their fits run in `.venv-lingam` (`benchmarks/start_lingam.sh`, with
causal-learn 0.1.4.8 beside lingam). The implementations differ in five parts, each named in its
adapter's docstring rather than tuned away:

| part | Andrey | lingam | causal-learn |
|---|---|---|---|
| regression | additive model of thin plate splines, 10 basis functions per term, GCV | pygam `LinearGAM()` | pygam `LinearGAM()` |
| HSIC bandwidth | median distance over every row | median distance over the first 100 rows | median distance over every row |
| HSIC null variance | product form | bias-corrected | bias-corrected |
| order the sets are visited in | a rank order computed from the data | the column order | the column order |
| removing candidates that are not parents | one at a time, then edges on directed cycles dropped | together | together |

## Scores

Every estimate is a PAG over the observed variables. Each fit gets these scores, added by the driver
from the stored estimates:

| column | what it scores |
|---|---|
| `directed_direct_errors` | the directed edges against the edges CAM-UV can identify as direct, those between observed variables that no unobserved causal or backdoor path joins: a missing or an extra edge costs 1, a reversed one 2 |
| `shd_endpoint` | the endpoint SHD against the PAG of the observed variables |
| `shd_endpoint_bidirected` | the same with the marked pairs drawn `<->`, the references' mark, so the packages' graphs are compared without the mark convention |
| `confounded_unobserved_path_*` | the marked pairs against the pairs an unobserved causal or backdoor path joins: precision and recall |

The verdict reads the directed-edge errors. The paper's comparison also "only used directed
edges", the 0.7.0 CAM-UV lane read the same errors, and they do not depend on the mark. Two graphs
are the same when they have the same directed edges and the same marked pairs, whatever the mark. A
pair is one dataset fitted by both solutions in one run. D is Andrey's errors minus the other
solution's, and its interval the percentile bootstrap 95% interval of the mean D over the pairs
(10,000 resamples). A cell in claim is worse when the interval lies above 0, better when it lies
below, and a tie otherwise; a cell worse or better is run again on 20 fresh seeds and keeps its
verdict only if that interval lies on the same side of 0. A fit stopped at the wall cap is a
timeout: it is counted in the coverage and never scored.

## Results

The numbers come from [`published/0.13.0/camuv.json`](../published/0.13.0/camuv.json): an Andrey
pre-release build of the 0.13.0 branch (1 November 2026) against lingam 1.13.0 and causal-learn
0.1.4.8, on one Intel Core i7-7820X, one core and one BLAS thread per fit, four fits at a time.
Every setting in claim has 20 datasets (seeds 0 to 19), every one out of claim 10 (seeds 0 to 9),
each fitted once with a 3,600 s cap; every fit finished under it, so no setting has a timeout.
Errors are means over the datasets, CPU s the median CPU time of a fit, and the CPU ratio the
median over the datasets of a reference's CPU time over Andrey's.

In claim, every setting is a tie with both references once replicated; none is worse. On the
paper's simulation Andrey made fewer errors than both on seeds 0 to 19, at 500 and at 1,000 rows,
but not on seeds 20 to 39 (the replication below), so both count as ties. Andrey ran 2.1 to 3.1
times faster than lingam and 7.4 to 13.5 times faster than causal-learn.

| setting | observed + hidden | rows | same graph, lingam / causal-learn | errors Andrey | errors lingam | errors causal-learn | errors empty | D vs lingam [95% CI] | D vs causal-learn [95% CI] | D vs empty [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| one in five hidden | 4 + 1 | 2,000 | 11 / 12 | 1.95 | 1.85 | 1.80 | 1.70 | +0.10 [-0.30, +0.55] | +0.15 [-0.30, +0.60] | +0.25 [-0.75, +1.35] |
| one in five hidden | 8 + 2 | 2,000 | 4 / 4 | 5.05 | 5.45 | 5.40 | 3.55 | -0.40 [-1.15, +0.25] | -0.35 [-1.15, +0.35] | +1.50 [+0.30, +2.75] |
| one in five hidden | 12 + 3 | 2,000 | 1 / 1 | 7.35 | 7.70 | 7.95 | 6.05 | -0.35 [-1.80, +1.00] | -0.60 [-2.00, +0.70] | +1.30 [-0.60, +3.25] |
| one in ten hidden | 10 + 1 | 1,000 | 1 / 2 | 6.60 | 6.75 | 7.05 | 7.00 | -0.15 [-1.35, +1.05] | -0.45 [-1.80, +0.85] | -0.40 [-2.60, +1.90] |
| one in ten hidden | 20 + 2 | 1,000 | 0 / 0 | 10.40 | 12.05 | 10.85 | 15.30 | -1.65 [-3.55, +0.20] | -0.45 [-2.30, +1.45] | -4.90 [-6.75, -2.95] |
| one in ten hidden | 50 + 5 | 1,000 | 0 / 0 | 27.90 | 28.90 | 30.05 | 39.60 | -1.00 [-5.70, +3.80] | -2.15 [-6.55, +2.40] | -11.70 [-15.35, -7.55] |
| the paper's simulation | 10 + 4 | 500 | 6 / 6 | 6.90 | 7.95 | 8.10 | 9.35 | -1.05 [-2.00, -0.25] | -1.20 [-2.10, -0.40] | -2.45 [-3.80, -0.80] |
| the paper's simulation | 10 + 4 | 1,000 | 3 / 2 | 5.55 | 7.55 | 7.00 | 9.35 | -2.00 [-3.35, -0.80] | -1.45 [-2.65, -0.45] | -3.80 [-4.75, -2.85] |

With one node in five hidden, the hidden nodes are an end of about half of the edges, and no CAM-UV
makes fewer errors than the empty graph on average: at 8 + 2 variables Andrey's interval against it
lies above 0, and the references tie with Andrey. With one node in ten hidden every package makes
fewer errors than the empty graph from 20 observed variables on, and on the paper's simulation at
both sizes. At 50 observed variables the intervals are wide, so the tie there is a wide one.

The endpoint SHD reads the marked pairs too. Andrey's marks are `o-o` and the references' `<->`,
so the intervals compare Andrey's graph with its pairs drawn `<->`; none lies above 0.

| setting | observed + hidden | rows | endpoint SHD Andrey, as returned / as `<->` | lingam | causal-learn | empty | D vs lingam, as `<->` [95% CI] | D vs causal-learn, as `<->` [95% CI] |
|---|---|---|---|---|---|---|---|---|
| one in five hidden | 4 + 1 | 2,000 | 6.30 / 7.00 | 6.85 | 7.05 | 8.30 | +0.15 [-0.15, +0.60] | -0.05 [-0.15, +0.00] |
| one in five hidden | 8 + 2 | 2,000 | 12.85 / 14.25 | 14.25 | 14.10 | 18.80 | +0.00 [-1.20, +1.10] | +0.15 [-1.10, +1.35] |
| one in five hidden | 12 + 3 | 2,000 | 19.40 / 18.75 | 19.40 | 20.05 | 27.80 | -0.65 [-3.10, +1.70] | -1.30 [-3.70, +0.95] |
| one in ten hidden | 10 + 1 | 1,000 | 19.70 / 20.30 | 19.80 | 20.00 | 24.10 | +0.50 [-1.30, +2.10] | +0.30 [-1.55, +2.00] |
| one in ten hidden | 20 + 2 | 1,000 | 38.95 / 38.50 | 40.15 | 37.90 | 44.30 | -1.65 [-8.35, +6.00] | +0.60 [-6.75, +8.85] |
| one in ten hidden | 50 + 5 | 1,000 | 111.15 / 108.00 | 101.30 | 102.80 | 115.90 | +6.70 [-3.90, +17.75] | +5.20 [-4.95, +15.50] |
| the paper's simulation | 10 + 4 | 500 | 23.25 / 22.65 | 23.70 | 24.00 | 28.70 | -1.05 [-2.80, +0.40] | -1.35 [-2.85, -0.05] |
| the paper's simulation | 10 + 4 | 1,000 | 21.05 / 20.95 | 22.65 | 22.45 | 28.70 | -1.70 [-3.95, +0.05] | -1.50 [-3.85, +0.50] |

The marked pairs are scored against the pairs an unobserved causal or backdoor path joins, pooled
over the datasets of a setting. At 50 observed variables Andrey marks more pairs than either
reference, at a lower precision and a higher recall.

| setting | observed + hidden | rows | true pairs per graph | marked, Andrey / lingam / causal-learn | precision | recall | CPU s, Andrey / lingam / causal-learn | CPU ratio, lingam / causal-learn |
|---|---|---|---|---|---|---|---|---|
| one in five hidden | 4 + 1 | 2,000 | 2.40 | 16 / 16 / 18 | 0.81 / 0.62 / 0.67 | 0.27 / 0.21 / 0.25 | 4.0 / 8.3 / 40 | 2.1 / 10.0 |
| one in five hidden | 8 + 2 | 2,000 | 5.80 | 40 / 35 / 40 | 0.70 / 0.83 / 0.75 | 0.24 / 0.25 / 0.26 | 10 / 26 / 125 | 2.5 / 12.0 |
| one in five hidden | 12 + 3 | 2,000 | 7.90 | 52 / 53 / 50 | 0.73 / 0.66 / 0.66 | 0.24 / 0.22 / 0.21 | 19 / 49 / 247 | 2.8 / 13.5 |
| one in ten hidden | 10 + 1 | 1,000 | 5.15 | 50 / 27 / 26 | 0.34 / 0.30 / 0.27 | 0.17 / 0.08 / 0.07 | 5.1 / 15 / 47 | 2.6 / 8.6 |
| one in ten hidden | 20 + 2 | 1,000 | 7.55 | 103 / 101 / 92 | 0.33 / 0.31 / 0.30 | 0.23 / 0.20 / 0.18 | 16 / 52 / 165 | 3.1 / 10.6 |
| one in ten hidden | 50 + 5 | 1,000 | 21.75 | 336 / 237 / 228 | 0.35 / 0.41 / 0.40 | 0.27 / 0.22 / 0.21 | 91 / 296 / 964 | 3.1 / 10.3 |
| the paper's simulation | 10 + 4 | 500 | 6.65 | 42 / 47 / 47 | 0.36 / 0.32 / 0.30 | 0.11 / 0.11 / 0.10 | 1.8 / 4.8 / 12 | 2.9 / 7.4 |
| the paper's simulation | 10 + 4 | 1,000 | 6.65 | 62 / 51 / 56 | 0.63 / 0.45 / 0.50 | 0.29 / 0.17 / 0.21 | 5.2 / 16 / 49 | 2.9 / 8.6 |

## Replication

The two settings better on seeds 0 to 19, the paper's simulation at 500 and 1,000 rows, were run
again on seeds 20 to 39. Neither interval lies below 0 there, so both count as ties; no setting in
claim was worse, so none needed a replication for that.

| setting | observed + hidden | rows | seeds | same graph, lingam / causal-learn | errors Andrey | errors lingam | errors causal-learn | errors empty | D vs lingam [95% CI] | D vs causal-learn [95% CI] | verdict, lingam / causal-learn |
|---|---|---|---|---|---|---|---|---|---|---|---|
| the paper's simulation | 10 + 4 | 500 | 0 to 19 | 6 / 6 | 6.90 | 7.95 | 8.10 | 9.35 | -1.05 [-2.00, -0.25] | -1.20 [-2.10, -0.40] | better / better |
| the paper's simulation | 10 + 4 | 500 | 20 to 39 | 2 / 2 | 8.65 | 8.20 | 8.15 | 11.00 | +0.45 [-0.90, +1.85] | +0.50 [-0.80, +1.85] | tie / tie |
| the paper's simulation | 10 + 4 | 1,000 | 0 to 19 | 3 / 2 | 5.55 | 7.55 | 7.00 | 9.35 | -2.00 [-3.35, -0.80] | -1.45 [-2.65, -0.45] | better / better |
| the paper's simulation | 10 + 4 | 1,000 | 20 to 39 | 3 / 4 | 7.80 | 7.85 | 7.95 | 11.00 | -0.05 [-1.50, +1.35] | -0.15 [-1.55, +1.25] | tie / tie |

## Differences

Two graphs are the same here when they have the same directed edges and the same marked pairs,
whatever the mark; with the marks as returned, `o-o` against `<->`, two graphs are equal only where
neither marks a pair. Each dataset where Andrey's graph differs from a reference's was refitted
along one path of switches from lingam's CAM-UV to Andrey's, each switch changing one part of the
table in [Settings](#settings): lingam run on the columns in Andrey's rank order; then with
Andrey's HSIC test; then with Andrey's regressions; then Andrey's engine with lingam's algorithm,
which removes a variable's non-parents together and drops no edge on a directed cycle; then
Andrey's phase 2, one candidate at a time; then the cycle drop, which is `andrey.camuv`. At 50
observed variables 5 of the 20 datasets were traced, at every other size all of them. Three facts
close the question:

- lingam's own code run with Andrey's test and regressions returns the graph of Andrey's engine
  run with lingam's algorithm on every traced dataset, so each difference is made of the five
  switches and nothing else;
- causal-learn run with lingam's HSIC test returns lingam's graph on every traced dataset where the
  references differ: their difference is the HSIC implementation, whose bandwidth reads every row
  in causal-learn and the first 100 rows in lingam;
- no traced lingam fit made two variables each other's parent, which the adapters' marks would hold
  as one edge, and the refit lingam graph is the lane's on every traced dataset.

The table counts, per setting, the switch after which the graph is Andrey's. On the additive-noise
data with up to 12 observed variables the visiting order alone, lingam run in Andrey's rank order,
most often gives Andrey's graph. On the paper's simulation, over its four runs, and at 20 observed
variables, the regressions most often do. At 50 every traced graph needs phase 2 or the cycle
drop.

| setting | observed + hidden | rows | seeds | graphs differ | traced | first switch that gives Andrey's graph: order / test / regressions / phase 2 / cycle drop | references differ, the HSIC test explains |
|---|---|---|---|---|---|---|---|
| one in five hidden | 4 + 1 | 2,000 | 0 to 19 | 9 | 9 | 7 / 0 / 1 / 1 / 0 | 3 of 3 |
| one in five hidden | 8 + 2 | 2,000 | 0 to 19 | 16 | 16 | 6 / 1 / 4 / 4 / 1 | 3 of 3 |
| one in five hidden | 12 + 3 | 2,000 | 0 to 19 | 19 | 19 | 7 / 2 / 6 / 2 / 2 | 9 of 9 |
| one in ten hidden | 10 + 1 | 1,000 | 0 to 19 | 19 | 19 | 8 / 1 / 7 / 2 / 1 | 2 of 2 |
| one in ten hidden | 20 + 2 | 1,000 | 0 to 19 | 20 | 20 | 3 / 3 / 9 / 2 / 3 | 9 of 9 |
| one in ten hidden | 50 + 5 | 1,000 | 0 to 19 | 20 | 5 | 0 / 0 / 0 / 3 / 2 | 5 of 5 |
| the paper's simulation | 10 + 4 | 500 | 0 to 19 | 14 | 14 | 1 / 1 / 11 / 1 / 0 | 6 of 6 |
| the paper's simulation | 10 + 4 | 1,000 | 0 to 19 | 19 | 19 | 6 / 5 / 4 / 2 / 0 | 8 of 8 |
| the paper's simulation | 10 + 4 | 500 | 20 to 39 | 18 | 18 | 5 / 1 / 8 / 4 / 0 | 8 of 8 |
| the paper's simulation | 10 + 4 | 1,000 | 20 to 39 | 17 | 17 | 4 / 2 / 9 / 2 / 0 | 5 of 5 |
| nothing hidden | 20 | 1,000 | 0 to 9 | 10 | 10 | 1 / 1 / 4 / 4 / 0 | 4 of 4 |
| linear Gaussian, one in ten hidden | 20 + 2 | 1,000 | 0 to 9 | 10 | 10 | 0 / 0 / 3 / 5 / 2 | 6 of 6 |

## Out of claim

Out of claim, at ten datasets per setting: nothing hidden, and linear Gaussian effects with one
node in ten hidden. With nothing hidden every marked pair is false, and each package marks three
or four per graph. With linear effects every CAM-UV makes more errors than the empty graph, by 7 or
more per graph.

| setting | observed + hidden | rows | same graph, lingam / causal-learn | errors Andrey | errors lingam | errors causal-learn | errors empty | D vs lingam [95% CI] | D vs causal-learn [95% CI] | D vs empty [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| nothing hidden | 20 | 1,000 | 0 / 0 | 11.00 | 9.40 | 10.20 | 20.00 | +1.60 [-0.90, +4.20] | +0.80 [-2.10, +3.40] | -9.00 [-12.40, -5.20] |
| linear Gaussian, one in ten hidden | 20 + 2 | 1,000 | 0 / 0 | 21.90 | 24.30 | 25.70 | 14.80 | -2.40 [-5.30, +0.80] | -3.80 [-7.00, -0.50] | +7.10 [+5.50, +9.10] |

| setting | observed + hidden | rows | true pairs per graph | marked, Andrey / lingam / causal-learn | precision | recall | CPU s, Andrey / lingam / causal-learn | CPU ratio, lingam / causal-learn |
|---|---|---|---|---|---|---|---|---|
| nothing hidden | 20 | 1,000 | 0.00 | 35 / 31 / 34 | 0.00 / 0.00 / 0.00 | - / - / - | 17 / 48 / 149 | 2.9 / 10.0 |
| linear Gaussian, one in ten hidden | 20 + 2 | 1,000 | 9.90 | 48 / 31 / 32 | 0.35 / 0.32 / 0.28 | 0.17 / 0.10 / 0.09 | 14 / 44 / 131 | 3.1 / 9.6 |

## Reproduce

Build lingam's environment, fill the shared dataset store once, then cut the campaign into runs
with their own `--out`; pairs form inside one run, so keep a run's whole solution list:

```shell
benchmarks/start_lingam.sh
export ANDREY_BENCH_LINGAM_PYTHON=$PWD/benchmarks/.venv-lingam/bin/python
export ANDREY_BENCH_DATA=$PWD/outputs/camuv-data
$ANDREY_BENCH_PYTHON benchmarks/camuv_cpu.py --out outputs/camuv-fifth \
    --regimes latent_anm_er --seeds 20 --cores 1 --cap-wall-s 3600
$ANDREY_BENCH_PYTHON benchmarks/camuv_cpu.py --out outputs/camuv-tenth \
    --regimes latent_anm_er_tenth --seeds 20 --cores 1 --cap-wall-s 3600
$ANDREY_BENCH_PYTHON benchmarks/camuv_cpu.py --out outputs/camuv-paper \
    --regimes camuv_paper --n 1000 --seeds 20 --cores 1 --cap-wall-s 3600
```

`--n 500` runs the paper's simulation at 500 rows; `--seed-start 20` runs a replication;
`--ladder 55 --seeds 5 --seed-start 0` cuts the 50-variable setting into runs of five seeds; and
`--regimes anm_er` or `--regimes latent_gauss_er` with `--seeds 10` runs the settings out of
claim. `--rescore` recomputes the score columns of a run without fitting.
