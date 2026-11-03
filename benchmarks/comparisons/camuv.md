# CAM-UV

`camuv_cpu.py` compares Andrey's CAM-UV with lingam's `CAMUV` and causal-learn's `CAMUV`, with the
empty graph as the floor. The claim: on data from CAM-UV's model class, nonlinear additive effects
with any number of hidden variables, none included, at the paper's settings, Andrey's graphs make
no more directed-edge errors than either reference's, lingam's at the paper's set size and at its
own default.

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

The driver runs four settings in claim and one out of claim, `d` counting the hidden nodes:

| regime | effects | hidden | `d` (observed + hidden) | rows | claim |
|---|---|---|---|---|---|
| `latent_anm_er` | `w (z + sin z)`, Gaussian noise | one node in five | 5 (4 + 1), 10 (8 + 2), 15 (12 + 3) | 2,000 | in |
| `latent_anm_er_tenth` | `w (z + sin z)`, Gaussian noise | one node in ten | 11 (10 + 1), 22 (20 + 2), 55 (50 + 5) | 1,000 | in |
| `camuv_paper` | the paper's simulation | two common causes, two intermediates | 14 (10 + 4) | 500, 1,000 | in |
| `anm_er` | `w (z + sin z)`, Gaussian noise | none | 20 | 1,000 | in |
| `latent_gauss_er` | linear, Gaussian noise | one node in ten | 22 (20 + 2) | 1,000 | out |

The first two regimes draw additive-noise data on an Erdos-Renyi graph of mean degree 2 with
`andrey.data`, standardized, and hide nodes chosen among those with two or more children, so a
hidden node is a common cause and, when it has a parent, an intermediate too. They lie in the
model class: the effects are nonlinear, the noise independent, and some causes hidden. They differ
from the paper's simulation in the effects, close to linear where `z` is large, the Gaussian noise,
the sparser graph, and the share of hidden nodes. `camuv_paper` is the paper's simulation
(`andrey_bench/regime_models.py`): the paper inserts each intermediate "between each variable pair"
without saying whether the pair keeps its edge, and here the intermediate takes the place of an
observed edge drawn at random. `anm_er` hides nothing, so every marked pair is false; the paper's
model allows no hidden variable, and its test on simulated fMRI data includes runs with none
omitted, so the setting is in claim. Out of claim, `latent_gauss_er` has linear effects, where a
path through a hidden variable can be regressed away, which the method assumes it cannot.

## Settings

Every package runs the paper's settings, which are Andrey's defaults:

| Andrey (`andrey.camuv`) | lingam 1.13.0 `CAMUV` | causal-learn 0.1.4.8 `CAMUV.execute` |
|---|---|---|
| `alpha=0.01` | `alpha=0.01` | `alpha=0.01` |
| `max_depth=2`, sets of up to 3 variables | `num_explanatory_vals=3` (default 2) | `num_explanatory_vals=3` (no default) |
| a pair marked `o-o` | a pair marked `<->` (`NaN` in `adjacency_matrix_`) | a pair marked `<->` (its `U`) |

lingam's documented default is `num_explanatory_vals=2`, sets of up to 2 variables. The lane runs
it at 3, the paper's d (Sect. 5), so the three packages search the same sets, and at its default
too, as `lingam.camuv.default`, on every setting in claim and judged the same way. Those fits ran
on the same stored datasets in runs of their own, two days after the others, so a CPU ratio against
them pairs fits from different runs.

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
| `directed_latent_projection_*` | the directed edges against the paper's truth below: errors, a missing or an extra edge 1 and a reversed one 2, and precision, recall, and F1 |

The verdict reads the directed-edge errors, which do not depend on the mark; the paper's comparison
also "only used directed edges", and the 0.7.0 CAM-UV lane read the same errors. The paper scores
them against another truth (Sect. 5): `x_j -> x_i` wherever a directed path from `x_j` to `x_i` has
no other observed variable on it. That truth also holds the pairs joined through a hidden
intermediate and the direct edges a hidden common cause confounds, pairs CAM-UV marks rather than
orients. The verdict scores only the edges CAM-UV can identify, a truth with fewer edges, so the
empty graph makes fewer errors under it than under the paper's; `directed_latent_projection_*`
scores the paper's truth too. Two graphs are the same when they have the same directed edges and
the same marked pairs, whatever the mark. A pair is one dataset fitted by both solutions. D is
Andrey's errors minus the other solution's, and its interval the percentile bootstrap 95% interval
of the mean D over the pairs (10,000 resamples). A cell in claim is worse when the interval lies
above 0, better when it lies below, and a tie otherwise; a cell worse or better is run again on 20
fresh seeds and keeps its verdict only if that interval lies on the same side of 0. A fit stopped at
the wall cap is a timeout: it is counted in the coverage and never scored.

## Results

The numbers come from [`published/0.14.0/camuv.json`](../published/0.14.0/camuv.json): an Andrey
pre-release build of the 0.13.0 branch (1 November 2026), whose CAM-UV code is the 0.13.0
release's, against lingam 1.13.0 and causal-learn 0.1.4.8, on one Intel Core i7-7820X, one core and
one BLAS thread per fit, four fits at a time. The fits on fresh seeds at 8 + 2 variables and on
seeds 10 to 19 with nothing hidden ran on a pre-release build of the 0.14.0 branch (3 November
2026), whose CAM-UV code and column ranks are the same on every dataset of the lane. Every setting
in claim has 20 datasets (seeds 0 to 19), the one out of claim 10 (seeds 0 to 9), each fitted once
with a 3,600 s cap; every fit finished under it, so no setting has a timeout.
Errors are means over the datasets, CPU s the median CPU time of a fit, and the CPU ratio the
median over the datasets of a reference's CPU time over Andrey's.

In claim, every setting is a tie with each reference once replicated, lingam at its default
included; none is worse. On the paper's simulation Andrey made fewer errors than each on seeds 0 to
19, at 500 and at 1,000 rows, but not on seeds 20 to 39 (the replication below), so both count as
ties. Andrey ran 2.1 to 3.1 times faster than lingam and 7.4 to 13.5 times faster than
causal-learn, each searching sets of up to 3 variables.

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
| nothing hidden | 20 | 1,000 | 0 / 0 | 11.55 | 10.25 | 10.85 | 20.00 | +1.30 [-0.95, +3.60] | +0.70 [-1.70, +3.10] | -8.45 [-11.05, -5.65] |

With one node in five hidden no CAM-UV makes fewer errors than the empty graph on average, at any
of the three sizes; the floor has a section of its own below. Every package makes fewer errors than
the empty graph from 20 observed variables on, with nothing hidden, and on the paper's simulation
at both sizes. The share of true edges with a hidden end does not separate these settings: it is
0.52 to 0.57 with one node in five hidden, 0.32 at 10 + 1, 0.27 from 20 + 2 on, and 0.40 to 0.43 on
the paper's simulation. At 50 observed variables the intervals are wide, so the tie there is a wide
one.

With nothing hidden every marked pair is false, and each package marks 3.2 to 3.6 per graph;
Andrey's mean errors there are the highest, with each interval against a reference including 0.

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
| nothing hidden | 20 | 1,000 | 37.95 / 38.25 | 36.75 | 37.80 | 40.00 | +1.50 [-2.80, +5.60] | +0.45 [-4.00, +4.75] |

The marked pairs are scored against the pairs an unobserved causal or backdoor path joins, pooled
over the datasets of a setting. At 50 observed variables Andrey marks more pairs than either
reference, at a higher recall, whose interval against each lies above 0, and a lower precision,
whose interval includes 0.

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
| nothing hidden | 20 | 1,000 | 0.00 | 71 / 64 / 68 | 0.00 / 0.00 / 0.00 | - / - / - | 17 / 48 / 155 | 3.0 / 9.9 |

## lingam at its default

lingam at its default set size returns the graph of lingam with sets of up to 3 on 17 to 20 of the
20 datasets of each setting, so the verdicts against it follow those against lingam: a tie in
every setting once replicated. Andrey ran 1.9 to 3.2 times faster than lingam at its default.

| setting | observed + hidden | rows | seeds | same graph as lingam with sets of 3 | errors lingam at its default | errors Andrey | D vs lingam at its default [95% CI] | verdict | CPU ratio |
|---|---|---|---|---|---|---|---|---|---|
| one in five hidden | 4 + 1 | 2,000 | 0 to 19 | 19 | 1.75 | 1.95 | +0.20 [-0.25, +0.70] | tie | 1.9 |
| one in five hidden | 8 + 2 | 2,000 | 0 to 19 | 19 | 5.40 | 5.05 | -0.35 [-1.15, +0.35] | tie | 2.6 |
| one in five hidden | 8 + 2 | 2,000 | 20 to 40 | 19 | 5.15 | 5.15 | +0.00 [-0.90, +0.90] | tie | 2.4 |
| one in five hidden | 12 + 3 | 2,000 | 0 to 19 | 19 | 7.85 | 7.35 | -0.50 [-2.00, +0.90] | tie | 2.9 |
| one in ten hidden | 10 + 1 | 1,000 | 0 to 19 | 18 | 6.70 | 6.60 | -0.10 [-1.40, +1.15] | tie | 2.7 |
| one in ten hidden | 20 + 2 | 1,000 | 0 to 19 | 19 | 12.00 | 10.40 | -1.60 [-3.50, +0.20] | tie | 3.2 |
| one in ten hidden | 50 + 5 | 1,000 | 0 to 19 | 17 | 28.75 | 27.90 | -0.85 [-5.80, +4.25] | tie | 2.5 |
| the paper's simulation | 10 + 4 | 500 | 0 to 19 | 20 | 7.95 | 6.90 | -1.05 [-2.00, -0.25] | better | 2.7 |
| the paper's simulation | 10 + 4 | 500 | 20 to 39 | 19 | 8.20 | 8.65 | +0.45 [-0.90, +1.85] | tie | 3.5 |
| the paper's simulation | 10 + 4 | 1,000 | 0 to 19 | 17 | 7.65 | 5.55 | -2.10 [-3.35, -1.05] | better | 2.7 |
| the paper's simulation | 10 + 4 | 1,000 | 20 to 39 | 18 | 7.70 | 7.80 | +0.10 [-1.20, +1.40] | tie | 2.8 |
| nothing hidden | 20 | 1,000 | 0 to 19 | 20 | 10.25 | 11.55 | +1.30 [-0.95, +3.60] | tie | 2.9 |

## The empty-graph floor

With one node in five hidden at 8 + 2 variables, every CAM-UV made more errors than the empty
graph on seeds 0 to 19, each interval above 0. On fresh seeds, 20 to 40 with seed 31 skipped as one
that cannot hide its nodes, every mean stays above the empty graph's but every interval includes 0,
so by the replication rule the floor failure does not repeat. Scored against the paper's truth,
which also holds the edges CAM-UV marks, every interval at 8 + 2 includes 0 on both sets of seeds.
D here is each package's errors minus the empty graph's.

| setting | observed + hidden | rows | seeds | truth | errors empty | D Andrey [95% CI] | D lingam [95% CI] | D causal-learn [95% CI] | D lingam at its default [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| one in five hidden | 4 + 1 | 2,000 | 0 to 19 | CAM-UV's | 1.70 | +0.25 [-0.75, +1.35] | +0.15 [-0.80, +1.25] | +0.10 [-0.80, +1.10] | +0.05 [-0.90, +1.15] |
| one in five hidden | 4 + 1 | 2,000 | 0 to 19 | the paper's | 3.00 | -0.85 [-1.70, +0.15] | -0.95 [-1.65, -0.15] | -1.00 [-1.65, -0.30] | -0.85 [-1.55, -0.05] |
| one in five hidden | 8 + 2 | 2,000 | 0 to 19 | CAM-UV's | 3.55 | +1.50 [+0.30, +2.75] | +1.90 [+0.70, +3.00] | +1.85 [+0.75, +2.90] | +1.85 [+0.65, +3.00] |
| one in five hidden | 8 + 2 | 2,000 | 0 to 19 | the paper's | 5.45 | +0.20 [-1.20, +1.50] | +0.40 [-0.90, +1.65] | +0.45 [-0.70, +1.55] | +0.45 [-0.80, +1.65] |
| one in five hidden | 8 + 2 | 2,000 | 20 to 40 | CAM-UV's | 3.95 | +1.20 [-0.20, +2.60] | +1.25 [-0.25, +2.80] | +1.30 [-0.20, +2.80] | +1.20 [-0.30, +2.75] |
| one in five hidden | 8 + 2 | 2,000 | 20 to 40 | the paper's | 5.90 | -0.30 [-1.90, +1.30] | +0.05 [-1.45, +1.60] | +0.00 [-1.50, +1.55] | +0.10 [-1.40, +1.65] |
| one in five hidden | 12 + 3 | 2,000 | 0 to 19 | CAM-UV's | 6.05 | +1.30 [-0.60, +3.25] | +1.65 [-0.50, +3.90] | +1.90 [-0.20, +4.10] | +1.80 [-0.30, +4.00] |
| one in five hidden | 12 + 3 | 2,000 | 0 to 19 | the paper's | 8.10 | -0.50 [-2.00, +1.00] | -0.15 [-2.05, +1.90] | +0.00 [-1.80, +1.90] | +0.00 [-1.90, +2.00] |
| one in ten hidden | 10 + 1 | 1,000 | 0 to 19 | CAM-UV's | 7.00 | -0.40 [-2.60, +1.90] | -0.25 [-2.05, +1.75] | +0.05 [-1.65, +1.95] | -0.30 [-2.05, +1.65] |
| one in ten hidden | 10 + 1 | 1,000 | 0 to 19 | the paper's | 8.35 | -1.10 [-3.05, +0.95] | -1.25 [-2.85, +0.55] | -0.95 [-2.50, +0.80] | -1.20 [-2.80, +0.60] |
| one in ten hidden | 20 + 2 | 1,000 | 0 to 19 | CAM-UV's | 15.30 | -4.90 [-6.75, -2.95] | -3.25 [-5.75, -0.45] | -4.45 [-6.50, -2.25] | -3.30 [-5.80, -0.55] |
| one in ten hidden | 20 + 2 | 1,000 | 0 to 19 | the paper's | 18.00 | -5.90 [-8.05, -3.70] | -4.35 [-6.60, -2.00] | -5.55 [-7.55, -3.30] | -4.40 [-6.60, -2.05] |
| one in ten hidden | 50 + 5 | 1,000 | 0 to 19 | CAM-UV's | 39.60 | -11.70 [-15.35, -7.55] | -10.70 [-14.95, -6.55] | -9.55 [-13.90, -5.10] | -10.85 [-15.00, -6.75] |
| one in ten hidden | 50 + 5 | 1,000 | 0 to 19 | the paper's | 45.85 | -15.30 [-18.70, -11.50] | -14.70 [-18.65, -10.75] | -13.65 [-17.80, -9.60] | -14.75 [-18.70, -10.90] |
| the paper's simulation | 10 + 4 | 500 | 0 to 19 | CAM-UV's | 9.35 | -2.45 [-3.80, -0.80] | -1.40 [-2.90, +0.35] | -1.25 [-2.75, +0.40] | -1.40 [-2.90, +0.35] |
| the paper's simulation | 10 + 4 | 500 | 0 to 19 | the paper's | 12.65 | -5.25 [-6.45, -3.75] | -3.90 [-5.25, -2.35] | -3.75 [-5.05, -2.30] | -3.90 [-5.25, -2.35] |
| the paper's simulation | 10 + 4 | 500 | 20 to 39 | CAM-UV's | 11.00 | -2.35 [-3.50, -1.25] | -2.80 [-4.35, -1.15] | -2.85 [-4.35, -1.25] | -2.80 [-4.35, -1.20] |
| the paper's simulation | 10 + 4 | 500 | 20 to 39 | the paper's | 14.25 | -4.55 [-5.75, -3.35] | -4.90 [-6.40, -3.30] | -4.85 [-6.40, -3.20] | -4.90 [-6.40, -3.35] |
| the paper's simulation | 10 + 4 | 1,000 | 0 to 19 | CAM-UV's | 9.35 | -3.80 [-4.75, -2.85] | -1.80 [-3.05, -0.55] | -2.35 [-3.65, -1.05] | -1.70 [-2.85, -0.50] |
| the paper's simulation | 10 + 4 | 1,000 | 0 to 19 | the paper's | 12.65 | -5.60 [-6.50, -4.65] | -4.70 [-5.65, -3.75] | -4.95 [-6.10, -3.85] | -4.40 [-5.45, -3.35] |
| the paper's simulation | 10 + 4 | 1,000 | 20 to 39 | CAM-UV's | 11.00 | -3.20 [-4.50, -1.85] | -3.15 [-4.65, -1.60] | -3.05 [-4.50, -1.55] | -3.30 [-4.70, -1.90] |
| the paper's simulation | 10 + 4 | 1,000 | 20 to 39 | the paper's | 14.25 | -5.00 [-6.25, -3.60] | -5.25 [-6.60, -3.70] | -5.15 [-6.50, -3.70] | -5.30 [-6.65, -3.80] |
| nothing hidden | 20 | 1,000 | 0 to 19 | CAM-UV's | 20.00 | -8.45 [-11.05, -5.65] | -9.75 [-12.25, -7.20] | -9.15 [-11.75, -6.45] | -9.75 [-12.25, -7.10] |
| nothing hidden | 20 | 1,000 | 0 to 19 | the paper's | 20.00 | -8.45 [-11.05, -5.65] | -9.75 [-12.25, -7.20] | -9.15 [-11.75, -6.45] | -9.75 [-12.25, -7.10] |

## Replication

The two settings better on seeds 0 to 19, the paper's simulation at 500 and 1,000 rows, were run
again on seeds 20 to 39. Neither interval lies below 0 there, so both count as ties; no setting in
claim was worse, so none needed a replication for that. The setting with one node in five hidden at
8 + 2 variables was run again for the floor.

| setting | observed + hidden | rows | seeds | same graph, lingam / causal-learn | errors Andrey | errors lingam | errors causal-learn | errors empty | D vs lingam [95% CI] | D vs causal-learn [95% CI] | verdict, lingam / causal-learn |
|---|---|---|---|---|---|---|---|---|---|---|---|
| one in five hidden | 8 + 2 | 2,000 | 0 to 19 | 4 / 4 | 5.05 | 5.45 | 5.40 | 3.55 | -0.40 [-1.15, +0.25] | -0.35 [-1.15, +0.35] | tie / tie |
| one in five hidden | 8 + 2 | 2,000 | 20 to 40 | 4 / 5 | 5.15 | 5.20 | 5.25 | 3.95 | -0.05 [-0.95, +0.85] | -0.10 [-1.00, +0.80] | tie / tie |
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
observed variables 5 of the 20 datasets were traced, and the fresh seeds at 8 + 2 variables and
seeds 10 to 19 with nothing hidden none; at every other size, every dataset where a graph differed.
Three facts close the question:

- lingam's own code run with Andrey's test and regressions returns the graph of Andrey's engine
  run with lingam's algorithm on every traced dataset, so each difference is made of the five
  switches and nothing else;
- causal-learn run with lingam's HSIC test returns lingam's graph on every traced dataset where the
  references differ: their difference is the HSIC implementation, whose bandwidth reads every row
  in causal-learn and the first 100 rows in lingam;
- no traced lingam fit made two variables each other's parent, which the adapters' marks would hold
  as one edge, and a lingam fit in an environment of the same package versions returns the lane's
  graph on every traced dataset.

The traces start from lingam with sets of up to 3; lingam at its default returns that graph on most
datasets, as the section above counts.

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
| one in five hidden | 8 + 2 | 2,000 | 20 to 40 | 16 | 0 | 0 / 0 / 0 / 0 / 0 | 0 of 0 |
| one in ten hidden | 10 + 1 | 1,000 | 0 to 19 | 19 | 19 | 8 / 1 / 7 / 2 / 1 | 2 of 2 |
| one in ten hidden | 20 + 2 | 1,000 | 0 to 19 | 20 | 20 | 3 / 3 / 9 / 2 / 3 | 9 of 9 |
| one in ten hidden | 50 + 5 | 1,000 | 0 to 19 | 20 | 5 | 0 / 0 / 0 / 3 / 2 | 5 of 5 |
| the paper's simulation | 10 + 4 | 500 | 0 to 19 | 14 | 14 | 1 / 1 / 11 / 1 / 0 | 6 of 6 |
| the paper's simulation | 10 + 4 | 1,000 | 0 to 19 | 19 | 19 | 6 / 5 / 4 / 2 / 0 | 8 of 8 |
| the paper's simulation | 10 + 4 | 500 | 20 to 39 | 18 | 18 | 5 / 1 / 8 / 4 / 0 | 8 of 8 |
| the paper's simulation | 10 + 4 | 1,000 | 20 to 39 | 17 | 17 | 4 / 2 / 9 / 2 / 0 | 5 of 5 |
| nothing hidden | 20 | 1,000 | 0 to 19 | 20 | 10 | 1 / 1 / 4 / 4 / 0 | 4 of 4 |
| linear Gaussian, one in ten hidden | 20 + 2 | 1,000 | 0 to 9 | 10 | 10 | 0 / 0 / 3 / 5 / 2 | 6 of 6 |

On the paper's simulation at 1,000 rows, seeds 0 to 19, 2 of the 19 traced datasets have Andrey's
graph equal to lingam's and different from causal-learn's. The switches start from lingam's graph,
so these 2 have none, and the first-switch counts sum to 17; the HSIC test explains both, among the
8 where the references differ.

## Out of claim

Out of claim, at ten datasets: linear Gaussian effects with one node in ten hidden. Every CAM-UV
makes more errors than the empty graph, by 7 or more per graph.

| setting | observed + hidden | rows | same graph, lingam / causal-learn | errors Andrey | errors lingam | errors causal-learn | errors empty | D vs lingam [95% CI] | D vs causal-learn [95% CI] | D vs empty [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| linear Gaussian, one in ten hidden | 20 + 2 | 1,000 | 0 / 0 | 21.90 | 24.30 | 25.70 | 14.80 | -2.40 [-5.30, +0.80] | -3.80 [-7.00, -0.50] | +7.10 [+5.50, +9.10] |

| setting | observed + hidden | rows | true pairs per graph | marked, Andrey / lingam / causal-learn | precision | recall | CPU s, Andrey / lingam / causal-learn | CPU ratio, lingam / causal-learn |
|---|---|---|---|---|---|---|---|---|
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
`--ladder 55 --seeds 5 --seed-start 0` cuts the 50-variable setting into runs of five seeds;
`--regimes anm_er --seeds 20` runs the setting with nothing hidden; `--solutions
lingam.camuv.default` runs lingam at its default on the stored datasets; and `--regimes
latent_gauss_er --seeds 10` runs the setting out of claim. `--rescore` recomputes the score columns
of a run without fitting.
