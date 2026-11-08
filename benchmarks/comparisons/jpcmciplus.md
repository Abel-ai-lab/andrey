# J-PCMCI+

`jpcmciplus_cpu.py` compares Andrey's `jpcmciplus` with `JPCMCIplus` from tigramite 5.2.10.1, the
package of the method's authors (Gunther, Ninad, and Runge 2023), on the model their experiment
script draws: time series from several datasets of the same system variables that differ through
time and space contexts, some of them hidden. Andrey's `pcmciplus` on the datasets pooled without
an index, the paper's baseline, runs beside them as context: reported, never judged.

## Data

`andrey_bench.context_models.jpcmciplus2023` draws the datasets as the authors' script
(`generate_data.py`) does, in the order of its random draws, so a seed gives the model the script
draws for that seed. A draw is one seed's datasets, all from one model:

| Part | Draw |
|---|---|
| System variables | 5, each with a link to its own past at lag 1 (weight 0.5 or 0.7), and 5 cross-links along one random causal order (weight -0.5 or 0.5) |
| Time contexts | 2, each with a link to its own past at lag 1 and one link between them; one series, the same in every dataset |
| Space contexts | 2, with one same-step link between them; one draw per dataset |
| Context links | twice as many as the context slots that are not a parent of another context, weight -0.5 or 0.5, at lag 0 to 2 (a space context's at lag 0); a variable can have several context parents |
| Noise | Gaussian, standard deviation 0.5 or 2 per variable; a burn-in of 20% of the steps, dropped; no rescaling |

The script asks for every cross-link at lag 0, but its lag draw compares a pair with a list of
triples, a test that never holds, so each cross-link gets a lag of 1 or 2. Run on 200 seeds, the
script drew 1,000 of 1,000 cross-links lagged, and the lane's model matched the script's for 600 of
600 seeds at 5 and 10 system variables. The paper's figures come from these data, and we follow the
script, so the claimed cells have no same-step links between system variables and the same-step
part of an error counts false links only. We measure same-step recovery outside the claim, on
`andrey.data.sample_context_series`, which draws half its links at lag 0.

The paper's text describes other data than the script draws: half the links at lag 0, the data
rescaled, 3 contexts, no context with a context parent, and at most one context parent per system
variable. The script's links between contexts, lagged between the two time contexts and same-step
between the two space contexts, are what can break Assumption 3 below, and with both space contexts
observed, the same-step one is what makes tigramite raise without its fix (Known differences).

With every context observed, the datasets hold both contexts of each kind; with one of each kind,
one time and one space context, chosen as the script chooses them. The paper's Assumption 3 rules
out a hidden context that confounds an observed context and a system variable. In the script's
model that happens when a hidden context is a parent of an observed one and reaches a system
variable through hidden contexts only. The model and the contexts observed follow the seed alone,
not the numbers of datasets and steps, so of the 40 models with hidden contexts, seeds 0 to 39, the
same 19 break Assumption 3 in every cell. Their draws form rows of their own, reported and not
judged, so the cells with hidden contexts run 40 draws to keep about 20 in the judged row.

## Cells

| Cell | Model | System variables | Datasets x steps | Contexts observed | Claim |
|---|---|---|---|---|---|
| all-M10-T10 to T200 | the script's | 5 | 10 x 10, 20, 30, 50, 100, 150, 200 | all | in |
| half-M10-T10 to T200 | the script's | 5 | 10 x 10, 20, 30, 50, 100, 150, 200 | one of each kind | in |
| half-M5-T10 to M200 | the script's | 5 | 5, 20, 30, 50, 100, 150, 200 x 10 | one of each kind | in |
| current-* | `sample_context_series`: 1 time and 2 space contexts, half the links at lag 0 | 5 | 5 x 100; 10 x 50, 100, 200; 20 x 50, 100; 50 x 10 | all | out |
| all-N10-M10-T100, all-N15-M10-T100 | the script's | 10, 15 | 10 x 100 | all | out |
| gpdc-M5-T20 to M60 | the script's nonlinear model: 1 time and 1 space context, a squared link, standard normal noise | 2 | 5, 10, 20, 30, 50, 60 x 20 | all | out |

In claim are the settings of the paper's Figures 2 and 3: every context observed with 10 datasets of
10 to 200 steps, and one context of each kind observed with 10 datasets of 10 to 200 steps or with
5 to 200 datasets of 10 steps, judged at alpha 0.05, the paper's level. Out of claim are the model
of `sample_context_series` at the sizes of an earlier comparison, the script's model with 10 and 15
system variables, and its nonlinear model, the data of the supplement's Figure 6, with GPDC. Each
cell runs the draws of seeds 0 to 19, and the cells with hidden contexts those of seeds 0 to 39.

## Settings

Both packages run with the dummy of the dataset and that of the time step, `tau_max=2`, at alpha
0.05, the paper's level, and 0.01, both packages' default, and every other argument at its default,
the majority collider rule among them:

| Andrey | tigramite |
|---|---|
| `jpcmciplus(datasets, time_contexts=..., space_contexts=..., time_dummy=True, tau_max=2, alpha=a)`, the Fisher z test, its default on continuous data | `JPCMCIplus(dataframe=DataFrame(blocks, analysis_mode="multiple", vector_vars=...), cond_ind_test=ParCorrMult(), node_classification=...).run_jpcmciplus(tau_max=2, pc_alpha=a)` |
| `indep_test="gpdc"` in the GPDC cells | `cond_ind_test=GPDC()` |

tigramite's blocks hold each dataset's columns, then the one-hot dummy of the time step and that of
the dataset, each one vector variable, as the authors' experiment code builds them. Andrey takes the
datasets as a list and builds its dummies itself. Its `fdr_method="fdr_bh"` adjusts the reported
p-values and changes no link.

The context arms:

| Arm | Cells | What it shows |
|---|---|---|
| Andrey `pcmciplus` on the system variables of the datasets one after another, without an index | all | the paper's baseline, pooling without the contexts |
| tigramite without the fix, at alpha 0.05 | where every context is observed, but GPDC | whether the fix changes a fit that finishes without it |

## Scores

The parent scores every fit with one function.

| Metric | Meaning |
|---|---|
| `shd` | the endpoint SHD over the links between system variables at lags 0 to 2: a missing or extra link costs its two marks, a wrong mark one (judged) |
| `context_error` | where every context is observed, the links from the contexts into the system variables that a fit misses plus those it adds (judged in the claim); links between contexts are not scored |
| `shd_lag0`, `shd_lagged` | the same-step and the lagged part of `shd` |
| `adjacency_*` | precision, recall, and F1 of the links between system variables, whatever their marks |
| `wall_s` | the timed fit, from the datasets in memory to the returned graph; for tigramite its dummies, the DataFrame, the class, and the run |
| empty | the empty graph's scores |

A row is a cell, a level, an error, and, in a cell with hidden contexts, the draws where Assumption
3 holds or those where it breaks. Per draw, D is Andrey's error minus tigramite's; the row's
interval is the percentile bootstrap 95% interval of the mean D over the paired draws (10,000
resamples). A row in the claim, at alpha 0.05 where Assumption 3 holds, is judged only where
tigramite beats the empty graph (its own interval against the empty graph lies below 0) with at
least 10 paired draws. It is worse when the interval lies above 0 and better when below, and
either counts only if a rerun on seeds 100 to 119, or 100 to 139 with hidden contexts, agrees. A
row also fails when an Andrey fit did not finish where tigramite's did, or when Andrey does not beat
the empty graph where tigramite does. A timeout is never scored as an empty graph.

## Results

Andrey's pre-release build of 8 November 2026 against tigramite 5.2.10.1 with the fix, both on
Python 3.13.11 and NumPy 2.5.4, one fit at a time on one core with one BLAS thread: on an AMD EPYC
7713 processor, and for some GPDC fits at 60 datasets on an Intel Xeon Gold 6542Y. Every number
below is in [`published/0.15.0/jpcmciplus.json`](../published/0.15.0/jpcmciplus.json), with the
context arms and the coverage of every row. Every fit of Andrey and of tigramite with the fix
finished, with no timeout or error. Refitted on the 0.15.0 release branch, Andrey returned the same
graph on all 1,760 fits of the linear cells.

### In the claim

Means over the paired draws of the endpoint SHD over the links between system variables at lags 0
to 2. With every context observed, the cells pair 20 draws; with one of each kind, the 21 where
Assumption 3 holds. D is Andrey's SHD minus tigramite's, with its 95% interval, and Same graph
counts the draws where the two returned the same marks over every column, the context links
included.

| Observed | Datasets x steps | Andrey | tigramite | Empty | D [95%] | Same graph |
|---|---|---|---|---|---|---|
| all | 10 x 10 | 7.7 | 7.6 | 20 | 0.1 [0, 0.3] | 15 of 20 |
| all | 10 x 20 | 4.9 | 5 | 20 | -0.1 [-0.3, 0] | 17 of 20 |
| all | 10 x 30 | 4 | 3.9 | 20 | 0.1 [0, 0.3] | 17 of 20 |
| all | 10 x 50 | 3.7 | 3.6 | 20 | 0.1 [-0.2, 0.5] | 15 of 20 |
| all | 10 x 100 | 3.1 | 3.1 | 20 | 0 [0, 0] | 18 of 20 |
| all | 10 x 150 | 3.6 | 3.6 | 20 | 0 [0, 0] | 19 of 20 |
| all | 10 x 200 | 3.1 | 3.1 | 20 | 0 [0, 0] | 18 of 20 |
| one of each | 10 x 10 | 7.24 | 7.33 | 20 | -0.1 [-0.29, 0] | 20 of 21 |
| one of each | 10 x 20 | 4.67 | 5.05 | 20 | -0.38 [-0.86, 0] | 17 of 21 |
| one of each | 10 x 30 | 5.52 | 5.62 | 20 | -0.1 [-0.29, 0] | 18 of 21 |
| one of each | 10 x 50 | 4.67 | 4.86 | 20 | -0.19 [-0.95, 0.38] | 16 of 21 |
| one of each | 10 x 100 | 3.71 | 4.1 | 20 | -0.38 [-0.86, 0] | 17 of 21 |
| one of each | 10 x 150 | 3.62 | 3.81 | 20 | -0.19 [-0.67, 0.29] | 11 of 21 |
| one of each | 10 x 200 | 3.71 | 3.43 | 20 | 0.29 [0, 0.57] | 17 of 21 |
| one of each | 5 x 10 | 12.95 | 12.76 | 20 | 0.19 [-0.48, 0.86] | 12 of 21 |
| one of each | 20 x 10 | 5.9 | 5.81 | 20 | 0.1 [-0.19, 0.38] | 18 of 21 |
| one of each | 30 x 10 | 6.19 | 5.71 | 20 | 0.48 [-0.1, 1.24] | 13 of 21 |
| one of each | 50 x 10 | 5.33 | 5.33 | 20 | 0 [-0.29, 0.29] | 19 of 21 |
| one of each | 100 x 10 | 5.05 | 5.05 | 20 | 0 [0, 0] | 21 of 21 |
| one of each | 150 x 10 | 5.33 | 5.14 | 20 | 0.19 [0, 0.48] | 14 of 21 |
| one of each | 200 x 10 | 4.76 | 4.76 | 20 | 0 [-0.57, 0.48] | 12 of 21 |

The context links missed plus added, where every context is observed, over the same draws:

| Datasets x steps | Andrey | tigramite | Empty | D [95%] |
|---|---|---|---|---|
| 10 x 10 | 7.75 | 7.75 | 8.9 | 0 [0, 0] |
| 10 x 20 | 4.3 | 4.3 | 8.9 | 0 [0, 0] |
| 10 x 30 | 3.85 | 3.85 | 8.9 | 0 [0, 0] |
| 10 x 50 | 2.85 | 2.85 | 8.9 | 0 [0, 0] |
| 10 x 100 | 1.65 | 1.65 | 8.9 | 0 [0, 0] |
| 10 x 150 | 1.45 | 1.45 | 8.9 | 0 [0, 0] |
| 10 x 200 | 1.05 | 1.05 | 8.9 | 0 [0, 0] |

All 28 rows tie, so no row needed a rerun. With the true graph holding no same-step link between
system variables, the same-step part of the SHD, its false same-step links, averaged 0.5 to 1.2 for
each package with every context observed.

### Floors

tigramite's errors were below the empty graph's in all 28 rows in the claim, so the floor applies in
each, and Andrey's were below it too: the upper end of its interval of the error minus the empty
graph's was -0.35 or lower. Out of the claim the floor applied in all rows but one, at alpha 0.01
with GPDC at 5 datasets, where tigramite's interval reaches above 0, and Andrey passed each.

### Out of the claim

The claimed cells at alpha 0.01, both packages' default, where Assumption 3 holds:

| Observed | Datasets x steps | Andrey | tigramite | D [95%] | Context links D [95%] |
|---|---|---|---|---|---|
| all | 10 x 10 | 8 | 8.4 | -0.4 [-1.3, 0.2] | 0 [0, 0] |
| all | 10 x 20 | 5.1 | 5 | 0.1 [0, 0.3] | -0.05 [-0.15, 0] |
| all | 10 x 30 | 4 | 3.9 | 0.1 [0, 0.3] | 0 [0, 0] |
| all | 10 x 50 | 3.2 | 3.2 | 0 [0, 0] | 0 [0, 0] |
| all | 10 x 100 | 3.1 | 3.1 | 0 [0, 0] | 0 [0, 0] |
| all | 10 x 150 | 2.9 | 2.9 | 0 [0, 0] | 0 [0, 0] |
| all | 10 x 200 | 1.9 | 2 | -0.1 [-0.3, 0] | 0 [0, 0] |
| one of each | 10 x 10 | 8.19 | 8.57 | -0.38 [-0.76, -0.1] |  |
| one of each | 10 x 20 | 5.05 | 5.24 | -0.19 [-0.48, 0] |  |
| one of each | 10 x 30 | 5.14 | 5.05 | 0.1 [-0.19, 0.38] |  |
| one of each | 10 x 50 | 4.1 | 4.1 | 0 [0, 0] |  |
| one of each | 10 x 100 | 4.29 | 4.38 | -0.1 [-0.29, 0] |  |
| one of each | 10 x 150 | 2.76 | 3.24 | -0.48 [-0.95, -0.1] |  |
| one of each | 10 x 200 | 2.48 | 2.57 | -0.1 [-0.29, 0] |  |
| one of each | 5 x 10 | 13.43 | 13.62 | -0.19 [-1.05, 0.67] |  |
| one of each | 20 x 10 | 4.76 | 4.86 | -0.1 [-0.29, 0] |  |
| one of each | 30 x 10 | 4.48 | 4.57 | -0.1 [-0.29, 0] |  |
| one of each | 50 x 10 | 4.29 | 4.29 | 0 [0, 0] |  |
| one of each | 100 x 10 | 3.52 | 3.52 | 0 [0, 0] |  |
| one of each | 150 x 10 | 3.14 | 3.24 | -0.1 [-0.29, 0] |  |
| one of each | 200 x 10 | 3.05 | 3.05 | 0 [0, 0] |  |

Two rows lie below 0, with one of each kind observed at 10 datasets of 10 and of 150 steps: Andrey
made fewer errors there. The draws that break Assumption 3, 19 in each cell:

| Datasets x steps | Paired | alpha 0.05: Andrey | tigramite | D [95%] | alpha 0.01: D [95%] |
|---|---|---|---|---|---|
| 10 x 10 | 19 | 7.79 | 7.79 | 0 [-0.63, 0.53] | -0.32 [-1.26, 0.32] |
| 10 x 20 | 19 | 6.74 | 6.53 | 0.21 [-0.21, 0.63] | -0.32 [-0.84, 0] |
| 10 x 30 | 19 | 4.95 | 4.74 | 0.21 [-0.21, 0.63] | -0.21 [-0.63, 0.21] |
| 10 x 50 | 19 | 4.95 | 4.95 | 0 [-0.32, 0.32] | -0.42 [-0.95, 0] |
| 10 x 100 | 19 | 3.79 | 4 | -0.21 [-0.63, 0.21] | -0.32 [-0.63, 0] |
| 10 x 150 | 19 | 4 | 5.05 | -1.05 [-2.74, 0.11] | -0.21 [-0.53, 0] |
| 10 x 200 | 19 | 3.79 | 4.74 | -0.95 [-2.32, -0.11] | -0.63 [-1.16, -0.21] |
| 5 x 10 | 19 | 10.74 | 10.95 | -0.21 [-0.84, 0.42] | 0 [-0.32, 0.32] |
| 20 x 10 | 19 | 6.32 | 6.32 | 0 [-0.32, 0.32] | -0.21 [-0.74, 0.21] |
| 30 x 10 | 19 | 6.74 | 6.74 | 0 [-0.32, 0.32] | -0.11 [-0.32, 0] |
| 50 x 10 | 19 | 4.84 | 4.95 | -0.11 [-0.32, 0] | -0.21 [-0.74, 0.21] |
| 100 x 10 | 19 | 7.58 | 7.58 | 0 [0, 0] | 0.11 [0, 0.32] |
| 150 x 10 | 19 | 5.26 | 5.26 | 0 [-0.63, 0.63] | 0 [0, 0] |
| 200 x 10 | 19 | 6.32 | 6.32 | 0 [-0.42, 0.42] | -0.42 [-1.26, 0] |

At 10 datasets of 200 steps, Andrey made fewer errors at both levels. The model of
`sample_context_series`, with half the links at lag 0, and the script's model with 10 and 15 system
variables:

| Cell | alpha | Andrey | tigramite | Empty | D [95%] | Context links D [95%] |
|---|---|---|---|---|---|---|
| current-M5-T100 | 0.05 | 1.5 | 1.6 | 20 | -0.1 [-0.3, 0] | 0 [0, 0] |
| current-M5-T100 | 0.01 | 1.1 | 1.1 | 20 | 0 [0, 0] | 0 [0, 0] |
| current-M10-T50 | 0.05 | 0.95 | 0.95 | 20 | 0 [0, 0] | 0 [0, 0] |
| current-M10-T50 | 0.01 | 0.55 | 0.55 | 20 | 0 [0, 0] | 0 [0, 0] |
| current-M10-T100 | 0.05 | 1.05 | 1.05 | 20 | 0 [0, 0] | 0 [0, 0] |
| current-M10-T100 | 0.01 | 0.85 | 0.85 | 20 | 0 [0, 0] | 0 [0, 0] |
| current-M10-T200 | 0.05 | 0.95 | 0.95 | 20 | 0 [0, 0] | 0 [0, 0] |
| current-M10-T200 | 0.01 | 0.55 | 0.55 | 20 | 0 [0, 0] | 0 [0, 0] |
| current-M20-T50 | 0.05 | 1.4 | 1.5 | 20 | -0.1 [-0.3, 0] | 0 [0, 0] |
| current-M20-T50 | 0.01 | 0.85 | 0.85 | 20 | 0 [0, 0] | 0 [0, 0] |
| current-M20-T100 | 0.05 | 0.9 | 0.9 | 20 | 0 [0, 0] | 0 [0, 0] |
| current-M20-T100 | 0.01 | 0.4 | 0.4 | 20 | 0 [0, 0] | 0 [0, 0] |
| current-M50-T10 | 0.05 | 1.8 | 1.65 | 20 | 0.15 [0, 0.4] | 0 [0, 0] |
| current-M50-T10 | 0.01 | 1.6 | 1.55 | 20 | 0.05 [0, 0.15] | 0 [0, 0] |
| all-N10-M10-T100 | 0.05 | 8.1 | 8.1 | 40 | 0 [0, 0] | 0 [0, 0] |
| all-N10-M10-T100 | 0.01 | 3.6 | 3.6 | 40 | 0 [0, 0] | 0 [0, 0] |
| all-N15-M10-T100 | 0.05 | 18.8 | 18.9 | 60 | -0.1 [-0.3, 0] | 0 [0, 0] |
| all-N15-M10-T100 | 0.01 | 9 | 8.7 | 60 | 0.3 [0, 0.8] | 0 [0, 0] |

No interval lies above or below 0, so the two tie in every cell, including those of
`sample_context_series`, where half the true links are at lag 0.

With GPDC on the script's nonlinear model, 2 system variables with one time and one space context:

| Datasets x steps | alpha | Paired | Andrey | tigramite | Empty | D [95%] | Context links D [95%] |
|---|---|---|---|---|---|---|---|
| 5 x 20 | 0.05 | 20 | 3 | 3 | 4 | 0 [-0.4, 0.4] | 0.15 [-0.05, 0.35] |
| 5 x 20 | 0.01 | 20 | 3.4 | 3.6 | 4 | -0.2 [-0.9, 0.4] | -0.15 [-0.5, 0.2] |
| 10 x 20 | 0.05 | 20 | 2.1 | 2.6 | 4 | -0.5 [-1.2, 0.1] | -0.15 [-0.4, 0] |
| 10 x 20 | 0.01 | 20 | 2.8 | 2.7 | 4 | 0.1 [-0.5, 0.7] | -0.15 [-0.3, 0] |
| 20 x 20 | 0.05 | 20 | 1 | 1.2 | 4 | -0.2 [-0.6, 0.2] | -0.05 [-0.2, 0.1] |
| 20 x 20 | 0.01 | 20 | 2.5 | 1.7 | 4 | 0.8 [0.1, 1.5] | 0.1 [0, 0.25] |
| 30 x 20 | 0.05 | 20 | 0.9 | 1.2 | 4 | -0.3 [-0.7, 0.1] | 0 [0, 0] |
| 30 x 20 | 0.01 | 20 | 1.6 | 1.6 | 4 | 0 [-0.5, 0.5] | -0.1 [-0.25, 0] |
| 50 x 20 | 0.05 | 20 | 0.8 | 2 | 4 | -1.2 [-2, -0.4] | 0.1 [-0.1, 0.3] |
| 50 x 20 | 0.01 | 20 | 1.5 | 1.4 | 4 | 0.1 [-0.6, 0.7] | 0.8 [0.6, 0.95] |
| 60 x 20 | 0.05 | 20 | 0.2 | 1.8 | 4 | -1.6 [-2.8, -0.7] | 0 [0, 0] |
| 60 x 20 | 0.01 | 20 | 0.8 | 1.4 | 4 | -0.6 [-1.3, -0.1] | 0 [-0.15, 0.15] |

Andrey made fewer errors at 50 and 60 datasets at alpha 0.05 and at 60 at alpha 0.01. At alpha 0.01
it made more at 20 datasets, and more context-link errors at 50. tigramite's GPDC reads one column
of a vector variable, one level of each dummy (Known differences below), so the two packages do not
run the same tests here.

### Context

On the system variables pooled without an index, Andrey's `pcmciplus`, the paper's baseline, made
15.6 to 23.0 errors on average with every context observed at alpha 0.05, against J-PCMCI+'s 3.1 to
7.7 and the empty graph's 20: the contexts it cannot see add links between the system variables.

Where every context is observed, tigramite without the fix raised "Inconsistent link assumptions" on
243 of 320 fits at alpha 0.05, including every fit at 10 and 15 system variables; the 77 it
finished, 15 of them in the claim, gave the same graph as with the fix.

### Speed

Each package's time runs from the datasets in memory to the returned graph; tigramite's includes its
dummies and DataFrame. Median fit times at alpha 0.05 and, per row, the median over the paired draws
of tigramite's time over Andrey's:

| Cell | Andrey | tigramite | Ratio |
|---|---|---|---|
| all-M10-T10 | 0.026 | 0.47 | 20 |
| all-M10-T20 | 0.032 | 0.89 | 26 |
| all-M10-T30 | 0.032 | 1.03 | 30 |
| all-M10-T50 | 0.035 | 1.32 | 39 |
| all-M10-T100 | 0.057 | 2.13 | 42 |
| all-M10-T150 | 0.067 | 2.2 | 41 |
| all-M10-T200 | 0.08 | 2.84 | 38 |
| half-M10-T10 | 0.021 | 0.42 | 18 |
| half-M10-T20 | 0.028 | 0.74 | 27 |
| half-M10-T30 | 0.035 | 1.12 | 30 |
| half-M10-T50 | 0.072 | 1.58 | 19 |
| half-M10-T100 | 0.098 | 3.64 | 38 |
| half-M10-T150 | 0.148 | 6.56 | 45 |
| half-M10-T200 | 0.171 | 12 | 52 |
| half-M5-T10 | 0.017 | 0.24 | 14 |
| half-M20-T10 | 0.025 | 0.8 | 31 |
| half-M30-T10 | 0.026 | 1.16 | 44 |
| half-M50-T10 | 0.035 | 2.25 | 71 |
| half-M100-T10 | 0.037 | 4.82 | 110 |
| half-M150-T10 | 0.054 | 8.83 | 168 |
| half-M200-T10 | 0.074 | 11 | 166 |
| current-M5-T100 | 0.085 | 0.92 | 13 |
| current-M10-T50 | 0.064 | 0.98 | 14 |
| current-M10-T100 | 0.052 | 1.26 | 23 |
| current-M10-T200 | 0.111 | 1.95 | 18 |
| current-M20-T50 | 0.033 | 1.02 | 34 |
| current-M20-T100 | 0.05 | 1.47 | 34 |
| current-M50-T10 | 0.028 | 1.33 | 50 |
| all-N10-M10-T100 | 0.3 | 20 | 68 |
| all-N15-M10-T100 | 11.5 | 664 | 64 |
| gpdc-M5-T20 | 4.8 | 10 | 2.2 |
| gpdc-M10-T20 | 7.6 | 12 | 1.6 |
| gpdc-M20-T20 | 32.4 | 26 | 0.8 |
| gpdc-M30-T20 | 90.7 | 55 | 0.6 |
| gpdc-M50-T20 | 384.7 | 203 | 0.6 |
| gpdc-M60-T20 | 754.1 | 313 | 0.4 |

In the claim Andrey was 14 to 168 times faster: 14 times with 5 datasets, 18 to 52 times with 10,
and up to 168 times with more, as tigramite's fits grew with the datasets and Andrey's median stayed
under 0.1 s. With GPDC Andrey was faster at 5 and 10 datasets and slower from 20: at 50 datasets its
median fit took 385 s against tigramite's 203 s.

## Run

```shell
benchmarks/start.sh
benchmarks/start_tigramite.sh
export ANDREY_BENCH_TIGRAMITE_PYTHON=$PWD/benchmarks/.venv-tigramite/bin/python
$ANDREY_BENCH_PYTHON benchmarks/jpcmciplus_cpu.py run --out outputs/jpcmciplus \
    --cells all-M10-T100 half-M10-T100 --seeds 0-39
$ANDREY_BENCH_PYTHON benchmarks/jpcmciplus_cpu.py summarize --runs outputs/jpcmciplus \
    --out outputs/jpcmciplus-summary.json
```

`run` draws each dataset and fits every arm of the cell on it, each fit in a fresh interpreter on
one core with one BLAS thread, after one untimed warm-up fit; with GPDC, and for tigramite above 5
system variables, where a fit runs for minutes, the warm-up is skipped. A fit past its cap, 1,800 s
with imports and warm-up or 16,000 MB, is a timeout, and an Andrey fit that raises `ValueError` is
recorded as refused. `--resume` continues a run; `--arms`, `--alphas`, and `--seeds` take part of a
cell. `summarize` pairs the records of one or more runs into the rows above, and `--replications`
adds each row's rerun.

## Known differences

- **The fix.** tigramite 5.2.10.1's `JPCMCIplus` gives both ends of a same-step link between two
  observed contexts the assumption `-->` and then raises "Inconsistent link assumptions". It does
  so whenever such a link is found, as in most draws with both space contexts observed.
  `patches/tigramite-jpcmciplus-context-links.patch` leaves that link `o?o`, and the lane applies
  it in memory. Every fit that finished without the fix gave the same graph with it (Context
  above). Links between contexts are not scored, and Andrey does not test them, following the
  paper's text.
- **No same-step links in the claim.** The authors' script draws every link between system
  variables at lag 1 or 2, so in the claimed cells, as in the paper's figures, the same-step part
  of the error counts false links only. Same-step recovery is measured outside the claim, on
  `sample_context_series`.
- **GPDC on a vector variable.** tigramite's stock `GPDC` regresses the first column of a vector
  variable only, so its tests of a dummy read one level of it, the first time step or the first
  dataset. Andrey tests each level of a dummy against the rest, with a Bonferroni factor. The GPDC
  rows compare the two tests as each package runs them.
- **A time context its own lags determine.** A deterministic time context, such as a sine and
  cosine of one period, is a function of its own lags, which breaks faithfulness (the paper's
  supplement, C). A test of its link given those lags cannot tell anything apart: Andrey keeps the
  links it cannot test, its own rule, and tigramite reads such a test as independence and removes
  the link. The two agree once the contexts carry noise, as every context of the lane does.
- **The paper's code.** The paper's figures came from the authors' `J_PCMCI` class with
  `ParCorrMultNew`, which drops the constant dummy rows, under tigramite 5.2.1.8; the reference here
  is the released `JPCMCIplus` with the stock `ParCorrMult`.

## References

- Gunther, W., Ninad, U., and Runge, J. (2023). Causal discovery for time series from multiple
  datasets with latent contexts. Proceedings of the 39th Conference on Uncertainty in Artificial
  Intelligence (UAI), PMLR 216, 766-776.
- Runge, J. (2020). Discovering contemporaneous and lagged causal relations in autocorrelated
  nonlinear time series datasets. Proceedings of the 36th Conference on Uncertainty in Artificial
  Intelligence (UAI), PMLR 124, 1388-1397.
