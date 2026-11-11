# NOTEARS and DAGMA

`notears_cpu.py` measures `andrey.notears` (Zheng et al., 2018) and `andrey.dagma` (Bello et al.,
2022) in the setting their papers claim, each against its authors' code, with the empty graph as
the floor and two sortability baselines as controls. `notears_summary.py` pairs the records into the
rows of [`published/0.17.0/notears-dagma.json`](../published/0.17.0/notears-dagma.json).

## Data

Linear structural equation models from `andrey.data.sample_scm`: weights drawn from
+-[0.5, 2], unit noise variances, 1,000 rows, columns shuffled.

- **Graphs.** `ER2`, Erdos-Renyi with `2 d` edges, and `SF4`, scale-free with `4 d` edges, the
  densities the two papers show.
- **Sizes.** 10, 20, and 50 variables on every scale, and 100 on the standardized scale.
- **Noise.** Gaussian, exponential, and Gumbel, each centered and with unit variance.
- **Scales.** `raw` is the papers' own scale. `standardized` z-scores each column after the sample
  is drawn. `rescaled` divides each variable by its standard deviation as it is generated, in
  causal order, so its children read the scaled value.
- **Seeds.** 0 to 19 in every cell; a cell that came out worse than the reference is run again on
  seeds 20 to 39.

On the raw scale the variance of a variable grows along the causal order (var-sortability 0.96 to
0.99; Reisach et al., 2021), so a method can read the order from the columns' units. The
standardized and rescaled scales remove that (var-sortability 0.50). Standardizing leaves the
R2-sortability of the raw data (0.81 to 0.97; Reisach et al., 2023), and rescaling removes it too
(0.41 to 0.48). The papers' experiments are on the raw scale; neither paper makes the units part of
the method, so the standardized cells decide.

## Arms

Every arm fits the same files in the same run:

- `andrey.notears` and `andrey.dagma` at their defaults;
- the references (below);
- each method with `standardize=True`, on the raw and rescaled cells;
- each method with `robust=True`, on the standardized cells up to 20 variables, reported only;
- R2-SortnRegress and Var-SortnRegress of CausalDisco 0.3.0, the sortability controls;
- the empty graph, the floor.

An arm that raises or runs past the cap of 1,800 s per fit is recorded as an error or a timeout and
is never scored as a graph. No fit of any arm failed or timed out: all 10,320 records of the first
run and the 240 of the reruns finished.

## Reference implementations

| method | reference | version | setting |
|---|---|---|---|
| `notears` | the authors' `notears_linear` | `xunzheng/notears` at commit `4a9ab19f` | `lambda1 = 0.1`, `loss_type = "l2"`, `w_threshold = 0.3` |
| `dagma` | the authors' package, `DagmaLinear` | dagma 1.1.1 | `loss_type = "l2"`, `lambda1 = 0.03`, `w_threshold = 0.3` |

Both run unmodified in the environment `start_notears.sh` builds; it downloads the NOTEARS file at
its pinned commit and checks its SHA-256. Both return thresholded weights that nothing in their code
makes acyclic. The benchmark turns a cyclic output into a DAG outside the reference's code, by
dropping the weakest edges until no cycle is left, the rule Andrey applies to its own fits.
The run did not keep those counts, so the recorded datasets were fitted again to count them.
Every refit of the authors' NOTEARS code returned its recorded graph. Of the 1,200 refits of the
authors' dagma, on other hosts of the same processor model, 1,063 did (55 of 120 on the
standardized cells at 50 variables): its fit is not reproducible to the graph across hosts, and its
counts describe the refits.

| reference | scale | cyclic outputs | edges dropped from a cyclic output |
|---|---|---|---|
| NOTEARS code | standardized | 9 of 120 at 10 variables, 79 of 120 at 20, 89 of 120 at 50 | median 1, 4, and 12; at most 1, 12, and 37, of a mean of 11, 23, and 59 edges |
| NOTEARS code | rescaled | 5 of 120 at 10 variables, 20 of 120 at 20 | median 1 and 3; at most 2 and 8 |
| NOTEARS code | raw | 0 of 240 at 10 and 20 variables | |
| dagma | standardized | 0 of 120 at 10 variables, 5 of 120 at 20, 5 of 120 at 50, 0 of 120 at 100 | median 2 and 3; at most 5 and 16 |
| dagma | raw, rescaled | 0 of 720 | |

The NOTEARS code's fits at 100 variables, and at 50 on the raw and rescaled scales, were not
fitted again. So on the standardized cells from 20 variables on, most graphs of the NOTEARS
reference are its output after this repair, which drops edges weakest first whether or not they
lie on a cycle. Andrey's fits apply the same rule and report it in `metadata["edges_removed"]`: on
the 20 standardized `ER2` datasets with Gaussian noise at 20 variables, its NOTEARS fit removed
edges in 9 and its DAGMA fit in none.

## Scores

SHD against the true DAG is the papers' measure and decides. The SHD between the CPDAGs of the
estimate and of the truth (MEC SHD) is reported beside it: with Gaussian noise on standardized
columns the DAG is identified only up to its equivalence class.

A comparison is the per-dataset difference `D = SHD(a) - SHD(b)` on the datasets both arms
finished, with the 95% percentile-bootstrap interval of its mean over 10,000 resamples: worse when
the interval lies wholly above 0, better when wholly below, a tie otherwise. Fewer than 10 paired
datasets are reported and not judged.

The rule, fixed before the run. The decision cells are the standardized ones, 24 per method. A
method is supported when both bars hold:

- **Reference.** In no decision cell is Andrey worse than its reference on the first seeds and
  again on the fresh ones.
- **Floor.** Andrey's interval against the empty graph lies wholly below 0 in more than half of the
  decision cells.

A method that fails the floor is not supported, whatever its reference does. The raw and rescaled
cells, the controls, MEC SHD, and the `robust=True` arms are reported and do not decide.

## Results

Every number is on simulated data, one core and one BLAS thread per fit, on AMD EPYC 7713
processors. The first run holds 10,320 fits in 60.8 CPU-hours of fit time, and the reruns 240
more.

| method | reference | better / tie / worse, 24 decision cells | worse again | below the empty graph | reference below the empty graph | median fit time, reference / Andrey | verdict |
|---|---|---|---|---|---|---|---|
| `notears` | the authors' code | 1 / 23 / 0 | 0 | 15 of 24 | 16 of 24 | 0.66 to 1.64 | meets both bars |
| `dagma` | the authors' package | 0 / 20 / 4 | 3 of 4 | 0 of 24 | 0 of 24 | 0.68 to 0.94 | fails both bars |

By size, NOTEARS is below the empty graph in 4 of 6 decision cells at 10 variables, 3 of 6 at 20,
4 of 6 at 50, and 4 of 6 at 100: in all 12 `ER2` cells and in 3 of the 12 `SF4` cells. DAGMA is
below it in none and above it in 15, the cells from 20 variables on `SF4` and from 50 on `ER2`;
its authors' package is below it in none and above it in 16.

### Decision cells

Mean SHD over the 20 datasets of each cell; differences with their 95% intervals. "Same graph" is
the share of datasets where Andrey and the reference return the same DAG.

#### NOTEARS

| graph | d | noise | Andrey | reference | empty graph | Andrey - reference | same graph | Andrey - empty | reference - empty | result |
|---|---|---|---|---|---|---|---|---|---|---|
| ER2 | 10 | exponential | 18.2 | 18.1 | 20.0 | +0.2 [+0.0, +0.5] | 80% | -1.8 [-2.5, -1.1] | -1.9 [-2.8, -1.2] | tie |
| ER2 | 10 | gaussian | 18.4 | 18.3 | 20.0 | +0.1 [+0.0, +0.4] | 90% | -1.6 [-2.5, -0.8] | -1.7 [-2.5, -0.9] | tie |
| ER2 | 10 | gumbel | 18.8 | 18.6 | 20.0 | +0.2 [+0.0, +0.5] | 75% | -1.2 [-2.1, -0.3] | -1.4 [-2.2, -0.6] | tie |
| ER2 | 20 | exponential | 37.2 | 37.2 | 40.0 | +0.0 [-0.3, +0.3] | 75% | -2.8 [-4.5, -1.2] | -2.8 [-4.5, -1.1] | tie |
| ER2 | 20 | gaussian | 37.4 | 37.5 | 40.0 | -0.1 [-0.1, +0.0] | 75% | -2.6 [-4.7, -0.8] | -2.5 [-4.6, -0.7] | tie |
| ER2 | 20 | gumbel | 37.2 | 37.0 | 40.0 | +0.1 [+0.0, +0.5] | 85% | -2.8 [-4.8, -0.8] | -3.0 [-4.8, -1.1] | tie |
| ER2 | 50 | exponential | 86.9 | 86.8 | 100.0 | +0.1 [-0.4, +0.6] | 35% | -13.1 [-16.1, -10.2] | -13.2 [-16.1, -10.3] | tie |
| ER2 | 50 | gaussian | 88.0 | 87.9 | 100.0 | +0.1 [-0.3, +0.6] | 50% | -12.0 [-15.3, -8.8] | -12.1 [-15.2, -9.1] | tie |
| ER2 | 50 | gumbel | 87.0 | 86.8 | 100.0 | +0.2 [-0.1, +0.6] | 55% | -13.0 [-16.0, -10.0] | -13.2 [-16.3, -10.2] | tie |
| ER2 | 100 | exponential | 157.7 | 157.4 | 200.0 | +0.2 [-0.2, +0.7] | 40% | -42.4 [-47.5, -37.4] | -42.5 [-47.8, -37.5] | tie |
| ER2 | 100 | gaussian | 157.2 | 156.7 | 200.0 | +0.6 [-0.1, +1.2] | 35% | -42.8 [-47.2, -38.1] | -43.3 [-47.9, -38.8] | tie |
| ER2 | 100 | gumbel | 156.7 | 157.0 | 200.0 | -0.3 [-0.9, +0.2] | 30% | -43.3 [-47.3, -39.2] | -43.0 [-47.0, -39.0] | tie |
| SF4 | 10 | exponential | 23.1 | 23.3 | 24.0 | -0.1 [-0.5, +0.0] | 90% | -0.8 [-1.9, +0.1] | -0.7 [-1.8, +0.4] | tie |
| SF4 | 10 | gaussian | 23.1 | 23.3 | 24.0 | -0.1 [-0.3, +0.0] | 75% | -0.8 [-1.9, +0.2] | -0.7 [-1.8, +0.4] | tie |
| SF4 | 10 | gumbel | 22.9 | 22.9 | 24.0 | +0.0 [+0.0, +0.0] | 85% | -1.1 [-1.9, -0.1] | -1.1 [-1.9, -0.1] | tie |
| SF4 | 20 | exponential | 63.6 | 63.8 | 64.0 | -0.1 [-0.6, +0.2] | 60% | -0.3 [-1.6, +0.8] | -0.2 [-1.4, +0.9] | tie |
| SF4 | 20 | gaussian | 64.2 | 64.0 | 64.0 | +0.2 [-0.1, +0.6] | 50% | +0.2 [-0.7, +1.2] | +0.0 [-0.8, +0.9] | tie |
| SF4 | 20 | gumbel | 63.3 | 63.2 | 64.0 | +0.1 [-0.5, +0.6] | 65% | -0.7 [-1.9, +0.6] | -0.8 [-1.9, +0.6] | tie |
| SF4 | 50 | exponential | 181.3 | 181.3 | 184.0 | +0.0 [-0.4, +0.5] | 50% | -2.6 [-5.3, +0.1] | -2.6 [-5.2, -0.1] | tie |
| SF4 | 50 | gaussian | 180.8 | 181.0 | 184.0 | -0.2 [-0.8, +0.5] | 40% | -3.2 [-5.8, -0.6] | -3.0 [-5.5, -0.5] | tie |
| SF4 | 50 | gumbel | 182.6 | 182.6 | 184.0 | +0.0 [-0.5, +0.5] | 45% | -1.4 [-4.4, +1.9] | -1.4 [-4.3, +1.7] | tie |
| SF4 | 100 | exponential | 378.0 | 378.6 | 384.0 | -0.7 [-1.8, +0.5] | 20% | -6.0 [-10.6, -1.2] | -5.3 [-10.1, -0.6] | tie |
| SF4 | 100 | gaussian | 379.4 | 379.9 | 384.0 | -0.5 [-1.5, +0.7] | 10% | -4.6 [-10.3, +1.2] | -4.2 [-10.4, +2.1] | tie |
| SF4 | 100 | gumbel | 379.1 | 380.4 | 384.0 | -1.4 [-2.8, -0.2] | 5% | -4.9 [-10.0, +0.2] | -3.5 [-8.8, +1.7] | better |

#### DAGMA

| graph | d | noise | Andrey | reference | empty graph | Andrey - reference | same graph | Andrey - empty | reference - empty | result |
|---|---|---|---|---|---|---|---|---|---|---|
| ER2 | 10 | exponential | 19.0 | 18.8 | 20.0 | +0.2 [-0.7, +1.3] | 50% | -1.0 [-2.2, +0.3] | -1.2 [-2.6, +0.1] | tie |
| ER2 | 10 | gaussian | 19.1 | 18.9 | 20.0 | +0.1 [-0.3, +0.7] | 55% | -0.9 [-1.9, +0.1] | -1.1 [-2.1, +0.1] | tie |
| ER2 | 10 | gumbel | 19.4 | 19.6 | 20.0 | -0.1 [-0.8, +0.3] | 80% | -0.6 [-1.9, +0.8] | -0.5 [-1.8, +0.9] | tie |
| ER2 | 20 | exponential | 42.1 | 41.8 | 40.0 | +0.3 [-0.4, +1.1] | 45% | +2.1 [-0.8, +5.0] | +1.8 [-0.8, +4.4] | tie |
| ER2 | 20 | gaussian | 42.5 | 42.8 | 40.0 | -0.2 [-0.8, +0.2] | 45% | +2.5 [-0.2, +5.2] | +2.8 [+0.1, +5.5] | tie |
| ER2 | 20 | gumbel | 42.4 | 42.5 | 40.0 | -0.1 [-0.8, +0.7] | 45% | +2.4 [-0.3, +5.2] | +2.5 [-0.3, +5.3] | tie |
| ER2 | 50 | exponential | 112.5 | 110.6 | 100.0 | +1.9 [+0.2, +3.4] | 5% | +12.4 [+8.9, +16.0] | +10.6 [+6.5, +14.6] | worse; rerun +2.1 [+0.4, +4.0] |
| ER2 | 50 | gaussian | 113.5 | 110.0 | 100.0 | +3.5 [+1.7, +5.4] | 5% | +13.6 [+9.1, +18.3] | +10.1 [+6.0, +14.4] | worse; rerun +2.3 [+0.7, +4.2] |
| ER2 | 50 | gumbel | 111.5 | 111.3 | 100.0 | +0.2 [-2.5, +2.6] | 10% | +11.5 [+7.1, +15.9] | +11.3 [+6.7, +15.8] | tie |
| ER2 | 100 | exponential | 211.2 | 206.8 | 200.0 | +4.5 [+1.8, +7.0] | 5% | +11.2 [+6.8, +15.5] | +6.8 [+2.8, +10.8] | tie (did not replicate); rerun +3.9 [-0.1, +7.7] |
| ER2 | 100 | gaussian | 211.2 | 208.9 | 200.0 | +2.2 [-0.1, +4.4] | 10% | +11.2 [+6.8, +15.6] | +8.9 [+5.0, +12.8] | tie |
| ER2 | 100 | gumbel | 212.6 | 207.5 | 200.0 | +5.0 [+2.5, +7.5] | 0% | +12.6 [+8.7, +16.4] | +7.5 [+4.0, +10.8] | worse; rerun +2.7 [+0.2, +5.4] |
| SF4 | 10 | exponential | 23.4 | 23.5 | 24.0 | -0.1 [-0.3, +0.0] | 80% | -0.7 [-1.9, +0.5] | -0.5 [-1.8, +0.7] | tie |
| SF4 | 10 | gaussian | 23.5 | 23.2 | 24.0 | +0.3 [-0.2, +0.9] | 55% | -0.5 [-1.6, +0.7] | -0.8 [-1.9, +0.3] | tie |
| SF4 | 10 | gumbel | 23.6 | 23.4 | 24.0 | +0.3 [-0.2, +1.0] | 55% | -0.3 [-1.4, +0.8] | -0.7 [-1.6, +0.2] | tie |
| SF4 | 20 | exponential | 68.0 | 67.3 | 64.0 | +0.7 [-0.1, +1.8] | 50% | +4.0 [+1.8, +6.3] | +3.3 [+1.1, +5.6] | tie |
| SF4 | 20 | gaussian | 67.3 | 67.5 | 64.0 | -0.2 [-1.2, +0.9] | 25% | +3.3 [+0.9, +5.8] | +3.5 [+1.1, +5.8] | tie |
| SF4 | 20 | gumbel | 68.5 | 67.8 | 64.0 | +0.8 [-0.1, +1.8] | 50% | +4.5 [+1.9, +7.1] | +3.8 [+1.1, +6.3] | tie |
| SF4 | 50 | exponential | 198.9 | 198.3 | 184.0 | +0.6 [-1.0, +2.2] | 15% | +14.9 [+9.3, +20.6] | +14.3 [+9.2, +19.4] | tie |
| SF4 | 50 | gaussian | 198.8 | 197.5 | 184.0 | +1.3 [-0.4, +2.9] | 15% | +14.8 [+9.7, +19.8] | +13.5 [+8.3, +18.6] | tie |
| SF4 | 50 | gumbel | 198.7 | 197.9 | 184.0 | +0.8 [-0.9, +2.2] | 10% | +14.7 [+10.2, +19.1] | +13.9 [+9.6, +18.2] | tie |
| SF4 | 100 | exponential | 409.4 | 408.4 | 384.0 | +0.9 [-1.6, +3.4] | 5% | +25.4 [+19.7, +31.4] | +24.4 [+18.9, +29.9] | tie |
| SF4 | 100 | gaussian | 409.6 | 407.5 | 384.0 | +2.1 [-0.6, +4.5] | 5% | +25.6 [+19.8, +31.2] | +23.5 [+18.5, +28.4] | tie |
| SF4 | 100 | gumbel | 407.7 | 407.5 | 384.0 | +0.2 [-1.4, +1.8] | 0% | +23.7 [+18.6, +28.4] | +23.5 [+18.9, +28.1] | tie |

The four DAGMA cells that were worse on the first seeds ran again on seeds 20 to 39; "rerun" is
that run's difference from the reference.

These DAGMA rows are of the first build, which held the diagonal of the weight matrix at zero;
[DAGMA with a free diagonal](#dagma-with-a-free-diagonal) has the rows of the build that leaves it
free, which ties the reference in all 24 cells.

### Raw, rescaled, and standardized cells beside the controls

Means over the three noises of a graph and size; the counts are over those three cells.

#### Raw

| method | graph | d | var-sortability | R2-sortability | Andrey | reference | empty graph | Var-SortnRegress | R2-SortnRegress | below empty | against reference (better/tie/worse) | below Var-SortnRegress | below R2-SortnRegress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NOTEARS | ER2 | 10 | 0.97 | 0.81 | 2.5 | 2.5 | 20.0 | 4.3 | 17.5 | 3 of 3 | 0/3/0 | 1 of 3 | 3 of 3 |
| NOTEARS | ER2 | 20 | 0.98 | 0.85 | 5.7 | 6.0 | 40.0 | 14.3 | 62.6 | 3 of 3 | 0/3/0 | 3 of 3 | 3 of 3 |
| NOTEARS | ER2 | 50 | 0.98 | 0.90 | 14.4 | 14.6 | 100.0 | 42.7 | 206.3 | 3 of 3 | 0/3/0 | 3 of 3 | 3 of 3 |
| NOTEARS | SF4 | 10 | 0.97 | 0.83 | 5.9 | 5.9 | 24.0 | 5.0 | 17.2 | 3 of 3 | 0/3/0 | 0 of 3 | 3 of 3 |
| NOTEARS | SF4 | 20 | 0.98 | 0.91 | 21.1 | 22.0 | 64.0 | 34.8 | 82.3 | 3 of 3 | 1/2/0 | 3 of 3 | 3 of 3 |
| NOTEARS | SF4 | 50 | 0.99 | 0.95 | 83.5 | 82.8 | 184.0 | 149.9 | 451.7 | 3 of 3 | 0/3/0 | 3 of 3 | 3 of 3 |
| DAGMA | ER2 | 10 | 0.97 | 0.81 | 0.3 | 0.6 | 20.0 | 4.3 | 17.5 | 3 of 3 | 1/2/0 | 3 of 3 | 3 of 3 |
| DAGMA | ER2 | 20 | 0.98 | 0.85 | 1.0 | 0.8 | 40.0 | 14.3 | 62.6 | 3 of 3 | 0/3/0 | 3 of 3 | 3 of 3 |
| DAGMA | ER2 | 50 | 0.98 | 0.90 | 2.4 | 2.7 | 100.0 | 42.7 | 206.3 | 3 of 3 | 0/3/0 | 3 of 3 | 3 of 3 |
| DAGMA | SF4 | 10 | 0.97 | 0.83 | 1.1 | 1.4 | 24.0 | 5.0 | 17.2 | 3 of 3 | 0/3/0 | 3 of 3 | 3 of 3 |
| DAGMA | SF4 | 20 | 0.98 | 0.91 | 6.9 | 6.8 | 64.0 | 34.8 | 82.3 | 3 of 3 | 0/3/0 | 3 of 3 | 3 of 3 |
| DAGMA | SF4 | 50 | 0.99 | 0.95 | 27.2 | 26.6 | 184.0 | 149.9 | 451.7 | 3 of 3 | 0/3/0 | 3 of 3 | 3 of 3 |

#### Rescaled

| method | graph | d | var-sortability | R2-sortability | Andrey | reference | empty graph | Var-SortnRegress | R2-SortnRegress | below empty | against reference (better/tie/worse) | below Var-SortnRegress | below R2-SortnRegress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NOTEARS | ER2 | 10 | 0.50 | 0.46 | 15.7 | 15.7 | 20.0 | 24.3 | 28.0 | 3 of 3 | 0/3/0 | 3 of 3 | 3 of 3 |
| NOTEARS | ER2 | 20 | 0.50 | 0.47 | 28.0 | 27.9 | 40.0 | 83.4 | 92.5 | 3 of 3 | 0/3/0 | 3 of 3 | 3 of 3 |
| NOTEARS | ER2 | 50 | 0.50 | 0.47 | 62.1 | 62.0 | 100.0 | 336.3 | 410.3 | 3 of 3 | 0/3/0 | 3 of 3 | 3 of 3 |
| NOTEARS | SF4 | 10 | 0.50 | 0.42 | 19.7 | 19.6 | 24.0 | 23.9 | 28.8 | 3 of 3 | 0/3/0 | 2 of 3 | 3 of 3 |
| NOTEARS | SF4 | 20 | 0.50 | 0.44 | 54.6 | 54.6 | 64.0 | 108.8 | 123.9 | 3 of 3 | 0/3/0 | 3 of 3 | 3 of 3 |
| NOTEARS | SF4 | 50 | 0.50 | 0.47 | 147.3 | 147.4 | 184.0 | 587.1 | 682.7 | 3 of 3 | 0/3/0 | 3 of 3 | 3 of 3 |
| DAGMA | ER2 | 10 | 0.50 | 0.46 | 14.7 | 14.8 | 20.0 | 24.3 | 28.0 | 3 of 3 | 0/3/0 | 3 of 3 | 3 of 3 |
| DAGMA | ER2 | 20 | 0.50 | 0.47 | 34.0 | 34.0 | 40.0 | 83.4 | 92.5 | 3 of 3 | 0/3/0 | 3 of 3 | 3 of 3 |
| DAGMA | ER2 | 50 | 0.50 | 0.47 | 85.9 | 83.5 | 100.0 | 336.3 | 410.3 | 3 of 3 | 0/0/3 | 3 of 3 | 3 of 3 |
| DAGMA | SF4 | 10 | 0.50 | 0.42 | 18.8 | 18.6 | 24.0 | 23.9 | 28.8 | 3 of 3 | 0/3/0 | 3 of 3 | 3 of 3 |
| DAGMA | SF4 | 20 | 0.50 | 0.44 | 53.2 | 53.3 | 64.0 | 108.8 | 123.9 | 3 of 3 | 0/3/0 | 3 of 3 | 3 of 3 |
| DAGMA | SF4 | 50 | 0.50 | 0.47 | 145.4 | 143.1 | 184.0 | 587.1 | 682.7 | 3 of 3 | 0/2/1 | 3 of 3 | 3 of 3 |

#### Standardized

| method | graph | d | var-sortability | R2-sortability | Andrey | reference | empty graph | Var-SortnRegress | R2-SortnRegress | below empty | against reference (better/tie/worse) | below Var-SortnRegress | below R2-SortnRegress |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NOTEARS | ER2 | 10 | 0.50 | 0.81 | 18.5 | 18.3 | 20.0 | 24.9 | 17.5 | 3 of 3 | 0/3/0 | 3 of 3 | 0 of 3 |
| NOTEARS | ER2 | 20 | 0.50 | 0.85 | 37.3 | 37.2 | 40.0 | 86.9 | 62.6 | 3 of 3 | 0/3/0 | 3 of 3 | 3 of 3 |
| NOTEARS | ER2 | 50 | 0.50 | 0.90 | 87.3 | 87.2 | 100.0 | 405.4 | 206.3 | 3 of 3 | 0/3/0 | 3 of 3 | 3 of 3 |
| NOTEARS | ER2 | 100 | 0.50 | 0.91 | 157.2 | 157.0 | 200.0 | 1117.2 | 529.1 | 3 of 3 | 0/3/0 | 3 of 3 | 3 of 3 |
| NOTEARS | SF4 | 10 | 0.50 | 0.83 | 23.1 | 23.2 | 24.0 | 25.1 | 17.2 | 1 of 3 | 0/3/0 | 1 of 3 | 0 of 3 |
| NOTEARS | SF4 | 20 | 0.50 | 0.91 | 63.7 | 63.7 | 64.0 | 119.5 | 82.3 | 0 of 3 | 0/3/0 | 3 of 3 | 3 of 3 |
| NOTEARS | SF4 | 50 | 0.50 | 0.95 | 181.6 | 181.7 | 184.0 | 712.9 | 451.7 | 1 of 3 | 0/3/0 | 3 of 3 | 3 of 3 |
| NOTEARS | SF4 | 100 | 0.50 | 0.97 | 378.8 | 379.6 | 384.0 | 2359.9 | 1472.7 | 1 of 3 | 1/2/0 | 3 of 3 | 3 of 3 |
| DAGMA | ER2 | 10 | 0.50 | 0.81 | 19.1 | 19.1 | 20.0 | 24.9 | 17.5 | 0 of 3 | 0/3/0 | 3 of 3 | 0 of 3 |
| DAGMA | ER2 | 20 | 0.50 | 0.85 | 42.3 | 42.3 | 40.0 | 86.9 | 62.6 | 0 of 3 | 0/3/0 | 3 of 3 | 3 of 3 |
| DAGMA | ER2 | 50 | 0.50 | 0.90 | 112.5 | 110.6 | 100.0 | 405.4 | 206.3 | 0 of 3 | 0/1/2 | 3 of 3 | 3 of 3 |
| DAGMA | ER2 | 100 | 0.50 | 0.91 | 211.7 | 207.7 | 200.0 | 1117.2 | 529.1 | 0 of 3 | 0/1/2 | 3 of 3 | 3 of 3 |
| DAGMA | SF4 | 10 | 0.50 | 0.83 | 23.5 | 23.4 | 24.0 | 25.1 | 17.2 | 0 of 3 | 0/3/0 | 1 of 3 | 0 of 3 |
| DAGMA | SF4 | 20 | 0.50 | 0.91 | 67.9 | 67.5 | 64.0 | 119.5 | 82.3 | 0 of 3 | 0/3/0 | 3 of 3 | 3 of 3 |
| DAGMA | SF4 | 50 | 0.50 | 0.95 | 198.8 | 197.9 | 184.0 | 712.9 | 451.7 | 0 of 3 | 0/3/0 | 3 of 3 | 3 of 3 |
| DAGMA | SF4 | 100 | 0.50 | 0.97 | 408.9 | 407.8 | 384.0 | 2359.9 | 1472.7 | 0 of 3 | 0/3/0 | 3 of 3 | 3 of 3 |

- **Raw.** Both methods are below the empty graph in all 18 cells and far below it: NOTEARS at 2.5
  to 83.5 against 20 to 184, DAGMA at 0.3 to 27.2. Var-SortnRegress, which only sorts the columns
  by variance and regresses, reaches 4.3 to 150 on the same files, so most of that accuracy is
  available from the units alone; NOTEARS is below it in 13 of the 18 cells and DAGMA in all 18.
  Against the references NOTEARS has 1 cell better and 17 ties, and DAGMA the same.
- **Rescaled.** Both are below the empty graph in all 18 cells, by 14 to 39% for NOTEARS (15.7
  against 20 at 10 variables on `ER2`, 62.1 against 100 at 50) and 13 to 28% for DAGMA, and both
  controls are above it. NOTEARS ties its reference in all 18; DAGMA is worse than its reference in
  4 (the three `ER2` cells at 50 variables and one `SF4` cell there), on the first seeds only.
- **Standardized.** R2-SortnRegress has fewer errors than both methods at 10 variables on `SF4`
  (17.2 against 23.1 and 23.5) and ties them on `ER2`; from 20 variables on it is far above the
  empty graph and both methods are below it.

### MEC SHD

On MEC SHD NOTEARS is below the empty graph in 5 of the 24 decision cells (`ER2` at 50 and 100
variables) and above it in the 9 `SF4` cells from 20 variables on; its reference has the same
counts. DAGMA is below it in none and above it in 19.

| method | graph | d | Andrey below empty (SHD) | Andrey below empty (MEC SHD) | Andrey above empty (MEC SHD) | reference below empty (SHD) | reference below empty (MEC SHD) |
|---|---|---|---|---|---|---|---|
| NOTEARS | ER2 | 10 | 3 | 0 | 0 | 3 | 0 |
| NOTEARS | ER2 | 20 | 3 | 0 | 0 | 3 | 0 |
| NOTEARS | ER2 | 50 | 3 | 2 | 0 | 3 | 2 |
| NOTEARS | ER2 | 100 | 3 | 3 | 0 | 3 | 3 |
| NOTEARS | SF4 | 10 | 1 | 0 | 0 | 1 | 0 |
| NOTEARS | SF4 | 20 | 0 | 0 | 3 | 0 | 0 |
| NOTEARS | SF4 | 50 | 1 | 0 | 3 | 2 | 0 |
| NOTEARS | SF4 | 100 | 1 | 0 | 3 | 1 | 0 |
| DAGMA | ER2 | 10 | 0 | 0 | 0 | 0 | 0 |
| DAGMA | ER2 | 20 | 0 | 0 | 3 | 0 | 0 |
| DAGMA | ER2 | 50 | 0 | 0 | 3 | 0 | 0 |
| DAGMA | ER2 | 100 | 0 | 0 | 3 | 0 | 0 |
| DAGMA | SF4 | 10 | 0 | 0 | 1 | 0 | 0 |
| DAGMA | SF4 | 20 | 0 | 0 | 3 | 0 | 0 |
| DAGMA | SF4 | 50 | 0 | 0 | 3 | 0 | 0 |
| DAGMA | SF4 | 100 | 0 | 0 | 3 | 0 | 0 |

### `standardize=True` and `robust=True`

`standardize=True` on a raw file should return the graph of the default fit on the same seed's
standardized file. Its mean SHD is within 0.5 of that fit's in every cell, and the two graphs are
the same in 30 to 95% of a cell's datasets for NOTEARS and 70 to 100% for DAGMA; why they differ
is not traced. On rescaled files, whose columns already have unit variance, it changes the mean
SHD by at most 0.7.

`robust=True` has no reference arm here. On the 12 standardized cells up to 20 variables, NOTEARS
with it is below the empty graph in 7 and better than the default fit in 3 (`ER2` at 20
variables: 33.0 against 37.3); DAGMA with it is below the empty graph in none and better than the
default fit in 4. Its median fit takes 64 to 96 s for NOTEARS and 16 to 24 s for DAGMA.

| method | graph | d | default, raw | standardize=True, raw | default, standardized | same graph as the standardized fit | standardize=True, rescaled | default, rescaled | robust=True, standardized | robust below empty | robust against default (better/tie/worse) | robust median s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NOTEARS | ER2 | 10 | 2.5 | 18.5 | 18.5 | 83% | 15.7 | 15.7 | 18.8 | 1 of 3 | 0/3/0 | 67 |
| NOTEARS | ER2 | 20 | 5.7 | 37.2 | 37.3 | 78% | 28.0 | 28.0 | 33.0 | 3 of 3 | 3/0/0 | 96 |
| NOTEARS | ER2 | 50 | 14.4 | 87.2 | 87.3 | 45% | 62.1 | 62.1 | - | - | - | - |
| NOTEARS | SF4 | 10 | 5.9 | 23.2 | 23.1 | 82% | 19.7 | 19.7 | 22.2 | 3 of 3 | 0/3/0 | 64 |
| NOTEARS | SF4 | 20 | 21.1 | 63.9 | 63.7 | 55% | 54.6 | 54.6 | 62.1 | 0 of 3 | 0/3/0 | 79 |
| NOTEARS | SF4 | 50 | 83.5 | 181.6 | 181.6 | 30% | 147.5 | 147.3 | - | - | - | - |
| DAGMA | ER2 | 10 | 0.3 | 19.1 | 19.1 | 100% | 14.7 | 14.7 | 19.1 | 0 of 3 | 0/3/0 | 16 |
| DAGMA | ER2 | 20 | 1.0 | 42.3 | 42.3 | 92% | 34.0 | 34.0 | 40.1 | 0 of 3 | 2/1/0 | 24 |
| DAGMA | ER2 | 50 | 2.4 | 112.7 | 112.5 | 90% | 85.9 | 85.9 | - | - | - | - |
| DAGMA | SF4 | 10 | 1.1 | 23.5 | 23.5 | 100% | 18.8 | 18.8 | 23.0 | 0 of 3 | 0/3/0 | 16 |
| DAGMA | SF4 | 20 | 6.9 | 68.0 | 67.9 | 93% | 53.2 | 53.2 | 65.5 | 0 of 3 | 2/1/0 | 21 |
| DAGMA | SF4 | 50 | 27.2 | 198.7 | 198.8 | 75% | 145.3 | 145.4 | - | - | - | - |

### Fit times

Median seconds per fit; each entry is the median over a scale and size's six cells.

| method | scale | d | Andrey | reference | reference / Andrey |
|---|---|---|---|---|---|
| NOTEARS | raw | 10 | 1.9 | 1.4 | 0.74 |
| NOTEARS | raw | 20 | 9.3 | 10.7 | 1.15 |
| NOTEARS | raw | 50 | 110.0 | 169.2 | 1.54 |
| NOTEARS | rescaled | 10 | 1.1 | 0.7 | 0.66 |
| NOTEARS | rescaled | 20 | 5.3 | 6.4 | 1.20 |
| NOTEARS | rescaled | 50 | 26.2 | 41.2 | 1.57 |
| NOTEARS | standardized | 10 | 2.1 | 1.6 | 0.78 |
| NOTEARS | standardized | 20 | 9.4 | 11.4 | 1.22 |
| NOTEARS | standardized | 50 | 38.6 | 63.4 | 1.64 |
| NOTEARS | standardized | 100 | 291.2 | 396.3 | 1.36 |
| DAGMA | raw | 10 | 3.0 | 2.1 | 0.68 |
| DAGMA | raw | 20 | 5.0 | 4.4 | 0.88 |
| DAGMA | raw | 50 | 15.3 | 13.5 | 0.88 |
| DAGMA | rescaled | 10 | 2.4 | 1.7 | 0.70 |
| DAGMA | rescaled | 20 | 3.3 | 2.6 | 0.79 |
| DAGMA | rescaled | 50 | 8.2 | 7.7 | 0.94 |
| DAGMA | standardized | 10 | 2.9 | 2.1 | 0.73 |
| DAGMA | standardized | 20 | 4.3 | 3.7 | 0.86 |
| DAGMA | standardized | 50 | 10.7 | 9.8 | 0.91 |
| DAGMA | standardized | 100 | 46.7 | 38.2 | 0.82 |

## DAGMA with a free diagonal

The DAGMA rows above are of the first build, whose fit held the diagonal of the weight matrix at
zero at every step. The paper's algorithm (Bello et al., 2022, Algorithm 1) solves each stage as an
unconstrained problem over the whole matrix, started at zero: nothing fixes the diagonal, and the
acyclicity function, which counts a self-weight as a cycle, brings it back toward zero as the path
proceeds. The authors' package does the same: its Adam step updates every entry, and the result is
thresholded and nothing else, so a diagonal entry goes only because it is small (at most 5e-5 in
the fits traced). The diagonal changes the path the other weights take through the first stage, and
the path ends at another solution. `andrey.dagma` now leaves the diagonal free in its
least-squares fit; its graph and its returned weights carry no diagonal entry. The likelihood steps
of `robust=True` hold the diagonal at zero, as before.

The rows below are `andrey.dagma` of the build with that change (commit `6c70976ab`), at its
defaults on the same datasets, against the recorded rows of the authors' package and of the empty
graph: 1,200 fits, and 240 more with `robust=True`.

| scale | cells | better / tie / worse than the reference | below / tied with / above the empty graph | reference below / tied with / above the empty graph | better / tie / worse than the first build |
|---|---|---|---|---|---|
| standardized (decision cells) | 24 | 0 / 24 / 0 | 0 / 8 / 16 | 0 / 8 / 16 | 4 / 20 / 0 |
| raw | 18 | 0 / 18 / 0 | 18 / 0 / 0 | 18 / 0 / 0 | 0 / 17 / 1 |
| rescaled | 18 | 0 / 17 / 1 | 18 / 0 / 0 | 18 / 0 / 0 | 4 / 14 / 0 |

On the decision cells DAGMA now ties its authors' package in all 24, where the first build was
worse in 4 on the first seeds and in 3 again on fresh ones. It returns the reference's recorded
graph in every dataset at 10 variables, 96% of them at 20, 84% at 50, and 57% at 100, against 0 to
80% of a cell's datasets in the first build. The floor does not move: it is below the empty graph
in none of the 24 cells and above it in 16, exactly as the authors' package is. The one rescaled
cell worse than the reference, `SF4` at 50 variables with Gaussian noise (+0.6 [+0.1, +1.2]), ran
again on seeds 20 to 39: +0.25 [+0.0, +0.7], a tie, so no cell is worse than the reference twice.
The one raw cell where the first build was better than the reference, `ER2` at 10 variables with
Gumbel noise (0.5 against 1.0), is now a tie at 1.0: the build makes the errors its reference
makes.

Standardized cells, mean SHD over the 20 datasets of each cell; "before the change" is the first
build on the same datasets:

| graph | d | noise | Andrey | reference | empty graph | Andrey - reference | same graph | Andrey - empty | reference - empty | before the change | Andrey - before | result |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ER2 | 10 | exponential | 18.8 | 18.8 | 20.0 | +0.0 [+0.0, +0.0] | 100% | -1.2 [-2.6, +0.1] | -1.2 [-2.6, +0.1] | 19.0 | -0.2 [-1.3, +0.7] | tie |
| ER2 | 10 | gaussian | 18.9 | 18.9 | 20.0 | +0.0 [+0.0, +0.0] | 100% | -1.1 [-2.1, +0.1] | -1.1 [-2.1, +0.1] | 19.1 | -0.1 [-0.7, +0.3] | tie |
| ER2 | 10 | gumbel | 19.6 | 19.6 | 20.0 | +0.0 [+0.0, +0.0] | 100% | -0.5 [-1.8, +0.9] | -0.5 [-1.8, +0.9] | 19.4 | +0.1 [-0.3, +0.8] | tie |
| ER2 | 20 | exponential | 41.8 | 41.8 | 40.0 | +0.0 [+0.0, +0.0] | 100% | +1.8 [-0.8, +4.4] | +1.8 [-0.8, +4.4] | 42.1 | -0.3 [-1.1, +0.4] | tie |
| ER2 | 20 | gaussian | 42.7 | 42.8 | 40.0 | -0.1 [-0.1, +0.0] | 95% | +2.7 [+0.1, +5.5] | +2.8 [+0.1, +5.5] | 42.5 | +0.2 [-0.3, +0.8] | tie |
| ER2 | 20 | gumbel | 42.5 | 42.5 | 40.0 | +0.0 [+0.0, +0.0] | 100% | +2.5 [-0.3, +5.4] | +2.5 [-0.3, +5.3] | 42.4 | +0.1 [-0.7, +0.8] | tie |
| ER2 | 50 | exponential | 110.8 | 110.6 | 100.0 | +0.2 [-0.1, +0.7] | 80% | +10.8 [+6.7, +14.8] | +10.6 [+6.5, +14.6] | 112.5 | -1.6 [-3.2, +0.1] | tie |
| ER2 | 50 | gaussian | 110.0 | 110.0 | 100.0 | +0.0 [-0.1, +0.1] | 90% | +10.1 [+6.0, +14.2] | +10.1 [+6.0, +14.4] | 113.5 | -3.5 [-5.5, -1.8] | tie |
| ER2 | 50 | gumbel | 111.2 | 111.3 | 100.0 | -0.1 [-0.4, +0.0] | 90% | +11.2 [+6.4, +15.7] | +11.3 [+6.7, +15.8] | 111.5 | -0.3 [-2.8, +2.4] | tie |
| ER2 | 100 | exponential | 206.9 | 206.8 | 200.0 | +0.1 [-0.1, +0.4] | 75% | +6.9 [+2.9, +10.9] | +6.8 [+2.8, +10.8] | 211.2 | -4.3 [-6.8, -1.8] | tie |
| ER2 | 100 | gaussian | 208.6 | 208.9 | 200.0 | -0.3 [-0.9, +0.0] | 80% | +8.6 [+4.6, +12.5] | +8.9 [+5.0, +12.8] | 211.2 | -2.6 [-4.7, -0.3] | tie |
| ER2 | 100 | gumbel | 207.4 | 207.5 | 200.0 | -0.1 [-0.3, +0.2] | 60% | +7.5 [+4.0, +10.8] | +7.5 [+4.0, +10.8] | 212.6 | -5.1 [-7.6, -2.6] | tie |
| SF4 | 10 | exponential | 23.5 | 23.5 | 24.0 | +0.0 [+0.0, +0.0] | 100% | -0.5 [-1.8, +0.7] | -0.5 [-1.8, +0.7] | 23.4 | +0.1 [+0.0, +0.3] | tie |
| SF4 | 10 | gaussian | 23.2 | 23.2 | 24.0 | +0.0 [+0.0, +0.0] | 100% | -0.8 [-1.9, +0.3] | -0.8 [-1.9, +0.3] | 23.5 | -0.3 [-0.9, +0.2] | tie |
| SF4 | 10 | gumbel | 23.4 | 23.4 | 24.0 | +0.0 [+0.0, +0.0] | 100% | -0.7 [-1.6, +0.2] | -0.7 [-1.6, +0.2] | 23.6 | -0.3 [-1.0, +0.2] | tie |
| SF4 | 20 | exponential | 67.3 | 67.3 | 64.0 | +0.1 [+0.0, +0.1] | 95% | +3.4 [+1.1, +5.5] | +3.3 [+1.1, +5.6] | 68.0 | -0.7 [-1.8, +0.2] | tie |
| SF4 | 20 | gaussian | 67.8 | 67.5 | 64.0 | +0.3 [+0.0, +0.9] | 95% | +3.8 [+1.4, +6.2] | +3.5 [+1.1, +5.8] | 67.3 | +0.5 [-0.7, +1.6] | tie |
| SF4 | 20 | gumbel | 67.8 | 67.8 | 64.0 | +0.0 [+0.0, +0.0] | 90% | +3.8 [+1.1, +6.2] | +3.8 [+1.1, +6.3] | 68.5 | -0.8 [-1.7, +0.1] | tie |
| SF4 | 50 | exponential | 197.8 | 198.3 | 184.0 | -0.6 [-1.7, +0.2] | 80% | +13.8 [+8.4, +19.1] | +14.3 [+9.2, +19.4] | 198.9 | -1.1 [-2.9, +0.6] | tie |
| SF4 | 50 | gaussian | 197.5 | 197.5 | 184.0 | +0.0 [-0.3, +0.3] | 80% | +13.5 [+8.2, +18.8] | +13.5 [+8.3, +18.6] | 198.8 | -1.3 [-2.9, +0.3] | tie |
| SF4 | 50 | gumbel | 197.8 | 197.9 | 184.0 | -0.1 [-0.5, +0.4] | 85% | +13.8 [+9.4, +18.1] | +13.9 [+9.6, +18.2] | 198.7 | -0.8 [-2.2, +0.9] | tie |
| SF4 | 100 | exponential | 408.4 | 408.4 | 384.0 | +0.0 [-0.7, +0.7] | 35% | +24.4 [+19.1, +29.8] | +24.4 [+18.9, +29.9] | 409.4 | -0.9 [-3.4, +1.5] | tie |
| SF4 | 100 | gaussian | 408.1 | 407.5 | 384.0 | +0.6 [-0.1, +1.4] | 55% | +24.1 [+18.9, +29.1] | +23.5 [+18.5, +28.4] | 409.6 | -1.6 [-3.7, +0.8] | tie |
| SF4 | 100 | gumbel | 407.7 | 407.5 | 384.0 | +0.2 [-0.8, +1.4] | 35% | +23.7 [+18.8, +28.4] | +23.5 [+18.9, +28.1] | 407.7 | +0.0 [-1.6, +1.6] | tie |

Raw and rescaled cells, mean SHD over the three noises (60 datasets a row):

Raw:

| graph | d | Andrey | reference | before the change | empty graph |
|---|---|---|---|---|---|
| ER2 | 10 | 0.6 | 0.6 | 0.3 | 20 |
| ER2 | 20 | 0.8 | 0.8 | 1.0 | 40 |
| ER2 | 50 | 2.7 | 2.7 | 2.4 | 100 |
| SF4 | 10 | 1.4 | 1.4 | 1.1 | 24 |
| SF4 | 20 | 6.8 | 6.8 | 6.9 | 64 |
| SF4 | 50 | 26.1 | 26.6 | 27.2 | 184 |

Rescaled:

| graph | d | Andrey | reference | before the change | empty graph |
|---|---|---|---|---|---|
| ER2 | 10 | 14.8 | 14.8 | 14.7 | 20 |
| ER2 | 20 | 34.0 | 34.0 | 34.0 | 40 |
| ER2 | 50 | 83.5 | 83.5 | 85.9 | 100 |
| SF4 | 10 | 18.6 | 18.6 | 18.8 | 24 |
| SF4 | 20 | 53.3 | 53.3 | 53.2 | 64 |
| SF4 | 50 | 143.3 | 143.1 | 145.4 | 184 |

With `robust=True` on the standardized cells at 10 and 20 variables the mean SHD is 18.8 and 40.0
on `ER2` and 22.9 and 65.2 on `SF4`: below the empty graph in 1 of the 12 cells, above it in none,
and below the default fit in 5.

### At the paper's penalty

The package default of the `l1` penalty is 0.03, in the authors' package and in Andrey; the
experiments of the DAGMA paper use 0.05. `--paper-penalty` runs both codes at 0.05 on the
standardized cells, seeds 0 to 19. This measures the paper's setting beside the default's; it is
not a default.

| penalty | better / tie / worse than the reference | below / tied with / above the empty graph | reference below / tied with / above the empty graph |
|---|---|---|---|
| 0.03, the default | 0 / 24 / 0 | 0 / 8 / 16 | 0 / 8 / 16 |
| 0.05, the paper's | 0 / 24 / 0 | 6 / 12 / 6 | 6 / 12 / 6 |

At 0.05 DAGMA is below the empty graph on `ER2` at 100 variables (175.7 against 200) and in 3 of
the 9 smaller `ER2` cells, tied with it on `SF4` at 10 and 20 variables, and above it on `SF4` at 50
and 100 (189.4 against 184, 392.7 against 384). Andrey ties the authors' package in every cell.

| graph | d | noise | Andrey | reference | empty graph | Andrey - reference | same graph | Andrey - empty | reference - empty | result |
|---|---|---|---|---|---|---|---|---|---|---|
| ER2 | 10 | exponential | 18.9 | 18.9 | 20.0 | +0.0 [+0.0, +0.0] | 100% | -1.1 [-2.4, +0.2] | -1.1 [-2.4, +0.2] | tie |
| ER2 | 10 | gaussian | 18.6 | 18.6 | 20.0 | +0.0 [+0.0, +0.0] | 100% | -1.4 [-2.4, -0.3] | -1.4 [-2.4, -0.3] | tie |
| ER2 | 10 | gumbel | 18.6 | 18.6 | 20.0 | +0.0 [+0.0, +0.0] | 95% | -1.4 [-2.6, -0.1] | -1.4 [-2.6, -0.1] | tie |
| ER2 | 20 | exponential | 39.0 | 38.8 | 40.0 | +0.2 [+0.0, +0.5] | 85% | -1.0 [-3.6, +1.6] | -1.2 [-4.0, +1.5] | tie |
| ER2 | 20 | gaussian | 38.7 | 38.8 | 40.0 | -0.1 [-0.2, +0.0] | 90% | -1.3 [-3.6, +1.1] | -1.2 [-3.5, +1.1] | tie |
| ER2 | 20 | gumbel | 37.6 | 37.6 | 40.0 | +0.0 [+0.0, +0.0] | 100% | -2.4 [-4.7, +0.1] | -2.4 [-4.7, +0.1] | tie |
| ER2 | 50 | exponential | 97.6 | 97.6 | 100.0 | +0.0 [+0.0, +0.0] | 100% | -2.4 [-6.0, +1.1] | -2.4 [-6.0, +1.1] | tie |
| ER2 | 50 | gaussian | 97.2 | 97.0 | 100.0 | +0.1 [-0.1, +0.6] | 90% | -2.9 [-6.5, +0.8] | -3.0 [-6.7, +0.6] | tie |
| ER2 | 50 | gumbel | 95.4 | 95.3 | 100.0 | +0.1 [+0.0, +0.3] | 95% | -4.6 [-8.7, -0.7] | -4.7 [-8.8, -0.8] | tie |
| ER2 | 100 | exponential | 176.2 | 176.2 | 200.0 | -0.1 [-0.3, +0.1] | 80% | -23.9 [-29.0, -19.1] | -23.8 [-29.0, -18.9] | tie |
| ER2 | 100 | gaussian | 174.6 | 174.3 | 200.0 | +0.2 [-0.1, +0.7] | 70% | -25.4 [-30.2, -21.1] | -25.6 [-30.4, -21.2] | tie |
| ER2 | 100 | gumbel | 176.4 | 176.5 | 200.0 | -0.1 [-0.5, +0.2] | 80% | -23.6 [-28.0, -18.9] | -23.5 [-28.0, -18.7] | tie |
| SF4 | 10 | exponential | 23.1 | 23.1 | 24.0 | +0.0 [+0.0, +0.0] | 100% | -0.8 [-1.9, +0.3] | -0.8 [-1.9, +0.3] | tie |
| SF4 | 10 | gaussian | 23.1 | 23.1 | 24.0 | +0.0 [+0.0, +0.0] | 100% | -0.9 [-1.9, +0.2] | -0.9 [-1.9, +0.2] | tie |
| SF4 | 10 | gumbel | 23.4 | 23.4 | 24.0 | +0.0 [+0.0, +0.0] | 100% | -0.6 [-1.7, +0.5] | -0.6 [-1.7, +0.5] | tie |
| SF4 | 20 | exponential | 65.0 | 65.0 | 64.0 | +0.1 [+0.0, +0.1] | 95% | +1.0 [-1.1, +3.0] | +0.9 [-1.1, +3.0] | tie |
| SF4 | 20 | gaussian | 65.5 | 65.3 | 64.0 | +0.1 [+0.0, +0.2] | 90% | +1.4 [-0.4, +3.3] | +1.4 [-0.5, +3.2] | tie |
| SF4 | 20 | gumbel | 65.5 | 65.5 | 64.0 | +0.0 [+0.0, +0.0] | 95% | +1.5 [-0.3, +3.5] | +1.5 [-0.3, +3.5] | tie |
| SF4 | 50 | exponential | 188.8 | 189.3 | 184.0 | -0.5 [-1.4, +0.0] | 90% | +4.8 [+1.6, +8.3] | +5.3 [+2.1, +8.7] | tie |
| SF4 | 50 | gaussian | 189.4 | 189.4 | 184.0 | -0.1 [-0.3, +0.1] | 90% | +5.4 [+1.3, +9.6] | +5.5 [+1.4, +9.6] | tie |
| SF4 | 50 | gumbel | 189.9 | 189.9 | 184.0 | +0.1 [-0.1, +0.2] | 75% | +6.0 [+2.7, +9.2] | +5.9 [+2.7, +9.2] | tie |
| SF4 | 100 | exponential | 393.5 | 393.4 | 384.0 | +0.1 [-0.5, +0.8] | 55% | +9.5 [+3.9, +15.4] | +9.3 [+3.7, +15.3] | tie |
| SF4 | 100 | gaussian | 390.9 | 390.9 | 384.0 | -0.1 [-0.7, +0.6] | 60% | +6.9 [+1.9, +12.6] | +7.0 [+1.8, +12.8] | tie |
| SF4 | 100 | gumbel | 393.8 | 393.4 | 384.0 | +0.5 [-0.4, +1.6] | 70% | +9.8 [+4.5, +15.3] | +9.3 [+3.8, +14.9] | tie |

## Why the least-squares score fails on standardized columns

NOTEARS and DAGMA minimize the same score, half the mean squared residual summed over the variables
plus an `l1` penalty, and differ in how they enforce acyclicity and in their default penalty (0.1
and 0.03). On the raw columns of this simulation a variable's variance grows along the causal order:
92 to 94% of the true edges point from the lower to the higher variance, and a child's standard
deviation is 2.3 to 4.9 times its parent's (median). The score reads that order, and so does sorting
the columns by variance (Var-SortnRegress, above). Z-scoring removes it, and the score's lowest
value then belongs to another graph. The tables below trace this on the standardized cells with the
authors' code of both methods and the weights each returns before its threshold (refits of the
recorded datasets, the build before the diagonal change; 60 datasets a row unless a table says
otherwise).

**The objective prefers the returned graph to the true one.** Each method's own penalized score
(DAGMA at penalty 0.03, NOTEARS at 0.1), per dataset: of the empty graph, of the true weights, of
the best weights on the true DAG, of the method's solution before the threshold, and of the best
weights on the DAG it returns; then the share of datasets in which the solution scores below the
true weights, and the returned DAG below the true DAG.

| method | graph | d | empty | true weights | true DAG, best weights | solution | returned DAG, best weights | solution below true weights | returned DAG below true DAG | datasets |
|---|---|---|---|---|---|---|---|---|---|---|
| DAGMA | ER2 | 10 | 5.00 | 2.06 | 2.01 | 1.74 | 1.78 | 100% | 98% | 60 |
| DAGMA | ER2 | 20 | 10.00 | 4.03 | 3.97 | 3.30 | 3.62 | 100% | 92% | 60 |
| DAGMA | ER2 | 50 | 25.00 | 10.67 | 10.59 | 8.69 | 9.69 | 100% | 97% | 30 |
| DAGMA | SF4 | 10 | 5.00 | 1.91 | 1.84 | 1.59 | 1.64 | 100% | 100% | 60 |
| DAGMA | SF4 | 20 | 10.00 | 3.30 | 3.06 | 2.37 | 2.69 | 100% | 92% | 60 |
| DAGMA | SF4 | 50 | 25.00 | 7.30 | 6.84 | 5.33 | 7.35 | 100% | 43% | 30 |
| NOTEARS | ER2 | 10 | 5.00 | 2.91 | 2.60 | 2.46 | 2.53 | 100% | 82% | 60 |
| NOTEARS | ER2 | 20 | 10.00 | 5.56 | 5.15 | 4.85 | 5.31 | 100% | 33% | 60 |
| NOTEARS | ER2 | 50 | 25.00 | 14.18 | 13.46 | 12.79 | 14.12 | 100% | 17% | 30 |
| NOTEARS | SF4 | 10 | 5.00 | 2.86 | 2.45 | 2.30 | 2.36 | 100% | 80% | 60 |
| NOTEARS | SF4 | 20 | 10.00 | 5.55 | 4.37 | 3.87 | 4.50 | 100% | 40% | 60 |
| NOTEARS | SF4 | 50 | 25.00 | 12.69 | 10.22 | 9.12 | 11.40 | 100% | 10% | 30 |

Both solutions score below the true weights in every dataset. A DAGMA run started at the true
standardized weights leaves them and ends at the error of a run started at zero (SHD 113.2 against
116.6 on `ER2` at 50 variables, where the true weights thresholded would have 32.9; 30 datasets).
So the optimizer is not the cause: the score is.

**The threshold is not the cause.** Mean SHD of the authors' code at its default penalty as the
threshold falls from the default 0.3:

| method | graph | d | 0.05 | 0.1 | 0.15 | 0.2 | 0.25 | 0.3 | empty graph | datasets |
|---|---|---|---|---|---|---|---|---|---|---|
| DAGMA | ER2 | 10 | 21.9 | 20.7 | 20.1 | 19.6 | 19.3 | 19.1 | 20.0 | 60 |
| DAGMA | ER2 | 20 | 55.8 | 51.6 | 48.6 | 46.4 | 44.5 | 42.9 | 40.0 | 60 |
| DAGMA | ER2 | 50 | 135.2 | 127.4 | 122.5 | 118.9 | 115.2 | 111.8 | 100.0 | 60 |
| DAGMA | ER2 | 100 | 293.9 | 253.8 | 235.1 | 221.7 | 212.2 | 205.8 | 200.0 | 40 |
| DAGMA | SF4 | 10 | 26.0 | 25.2 | 24.5 | 24.0 | 23.6 | 23.4 | 24.0 | 60 |
| DAGMA | SF4 | 20 | 78.3 | 74.7 | 72.5 | 70.6 | 68.8 | 68.2 | 64.0 | 60 |
| DAGMA | SF4 | 50 | 227.7 | 220.3 | 214.9 | 209.8 | 203.3 | 198.2 | 184.0 | 60 |
| DAGMA | SF4 | 100 | 440.9 | 440.9 | 440.8 | 432.9 | 414.4 | 401.6 | 384.0 | 30 |
| NOTEARS | ER2 | 10 | 19.6 | 19.3 | 19.2 | 18.8 | 18.4 | 18.3 | 20.0 | 60 |
| NOTEARS | ER2 | 20 | 39.5 | 38.6 | 38.0 | 37.5 | 37.4 | 37.2 | 40.0 | 60 |
| NOTEARS | ER2 | 50 | 86.5 | 86.5 | 86.5 | 86.5 | 86.4 | 86.0 | 100.0 | 30 |
| NOTEARS | SF4 | 10 | 24.2 | 23.7 | 23.4 | 23.2 | 23.3 | 23.2 | 24.0 | 60 |
| NOTEARS | SF4 | 20 | 64.3 | 64.2 | 64.0 | 63.9 | 63.7 | 63.7 | 64.0 | 60 |
| NOTEARS | SF4 | 50 | 180.0 | 180.0 | 180.0 | 180.0 | 179.7 | 179.4 | 184.0 | 30 |

A lower threshold trades fewer missing edges for more extra and more reversed ones: DAGMA is below
the empty graph in none of its 24 cells at any of these thresholds. Of the true edges it keeps
adjacent, DAGMA orients 39 to 52% the right way and NOTEARS 38 to 70%.

**The penalty separates the two methods.** Mean SHD of the authors' code at threshold 0.3 as the
`l1` penalty changes, seeds 0 to 9 (30 datasets a row; NOTEARS at 50 variables was fit at 0.03 and
0.1 only). This is a sensitivity curve on 10 seeds, not a recommended setting:

| graph | d | method | 0 | 0.01 | 0.03 | 0.05 | 0.1 | empty graph |
|---|---|---|---|---|---|---|---|---|
| ER2 | 20 | DAGMA | 76.2 | 52.7 | 43.1 | 38.8 | 35.0 | 40.0 |
| ER2 | 20 | NOTEARS | 72.7 | 52.8 | 42.1 | 40.7 | 36.8 | 40.0 |
| ER2 | 50 | DAGMA | 321.7 | 152.5 | 116.7 | 98.0 | 81.8 | 100.0 |
| ER2 | 50 | NOTEARS | | | 110.2 | | 86.0 | 100.0 |
| SF4 | 20 | DAGMA | 113.4 | 73.4 | 67.4 | 63.5 | 63.6 | 64.0 |
| SF4 | 20 | NOTEARS | 110.6 | 71.6 | 68.6 | 64.7 | 63.3 | 64.0 |
| SF4 | 50 | DAGMA | 637.2 | 226.4 | 194.7 | 188.4 | 180.8 | 184.0 |
| SF4 | 50 | NOTEARS | | | 189.7 | | 179.4 | 184.0 |

At the same penalty the two methods are within a few edges of each other, so DAGMA's result above
the empty graph and NOTEARS's below it in the decision cells separate two default penalties, not
two methods. A larger penalty returns fewer edges: at 0.1 both are below the empty graph on `ER2`
and at it on `SF4`.

**What the literature reports.** Reisach et al. (2021) define varsortability, the share of
directed paths that run from a lower to a higher variance, find it above 0.94 in the usual
simulations and at one half after standardization, and report that NOTEARS is "excellent on raw
data but far worse on standardized data". Kaiser and Sipos (2022) show NOTEARS orienting an edge
differently after the data are normalized. Ng et al. (2024) trace it to the score: least squares is
the Gaussian likelihood with equal noise variances, and after standardization the noise variances
are far from equal, so least-squares methods "are not expected to perform well after standardizing
the data". Deng et al. (2024), with two of DAGMA's authors, report for standardized data that "the
SHD values are not any better than an empty graph" for GOLEM, NOTEARS and DAGMA. The DAGMA paper's
own experiments use the raw simulated columns only, with a penalty of 0.05.

## Status

The bar: never less accurate than the reference in a decision cell that repeats, and below the
empty graph in more than half of the decision cells.

- `notears` meets both bars. Against the authors' code it is better in 1 decision cell, tied in
  23, and worse in none, so no cell needed a rerun, and it is below the empty graph in 15 of 24,
  with the small margins listed under Limits.
- `dagma` meets the reference bar and not the floor. With the free diagonal it ties the authors'
  package in all 24 decision cells. It is below the empty graph in none and above it in 16, as the
  authors' package is, so the floor fails for the method on these data and not for Andrey's
  implementation. On the rescaled cells it is below the empty graph in all 18. It stays
  experimental, with a stated scope: data whose units are meaningful, where the columns' variances
  follow the causal order. On standardized columns, or data whose units are arbitrary, it does no
  better than the empty graph at its default penalty; at the paper's 0.05 it is below the empty
  graph in 6 of 24 cells and above it in 6; and at NOTEARS's penalty it matches NOTEARS (the
  sensitivity curve above).
- In the package `notears` is supported from 0.17.0: its rows are in the release's summary,
  [`published/0.17.0/summary.json`](../published/0.17.0/summary.json), and the
  [0.17.0 release card](../../docs/docs/benchmarks/0.17.0.md#notears) has its
  section. `dagma` is experimental, and the card's
  [DAGMA part](../../docs/docs/benchmarks/0.17.0.md#dagma) states its scope.

## Limits

- **Small margins on standardized columns.** On `ER2` NOTEARS' mean SHD is 6 to 22% below the
  empty graph's (18.4 against 20 at 10 variables with Gaussian noise, 157 against 200 at 100). On
  `SF4` it is within 5% of the empty graph in all 12 cells and meets the floor in 3, and on MEC
  SHD it is above the empty graph there from 20 variables on.
- **One scale-free cell at the empty graph's level.** On `SF4` at 50 variables with exponential
  noise, the reference's interval against the empty graph lies below 0 on the first seeds and
  Andrey's does not, at the same mean (-2.65, [-5.25, -0.15] and [-5.3, +0.05]). The cell was
  rerun on seeds 20 to 39, where neither is below the empty graph and Andrey is worse than the
  reference (+0.65 [+0.1, +1.5], 185.55 against 184.9), and then on seeds 40 to 59, where the two
  tie (+0.25 [-0.6, +1.1], 183.95 against 183.7). Over the 40 fresh datasets the difference is
  +0.45 [-0.1, +1.0], a tie, with Andrey 0.75 above the empty graph ([-1.1, +2.75]) and the
  reference 0.3 above it ([-1.6, +2.4]); over all 60 it is +0.3 [-0.1, +0.7]. The release summary
  records the 40 fresh datasets as the row's replication, with each block of 20.
- **Cycle repair in the NOTEARS comparison.** On the standardized cells the NOTEARS code's
  thresholded output is cyclic in 9 of 120 refits at 10 variables, 79 of 120 at 20, and 89 of 120
  at 50, and the repair drops a median of 12 of about 59 edges at 50. The comparison is of
  Andrey's NOTEARS, which includes that repair, with the authors' code followed by the same
  repair; the authors' code alone does not return a DAG there.
- **Graphs that differ between machines.** The NOTEARS fit ends at a local solution that the last
  bits of the input can change. On the standardized `ER2` cell with Gaussian noise at 50 variables,
  changing every value by a relative 1e-12 gave another graph in 47 of 100 fits of Andrey's NOTEARS
  (ten datasets, ten changes each), and the median SHD over the datasets moved by at most 1. So a
  graph can differ between hosts where the errors do not; the quality gate checks the median.
- **Units.** Both methods' accuracy on raw columns depends on the data's units: the same data
  standardized give 18.5 in place of 2.5 at 10 variables on `ER2` for NOTEARS, and 19.1 in place
  of 0.3 for DAGMA.
- **DAGMA against its reference.** Andrey's DAGMA is worse than the authors' package twice in 3
  decision cells (`ER2`: exponential and Gaussian noise at 50 variables, Gumbel at 100) by 1.9 to
  5.0 SHD of about 110 to 210, and on the first seeds in 4 rescaled cells at 50 variables. The two
  return the same graph in 0 to 15% of the datasets at 50 and 100 variables. The gap is not traced.
- **Coverage.** One sample size (1,000 rows), two densities, linear models. `ges` and `calm` are
  not arms of this lane. The raw and rescaled cells stop at 50 variables.

## Run

```shell
benchmarks/start.sh && benchmarks/start_notears.sh
$ANDREY_BENCH_PYTHON benchmarks/notears_cpu.py --robust --part 0 --parts 24 \
    --out outputs/part-0 --resume
$ANDREY_BENCH_PYTHON benchmarks/notears_cpu.py --sizes 100 --scales standardized --part 0 \
    --parts 16 --out outputs-d100/part-0 --resume
$ANDREY_BENCH_PYTHON benchmarks/notears_cpu.py --first-seed 20 --scales standardized --sizes 50 \
    --graphs er2 --noises exponential gaussian \
    --solutions andrey.dagma bello.dagma baseline.empty --out outputs-rerun/part-0 --resume
$ANDREY_BENCH_PYTHON benchmarks/notears_summary.py --runs outputs/*/ outputs-d100/*/ \
    --replications outputs-rerun/*/ --out notears-dagma.json
$ANDREY_BENCH_PYTHON benchmarks/notears_cpu.py --paper-penalty --sizes 10 20 50 100 \
    --scales standardized --solutions andrey.dagma.paper bello.dagma.paper \
    --out outputs-paper/part-0 --resume
```

`--part i --parts k` splits the grid into independent jobs, and `--first-seed` moves the seeds for
a rerun. `--paper-penalty` adds both DAGMA codes at `lambda1 = 0.05` on the standardized cells.
`start_notears.sh` prints the environment variables the reference arms read.

## Known differences

- **Noise.** The generator centers the exponential and Gumbel noise and gives it unit variance; the
  papers' draws have scale 1 and a nonzero mean. Every fit centers the columns.
- **Timing.** A reference's time is its fit alone. The two controls import scikit-learn inside the
  timed call (1.5 to 2.9 s), which no comparison here reads.
- **dagma without PyTorch.** dagma declares PyTorch as a dependency; its linear model does not
  import it, and the environment installs the package without it.

## References

- Bello, K., Aragam, B., and Ravikumar, P. (2022). DAGMA: Learning DAGs via M-matrices and a
  log-determinant acyclicity characterization. Advances in Neural Information Processing Systems
  35.
- Deng, C., Bello, K., Ravikumar, P., and Aragam, B. (2024). Markov equivalence and consistency in
  differentiable structure learning. Advances in Neural Information Processing Systems 37.
- Kaiser, M. and Sipos, M. (2022). Unsuitability of NOTEARS for causal graph discovery when
  dealing with dimensional quantities. Neural Processing Letters 54(3), 1587-1595.
- Ng, I., Huang, B., and Zhang, K. (2024). Structure learning with continuous optimization: A
  sober look and beyond. CLeaR 2024, PMLR 236, 71-105.
- Reisach, A. G., Seiler, C., and Weichwald, S. (2021). Beware of the simulated DAG! Causal
  discovery benchmarks may be easy to game. Advances in Neural Information Processing Systems 34.
- Reisach, A. G., Tami, M., Seiler, C., Chambaz, A., and Weichwald, S. (2023). A scale-invariant
  sorting criterion to find a causal order in additive noise models. Advances in Neural Information
  Processing Systems 36.
- Zheng, X., Aragam, B., Ravikumar, P., and Xing, E. P. (2018). DAGs with NO TEARS: Continuous
  optimization for structure learning. Advances in Neural Information Processing Systems 31.
