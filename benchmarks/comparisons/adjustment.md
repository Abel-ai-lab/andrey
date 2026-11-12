# Treatment-outcome methods

`adjustment_cpu.py` measures the methods that answer for a treatment and an outcome: whether the
effect is nonzero, whether it is identifiable, and the adjustment sets to estimate it with. It runs
every Andrey method of the group that is installed, Andrey's PC read at the pair, two baselines, and
the authors' code of the reference methods, on models planted as the method's own paper describes
them, and scores the answers against the truth the method's definition gives.

The lane is built before the methods: the seven Andrey methods fill a table row each
([Adding a method](#adding-a-method)), and no row has run in a campaign yet.

## Data

Each cell of the lane is `setting@n` with optional `key=value` parameters, such as
`locpc-identifiable@5000,v=50`. A model is a function of the setting, its parameters, and the seed;
its rows are a function of the model, `n`, and a stream, so the fresh sample of the effect estimates
is a second sample of the same model (`andrey_bench/adjustment_models.py`). Every generator puts the
observed columns in an order drawn from the seed and the hidden variables after them, and gives the
pair asked about. A constant a paper leaves open is the smallest valid value and is named in the
generator's docstring; where the authors' code differs from the paper, the paper is followed.

| Setting | Paper | Model |
|---|---|---|
| `entner-effect-admissible`, `entner-effect-none`, `entner-zero-admissible`, `entner-zero-none` | Entner, Hoyer, and Spirtes (2013) | 10 observed and 5 hidden covariates before the treatment and the outcome, linear Gaussian; models are redrawn until the task's property holds on the graph (a nonzero or zero effect, and some set of observed covariates valid or none). |
| `ldp` | Maasch et al. (2024) | The 10-node DAG of the paper's Figure 3, one node per partition (confounder, collider, mediator, cause of the outcome, instrument, and the rest), every coefficient 1 and the effect 2.75; `latent=True` is the 18-node DAG of Figure E.5 with the dashed nodes hidden by `hide=B1+Z1`. |
| `ld3-dag` | Maasch et al. (2025) | The 18-node DAG of Figure D.1, the direct effect 1.25 and the other coefficients uniform on `[0.75, 1.1]`; `hide=` hides the one-at-a-time latent variants. |
| `ld3-er` | Maasch et al. (2025) | A random DAG with `1.5 * nodes` arcs whose outcome is the leaf with the most parents (a sink), the treatment one of its parents. |
| `ldecc` | Gupta, Childers, and Lipton (2023) | Appendix C.1: 20 covariates, the treatment, 3 mediators, and the outcome with the paper's edge probabilities, weights, and noise variances. |
| `locpc-identifiable`, `locpc-nonidentifiable` | Loranchet and Assaad (2026) | Erdos-Renyi graphs with two neighbors on average and a pair whose controlled direct effect is identifiable, or not, in the true CPDAG; `family=binary` is the paper's logistic model. |
| `random-graph` | Schubert, Claassen, and Magliacane (2025, 2026) | Random graphs of `nodes` nodes, expected degree `degree`, maximum degree 10, weights `+-[0.5, 3]`, and a pair of which one is an explicit ancestor of the other with an identifiable effect. |
| `magic-niab` | Gupta, Childers, and Lipton (2023) | The fitted linear-Gaussian network (44 nodes, 66 arcs): treatment `G266`, outcome `HT`, or `pair=random`. |
| `sangiovese` | Maasch et al. (2025) | The fitted network without its discrete root `Treatment`, fixed at its most probable level: treatment `pH`, outcome `GrapeW`, the direct effect. |
| `mildew` | Maasch et al. (2024) | The discrete network drawn from its tables: treatment `mikro_1`, outcome `meldug_2`, `dm_1` and `foto_1` hidden. |

Where the sources differ or are silent, the generators say so; the points that matter for reading a
row:

- **Entner.** The paper gives neither the edge density, the weights, nor how a model is brought to a
  task. The lane draws each arc with probability 0.25, weights with a random sign and a magnitude
  uniform on `[0.5, 1]`, and unit noise, and keeps the models that have the task's property.
- **LDP.** The authors' script uses the effect 1 and the paper 2.75; the paper is followed. The
  paper's Figure 6 and Table G.8 give the level of the chi-square test as 0.001 and as 0.005.
- **LD3.** The paper's Figure D.1 draws 16 nodes and an arc `Z4a -> Z4b` the authors' code lacks;
  the code adds `Z7` and an isolated `Z8`. The lane has the figure's arcs and the code's 18 nodes.
  The paper does not say how it handled the discrete root of SANGIOVESE or why it chose the pair.
- **LDECC.** The authors' notebook draws one sign for a whole graph and gives the arcs from the
  treatment to the mediators probability 0.05; the paper's independent signs and 0.1 are followed.
- **LocPC-CDE.** The paper draws the noise variances uniform on `[0.8, 1]` and the authors' script
  the standard deviations; the paper is followed. The authors' script requires the treatment to be a
  compelled parent of the outcome in the CPDAG; the lane follows the paper, a parent in the DAG.
- **SNAP and LOAD.** The SNAP authors' code draws a degree sequence; the lane draws the Erdos-Renyi
  graphs the LOAD paper names, and removes arcs from a node above the maximum degree. Neither paper
  says how a qualifying pair is drawn: the lane takes a node, then one of its explicit descendants,
  uniformly, until the effect is identifiable.

## Solutions

A solution runs on the cells that ask for its estimand and, when it identifies the effect from the
Markov equivalence class, on the cells without hidden variables. Every setting is passed to the
solution, never defaulted; `--alpha` replaces the level of every solution at once.

- `andrey.<method>`: the Andrey methods of the group, one per row of `METHODS` in
  `andrey_bench/adapters/andrey_adjustment.py`, at the level of the method's paper and with the
  cell's test.
- `andrey.pc.adjustment`: the global baseline of every paper. PC's CPDAG, the optimal adjustment set
  where the class identifies the total effect and the local parent sets of the treatment where it
  does not; for a direct effect, identifiable when no undirected edge meets the outcome.
- `baseline.adjust-none`, the unadjusted regression, and `baseline.adjust-all`, the regression on
  every other variable. Neither says whether the effect is zero or identifiable.
- The authors' code, each in an environment of its own that a `start_*.sh` script builds from a
  pinned commit; see [References](#references).

## Measures

The answers are scored by `andrey.metrics`, and the lane adds nothing to their definitions.

- **Answers.** `effect_oracle` gives the records the method's definition returns on the true graph,
  and `effect_scores` scores the answer against them: `nonzero`, `identifiable`, and the sets, each
  as correct, wrong, incomplete, or abstained, with the answer rates and the floor of the constant
  answers. The oracle is built per method: the total effect from the Markov equivalence class
  (LDECC, SNAP, LOAD, the PC arm) or from observed sets of the true graph (Entner, LDP), the sets as
  the method defines them (the optimal set, the valid sets, the local parent sets, the outcome's
  parents). A method run with a d-separation test in place of its own, which defines Entner's `?`
  and LDP's "not identifiable" exactly, is a later arm.
- **Sets.** `adjustment_scores` gives, for each returned set of a total effect, whether it is valid
  in the true graph and, without hidden variables, its precision, recall, and F1 against the optimal
  set.
- **Effect error.** Least squares of the outcome on an intercept, the treatment, and a returned set,
  on a fresh sample of the model, against the true effect; both in standardized units, the estimate
  with the fresh sample's standard deviations and the truth with the model's. The error of a record
  is the mean over its sets of the absolute difference (the intervention distance of Schubert et al.,
  2025, Definition 9.1), and a record that says the effect is zero estimates 0. A record with no set
  that does not say the effect is zero abstains and has no error: it is never scored as the empty
  set. The summary reports the answer rate, and compares the error with `baseline.adjust-none` on
  the fits the solution answered, paired per dataset. The records keep every estimate, so the worst
  set's error, and the Hausdorff distance LDECC's paper reports, can be recomputed.
- **Cost.** `n_tests` where the method reports it, and the wall time.

A fit that passes `--cap-wall-s` is a `timeout` with no answer and fails its cell, never an empty
answer.

## Run

```shell
$ANDREY_BENCH_PYTHON benchmarks/adjustment_cpu.py --out outputs/adj --smoke
```

The defaults are the claimed settings of the papers at their sizes: 20 seeds of each cell, every
solution that is installed. `--cells` takes cells, `--solutions` names, `--alpha` and `--max-depth`
set the same value for every solution that takes it, and `--part` and `--parts` split a campaign
over jobs. `--resume` continues a run and `--rescore` recomputes every score from the stored
answers. `records.jsonl` holds one line per fit with the answer, the scores, and the settings.

MAGIC-NIAB and SANGIOVESE are read once in the R lane's session
([The R lane](../README.md#the-r-lane), with bnlearn in the library), checked against a SHA-256, and
kept as a small JSON file in the networks folder; later runs need no R.

## Adding a method

A method is a row of `METHODS` in `andrey_bench/adapters/andrey_adjustment.py`: what it answers (the
estimand and the kind of set), how its effect is identified, whether it is told the treatment, and
the level of its tests. The adapter finds `andrey.<method>`, passes `treatment` and `outcome` (or
`variables`), `alpha`, `indep_test`, and `max_depth`, and reads `metadata["effects"]` and
`metadata["n_tests"]`. `test_the_table_says_what_the_facade_offers` checks each row against the
installed facade. A solution outside Andrey is a class that fills `PairAdapter`
(`andrey_bench/adjustment_lane.py`): `fit(data, params)` returns `pack_effects(records, d, n_tests)`,
or a packed graph with a `records(native, params)` method that reads the sets where Andrey is
installed, for a reference whose environment is not.
