# Markov blankets, and parents and children

`blanket_cpu.py` compares Andrey's learners of Markov blankets (`iamb`, `inter_iamb`) and of parents
and children (`mmpc`) with bnlearn's, on the set each learns at a target and on the graph the sets
give. A learner joins the lane by being named in `LEARNERS` of
`andrey_bench/adapters/andrey_blanket.py`, and its reference by being named in `COUNTERPART` of
`andrey_bench/adapters/bnlearn_blanket.py`; the driver runs every learner the installed Andrey has.

## Data

A cell is one data setting, written as the driver's `--cells` take it:

- `net:<name>:<n>`: `n` rows sampled from a published network's own tables, discrete
  (`andrey.data`'s networks, such as `child`, `insurance`, and `alarm`);
- `gauss:<d>:<n>[:<degree>]`: `n` rows of a linear-Gaussian graph on `d` variables with `degree`
  expected edges per variable (2 by default), standardized.

The columns come in an order drawn from the seed, so no learner reads the generating order. The
truth is the generating DAG: its parents and children are the parents-and-children sets, those and
its children's other parents the Markov blankets, and the moral graph and skeleton follow from them.
The default cells are three networks and two Gaussian sizes, each at two sample sizes, on 20 seeds.

## Solutions

| Andrey | bnlearn 5.2.1, at a target | bnlearn, for the whole graph |
|---|---|---|
| `iamb` | `learn.mb(method = "iamb")` | `iamb` |
| `inter_iamb` | `learn.mb(method = "inter.iamb")` | `inter.iamb` |
| `mmpc` | `learn.nbr(method = "mmpc")`, kept to members whose own set holds the target | `mmpc` |
| `gs` | `learn.mb(method = "gs")` | `gs` |
| `fast_iamb` | `learn.mb(method = "fast.iamb")` | `fast.iamb` |
| `hiton_pc` | `learn.nbr(method = "si.hiton.pc")`, kept as for `mmpc` | `si.hiton.pc` |

The last three rows are for learners the lane is ready for: their bnlearn side runs now, and their
Andrey side once the installed Andrey has them. `hiton_mb`, `mmmb`, and `pcmb` have
no bnlearn function.

bnlearn's `pc.stable` is read at a target as well, once per fit: `mb()` of its CPDAG for a
blanket, `nbr()` for parents and children, the answer a graph learner gives to the same question.
`baseline.empty` answers the empty set and the empty graph, the floor each score is read against.

bnlearn is pinned to 5.2.1, the release the lane was written against (`fast.iamb` is deprecated in
it and removed in 2027). The driver stops when the R library holds another version; every row names
the version it ran.

## The same input

Both sides get the same data, the same test, the same `alpha`, and the same largest conditioning
set (`--max-depth`, Andrey's `max_depth` and bnlearn's `max.sx`), and every row records them. Every
setting is passed, never defaulted: a default of one side that the other does not share, such as
bnlearn's t-test for continuous data or the limit a learner of Andrey may carry, would otherwise
change what is compared. `adapters/bnlearn_blanket.py` lists each place bnlearn's defaults differ
from Andrey's and how the lane lines them up. The ones that decide a comparison:

- **The test.** Continuous data runs Fisher's z (`fisherz` and bnlearn's `zf`). Discrete data runs
  the G-square or chi-square test (`--indep-test`, `gsq` or `chisq`) against bnlearn's `mi` or
  `x2` form. `--discrete-test adf` picks `mi-adf` and `x2-adf`, which assume independence when a
  table averages fewer than 5 rows per cell and take empty rows and columns out of the degrees of
  freedom; `--discrete-test plain` picks `mi` and `x2`, which do neither. Andrey's tests have no
  such rule, so with `adf` a sparse table can end a search earlier in bnlearn. The test used, in
  both names, and whether the rule is on, are in every row (`indep_test`, `bnlearn_test`,
  `sparse_rule`).
- **The limit.** With `--max-depth`, bnlearn's `mmpc` and `si.hiton.pc` also stop growing the
  candidate set once it holds more members than the limit, and Andrey's `mmpc` caps only the size of
  the conditioning sets, so the parents-and-children learners agree only with no limit. A blanket
  learner stops growing at the same size on both sides.
- **Parents and children are symmetric.** bnlearn's `learn.nbr` is a one-sided search, and
  Andrey's `mmpc(target=)` keeps a member only when the member's own search holds the target. The
  lane applies the same rule to bnlearn, and counts the tests of those searches. A blanket has no
  such rule on either side.
- **Ties.** bnlearn orders candidates of equal strength by column, Andrey by a rank computed from
  the data, so where several p-values agree, for example when they underflow to zero, the two can
  add variables in a different order and end with different sets. This is a known cause of
  differences, and a difference is a finding to trace, not a setting to tune.

## Measures

`--mode target` (the default) fits each solution once per dataset and answers for `--targets` of
them: every column (`all`), or a number drawn from the seed. The time of a fit is the time for all
its targets. A set is scored against the true set of its kind, per target and pooled:

- `precision`, `recall`, and `f1` of the set; `distance`, the Euclidean distance of (recall,
  precision) to (1, 1); and `exact`, the share of targets whose set is the true one. A ratio with
  a zero denominator is undefined, so an empty set has no precision and an isolated target has no
  recall; the pooled ratios add the counts of every target (`tp`, `est`, `true`) before dividing,
  and `mean_` takes each ratio over the targets where it is defined.
- the number of tests of each search where the package reports it. bnlearn reports them per
  search; a package that reports none has `NO_COUNT` (-1) in the row.
- `same`, in the summary: the targets where a learner's set is the set of its bnlearn counterpart.

`--mode graph` fits once per dataset and `--structures` (`moral`, `skeleton`, and `cpdag`, each
where a learner offers it) and scores the graph's adjacencies, `precision`, `recall`, `f1`, and
`shd` against the empty graph's. The CPDAG's `shd` is `andrey.metrics`', against the true CPDAG.

## Run

Load R as [The R lane](../README.md#the-r-lane) describes, with bnlearn 5.2.1 in the library, then:

```shell
$ANDREY_BENCH_PYTHON benchmarks/blanket_cpu.py --out outputs/blanket-smoke --smoke
$ANDREY_BENCH_PYTHON benchmarks/blanket_cpu.py --out outputs/blanket-cpu --max-depth none
$ANDREY_BENCH_PYTHON benchmarks/blanket_cpu.py --out outputs/blanket-graph --mode graph
```

The defaults are ten cells, 20 seeds from `--first-seed 0`, every target, `--indep-test gsq`,
`--discrete-test adf`, `--alpha 0.05`, no limit on the conditioning set, one untimed warm-up fit,
one timed fit, and `--cap-wall-s` of 1800. `--cells net:alarm:5000 gauss:50:1000:4` selects cells,
`--solutions andrey.iamb,bnlearn.iamb` solutions, and `--part` with `--parts` splits the cells over
jobs. The records go to `records.jsonl`, one line per fit, with each answer, so `--rescore`
recomputes the scores without fitting, and the build and machine to `provenance.json`. A fit starts
a process, so on a small cell most of its wall time is that start, several seconds for R.
