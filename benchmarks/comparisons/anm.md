# ANM direction test

`anm_pairs.py` compares Andrey's `anm` with causal-learn's `ANM` on cause-effect pairs from the
`anm_pair` regime: two nodes at density 1.0, so every pair has one edge, in a direction the seed
draws, with a nonlinear additive-noise mechanism and Gaussian noise. The ladder runs over the
number of rows, not over `d`, since a pair always has two columns. causal-learn returns the two
p-values only, so its adapter decides the edge with Andrey's function: both solutions draw the
direction with the larger p-value. On equal p-values, Andrey takes the direction with the smaller
HSIC statistic relative to its mean under independence, and causal-learn draws no edge.

The defaults are sizes of 200, 500, 1000, and 2000 rows, 20 seeds, one repeat without warm-up, and
`--cap-wall-s` of 1800. The Gaussian-process regression's time grows with the cube of the rows. A
smoke run fits one pair of 100 rows:

```shell
$ANDREY_BENCH_PYTHON benchmarks/anm_pairs.py --out outputs/anm-smoke --smoke
```

Each record gets `direction_correct`, `direction_reversed`, and `direction_none`, one of them 1.0,
from the estimate stored in `structures/`. The summary prints, per size and solution, the median
wall time, the share of pairs with the true direction, and `agree`, the share of pairs on which the
two solutions return the same graph. A rescore recomputes the direction columns without fitting
anything, as in [Hidden-confounder methods](hidden-confounder-methods.md); pass the first run's flags:

```shell
$ANDREY_BENCH_PYTHON benchmarks/anm_pairs.py --out outputs/anm-pairs --rescore
```

Both packages fit the same Gaussian process and the same HSIC test. Andrey standardizes each
column first, so its answer does not depend on the units; causal-learn fits the data as given. The
regime's data are standardized, so the two compute the same p-values here up to rounding, and the
comparison measures time.
