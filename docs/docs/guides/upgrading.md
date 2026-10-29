---
name: upgrading
description: What to change in your code when you upgrade Andrey, with the common calls before and
  after each release that renames arguments or changes defaults.
meta:
  type: guide
---

# Upgrading

Andrey is in alpha, so a release may rename an argument or change a default. A renamed argument
keeps its former name for one release: the call still works and warns with
`andrey.DeprecatedArgumentWarning`, which gives the new name. The [changelog](../changelog.md) lists
each release's renamed arguments, changed defaults, and changed outputs under **Breaking**.

To find every former name your code passes, turn the warning into an error and run your code or
your tests:

```python
import warnings

import andrey

warnings.simplefilter("error", andrey.DeprecatedArgumentWarning)
```

## From 0.11 to 0.12

Andrey 0.12.0 renames four settings and reads a column's type only from what you declare. A former
name in the table below still works in 0.12, with an `andrey.DeprecatedArgumentWarning`, and stops
working in the next release:

| Method | Former name | Name now |
| --- | --- | --- |
| `grasp` | `depth` | `grasp_depth` |
| `pcmci` | `pc_alpha` | `pc1_alpha` |
| `pcmciplus` | `collider_rule="none"` | `collider_rule="sepsets"` |
| `andrey.bootstrap` | `n_boot` (`--n-boot`) | `n_resamples` (`--n-resamples`) |
| `andrey.config` | `num_workers` | `n_jobs` |
| environment | `ANDREY_NUM_WORKERS` | `ANDREY_N_JOBS` |
| every method with `categorical` | `max_categories` | none: ignored; declare the column types |

```python
# 0.11
out = andrey.grasp(data, depth=2)
out = andrey.pcmci(series, pc_alpha=0.1)
with andrey.config(num_workers=8):
    boot = andrey.bootstrap(andrey.pc, data, n_boot=200)

# 0.12
out = andrey.grasp(data, grasp_depth=2)
out = andrey.pcmci(series, pc1_alpha=0.1)
with andrey.config(n_jobs=8):
    boot = andrey.bootstrap(andrey.pc, data, n_resamples=200)
```

### Column types

A mixed-data test or score you name, such as `indep_test="cond_gauss"`, read every column of at
most `max_categories` distinct whole numbers as categorical. It now reads only the columns you
declare, with `categorical` or a DataFrame's `category`, `bool`, or text type, as `None` does, and
warns about the others that they may hold category codes. Name such columns to keep the former
result:

```python
# 0.11: column "smoker" holds 0 and 1, read as categorical from its values
out = andrey.pc(df, indep_test="cond_gauss")

# 0.12
out = andrey.pc(df, indep_test="cond_gauss", categorical=["smoker"])
out = andrey.pc(df.astype({"smoker": "category"}))  # None picks cond_gauss from the type
```

A test or score for continuous data, such as `"fisherz"` or `"bic"`, raises `ValueError` on a
column you declare categorical, by its type as well as by `categorical`. Leave the setting as `None`
to pick one that reads categories, or pass `categorical=[]` to read every column as numbers.

`ebic_gamma` takes values from 0 to 1 and only with `score_func="bic"`; drop it from a call with
another score.

## From 0.10 to 0.11

Andrey 0.11.0 gives each setting one name in every method, such as `seed`, `alpha`,
`penalty_discount`, and `tau_max`, and gives the scores short names. The tables in the changelog's
0.11.0 section list every renamed argument and value, changed default, and changed output; the
common calls are below.

### Significance level and test

`pc` keeps `alpha`, and `pcmci`, `pcmciplus`, `lpcmci`, and `rcd` take it too. `indep_test=None`,
the default, picks the test from the column types you declare, with `categorical` or a DataFrame's
`category`, `bool`, or text columns, and `metadata["indep_test"]` names the test the fit used:

```python
# 0.10
out = andrey.pc(df, alpha=0.01)  # fisherz, whatever the column types
out = andrey.pcmci(series, alpha_level=0.05)

# 0.11
out = andrey.pc(df, alpha=0.01)  # gsq when every column is categorical, cond_gauss when some are
out = andrey.pc(df, alpha=0.01, indep_test="fisherz")  # the test of 0.10
out.metadata["indep_test"]
out = andrey.pcmci(series, alpha=0.05)
```

`pcmciplus` and `lpcmci` take `alpha` for `pc_alpha`, and `rcd` for `ind_alpha`.

### Score searches

`lambda_value` is `penalty_discount`, and each score has a short name: `"bic"`, `"discrete_bic"`,
`"bdeu"`, `"cond_gauss"`, and `"degen_gauss"`. On continuous data, the score searches add the
extended-BIC term, `ebic_gamma=1.0`, so their graphs keep fewer false edges; `ebic_gamma=0.0` gives
the graphs of 0.10:

```python
# 0.10
out = andrey.ges(data, score_func="local_score_BIC", lambda_value=2.0)

# 0.11
out = andrey.ges(data, score_func="bic", penalty_discount=2.0)
out = andrey.ges(data, score_func="bic", penalty_discount=2.0, ebic_gamma=0.0)  # as in 0.10
```

### Seeds

```python
# 0.10
out = andrey.ica_lingam(data, random_state=0)

# 0.11
out = andrey.ica_lingam(data, seed=0)
```

`direct_lingam` and `multi_group_direct_lingam` ignored their `random_state`; drop it.

### Lags

`granger` and `longitudinal_lingam` take `tau_max`, the largest lag, as `pcmci`, `pcmciplus`, and
`lpcmci` do, and `granger`'s default is 1, as theirs is, where `maxlag` defaulted to 2:

```python
# 0.10
out = andrey.granger(series, maxlag=2)

# 0.11
out = andrey.granger(series, tau_max=2)
```

`longitudinal_lingam` and `multi_group_direct_lingam` take their list of datasets as `data`, for
`data_list` and `data_groups`.

### Workers

`andrey.bootstrap` runs its fits on the workers `andrey.config` sets, as every method does, and
takes no `num_workers`; the variable `ANDREY_NUM_WORKERS` sets them too:

```python
# 0.11 pre-release builds
boot = andrey.bootstrap(andrey.pc, data, num_workers=8)

# 0.11
with andrey.config(num_workers=8):
    boot = andrey.bootstrap(andrey.pc, data)
```
