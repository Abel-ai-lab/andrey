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

`tau_max` is the largest lag of every time-series method, and `granger`'s default is 1, as in the
others, where `maxlag` defaulted to 2:

```python
# 0.10
out = andrey.granger(series, maxlag=2)

# 0.11
out = andrey.granger(series, tau_max=2)
```

`longitudinal_lingam` takes `tau_max` for `n_lags`, and it and `multi_group_direct_lingam` take
their list of datasets as `data`.

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
