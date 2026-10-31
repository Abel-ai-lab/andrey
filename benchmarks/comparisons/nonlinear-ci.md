# Nonlinear conditional-independence tests

`nonlinear_ci_cpu.py` runs PC, or with `--method fci` FCI, with each nonlinear CI test on additive
nonlinear noise, with rows as an axis of their own. `--regime nonmonotone_er` makes each effect
`w cos(parent)`, an even function partial correlation does not see. Every reference runs at
Andrey's defaults under its own names:

| Andrey | causal-learn | tigramite | RCIT (R) |
|---|---|---|---|
| `indep_test="kci"` | `indep_test="kci"` | - | `KCIT(Bootstrap = FALSE)` |
| `indep_test="rcot"` | `indep_test="rcit"`, `rcit=False`, `approx="hbe"` | - | `RCoT(approx = "hbe")` |
| RCoT `n_features=5`, `n_features_cond=100` | `num_f2=5`, `num_f=100` | - | `num_f2 = 5`, `num_f = 100` |
| `indep_test="cmiknn"` | - | `CMIknn`, `significance="shuffle_test"`, `sig_blocklength=1` | - |
| CMIknn `k_cmi=0.1`, `k_perm=5`, `n_permutations=500` | - | `knn=0.1`, `shuffle_neighbors=5`, `sig_samples=500` | - |
| `indep_test="gpdc"` | - | `GPDC`, `significance="analytic"` | - |
| `alpha=0.05` | `alpha=0.05` | `pc_alpha=0.05`, `contemp_collider_rule="none"` | pcalg `alpha = 0.05` |

pcalg's PC with RCIT's RCoT (`pcalg.pc.rcot`) needs the R library of
[`start_r.sh`](../README.md#the-r-lane), which installs RCIT from its GitHub repository at a fixed
commit. `nonlinear_ci_queries.py` compares the tests themselves: every contestant answers the same
queries on the same seeded draws, independent and dependent, and the summary gives each one's level,
power, distance of its p-values to the uniform under independence, agreement with Andrey's test, and
seconds per query.
