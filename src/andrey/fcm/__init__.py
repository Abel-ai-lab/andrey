"""Functional causal models: methods that fit each variable as a function of its parents.

The nonlinear additive-noise family models each variable as ``x_j = f_j(parents) + e_j`` with
nonlinear ``f_j`` and noise independent of the parents, which identifies the DAG from the
observational distribution (Hoyer et al. 2009; Peters et al. 2014). :mod:`andrey.fcm.cam` holds
CAM. The family's shared pieces: :mod:`andrey.fcm.gam` fits additive models of penalized thin
plate regression splines, and :mod:`andrey.fcm.pruning` removes the edges of a candidate DAG whose
smooth terms test as insignificant.

References
----------
Hoyer, Janzing, Mooij, Peters, Scholkopf. "Nonlinear causal discovery with additive noise
models." NeurIPS 21, 689-696, 2009.

Peters, Mooij, Janzing, Scholkopf. "Causal discovery with continuous additive noise models."
JMLR 15:2009-2053, 2014.
"""
