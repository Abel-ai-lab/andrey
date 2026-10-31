"""Functional causal models: methods that fit each variable as a function of its parents.

A functional causal model writes each variable as a function of its parents and a noise term
independent of them, ``x_j = f_j(parents, e_j)``. A restricted class of functions identifies more
of the DAG than the Markov equivalence class that independence tests and scores find (Glymour,
Zhang, and Spirtes 2019). The family's methods take one of two classes. Linear effects with
non-Gaussian noise identify the DAG: :mod:`andrey.lingam` holds the LiNGAM methods. Nonlinear
effects with additive noise, ``x_j = f_j(parents) + e_j``, identify it too (Hoyer et al. 2009;
Peters et al. 2014): :mod:`andrey.fcm.cam` holds CAM and :mod:`andrey.fcm.resit` RESIT. Methods
for time series, hidden variables, and a single pair of variables keep families of their own.

The shared pieces of the nonlinear fits: :mod:`andrey.fcm.gam` fits additive models of penalized
thin plate regression splines, and :mod:`andrey.fcm.pruning` removes the edges of a candidate DAG
whose smooth terms test as insignificant.

References
----------
Glymour, Zhang, Spirtes. "Review of causal discovery methods based on graphical models."
Frontiers in Genetics 10:524, 2019.

Hoyer, Janzing, Mooij, Peters, Scholkopf. "Nonlinear causal discovery with additive noise
models." NeurIPS 21, 689-696, 2009.

Peters, Mooij, Janzing, Scholkopf. "Causal discovery with continuous additive noise models."
JMLR 15:2009-2053, 2014.
"""
