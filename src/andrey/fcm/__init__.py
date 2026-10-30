"""Functional causal models: methods that fit each variable as a function of its parents.

The family's shared pieces live here before its methods do: :mod:`andrey.fcm.gam` fits additive
models of penalized thin plate regression splines, and :mod:`andrey.fcm.pruning` removes the edges
of a candidate DAG whose smooth terms test as insignificant.
"""
