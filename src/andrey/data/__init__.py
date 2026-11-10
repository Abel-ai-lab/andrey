"""Real and synthetic data for causal discovery.

:func:`load_dataset` returns a real dataset, or data simulated on a published network, with its
reference graph; it downloads the file on first use, checks its SHA-256, and caches it in
``ANDREY_DATA_DIR`` or ``~/.cache/andrey``. :func:`list_datasets` names them. A structural causal
model built from a ``graph``, a ``functional`` form, and ``noise`` samples synthetic data with its
true graph:

>>> import andrey.data as data
>>> scm = data.SCM(graph=data.graphs.erdos_renyi(d=10, avg_degree=2),
...                functional=data.functional.linear(),
...                noise=data.noise.gaussian())
>>> ds = scm.sample(n=500, seed=0)
>>> ds.data.shape, ds.graph.kind
((500, 10), 'dag')

:func:`sample_discrete` samples category codes from a discrete Bayesian network with random
conditional probability tables, the data for the discrete tests and scores. :func:`sample_mixed`
samples categorical and continuous variables from a conditional linear Gaussian network, the data
for the mixed-data tests and scores. :meth:`SCM.do` and ``envs=`` on :meth:`SCM.sample`,
:func:`sample_discrete`, and :func:`sample_mixed` draw data under the hard and soft interventions
of :mod:`interventions`. ``missing=`` on :meth:`SCM.sample` and :func:`sample_scm` draws missing
values with a known missingness graph, from :mod:`missingness`.
:func:`sample_context_series` draws time series from several datasets with shared contexts, the
data for :func:`~andrey.jpcmciplus`. ``measurements=`` on :meth:`SCM.sample` and
:func:`sample_scm` appends noisy measurement columns of chosen variables, from :mod:`measurement`.
"""

from __future__ import annotations

from . import (
    benchmarks,
    functional,
    graphs,
    interventions,
    latent,
    measurement,
    missingness,
    noise,
    qa,
    sortability,
    temporal,
)
from .contexts import ContextSeries, sample_context_series
from .dataset import CausalDataset, Measurements, Missingness, QAReport, SCMParams
from .discrete import sample_discrete
from .generate import Ensemble, generate
from .mixed import sample_mixed
from .qa import FaithfulnessScreen
from .real import RealDataset, list_datasets, load_dataset
from .scm import SCM, sample_scm

__all__ = [
    "SCM",
    "sample_scm",
    "sample_discrete",
    "sample_mixed",
    "sample_context_series",
    "ContextSeries",
    "Ensemble",
    "generate",
    "CausalDataset",
    "Missingness",
    "Measurements",
    "QAReport",
    "SCMParams",
    "FaithfulnessScreen",
    "RealDataset",
    "list_datasets",
    "load_dataset",
    "graphs",
    "functional",
    "noise",
    "interventions",
    "missingness",
    "measurement",
    "latent",
    "sortability",
    "qa",
    "benchmarks",
    "temporal",
]
