"""What every adapter owes the harness, checked across all of them at once.

``params()`` is published on the record as text (``json.dumps(..., sort_keys=True)``), so a value that
does not survive json takes the benchmark down at record assembly — after the fit it has already paid
for. The adapter itself is pickled into the child's job, so one that cannot be pickled fails every
fit instead. The list below is written out because there is no registry to enumerate: a new adapter
joins it here.

Each fit case runs in this process on 6 variables x 200 samples: the fit must return the array its
``to_structure`` reads, and the conversion must give a valid graph of the kind the adapter declares.
Correctness only: no timing, no comparison between packages.
"""

from __future__ import annotations

import json
import pickle
import warnings

import numpy as np
import pytest

from andrey import GraphStructure
from andrey_bench.adapters.andrey_boss import AndreyBOSS
from andrey_bench.adapters.andrey_calm import AndreyCALM
from andrey_bench.adapters.andrey_direct_lingam import andrey_direct_lingam_adapters
from andrey_bench.adapters.andrey_exact_search import AndreyExactSearch
from andrey_bench.adapters.andrey_fci import AndreyFCI
from andrey_bench.adapters.andrey_ges import andrey_ges_adapters
from andrey_bench.adapters.andrey_gfci import AndreyGFCI
from andrey_bench.adapters.andrey_gies import AndreyGIES
from andrey_bench.adapters.andrey_grasp import AndreyGRaSP
from andrey_bench.adapters.andrey_hc import andrey_hc_adapters
from andrey_bench.adapters.andrey_ica_lingam import AndreyICALiNGAM
from andrey_bench.adapters.andrey_pc import AndreyPC
from andrey_bench.adapters.andrey_tabu import andrey_tabu_adapters
from andrey_bench.adapters.andrey_xges import AndreyXGES
from andrey_bench.adapters.baseline_empty import BaselineEmpty
from andrey_bench.adapters.bnlearn_hc import BnlearnHC
from andrey_bench.adapters.bnlearn_pc_stable import BnlearnPCStable
from andrey_bench.adapters.bnlearn_tabu import BnlearnTabu
from andrey_bench.adapters.causal_learn_boss import CausalLearnBOSS
from andrey_bench.adapters.causal_learn_calm import CausalLearnCALM
from andrey_bench.adapters.causal_learn_direct_lingam import CausalLearnDirectLiNGAM
from andrey_bench.adapters.causal_learn_exact_search import CausalLearnExactSearch
from andrey_bench.adapters.causal_learn_fci import CausalLearnFCI
from andrey_bench.adapters.causal_learn_ges import CausalLearnGES
from andrey_bench.adapters.causal_learn_grasp import CausalLearnGRaSP
from andrey_bench.adapters.causal_learn_ica_lingam import CausalLearnICALiNGAM
from andrey_bench.adapters.causal_learn_pc import CausalLearnPC
from andrey_bench.adapters.lingam_direct import LingamDirect
from andrey_bench.adapters.pcalg_fci import PcalgFCI
from andrey_bench.adapters.pcalg_ges import PcalgGES
from andrey_bench.adapters.pcalg_gies import PcalgGIES
from andrey_bench.adapters.pcalg_pc import PcalgPC
from andrey_bench.adapters.pcalg_rfci import PcalgRFCI
from andrey_bench.adapters.tetrad_gfci import TetradGFCI
from andrey_bench.adapters.tigramite_pc import TigramitePC
from andrey_bench.contracts import SolutionAdapter
from andrey_bench.oracle import OracleGES

ALL_ADAPTERS: list[SolutionAdapter] = [
    *andrey_ges_adapters(),
    *andrey_hc_adapters(),
    *andrey_tabu_adapters(),
    AndreyPC(),
    *[AndreyPC(test) for test in ("kci", "rcot", "cmiknn", "gpdc")],
    AndreyFCI(),
    AndreyFCI("majority"),
    AndreyGFCI(),
    AndreyGFCI("parallel"),
    AndreyCALM(),
    AndreyCALM("paper"),
    AndreyBOSS(),
    AndreyGRaSP(),
    *andrey_direct_lingam_adapters(),
    AndreyICALiNGAM(),
    AndreyXGES(),
    AndreyXGES(name="andrey.xges.numpy.no_extended_search", extended_search=False),
    AndreyGIES(),
    AndreyExactSearch(),
    CausalLearnGES(),
    CausalLearnPC(),
    CausalLearnPC("kci"),
    CausalLearnPC("rcot"),
    CausalLearnFCI(),
    CausalLearnCALM(),
    CausalLearnBOSS(),
    CausalLearnGRaSP(),
    CausalLearnDirectLiNGAM(),
    CausalLearnICALiNGAM(),
    CausalLearnExactSearch(),
    LingamDirect(),
    PcalgGES(),
    PcalgGIES(),
    PcalgPC(),
    PcalgFCI(),
    PcalgRFCI(),
    TetradGFCI(),
    BnlearnHC(),
    BnlearnTabu(),
    BnlearnPCStable(),
    TigramitePC("cmiknn"),
    TigramitePC("gpdc"),
    OracleGES(),
    BaselineEmpty(),
]


@pytest.mark.parametrize("adapter", ALL_ADAPTERS, ids=lambda a: a.name)
def test_every_adapter_declares_serializable_params(adapter: SolutionAdapter) -> None:
    assert isinstance(adapter, SolutionAdapter)
    params = adapter.params()
    assert isinstance(params, dict), f"{adapter.name} params must be a dict, got {params!r}"
    json.dumps(params, sort_keys=True)  # raises if a value is not JSON-serializable


def test_every_adapter_has_its_own_name() -> None:
    """The name identifies a solution's records, so two adapters cannot share one."""
    names = [a.name for a in ALL_ADAPTERS]
    assert len(set(names)) == len(names)


def _identity(adapter: SolutionAdapter) -> tuple:
    return (
        adapter.name,
        adapter.backend,
        adapter.output_type,
        adapter.params(),
        getattr(adapter, "device", None),
    )


@pytest.mark.parametrize("adapter", ALL_ADAPTERS, ids=lambda a: a.name)
def test_every_adapter_survives_the_pickle_the_runner_puts_it_through(
    adapter: SolutionAdapter,
) -> None:
    """Adapters must survive pickling into a child process.

    Unpicklable handles, such as open sessions or compiled callables, fail every fit.
    """
    assert _identity(pickle.loads(pickle.dumps(adapter))) == _identity(adapter)


#: The module each adapter's ``fit`` imports on first use (the ``andrey.<method>`` facades load
#: theirs lazily). Setup must import it, so the first timed fit at ``--warmup 0`` does not pay for
#: the import. Every Andrey adapter must be listed.
DEFERRED_MODULES = {
    "andrey.ges.numpy.serial": "andrey.search.ges",
    "andrey.ges.numpy.parallel": "andrey.search.ges",
    "andrey.hc.numpy.serial": "andrey.search.hc",
    "andrey.tabu.numpy.serial": "andrey.search.tabu",
    "andrey.pc.numpy": "andrey.constraint.pc",
    "andrey.pc.kci.numpy": "andrey.core.ci_kernel",
    "andrey.pc.rcot.numpy": "andrey.core.ci_kernel",
    "andrey.pc.cmiknn.numpy": "andrey.core.ci_cmiknn",
    "andrey.pc.gpdc.numpy": "andrey.core.ci_gpdc",
    "andrey.fci.numpy": "andrey.constraint.fci",
    "andrey.fci.majority.numpy": "andrey.constraint.fci",
    "andrey.gfci.numpy": "andrey.search.gfci",
    "andrey.gfci.numpy.parallel": "andrey.search.gfci",
    "andrey.calm.torch-cpu": "andrey.search.calm",
    "andrey.calm.torch-cpu.paper_schedule": "andrey.search.calm",
    "andrey.boss.numpy": "andrey.search.boss",
    "andrey.grasp.numpy": "andrey.search.grasp",
    "andrey.direct_lingam.numpy": "andrey.lingam.direct",
    "andrey.direct_lingam.torch-cuda": "andrey.lingam.direct",
    "andrey.ica_lingam.numpy": "andrey.lingam.ica",
    "andrey.xges.numpy": "andrey.search.xges",
    "andrey.xges.numpy.no_extended_search": "andrey.search.xges",
    "andrey.gies.numpy": "andrey.search.gies",
    "andrey.exact_search.numpy": "andrey.search.exact",
    "causal-learn.exact_search": "causallearn.search.ScoreBased.ExactSearch",
    "causal-learn.boss": "causallearn.search.PermutationBased.BOSS",
    "causal-learn.grasp": "causallearn.search.PermutationBased.GRaSP",
}


#: The module each comparator package imports as. Its fit cases skip where it is not installed.
COMPARATOR_MODULES = {"causal-learn": "causallearn", "lingam": "lingam"}


def _skip_without_package(adapter: SolutionAdapter) -> None:
    module = COMPARATOR_MODULES.get(adapter.package)
    if module is not None:
        pytest.importorskip(module, reason=f"{adapter.package} is not installed")


@pytest.mark.parametrize(
    "adapter",
    [a for a in ALL_ADAPTERS if a.package == "andrey" or a.name in DEFERRED_MODULES],
    ids=lambda a: a.name,
)
def test_setup_imports_the_module_fit_defers(adapter: SolutionAdapter) -> None:
    import sys

    _skip_without_package(adapter)
    module = DEFERRED_MODULES[adapter.name]
    sys.modules.pop(module, None)
    adapter.setup()
    assert module in sys.modules, (
        f"{adapter.name}: setup() left {module} for the first fit to import"
    )


@pytest.mark.parametrize(("pinned", "workers"), [("1", 1), ("4", 4), ("-1", -1), ("", 2), ("x", 2)])
def test_a_parallel_adapter_runs_the_workers_the_runner_pinned(monkeypatch, pinned, workers):
    """`split_cores` pins `ANDREY_NUM_WORKERS` to the core budget, so at `--cores 1` a parallel
    adapter runs one worker, as its record says. The fallback covers only an unset or unreadable
    value."""
    from andrey_bench.adapters import andrey_ges

    monkeypatch.setenv("ANDREY_NUM_WORKERS", pinned)
    assert andrey_ges._parallel_workers() == workers


def test_the_cuda_directlingam_variant_pins_andrey_to_cuda() -> None:
    """The device pin is what makes the torch-cuda variant's records GPU runs."""
    (gpu,) = [a for a in andrey_direct_lingam_adapters() if a.backend == "torch-cuda"]
    assert gpu.device == "cuda"


# --- fit -> to_structure on tiny data -------------------------------------------------------------

#: Values a native array may hold: Andrey endpoint marks (NULL, TAIL, ARROW, and CIRCLE in a PAG),
#: or a plain 0/1 adjacency.
_MARKS, _PAG_MARKS, _ADJACENCY = {0, 1, 2}, {0, 1, 2, 3}, {0, 1}


def _case(adapter: SolutionAdapter, scm: str, kind: str, values: set[int]):
    return pytest.param(adapter, scm, kind, values, id=adapter.name)


#: ``scm`` names an ``andrey.data.benchmarks`` regime: Gaussian ER, latent-confounded ER (one of the
#: 6 nodes hidden), or non-Gaussian scale-free, which LiNGAM needs to identify a direction.
FIT_CASES = [
    *[_case(a, "linear_gauss_er", "cpdag", _MARKS) for a in andrey_ges_adapters()],
    *[_case(a, "linear_gauss_er", "cpdag", _MARKS) for a in andrey_tabu_adapters()],
    _case(CausalLearnGES(), "linear_gauss_er", "cpdag", _ADJACENCY),
    _case(AndreyPC(), "linear_gauss_er", "cpdag", _MARKS),
    _case(CausalLearnPC(), "linear_gauss_er", "cpdag", _ADJACENCY),
    _case(AndreyBOSS(), "linear_gauss_er", "cpdag", _MARKS),
    _case(AndreyGRaSP(), "linear_gauss_er", "cpdag", _MARKS),
    _case(CausalLearnBOSS(), "linear_gauss_er", "cpdag", _ADJACENCY),
    _case(CausalLearnGRaSP(), "linear_gauss_er", "cpdag", _ADJACENCY),
    _case(AndreyXGES(), "linear_gauss_er", "cpdag", _MARKS),
    _case(AndreyExactSearch(), "linear_gauss_er", "dag", _MARKS),
    _case(CausalLearnExactSearch(), "linear_gauss_er", "dag", _ADJACENCY),
    _case(AndreyFCI(), "latent_confounded_er", "pag", _PAG_MARKS),
    _case(AndreyGFCI(), "latent_confounded_er", "pag", _PAG_MARKS),
    _case(CausalLearnFCI(), "latent_confounded_er", "pag", _PAG_MARKS),
    *[
        _case(a, "lingam_sf", "dag", _ADJACENCY)
        for a in andrey_direct_lingam_adapters()
        if a.backend == "numpy"
    ],
    _case(AndreyICALiNGAM(), "lingam_sf", "dag", _ADJACENCY),
    _case(CausalLearnDirectLiNGAM(), "lingam_sf", "dag", _ADJACENCY),
    _case(CausalLearnICALiNGAM(), "lingam_sf", "dag", _ADJACENCY),
    _case(LingamDirect(), "lingam_sf", "dag", _ADJACENCY),
]


@pytest.mark.parametrize(("adapter", "scm", "kind", "values"), FIT_CASES)
def test_a_fit_converts_to_a_valid_graph(
    adapter: SolutionAdapter, scm: str, kind: str, values: set[int]
) -> None:
    import andrey.data as data

    _skip_without_package(adapter)
    x = np.asarray(data.benchmarks.scm(scm, 6).sample(n=200, seed=0).data, dtype=np.float64)
    d = x.shape[1]
    adapter.setup()
    native = np.asarray(adapter.fit(x, adapter.params()))  # what a benchmark runs it with

    assert native.shape == (d, d)
    assert set(np.unique(native).tolist()) <= values
    assert not native.diagonal().any()
    if kind == "dag":
        assert not (native & native.T).any()  # oriented: no 2-cycles

    structure = adapter.to_structure(native)
    assert isinstance(structure, GraphStructure)
    assert (adapter.output_type, structure.kind, structure.n_nodes) == (kind, kind, d)
    structure.validate()  # raises on a malformed graph


def test_the_cuda_direct_lingam_runs_or_falls_back_to_the_numpy_answer() -> None:
    """The CUDA-pinned DirectLiNGAM fits on a GPU when one is present; without one, the pinned
    device warns once and falls back to NumPy, the exact reference, without raising."""
    import andrey.data as data
    from andrey.core import backend

    variants = {a.backend: a for a in andrey_direct_lingam_adapters()}
    cuda, reference = variants["torch-cuda"], variants["numpy"]
    x = np.asarray(data.benchmarks.scm("lingam_sf", 6).sample(n=200, seed=0).data, dtype=np.float64)
    cuda.setup()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # the pinned, unavailable device's warning
        native = np.asarray(cuda.fit(x, cuda.params()))

    assert native.shape == (6, 6) and set(np.unique(native).tolist()) <= _ADJACENCY
    assert not (native & native.T).any()
    cuda.to_structure(native).validate()
    if not backend._is_available("cuda"):
        reference.setup()
        assert np.array_equal(native, np.asarray(reference.fit(x, reference.params())))
