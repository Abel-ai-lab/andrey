"""Real datasets and published networks, each with its reference graph, downloaded on first use.

``load_dataset(name)`` downloads the dataset's file the first time, checks its SHA-256, and keeps
it in a cache directory, so later loads need no network. The cache directory is ``data_home``
when given, else ``ANDREY_DATA_DIR``, else ``$XDG_CACHE_HOME/andrey`` (``~/.cache/andrey`` when
``XDG_CACHE_HOME`` is unset). Without network access, put the file at the path the error names.

``sachs`` is measured data. ``asia``, ``alarm``, ``hepar2``, and ``andes`` are published
Bayesian-network structures from the bnlearn repository; their data are simulated on the graph by
a seeded linear-Gaussian model, since the networks' own tables are discrete.

>>> from andrey.data import list_datasets
>>> list_datasets()
('sachs', 'asia', 'alarm', 'hepar2', 'andes')
"""

from __future__ import annotations

import gzip
import hashlib
import numbers
import os
import re
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from andrey.core import GraphStructure, env

from . import functional, noise
from .graphs import FixedDAG
from .scm import SCM

__all__ = ["RealDataset", "list_datasets", "load_dataset"]


@dataclass(frozen=True)
class RealDataset:
    """A dataset from :func:`load_dataset`: its observations and the graph they are scored against.

    Attributes
    ----------
    data : pandas.DataFrame or numpy.ndarray of shape (n_samples, n_variables)
        The observations, one row per sample and one column per variable. A DataFrame's column
        names become the node labels of a graph learned from it. With ``as_frame=False``,
        :func:`load_dataset` gives a float64 array in the same column order.
    graph : GraphStructure
        The reference DAG (``kind="dag"``), its nodes labeled with ``feature_names``.
    feature_names : tuple of str
        The variable names, in column order.
    description : str
        What the data are, whether they are measured or simulated, where they come from, and how
        to cite them.
    """

    data: Any
    graph: GraphStructure
    feature_names: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class Contents:
    """What a dataset's file gives: its variables, the reference graph's arcs, and any rows."""

    nodes: tuple[str, ...]
    arcs: tuple[tuple[str, str], ...]
    #: The measurements, or ``None`` for a published structure whose data are simulated.
    values: np.ndarray | None


@dataclass(frozen=True)
class Remote:
    """A downloadable file: the URLs to try in order, its SHA-256, and how to read it."""

    filename: str
    sources: tuple[str, ...]
    sha256: str
    read: Callable[[bytes], Contents]
    description: str
    #: The file holds measurements; otherwise the data are simulated on its graph.
    measured: bool = False


#: The 11 measured proteins and phospholipids, in the file's column order.
SACHS_NODES = ("raf", "mek", "plc", "pip2", "pip3", "erk", "akt", "pka", "pkc", "p38", "jnk")
#: The 17-arc consensus network usually scored against the 853 observational rows.
SACHS_ARCS = (
    ("erk", "akt"),
    ("pka", "akt"),
    ("mek", "erk"),
    ("pka", "erk"),
    ("pka", "jnk"),
    ("pkc", "jnk"),
    ("pka", "mek"),
    ("pkc", "mek"),
    ("raf", "mek"),
    ("pka", "p38"),
    ("pkc", "p38"),
    ("pip3", "pip2"),
    ("plc", "pip2"),
    ("plc", "pip3"),
    ("pkc", "pka"),
    ("pka", "raf"),
    ("pkc", "raf"),
)
#: The file's first condition, with no intervention.
SACHS_ROWS = 853
#: Rows simulated on a published structure when ``n`` is not given.
SIMULATED_ROWS = 1000

_SACHS_DESCRIPTION = """\
Sachs et al. (2005): flow cytometry of 11 phosphorylated proteins and phospholipids in single human
immune-system cells. These are the 853 observational measurements (the first condition, with no
intervention), with the 17-arc consensus signalling network they are usually scored against.

Source: cmu-phil/example-causal-datasets (CC0-1.0), real/sachs/data/sachs.2005.continuous.txt.
Cite: K. Sachs, O. Perez, D. Pe'er, D. A. Lauffenburger, G. P. Nolan. Causal protein-signaling
networks derived from multiparameter single-cell data. Science 308(5721):523-529, 2005.
"""


def _read_sachs(raw: bytes) -> Contents:
    lines = raw.decode("utf-8").splitlines()
    if tuple(lines[0].split()) != SACHS_NODES:
        raise ValueError(f"the Sachs columns {lines[0].split()} are not {SACHS_NODES}")
    values = np.loadtxt(lines[1 : SACHS_ROWS + 1], dtype=np.float64)
    return Contents(SACHS_NODES, SACHS_ARCS, values)


_BIF_VARIABLE = re.compile(r"^\s*variable\s+([^\s{]+)\s*\{", re.MULTILINE)
_BIF_PROBABILITY = re.compile(r"^\s*probability\s*\(([^)]*)\)", re.MULTILINE)


def read_bif(text: str) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    """Return a BIF network's variables, in file order, and its arcs as ``(parent, child)``.

    Only the structure is read: the ``variable`` names, and the parents in each
    ``probability ( child | parent, ... )`` line. The probability tables are ignored.

    Raises
    ------
    ValueError
        If a variable has no probability line or more than one, or a line names an undeclared
        variable.
    """
    nodes = tuple(_BIF_VARIABLE.findall(text))
    declared = set(nodes)
    arcs, children = [], []
    for inside in _BIF_PROBABILITY.findall(text):
        child, _, parents = inside.partition("|")
        child = child.strip()
        children.append(child)
        for parent in (p.strip() for p in parents.split(",") if p.strip()):
            arcs.append((parent, child))
    unknown = sorted({v for arc in arcs for v in arc} - declared | set(children) - declared)
    if unknown:
        raise ValueError(f"the BIF probability lines name undeclared variables {unknown}")
    if sorted(children) != sorted(nodes):
        raise ValueError("every BIF variable needs exactly one probability line")
    return nodes, tuple(arcs)


def _read_bif_gz(raw: bytes) -> Contents:
    nodes, arcs = read_bif(gzip.decompress(raw).decode("utf-8"))
    return Contents(nodes, arcs, None)


_BNLEARN = "https://www.bnlearn.com/bnrepository/{name}/{name}.bif.gz"
_PGMPY = (
    "https://raw.githubusercontent.com/pgmpy/example_models/"
    "920f4e03fce2a319af0fbc8dc6fad1ac99ff4318/discrete/{name}.bif.gz"
)
_SIMULATED = """\
The graph is the published network's structure; the data are simulated on it, not measured. The
network's own probability tables are not used: the rows come from a linear-Gaussian structural
causal model on the graph ({simulation}).

Source: the bnlearn Bayesian network repository (M. Scutari, www.bnlearn.com/bnrepository,
CC BY-SA 3.0), also in pgmpy/example_models (MIT).
Cite: {cite}
"""


def _bnlearn(name: str, sha256: str, summary: str, cite: str) -> Remote:
    return Remote(
        filename=f"{name}.bif.gz",
        sources=(_BNLEARN.format(name=name), _PGMPY.format(name=name)),
        sha256=sha256,
        read=_read_bif_gz,
        description=summary + "\n\n" + _SIMULATED.replace("{cite}", cite),
    )


DATASETS: dict[str, Remote] = {
    "sachs": Remote(
        filename="sachs.2005.continuous.txt",
        sources=(
            "https://raw.githubusercontent.com/cmu-phil/example-causal-datasets/"
            "4ba0565b8164b46063daf0f6fba6b94e5a99baf6/real/sachs/data/sachs.2005.continuous.txt",
        ),
        sha256="a488589b0f021b2a261ff2c696a908c6823051b0d98693e0fa0e78bb12097063",
        read=_read_sachs,
        description=_SACHS_DESCRIPTION,
        measured=True,
    ),
    "asia": _bnlearn(
        "asia",
        "7d5e7b8549834824c1b05f367eb7860fa3fca3adaf3649f0c7be6035af197197",
        "ASIA (Lauritzen and Spiegelhalter, 1988): a fictional diagnosis of tuberculosis, lung "
        "cancer, and bronchitis; 8 variables and 8 arcs.",
        "S. L. Lauritzen, D. J. Spiegelhalter. Local computations with probabilities on graphical "
        "structures and their application to expert systems. Journal of the Royal Statistical "
        "Society B 50(2):157-224, 1988.",
    ),
    "alarm": _bnlearn(
        "alarm",
        "f5860caa7137b5817e0c4f169cce0f1451ce6c246b96f5ce4fb4a8167993be29",
        "ALARM (Beinlich et al., 1989): a patient-monitoring network for anaesthesia; 37 variables "
        "and 46 arcs.",
        "I. A. Beinlich, H. J. Suermondt, R. M. Chavez, G. F. Cooper. The ALARM monitoring system: "
        "a case study with two probabilistic inference techniques for belief networks. "
        "Proceedings of the 2nd European Conference on Artificial Intelligence in Medicine, "
        "247-256, 1989.",
    ),
    "hepar2": _bnlearn(
        "hepar2",
        "f98d4a65c496b79d76b15797e6a6cbeafaff32a17d66a844d8e5b9506d379fcd",
        "HEPAR2 (Onisko, 2003): a network for the diagnosis of liver disorders; 70 variables and "
        "123 arcs.",
        "A. Onisko. Probabilistic Causal Models in Medicine: Application to Diagnosis of Liver "
        "Disorders. PhD thesis, Institute of Biocybernetics and Biomedical Engineering, Polish "
        "Academy of Sciences, 2003.",
    ),
    "andes": _bnlearn(
        "andes",
        "dc9cd41e8658c4fd2b96305cc60392224032f02934d27e43123a61c2eab9dab7",
        "ANDES (Conati et al., 1997): a student model for physics problem solving; 223 variables "
        "and 338 arcs.",
        "C. Conati, A. S. Gertner, K. VanLehn, M. J. Druzdzel. On-line student modeling for "
        "coached problem solving using Bayesian networks. Proceedings of the 6th International "
        "Conference on User Modeling, 231-242, 1997.",
    ),
}


def list_datasets() -> tuple[str, ...]:
    """Return the names :func:`load_dataset` accepts.

    Returns
    -------
    tuple of str
        The dataset names: measured data first, then the published networks from fewest
        variables to most.

    Examples
    --------
    >>> from andrey.data import list_datasets
    >>> list_datasets()
    ('sachs', 'asia', 'alarm', 'hepar2', 'andes')
    """
    return tuple(DATASETS)


def cache_dir(data_home: str | os.PathLike[str] | None = None) -> Path:
    """Return ``data_home``, else ``ANDREY_DATA_DIR``, else ``$XDG_CACHE_HOME/andrey``."""
    if data_home is not None:
        return Path(data_home).expanduser()
    configured = env.DATA_DIR.read()
    if configured is not None:
        return configured
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "andrey"


def fetch(url: str) -> bytes:
    """Return the bytes at ``url``, an ``https://`` address."""
    if not url.startswith("https://"):
        raise ValueError(f"a dataset source must be an https:// URL, got {url!r}")
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def _download(remote: Remote, target: Path, cached: bool) -> bytes:
    """Try each source in order; keep and return the first bytes with the recorded SHA-256."""
    tried = []
    for url in remote.sources:
        try:
            raw = fetch(url)
        except OSError as err:
            tried.append(f"  {url}: {err}")
            continue
        digest = hashlib.sha256(raw).hexdigest()
        if digest != remote.sha256:
            tried.append(f"  {url}: SHA-256 {digest}, not {remote.sha256}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        partial.write_bytes(raw)
        partial.replace(target)
        return raw
    stale = (
        f"\nThe cached file at {target} does not match the SHA-256 and was left in place; "
        "delete or replace it."
        if cached
        else ""
    )
    raise OSError(
        f"could not download {remote.filename}; tried:\n" + "\n".join(tried) + "\nWithout "
        f"network access, download it elsewhere and put it at {target}" + stale
    )


def load_dataset(
    name: str,
    *,
    n: int | None = None,
    seed: int | None = None,
    as_frame: bool = True,
    data_home: str | os.PathLike[str] | None = None,
) -> RealDataset:
    """Load a real dataset, or a published network with data simulated on it, and its graph.

    The first load downloads the dataset's file and keeps it in the cache directory; later loads
    read the cached copy and need no network. Every load checks the file's SHA-256, so a changed
    or truncated file is downloaded again rather than read.

    Parameters
    ----------
    name : str
        The dataset; :func:`list_datasets` returns every name.

        - ``"sachs"``: the protein-signaling measurements of Sachs et al. (2005) [1]_, 853
          observational cells by 11 proteins and phospholipids, with the 17-arc consensus
          network.
        - ``"asia"`` (8 variables, 8 arcs), ``"alarm"`` (37, 46), ``"hepar2"`` (70, 123), and
          ``"andes"`` (223, 338): published Bayesian networks [2]_. The graph is the network's
          structure; the data are simulated on it by a linear-Gaussian structural causal model,
          standardized, since the networks' own tables are discrete.
    n : int or None, default=None
        Rows to simulate on a published network; ``None`` gives 1000. Measured data such as
        ``"sachs"`` take no ``n``.
    seed : int or None, default=None
        Seed of the simulation's weights and noise; ``None`` gives 0. The same seed and ``n``
        give the same rows. Measured data take no ``seed``.
    as_frame : bool, default=True
        Return the observations as a pandas DataFrame named by variable, so a method run on them
        names its nodes. ``False`` returns a float64 NumPy array in the same column order.
    data_home : str or path-like or None, default=None
        The cache directory. ``None`` uses ``ANDREY_DATA_DIR`` when set, else
        ``$XDG_CACHE_HOME/andrey`` (``~/.cache/andrey`` when ``XDG_CACHE_HOME`` is unset).

    Returns
    -------
    RealDataset
        ``data``, ``graph`` (the reference DAG over the same names), ``feature_names``, and
        ``description``, which gives the source and the citation.

    Raises
    ------
    ValueError
        If ``name`` is not a dataset, ``n`` or ``seed`` is given for measured data, ``n`` is not
        an integer of at least 2, or ``seed`` is not a non-negative integer. These are checked
        before any download.
    OSError
        If no source gives the file with its recorded SHA-256, for example without network
        access. The message lists each URL tried and the path to put the file at by hand, and
        names a cached file that fails the check; that file is left in place.

    Notes
    -----
    Each dataset's ``description`` gives its source, license, and citation.

    References
    ----------
    .. [1] K. Sachs, O. Perez, D. Pe'er, D. A. Lauffenburger, G. P. Nolan. Causal protein-signaling
       networks derived from multiparameter single-cell data. Science 308(5721):523-529, 2005.
    .. [2] M. Scutari. The bnlearn Bayesian network repository,
       https://www.bnlearn.com/bnrepository. The same files are mirrored in pgmpy/example_models.

    Examples
    --------
    >>> from andrey.data import load_dataset
    >>> sachs = load_dataset("sachs")  # doctest: +SKIP
    >>> sachs.data.shape, len(sachs.graph.oriented_edges())  # doctest: +SKIP
    ((853, 11), 17)
    >>> alarm = load_dataset("alarm", n=2000, seed=1)  # doctest: +SKIP
    >>> alarm.data.shape, len(alarm.graph.oriented_edges())  # doctest: +SKIP
    ((2000, 37), 46)
    """
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; choose one of {list_datasets()}")
    remote = DATASETS[name]
    if remote.measured and (n is not None or seed is not None):
        raise ValueError(f"{name!r} is measured data: n and seed apply to simulated datasets")
    if n is not None and not (_is_int(n) and n >= 2):
        raise ValueError(f"n must be an integer of at least 2, got {n!r}")
    if seed is not None and not (_is_int(seed) and seed >= 0):
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
    target = cache_dir(data_home) / name / remote.filename
    cached = target.is_file()
    raw = target.read_bytes() if cached else b""
    if hashlib.sha256(raw).hexdigest() != remote.sha256:
        raw = _download(remote, target, cached)
    contents = remote.read(raw)
    description = remote.description
    values = contents.values
    if values is None:
        n, seed = SIMULATED_ROWS if n is None else n, 0 if seed is None else seed
        scm = SCM(
            graph=FixedDAG(nodes=contents.nodes, arcs=contents.arcs),
            functional=functional.linear(),
            noise=noise.gaussian(),
        )
        values = scm.sample(n=n, seed=seed, scale="standardize").data
        description = description.replace("{simulation}", f"n={n}, seed={seed}")
    data: Any = values
    if as_frame:
        import pandas as pd

        data = pd.DataFrame(values, columns=list(contents.nodes))
    return RealDataset(data, _graph(contents), contents.nodes, description)


def _is_int(value: object) -> bool:
    """Whether ``value`` is an integer, a NumPy one included, and not a bool."""
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _graph(contents: Contents) -> GraphStructure:
    """Return the reference DAG over the dataset's variable names."""
    index = {name: i for i, name in enumerate(contents.nodes)}
    marks = np.zeros((len(contents.nodes), len(contents.nodes)), dtype=np.int8)
    for parent, child in contents.arcs:
        marks[index[parent], index[child]] = 1  # a tail at the parent
        marks[index[child], index[parent]] = 2  # an arrowhead at the child
    return GraphStructure.from_numpy(marks, kind="dag", labels=contents.nodes)
