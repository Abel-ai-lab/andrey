"""The datasets and the counts of the lane of target-centered methods on categorical networks.

MB-by-MB and CMB answer for one target: the edges it has and their directions. The lane scores the
answer at the target against the true graph in the two readings of the methods' papers.

- :func:`dag_counts`, the MB-by-MB paper's (Wang et al., 2014, Tables 1-3): the answer's edges at
  the target against the true DAG, as ``undirected``, ``miss``, ``extra``, and ``reverse``, with
  ``shd`` their sum. An edge the answer leaves undirected counts once, as ``undirected``, whatever
  the truth's direction; a directed edge with the wrong direction is a ``reverse``.
- :func:`cpdag_counts`, the CMB paper's (Gao and Ji, 2015, Table 1): the same edges against the
  true CPDAG, every DAG of the true class being a right answer, as ``extra``, ``missing``, and
  ``reversed``, the last split into ``wrongdir`` (a directed edge of the opposite direction),
  ``over`` (a directed edge where the CPDAG's is undirected), and ``under`` (an undirected edge
  where the CPDAG's is directed); ``total`` is ``extra + missing + reversed``.

A cell draws ``n`` rows of a network (:func:`draw`): ``alarm`` from :mod:`andrey.data`, the
chessboard of :mod:`andrey_bench.chessboard`, and any other published network from
:func:`andrey_bench.networks.load_network`, its columns permuted by the seed. The truth is the
same for every ``n`` (:func:`truth`). The counts import NumPy only; :func:`draw` and
:func:`truth` import Andrey when called.

References
----------
Wang, C., Zhou, Y., Zhao, Q., and Geng, Z. (2014). Discovering and orienting the edges connected to
a target variable in a DAG via a sequential local learning approach. Computational Statistics and
Data Analysis 77, 252-266.

Gao, T., and Ji, Q. (2015). Local causal discovery of direct causes and effects. Advances in
Neural Information Processing Systems 28, 2512-2520.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from andrey_bench import chessboard

#: The name of the chessboard, the one network of the lane that is not a published file.
CHESSBOARD = "chess45"

# Andrey endpoint marks (mirrors contracts.py).
_NULL, _TAIL, _ARROW = 0, 1, 2


@dataclass(frozen=True)
class Draw:
    """A dataset and its truth, in the data's column order."""

    #: ``(n, d)`` ``float64`` category codes.
    data: np.ndarray
    #: ``dag[i, j] = 1`` for the true edge ``i -> j``, ``int8``.
    dag: np.ndarray
    #: The variables' names, one per column.
    labels: list[str]
    #: The columns the lane answers for: every column, or the chessboard's ten squares.
    targets: list[int]


def marks_of(dag: np.ndarray) -> np.ndarray:
    """The endpoint marks of the DAG ``dag``: ``marks[i, j]`` is the mark at ``i``.

    An edge ``i -> j`` is a tail at ``i`` and an arrowhead at ``j``.
    """
    d = dag.shape[0]
    marks = np.zeros((d, d), dtype=np.int8)
    i, j = np.nonzero(dag)
    marks[i, j] = _TAIL
    marks[j, i] = _ARROW
    return marks


def cpdag_of(dag: np.ndarray) -> np.ndarray:
    """The endpoint marks of the CPDAG of the DAG ``dag``."""
    from andrey.core.orient import dag2cpdag

    return dag2cpdag(marks_of(dag))


def star_at(marks: np.ndarray, target: int) -> np.ndarray:
    """``marks`` with every edge away from ``target`` removed: its row and its column only."""
    marks = np.asarray(marks)
    star = np.zeros_like(marks)
    star[target], star[:, target] = marks[target], marks[:, target]
    return star


def _neighbors(marks: np.ndarray, target: int) -> set[int]:
    return set(np.flatnonzero(marks[target] != _NULL).tolist())


def dag_counts(marks: np.ndarray, dag_marks: np.ndarray, target: int) -> dict[str, int]:
    """Count the answer's errors at ``target`` against the true DAG, Wang et al. (2014).

    ``marks`` is the answer and ``dag_marks`` the truth, both endpoint marks as :func:`marks_of`
    gives them. Returns ``undirected``, ``miss``, ``extra``, ``reverse``, and ``shd``, their sum.
    """
    true, found = _neighbors(dag_marks, target), _neighbors(marks, target)
    undirected = reverse = 0
    for j in found & true:
        if marks[target, j] == _TAIL and marks[j, target] == _TAIL:
            undirected += 1
        elif (marks[target, j], marks[j, target]) != (dag_marks[target, j], dag_marks[j, target]):
            reverse += 1
    counts = {
        "undirected": undirected,
        "miss": len(true - found),
        "extra": len(found - true),
        "reverse": reverse,
    }
    counts["shd"] = sum(counts.values())
    return counts


def cpdag_counts(marks: np.ndarray, cpdag_marks: np.ndarray, target: int) -> dict[str, int]:
    """Count the answer's errors at ``target`` against the true CPDAG, Gao and Ji (2015).

    ``marks`` is the answer and ``cpdag_marks`` the truth's CPDAG, endpoint marks. Returns
    ``extra``, ``missing``, ``reversed``, its parts ``wrongdir``, ``over``, and ``under``, and
    ``total``, the sum of the first three.
    """
    true, found = _neighbors(cpdag_marks, target), _neighbors(marks, target)
    wrongdir = over = under = 0
    for j in found & true:
        mine = (marks[target, j], marks[j, target])
        theirs = (cpdag_marks[target, j], cpdag_marks[j, target])
        if mine == theirs:
            continue
        if theirs == (_TAIL, _TAIL):
            over += 1
        elif mine == (_TAIL, _TAIL):
            under += 1
        else:
            wrongdir += 1
    counts = {
        "extra": len(found - true),
        "missing": len(true - found),
        "reversed": wrongdir + over + under,
        "wrongdir": wrongdir,
        "over": over,
        "under": under,
    }
    counts["total"] = counts["extra"] + counts["missing"] + counts["reversed"]
    return counts


def scores(marks: np.ndarray, dag_marks: np.ndarray, cpdag_marks: np.ndarray, target: int) -> dict:
    """Both readings' counts of the answer at ``target``: ``{"dag": ..., "cpdag": ...}``."""
    return {
        "dag": dag_counts(marks, dag_marks, target),
        "cpdag": cpdag_counts(marks, cpdag_marks, target),
    }


def draw(network: str, n: int, seed: int) -> Draw:
    """``n`` rows of ``network`` with the columns shuffled by ``seed``, and the truth.

    ``alarm`` is :func:`andrey.data.load_dataset`'s, already shuffled. The chessboard
    (:data:`CHESSBOARD`) and any other published network are drawn in the file's order and their
    columns permuted by ``numpy.random.default_rng([seed, 99])``, the truth and the labels with
    them. Every column is a target, but the chessboard's.
    """
    if network == "alarm":
        from andrey.data import load_dataset

        ds = load_dataset("alarm", n=n, seed=seed, discrete=True, as_frame=False)
        marks = ds.graph.to_numpy()
        dag = ((marks == _TAIL) & (marks.T == _ARROW)).astype(np.int8)
        return Draw(ds.data, dag, list(ds.graph.labels), list(range(dag.shape[0])))
    if network == CHESSBOARD:
        dag = chessboard.adjacency()
        data, labels = chessboard.sample(dag, n, seed), chessboard.labels()
        targets: list[int] | None = chessboard.targets()
    else:
        from andrey_bench import networks

        net = networks.load_network(network)
        data, dag, labels, targets = net.sample(n, seed), net.adjacency, list(net.nodes), None
    permutation = np.random.default_rng([seed, 99]).permutation(dag.shape[0])
    position = np.argsort(permutation)
    if targets is None:
        targets = list(range(dag.shape[0]))
    return Draw(
        data[:, permutation],
        dag[np.ix_(permutation, permutation)],
        [labels[p] for p in permutation],
        sorted(int(position[t]) for t in targets),
    )


def truth(network: str, seed: int) -> Draw:
    """The truth of ``network`` at ``seed``: :func:`draw` of two rows, which every ``n`` shares."""
    return draw(network, 2, seed)
