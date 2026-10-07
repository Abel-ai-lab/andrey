"""Fit adapters; imports, conversion, and scoring stay outside the timed call."""

from __future__ import annotations

import numpy as np

from andrey import GraphStructure

PACKAGES = {
    "PC": ("Andrey", "causal-learn", "gCastle"),
    "DirectLiNGAM": ("Andrey", "causal-learn", "lingam"),
    "FCI": ("Andrey", "causal-learn"),
}
KINDS = {"PC": "cpdag", "DirectLiNGAM": "dag", "GES": "cpdag", "FCI": "pag"}


def signed_marks(matrix):
    """Map causal-learn's marks without changing endpoint orientation."""
    matrix = np.asarray(matrix)
    if not np.isin(matrix, [-1, 0, 1, 2]).all():
        raise ValueError("Unsupported endpoint code.")
    marks = np.zeros(matrix.shape, dtype=np.uint8)
    for old, new in [(-1, 1), (1, 2), (2, 3)]:
        marks[matrix == old] = new
    return marks


def adjacency_marks(matrix):
    """Convert row-source adjacency to Andrey endpoint marks."""
    adjacency = np.asarray(matrix) != 0
    marks = np.zeros(adjacency.shape, dtype=np.uint8)
    marks[adjacency] = 1
    marks[(adjacency & ~adjacency.T).T] = 2
    np.fill_diagonal(marks, 0)
    return marks


def prepare(package, method, data):
    """Return a zero-argument fit and a separate result converter."""
    if package == "Andrey":
        import andrey
        from andrey.constraint.fci import fci  # noqa: F401
        from andrey.constraint.pc import pc  # noqa: F401
        from andrey.lingam.direct import direct_lingam  # noqa: F401
        from andrey.search.ges import ges  # noqa: F401

        functions = {
            "PC": lambda: andrey.pc(data, alpha=0.05, indep_test="fisherz"),
            "DirectLiNGAM": lambda: andrey.direct_lingam(data, measure="pwling"),
            "GES": lambda: andrey.ges(data),
            "FCI": lambda: andrey.fci(data, alpha=0.05),
        }
        return functions[method], lambda result: result.structure.to_numpy()
    if package == "causal-learn" and method == "PC":
        from causallearn.search.ConstraintBased.PC import pc

        return (
            lambda: pc(data, alpha=0.05, indep_test="fisherz", stable=True, show_progress=False),
            lambda result: signed_marks(result.G.graph),
        )
    if package == "causal-learn" and method == "FCI":
        from causallearn.search.ConstraintBased.FCI import fci

        return (
            lambda: fci(data, independence_test_method="fisherz", alpha=0.05, show_progress=False),
            lambda result: signed_marks(result[0].graph),
        )
    if package == "gCastle" and method == "PC":
        from castle.algorithms import PC

        model = PC(alpha=0.05, ci_test="fisherz", variant="stable")
        return lambda: model.learn(data), lambda _: adjacency_marks(model.causal_matrix)
    if method == "DirectLiNGAM":
        if package == "causal-learn":
            from causallearn.search.FCMBased.lingam import DirectLiNGAM
        elif package == "lingam":
            from lingam import DirectLiNGAM
        else:
            raise ValueError("Unknown package.")
        model = DirectLiNGAM(measure="pwling", random_state=0)
        return (
            lambda: model.fit(data),
            lambda _: adjacency_marks(model.adjacency_matrix_.T),
        )
    raise ValueError("Unsupported package and method.")


def structure(marks, method, labels=None):
    """Validate a worker's graph and attach the data's column labels."""
    return GraphStructure.from_numpy(
        np.asarray(marks, dtype=np.uint8), kind=KINDS[method], labels=labels
    )
