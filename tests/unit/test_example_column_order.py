"""The example notebooks' hand-built data list their columns in an order drawn from the seed.

Each hand-built example draws its data with a function of the seed that takes ``shuffle`` and names
its true edges, cause first, in a constant; hidden causes are names with no column. The test runs
only those definitions, not the notebook. Notebooks that draw from ``andrey.data`` take its column
shuffle, which ``test_data_scm.py`` checks.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from andrey.data.sortability import index_correlation

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"

#: Each notebook's hand-built examples: the function that draws one, and the name of its edges.
DRAWS = {
    "bottom_up_parce_lingam": {"sample": "EDGES"},
    "constraint_based": {
        "sample_chain": "CHAIN",
        "sample_hidden": "HIDDEN",
        "sample_domains": "DOMAINS",
    },
    "ges_quickstart": {"sample": "CHAIN"},
    "lingam": {"sample_groups": "CHAIN", "sample_indicators": "INDICATORS"},
    "rcd": {"sample": "EDGES"},
    "score_search": {"sample_intervened": "CHAIN"},
    "time_series": {"sample_panel": "CHAIN"},
    "viz": {"sample": "EDGES"},
}
SEEDS = range(200)


def _definitions(notebook: str) -> dict:
    """Run the notebook's function definitions and its edges constants, nothing else."""
    namespace: dict = {"np": np, "pd": pd}
    cells = json.loads((EXAMPLES / f"{notebook}.ipynb").read_text())["cells"]
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        lines = "".join(cell["source"]).splitlines()
        tree = ast.parse("\n".join(line for line in lines if not line.lstrip().startswith("!")))
        keep = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) or _assigns(node, set(DRAWS[notebook].values()))
        ]
        exec(compile(ast.Module(body=keep, type_ignores=[]), notebook, "exec"), namespace)
    return namespace


def _assigns(node: ast.stmt, names: set[str]) -> bool:
    """Whether ``node`` assigns, or adds to, one of ``names``."""
    targets = getattr(node, "targets", None) or [getattr(node, "target", None)]
    return isinstance(node, (ast.Assign, ast.AugAssign)) and all(
        isinstance(t, ast.Name) and t.id in names for t in targets
    )


def _correlation(frame: pd.DataFrame, edges: list[tuple[str, str]]) -> tuple[float, int]:
    """The rank correlation of column index and causal depth, and the number of columns."""
    names = list(frame.columns)
    names += sorted({name for edge in edges for name in edge} - set(names))  # hidden causes last
    node = {name: j for j, name in enumerate(names)}
    pairs = np.array([(node[a], node[b]) for a, b in edges])
    return index_correlation(pairs, len(names), n_observed=frame.shape[1]), frame.shape[1]


def _frame(drawn) -> pd.DataFrame:
    """The data of a draw: the frame itself, or the first of several frames or of a pair."""
    return drawn if isinstance(drawn, pd.DataFrame) else drawn[0]


@pytest.mark.parametrize(
    ("notebook", "draw"), [(nb, draw) for nb, draws in DRAWS.items() for draw in draws]
)
def test_an_examples_column_order_says_nothing_of_the_causal_order(notebook, draw):
    """Across seeds the mean rank correlation of column index and causal depth is near zero.

    Under a uniform column order the correlation on ``k`` columns has mean 0 and variance
    ``1 / (k - 1)``, so over 200 seeds its mean stays within four standard deviations,
    ``4 / sqrt(200 (k - 1))``. The order the variables are written in, the positive control, is
    outside that bound.
    """
    namespace = _definitions(notebook)
    sample, edges = namespace[draw], namespace[DRAWS[notebook][draw]]
    drawn = [_correlation(_frame(sample(seed)), edges) for seed in SEEDS]
    k = drawn[0][1]
    bound = 4 / np.sqrt(len(SEEDS) * (k - 1))
    assert abs(np.mean([rho for rho, _ in drawn])) < bound
    assert abs(_correlation(_frame(sample(0, shuffle=False)), edges)[0]) > bound


def test_every_frame_of_a_draw_lists_one_column_order():
    """Groups, panels, and an intervention mask follow the same order as the first frame."""
    groups = _definitions("lingam")["sample_groups"](1)
    assert all(list(g.columns) == list(groups[0].columns) for g in groups)
    panel = _definitions("time_series")["sample_panel"](1)
    assert all(list(p.columns) == list(panel[0].columns) for p in panel)
    frame, intervened = _definitions("score_search")["sample_intervened"](1)
    assert intervened[:, frame.columns.get_loc("x1")].sum() == 1000 == intervened.sum()
