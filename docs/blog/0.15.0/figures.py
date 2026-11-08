"""The 0.15.0 post's figures, written next to it as HTML that the post includes.

- ``environments.html``, after Figure 1 of Huang et al. (2020): a chain ``V1 -> V2 -> V3 -> V4``
  whose environments shift ``V2`` and ``V4`` through one shared change ``g``:
  ``V2 = 0.8 V1 + g + e2`` and ``V4 = 0.8 V3 - 2 g + e4``, the other variables as in the chain,
  Gaussian noise of standard deviation 0.6, four environments of 300 rows with ``g`` at four evenly
  spaced values in ``[-1.5, 1.5]``. It draws the data's graph with the environment as ``C``, PC on
  one environment, PC on the pooled rows, and CD-NOD with the index (the conditional Gaussian test
  reading the environments as categories). On seeds 0 to 19, PC on the pooled rows drew the edges
  ``V1 - V4`` and ``V2 - V4`` every time, and CD-NOD's graph over the four variables was the chain
  every time, its index adjacent to exactly ``V2`` and ``V4`` on 16.
- ``contexts.html``, after Figure 1 of Mooij, Magliacane, and Claassen (2020): ``X1``, hours of
  violent games, and ``X2``, aggressive behavior, with ``C``, a ban on violent games, in force in
  half of 2,000 rows, lowering ``X1`` by 1. Either ``X1`` causes ``X2`` (``X2 = 0.8 X1 + e``) or a
  hidden ``U`` causes both and ``X1`` does not cause ``X2``; noise of standard deviation 0.7. FCI on
  the rows without the ban returns the same PAG for both; FCI-JCI with ``C`` (set from outside,
  ``confounded_index=False``) tells them apart. On seeds 0 to 19 FCI-JCI returned ``X1 -> X2`` for
  the first on 18 and ``X2 o-> X1`` for the second on 19.
- ``latent.html``, after Section 4 of Ramsey, Andrews, and Spirtes (2025): ``X -> Y <- L -> Z <- W``
  with ``L`` hidden, coefficients 0.8, noise of standard deviation 0.6, 2,000 rows. BOSS's CPDAG
  over the measured variables has one adjacency more than the PAG, ``X - Z`` or ``W - Y``, to cover
  the collider at ``Y`` or ``Z``; BOSS-FCI's test of the pair removes it. On seeds 0 to 19 BOSS
  added one of the two every time, and BOSS-FCI returned the true PAG every time.
- ``sachs.html``: Sachs's 11 proteins and phospholipids under nine conditions, the consensus
  network beside the graph PC learns without ``env`` and the one CD-NOD learns with it, over the
  same node positions, an edge between a pair the consensus network does not join highlighted.

Every figure draws seed 0. Every color is an ``--andrey-*`` token, so the figures follow the page's
light or dark theme. Run with ``uv run python docs/blog/0.15.0/figures.py``.

References
----------
Huang, B., Zhang, K., Zhang, J., Ramsey, J., Sanchez-Romero, R., Glymour, C., and Schölkopf, B.
(2020). Causal discovery from heterogeneous/nonstationary data. JMLR 21(89), 1-53.

Mooij, J. M., Magliacane, S., and Claassen, T. (2020). Joint causal inference from multiple
contexts. JMLR 21(99), 1-108.

Ramsey, J., Andrews, B., and Spirtes, P. (2025). Efficient latent variable causal discovery:
combining score search and targeted testing. arXiv:2510.04263.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

import andrey
from andrey import GraphStructure
from andrey.core import ARROW, TAIL
from andrey.data import load_dataset
from andrey.viz import LATENT, LIGHT, Palette, draw

HERE = Path(__file__).parent
SEED = 0
PALETTE = Palette(
    **{
        field.name: (
            getattr(LIGHT, field.name)
            if isinstance(getattr(LIGHT, field.name), tuple)
            else f"var(--andrey-{field.name.replace('_', '-')})"
        )
        for field in dataclasses.fields(LIGHT)
    }
)
STYLE = (
    "<style>"
    ".pf-figure .pf-row{display:flex;flex-wrap:wrap;gap:8px 12px;justify-content:center}"
    ".pf-figure .pf-panel{flex:1 1 140px}"
    ".pf-figure .pf-panel svg{width:100%;height:auto;display:block}"
    ".pf-figure .pf-title{font-family:var(--andrey-font-sans);font-size:12px;font-weight:600;"
    "text-align:center;margin:0;color:var(--andrey-ink)}"
    "</style>"
)


def directed(n: int, edges: list[tuple[int, int]]) -> np.ndarray:
    """The marks of the DAG on ``n`` nodes with ``edges``; ``M[i, j]`` is the mark at ``i``."""
    marks = np.zeros((n, n), dtype=np.int8)
    for i, j in edges:
        marks[i, j], marks[j, i] = TAIL, ARROW
    return marks


def grown(marks: np.ndarray, n: int) -> np.ndarray:
    """``marks`` on its nodes, padded with nodes that have no edges up to ``n``."""
    full = np.zeros((n, n), dtype=marks.dtype)
    full[: len(marks), : len(marks)] = marks
    return full


def with_index(marks: np.ndarray, index: int, index_edges: list) -> np.ndarray:
    """``marks`` with an edge from node ``index`` into each node a result's ``index_edges`` names.

    The index edges here are ``directed`` (CD-NOD) or come from FCI-JCI with
    ``confounded_index=False``, which points every edge the index keeps out of it.
    """
    full = marks.copy()
    for variable, kind in index_edges:
        assert kind == "directed", "an index set from outside only causes"
        full[index, variable], full[variable, index] = TAIL, ARROW
    return full


def panel(
    title: str,
    marks: np.ndarray,
    kind: str,
    labels: tuple[str, ...],
    positions: dict,
    size: tuple[int, int],
    hidden: tuple[int, ...] = (),
    highlight: list | None = None,
) -> str:
    """One titled graph; nodes in ``hidden`` are latent, edges in ``highlight`` lit at both ends."""
    types = np.array([1 if k in hidden else 0 for k in range(len(labels))]) * LATENT
    structure = GraphStructure.from_numpy(marks, kind=kind, labels=labels, node_types=types)
    drawn = draw(
        structure,
        palette=PALETTE,
        positions=positions,
        background=False,
        width=size[0],
        height=size[1],
        highlight=[end for i, j in highlight or [] for end in ((i, j), (j, i))],
    )
    svg = str(drawn).split("?>", 1)[-1].strip()
    return f'<div class="pf-panel"><p class="pf-title">{title}</p>{svg}</div>'


def figure(rows: list[list[str]], caption: str, width: int) -> str:
    """Panels in rows, each row wrapping on narrow pages, with a caption."""
    body = "".join(f'<div class="pf-row">{"".join(row)}</div>' for row in rows)
    return (
        f'<figure class="pf-figure" style="max-width:{width}px;margin-left:auto;'
        f'margin-right:auto">{body}<figcaption>{caption}</figcaption></figure>\n'
    )


def extra(marks: np.ndarray, truth: np.ndarray) -> list[tuple[int, int]]:
    """The adjacencies of ``marks`` that ``truth`` does not have."""
    n = len(marks)
    return [(i, j) for i in range(n) for j in range(i + 1, n) if marks[i, j] and not truth[i, j]]


def environments() -> str:
    """``environments.html``: the environment index removes edges and orients the rest."""
    rng = np.random.default_rng(SEED)
    parts = []
    for g in rng.permutation(np.linspace(-1.5, 1.5, 4)):
        v1 = rng.normal(size=300)
        v2 = 0.8 * v1 + g + 0.6 * rng.normal(size=300)
        v3 = 0.8 * v2 + 0.6 * rng.normal(size=300)
        v4 = 0.8 * v3 - 2 * g + 0.6 * rng.normal(size=300)
        parts.append(np.column_stack([v1, v2, v3, v4]))
    data, env = np.vstack(parts), np.repeat(np.arange(4), 300)
    labels = ("V1", "V2", "V3", "V4", "C")
    # The chain along the bottom, C above it; every panel draws C, so all share one scale.
    positions = {0: (0.0, 0.8), 1: (0.33, 0.8), 2: (0.67, 0.8), 3: (1.0, 0.8), 4: (0.67, 0.25)}
    truth = directed(5, [(0, 1), (1, 2), (2, 3), (4, 1), (4, 3)])
    cdnod = andrey.cdnod(data, env, indep_test="cond_gauss")
    one = grown(andrey.pc(data[env == 0]).structure.adjacency, 5)
    pooled = grown(andrey.pc(data).structure.adjacency, 5)
    learned = with_index(grown(cdnod.structure.adjacency, 5), 4, cdnod.metadata["index_edges"])

    def show(title: str, marks: np.ndarray, hidden: tuple[int, ...] = ()) -> str:
        wrong = extra(marks, truth)
        return panel(title, marks, "cpdag", labels, positions, (190, 140), hidden, wrong)

    return figure(
        [
            [
                show("(a) The data's graph", truth),
                show("(b) PC, one environment", one, hidden=(4,)),
                show("(c) PC, pooled", pooled, hidden=(4,)),
                show("(d) CD-NOD, with C", learned),
            ]
        ],
        "A chain whose environments shift V2 and V4 through one shared change, after Figure 1 of "
        "Huang et al. (2020). C is the environment, dashed where the method is not given it. In "
        "(c), the edges between pairs that (a) does not join are highlighted.",
        800,
    )


def contexts() -> str:
    """``contexts.html``: a context variable tells a cause from a hidden common cause."""
    labels = ("X1", "X2", "C", "U")
    # C, X1, X2 along the bottom, U above X1 and X2 in the second row. At 170 x 130 units both rows
    # draw at the scale their width allows, so the nodes keep their spacing.
    positions = {2: (0.0, 0.8), 0: (0.5, 0.8), 1: (1.0, 0.8), 3: (0.75, 0.2)}
    rows = []
    for confounded in (False, True):
        rng = np.random.default_rng(SEED)
        env = np.repeat([0, 1], 1000)
        u = rng.normal(size=2000)
        x1 = -1.0 * env + (u if confounded else 0) + 0.7 * rng.normal(size=2000)
        x2 = (u if confounded else 0.8 * x1) + 0.7 * rng.normal(size=2000)
        data = np.column_stack([x1, x2])
        n = 4 if confounded else 3  # the second row draws the hidden U
        fci = grown(andrey.fci(data[env == 0]).structure.adjacency, n)
        jci = andrey.fci_jci(data, env, confounded_index=False)
        learned = with_index(grown(jci.structure.adjacency, n), 2, jci.metadata["index_edges"])
        if confounded:
            truth, name, tags = directed(4, [(2, 0), (3, 0), (3, 1)]), "U causes both", "def"
        else:
            truth, name, tags = directed(3, [(2, 0), (0, 1)]), "X1 causes X2", "abc"
        place = {k: positions[k] for k in range(n)}
        u_hidden = (3,) if confounded else ()

        def show(title: str, marks: np.ndarray, kind: str, hidden: tuple[int, ...]) -> str:
            return panel(title, marks, kind, labels[:n], place, (170, 130), hidden)

        rows.append(
            [
                show(f"({tags[0]}) Truth: {name}", truth, "dag", u_hidden),
                show(f"({tags[1]}) FCI, no ban", fci, "pag", (2, *u_hidden)),
                show(f"({tags[2]}) FCI-JCI, with C", learned, "pag", u_hidden),
            ]
        )
    return figure(
        rows,
        "After Figure 1 of Mooij, Magliacane, and Claassen (2020): X1 is hours of violent games, "
        "X2 aggressive behavior, and C a ban on violent games. Without the ban, FCI returns the "
        "same PAG in both rows. With C, FCI-JCI finds that X1 causes X2 in (c) and that it does "
        "not in (f). Dashed nodes are not given to the method.",
        620,
    )


def latent() -> str:
    """``latent.html``: a DAG search adds an adjacency for a hidden cause; a test removes it."""
    rng = np.random.default_rng(SEED)
    x, w, hidden = (rng.normal(size=2000) for _ in range(3))
    y = 0.8 * x + 0.8 * hidden + 0.6 * rng.normal(size=2000)
    z = 0.8 * hidden + 0.8 * w + 0.6 * rng.normal(size=2000)
    data = np.column_stack([x, y, z, w])
    labels = ("X", "Y", "Z", "W", "L")
    positions = {0: (0.0, 0.8), 1: (0.33, 0.8), 2: (0.67, 0.8), 3: (1.0, 0.8), 4: (0.5, 0.2)}
    truth = directed(5, [(0, 1), (4, 1), (4, 2), (3, 2)])
    # The adjacencies of the true PAG over the measured variables: Y and Z share L, so they are
    # adjacent there, and X and Z, or W and Y, are not.
    adjacent = directed(5, [(0, 1), (1, 2), (3, 2)])
    boss = grown(andrey.boss(data).structure.adjacency, 5)
    pag = grown(andrey.boss_fci(data).structure.adjacency, 5)

    def show(title: str, marks: np.ndarray, kind: str) -> str:
        wrong = extra(marks, adjacent) if kind != "dag" else []
        return panel(title, marks, kind, labels, positions, (220, 160), (4,), wrong)

    return figure(
        [
            [
                show("(a) Truth, L hidden", truth, "dag"),
                show("(b) BOSS", boss, "cpdag"),
                show("(c) BOSS-FCI", pag, "pag"),
            ]
        ],
        "After Section 4 of Ramsey, Andrews, and Spirtes (2025). L is hidden. BOSS's CPDAG over "
        "the measured variables joins a pair that (a) does not (highlighted) to cover a collider; "
        "BOSS-FCI's test removes it and returns the PAG: Y and Z share a hidden common cause.",
        720,
    )


# Sachs's nodes in the unit square, y downward: the kinase cascade runs left to right below the
# phospholipids, with pkc and pka on the left as the regulators the consensus network has.
SACHS_PLACES = {
    "plc": (0.0, 0.0),
    "pip3": (0.2, 0.12),
    "pip2": (0.4, 0.0),
    "pkc": (0.0, 0.5),
    "pka": (0.2, 0.75),
    "raf": (0.35, 0.33),
    "mek": (0.6, 0.33),
    "erk": (0.8, 0.55),
    "akt": (1.0, 0.75),
    "jnk": (0.3, 1.0),
    "p38": (0.55, 1.0),
}


def sachs() -> str:
    """``sachs.html``: PC on the pooled conditions beside CD-NOD with ``env``."""
    loaded = load_dataset("sachs", conditions=True)
    data, env, consensus = loaded.data, loaded.env, loaded.graph
    labels = consensus.labels
    positions = {k: SACHS_PLACES[label] for k, label in enumerate(labels)}
    truth = consensus.adjacency

    def show(title: str, marks: np.ndarray, kind: str) -> str:
        wrong = extra(marks, truth)
        edges = int(np.count_nonzero(np.triu(marks + marks.T)))
        title = f"{title}: {edges} edges" + (f", {len(wrong)} not in (a)" if kind != "dag" else "")
        return panel(title, marks, kind, labels, positions, (300, 290), (), wrong)

    pooled = andrey.pc(data).structure.adjacency
    indexed = andrey.cdnod(data, env, indep_test="cond_gauss").structure.adjacency
    return figure(
        [
            [
                show("(a) Consensus network", truth, "dag"),
                show("(b) PC, pooled", pooled, "cpdag"),
                show("(c) CD-NOD, with env", indexed, "cpdag"),
            ]
        ],
        "Sachs's 11 proteins and phospholipids under nine conditions. An edge between a pair the "
        "consensus network does not join is highlighted. CD-NOD's index, adjacent to all 11 "
        "variables, is not drawn.",
        960,
    )


def main() -> None:
    for name, build in (
        ("environments", environments),
        ("contexts", contexts),
        ("latent", latent),
        ("sachs", sachs),
    ):
        (HERE / f"{name}.html").write_text(STYLE + build())


if __name__ == "__main__":
    main()
