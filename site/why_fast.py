"""The "Why Andrey is fast" scenes: one animation per idea behind Andrey's speed.

Each scene runs the same illustrative work in two lanes: Andrey's, and a quieter lane for other
packages, which are never named. The scenes appear under the headings of the blog post "Why Andrey
is fast". The markup is complete without script: each counter already shows its lane's final count.
``site/assets/why-fast.js`` plays the scenes with Motion; they are illustrations of the ideas, not
measurements.
"""

from __future__ import annotations

from html import escape

# Each scene: its title and narration, its one-line summary for places without the animation (the
# README), what each lane does, the counter's unit, and the illustration's size, which the script
# reads from the markup and which fixes the final counts.
SCENES = (
    dict(
        key="parallel",
        title="Do many things at once",
        text="Within one round of PC, each independence test stands alone, and within one pass of "
        "GES, so does each candidate move. Andrey runs them as one batched array operation, "
        "thousands at a time, with the same results as one at a time.",
        line="thousands of independence tests run as one array operation, with the same results.",
        andrey="one call for the batch",
        others="one call per test",
        unit=("call", "calls"),
        size={"n": 8},
    ),
    dict(
        key="reuse",
        title="Never compute twice",
        text="Andrey computes the correlation matrix once, up front, and caches it. Each test "
        "reads the few entries it needs instead of recomputing them, and GES memoizes its local "
        "scores the same way.",
        line="the correlation matrix is computed once and cached; GES memoizes its local scores.",
        andrey="computed once, then read",
        others="recomputed for each test",
        unit=("computation", "computations"),
        size={"n": 6},
    ),
    dict(
        key="skip",
        title="Skip what can't matter",
        text="Tests given zero or one other variable have a closed form, so a quick pass settles "
        "most edges before any matrix is inverted. GES rules out invalid candidates with bit masks "
        "before scoring them.",
        line="tests with a closed form settle most edges before any matrix is inverted.",
        andrey="a quick check first",
        others="every test inverts a matrix",
        unit=("inversion", "inversions"),
        size={"n": 10, "costly": (2, 6, 9)},
    ),
    dict(
        key="hardware",
        title="Use the hardware you have",
        text="With the optional PyTorch extra, a CUDA GPU computes large entropy tables many cells "
        "at a time, once they are large enough to repay the transfer.",
        line="with the optional PyTorch extra, a GPU computes large entropy tables.",
        andrey="a GPU, many cells per step",
        others="one core, one cell per step",
        unit=("step", "steps"),
        size={"rows": 4, "cols": 12, "waves": 3},
    ),
)
BY_KEY = {scene["key"]: scene for scene in SCENES}


def icon(name: str, body: str) -> str:
    return f'<svg class="i-{name}" viewBox="0 0 16 16" aria-hidden="true">{body}</svg>'


STROKE = 'fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"'
CHECK = icon("check", f'<path d="M4 8.4l2.6 2.6L12 5.4" stroke-width="2" {STROKE}/>')
REPLAY = icon(
    "replay",
    f'<path d="M3.2 8a4.8 4.8 0 1 0 1.5-3.5" stroke-width="1.8" {STROKE}/>'
    f'<path d="M4.4 1.9v2.9h2.9" stroke-width="1.8" {STROKE}/>',
)
PLUS = icon("plus", f'<path d="M8 2v12M2 8h12" stroke-width="2.2" {STROKE}/>')
CLOSE = icon("close", f'<path d="M3 3l10 10M13 3 3 13" stroke-width="2" {STROKE}/>')


def final_counts(scene: dict) -> tuple[int, int]:
    """Each lane's count when its scene ends: Andrey's, then the other packages'."""
    size = scene["size"]
    return {
        "parallel": lambda: (1, size["n"]),
        "reuse": lambda: (1, size["n"]),
        "skip": lambda: (len(size["costly"]), size["n"]),
        "hardware": lambda: (size["waves"], size["rows"] * size["cols"]),
    }[scene["key"]]()


def unit(scene: dict, count: int) -> str:
    return scene["unit"][0] if count == 1 else scene["unit"][1]


def lane(scene: dict, who: str, name: str, count: int) -> str:
    return (
        f'<div class="wf-lane {who}"><p class="wf-who">{name}<small>{escape(scene[who])}</small>'
        '</p><div class="wf-track"></div>'
        f'<p class="wf-count"><span class="wf-done">{CHECK}</span><b>{count}</b>'
        f"<small>{unit(scene, count)}</small></p></div>"
    )


def stage(scene: dict) -> str:
    """Both lanes at their final counts, and the same counts as a sentence for screen readers."""
    andrey, others = final_counts(scene)
    summary = (
        f"Andrey: {andrey} {unit(scene, andrey)}, {scene['andrey']}. "
        f"Others: {others} {unit(scene, others)}, {scene['others']}."
    )
    return (
        '<div class="wf-stage"><div class="wf-lanes" aria-hidden="true">'
        f"{lane(scene, 'andrey', 'Andrey', andrey)}{lane(scene, 'others', 'Others', others)}"
        f'</div></div><p class="wf-sr">{escape(summary)}</p>'
    )


def figure(scene: dict) -> str:
    """One scene's stage: it plays once when it scrolls into view, then offers a replay."""
    size = " ".join(
        f'data-{name}="{" ".join(map(str, value)) if isinstance(value, tuple) else value}"'
        for name, value in scene["size"].items()
    )
    replay = (
        '<div class="wf-after"><button class="wf-replay" type="button" aria-label="Replay" hidden>'
        f"{REPLAY}</button></div>"
    )
    return (
        f'<figure class="wf wf-scene" data-scene="{scene["key"]}" data-play="view" {size} '
        f'data-unit="{" ".join(scene["unit"])}">{stage(scene)}{replay}</figure>'
    )


def fragment(key: str) -> str:
    """A scene for the "Why Andrey is fast" post, under a heading of the page's own."""
    return figure(BY_KEY[key])
