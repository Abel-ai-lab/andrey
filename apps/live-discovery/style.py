"""Gradio theme and CSS derived from Andrey's active palette."""

import dataclasses

import gradio as gr

from andrey.viz.theme import DARK, LIGHT


def theme():
    """Use the cobalt ramp for controls and explicit light/dark semantic tokens."""
    ramp = LIGHT.ramp
    cobalt = gr.themes.Color(
        name="andrey",
        **dict(
            zip(
                [f"c{n}" for n in (50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950)],
                [
                    ramp[0],
                    ramp[0],
                    ramp[1],
                    ramp[1],
                    ramp[2],
                    ramp[3],
                    ramp[3],
                    ramp[4],
                    ramp[4],
                    ramp[4],
                    ramp[4],
                ],
            )
        ),
    )
    selected = gr.themes.Base(
        primary_hue=cobalt,
        secondary_hue=cobalt,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("DM Mono"), "monospace"],
    )
    values = {}
    for suffix, palette in (("", LIGHT), ("_dark", DARK)):
        for token, color in {
            "body_background_fill": palette.paper,
            "body_text_color": palette.ink,
            "body_text_color_subdued": palette.muted,
            "block_background_fill": palette.surface,
            "block_border_color": palette.line,
            "input_background_fill": palette.surface,
            "input_border_color": palette.line,
            "button_primary_background_fill": palette.accent,
            "button_primary_background_fill_hover": palette.ramp[4],
            "button_primary_text_color": LIGHT.surface if not suffix else DARK.paper,
            "button_secondary_background_fill": palette.surface_2,
            "button_secondary_text_color": palette.ink,
        }.items():
            values[token + suffix] = color
    return selected.set(**values)


def css():
    """Map graph and result colors to the same active Gradio theme."""
    blocks = []
    for selector, palette in (
        (":root,.gradio-container", LIGHT),
        (".dark,.dark .gradio-container", DARK),
    ):
        tokens = ";".join(
            f"--andrey-{f.name.replace('_', '-')}:{getattr(palette, f.name)}"
            for f in dataclasses.fields(palette)
            if isinstance(getattr(palette, f.name), str)
        )
        blocks.append(f"{selector}{{{tokens}}}")
    return (
        "\n".join(blocks)
        + """
    .gradio-container{max-width:1280px!important;margin:auto!important}
    #hero{padding:28px 0 22px}#hero h1{font-size:clamp(32px,5vw,54px);line-height:1.1;
    font-weight:600;letter-spacing:-.04em;margin:24px 0 14px;max-width:20ch}
    #hero p{font-size:18px;max-width:65ch;color:var(--andrey-muted)}
    .wordmark svg{width:150px;height:auto}.wordmark-dark{display:none}
    .dark .wordmark-light{display:none}.dark .wordmark-dark{display:block}
    .live-label{padding:12px 16px;border-left:3px solid var(--andrey-accent);
    background:var(--andrey-surface-2);color:var(--andrey-muted)}
    .race-table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
    .race-table th,.race-table td{text-align:right;padding:13px 10px;
    border:0;border-bottom:1px solid var(--andrey-line)}.race-table th:first-child,
    .race-table td:first-child{text-align:left}.race-table .ours td,
    .race-table .ours strong{color:var(--andrey-accent)}
    .table-scroll{overflow-x:auto}.graph-pair{padding:18px 0;max-width:900px}
    .graph-pair svg{max-height:480px}.quiet{color:var(--andrey-muted)}
    .graph-columns{display:grid;grid-template-columns:1fr 1fr;gap:24px}
    .graph-columns svg{width:100%;height:auto;max-height:480px}
    @media(max-width:640px){.graph-columns{grid-template-columns:1fr}}
    .bar-track{height:6px;background:var(--andrey-surface-2);margin-top:8px}
    .bar-fill{height:100%;background:var(--andrey-muted)}
    .ours .bar-fill{background:var(--andrey-accent)}
    @media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
    """
    )
