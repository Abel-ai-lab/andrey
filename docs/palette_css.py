"""Render an ``andrey.viz`` palette as CSS custom properties, for the docs and the homepage.

Each palette field becomes ``--andrey-<field>``; tuple fields are numbered from 1
(``--andrey-ramp-1`` ... ``--andrey-ramp-5``). The stylesheets name these properties, never a hex.
"""

from __future__ import annotations

import dataclasses

import andrey.viz


def palette_block(selector: str, palette: andrey.viz.Palette) -> str:
    """Render one theme's palette as a CSS rule of ``--andrey-*`` custom properties."""
    lines = [f"{selector} {{"]
    for field in dataclasses.fields(palette):
        value = getattr(palette, field.name)
        name = field.name.replace("_", "-")
        if isinstance(value, tuple):
            lines += [f"  --andrey-{name}-{i}: {v};" for i, v in enumerate(value, 1)]
        else:
            lines.append(f"  --andrey-{name}: {value};")
    lines.append("}")
    return "\n".join(lines)
