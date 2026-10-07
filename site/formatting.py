"""Number formats shared by the homepage, the launch material, and the live demo."""

from __future__ import annotations


def ratio_text(ratio: float) -> str:
    """Format a speed ratio as the docs do: 941x, 15x, 4.3x, 0.62x."""
    if round(ratio, 1) >= 10:
        return f"{ratio:,.0f}x"
    return f"{ratio:.1f}x" if round(ratio, 2) >= 1 else f"{ratio:.2f}x"


def seconds_text(seconds: float | None) -> str:
    """Format a median fit time: 617 s, 50.8 s, 0.82 s; a missing time ran past its cap."""
    if seconds is None:
        return "past cap"
    if seconds >= 100:
        return f"{seconds:,.0f} s"
    return f"{seconds:.3g} s" if seconds >= 1 else f"{seconds:.2g} s"


def number_text(value: float | None) -> str:
    """Format an SHD, a mean or median over datasets: 1,473, 792.5, 0.67."""
    if value is None:
        return "-"
    return f"{value:,.0f}" if value == int(value) else f"{value:,.2f}".rstrip("0")
