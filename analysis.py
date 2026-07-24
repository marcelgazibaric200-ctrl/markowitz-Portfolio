"""Number formatting helpers for the analysis view.

German formatting to match the reference (1.234,56). Kept separate from the
Streamlit layer so it stays unit-testable.
"""

from __future__ import annotations


def _de(number: float, decimals: int = 2) -> str:
    """Format a number with German separators: 1234567.89 -> '1.234.567,89'."""
    formatted = f"{number:,.{decimals}f}"  # e.g. '1,234,567.89'
    return formatted.replace(",", "\x01").replace(".", ",").replace("\x01", ".")


def fmt_usd(value: float | None) -> str:
    """Compact USD formatting with German units (Mio, Mrd, Bio)."""
    if value is None:
        return "n/a"
    magnitude = abs(value)
    if magnitude >= 1e12:
        return f"{_de(value / 1e12)} Bio $"
    if magnitude >= 1e9:
        return f"{_de(value / 1e9)} Mrd $"
    if magnitude >= 1e6:
        return f"{_de(value / 1e6)} Mio $"
    if magnitude >= 1000:
        return f"{_de(value, 0)} $"
    return f"{_de(value)} $"


def fmt_pct(fraction: float | None, signed: bool = False) -> str:
    """Format a fraction as a percentage: 0.382 -> '38,2 %'."""
    if fraction is None:
        return "n/a"
    sign = "+" if (signed and fraction >= 0) else ""
    return f"{sign}{_de(fraction * 100, 1)} %"
