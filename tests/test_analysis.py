"""Unit tests for the number formatting helpers."""

import analysis


def test_de_uses_german_separators():
    assert analysis._de(1234567.89) == "1.234.567,89"
    assert analysis._de(-1234.5) == "-1.234,50"


def test_fmt_usd_units():
    assert analysis.fmt_usd(62195) == "62.195 $"
    assert analysis.fmt_usd(7.8e6) == "7,80 Mio $"
    assert analysis.fmt_usd(1.248e12) == "1,25 Bio $"
    assert analysis.fmt_usd(None) == "n/a"


def test_fmt_pct():
    assert analysis.fmt_pct(0.382) == "38,2 %"
    assert analysis.fmt_pct(0.0123, signed=True) == "+1,2 %"
    assert analysis.fmt_pct(None) == "n/a"
