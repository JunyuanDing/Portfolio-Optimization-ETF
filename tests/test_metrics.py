import pytest

pd = pytest.importorskip("pandas")

from portfolio.metrics import effective_number_of_assets, herfindahl_index


def test_concentration_metrics_normalize_within_risky_sleeve():
    weights = pd.Series({"SPY": 0.05, "TLT": 0.05, "BIL": 0.90})

    assert herfindahl_index(weights, cash_ticker="BIL") == pytest.approx(0.5)
    assert effective_number_of_assets(weights, cash_ticker="BIL") == pytest.approx(2.0)


def test_concentration_metrics_are_nan_when_fully_cash():
    weights = pd.Series({"SPY": 0.0, "TLT": 0.0, "BIL": 1.0})

    assert pd.isna(herfindahl_index(weights, cash_ticker="BIL"))
    assert pd.isna(effective_number_of_assets(weights, cash_ticker="BIL"))
