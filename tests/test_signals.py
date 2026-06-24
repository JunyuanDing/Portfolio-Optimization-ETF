import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from portfolio.signals import (
    SignalConfig,
    alpha_from_scores,
    compute_signal_scores,
    cross_sectional_zscore_ex_cash,
    momentum_12_1,
    trend_signal,
)


def synthetic_prices(periods=320):
    dates = pd.bdate_range("2020-01-01", periods=periods)
    base = pd.Series(np.arange(1, periods + 1, dtype=float), index=dates)
    prices = pd.DataFrame(
        {
            "SPY": 100 + base,
            "TLT": 100 + 0.5 * base,
            "GLD": 120 + 0.2 * base,
            "BIL": 100 + 0.01 * base,
        },
        index=dates,
    )
    return prices


def test_momentum_12_1_skips_most_recent_month():
    prices = synthetic_prices(300)
    momentum = momentum_12_1(prices, lookback_days=252, skip_days=21)
    date = prices.index[-1]
    expected = prices.loc[prices.index[-22], "SPY"] / prices.loc[prices.index[-253], "SPY"] - 1.0
    assert momentum.loc[date, "SPY"] == pytest.approx(expected)


def test_trend_signal_is_binary_not_zscored():
    prices = synthetic_prices(260)
    trend = trend_signal(prices, window_days=200)
    valid = trend.dropna()
    assert set(valid.stack().unique()).issubset({-1.0, 1.0})


def test_cash_is_excluded_from_cross_sectional_zscore():
    values = pd.DataFrame(
        {"SPY": [1.0], "TLT": [2.0], "GLD": [3.0], "BIL": [999.0]},
        index=[pd.Timestamp("2021-01-01")],
    )
    zscores = cross_sectional_zscore_ex_cash(values, "BIL")
    assert zscores.loc[values.index[0], "BIL"] == 0.0
    assert zscores[["SPY", "TLT", "GLD"]].mean(axis=1).iloc[0] == pytest.approx(0.0)


def test_alpha_scaling_and_cash_alpha_zero():
    scores = pd.Series({"SPY": 1.5, "TLT": -0.5, "BIL": 10.0})
    alpha = alpha_from_scores(scores, alpha_scale=0.03, cash_ticker="BIL")
    assert alpha.loc["SPY"] == pytest.approx(0.045)
    assert alpha.loc["TLT"] == pytest.approx(-0.015)
    assert "BIL" not in alpha.index


def test_combined_score_excludes_cash_column():
    prices = synthetic_prices(320)
    scores = compute_signal_scores(prices, SignalConfig(cash_ticker="BIL"))
    assert "BIL" not in scores.columns
    assert set(scores.columns) == {"SPY", "TLT", "GLD"}
