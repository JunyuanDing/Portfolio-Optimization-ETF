import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from portfolio.backtest import (
    BacktestConfig,
    StrategySpec,
    default_strategy_specs,
    monthly_rebalance_dates,
    run_backtest,
)
from portfolio.data import CASH_TICKER, RISKY_TICKERS
from portfolio.validation import assert_no_lookahead


def test_rebalance_dates_start_after_lookback():
    dates = pd.bdate_range("2020-01-01", periods=300)
    returns = pd.DataFrame({"A": 0.001, "B": 0.0}, index=dates)
    rebalance_dates = monthly_rebalance_dates(returns, lookback_days=100)
    assert min(rebalance_dates) >= returns.index[100]


def test_no_lookahead_guard_raises_on_bad_dates():
    with pytest.raises(ValueError):
        assert_no_lookahead(pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-02"))


def test_backtest_windows_are_separated_and_execution_is_out_of_sample():
    dates = pd.bdate_range("2020-01-01", periods=340)
    prices = pd.DataFrame(index=dates)
    for idx, ticker in enumerate(RISKY_TICKERS):
        returns = 0.0002 + 0.00001 * idx + 0.001 * np.sin(np.arange(len(dates)) / (10 + idx))
        prices[ticker] = 100.0 * np.cumprod(1.0 + returns)
    prices[CASH_TICKER] = 100.0 * np.cumprod(np.full(len(dates), 1.0 + 0.00005))

    config = BacktestConfig(
        momentum_lookback_days=252,
        momentum_skip_days=21,
        trend_window_days=200,
        covariance_lookback_days=126,
        stress_ewma_span_days=20,
        stress_short_window_days=63,
        stress_reference_window_days=126,
        covariance_estimator="sample",
    )
    result = run_backtest(
        prices,
        strategies=[StrategySpec("Equal Weight", "equal_weight")],
        config=config,
    )

    assert not result.returns.empty
    assert set(result.diagnostics["momentum_lookback_days"]) == {252}
    assert set(result.diagnostics["trend_window_days"]) == {200}
    assert set(result.diagnostics["covariance_lookback_days"]) == {126}
    assert set(result.diagnostics["stress_ewma_span_days"]) == {20}
    assert set(result.diagnostics["stress_reference_window_days"]) == {126}

    first_rebalance_date = pd.Timestamp(result.diagnostics["date"].min())
    assert result.returns.index.min() > first_rebalance_date


def test_strategy_specific_target_volatility_diagnostics_stays_at_ten_percent():
    dates = pd.bdate_range("2020-01-01", periods=340)
    prices = pd.DataFrame(index=dates)
    for idx, ticker in enumerate(RISKY_TICKERS):
        returns = 0.0002 + 0.00001 * idx + 0.0008 * np.sin(np.arange(len(dates)) / (12 + idx))
        prices[ticker] = 100.0 * np.cumprod(1.0 + returns)
    prices[CASH_TICKER] = 100.0 * np.cumprod(np.full(len(dates), 1.0 + 0.00005))

    result = run_backtest(
        prices,
        strategies=[
            StrategySpec(
                "Proposed sigma*=10%",
                "cash_allowed_target_vol_robust",
                {"target_volatility": 0.10},
            ),
        ],
        config=BacktestConfig(covariance_estimator="sample"),
    )

    target_vols = result.diagnostics.groupby("strategy")["target_volatility"].first()
    assert target_vols.loc["Proposed sigma*=10%"] == pytest.approx(0.10)
    assert (
        result.diagnostics["predicted_volatility"]
        <= result.diagnostics["target_volatility"] + 1e-5
    ).all()


def test_default_final_strategy_uses_ewma_risk_and_dynamic_target_volatility():
    dates = pd.bdate_range("2020-01-01", periods=340)
    prices = pd.DataFrame(index=dates)
    for idx, ticker in enumerate(RISKY_TICKERS):
        returns = 0.0002 + 0.00001 * idx + 0.001 * np.sin(np.arange(len(dates)) / (8 + idx))
        returns[-25:] += 0.006 * np.sin(np.arange(25) * (idx + 1))
        prices[ticker] = 100.0 * np.cumprod(1.0 + returns)
    prices[CASH_TICKER] = 100.0 * np.cumprod(np.full(len(dates), 1.0 + 0.00005))

    result = run_backtest(
        prices,
        strategies=default_strategy_specs(),
        config=BacktestConfig(covariance_estimator="sample"),
    )

    final = result.diagnostics[result.diagnostics["strategy"] == "Proposed Strategy"]
    assert not final.empty
    assert list(result.returns.columns) == [
        "Equal Weight",
        "Minimum Variance",
        "Traditional MVO",
        "Proposed Strategy",
    ]
    assert set(result.diagnostics["strategy"]) == {
        "Equal Weight",
        "Minimum Variance",
        "Traditional MVO",
        "Proposed Strategy",
    }
    assert set(final["market_state_mode"]) == {"ewma"}
    assert set(final["target_volatility_mode"]) == {"inverse_vt"}
    assert set(final["covariance_estimator"]) == {"ewma"}
    assert set(final["covariance_lookback_days"]) == {20}
    expected_target = (0.10 / final["v_t"]).clip(lower=0.04, upper=0.16)
    assert np.allclose(final["effective_target_volatility"], expected_target)
    assert np.allclose(final["rho_t"], (0.08 * final["v_t"]).clip(lower=0.03, upper=0.25))
