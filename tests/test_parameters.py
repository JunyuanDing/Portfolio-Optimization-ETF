import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from portfolio.parameters import (
    compute_market_state,
    dynamic_rho,
    ewma_volatility,
    ewma_volatility_stress_ratio,
    risky_equal_weight_returns,
    volatility_stress_ratio,
)


def test_risky_equal_weight_returns_excludes_bil():
    returns = pd.DataFrame(
        {
            "SPY": [0.01, 0.03],
            "TLT": [0.02, 0.04],
            "BIL": [0.99, 0.99],
        }
    )
    ew = risky_equal_weight_returns(returns, risky_tickers=("SPY", "TLT"), cash_ticker="BIL")
    expected = returns[["SPY", "TLT"]].mean(axis=1)
    assert ew.equals(expected)


def test_volatility_stress_ratio_uses_historical_risky_returns_and_epsilon():
    dates = pd.bdate_range("2021-01-01", periods=252)
    returns = pd.DataFrame(
        {
            "SPY": np.zeros(252),
            "TLT": np.zeros(252),
            "BIL": np.linspace(0.0, 0.10, 252),
        },
        index=dates,
    )
    sigma_63, sigma_252, v_t = volatility_stress_ratio(
        returns,
        short_window=63,
        long_window=252,
        risky_tickers=("SPY", "TLT"),
        cash_ticker="BIL",
        epsilon=1e-12,
    )
    assert sigma_63 == pytest.approx(0.0)
    assert sigma_252 == pytest.approx(0.0)
    assert v_t == pytest.approx(0.5)


def test_dynamic_rho_increases_with_volatility_stress():
    assert dynamic_rho(0.5) < dynamic_rho(1.0) < dynamic_rho(2.0)
    assert dynamic_rho(10.0) == pytest.approx(0.15)


def test_ewma_volatility_stress_ratio_uses_20_day_ewma_over_252():
    dates = pd.bdate_range("2021-01-01", periods=252)
    calm = np.full(231, 0.0005)
    volatile = np.array([0.02, -0.02] * 11)[:21]
    ew_returns = np.concatenate([calm, volatile])
    returns = pd.DataFrame(
        {
            "SPY": ew_returns,
            "TLT": ew_returns,
            "BIL": np.linspace(0.0, 0.10, 252),
        },
        index=dates,
    )

    sigma_ewma, sigma_63, sigma_252, v_t = ewma_volatility_stress_ratio(
        returns,
        ewma_span=20,
        short_window=63,
        long_window=252,
        risky_tickers=("SPY", "TLT"),
        cash_ticker="BIL",
    )
    expected_sigma = ewma_volatility(pd.Series(ew_returns, index=dates), span=20)
    expected = np.clip(expected_sigma / sigma_252, 0.5, 2.5)

    assert sigma_ewma == pytest.approx(expected_sigma)
    assert v_t == pytest.approx(expected)


def test_ewma_market_state_uses_stronger_rho_mapping():
    dates = pd.bdate_range("2021-01-01", periods=252)
    returns = pd.DataFrame(
        {
            "SPY": np.linspace(-0.01, 0.01, 252),
            "TLT": np.linspace(0.01, -0.01, 252),
        },
        index=dates,
    )

    state = compute_market_state(
        returns,
        mode="ewma",
        ewma_span=20,
        short_window=63,
        long_window=252,
        rho_base=0.08,
        rho_lower=0.03,
        rho_upper=0.25,
        risky_tickers=("SPY", "TLT"),
    )

    assert state.sigma_ewma_short >= 0.0
    assert state.rho_t == pytest.approx(dynamic_rho(state.v_t, base=0.08, lower=0.03, upper=0.25))
