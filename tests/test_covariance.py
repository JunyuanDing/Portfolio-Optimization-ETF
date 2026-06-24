import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from portfolio.covariance import (
    ewma_covariance,
    is_psd,
    make_psd,
    risky_returns_frame,
    sample_covariance,
)


def test_sample_covariance_is_annualized_and_psd_after_repair():
    returns = pd.DataFrame(
        {
            "A": [0.01, 0.02, -0.01, 0.00],
            "B": [0.00, 0.01, -0.02, 0.01],
        }
    )
    covariance = sample_covariance(returns, periods_per_year=252)
    expected = returns.cov() * 252
    repaired = make_psd(covariance)
    assert covariance.shape == (2, 2)
    assert np.allclose(covariance.values, expected.values)
    assert is_psd(repaired)


def test_ewma_covariance_is_psd_after_repair():
    rng = np.random.default_rng(7)
    returns = pd.DataFrame(rng.normal(0, 0.01, size=(100, 3)), columns=["A", "B", "C"])
    covariance = ewma_covariance(returns)
    assert is_psd(make_psd(covariance))


def test_risky_returns_frame_excludes_bil_from_covariance_inputs():
    returns = pd.DataFrame(
        {
            "SPY": [0.01, 0.02],
            "TLT": [0.00, 0.01],
            "BIL": [0.0001, 0.0001],
        }
    )
    risky = risky_returns_frame(returns, risky_tickers=("SPY", "TLT"), cash_ticker="BIL")
    assert list(risky.columns) == ["SPY", "TLT"]
    assert "BIL" not in risky.columns
