import pytest

pd = pytest.importorskip("pandas")

from portfolio.costs import one_way_turnover, simulate_holding_period, transaction_cost


def test_full_vector_turnover_includes_cash_weight_change():
    old = pd.Series({"SPY": 0.20, "TLT": 0.10, "BIL": 0.70})
    new = pd.Series({"SPY": 0.25, "TLT": 0.25, "BIL": 0.50})
    expected = 0.5 * (0.05 + 0.15 + 0.20)
    assert one_way_turnover(new, old) == pytest.approx(expected)
    assert transaction_cost(new, old, 0.001) == pytest.approx(0.001 * expected)


def test_transaction_cost_net_return_not_above_gross_return():
    start = pd.Series({"SPY": 0.50, "BIL": 0.50})
    returns = pd.DataFrame(
        {"SPY": [0.01, -0.005], "BIL": [0.0001, 0.0001]},
        index=pd.bdate_range("2021-01-01", periods=2),
    )
    net, gross, _ = simulate_holding_period(start, returns, initial_cost=0.001)
    assert net.iloc[0] <= gross.iloc[0] + 1e-12
    assert net.iloc[1] == pytest.approx(gross.iloc[1])
