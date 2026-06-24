import inspect

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("cvxpy")

from portfolio.costs import one_way_turnover
from portfolio.robust_optimization import (
    CashAllowedOptimizationConfig,
    solve_cash_allowed_target_vol_robust,
)
from portfolio.validation import validate_cash_allowed_weights


def diagonal_covariance(assets, variances):
    return pd.DataFrame(np.diag(variances), index=assets, columns=assets)


def test_cash_allowed_optimizer_satisfies_final_constraints():
    assets = ["SPY", "TLT", "GLD"]
    alpha = pd.Series([0.06, 0.02, 0.01], index=assets)
    covariance = diagonal_covariance(assets, [0.04, 0.01, 0.0225])
    pre_risky = pd.Series([0.10, 0.20, 0.10], index=assets)
    pre_cash = 0.60
    config = CashAllowedOptimizationConfig(
        target_volatility=0.10,
        lambda_tc=0.001,
        eta_l2=0.01,
        upper_bound=0.25,
    )

    result = solve_cash_allowed_target_vol_robust(
        alpha=alpha,
        covariance=covariance,
        pre_risky_weights=pre_risky,
        pre_cash_weight=pre_cash,
        rho_t=0.05,
        config=config,
        cash_ticker="BIL",
    )

    validation = validate_cash_allowed_weights(
        result.risky_weights,
        result.cash_weight,
        covariance,
        upper_bound=0.25,
        target_volatility=0.10,
        pre_full_weights=pd.concat([pre_risky, pd.Series({"BIL": pre_cash})]),
        cash_ticker="BIL",
        tolerance=1e-5,
    )
    assert validation.is_valid
    assert result.diagnostics["lambda_tc"] == pytest.approx(0.001)
    assert result.diagnostics["robust_penalty"] == pytest.approx(
        0.05 * validation.predicted_volatility, rel=1e-5
    )


def test_turnover_penalty_uses_full_vector_including_cash():
    assets = ["SPY", "TLT", "GLD"]
    alpha = pd.Series([0.05, 0.04, 0.03], index=assets)
    covariance = diagonal_covariance(assets, [0.02, 0.015, 0.01])
    pre_risky = pd.Series([0.20, 0.10, 0.00], index=assets)
    pre_cash = 0.70
    pre_full = pd.concat([pre_risky, pd.Series({"BIL": pre_cash})])

    result = solve_cash_allowed_target_vol_robust(
        alpha=alpha,
        covariance=covariance,
        pre_risky_weights=pre_risky,
        pre_cash_weight=pre_cash,
        rho_t=0.03,
        config=CashAllowedOptimizationConfig(target_volatility=0.10),
        cash_ticker="BIL",
    )

    assert result.diagnostics["turnover_penalty"] == pytest.approx(
        one_way_turnover(result.full_weights, pre_full), rel=1e-6
    )


def test_negative_alpha_pushes_optimizer_to_cash():
    assets = ["SPY", "TLT", "GLD"]
    alpha = pd.Series([-0.04, -0.03, -0.02], index=assets)
    covariance = diagonal_covariance(assets, [0.01, 0.01, 0.01])

    result = solve_cash_allowed_target_vol_robust(
        alpha=alpha,
        covariance=covariance,
        pre_risky_weights=pd.Series(1.0 / 3.0, index=assets),
        pre_cash_weight=0.0,
        rho_t=0.05,
        config=CashAllowedOptimizationConfig(target_volatility=0.10),
        cash_ticker="BIL",
    )

    assert result.cash_weight >= 0.95
    assert result.risky_weights.sum() <= 0.05


def test_cash_allowed_optimizer_uses_soc_norm_risk_expression():
    source = inspect.getsource(solve_cash_allowed_target_vol_robust)
    assert "cp.norm(factor @ x, 2)" in source
    assert "cp.sqrt" not in source
    assert "quad_form" not in source
