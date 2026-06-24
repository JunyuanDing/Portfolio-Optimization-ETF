"""Transaction cost, turnover, and drift-weight utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def align_weights(weights: pd.Series, columns: pd.Index | list[str]) -> pd.Series:
    """Align a weight vector to a return or price column index."""
    return weights.reindex(columns).fillna(0.0).astype(float)


def drift_weights(weights: pd.Series, asset_returns: pd.Series) -> pd.Series:
    """Drift portfolio weights after one period of asset returns."""
    aligned_returns = asset_returns.reindex(weights.index).fillna(0.0).astype(float)
    gross_asset_values = weights.astype(float) * (1.0 + aligned_returns)
    portfolio_gross_return = float(gross_asset_values.sum())
    if not np.isfinite(portfolio_gross_return) or portfolio_gross_return <= 0.0:
        raise ValueError("Portfolio gross value is non-positive after drift.")
    return gross_asset_values / portfolio_gross_return


def drift_weights_over_period(
    weights: pd.Series,
    returns: pd.DataFrame,
) -> pd.Series:
    """Drift weights over a multi-day holding period without rebalancing."""
    current = weights.copy().astype(float)
    for _, row in returns.iterrows():
        current = drift_weights(current, row)
    return current


def l1_turnover(new_weights: pd.Series, old_weights: pd.Series) -> float:
    """Return double-sided L1 turnover for the full portfolio vector."""
    old_aligned = old_weights.reindex(new_weights.index).fillna(0.0)
    return float((new_weights.astype(float) - old_aligned.astype(float)).abs().sum())


def one_way_turnover(new_weights: pd.Series, old_weights: pd.Series) -> float:
    """Return one-way turnover: 0.5 * ||w_new - w_old||_1."""
    return 0.5 * l1_turnover(new_weights, old_weights)


def transaction_cost(
    new_weights: pd.Series,
    old_weights: pd.Series,
    one_way_cost_rate: float,
) -> float:
    """Compute transaction cost from full-vector one-way turnover and rate.

    Example: one_way_cost_rate=0.001 represents 10 bps.
    """
    if one_way_cost_rate < 0.0:
        raise ValueError("one_way_cost_rate must be nonnegative.")
    return one_way_turnover(new_weights, old_weights) * one_way_cost_rate


def simulate_holding_period(
    start_weights: pd.Series,
    returns: pd.DataFrame,
    initial_cost: float = 0.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Simulate buy-and-hold returns, gross returns, and ending weights."""
    current_weights = start_weights.copy().astype(float)
    net_returns = []
    gross_returns = []
    for step, (date, row) in enumerate(returns.iterrows()):
        aligned = row.reindex(current_weights.index).fillna(0.0)
        gross_return = float(current_weights.dot(aligned))
        net_return = gross_return - initial_cost if step == 0 else gross_return
        gross_returns.append((date, gross_return))
        net_returns.append((date, net_return))
        current_weights = drift_weights(current_weights, aligned)
    net_series = pd.Series(
        data=[value for _, value in net_returns],
        index=[date for date, _ in net_returns],
        dtype=float,
    )
    gross_series = pd.Series(
        data=[value for _, value in gross_returns],
        index=[date for date, _ in gross_returns],
        dtype=float,
    )
    return net_series, gross_series, current_weights
