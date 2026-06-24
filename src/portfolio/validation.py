"""Validation utilities for optimized portfolios and backtests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .costs import l1_turnover, one_way_turnover
from .data import CASH_TICKER

ACCEPTABLE_SOLVER_STATUSES = {"optimal", "optimal_inaccurate"}


@dataclass(frozen=True)
class WeightValidation:
    """Portfolio validation diagnostics."""

    is_valid: bool
    weight_sum: float
    min_weight: float
    max_weight: float
    budget_error: float
    min_weight_violation: float
    upper_bound_violation: float
    turnover: float | None
    turnover_violation: float | None


@dataclass(frozen=True)
class CashAllowedValidation:
    """Validation diagnostics for risky plus cash portfolios."""

    is_valid: bool
    full_weight_sum: float
    risky_weight_sum: float
    cash_weight: float
    min_risky_weight: float
    max_risky_weight: float
    budget_error: float
    cash_violation: float
    min_risky_violation: float
    upper_bound_violation: float
    target_volatility_violation: float
    predicted_volatility: float
    one_way_turnover: float | None
    l1_turnover: float | None


def validate_weights(
    weights: pd.Series,
    pre_weights: pd.Series | None = None,
    upper_bound: float = 0.25,
    turnover_limit: float | None = 0.40,
    tolerance: float = 1e-6,
) -> WeightValidation:
    """Validate budget, long-only, upper-bound, and optional L1 turnover constraints."""
    clean = weights.astype(float)
    weight_sum = float(clean.sum())
    min_weight = float(clean.min())
    max_weight = float(clean.max())
    budget_error = abs(weight_sum - 1.0)
    min_violation = max(0.0, -min_weight)
    upper_violation = max(0.0, max_weight - upper_bound)

    turnover = None
    turnover_violation = None
    if turnover_limit is not None and pre_weights is not None:
        turnover = l1_turnover(clean, pre_weights.reindex(clean.index).fillna(0.0))
        turnover_violation = max(0.0, turnover - turnover_limit)

    is_valid = (
        np.isfinite(clean.values).all()
        and budget_error <= tolerance
        and min_violation <= tolerance
        and upper_violation <= tolerance
        and (turnover_violation is None or turnover_violation <= tolerance)
    )

    return WeightValidation(
        is_valid=is_valid,
        weight_sum=weight_sum,
        min_weight=min_weight,
        max_weight=max_weight,
        budget_error=budget_error,
        min_weight_violation=min_violation,
        upper_bound_violation=upper_violation,
        turnover=turnover,
        turnover_violation=turnover_violation,
    )


def validate_cash_allowed_weights(
    risky_weights: pd.Series,
    cash_weight: float,
    covariance: pd.DataFrame,
    upper_bound: float = 0.25,
    target_volatility: float = 0.10,
    pre_full_weights: pd.Series | None = None,
    cash_ticker: str = CASH_TICKER,
    tolerance: float = 1e-6,
) -> CashAllowedValidation:
    """Validate budget, cash, risky bounds, and target volatility."""
    risky = risky_weights.astype(float)
    sigma = covariance.reindex(index=risky.index, columns=risky.index)
    predicted_variance = float(risky.values @ sigma.values @ risky.values)
    predicted_volatility = float(np.sqrt(max(predicted_variance, 0.0)))
    full = risky.copy()
    full.loc[cash_ticker] = float(cash_weight)
    full_weight_sum = float(full.sum())
    risky_weight_sum = float(risky.sum())
    min_risky = float(risky.min()) if not risky.empty else 0.0
    max_risky = float(risky.max()) if not risky.empty else 0.0
    budget_error = abs(full_weight_sum - 1.0)
    cash_violation = max(0.0, -float(cash_weight))
    min_risky_violation = max(0.0, -min_risky)
    upper_bound_violation = max(0.0, max_risky - upper_bound)
    target_volatility_violation = max(0.0, predicted_volatility - target_volatility)
    ow_turnover = None
    l1 = None
    if pre_full_weights is not None:
        aligned_pre = pre_full_weights.reindex(full.index).fillna(0.0)
        ow_turnover = one_way_turnover(full, aligned_pre)
        l1 = l1_turnover(full, aligned_pre)
    is_valid = (
        np.isfinite(full.values).all()
        and budget_error <= tolerance
        and cash_violation <= tolerance
        and min_risky_violation <= tolerance
        and upper_bound_violation <= tolerance
        and target_volatility_violation <= tolerance
    )
    return CashAllowedValidation(
        is_valid=is_valid,
        full_weight_sum=full_weight_sum,
        risky_weight_sum=risky_weight_sum,
        cash_weight=float(cash_weight),
        min_risky_weight=min_risky,
        max_risky_weight=max_risky,
        budget_error=budget_error,
        cash_violation=cash_violation,
        min_risky_violation=min_risky_violation,
        upper_bound_violation=upper_bound_violation,
        target_volatility_violation=target_volatility_violation,
        predicted_volatility=predicted_volatility,
        one_way_turnover=ow_turnover,
        l1_turnover=l1,
    )


def is_acceptable_solver_status(status: str | None) -> bool:
    """Return whether a cvxpy solver status is acceptable for research output."""
    return status in ACCEPTABLE_SOLVER_STATUSES


def assert_no_lookahead(
    estimation_end: pd.Timestamp,
    holding_start: pd.Timestamp,
) -> None:
    """Raise if a holding period starts before the estimation window ends."""
    if pd.Timestamp(estimation_end) >= pd.Timestamp(holding_start):
        raise ValueError(
            "Potential look-ahead: holding_start must be after estimation_end."
        )


def assert_excludes_cash(
    labels: Sequence[str] | pd.Index,
    cash_ticker: str = CASH_TICKER,
    label_name: str = "labels",
) -> None:
    """Raise if a risky-only object includes the cash ticker."""
    if cash_ticker in set(labels):
        raise ValueError(f"{label_name} must exclude cash ticker {cash_ticker}.")


def assert_fixed_lambda_tc(lambda_tc: float, expected: float = 0.001) -> None:
    """Raise if the turnover penalty differs from the fixed main-spec value."""
    if abs(lambda_tc - expected) > 1e-12:
        raise ValueError("lambda_tc must remain fixed at 0.001 in the main strategy.")
