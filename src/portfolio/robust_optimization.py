"""Convex portfolio optimizers and baseline allocation rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from .covariance import covariance_sqrt, make_psd
from .data import CASH_TICKER, RISKY_TICKERS
from .validation import is_acceptable_solver_status


@dataclass(frozen=True)
class CashAllowedOptimizationConfig:
    """Configuration for the final cash-allowed target-volatility optimizer."""

    target_volatility: float = 0.10
    lambda_tc: float = 0.001
    eta_l2: float = 0.01
    upper_bound: float = 0.25
    solver: str | None = None
    fallback_to_cash: bool = True
    psd_floor: float = 1e-8


@dataclass
class OptimizationResult:
    """Optimization output with diagnostics needed for research auditing."""

    weights: pd.Series
    status: str | None
    objective_value: float | None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class CashAllowedOptimizationResult:
    """Cash-allowed optimization result."""

    risky_weights: pd.Series
    cash_weight: float
    full_weights: pd.Series
    status: str | None
    objective_value: float | None
    diagnostics: dict[str, Any] = field(default_factory=dict)


def full_weight_vector(
    risky_weights: pd.Series,
    cash_weight: float,
    cash_ticker: str = CASH_TICKER,
) -> pd.Series:
    """Combine risky weights and cash weight into a full portfolio vector."""
    full = risky_weights.copy().astype(float)
    full.loc[cash_ticker] = float(cash_weight)
    return full


def equal_weight(assets: list[str] | pd.Index) -> pd.Series:
    """Return an equal-weight portfolio."""
    asset_list = list(assets)
    return pd.Series(1.0 / len(asset_list), index=asset_list, dtype=float)


def equal_weight_risky(
    risky_assets: list[str] | pd.Index = RISKY_TICKERS,
    cash_ticker: str = CASH_TICKER,
) -> pd.Series:
    """Return a full vector with equal weights across risky assets and zero cash."""
    risky = list(risky_assets)
    weights = pd.Series(1.0 / len(risky), index=risky, dtype=float)
    return full_weight_vector(weights, 0.0, cash_ticker)


def solve_cash_allowed_target_vol_robust(
    alpha: pd.Series,
    covariance: pd.DataFrame,
    pre_risky_weights: pd.Series | None,
    pre_cash_weight: float,
    rho_t: float,
    config: CashAllowedOptimizationConfig | None = None,
    cash_ticker: str = CASH_TICKER,
) -> CashAllowedOptimizationResult:
    """Solve the final cash-allowed robust target-volatility convex problem."""
    try:
        import cvxpy as cp
    except ImportError as exc:
        raise ImportError(
            "cvxpy is required for optimization. Install requirements.txt first."
        ) from exc

    cfg = config or CashAllowedOptimizationConfig()
    risky_assets = list(alpha.index)
    sigma = make_psd(covariance.reindex(index=risky_assets, columns=risky_assets), cfg.psd_floor)
    alpha_vector = alpha.reindex(risky_assets).fillna(0.0).astype(float).values
    if pre_risky_weights is None:
        pre_risky = pd.Series(0.0, index=risky_assets, dtype=float)
    else:
        pre_risky = pre_risky_weights.reindex(risky_assets).fillna(0.0).astype(float)
    pre_risky_vector = pre_risky.values
    pre_cash = float(pre_cash_weight)
    pre_full_vector = np.append(pre_risky_vector, pre_cash)

    x = cp.Variable(len(risky_assets))
    c = cp.Variable()
    factor = covariance_sqrt(sigma, floor=cfg.psd_floor)
    predicted_vol_expr = cp.norm(factor @ x, 2)
    full_weight_expr = cp.hstack([x, c])
    turnover_penalty_expr = 0.5 * cp.norm1(full_weight_expr - pre_full_vector)

    objective = cp.Maximize(
        alpha_vector @ x
        - cfg.lambda_tc * turnover_penalty_expr
        - cfg.eta_l2 * cp.sum_squares(x)
        - rho_t * predicted_vol_expr
    )
    constraints = [
        cp.sum(x) + c == 1.0,
        x >= 0.0,
        c >= 0.0,
        x <= cfg.upper_bound,
        predicted_vol_expr <= cfg.target_volatility,
    ]
    problem = cp.Problem(objective, constraints)

    status = None
    solve_error = None
    solve_start = perf_counter()
    for solver in _solver_candidates(cp, cfg.solver):
        try:
            if solver is None:
                problem.solve(verbose=False)
            else:
                problem.solve(solver=solver, verbose=False)
            status = problem.status
            if x.value is not None and c.value is not None:
                break
        except cp.SolverError as exc:
            solve_error = str(exc)
            continue
    solver_time_seconds = perf_counter() - solve_start

    if x.value is None or c.value is None or not is_acceptable_solver_status(status):
        if cfg.fallback_to_cash:
            risky_weights = pd.Series(0.0, index=risky_assets, dtype=float)
            cash_weight = 1.0
        else:
            risky_weights = pre_risky.clip(lower=0.0)
            total_risky = float(risky_weights.sum())
            cash_weight = max(0.0, 1.0 - total_risky)
            total = total_risky + cash_weight
            risky_weights = risky_weights / total
            cash_weight = cash_weight / total
        full_weights = full_weight_vector(risky_weights, cash_weight, cash_ticker)
        return CashAllowedOptimizationResult(
            risky_weights=risky_weights,
            cash_weight=cash_weight,
            full_weights=full_weights,
            status=status,
            objective_value=None,
            diagnostics={
                "success": False,
                "solver_success": False,
                "solve_error": solve_error,
                "fallback_used": True,
                "predicted_volatility": 0.0,
                "rho_t": rho_t,
                "lambda_tc": cfg.lambda_tc,
                "eta_l2": cfg.eta_l2,
                "target_volatility": cfg.target_volatility,
                "solver_time_seconds": solver_time_seconds,
            },
        )

    risky_values = np.asarray(x.value).ravel()
    risky_values[np.abs(risky_values) < 1e-10] = 0.0
    risky_weights = pd.Series(risky_values, index=risky_assets, dtype=float).clip(lower=0.0)
    cash_weight = float(max(0.0, c.value))
    total = float(risky_weights.sum() + cash_weight)
    if total <= 0.0 or not np.isfinite(total):
        risky_weights = pd.Series(0.0, index=risky_assets, dtype=float)
        cash_weight = 1.0
    else:
        risky_weights = risky_weights / total
        cash_weight = cash_weight / total

    full_weights = full_weight_vector(risky_weights, cash_weight, cash_ticker)
    predicted_variance = float(risky_weights.values @ sigma.values @ risky_weights.values)
    predicted_volatility = float(np.sqrt(max(predicted_variance, 0.0)))
    turnover_penalty = 0.5 * float(
        np.abs(np.append(risky_weights.values, cash_weight) - pre_full_vector).sum()
    )
    diagnostics = {
        "success": True,
        "solver_success": True,
        "fallback_used": False,
        "weight_sum": float(full_weights.sum()),
        "risky_weight_sum": float(risky_weights.sum()),
        "cash_weight": cash_weight,
        "min_risky_weight": float(risky_weights.min()),
        "max_risky_weight": float(risky_weights.max()),
        "predicted_volatility": predicted_volatility,
        "target_volatility": cfg.target_volatility,
        "rho_t": rho_t,
        "lambda_tc": cfg.lambda_tc,
        "eta_l2": cfg.eta_l2,
        "solver_time_seconds": solver_time_seconds,
        "robust_penalty": float(rho_t * predicted_volatility),
        "turnover_penalty": turnover_penalty,
        "turnover_penalty_l1_full": float(2.0 * turnover_penalty),
    }
    return CashAllowedOptimizationResult(
        risky_weights=risky_weights,
        cash_weight=cash_weight,
        full_weights=full_weights,
        status=status,
        objective_value=None if problem.value is None else float(problem.value),
        diagnostics=diagnostics,
    )


def solve_minimum_variance_long_only(
    covariance: pd.DataFrame,
    upper_bound: float = 0.25,
    solver: str | None = None,
    psd_floor: float = 1e-8,
) -> OptimizationResult:
    """Solve a fully invested long-only risky minimum-variance portfolio."""
    try:
        import cvxpy as cp
    except ImportError as exc:
        raise ImportError(
            "cvxpy is required for optimization. Install requirements.txt first."
        ) from exc

    assets = list(covariance.index)
    sigma = make_psd(covariance.reindex(index=assets, columns=assets), psd_floor)
    x = cp.Variable(len(assets))
    problem = cp.Problem(
        cp.Minimize(cp.quad_form(x, sigma.values)),
        [cp.sum(x) == 1.0, x >= 0.0, x <= upper_bound],
    )
    status = None
    solve_error = None
    solve_start = perf_counter()
    for candidate in _solver_candidates(cp, solver):
        try:
            if candidate is None:
                problem.solve(verbose=False)
            else:
                problem.solve(solver=candidate, verbose=False)
            status = problem.status
            if x.value is not None:
                break
        except cp.SolverError as exc:
            solve_error = str(exc)
            continue
    solver_time_seconds = perf_counter() - solve_start
    if x.value is None or not is_acceptable_solver_status(status):
        weights = equal_weight(assets)
        return OptimizationResult(
            weights=weights,
            status=status,
            objective_value=None,
            diagnostics={
                "success": False,
                "solver_success": False,
                "solve_error": solve_error,
                "fallback_used": True,
                "solver_time_seconds": solver_time_seconds,
            },
        )
    values = np.asarray(x.value).ravel()
    values[np.abs(values) < 1e-10] = 0.0
    weights = pd.Series(values, index=assets, dtype=float).clip(lower=0.0)
    weights = weights / weights.sum()
    predicted_variance = float(weights.values @ sigma.values @ weights.values)
    return OptimizationResult(
        weights=weights,
        status=status,
        objective_value=None if problem.value is None else float(problem.value),
        diagnostics={
            "success": True,
            "solver_success": True,
            "fallback_used": False,
            "predicted_volatility": float(np.sqrt(max(predicted_variance, 0.0))),
            "solver_time_seconds": solver_time_seconds,
        },
    )


def solve_traditional_mvo_long_only(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_aversion: float = 1.0,
    upper_bound: float = 0.25,
    solver: str | None = None,
    psd_floor: float = 1e-8,
) -> OptimizationResult:
    """Solve a long-only traditional mean-variance portfolio."""
    try:
        import cvxpy as cp
    except ImportError as exc:
        raise ImportError(
            "cvxpy is required for optimization. Install requirements.txt first."
        ) from exc

    if risk_aversion < 0.0:
        raise ValueError("risk_aversion must be non-negative.")

    assets = list(expected_returns.index)
    sigma = make_psd(covariance.reindex(index=assets, columns=assets), psd_floor)
    mu = expected_returns.reindex(assets).fillna(0.0).astype(float).values
    x = cp.Variable(len(assets))
    problem = cp.Problem(
        cp.Maximize(mu @ x - risk_aversion * cp.quad_form(x, sigma.values)),
        [cp.sum(x) == 1.0, x >= 0.0, x <= upper_bound],
    )
    status = None
    solve_error = None
    solve_start = perf_counter()
    for candidate in _solver_candidates(cp, solver):
        try:
            if candidate is None:
                problem.solve(verbose=False)
            else:
                problem.solve(solver=candidate, verbose=False)
            status = problem.status
            if x.value is not None:
                break
        except cp.SolverError as exc:
            solve_error = str(exc)
            continue
    solver_time_seconds = perf_counter() - solve_start
    if x.value is None or not is_acceptable_solver_status(status):
        weights = equal_weight(assets)
        return OptimizationResult(
            weights=weights,
            status=status,
            objective_value=None,
            diagnostics={
                "success": False,
                "solver_success": False,
                "solve_error": solve_error,
                "fallback_used": True,
                "solver_time_seconds": solver_time_seconds,
            },
        )
    values = np.asarray(x.value).ravel()
    values[np.abs(values) < 1e-10] = 0.0
    weights = pd.Series(values, index=assets, dtype=float).clip(lower=0.0)
    weights = weights / weights.sum()
    predicted_variance = float(weights.values @ sigma.values @ weights.values)
    return OptimizationResult(
        weights=weights,
        status=status,
        objective_value=None if problem.value is None else float(problem.value),
        diagnostics={
            "success": True,
            "solver_success": True,
            "fallback_used": False,
            "predicted_volatility": float(np.sqrt(max(predicted_variance, 0.0))),
            "solver_time_seconds": solver_time_seconds,
            "risk_aversion": risk_aversion,
        },
    )


def solve_minimum_variance(
    covariance: pd.DataFrame,
    pre_weights: pd.Series | None = None,
    upper_bound: float = 0.25,
) -> OptimizationResult:
    """Compatibility wrapper for the minimum-variance baseline."""
    return solve_minimum_variance_long_only(covariance, upper_bound=upper_bound)


def _solver_candidates(cp, requested_solver: str | None) -> list[str | None]:
    installed = set(cp.installed_solvers())
    if requested_solver is not None:
        return [requested_solver, None]
    ordered = ["CLARABEL", "ECOS", "SCS"]
    candidates = [solver for solver in ordered if solver in installed]
    candidates.append(None)
    return candidates
