"""Portfolio performance and allocation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import CASH_TICKER


def cumulative_returns(returns: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Convert simple returns to cumulative wealth."""
    return (1.0 + returns).cumprod()


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Compute geometric annualized return."""
    clean = returns.dropna()
    if clean.empty:
        return np.nan
    total_return = float((1.0 + clean).prod())
    return total_return ** (periods_per_year / len(clean)) - 1.0


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Compute annualized return volatility."""
    return float(returns.dropna().std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Compute annualized Sharpe ratio using a constant annual risk-free rate."""
    clean = returns.dropna()
    if clean.empty:
        return np.nan
    daily_rf = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess = clean - daily_rf
    vol = excess.std(ddof=1)
    if vol == 0 or not np.isfinite(vol):
        return np.nan
    return float(excess.mean() / vol * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series,
    target_return: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Compute annualized Sortino ratio."""
    clean = returns.dropna()
    downside = clean[clean < target_return] - target_return
    downside_vol = downside.std(ddof=1)
    if downside_vol == 0 or not np.isfinite(downside_vol):
        return np.nan
    annual_excess = annualized_return(clean, periods_per_year) - target_return
    return float(annual_excess / (downside_vol * np.sqrt(periods_per_year)))


def drawdown(returns: pd.Series) -> pd.Series:
    """Compute drawdown from simple returns."""
    wealth = cumulative_returns(returns.dropna())
    running_max = wealth.cummax()
    return wealth / running_max - 1.0


def max_drawdown(returns: pd.Series) -> float:
    """Compute maximum drawdown."""
    dd = drawdown(returns)
    return float(dd.min()) if not dd.empty else np.nan


def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Compute Calmar ratio."""
    mdd = abs(max_drawdown(returns))
    if mdd == 0 or not np.isfinite(mdd):
        return np.nan
    return float(annualized_return(returns, periods_per_year) / mdd)


def hit_rate(returns: pd.Series) -> float:
    """Fraction of periods with positive returns."""
    clean = returns.dropna()
    if clean.empty:
        return np.nan
    return float((clean > 0.0).mean())


def herfindahl_index(weights: pd.Series, cash_ticker: str = CASH_TICKER) -> float:
    """Compute concentration within the risky sleeve, excluding cash."""
    clean = weights.drop(labels=[cash_ticker], errors="ignore").dropna().astype(float)
    risky_exposure = float(clean.sum())
    if risky_exposure <= 1e-12:
        return np.nan
    normalized = clean / risky_exposure
    return float((normalized**2).sum())


def effective_number_of_assets(weights: pd.Series, cash_ticker: str = CASH_TICKER) -> float:
    """Compute 1 / risky-sleeve Herfindahl index."""
    hhi = herfindahl_index(weights, cash_ticker)
    if hhi <= 0.0:
        return np.nan
    return float(1.0 / hhi)


def rolling_sharpe(
    returns: pd.Series,
    window: int = 252,
    periods_per_year: int = 252,
) -> pd.Series:
    """Compute rolling annualized Sharpe ratio."""
    mean = returns.rolling(window).mean()
    vol = returns.rolling(window).std(ddof=1)
    return mean.div(vol).mul(np.sqrt(periods_per_year))


def performance_summary(
    returns: pd.DataFrame,
    turnover: pd.DataFrame | None = None,
    weights: pd.DataFrame | None = None,
    diagnostics: pd.DataFrame | None = None,
    periods_per_year: int = 252,
    cash_ticker: str = CASH_TICKER,
) -> pd.DataFrame:
    """Build a performance table for strategies."""
    rows = []
    for strategy in returns.columns:
        series = returns[strategy].dropna()
        row = {
            "strategy": strategy,
            "annualized_return": annualized_return(series, periods_per_year),
            "annualized_volatility": annualized_volatility(series, periods_per_year),
            "sharpe_ratio": sharpe_ratio(series, periods_per_year=periods_per_year),
            "sortino_ratio": sortino_ratio(series, periods_per_year=periods_per_year),
            "max_drawdown": max_drawdown(series),
            "calmar_ratio": calmar_ratio(series, periods_per_year),
            "hit_rate": hit_rate(series),
            "realized_volatility": annualized_volatility(series, periods_per_year),
        }
        if turnover is not None and strategy in turnover.columns:
            row["average_turnover"] = float(turnover[strategy].dropna().mean())
        if weights is not None and not weights.empty and "strategy" in weights.columns:
            strategy_weights = weights[weights["strategy"] == strategy]
            if not strategy_weights.empty:
                by_date = strategy_weights.pivot(index="date", columns="asset", values="weight")
                row["average_cash_weight"] = float(by_date.get(cash_ticker, pd.Series(index=by_date.index, data=0.0)).mean())
                row["maximum_cash_weight"] = float(by_date.get(cash_ticker, pd.Series(index=by_date.index, data=0.0)).max())
                row["minimum_cash_weight"] = float(by_date.get(cash_ticker, pd.Series(index=by_date.index, data=0.0)).min())
                row["average_herfindahl"] = float(by_date.apply(lambda x: herfindahl_index(x, cash_ticker), axis=1).mean())
                row["average_effective_assets"] = float(by_date.apply(lambda x: effective_number_of_assets(x, cash_ticker), axis=1).mean())
        if diagnostics is not None and not diagnostics.empty:
            diag = diagnostics[diagnostics["strategy"] == strategy]
            if not diag.empty:
                for source, output in [
                    ("predicted_volatility", "average_predicted_volatility"),
                    ("rho_t", "average_rho_t"),
                    ("v_t", "average_v_t"),
                ]:
                    if source in diag.columns:
                        row[output] = float(pd.to_numeric(diag[source], errors="coerce").mean())
                if "solver_success" in diag.columns:
                    row["solver_failure_count"] = int((diag["solver_success"] == False).sum())
                if "constraint_valid" in diag.columns:
                    row["constraint_violation_count"] = int((diag["constraint_valid"] == False).sum())
        rows.append(row)
    return pd.DataFrame(rows).set_index("strategy")
