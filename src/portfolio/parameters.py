"""Dynamic market-pressure and robust-penalty parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .data import CASH_TICKER, RISKY_TICKERS


@dataclass(frozen=True)
class MarketState:
    """Volatility stress state used to set the robust penalty."""

    sigma_ewma_short: float
    sigma_ew_63d: float
    sigma_ew_252d: float
    v_t: float
    rho_t: float


def risky_returns_only(
    returns: pd.DataFrame,
    risky_tickers: Sequence[str] = RISKY_TICKERS,
    cash_ticker: str = CASH_TICKER,
) -> pd.DataFrame:
    """Return risky ETF returns only, explicitly excluding the cash ticker."""
    columns = [ticker for ticker in risky_tickers if ticker in returns.columns and ticker != cash_ticker]
    if not columns:
        raise ValueError("No risky return columns are available.")
    return returns.loc[:, columns]


def risky_equal_weight_returns(
    returns: pd.DataFrame,
    risky_tickers: Sequence[str] = RISKY_TICKERS,
    cash_ticker: str = CASH_TICKER,
) -> pd.Series:
    """Compute equal-weight risky ETF returns, excluding BIL/cash."""
    risky = risky_returns_only(returns, risky_tickers, cash_ticker)
    return risky.mean(axis=1)


def annualized_volatility(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Annualize the standard deviation of daily returns."""
    clean = returns.dropna()
    if clean.empty:
        raise ValueError("Cannot compute volatility from an empty return series.")
    return float(clean.std(ddof=1) * np.sqrt(periods_per_year))


def volatility_stress_ratio(
    returns_window: pd.DataFrame,
    short_window: int = 63,
    long_window: int = 252,
    periods_per_year: int = 252,
    lower: float = 0.5,
    upper: float = 2.5,
    epsilon: float = 1e-12,
    risky_tickers: Sequence[str] = RISKY_TICKERS,
    cash_ticker: str = CASH_TICKER,
) -> tuple[float, float, float]:
    """Compute clipped sigma_63 / sigma_252 using risky equal-weight returns."""
    if len(returns_window) < long_window:
        raise ValueError("returns_window must contain at least long_window observations.")
    ew_returns = risky_equal_weight_returns(returns_window.tail(long_window), risky_tickers, cash_ticker)
    sigma_short = annualized_volatility(ew_returns.tail(short_window), periods_per_year)
    sigma_long = annualized_volatility(ew_returns.tail(long_window), periods_per_year)
    denominator = max(float(sigma_long), epsilon)
    raw_ratio = sigma_short / denominator
    v_t = float(np.clip(raw_ratio, lower, upper))
    return sigma_short, sigma_long, v_t


def ewma_volatility(
    returns: pd.Series,
    span: int = 20,
    periods_per_year: int = 252,
) -> float:
    """Compute the latest annualized EWMA volatility."""
    if span <= 1:
        raise ValueError("span must be greater than one.")
    clean = returns.dropna()
    if len(clean) < span:
        raise ValueError("Not enough observations for EWMA volatility.")
    variance = clean.ewm(span=span, adjust=False).var(bias=False).iloc[-1]
    return float(np.sqrt(max(float(variance), 0.0)) * np.sqrt(periods_per_year))


def ewma_volatility_stress_ratio(
    returns_window: pd.DataFrame,
    ewma_span: int = 20,
    short_window: int = 63,
    long_window: int = 252,
    periods_per_year: int = 252,
    lower: float = 0.5,
    upper: float = 2.5,
    epsilon: float = 1e-12,
    risky_tickers: Sequence[str] = RISKY_TICKERS,
    cash_ticker: str = CASH_TICKER,
) -> tuple[float, float, float, float]:
    """Compute clipped short-window EWMA volatility / sigma_252 for market pressure."""
    if len(returns_window) < long_window:
        raise ValueError("returns_window must contain at least long_window observations.")
    ew_returns = risky_equal_weight_returns(returns_window.tail(long_window), risky_tickers, cash_ticker)
    sigma_ewma = ewma_volatility(ew_returns, span=ewma_span, periods_per_year=periods_per_year)
    sigma_short = annualized_volatility(ew_returns.tail(short_window), periods_per_year)
    sigma_long = annualized_volatility(ew_returns.tail(long_window), periods_per_year)
    denominator = max(float(sigma_long), epsilon)
    raw_ratio = sigma_ewma / denominator
    v_t = float(np.clip(raw_ratio, lower, upper))
    return sigma_ewma, sigma_short, sigma_long, v_t


def dynamic_rho(
    v_t: float,
    base: float = 0.05,
    lower: float = 0.02,
    upper: float = 0.15,
) -> float:
    """Compute rho_t = clip(base * v_t, lower, upper)."""
    return float(np.clip(base * v_t, lower, upper))


def compute_market_state(
    returns_window: pd.DataFrame,
    mode: str = "standard",
    ewma_span: int = 20,
    short_window: int = 63,
    long_window: int = 252,
    periods_per_year: int = 252,
    rho_base: float = 0.05,
    rho_lower: float = 0.02,
    rho_upper: float = 0.15,
    epsilon: float = 1e-12,
    risky_tickers: Sequence[str] = RISKY_TICKERS,
    cash_ticker: str = CASH_TICKER,
) -> MarketState:
    """Compute volatility stress v_t and dynamic robust penalty rho_t."""
    if mode == "standard":
        sigma_ewma = np.nan
        sigma_short, sigma_long, v_t = volatility_stress_ratio(
            returns_window=returns_window,
            short_window=short_window,
            long_window=long_window,
            periods_per_year=periods_per_year,
            epsilon=epsilon,
            risky_tickers=risky_tickers,
            cash_ticker=cash_ticker,
        )
    elif mode == "ewma":
        sigma_ewma, sigma_short, sigma_long, v_t = ewma_volatility_stress_ratio(
            returns_window=returns_window,
            ewma_span=ewma_span,
            short_window=short_window,
            long_window=long_window,
            periods_per_year=periods_per_year,
            epsilon=epsilon,
            risky_tickers=risky_tickers,
            cash_ticker=cash_ticker,
        )
    else:
        raise ValueError(f"Unknown market_state mode: {mode}")
    rho_t = dynamic_rho(v_t, base=rho_base, lower=rho_lower, upper=rho_upper)
    return MarketState(
        sigma_ewma_short=sigma_ewma,
        sigma_ew_63d=sigma_short,
        sigma_ew_252d=sigma_long,
        v_t=v_t,
        rho_t=rho_t,
    )
