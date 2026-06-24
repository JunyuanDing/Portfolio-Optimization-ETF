"""Forecast signal construction for ETF allocation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from .data import CASH_TICKER, RISKY_TICKERS

SignalMode = Literal["both", "momentum", "trend", "none"]


@dataclass(frozen=True)
class SignalConfig:
    """Configuration for risky-asset signal and alpha construction."""

    risky_tickers: Sequence[str] = tuple(RISKY_TICKERS)
    cash_ticker: str = CASH_TICKER
    momentum_lookback_days: int = 252
    momentum_skip_days: int = 21
    trend_window_days: int = 200
    trend_weight: float = 0.5
    alpha_scale: float = 0.03
    mode: SignalMode = "both"
    zscore_epsilon: float = 1e-12


def risky_columns(
    columns: pd.Index,
    risky_tickers: Sequence[str] = RISKY_TICKERS,
    cash_ticker: str = CASH_TICKER,
) -> list[str]:
    """Return risky asset columns, explicitly excluding the cash ticker."""
    available = set(columns)
    return [ticker for ticker in risky_tickers if ticker in available and ticker != cash_ticker]


def momentum_12_1(
    prices: pd.DataFrame,
    lookback_days: int = 252,
    skip_days: int = 21,
) -> pd.DataFrame:
    """Compute 12-1 momentum using price[t-skip] / price[t-lookback] - 1."""
    if lookback_days <= skip_days:
        raise ValueError("lookback_days must be larger than skip_days.")
    return prices.shift(skip_days) / prices.shift(lookback_days) - 1.0


def cross_sectional_zscore(
    values: pd.DataFrame,
    epsilon: float = 1e-12,
) -> pd.DataFrame:
    """Z-score each row across assets, returning zero when dispersion is tiny."""
    row_mean = values.mean(axis=1)
    row_std = values.std(axis=1, ddof=0)
    centered = values.sub(row_mean, axis=0)
    zscores = centered.div(row_std.replace(0.0, np.nan), axis=0)
    small_dispersion = row_std.abs() <= epsilon
    if small_dispersion.any():
        zscores.loc[small_dispersion, :] = 0.0
    return zscores.fillna(0.0)


def cross_sectional_zscore_ex_cash(
    values: pd.DataFrame,
    cash_ticker: str = CASH_TICKER,
    risky_tickers: Sequence[str] = RISKY_TICKERS,
    epsilon: float = 1e-12,
) -> pd.DataFrame:
    """Compatibility wrapper: z-score risky assets and set cash to zero if present."""
    zscores = pd.DataFrame(0.0, index=values.index, columns=values.columns, dtype=float)
    risky = risky_columns(values.columns, risky_tickers, cash_ticker)
    if risky:
        zscores.loc[:, risky] = cross_sectional_zscore(values.loc[:, risky], epsilon)
    return zscores


def trend_signal(
    prices: pd.DataFrame,
    window_days: int = 200,
) -> pd.DataFrame:
    """Return +1 if price is above its moving average, otherwise -1."""
    moving_average = prices.rolling(window_days).mean()
    trend = pd.DataFrame(
        np.where(prices > moving_average, 1.0, -1.0),
        index=prices.index,
        columns=prices.columns,
    )
    return trend.where(moving_average.notna())


def compute_signal_scores(
    prices: pd.DataFrame,
    config: SignalConfig | None = None,
) -> pd.DataFrame:
    """Compute risky-only scores from raw 12-1 momentum and 200-day trend."""
    cfg = config or SignalConfig()
    risky = risky_columns(pd.Index(prices.columns), cfg.risky_tickers, cfg.cash_ticker)
    if not risky:
        raise ValueError("No risky asset columns are available for signal construction.")
    risky_prices = prices.loc[:, risky]
    momentum = momentum_12_1(
        risky_prices,
        lookback_days=cfg.momentum_lookback_days,
        skip_days=cfg.momentum_skip_days,
    )
    z_momentum = cross_sectional_zscore(momentum, epsilon=cfg.zscore_epsilon)
    trend = trend_signal(risky_prices, cfg.trend_window_days)

    if cfg.mode == "both":
        scores = z_momentum + cfg.trend_weight * trend
    elif cfg.mode == "momentum":
        scores = z_momentum.copy()
    elif cfg.mode == "trend":
        scores = cfg.trend_weight * trend
    elif cfg.mode == "none":
        scores = pd.DataFrame(0.0, index=risky_prices.index, columns=risky)
    else:
        raise ValueError(f"Unknown signal mode: {cfg.mode}")
    return scores.loc[:, risky]


def alpha_from_scores(
    scores: pd.DataFrame | pd.Series,
    alpha_scale: float = 0.03,
    cash_ticker: str = CASH_TICKER,
) -> pd.DataFrame | pd.Series:
    """Convert risky signal scores into annualized alpha forecasts."""
    alpha = alpha_scale * scores
    if isinstance(alpha, pd.Series) and cash_ticker in alpha.index:
        alpha = alpha.drop(index=cash_ticker)
    elif isinstance(alpha, pd.DataFrame) and cash_ticker in alpha.columns:
        alpha = alpha.drop(columns=[cash_ticker])
    return alpha


def signal_at_date(
    prices: pd.DataFrame,
    as_of_date: pd.Timestamp,
    config: SignalConfig | None = None,
) -> pd.Series:
    """Compute the latest risky-only score at or before `as_of_date`."""
    history = prices.loc[:as_of_date]
    if history.empty:
        raise ValueError("No price history is available at the requested date.")
    scores = compute_signal_scores(history, config)
    valid = scores.dropna(how="all")
    if valid.empty:
        raise ValueError("Insufficient history to compute signal scores.")
    return valid.iloc[-1].reindex(scores.columns)
