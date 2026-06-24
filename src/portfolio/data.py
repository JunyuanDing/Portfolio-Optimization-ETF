"""Data loading, downloading, and preprocessing utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

CASH_TICKER = "BIL"

RISKY_TICKERS = [
    "SPY",
    "QQQ",
    "IWM",
    "EFA",
    "EEM",
    "TLT",
    "IEF",
    "SHY",
    "LQD",
    "HYG",
    "GLD",
    "DBC",
    "VNQ",
]

DEFAULT_TICKERS = RISKY_TICKERS + [CASH_TICKER]

ASSET_GROUPS = {
    "Equity": ["SPY", "QQQ", "IWM", "EFA", "EEM"],
    "Rates": ["TLT", "IEF", "SHY"],
    "Credit": ["LQD", "HYG"],
    "RealAssets": ["GLD", "DBC", "VNQ"],
    "CashLike": [CASH_TICKER],
}


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def risky_tickers_from_columns(
    columns: Iterable[str],
    cash_ticker: str = CASH_TICKER,
) -> list[str]:
    """Return known risky tickers present in a column collection."""
    available = set(columns)
    return [ticker for ticker in RISKY_TICKERS if ticker in available and ticker != cash_ticker]


def download_adjusted_close(
    tickers: Sequence[str] = DEFAULT_TICKERS,
    start: str = "2010-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Download adjusted close prices from Yahoo Finance through yfinance.

    The function intentionally returns only prices. The caller is responsible for
    saving the raw file before any transformations.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "yfinance is required for data download. Install requirements.txt first."
        ) from exc

    raw = yf.download(
        list(tickers),
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    if raw.empty:
        raise ValueError("Downloaded price data is empty.")

    if isinstance(raw.columns, pd.MultiIndex):
        if "Adj Close" in raw.columns.get_level_values(0):
            prices = raw["Adj Close"].copy()
        elif "Close" in raw.columns.get_level_values(0):
            prices = raw["Close"].copy()
        else:
            raise ValueError("Downloaded data has neither Adj Close nor Close.")
    else:
        column = "Adj Close" if "Adj Close" in raw.columns else "Close"
        prices = raw[[column]].copy()
        prices.columns = list(tickers)

    prices = prices.reindex(columns=list(tickers))
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    prices = prices.dropna(how="all")
    return prices


def save_frame(frame: pd.DataFrame, path: str | Path) -> None:
    """Save a DataFrame as CSV with an index label."""
    output_path = Path(path)
    ensure_directory(output_path.parent)
    frame.to_csv(output_path, index_label="Date")


def load_frame(path: str | Path) -> pd.DataFrame:
    """Load a CSV DataFrame with a Date index."""
    frame = pd.read_csv(path, index_col="Date", parse_dates=True)
    frame.index = pd.to_datetime(frame.index)
    return frame.sort_index()


def clean_prices(
    prices: pd.DataFrame,
    tickers: Iterable[str] | None = None,
    min_non_missing: float = 0.95,
) -> pd.DataFrame:
    """Forward-fill prices and drop rows or assets with insufficient history."""
    cleaned = prices.copy()
    if tickers is not None:
        cleaned = cleaned.reindex(columns=list(tickers))

    availability = cleaned.notna().mean(axis=0)
    keep = availability[availability >= min_non_missing].index
    cleaned = cleaned.loc[:, keep]
    cleaned = cleaned.ffill().dropna(how="any")

    if cleaned.empty:
        raise ValueError("No usable prices remain after cleaning.")
    return cleaned


def prices_to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert adjusted close prices to simple daily returns."""
    returns = prices.pct_change()
    returns = returns.replace([float("inf"), float("-inf")], pd.NA)
    return returns.dropna(how="any")
